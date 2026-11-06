//! Trainer state this worker holds for each bound model, and the tasks that
//! change it. Each model runs one call at a time.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use error_stack::Report;
use metrl_object::ObjectError;
use metrl_proto::worker::{
    self, BindRequest, ExecuteRequest, ExecuteResponse, ForwardBackwardDone, OptimStepDone,
    TaskFailed, execute_request, execute_response,
};
use metrl_types::{AdamParams, Generation, ModelId, Seq};
use parking_lot::Mutex;

use crate::{Trainer, WorkerError, tinker};

/// The server's worker client gives up after 30 seconds. A call that waits
/// longer for the model's slot answers `Unavailable` before then, so the
/// server retries instead of holding the call. Engine calls are never cut
/// short: an abandoned call keeps running in the engine, so the slot stays
/// held until it returns and a retry finds its response.
const SLOT_WAIT: Duration = Duration::from_secs(25);

pub(crate) struct Training<T> {
    trainer: T,
    object: metrl_object::Store,
    /// The outer lock only finds a model's slot; the slot's own lock is held
    /// across engine calls.
    models: Mutex<HashMap<ModelId, Arc<tokio::sync::Mutex<Option<Bound>>>>>,
}

/// A model whose trainer state lives on this worker.
struct Bound {
    generation: Generation,
    /// The last task that ran, so a repeated call returns its response
    /// instead of applying it twice.
    last: Option<Ran>,
}

/// A task the engine ran, kept as soon as the engine returns so that nothing
/// after it can make a retry run the task again.
struct Ran {
    seq: Seq,
    response: ExecuteResponse,
    /// Output not yet in the object store; a repeated call writes it.
    unwritten: Option<Output>,
}

struct Output {
    key: String,
    body: Bytes,
}

impl<T: Trainer> Training<T> {
    pub(crate) fn new(trainer: T, object: metrl_object::Store) -> Self {
        Self {
            trainer,
            object,
            models: Mutex::new(HashMap::new()),
        }
    }

    fn slot(&self, model: ModelId) -> Arc<tokio::sync::Mutex<Option<Bound>>> {
        let mut models = self.models.lock();
        Arc::clone(models.entry(model).or_default())
    }

    pub(crate) async fn bind(&self, request: BindRequest) -> Result<(), Report<WorkerError>> {
        let model = model_id(&request.model_id)?;
        let generation = Generation(request.generation);
        let slot = self.slot(model);
        let mut guard = tokio::time::timeout(SLOT_WAIT, slot.lock())
            .await
            .map_err(|error| Report::new(WorkerError::Unavailable).attach(error))?;
        match guard.as_mut() {
            Some(bound) if bound.generation > generation => {
                return Err(Report::new(WorkerError::Superseded));
            }
            Some(bound) => bound.generation = generation,
            None => {
                self.trainer
                    .create(model, &request.base, request.lora_rank, request.seed)
                    .await?;
                *guard = Some(Bound {
                    generation,
                    last: None,
                });
            }
        }
        drop(guard);
        Ok(())
    }

    pub(crate) async fn execute(
        &self, request: ExecuteRequest,
    ) -> Result<ExecuteResponse, Report<WorkerError>> {
        let model = model_id(&request.model_id)?;
        let generation = Generation(request.generation);
        let seq = Seq(request.seq);
        let slot = self.slot(model);
        let mut guard = tokio::time::timeout(SLOT_WAIT, slot.lock())
            .await
            .map_err(|error| Report::new(WorkerError::Unavailable).attach(error))?;
        let Some(bound) = guard.as_mut() else {
            return Err(Report::new(WorkerError::Unbound));
        };
        if bound.generation > generation {
            return Err(Report::new(WorkerError::Superseded));
        }
        if bound.generation < generation {
            return Err(Report::new(WorkerError::Unbound));
        }
        match &bound.last {
            Some(last) if last.seq == seq => {}
            Some(last) if last.seq > seq => return Err(Report::new(WorkerError::Superseded)),
            _ => {
                let task = request.task.ok_or_else(|| {
                    Report::new(WorkerError::Failed).attach("execute request has no task")
                })?;
                bound.last = Some(self.run(model, seq, task).await?);
            }
        }
        let last = bound
            .last
            .as_mut()
            .expect("the task that ran or repeated is the last one");
        if let Some(output) = &last.unwritten {
            match self.object.put(&output.key, output.body.clone()).await {
                Ok(()) => last.unwritten = None,
                // Retrying cannot store the output, so the step fails rather
                // than hold the model forever.
                Err(report) if matches!(report.current_context(), ObjectError::Rejected) => {
                    let report = report.change_context(WorkerError::Failed);
                    last.response = ExecuteResponse {
                        outcome: Some(failed(seq, &report)),
                    };
                    last.unwritten = None;
                }
                Err(report) => return Err(report.change_context(WorkerError::Unavailable)),
            }
        }
        let response = last.response.clone();
        drop(guard);
        Ok(response)
    }

    /// Runs the task on the engine. Fails, keeping nothing, only on an object
    /// store failure that a retry may get past; every other outcome is kept.
    async fn run(
        &self, model: ModelId, seq: Seq, task: execute_request::Task,
    ) -> Result<Ran, Report<WorkerError>> {
        let (outcome, unwritten) = match task {
            execute_request::Task::ForwardBackward(task) => {
                match self.object.get(&task.input_key).await {
                    Ok(input) => {
                        match self.forward_backward(model, &input).await {
                            Ok(body) => {
                                let done = execute_response::Outcome::ForwardBackward(
                                    ForwardBackwardDone {
                                        output_key: task.output_key.clone(),
                                    },
                                );
                                let output = Output {
                                    key: task.output_key,
                                    body,
                                };
                                (done, Some(output))
                            }
                            Err(report) => (failed(seq, &report), None),
                        }
                    }
                    // A missing or forbidden input fails the task instead of
                    // blocking every later one behind retries.
                    Err(report) if matches!(report.current_context(), ObjectError::Rejected) => {
                        (
                            failed(seq, &report.change_context(WorkerError::Failed)),
                            None,
                        )
                    }
                    Err(report) => return Err(report.change_context(WorkerError::Unavailable)),
                }
            }
            execute_request::Task::OptimStep(params) => {
                let params = AdamParams {
                    learning_rate: params.learning_rate,
                    beta1: params.beta1,
                    beta2: params.beta2,
                    eps: params.eps,
                    weight_decay: params.weight_decay,
                    grad_clip_norm: params.grad_clip_norm,
                };
                let outcome = self.optim_step(model, params).await.map_or_else(
                    |report| failed(seq, &report),
                    |metrics| {
                        execute_response::Outcome::OptimStep(OptimStepDone {
                            metrics: metrics.into_iter().collect(),
                        })
                    },
                );
                (outcome, None)
            }
        };
        Ok(Ran {
            seq,
            response: ExecuteResponse {
                outcome: Some(outcome),
            },
            unwritten,
        })
    }

    /// The encoded output of a forward-backward pass.
    async fn forward_backward(
        &self, model: ModelId, input: &[u8],
    ) -> Result<Bytes, Report<WorkerError>> {
        let request = tinker::decode(input)?;
        let data = request.data.len();
        let output = self.trainer.forward_backward(model, request).await?;
        if output.loss_fn_outputs.len() != data {
            return Err(Report::new(WorkerError::Failed).attach(format!(
                "engine returned {} outputs for {data} datums",
                output.loss_fn_outputs.len()
            )));
        }
        tinker::encode(output)
    }

    async fn optim_step(
        &self, model: ModelId, params: AdamParams,
    ) -> Result<BTreeMap<String, f64>, Report<WorkerError>> {
        let metrics = self.trainer.optim_step(model, params).await?;
        if let Some((key, value)) = metrics.iter().find(|(_, value)| !value.is_finite()) {
            return Err(Report::new(WorkerError::Failed)
                .attach(format!("engine returned metric {key} = {value}")));
        }
        Ok(metrics)
    }
}

/// The outcome of a task the engine rejected or failed.
fn failed(seq: Seq, report: &Report<WorkerError>) -> execute_response::Outcome {
    tracing::warn!(error = ?report, seq = seq.0, "task failed");
    let (category, message) = match report.current_context() {
        WorkerError::Invalid(message) => (worker::ErrorCategory::User, message.clone()),
        other => (worker::ErrorCategory::Server, other.to_string()),
    };
    execute_response::Outcome::Failed(TaskFailed {
        category: category as i32,
        message,
    })
}

fn model_id(id: &str) -> Result<ModelId, Report<WorkerError>> {
    use error_stack::ResultExt;
    uuid::Uuid::parse_str(id)
        .map(ModelId)
        .change_context_lazy(|| WorkerError::Invalid("model_id must be a UUID".to_owned()))
}
