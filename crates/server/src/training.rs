//! Training models and their task logs as the API sees them: creating
//! models, appending tasks, and reading their results. Model actors run the
//! tasks.

use std::collections::BTreeMap;
use std::time::SystemTime;

use bytes::Bytes;
use error_stack::{Report, ResultExt};
use metrl_meta::{Store, Task, View};
use metrl_object::ObjectError;
use metrl_types::{
    ErrorCategory, Model, ModelId, OrgId, Outcome, RequestId, Role, Seq, SessionId, TaskInput,
};
use uuid::Uuid;

use crate::{Sampling, ServerError, Workers};

/// A model takes no new task while this many wait, except the one it runs
/// next: seq 1, or a seq whose predecessor has an outcome.
/// `parallel_fwdbwd_chunks` is false, so the SDK sends chunks in seq order.
/// Each exempt task is pending itself, so the next seq is not exempt.
const MAX_PENDING: usize = 64;

const DATABASE: ServerError = ServerError::Unavailable("database unavailable");

/// An object store failure as the API reports it: one a retry may get past
/// is 503; a missing or forbidden object is 500 and logged as an error,
/// because it needs an operator. The SDK retries both alike, so neither
/// ends a submission; `retrieve` reports an unreadable output as a failed
/// task instead.
fn object_store(report: Report<ObjectError>) -> Report<ServerError> {
    let context = match report.current_context() {
        ObjectError::Rejected => ServerError::Internal("object store rejected the request"),
        ObjectError::Config | ObjectError::Request => {
            ServerError::Unavailable("object store unavailable")
        }
    };
    report.change_context(context)
}

/// What a `create_model` call asks for.
pub struct NewModel {
    pub session: SessionId,
    pub session_seq: u64,
    pub base: String,
    pub lora_rank: u32,
    /// Picked by the server when `None`.
    pub seed: Option<u64>,
}

pub enum TrainingResult {
    Pending,
    Created(ModelId),
    /// A Tinker `ForwardBackwardOutput`.
    ForwardBackward(Bytes),
    OptimStep(BTreeMap<String, f64>),
    Failed {
        category: ErrorCategory,
        message: String,
    },
}

#[derive(Clone)]
pub struct Training {
    store: Store,
    view: View,
    workers: Workers,
    object: metrl_object::Store,
}

impl Training {
    pub fn new(store: Store, view: View, workers: Workers, object: metrl_object::Store) -> Self {
        Self {
            store,
            view,
            workers,
            object,
        }
    }

    /// Waits until the view has changed since this handle last waited.
    pub async fn changed(&mut self) {
        self.view.changed().await;
    }

    /// Creates a model on a base that a registered trainer serves. A retry
    /// returns the model the first call created, even if it reaches another
    /// server, whose memory lacks the session, or the trainer left.
    pub async fn create_model(
        &self, org: &OrgId, sessions: &Sampling, request: NewModel,
    ) -> Result<ModelId, Report<ServerError>> {
        let existing = self
            .store
            .model_by_session(org, request.session, request.session_seq)
            .await
            .change_context(DATABASE)
            .attach(format!(
                "session {}, seq {}",
                request.session.0, request.session_seq
            ))?;
        let model = if let Some(model) = existing {
            model
        } else {
            sessions.heartbeat(org, request.session)?;
            let base = self
                .workers
                .base(&request.base, Role::Trainer)
                .ok_or_else(|| {
                    Report::new(ServerError::Invalid(
                        "base model is not served by a registered trainer",
                    ))
                })?;
            if !(1..=base.max_lora_rank).contains(&request.lora_rank) {
                return Err(Report::new(ServerError::Invalid(
                    "lora rank must be between 1 and the base's max_lora_rank",
                ))
                .attach(format!("max_lora_rank {}", base.max_lora_rank)));
            }
            let id = ModelId(Uuid::now_v7());
            let model = Model {
                id,
                org: org.clone(),
                session: request.session,
                session_seq: request.session_seq,
                base: base.name,
                base_revision: base.revision,
                arch: base.arch,
                tokenizer_id: base.tokenizer_id,
                lora_rank: request.lora_rank,
                seed: request.seed.unwrap_or_else(|| id.0.as_u64_pair().1),
            };
            // Another server may have created it for a concurrent retry.
            self.store
                .create_model(&model)
                .await
                .change_context(DATABASE)
                .attach(format!("model {}", model.id.0))?
        };
        let same = model.base == request.base
            && model.lora_rank == request.lora_rank
            && request.seed.is_none_or(|seed| seed == model.seed);
        if !same {
            return Err(Report::new(ServerError::Invalid(
                "create_model retry changed the model",
            )));
        }
        Ok(model.id)
    }

    pub async fn model(&self, org: &OrgId, id: ModelId) -> Result<Model, Report<ServerError>> {
        self.store
            .model(org, id)
            .await
            .change_context(DATABASE)
            .attach(format!("model {}", id.0))?
            .ok_or_else(|| Report::new(ServerError::NotFound))
    }

    /// Appends a task to the model's log. `body` is the forward-backward
    /// request, stored as the task's `input.pb`. A retry of the same seq
    /// returns the request that took it, and fails if its input changed.
    pub async fn submit(
        &self, org: &OrgId, model: ModelId, seq: Seq, input: TaskInput, body: Option<Bytes>,
    ) -> Result<RequestId, Report<ServerError>> {
        // Seqs are stored as bigint.
        if seq == Seq(0) || i64::try_from(seq.0).is_err() {
            return Err(Report::new(ServerError::Invalid(
                "seq_id must be between 1 and 2^63 - 1",
            )));
        }
        if let Some(task) = self.view.task(model, seq) {
            return self.retried(org, task, &input, body.as_ref()).await;
        }
        self.model(org, model).await?;
        let runs_next = seq == Seq(1)
            || self
                .view
                .task(model, Seq(seq.0 - 1))
                .is_some_and(|previous| previous.outcome.is_some());
        if !runs_next && self.view.pending_tasks(model) >= MAX_PENDING {
            return Err(Report::new(ServerError::Capacity)
                .attach(format!("model {} has {MAX_PENDING} pending tasks", model.0)));
        }
        let request = RequestId(Uuid::now_v7());
        if let Some(body) = &body {
            let key = metrl_object::task_input(org, model, seq, request);
            self.object
                .put(&key, body.clone())
                .await
                .map_err(object_store)
                .attach(format!("model {}, seq {}", model.0, seq.0))?;
        }
        let task = Task {
            model,
            seq,
            request,
            org: org.clone(),
            input,
            outcome: None,
            created_at: SystemTime::now(),
        };
        if self
            .store
            .insert_task(&task)
            .await
            .change_context(DATABASE)
            .attach(format!("model {}, seq {}", model.0, seq.0))?
        {
            return Ok(request);
        }
        // The seq was taken after the view was read, or before it caught up.
        let taken = self
            .store
            .task(org, model, seq)
            .await
            .change_context(DATABASE)
            .attach(format!("model {}, seq {}", model.0, seq.0))?
            .ok_or_else(|| Report::new(DATABASE).attach("the task holding a taken seq is gone"))?;
        self.retried(org, taken, &task.input, body.as_ref()).await
    }

    /// The request of `existing` if it holds the same input. Bodies are
    /// compared byte for byte, so callers must encode them canonically.
    async fn retried(
        &self, org: &OrgId, existing: Task, input: &TaskInput, body: Option<&Bytes>,
    ) -> Result<RequestId, Report<ServerError>> {
        if existing.org != *org {
            return Err(Report::new(ServerError::NotFound));
        }
        let changed = || Report::new(ServerError::Invalid("task retry changed input"));
        if existing.input != *input {
            return Err(changed());
        }
        if let Some(body) = body {
            let key = metrl_object::task_input(
                &existing.org,
                existing.model,
                existing.seq,
                existing.request,
            );
            let stored = self
                .object
                .get(&key)
                .await
                .map_err(object_store)
                .attach_with(|| format!("model {}, seq {}", existing.model.0, existing.seq.0))?;
            if stored != body {
                return Err(changed());
            }
        }
        Ok(existing.request)
    }

    /// The result of a `create_model` call or a task.
    pub async fn retrieve(
        &self, org: &OrgId, request: RequestId,
    ) -> Result<TrainingResult, Report<ServerError>> {
        let task = if let Some(task) = self.view.task_by_request(request) {
            Some(task)
        } else {
            // `models` is not published, and the view may not hold a task
            // this server has just accepted.
            let model = self
                .store
                .model(org, ModelId(request.0))
                .await
                .change_context(DATABASE)
                .attach(format!("request {}", request.0))?;
            if let Some(model) = model {
                return Ok(TrainingResult::Created(model.id));
            }
            self.store
                .task_by_request(org, request)
                .await
                .change_context(DATABASE)
                .attach(format!("request {}", request.0))?
        };
        let task = task
            .filter(|task| task.org == *org)
            .ok_or_else(|| Report::new(ServerError::NotFound))?;
        Ok(match task.outcome {
            None => TrainingResult::Pending,
            Some(Outcome::ForwardBackward { output_key }) => {
                match self.object.get(&output_key).await {
                    Ok(output) => TrainingResult::ForwardBackward(output),
                    // The SDK retries a failed retrieve without limit, so an
                    // output that cannot be read answers as a failed task.
                    Err(report) if matches!(report.current_context(), ObjectError::Rejected) => {
                        tracing::error!(
                            error = ?report,
                            model_id = %task.model.0,
                            seq = task.seq.0,
                            "task output cannot be read"
                        );
                        TrainingResult::Failed {
                            category: ErrorCategory::Server,
                            message: "the task's output cannot be read".to_owned(),
                        }
                    }
                    Err(report) => {
                        return Err(object_store(report))
                            .attach(format!("model {}, seq {}", task.model.0, task.seq.0));
                    }
                }
            }
            Some(Outcome::OptimStep { metrics }) => TrainingResult::OptimStep(metrics),
            Some(Outcome::Failed { category, message }) => {
                TrainingResult::Failed { category, message }
            }
        })
    }
}
