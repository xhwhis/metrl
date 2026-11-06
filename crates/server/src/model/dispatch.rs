//! Runs one task on a trainer and records its outcome, fenced by the
//! generation that dispatched it.

use std::time::Duration;

use metrl_meta::{Lease, Store, Task};
use metrl_proto::worker::{self, BindRequest, ExecuteRequest, execute_request, execute_response};
use metrl_types::{ErrorCategory, Outcome, Role, Seq, TaskInput, WorkerId};
use tonic::{Code, Status};

use crate::Workers;

/// What a dispatched task needs from the actor that owns the model.
#[derive(Clone)]
pub(super) struct Owner {
    pub(super) lease: Lease,
    pub(super) store: Store,
    pub(super) workers: Workers,
}

/// The trainer this generation runs the model's tasks on.
pub(super) struct Binding {
    worker: Option<WorkerId>,
    /// The worker accepted this generation's Bind.
    bound: bool,
    /// The worker may hold state no other worker has, so the model cannot
    /// move to another one.
    has_state: bool,
}

impl Binding {
    pub(super) fn new(worker: Option<WorkerId>, applied: Seq) -> Self {
        Self {
            worker,
            bound: false,
            has_state: applied > Seq(0),
        }
    }
}

pub(super) enum Finished {
    /// The task's outcome is recorded and `applied` advanced to it.
    Executed { seq: Seq, binding: Binding },
    /// A newer generation owns the model.
    Fenced,
}

/// What one attempt to run the task came to.
enum Attempt {
    Done(Outcome),
    Retry { after: Duration },
    Fenced,
}

/// Runs the task on the model's trainer until its outcome is recorded.
pub(super) async fn execute(owner: Owner, task: Task, mut binding: Binding) -> Finished {
    let sent = match &task.input {
        TaskInput::ForwardBackward => {
            execute_request::Task::ForwardBackward(worker::ForwardBackward {
                input_key: metrl_object::task_input(&task.org, task.model, task.seq, task.request),
                output_key: metrl_object::task_output(
                    &task.org,
                    task.model,
                    task.seq,
                    task.request,
                    owner.lease.generation,
                ),
            })
        }
        TaskInput::OptimStep(params) => {
            execute_request::Task::OptimStep(worker::AdamParams {
                learning_rate: params.learning_rate,
                beta1: params.beta1,
                beta2: params.beta2,
                eps: params.eps,
                weight_decay: params.weight_decay,
                grad_clip_norm: params.grad_clip_norm,
            })
        }
    };
    let outcome = loop {
        match attempt(&owner, task.seq, &sent, &mut binding).await {
            Attempt::Done(outcome) => break outcome,
            Attempt::Retry { after } => tokio::time::sleep(after).await,
            Attempt::Fenced => return Finished::Fenced,
        }
    };
    if let Outcome::Failed { category, message } = &outcome {
        tracing::info!(?category, message, "task failed");
    }
    loop {
        match owner
            .store
            .complete_task(&owner.lease, task.seq, &outcome)
            .await
        {
            Ok(true) => {
                return Finished::Executed {
                    seq: task.seq,
                    binding,
                };
            }
            Ok(false) => return Finished::Fenced,
            Err(report) => {
                tracing::warn!(error = ?report, "cannot record the task's outcome");
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    }
}

/// Records a task as failed without running it, retrying until the database
/// answers. False once a newer generation owns the model.
pub(super) async fn fail(owner: &Owner, seq: Seq, outcome: &Outcome) -> bool {
    loop {
        match owner.store.fail_task(&owner.lease, seq, outcome).await {
            Ok(current) => return current,
            Err(report) => {
                tracing::warn!(error = ?report, seq = seq.0, "cannot record the failed task");
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    }
}

/// Binds the trainer if this generation has not, then runs the task on it.
async fn attempt(
    owner: &Owner, seq: Seq, sent: &execute_request::Task, binding: &mut Binding,
) -> Attempt {
    let lease = &owner.lease;
    let live = match binding.worker.map(|id| owner.workers.get(id)) {
        Some(Some(live)) => live,
        Some(None) if binding.has_state => {
            return Attempt::Done(Outcome::Failed {
                category: ErrorCategory::Server,
                message: "the trainer that held this model's state is gone, and the state cannot \
                          be recovered"
                    .to_owned(),
            });
        }
        Some(None) | None => {
            match owner.workers.pick(
                &lease.base,
                &lease.base_revision,
                Role::Trainer,
                lease.model.0.as_u64_pair().1,
            ) {
                Ok(live) => {
                    *binding = Binding {
                        worker: Some(live.id),
                        bound: false,
                        has_state: false,
                    };
                    live
                }
                Err(report) => {
                    tracing::warn!(error = ?report, "waiting for a trainer");
                    return Attempt::Retry {
                        after: Duration::from_secs(1),
                    };
                }
            }
        }
    };
    let mut client = live.client;
    if !binding.bound {
        let request = BindRequest {
            worker_id: live.id.0.to_string(),
            model_id: lease.model.0.to_string(),
            generation: lease.generation.0,
            base: lease.base.clone(),
            lora_rank: lease.lora_rank,
            seed: lease.seed,
            revision: lease.base_revision.clone(),
        };
        if let Err(status) = client.bind(request).await {
            return rejected(&status, binding);
        }
        match owner.store.record_worker(lease, live.id).await {
            Ok(true) => binding.bound = true,
            Ok(false) => return Attempt::Fenced,
            Err(report) => {
                tracing::warn!(error = ?report, "cannot record the model's trainer");
                return Attempt::Retry {
                    after: Duration::from_secs(1),
                };
            }
        }
    }
    // The worker may apply the task even if its answer is lost.
    binding.has_state = true;
    let request = ExecuteRequest {
        worker_id: live.id.0.to_string(),
        model_id: lease.model.0.to_string(),
        generation: lease.generation.0,
        seq: seq.0,
        task: Some(sent.clone()),
    };
    match client.execute(request).await {
        Ok(response) => Attempt::Done(outcome(response.into_inner())),
        Err(status) => rejected(&status, binding),
    }
}

/// What a trainer call that returned an error comes to.
fn rejected(status: &Status, binding: &mut Binding) -> Attempt {
    tracing::warn!(%status, "trainer call failed");
    match status.code() {
        Code::Unavailable | Code::DeadlineExceeded | Code::Cancelled | Code::Unknown => {
            Attempt::Retry {
                after: Duration::from_secs(1),
            }
        }
        // A newer generation has run on the worker; the lease supervisor
        // stops this actor once it learns the same.
        Code::Aborted => {
            Attempt::Retry {
                after: Duration::from_secs(3),
            }
        }
        // The worker has not seen this generation's Bind, or another worker
        // now answers at its address.
        Code::FailedPrecondition => {
            binding.bound = false;
            Attempt::Retry {
                after: Duration::from_secs(1),
            }
        }
        Code::InvalidArgument => {
            Attempt::Done(Outcome::Failed {
                category: ErrorCategory::User,
                message: status.message().to_owned(),
            })
        }
        _ => {
            Attempt::Done(Outcome::Failed {
                category: ErrorCategory::Server,
                message: "the trainer failed".to_owned(),
            })
        }
    }
}

fn outcome(response: worker::ExecuteResponse) -> Outcome {
    match response.outcome {
        Some(execute_response::Outcome::ForwardBackward(done)) => {
            Outcome::ForwardBackward {
                output_key: done.output_key,
            }
        }
        Some(execute_response::Outcome::OptimStep(done)) => {
            Outcome::OptimStep {
                metrics: done.metrics.into_iter().collect(),
            }
        }
        Some(execute_response::Outcome::Failed(failed)) => {
            Outcome::Failed {
                category: match failed.category() {
                    worker::ErrorCategory::User => ErrorCategory::User,
                    worker::ErrorCategory::Server | worker::ErrorCategory::Unspecified => {
                        ErrorCategory::Server
                    }
                },
                message: failed.message,
            }
        }
        None => {
            Outcome::Failed {
                category: ErrorCategory::Server,
                message: "the trainer returned no outcome".to_owned(),
            }
        }
    }
}
