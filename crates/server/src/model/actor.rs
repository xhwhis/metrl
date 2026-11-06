use std::ops::ControlFlow;
use std::time::{Instant, SystemTime};

use metrl_meta::{Lease, Store, View};
use metrl_scheduler::model::{self, Effect, Event, State};
use metrl_types::{ErrorCategory, Outcome};
use tokio::task::JoinSet;

use crate::Workers;
use crate::model::dispatch::{self, Binding, Finished, Owner};

struct Actor {
    owner: Owner,
    view: View,
    state: State,
    /// Taken by the running task and handed back when it finishes.
    binding: Option<Binding>,
    running: JoinSet<Finished>,
}

/// Runs the model's task log for the lease's generation. Returns once a
/// newer generation owns the model; the lease supervisor aborts it sooner if
/// the lease lapses.
pub(super) async fn run(lease: Lease, store: Store, mut view: View, workers: Workers) {
    tracing::info!(applied = lease.applied.0, "owning model");
    // The lease was taken before this instant. Until the view holds every
    // commit made before it, it may still show a task as pending that the
    // previous owner already failed, and running that task would change the
    // model behind a failure the user was told about.
    let acquired = Instant::now();
    while !view.applied_through(acquired) {
        view.changed().await;
    }
    let mut state = State::default();
    let acquired = Event::OwnershipAcquired {
        generation: lease.generation,
        applied: lease.applied,
    };
    let effects = model::on_event(&mut state, Instant::now(), acquired);
    let mut actor = Actor {
        binding: Some(Binding::new(lease.worker, lease.applied)),
        owner: Owner {
            lease,
            store,
            workers,
        },
        view,
        state,
        running: JoinSet::new(),
    };
    if actor.apply(effects).await.is_break() {
        return;
    }
    while actor.next().await.is_continue() {}
    tracing::info!("a newer generation owns the model");
}

impl Actor {
    /// Feeds the log, then waits for and handles one wakeup.
    async fn next(&mut self) -> ControlFlow<()> {
        let effects = self.feed();
        self.apply(effects).await?;
        let deadline = model::next_deadline(&self.state);
        tokio::select! {
            () = self.view.changed() => ControlFlow::Continue(()),
            Some(finished) = self.running.join_next() => {
                match finished.expect("dispatch tasks are never aborted while the actor runs") {
                    Finished::Executed { seq, binding } => {
                        self.binding = Some(binding);
                        let executed = Event::Executed {
                            seq,
                            generation: self.owner.lease.generation,
                        };
                        let effects = model::on_event(&mut self.state, Instant::now(), executed);
                        self.apply(effects).await
                    }
                    Finished::Fenced => ControlFlow::Break(()),
                }
            }
            () = async {
                match deadline {
                    Some(at) => tokio::time::sleep_until(tokio::time::Instant::from_std(at)).await,
                    None => std::future::pending().await,
                }
            } => {
                let effects = model::on_event(&mut self.state, Instant::now(), Event::Tick);
                self.apply(effects).await
            }
        }
    }

    /// Hands the scheduler every task the log holds after `applied`. Tasks
    /// repeat on every call; the scheduler ignores the ones it knows.
    fn feed(&mut self) -> Vec<Effect> {
        let tasks = self
            .view
            .tasks_after(self.owner.lease.model, self.state.applied());
        let now = SystemTime::now();
        tasks
            .into_iter()
            .flat_map(|task| {
                // Clocks of the servers that appended tasks may run ahead.
                let waited = now.duration_since(task.created_at).unwrap_or_default();
                let event = match task.outcome {
                    Some(Outcome::Failed { .. }) => {
                        Event::TaskFailed {
                            seq: task.seq,
                            waited,
                        }
                    }
                    // A task that already ran is dispatched again only after a
                    // takeover raced its commit; the worker returns the
                    // response it kept.
                    None | Some(_) => {
                        Event::TaskArrived {
                            seq: task.seq,
                            request: task.request,
                            waited,
                        }
                    }
                };
                model::on_event(&mut self.state, Instant::now(), event)
            })
            .collect()
    }

    async fn apply(&mut self, effects: Vec<Effect>) -> ControlFlow<()> {
        use tracing::Instrument;
        for effect in effects {
            match effect {
                Effect::Execute { seq, request, .. } => {
                    let Some(task) = self.view.task(self.owner.lease.model, seq) else {
                        // Only a reset of the database removes tasks; the
                        // model's lease goes with them.
                        tracing::warn!(seq = seq.0, "dispatched task is gone from the log");
                        continue;
                    };
                    let binding = self
                        .binding
                        .take()
                        .expect("the scheduler runs one task at a time");
                    let span = tracing::info_span!("task", seq = seq.0, request_id = %request.0);
                    self.running.spawn(
                        dispatch::execute(self.owner.clone(), task, binding).instrument(span),
                    );
                }
                Effect::Fail { seq, missing, .. } => {
                    let outcome = Outcome::Failed {
                        category: ErrorCategory::User,
                        message: format!(
                            "seq {} was never received; later requests cannot run before it",
                            missing.0
                        ),
                    };
                    if !dispatch::fail(&self.owner, seq, &outcome).await {
                        return ControlFlow::Break(());
                    }
                }
            }
        }
        ControlFlow::Continue(())
    }
}
