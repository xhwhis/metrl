use std::time::Duration;

use metrl_types::{Generation, RequestId, Seq};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Event {
    /// This server won the model's lease. `applied` is the log position
    /// recorded with it. The actor then feeds every task after `applied`
    /// from a scan of the log, because the scheduler keeps no tasks across
    /// ownership changes. Each takeover carries a new generation.
    OwnershipAcquired {
        generation: Generation,
        applied: Seq,
    },
    /// This server no longer holds the lease; nothing runs until it wins it
    /// again.
    OwnershipLost,
    /// A task in the log, from the takeover scan or CDC. Tasks may repeat
    /// and arrive out of order. `waited` is how long ago it entered the log,
    /// so a gap's age survives a takeover.
    TaskArrived {
        seq: Seq,
        request: RequestId,
        waited: Duration,
    },
    /// A task the log already records as failed before it ran. The log
    /// passes over it once every earlier task is applied.
    TaskFailed { seq: Seq, waited: Duration },
    /// The task dispatched by [`Effect::Execute`] finished, and its result,
    /// success or failure, is recorded.
    Executed { seq: Seq, generation: Generation },
    /// Time passed; the actor sends it at
    /// [`next_deadline`](super::next_deadline).
    Tick,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Effect {
    /// Run the task, then record its result, success or failure, together
    /// with `applied = seq`, fenced by `generation`, and report
    /// [`Event::Executed`]. A failed task is a no-op in the log.
    Execute {
        seq: Seq,
        request: RequestId,
        generation: Generation,
    },
    /// Record the task as failed without running it, fenced by
    /// `generation`: `missing` did not arrive in time, and tasks never skip
    /// a seq.
    Fail {
        seq: Seq,
        generation: Generation,
        missing: Seq,
    },
}
