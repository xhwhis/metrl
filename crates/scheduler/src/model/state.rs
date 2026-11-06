use std::collections::BTreeMap;
use std::time::Instant;

use metrl_types::{Generation, RequestId, Seq};

/// One model's task log as seen by the server that may own it.
#[derive(Debug, Default)]
pub struct State {
    /// Every task up to here has run or failed.
    pub(super) applied: Seq,
    /// The lease this server holds; `None` while another server may own the
    /// model.
    pub(super) owned: Option<Generation>,
    /// Known tasks after `applied`, while owned.
    pub(super) tasks: BTreeMap<Seq, Task>,
}

/// A known task. `due` is when it has waited the gap timeout since it entered
/// the log; once the next seq is missing, the earliest `due` of the tasks
/// behind it fails every waiting one. Failed tasks keep theirs, so a task that
/// arrives behind a gap that already timed out fails at once.
#[derive(Debug)]
pub(super) enum Task {
    Waiting { request: RequestId, due: Instant },
    Running,
    Failed { due: Instant },
}

impl State {
    /// Every task up to here has run or failed. The actor feeds tasks after
    /// it.
    pub fn applied(&self) -> Seq {
        self.applied
    }

    pub(super) fn next(&self) -> Seq {
        Seq(self.applied.0 + 1)
    }

    pub(super) fn assert_invariants(&self) {
        debug_assert!(
            self.tasks.keys().all(|seq| *seq > self.applied),
            "tasks up to applied are dropped as the log advances"
        );
        debug_assert!(
            self.owned.is_some() || self.tasks.is_empty(),
            "a server that does not own the model keeps no tasks"
        );
        debug_assert!(
            self.tasks
                .iter()
                .all(|(seq, task)| !matches!(task, Task::Running) || *seq == self.next()),
            "only the task right after applied runs"
        );
        debug_assert!(
            matches!(self.tasks.get(&self.next()), None | Some(Task::Running)),
            "an owner runs or passes over the next task as soon as it is known"
        );
    }
}
