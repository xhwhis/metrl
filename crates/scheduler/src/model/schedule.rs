use std::time::{Duration, Instant};

use crate::model::state::Task;
use crate::model::{Effect, Event, State};

/// How long a missing seq may block later tasks before they fail. The SDK
/// sends seqs in order, so a gap this old means the request was lost.
const GAP_TIMEOUT: Duration = Duration::from_secs(600);

/// Applies `event` at `now` and returns what the actor must do. Timers that
/// are due fire before the event, so the effects, in order, do not depend on
/// when [`Event::Tick`]s arrive.
pub fn on_event(state: &mut State, now: Instant, event: Event) -> Vec<Effect> {
    let mut effects = fail_after_gap(state, now);
    match event {
        Event::OwnershipAcquired {
            generation,
            applied,
        } => {
            state.owned = Some(generation);
            state.applied = applied;
            state.tasks.clear();
        }
        Event::OwnershipLost => {
            state.owned = None;
            state.tasks.clear();
        }
        Event::TaskArrived {
            seq,
            request,
            waited,
        } => {
            if state.owned.is_some() && seq > state.applied {
                let due = now + GAP_TIMEOUT.saturating_sub(waited);
                state
                    .tasks
                    .entry(seq)
                    .or_insert(Task::Waiting { request, due });
            }
        }
        Event::TaskFailed { seq, waited } => {
            if state.owned.is_some() && seq > state.applied {
                let due = now + GAP_TIMEOUT.saturating_sub(waited);
                let task = state.tasks.entry(seq).or_insert(Task::Failed { due });
                if let Task::Waiting { due, .. } = *task {
                    *task = Task::Failed { due };
                }
            }
        }
        Event::Executed { seq, generation } => {
            if state.owned == Some(generation)
                && matches!(state.tasks.get(&seq), Some(Task::Running))
            {
                state.tasks.remove(&seq);
                state.applied = seq;
            }
        }
        Event::Tick => {}
    }
    effects.extend(advance(state));
    // A task that arrived behind a gap that is already too old fails now.
    effects.extend(fail_after_gap(state, now));
    state.assert_invariants();
    effects
}

/// When the actor must send [`Event::Tick`] next: once the earliest task
/// behind a missing seq has waited the gap timeout, while any still waits.
pub fn next_deadline(state: &State) -> Option<Instant> {
    let behind_gap = !state.tasks.contains_key(&state.next());
    let waiting = state
        .tasks
        .values()
        .any(|task| matches!(task, Task::Waiting { .. }));
    if !(behind_gap && waiting) {
        return None;
    }
    state
        .tasks
        .values()
        .filter_map(|task| {
            match task {
                Task::Waiting { due, .. } | Task::Failed { due } => Some(*due),
                Task::Running => None,
            }
        })
        .min()
}

/// Fails every waiting task once the gap before them is too old.
fn fail_after_gap(state: &mut State, now: Instant) -> Vec<Effect> {
    let (Some(generation), Some(deadline)) = (state.owned, next_deadline(state)) else {
        return Vec::new();
    };
    if now < deadline {
        return Vec::new();
    }
    let missing = state.next();
    state
        .tasks
        .iter_mut()
        .filter_map(|(seq, task)| {
            let Task::Waiting { due, .. } = *task else {
                return None;
            };
            *task = Task::Failed { due };
            Some(Effect::Fail {
                seq: *seq,
                generation,
                missing,
            })
        })
        .collect()
}

/// Passes over failed tasks after `applied` and dispatches the next one.
fn advance(state: &mut State) -> Option<Effect> {
    let generation = state.owned?;
    while let Some(Task::Failed { .. }) = state.tasks.get(&state.next()) {
        state.applied = state.next();
        state.tasks.remove(&state.applied);
    }
    let next = state.next();
    let Some(Task::Waiting { request, .. }) = state.tasks.get(&next) else {
        return None;
    };
    let request = *request;
    state.tasks.insert(next, Task::Running);
    Some(Effect::Execute {
        seq: next,
        request,
        generation,
    })
}
