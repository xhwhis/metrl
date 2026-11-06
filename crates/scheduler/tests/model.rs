use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use metrl_scheduler::model::{self, Effect, Event, State};
use metrl_types::{Generation, RequestId, Seq};
use proptest::prelude::*;
use uuid::Uuid;

fn request(seq: u64) -> RequestId {
    RequestId(Uuid::from_u128(u128::from(seq)))
}

fn arrive(seq: u64) -> Event {
    arrive_after(seq, 0)
}

/// A task that entered the log `waited` seconds before the event.
fn arrive_after(seq: u64, waited: u64) -> Event {
    Event::TaskArrived {
        seq: Seq(seq),
        request: request(seq),
        waited: Duration::from_secs(waited),
    }
}

fn failed(seq: u64, waited: u64) -> Event {
    Event::TaskFailed {
        seq: Seq(seq),
        waited: Duration::from_secs(waited),
    }
}

fn acquire(generation: u64, applied: u64) -> Event {
    Event::OwnershipAcquired {
        generation: Generation(generation),
        applied: Seq(applied),
    }
}

fn executed(seq: u64, generation: u64) -> Event {
    Event::Executed {
        seq: Seq(seq),
        generation: Generation(generation),
    }
}

/// Feeds `(seconds, event)` pairs to a fresh state and renders one line per
/// event: its effects, then when the actor must tick next.
fn trace(events: Vec<(u64, Event)>) -> String {
    let start = Instant::now();
    let mut state = State::default();
    events
        .into_iter()
        .map(|(secs, event)| {
            let label = match &event {
                Event::OwnershipAcquired {
                    generation,
                    applied,
                } => format!("acquire g{} at {}", generation.0, applied.0),
                Event::OwnershipLost => "lost".to_owned(),
                Event::TaskArrived { seq, waited, .. } => {
                    format!("arrive {}{}", seq.0, ago(*waited))
                }
                Event::TaskFailed { seq, waited } => format!("failed {}{}", seq.0, ago(*waited)),
                Event::Executed { seq, generation } => {
                    format!("executed {}@g{}", seq.0, generation.0)
                }
                Event::Tick => "tick".to_owned(),
            };
            let effects: Vec<_> =
                model::on_event(&mut state, start + Duration::from_secs(secs), event)
                    .iter()
                    .map(|effect| {
                        match effect {
                            Effect::Execute {
                                seq,
                                request: id,
                                generation,
                            } => {
                                assert_eq!(*id, request(seq.0), "a task runs with its own request");
                                format!("execute {}@g{}", seq.0, generation.0)
                            }
                            Effect::Fail {
                                seq,
                                generation,
                                missing,
                            } => format!("fail {}@g{} missing {}", seq.0, generation.0, missing.0),
                        }
                    })
                    .collect();
            let effects = if effects.is_empty() {
                "-".to_owned()
            } else {
                effects.join(", ")
            };
            let tick = model::next_deadline(&state).map_or_else(String::new, |deadline| {
                format!(" (tick at {}s)", deadline.duration_since(start).as_secs())
            });
            format!("{secs}s {label}: {effects}{tick}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn ago(waited: Duration) -> String {
    if waited.is_zero() {
        String::new()
    } else {
        format!(" from {}s ago", waited.as_secs())
    }
}

// Catches a gap timer that restarts on every arrival, a late task that waits
// another ten minutes behind a gap that already timed out, and skipping the
// missing seq instead of running it when it finally arrives.
#[test]
fn gap_fails_waiting_tasks_but_never_skips_the_missing_seq() {
    let trace = trace(vec![
        (0, acquire(1, 0)),
        (0, arrive(2)),
        (599, arrive(3)),
        (600, Event::Tick),
        (700, arrive(4)),
        (800, arrive(1)),
        (801, executed(1, 1)),
        (900, arrive(5)),
    ]);
    insta::assert_snapshot!(trace, @r"
    0s acquire g1 at 0: -
    0s arrive 2: - (tick at 600s)
    599s arrive 3: - (tick at 600s)
    600s tick: fail 2@g1 missing 1, fail 3@g1 missing 1
    700s arrive 4: fail 4@g1 missing 1
    800s arrive 1: execute 1@g1
    801s executed 1@g1: -
    900s arrive 5: execute 5@g1
    ");
}

// Catches a late result from a lost generation advancing the log while the
// new owner runs the same task, and running a task after a takeover when the
// log reports it failed only after it arrived as waiting.
#[test]
fn takeover_ignores_results_of_the_old_generation() {
    let trace = trace(vec![
        (0, acquire(1, 0)),
        (0, arrive(1)),
        (1, Event::OwnershipLost),
        (1, arrive(2)),
        (2, acquire(2, 0)),
        (2, arrive(2)),
        (2, arrive(3)),
        (2, failed(3, 0)),
        (2, arrive(1)),
        (3, executed(1, 1)),
        (4, executed(1, 2)),
        (5, executed(2, 2)),
        (6, arrive(4)),
    ]);
    insta::assert_snapshot!(trace, @r"
    0s acquire g1 at 0: -
    0s arrive 1: execute 1@g1
    1s lost: -
    1s arrive 2: -
    2s acquire g2 at 0: -
    2s arrive 2: - (tick at 602s)
    2s arrive 3: - (tick at 602s)
    2s failed 3: - (tick at 602s)
    2s arrive 1: execute 1@g2
    3s executed 1@g1: -
    4s executed 1@g2: execute 2@g2
    5s executed 2@g2: -
    6s arrive 4: execute 4@g2
    ");
}

// Catches a takeover restarting the gap timer: a model whose owner changes
// more often than the timeout would otherwise block its waiting tasks forever.
#[test]
fn gap_age_survives_takeover() {
    let trace = trace(vec![
        (0, acquire(1, 0)),
        (0, arrive(2)),
        (400, Event::OwnershipLost),
        (400, acquire(2, 0)),
        (400, arrive_after(2, 400)),
        (400, arrive(3)),
        (600, Event::Tick),
        (700, Event::OwnershipLost),
        (700, acquire(3, 0)),
        (700, failed(2, 700)),
        (700, failed(3, 300)),
        (700, arrive(4)),
    ]);
    insta::assert_snapshot!(trace, @r"
    0s acquire g1 at 0: -
    0s arrive 2: - (tick at 600s)
    400s lost: -
    400s acquire g2 at 0: -
    400s arrive 2 from 400s ago: - (tick at 600s)
    400s arrive 3: - (tick at 600s)
    600s tick: fail 2@g2 missing 1, fail 3@g2 missing 1
    700s lost: -
    700s acquire g3 at 0: -
    700s failed 2 from 700s ago: -
    700s failed 3 from 300s ago: -
    700s arrive 4: fail 4@g3 missing 1
    ");
}

/// `1..=n` in any order, with some seqs repeated.
fn arrivals() -> impl Strategy<Value = (u64, Vec<u64>)> {
    (1_u64..=24)
        .prop_flat_map(|n| (Just(n), proptest::collection::vec(1..=n, 0..8)))
        .prop_flat_map(|(n, repeats)| {
            let all: Vec<_> = (1..=n).chain(repeats).collect();
            (Just(n), Just(all).prop_shuffle())
        })
}

proptest! {
    // Catches running a task before its predecessor, running a repeated
    // task twice, running two tasks at once, and losing a task that arrived
    // while another one ran.
    #[test]
    fn every_seq_runs_once_in_order(
        (n, arrivals) in arrivals(),
        replies_first in proptest::collection::vec(any::<bool>(), 64),
    ) {
        let now = Instant::now();
        let mut state = State::default();
        let mut arrivals = arrivals.into_iter().peekable();
        let mut replies_first = replies_first.into_iter();
        let mut running = None;
        let mut ran = Vec::new();
        let mut event = Some(acquire(1, 0));
        while let Some(current) = event {
            for effect in model::on_event(&mut state, now, current) {
                let Effect::Execute { seq, .. } = effect else {
                    return Err(TestCaseError::fail(format!(
                        "no task fails while time stands still: {effect:?}"
                    )));
                };
                prop_assert!(running.replace(seq.0).is_none(), "two tasks ran at once");
                ran.push(seq.0);
            }
            let reply = running.is_some()
                && (arrivals.peek().is_none() || replies_first.next().unwrap_or(true));
            event = if reply {
                running.take().map(|seq| executed(seq, 1))
            } else {
                arrivals.next().map(arrive)
            };
        }
        prop_assert_eq!(ran, (1..=n).collect::<Vec<_>>(), "every seq up to {} runs once, in order", n);
    }

    // Catches skipping or repeating a seq under ownership churn, late and
    // repeated results, and failures; running two tasks at once; effects
    // fenced by the wrong generation; and timeouts whose effects depend on
    // when the actor happens to tick.
    #[test]
    fn arbitrary_events_never_skip_or_repeat_a_seq(
        ops in proptest::collection::vec((0_u64..400, op()), 1..80),
        ticks in proptest::collection::vec(proptest::option::of(0_u64..100), 80),
    ) {
        let start = Instant::now();
        let plain = run(start, &ops, &[]);
        let ticked = run(start, &ops, &ticks);
        prop_assert_eq!(&plain, &ticked, "extra ticks changed the effects");

        let mut owner: Option<Owner> = None;
        for (event, effects) in &plain {
            if let Some(owner) = &mut owner {
                match event {
                    Event::TaskFailed { seq, .. } => {
                        owner.settled.insert(*seq);
                    }
                    Event::Executed { seq, generation }
                        if *generation == owner.generation && owner.running == Some(*seq) =>
                    {
                        owner.running = None;
                    }
                    _ => {}
                }
            }
            // An ownership change has only the effects of timers that fired
            // before it.
            for effect in effects {
                let Some(owner) = &mut owner else {
                    return Err(TestCaseError::fail(format!("{effect:?} while not owning")));
                };
                let (Effect::Execute { seq, generation, .. }
                | Effect::Fail { seq, generation, .. }) = effect;
                prop_assert_eq!(*generation, owner.generation, "{:?} has the wrong fence", effect);
                prop_assert!(*seq > owner.applied, "{effect:?} before {:?}", owner.applied);
                if let Effect::Execute { .. } = effect {
                    prop_assert!(
                        (owner.applied.0 + 1..seq.0).all(|seq| owner.settled.contains(&Seq(seq))),
                        "{effect:?} skipped a seq; settled {:?}", owner.settled
                    );
                    prop_assert!(!owner.settled.contains(seq), "{effect:?} repeats a seq");
                    prop_assert!(owner.running.replace(*seq).is_none(), "{effect:?} ran beside another task");
                }
                owner.settled.insert(*seq);
            }
            match event {
                Event::OwnershipAcquired { generation, applied } => {
                    owner = Some(Owner {
                        generation: *generation,
                        applied: *applied,
                        settled: BTreeSet::new(),
                        running: None,
                    });
                }
                Event::OwnershipLost => owner = None,
                Event::TaskArrived { .. }
                | Event::TaskFailed { .. }
                | Event::Executed { .. }
                | Event::Tick => {}
            }
        }
    }
}

/// What the property expects of the current owner.
struct Owner {
    generation: Generation,
    /// The log position it took over at.
    applied: Seq,
    /// Seqs it ran or failed, or that the log reported failed, since then.
    settled: BTreeSet<Seq>,
    /// The task it dispatched and has not seen this generation's result for.
    running: Option<Seq>,
}

#[derive(Clone, Debug)]
enum Op {
    Acquire {
        applied: u64,
    },
    Lost,
    /// A task's seq and how many seconds ago it entered the log.
    Arrive(u64, u64),
    Failed(u64, u64),
    /// The result of an earlier dispatch, picked by index; possibly late or
    /// repeated.
    Reply(usize),
    Tick,
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        1 => (0_u64..4).prop_map(|applied| Op::Acquire { applied }),
        1 => Just(Op::Lost),
        6 => (1_u64..10, waited()).prop_map(|(seq, waited)| Op::Arrive(seq, waited)),
        1 => (1_u64..10, waited()).prop_map(|(seq, waited)| Op::Failed(seq, waited)),
        4 => any::<usize>().prop_map(Op::Reply),
        1 => Just(Op::Tick),
    ]
}

/// Mostly fresh tasks, and some that waited past the gap timeout before the
/// owner saw them.
fn waited() -> impl Strategy<Value = u64> {
    prop_oneof![3 => Just(0_u64), 1 => 0_u64..900]
}

/// Applies `ops` in order, each `delay` seconds after the previous one.
/// `ticks[i]`, when set, sends an extra tick that percentage of the way
/// through op `i`'s delay. Returns each op's event with its effects; an
/// extra tick's effects are joined to the op after it.
fn run(start: Instant, ops: &[(u64, Op)], ticks: &[Option<u64>]) -> Vec<(Event, Vec<Effect>)> {
    let mut state = State::default();
    let mut generation = 0;
    let mut now = start;
    let mut dispatched = Vec::new();
    ops.iter()
        .enumerate()
        .map(|(index, (delay, op))| {
            let mut effects = Vec::new();
            if let Some(Some(percent)) = ticks.get(index) {
                let tick_at = now + Duration::from_secs(delay * percent / 100);
                effects.extend(model::on_event(&mut state, tick_at, Event::Tick));
            }
            now += Duration::from_secs(*delay);
            let event = match op {
                Op::Acquire { applied } => {
                    generation += 1;
                    acquire(generation, *applied)
                }
                Op::Lost => Event::OwnershipLost,
                Op::Arrive(seq, waited) => arrive_after(*seq, *waited),
                Op::Failed(seq, waited) => failed(*seq, *waited),
                Op::Reply(pick) => {
                    dispatched
                        .get(pick.checked_rem(dispatched.len()).unwrap_or(0))
                        .map_or(Event::Tick, |(seq, generation)| {
                            Event::Executed {
                                seq: *seq,
                                generation: *generation,
                            }
                        })
                }
                Op::Tick => Event::Tick,
            };
            effects.extend(model::on_event(&mut state, now, event));
            dispatched.extend(effects.iter().filter_map(|effect| {
                match effect {
                    Effect::Execute {
                        seq, generation, ..
                    } => Some((*seq, *generation)),
                    Effect::Fail { .. } => None,
                }
            }));
            (event, effects)
        })
        .collect()
}
