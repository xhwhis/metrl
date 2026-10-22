# 0006. Pure scheduler

## Context

Ordering, gaps, timeouts, replay, placement, and fairness are the subtle parts of
Metrl. Mixing them with I/O makes them hard to test and impossible to replay
deterministically.

## Decision

- The scheduler is sans-I/O: `on_event(&mut State, now, Event) -> Vec<Effect>` and
  `next_deadline(&State)`.
- It has two parts: `model/` for the per-model task log and `cluster/` for placement
  and fairness.
- Model actors in `server` feed events, execute effects, and own all I/O.
- `debug_assert!(state.invariants())` runs after every event.

## Alternatives

- Scheduler doing Postgres, RPC, or object I/O: correctness depends on timing.
- Reading the wall clock inside: timers can't be tested without sleeping.

## Consequences

Scheduler behavior is tested with events alone, including property tests over
interleavings. New behavior must be expressed as new events or effects.
