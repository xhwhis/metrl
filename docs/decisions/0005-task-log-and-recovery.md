# 0005. Task log and recovery

## Context

Training is a sequence of stateful steps on one model. After a crash the new owner must
continue without losing or repeating steps, and SDK retries must not double-apply.

## Decision

- The task sequence is the SDK `seq_id`. Tasks are inserted with
  `ON CONFLICT (model_id, seq) DO NOTHING`; a retry returns the existing `request_id`,
  and a retry whose input differs gets 400. Forward-backward inputs compare as
  re-encoded protobufs with sorted maps.
- Only `seq == applied + 1` is dispatched, so a model runs one task at a time.
- A gap is timed from when the earliest task behind the missing `applied + 1` entered
  the log (`tasks.created_at`), so a takeover does not restart it. After 10 minutes
  every waiting task fails with 400
  "missing seq_id N", and so does each task that arrives behind the same gap later.
  The missing seq itself is never skipped: when it arrives it runs, and the log moves
  past the failed tasks after it.
- A failed task is a no-op: `applied` advances and replay skips it.
- Ownership changes reset the scheduler: the new owner starts from the `applied`
  recorded in Postgres and rescans the log, so it never trusts another generation's
  in-memory state. It dispatches nothing until its view holds every commit made
  before the takeover, so a task the previous owner failed is never run.
- The worker keeps the last task's response from the moment the engine returns, and
  never abandons an engine call: a timed-out or repeated call waits for the model and
  gets that response. A transient object store failure returns `UNAVAILABLE`: on the
  input it keeps nothing, and on the output a repeated call writes it again. A
  missing or forbidden object fails the task instead, so it cannot block the model.
- The model's trainer state stays on the worker recorded in `models.worker_id`, and a
  new owner binds the same worker. A model moves to another worker only before any
  task has applied.
- Recovery loads the latest checkpoint and replays later tasks. It is semantically
  equivalent, not bit-exact. Until checkpoints exist, a model whose worker is gone
  fails every later task with a server error.
- Auto checkpoints every 20 optimizer steps and on worker SIGTERM; keep the latest 2.

## Alternatives

- Server-assigned sequence numbers: break SDK retries and idempotency.
- Snapshot after every step: too much I/O for large weights.

## Consequences

Replay time is bounded by the checkpoint interval. Task inputs must stay in the object
store until a later checkpoint covers them.
