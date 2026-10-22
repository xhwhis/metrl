# 0005. Task log and recovery

## Context

Training is a sequence of stateful steps on one model. After a crash the new owner must
continue without losing or repeating steps, and SDK retries must not double-apply.

## Decision

- The task sequence is the SDK `seq_id`. Tasks are inserted with
  `ON CONFLICT (model_id, seq) DO NOTHING`; a retry returns the existing `request_id`.
- Only `seq == applied + 1` is dispatched. A gap older than 10 minutes fails waiting
  tasks with 400.
- A failed task is a no-op: `applied` advances and replay skips it.
- Recovery loads the latest checkpoint and replays later tasks. It is semantically
  equivalent, not bit-exact.
- Auto checkpoints every 20 optimizer steps and on worker SIGTERM; keep the latest 2.

## Alternatives

- Server-assigned sequence numbers: break SDK retries and idempotency.
- Snapshot after every step: too much I/O for large weights.

## Consequences

Replay time is bounded by the checkpoint interval. Task inputs must stay in the object
store until a later checkpoint covers them.
