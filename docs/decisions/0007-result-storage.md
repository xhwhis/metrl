# 0007. Result storage

## Context

Results range from a few numbers to large tensors, and sampling produces many short-lived
results. Writing large tensors into Postgres bloats it; persisting every sample is wasted
I/O for a result the SDK can resubmit.

## Decision

- Small training results go in `tasks.result jsonb`, with no size cutoff for them.
- Task inputs and fwd/bwd outputs go to the object store under `orgs/{org}/tasks/...`.
- Sample results stay in the receiving server's memory with a 1h TTL and a byte cap.
- Session heartbeats are kept in memory and flushed to Postgres every 30s.
- Task outputs expire after 1h and then return 400. Sweep runs under a Postgres
  advisory lock.

## Alternatives

- Everything in Postgres: table bloat and slow vacuum on tensor payloads.
- Everything in the object store: a write and read per sample.

## Consequences

Postgres stays small. Sample results are lost when their server dies, so the SDK gets
410 and resubmits (0003).
