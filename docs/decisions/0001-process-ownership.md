# 0001. Process ownership

## Context

An RL service has two kinds of state with different failure modes: the task log and
history, which must survive any process, and model weights and optimizer state, which
live in accelerator memory. Mixing them in one language or process makes recovery and
fencing depend on ML code.

## Decision

- Rust servers own the API, scheduling, tasks, and execution history.
- Python engines own compute and memory state, and nothing else.
- Only the server talks to Postgres. Workers, engines, and the SDK never connect to it.
- A worker is a Rust runtime embedding one Python engine; the runtime does gRPC, object
  I/O, and fencing, so engines only see tensors in memory.

## Alternatives

- Python orchestration: weaker typing and concurrency for the control plane; fencing
  would depend on engine code.
- Workers writing to Postgres: spreads the control boundary across every worker and
  makes stale-owner writes possible.

## Consequences

Engines stay small and replaceable. Every state change goes through a server, so the
server needs gRPC to every worker and must carry results back to Postgres itself.
