# 0004. Leases and fencing

## Context

Each model's task log must be executed by exactly one server at a time, but servers
crash, pause, and lose network. A server that believes it still owns a model must not
corrupt state after another server takes over.

## Decision

- A model has one owner server, held by a 10s lease in Postgres, renewed every 3s and
  timed by Postgres `now()`. Each renewal is one statement that extends the server's
  leases and takes every free or expired one.
- A server that has not sent a renewal for 7s stops its model actors. A renewal sent
  before then gives up after 2s, so it lands before the lease it extends lapses.
- Taking over increments `models.generation`. Generation is the only fencing token.
- Binds, executes, and commits carry the generation; workers and Postgres reject any
  older generation. A commit locks the model row with `FOR NO KEY UPDATE`, so it
  serializes with takeovers but not with task inserts.
- A worker keeps each model's last task response. When a new owner repeats that seq,
  the worker returns the response instead of running the task again.

## Alternatives

- Worker-held leases: ownership belongs to the task log, which servers manage.
- etcd or a consensus library for leader election: a second coordination plane (0002).
- Lease time alone without fencing: a paused owner can still write after expiry.

## Consequences

Failover takes up to one lease period. Every write path must carry and check the
generation, and work rejected for an old generation is dropped silently. A commit that
Postgres received before the fence can still land after it; the worker's kept response
makes the new owner's repeat of that seq harmless.
