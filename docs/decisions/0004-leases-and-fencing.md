# 0004. Leases and fencing

## Context

Each model's task log must be executed by exactly one server at a time, but servers
crash, pause, and lose network. A server that believes it still owns a model must not
corrupt state after another server takes over.

## Decision

- A model has one owner server, held by a 10s lease in Postgres, renewed every 3s and
  timed by Postgres `now()`.
- Taking over increments `models.generation`. Generation is the only fencing token.
- Binds, executes, and commits carry the generation; workers and Postgres reject any
  older generation.

## Alternatives

- Worker-held leases: ownership belongs to the task log, which servers manage.
- etcd or a consensus library for leader election: a second coordination plane (0002).
- Lease time alone without fencing: a paused owner can still write after expiry.

## Consequences

Failover takes up to one lease period. Every write path must carry and check the
generation, and work rejected for an old generation is dropped silently.
