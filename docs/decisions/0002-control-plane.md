# 0002. Control plane

## Context

Multiple servers need a consistent, durable view of orgs, models, tasks, checkpoints,
and workers, and need to learn about changes made by other servers without polling.

## Decision

- Postgres is the control plane and the single source of truth.
- Servers learn about changes through logical replication: pgwire-replication plus our
  own pgoutput decoder. Each server owns slot `metrl_<server_id>`.
- Startup: create the slot, scan the published tables, then stream from the slot's
  start position. A change is acked only after it is applied to the in-memory view.
- Every write goes to Postgres first and reaches all servers, including the writer,
  through CDC. CDC rows carry the full record, so servers don't query back.
- One migration, `0001_init.sql`, edited in place.

## Alternatives

- etcd: a second coordination system next to the database we need anyway.
- `LISTEN/NOTIFY`: drops notifications across disconnects, limits payload size, and
  needs a periodic rescan to be correct.
- Polling: latency and load grow with the number of servers.

## Consequences

Change delivery is ordered and lossless. Each server holds a replication slot, so a dead
server's slot must be dropped or WAL accumulates. Tables that change often but don't need
replication (`sessions`) stay out of the publication.
