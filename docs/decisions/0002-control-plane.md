# 0002. Control plane

## Context

Multiple servers need a consistent, durable view of orgs, models, tasks, checkpoints,
and workers, and need to learn about changes made by other servers without polling.

## Decision

- Postgres is the control plane and the single source of truth.
- Servers learn about changes through logical replication: pgwire-replication plus our
  own decoder for binary pgoutput. Each connection attempt uses a new temporary slot
  `metrl_<server_id>_<attempt>`, because after a link is lost without a reset Postgres
  holds the old slot until `wal_sender_timeout`. Replication connections set it to 10s
  so that slot goes away soon. It only applies once streaming starts; a session lost
  while idle in the slot's snapshot transaction, during the scan, holds the slot and its
  `catalog_xmin` until `idle_in_transaction_session_timeout`, set to 30s, longer than
  the 15s scan timeout. A session lost before creating the slot holds a walsender until
  `idle_session_timeout`, also 30s. Creating the slot waits for every transaction
  running in the cluster to end without reading the socket, so `lock_timeout` is 10s:
  otherwise each attempt given up during a long transaction would leave a walsender and
  its slot until that transaction ends, and retries would exhaust `max_wal_senders`.
- Startup and every reconnect: create the slot exporting a snapshot, scan the published
  tables in that snapshot, then stream from the slot. A transaction is applied to the
  in-memory view at its commit, then acked. Keepalives ack the server's WAL position
  too, because transactions that touch no published table send no commit.
- A link that freezes without a reset raises no error, so each server writes a
  non-transactional logical message every 5s (`pg_logical_emit_message` with `flush`,
  Postgres 17+) and reads its own back from its stream. Messages decode in WAL order, so
  reading one back proves every earlier commit is applied. A stream that reads none of
  its own for 15s is dropped and resynced. Any role can write a logical message, so a
  heartbeat that claims a time in the future is ignored.
- Heartbeats and scans use their own two-connection pool, so request queries
  cannot starve or poison them. A frozen link also leaves pooled connections open but
  silent, so both pools verify each checkout with an empty query and bound waiting,
  connecting, and verifying (5s, 5s, 2s); a connection that fails is dropped, not
  reused. Neither connecting, authenticating, nor `CREATE_REPLICATION_SLOT` times out
  by itself, so creating the slot and the scan each give up after 15s. Every run
  ends by aborting its replication client: a worker blocked on a frozen link ignores a
  graceful stop, and the temporary slot needs none.
- The view is fresh as of its last scan or heartbeat read back, and keeps its last state
  meanwhile. Once that is more than 30s ago, the server answers authenticated requests
  and `/api/v1/healthz` with 503, because the view may still hold revoked keys and
  removed workers.
- Every write goes to Postgres first and reaches all servers, including the writer,
  through CDC. CDC rows carry the full record, so servers don't query back.
- One schema file, `sql/schema.sql`, edited in place. Every statement lives in
  `crates/meta/sql/`.

## Alternatives

- etcd: a second coordination system next to the database we need anyway.
- `LISTEN/NOTIFY`: drops notifications across disconnects, limits payload size, and
  needs a periodic rescan to be correct.
- TCP keepalive or `tcp_user_timeout` on the replication socket: pgwire-replication
  exposes neither, and both detect only a dead peer, not a stream that fell behind.
- Persistent slots resumed after reconnect: avoid rescans, but a crashed server's slot
  retains WAL until someone drops it, and resuming needs the view to survive restarts.
- Polling: latency and load grow with the number of servers.

## Consequences

Change delivery is ordered and lossless while connected; a reconnect rescans instead of
resuming, so the view never depends on slot state that outlived its server. A dead
server's slot disappears with its connection, so no WAL accumulates. A rescan reads the
published tables once, so tables that change often but don't need replication stay out
of the publication: `sessions`, and `models`, whose lease columns every owner rewrites
every 3s. Owners learn their models from lease renewals instead.
