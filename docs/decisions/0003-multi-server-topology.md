# 0003. Multi-server topology

## Context

Metrl must scale out behind a load balancer: any server may receive any request, and
workers must keep serving when a server dies.

## Decision

- Servers are peers with no leader. Each one accepts every API request.
- A worker registers with any server; registration lands in Postgres and reaches every
  server through CDC, and every server dials the worker.
- Training requests are routed through Postgres to the owner of the model (0004).
- The only server-to-server RPC is `GetSampleResult`, because sample results live in the
  memory of the server that ran them. If that server is gone the answer is 410.

## Alternatives

- Pinning workers to one server: a server failure strands its workers.
- General request forwarding between servers: more coupling and failure modes.
- Sample results in the object store: removes the peer RPC but adds a write and read
  per sample.
- HTTP redirects to the owner: exposes internal addresses and depends on SDK behavior.
- Dynamo-style soft-state events between peers: Metrl already has an authoritative,
  ordered change stream in CDC.

## Consequences

Adding a server needs no coordination. Each worker accepts connections from every
server, so connection count grows as servers × workers.
