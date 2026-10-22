# 0015. Multi-tenancy and admission

## Context

Several orgs share servers and workers. One org must never see another's data, and one
org's load must not starve the others.

## Decision

- The org is the isolation boundary: `org_id` on every row and query, object keys under
  `orgs/{org_id}/`, and 404 for cross-org access.
- API keys are static and belong to an owner in an org; Postgres stores their sha256
  hashes, and servers cache them through CDC.
- Requests are accepted, then queued. Limits: 413 above 256 MiB; 429 on model, pending,
  sampling, or storage quotas; 408 `paused_capacity` or `paused_rate_limit` while queued.
- Workers are shared and scheduled weighted round-robin per org.

## Alternatives

- Keys in config files: no revocation without restarts.
- Projects as the boundary: in Tinker a project is a grouping inside an org.
- Dedicated workers per org by default: loses shared bases; can come later with labels.

## Consequences

Every query and key builder must take `org_id`. Fairness lives in the scheduler's
`cluster/`.
