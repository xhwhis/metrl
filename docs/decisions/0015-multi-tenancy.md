# 0015. Multi-tenancy and admission

## Context

Several orgs share servers and workers. One org must never see another's data, and one
org's load must not starve the others.

## Decision

- The org is the isolation boundary: `org_id` on every tenant row and query, object keys
  under `orgs/{org_id}/`, and 404 for cross-org access. Org ids match
  `[A-Za-z0-9_-]+`, so no id reaches outside its key prefix. Shared cluster rows (`workers`)
  have no `org_id`; the CDC scan reads every org's key hashes to authenticate requests.
- API keys are static and belong to an owner in an org; Postgres stores their sha256
  hashes, and servers cache them through CDC.
- Requests are accepted, then queued. Limits: 413 above 256 MiB for `forward_backward`
  and 2 MiB elsewhere, checked after the API key; 429 on model, pending, sampling, or
  storage quotas; 408 `paused_capacity` or `paused_rate_limit` while queued.
  A model's pending limit never rejects the seq it runs next, since every waiting task
  may sit behind it. `/client/config` turns off `parallel_fwdbwd_chunks`: the SDK would
  otherwise send a split call's first chunk last, and 64 later chunks would leave it
  rejected for good.
- Workers are shared and scheduled weighted round-robin per org.

## Alternatives

- Keys in config files: no revocation without restarts.
- Projects as the boundary: in Tinker a project is a grouping inside an org.
- Dedicated workers per org by default: loses shared bases; can come later with labels.

## Consequences

Every tenant query and key builder must take `org_id`. Fairness lives in the scheduler's
`cluster/`.
