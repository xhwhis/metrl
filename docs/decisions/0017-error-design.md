# 0017. Error design

## Context

Errors serve two readers. Callers need to know what to do (retry, give up, return 4xx);
operators need to know what happened, with business context. Forwarding source errors
with `?` serves neither.

## Decision

- `ErrorKind` is defined by the action it implies: `InvalidArgument` (400), `NotFound`
  (404, including cross-org), `Expired` (400), `Gone` (410), `Overloaded` (429 with
  retry-after), `TooLarge` (413), `Unavailable` (503), `Fenced` (drop silently),
  `Internal` (500).
- Errors use `error-stack` with `default-features = false, features = ["std"]`: no
  backtraces, no span traces.
- Each crate has one error context `{kind, message}` and returns
  `Result<T, Report<Context>>`.
- Crossing a crate boundary requires `change_context` and attaches business context such
  as `model_id` and `seq`.
- Only `api` and the worker RPC map kinds to protocol statuses. The full report is logged
  inside the request span.
- No `anyhow`, no `#[from]`. `thiserror` only derives `Display`.

## Alternatives

- Source-shaped enums with `#[from]`: say where an error came from, not what to do.
- `anyhow` or `eyre`: type erasure doesn't force boundary context.
- `tracing-error`: span traces only, and stale.
- `exn`: viable, but `error-stack` is more established.
- `snafu`: heavier, variant-per-source style.

## Consequences

Every boundary needs a deliberate `change_context`, which is more code but makes reports
readable. Python exceptions from engines are mapped to kinds in `metrl-py`.
