# 0011. Python binding

## Context

The worker runtime is Rust but engines are Python, and Python code needs accurate types
for the native module.

## Decision

- `crates/bindings/python` (`metrl-py`) builds the `metrl` wheel with module
  `metrl._core`.
- Async Rust functions are exposed as `#[pyfunction] async fn` and awaited from Python.
  Python cancellation drops the Rust future, which must stop its work.
- Stubs come from `maturin develop --generate-stubs` (pyo3 `experimental-inspect`) and
  are not committed.

## Alternatives

- A hand-written, committed `_core.pyi` as in Dynamo: drifts from the Rust signatures
  with nothing to catch it.
- Committing generated stubs: the Rust signature already is the source.

## Consequences

`ty` needs a local build first (`just setup`). Signatures that pyo3 can't infer are
annotated in `#[pyo3(signature = ...)]`.
