# 0012. Engine model

## Context

Engines wrap ML libraries that block for seconds, while the worker must keep serving
RPCs, heartbeats, and cancellation. Local development happens on Macs.

## Decision

- Engines implement async `Trainer` and `Sampler` Protocols. Blocking compute runs on a
  shared `ComputeThread` via `await compute.run(fn)`.
- Python owns the event loop (uvloop).
- One package per engine under `engines/`, imported as `metrl.engines.<name>` through
  PEP 420 namespaces and selected by `METRL_ENGINE`.
- First engines: `mlx` for Apple silicon and `mock`, a deterministic engine with no
  dependencies whose outputs carry a state hash chain.

## Alternatives

- Synchronous engine API: blocks RPC handling during compute.
- Rust owning the Python loop: fights the Python async ecosystem.
- All engines in one package: every worker installs every ML stack.
- Starting with a real engine only: fast deterministic E2E and chaos tests need a mock.

## Consequences

Adding an engine is adding a package. Engines can't use module-level state because
models from different orgs share one process.
