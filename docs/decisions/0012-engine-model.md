# 0012. Engine model

## Context

Engines wrap ML libraries that block for seconds, while the worker must keep serving
RPCs, heartbeats, and cancellation. Local development happens on Macs.

## Decision

- An engine package exports `async load(settings, bases, compute) -> Engine`. It gets
  its `[engine.<name>]` table untouched, the `[[bases]]` entries (`name`, `path`), and
  the worker's `ComputeThread`, and returns the loaded `Base` for each configured base
  (revision, arch, tokenizer, limits) with a `Trainer` and a `Sampler`.
- `Trainer` and `Sampler` are async Protocols whose methods follow Tinker names:
  `create_model`, `forward`, `forward_backward`, `optim_step`,
  `save_weights_for_sampler`, and `sample`. Every call carries `model_id`.
- Blocking compute runs on the one `ComputeThread` per worker via
  `await compute.run(fn)`, in submission order. A cancelled await leaves the call
  running to the end, so engine state never sees half a call.
- Engines exchange tensors and metadata only: `save_weights_for_sampler` returns
  `AdapterWeights` (rank, alpha, target modules, and `TensorData` by PEFT name), and the
  worker's Rust runtime encodes and stores them; `sample` gets an `Adapter` whose
  `load()` returns them decoded. Python never touches storage.
- An engine raises `ValueError` for a request it cannot run as given; the message
  reaches the user. Any other exception is a server error, and its text stays in the
  worker log.
- Python owns the event loop (uvloop).
- One package per engine under `engines/`, imported as `metrl.engines.<name>` through
  PEP 420 namespaces and selected by `[engine] name` in `worker.toml`.
- First engines: `mlx` for Apple silicon and `mock`, a deterministic engine with no
  dependencies whose outputs carry a state hash chain.

## Alternatives

- Synchronous engine API with a host-managed thread: two calling conventions, and
  samplers that batch must be async anyway.
- Rust owning the Python loop: fights the Python async ecosystem.
- All engines in one package: every worker installs every ML stack.
- Starting with a real engine only: fast deterministic E2E and chaos tests need a mock.
- Rust parsing engine settings: every engine option would need a Rust field.

## Consequences

Adding an engine is adding a package. Engines can't use module-level state because
models from different orgs share one process. Settings overridden through `METRL_`
variables arrive as strings, so engines parse their own types.
