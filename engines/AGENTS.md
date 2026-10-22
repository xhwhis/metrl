# engines

One package per engine: dist `metrl-engine-<name>`, module `metrl.engines.<name>`,
selected by `METRL_ENGINE`.

- `metrl/` and `metrl/engines/` are PEP 420 namespaces. Never add `__init__.py` there.
- Implement the `Trainer` and `Sampler` Protocols from `metrl.worker`. Methods are
  `async`; blocking compute goes through `ComputeThread`.
- Every call carries `model_id`. Adapters from different orgs may share one process, so
  keep no global mutable state and never let one model's state leak into another.
- Declare bases in `worker.toml` `[[bases]]` (`name`, `path`, `roles`). Report a real
  `revision`; servers reject mismatched revisions for the same name.
- `mock` has no dependencies and must match the observable semantics of real engines.
  Faults are injected only through `METRL_MOCK_FAULT`.
- When a Protocol changes, update every engine in the same change.
