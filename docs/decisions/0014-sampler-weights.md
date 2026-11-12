# 0014. Sampler weights

## Context

RL alternates training and sampling on the latest weights. LoRA adapters are megabytes;
bases are gigabytes and must not be reloaded per adapter.

## Decision

- `save_weights_for_sampler` writes an immutable, manifest-complete adapter path that
  sampling sessions reference.
- Samplers keep bases resident and load adapters lazily from the object store into an
  LRU of 8. The engine owns the cache: the worker hands it an `Adapter` whose `key` is
  the checkpoint's immutable prefix and whose `load()` downloads and decodes the files
  in Rust, so engines never touch storage. Requests that miss while the adapter loads
  wait for that load, so a burst of samples after a save downloads it once.
- Sampler choice: rendezvous hashing over the workers that serve the base revision,
  keyed by the checkpoint id, or by the sampling session for a bare base. Every server
  picks the same worker without coordinating, and a worker joining or leaving moves only
  the keys it ranks first for. Load-aware placement comes later.
- One worker may hold both roles over a shared base, or roles may run separately.

## Alternatives

- Direct trainer-to-sampler transfer: doesn't survive restarts or scale-out.
- NCCL or RDMA weight transfer: unneeded at LoRA sizes.
- KV-cache-aware routing as in Dynamo: more complexity than adapter affinity needs now.
- Modulo hashing over a sorted worker list: a worker joining or leaving moves almost
  every adapter, and every sampler reloads.
- Rust passing decoded weights on every call: a download per request even on a cache
  hit.
- A cache in Rust: the engine still holds the weights it applies, so the memory would
  count twice.

## Consequences

Sampler reads go through the object store. Adapter churn beyond the LRU costs a reload.
