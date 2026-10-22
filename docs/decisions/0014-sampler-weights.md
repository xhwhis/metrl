# 0014. Sampler weights

## Context

RL alternates training and sampling on the latest weights. LoRA adapters are megabytes;
bases are gigabytes and must not be reloaded per adapter.

## Decision

- `save_weights_for_sampler` writes an immutable, manifest-complete adapter path that
  sampling sessions reference.
- Samplers keep bases resident and load adapters lazily from the object store into an
  LRU of 8.
- Sampler choice: adapter affinity, then load, with session stickiness.
- One worker may hold both roles over a shared base, or roles may run separately.

## Alternatives

- Direct trainer-to-sampler transfer: doesn't survive restarts or scale-out.
- NCCL or RDMA weight transfer: unneeded at LoRA sizes.
- KV-cache-aware routing as in Dynamo: more complexity than adapter affinity needs now.

## Consequences

Sampler reads go through the object store. Adapter churn beyond the LRU costs a reload.
