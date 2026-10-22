# 0010. Worker protocol

## Context

Servers and workers exchange control messages and tensors. Inside a worker, tensors
cross from Rust to Python.

## Decision

- Server, worker, and peer RPCs are gRPC with protobuf. All `.proto` files live flat in
  `proto/`; each consuming crate compiles them with `tonic-prost-build` and the system
  `protoc`. Generated code is not committed.
- Every call carries `ctx`: `request_id`, `model_id`, `seq`, `trace_id`, `deadline`,
  and cancellation.
- Large payloads travel through the object store by key; RPCs carry keys, not tensors.
- Inside a worker, tensors cross as `TensorData` through the buffer protocol.

## Alternatives

- protox (pure-Rust protoc): one more dependency where the system protoc works.
- Shared files between runtime and engine: extra I/O and lifecycle for data that is
  already in memory.
- rust-numpy: an extra layer the buffer protocol makes unnecessary.

## Consequences

Builds need `protoc` installed. Tensor transfer inside a worker is zero-copy.
