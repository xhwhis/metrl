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
  Forward-backward inputs and outputs keep the Tinker protobuf format, so the server
  stores request bodies as sent and serves outputs as written.
- Inside a worker, tensors cross as `TensorData` through the buffer protocol.
- A task that fails is an `Execute` response with a user or server category. gRPC
  errors mean the call did not run as asked: `ABORTED` for a newer generation or a
  later seq, `FAILED_PRECONDITION` for an unbound generation or another worker at the
  address, and `INVALID_ARGUMENT` for a Bind the engine rejects.

## Alternatives

- protox (pure-Rust protoc): one more dependency where the system protoc works.
- Shared files between runtime and engine: extra I/O and lifecycle for data that is
  already in memory.
- rust-numpy: an extra layer the buffer protocol makes unnecessary.

## Consequences

Builds need `protoc` installed. Tensor transfer inside a worker is zero-copy.
