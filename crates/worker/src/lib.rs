//! Worker runtime: gRPC service, object I/O, and the engine seam.

mod runtime;

pub use runtime::{Sampler, WorkerError, serve};
