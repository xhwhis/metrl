//! Worker runtime: gRPC service, object I/O, and the engine seam.

mod adapter;
mod engine;
mod error;
mod runtime;
mod service;
mod tinker;
#[allow(clippy::doc_markdown, clippy::trivially_copy_pass_by_ref)]
pub mod tinker_public;
mod training;

pub use engine::{
    Adapter, AdapterWeights, DType, Datum, ForwardBackward, ForwardBackwardOutput, Sampler, Tensor,
    Trainer,
};
pub use error::WorkerError;
pub use runtime::{Identity, serve};
