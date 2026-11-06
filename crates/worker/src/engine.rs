//! The seam between the worker runtime and the engine that computes. Trainer
//! data crosses it as plain tensors, decoded from and encoded to the Tinker
//! wire format by the runtime.

use std::collections::BTreeMap;

use bytes::Bytes;
use error_stack::Report;
use metrl_proto::worker::{SampleRequest, SampleResponse};
use metrl_types::{AdamParams, ModelId};

use crate::WorkerError;

pub trait Sampler: Send + Sync + 'static {
    fn sample(
        &self, request: SampleRequest,
    ) -> impl Future<Output = Result<SampleResponse, Report<WorkerError>>> + Send;
}

/// Holds each model's trainer state. Calls for one model never overlap.
pub trait Trainer: Send + Sync + 'static {
    /// Creates the model's fresh trainer state.
    fn create(
        &self, model: ModelId, base: &str, lora_rank: u32, seed: u64,
    ) -> impl Future<Output = Result<(), Report<WorkerError>>> + Send;

    fn forward_backward(
        &self, model: ModelId, request: ForwardBackward,
    ) -> impl Future<Output = Result<ForwardBackwardOutput, Report<WorkerError>>> + Send;

    /// Applies the gradients accumulated since the last step.
    fn optim_step(
        &self, model: ModelId, params: AdamParams,
    ) -> impl Future<Output = Result<BTreeMap<String, f64>, Report<WorkerError>>> + Send;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    Float32,
    Int64,
}

impl DType {
    pub(crate) fn size(self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Int64 => 8,
        }
    }
}

/// A dense little-endian tensor whose data fills its shape.
#[derive(Clone, Debug)]
pub struct Tensor {
    data: Bytes,
    dtype: DType,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Bytes, dtype: DType, shape: Vec<usize>) -> Result<Self, Report<WorkerError>> {
        let bytes = shape
            .iter()
            .try_fold(dtype.size(), |bytes, dim| bytes.checked_mul(*dim));
        if bytes != Some(data.len()) {
            return Err(Report::new(WorkerError::Invalid(format!(
                "{} bytes do not fill a {dtype:?} tensor of shape {shape:?}",
                data.len()
            ))));
        }
        Ok(Self { data, dtype, shape })
    }

    pub fn data(&self) -> &Bytes {
        &self.data
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

/// Tinker `Datum`: the model input's tokens and the loss function's inputs.
#[derive(Clone, Debug)]
pub struct Datum {
    pub tokens: Vec<i32>,
    pub loss_fn_inputs: BTreeMap<String, Tensor>,
}

/// Tinker `ForwardBackwardInput`.
#[derive(Clone, Debug)]
pub struct ForwardBackward {
    pub data: Vec<Datum>,
    pub loss_fn: String,
    /// Computes the outputs without accumulating gradients.
    pub forward_only: bool,
}

/// Tinker `ForwardBackwardOutput`: one output map per input datum.
#[derive(Clone, Debug)]
pub struct ForwardBackwardOutput {
    pub loss_fn_outputs: Vec<BTreeMap<String, Tensor>>,
    /// Each key is `name:reduction`, which tells the SDK how to combine it
    /// across chunks.
    pub metrics: BTreeMap<String, f64>,
}
