//! The `metrl._core` extension module.

mod base;
mod config;
mod engine;
mod runtime;
mod sampler;
mod tensor;
mod trainer;

use pyo3::prelude::*;

#[pymodule]
mod _core {
    #[pymodule_export]
    use crate::base::Base;
    #[pymodule_export]
    use crate::config::{BaseConfig, Config};
    #[pymodule_export]
    use crate::runtime::serve;
    #[pymodule_export]
    use crate::sampler::{Adapter, SampleResponse, SampledSequence, SamplingParams};
    #[pymodule_export]
    use crate::tensor::TensorData;
    #[pymodule_export]
    use crate::trainer::{AdamParams, AdapterWeights, Datum, ForwardBackwardOutput};
}
