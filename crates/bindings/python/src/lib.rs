//! The `metrl._core` extension module.

mod runtime;
mod sampler;
mod tensor;
mod trainer;

use pyo3::prelude::*;

#[pymodule]
mod _core {
    #[pymodule_export]
    use crate::runtime::{Config, serve};
    #[pymodule_export]
    use crate::tensor::TensorData;
    #[pymodule_export]
    use crate::trainer::{AdamParams, Datum, ForwardBackwardOutput};
}
