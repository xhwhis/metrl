//! The `metrl._core` extension module.

mod runtime;
mod sampler;

use pyo3::prelude::*;

#[pymodule]
mod _core {
    #[pymodule_export]
    use crate::runtime::serve;
}
