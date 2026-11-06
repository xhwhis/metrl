use std::ffi::c_int;

use bytes::Bytes;
use metrl_worker::{DType, Tensor};
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyMemoryView;

/// A dense little-endian tensor. Its bytes are shared with Rust, never
/// copied; read them through the buffer protocol, for example
/// `memoryview(tensor).cast("f")`.
#[pyclass(frozen, module = "metrl._core")]
pub(crate) struct TensorData(pub(crate) Tensor);

#[pymethods]
impl TensorData {
    /// `dtype` is `"float32"` or `"int64"`.
    #[new]
    fn new(data: PyBackedBytes, dtype: &str, shape: Vec<usize>) -> PyResult<Self> {
        let dtype = match dtype {
            "float32" => DType::Float32,
            "int64" => DType::Int64,
            other => return Err(PyValueError::new_err(format!("unsupported dtype {other}"))),
        };
        Tensor::new(Bytes::from_owner(data), dtype, shape)
            .map(Self)
            .map_err(|report| PyValueError::new_err(report.current_context().to_string()))
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        match self.0.dtype() {
            DType::Float32 => "float32",
            DType::Int64 => "int64",
        }
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.0.shape().to_vec()
    }

    /// The bytes, as a read-only memoryview.
    fn view<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyMemoryView>> {
        PyMemoryView::from(slf.as_any())
    }

    unsafe fn __getbuffer__(
        slf: &Bound<'_, Self>, view: *mut ffi::Py_buffer, flags: c_int,
    ) -> PyResult<()> {
        let data = slf.get().0.data();
        // SAFETY: the buffer keeps `slf` alive, and `Bytes` never moves or
        // mutates its data while a reference to it exists. The view is
        // read-only, so the cast to a mutable pointer is never written
        // through.
        let filled = unsafe {
            ffi::PyBuffer_FillInfo(
                view,
                slf.as_ptr(),
                data.as_ptr().cast_mut().cast(),
                data.len() as ffi::Py_ssize_t,
                1,
                flags,
            )
        };
        if filled == 0 {
            Ok(())
        } else {
            Err(PyErr::fetch(slf.py()))
        }
    }
}
