use std::path::Path;

use metrl_config::WorkerConfig;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value;

/// Worker configuration loaded by Rust from `worker.toml` and `METRL_`
/// environment overrides.
#[pyclass(frozen, module = "metrl._core")]
pub(crate) struct Config(pub(crate) WorkerConfig);

#[pymethods]
impl Config {
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        metrl_config::load(Path::new(path))
            .map(Self)
            .map_err(|report| PyValueError::new_err(format!("{report:?}")))
    }

    /// Name of the `metrl.engines.<name>` package to load.
    #[getter]
    fn engine(&self) -> &str {
        &self.0.engine.name
    }

    /// The selected engine's `[engine.<name>]` table, empty when absent.
    /// Values set through the environment arrive as strings.
    #[pyo3(signature = () -> "dict[str, object]")]
    fn engine_settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        if let Some(section) = self.0.engine.sections.get(&self.0.engine.name) {
            for (key, value) in section {
                dict.set_item(key, to_python(py, value)?)?;
            }
        }
        Ok(dict)
    }

    /// The `[[bases]]` entries, in file order.
    #[getter]
    fn bases(&self) -> Vec<BaseConfig> {
        self.0.bases.iter().cloned().map(BaseConfig).collect()
    }
}

/// A `[[bases]]` entry of `worker.toml`.
#[pyclass(frozen, module = "metrl._core")]
pub(crate) struct BaseConfig(metrl_config::BaseConfig);

#[pymethods]
impl BaseConfig {
    /// The name clients use.
    #[getter]
    fn name(&self) -> &str {
        &self.0.name
    }

    /// What to load, such as a Hugging Face repository or a local directory;
    /// `None` loads `name`.
    #[getter]
    fn path(&self) -> Option<&str> {
        self.0.path.as_deref()
    }
}

/// TOML integers are `i64`, so every other number is a float.
fn to_python<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::IntoPyObjectExt;
    match value {
        Value::Null => Ok(py.None().into_bound(py)),
        Value::Bool(value) => value.into_bound_py_any(py),
        Value::Number(number) => {
            match number.as_i64() {
                Some(value) => value.into_bound_py_any(py),
                None => number.as_f64().into_bound_py_any(py),
            }
        }
        Value::String(value) => value.into_bound_py_any(py),
        Value::Array(items) => {
            let items = items
                .iter()
                .map(|item| to_python(py, item))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, items)?.into_any())
        }
        Value::Object(fields) => {
            let dict = PyDict::new(py);
            for (key, value) in fields {
                dict.set_item(key, to_python(py, value)?)?;
            }
            Ok(dict.into_any())
        }
    }
}
