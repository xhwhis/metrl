use error_stack::Report;
use metrl_worker::WorkerError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_async_runtimes::TaskLocals;

/// A Python trainer or sampler whose coroutines run on the worker's event
/// loop.
pub(crate) struct Engine {
    pub(crate) object: Py<PyAny>,
    pub(crate) locals: TaskLocals,
}

impl Engine {
    /// Awaits the coroutine that `call` starts on the engine object.
    pub(crate) async fn call(
        &self, call: impl FnOnce(Python<'_>, &Py<PyAny>) -> PyResult<Py<PyAny>> + Send,
    ) -> Result<Py<PyAny>, Report<WorkerError>> {
        let future = Python::attach(|py| {
            let coroutine = call(py, &self.object)?;
            pyo3_async_runtimes::into_future_with_locals(&self.locals, coroutine.into_bound(py))
        })
        .map_err(engine_error)?;
        future.await.map_err(engine_error)
    }
}

/// An engine raises `ValueError` for a request it cannot run as given; any
/// other exception is the engine's own failure.
pub(crate) fn engine_error(error: PyErr) -> Report<WorkerError> {
    let invalid = Python::attach(|py| {
        error
            .is_instance_of::<PyValueError>(py)
            .then(|| error.value(py).to_string())
    });
    match invalid {
        Some(message) => Report::new(error).change_context(WorkerError::Invalid(message)),
        None => Report::new(error).change_context(WorkerError::Failed),
    }
}
