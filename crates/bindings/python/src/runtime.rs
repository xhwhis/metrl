use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tokio_util::sync::CancellationToken;

use crate::sampler::PythonSampler;

/// Serves the worker gRPC API on `addr` until cancelled.
#[pyfunction]
pub(crate) async fn serve(
    worker_id: String, base_model: String, addr: String, engine: Py<PyAny>,
) -> PyResult<()> {
    use tracing::Instrument;
    let locals = Python::attach(pyo3_async_runtimes::tokio::get_current_locals)?;
    let addr = addr
        .parse()
        .map_err(|error| PyValueError::new_err(format!("invalid address {addr:?}: {error}")))?;
    // The guard stops the server when Python cancels and drops this future.
    let shutdown = CancellationToken::new();
    let _guard = shutdown.clone().drop_guard();
    pyo3_async_runtimes::tokio::get_runtime()
        .spawn(
            metrl_worker::serve(
                worker_id,
                base_model,
                addr,
                PythonSampler { engine, locals },
                shutdown.cancelled_owned(),
            )
            .instrument(tracing::info_span!("worker")),
        )
        .await
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
        .map_err(|report| PyRuntimeError::new_err(format!("{report:?}")))
}
