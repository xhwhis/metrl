//! The `metrl._core` extension module.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tokio_util::sync::CancellationToken;

/// Serves the worker gRPC API on `addr` until cancelled.
#[pyfunction]
async fn serve(worker_id: String, addr: String) -> PyResult<()> {
    let addr = addr
        .parse()
        .map_err(|error| PyValueError::new_err(format!("invalid address {addr:?}: {error}")))?;
    // Cancelling the Python task drops this future, and the guard stops the
    // server.
    let shutdown = CancellationToken::new();
    let _guard = shutdown.clone().drop_guard();
    pyo3_async_runtimes::tokio::get_runtime()
        .spawn(metrl_worker::serve(
            worker_id,
            addr,
            shutdown.cancelled_owned(),
        ))
        .await
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
        .map_err(|report| PyRuntimeError::new_err(format!("{report:?}")))
}

#[pymodule]
mod _core {
    #[pymodule_export]
    use super::serve;
}
