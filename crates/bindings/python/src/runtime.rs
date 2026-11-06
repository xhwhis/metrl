use std::path::Path;

use metrl_config::WorkerConfig;
use metrl_proto::worker;
use metrl_types::{Role, WorkerId};
use pyo3::coroutine::CancelHandle;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tokio_util::sync::CancellationToken;

use crate::sampler::PythonSampler;
use crate::trainer::PythonTrainer;

/// Worker configuration loaded by Rust from `worker.toml` and `METRL_`
/// environment overrides.
#[pyclass(frozen, module = "metrl._core")]
pub(crate) struct Config(WorkerConfig);

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
}

/// Serves the worker gRPC API and keeps it registered until cancelled.
#[pyfunction]
pub(crate) async fn serve(
    worker_id: String, config: Py<Config>, sampler: Py<PyAny>, trainer: Py<PyAny>,
    #[pyo3(cancel_handle)] mut cancel: CancelHandle,
) -> PyResult<()> {
    use tracing::Instrument;
    // Fails only if the process already installed a subscriber, which then
    // keeps receiving the worker's logs.
    metrl_telemetry::init_logs().ok();
    let id = uuid::Uuid::parse_str(&worker_id)
        .map(WorkerId)
        .map_err(|error| PyValueError::new_err(format!("worker_id must be a UUID: {error}")))?;
    let config = config.get().0.clone();
    let locals = Python::attach(pyo3_async_runtimes::tokio::get_current_locals)?;
    let identity = metrl_worker::Identity {
        id,
        advertise_addr: config.advertise_addr,
        servers: config.servers,
        bases: config
            .bases
            .into_iter()
            .map(|base| {
                // Real revisions and limits come from engines that load
                // weights. The tokenizer is a small real one, so that SDK
                // tokenizer calls work against fake bases.
                worker::Base {
                    revision: format!("{}-1", config.engine.name),
                    arch: config.engine.name.clone(),
                    tokenizer_id: "Qwen/Qwen3-0.6B".to_owned(),
                    max_context_length: 4096,
                    max_lora_rank: 64,
                    roles: base
                        .roles
                        .iter()
                        .map(|role| {
                            match role {
                                Role::Trainer => worker::Role::Trainer as i32,
                                Role::Sampler => worker::Role::Sampler as i32,
                            }
                        })
                        .collect(),
                    name: base.name,
                }
            })
            .collect(),
    };
    let shutdown = CancellationToken::new();
    // Python may drop this coroutine without cancelling it, for example when
    // its loop closes; the detached worker must still unregister and stop.
    let _guard = shutdown.clone().drop_guard();
    let sampler = PythonSampler {
        engine: sampler,
        locals: locals.clone(),
    };
    let trainer = PythonTrainer {
        engine: trainer,
        locals,
    };
    let stop = shutdown.clone();
    let mut worker = pyo3_async_runtimes::tokio::get_runtime().spawn(
        async move {
            metrl_worker::serve(
                identity,
                config.grpc_addr,
                &config.object,
                sampler,
                trainer,
                stop,
            )
            .await
        }
        .instrument(tracing::info_span!("worker")),
    );
    // On Python cancellation, unregister, then stop serving, before
    // re-raising.
    let (served, cancelled) = tokio::select! {
        served = &mut worker => (served, None),
        exception = cancel.cancelled() => {
            shutdown.cancel();
            (worker.await, Some(exception))
        }
    };
    let served = served
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
        .map_err(|report| PyRuntimeError::new_err(format!("{report:?}")));
    match cancelled {
        Some(exception) => {
            Err(Python::attach(|py| {
                PyErr::from_value(exception.into_bound(py))
            }))
        }
        None => served,
    }
}
