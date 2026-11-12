use metrl_proto::worker;
use metrl_types::{Role, WorkerId};
use pyo3::coroutine::CancelHandle;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tokio_util::sync::CancellationToken;

use crate::base::Base;
use crate::config::Config;
use crate::engine::Engine;
use crate::sampler::PythonSampler;
use crate::trainer::PythonTrainer;

/// Serves the worker gRPC API and keeps it registered until cancelled.
/// `bases` holds what the engine loaded for each configured base, in order.
#[pyfunction]
pub(crate) async fn serve(
    worker_id: String, config: Py<Config>, bases: Vec<Py<Base>>, sampler: Py<PyAny>,
    trainer: Py<PyAny>, #[pyo3(cancel_handle)] mut cancel: CancelHandle,
) -> PyResult<()> {
    use tracing::Instrument;
    // Fails only if the process already installed a subscriber, which then
    // keeps receiving the worker's logs.
    metrl_telemetry::init_logs().ok();
    let id = uuid::Uuid::parse_str(&worker_id)
        .map(WorkerId)
        .map_err(|error| PyValueError::new_err(format!("worker_id must be a UUID: {error}")))?;
    let config = config.get().0.clone();
    if bases.len() != config.bases.len() {
        return Err(PyValueError::new_err(format!(
            "the engine loaded {} bases for {} configured",
            bases.len(),
            config.bases.len()
        )));
    }
    let bases = config
        .bases
        .iter()
        .zip(bases)
        .map(|(configured, loaded)| {
            let loaded = loaded.get();
            if loaded.name != configured.name {
                return Err(PyValueError::new_err(format!(
                    "the engine loaded base {:?} for configured base {:?}",
                    loaded.name, configured.name
                )));
            }
            Ok(worker::Base {
                name: loaded.name.clone(),
                revision: loaded.revision.clone(),
                arch: loaded.arch.clone(),
                tokenizer_id: loaded.tokenizer_id.clone(),
                max_context_length: loaded.max_context_length,
                max_lora_rank: loaded.max_lora_rank,
                roles: configured
                    .roles
                    .iter()
                    .map(|role| {
                        match role {
                            Role::Trainer => worker::Role::Trainer as i32,
                            Role::Sampler => worker::Role::Sampler as i32,
                        }
                    })
                    .collect(),
            })
        })
        .collect::<PyResult<_>>()?;
    let locals = Python::attach(pyo3_async_runtimes::tokio::get_current_locals)?;
    let identity = metrl_worker::Identity {
        id,
        advertise_addr: config.advertise_addr,
        servers: config.servers,
        bases,
    };
    let shutdown = CancellationToken::new();
    // Python may drop this coroutine without cancelling it, for example when
    // its loop closes; the detached worker must still unregister and stop.
    let _guard = shutdown.clone().drop_guard();
    let sampler = PythonSampler(Engine {
        object: sampler,
        locals: locals.clone(),
    });
    let trainer = PythonTrainer(Engine {
        object: trainer,
        locals,
    });
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
