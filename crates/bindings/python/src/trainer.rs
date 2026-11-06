use std::collections::BTreeMap;

use error_stack::Report;
use metrl_types::ModelId;
use metrl_worker::WorkerError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_async_runtimes::TaskLocals;

use crate::tensor::TensorData;

/// Tinker `Datum`: the model input's tokens and the loss function's inputs.
#[pyclass(frozen, module = "metrl._core")]
pub(crate) struct Datum(metrl_worker::Datum);

#[pymethods]
impl Datum {
    #[getter]
    fn tokens(&self) -> Vec<i32> {
        self.0.tokens.clone()
    }

    #[getter]
    fn loss_fn_inputs(&self) -> BTreeMap<String, TensorData> {
        self.0
            .loss_fn_inputs
            .iter()
            .map(|(name, tensor)| (name.clone(), TensorData(tensor.clone())))
            .collect()
    }
}

/// What `Trainer.forward_backward` returns: one output map per input datum,
/// each with the same fields, dtypes, and trailing shapes, and metrics keyed
/// `name:reduction`.
#[pyclass(frozen, module = "metrl._core")]
pub(crate) struct ForwardBackwardOutput(metrl_worker::ForwardBackwardOutput);

#[pymethods]
impl ForwardBackwardOutput {
    #[new]
    fn new(
        loss_fn_outputs: Vec<BTreeMap<String, Py<TensorData>>>, metrics: BTreeMap<String, f64>,
    ) -> Self {
        Self(metrl_worker::ForwardBackwardOutput {
            loss_fn_outputs: loss_fn_outputs
                .into_iter()
                .map(|fields| {
                    fields
                        .into_iter()
                        .map(|(name, tensor)| (name, tensor.get().0.clone()))
                        .collect()
                })
                .collect(),
            metrics,
        })
    }
}

/// Tinker `AdamParams`.
#[pyclass(frozen, get_all, module = "metrl._core")]
pub(crate) struct AdamParams {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    grad_clip_norm: f64,
}

pub(crate) struct PythonTrainer {
    pub(crate) engine: Py<PyAny>,
    pub(crate) locals: TaskLocals,
}

impl PythonTrainer {
    /// Awaits the coroutine that `call` starts on the engine, on the
    /// engine's event loop.
    async fn call(
        &self, call: impl FnOnce(Python<'_>, &Py<PyAny>) -> PyResult<Py<PyAny>> + Send,
    ) -> Result<Py<PyAny>, Report<WorkerError>> {
        let future = Python::attach(|py| {
            let coroutine = call(py, &self.engine)?;
            pyo3_async_runtimes::into_future_with_locals(&self.locals, coroutine.into_bound(py))
        })
        .map_err(engine_error)?;
        future.await.map_err(engine_error)
    }
}

impl metrl_worker::Trainer for PythonTrainer {
    async fn create(
        &self, model: ModelId, base: &str, lora_rank: u32, seed: u64,
    ) -> Result<(), Report<WorkerError>> {
        self.call(|py, engine| {
            engine.call_method1(py, "create", (model.0.to_string(), base, lora_rank, seed))
        })
        .await
        .map(drop)
    }

    async fn forward_backward(
        &self, model: ModelId, request: metrl_worker::ForwardBackward,
    ) -> Result<metrl_worker::ForwardBackwardOutput, Report<WorkerError>> {
        let output = self
            .call(|py, engine| {
                let data = request
                    .data
                    .into_iter()
                    .map(|datum| Py::new(py, Datum(datum)))
                    .collect::<PyResult<Vec<_>>>()?;
                engine.call_method1(
                    py,
                    "forward_backward",
                    (
                        model.0.to_string(),
                        data,
                        request.loss_fn,
                        request.forward_only,
                    ),
                )
            })
            .await?;
        Python::attach(|py| {
            output
                .bind(py)
                .cast::<ForwardBackwardOutput>()
                .map(|output| output.get().0.clone())
                .map_err(PyErr::from)
        })
        .map_err(engine_error)
    }

    async fn optim_step(
        &self, model: ModelId, params: metrl_types::AdamParams,
    ) -> Result<BTreeMap<String, f64>, Report<WorkerError>> {
        let metrics = self
            .call(|py, engine| {
                let params = AdamParams {
                    learning_rate: params.learning_rate,
                    beta1: params.beta1,
                    beta2: params.beta2,
                    eps: params.eps,
                    weight_decay: params.weight_decay,
                    grad_clip_norm: params.grad_clip_norm,
                };
                engine.call_method1(py, "optim_step", (model.0.to_string(), params))
            })
            .await?;
        Python::attach(|py| metrics.extract(py)).map_err(engine_error)
    }
}

/// An engine raises `ValueError` for a request it cannot run as given; any
/// other exception is the engine's own failure.
fn engine_error(error: PyErr) -> Report<WorkerError> {
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
