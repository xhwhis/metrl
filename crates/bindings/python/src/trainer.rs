use std::collections::BTreeMap;

use error_stack::Report;
use metrl_types::ModelId;
use metrl_worker::WorkerError;
use pyo3::prelude::*;

use crate::engine::{Engine, engine_error};
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

/// What `Trainer.forward` and `Trainer.forward_backward` return: one output
/// map per input datum, each with the same fields, dtypes, and trailing
/// shapes, and metrics keyed `name:reduction`.
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

/// What `Trainer.save_weights_for_sampler` returns and `Adapter.load`
/// reads back: a low-rank adapter in PEFT layout, with tensors named as in
/// `adapter_model.safetensors`.
#[pyclass(frozen, module = "metrl._core")]
pub(crate) struct AdapterWeights(pub(crate) metrl_worker::AdapterWeights);

#[pymethods]
impl AdapterWeights {
    #[new]
    #[pyo3(signature = (*, rank, alpha, target_modules, tensors))]
    fn new(
        rank: u32, alpha: f64, target_modules: Vec<String>,
        tensors: BTreeMap<String, Py<TensorData>>,
    ) -> Self {
        Self(metrl_worker::AdapterWeights {
            rank,
            alpha,
            target_modules,
            tensors: tensors
                .into_iter()
                .map(|(name, tensor)| (name, tensor.get().0.clone()))
                .collect(),
        })
    }

    #[getter]
    fn rank(&self) -> u32 {
        self.0.rank
    }

    #[getter]
    fn alpha(&self) -> f64 {
        self.0.alpha
    }

    #[getter]
    fn target_modules(&self) -> Vec<String> {
        self.0.target_modules.clone()
    }

    /// In name order.
    #[getter]
    fn tensors(&self) -> BTreeMap<String, TensorData> {
        self.0
            .tensors
            .iter()
            .map(|(name, tensor)| (name.clone(), TensorData(tensor.clone())))
            .collect()
    }
}

pub(crate) struct PythonTrainer(pub(crate) Engine);

impl metrl_worker::Trainer for PythonTrainer {
    async fn create_model(
        &self, model: ModelId, base_model: &str, rank: u32, seed: u64,
    ) -> Result<(), Report<WorkerError>> {
        self.0
            .call(|py, engine| {
                engine.call_method1(
                    py,
                    "create_model",
                    (model.0.to_string(), base_model, rank, seed),
                )
            })
            .await
            .map(drop)
    }

    async fn forward_backward(
        &self, model: ModelId, request: metrl_worker::ForwardBackward,
    ) -> Result<metrl_worker::ForwardBackwardOutput, Report<WorkerError>> {
        let method = if request.forward_only {
            "forward"
        } else {
            "forward_backward"
        };
        let output = self
            .0
            .call(|py, engine| {
                let data = request
                    .data
                    .into_iter()
                    .map(|datum| Py::new(py, Datum(datum)))
                    .collect::<PyResult<Vec<_>>>()?;
                engine.call_method1(py, method, (model.0.to_string(), data, request.loss_fn))
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
        &self, model: ModelId, adam_params: metrl_types::AdamParams,
    ) -> Result<BTreeMap<String, f64>, Report<WorkerError>> {
        let metrics = self
            .0
            .call(|py, engine| {
                let adam_params = AdamParams {
                    learning_rate: adam_params.learning_rate,
                    beta1: adam_params.beta1,
                    beta2: adam_params.beta2,
                    eps: adam_params.eps,
                    weight_decay: adam_params.weight_decay,
                    grad_clip_norm: adam_params.grad_clip_norm,
                };
                engine.call_method1(py, "optim_step", (model.0.to_string(), adam_params))
            })
            .await?;
        Python::attach(|py| metrics.extract(py)).map_err(engine_error)
    }

    async fn save_weights_for_sampler(
        &self, model: ModelId,
    ) -> Result<metrl_worker::AdapterWeights, Report<WorkerError>> {
        let weights = self
            .0
            .call(|py, engine| {
                engine.call_method1(py, "save_weights_for_sampler", (model.0.to_string(),))
            })
            .await?;
        Python::attach(|py| {
            weights
                .bind(py)
                .cast::<AdapterWeights>()
                .map(|weights| weights.get().0.clone())
                .map_err(PyErr::from)
        })
        .map_err(engine_error)
    }
}
