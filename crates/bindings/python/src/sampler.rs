use error_stack::Report;
use metrl_proto::worker;
use metrl_worker::WorkerError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::engine::{Engine, engine_error};
use crate::trainer::AdapterWeights;

/// Tinker `SamplingParams`. `seed` is `None` for a different draw on every
/// call. `stop` is `None` to keep the implicit end-of-sequence stop, and an
/// empty list to stop only at `max_tokens`.
#[pyclass(frozen, get_all, module = "metrl._core")]
pub(crate) struct SamplingParams {
    max_tokens: u32,
    seed: Option<u64>,
    temperature: f64,
    top_k: i32,
    top_p: f64,
    stop: Option<Stop>,
}

/// Stop token ids or stop strings; a request never mixes them.
#[derive(Clone, IntoPyObject)]
pub(crate) enum Stop {
    Tokens(Vec<i32>),
    Strings(Vec<String>),
}

/// A saved low-rank adapter to sample with. Its files never change, so engines
/// may cache what `load` returns under `key`.
#[pyclass(frozen, module = "metrl._core")]
pub(crate) struct Adapter(metrl_worker::Adapter);

#[pymethods]
impl Adapter {
    #[getter]
    fn key(&self) -> &str {
        self.0.key()
    }

    /// Downloads and decodes the adapter.
    async fn load(&self) -> PyResult<AdapterWeights> {
        use tracing::Instrument;
        let adapter = self.0.clone();
        let span = tracing::info_span!("load_adapter", key = adapter.key());
        // Object store clients need the Tokio runtime; dropping the handle
        // when Python cancels the call aborts the download.
        let loading = tokio_util::task::AbortOnDropHandle::new(
            pyo3_async_runtimes::tokio::get_runtime()
                .spawn(async move { adapter.load().await }.instrument(span)),
        );
        loading
            .await
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
            .map(AdapterWeights)
            .map_err(|report| PyRuntimeError::new_err(format!("{report:?}")))
    }
}

/// Tinker `SampledSequence`: the generated tokens, each one's logprob under
/// the distribution it was drawn from, and why generation stopped,
/// `"length"` or `"stop"`.
#[pyclass(frozen, module = "metrl._core")]
pub(crate) struct SampledSequence(worker::SampledSequence);

#[pymethods]
impl SampledSequence {
    #[new]
    #[pyo3(signature = (*, tokens, logprobs, stop_reason))]
    fn new(tokens: Vec<i32>, logprobs: Vec<f32>, stop_reason: &str) -> PyResult<Self> {
        let stop_reason = match stop_reason {
            "length" => worker::StopReason::Length,
            "stop" => worker::StopReason::Stop,
            other => {
                return Err(PyValueError::new_err(format!(
                    "stop_reason must be \"length\" or \"stop\", not {other:?}"
                )));
            }
        };
        if logprobs.len() != tokens.len() {
            return Err(PyValueError::new_err(format!(
                "{} logprobs for {} tokens",
                logprobs.len(),
                tokens.len()
            )));
        }
        Ok(Self(worker::SampledSequence {
            tokens,
            logprobs,
            stop_reason: stop_reason as i32,
        }))
    }
}

/// What `Sampler.sample` returns: one sequence per requested sample and,
/// when requested, one logprob per prompt token, `None` for the first.
#[pyclass(frozen, module = "metrl._core")]
pub(crate) struct SampleResponse(worker::SampleResponse);

#[pymethods]
impl SampleResponse {
    #[new]
    #[pyo3(signature = (sequences, prompt_logprobs = None))]
    fn new(sequences: Vec<Py<SampledSequence>>, prompt_logprobs: Option<Vec<Option<f32>>>) -> Self {
        Self(worker::SampleResponse {
            sequences: sequences
                .into_iter()
                .map(|sequence| sequence.get().0.clone())
                .collect(),
            prompt_logprobs: prompt_logprobs
                .unwrap_or_default()
                .into_iter()
                .map(|logprob| logprob.unwrap_or(f32::NAN))
                .collect(),
        })
    }
}

pub(crate) struct PythonSampler(pub(crate) Engine);

impl metrl_worker::Sampler for PythonSampler {
    async fn sample(
        &self, request: worker::SampleRequest, adapter: Option<metrl_worker::Adapter>,
    ) -> Result<worker::SampleResponse, Report<WorkerError>> {
        let response = self
            .0
            .call(|py, engine| {
                let params = SamplingParams {
                    max_tokens: request.max_tokens,
                    seed: request.seed,
                    temperature: request.temperature,
                    top_k: request.top_k,
                    top_p: request.top_p,
                    stop: request.stop.map(|stop| {
                        if stop.strings.is_empty() {
                            Stop::Tokens(stop.tokens)
                        } else {
                            Stop::Strings(stop.strings)
                        }
                    }),
                };
                engine.call_method1(
                    py,
                    "sample",
                    (
                        request.model_id,
                        request.base_model,
                        adapter.map(Adapter),
                        request.prompt,
                        request.num_samples,
                        params,
                        request.prompt_logprobs,
                    ),
                )
            })
            .await?;
        Python::attach(|py| {
            response
                .bind(py)
                .cast::<SampleResponse>()
                .map(|response| response.get().0.clone())
                .map_err(PyErr::from)
        })
        .map_err(engine_error)
    }
}
