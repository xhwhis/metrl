use error_stack::Report;
use metrl_proto::worker::{SampleRequest, SampleResponse, SampledSequence};
use metrl_worker::WorkerError;
use pyo3::prelude::*;
use pyo3_async_runtimes::TaskLocals;

pub(crate) struct PythonSampler {
    pub(crate) engine: Py<PyAny>,
    pub(crate) locals: TaskLocals,
}

impl metrl_worker::Sampler for PythonSampler {
    async fn sample(&self, request: SampleRequest) -> Result<SampleResponse, Report<WorkerError>> {
        use error_stack::ResultExt;
        let future = Python::attach(|py| {
            let coroutine = self.engine.call_method1(
                py,
                "sample",
                (
                    request.model_id,
                    request.prompt,
                    request.num_samples,
                    request.max_length,
                    request.seed,
                ),
            )?;
            pyo3_async_runtimes::into_future_with_locals(&self.locals, coroutine.into_bound(py))
        })
        .change_context(WorkerError)?;
        let result = future.await.change_context(WorkerError)?;
        let sequences: Vec<Vec<i32>> =
            Python::attach(|py| result.extract(py)).change_context(WorkerError)?;
        Ok(SampleResponse {
            sequences: sequences
                .into_iter()
                .map(|tokens| SampledSequence { tokens })
                .collect(),
        })
    }
}
