use std::sync::Arc;

use error_stack::Report;
use metrl_proto::worker::worker_service_server::WorkerService;
use metrl_proto::worker::{
    self, BindRequest, BindResponse, DescribeRequest, DescribeResponse, ExecuteRequest,
    ExecuteResponse, SampleRequest, SampleResponse,
};
use metrl_types::WorkerId;
use tonic::{Request, Response, Status};

use crate::training::Training;
use crate::{Sampler, Trainer, WorkerError};

pub(crate) struct Worker<S, T> {
    pub(crate) id: WorkerId,
    pub(crate) bases: Vec<worker::Base>,
    pub(crate) sampler: S,
    pub(crate) training: Arc<Training<T>>,
}

impl<S, T> Worker<S, T> {
    /// Rejects a request a server meant for a worker that used to listen at
    /// this address.
    fn addressed(&self, worker_id: &str) -> Result<(), Status> {
        if worker_id == self.id.0.to_string() {
            Ok(())
        } else {
            Err(Status::failed_precondition(
                "request is addressed to another worker",
            ))
        }
    }
}

#[tonic::async_trait]
impl<S: Sampler, T: Trainer> WorkerService for Worker<S, T> {
    async fn describe(
        &self, _request: Request<DescribeRequest>,
    ) -> Result<Response<DescribeResponse>, Status> {
        Ok(Response::new(DescribeResponse {
            worker_id: self.id.0.to_string(),
            bases: self.bases.clone(),
        }))
    }

    async fn sample(
        &self, request: Request<SampleRequest>,
    ) -> Result<Response<SampleResponse>, Status> {
        let request = request.into_inner();
        self.addressed(&request.worker_id)?;
        self.sampler
            .sample(request)
            .await
            .map(Response::new)
            .map_err(|report| status(&report))
    }

    async fn bind(&self, request: Request<BindRequest>) -> Result<Response<BindResponse>, Status> {
        use tracing::Instrument;
        let request = request.into_inner();
        self.addressed(&request.worker_id)?;
        if !self.bases.iter().any(|base| {
            base.name == request.base
                && base.revision == request.revision
                && base.roles().any(|role| role == worker::Role::Trainer)
        }) {
            return Err(Status::failed_precondition(
                "worker does not serve the requested base revision",
            ));
        }
        let span = tracing::info_span!("bind", model_id = %request.model_id, generation = request.generation);
        let training = Arc::clone(&self.training);
        // Runs to completion even if the server gives up, so the engine never
        // sees a half-created model.
        tokio::spawn(async move { training.bind(request).await }.instrument(span))
            .await
            .map_err(|error| Status::internal(error.to_string()))?
            .map(|()| Response::new(BindResponse {}))
            .map_err(|report| status(&report))
    }

    async fn execute(
        &self, request: Request<ExecuteRequest>,
    ) -> Result<Response<ExecuteResponse>, Status> {
        use tracing::Instrument;
        let request = request.into_inner();
        self.addressed(&request.worker_id)?;
        let span = tracing::info_span!(
            "execute",
            model_id = %request.model_id,
            generation = request.generation,
            seq = request.seq
        );
        let training = Arc::clone(&self.training);
        // Runs to completion even if the server gives up, so the engine never
        // sees a half-applied task and a retry finds its response.
        tokio::spawn(async move { training.execute(request).await }.instrument(span))
            .await
            .map_err(|error| Status::internal(error.to_string()))?
            .map(Response::new)
            .map_err(|report| status(&report))
    }
}

/// The status a server sees for a failed call. The full report is logged
/// here, because the server only sees the status.
fn status(report: &Report<WorkerError>) -> Status {
    let error = report.current_context();
    match error {
        WorkerError::Invalid(message) => Status::invalid_argument(message.clone()),
        WorkerError::Superseded => Status::aborted(error.to_string()),
        WorkerError::Unbound => Status::failed_precondition(error.to_string()),
        WorkerError::Unavailable => {
            tracing::warn!(error = ?report, "worker call can be retried");
            Status::unavailable(error.to_string())
        }
        WorkerError::Failed => {
            tracing::error!(error = ?report, "worker call failed");
            Status::internal(error.to_string())
        }
    }
}
