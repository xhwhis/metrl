use std::net::SocketAddr;

use error_stack::Report;
use metrl_proto::worker::worker_service_server::{WorkerService, WorkerServiceServer};
use metrl_proto::worker::{
    Base, DescribeRequest, DescribeResponse, Role, SampleRequest, SampleResponse,
};
use tonic::{Request, Response, Status};

#[derive(Debug, thiserror::Error)]
#[error("worker operation failed")]
pub struct WorkerError;

pub trait Sampler: Send + Sync + 'static {
    fn sample(
        &self, request: SampleRequest,
    ) -> impl Future<Output = Result<SampleResponse, Report<WorkerError>>> + Send;
}

struct Worker<S> {
    id: String,
    base_model: String,
    sampler: S,
}

#[tonic::async_trait]
impl<S: Sampler> WorkerService for Worker<S> {
    async fn describe(
        &self, _request: Request<DescribeRequest>,
    ) -> Result<Response<DescribeResponse>, Status> {
        Ok(Response::new(DescribeResponse {
            worker_id: self.id.clone(),
            bases: vec![Base {
                name: self.base_model.clone(),
                revision: "mock-1".to_owned(),
                arch: "mock".to_owned(),
                tokenizer_id: self.base_model.clone(),
                max_context_length: 4096,
                max_lora_rank: 0,
                roles: vec![Role::Sampler as i32],
            }],
        }))
    }

    async fn sample(
        &self, request: Request<SampleRequest>,
    ) -> Result<Response<SampleResponse>, Status> {
        self.sampler
            .sample(request.into_inner())
            .await
            .map(Response::new)
            .map_err(|report| {
                tracing::error!(error = ?report, "engine sample failed");
                Status::internal("engine sample failed")
            })
    }
}

pub async fn serve<S: Sampler>(
    worker_id: String, base_model: String, addr: SocketAddr, sampler: S,
    shutdown: impl Future<Output = ()> + Send,
) -> Result<(), Report<WorkerError>> {
    use error_stack::ResultExt;
    tracing::info!(%addr, %worker_id, "worker listening");
    tonic::transport::Server::builder()
        .add_service(WorkerServiceServer::new(Worker {
            id: worker_id,
            base_model,
            sampler,
        }))
        .serve_with_shutdown(addr, shutdown)
        .await
        .change_context(WorkerError)
}
