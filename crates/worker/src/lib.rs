//! Worker runtime: gRPC service, object I/O, and the engine seam.

use std::net::SocketAddr;

use error_stack::{Report, ResultExt};
use metrl_proto::worker::worker_service_server::{WorkerService, WorkerServiceServer};
use metrl_proto::worker::{DescribeRequest, DescribeResponse};
use tonic::{Request, Response, Status};

#[derive(Debug, thiserror::Error)]
#[error("worker gRPC server failed on {addr}")]
pub struct WorkerError {
    addr: SocketAddr,
}

struct Worker {
    worker_id: String,
}

#[tonic::async_trait]
impl WorkerService for Worker {
    async fn describe(
        &self, _request: Request<DescribeRequest>,
    ) -> Result<Response<DescribeResponse>, Status> {
        Ok(Response::new(DescribeResponse {
            worker_id: self.worker_id.clone(),
            bases: Vec::new(),
        }))
    }
}

/// Serves the worker gRPC API until `shutdown` resolves.
pub async fn serve(
    worker_id: String, addr: SocketAddr, shutdown: impl Future<Output = ()> + Send,
) -> Result<(), Report<WorkerError>> {
    tracing::info!(%addr, %worker_id, "worker listening");
    tonic::transport::Server::builder()
        .add_service(WorkerServiceServer::new(Worker { worker_id }))
        .serve_with_shutdown(addr, shutdown)
        .await
        .change_context(WorkerError { addr })
}
