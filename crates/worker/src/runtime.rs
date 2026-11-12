use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use error_stack::Report;
use metrl_config::ObjectConfig;
use metrl_proto::server::server_service_client::ServerServiceClient;
use metrl_proto::server::{RegisterRequest, UnregisterRequest};
use metrl_proto::worker;
use metrl_proto::worker::worker_service_server::WorkerServiceServer;
use metrl_types::WorkerId;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tonic::Status;
use tonic::transport::server::TcpIncoming;

use crate::service::Worker;
use crate::training::Training;
use crate::{Sampler, Trainer, WorkerError};

/// How often a worker re-registers. Registration is idempotent, so this
/// restores a worker that a server removed, for example after Postgres was
/// reset.
const REGISTER_INTERVAL: Duration = Duration::from_secs(10);

/// Bounds each call to a server. Longer than the server's own registration
/// timeout, so a registration this worker gave up on is no longer running on
/// the server and cannot commit after the worker unregisters.
const CALL_TIMEOUT: Duration = Duration::from_secs(10);

/// How long other servers may still route here after unregistering, until
/// the removal reaches them through Postgres.
const UNREGISTER_GRACE: Duration = Duration::from_secs(1);

/// How long requests in flight at shutdown may finish; servers give up on a
/// request after the same time.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(30);

/// What a worker serves and where servers reach it.
pub struct Identity {
    pub id: WorkerId,
    /// Address servers dial; `None` uses the bound address.
    pub advertise_addr: Option<String>,
    pub servers: Vec<String>,
    pub bases: Vec<worker::Base>,
}

/// Serves the worker gRPC API and keeps it registered until `shutdown` is
/// cancelled. It then unregisters while still serving, so no server routes to
/// a closed listener, and lets requests in flight finish.
pub async fn serve<S: Sampler, T: Trainer>(
    identity: Identity, addr: SocketAddr, object: &ObjectConfig, sampler: S, trainer: T,
    shutdown: CancellationToken,
) -> Result<(), Report<WorkerError>> {
    use error_stack::ResultExt;
    use tracing::Instrument;
    let object = metrl_object::open(object)
        .change_context(WorkerError::Failed)
        .attach_with(|| format!("open object store {object:?}"))?;
    let listener = TcpListener::bind(addr)
        .await
        .change_context(WorkerError::Failed)
        .attach_with(|| format!("bind {addr}"))?;
    let bound = listener.local_addr().change_context(WorkerError::Failed)?;
    let advertise = identity
        .advertise_addr
        .clone()
        .unwrap_or_else(|| bound.to_string());
    // Every host resolves a wildcard address to itself, so servers would
    // reach whatever worker runs on their own host.
    if advertise
        .parse::<SocketAddr>()
        .is_ok_and(|addr| addr.ip().is_unspecified())
    {
        return Err(Report::new(WorkerError::Failed).attach(format!(
            "cannot advertise wildcard address {advertise}; set advertise_addr"
        )));
    }
    tracing::info!(%bound, %advertise, worker_id = %identity.id.0, "worker listening");

    // Cancelled once the worker has unregistered, or if registration panics.
    let unregistered = CancellationToken::new();
    let registration = tokio::spawn(
        stay_registered(
            identity.id,
            advertise,
            identity.servers,
            shutdown.clone(),
            unregistered.clone(),
        )
        .instrument(tracing::info_span!("register")),
    );
    let listening = tonic::transport::Server::builder()
        .add_service(WorkerServiceServer::new(Worker {
            id: identity.id,
            bases: identity.bases,
            sampler,
            training: Arc::new(Training::new(trainer, object.clone())),
            object,
        }))
        .serve_with_incoming_shutdown(TcpIncoming::from(listener), unregistered.cancelled());
    let served = tokio::select! {
        result = listening => result.change_context(WorkerError::Failed),
        () = async {
            unregistered.cancelled().await;
            tokio::time::sleep(DRAIN_TIMEOUT).await;
        } => {
            tracing::warn!("requests still running after the drain timeout were dropped");
            Ok(())
        }
    };
    shutdown.cancel();
    registration.await.ok();
    served
}

/// Registers until `shutdown`, then unregisters and cancels `unregistered`.
async fn stay_registered(
    id: WorkerId, addr: String, servers: Vec<String>, shutdown: CancellationToken,
    unregistered: CancellationToken,
) {
    let _unregistered = unregistered.drop_guard();
    let request = RegisterRequest {
        worker_id: id.0.to_string(),
        addr: addr.clone(),
    };
    let mut registered = false;
    loop {
        // Never cancelled midway: a dropped call could still commit on the
        // server after the unregistration below.
        let accepted = first_server(&servers, |mut client| {
            let request = request.clone();
            async move { client.register(request).await }
        })
        .await;
        match (&accepted, registered) {
            (Ok(server), false) => tracing::info!(%server, "registered"),
            (Err(status), _) => {
                tracing::warn!(?servers, %status, "no server accepted the registration");
            }
            (Ok(_), true) => {}
        }
        registered |= accepted.is_ok();
        let delay = if registered {
            REGISTER_INTERVAL
        } else {
            Duration::from_secs(1)
        };
        tokio::select! {
            () = shutdown.cancelled() => break,
            () = tokio::time::sleep(delay) => {}
        }
    }
    let request = UnregisterRequest {
        worker_id: id.0.to_string(),
        addr,
    };
    let removed = first_server(&servers, |mut client| {
        let request = request.clone();
        async move { client.unregister(request).await }
    })
    .await;
    match removed {
        Ok(server) => tracing::info!(%server, "unregistered"),
        Err(status) => tracing::warn!(%status, "no server accepted the unregistration"),
    }
    tokio::time::sleep(UNREGISTER_GRACE).await;
}

/// Calls the servers in order and returns the first that answered, or the
/// last failure.
async fn first_server<F, T>(
    servers: &[String], call: impl Fn(ServerServiceClient<tonic::transport::Channel>) -> F,
) -> Result<&str, Status>
where
    F: Future<Output = Result<T, Status>>,
{
    let mut failure = Status::failed_precondition("no valid server endpoint");
    for server in servers {
        let client = tonic::transport::Endpoint::from_shared(server.clone()).map(|endpoint| {
            endpoint
                .connect_timeout(Duration::from_secs(1))
                .timeout(CALL_TIMEOUT)
        });
        let Ok(endpoint) = client else {
            tracing::warn!(%server, "invalid server endpoint");
            continue;
        };
        let result = match endpoint.connect().await {
            Ok(channel) => call(ServerServiceClient::new(channel)).await.map(|_| ()),
            // `Display` of a transport error omits its cause.
            Err(error) => Err(Status::unavailable(format!("{error:?}"))),
        };
        match result {
            Ok(()) => return Ok(server),
            Err(status) => {
                tracing::debug!(%server, %status, "server call failed");
                failure = status;
            }
        }
    }
    Err(failure)
}
