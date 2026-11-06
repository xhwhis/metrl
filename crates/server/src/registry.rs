use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use error_stack::Report;
use metrl_meta::{Store, View};
use metrl_proto::server::server_service_server::{ServerService, ServerServiceServer};
use metrl_proto::server::{
    RegisterRequest, RegisterResponse, UnregisterRequest, UnregisterResponse,
};
use metrl_proto::worker::worker_service_client::WorkerServiceClient;
use metrl_proto::worker::{self, DescribeRequest};
use metrl_types::{Base, Role, WorkerId};
use parking_lot::Mutex;
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, Response, Status};
use uuid::Uuid;

use crate::ServerError;

/// Bounds a registration from dialing the worker to committing it. Workers
/// wait longer, so a registration a worker gave up on is no longer running
/// here.
const REGISTER_TIMEOUT: Duration = Duration::from_secs(5);

/// Registered workers, as replicated through Postgres, with a lazily opened
/// channel to each address.
#[derive(Clone)]
pub struct Workers {
    view: View,
    channels: Arc<Mutex<HashMap<String, Channel>>>,
}

pub(crate) struct Live {
    pub(crate) id: WorkerId,
    pub(crate) client: WorkerServiceClient<Channel>,
    bases: Vec<Base>,
}

impl Workers {
    pub fn new(view: View) -> Self {
        Self {
            view,
            channels: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Workers currently registered, each with a client.
    fn live(&self) -> Vec<Live> {
        let workers = self.view.workers();
        let mut channels = self.channels.lock();
        channels.retain(|addr, _| workers.iter().any(|worker| worker.addr == *addr));
        let live = workers
            .into_iter()
            .filter_map(|worker| {
                let channel = if let Some(channel) = channels.get(&worker.addr) {
                    channel.clone()
                } else {
                    let channel = Endpoint::from_shared(format!("http://{}", worker.addr))
                        .inspect_err(|error| {
                            tracing::warn!(%error, addr = %worker.addr, "invalid worker address");
                        })
                        .ok()?
                        .connect_timeout(Duration::from_secs(3))
                        .timeout(Duration::from_secs(30))
                        .connect_lazy();
                    channels.insert(worker.addr, channel.clone());
                    channel
                };
                Some(Live {
                    id: worker.id,
                    client: WorkerServiceClient::new(channel),
                    bases: worker.bases,
                })
            })
            .collect();
        drop(channels);
        live
    }

    /// `name` as a registered worker that serves it for `role` reports it.
    pub(crate) fn base(&self, name: &str, role: Role) -> Option<Base> {
        self.view
            .workers()
            .into_iter()
            .flat_map(|worker| worker.bases)
            .find(|base| base.name == name && base.roles.contains(&role))
    }

    /// The registered worker with this id, if it is still registered.
    pub(crate) fn get(&self, id: WorkerId) -> Option<Live> {
        self.live().into_iter().find(|live| live.id == id)
    }

    /// Bases of all registered workers, one entry per name with the roles of
    /// every worker that serves it and the smallest context length any of
    /// them accepts.
    pub(crate) fn bases(&self) -> Vec<Base> {
        let mut bases: Vec<Base> = self
            .view
            .workers()
            .into_iter()
            .flat_map(|worker| worker.bases)
            .collect();
        bases.sort_by(|a, b| a.name.cmp(&b.name));
        // `dedup_by` passes the duplicate first and the kept entry second.
        bases.dedup_by(|duplicate, kept| {
            let same = duplicate.name == kept.name;
            if same {
                kept.roles.append(&mut duplicate.roles);
                kept.max_context_length = kept.max_context_length.min(duplicate.max_context_length);
            }
            same
        });
        for base in &mut bases {
            base.roles.sort();
            base.roles.dedup();
        }
        bases
    }

    /// A worker serving `base` at `revision` for `role`. Picks by `spread` so
    /// that callers spread load without shared counters.
    pub(crate) fn pick(
        &self, base: &str, revision: &str, role: Role, spread: u64,
    ) -> Result<Live, Report<ServerError>> {
        let candidates: Vec<_> = self
            .live()
            .into_iter()
            .filter(|live| {
                live.bases.iter().any(|known| {
                    known.name == base && known.revision == revision && known.roles.contains(&role)
                })
            })
            .collect();
        if candidates.is_empty() {
            return Err(
                Report::new(ServerError::Unavailable("no worker serves the base"))
                    .attach(format!("base {base} revision {revision}, role {role:?}")),
            );
        }
        let index = (spread % candidates.len() as u64) as usize;
        Ok(candidates
            .into_iter()
            .nth(index)
            .expect("the index is reduced modulo the number of candidates"))
    }
}

/// Serves worker registration on the server gRPC port bound to `listen`.
pub fn registration(store: Store, listen: SocketAddr) -> ServerServiceServer<Registration> {
    ServerServiceServer::new(Registration { store, listen })
}

pub struct Registration {
    store: Store,
    listen: SocketAddr,
}

impl Registration {
    /// Records the worker once its address is shown to reach it.
    async fn admit(&self, request: RegisterRequest) -> Result<(), Report<ServerError>> {
        use error_stack::ResultExt;
        let id = worker_id(&request.worker_id)?;
        // Servers dial the address from their own hosts. A wildcard address
        // reaches each server's own host, and so does a loopback address,
        // which only means one worker when every server shares this host.
        let resolved: Vec<_> = tokio::net::lookup_host(&request.addr)
            .await
            .change_context(ServerError::Invalid("addr must resolve as host:port"))
            .attach_with(|| format!("addr {}", request.addr))?
            // `::ffff:127.0.0.1` reaches the same host as `127.0.0.1`.
            .map(|addr| addr.ip().to_canonical())
            .collect();
        if resolved.iter().any(IpAddr::is_unspecified) {
            return Err(
                Report::new(ServerError::Invalid("addr must not be a wildcard address"))
                    .attach(format!("addr {}", request.addr)),
            );
        }
        if !self.listen.ip().to_canonical().is_loopback()
            && resolved.iter().any(IpAddr::is_loopback)
        {
            return Err(Report::new(ServerError::Invalid(
                "addr must not be a loopback address unless servers listen on loopback",
            ))
            .attach(format!("addr {}", request.addr)));
        }
        // Ask the worker itself, which proves the address reaches it before any
        // server routes to it.
        let channel = Endpoint::from_shared(format!("http://{}", request.addr))
            .change_context(ServerError::Invalid("addr must be host:port"))?
            .connect_timeout(Duration::from_secs(2))
            .connect()
            .await
            .change_context(ServerError::Unavailable("cannot reach worker"))
            .attach_with(|| format!("addr {}", request.addr))?;
        let described = WorkerServiceClient::new(channel)
            .describe(DescribeRequest {})
            .await
            .change_context(ServerError::Unavailable("worker did not describe itself"))?
            .into_inner();
        if described.worker_id != request.worker_id {
            return Err(Report::new(ServerError::Invalid(
                "another worker answers at this address",
            ))
            .attach(format!("addr {}", request.addr)));
        }
        let bases = described
            .bases
            .into_iter()
            .map(base)
            .collect::<Result<Vec<_>, _>>()?;
        self.store
            .register_worker(id, &request.addr, &bases)
            .await
            .change_context(ServerError::Unavailable("database unavailable"))
            .attach_with(|| format!("worker {} at {}", id.0, request.addr))
    }
}

#[tonic::async_trait]
impl ServerService for Registration {
    async fn register(
        &self, request: Request<RegisterRequest>,
    ) -> Result<Response<RegisterResponse>, Status> {
        let admitted = tokio::time::timeout(REGISTER_TIMEOUT, self.admit(request.into_inner()))
            .await
            .unwrap_or_else(|_| {
                Err(Report::new(ServerError::Unavailable(
                    "registration timed out",
                )))
            });
        admitted
            .map(|()| Response::new(RegisterResponse {}))
            .map_err(|report| status(&report))
    }

    async fn unregister(
        &self, request: Request<UnregisterRequest>,
    ) -> Result<Response<UnregisterResponse>, Status> {
        use error_stack::ResultExt;
        let request = request.into_inner();
        let id = worker_id(&request.worker_id).map_err(|report| status(&report))?;
        self.store
            .remove_worker(id, &request.addr)
            .await
            .change_context(ServerError::Unavailable("database unavailable"))
            .attach_with(|| format!("worker {} at {}", id.0, request.addr))
            .map(|()| Response::new(UnregisterResponse {}))
            .map_err(|report| status(&report))
    }
}

/// The status a worker sees for a failed registration call. The full report
/// is logged here, because the worker only logs the status.
fn status(report: &Report<ServerError>) -> Status {
    tracing::warn!(error = ?report, "worker registration call failed");
    let message = report.current_context().to_string();
    match report.current_context() {
        ServerError::Invalid(_) => Status::invalid_argument(message),
        ServerError::NotFound | ServerError::Gone => Status::not_found(message),
        ServerError::Unauthorized => Status::unauthenticated(message),
        ServerError::Capacity => Status::resource_exhausted(message),
        ServerError::Unavailable(_) => Status::unavailable(message),
        ServerError::Internal(_) => Status::internal(message),
    }
}

fn worker_id(id: &str) -> Result<WorkerId, Report<ServerError>> {
    use error_stack::ResultExt;
    Uuid::parse_str(id)
        .map(WorkerId)
        .change_context(ServerError::Invalid("worker_id must be a UUID"))
}

fn base(base: worker::Base) -> Result<Base, Report<ServerError>> {
    let roles = base
        .roles()
        .map(|role| {
            match role {
                worker::Role::Trainer => Ok(Role::Trainer),
                worker::Role::Sampler => Ok(Role::Sampler),
                worker::Role::Unspecified => {
                    Err(Report::new(ServerError::Invalid("base without a role")))
                }
            }
        })
        .collect::<Result<_, _>>()?;
    Ok(Base {
        name: base.name,
        revision: base.revision,
        arch: base.arch,
        tokenizer_id: base.tokenizer_id,
        max_context_length: base.max_context_length,
        max_lora_rank: base.max_lora_rank,
        roles,
    })
}
