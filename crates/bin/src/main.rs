use std::net::SocketAddr;
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use error_stack::{Report, ResultExt};
use metrl_config::ServerConfig;
use metrl_types::{KeyHash, OrgId, ServerId};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
#[error("metrl-server failed")]
struct ServerError;

#[derive(Parser)]
#[command(about = "Metrl control-plane server")]
struct Args {
    #[arg(long, default_value = "config/server.toml")]
    config: PathBuf,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Serve the API (default).
    Serve,
    /// Create the schema in an empty database.
    Migrate,
    /// Manage API keys.
    Key {
        #[command(subcommand)]
        command: KeyCommand,
    },
}

#[derive(Subcommand)]
enum KeyCommand {
    /// Create an API key for an org, creating the org if needed. Prints the key
    /// once; only its hash is stored.
    Create {
        #[arg(long)]
        org: String,
    },
}

async fn serve_http(
    addr: SocketAddr, router: axum::Router, shutdown: CancellationToken,
) -> Result<(), Report<ServerError>> {
    let listener = TcpListener::bind(addr)
        .await
        .change_context(ServerError)
        .attach_with(|| format!("bind {addr}"))?;
    tracing::info!(%addr, "listening");
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown.cancelled_owned())
        .await
        .change_context(ServerError)
}

async fn serve_grpc(
    addr: SocketAddr, store: metrl_meta::Store, shutdown: CancellationToken,
) -> Result<(), Report<ServerError>> {
    tracing::info!(%addr, "listening");
    tonic::transport::Server::builder()
        .add_service(metrl_server::registration(store, addr))
        .serve_with_shutdown(addr, shutdown.cancelled_owned())
        .await
        .change_context(ServerError)
        .attach_with(|| format!("serve gRPC on {addr}"))
}

async fn serve(config: ServerConfig, store: metrl_meta::Store) -> Result<(), Report<ServerError>> {
    use tracing::Instrument;
    let system = metrl_telemetry::init().change_context(ServerError)?;
    // Only a `failpoints` build reads `FAILPOINTS`. The configuration lasts
    // as long as the process, so its guard is never dropped.
    if fail::has_failpoints() {
        std::mem::forget(fail::FailScenario::setup());
    }
    let server = ServerId(Uuid::now_v7());
    let mut cdc = metrl_meta::follow(store.clone(), server);
    cdc.ready().await;
    let view = cdc.view();
    let workers = metrl_server::Workers::new(view.clone());
    // Detached: models stop with the process, and their leases lapse.
    tokio::spawn(
        metrl_server::own_models(server, store.clone(), view.clone(), workers.clone())
            .instrument(tracing::info_span!("models", server_id = %server.0)),
    );
    let object = metrl_object::open(&config.object)
        .change_context(ServerError)
        .attach_with(|| format!("open object store {:?}", config.object))?;
    let training =
        metrl_server::Training::new(store.clone(), view.clone(), workers.clone(), object);
    let sampling = metrl_server::Sampling::new(view, workers, store.clone());
    // The API also ends its long polls on this token; the HTTP drain would
    // otherwise wait for them.
    let shutdown = CancellationToken::new();
    let router = metrl_api::router(sampling, training, shutdown.clone());
    tokio::try_join!(
        async {
            metrl_utils::shutdown_signal().await;
            shutdown.cancel();
            Ok::<(), Report<ServerError>>(())
        },
        serve_http(config.http_addr, router, shutdown.clone()),
        serve_grpc(config.grpc_addr, store, shutdown.clone()),
        serve_http(config.system_addr, system, shutdown.clone()),
    )?;
    Ok(())
}

fn new_key() -> Result<String, Report<ServerError>> {
    use std::fmt::Write;
    let mut bytes = [0_u8; 24];
    getrandom::fill(&mut bytes)
        .map_err(|error| Report::new(ServerError).attach(error.to_string()))?;
    Ok(bytes.iter().fold(String::from("tml-"), |mut key, byte| {
        write!(key, "{byte:02x}").expect("writing to a String cannot fail");
        key
    }))
}

#[tokio::main]
async fn main() -> Result<(), Report<ServerError>> {
    let args = Args::parse();
    let config: ServerConfig = metrl_config::load(&args.config).change_context(ServerError)?;
    let store = metrl_meta::connect(&config.database_url).change_context(ServerError)?;
    match args.command.unwrap_or(Command::Serve) {
        Command::Serve => serve(config, store).await,
        Command::Migrate => store.migrate().await.change_context(ServerError),
        Command::Key {
            command: KeyCommand::Create { org },
        } => {
            let key = new_key()?;
            let org = OrgId(org);
            store
                .create_api_key(&org, KeyHash::of(&key))
                .await
                .change_context(ServerError)
                .attach_with(|| format!("create API key for org {}", org.0))?;
            println!("{key}");
            Ok(())
        }
    }
}
