use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;
use error_stack::{Report, ResultExt};
use metrl_config::ServerConfig;
use tokio::net::TcpListener;

#[derive(Debug, thiserror::Error)]
#[error("metrl-server failed")]
struct ServerError;

#[derive(Parser)]
#[command(about = "Metrl control-plane server")]
struct Args {
    #[arg(long, default_value = "config/server.toml")]
    config: PathBuf,
}

async fn serve(addr: SocketAddr, router: axum::Router) -> Result<(), Report<ServerError>> {
    let listener = TcpListener::bind(addr)
        .await
        .change_context(ServerError)
        .attach_with(|| format!("bind {addr}"))?;
    tracing::info!(%addr, "listening");
    axum::serve(listener, router)
        .with_graceful_shutdown(metrl_utils::shutdown_signal())
        .await
        .change_context(ServerError)
}

#[tokio::main]
async fn main() -> Result<(), Report<ServerError>> {
    let args = Args::parse();
    let system = metrl_telemetry::init().change_context(ServerError)?;
    let config: ServerConfig = metrl_config::load(&args.config).change_context(ServerError)?;
    tokio::try_join!(
        serve(config.http_addr, metrl_api::router()),
        serve(config.system_addr, system),
    )?;
    Ok(())
}
