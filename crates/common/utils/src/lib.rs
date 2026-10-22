//! Project-internal functions that don't belong to any business module.

use tokio::signal::unix::{SignalKind, signal};

/// Resolves on the first SIGINT or SIGTERM.
pub async fn shutdown_signal() {
    let mut terminate =
        signal(SignalKind::terminate()).expect("SIGTERM handler installs once per call");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => tracing::info!("received SIGINT"),
        _ = terminate.recv() => tracing::info!("received SIGTERM"),
    }
}
