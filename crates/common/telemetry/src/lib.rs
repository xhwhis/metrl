//! Logging, metrics, and the system endpoints served on the system port.

use axum::Router;
use axum::routing::get;
use error_stack::{Report, ResultExt};
use metrics_exporter_prometheus::PrometheusBuilder;
use tracing_subscriber::EnvFilter;

#[derive(Debug, thiserror::Error)]
#[error("failed to initialize telemetry")]
pub struct TelemetryError;

/// Installs the global log subscriber and metrics recorder, and returns the
/// router for the system port.
pub fn init() -> Result<Router, Report<TelemetryError>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .try_init()
        .map_err(|error| Report::new(TelemetryError).attach(error.to_string()))?;
    let prometheus = PrometheusBuilder::new()
        .install_recorder()
        .change_context(TelemetryError)?;
    Ok(Router::new().route("/metrics", get(move || async move { prometheus.render() })))
}
