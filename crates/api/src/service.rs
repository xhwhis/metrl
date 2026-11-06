use axum::extract::{DefaultBodyLimit, FromRef, State};
use axum::{Json, Router, routing};
use metrl_server::{Sampling, Training};
use serde::Serialize;
use tokio_util::sync::CancellationToken;

use crate::error::ApiError;
use crate::{futures, sampling, training};

/// Larger request bodies get 413.
const MAX_BODY: usize = 256 << 20;

#[derive(Clone)]
struct Services {
    sampling: Sampling,
    training: Training,
    /// Cancelled when the server starts shutting down.
    shutdown: CancellationToken,
}

impl FromRef<Services> for Sampling {
    fn from_ref(services: &Services) -> Self {
        services.sampling.clone()
    }
}

impl FromRef<Services> for Training {
    fn from_ref(services: &Services) -> Self {
        services.training.clone()
    }
}

impl FromRef<Services> for CancellationToken {
    fn from_ref(services: &Services) -> Self {
        services.shutdown.clone()
    }
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
}

async fn healthz(State(service): State<Sampling>) -> Result<Json<HealthResponse>, ApiError> {
    service.ensure_fresh()?;
    Ok(Json(HealthResponse { status: "ok" }))
}

pub fn router(sampling: Sampling, training: Training, shutdown: CancellationToken) -> Router {
    let api = Router::new()
        .route("/healthz", routing::get(healthz))
        .route("/client/config", routing::post(sampling::config))
        .route("/create_session", routing::post(sampling::create_session))
        .route("/session_heartbeat", routing::post(sampling::heartbeat))
        .route(
            "/create_sampling_session",
            routing::post(sampling::create_sampler),
        )
        .route(
            "/get_server_capabilities",
            routing::get(sampling::capabilities),
        )
        .route("/asample", routing::post(sampling::sample))
        .route("/create_model", routing::post(training::create_model))
        .route("/get_info", routing::post(training::get_info))
        .route(
            "/forward_backward",
            routing::post(training::forward_backward).layer(DefaultBodyLimit::max(MAX_BODY)),
        )
        .route("/optim_step", routing::post(training::optim_step))
        .route("/retrieve_future", routing::post(futures::retrieve))
        .with_state(Services {
            sampling,
            training,
            shutdown,
        });
    Router::new().nest("/api/v1", api)
}
