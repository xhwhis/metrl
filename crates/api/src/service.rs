use axum::routing::{get, post};
use axum::{Json, Router};
use metrl_server::Sampling;
use serde::Serialize;

use crate::sampling;

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
}

async fn healthz() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

pub fn router(sampling: Sampling) -> Router {
    let api = Router::new()
        .route("/healthz", get(healthz))
        .route("/client/config", post(sampling::config))
        .route("/create_session", post(sampling::create_session))
        .route("/session_heartbeat", post(sampling::heartbeat))
        .route("/create_sampling_session", post(sampling::create_sampler))
        .route("/get_server_capabilities", get(sampling::capabilities))
        .route("/asample", post(sampling::sample))
        .route("/retrieve_future", post(sampling::retrieve))
        .with_state(sampling);
    Router::new().nest("/api/v1", api)
}
