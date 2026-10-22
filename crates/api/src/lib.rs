//! Tinker-compatible HTTP API.

mod service;
#[allow(clippy::doc_markdown)]
pub mod tinker_public;

use axum::Router;

pub fn router() -> Router {
    Router::new().nest("/api/v1", service::router())
}
