//! Tinker-compatible HTTP API.

mod sampling;
mod service;
#[allow(clippy::doc_markdown, clippy::trivially_copy_pass_by_ref)]
pub mod tinker_public;

pub use service::router;
