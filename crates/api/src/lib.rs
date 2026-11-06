//! Tinker-compatible HTTP API.

mod error;
mod extract;
mod futures;
mod sampling;
mod service;
#[allow(clippy::doc_markdown, clippy::trivially_copy_pass_by_ref)]
pub mod tinker_public;
mod training;

pub use service::router;
