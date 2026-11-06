//! Shared domain vocabulary: identifiers and kinds.

mod base;
mod ids;
mod key;
mod model;
mod task;

pub use base::{Base, Role};
pub use ids::{ModelId, OrgId, RequestId, ServerId, SessionId, WorkerId};
pub use key::KeyHash;
pub use model::Model;
pub use task::{AdamParams, ErrorCategory, Generation, Outcome, Seq, TaskInput};
