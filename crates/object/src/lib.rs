//! Object store keys, checkpoint manifests, and sweep.

mod error;
mod keys;
mod store;

pub use error::ObjectError;
pub use keys::{task_input, task_output};
pub use store::{Store, open};
