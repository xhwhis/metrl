//! Model actors, CDC view, leases, worker clients, and peer gRPC.

mod error;
mod model;
mod registry;
mod sampling;
mod training;

pub use error::ServerError;
pub use model::own_models;
pub use registry::{Registration, Workers, registration};
pub use sampling::{SampleResult, Sampling};
pub use training::{NewModel, Training, TrainingResult};
