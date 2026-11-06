//! Model actors: each model this server holds a lease on runs its task log
//! in one actor.

mod actor;
mod dispatch;
mod lease;

pub use lease::own_models;
