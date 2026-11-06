#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    /// The request cannot run as given; the message is shown to the user.
    #[error("{0}")]
    Invalid(String),
    /// A newer generation or a later seq has run on this worker.
    #[error("superseded by a newer owner or a later task")]
    Superseded,
    /// The generation has not bound the model on this worker.
    #[error("model is not bound by this generation")]
    Unbound,
    /// The model is busy or the object store failed in a way a retry may get
    /// past; the same call can be retried.
    #[error("worker is busy or cannot reach the object store")]
    Unavailable,
    #[error("worker operation failed")]
    Failed,
}
