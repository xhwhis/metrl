#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("invalid request: {0}")]
    Invalid(&'static str),
    #[error("resource not found")]
    NotFound,
    #[error("sampling result expired")]
    Gone,
    #[error("invalid API key")]
    Unauthorized,
    #[error("capacity exhausted")]
    Capacity,
    #[error("unavailable: {0}")]
    Unavailable(&'static str),
    /// A failure a retry cannot get past, such as a deployment whose object
    /// store denies access.
    #[error("internal error: {0}")]
    Internal(&'static str),
}

/// A Postgres failure as the API reports it; the SDK retries it.
pub(crate) const DATABASE: ServerError = ServerError::Unavailable("database unavailable");
