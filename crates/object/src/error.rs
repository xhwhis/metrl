#[derive(Debug, thiserror::Error)]
pub enum ObjectError {
    #[error("invalid object store configuration")]
    Config,
    /// A transport or server failure; the same request may succeed later.
    #[error("object store request failed")]
    Request,
    /// The object is missing or access is denied; repeating the request
    /// cannot succeed until the deployment changes.
    #[error("object store rejected the request")]
    Rejected,
}
