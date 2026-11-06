#[derive(Debug, thiserror::Error)]
pub enum MetaError {
    #[error("invalid database configuration")]
    Config,
    #[error("database unavailable")]
    Unavailable,
    #[error("database query failed")]
    Query,
    #[error("change stream failed")]
    Cdc,
}
