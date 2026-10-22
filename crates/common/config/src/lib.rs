//! Configuration files with `METRL_` environment overrides. Nested keys use
//! `__` as the separator, e.g. `METRL_HTTP_ADDR` or `METRL_ENGINE__MLX__MODEL`.

use std::net::SocketAddr;
use std::path::Path;

use config::{Config, Environment, File, FileFormat};
use error_stack::{Report, ResultExt};
use serde::Deserialize;
use serde::de::DeserializeOwned;

#[derive(Debug, thiserror::Error)]
#[error("failed to load config from {path}")]
pub struct ConfigError {
    path: String,
}

#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    pub http_addr: SocketAddr,
    pub system_addr: SocketAddr,
}

pub fn load<T: DeserializeOwned>(path: &Path) -> Result<T, Report<ConfigError>> {
    let error = || {
        ConfigError {
            path: path.display().to_string(),
        }
    };
    Config::builder()
        .add_source(File::from(path).format(FileFormat::Toml))
        .add_source(
            Environment::with_prefix("METRL")
                .prefix_separator("_")
                .separator("__"),
        )
        .build()
        .and_then(Config::try_deserialize)
        .change_context_lazy(error)
}
