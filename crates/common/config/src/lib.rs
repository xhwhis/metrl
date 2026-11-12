//! Configuration files with `METRL_` environment overrides. Nested keys use
//! `__` as the separator, e.g. `METRL_HTTP_ADDR` or `METRL_ENGINE__NAME`.

use std::collections::BTreeMap;
use std::fmt;
use std::net::SocketAddr;
use std::path::Path;

use config::{Config, Environment, File, FileFormat};
use error_stack::{Report, ResultExt};
use metrl_types::Role;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer};

#[derive(Debug, thiserror::Error)]
#[error("failed to load config from {path}")]
pub struct ConfigError {
    path: String,
}

#[derive(Deserialize)]
pub struct ServerConfig {
    pub http_addr: SocketAddr,
    pub grpc_addr: SocketAddr,
    pub system_addr: SocketAddr,
    pub database_url: String,
    pub object: ObjectConfig,
}

/// Leaves out `database_url`, whose password may be inline, so a config can
/// be logged.
impl fmt::Debug for ServerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ServerConfig")
            .field("http_addr", &self.http_addr)
            .field("grpc_addr", &self.grpc_addr)
            .field("system_addr", &self.system_addr)
            .field("object", &self.object)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct WorkerConfig {
    pub grpc_addr: SocketAddr,
    /// Address servers dial; defaults to the bound gRPC address. Required when
    /// `grpc_addr` binds a wildcard address such as `0.0.0.0`.
    pub advertise_addr: Option<String>,
    /// Server gRPC endpoints, tried in order when registering.
    #[serde(deserialize_with = "comma_separated")]
    pub servers: Vec<String>,
    pub engine: EngineConfig,
    pub bases: Vec<BaseConfig>,
    pub object: ObjectConfig,
}

/// Where task inputs, outputs, and checkpoints live. Credentials come from
/// the environment, e.g. `METRL_OBJECT__SECRET_ACCESS_KEY`.
#[derive(Clone, Deserialize)]
pub struct ObjectConfig {
    /// `s3://bucket/prefix` or `file:///absolute/path`.
    pub url: String,
    /// S3-compatible endpoint such as `http://127.0.0.1:9000`; plain HTTP is
    /// allowed for it.
    pub endpoint: Option<String>,
    pub region: Option<String>,
    pub access_key_id: Option<String>,
    pub secret_access_key: Option<String>,
}

/// Leaves out the secret key, so a config can be logged.
impl fmt::Debug for ObjectConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ObjectConfig")
            .field("url", &self.url)
            .field("endpoint", &self.endpoint)
            .field("region", &self.region)
            .field("access_key_id", &self.access_key_id)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct EngineConfig {
    /// Selects the `metrl.engines.<name>` package.
    pub name: String,
    /// `[engine.<name>]` tables. The selected engine gets its own as is, so
    /// Rust never needs to know an engine's settings.
    #[serde(flatten)]
    pub sections: BTreeMap<String, serde_json::Map<String, serde_json::Value>>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BaseConfig {
    /// The name clients use.
    pub name: String,
    /// What the engine loads, such as a Hugging Face repository or a local
    /// directory; `None` loads `name`.
    pub path: Option<String>,
    pub roles: Vec<Role>,
}

/// A list from the file, or one comma-separated string from the environment,
/// such as `METRL_SERVERS=http://a:7581,http://b:7581`.
fn comma_separated<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum List {
        Items(Vec<String>),
        Joined(String),
    }
    Ok(match List::deserialize(deserializer)? {
        List::Items(items) => items,
        List::Joined(joined) => {
            joined
                .split(',')
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(str::to_owned)
                .collect()
        }
    })
}

pub fn load<T: DeserializeOwned>(path: &Path) -> Result<T, Report<ConfigError>> {
    let error = || {
        ConfigError {
            path: path.display().to_string(),
        }
    };
    Config::builder()
        .add_source(File::from(path).format(FileFormat::Toml))
        // Values stay strings, so a secret such as `00123456` is never read
        // as a number; typed keys parse when they are deserialized.
        .add_source(
            Environment::with_prefix("METRL")
                .prefix_separator("_")
                .separator("__"),
        )
        .build()
        .and_then(Config::try_deserialize)
        .change_context_lazy(error)
}
