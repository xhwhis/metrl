use std::sync::Arc;

use bytes::Bytes;
use error_stack::{Report, ResultExt};
use metrl_config::ObjectConfig;
use object_store::ObjectStore;
use object_store::path::Path;
use object_store::prefix::PrefixStore;
use url::Url;

use crate::ObjectError;

/// The object store, rooted at the configured URL's path.
#[derive(Clone)]
pub struct Store(Arc<dyn ObjectStore>);

pub fn open(config: &ObjectConfig) -> Result<Store, Report<ObjectError>> {
    let url = Url::parse(&config.url)
        .change_context(ObjectError::Config)
        .attach_with(|| format!("url {}", config.url))?;
    let options = [
        ("endpoint", &config.endpoint),
        ("region", &config.region),
        ("access_key_id", &config.access_key_id),
        ("secret_access_key", &config.secret_access_key),
    ];
    let mut options: Vec<(&str, String)> = options
        .into_iter()
        .filter_map(|(key, value)| value.clone().map(|value| (key, value)))
        .collect();
    if config
        .endpoint
        .as_ref()
        .is_some_and(|endpoint| endpoint.starts_with("http://"))
    {
        options.push(("allow_http", "true".to_owned()));
    }
    let (store, root) = object_store::parse_url_opts(&url, options)
        .change_context(ObjectError::Config)
        .attach_with(|| format!("url {}", config.url))?;
    Ok(Store(Arc::new(PrefixStore::new(store, root))))
}

impl Store {
    pub async fn put(&self, key: &str, bytes: Bytes) -> Result<(), Report<ObjectError>> {
        use object_store::ObjectStoreExt;
        self.0
            .put(&Path::from(key), bytes.into())
            .await
            .map_err(report)
            .attach_with(|| format!("put {key}"))
            .map(|_| ())
    }

    pub async fn get(&self, key: &str) -> Result<Bytes, Report<ObjectError>> {
        use object_store::ObjectStoreExt;
        let attach = || format!("get {key}");
        self.0
            .get(&Path::from(key))
            .await
            .map_err(report)
            .attach_with(attach)?
            .bytes()
            .await
            .map_err(report)
            .attach_with(attach)
    }
}

fn report(error: object_store::Error) -> Report<ObjectError> {
    let context = match error {
        object_store::Error::NotFound { .. }
        | object_store::Error::PermissionDenied { .. }
        | object_store::Error::Unauthenticated { .. } => ObjectError::Rejected,
        _ => ObjectError::Request,
    };
    Report::new(error).change_context(context)
}
