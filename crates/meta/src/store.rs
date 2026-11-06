use std::time::Duration;

use deadpool_postgres::{Manager, ManagerConfig, Pool, RecyclingMethod, Runtime, Timeouts};
use error_stack::{Report, ResultExt};
use metrl_types::{Base, KeyHash, OrgId, WorkerId};
use tokio_postgres::NoTls;
use tokio_postgres::types::Json;

use crate::MetaError;

/// Bounds every pool checkout. A link that freezes without a reset leaves
/// connections that are open but never answer, so each checkout first runs
/// an empty query and drops a connection that does not answer in time.
const POOL_TIMEOUTS: Timeouts = Timeouts {
    wait: Some(Duration::from_secs(5)),
    create: Some(Duration::from_secs(5)),
    recycle: Some(Duration::from_secs(2)),
};

/// The Postgres control plane. Every statement comes from `sql/`.
#[derive(Clone)]
pub struct Store {
    pool: Pool,
    /// Serves only the change follower, so that request load cannot starve
    /// or poison the connections that keep every view fresh.
    cdc_pool: Pool,
    config: tokio_postgres::Config,
}

pub fn connect(database_url: &str) -> Result<Store, Report<MetaError>> {
    use std::str::FromStr;
    let config = tokio_postgres::Config::from_str(database_url)
        .change_context(MetaError::Config)
        .attach("database_url is not a valid connection string")?;
    // The change stream opens its own connection from these parts, so none of
    // them may fall back to a default the pool resolves differently.
    let complete = matches!(config.get_hosts(), [tokio_postgres::config::Host::Tcp(_)])
        && config.get_user().is_some()
        && config.get_dbname().is_some();
    if !complete {
        return Err(Report::new(MetaError::Config)
            .attach("database_url must name a user, exactly one TCP host, and a database"));
    }
    Ok(Store {
        pool: pool(&config, 16)?,
        // One connection for heartbeats and one for a resync scan.
        cdc_pool: pool(&config, 2)?,
        config,
    })
}

fn pool(config: &tokio_postgres::Config, max_size: usize) -> Result<Pool, Report<MetaError>> {
    let manager = Manager::from_config(
        config.clone(),
        NoTls,
        ManagerConfig {
            recycling_method: RecyclingMethod::Verified,
        },
    );
    Pool::builder(manager)
        .max_size(max_size)
        .timeouts(POOL_TIMEOUTS)
        // Timeouts fail every checkout without a runtime to run them on.
        .runtime(Runtime::Tokio1)
        .build()
        .change_context(MetaError::Config)
}

impl Store {
    pub(crate) fn config(&self) -> &tokio_postgres::Config {
        &self.config
    }

    pub(crate) async fn client(&self) -> Result<deadpool_postgres::Client, Report<MetaError>> {
        self.pool.get().await.change_context(MetaError::Unavailable)
    }

    /// A connection for the change follower's heartbeats and scans.
    pub(crate) async fn cdc_client(&self) -> Result<deadpool_postgres::Client, Report<MetaError>> {
        self.cdc_pool
            .get()
            .await
            .change_context(MetaError::Unavailable)
    }

    /// Creates the schema in an empty database.
    pub async fn migrate(&self) -> Result<(), Report<MetaError>> {
        self.client()
            .await?
            .batch_execute(include_str!("../sql/schema.sql"))
            .await
            .change_context(MetaError::Query)
            .attach("apply sql/schema.sql")
    }

    /// Creates the org if needed and stores the key's hash.
    pub async fn create_api_key(&self, org: &OrgId, key: KeyHash) -> Result<(), Report<MetaError>> {
        let mut client = self.client().await?;
        let transaction = client
            .transaction()
            .await
            .change_context(MetaError::Query)?;
        transaction
            .execute(include_str!("../sql/insert_org.sql"), &[&org.0])
            .await
            .change_context(MetaError::Query)?;
        transaction
            .execute(
                include_str!("../sql/insert_api_key.sql"),
                &[&key.0.as_slice(), &org.0],
            )
            .await
            .change_context(MetaError::Query)?;
        transaction.commit().await.change_context(MetaError::Query)
    }

    /// Records a worker; a previous registration on the same address is
    /// replaced because that process is gone.
    pub async fn register_worker(
        &self, id: WorkerId, addr: &str, bases: &[Base],
    ) -> Result<(), Report<MetaError>> {
        let mut client = self.client().await?;
        let transaction = client
            .transaction()
            .await
            .change_context(MetaError::Query)?;
        transaction
            .execute(
                include_str!("../sql/delete_superseded_workers.sql"),
                &[&addr, &id.0],
            )
            .await
            .change_context(MetaError::Query)?;
        transaction
            .execute(
                include_str!("../sql/upsert_worker.sql"),
                &[&id.0, &addr, &Json(bases)],
            )
            .await
            .change_context(MetaError::Query)?;
        transaction.commit().await.change_context(MetaError::Query)
    }

    /// Writes a logical message that every change stream receives. It is
    /// flushed at once because walsenders only decode flushed WAL.
    pub(crate) async fn emit_heartbeat(
        &self, prefix: &str, content: &str,
    ) -> Result<(), Report<MetaError>> {
        self.cdc_client()
            .await?
            .execute(
                include_str!("../sql/emit_heartbeat.sql"),
                &[&prefix, &content],
            )
            .await
            .change_context(MetaError::Query)
            .map(|_| ())
    }

    /// Removes a worker only if it is still registered at `addr`.
    pub async fn remove_worker(&self, id: WorkerId, addr: &str) -> Result<(), Report<MetaError>> {
        self.client()
            .await?
            .execute(include_str!("../sql/delete_worker.sql"), &[&id.0, &addr])
            .await
            .change_context(MetaError::Query)
            .map(|_| ())
    }
}
