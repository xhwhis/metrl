//! Model leases and the writes they fence. A server owns a model while its
//! lease runs; every write an owner makes names its generation and is
//! rejected once a newer generation took the model over.

use std::collections::BTreeMap;
use std::time::Duration;

use error_stack::{Report, ResultExt};
use metrl_types::{Generation, ModelId, OrgId, Outcome, RequestId, Seq, ServerId, WorkerId};
use tokio_postgres::types::Json;

use crate::view::Columns;
use crate::{MetaError, Store};

/// A model this server owns, as of its last lease renewal.
#[derive(Clone, Debug)]
pub struct Lease {
    pub model: ModelId,
    org: OrgId,
    pub generation: Generation,
    pub applied: Seq,
    pub base: String,
    pub base_revision: String,
    pub lora_rank: u32,
    pub seed: u64,
    /// The trainer that holds the model's state, if any task has run.
    pub worker: Option<WorkerId>,
}

/// The checkpoint a `save_weights_for_sampler` task wrote. Its base, revision,
/// and rank are the model's.
pub struct NewCheckpoint<'a> {
    /// The save's request.
    pub id: RequestId,
    pub name: Option<&'a str>,
    pub prefix: &'a str,
    pub size_bytes: u64,
    pub ttl_seconds: Option<u64>,
    pub user_metadata: &'a BTreeMap<String, String>,
}

/// Bounds each owner write. A link that freezes mid-query would otherwise
/// hold the write until TCP gives up, while renewals on other connections
/// keep the lease and no other server can take the model over. Every write
/// it bounds is safe to repeat if it committed before the timeout.
const WRITE_TIMEOUT: Duration = Duration::from_secs(5);

impl Store {
    /// Extends this server's leases by `ttl`, takes over every model whose
    /// lease is free or expired, and returns every model it now owns.
    pub async fn renew_leases(
        &self, server: ServerId, ttl: Duration,
    ) -> Result<Vec<Lease>, Report<MetaError>> {
        let rows = self
            .client()
            .await?
            .query(
                include_str!("../sql/renew_leases.sql"),
                &[&server.0, &ttl.as_secs_f64()],
            )
            .await
            .change_context(MetaError::Query)?;
        rows.iter().map(lease).collect()
    }

    /// Records the trainer that holds the model's state. False once a newer
    /// generation owns the model.
    pub async fn record_worker(
        &self, lease: &Lease, worker: WorkerId,
    ) -> Result<bool, Report<MetaError>> {
        bounded(async {
            self.client()
                .await?
                .execute(
                    include_str!("../sql/record_worker.sql"),
                    &[
                        &lease.model.0,
                        &(lease.generation.0 as i64),
                        &worker.0,
                        &lease.org.0,
                    ],
                )
                .await
                .change_context(MetaError::Query)
                .map(|updated| updated > 0)
        })
        .await
    }

    /// Records the outcome of a task that ran and advances `applied` to it.
    /// False once a newer generation owns the model.
    pub async fn complete_task(
        &self, lease: &Lease, seq: Seq, outcome: &Outcome,
    ) -> Result<bool, Report<MetaError>> {
        self.fenced_write(
            include_str!("../sql/complete_task.sql"),
            lease,
            seq,
            outcome,
        )
        .await
    }

    /// Records a finished save with the checkpoint it wrote and advances
    /// `applied` to it. When another of the model's checkpoints holds the
    /// name, records `name_taken` instead and no checkpoint. False once a
    /// newer generation owns the model.
    pub async fn complete_save(
        &self, lease: &Lease, seq: Seq, saved: &Outcome, name_taken: &Outcome,
        checkpoint: &NewCheckpoint<'_>,
    ) -> Result<bool, Report<MetaError>> {
        // Sizes and ttls are stored as bigint; the casts round-trip.
        bounded(async {
            let row = self
                .client()
                .await?
                .query_one(
                    include_str!("../sql/complete_save.sql"),
                    &[
                        &lease.model.0,
                        &(lease.generation.0 as i64),
                        &(seq.0 as i64),
                        &Json(saved),
                        &Json(name_taken),
                        &lease.org.0,
                        &checkpoint.id.0,
                        &checkpoint.name,
                        &checkpoint.prefix,
                        &(checkpoint.size_bytes as i64),
                        &checkpoint.ttl_seconds.map(|ttl| ttl as i64),
                        &Json(checkpoint.user_metadata),
                    ],
                )
                .await
                .change_context(MetaError::Query)?;
            Columns::get(&row, "owned")
        })
        .await
    }

    /// Records a pending task as failed without running it. False once a
    /// newer generation owns the model.
    pub async fn fail_task(
        &self, lease: &Lease, seq: Seq, outcome: &Outcome,
    ) -> Result<bool, Report<MetaError>> {
        self.fenced_write(include_str!("../sql/fail_task.sql"), lease, seq, outcome)
            .await
    }

    async fn fenced_write(
        &self, statement: &str, lease: &Lease, seq: Seq, outcome: &Outcome,
    ) -> Result<bool, Report<MetaError>> {
        bounded(async {
            let row = self
                .client()
                .await?
                .query_one(
                    statement,
                    &[
                        &lease.model.0,
                        &(lease.generation.0 as i64),
                        &(seq.0 as i64),
                        &Json(outcome),
                        &lease.org.0,
                    ],
                )
                .await
                .change_context(MetaError::Query)?;
            Columns::get(&row, "owned")
        })
        .await
    }
}

async fn bounded<T>(
    write: impl Future<Output = Result<T, Report<MetaError>>>,
) -> Result<T, Report<MetaError>> {
    tokio::time::timeout(WRITE_TIMEOUT, write)
        .await
        .change_context(MetaError::Unavailable)
        .attach(format!("no answer within {}s", WRITE_TIMEOUT.as_secs()))?
}

// Generations, seqs, and seeds are stored as bigint; the casts round-trip.
fn lease(row: &impl Columns) -> Result<Lease, Report<MetaError>> {
    Ok(Lease {
        model: ModelId(row.get("id")?),
        org: OrgId(row.get("org_id")?),
        generation: Generation(row.get::<i64>("generation")? as u64),
        applied: Seq(row.get::<i64>("applied")? as u64),
        base: row.get("base")?,
        base_revision: row.get("base_revision")?,
        lora_rank: row.get::<i32>("lora_rank")? as u32,
        seed: row.get::<i64>("seed")? as u64,
        worker: row.get::<Option<_>>("worker_id")?.map(WorkerId),
    })
}
