use std::collections::{BTreeMap, HashMap};
use std::ops::Bound;
use std::time::{Duration, Instant, SystemTime};

use error_stack::Report;
use metrl_types::{Base, KeyHash, ModelId, OrgId, Outcome, RequestId, Seq, TaskInput, WorkerId};
use tokio::sync::watch;
use tokio_postgres::types::{FromSql, Json, Type};

use crate::MetaError;
use crate::pgoutput::{Relation, Value};

#[derive(Clone, Debug)]
pub struct Worker {
    pub id: WorkerId,
    pub addr: String,
    pub bases: Vec<Base>,
}

/// A task in its model's log.
#[derive(Clone, Debug)]
pub struct Task {
    pub model: ModelId,
    pub seq: Seq,
    pub request: RequestId,
    pub org: OrgId,
    pub input: TaskInput,
    /// `None` while the task is pending.
    pub outcome: Option<Outcome>,
    /// When the receiving server appended it, by that server's clock.
    pub created_at: SystemTime,
}

/// How far behind the database the view may be before readers must stop
/// trusting it.
const STALE_AFTER: Duration = Duration::from_secs(30);

#[derive(Clone, Copy)]
struct TaskKey {
    model: ModelId,
    seq: Seq,
}

/// The replicated control state, as of the last applied transaction.
#[derive(Default)]
pub(crate) struct State {
    keys: HashMap<KeyHash, OrgId>,
    workers: HashMap<WorkerId, Worker>,
    tasks: HashMap<ModelId, BTreeMap<Seq, Task>>,
    requests: HashMap<RequestId, TaskKey>,
    /// How many of each model's tasks have no outcome, so admission does not
    /// scan the whole log.
    pending: HashMap<ModelId, usize>,
    /// Every commit made before this instant has been applied; set by scans
    /// and heartbeats.
    pub(crate) fresh_at: Option<Instant>,
}

/// One row change, decoded from a scan or the change stream.
pub(crate) enum Change {
    PutKey { hash: KeyHash, org: OrgId },
    DeleteKey { hash: KeyHash },
    PutWorker(Worker),
    DeleteWorker { id: WorkerId },
    PutTask(Task),
    DeleteTask { model: ModelId, seq: Seq },
    ClearKeys,
    ClearWorkers,
    ClearTasks,
}

impl State {
    pub(crate) fn apply(&mut self, change: Change) {
        match change {
            Change::PutKey { hash, org } => {
                self.keys.insert(hash, org);
            }
            Change::DeleteKey { hash } => {
                self.keys.remove(&hash);
            }
            Change::PutWorker(worker) => {
                self.workers.insert(worker.id, worker);
            }
            Change::DeleteWorker { id } => {
                self.workers.remove(&id);
            }
            Change::PutTask(task) => {
                let model = task.model;
                let key = TaskKey {
                    model,
                    seq: task.seq,
                };
                let request = task.request;
                let pending = task.outcome.is_none();
                let replaced = self.tasks.entry(model).or_default().insert(task.seq, task);
                if let Some(replaced) = &replaced {
                    self.requests.remove(&replaced.request);
                }
                self.requests.insert(request, key);
                let count = self.pending.entry(model).or_default();
                // A replaced pending task was counted, so this cannot
                // underflow.
                *count = *count + usize::from(pending)
                    - usize::from(replaced.is_some_and(|task| task.outcome.is_none()));
            }
            Change::DeleteTask { model, seq } => {
                if let Some(task) = self.tasks.get_mut(&model).and_then(|log| log.remove(&seq)) {
                    self.requests.remove(&task.request);
                    if task.outcome.is_none() {
                        *self
                            .pending
                            .get_mut(&model)
                            .expect("a pending task in the log was counted") -= 1;
                    }
                }
            }
            Change::ClearKeys => self.keys.clear(),
            Change::ClearWorkers => self.workers.clear(),
            Change::ClearTasks => {
                self.tasks.clear();
                self.requests.clear();
                self.pending.clear();
            }
        }
    }
}

/// Read access to the replicated control state.
#[derive(Clone)]
pub struct View(pub(crate) watch::Receiver<State>);

impl View {
    /// A view of an empty control plane that never changes.
    pub fn empty() -> Self {
        Self(watch::channel(State::default()).1)
    }

    /// Fails once the view may be more than [`STALE_AFTER`] behind the
    /// database, because revoked keys and removed workers may still appear.
    pub fn ensure_fresh(&self) -> Result<(), Report<MetaError>> {
        let fresh_at = self.0.borrow().fresh_at;
        match fresh_at.map(|at| at.elapsed()) {
            Some(elapsed) if elapsed <= STALE_AFTER => Ok(()),
            Some(elapsed) => {
                Err(Report::new(MetaError::Unavailable).attach(format!(
                    "control state last confirmed {}s ago",
                    elapsed.as_secs()
                )))
            }
            None => Err(Report::new(MetaError::Unavailable).attach("control state never loaded")),
        }
    }

    /// Whether every commit made before `at` has been applied.
    pub fn applied_through(&self, at: Instant) -> bool {
        self.0
            .borrow()
            .fresh_at
            .is_some_and(|fresh_at| fresh_at >= at)
    }

    pub fn org(&self, key: &KeyHash) -> Option<OrgId> {
        self.0.borrow().keys.get(key).cloned()
    }

    pub fn workers(&self) -> Vec<Worker> {
        self.0.borrow().workers.values().cloned().collect()
    }

    /// The model's tasks after `seq`, in seq order.
    pub fn tasks_after(&self, model: ModelId, seq: Seq) -> Vec<Task> {
        self.0
            .borrow()
            .tasks
            .get(&model)
            .map_or_else(Vec::new, |log| {
                log.range((Bound::Excluded(seq), Bound::Unbounded))
                    .map(|(_, task)| task.clone())
                    .collect()
            })
    }

    pub fn task(&self, model: ModelId, seq: Seq) -> Option<Task> {
        self.0.borrow().tasks.get(&model)?.get(&seq).cloned()
    }

    pub fn task_by_request(&self, request: RequestId) -> Option<Task> {
        let state = self.0.borrow();
        let key = state.requests.get(&request)?;
        state.tasks.get(&key.model)?.get(&key.seq).cloned()
    }

    /// How many of the model's tasks have no outcome yet.
    pub fn pending_tasks(&self, model: ModelId) -> usize {
        self.0.borrow().pending.get(&model).copied().unwrap_or(0)
    }

    /// Resolves once the view has changed since this handle last waited.
    pub async fn changed(&mut self) {
        // The sender lives as long as the follower task, which never returns;
        // a view that can no longer change never resolves.
        if self.0.changed().await.is_err() {
            std::future::pending::<()>().await;
        }
    }
}

/// A row from either a query or the change stream.
pub(crate) trait Columns {
    fn get<'a, T: FromSql<'a>>(&'a self, name: &str) -> Result<T, Report<MetaError>>;
}

impl Columns for tokio_postgres::Row {
    fn get<'a, T: FromSql<'a>>(&'a self, name: &str) -> Result<T, Report<MetaError>> {
        use error_stack::ResultExt;
        self.try_get(name)
            .change_context(MetaError::Query)
            .attach_with(|| format!("column {name}"))
    }
}

pub(crate) struct ChangedRow<'r> {
    pub(crate) relation: &'r Relation,
    pub(crate) values: &'r [Value],
}

impl Columns for ChangedRow<'_> {
    fn get<'a, T: FromSql<'a>>(&'a self, name: &str) -> Result<T, Report<MetaError>> {
        let error =
            || Report::new(MetaError::Cdc).attach(format!("{}.{name}", self.relation.table));
        let index = self
            .relation
            .columns
            .iter()
            .position(|column| column.name == name)
            .ok_or_else(error)?;
        let oid = self.relation.columns[index].type_oid;
        let ty = Type::from_oid(oid)
            .filter(T::accepts)
            .ok_or_else(|| error().attach(format!("unexpected type oid {oid}")))?;
        match self.values.get(index) {
            Some(Value::Binary(bytes)) => T::from_sql(&ty, bytes),
            Some(Value::Null) => T::from_sql_null(&ty),
            // An omitted TOAST value cannot be applied; the stream resyncs.
            Some(Value::Unchanged) | None => return Err(error().attach("value not sent")),
        }
        .map_err(|cause| error().attach(cause.to_string()))
    }
}

pub(crate) fn key_hash(row: &impl Columns) -> Result<KeyHash, Report<MetaError>> {
    use error_stack::ResultExt;
    let bytes: &[u8] = row.get("key_hash")?;
    bytes
        .try_into()
        .map(KeyHash)
        .change_context(MetaError::Cdc)
        .attach("key_hash must be 32 bytes")
}

pub(crate) fn api_key(row: &impl Columns) -> Result<Change, Report<MetaError>> {
    Ok(Change::PutKey {
        hash: key_hash(row)?,
        org: OrgId(row.get("org_id")?),
    })
}

/// Seqs are stored as bigint; the cast round-trips.
pub(crate) fn task(row: &impl Columns) -> Result<Task, Report<MetaError>> {
    let Json(input) = row.get("input")?;
    let outcome: Option<Json<Outcome>> = row.get("outcome")?;
    Ok(Task {
        model: ModelId(row.get("model_id")?),
        seq: Seq(row.get::<i64>("seq")? as u64),
        request: RequestId(row.get("request_id")?),
        org: OrgId(row.get("org_id")?),
        input,
        outcome: outcome.map(|Json(outcome)| outcome),
        created_at: row.get("created_at")?,
    })
}

pub(crate) fn worker(row: &impl Columns) -> Result<Change, Report<MetaError>> {
    let Json(bases) = row.get("bases")?;
    Ok(Change::PutWorker(Worker {
        id: WorkerId(row.get("id")?),
        addr: row.get("addr")?,
        bases,
    }))
}
