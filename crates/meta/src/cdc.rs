//! Follows Postgres through logical replication and keeps a [`View`] current.
//!
//! Each run creates a temporary slot that exports a snapshot, scans the
//! published tables in that snapshot, then streams changes committed after it.
//! The slot dies with the connection, so a crashed server leaves no WAL behind;
//! after a disconnect the next run rebuilds the view from a fresh scan. A run
//! that reads none of its own [`heartbeat`]s for [`heartbeat::TIMEOUT`] is
//! treated as disconnected.
//!
//! Every run names a new slot. After a link is lost without a reset, Postgres
//! keeps the old walsender and its slot until `wal_sender_timeout`; reusing
//! the name would fail until then.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use error_stack::Report;
use metrl_types::{ModelId, Seq, ServerId, WorkerId};
use pgwire_replication::{ReplicationClient, ReplicationConfig, ReplicationEvent};
use tokio::sync::watch;
use tokio_postgres::config::Host;

use crate::heartbeat::{self, Heartbeat};
use crate::pgoutput::{self, Message, Relation, Value};
use crate::view::{self, Change, ChangedRow, State};
use crate::{MetaError, Store, View};

/// Creating the slot or scanning still running after this long is stuck, for
/// example on a frozen link, and the run gives up. Neither connecting nor a
/// replication command times out by itself. Creating the slot also waits for
/// every transaction running in the cluster to end, which can take longer
/// than this; see the `lock_timeout` in [`replication_config`].
const SETUP_TIMEOUT: Duration = Duration::from_secs(15);

/// Handle to the background change follower.
pub struct Cdc {
    view: View,
    ready: watch::Receiver<bool>,
}

impl Cdc {
    pub fn view(&self) -> View {
        self.view.clone()
    }

    /// Resolves once the view reflects a scan of the database.
    pub async fn ready(&mut self) {
        // The sender lives as long as the follower task, which never returns.
        self.ready.wait_for(|ready| *ready).await.ok();
    }
}

/// Starts following the database in the background.
pub fn follow(store: Store, server: ServerId) -> Cdc {
    use tracing::Instrument;
    let (state, view) = watch::channel(State::default());
    let (ready, ready_rx) = watch::channel(false);
    let heartbeat = Heartbeat::new(server);
    tokio::spawn({
        let heartbeat = heartbeat.clone();
        let store = store.clone();
        async move { heartbeat.emit(&store).await }
            .instrument(tracing::info_span!("heartbeat", server = %server.0))
    });
    tokio::spawn(
        async move {
            let mut attempt = 0_u64;
            loop {
                attempt += 1;
                let slot = format!("metrl_{}_{attempt}", server.0.simple());
                if let Err(report) = run(&store, &slot, &heartbeat, &state, &ready).await {
                    tracing::warn!(error = ?report, "change stream ended, resyncing");
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
        .instrument(tracing::info_span!("cdc", server = %server.0)),
    );
    Cdc {
        view: View(view),
        ready: ready_rx,
    }
}

async fn run(
    store: &Store, slot: &str, heartbeat: &Heartbeat, state: &watch::Sender<State>,
    ready: &watch::Sender<bool>,
) -> Result<(), Report<MetaError>> {
    use error_stack::ResultExt;
    let mut stream = ReplicationClient::connect_gated(replication_config(store.config(), slot))
        .await
        .change_context(MetaError::Cdc)?;
    let result = replicate(store, slot, heartbeat, state, ready, &mut stream).await;
    // Dropping the client only asks its worker to stop, and a worker blocked
    // on a frozen link never notices. The slot is temporary, so there is
    // nothing to stop gracefully.
    stream.abort();
    result
}

/// Creates the slot, scans its snapshot, then applies streamed changes until
/// the stream fails.
async fn replicate(
    store: &Store, slot: &str, heartbeat: &Heartbeat, state: &watch::Sender<State>,
    ready: &watch::Sender<bool>, stream: &mut ReplicationClient,
) -> Result<(), Report<MetaError>> {
    use error_stack::ResultExt;
    // The slot's snapshot includes every commit made before it was requested.
    let requested_at = Instant::now();
    let create_slot = format!(include_str!("../sql/create_slot.sql"), slot = slot);
    let created = tokio::time::timeout(SETUP_TIMEOUT, stream.simple_query(&create_slot))
        .await
        .change_context(MetaError::Cdc)
        .attach("create slot timed out")?
        .change_context(MetaError::Cdc)
        .attach_with(|| format!("create slot {slot}"))?;
    let snapshot = created
        .first()
        .and_then(|row| row.get(2).cloned().flatten())
        .ok_or_else(|| Report::new(MetaError::Cdc).attach("slot exported no snapshot"))?;
    let snapshot = str::from_utf8(&snapshot).change_context(MetaError::Cdc)?;

    let mut scanned = State::default();
    // A scan stuck on a pooled connection leaves it to the next checkout,
    // which drops it.
    tokio::time::timeout(SETUP_TIMEOUT, scan(store, snapshot, &mut scanned))
        .await
        .change_context(MetaError::Cdc)
        .attach("scan timed out")??;
    scanned.fresh_at = Some(requested_at);
    state.send_replace(scanned);
    ready.send_replace(true);
    tracing::info!(%slot, "view loaded, streaming changes");

    stream.start().await.change_context(MetaError::Cdc)?;
    let mut relations = HashMap::new();
    let mut pending = Vec::new();
    let mut deadline = tokio::time::Instant::now() + heartbeat::TIMEOUT;
    loop {
        let Ok(event) = tokio::time::timeout_at(deadline, stream.recv()).await else {
            return Err(Report::new(MetaError::Cdc).attach(format!(
                "no heartbeat within {}s",
                heartbeat::TIMEOUT.as_secs()
            )));
        };
        let Some(event) = event.change_context(MetaError::Cdc)? else {
            break;
        };
        match event {
            ReplicationEvent::XLogData { data, .. } => {
                decode(data, &mut relations, &mut pending)?;
            }
            ReplicationEvent::Commit { end_lsn, .. } => {
                state.send_modify(|state| pending.drain(..).for_each(|change| state.apply(change)));
                stream.update_applied_lsn(end_lsn);
            }
            // Transactions that touch no published table send no Commit, so
            // only keepalives let the slot release their WAL. A keepalive
            // never acks an unapplied commit: the walsender sends committed
            // output before any later keepalive on this ordered stream.
            ReplicationEvent::KeepAlive { wal_end, .. } => stream.update_applied_lsn(wal_end),
            ReplicationEvent::Message {
                prefix, content, ..
            } => {
                if let Some(emitted_at) = heartbeat.emitted_at(&prefix, &content) {
                    state
                        .send_modify(|state| state.fresh_at = state.fresh_at.max(Some(emitted_at)));
                    deadline = tokio::time::Instant::now() + heartbeat::TIMEOUT;
                }
            }
            ReplicationEvent::Begin { .. } | ReplicationEvent::StoppedAt { .. } => {}
        }
    }
    Err(Report::new(MetaError::Cdc).attach("change stream closed"))
}

fn replication_config(config: &tokio_postgres::Config, slot: &str) -> ReplicationConfig {
    let [Host::Tcp(host)] = config.get_hosts() else {
        unreachable!("connect accepts only a database_url with exactly one TCP host");
    };
    let password = config.get_password().map_or_else(String::new, |password| {
        String::from_utf8_lossy(password).into_owned()
    });
    ReplicationConfig::new(
        host.as_str(),
        config
            .get_user()
            .expect("connect accepts only a database_url with a user"),
        password,
        config
            .get_dbname()
            .expect("connect accepts only a database_url with a dbname"),
        slot,
        "metrl",
    )
    .with_port(config.get_ports().first().copied().unwrap_or(5432))
    .with_binary(true)
    .with_application_name(slot)
    // Acks release WAL; the 10s default leaves it pinned for no benefit.
    .with_status_interval(Duration::from_secs(1))
    // Postgres drops a walsender whose client stopped answering, and with it
    // the slot, after `wal_sender_timeout` instead of the 60s default. That
    // only runs once streaming starts. Before it, the session idles in the
    // slot's snapshot transaction while the scan runs, so a session abandoned
    // there ends after `idle_in_transaction_session_timeout`, which must
    // outlast `SETUP_TIMEOUT`; one abandoned before creating the slot ends
    // after `idle_session_timeout` instead of holding a walsender. Creating
    // the slot waits on running transactions without reading the socket, so
    // without `lock_timeout` each run that gives up during a long transaction
    // leaves a walsender and its slot behind until that transaction ends.
    .with_options(
        "-c wal_sender_timeout=10s -c idle_in_transaction_session_timeout=30s \
         -c idle_session_timeout=30s -c lock_timeout=10s",
    )
}

async fn scan(store: &Store, snapshot: &str, state: &mut State) -> Result<(), Report<MetaError>> {
    use error_stack::ResultExt;
    let mut client = store.cdc_client().await?;
    let transaction = client
        .build_transaction()
        .isolation_level(tokio_postgres::IsolationLevel::RepeatableRead)
        .read_only(true)
        .start()
        .await
        .change_context(MetaError::Query)?;
    transaction
        .batch_execute(&format!(
            include_str!("../sql/set_snapshot.sql"),
            snapshot = snapshot
        ))
        .await
        .change_context(MetaError::Query)?;
    let keys = transaction
        .query(include_str!("../sql/select_api_keys.sql"), &[])
        .await
        .change_context(MetaError::Query)?;
    let workers = transaction
        .query(include_str!("../sql/select_workers.sql"), &[])
        .await
        .change_context(MetaError::Query)?;
    let tasks = transaction
        .query(include_str!("../sql/select_tasks.sql"), &[])
        .await
        .change_context(MetaError::Query)?;
    transaction
        .commit()
        .await
        .change_context(MetaError::Query)?;
    for row in &keys {
        state.apply(view::api_key(row)?);
    }
    for row in &workers {
        state.apply(view::worker(row)?);
    }
    for row in &tasks {
        state.apply(Change::PutTask(view::task(row)?));
    }
    Ok(())
}

fn decode(
    data: bytes::Bytes, relations: &mut HashMap<u32, Relation>, pending: &mut Vec<Change>,
) -> Result<(), Report<MetaError>> {
    let relation = |id: u32| {
        relations
            .get(&id)
            .ok_or_else(|| Report::new(MetaError::Cdc).attach(format!("unknown relation {id}")))
    };
    match pgoutput::decode(data)? {
        Message::Relation(relation) => {
            relations.insert(relation.id, relation);
        }
        Message::Insert { relation: id, row } => {
            pending.push(put(relation(id)?, &row)?);
        }
        Message::Update {
            relation: id,
            old_key,
            row,
        } => {
            let relation = relation(id)?;
            // A changed key moves the row: the old key must stop resolving.
            if let Some(key) = old_key {
                pending.push(delete(relation, &key)?);
            }
            pending.push(put(relation, &row)?);
        }
        Message::Delete { relation: id, key } => {
            pending.push(delete(relation(id)?, &key)?);
        }
        Message::Truncate { relations: ids } => {
            for id in ids {
                pending.push(match relation(id)?.table.as_str() {
                    "api_keys" => Change::ClearKeys,
                    "workers" => Change::ClearWorkers,
                    "tasks" => Change::ClearTasks,
                    table => return Err(unpublished(table)),
                });
            }
        }
        Message::Ignored => {}
    }
    Ok(())
}

/// The change that stores `row`, a full new tuple.
fn put(relation: &Relation, row: &[Value]) -> Result<Change, Report<MetaError>> {
    let row = ChangedRow {
        relation,
        values: row,
    };
    match relation.table.as_str() {
        "api_keys" => view::api_key(&row),
        "workers" => view::worker(&row),
        "tasks" => view::task(&row).map(Change::PutTask),
        table => Err(unpublished(table)),
    }
}

/// The change that removes the row identified by `key`, a key or old tuple.
fn delete(relation: &Relation, key: &[Value]) -> Result<Change, Report<MetaError>> {
    use crate::view::Columns;
    let row = ChangedRow {
        relation,
        values: key,
    };
    match relation.table.as_str() {
        "api_keys" => {
            Ok(Change::DeleteKey {
                hash: view::key_hash(&row)?,
            })
        }
        "workers" => {
            Ok(Change::DeleteWorker {
                id: WorkerId(row.get("id")?),
            })
        }
        "tasks" => {
            Ok(Change::DeleteTask {
                model: ModelId(row.get("model_id")?),
                seq: Seq(row.get::<i64>("seq")? as u64),
            })
        }
        table => Err(unpublished(table)),
    }
}

fn unpublished(table: &str) -> Report<MetaError> {
    Report::new(MetaError::Cdc).attach(format!("change for unpublished table {table}"))
}
