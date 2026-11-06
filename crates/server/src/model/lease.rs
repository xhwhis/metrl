use std::collections::HashMap;
use std::time::{Duration, Instant};

use metrl_meta::{Lease, Store, View};
use metrl_types::{Generation, ModelId, ServerId};
use tokio::task::JoinHandle;

use crate::Workers;
use crate::model::actor;

/// How long a lease runs past its last renewal.
const LEASE_TTL: Duration = Duration::from_secs(10);

const RENEW_INTERVAL: Duration = Duration::from_secs(3);

/// Bounds each renewal, so a frozen connection cannot hold leases past the
/// fence.
const RENEW_TIMEOUT: Duration = Duration::from_secs(2);

/// Without a renewal sent within this long, this server stops its actors.
/// A renewal sent earlier finishes before the lease it extends could lapse:
/// `FENCE_AFTER + RENEW_TIMEOUT < LEASE_TTL`.
const FENCE_AFTER: Duration = Duration::from_secs(7);

/// The actor running one owned model, at the generation it started with.
struct Actor {
    generation: Generation,
    handle: JoinHandle<()>,
}

/// Renews this server's leases, takes over every model whose lease is free or
/// expired, and runs one actor per owned model. Runs for the life of the
/// process.
pub async fn own_models(server: ServerId, store: Store, view: View, workers: Workers) {
    let mut actors: HashMap<ModelId, Actor> = HashMap::new();
    let mut renewed: Option<Instant> = None;
    let mut interval = tokio::time::interval(RENEW_INTERVAL);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            // A renewal is never sent past the fence while actors still run.
            biased;
            () = fence(renewed) => {
                stop_all(&mut actors);
                renewed = None;
                continue;
            }
            _ = interval.tick() => {}
        }
        // An owner runs tasks only from a fresh view. Without one it lets its
        // leases lapse, so a server whose change stream works takes over
        // instead of the models waiting on this one.
        if let Err(report) = view.ensure_fresh() {
            tracing::warn!(error = ?report, "change stream is stale; not renewing leases");
            continue;
        }
        // The lease runs from when the database applies the renewal, which
        // is after this.
        let sent = Instant::now();
        let renewal = tokio::time::timeout(RENEW_TIMEOUT, store.renew_leases(server, LEASE_TTL));
        // The fence also holds while a renewal is in flight, so actors stop at
        // `FENCE_AFTER` even when the renewal that would extend it stalls.
        let renewal = tokio::select! {
            biased;
            () = fence(renewed) => {
                stop_all(&mut actors);
                renewed = None;
                continue;
            }
            renewal = renewal => renewal,
        };
        match renewal {
            Ok(Ok(leases)) => {
                renewed = Some(sent);
                reconcile(&mut actors, leases, &store, &view, &workers);
            }
            Ok(Err(report)) => tracing::warn!(error = ?report, "lease renewal failed"),
            Err(_) => tracing::warn!("lease renewal timed out"),
        }
    }
}

/// Resolves [`FENCE_AFTER`] after the last renewal was sent; never while no
/// renewal holds a lease.
async fn fence(renewed: Option<Instant>) {
    match renewed {
        Some(at) => {
            tokio::time::sleep_until(tokio::time::Instant::from_std(at + FENCE_AFTER)).await;
        }
        None => std::future::pending().await,
    }
}

fn stop_all(actors: &mut HashMap<ModelId, Actor>) {
    tracing::error!(
        models = actors.len(),
        "leases not renewed in time; stopping every model"
    );
    for (_, actor) in actors.drain() {
        actor.handle.abort();
    }
}

/// Keeps each actor whose generation still owns its model, stops the rest,
/// and starts an actor for every newly owned model.
fn reconcile(
    actors: &mut HashMap<ModelId, Actor>, leases: Vec<Lease>, store: &Store, view: &View,
    workers: &Workers,
) {
    use tracing::Instrument;
    let mut leases: HashMap<ModelId, Lease> = leases
        .into_iter()
        .map(|lease| (lease.model, lease))
        .collect();
    actors.retain(|model, actor| {
        let current = !actor.handle.is_finished()
            && leases
                .get(model)
                .is_some_and(|lease| lease.generation == actor.generation);
        if current {
            leases.remove(model);
        } else {
            actor.handle.abort();
        }
        current
    });
    for (model, lease) in leases {
        let span =
            tracing::info_span!("model", model_id = %model.0, generation = lease.generation.0);
        let generation = lease.generation;
        let handle = tokio::spawn(
            actor::run(lease, store.clone(), view.clone(), workers.clone()).instrument(span),
        );
        actors.insert(model, Actor { generation, handle });
    }
}
