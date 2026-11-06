//! Proves that a change stream is live and caught up.
//!
//! A link that freezes without a reset looks like an idle stream: no error
//! arrives and the view silently stops changing. So each server writes a
//! logical message every [`INTERVAL`] and expects to read it back on its own
//! stream. Messages are decoded in WAL order, so reading one back means every
//! commit made before it was written has been applied.

use std::time::{Duration, Instant};

use metrl_types::ServerId;

use crate::Store;

const INTERVAL: Duration = Duration::from_secs(5);

/// A stream that has read none of its own heartbeats for this long is dead;
/// it tolerates two missed ones.
pub(crate) const TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Clone)]
pub(crate) struct Heartbeat {
    /// Every server's slot receives every server's heartbeats; the prefix
    /// tells this server's apart.
    prefix: String,
    /// Content is the microseconds since `origin`, so the reader can recover
    /// when a heartbeat was written.
    origin: Instant,
}

impl Heartbeat {
    pub(crate) fn new(server: ServerId) -> Self {
        Self {
            prefix: format!("metrl.heartbeat.{}", server.0.simple()),
            origin: Instant::now(),
        }
    }

    /// Writes a heartbeat every [`INTERVAL`], forever.
    pub(crate) async fn emit(&self, store: &Store) {
        let mut ticks = tokio::time::interval(INTERVAL);
        loop {
            ticks.tick().await;
            let content = self.origin.elapsed().as_micros().to_string();
            match tokio::time::timeout(INTERVAL, store.emit_heartbeat(&self.prefix, &content)).await
            {
                Ok(Ok(())) => {}
                Ok(Err(report)) => tracing::warn!(error = ?report, "heartbeat not written"),
                Err(_) => tracing::warn!("heartbeat write timed out"),
            }
        }
    }

    /// When this server wrote the heartbeat; `None` for another server's.
    pub(crate) fn emitted_at(&self, prefix: &str, content: &[u8]) -> Option<Instant> {
        if prefix != self.prefix {
            return None;
        }
        let micros = str::from_utf8(content).ok()?.parse().ok()?;
        // Any role can write a logical message with this prefix. A future
        // instant would keep the view fresh long after the stream stopped.
        self.origin
            .checked_add(Duration::from_micros(micros))
            .filter(|at| *at <= Instant::now())
    }
}
