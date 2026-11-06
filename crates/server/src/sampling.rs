use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use error_stack::Report;
use metrl_meta::View;
use metrl_proto::worker::{SampleRequest, SampleResponse};
use metrl_types::{Base, KeyHash, ModelId, OrgId, RequestId, Role, SessionId};
use parking_lot::Mutex;
use tokio::sync::Notify;
use tokio::sync::futures::Notified;
use uuid::Uuid;

use crate::{ServerError, Workers};

/// Samples one org may hold, so one org cannot take every slot.
const MAX_SAMPLES_PER_ORG: usize = 1024;

/// Samples and expired-sample records this server holds across orgs.
const MAX_SAMPLES: usize = 16 * 1024;

/// Sessions one org may hold, so one org cannot take every slot.
const MAX_SESSIONS_PER_ORG: usize = 1024;

/// Sessions this server holds across orgs.
const MAX_SESSIONS: usize = 16 * 1024;

/// Sampling sessions one org may hold. One session can create any number,
/// so the session cap alone does not stop one org from taking every slot.
const MAX_SAMPLERS_PER_ORG: usize = 1024;

/// Sampling sessions this server holds across orgs.
const MAX_SAMPLERS: usize = 16 * 1024;

/// How long a sample stays unread before it expires.
const SAMPLE_TTL: Duration = Duration::from_secs(3600);

/// How long a finished sample stays after it is read, for a client whose
/// response was lost; after that it is gone (410), and the SDK resubmits.
const READ_SAMPLE_TTL: Duration = Duration::from_secs(60);

#[derive(Clone)]
pub enum SampleResult {
    Pending,
    Complete(SampleResponse),
    Failed,
}

struct Session {
    org: OrgId,
    expires: Instant,
}

struct SamplingSession {
    parent: SessionId,
    seq: u64,
    base: String,
    revision: String,
}

struct Sample {
    org: OrgId,
    model: ModelId,
    seq: u64,
    request: SampleRequest,
    result: SampleResult,
    expires: Instant,
}

struct ExpiredSample {
    org: OrgId,
    expires: Instant,
}

#[derive(Default)]
struct State {
    sessions: HashMap<SessionId, Session>,
    samplers: HashMap<ModelId, SamplingSession>,
    samples: HashMap<RequestId, Sample>,
    expired_samples: HashMap<RequestId, ExpiredSample>,
}

impl State {
    fn expire(&mut self) {
        let now = Instant::now();
        self.sessions.retain(|_, session| session.expires > now);
        self.samplers
            .retain(|_, sampler| self.sessions.contains_key(&sampler.parent));
        let expired = self
            .samples
            .extract_if(|_, sample| sample.expires <= now)
            .map(|(id, sample)| (id, sample.org));
        for (id, org) in expired {
            if self.expired_samples.len() >= MAX_SAMPLES {
                let oldest = self
                    .expired_samples
                    .iter()
                    .min_by_key(|(_, sample)| sample.expires)
                    .map(|(id, _)| *id)
                    .expect("expired sample map is non-empty at capacity");
                self.expired_samples.remove(&oldest);
            }
            self.expired_samples.insert(
                id,
                ExpiredSample {
                    org,
                    expires: now + SAMPLE_TTL,
                },
            );
        }
        self.expired_samples
            .retain(|_, sample| sample.expires > now);
    }
}

#[derive(Clone)]
pub struct Sampling {
    view: View,
    workers: Workers,
    state: Arc<Mutex<State>>,
    finished: Arc<Notify>,
}

impl Sampling {
    pub fn new(view: View, workers: Workers) -> Self {
        Self {
            view,
            workers,
            state: Arc::new(Mutex::new(State::default())),
            finished: Arc::new(Notify::new()),
        }
    }

    /// Resolves once any sample finishes after this call.
    pub fn finished(&self) -> Notified<'_> {
        self.finished.notified()
    }

    pub fn bases(&self) -> Vec<Base> {
        self.workers.bases()
    }

    /// Fails while the control state may be missing revocations.
    pub fn ensure_fresh(&self) -> Result<(), Report<ServerError>> {
        use error_stack::ResultExt;
        self.view
            .ensure_fresh()
            .change_context(ServerError::Unavailable("control state is stale"))
    }

    pub fn authenticate(&self, key: &str) -> Result<OrgId, Report<ServerError>> {
        self.ensure_fresh()?;
        self.view
            .org(&KeyHash::of(key))
            .ok_or_else(|| Report::new(ServerError::Unauthorized))
    }

    pub fn create_session(&self, org: OrgId) -> Result<SessionId, Report<ServerError>> {
        let mut state = self.state.lock();
        state.expire();
        let held = state.sessions.values().filter(|s| s.org == org).count();
        if held >= MAX_SESSIONS_PER_ORG || state.sessions.len() >= MAX_SESSIONS {
            return Err(Report::new(ServerError::Capacity));
        }
        let id = SessionId(Uuid::now_v7());
        state.sessions.insert(
            id,
            Session {
                org,
                expires: Instant::now() + Duration::from_secs(3600),
            },
        );
        drop(state);
        Ok(id)
    }

    pub fn heartbeat(&self, org: &OrgId, session: SessionId) -> Result<(), Report<ServerError>> {
        let mut state = self.state.lock();
        state.expire();
        let entry = state
            .sessions
            .get_mut(&session)
            .filter(|s| &s.org == org)
            .ok_or_else(|| Report::new(ServerError::NotFound))?;
        entry.expires = Instant::now() + Duration::from_secs(3600);
        drop(state);
        Ok(())
    }

    pub fn create_sampler(
        &self, org: &OrgId, session: SessionId, seq: u64, base: &str,
    ) -> Result<ModelId, Report<ServerError>> {
        self.heartbeat(org, session)?;
        let mut state = self.state.lock();
        // A retry returns the existing sampler even if its worker has left
        // since.
        if let Some((id, sampler)) = state
            .samplers
            .iter()
            .find(|(_, s)| s.parent == session && s.seq == seq)
        {
            if sampler.base != base {
                return Err(Report::new(ServerError::Invalid(
                    "sampling session retry changed base model",
                )));
            }
            return Ok(*id);
        }
        let known = self.workers.base(base, Role::Sampler).ok_or_else(|| {
            Report::new(ServerError::Invalid(
                "base model is not served by a registered worker",
            ))
        })?;
        let held = state
            .samplers
            .values()
            .filter(|s| state.sessions.get(&s.parent).is_some_and(|s| &s.org == org))
            .count();
        if held >= MAX_SAMPLERS_PER_ORG || state.samplers.len() >= MAX_SAMPLERS {
            return Err(Report::new(ServerError::Capacity));
        }
        let id = ModelId(Uuid::now_v7());
        state.samplers.insert(
            id,
            SamplingSession {
                parent: session,
                seq,
                base: known.name,
                revision: known.revision,
            },
        );
        drop(state);
        Ok(id)
    }

    pub fn sample(
        &self, org: &OrgId, model: ModelId, seq: u64, mut request: SampleRequest,
    ) -> Result<RequestId, Report<ServerError>> {
        use tracing::Instrument;
        let id = RequestId(Uuid::now_v7());
        let mut state = self.state.lock();
        state.expire();
        let (base, revision) = state
            .samplers
            .get(&model)
            .filter(|s| state.sessions.get(&s.parent).is_some_and(|s| &s.org == org))
            .map(|sampler| (sampler.base.clone(), sampler.revision.clone()))
            .ok_or_else(|| Report::new(ServerError::NotFound))?;
        request.model_id = model.0.to_string();
        if let Some((id, sample)) = state
            .samples
            .iter()
            .find(|(_, s)| &s.org == org && s.model == model && s.seq == seq)
        {
            if sample.request != request {
                return Err(Report::new(ServerError::Invalid(
                    "sample retry changed input",
                )));
            }
            return Ok(*id);
        }
        let held = state.samples.values().filter(|s| &s.org == org).count();
        if held >= MAX_SAMPLES_PER_ORG || state.samples.len() >= MAX_SAMPLES {
            return Err(Report::new(ServerError::Capacity));
        }
        state.samples.insert(
            id,
            Sample {
                org: org.clone(),
                model,
                seq,
                request: request.clone(),
                result: SampleResult::Pending,
                expires: Instant::now() + SAMPLE_TTL,
            },
        );
        drop(state);
        let worker = self
            .workers
            .pick(&base, &revision, Role::Sampler, id.0.as_u64_pair().1);
        let service = self.clone();
        tokio::spawn(
            async move {
                let sampled = match worker {
                    Ok(mut worker) => {
                        // A different process may listen at the address by now;
                        // it rejects a request addressed to another worker.
                        request.worker_id = worker.id.0.to_string();
                        worker
                            .client
                            .sample(request)
                            .await
                            .map_err(|status| status.to_string())
                    }
                    Err(report) => Err(format!("{report:?}")),
                };
                let result = match sampled {
                    Ok(response) => SampleResult::Complete(response.into_inner()),
                    Err(error) => {
                        tracing::error!(%error, "sample failed");
                        SampleResult::Failed
                    }
                };
                let mut state = service.state.lock();
                if let Some(sample) = state.samples.get_mut(&id) {
                    sample.result = result;
                }
                drop(state);
                service.finished.notify_waiters();
            }
            .instrument(tracing::info_span!("sample", request_id = %id.0)),
        );
        Ok(id)
    }

    /// Looks up one sample and leaves the rest to the write paths' sweep:
    /// every held `retrieve_future` calls this whenever any sample finishes,
    /// so sweeping here would cost held polls times held state.
    pub fn retrieve(
        &self, org: &OrgId, id: RequestId,
    ) -> Result<SampleResult, Report<ServerError>> {
        let now = Instant::now();
        let mut guard = self.state.lock();
        let state = &mut *guard;
        let result = match state.samples.get_mut(&id) {
            Some(sample) if sample.org != *org => Err(Report::new(ServerError::NotFound)),
            Some(sample) if sample.expires <= now => Err(Report::new(ServerError::Gone)),
            Some(sample) => {
                if !matches!(sample.result, SampleResult::Pending) {
                    sample.expires = sample.expires.min(now + READ_SAMPLE_TTL);
                }
                Ok(sample.result.clone())
            }
            None if state
                .expired_samples
                .get(&id)
                .is_some_and(|sample| sample.org == *org && sample.expires > now) =>
            {
                Err(Report::new(ServerError::Gone))
            }
            None => Err(Report::new(ServerError::NotFound)),
        };
        drop(guard);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn expired_sample_is_gone_to_owner_and_not_found_to_other_org() {
        let view = View::empty();
        let service = Sampling::new(view.clone(), Workers::new(view));
        let id = RequestId(Uuid::now_v7());
        let model = ModelId(Uuid::now_v7());
        service.state.lock().samples.insert(
            id,
            Sample {
                org: OrgId("alpha".to_owned()),
                model,
                seq: 0,
                request: SampleRequest::default(),
                result: SampleResult::Pending,
                expires: Instant::now()
                    .checked_sub(Duration::from_secs(1))
                    .expect("the test clock is after one second of uptime"),
            },
        );

        let owner = service.retrieve(&OrgId("alpha".to_owned()), id);
        assert!(
            matches!(
                owner,
                Err(report) if matches!(report.current_context(), ServerError::Gone)
            ),
            "the owner must be told that the sampling result expired"
        );
        let other = service.retrieve(&OrgId("beta".to_owned()), id);
        assert!(
            matches!(
                other,
                Err(report) if matches!(report.current_context(), ServerError::NotFound)
            ),
            "cross-org access to an expired result must remain hidden"
        );
    }
}
