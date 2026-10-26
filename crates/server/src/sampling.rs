use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use error_stack::Report;
use metrl_proto::worker::worker_service_client::WorkerServiceClient;
use metrl_proto::worker::{Base, DescribeRequest, SampleRequest, SampleResponse};
use metrl_types::{ModelId, OrgId, RequestId, SessionId};
use tonic::transport::{Channel, Endpoint};
use uuid::Uuid;

const MAX_SAMPLES: usize = 1024;

#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("invalid request: {0}")]
    Invalid(&'static str),
    #[error("resource not found")]
    NotFound,
    #[error("sampling result expired")]
    Gone,
    #[error("invalid API key")]
    Unauthorized,
    #[error("sampling capacity exhausted")]
    Capacity,
    #[error("worker unavailable")]
    Unavailable,
}

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
                    expires: now + Duration::from_secs(3600),
                },
            );
        }
        self.expired_samples
            .retain(|_, sample| sample.expires > now);
    }
}

#[derive(Clone)]
pub struct Sampling {
    worker: WorkerServiceClient<Channel>,
    keys: Arc<HashMap<String, OrgId>>,
    state: Arc<Mutex<State>>,
}

impl Sampling {
    pub fn new(endpoint: &str, auth_keys: &str) -> Result<Self, Report<ServerError>> {
        use error_stack::ResultExt;
        let keys: HashMap<_, _> = auth_keys
            .split(',')
            .map(|entry| {
                let (org, key) = entry
                    .split_once(':')
                    .filter(|(org, key)| !org.is_empty() && !key.is_empty())
                    .ok_or_else(|| {
                        Report::new(ServerError::Invalid(
                            "METRL_AUTH_KEYS must contain org:key pairs",
                        ))
                    })?;
                Ok((key.to_owned(), OrgId(org.to_owned())))
            })
            .collect::<Result<_, Report<ServerError>>>()?;
        let channel = Endpoint::from_shared(endpoint.to_owned())
            .change_context(ServerError::Invalid("invalid worker endpoint"))?
            .connect_timeout(Duration::from_secs(3))
            .timeout(Duration::from_secs(30))
            .connect_lazy();
        Ok(Self {
            worker: WorkerServiceClient::new(channel),
            keys: Arc::new(keys),
            state: Arc::new(Mutex::new(State::default())),
        })
    }

    pub fn authenticate(&self, key: &str) -> Result<OrgId, Report<ServerError>> {
        self.keys
            .get(key)
            .cloned()
            .ok_or_else(|| Report::new(ServerError::Unauthorized))
    }

    pub async fn bases(&self) -> Result<Vec<Base>, Report<ServerError>> {
        use error_stack::ResultExt;
        Ok(self
            .worker
            .clone()
            .describe(DescribeRequest {})
            .await
            .change_context(ServerError::Unavailable)?
            .into_inner()
            .bases)
    }

    pub fn create_session(&self, org: OrgId) -> Result<SessionId, Report<ServerError>> {
        let mut state = self
            .state
            .lock()
            .expect("sampling state is never locked across engine or user code");
        state.expire();
        if state.sessions.len() >= 4096 {
            return Err(Report::new(ServerError::Capacity));
        }
        let id = SessionId(Uuid::new_v4());
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
        let mut state = self
            .state
            .lock()
            .expect("sampling state is never locked across engine or user code");
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

    pub async fn create_sampler(
        &self, org: &OrgId, session: SessionId, seq: u64, base: String,
    ) -> Result<ModelId, Report<ServerError>> {
        self.heartbeat(org, session)?;

        {
            let state = self
                .state
                .lock()
                .expect("sampling state is never locked across engine or user code");
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
        }

        if !self.bases().await?.iter().any(|b| b.name == base) {
            return Err(Report::new(ServerError::Invalid(
                "base model is not served by a live worker",
            )));
        }
        let mut state = self
            .state
            .lock()
            .expect("sampling state is never locked across engine or user code");

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
        if state.samplers.len() >= 4096 {
            return Err(Report::new(ServerError::Capacity));
        }
        let id = ModelId(Uuid::new_v4());
        state.samplers.insert(
            id,
            SamplingSession {
                parent: session,
                seq,
                base,
            },
        );
        drop(state);
        Ok(id)
    }

    pub fn sample(
        &self, org: &OrgId, model: ModelId, seq: u64, mut request: SampleRequest,
    ) -> Result<RequestId, Report<ServerError>> {
        use tracing::Instrument;
        let mut state = self
            .state
            .lock()
            .expect("sampling state is never locked across engine or user code");
        state.expire();
        state
            .samplers
            .get(&model)
            .filter(|s| state.sessions.get(&s.parent).is_some_and(|s| &s.org == org))
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
        if state.samples.len() >= MAX_SAMPLES {
            return Err(Report::new(ServerError::Capacity));
        }
        let id = RequestId(Uuid::new_v4());
        state.samples.insert(
            id,
            Sample {
                org: org.clone(),
                model,
                seq,
                request: request.clone(),
                result: SampleResult::Pending,
                expires: Instant::now() + Duration::from_secs(3600),
            },
        );
        drop(state);
        let service = self.clone();
        tokio::spawn(
            async move {
                let result = match service.worker.clone().sample(request).await {
                    Ok(response) => SampleResult::Complete(response.into_inner()),
                    Err(error) => {
                        tracing::error!(%error, request_id = %id.0, "sample failed");
                        SampleResult::Failed
                    }
                };
                let mut state = service
                    .state
                    .lock()
                    .expect("sampling state is never locked across engine or user code");
                if let Some(sample) = state.samples.get_mut(&id) {
                    sample.result = result;
                }
            }
            .instrument(tracing::info_span!("sample", request_id = %id.0)),
        );
        Ok(id)
    }

    pub fn retrieve(
        &self, org: &OrgId, id: RequestId,
    ) -> Result<SampleResult, Report<ServerError>> {
        let mut state = self
            .state
            .lock()
            .expect("sampling state is never locked across engine or user code");
        state.expire();
        let result = if let Some(sample) = state.samples.get(&id) {
            sample
                .org
                .eq(org)
                .then(|| sample.result.clone())
                .ok_or_else(|| Report::new(ServerError::NotFound))
        } else if state
            .expired_samples
            .get(&id)
            .is_some_and(|sample| sample.org == *org)
        {
            Err(Report::new(ServerError::Gone))
        } else {
            Err(Report::new(ServerError::NotFound))
        };
        drop(state);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn expired_sample_is_gone_to_owner_and_not_found_to_other_org() {
        let service = Sampling::new("http://127.0.0.1:1", "alpha:key")
            .expect("test endpoint and auth key are valid");
        let id = RequestId(Uuid::new_v4());
        let model = ModelId(Uuid::new_v4());
        service
            .state
            .lock()
            .expect("sampling state is never locked across engine or user code")
            .samples
            .insert(
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
