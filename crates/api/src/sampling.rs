use axum::Json;
use axum::extract::{FromRequest, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use error_stack::Report;
use metrl_proto::worker::SampleRequest;
use metrl_server::{SampleResult, Sampling, ServerError};
use metrl_types::{ModelId, OrgId, RequestId, SessionId};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

pub(crate) struct ApiJson<T>(T);

impl<T: DeserializeOwned, S: Send + Sync> FromRequest<S> for ApiJson<T> {
    type Rejection = Response;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        Json::<T>::from_request(request, state)
            .await
            .map(|Json(value)| Self(value))
            .map_err(|error| {
                let status = match error.status() {
                    StatusCode::UNPROCESSABLE_ENTITY => StatusCode::BAD_REQUEST,
                    status => status,
                };
                (status, Json(json!({"error": error.body_text()}))).into_response()
            })
    }
}

pub(crate) struct ApiError(Report<ServerError>);

impl From<Report<ServerError>> for ApiError {
    fn from(report: Report<ServerError>) -> Self {
        Self(report)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self.0.current_context() {
            ServerError::Invalid(_) => StatusCode::BAD_REQUEST,
            ServerError::NotFound => StatusCode::NOT_FOUND,
            ServerError::Gone => StatusCode::GONE,
            ServerError::Unauthorized => StatusCode::UNAUTHORIZED,
            ServerError::Capacity => StatusCode::TOO_MANY_REQUESTS,
            ServerError::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        };
        (
            status,
            Json(json!({"error": self.0.current_context().to_string()})),
        )
            .into_response()
    }
}

fn org(service: &Sampling, headers: &HeaderMap) -> Result<OrgId, ApiError> {
    let key = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    Ok(service.authenticate(key)?)
}

pub(crate) async fn config(
    State(service): State<Sampling>, headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    org(&service, &headers)?;
    Ok(Json(json!({"use_pyqwest_transport": false})))
}

pub(crate) async fn create_session(
    State(service): State<Sampling>, headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    let id = service.create_session(org(&service, &headers)?)?;
    Ok(Json(json!({"session_id": id, "type": "create_session"})))
}

#[derive(Deserialize)]
pub(crate) struct Heartbeat {
    session_id: SessionId,
}

pub(crate) async fn heartbeat(
    State(service): State<Sampling>, headers: HeaderMap, ApiJson(request): ApiJson<Heartbeat>,
) -> Result<Json<Value>, ApiError> {
    service.heartbeat(&org(&service, &headers)?, request.session_id)?;
    Ok(Json(json!({})))
}

#[derive(Deserialize)]
pub(crate) struct CreateSampler {
    session_id: SessionId,
    sampling_session_seq_id: u64,
    base_model: Option<String>,
    model_path: Option<String>,
}

pub(crate) async fn create_sampler(
    State(service): State<Sampling>, headers: HeaderMap, ApiJson(request): ApiJson<CreateSampler>,
) -> Result<Json<Value>, ApiError> {
    let org = org(&service, &headers)?;
    if request.model_path.is_some() {
        return Err(Report::new(ServerError::Invalid(
            "checkpoint sampling is not implemented",
        ))
        .into());
    }
    let base = request
        .base_model
        .ok_or_else(|| Report::new(ServerError::Invalid("base_model is required")))?;
    let id = service
        .create_sampler(
            &org,
            request.session_id,
            request.sampling_session_seq_id,
            base,
        )
        .await?;
    Ok(Json(
        json!({"sampling_session_id": id, "type": "create_sampling_session"}),
    ))
}

pub(crate) async fn capabilities(
    State(service): State<Sampling>, headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    org(&service, &headers)?;
    let models: Vec<_> = service
        .bases()
        .await?
        .into_iter()
        .map(|base| {
            json!({
                "model_name": base.name, "max_context_length": base.max_context_length,
                "trainable": false, "sampleable": true,
            })
        })
        .collect();
    Ok(Json(json!({"supported_models": models})))
}

#[derive(Deserialize)]
pub(crate) struct Sample {
    sampling_session_id: ModelId,
    seq_id: u64,
    num_samples: u32,
    prompt: Prompt,
    sampling_params: Params,
    #[serde(default)]
    prompt_logprobs: bool,
    #[serde(default)]
    topk_prompt_logprobs: u32,
    #[serde(default)]
    topk_sample_logprobs: u32,
    target_prompt_logprobs: Option<Value>,
    #[serde(default)]
    prompt_alt_tokens_k: u32,
    prompt_logprobs_last_n: Option<u32>,
    #[serde(default)]
    record_stability_info: bool,
}

#[derive(Deserialize)]
struct Prompt {
    chunks: Vec<Chunk>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum Chunk {
    #[serde(rename = "encoded_text")]
    EncodedText { tokens: Vec<i32> },
}

#[derive(Deserialize)]
struct Params {
    max_tokens: Option<u32>,
    seed: Option<u64>,
    stop: Option<Value>,
    temperature: f64,
    top_k: i32,
    top_p: f64,
}

pub(crate) async fn sample(
    State(service): State<Sampling>, headers: HeaderMap, ApiJson(request): ApiJson<Sample>,
) -> Result<Json<Value>, ApiError> {
    let org = org(&service, &headers)?;
    let prompt: Vec<_> = request
        .prompt
        .chunks
        .into_iter()
        .flat_map(|chunk| {
            match chunk {
                Chunk::EncodedText { tokens } => tokens,
            }
        })
        .collect();
    let max_length = request.sampling_params.max_tokens.unwrap_or(16);
    if request.num_samples == 0
        || request.num_samples > 16
        || max_length == 0
        || max_length > 256
        || prompt.is_empty()
        || prompt.len() > 4096
        || prompt.iter().any(|&t| t < 0)
        || !request.sampling_params.temperature.is_finite()
        || request.sampling_params.temperature < 0.0
        || request.sampling_params.top_k < -1
        || !(0.0..=1.0).contains(&request.sampling_params.top_p)
    {
        return Err(Report::new(ServerError::Invalid(
            "invalid sampling limits or parameters",
        ))
        .into());
    }
    if request.prompt_logprobs
        || request.topk_prompt_logprobs > 0
        || request.topk_sample_logprobs > 0
        || request.target_prompt_logprobs.is_some()
        || request.prompt_alt_tokens_k > 0
        || request.prompt_logprobs_last_n.is_some()
        || request.record_stability_info
        || request.sampling_params.stop.is_some()
        || !matches!(request.sampling_params.temperature, 1.0)
        || request.sampling_params.top_k != -1
        || !matches!(request.sampling_params.top_p, 1.0)
    {
        return Err(Report::new(ServerError::Invalid(
            "this sampling option is not implemented by the mock engine",
        ))
        .into());
    }
    let id = service.sample(
        &org,
        request.sampling_session_id,
        request.seq_id,
        SampleRequest {
            model_id: String::new(),
            prompt,
            num_samples: request.num_samples,
            max_length,
            seed: request.sampling_params.seed.unwrap_or(0),
        },
    )?;
    let sequence_ids: Vec<_> = (0..request.num_samples)
        .map(|i| format!("{}-{i}", id.0))
        .collect();
    Ok(Json(
        json!({"request_id": id, "sample_sequence_ids": sequence_ids}),
    ))
}

#[derive(Deserialize)]
pub(crate) struct Retrieve {
    request_id: RequestId,
}

pub(crate) async fn retrieve(
    State(service): State<Sampling>, headers: HeaderMap, ApiJson(request): ApiJson<Retrieve>,
) -> Result<Response, ApiError> {
    use prost::Message;
    match service.retrieve(&org(&service, &headers)?, request.request_id)? {
        SampleResult::Pending => {
            Ok((
                StatusCode::REQUEST_TIMEOUT,
                Json(json!({"type": "try_again", "queue_state": "active"})),
            )
                .into_response())
        }
        SampleResult::Failed => {
            Ok(
                Json(json!({"error": "worker sampling failed", "category": "server"}))
                    .into_response(),
            )
        }
        SampleResult::Complete(result) => {
            let result = crate::tinker_public::SampleResponse {
                sequences: result
                    .sequences
                    .into_iter()
                    .map(|sequence| {
                        crate::tinker_public::SampledSequence {
                            stop_reason: crate::tinker_public::StopReason::Length as i32,
                            tokens: sequence
                                .tokens
                                .into_iter()
                                .flat_map(i32::to_le_bytes)
                                .collect(),
                            ..Default::default()
                        }
                    })
                    .collect(),
                ..Default::default()
            };
            Ok((
                [("content-type", "application/x-protobuf")],
                result.encode_to_vec(),
            )
                .into_response())
        }
    }
}
