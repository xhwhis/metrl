use axum::Json;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use error_stack::Report;
use metrl_proto::worker::SampleRequest;
use metrl_server::{SampleResult, Sampling, ServerError};
use metrl_types::{ModelId, Role, SessionId};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::ApiError;
use crate::extract::{ApiJson, Org};

/// With `parallel_fwdbwd_chunks` the SDK sends a split call's first chunk
/// last, so 64 later chunks waiting for it would leave it rejected by the
/// pending limit forever.
pub(crate) async fn config(_: Org) -> Json<Value> {
    Json(json!({"use_pyqwest_transport": false, "parallel_fwdbwd_chunks": false}))
}

pub(crate) async fn create_session(
    Org(org): Org, State(service): State<Sampling>,
) -> Result<Json<Value>, ApiError> {
    let id = service.create_session(org)?;
    Ok(Json(json!({"session_id": id, "type": "create_session"})))
}

#[derive(Deserialize)]
pub(crate) struct Heartbeat {
    session_id: SessionId,
}

pub(crate) async fn heartbeat(
    Org(org): Org, State(service): State<Sampling>, ApiJson(request): ApiJson<Heartbeat>,
) -> Result<Json<Value>, ApiError> {
    service.heartbeat(&org, request.session_id)?;
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
    Org(org): Org, State(service): State<Sampling>, ApiJson(request): ApiJson<CreateSampler>,
) -> Result<Json<Value>, ApiError> {
    if request.model_path.is_some() {
        return Err(Report::new(ServerError::Invalid(
            "checkpoint sampling is not implemented",
        ))
        .into());
    }
    let base = request
        .base_model
        .ok_or_else(|| Report::new(ServerError::Invalid("base_model is required")))?;
    let id = service.create_sampler(
        &org,
        request.session_id,
        request.sampling_session_seq_id,
        &base,
    )?;
    Ok(Json(
        json!({"sampling_session_id": id, "type": "create_sampling_session"}),
    ))
}

pub(crate) async fn capabilities(_: Org, State(service): State<Sampling>) -> Json<Value> {
    let models: Vec<_> = service
        .bases()
        .into_iter()
        .map(|base| {
            json!({
                "model_name": base.name, "max_context_length": base.max_context_length,
                "trainable": base.roles.contains(&Role::Trainer),
                "sampleable": base.roles.contains(&Role::Sampler),
            })
        })
        .collect();
    Json(json!({"supported_models": models}))
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
    Org(org): Org, State(service): State<Sampling>, ApiJson(request): ApiJson<Sample>,
) -> Result<Json<Value>, ApiError> {
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
            worker_id: String::new(),
        },
    )?;
    let sequence_ids: Vec<_> = (0..request.num_samples)
        .map(|i| format!("{}-{i}", id.0))
        .collect();
    Ok(Json(
        json!({"request_id": id, "sample_sequence_ids": sequence_ids}),
    ))
}

/// The answer for a sample, or `None` while it is pending.
pub(crate) fn response(result: SampleResult) -> Option<Response> {
    use prost::Message;
    match result {
        SampleResult::Pending => None,
        SampleResult::Failed => {
            Some(
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
            Some(
                (
                    [("content-type", "application/x-protobuf")],
                    result.encode_to_vec(),
                )
                    .into_response(),
            )
        }
    }
}
