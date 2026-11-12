use std::collections::BTreeMap;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use error_stack::{Report, ResultExt};
use metrl_server::{NewModel, Sampling, ServerError, Training, TrainingResult};
use metrl_types::{AdamParams, ModelId, Seq, SessionId, TaskInput};
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::error::ApiError;
use crate::extract::{ApiJson, Org};

#[derive(Deserialize)]
pub(crate) struct CreateModel {
    session_id: SessionId,
    model_seq_id: u64,
    base_model: String,
    lora_config: Option<LoraConfig>,
    optimizer_config: Option<OptimizerConfig>,
}

#[derive(Deserialize)]
struct LoraConfig {
    rank: u32,
    seed: Option<u64>,
    train_unembed: Option<bool>,
    train_mlp: Option<bool>,
    train_attn: Option<bool>,
}

#[derive(Deserialize)]
struct OptimizerConfig {
    #[serde(rename = "type")]
    kind: String,
}

pub(crate) async fn create_model(
    Org(org): Org, State(sampling): State<Sampling>, State(training): State<Training>,
    ApiJson(request): ApiJson<CreateModel>,
) -> Result<Json<Value>, ApiError> {
    let lora = request.lora_config.ok_or_else(|| {
        Report::new(ServerError::Invalid(
            "lora_config is required; full fine-tuning is not implemented",
        ))
    })?;
    // The trainer adapts every layer; a narrower request cannot run as given.
    if [lora.train_unembed, lora.train_mlp, lora.train_attn].contains(&Some(false)) {
        return Err(Report::new(ServerError::Invalid(
            "train_unembed, train_mlp, and train_attn must be true",
        ))
        .into());
    }
    if request
        .optimizer_config
        .is_some_and(|optimizer| optimizer.kind != "adamw")
    {
        return Err(Report::new(ServerError::Invalid(
            "only the adamw optimizer is supported",
        ))
        .into());
    }
    let id = training
        .create_model(
            &org,
            &sampling,
            NewModel {
                session: request.session_id,
                session_seq: request.model_seq_id,
                base: request.base_model,
                lora_rank: lora.rank,
                seed: lora.seed,
            },
        )
        .await?;
    // The model is created before this returns, so its future is done and
    // retrieving it needs no request of its own.
    Ok(Json(json!({"request_id": id, "model_id": id})))
}

#[derive(Deserialize)]
pub(crate) struct GetInfo {
    model_id: ModelId,
}

pub(crate) async fn get_info(
    Org(org): Org, State(training): State<Training>, ApiJson(request): ApiJson<GetInfo>,
) -> Result<Json<Value>, ApiError> {
    let model = training.model(&org, request.model_id).await?;
    Ok(Json(json!({
        "type": "get_info",
        "model_id": model.id,
        "model_data": {
            "arch": model.arch, "model_name": model.base, "tokenizer_id": model.tokenizer_id,
        },
        "optimizer_config": {"type": "adamw"},
        "is_lora": true,
        "lora_rank": model.lora_rank,
        "model_name": model.base,
    })))
}

/// Forward and forward-backward share this route; the body is a Tinker
/// `ForwardBackwardRequest` whose `forward_only` tells them apart.
pub(crate) async fn forward_backward(
    Org(org): Org, State(training): State<Training>, body: Bytes,
) -> Result<Json<Value>, ApiError> {
    use prost::Message;
    let request = crate::tinker_public::ForwardBackwardRequest::decode(body).change_context(
        ServerError::Invalid("body is not a Tinker ForwardBackwardRequest"),
    )?;
    let model = Uuid::parse_str(&request.model_id)
        .map(ModelId)
        .change_context(ServerError::Invalid("model_id must be a UUID"))?;
    let seq = u64::try_from(request.seq_id)
        .change_context(ServerError::Invalid("seq_id must be at least 1"))?;
    // Re-encoded so that a retry compares equal however its client ordered
    // the maps.
    let input = Bytes::from(request.encode_to_vec());
    let id = training
        .submit(
            &org,
            model,
            Seq(seq),
            TaskInput::ForwardBackward,
            Some(input),
        )
        .await?;
    Ok(Json(json!({"request_id": id, "model_id": model})))
}

#[derive(Deserialize)]
pub(crate) struct OptimStep {
    model_id: ModelId,
    seq_id: Option<u64>,
    adam_params: Option<AdamParams>,
}

pub(crate) async fn optim_step(
    Org(org): Org, State(training): State<Training>, ApiJson(request): ApiJson<OptimStep>,
) -> Result<Json<Value>, ApiError> {
    let params = request.adam_params.ok_or_else(|| {
        Report::new(ServerError::Invalid(
            "adam_params is required; other optimizers are not supported",
        ))
    })?;
    let seq = request
        .seq_id
        .ok_or_else(|| Report::new(ServerError::Invalid("seq_id is required")))?;
    let id = training
        .submit(
            &org,
            request.model_id,
            Seq(seq),
            TaskInput::OptimStep(params),
            None,
        )
        .await?;
    Ok(Json(
        json!({"request_id": id, "model_id": request.model_id}),
    ))
}

/// The longest ttl Tinker accepts: ten years.
const MAX_TTL_SECONDS: u64 = 10 * 365 * 24 * 3600;

#[derive(Deserialize)]
pub(crate) struct SaveWeightsForSampler {
    model_id: ModelId,
    seq_id: Option<u64>,
    /// The checkpoint's name; unnamed saves send `sampling_session_seq_id`.
    path: Option<String>,
    sampling_session_seq_id: Option<u64>,
    kind: Option<String>,
    ttl_seconds: Option<u64>,
    user_metadata: Option<BTreeMap<String, String>>,
}

pub(crate) async fn save_weights_for_sampler(
    Org(org): Org, State(training): State<Training>,
    ApiJson(request): ApiJson<SaveWeightsForSampler>,
) -> Result<Json<Value>, ApiError> {
    if request.kind.is_some_and(|kind| kind != "snapshot") {
        return Err(Report::new(ServerError::Invalid(
            "only snapshot sampling clients are supported",
        ))
        .into());
    }
    let seq = request
        .seq_id
        .ok_or_else(|| Report::new(ServerError::Invalid("seq_id is required")))?;
    if request.path.is_some() == request.sampling_session_seq_id.is_some() {
        return Err(Report::new(ServerError::Invalid(
            "exactly one of path and sampling_session_seq_id must be set",
        ))
        .into());
    }
    // The name ends a `tinker://` path, so it cannot hold a `/`.
    let valid_name = |name: &String| {
        !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    };
    if request.path.as_ref().is_some_and(|name| !valid_name(name)) {
        return Err(Report::new(ServerError::Invalid(
            "path may only hold ASCII letters, digits, '-', '_', and '.'",
        ))
        .into());
    }
    if request
        .ttl_seconds
        .is_some_and(|ttl| !(3600..=MAX_TTL_SECONDS).contains(&ttl))
    {
        return Err(Report::new(ServerError::Invalid(
            "ttl_seconds must be between 3600 and ten years",
        ))
        .into());
    }
    let input = TaskInput::SaveWeightsForSampler {
        name: request.path,
        sampling_session_seq_id: request.sampling_session_seq_id,
        ttl_seconds: request.ttl_seconds,
        user_metadata: request.user_metadata.unwrap_or_default(),
    };
    let id = training
        .submit(&org, request.model_id, Seq(seq), input, None)
        .await?;
    Ok(Json(
        json!({"request_id": id, "model_id": request.model_id}),
    ))
}

/// The answer for a finished training request, or `None` while it is pending.
pub(crate) fn response(result: TrainingResult) -> Option<Response> {
    let response = match result {
        TrainingResult::Pending => return None,
        TrainingResult::Created(model) => {
            Json(json!({"model_id": model, "type": "create_model"})).into_response()
        }
        TrainingResult::ForwardBackward(output) => {
            ([("content-type", "application/x-protobuf")], output).into_response()
        }
        TrainingResult::OptimStep(metrics) => Json(json!({"metrics": metrics})).into_response(),
        TrainingResult::SamplerPath(path) => {
            Json(json!({"path": path, "type": "save_weights_for_sampler"})).into_response()
        }
        TrainingResult::SamplingSession(id) => {
            Json(json!({"sampling_session_id": id, "type": "save_weights_for_sampler"}))
                .into_response()
        }
        TrainingResult::Failed { category, message } => {
            Json(json!({"error": message, "category": category})).into_response()
        }
    };
    Some(response)
}
