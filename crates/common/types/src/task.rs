use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Position of a task in its model's log: the SDK's `seq_id`, starting at 1.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Seq(pub u64);

/// Ownership epoch of a model. Each takeover increments it; writes from an
/// older generation are rejected.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Generation(pub u64);

/// What a task runs. Forward-backward data travels in the task's `input.pb`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TaskInput {
    ForwardBackward,
    OptimStep(AdamParams),
    SaveWeightsForSampler {
        /// Set for a named save, which answers with a `tinker://` path;
        /// exactly one of this and `sampling_session_seq_id` is set.
        name: Option<String>,
        /// Set for an unnamed save, which answers with a sampling session.
        sampling_session_seq_id: Option<u64>,
        /// `None` never expires.
        ttl_seconds: Option<u64>,
        user_metadata: BTreeMap<String, String>,
    },
}

/// Tinker `AdamParams`.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct AdamParams {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub grad_clip_norm: f64,
}

/// How a task ended.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Outcome {
    /// The output is the object at `output_key`.
    ForwardBackward { output_key: String },
    /// Metric values are finite, because the column is JSON.
    OptimStep { metrics: BTreeMap<String, f64> },
    /// The adapter's files are under `prefix`, and the task's checkpoint row
    /// points to them.
    SaveWeightsForSampler { prefix: String, size_bytes: u64 },
    Failed {
        category: ErrorCategory,
        message: String,
    },
}

/// Tinker `RequestErrorCategory`: whether the user or the server is at fault.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCategory {
    User,
    Server,
}
