use crate::{ModelId, OrgId, SessionId};

/// A low-rank adapter on a base, fixed at creation. Its id is also the
/// request id of the `create_model` call that created it.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub id: ModelId,
    pub org: OrgId,
    /// The SDK names a model by its session and `model_seq_id`, so a retried
    /// `create_model` finds the model it created first.
    pub session: SessionId,
    pub session_seq: u64,
    /// The base as the trainer reported it at creation.
    pub base: String,
    pub base_revision: String,
    pub arch: String,
    pub tokenizer_id: String,
    pub lora_rank: u32,
    pub seed: u64,
}
