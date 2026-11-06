use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Trainer,
    Sampler,
}

/// A base model as loaded by a worker.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Base {
    pub name: String,
    pub revision: String,
    pub arch: String,
    pub tokenizer_id: String,
    pub max_context_length: u32,
    pub max_lora_rank: u32,
    pub roles: Vec<Role>,
}
