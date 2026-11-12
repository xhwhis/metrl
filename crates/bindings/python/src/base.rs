use pyo3::prelude::*;

/// A base as the engine loaded it. An engine returns one for each configured
/// base; the worker reports them to servers.
#[pyclass(frozen, module = "metrl._core")]
pub(crate) struct Base {
    pub(crate) name: String,
    pub(crate) revision: String,
    pub(crate) arch: String,
    pub(crate) tokenizer_id: String,
    pub(crate) max_context_length: u32,
    pub(crate) max_lora_rank: u32,
}

#[pymethods]
impl Base {
    /// `revision` identifies the loaded weights; servers reject a different
    /// revision for a name another worker already serves.
    #[new]
    #[pyo3(signature = (*, name, revision, arch, tokenizer_id, max_context_length, max_lora_rank))]
    fn new(
        name: String, revision: String, arch: String, tokenizer_id: String,
        max_context_length: u32, max_lora_rank: u32,
    ) -> Self {
        Self {
            name,
            revision,
            arch,
            tokenizer_id,
            max_context_length,
            max_lora_rank,
        }
    }
}
