//! `tinker://<model>/sampler_weights/<name>`, the path a named
//! `save_weights_for_sampler` answers with. Paths are logical; the
//! `checkpoints` table maps them to objects.

use metrl_types::ModelId;
use uuid::Uuid;

const SCHEME: &str = "tinker://";
const KIND: &str = "sampler_weights";

/// A named checkpoint as a path names it.
pub(crate) struct SamplerPath<'a> {
    pub(crate) model: ModelId,
    pub(crate) name: &'a str,
}

pub(crate) fn format(model: ModelId, name: &str) -> String {
    format!("{SCHEME}{}/{KIND}/{name}", model.0)
}

/// `None` unless `path` is a sampler weights path.
pub(crate) fn parse(path: &str) -> Option<SamplerPath<'_>> {
    let mut parts = path.strip_prefix(SCHEME)?.split('/');
    let (Some(model), Some(KIND), Some(name), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return None;
    };
    let model = Uuid::parse_str(model).ok().map(ModelId)?;
    (!name.is_empty()).then_some(SamplerPath { model, name })
}
