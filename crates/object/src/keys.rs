//! Every key starts with `orgs/{org}/`. The zero-padded seq keeps a model's
//! tasks in log order when listed.

use metrl_types::{Generation, ModelId, OrgId, RequestId, Seq};

/// The task's request body, written once by the server that accepted it.
pub fn task_input(org: &OrgId, model: ModelId, seq: Seq, request: RequestId) -> String {
    format!(
        "orgs/{}/tasks/{}/{:020}/{}/input.pb",
        org.0, model.0, seq.0, request.0
    )
}

/// The task's forward-backward output. Each generation writes its own, so a
/// fenced owner never overwrites the output its successor recorded.
pub fn task_output(
    org: &OrgId, model: ModelId, seq: Seq, request: RequestId, generation: Generation,
) -> String {
    format!(
        "orgs/{}/tasks/{}/{:020}/{}/g{}/output.pb",
        org.0, model.0, seq.0, request.0, generation.0
    )
}

/// The prefix of a `save_weights_for_sampler` call's adapter files. Each
/// generation writes its own, so a fenced owner never overwrites the files its
/// successor recorded.
pub fn sampler_weights(
    org: &OrgId, model: ModelId, request: RequestId, generation: Generation,
) -> String {
    format!(
        "orgs/{}/models/{}/sampler_weights/{}/g{}/",
        org.0, model.0, request.0, generation.0
    )
}
