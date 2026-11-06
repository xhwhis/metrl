//! Models as created through the API. Lease and fencing state is in
//! [`crate::lease`].

use error_stack::{Report, ResultExt};
use metrl_types::{Model, ModelId, OrgId, SessionId};

use crate::view::Columns;
use crate::{MetaError, Store};

impl Store {
    /// Creates the model unless its session and seq already created one, and
    /// returns the model that holds them.
    pub async fn create_model(&self, model: &Model) -> Result<Model, Report<MetaError>> {
        let client = self.client().await?;
        client
            .execute(
                include_str!("../sql/create_model.sql"),
                &[
                    &model.id.0,
                    &model.org.0,
                    &model.session.0,
                    &(model.session_seq as i64),
                    &model.base,
                    &model.base_revision,
                    &model.arch,
                    &model.tokenizer_id,
                    &(model.lora_rank as i32),
                    &(model.seed as i64),
                ],
            )
            .await
            .change_context(MetaError::Query)?;
        // A separate statement, so its snapshot includes a concurrent insert
        // that the conflict waited for.
        let row = client
            .query_opt(
                include_str!("../sql/select_model_by_session.sql"),
                &[&model.org.0, &model.session.0, &(model.session_seq as i64)],
            )
            .await
            .change_context(MetaError::Query)?
            .ok_or_else(|| Report::new(MetaError::Query).attach("the created model is gone"))?;
        self::model(&row)
    }

    pub async fn model(
        &self, org: &OrgId, id: ModelId,
    ) -> Result<Option<Model>, Report<MetaError>> {
        self.client()
            .await?
            .query_opt(include_str!("../sql/select_model.sql"), &[&id.0, &org.0])
            .await
            .change_context(MetaError::Query)?
            .as_ref()
            .map(model)
            .transpose()
    }

    /// The model that this session's `model_seq_id` created.
    pub async fn model_by_session(
        &self, org: &OrgId, session: SessionId, seq: u64,
    ) -> Result<Option<Model>, Report<MetaError>> {
        self.client()
            .await?
            .query_opt(
                include_str!("../sql/select_model_by_session.sql"),
                &[&org.0, &session.0, &(seq as i64)],
            )
            .await
            .change_context(MetaError::Query)?
            .as_ref()
            .map(model)
            .transpose()
    }
}

// Seqs, ranks, and seeds are stored as signed integers; the casts round-trip.
fn model(row: &impl Columns) -> Result<Model, Report<MetaError>> {
    Ok(Model {
        id: ModelId(row.get("id")?),
        org: OrgId(row.get("org_id")?),
        session: SessionId(row.get("session_id")?),
        session_seq: row.get::<i64>("session_seq")? as u64,
        base: row.get("base")?,
        base_revision: row.get("base_revision")?,
        arch: row.get("arch")?,
        tokenizer_id: row.get("tokenizer_id")?,
        lora_rank: row.get::<i32>("lora_rank")? as u32,
        seed: row.get::<i64>("seed")? as u64,
    })
}
