//! Checkpoints as Postgres holds them, for readers whose view may lag a save
//! that another server answered.

use error_stack::{Report, ResultExt};
use metrl_types::{ModelId, OrgId, RequestId};

use crate::view::{self, Checkpoint};
use crate::{MetaError, Store};

impl Store {
    pub async fn checkpoint(
        &self, org: &OrgId, id: RequestId,
    ) -> Result<Option<Checkpoint>, Report<MetaError>> {
        self.client()
            .await?
            .query_opt(
                include_str!("../sql/select_checkpoint.sql"),
                &[&id.0, &org.0],
            )
            .await
            .change_context(MetaError::Query)?
            .as_ref()
            .map(view::checkpoint)
            .transpose()
    }

    pub async fn named_checkpoint(
        &self, org: &OrgId, model: ModelId, name: &str,
    ) -> Result<Option<Checkpoint>, Report<MetaError>> {
        self.client()
            .await?
            .query_opt(
                include_str!("../sql/select_named_checkpoint.sql"),
                &[&model.0, &name, &org.0],
            )
            .await
            .change_context(MetaError::Query)?
            .as_ref()
            .map(view::checkpoint)
            .transpose()
    }
}
