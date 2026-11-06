//! Appending tasks to a model's log and reading them back before the change
//! stream delivers them. Outcomes are written in [`crate::lease`].

use error_stack::{Report, ResultExt};
use metrl_types::{ModelId, OrgId, RequestId, Seq};
use tokio_postgres::types::Json;

use crate::view::{self, Task};
use crate::{MetaError, Store};

impl Store {
    /// Appends a pending task to its model's log. False when the seq is
    /// taken.
    pub async fn insert_task(&self, task: &Task) -> Result<bool, Report<MetaError>> {
        self.client()
            .await?
            .execute(
                include_str!("../sql/insert_task.sql"),
                &[
                    &task.model.0,
                    &(task.seq.0 as i64),
                    &task.request.0,
                    &task.org.0,
                    &Json(&task.input),
                    &task.created_at,
                ],
            )
            .await
            .change_context(MetaError::Query)
            .map(|inserted| inserted > 0)
    }

    pub async fn task(
        &self, org: &OrgId, model: ModelId, seq: Seq,
    ) -> Result<Option<Task>, Report<MetaError>> {
        self.client()
            .await?
            .query_opt(
                include_str!("../sql/select_task.sql"),
                &[&model.0, &(seq.0 as i64), &org.0],
            )
            .await
            .change_context(MetaError::Query)?
            .as_ref()
            .map(view::task)
            .transpose()
    }

    pub async fn task_by_request(
        &self, org: &OrgId, request: RequestId,
    ) -> Result<Option<Task>, Report<MetaError>> {
        self.client()
            .await?
            .query_opt(
                include_str!("../sql/select_task_by_request.sql"),
                &[&request.0, &org.0],
            )
            .await
            .change_context(MetaError::Query)?
            .as_ref()
            .map(view::task)
            .transpose()
    }
}
