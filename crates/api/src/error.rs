use axum::Json;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use error_stack::Report;
use metrl_server::ServerError;
use serde_json::json;

pub(crate) struct ApiError(Report<ServerError>);

impl From<Report<ServerError>> for ApiError {
    fn from(report: Report<ServerError>) -> Self {
        Self(report)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let context = self.0.current_context();
        if !matches!(
            context,
            ServerError::Unavailable(_) | ServerError::Internal(_)
        ) {
            tracing::debug!(error = ?self.0, "request rejected");
        }
        let body = Json(json!({"error": context.to_string()}));
        let status = match context {
            ServerError::Invalid(_) => StatusCode::BAD_REQUEST,
            ServerError::NotFound => StatusCode::NOT_FOUND,
            ServerError::Gone => StatusCode::GONE,
            ServerError::Unauthorized => StatusCode::UNAUTHORIZED,
            ServerError::Capacity => {
                return (
                    StatusCode::TOO_MANY_REQUESTS,
                    [(header::RETRY_AFTER, "1")],
                    body,
                )
                    .into_response();
            }
            ServerError::Unavailable(_) => {
                tracing::warn!(error = ?self.0, "request failed");
                StatusCode::SERVICE_UNAVAILABLE
            }
            ServerError::Internal(_) => {
                tracing::error!(error = ?self.0, "request failed");
                StatusCode::INTERNAL_SERVER_ERROR
            }
        };
        (status, body).into_response()
    }
}
