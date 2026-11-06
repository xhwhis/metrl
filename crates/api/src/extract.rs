use axum::Json;
use axum::extract::{FromRef, FromRequest, FromRequestParts, Request};
use axum::http::request::Parts;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use error_stack::Report;
use metrl_server::{Sampling, ServerError};
use metrl_types::OrgId;
use serde::de::DeserializeOwned;
use serde_json::json;

use crate::error::ApiError;

/// A JSON body whose rejections are 400 with an `error` message, as Tinker
/// returns them.
pub(crate) struct ApiJson<T>(pub(crate) T);

impl<T: DeserializeOwned, S: Send + Sync> FromRequest<S> for ApiJson<T> {
    type Rejection = Response;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        Json::<T>::from_request(request, state)
            .await
            .map(|Json(value)| Self(value))
            .map_err(|error| {
                let status = match error.status() {
                    StatusCode::UNPROCESSABLE_ENTITY => StatusCode::BAD_REQUEST,
                    status => status,
                };
                (status, Json(json!({"error": error.body_text()}))).into_response()
            })
    }
}

/// The org of the request's API key. As an extractor it runs before the
/// body is read, so a request without a valid key is never buffered.
pub(crate) struct Org(pub(crate) OrgId);

impl<S: Send + Sync> FromRequestParts<S> for Org
where
    Sampling: FromRef<S>,
{
    type Rejection = ApiError;

    fn from_request_parts(
        parts: &mut Parts, state: &S,
    ) -> impl Future<Output = Result<Self, Self::Rejection>> + Send {
        let org = org(&Sampling::from_ref(state), &parts.headers);
        std::future::ready(org.map(Self).map_err(ApiError::from))
    }
}

/// The org that the API key in `headers` belongs to.
pub(crate) fn org(service: &Sampling, headers: &HeaderMap) -> Result<OrgId, Report<ServerError>> {
    let key = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    service.authenticate(key)
}
