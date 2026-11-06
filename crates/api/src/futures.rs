use std::time::Duration;

use axum::Json;
use axum::extract::{FromRequest, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use error_stack::Report;
use metrl_server::{Sampling, ServerError, Training};
use metrl_types::RequestId;
use serde::Deserialize;
use serde_json::json;
use tokio::time;
use tokio_util::sync::CancellationToken;

use crate::error::ApiError;
use crate::extract::{self, ApiJson};
use crate::{sampling, training};

/// How long `retrieve_future` holds a request it cannot answer yet. The SDK
/// retries 408 and 5xx at once, so answering sooner makes it spin; it gives
/// each call 45 seconds.
const POLL: Duration = Duration::from_secs(25);

#[derive(Deserialize)]
pub(crate) struct Retrieve {
    request_id: RequestId,
}

/// Waits until the request is done, [`POLL`] passes, or the server starts
/// shutting down, whose drain would otherwise wait for every poll. A pending
/// request and an unavailable view, database, or object store are retried on
/// every finished sample and view change.
///
/// A rejected key answers before the body is read. While the view is stale
/// the key cannot be judged, so it is checked again on each try; answering
/// the stale view with 503 at once would make the SDK retry in a loop.
pub(crate) async fn retrieve(
    State(service): State<Sampling>, State(mut training): State<Training>,
    State(shutdown): State<CancellationToken>, headers: HeaderMap, body: Request,
) -> Result<Response, ApiError> {
    if let Err(report) = extract::org(&service, &headers)
        && !matches!(report.current_context(), ServerError::Unavailable(_))
    {
        return Err(report.into());
    }
    let request = match ApiJson::<Retrieve>::from_request(body, &()).await {
        Ok(ApiJson(request)) => request,
        Err(rejection) => return Ok(rejection),
    };
    let deadline = time::Instant::now() + POLL;
    loop {
        // Created before reading, so a sample that finishes in between wakes
        // it.
        let finished = service.finished();
        let answer = poll(&service, &training, &headers, request.request_id).await;
        let waiting = match &answer {
            Ok(response) => response.is_none(),
            Err(report) => matches!(report.current_context(), ServerError::Unavailable(_)),
        };
        if !waiting || time::Instant::now() >= deadline || shutdown.is_cancelled() {
            return answer
                .map(|response| response.unwrap_or_else(|| try_again(request.request_id)))
                .map_err(ApiError::from);
        }
        tokio::select! {
            () = finished => {}
            () = training.changed() => {}
            () = shutdown.cancelled() => {}
            () = time::sleep_until(deadline) => {}
        }
    }
}

/// The answer for a finished request, or `None` while it is pending.
/// Sampling results live in the memory of the server that took them;
/// training results in the task log.
async fn poll(
    service: &Sampling, training: &Training, headers: &HeaderMap, request: RequestId,
) -> Result<Option<Response>, Report<ServerError>> {
    let org = extract::org(service, headers)?;
    match service.retrieve(&org, request) {
        Ok(result) => Ok(sampling::response(result)),
        Err(report) if matches!(report.current_context(), ServerError::NotFound) => {
            Ok(training::response(training.retrieve(&org, request).await?))
        }
        Err(report) => Err(report),
    }
}

/// Tinker's answer for a request that is still queued; the SDK polls again.
fn try_again(request: RequestId) -> Response {
    (
        StatusCode::REQUEST_TIMEOUT,
        Json(json!({"type": "try_again", "request_id": request, "queue_state": "active"})),
    )
        .into_response()
}
