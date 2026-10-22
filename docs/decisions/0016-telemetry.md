# 0016. Telemetry

## Context

A request crosses the SDK, a server, possibly the owner server, a worker, and an engine.
Operators need one trace per request and standard metrics.

## Decision

- `tracing` with `tracing-opentelemetry`; W3C `traceparent` across HTTP and gRPC.
- Tasks store `trace_ctx`, so the owner continues the submitter's trace.
- OTLP export is optional, enabled by configuration.
- Metrics use the `metrics` crate with a Prometheus exporter at `/metrics` on the system
  port. Labels are limited to `kind`, `role`, and `status`.

## Alternatives

- fastrace: lower span overhead doesn't matter next to ML step times, and it splits from
  the tonic, hyper, and tower ecosystem.
- OpenTelemetry metrics export: version churn, and Prometheus pull is the norm.
- OpenTelemetry in Python: Rust-side spans cover engine calls.

## Consequences

Trace context must be propagated by hand across CDC and actor boundaries.
