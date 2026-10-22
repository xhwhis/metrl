# 0009. Tinker compatibility

## Context

Users already train with the Tinker SDK. A new API would force them to change code and
would give us nothing they need.

## Decision

- The Tinker SDK HTTP API is the only external contract: routes, payloads, status codes,
  and naming follow it.
- Track the latest `tinker` release; `uv.lock` pins the version used in tests.
- `proto/tinker.proto` is generated from the SDK's wire descriptor by
  `just tinker-proto`; on upgrade, regenerate it and diff the SDK's `types/` and routes
  against the API.
- Everything internal may change without compatibility.

## Alternatives

- A Metrl-specific API: users must port code.
- Pinning one SDK version: drifts from what users install.

## Consequences

E2E tests run through the real SDK. Each SDK release may require API work.
