# tests

End-to-end and chaos tests drive real `metrl-server` and `metrl-worker` processes
through the Tinker SDK.

- Default engine is `mock`; an MLX run executes the same flows on tiny Qwen3.
- Each test gets its own Postgres database cloned from a template, and its own
  object store prefix.
- Chaos uses `fail` points in Rust (`failpoints` feature) and `METRL_MOCK_FAULT` in the
  mock engine. Assert on recovered results, not on logs. Seeded runs print their seed.
- Multi-tenant flows must interleave at least two orgs and match isolated runs.
- Add a test for a user-visible behavior or a real failure mode. Do not add tests for
  coverage or to restate an implementation.
