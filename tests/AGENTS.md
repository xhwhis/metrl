# tests

End-to-end and chaos tests drive real `metrl-server` and `metrl-worker` processes
through the Tinker SDK.

- Default engine is `mock`. `test_mlx.py` runs the RL loop and sampling flows on the
  MLX engine with a tiny Qwen3; it runs only on macOS and is skipped in CI.
- Each test gets its own Postgres database cloned from a template, and its own
  object store prefix.
- Chaos uses `fail` points in Rust (`failpoints` feature) and `METRL_MOCK_FAULT` in the
  mock engine. Assert on recovered results, not on logs. Seeded runs print their seed.
- `just test` builds `metrl-server` with `--features failpoints`; a test turns fail
  points on by setting `FAILPOINTS` (`name=action;…`) for a process it spawns, and
  checks that the fault took effect, so a build without the feature fails instead of
  passing without the fault.
- Multi-tenant flows must interleave at least two orgs and match isolated runs.
- Add a test for a user-visible behavior or a real failure mode. Do not add tests for
  coverage or to restate an implementation.
