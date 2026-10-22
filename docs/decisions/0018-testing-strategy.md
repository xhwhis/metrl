# 0018. Testing strategy

## Context

Agent-written code tends to accumulate many unit tests that restate the implementation
and still miss real failures, which in Metrl are distributed: lease takeover, CDC
ordering, retries, and recovery.

## Decision

- E2E through the Tinker SDK against real processes and the mock engine comes first;
  chaos tests inject faults with fail points, process kills, and `METRL_MOCK_FAULT`.
- The mock engine's outputs carry a state hash chain. A clean run, a fault-injected run,
  and a two-org interleaved run must produce the same hashes.
- Scheduler logic is tested through `on_event`, with proptest over event interleavings
  and invariants asserted after every event. Actor timers use paused tokio time.
- Snapshot tests use insta. Assertions target observable results and metrics.
- No internal mocks, no coverage targets. A new test must name the plausible bug it
  catches.
- Lint levels live only in root configs, and source code never suppresses a check.
- Rust code builds and tests on stable in CI; daily development uses nightly rustfmt.
- Benchmarks come in M5 as ~300ms scenarios reported in integer milliseconds, outside CI.

## Alternatives

- loom: checks low-level atomics and locks, not Metrl's distributed races, doesn't model
  tokio I/O, and is unmaintained. shuttle is allowed for a hand-written concurrent
  structure if one appears.
- Deterministic simulation (turmoil, madsim): too costly with real Postgres and CDC.
- Coverage targets: optimize a number, not confidence.

## Consequences

Tests are fewer and slower but exercise real failure modes. The mock engine must stay
faithful to real engine semantics.
