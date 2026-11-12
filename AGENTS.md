# Metrl

RL infra orchestration layer with a Tinker-compatible HTTP API. Rust owns the API,
scheduling, tasks, and execution history. Python owns compute and memory state.
Read ARCHITECTURE.md for the design and docs/decisions/ for why.

## Layout

- `crates/common/{config,error,proto,telemetry,types,utils}`: crates shared across the project.
  `utils` holds project-internal functions that don't fit business logic.
- `crates/scheduler`: pure decisions only (`model/`, `cluster/`). No I/O, no clock reads.
- `crates/meta`: Postgres access, schema, and CDC. All server SQL lives in `crates/meta/sql/`.
- `crates/object`: object store keys, checkpoints, and sweep.
- `crates/api`: Tinker HTTP API.
- `crates/server`: wires api, scheduler, meta, and object; runs model actors and peers.
- `crates/worker`: worker runtime driven from Python.
- `crates/bin`: the `metrl-server` binary.
- `crates/bindings/python`: `metrl-py`, builds the `metrl` wheel (`metrl._core`).
- `python/metrl/worker`: engine Protocols, `serve()`, and the `metrl-worker` script.
- `engines/{mlx,mock}`: one package per engine, imported as `metrl.engines.<name>`.
- `proto/`: all `.proto` files, flat. Each consuming crate compiles them in its `build.rs`.
- `config/`: default config files.

## Commands

Prerequisites: Rust nightly, Rust 1.99.0 with `cargo-hawk`, `protoc`, `uv`, `just`, `prek`,
`taplo`, `typos`, `cargo-shear`, `cargo-nextest`, Docker, `psql`; `samply` for profiling.

`just setup | up | db-reset | dev | worker | key | rl | fmt | lint | test | test-mlx |
check | tinker-proto | profile`

`just check` runs everything CI runs except the stable-toolchain build and the macOS job,
which `just test-mlx` runs.

## Rules

- Breaking changes are fine. No backward compatibility, versioning, or deprecation
  shims; ship the best implementation. The only external contract is the Tinker SDK
  HTTP API. Track the latest `tinker` release; on upgrade, run `just tinker-proto` and
  diff the SDK's `types/` and routes against the API.
- SQL lives only in `.sql` files, one statement per file: server SQL in
  `crates/meta/sql/`, loaded with `include_str!`; test SQL in `tests/sql/`, run with
  `psql()` from `tests/conftest.py`. No SQL strings in Rust, Python, or scripts. Tests
  use `metrl-server` commands or the API first and SQL only for what those cannot
  reach, such as per-test databases, writes that bypass Metrl, or replication slots.
  The schema is the one exception: all of it is in `crates/meta/sql/schema.sql`,
  edited in place; reset or backfill data instead of writing migrations.
- Dependency boundaries: only `server` depends on `scheduler`; only `server` and `bin`
  depend on `meta`; `worker` and `metrl-py` never reach `scheduler`, `meta`, or `server`.
- Dependency versions live only in `[workspace.dependencies]` in the root `Cargo.toml`.
  Members write `{ workspace = true }` and may add `features` or `optional`; `taplo check`
  enforces this. Before adding a dependency, check crates.io or PyPI for the latest
  version and recent maintenance. Use caret ranges and commit lockfiles.
- Multi-tenancy: every tenant row carries `org_id`, every tenant query filters by it,
  cross-org access returns 404, and object keys start with `orgs/{org_id}/`. Shared
  cluster rows (`workers`) have no `org_id`.
- Lint and type-check levels live only in the root config: `[workspace.lints]` in
  `Cargo.toml` and `clippy.toml`, and `[tool.ruff]` / `[tool.ty]` in `pyproject.toml`.
  Do not add suppressions to source code: no `#[allow]`, `#[expect]`, `#[warn]`,
  `#[deny]`, `#[forbid]` or their `cfg_attr` forms in Rust; no `# noqa`,
  `# type: ignore` or `# ty: ignore` in Python. When a check fires, fix the code;
  silencing it to make `just check` pass is not a fix. If a rule is wrong for this
  project, propose a root config change and stop for review. The only exception is
  the module that includes generated protobuf code.
- Code must build on stable Rust. Nightly is used only for rustfmt options.

## Code style

- Plain data and functions. No builders. Add an abstraction or helper only when it names
  a real concept; repetition beats a vague `*_helper`.
- Prefer more modules over large files. Non-leaf modules hold only `mod` declarations
  and re-exports.
- IDs are newtypes in `common/types`, never bare `String` or `Uuid`.
- Use structs with named fields instead of tuples; single-field newtypes are fine.
- A `bool` parameter is fine for a true two-sided flag; use an enum when the meaning is
  not obvious at the call site.
- No `impl Into<T>` parameters; callers convert.
- Call free functions through their module (`mem::take`, not `take`). Import traits
  inside the function that uses them unless the whole file does.
- A generated proto type that shares a name with a domain type is always written
  through its package module (`worker::Base`), never imported by name or renamed with
  `as`, so a bare `Base` always means the domain type.
- Keep the functional style: iterator chains, `match`, and expressions over mutable
  loops where they read clearly.
- `Arc` and `.clone()` are fine where ownership is really shared; never clone or wrap
  in `Arc` just to silence a borrow or lifetime error. Fix the ownership instead.
- `assert!` always carries a message that says why the condition holds.
- Scheduler: `on_event(&mut State, now, Event) -> Vec<Effect>` and `next_deadline(&State)`.
  The model actor executes effects; all I/O lives there.
- Errors use `error-stack`. Each crate has one error context and returns
  `Result<T, Report<Context>>`. Crossing a crate boundary calls `change_context` and
  attaches business context such as `model_id` and `seq`. Only protocol boundaries
  map error kinds to statuses, each service in one place: `api` for HTTP, and each gRPC
  service implementation. No `anyhow`, no `#[from]`, no `unwrap`.
  `expect` is fine for invariants that only a caller bug can break; state the invariant.
- Code explains what; comments explain why. No comments that restate code.
- Every `tokio::spawn` carries `.instrument(span)`.
- Metric labels are limited to `kind`, `role`, and `status`.
- Write the project name as "Metrl" in prose, docs, and comments; identifiers stay lowercase.

## Python

- Python 3.14, fully typed, checked by `ty` and `ruff`.
- Plain data uses `@dataclass(frozen=True, slots=True)`. No pydantic; engines and the
  worker package stay dependency-light.
- Interfaces are `typing.Protocol`, not ABCs. No `Any` in public signatures.
- Engine methods are `async def`. Blocking compute runs on the shared
  `ComputeThread` via `await compute.run(fn)`, never on the event loop.
- No module-level mutable state. Configuration is passed in, never read from globals.
  Every engine call carries `model_id`.
- Absolute imports only (`from metrl.worker import ...`).

## Testing

- E2E through the Tinker SDK against the mock engine, plus chaos tests, come first.
- Write a unit test only for logic hard to reach end to end. No coverage-driven tests,
  and no regression test unless it catches a plausible future bug.
- Before adding a test, name the plausible bug it catches. If an existing check table,
  property, or E2E scenario covers it, extend that instead of adding a test.
- Assert observable results: API responses, outputs, engine state hash, metrics.
- No mocks of internal components; the mock engine is the only fake.
- Do not use loom. If hand-written concurrent structures appear, such as an in-memory
  result cache or per-model actor mailbox, use shuttle only for that structure.

## Docs

- Record decisions in `docs/decisions/NNNN-title.md` and edit them in place. Update
  ARCHITECTURE.md when structure changes.
- Every AGENTS.md has a sibling CLAUDE.md whose only content is `@AGENTS.md`.
- Skills live in `.agents/skills/` (`.claude/skills` is a symlink). Add a skill only
  after the workflow has been done once for real.

## Git

- Agents must not commit or push. When a commit is needed, provide a suggested
  commit message only; the owner commits and pushes.
- Commits: `<type>(<scope>): <summary>`, e.g. `feat(meta): decode pgoutput inserts`.
  - `type`: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`, `ci`, `chore`.
  - `scope`: crate or package name without the `metrl-` prefix (`scheduler`, `meta`,
    `api`, `worker`, `py`, `engine-mlx`), or `repo` for cross-cutting changes.
  - Summary in imperative mood, lowercase, no trailing period.
