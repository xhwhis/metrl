# Architecture

Metrl is an RL orchestration layer that serves the Tinker HTTP API. Rust servers own the
API, scheduling, tasks, and execution history. Python engines own compute and memory
state. The task log makes training recoverable; user saves are durable snapshots.

## Current implementation

The runnable slice is token-only sampling and LoRA training on the mock engine through
the real Tinker SDK: HTTP → Rust server → task log in Postgres → model actor → gRPC
worker → PyO3 → async Python mock engine → protobuf or JSON future result.

- Control plane: Postgres holds orgs, API keys (sha256 only), workers, models, and
  tasks. Each server follows keys, workers, and tasks through CDC into an in-memory
  view, so changes apply without restarts. All server SQL is in `crates/meta/sql/`;
  test SQL is in `tests/sql/`.
- Workers: a worker registers with the first reachable server in `servers`; the server
  confirms the worker answers `Describe` at its address before recording it. Every
  server dials that address from its own host, so servers reject wildcard addresses
  such as `0.0.0.0`, and loopback addresses unless their gRPC port listens on loopback;
  a worker bound to a wildcard address must set `advertise_addr`. Workers re-register
  every 10s. On SIGINT/SIGTERM a worker unregisters, keeps serving for 1s while the
  removal reaches every server, then lets requests in flight finish for up to 30s; a
  second signal exits at once. Sampling picks a registered worker that serves the base
  and names it in the request, so a different process at that address rejects it.
- Sampling state: sessions, sampling sessions, and results are in the receiving
  server's memory, isolated by org. Retrying the same sampling session and sequence
  returns the same request; changed inputs return 400. Results expire one hour after
  submission, or 60s after a finished result is first read; then they are gone (410)
  and the SDK resubmits. Each org holds at most 1024 sessions, sampling sessions,
  and samples per server.
- Mock sampler: `max_tokens` (1–256), `num_samples` (1–16), and `seed`, with prompts of
  1–4096 nonnegative token IDs. Other options must stay at SDK defaults; other values
  return 400. Output is deterministic tokens, not language-model output.
- Task log: every server renews its model leases every 3s, takes over every model whose
  lease is free or expired, and runs one actor per owned model; it stops all of them
  when no renewal was sent for 7s. A server whose view is stale (30s without a
  heartbeat) renews nothing, so its models move to a server whose stream works. Every
  fenced write and trainer record is bounded by 5s. An actor runs `scheduler::model`
  over the tasks CDC delivers. It binds a trainer that serves the base, records it in
  `models.worker_id`, sends `Execute` for `applied + 1`, and commits the outcome and
  `applied` fenced by generation. Workers keep each model's last response, so an
  owner that repeats a seq after a takeover gets the earlier result instead of a
  second step. Forward-backward inputs and outputs are Tinker protobufs in the object
  store; the worker decodes them into `TensorData`, which crosses to Python through
  the buffer protocol.
- Mock trainer: `cross_entropy` only, with deterministic logprobs. `optim_step` returns
  `state_hash:unique`, a digest of every applied step, so tests can tell whether each
  step applied exactly once.
- Training API: `create_model` checks the base and rank against a registered trainer,
  stores the model, and answers with a finished future; a retry with the same session
  and `model_seq_id` returns the same model. `forward_backward` (also `forward`) and
  `optim_step` append to the task log as described in Training path. `get_info` and
  `retrieve_future` read Postgres or the view. The mock base reports the
  `Qwen/Qwen3-0.6B` tokenizer so that SDK tokenizer calls work, and LoRA rank up to 64.

Not implemented yet: checkpoints and weight saving, so a model whose trainer leaves
fails every later task; loss functions other than `cross_entropy`, `loss_fn_config`,
and optimizers other than AdamW; zstd request bodies; task output expiry;
lease release on shutdown, so a restarted server's models wait up to 10s; placement
beyond hashing the model id; peer routing, MLX, revision conflict checks, dead-worker
detection for workers that exit without unregistering, TLS to Postgres,
authenticated worker registration, and the 408 `paused_capacity` /
`paused_rate_limit` queue states (408 always reports `active`). A server restart loses sampling state. The sections
below describe the target architecture.

Local run from the repository root:

```sh
just setup
just db-reset           # Postgres with an empty schema
just dev                # server on 7580 (HTTP) and 7581 (gRPC)
just worker             # mock worker from config/worker.toml
just key local          # prints a new tml-… key for org "local"
```

Connect with `tinker.ServiceClient(base_url="http://127.0.0.1:7580", api_key=<key>)`
and create sampling or LoRA training clients with `base_model="mock"`. `just up`
creates the `metrl` bucket in the local RustFS. Local Postgres uses trust auth on
127.0.0.1; deployments pass credentials in `METRL_DATABASE_URL`.

`just test` runs the E2E suite: each test clones a migrated template database, uses its
own prefix in the local RustFS, and starts real server and worker processes. Training
tests download the `Qwen/Qwen3-0.6B` tokenizer from Hugging Face once.

## Target topology

```diagram
┌──────────┐ HTTP  ┌────────────────────┐ gRPC  ┌────────────────────────────┐
│ Tinker   │──────▶│ server × N         │──────▶│ worker × M                 │
│ SDK      │       │ api · model actors │◀──────│ Rust runtime + Python      │
└──────────┘       │ scheduler · meta   │ peer  │ engine (trainer / sampler) │
                   └──┬──────────────┬──┘       └─────────────┬──────────────┘
                 SQL+CDC         input blobs          weights, outputs
                   ┌──▼───────┐   ┌──▼────────────────────────▼──┐
                   │ Postgres │   │ object store                 │
                   └──────────┘   └──────────────────────────────┘
```

## Components

- server: Tinker HTTP API (7580), worker and peer gRPC (7581), system endpoints
  (7582). The only process that touches Postgres. Runs one actor per owned model.
- scheduler: pure `on_event(&mut State, now, Event) -> Vec<Effect>`. Actors execute
  effects; all I/O lives in the actor.
- worker: a Rust runtime (gRPC 7590, system 7591, object I/O, safetensors, fencing)
  hosting a Python engine. Engines only compute on tensors in memory.
- Postgres: control state (orgs, keys, models, tasks, checkpoints, workers, sessions).
  Servers keep an in-memory view fed by CDC.
- object store: task inputs and outputs, checkpoints. Write-once; Postgres is the index.

## Repository

```diagram
crates/
  common/{config,error,proto,telemetry,types,utils}   shared by every crate
  scheduler   pure decisions            ◀── server
  meta        Postgres, schema, CDC     ◀── server, bin
  object      object keys, checkpoints  ◀── server, worker
  api         Tinker HTTP routes        ◀── bin
  server      actors, leases, peers     ◀── api, bin
  worker      worker gRPC runtime       ◀── bindings/python
  bin         metrl-server
  bindings/python   metrl-py → metrl._core
python/metrl/worker   Protocols, serve(), metrl-worker
engines/{mlx,mock}    metrl.engines.<name>
proto/                worker.proto, server.proto, peer.proto, tinker.proto
```

Crates are split by ownership boundary, not by layer: the scheduler is pure so it can be
tested by events alone, `meta` is the only Postgres client, and the worker side never
links the scheduler or Postgres. Protos live in one flat directory; each consuming crate
compiles what it needs with `tonic-prost-build` and the system `protoc`.

## Control plane

Each server process gets a random server ID and, per connection attempt, a temporary
logical replication slot `metrl_<server_id>_<attempt>`. The slot exports a snapshot;
the server scans the published tables in that snapshot, then streams binary pgoutput
from the slot. A transaction is applied to the view at its commit and acked after.
Temporary slots die with their connection, so a crashed server leaves no WAL behind;
after any stream error the server rebuilds the view from a new slot and scan. Writes go
to Postgres first and reach every server, including the writer, through CDC.

Each server also writes heartbeat logical messages (Postgres 17+) and reads its own
back, which proves the stream is live and caught up. Authenticated requests and
`/api/v1/healthz` return 503 once the view is not known to be recent. Replication
connections set Postgres timeouts so that a lost link never holds a slot or walsender
for long, and every setup step and pool checkout is bounded. The intervals and timeouts
are in decision 0002.

Target design: Tinker session heartbeats stay in memory and are flushed every 30s;
`sessions` is not published.

## Training path

1. The SDK assigns `seq_id` per model. Any server accepts the request, writes
   `input.pb`, and inserts the task with `ON CONFLICT (model_id, seq) DO NOTHING`; a
   retry returns the existing `request_id`, and a retry with a different input gets
   400. Forward-backward bodies are re-encoded with sorted maps, so retries compare
   byte for byte.
2. CDC delivers the task to the owning server's model actor.
3. The actor dispatches only `seq == applied + 1` as `Execute{seq, generation, keys}`.
4. The worker reads the input, runs the engine, writes `g{gen}/output.pb`, and replies.
   Repeating its last seq returns the kept response, which names the generation that
   ran it.
5. The owner commits the outcome and `applied`, conditioned on `models.generation`.
   Small results go in `tasks.outcome jsonb`; fwd/bwd outputs stay in `output.pb`.
6. `retrieve_future` is answered from any server's view, falling back to Postgres for
   a task the view has not caught up with. The SDK retries 408 and 5xx at once, so the
   server holds a pending or unavailable request for up to 25s, retrying on each view
   change and finished sample, before answering 408 with `queue_state` or the error.
   Shutdown answers held requests at once, so the HTTP drain does not wait for them.

A missing seq that blocks later tasks for 10 minutes fails them with 400, but is never
skipped itself. A failed task is a no-op in the log: `applied` advances and replay skips
it. The ordering rules are the pure `scheduler::model` (decision 0005); the actor feeds
it events and carries out its effects.

## Sampling path

Sampling skips the task log. The receiving server picks a sampler by adapter affinity,
load, and session stickiness, and keeps the result in memory (1h TTL, byte cap). Other
servers fetch it with peer `GetSampleResult`; if the holder is gone they return 410 and
the SDK resubmits. Bases stay resident; LoRA adapters load lazily from immutable paths
into an LRU of 8.

## Ownership and recovery

A model has one owner server, held by a 10s lease renewed every 3s on Postgres `now()`.
Takeover increments `models.generation`, the only fencing token: stale owners' binds,
executes, and commits are rejected. The new owner binds a worker, loads the latest
checkpoint, and replays later tasks. Recovery is semantically equivalent, not bit-exact.
Auto checkpoints run every 20 optimizer steps and on worker SIGTERM; the latest 2 are kept.

## Storage

| Data | Where |
|---|---|
| Control state, small task results (`outcome jsonb`) | Postgres |
| Task inputs, fwd/bwd outputs | `orgs/{org}/tasks/{model}/{seq:020}/{request}/…` |
| Checkpoints | `orgs/{org}/models/{model}/{weights,sampler_weights,checkpoints/auto}/…` |
| Sample results | server memory |

Files are written first and `manifest.json` last; a checkpoint exists only when its
manifest exists and a Postgres row points to it. `tinker://` paths are logical and map to
objects through the `checkpoints` table. Sweep runs under an advisory lock; task outputs
expire after 1h and then return 400.

## Workers and base models

A worker registers with any server; every server dials it. Registration reports bases
from `worker.toml` (`name`, `revision`, `arch`, `tokenizer_id`, `max_context_length`,
`max_lora_rank`, `roles`). Conflicting revisions for the same name are rejected.
`get_server_capabilities` aggregates live workers. CreateModel returns 400 when no live
worker serves the base. Models snapshot `base_model` and `base_revision`. One process
can hold both roles over a shared base, or roles can run separately.

## Rust and Python

`metrl-py` exposes `metrl._core`. Python owns the event loop (uvloop); Rust async
functions are awaitable from Python and run on the pyo3-async-runtimes tokio runtime.
Tensors cross as `TensorData` through the buffer protocol. Engines implement async
`Trainer` and `Sampler` Protocols and run blocking compute on `ComputeThread`. Every
call carries `ctx` (request_id, model_id, seq, trace_id, deadline, `cancelled()`).

## Tenancy and admission

The org is the isolation boundary: `org_id` on every tenant row and query,
`orgs/{org_id}/` on every key, 404 across orgs. Shared cluster rows (`workers`) have no
`org_id`, and the CDC scan reads every org's key hashes. API keys are stored as sha256
hashes and cached via CDC. The API key is checked before a body is read. Requests are
accepted then queued: 413 above 256 MiB for `forward_backward` and 2 MiB elsewhere, 429
on model, pending, sampling, or storage limits (today: 64 pending tasks per model, except
the seq the model runs next; `parallel_fwdbwd_chunks` is off so a split call's chunks
arrive in seq order), and 408 `paused_capacity` / `paused_rate_limit` while waiting.
Workers are shared and scheduled weighted round-robin per org.

## Configuration

`config/server.toml` and `config/worker.toml`, overridden by `METRL_` environment
variables with `__` between nested keys (`METRL_HTTP_ADDR`, `METRL_ENGINE__NAME`).
Secrets come only from the environment. Rust validates its own keys; `[engine] name`
selects the engine package, and the rest of `[engine]` is to be passed through to it.
Ports: server 7580 HTTP, 7581 gRPC, 7582 system; worker 7590 gRPC, 7591 system. System
ports are cluster-internal and unauthenticated. The server gRPC port (worker
registration) is also cluster-internal and unauthenticated, and binds loopback by
default.

## Observability

tracing with W3C traceparent across HTTP and gRPC. Tasks store `trace_ctx` so the owner
continues the submitter's trace. OTLP export is enabled by `[telemetry] otlp_endpoint`.
Prometheus metrics are served at `/metrics` on the system port.

## Operations

- CPU: binaries keep frame pointers and line tables; `just profile` records the server
  with samply on the `profiling` profile.
- Memory: the main in-memory holders (sample results, request bodies, CDC backlog)
  export byte gauges. If RSS keeps diverging from the gauged total in a long Linux run,
  switch the server to jemalloc and add `/debug/pprof/heap` on the system port.
- Continuous profiling with eBPF is deployed with Kubernetes, not built into binaries.
