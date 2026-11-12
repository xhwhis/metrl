# Architecture

Metrl is an RL orchestration layer that serves the Tinker HTTP API. Rust servers own the
API, scheduling, tasks, and execution history. Python engines own compute and memory
state. The task log makes training recoverable; user saves are durable snapshots.

## Current implementation

The runnable slice is token-only sampling of bases and saved adapters, and LoRA training,
on the mock engine, and on the MLX engine on Apple silicon, through
the real Tinker SDK: HTTP → Rust server → task log in Postgres → model actor → gRPC
worker → PyO3 → async Python engine → protobuf or JSON future result.

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
  second signal exits at once. Sampling picks the registered worker that serves the
  base and ranks first for the checkpoint id, or for the sampling session on a bare base
  (rendezvous hashing, so every server agrees and a joining or leaving worker moves only
  its own keys), and names it in the request, so a different process at that address
  rejects it. The worker gives the engine an `Adapter` whose `key` is the checkpoint's
  immutable prefix and whose `load()` downloads and decodes it in Rust; engines cache
  loaded adapters by key.
- Sampling state: sessions, sampling sessions, and results are in the receiving
  server's memory, isolated by org. Retrying the same sampling session and sequence
  returns the same request; changed inputs return 400. Results expire one hour after
  submission, or 60s after a finished result is first read; then they are gone (410)
  and the SDK resubmits. Each org holds at most 1024 sessions, sampling sessions,
  and samples per server.
- Sampling options: `max_tokens` (1–256), `num_samples` (1–16), `seed`, `temperature`
  (≥ 0), `top_k` (-1 or ≥ 1), `top_p` (0, 1], `stop` (at most 64 strings or token ids),
  and `prompt_logprobs`, with prompts of 1–4096 nonnegative token IDs. Other values
  return 400, as do top-k logprobs, target or alternative prompt logprobs,
  `prompt_logprobs_last_n`, and stability info. Sampled logprobs are under the
  distribution the token was drawn from, after temperature, `top_k`, and `top_p`; prompt
  logprobs are under the model's own distribution. An engine's rejection, such as a
  string stop on the mock, fails the future as a user error with the engine's message.
- Mock sampler: a fake language model over token IDs below 256 whose logits follow from
  the weights (base name, or adapter digest) and the tokens before. A seed reproduces
  samples, and no seed draws a fresh one. It has no tokenizer and no end-of-sequence
  token, so only token-id stops end a sequence early, and prompt logprobs need prompt
  IDs below 256. It keeps the 8 most recently used adapters, and requests that arrive
  while one loads wait for that load.
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
- Mock trainer: `cross_entropy` (`target_tokens`, `weights`) and `importance_sampling`
  (`target_tokens`, sampler `logprobs`, `advantages`; loss `-Σ exp(logprob − sampler
  logprob) · advantage`), both returning `logprobs` and `loss:sum`. Logprobs come from
  the mock sampler's language model under the adapter the trainer would save, so
  on-policy logprobs match the sampler's and every importance ratio is 1; targets must be
  below 256. `optim_step` returns `state_hash:unique`, a digest of every applied step, so
  tests can tell whether each step applied exactly once.
- MLX engine (Apple silicon): loads each base with `mlx_lm` once for both roles; its
  revision is a digest of the loaded weights and config, and its `tokenizer_id` is the
  Hugging Face repository it loaded, or the name for a local directory. LoRA of rank
  up to 64 covers every attention and MLP linear (not the unembedding yet), scaled by
  32 / rank as in Tinker, and trains with AdamW (bias-corrected, global grad-norm
  clipping); losses as on the mock, with logprobs in float32. All models share the
  resident base: each call installs its own low-rank deltas first. Sampling decodes all
  samples of a request as one batch over the prompt's KV cache; without `stop` it ends
  at the tokenizer's end-of-sequence tokens, and stop strings end a sequence once its
  decoded completion contains one. Inputs shorter than 9 tokens are padded for training,
  since MLX 0.32's short-query attention kernels have no backward pass.
- Training API: `create_model` checks the base and rank against a registered trainer,
  stores the model, and answers with a finished future; a retry with the same session
  and `model_seq_id` returns the same model. `forward_backward` (also `forward`) and
  `optim_step` append to the task log as described in Training path. `get_info` and
  `retrieve_future` read Postgres or the view. The mock base reports the
  `Qwen/Qwen3-0.6B` tokenizer so that SDK tokenizer calls work, and LoRA rank up to 64.
- Sampler saves: `save_weights_for_sampler` is a task. The trainer writes
  `adapter_model.safetensors`, `adapter_config.json`, then `manifest.json` under
  `orgs/{org}/models/{model}/sampler_weights/{request}/g{generation}/`, and the owner
  inserts the `checkpoints` row in the statement that commits the outcome. A named
  save answers with `tinker://{model}/sampler_weights/{name}`, and a name already
  used by the model fails the task as a user error. An unnamed save answers with the
  checkpoint's id as a sampling session. `create_sampling_session` and `asample`
  find a checkpoint in the view, or in Postgres when the view lags a save that
  `retrieve_future` already answered from Postgres. `GET /samplers/{id}` answers
  with the session's base model and, for a named save, its `tinker://` path; the SDK
  calls it when it creates a sampling client from a path. Only `snapshot` saves are
  supported.

Not implemented yet: training-state saves, and recovery from a
checkpoint, so a model whose trainer leaves fails every later task; checkpoint expiry
and sweep; loss functions other than `cross_entropy` and `importance_sampling`,
`loss_fn_config`,
and optimizers other than AdamW; zstd request bodies; task output expiry;
lease release on shutdown, so a restarted server's models wait up to 10s; placement
by load; peer routing, revision conflict checks, dead-worker
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
just worker config/worker-mlx.toml   # or MLX on Qwen3-0.6B, on Apple silicon
just key local          # prints a new tml-… key for org "local"
```

Connect with `tinker.ServiceClient(base_url="http://127.0.0.1:7580", api_key=<key>)`
and create sampling or LoRA training clients with `base_model="mock"` (or
`"Qwen/Qwen3-0.6B"` on the MLX worker). With the MLX worker running, `just rl <key>`
runs `scripts/rl_loop.py`, the M2 acceptance: Qwen3-0.6B learns two-digit
multiplication by RL through the SDK, and it fails unless the reward on held-out
problems rises by 0.1. `just up`
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
then session stickiness, then load, and keeps the result in memory (1h TTL, byte cap). Other
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
Tensors cross as `TensorData` through the buffer protocol. An engine package exports
`load(settings, bases, compute)`, which gets its `[engine.<name>]` table, the configured
bases, and the worker's one `ComputeThread`, and returns the loaded `Base` for each
configured base plus a `Trainer` and a `Sampler`. Their async methods follow Tinker
names (`create_model`, `forward`, `forward_backward`, `optim_step`,
`save_weights_for_sampler`, `sample`), carry `model_id`, and run blocking compute on
the compute thread with `await compute.run(fn)`, one call at a time. Engines return
tensors and adapter metadata only; the worker's Rust runtime encodes and stores them.
Every call is to carry `ctx` (request_id, model_id, seq, trace_id, deadline,
`cancelled()`); today calls carry `model_id` only.

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
selects the engine package, which gets its `[engine.<name>]` table as is and validates
it.
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
