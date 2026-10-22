# Architecture

Metrl is an RL orchestration layer that serves the Tinker HTTP API. Rust servers own the
API, scheduling, tasks, and execution history. Python engines own compute and memory
state. The task log makes training recoverable; user saves are durable snapshots.

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
  meta        Postgres, migrations, CDC ◀── server, bin
  object      object keys, checkpoints  ◀── server, worker
  api         Tinker HTTP routes        ◀── bin
  server      actors, leases, peers     ◀── api, bin
  worker      worker gRPC runtime       ◀── bindings/python
  bin         metrl-server
  bindings/python   metrl-py → metrl._core
python/metrl/worker   Protocols, serve(), metrl-worker
engines/{mlx,mock}    metrl.engines.<name>
proto/                worker.proto, peer.proto, tinker.proto
```

Crates are split by ownership boundary, not by layer: the scheduler is pure so it can be
tested by events alone, `meta` is the only Postgres client, and the worker side never
links the scheduler or Postgres. Protos live in one flat directory; each consuming crate
compiles what it needs with `tonic-prost-build` and the system `protoc`.

## Control plane

Each server owns a logical replication slot `metrl_<server_id>`. Startup creates the
slot, scans the tables, then streams pgoutput; changes are acked after they are applied.
Writes go to Postgres first and reach every server, including the writer, through CDC.
Session heartbeats stay in memory and are flushed every 30s; `sessions` is not published.

## Training path

1. The SDK assigns `seq_id` per model. Any server accepts the request, writes
   `input.pb`, and inserts the task with `ON CONFLICT (model_id, seq) DO NOTHING`; a
   retry returns the existing `request_id`.
2. CDC delivers the task to the owning server's model actor.
3. The actor dispatches only `seq == applied + 1` as `Execute{seq, generation, keys}`.
4. The worker reads the input, runs the engine, writes `g{gen}/output.pb`, and replies.
5. The owner commits the result and `applied`, conditioned on `models.generation`.
   Small results go in `tasks.result jsonb`; fwd/bwd outputs stay in `output.pb`.
6. `retrieve_future` is answered from any server's view. Pending returns 408 with
   `queue_state`.

A seq gap older than 10 minutes fails waiting tasks with 400. A failed task is a no-op
in the log: `applied` advances and replay skips it.

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
| Control state, small task results (`result jsonb`) | Postgres |
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

The org is the isolation boundary: `org_id` on every row, `orgs/{org_id}/` on every
key, 404 across orgs. API keys are stored as sha256 hashes and cached via CDC. Requests
are accepted then queued: 413 above 256 MiB, 429 on model, pending, sampling, or storage
limits, and 408 `paused_capacity` / `paused_rate_limit` while waiting. Workers are shared
and scheduled weighted round-robin per org.

## Configuration

`config/server.toml` and `config/worker.toml`, overridden by `METRL_` environment
variables with `__` between nested keys (`METRL_HTTP_ADDR`, `METRL_ENGINE__MLX__MODEL`).
Secrets come only from the environment. Rust validates its own keys; `[engine.<name>]`
is passed through to the engine. Ports: server 7580 HTTP, 7581 gRPC, 7582 system;
worker 7590 gRPC, 7591 system. System ports are cluster-internal and unauthenticated.

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
