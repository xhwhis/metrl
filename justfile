# Hawk links against compiler internals and only runs on this exact toolchain.
hawk_toolchain := "1.99.0"

default:
    @just --list

# Install Python deps, build metrl._core with stubs, and install git hooks
setup:
    uv sync
    uv run --no-sync maturin develop --uv --generate-stubs
    prek install

# Start Postgres and the object store, and create the bucket
up:
    docker compose up -d --wait
    docker compose run --rm rustfs-init

# Delete all local data and recreate the schema
db-reset:
    docker compose down --volumes
    docker compose up -d --wait
    docker compose run --rm rustfs-init
    cargo run -p metrl-bin --bin metrl-server -- migrate

# Run the server against local dependencies
dev: up
    METRL_OBJECT__SECRET_ACCESS_KEY=metrl-dev-secret cargo run -p metrl-bin --bin metrl-server

# Run a worker against the local object store; config/worker-mlx.toml runs MLX on Qwen3-0.6B
worker config="config/worker.toml":
    METRL_OBJECT__SECRET_ACCESS_KEY=metrl-dev-secret uv run --no-sync metrl-worker --config {{config}}

# Create an API key for an org and print it
key org:
    cargo run -q -p metrl-bin --bin metrl-server -- key create --org {{org}}

# RL acceptance on Qwen3-0.6B against `just dev` and `just worker config/worker-mlx.toml`
rl key:
    TINKER_API_KEY={{key}} uv run --no-sync python scripts/rl_loop.py

fmt:
    cargo fmt --all
    RUST_LOG=warn taplo fmt
    uv run --no-sync ruff check --fix .
    uv run --no-sync ruff format .

lint:
    cargo shear
    typos
    RUST_LOG=warn taplo check
    RUST_LOG=warn taplo fmt --check
    cargo fmt --all --check
    cargo clippy --workspace --all-targets --all-features -- -D warnings
    cargo +{{hawk_toolchain}} hawk check -D warnings --exclude-crate metrl_proto
    uv run --no-sync ruff check .
    uv run --no-sync ruff format --check .
    uv run --no-sync ty check

test: up
    cargo nextest run --workspace --all-targets --all-features --no-tests=pass
    cargo build -p metrl-bin --bin metrl-server --features failpoints
    uv run --no-sync maturin develop --uv --generate-stubs
    uv run --no-sync pytest -q

# E2E on the MLX engine with a tiny Qwen3; macOS only, outside `check`
test-mlx: up
    cargo build -p metrl-bin --bin metrl-server --features failpoints
    uv run --no-sync maturin develop --uv --generate-stubs
    uv run --no-sync pytest -q tests/test_mlx.py

# Everything CI checks except the stable toolchain
check: lint test

# Regenerate proto/tinker.proto from the locked tinker SDK
tinker-proto:
    uv run --no-sync python scripts/dump_tinker_proto.py

# Record a CPU profile of the server with samply
profile:
    cargo build --profile profiling -p metrl-bin --bin metrl-server
    samply record target/profiling/metrl-server
