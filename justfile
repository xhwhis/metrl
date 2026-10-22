# Hawk links against compiler internals and only runs on this exact toolchain.
hawk_toolchain := "1.99.0"

default:
    @just --list

# Install Python deps, build metrl._core with stubs, and install git hooks
setup:
    uv sync
    uv run --no-sync maturin develop --uv --generate-stubs
    prek install

# Start Postgres and the object store
up:
    docker compose up -d --wait

# Stop local dependencies and delete their data
db-reset:
    docker compose down --volumes

# Run the server against local dependencies
dev: up
    cargo run -p metrl-bin --bin metrl-server

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

test:
    cargo nextest run --workspace --all-targets --all-features --no-tests=pass

# Everything CI checks except the stable toolchain
check: lint test

# Regenerate proto/tinker.proto from the locked tinker SDK
tinker-proto:
    uv run --no-sync python scripts/dump_tinker_proto.py

# Record a CPU profile of the server with samply
profile:
    cargo build --profile profiling -p metrl-bin --bin metrl-server
    samply record target/profiling/metrl-server
