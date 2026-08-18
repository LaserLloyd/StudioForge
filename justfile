# StudioForge developer tasks. `just` (or `make -f Makefile`) runs these.
set windows-shell := ["bash", "-c"]

# Windows layout by default; on Linux/macOS override: `just python=.venv/bin/python ...`
python := ".venv/Scripts/python.exe"
# One story for the data dir: SF_DATA_DIR, else <repo>/data (gitignored).
data   := env_var_or_default("SF_DATA_DIR", "data")

default:
    @just --list

# Run the server (GUI on 8080, watchdog sidecar spawned).
run:
    SF_DATA_DIR="{{data}}" {{python}} -m studioforge serve

# Run without the watchdog, on an alternate port (1234 is often LM Studio's).
run-dev:
    SF_DATA_DIR="{{data}}" {{python}} -m studioforge serve --no-watchdog --port 1299

# Fast suite: no GPU, no engine, no network.
test:
    SF_DATA_DIR="{{data}}" {{python}} -m pytest tests/unit -q --timeout=300

# OpenAI parity suite against a real engine and real weights.
# Takes the GPUs: needs SF_RUN_CONTRACT=1 AND -m contract, both deliberate (D23).
test-contract:
    SF_RUN_CONTRACT=1 SF_DATA_DIR="{{data}}" {{python}} -m pytest -m contract tests/contract -q --timeout=1200

test-all: test test-contract

# The seven-scenario live acceptance matrix.
e2e:
    SF_DATA_DIR="{{data}}" {{python}} scripts/e2e_matrix.py

lint:
    {{python}} -m ruff check src/ tests/ packages/
    {{python}} -m ruff format --check src/ tests/ packages/
    {{python}} -m mypy src/studioforge/core src/studioforge/api src/studioforge/db.py src/studioforge/config.py src/studioforge/types.py

format:
    {{python}} -m ruff format src/ tests/ packages/
    {{python}} -m ruff check --fix src/ tests/ packages/

# Inventory the model library without starting a server.
scan:
    SF_DATA_DIR="{{data}}" {{python}} -m studioforge scan

# Show the effective config (secrets redacted).
config:
    SF_DATA_DIR="{{data}}" {{python}} -m studioforge config

# Install / verify the pinned llama.cpp engine.
engine:
    SF_DATA_DIR="{{data}}" {{python}} -m studioforge engine --smoke-test

check: lint test
