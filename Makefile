# Mirror of the justfile for environments without `just`.
PYTHON ?= .venv/Scripts/python.exe
# One story for the data dir: SF_DATA_DIR, else <repo>/data (gitignored).
SF_DATA_DIR ?= data
export SF_DATA_DIR

.PHONY: run run-dev test test-contract test-all e2e lint format scan config engine check
run: ; $(PYTHON) -m studioforge serve
run-dev: ; $(PYTHON) -m studioforge serve --no-watchdog --port 1299
test: ; $(PYTHON) -m pytest tests/unit -q --timeout=300
# Takes the GPUs: needs SF_RUN_CONTRACT=1 AND -m contract, both deliberate (D23).
test-contract: ; SF_RUN_CONTRACT=1 $(PYTHON) -m pytest -m contract tests/contract -q --timeout=1200
test-all: test test-contract
e2e: ; $(PYTHON) scripts/e2e_matrix.py
lint:
	$(PYTHON) -m ruff check src/ tests/ packages/
	$(PYTHON) -m mypy src/studioforge/core src/studioforge/api src/studioforge/db.py src/studioforge/config.py src/studioforge/types.py
format:
	$(PYTHON) -m ruff format src/ tests/ packages/
	$(PYTHON) -m ruff check --fix src/ tests/ packages/
scan: ; $(PYTHON) -m studioforge scan
config: ; $(PYTHON) -m studioforge config
engine: ; $(PYTHON) -m studioforge engine --smoke-test
check: lint test
