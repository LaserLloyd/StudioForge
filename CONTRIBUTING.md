# Contributing

## Environment

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"   # Linux: .venv/bin/python
```

`uv` is the only required tool. `just` is optional — `Makefile` mirrors every recipe for
environments without it.

## The three checks

```bash
just lint     # ruff check + mypy on the typed core
just test     # pytest tests/unit
just check    # both
```

or, without `just`:

```bash
.venv/Scripts/python.exe -m ruff check src/ tests/ packages/
.venv/Scripts/python.exe -m mypy src/studioforge/core src/studioforge/api \
    src/studioforge/db.py src/studioforge/config.py src/studioforge/types.py
.venv/Scripts/python.exe -m pytest tests/unit -q
```

`ruff format` is available (`just format`) but is not enforced in CI; matching the surrounding
style matters more than running it over a file you barely touched.

mypy runs in strict-ish mode (`disallow_untyped_defs`) over `core`, `api`, `db`, `config` and
`types`. The GUI is exempt (`studioforge.gui.*`), because NiceGUI's declarative style produces a
lot of untyped callbacks for no safety gain.

## Tests

**Run `pytest tests/unit`. Never a bare `pytest`, and never `pytest tests`.**

`tests/contract` starts a real gateway with a real engine and loads real weights onto real GPUs.
It is deselected by default (`addopts = -m 'not contract'`) *and* gated on `SF_RUN_CONTRACT=1`,
because an agent once ran `pytest tests` and left three `llama-server` children holding ~25 GiB of
VRAM after the run finished (DECISIONS.md D23). Keep both gates.

A few unit tests use real artefacts **when this machine happens to have them** and skip cleanly
when it does not:

| Test | Needs | Found via |
| --- | --- | --- |
| `test_gguf.py` "Real library" | a GGUF library to parse | `SF_TEST_MODELS_DIR`, else the auto-detected LM Studio library |
| `test_registry.py::test_live_real_library` | same | same |
| `test_engine.py` `needs_engine` | an installed `llama-server` | `<SF_DATA_DIR>/engines/<tag>/`, else `<repo>/data/engines/<tag>/` |
| `test_supervisor.py::test_live_real_llama_server` | engine **and** a tiny model — **this one loads onto a GPU** | as above, plus `SF_TEST_MODELS_DIR` |

Never hard-code an absolute path into a test. It is right on exactly one machine, it leaks whose
machine that was, and it turns "not installed here" into a failure instead of a skip.

## Writing tests

Tests assert behaviour, not implementation, and their names say what is guaranteed
(`test_a_reachable_watchdog_gets_the_job_and_the_reply_says_so`). When a test exists because
something once went wrong in production, say so in the docstring — that sentence is why the test
survives a refactor.

## DECISIONS.md

Architectural decisions go in [`DECISIONS.md`](DECISIONS.md), appended as `D<n>`, newest last.
The convention is one section per decision containing **the decision, and the measurement or
incident that produced it**. An entry that says what was chosen but not why is not worth writing;
several entries here exist purely because a number in a docstring turned out to be wrong when
someone finally measured it.

Reference decisions by number from code comments and docs (`see DECISIONS.md D17`) rather than
restating the reasoning in three places.

## Things that are deliberate

- **GPU-only.** No CPU offload flags anywhere. A model that does not fit is rejected with the
  arithmetic, not silently made twenty times slower.
- **Never implement inference.** Anything a `llama-server` flag can do belongs in the launch
  arguments, not in Python.
- **No telemetry, ever.** The only outbound requests are HuggingFace, GitHub release checks, and
  image URLs a request names.
- **No personal data in the repository.** No absolute home paths, no hostnames, no tailnet
  addresses, no PINs or tokens — in code, tests, docs or fixtures. Machine-specific settings go in
  `local-env.bat` (gitignored) or `SF_DATA_DIR`.

## Data directory

Everything the app writes lives in the data dir: `SF_DATA_DIR`, else `<repo>/data`, else the
platform directory (DECISIONS.md D25). `data/` is gitignored. Two servers cannot share one data
dir — the second becomes a read-only *secondary* (D24) — so stop one before starting the other.
