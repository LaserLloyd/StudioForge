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
just lint     # ruff check + ruff format --check + mypy on the typed core
just test     # pytest tests/unit
just check    # both
```

or, without `just`:

```bash
.venv/Scripts/python.exe -m ruff check src/ tests/ packages/
.venv/Scripts/python.exe -m ruff format --check src/ tests/ packages/
.venv/Scripts/python.exe -m mypy src/studioforge/core src/studioforge/api \
    src/studioforge/db.py src/studioforge/config.py src/studioforge/types.py
.venv/Scripts/python.exe -m pytest tests/unit -q
```

The tree is `ruff format`-clean and `just lint` checks it: run `just format` before committing.
`ruff` and `mypy` are pinned exactly in the `dev` extra because their output changes between
releases; bump the pin and reformat in one commit when moving to a newer one.

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
machine that was, and it turns "not installed here" into a failure instead of a skip. And never
let a test reach the developer's real home directory: anything that writes under `~` (the
`lmstudio://` handler's `mimeapps.list`, `~/.lmstudio`, the Startup folder) must be pointed at
`tmp_path` — one test once took over the developer's real URL handler.

CI (`.github/workflows/ci.yml`) runs `ruff check`, `ruff format --check` and `pytest tests/unit`
on Windows and Ubuntu with `SF_GPU_PROBE=null` (no GPU) and, on the headless Ubuntu leg,
`PYSTRAY_BACKEND=dummy` so the tray tests collect without an X display.

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

  This one is enforced, not trusted. `scripts/scrub_check.py` scans the tree, the index, a commit
  and commit messages; `sh scripts/install-hooks.sh` wires it into pre-commit, commit-msg and
  pre-push, and CI runs it on every push and pull request. Run it once after cloning — nothing
  installs a hook for you.

  It reads the **index**, not the working tree, because those differ: stage a secret, tidy the
  working copy, commit, and a working-tree scan sees nothing. The pre-push scan reads the
  **commits being pushed** for the same reason — a clean checkout says nothing about what is
  already committed behind it.

  Some things it flags are legitimate: this is a LAN server, so private addresses appear
  throughout the docs and the SSRF tests, and a redaction test needs a credential-shaped string.
  Those are exempt under `tests/`. A genuine false positive elsewhere gets an inline
  `scrub-ok: <why>` comment — the marker must be the first thing in the comment, so prose can
  never trigger it by accident. Vendor-prefixed keys (`sk-ant-`, `ghp_`, `AKIA…`), private-key
  blocks and JWTs are never exempt anywhere, including tests.

  Personal identifiers — a name, a machine, a tailnet — live in `scripts/scrub-rules.local.txt`,
  which is git-ignored: publishing the list of words that must never be published is its own leak.
  CI therefore runs with the generic patterns only, and says so in its output. Your clone is where
  the identifier rules actually fire. Every verdict — the clean line and `--selftest`'s `OK` — ends
  with how many identifier rules were loaded, so a "clean" can never be read as more than it was;
  `--require-local-rules` turns that caption into a gate (exit 2 on zero) for a release check on a
  machine that is supposed to have the file. Do not add it to CI, where the file never exists.

## Where things are

`src/studioforge/` is the app (`api/`, `core/`, `gui/`, `mcp/`, `tray/`, `watchdog/`);
`packages/studioforge-companion/` is `sfctl`, a separate package with its own `pyproject.toml`;
`launchers/` holds the Windows `.bat` launchers; `deploy/` the Linux systemd units; `docs/` the
user-facing documentation; `tests/unit/` the suite CI runs and `tests/contract/` the opt-in suite
that needs real GPUs. Decisions with a *why* go in `DECISIONS.md`, numbered.

## Data directory

Everything the app writes lives in the data dir: `SF_DATA_DIR`, else `<repo>/data`, else the
platform directory (DECISIONS.md D25). `data/` is gitignored. Two servers cannot share one data
dir — the second becomes a read-only *secondary* (D24) — so stop one before starting the other.

## Licensing of contributions

The project is **MIT** ([`LICENSE`](LICENSE)). Contributions are accepted under the same terms:
opening a pull request means you agree your work is licensed MIT and that you have the right to
licence it. There is no separate CLA.
