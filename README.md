# StudioForge

[![CI](https://github.com/LaserLloyd/StudioForge/actions/workflows/ci.yml/badge.svg)](https://github.com/LaserLloyd/StudioForge/actions/workflows/ci.yml)

A self-hosted, **GPU-only** LLM serving system: an OpenAI-compatible gateway over
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`, with a model registry, a VRAM
planner, a web control panel, and an MCP control plane for agents. It was built to replace LM
Studio as the backend for [OpenClaw](docs/OPENCLAW.md), and it listens on LM Studio's port, so
switching over is a host change rather than a rewrite.

It is a *management* layer — it does not implement inference. Each loaded model is a supervised
`llama-server` child on an internal port, and the gateway reverse-proxies OpenAI-shaped requests to
the right one. That buys per-model crash isolation, per-model engine pinning, and `SIGKILL` as a
guaranteed VRAM-reclaim path.

**GPU-only policy:** models run fully in VRAM or they do not run. There is no CPU inference and no
CPU layer offload anywhere in the codebase. A model that cannot fit entirely on GPU is rejected at
load time with the numbers and an actionable suggestion — never silently degraded into something
twenty times slower.

**Nothing leaves the box.** No telemetry, no analytics, no phone-home. The only outbound requests
are ones you ask for: HuggingFace when you search or download a model, GitHub when you check for a
newer llama.cpp engine, and the image URLs a vision request names.

---

## What it looks like

The control panel at `http://<host>:8080` — every number is the *actual* one the engine
reports (context, slots, VRAM per card), not the one that was asked for:

![Dashboard: GPUs, who holds the VRAM, standing GPU leases, and each loaded model with its live slots](docs/images/dashboard.png)

The Setup tab is a checklist, not a YAML file — it opens on a fresh install and stays green
afterwards; the Models tab is the library, indexed in place, with load / pin / test / benchmark
per row:

| Setup | Models |
| --- | --- |
| ![Setup checklist](docs/images/setup.png) | ![Model library](docs/images/models.png) |

On Windows, StudioForge also lives in the notification area. The tray starts the server, restarts
it if it crashes, and puts the everyday actions one right-click away:

<p align="center"><img src="docs/images/tray-menu.png" alt="The system tray menu: open the control panel, unload models, restart engines or the server, copy the MCP URL and PIN, start at login" width="420"></p>

### The companion: `sfctl`

StudioForge is built to be driven from *another* machine — the one running your agent. The
[`studioforge-companion`](packages/studioforge-companion/) package installs anywhere (no CUDA, no
server dependencies) and gives you:

- **`sfctl`** — a remote control for the rig: status, load/unload/pin, benchmark, download,
  logs, config, engine updates, and `sfctl recover` for when the server is wedged.
- **`sfctl mcp`** — one stdio MCP server that merges the server's management tools *and* the
  watchdog's recovery tools into a single toolset for [OpenClaw](docs/OPENCLAW.md) (or any MCP
  client). When the main server locks up, the agent still holds `restart_server`.

```bash
uv tool install ./studioforge_companion-<version>-py3-none-any.whl
sfctl servers add rig http://<studioforge-host>:1234 --api-key <pin-or-key> --use
sfctl status
```

## Requirements

| | |
| --- | --- |
| OS | Windows 10/11 or Linux |
| GPU | NVIDIA, with a driver new enough for the CUDA build you install (CUDA 13.3 binaries need a 580-series driver or newer) |
| Python | 3.12+ |
| Tooling | [`uv`](https://docs.astral.sh/uv/) |

On Windows no CUDA toolkit is needed: the prebuilt `llama-server` archive ships the runtime, and
StudioForge fetches, verifies and smoke-tests the right one for your driver on first run. On
Linux + NVIDIA the engine is **built from source** (upstream publishes no Linux CUDA archive), so
`git`, `cmake` and a CUDA toolkit with `nvcc` matching the driver are needed once per engine tag.

## Quickstart

```bash
git clone <repo> studioforge && cd studioforge
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"     # Linux: .venv/bin/python
studioforge serve --open
```

On Windows, double-clicking **launchers\Update StudioForge.bat** does the install (and keeps doing
it on later updates), and **launchers\Start StudioForge.bat** starts it. `--open` waits until the
control panel actually answers before opening a browser.

**First run** detects your GPUs, tunes a handful of defaults to them, installs the pinned
`llama-server` build and smoke-tests it before trusting it, and registers your existing model
library **in place** — nothing is copied or moved. The panel opens on its **Setup** tab: a live
checklist of everything still between this box and serving a model, each unmet item with the
button that fixes it, and every configuration key grouped by the decision it makes. You should
never need to edit `config.yaml` by hand. Three things to check there: the engine installed,
`models.dir` points at your GGUF library (**Detect LM Studio library** finds one, including a
relocated `downloadsFolder`), and the MCP pairing PIN you will need to connect an agent.

**[`docs/SETUP.md`](docs/SETUP.md) is the full walkthrough**, tab section by tab section, with
the equivalent YAML and CLI for a box with no browser.

## Ports

| Service | Default | Purpose |
| --- | --- | --- |
| Gateway / management API | `1234` | OpenAI-compatible endpoints + `/api/*` + MCP |
| Web GUI | `8080` | NiceGUI control panel |
| Watchdog | `1235` | Recovery MCP sidecar (separate process) |
| `llama-server` children | `18100–18200` | Internal, one port per loaded model |

> LM Studio also uses port `1234`. They can share the same model library on disk, but not the same
> port — quit LM Studio, or set `server.port` to something else.

With no `server.api_key` (the default) anyone on the LAN can read, chat and load/unload; anything
that changes the *box* — config, engines, files, restarts — needs a browser or client on the
machine itself, or the MCP PIN, on the API, the MCP path and the control panel alike
(DECISIONS.md D32). Set a key to manage it remotely.

## Data directory

Everything the app writes — `config.yaml`, `registry.sqlite3`, `engines/`, `logs/`, `downloads/` —
lives in **one** place, resolved in one order:

1. `SF_DATA_DIR` if it is set;
2. the directory a `--config` (or `SF_CONFIG`) file lives in, when one was named — `config.yaml`
   always sits in its data directory, and that is how the tray, the watchdog and autostart pass
   the location to the processes they spawn;
3. `<repo>/data` when you are running from a checkout (this is the normal case, and `.gitignore`
   keeps it out of the repository);
4. the platform data directory (`%LOCALAPPDATA%\studioforge`, `~/.local/share/studioforge`) for an
   installed wheel.

`config.yaml` itself never names the data directory (a `data_dir` key left by an older build is
ignored with a warning), so copying a config between installs cannot silently move one.

The `.bat` launchers, the `justfile`/`Makefile` and the CLI all follow that rule, so a
double-click and a typed command reach the same install.

**To point this checkout at a data directory that already exists** — an older install, or a second
drive — create `local-env.bat` in the repo root (template: `launchers\local-env.example.bat`):

```bat
set "SF_DATA_DIR=D:\path\to\an\existing\data"
```

Every launcher calls it before doing anything else, and it is gitignored, so machine-specific paths
can never be committed by accident. On Linux, export the same variable from your shell profile or a
systemd unit (see [`deploy/`](deploy/)).

> **One data directory serves one running instance.** Start a second server against the same data
> dir and it comes up as a **secondary**: it answers reads, and it runs no downloader, no TTL
> sweeper and no auto-load, because those want a single writer (DECISIONS.md D24). `GET /health`
> reports `"instance": "secondary"` and names the pid holding it. So before pointing a new checkout
> at an existing data directory, stop the old server and tray first.

## Project layout

| Path | What lives there |
| --- | --- |
| `src/studioforge/` | The application: `api/` (OpenAI-compatible gateway + management routes), `core/` (registry, VRAM planner, supervisor, engine, leases, benchmarks, downloader), `gui/` (the control panel: dashboard, models, download, benchmark, chat, logs, server, setup), `mcp/` (the agent control plane, 19 tools), `tray/` (Windows notification-area app), `watchdog/` (the recovery sidecar, 10 tools), `migrations/` (SQL schema, applied at startup) |
| `packages/studioforge-companion/` | `sfctl` — the thin remote-control CLI and the `sfctl mcp` stdio bridge for OpenClaw; installs anywhere, no CUDA dependencies |
| `launchers/` | Windows double-click launchers (below) |
| `deploy/` | Linux: systemd user units for the server and the watchdog |
| `docs/` | Setup, the catalog, OpenClaw integration, the benchmarking playbook, the runbook, engine features, limitations |
| `tests/unit/` | The suite CI runs (no GPU, no network); `tests/contract/` needs real engines and weights and is opt-in |
| `DECISIONS.md` | The running architectural decision log, D1 onward — the *why* behind every non-obvious rule |
| `config.example.yaml` | Every config key with its shipped default, annotated (a unit test keeps it so); the app writes its own `config.yaml` into the data dir |

## Windows: double-click launchers

Everything a Windows user needs is in [`launchers/`](launchers/) — five `.bat` files, none of which
need a terminal or admin rights:

| File | What it does |
| --- | --- |
| **Start StudioForge.bat** | Starts the server and opens the control panel once it is actually up |
| **StudioForge Tray.bat** | Puts StudioForge in the notification area: it starts the server, restarts it if it crashes, and offers start/stop, free VRAM and copy-the-MCP-URL from the icon |
| **Open StudioForge GUI.bat** | Opens the control panel of an already-running server |
| **StudioForge Autostart.bat** | Turns "start when I log in" on or off (tray, or server only) |
| **Update StudioForge.bat** | Pulls code (if a git remote exists), syncs dependencies, updates the llama.cpp engine, then verifies |

They resolve the repo from their own location, so they work from a shortcut on the desktop too.
`launchers\local-env.example.bat` is the template for keeping your data outside the checkout
(copy it to the repo root as `local-env.bat`; see *Data directory* above).

The same things from a terminal, on any platform:

```bash
studioforge serve --open          # start, and open the GUI when it is ready
studioforge gui                   # open the GUI of a running server
studioforge scan                  # inventory the model library without starting a server
studioforge config                # show the effective config, secrets redacted
studioforge engine --check        # is there a newer llama.cpp release?
studioforge engine --update       # install it, smoke-test it, then pin it
studioforge autostart enable      # start at login (Startup folder / systemd --user)
```

`engine --update` only repins after the new build passes its smoke test, so a broken release can
never become the default. Running instances keep the engine they started with; reload a model to
move it onto the new one.

---

## Using it from OpenClaw (or any OpenAI client)

Inference is a base-URL change; management is one stdio MCP server on the agent's machine:

```bash
OPENAI_BASE_URL=http://<studioforge-host>:1234/v1
OPENAI_API_KEY=<server.api_key, or any non-empty string when auth is disabled>
```

```bash
uv tool install ./studioforge_companion-<version>-py3-none-any.whl   # from `uv build --wheel`
sfctl servers add rig http://<studioforge-host>:1234 --api-key <PIN or server.api_key>
```

```json
{ "mcpServers": { "studioforge": { "command": "sfctl", "args": ["mcp"] } } }
```

`GET /v1/models` lists every **downloaded** model (loaded or not), exactly like LM Studio, and
naming an unloaded model in a chat request just-in-time loads it. `sfctl mcp` merges the server's
19 management tools and the watchdog's 10 recovery tools into one list, so the agent keeps
`restart_server`, `kill_model` and `tail_logs` even when the main server is wedged. There is no
inference tool by design; generation goes over `/v1/chat/completions`, which streams.

The loop an agent runs — `list_models` (every model with a `recommended` load and a per-context
`options` table, nothing left to compute), `load_recommended(model_id, ctx_size)` (loads at
**exactly** that context or refuses with numbers), `search_models` → `repo_details` →
`download_model`, `pin_model`, `reserve_gpus` — is in [`docs/OPENCLAW.md`](docs/OPENCLAW.md);
every catalog column and its formula is in [`docs/CATALOG.md`](docs/CATALOG.md). Set
`models.default_model` to serve requests that name no model at all.

---

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests/unit -q      # the fast suite: no GPU, no engine, no network
```

`tests/contract` starts a **real** gateway with a **real** engine and loads **real** weights onto
your GPUs; it is deselected by default and additionally gated on `SF_RUN_CONTRACT=1`
(DECISIONS.md D23). How to run it, which unit tests use real artefacts when the machine has them,
lint and types: [`CONTRIBUTING.md`](CONTRIBUTING.md) and
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

## Documentation

- [`docs/SETUP.md`](docs/SETUP.md) — first run, tab by tab; the GPU policy knobs; the engine
  (Windows download, Linux source build); network, credentials and the D32 admin rule;
  downloading models; the headless YAML and CLI
- [`docs/OPENCLAW.md`](docs/OPENCLAW.md) — pointing OpenClaw at it, the tool list, the loop an agent
  runs, pins and leases, default models, reliability guarantees, troubleshooting
- [`docs/OPENCLAW-SETUP.md`](docs/OPENCLAW-SETUP.md) — a step-by-step two-machine install with a
  verification after each step (all hostnames and addresses are placeholders)
- [`docs/CATALOG.md`](docs/CATALOG.md) — the model catalog an agent picks from: every column,
  the speed formulas, calibration, `planner.preference`
- [`docs/OPENCLAW-LONG-CONTEXT.md`](docs/OPENCLAW-LONG-CONTEXT.md) — what a long window really
  costs, with three models' whole option tables and the measured numbers behind them
- [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) — the benchmarking playbook for the agent: the
  two benchmarks, the exact calls, the three rules, and locking a result in with a lease
- [`docs/ENGINE-FEATURES.md`](docs/ENGINE-FEATURES.md) — the llama.cpp features StudioForge turns
  on and the ones it deliberately does not, each with its default, quality cost and measurement
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — the operator's runbook: what `/health` is telling you, and
  what to do when it will not start, a model will not load, VRAM is held, or a download stalls
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) — working on the code: scratch data dir, the shape
  of the code, adding a config key, headless Linux, the companion wheel
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) — known limitations, honestly
- [`docs/COMPARISON.md`](docs/COMPARISON.md) — what was borrowed from Ollama, oobabooga, KoboldCpp, vLLM
- [`DECISIONS.md`](DECISIONS.md) — architectural decisions, each with the measurement behind it

## License

**Not yet chosen — all rights reserved for now.** There is deliberately no `LICENSE` file: until
one is picked, no permission to copy, modify or redistribute is granted.
