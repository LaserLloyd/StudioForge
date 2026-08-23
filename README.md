# StudioForge

[![CI](https://github.com/LaserLloyd/StudioForge/actions/workflows/ci.yml/badge.svg)](https://github.com/LaserLloyd/StudioForge/actions/workflows/ci.yml)

A self-hosted, **GPU-only** LLM server: an OpenAI-compatible gateway over
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`, with a model registry, a VRAM
planner, a web control panel, a system tray, and an MCP control plane for agents. Built to replace
LM Studio as the backend for [OpenClaw](docs/OPENCLAW.md) — it listens on LM Studio's port, so
switching is a host change, not a rewrite.

- **GPU-only.** A model runs fully in VRAM or it does not run; a load that will not fit is refused
  with the numbers and a suggestion, never silently slowed down.
- **Nothing leaves the box.** No telemetry. The only outbound requests are the ones you ask for
  (HuggingFace downloads, llama.cpp release checks, image URLs a vision request names).

---

## Install

### Windows

1. Install [Git](https://git-scm.com/download/win), [Python 3.12+](https://www.python.org/downloads/)
   and [uv](https://docs.astral.sh/uv/getting-started/installation/). Have a current
   [NVIDIA driver](https://www.nvidia.com/drivers/) (no CUDA toolkit needed — the prebuilt engine
   ships its runtime).
2. Clone:

   ```bash
   git clone https://github.com/LaserLloyd/StudioForge.git
   ```

3. Double-click **`launchers\Update StudioForge.bat`** — creates the virtualenv, installs the
   dependencies, downloads and smoke-tests the pinned `llama-server` build for your driver.
4. Double-click **`launchers\Start StudioForge.bat`** (or **`launchers\StudioForge Tray.bat`** for
   a notification-area icon that keeps the server alive). The control panel opens at
   <http://127.0.0.1:8080> on its **Setup** tab — a checklist with a button for each unmet item.

### Linux

1. Install [Git](https://git-scm.com/downloads), [Python 3.12+](https://www.python.org/downloads/),
   [uv](https://docs.astral.sh/uv/getting-started/installation/), `cmake`, and the
   [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) whose `nvcc` matches your driver.
   Upstream ships no Linux CUDA archive, so the engine is **built from source once per version**.
2. Clone and install:

   ```bash
   git clone https://github.com/LaserLloyd/StudioForge.git && cd StudioForge
   uv venv --python 3.12 .venv
   uv pip install --python .venv/bin/python -e ".[dev]"
   ```

3. Start it (the first run builds the engine; watch the progress in the terminal):

   ```bash
   .venv/bin/studioforge serve --open
   ```

4. To run it as a service that survives logout, use the systemd user units in
   [`deploy/`](deploy/README.md).

### First run, either platform

The Setup tab asks you to confirm three things: the engine is installed, `models.dir` points at
your GGUF library (**Detect LM Studio library** finds an existing one — models are indexed in
place, never copied), and you have noted the MCP pairing PIN you will need for an agent. The full
walkthrough, with the headless YAML/CLI equivalent, is [`docs/SETUP.md`](docs/SETUP.md).

---

## Three ways to set it up

### A. Let an AI coding agent do it

Open [Claude Code](https://docs.anthropic.com/en/docs/claude-code), a DeepSeek-based harness, or
any coding agent that can run shell commands **in the cloned folder**, and paste:

> Install StudioForge on this machine by following `docs/SETUP.md`: create the virtualenv with uv,
> install the llama.cpp engine with `studioforge engine --update`, point `models.dir` at my GGUF
> library, start it with `studioforge serve --open`, and confirm `GET http://127.0.0.1:1234/health`
> reports `"can_serve": true`. Only run `tests/unit`, never `tests/contract`.

The repo is written for that: [`DECISIONS.md`](DECISIONS.md) explains every non-obvious rule,
[`docs/RUNBOOK.md`](docs/RUNBOOK.md) says what each `/health` field means, and
[`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) is a playbook an agent can follow verbatim.

### B. Manually, as an API (any OpenAI-compatible client)

Once the server is up, it *is* a cloud-style API on your own hardware. Point any OpenAI client at it:

```bash
export OPENAI_BASE_URL=http://<studioforge-host>:1234/v1
export OPENAI_API_KEY=not-required        # any non-empty string while server.api_key is unset
```

```bash
curl http://<studioforge-host>:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<id from /v1/models>", "messages": [{"role": "user", "content": "hello"}]}'
```

```python
from openai import OpenAI  # https://github.com/openai/openai-python
client = OpenAI(base_url="http://<studioforge-host>:1234/v1", api_key="not-required")
print(client.models.list())   # every downloaded model; naming an unloaded one loads it on demand
```

`GET /v1/models` lists every **downloaded** model, like LM Studio, and a request that names an
unloaded model just-in-time loads it. To use it from outside your LAN, set `server.api_key` on the
Setup tab and pass that key — see [`docs/SETUP.md` §6](docs/SETUP.md#6-network--access).

### C. With OpenClaw (or any MCP agent)

Inference is path B. Management — load, unload, pin, benchmark, download, recover — is one stdio
MCP server on the agent's machine, provided by the companion package:

```bash
uv build --wheel -o dist                                              # in packages/studioforge-companion, on the rig
uv tool install ./studioforge_companion-<version>-py3-none-any.whl    # on the agent's machine
sfctl servers add rig http://<studioforge-host>:1234 --api-key <PIN or server.api_key> --use
```

```json
{ "mcpServers": { "studioforge": { "command": "sfctl", "args": ["mcp"] } } }
```

That gives the agent 29 tools: the server's 19 management tools plus the watchdog's 10 recovery
tools, so it still holds `restart_server` when the main server is wedged. The step-by-step
two-machine install with a check after each step is
[`docs/OPENCLAW-SETUP.md`](docs/OPENCLAW-SETUP.md); the loop an agent actually runs is
[`docs/OPENCLAW.md`](docs/OPENCLAW.md).

---

## What it looks like

Every number in the panel is the *actual* one the engine reports — context, slots, VRAM per card:

![Dashboard: GPUs, who holds the VRAM, standing GPU leases, and each loaded model with its live slots](docs/images/dashboard.png)

| Setup — a checklist, not a YAML file | Models — the library, indexed in place |
| --- | --- |
| ![Setup checklist](docs/images/setup.png) | ![Model library](docs/images/models.png) |

On Windows, the tray starts the server, restarts it if it crashes, and keeps the everyday actions
one right-click away:

<p align="center"><img src="docs/images/tray-menu.png" alt="The system tray menu: open the control panel, unload models, restart engines or the server, copy the MCP URL and PIN, start at login" width="420"></p>

## The companion: `sfctl`

[`packages/studioforge-companion`](packages/studioforge-companion/) installs anywhere — no CUDA, no
server dependencies — and is the remote control for the rig: `sfctl status`, `models load/unload/pin`,
`download`, `logs`, `config`, `update`, and `sfctl recover` for when the server is wedged.
`sfctl mcp` is the stdio MCP bridge from path C.

## Ports

| Service | Default | Purpose |
| --- | --- | --- |
| Gateway / management API | `1234` | OpenAI-compatible endpoints, `/api/*`, `/mcp` |
| Web GUI | `8080` | The control panel |
| Watchdog | `1235` | Recovery MCP sidecar (separate process) |
| `llama-server` children | `18100–18200` | Internal, one port per loaded model |

LM Studio also uses `1234`: quit it, or set `server.port`. With no `server.api_key` (the default)
anyone on your LAN can read, chat and load/unload; anything that changes the *box* — config,
engines, files, restarts — needs a browser or client on the machine itself, or the MCP PIN
([DECISIONS.md](DECISIONS.md) D32). Set a key to manage it remotely.

## Data directory

Everything the app writes — `config.yaml`, `registry.sqlite3`, `engines/`, `logs/`, `downloads/` —
lives in one place: `SF_DATA_DIR` if set; else the folder a `--config` file lives in; else
`<repo>/data` in a checkout (gitignored); else the platform data directory. To keep your data
outside the checkout, copy [`launchers\local-env.example.bat`](launchers/local-env.example.bat) to
the repo root as `local-env.bat` and set `SF_DATA_DIR` there (Linux: export it in your shell or the
systemd unit). Every launcher reads it, and it is gitignored.

One data directory serves one running instance; a second server on the same directory comes up
read-only (`"instance": "secondary"` in `/health`). Stop the old server and tray before pointing a
new checkout at an existing data directory.

## Windows launchers

All in [`launchers/`](launchers/); none need a terminal or admin rights, and they work from a
desktop shortcut.

| File | What it does |
| --- | --- |
| **Start StudioForge.bat** | Starts the server and opens the control panel once it is actually up |
| **StudioForge Tray.bat** | Notification-area icon: starts the server, restarts it if it crashes, start/stop, free VRAM, copy the MCP URL |
| **Open StudioForge GUI.bat** | Opens the control panel of an already-running server |
| **StudioForge Autostart.bat** | Turns "start when I log in" on or off (tray, or server only) |
| **Update StudioForge.bat** | Pulls code (if a git remote exists), syncs dependencies, updates the llama.cpp engine, verifies |

The same from a terminal, on any platform: `studioforge serve --open`, `studioforge gui`,
`studioforge scan`, `studioforge config`, `studioforge engine --check | --update`,
`studioforge autostart enable`.

## Project layout

| Path | What lives there |
| --- | --- |
| `src/studioforge/` | The app: `api/` (gateway + management routes), `core/` (registry, VRAM planner, supervisor, engine, leases, benchmarks, downloader), `gui/`, `mcp/` (19 tools), `tray/`, `watchdog/` (10 tools), `migrations/` |
| `packages/studioforge-companion/` | `sfctl` and the `sfctl mcp` bridge |
| `launchers/` | Windows double-click launchers |
| `deploy/` | Linux systemd user units |
| `docs/` | Setup, OpenClaw, the catalog, benchmarking, the runbook, limitations |
| `tests/unit/` | The suite CI runs (no GPU, no network); `tests/contract/` needs real engines and weights, opt-in |
| `DECISIONS.md` | The architectural decision log, D1 onward — the *why* behind every rule |
| `config.example.yaml` | Every config key with its shipped default, annotated |

## Documentation

- [`docs/SETUP.md`](docs/SETUP.md) — first run, tab by tab; GPUs, engine, network, downloads; headless
- [`docs/OPENCLAW-SETUP.md`](docs/OPENCLAW-SETUP.md) — two-machine install, verified step by step
- [`docs/OPENCLAW.md`](docs/OPENCLAW.md) — the tool list and the loop an agent runs; pins and leases
- [`docs/CATALOG.md`](docs/CATALOG.md) — the model catalog an agent picks from, every column explained
- [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) — the benchmarking playbook
- [`docs/OPENCLAW-LONG-CONTEXT.md`](docs/OPENCLAW-LONG-CONTEXT.md) — what a long window really costs
- [`docs/ENGINE-FEATURES.md`](docs/ENGINE-FEATURES.md) — llama.cpp features on, off, and measured
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — what `/health` means and what to do when it is wrong
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) · [`CONTRIBUTING.md`](CONTRIBUTING.md) — working on the code, running the tests
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) · [`docs/COMPARISON.md`](docs/COMPARISON.md) — known limits; what was borrowed from Ollama, oobabooga, KoboldCpp, vLLM
- [`DECISIONS.md`](DECISIONS.md) — architectural decisions, each with the measurement behind it

## License

**Not yet chosen — all rights reserved for now.** There is deliberately no `LICENSE` file: until
one is picked, no permission to copy, modify or redistribute is granted.
