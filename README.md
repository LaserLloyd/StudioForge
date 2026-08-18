# StudioForge

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

## Requirements

| | |
| --- | --- |
| OS | Windows 10/11 or Linux |
| GPU | NVIDIA, with a driver new enough for the CUDA build you install (CUDA 13.3 binaries need a 580-series driver or newer) |
| Python | 3.12+ |
| Tooling | [`uv`](https://docs.astral.sh/uv/) |

No CUDA toolkit install is needed: the prebuilt `llama-server` binaries ship the runtime, and
StudioForge fetches and verifies the right one for your hardware on first run.

## Quickstart

```bash
git clone <repo> studioforge && cd studioforge
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"     # Linux: .venv/bin/python
```

On Windows, double-clicking **Update StudioForge.bat** does the same thing (and keeps doing it on
later updates). Then start it:

```bash
studioforge serve --open
```

`--open` waits until the control panel actually answers before opening a browser, so you never land
on a connection-refused page during a slow first start.

**First run** detects your GPUs, tunes a handful of defaults to them, fetches the pinned
`llama-server` build, smoke-tests it before trusting it, and registers your existing model library
**in place** — nothing is copied or moved. The panel opens on its **Setup** tab, which is where the
whole first run happens: a live checklist of everything still standing between this box and serving
a model, each unmet item with the button that fixes it, and every configuration key StudioForge has
— grouped by the decision it makes, not by the file it lives in. You should never need to edit
`config.yaml` by hand.

Three things are worth checking there afterwards:

1. **The engine installed.** If it did not, the checklist offers an Install button, and
   `studioforge engine --update` says why from a terminal.
2. **`models.dir` points at your GGUF library.** "Detect LM Studio library" probes for one
   (including the relocated `downloadsFolder` recorded in `~/.lmstudio/settings.json`, which is
   where most non-default installs actually live), but any directory of GGUFs works.
3. **The MCP pairing PIN**, shown masked on the Setup tab, printed in the startup banner and served
   at `GET /api/mcp/info`. You need it to connect an agent.

**[`docs/SETUP.md`](docs/SETUP.md) is the full walkthrough**, tab section by tab section, and it
gives the equivalent YAML and CLI for a box with no browser.

## Ports

| Service | Default | Purpose |
| --- | --- | --- |
| Gateway / management API | `1234` | OpenAI-compatible endpoints + `/api/*` + MCP |
| Web GUI | `8080` | NiceGUI control panel |
| Watchdog | `1235` | Recovery MCP sidecar (separate process) |
| `llama-server` children | `18100–18200` | Internal, one port per loaded model |

> LM Studio also uses port `1234`. They can share the same model library on disk, but not the same
> port — quit LM Studio, or set `server.port` to something else.

## Data directory

Everything the app writes — `config.yaml`, `registry.sqlite3`, `engines/`, `logs/`, `downloads/` —
lives in **one** place, resolved in one order:

1. `SF_DATA_DIR` if it is set;
2. `<repo>/data` when you are running from a checkout (this is the normal case, and `.gitignore`
   keeps it out of the repository);
3. the platform data directory (`%LOCALAPPDATA%\studioforge`, `~/.local/share/studioforge`) for an
   installed wheel.

The `.bat` launchers, the `justfile`/`Makefile` and the CLI all follow that rule, so a
double-click and a typed command reach the same install.

**To point this checkout at a data directory that already exists** — an older install, or a second
drive — create `local-env.bat` next to the launchers:

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

`config.example.yaml` is the generated default config with every important key annotated. The app
does not need it — it writes its own `config.yaml` into the data dir on first run — it is there to
read.

## Windows: double-click launchers

Five `.bat` files at the repo root, none of which need a terminal or admin rights:

| File | What it does |
| --- | --- |
| **Start StudioForge.bat** | Starts the server and opens the control panel once it is actually up |
| **Open StudioForge GUI.bat** | Opens the control panel of an already-running server |
| **Update StudioForge.bat** | Pulls code (if a git remote exists), syncs dependencies, updates the llama.cpp engine, then verifies |
| **StudioForge Autostart.bat** | Turns "start when I log in" on or off |
| **StudioForge Tray.bat** | Puts StudioForge in the notification area: start/stop the server, free VRAM, copy the MCP URL |

The same things from a terminal, on any platform:

```bash
studioforge serve --open          # start, and open the GUI when it is ready
studioforge gui                   # open the GUI of a running server
studioforge scan                  # inventory the model library without starting a server
studioforge config                # show the effective config, secrets redacted
studioforge engine --check        # is there a newer llama.cpp release?
studioforge engine --update       # install it, smoke-test it, then pin it
studioforge autostart enable      # start at login (Startup folder / systemd --user)
studioforge autostart disable
```

`engine --update` only repins after the new build passes its smoke test, so a broken release can
never become the default. Running instances keep the engine they started with; reload a model to
move it onto the new one.

---

## Using it from OpenClaw (or any OpenAI client)

### 1. Point inference at the gateway

```bash
OPENAI_BASE_URL=http://<studioforge-host>:1234/v1
OPENAI_API_KEY=<server.api_key, or any non-empty string when auth is disabled>
```

`GET /v1/models` lists every **downloaded** model (loaded or not), exactly like LM Studio, and
naming an unloaded model in a chat request just-in-time loads it.

### 2. Register the companion as a local stdio MCP server

`sfctl` is a separate, dependency-light package in
[`packages/studioforge-companion`](packages/studioforge-companion) that installs on the *agent's*
machine. Build the wheel (`uv build --wheel -o dist`), copy it across, then:

```bash
uv tool install ./studioforge_companion-<version>-py3-none-any.whl
sfctl servers add rig http://<studioforge-host>:1234 --api-key <PIN or server.api_key>
```

```json
{
  "mcpServers": {
    "studioforge": {
      "command": "sfctl",
      "args": ["mcp"]
    }
  }
}
```

`sfctl mcp` speaks MCP over stdio locally and proxies to **both** the server's management-plane MCP
(14 tools) and the watchdog's recovery MCP (10) as one merged toolset — so the agent keeps working
recovery tools (`restart_server`, `kill_model`, `gpu_status`, `tail_logs`,
`reclaim_orphan_engines`) even when the main server is wedged. There is no inference tool by
design; generation goes over `/v1/chat/completions`, which streams.

### 3. Let the agent pick its own model

`list_models` over MCP returns a **catalog**: every model in the library, newest download first,
each with a table of loading options and the exact arguments to load one.

```jsonc
{
  "id": "publisher/Some-122B-MoE-GGUF/...-Q5_K_M",
  "summary": "qwen35moe | 122B-A9.5B MoE | Q5_K_M | hybrid | tools+thinking | 82.9 GB | 262144 ctx train",
  "attention_kind": "hybrid",
  "downloaded_at": "2026-08-16T09:14:02Z",
  "options": [{
    "ctx_per_slot": 65536, "fits": true, "devices": [0, 1, 2, 3],
    "max_parallel": 4, "parallel_limited_by": "knee",
    "est_gen_tps": 37.0, "est_gen_tps_full_ctx": 31.2,
    "confidence": "calibrated", "recommended": true,
    "load_args": { "model_id": "...", "ctx_size": 65536, "parallel": 4, "kv_cache_type": "f16" }
  }]
}
```

The agent takes the `recommended` row and passes its `load_args` verbatim to `load_model`. Whether
it fits in the VRAM free *right now*, which GPUs it would use, how many conversations it can serve
at once, and roughly how fast at an ordinary turn (`est_gen_tps`) and with the window nearly full
(`est_gen_tps_full_ctx`) — all already in the row, so there is nothing left to compute or guess.
The recommendation never drops below the server's default context floor to buy a second slot: a
window that cannot hold the task is a failed task. The same data is on `GET /api/catalog`.

### 4. Let the agent find and fetch new models

```
search_models(query="qwen3")        # compact repo rows — HF publishes no file sizes here
repo_details(repo_id)               # real per-quant sizes, fit, and the context each GPU set reaches
download_model(repo_id, quant)      # queued; appears in list_models when it lands
```

`repo_details` reads the model's GGUF header remotely (seconds the first time, then cached), so the
answer is the planner's own — the same arithmetic a real load runs, not a rule of thumb.

### Optional: never name a model

```yaml
models:
  default_model: <model-id>
  preload_default_model: true
```

A request that omits `model` — or names `local-model`, `default`, `auto`, `current` — is served by
the default, and `preload_default_model` loads it at startup so the first request is warm.

The full sequence an agent runs is in
[`docs/OPENCLAW.md`](docs/OPENCLAW.md#the-loop-an-agent-actually-runs), along with the reliability
guarantees it can depend on and a troubleshooting section.

---

## Download models from HuggingFace

Click **Use this model → LM Studio** on any HuggingFace GGUF page and have it open StudioForge's
quant picker instead:

```bash
studioforge protocol register --takeover-lmstudio
```

Opt-in and reversible — LM Studio's handler is backed up first and restored by `studioforge
protocol unregister`. Without the flag only the `studioforge://` scheme is claimed and LM Studio is
left alone. You can also search from the GUI's Download tab, or from the agent's machine:

```bash
sfctl download lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF --quant Q4_K_M
```

Downloads are resumable, sha256-verified against what is actually on disk, survive a restart, and
land in your existing library using LM Studio's `publisher/repo/` layout.

---

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests/unit -q      # the whole fast suite, no GPU, no network
```

`tests/unit` is the suite. It needs no GPU, no engine and no network, and it is what `just test`
and `make test` run.

`tests/contract` is different: it starts a **real** gateway with a **real** engine and loads
**real** weights onto your GPUs. It is opt-in twice over — every item is marked `contract`,
`addopts = -m 'not contract'` deselects the mark by default, and `SF_RUN_CONTRACT=1` must also be
set. Both gates exist because an agent once ran `pytest tests` and left three `llama-server`
children holding ~25 GiB after the run "finished" (DECISIONS.md D23).

```bash
SF_RUN_CONTRACT=1 SF_TEST_MODELS_DIR=/path/to/gguf/library \
  .venv/Scripts/python.exe -m pytest -m contract tests/contract -q
```

A handful of unit tests exercise the *real* GGUF parser and the *real* engine binary when this
machine happens to have them (`SF_TEST_MODELS_DIR`, or an auto-detected LM Studio library; the
engine under `SF_DATA_DIR`). They skip cleanly when it does not, so a fresh checkout runs the whole
file. See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

## Project layout

```
src/studioforge/
  api/          FastAPI app: OpenAI routes, /api/* management, admin, auth
  core/         registry, planner, supervisor, engine, downloader, gpu, updater
  gui/          NiceGUI control panel (tabs/: dashboard, models, download, server, setup)
  mcp/          management-plane MCP server (14 tools)
  watchdog/     recovery sidecar: separate process, separate port, 10 MCP tools
  tray/         Windows notification-area app
  migrations/   SQL schema, applied at startup
packages/
  studioforge-companion/   sfctl: the client half, installed on the agent's machine
tests/unit/     the suite (no GPU, no network)
tests/contract/ real engine + real weights, opt-in
docs/           catalog formulas, OpenClaw setup, limitations, comparison
deploy/         systemd units
```

## Documentation

- [`docs/SETUP.md`](docs/SETUP.md) — first run, tab by tab: the checklist, the model library, the
  GPU policy knobs, the engine, ports and credentials — plus the equivalent YAML and CLI for a
  headless box
- [`docs/CATALOG.md`](docs/CATALOG.md) — the model catalog an agent picks from: how each column is
  computed, the speed formulas, calibration, and the `list_models` → `load_model` sequence
- [`docs/OPENCLAW.md`](docs/OPENCLAW.md) — pointing OpenClaw at it, the tool list, the loop an agent
  runs (catalog → load → inference → download → recover), default models, troubleshooting
- [`docs/OPENCLAW-LONG-CONTEXT.md`](docs/OPENCLAW-LONG-CONTEXT.md) — what a long window really
  costs, with three models' whole option tables and the measured numbers behind them
- [`docs/OPENCLAW-SETUP.md`](docs/OPENCLAW-SETUP.md) — a step-by-step two-machine install with a
  verification after each step (all hostnames and addresses are placeholders)
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) — working on the code: environment, lint, types,
  which tests touch real hardware, and how to run the GUI against a scratch data dir
- [`DECISIONS.md`](DECISIONS.md) — architectural decisions, each with the measurement behind it
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) — known limitations, honestly
- [`docs/COMPARISON.md`](docs/COMPARISON.md) — what was borrowed from Ollama, oobabooga, KoboldCpp, vLLM

## License

**Not yet chosen — all rights reserved for now.** There is deliberately no `LICENSE` file: until
one is picked, no permission to copy, modify or redistribute is granted.
