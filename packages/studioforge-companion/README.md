# studioforge-companion (`sfctl`)

Remote control for a StudioForge server:
a GPU-only, OpenAI-compatible LLM gateway over `llama.cpp`'s `llama-server`.

This package is a **thin client**. Every piece of state — the model registry, the VRAM
planner, the engine, the configuration — lives on the server and is reached over HTTP, so
`sfctl` installs anywhere (an agent host, a laptop, a NAS) without the server's CUDA-adjacent
dependency tree. It does **not** depend on the `studioforge` package, and it needs Python 3.11+
rather than the server's 3.12+.

## Install

```bash
uv tool install studioforge-companion     # recommended: isolated, on PATH
# or
pipx install studioforge-companion
# or, into the current environment
pip install studioforge-companion
```

Then point it at a rig:

```bash
sfctl servers add rig http://100.64.0.3:1234 --api-key sf-your-key --use
sfctl status
```

The profile is stored in `~/.config/studioforge/companion.toml`
(`%APPDATA%\studioforge\companion.toml` on Windows), written `0600` on POSIX because it holds
an API key. The key is never printed: everything that renders it redacts it first.

## Multiple servers

```toml
default = "rig"

[servers.rig]
url = "http://100.64.0.3:1234"
api_key = "sf-..."
watchdog_url = "http://100.64.0.3:1235"   # optional; derived from url + port 1235

[servers.laptop]
url = "http://192.168.1.50:1234"
```

```bash
sfctl -s laptop models list          # one-off against another profile
sfctl --url http://box:1234 status   # no profile at all
sfctl servers use laptop             # change the default
```

`SF_API_KEY` is honoured as an environment variable, and `--url` bypasses the config file
entirely — handy in CI.

## Commands

| Command | What it does |
| --- | --- |
| `sfctl status` | VRAM per GPU, loaded models with ctx/port/TTL countdown, queue depth, engine, uptime |
| `sfctl models list [--loaded] [--vision] [--kind K]` | Registry listing with capability badges |
| `sfctl models info <model>` | Details, plus **requested** settings beside the **actual** running values from `llama-server` |
| `sfctl models load <model> [--ctx N] [--kv-type T] [--parallel N] [--force]` | Shows the fit plan, then loads |
| `sfctl models unload <model>` | Free the VRAM |
| `sfctl models pin <model> [--off]` | Exempt from the idle TTL |
| `sfctl models test <model> [--prompt TEXT]` | Latency, tok/s and the reply |
| `sfctl models plan <model> [--ctx N]` | Fit verdict, per-GPU projection, suggestions — loads nothing |
| `sfctl models settings <model> [--set k=v ...]` | Per-model settings; server-side validation reported verbatim |
| `sfctl models delete <model> [--files] [--yes]` | Destructive: needs confirmation |
| `sfctl models scan [--force]` | Rescan the model directories |
| `sfctl download <hf-repo> [--quant Q] [--no-mmproj]` | Download with a live progress bar (MB/s + ETA) |
| `sfctl logs [model] [--follow] [-n N] [--level L]` | Server ring buffer or one model's `llama-server` output |
| `sfctl config get [key]` / `sfctl config set k=v ...` | **Server-side** config; secrets redacted |
| `sfctl config-local set k=v` / `sfctl config-local path` | The **local** companion config |
| `sfctl update [--engine TAG] [--check]` | Engine install / self-update; `--check` only reports |
| `sfctl chat <model> [--system TEXT] [--temp F]` | Streaming terminal chat with a running tok/s |
| `sfctl recover [--restart] [--kill MODEL] [--nuke]` | Talks to the **watchdog**, so it works when the server is wedged |
| `sfctl openclaw-setup [--reveal-key]` | The snippets to paste into OpenClaw |
| `sfctl mcp` | The stdio MCP server (see below) |
| `sfctl servers list\|add\|remove\|use` | Local profiles |

Add `--json` to any command for machine-readable output. `--no-color` drops the styling.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | success |
| `1` | API error — the server answered and refused (unknown model, bad flag, won't fit in VRAM) |
| `2` | usage error — bad arguments or bad local config |
| `3` | confirmation required — a destructive command had no `--yes` and stdin is not a terminal |
| `4` | server unreachable — refused, timed out, DNS |
| `5` | auth failed — missing or wrong API key |

Code `3` exists so scripts can distinguish "I needed you to confirm" from "the server said no".
An unreachable server prints one short line naming the URL and pointing at `sfctl recover` —
never a traceback.

## OpenClaw / MCP

Register the stdio bridge once:

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

Add `"args": ["-s", "rig", "mcp"]` to pin a profile.

`sfctl mcp` merges **two** upstream MCP servers into one list of **29 tools**.

**Management** (the main app, `<url>/mcp`) — 19 tools:

| Tool | What it does |
| --- | --- |
| `list_models` | **Start here.** The catalog: every model newest-download-first, each with a table of loading options and a `load_args` object |
| `model_options` | Every context tier for one model, when the recommended row is not what you need |
| `model_info` | One model in detail, including what the running engine *actually* reports |
| `load_model` | Load now; pass a catalog row's `load_args` verbatim |
| `load_recommended` | Name the model and the context; the server picks the GPUs, KV cache and slots and loads at exactly that window, or refuses with numbers |
| `unload_model` | Free its VRAM immediately |
| `pin_model` | Keep a model loaded at all times (no idle TTL, never evicted, auto-loaded and reloaded); `pinned=false` unpins |
| `reserve_gpus` | Give specific GPUs to one model (loaded there at its measured-fastest settings, auto-sized slots) or hold them for an outside program; auto-releases after `idle_ttl_s` |
| `release_gpus` | End a reservation early by its lease id |
| `test_model` | Smoke-test end to end on an idle server and report tokens/second |
| `benchmark_parallel` | Measure how many concurrent slots a model is worth running on a set of cards |
| `search_models` | Browse HuggingFace. One thin row per repo — **no sizes**, HF publishes none at search time |
| `repo_details` | One repo in full: per-quant sizes, fit verdicts, and the context each GPU placement reaches |
| `download_model` | Queue a download; it runs in the background and appears in `list_models` when it lands |
| `delete_model` | Destructive; requires `confirm=true` |
| `server_status` | VRAM per GPU, **who is holding it**, loaded models, queue, engine |
| `connection_info` | Every address this server answers on, best first |
| `get_config` / `set_config` | The server's configuration, secrets redacted |

**Recovery** (the watchdog, `<watchdog_url>/mcp`, default port 1235) — 10 tools:
`restart_server`, `kill_model`, `nuke_all_models`, `reclaim_orphan_engines`, `tail_logs`,
`gpu_status`, `rollback_update`, plus `recovery_health`, `recovery_get_config`,
`recovery_set_config`.

Naming rule: management tools keep their bare names; watchdog tools whose names would collide
get a `recovery_` prefix (so `get_config` is the server's config and `recovery_get_config` is the
watchdog's). Watchdog-only tools keep their own names.

Two flows worth knowing:

* **Getting a model:** `search_models` → `repo_details(repo_id)` → `download_model(repo_id, quant)`.
  Search knows nothing about size or fit; `repo_details` reads the model's GGUF header remotely
  (seconds, then cached) and answers exactly.
* **Getting VRAM back:** `server_status` classifies every `llama-server` on the box. A
  `vram_orphan_count` above zero means leaked engine processes with nothing waiting on them, and
  `reclaim_orphan_engines` kills exactly those — never somebody's live child.

The two upstreams are connected to lazily and independently, which is the entire point of the
split: **when the main server is down the watchdog's tools are still listed and still work**, so
the agent can call `restart_server` and fix it. Management tools stay visible in that state and
return a clear "main server unreachable — try `restart_server`" tool-result error instead of
killing the session.

Also point the agent's inference at the same box:

```
OPENAI_BASE_URL=http://100.64.0.3:1234/v1
OPENAI_API_KEY=sf-your-key
```

`sfctl openclaw-setup` prints both snippets, with the key redacted unless you pass
`--reveal-key`.
