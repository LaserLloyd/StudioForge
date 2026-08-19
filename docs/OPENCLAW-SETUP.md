# Setting up StudioForge for OpenClaw

A start-to-finish install guide for the **two-machine** deployment this project was built for: a
GPU rig serves the models, and the box running OpenClaw talks to it over Tailscale.

If you want the shorter, surface-level version of this page, see [OPENCLAW.md](OPENCLAW.md). This
page is the walk-through, with a verification after every step.

> **Every hostname, address and path on this page is a placeholder.** Substitute your own:
> `<rig>`/`<rig-ip>` is the GPU machine, `<openclaw-host>`/`<openclaw-ip>` is the machine running
> the agent, and `<models-dir>` is wherever your GGUF library lives. Nothing here depends on the
> particular values — a plain LAN address or a DNS name works exactly as well as a tailnet one.

---

## The two machines

| Role | Host | Tailscale IP | OS |
| --- | --- | --- | --- |
| GPU rig — runs StudioForge and the models | `<rig>` | `<rig-ip>` | Windows 11 |
| OpenClaw box — runs the agent and `sfctl` | `<openclaw-host>` | `<openclaw-ip>` | Linux |

Ports on the rig, all bound to `0.0.0.0` so they answer on the tailnet:

| Port | Surface | Who talks to it |
| --- | --- | --- |
| `1234` | OpenAI-compatible API + `/mcp` | OpenClaw (inference), `sfctl` |
| `8080` | Web control panel | your browser |
| `1235` | Watchdog recovery sidecar | `sfctl recover`, the MCP proxy |

Models live at `<models-dir>` on the rig, **in place** — the same folder LM Studio uses.
StudioForge never moves or copies your files.

> **Why port 1234?** It is the port LM Studio serves on. Migrating an existing OpenClaw config is
> a *host* change, not a host-and-port change. LM Studio and StudioForge cannot both hold it at
> once — quit LM Studio, or set `server.port` on the rig and adjust the URLs below to match.

---

## Before you start

**On the rig**, confirm the server is up:

```bash
curl -s http://127.0.0.1:1234/health
```

You want `{"status":"ok",...,"instance":"primary"}`. If nothing answers, launch
`Start StudioForge.bat` (or `StudioForge Autostart.bat` once, to have it start with Windows).

`instance` is worth a glance: `secondary` means another process already holds the data directory,
and this one is serving reads while doing no background work at all — no download resume, no TTL
sweeper, no auto-load. `instance_holder_pid` names the one that owns it.

**Get the MCP pairing PIN.** You need it in Step 4. It is printed in the startup banner, and it is
also served:

```bash
curl -s http://127.0.0.1:1234/api/mcp/info
```

The `pin` field is an 8-digit code. It is regenerated only if you delete it from `config.yaml`, so
it is stable across restarts — but if you ever rotate it, redo Step 4 with the new value.

---

## Step 1 — Prove the rig is reachable from the OpenClaw box

Do this first. Every later step assumes it, and it is the single most common thing to be wrong.

**On the OpenClaw box:**

```bash
curl -s -m 8 http://<rig-ip>:1234/health
```

Expected: `{"status":"ok","version":"0.2.0",...}`.

If it hangs or refuses:

- Check the tailnet is up on both ends: `tailscale status` should list `<rig>` as online.
- Windows Firewall must allow inbound on the Tailscale adapter's network profile. An inbound
  allow rule for the venv's `python.exe` on the Private (and, if the tailnet adapter is classed
  that way, Public) profile is what makes this work.
- Confirm the listener is actually bound wide, on the rig:
  `Get-NetTCPConnection -LocalPort 1234 -State Listen` should show `0.0.0.0`, not `127.0.0.1`.

---

## Step 2 — Point OpenClaw's inference at the rig

This is the whole inference setup. **On the OpenClaw box:**

```bash
export OPENAI_BASE_URL=http://<rig-ip>:1234/v1
```

```bash
export OPENAI_API_KEY=not-required
```

`server.api_key` is unset by default, so any non-empty string is accepted — but most
OpenAI clients refuse to start with an empty key, hence the placeholder. (To close this off
properly, see *Turning on a real API key* below.)

Put those two lines in whatever OpenClaw reads at startup — your shell profile, a systemd unit's
`Environment=`, or OpenClaw's own env config — so they survive a reboot.

**Verify:**

```bash
curl -s http://<rig-ip>:1234/v1/models | head -c 400
```

That lists every **downloaded** model, not just loaded ones. Naming an unloaded model in a request
just-in-time loads it; there is no separate "load" step for ordinary inference.

---

## Step 3 — Install the companion CLI on the OpenClaw box

`sfctl` gives OpenClaw model management as agent tools, and gives you a remote control for the rig.
It is **not published to PyPI**, so install the wheel that ships in this repo.

**On the rig**, the wheel is at:

```
packages\studioforge-companion\dist\studioforge_companion-<version>-py3-none-any.whl
```

Rebuild it any time with `uv build --wheel -o dist` from `packages/studioforge-companion`.

**Send it across.** Taildrop is easiest, since it needs no SSH:

```bash
tailscale file cp "<repo>/packages/studioforge-companion/dist/studioforge_companion-<version>-py3-none-any.whl" <openclaw-host>:
```

**On the OpenClaw box**, collect and install it. `uv tool install` puts `sfctl` on your PATH in userspace,
which is what you want on an immutable OS — nothing is layered into the base image:

```bash
tailscale file get ~/
```

```bash
uv tool install ~/studioforge_companion-<version>-py3-none-any.whl
```

**Verify:**

```bash
sfctl --help
```

If `sfctl` is not found, `uv tool install` put it in `~/.local/bin` — make sure that is on your
PATH (`uv tool update-shell` does this for you).

---

## Step 4 — Register the rig as a server profile

**The PIN is required here.** StudioForge's `/mcp` endpoint enforces the pairing PIN even when
`server.api_key` is unset, so the control plane is never the least-protected surface. Register the
profile *without* a key and inference will work fine while every MCP tool call returns 401.

**On the OpenClaw box**, using the PIN from *Before you start*:

```bash
sfctl servers add rig http://<rig-ip>:1234 --api-key <PIN> --use
```

Notes on that command:

- The URL is a **positional argument**. There is no `--url` option on `servers add`.
- `--use` makes it the default profile, so later commands need no `-s rig`.
- The watchdog URL is derived automatically as the same host on port `1235`. Override with
  `--watchdog-url` only if you changed `watchdog.port` on the rig.
- The PIN travels as a bearer token. The rig accepts it on `/mcp`; on `/v1` it is ignored, because
  no API key is configured there yet. When you later set a real `server.api_key`, put *that* here
  instead — it is valid on every surface, PIN included.

The profile is written to `~/.config/studioforge/companion.toml` (or `$XDG_CONFIG_HOME/studioforge/`),
and looks like this:

```toml
default = "rig"

[servers.rig]
url = "http://<rig-ip>:1234"
api_key = "<PIN>"
```

**Verify** — this reaches the rig, not localhost:

```bash
sfctl status
```

You should see every GPU the rig has — the reference rig reports 2× RTX 5090 `sm_120` and
2× RTX 3090 `sm_86` — with VRAM per card, loaded models and uptime.

---

## Step 5 — Register the MCP server with OpenClaw

Add this to OpenClaw's MCP configuration:

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

`sfctl mcp` is a **stdio** server — OpenClaw launches it as a child process, and it reads the
profile you created in Step 4 to find the rig. No ports, no extra credentials in the JSON.

It merges two upstream tool sets into one list of **26 tools**:

| Source | Tools | Works when the main server is wedged? |
| --- | --- | --- |
| Main app (16) | `list_models`, `model_options`, `model_info`, `load_model`, `load_recommended`, `unload_model`, `test_model`, `benchmark_parallel`, `search_models`, `repo_details`, `download_model`, `delete_model`, `server_status`, `connection_info`, `get_config`, `set_config` | no |
| Watchdog (10) | `restart_server`, `kill_model`, `nuke_all_models`, `reclaim_orphan_engines`, `tail_logs`, `gpu_status`, `rollback_update`, `recovery_health`, `recovery_get_config`, `recovery_set_config` | **yes** |

That split is the entire point of the separate sidecar: when the main server locks up, OpenClaw
still holds working tools to diagnose and restart it. Colliding names get a `recovery_` prefix so
you always know which plane you are talking to. Calling a main-app tool while the server is down
returns an error *result* naming `restart_server` — not a protocol error that would kill the
session.

There is no inference tool, deliberately: generation goes over `POST /v1/chat/completions`, which
streams. The three sequences the agent actually runs are in
[OPENCLAW.md](OPENCLAW.md#the-loop-an-agent-actually-runs) — choosing and loading a model
(`list_models` → `load_model`), getting a new one (`search_models` → `repo_details` →
`download_model`), and getting VRAM back (`server_status` → `reclaim_orphan_engines`).

`connection_info` is what to call when the network moves: it hands back the current LAN and
Tailscale addresses for direct connection.

---

## Step 6 — End-to-end check

**On the OpenClaw box**, confirm inference works through the exact URL OpenClaw will use:

```bash
curl -s http://<rig-ip>:1234/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF","messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":16}'
```

The first call on a cold model includes load time — that is the JIT load, and it is expected.

Then confirm the tools plane:

```bash
sfctl recover
```

With no flags this prints the watchdog's own health diagnosis. It goes to the **watchdog** on 1235, deliberately bypassing the main server, so a success here
proves the recovery path OpenClaw depends on is live.

Finally, restart OpenClaw and confirm it lists the 24 `studioforge` tools.

---

## Optional — a default model, so OpenClaw never has to name one

By default a request must name a model. To let OpenClaw just talk to the server, set a default on
the rig in `config.yaml`:

```yaml
models:
  default_model: lmstudio-community/gemma-4-31B-it-QAT-GGUF
  preload_default_model: true
```

With this set, requests that omit `model` — or that send `local-model`, `default`, `auto` or
`current`, the aliases other tools like to send — resolve to that model. `preload_default_model`
loads it at server startup, so the first agent request of the day does not pay the load cost.

---

## Recommended — turning on a real API key

Right now inference is open to anything on your tailnet. The PIN protects the MCP control plane
only; `POST /v1/chat/completions` needs no credential. On a private tailnet that may be an
acceptable trade, but closing it is two steps.

**On the rig**, set the key in `config.yaml` and restart:

```yaml
server:
  api_key: <a long random string>
```

**On the OpenClaw box**, update both sides to use it:

```bash
sfctl servers add rig http://<rig-ip>:1234 --api-key <the same string> --use
```

```bash
export OPENAI_API_KEY=<the same string>
```

The API key is valid everywhere — `/v1`, `/api`, `/mcp` and the watchdog — so it replaces the PIN
rather than sitting alongside it. Two behaviours worth knowing:

- `GET /health` stays open with no credential, so watchdogs and probes keep working.
- `GET /health?deep=true` does **not** — it runs a real completion against every loaded model, so
  it requires the key, and it fails closed on a malformed `deep` value.

---

## Reference

**Where things live**

| What | Path |
| --- | --- |
| Companion profile (OpenClaw box) | `~/.config/studioforge/companion.toml` |
| Override that path | `SF_COMPANION_CONFIG=/path/to/companion.toml` |
| Server config (rig) | `<data dir>\config.yaml`; the data dir is `<repo>\data` unless `SF_DATA_DIR` says otherwise |
| Override the data dir | `SF_DATA_DIR`, or a gitignored `local-env.bat` next to the launchers |
| Models (rig) | `<models-dir>` |
| Logs (rig) | `<data dir>\logs\`, per-model logs under `logs\models\` |

**`sfctl` global flags**

| Flag | Meaning |
| --- | --- |
| `-s, --server <name>` | Use a named profile |
| `--url <url>` | Bypass profiles entirely |
| `--api-key <key>` | Override the key (also `SF_API_KEY`; never echoed) |
| `--json` | Machine-readable output |
| `--no-color` | Disable styling |

**`sfctl` exit codes**, for scripting:

| Code | Meaning |
| --- | --- |
| `0` | success |
| `1` | API error — the server refused the request |
| `2` | usage error — bad arguments or bad local config |
| `3` | confirmation required — destructive command, no `--yes`, no tty |
| `4` | server unreachable — refused, timed out, or DNS |
| `5` | auth failed — missing or wrong API key |

**Handy commands**

```bash
sfctl openclaw-setup
```

prints the inference and MCP snippets pre-filled with the rig's current addresses — useful after a
network change, since it reads the live Tailscale and LAN addresses rather than guessing.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `curl` to `<rig-ip>:1234` hangs or refuses | tailnet down, or firewall | `tailscale status` on both ends; check the rig's inbound rules for `python.exe` |
| Inference works, every MCP tool returns 401 | profile registered without the PIN | redo Step 4 with `--api-key <PIN>` |
| `sfctl: command not found` | `~/.local/bin` not on PATH | `uv tool update-shell`, then reopen the shell |
| `no default server set and several are defined` | more than one profile, none default | `sfctl servers use rig` |
| Server exits at startup, port in use | LM Studio owns 1234 | quit LM Studio, or change `server.port` and update the URLs here |
| First request to a model is slow | JIT load of the weights | expected; use `preload_default_model`, or pin the model so TTL never evicts it |
| Model refuses to load, "does not fit" | GPU-only policy — no CPU offload, ever | pick a smaller quant, lower the context, or free VRAM with `sfctl models unload <id>` |
| VRAM is used but nothing of yours is loaded | leaked `llama-server` processes from an earlier run, or somebody else's live one | `server_status` names every holder; `vram_orphan_count > 0` → the watchdog's `reclaim_orphan_engines` (or the Dashboard's Reclaim button). A holder classed `child_of_live_process` is never killed — it belongs to something still running |
| Engine update offers a version that will not install | a llama.cpp prerelease tagged `vX.Y.Z` ships no `llama-server` build | nothing to do — `engine --check` filters non-`bNNNN` tags out now, and `engine --list` says so |
| Main server unresponsive | — | `sfctl recover` to diagnose, then `sfctl recover --restart`; the watchdog answers independently |

---

## Day to day

- **Update the engine or the app** on the rig with `Update StudioForge.bat`, or `sfctl update`
  remotely. App updates roll back automatically if the new version fails to come up; `rollback_update`
  is also exposed as an MCP tool.
- **After updating the rig**, rebuild and reinstall the companion wheel on the OpenClaw box if the companion
  changed, so both ends speak the same tool list.
- **Restarting** from the OpenClaw box: `sfctl recover --restart`, or the "Restart server" button in the
  web panel at `http://<rig-ip>:8080`.
