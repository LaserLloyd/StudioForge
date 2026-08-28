# Operator's runbook

What to do when something is wrong at 3 a.m. Every entry names the signal, the one command or
click that settles it, and the DECISIONS entry that explains why it behaves that way. Everything
here assumes the defaults: gateway `:1234`, control panel `:8080`, watchdog `:1235`, logs under
`<data dir>/logs/`.

## First: what state is it in?

```bash
curl -s http://127.0.0.1:1234/health
```

| Field | Meaning |
| --- | --- |
| connection refused | the process is not up, or is still binding: see *It will not start* |
| `boot.ready: false`, `boot.phase: ...` | the port is up and the slow half of startup is running (`scanning models`, `installing engine ...`, `starting model manager`). Wait; every other request queues behind the scan (D33) |
| `can_serve: false`, `cannot_serve_reason: ...` | up, but cannot load a model; the reason names the fix (engine, GPU/driver, `models.dir`) |
| `instance: secondary`, `instance_holder_pid` | another StudioForge owns this data directory; this one runs no downloads/TTL/auto-load (D24). Stop one of them |
| `draining: true` | a shutdown or restart is in progress |
| `restart_failed: {...}` | a restart was asked for and did not happen; the reason is in the object |
| `loaded_models: [...]` | what is resident right now |

The watchdog has its own open `/health` on `:1235` that keeps answering when the server does not:
`restart_in_progress` there means it is mid-restart.

## It will not start

* **`startup port conflict`** in the console / `launchers\Start StudioForge.bat` names the port and the
  holder ("LM Studio", "another StudioForge (pid N)"). Quit that, or change `server.port` on the
  Setup tab. Exit code 3; the tray waits up to 120 s for whoever holds the port to answer as a
  StudioForge server before saying so (D28).
* **`error: invalid configuration in .../config.yaml: <key> ...`** — one line, exit 2. The key is
  named; `config.yaml.bak` next to it is the previous good file. A YAML syntax error is reported,
  never overwritten; an *empty* file is treated as missing and regenerated with a warning (D31).
* **Nothing at all under `pythonw` / the tray** — logs are in `<data dir>/logs/studioforge.log`
  and `logs/tray-server.log`; a dead console cannot kill the server any more (D33), so if the log
  ends abruptly, look for a port conflict or a config error at the top of the next start.
* **Boot hangs at `installing engine`** on a fresh box: on Windows it is downloading ~600 MB
  from GitHub; on Linux + NVIDIA it is *compiling* the tag (upstream ships no Linux CUDA archive),
  which takes minutes and needs `git`, `cmake` and a CUDA toolkit with `nvcc` — a refusal names
  whichever is missing. `/health` shows the phase and percentage. Offline? `studioforge engine
  --list` says what it wants; the Setup tab's engine row offers Install once the boot has given
  up.

## A model will not load

* **507 `insufficient_vram`** — the body carries `required_bytes`, `per_gpu_free`,
  `max_ctx_that_fits`, `max_parallel_that_fits`, `suggestions` and `vram_holders` (who has the
  memory, ComfyUI by name). Take the first suggestion; the offered context/slot count is one the
  next load accepts (D30).
* **502 `model_load_failed`** with a stderr tail — `logs/models/<model>.log` has the child's
  whole output. `unknown argument` = a per-model `extra_flags`/setting the engine does not know
  (never retried); `CUDA error: out of memory` is retried once after evicting the LRU idle model.
* **A forced reload was refused** — the model that was serving is still serving; nothing was
  unloaded (D30).
* **`no llama-server engine is installed`** — Setup tab → Install, or `studioforge engine
  --update`. A model pinning `engine_tag` that is not installed says which model pins it.

## A load was refused and the box looks half empty

Read the refusal's `busy_models` and `retry_after_s`. A model serving a request is never evicted to
make room for another one (D36), so a box that is *busy* rather than *full* refuses a load that
would have succeeded ten seconds earlier — and says which model and how many requests it has in
flight. `GET /health` carries the same picture cheaply as
`busy: {active_requests, busy_models, loading, testing}`. Wait, or re-issue the load with
`force=true` if you know what you are interrupting. `loaded_by` on each loaded model
(`/api/status`, `server_status`) names whoever asked for it — MCP, a JIT inference request, the
GUI, the autoloader.

`test_model` refuses on the same grounds and additionally refuses a second concurrent test; it
loads cold models small (one slot, the default context) and unloads them again, so it never leaves
the rig rearranged — and a pinned model it unloaded comes back by itself, because a test's or a
benchmark's unload is housekeeping, not the deliberate unload that suppresses a pin (D41). Both
benchmarks refuse a busy rig the same way. `load_recommended` on a model
that is itself serving is a **503** with `retry_after_s` rather than a reload under its clients;
`load_model(force=true)` is the only thing that interrupts a stream.

**A 503 with `error.code: priority_hold`.** Not a failure and not a busy model: a chat- or
agent-tier load (D46) is in flight, and until it settles, loads and inference for worse-tier models
are refused so they drain off the cards instead of competing with the upload. The action is the
`Retry-After` header — the hold lifts by itself, typically in the seconds-to-minutes a load takes.
Nothing is wrong with the model that was refused.

Who is holding it: `GET /health` (or `/api/health`, or `server_status`) carries
`busy.priority_hold` as `{model_id, priority}` while a hold stands, and `null` otherwise; the same
block is repeated in the refusal's own `details`. `GET /api/status` and `sfctl status` list every
loaded model with the tier it loaded at (`priority`, the `Prio` column), which is who wins the
cards once the hold clears. `GET /api/evictions` is where to look afterwards: the ring says what
the tier-1 load displaced and why (`reason: plan`), and the recently-active victims are reloaded
automatically once it settles.

Holds became far more common with D48, because a tier now survives a restart: a model whose
`settings.priority` is 1 or 2 loads at that tier for ever after, including on the just-in-time
reload after a TTL unload. A caller sending `priority: 3` on its own requests does not escape that:
the tier it names governs its own admission, but the load it triggers runs at the better of the
requested tier and the standing one, so the reload still comes up at tier 1 or 2 and still holds
worse-tier traffic off while it runs. If a background job is being held off more than you expect,
check that model's saved `priority` (Models tab, or `sfctl models settings <model>`) — and expect
`reserve_gpus` to refuse a lease whose cards hold an idle chat- or agent-tier model, which
`force=true` overrides exactly as it does for a pin.

**A device OOMs while the plan said it fits.** The `load observation` INFO line now carries
`per_device_mb`, and a card holding more than 15% over its planned share logs *"a device holds more
than its planned share"* with both numbers (D40). llama.cpp puts the output layer on the **last**
device of the list and the planner charges it there; a persistent overrun on one model means that
charge is too small for it -- raise `planner.headroom_fraction` a little, or pin `device_override`
with the roomier card last.

## VRAM is held and nothing is loaded

`GET /api/vram/holders` (or the Dashboard's VRAM holders panel) names every process on every
GPU. Ours are marked; an `orphan` is a `llama-server` from a dead StudioForge — `POST
/api/vram/reclaim` (or the watchdog's `reclaim_orphan_engines`, or restart) kills only those. A
`child-of-live-process` belongs to something running (another instance, a test) and is left alone
(D23). On Linux, children die with a killed server (PDEATHSIG shim); on Windows the job object does
it.

**Which GPU is it actually on.** Each holder carries `per_gpu_bytes` (`{"0": 16663000000, "1":
15547000000}`) and a `gpu_indices` list of the devices holding at least 512 MiB, so a row reads
`llama-server.exe (pid 32188) · 30.44 GiB · CUDA0 15.5 GiB, CUDA1 14.5 GiB`. Check
`gpu_indices_source` before trusting it: `pdh` is a measurement, `nvml-context` means only NVML
could answer and the devices listed are the ones the process has a CUDA *context* on — llama.cpp
opens one on every visible card, so a two-GPU model looks like a four-GPU one (D39). A holder
classified `other-instance` is a `llama-server` from a different install; its `detail` gives the
`--alias`, `--port` and directory to go and stop it, and nothing here will ever kill it for you.

## The server is up but wedged

The watchdog on `:1235` answers when the server does not: `sfctl recover --restart` (or `POST
http://127.0.0.1:1235/restart` with the API key or PIN) kills the process tree and — unless a live
tray launched the server, in which case the tray does it — spawns the replacement. Exactly one
process respawns (D28). The watchdog's read-only diagnostics are on the same command and
are the right first move: `sfctl recover --gpus`, `--logs <n>` (add `--log-model <id>` for
one model's log) and `--config`. Then `--kill <model>` or `--nuke` to free VRAM without a
full restart.

The watchdog's `/mcp` and `POST /restart` demand a credential whenever **either** the API key
or the MCP PIN is set — the shipped default (PIN only, no key) included — with the same
wrong-credential lockout as the main app (D44) and `?pin=` refused. Only `GET /health` is open.
Verified 2026-08-26: an unauthenticated call to a destructive watchdog tool from off the box is
a 401, and `tests/unit/test_watchdog.py` pins it.

## Restarts and who brings it back

| You started it with | Restart from GUI/API/MCP does | Crash does |
| --- | --- | --- |
| the tray (`launchers\StudioForge Tray.bat`, autostart) | server exits 75, tray respawns, no crash counted | tray respawns (3 attempts within 60 s, then "Crashed — see the logs folder") |
| `studioforge serve` / `launchers\Start StudioForge.bat` | watchdog kills and respawns | watchdog does **not** auto-respawn (by design); relaunch |
| systemd | watchdog kills and respawns | `Restart=on-failure` in the unit |

## Downloads

* **`failed` with `gated or private`** — open the model page, accept the licence, set `hf.token`
  (Setup tab), Resume. Not retried on its own.
* **`not enough disk space on <drive>`** — refused up front with the shortfall; free space or move
  `models.dir`, then Resume. A transfer that hits ENOSPC keeps its `.part` and says the same.
* **`could not be moved into place`** — the file is complete and verified; a loaded model has the
  destination open (Windows). Unload it, Resume: it is published without re-downloading.
* **Resume after a crash** re-verifies the `.part` and continues from where it stopped; a complete
  `.part` is published without a request (D24).
* **`downloads cannot start from this process`** — this is a secondary instance; queue it on the
  primary (`instance_holder_pid` in `/health`).

## Security

* Anyone on the LAN can reach the gateway on the defaults. Set `server.api_key` on the Setup tab
  (the checklist's *Network exposure* row is red until you do or you bind to `127.0.0.1`).
* Without a key, routes that change the box are only accepted from the machine itself or with the
  MCP PIN (`403 remote_admin_requires_credential` otherwise) — D32. "The machine itself" means a
  loopback peer whose browser `Origin`, if any, is this server's own host and port: a page from
  another site or another local app is refused even from `127.0.0.1`, and so is the panel on
  `:8080` for a LAN viewer. A `403` on a loopback request is therefore a cross-site page, or a
  reverse proxy rewriting `Host` — the log line names which.
* The MCP path is under the same rule: with no key, every tool call from off the box needs the PIN
  even with `mcp.pin_required: false`; `GET /api/mcp/info` reports the `pin_required` a caller is
  really held to.
* The PIN is in the startup banner and `studioforge config` on the box; **Generate new PIN** on the
  Setup tab rotates it after a leak. Send it as `X-MCP-Pin` or the bearer token. `?pin=` in the URL
  is refused: a URL lands in proxy logs, browser history and shell history.
* Wrong credentials are rate-limited. Three free attempts per client address, then a doubling
  lockout (1s, 2s, 4s … capped at 5 minutes) that clears after 15 quiet minutes or on any
  success. A locked-out caller gets `429` with `Retry-After`. The main server and the watchdog
  count separately, so neither door can lock the other.
* Vision requests never fetch an image from loopback, link-local, private, ULA or CGNAT space
  (`100.64.0.0/10`, the tailnet) unless `gateway.allow_private_image_hosts` is on.

## Where the logs are

`<data dir>/logs/studioforge.log` (server), `logs/models/<model>.log` (each child's stdout/stderr,
the place a load failure is legible), `logs/watchdog.log`, `logs/tray-server.log` (a tray-launched
server's console). `GET /api/logs?n=500` tails the in-memory ring buffer; the watchdog's
`tail_logs` reads the files even when the server is down. Secrets (`server.api_key`, `hf.token`,
`mcp.pin`) are redacted in every log line and every config dump.
