# First run

Everything StudioForge needs to be told is on the control panel's **Setup** tab. This page walks
through it once, and then gives the equivalent YAML and CLI for a box with no browser.

You should not have to edit `config.yaml` by hand. If you find yourself doing it, that is a gap
worth reporting — the Setup tab is generated from the configuration model itself, so every key it
has is reachable from a form (DECISIONS.md D26).

---

## 1. Start it

```bash
git clone <repo> studioforge && cd studioforge
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"     # Linux: .venv/bin/python
studioforge serve --open
```

On Windows, **launchers\Start StudioForge.bat** does the same thing without a terminal.

The panel opens on `http://localhost:8080`. On a fresh install it lands on **Setup** rather than the
Dashboard, because a box with no engine and no model library has nothing to put on a dashboard.

---

## 2. Work down the checklist

The card at the top is computed live every time you open the tab or press **Re-check**. Each line
is either green (done), amber (**required** — the server cannot load a model until it passes) or
grey (**optional** — it changes what you can do, not whether the thing works). Unmet lines carry
the one button that fixes them.

| Check | What it means | Fix |
| --- | --- | --- |
| **Data directory** | `config.yaml`, `registry.sqlite3`, `engines/` and `logs/` all live here, and it must be writable. Proved by writing a probe file, not by asking the OS. | Point `SF_DATA_DIR` somewhere writable — see [Where things live](#7-startup--service) |
| **Model library** | `models.dir` exists and has at least one `.gguf` in it | **Detect LM Studio library**, or type a path |
| **Models indexed** | The registry has scanned that library. It is what `/v1/models`, the catalog and the planner read | **Rescan now** |
| **GPUs** | NVML sees at least one card. StudioForge is GPU-only: no GPU, no inference | Check the driver, then **Re-probe** |
| **llama.cpp engine** | A versioned engine is installed under `engines/<tag>/` and active | **Install engine b10425** |
| **Gateway port** | Something is listening on `server.port`, and that something is us rather than LM Studio or a second copy of StudioForge | Quit the other one, or change the port |
| **MCP pairing PIN** | Required while `mcp.pin_required` is on — and, with no `server.api_key`, for every MCP caller that is not on this machine whatever the toggle says (D32) | **Generate PIN** |
| *HuggingFace token* | Optional. Only gated or private repositories need one | Paste one into **Downloads & HuggingFace** |
| *Start at login* | Optional | **Enable** in **Startup & service** |

When every required line is green the card's headline reads **Ready to serve**.

---

## 3. Model library

`models.dir` is the root of your GGUF library. It is scanned **in place** — nothing is copied,
moved or renamed, so an existing LM Studio library keeps working in LM Studio.

**Detect LM Studio library** reuses the same probe order the app uses on first run:

1. the `downloadsFolder` recorded in `~/.lmstudio/settings.json` — this is where a relocated
   library actually lives, and it is the common case;
2. `~/.lmstudio/models`;
3. `~/.cache/lm-studio/models`;
4. `%USERPROFILE%\.lmstudio\models` on Windows.

**Show what was probed** lists all of them with what was found at each, so "detect found nothing"
can be told apart from "the drive that library was on is not mounted".

The line under the field validates as you go: whether the directory exists, how many `.gguf` files
are under it (counted up to 2000, then reported as a floor), and how much room is left on that
volume.

### The context ladder

Three fields, and they are the ones most worth understanding (DECISIONS.md D14):

- **`models.target_ctx`** — what every load *aims* for. Default `1048576`.
- **`models.default_ctx`** — the **floor**. The planner halves down from the target to the largest
  window that fits in VRAM and never goes below this. Default `8192`, raised on first run if your
  smallest card is big enough.
- **`models.thinking_default_ctx`** — the floor for reasoning models, which spend their budget
  thinking before they answer.

An explicit context — per model, or in the request — is always honoured verbatim.

**`planner.preference`** decides what the catalog's one *recommended* load per model optimises
for (D36): `quality` (the default) keeps the best KV cache type that reaches the floor and then
takes the largest window at that quality; `throughput` takes the largest window at or above the
floor, and a second slot if it fits, even with a quantized cache. [`CATALOG.md`](CATALOG.md) has
the measurements behind the default.

**`models.default_parallel`** is `auto` by default: the slot count is estimated per model and per
placement (D17). `llama-server`'s `--ctx-size` is the *total* budget shared by the slots, so
StudioForge multiplies the per-conversation window by the slot count when it launches the child.

### Never naming a model

Set **Default model** (and optionally **Preload that default at startup**) and any request that
omits `model` — or names `local-model`, `default`, `auto`, `current` — is served by it. The field
becomes a dropdown of the models actually in your registry once a scan has run.

---

## 4. GPUs & memory

The table is live: index, name, free/total VRAM, compute capability, and — where the platform can
tell us — the processes currently holding memory on that card.

Devices are referred to **by CUDA ordinal** (CUDA0, CUDA1, …) everywhere in StudioForge: in
`excluded_devices`, in `reserved_mb`, in every per-model device override and in every plan. Those
ordinals come from the driver and are **not stable across a hardware change** — adding, removing or
re-slotting a card can renumber the rest, and `CUDA_VISIBLE_DEVICES` renumbers them for one process.
After any of those, re-probe and re-read the exclusions: they are indices, not names.

Two per-GPU controls implement DECISIONS.md D19, for a box that shares its GPUs with something else:

- **never place models here** (`planner.excluded_devices`) — the planner will not use that card at
  all. Use it to hand a whole GPU to ComfyUI or a training job.
- **reserve (MiB)** (`planner.reserved_mb`) — hold back that much memory on that card and use the
  rest. The softer half of the same knob: *"leave ComfyUI 8 GB on CUDA3"* rather than *"never touch
  CUDA3"*.

Both default to empty, because reserving hardware is a decision about one specific machine and a
shipped default must not make it. A per-model `device_override` naming an excluded device still
wins, with a warning — the user naming a device is the user deciding. A reservation applies even to
a forced placement, because it describes the *neighbour's* memory.

`planner.headroom_fraction` is different again: it is a percentage held back from **every** card,
which is exactly why the two knobs above had to exist.

**`planner.rebalance`** (`auto` | `suggest` | `off`, default `auto`) is the D42 housekeeping pass:
once a minute, on a quiet box, a model idle for five minutes or more whose exact current settings
would now fit on fewer cards — or off a card it shares with another model — is reloaded there. It
evicts nothing, never touches a model that is serving, a persisted `device_override`, or a model
holding a GPU lease (D43), and moves each model at most once per half hour. `suggest` only logs
what it would do.

---

## 5. Engine

Engines are versioned artifacts under `engines/<tag>/`, never "whatever `llama-server` is on PATH"
(D3). The panel shows the active tag, every installed tag, and offers **smoke test** and
**activate + reload** per tag.

**`engine.cuda_variant`** is `auto`, which picks the highest CUDA build this driver can actually
run. The rule runs one way only: a build compiled against CUDA *X.Y* needs a driver advertising
*X.Y or newer* — CUDA compatibility runs forward, not back. Blackwell (sm_120) cards therefore need
a 13.x build; the 12.4 archives carry no sm_120 kernels at all (D2/D3). Pin it explicitly only when
auto-detection picks wrong; the panel prints your driver's CUDA level next to the field so you can
see what it had to work with.

**Check for update** asks GitHub for the newest release that has an asset *this* box can install —
verified by reading the asset list, not by taking whatever tag sorts first — and never calls a
downgrade an update. Installing a new engine does not disturb running models: each `llama-server`
child keeps the build it was launched with until it is reloaded.

**Linux + NVIDIA always builds from source.** Upstream publishes no Linux CUDA archive at any tag
(the `ubuntu` assets are cpu, vulkan, rocm, sycl and openvino), so on Linux every install path —
first run, **Install engine**, `engine --update`, the MCP tool — compiles the pinned tag with
`allow_source_build` (default on) and needs `git`, `cmake` and a CUDA toolkit whose `nvcc` matches
the driver. The build takes minutes, lands in `engines/<tag>-local/`, is reused on the next
install of the same tag, and is smoke-tested like a download. With `allow_source_build: false` the
refusal names those prerequisites. The ROCm tarball serves AMD cards the same way a Windows zip
does. `keep_versions` is how many old engine directories survive a prune.

---

## 6. Network & access

| Field | Notes |
| --- | --- |
| `server.host` | A **bind** address. `0.0.0.0` listens on every interface, LAN and tailnet included; `127.0.0.1` is this machine only, which also means no agent on another box can reach it |
| `server.port` | `1234` is LM Studio's default, which is the whole point — pointing an OpenAI client here is a host change, not a rewrite. They cannot share the port, though: quit LM Studio or move one of them |
| `gui.port` / `watchdog.port` | This panel, and the recovery sidecar (a separate process) |
| `server.api_key` | Guards inference **and** this panel. Blank is the LAN/tailnet-trust default: reads, inference and load/unload are open, but changing the box (config, files, engines, restarts, downloads) needs a browser on the machine itself, the same rule the API applies (D32) |
| `mcp.pin` | A short pairing code for the MCP path only — deliberately *not* the inference credential. **Generate new PIN** rotates it, which is the documented response to a leak; every already-paired agent then needs the new one |

All three secrets render masked, with a reveal button. Leaving a masked field untouched keeps the
current value: the panel never posts the placeholder back over a working credential.

**Network exposure** is a checklist item, not a footnote. Bound to `127.0.0.1` (all three
listeners: `server.host`, `gui.host`, `watchdog.host`) it is green and optional — nothing off this
box can reach the API. Any of them bound to `0.0.0.0` (or a LAN address) with no `server.api_key`
turns it **required and amber**: everyone on the LAN or tailnet can then load and unload models
and use the control panel. **Set API key** mints one and saves it; copy it into OpenClaw
(`Authorization: Bearer <key>`). The same rule decides whether `GET /api/mcp/info` and
`/api/openclaw-setup` will hand the PIN to a remote caller at all — with no key set they answer
only on the machine itself.

Even with no key, the routes that change the *box* — `PATCH /api/config`, restarts, engine and
app installs, deleting model files, queueing downloads, killing processes — are accepted only from
this machine or with the MCP PIN (`X-MCP-Pin`, or as the bearer token, which is how `sfctl` sends
it); anything else is `403 remote_admin_requires_credential` (D32). Reads, inference and
load/unload stay open, as LM Studio's do. "This machine" means the peer address *and* the browser
origin: a page from any other site or port that your browser sends to `127.0.0.1:1234` is
refused, and is not handed the PIN, even though it arrives from loopback. The MCP path follows
the same rule — with no key, a caller off the box needs the PIN for every tool call even when
`mcp.pin_required` is off (that toggle only relaxes same-machine callers), and `GET /api/mcp/info`
reports the `pin_required` the caller will actually be held to. Send the PIN as `X-MCP-Pin` or as
the bearer token; `?pin=` in the URL is still accepted for connectors that can only be configured
with a URL, but a URL ends up in proxy logs and shell history, so it is no longer advertised.

**Reachable at** lists the concrete addresses another machine can use — Tailscale first, because a
tailnet address survives a network change where a LAN address silently stops resolving.

**Point OpenClaw at this server** renders the ready-to-paste configuration, built by the same
`GET /api/openclaw-setup` endpoint `sfctl` and the MCP server use, so it is the values this instance
is really serving on. Secrets are masked on screen; the copy buttons copy the real text.

---

## 7. Startup & service

**Autostart** is a per-user mechanism, so neither half needs administrator rights: a hidden `.vbs`
shim in the Startup folder on Windows (which launches the tray, and the tray brings up the server),
a `systemd --user` unit on Linux. The system-wide units for a headless server are in [`deploy/`](../deploy).

**Where things live** is read-only, and the last row is the one people ask about — which of the
data-directory rules produced the directory in force (D25, D31):

1. `SF_DATA_DIR` if it is set;
2. the directory a `--config`/`SF_CONFIG` file lives in, when one was named (that is how the tray,
   the watchdog and autostart pass the location on);
3. `<repo>/data` when running from a source checkout;
4. the platform data directory, for an installed wheel.

`config.yaml` never carries a `data_dir` key — one left there by an older build is ignored with a
warning — so copying a config file between installs cannot silently move the install.

**Open data dir** and **Open logs** open a file manager **on the machine running StudioForge**, not
on the machine you are browsing from. **Restart server** confirms first: it is the one control that
takes the gateway down, and every client talking to it sees a connection error. Exactly one
process brings it back (D28): the tray, when the tray launched the server (the server exits and
the tray respawns it, without counting a crash); otherwise the watchdog. `GET /health` reports
`can_serve` and, when false, `cannot_serve_reason` with the next action — an engine to install, a
GPU the driver does not show, a library to point at.

The port answers **before** the slow half of startup runs (D33): `GET /health` carries
`boot: {phase, ready, elapsed_s}` — `scanning models`, `installing engine b10425: download 42%`,
`ready` — and every other request waits (bounded) for the library index, so an early client sees a
slow answer rather than an empty library. On a fresh box the first start downloads the engine in
that background phase; the checklist's engine row shows the progress instead of a second Install
button.

---

## 8. Advanced

Every remaining key, grouped by config section, generated from the pydantic model itself. A key
added to StudioForge appears here without anyone remembering to add a form row.

Three things are deliberately *not* in it, and each is a rule rather than a list:

- keys with a purpose-built control in a section above (one control per key, always);
- secrets, which need the masked widget and the "did this really change" guard;
- `planner.reserved_mb` and `planner.quant_affinity`, which are mappings — rendering a mapping as a
  text box silently destroys the entries you did not retype, so they get row widgets instead.

A **↻** marker means the change is saved immediately but only takes effect after a restart
(`RESTART_REQUIRED_KEYS`: the ports, the bind addresses, the data dir, the CORS origins and the MCP
mount path).

---

## 9. Downloading models

The **Download** tab searches HuggingFace and queues a quant; `sfctl download <repo> --quant
<Q>` does the same from the agent's machine, and an agent uses `search_models` → `repo_details` →
`download_model`. Downloads are resumable, verified (sha256 where the repository publishes one,
byte count always) against what is actually on disk, survive a restart, and land in your library
using LM Studio's `publisher/repo/` layout. Gated repositories need `hf.token`.

HuggingFace's **Use this model → LM Studio** button can open StudioForge's quant picker instead:

```bash
studioforge protocol register --takeover-lmstudio
```

Opt-in and reversible — LM Studio's handler is backed up first and restored by `studioforge
protocol unregister`. Without the flag only the `studioforge://` scheme is claimed and LM Studio is
left alone. The Server tab shows the handler's true current state.

---

## Headless: the same thing without a browser

`config.yaml` lives in the data directory. `config.example.yaml` in the repository root is the
generated default with every important key annotated — the app does not read it, it is there to
read.

```yaml
server:
  host: 0.0.0.0
  port: 1234
  api_key: null              # null = no auth (LAN/tailnet trust)
mcp:
  pin: "12345678"            # auto-generated on first run
  pin_required: true
models:
  dir: D:\LLM\Models         # your GGUF library; scanned in place
  default_ctx: 8192          # floor
  target_ctx: 1048576        # aim; the planner halves down to what fits
  thinking_default_ctx: 32768
  default_parallel: auto     # or an integer
  default_kv_cache_type: auto
  default_ttl_s: 1800        # 0 = never idle-unload
  default_model: null
engine:
  pinned_tag: b10425
  cuda_variant: auto         # or "13.3" / "12.4"
planner:
  headroom_fraction: 0.10    # held back on EVERY card
  excluded_devices: [3]      # CUDA indices the planner may never use
  reserved_mb: {3: 8192}     # MiB held back on that card for a neighbour
hf:
  token: null                # only for gated repos
logging:
  level: INFO
```

The equivalent commands:

```bash
studioforge config                  # the effective config, secrets redacted
studioforge scan                    # index the model library, no server needed
studioforge engine --check          # is there a newer llama.cpp release?
studioforge engine --update         # install it, smoke-test it, then pin it
studioforge autostart enable        # Startup folder / systemd --user
studioforge autostart disable
```

Anything the Setup tab can change, `PATCH /api/config` can change, with the same validation and the
same restart-required reporting — the panel calls that route rather than reimplementing it:

```bash
curl -X PATCH http://127.0.0.1:1234/api/config \
  -H 'content-type: application/json' \
  -d '{"planner.excluded_devices": [3], "planner.reserved_mb": {"3": 8192}}'
# -> {"updated": ["planner.excluded_devices","planner.reserved_mb"], "restart_required": []}
```

`sfctl` exposes the same surface from the agent's machine, and the management-plane MCP server
exposes it as a tool. All four are the one implementation.

---

## Where to go next

- [`OPENCLAW-SETUP.md`](OPENCLAW-SETUP.md) — a step-by-step two-machine install with a verification
  after every step
- [`CATALOG.md`](CATALOG.md) — how the catalog an agent picks from is computed
- [`DEVELOPMENT.md`](DEVELOPMENT.md) — running against a scratch data directory, and how to add a
  config key
- [`../DECISIONS.md`](../DECISIONS.md) — why any of the above is the way it is
