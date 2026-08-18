# Development

Working notes for changing the code. For the rules of the road — lint, types, the test policy, the
`DECISIONS.md` convention — see [`../CONTRIBUTING.md`](../CONTRIBUTING.md).

## Running it against a scratch data directory

The safest way to run a development build is to give it its own data dir, its own ports, and an
empty model library, so it cannot touch an install you rely on:

```bash
export SF_DATA_DIR=/tmp/sf-dev            # Windows: set "SF_DATA_DIR=%TEMP%\sf-dev"
.venv/Scripts/python.exe -m studioforge serve --no-watchdog --port 1299
```

`just run-dev` is exactly that, minus the data dir. A scratch instance still needs an engine: copy
(or junction) `engines/<tag>/` from an existing data dir rather than downloading a second 670 MB
copy, and copy `engines/active.json` alongside it.

Points worth knowing:

- **Redirect stdout and stderr to files, never close them.** The startup banner writes through a
  helper that tolerates a dead stream, but a plain `print` on a detached console raises
  `ValueError: I/O operation on closed file` and takes the whole process with it *after* startup
  has otherwise succeeded — the last log line is "management MCP mounted" and there is no
  traceback (DECISIONS.md D21).
- **Children die with the parent.** On Windows every `llama-server` child is assigned to a job
  object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, so killing the process tree reclaims the VRAM
  even on a hard kill (D23). After stopping a dev instance, confirm with
  `Get-Process llama-server -ErrorAction SilentlyContinue`.
- **One data dir, one primary.** A second server on the same data dir starts as a *secondary*: it
  serves reads and runs no downloader, no TTL sweeper and no auto-load (D24). `GET /health` reports
  `"instance"` and, for a secondary, `instance_holder_pid`.

## Ports used while developing

| Port | What |
| --- | --- |
| `1234` / `8080` / `1235` | the defaults — i.e. whatever install is already running |
| `1299` | `just run-dev`'s gateway |
| `18100–18200` | `llama-server` children of the default config |
| `19420–19460` | `test_supervisor.py`'s fake children, deliberately out of the production range |

Pick something else entirely for a scratch instance (`server.port`, `gui.port`, `watchdog.port` are
all validated against each other and against the child range at load time).

## The shape of the code

```
api/         FastAPI app and routes
  app.py           create_app, lifespan, state assembly
  openai_routes.py /v1/*  — the parity surface
  mgmt_routes.py   /api/* — catalog, models, downloads, status
  admin_routes.py  restart, update, connection info
  auth.py          one dependency; API key or MCP pin
core/
  registry.py      scans model dirs, owns the sqlite records
  gguf.py          GGUF header parser — geometry, quant, capabilities
  planner.py       VRAM arithmetic: what fits, where, at what context
  supervisor.py    llama-server children: launch args, health, kill
  engine.py        versioned engine artifacts, download, smoke test
  downloader.py    resumable HF downloads, one writer per .part
  gpu.py           NVML probe
gui/           NiceGUI panel; tabs/ is one module per tab
mcp/           management-plane MCP server
watchdog/      recovery sidecar: its own process, port and MCP server
```

The dependency direction is one-way: `api` and `gui` and `mcp` all depend on `core`, and `core`
depends on nothing above it. A change that makes `core` import from `api` is a change to reject.

## Adding a config key

1. Add the field to the right model in `config.py`, with a default and a comment saying *why* that
   default.
2. If it cannot take effect without a restart, add it to `RESTART_REQUIRED_KEYS`.
3. Regenerate `config.example.yaml` (it is produced by `load_config(create=True)` against a temp
   data dir, with `data_dir`, `models.dir` and `mcp.pin` blanked) and annotate the key in the
   header if it is one a new user has to think about.
4. **The GUI needs no change.** The Setup tab's Advanced section is generated from the pydantic
   model, so a new scalar key gets a type-aware widget automatically (DECISIONS.md D26). Two
   optional follow-ups: add a one-liner to `gui/state.CONFIG_FIELD_HELP` if the generated
   "`<kind>`, default `<x>`" is not explanation enough, and promote it out of Advanced into a named
   section by adding a `fields.row(...)` in `gui/tabs/setup.py` **and** listing it in that module's
   `COVERED_KEYS` — a test asserts every key has exactly one control.
5. A key whose type is a mapping (`dict[int, int]`) is deliberately *not* generated: it needs a row
   widget, like `planner.reserved_mb` has on the GPU card. Add it to
   `gui/state.CUSTOM_WIDGET_KEYS` along with the widget, or the same test fails.

## Building the companion wheel

```bash
cd packages/studioforge-companion
uv build --wheel -o dist
```

It deliberately does not depend on `studioforge`: it is a pure HTTP client, installable on a laptop
or an immutable OS with `uv tool install`. It targets Python 3.11 rather than 3.12 for the same
reason — the machine running the agent frequently lags the rig.
