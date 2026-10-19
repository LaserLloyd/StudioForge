# Development

Working notes for changing the code. For the rules of the road — lint, types, the test policy, the
`DECISIONS.md` convention — see [`../CONTRIBUTING.md`](../CONTRIBUTING.md). For cutting a release —
the four version strings, the tag, and the workflow that builds the zip and the wheels — see
[`RELEASING.md`](RELEASING.md).

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
- **Headless Linux and the tray tests.** `pystray` picks its backend at import and the xorg one
  opens an X display, failing with something that is *not* an `ImportError`. `test_tray.py` parks
  it on the inert backend itself (`PYSTRAY_BACKEND=dummy` off Windows, unless you set one), and
  CI's ubuntu leg sets the same variable; Windows keeps the real `_win32` backend, which is the
  only place a pywin32 regression would show.
- **Linux engines are source builds.** Upstream ships no Linux CUDA archive, so a Linux dev box
  needs `git`, `cmake` and a CUDA toolkit with `nvcc` for `engine --update`; the build is reused
  on the next install of the same tag (`engines/<tag>-local/`).

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
  mgmt_routes.py   /api/* — catalog, models, downloads, leases, status
  admin_routes.py  restart, update, connection info, /api/mcp/info
  health_routes.py /health
  auth.py          one dependency: API key, MCP pin, the D32 admin gate
netguard.py    which hosts a tenant URL may name (loopback, tailnet, allowed_hosts)
  vision.py        image fetching for multimodal requests (SSRF guard)
core/
  registry.py      scans model dirs, owns the sqlite records
  gguf.py          GGUF header parser — geometry, quant, capabilities
  planner.py       VRAM arithmetic: what fits, where, at what context
  catalog.py       the rows an agent picks from (docs/CATALOG.md)
  model_gate.py    "is what is loaded good enough?" — the routing gate (D52)
  manager.py       load/unload, pin reconciler (D41), rebalancer (D42)
  leases.py        GPU leases (D43), automatic placement (D78)
  tenants.py       the tenant registry and its health poller (D77)
  supervisor.py    llama-server children: launch args, health, kill
  engine.py        versioned engine artifacts, download, source build, smoke test
  benchmark.py / parallel_bench.py   the two benchmarks (docs/BENCHMARKING.md)
  downloader.py    resumable HF downloads, one writer per .part
  gpu.py           NVML probe
gui/           NiceGUI panel; nav.py is the four-tab registry (docs/panel.md); tabs/ is one
               module per tab or Advanced page (status, models, chat, advanced, ...)
mcp/           management-plane MCP server (23 tools)
tray/          Windows notification-area app
watchdog/      recovery sidecar: its own process, port and MCP server (10 tools)
```

The dependency direction is one-way: `api` and `gui` and `mcp` all depend on `core`, and `core`
depends on nothing above it. A change that makes `core` import from `api` is a change to reject.

## Adding a config key

1. Add the field to the right model in `config.py`, with a default and a comment saying *why* that
   default.
2. If it cannot take effect without a restart, add it to `RESTART_REQUIRED_KEYS`.
3. Add the key to `config.example.yaml` with its shipped default — the file promises every key
   the app understands, and `tests/unit/test_config_example.py` fails until it does, comparing
   the YAML against `Config(data_dir=tmp).to_yaml_dict()` (the exact dump `save()` writes, with
   `data_dir` excluded per D31). To regenerate the body wholesale, `yaml.safe_dump` that dict
   with `sort_keys=False` against a scratch `SF_DATA_DIR` and paste it under the hand-written
   header; then put back the inline comments (`engine.cache_ram_mb`, `planner.preference`,
   `update.repo`) and annotate the key in the header if it is one a new user has to think about.
4. **The GUI needs no change.** The Advanced > Config page is generated from the pydantic
   model, so a new scalar key gets a type-aware widget automatically (DECISIONS.md D26). Two
   optional follow-ups: add a one-liner to `gui/state.CONFIG_FIELD_HELP` if the generated
   "`<kind>`, default `<x>`" is not explanation enough, and promote it out of Advanced into a named
   section by adding a `fields.row(...)` in `gui/tabs/setup.py` **and** listing it in that module's
   `COVERED_KEYS` — a test asserts every key has exactly one control.
5. A key whose type is a mapping (`dict[int, int]`) or a list of nested models
   (`tenants.registry`) is deliberately *not* generated: it needs a row widget, like
   `planner.reserved_mb` has on the GPU card. Add it to `gui/state.CUSTOM_WIDGET_KEYS` along with
   the widget, or the same test fails.
6. **Regenerate the generated docs.** Run `uv run python scripts/gen_docs.py`: the key appears in
   `docs/reference/settings.md` and `docs/contracts/settings-schema.json` (the settings registry,
   `settings_schema.py`, walks the model). It needs a description - a comment above the field in
   `config.py` or in `config.example.yaml` - or `tests/unit/test_settings_schema.py` fails. The same
   command refreshes the OpenAPI, MCP tool, error and capability contracts; CI runs
   `scripts/gen_docs.py --check`.

## GUI theming

The panel's colours, type and component skin come from **ForgeUI**, the ThemeForge design system
([LaserLloyd/ThemeForge](https://github.com/LaserLloyd/ThemeForge), a separate repository, not
part of this one), not from anything hand-written here. `gui/theme/` is ForgeUI's **drop-in
bundle** (its `ui-theme/` folder), vendored whole: the same folder every app gets. It is never
hand-edited:

```
gui/theme/ui-theme.js            the runtime and the full theme registry
gui/theme/ui-theme-base.css      element layer + the text standard (.ui-text-*, .ui-markdown)
gui/theme/ui-theme.css           contract tokens for every theme
gui/theme/ui-components.css/js   the component layer (not loaded by the panel today)
gui/theme/adapters/quasar.js     keeps Quasar's dark-mode flag in step with the theme's ground
gui/theme/adapters/quasar.css    contract tokens onto Quasar's components
gui/theme/themes.json            the registry (app.py reads it for the picker's names)
gui/theme/locales/               string tables for the component helpers
gui/theme/update.py              installs/updates the bundle without pip; no page loads it
gui/theme/files.json             sha256 of every bundle file; what the vendor check verifies
gui/theme/VERSION, README.md     bundle version and plug-in notes
gui/theme/adapters/studioforge-extra.css   OURS: copied here from gui/overlays/ by `forgeui vendor`
```

StudioForge's own theme choice lives in **one place**, the manifest `forgeui.json` at the
repository root: `settings.themes` (the picker's themes, in order), `settings.default`
(`glacier`) and `settings.storage_key` (`studioforge.theme`), plus `adapters` (`["quasar"]`),
`version` (`"1.2"`: any 1.2.x) and `extra_files` (the overlay). `gui/app.py` reads it as
`THEME_SETTINGS` and writes it as `data-*` attributes on the `ui-theme.js` script tag; the wheel
carries a copy next to `gui/app.py` (`pyproject.toml` `force-include`; the Dockerfile copies it).
`app.py` serves `gui/theme/` at `/sf-theme/` (exempted from the auth gate, since it has to be
reachable from the unauthenticated login page too) and injects six tags into every page's
`<head>` exactly once, each with a `?v=<sha256 prefix>` cache-busting query string computed at
import: the runtime, the Quasar script, base, tokens, the Quasar CSS, and the overlay. The
header's theme picker (`_theme_picker`) is a `ui.select` that calls `UITheme.set(slug)` in the
browser; `adapters/quasar.js` keeps Quasar's own dark-mode flag in step with the active theme's
ground, so `ui.run_with(dark=True)` stays fixed in Python regardless of which theme is active.
To offer another theme, add its slug to `settings.themes` in `forgeui.json` (opt-in themes such
as `electric-yellow` are listed on purpose, never by default) and restart the panel.

**StudioForge's own CSS** (the `sf-well`, `sf-danger-card`, `sf-warning-card` and
`sf-theme-picker` helpers) lives in `gui/overlays/studioforge-extra.css`. It is never appended
into a bundle file: `forgeui vendor` copies it to `gui/theme/adapters/studioforge-extra.css`
(listed under `extra_files`), the head loads it after `adapters/quasar.css`, and `files.json`
keeps proving the bundle. Edit the overlay, then vendor.

**Updating the bundle** (no registry, no private tooling):

```bash
uv pip install "forgeui @ git+https://github.com/LaserLloyd/ThemeForge@v1.2.0"   # the release forgeui.json accepts
python -m forgeui vendor                 # write the new copy (VERSION last)
python -m forgeui vendor --check         # exit 1 on any drift: version, stale/edited file, overlay, settings
# offline: python -m forgeui vendor --source <ThemeForge checkout, ui-theme folder, zip or wheel>
```

(`forgeui` is installed from the ThemeForge release tag, not PyPI: that name belongs to another
project. It is a development tool, not a dependency of StudioForge.) A hand-edited bundle file is
refused unless you pass `--force`; move the edit to the overlay first. When ForgeUI moves to a new
minor (1.3), change `version` in `forgeui.json`, vendor, and re-run the GUI tests.

`tests/unit/test_forgeui_vendor.py` proves, with the standard library alone, that the folder
matches its `files.json`, the overlay matches its source, the version stamp is one the manifest
accepts and the settings name real themes; with `forgeui` installed it also runs
`forgeui vendor --check`. CI installs it and sets `FORGEUI_REQUIRED=1`, and has its own
`forgeui vendor --check` step. `ruff` skips `gui/theme/` (`extend-exclude` in `pyproject.toml`):
the bundle follows ForgeUI's own style and `files.json` checksums every file in it. The wheel
ships `gui/theme/**` and `forgeui.json`.

To change how a Quasar component looks for every app, change ForgeUI's `adapters/quasar.css`
(and, for JS behaviour, `adapters/quasar.js`) in the ThemeForge repository and vendor the
release; for StudioForge alone, use the overlay. Never edit `gui/theme/*` directly.

No restart is needed for new colours: `/sf-theme/` is served with `max_cache_age=0`, so browsers
revalidate on every load and get the new bytes even under the old `?v=` (computed once at
import, see `_asset_version`). Restart the server (`POST /api/restart/server` with
`{"confirm": true}`, or the tray's Restart server) when the bundle gains a theme you add to
`settings.themes`, since the picker's options are read at import.

Quasar's named palette classes (`.text-cyan`, `.bg-orange`, `.text-purple`, `.bg-teal`,
`.text-blue-grey`, `.text-grey`/`.bg-grey`, used via `color=` props and `gui/state.py`'s
`CAPABILITY_ICONS`/`STATUS_COLOURS` maps) are remapped in `adapters/quasar.css`'s `@layer
overrides` to distinct `--cat-*` categorical tokens, not to hue-matched colours — `.text-orange`
lands on `--cat-2` (teal), `.text-purple` on `--cat-3` (gold), and so on; only `--offline`
("grey") is actually named for what it is. The contract promises the `--cat-*` set is
distinguishable from itself and >=3:1 on `--surface-0`, not that "orange" looks orange, and that
distinctness is test-enforced — don't "fix" the hues to match the names. Because `--cat-*` (and
`--offline`) are tuned as a foreground colour rather than as a fill, `.bg-<name>.text-white` is
covered separately in the same layer and routed through `--text-inverse` (the theme's flipped
page ink) instead of trusting the categorical token's own hue to hold a literal white label.

## Building the companion wheel

```bash
cd packages/studioforge-companion
uv build --wheel -o dist
```

It deliberately does not depend on `studioforge`: it is a pure HTTP client, installable on a laptop
or an immutable OS with `uv tool install`. It targets Python 3.11 rather than 3.12 for the same
reason — the machine running the agent frequently lags the rig.
