# The control panel: four tabs

The panel (`:8080`) has exactly four top-level tabs, in this order:

| Tab | What it is for |
|---|---|
| **Status** | What a person checks first: health and boot phase, every GPU and what is on it, loaded models (busy or idle, tok/s), active requests, engine, version, uptime. Quick actions are limited to Unload, Open in Models and Open logs. No settings. |
| **Models** | One page for the whole library: search, filter chips (All, Loaded, Downloads), per-row state, a details drawer per model (load with the planner's recommended placement, VRAM fit bars before you press Load, pin, sampler defaults, presets, aliases, delete). Adding a model (Hugging Face repo id, URL or search) and the download queue live here too. |
| **Test Chat** | Pick a loaded model (or load one through the same drawer), optional system prompt, streaming reply with TTFT and tok/s, copy as curl. |
| **Advanced** | A left sub-navigation holding everything else (below). Nothing was removed: every control that existed in 2.0.x is reachable here or in Models. |

Advanced sub-pages: Serving & network, GPUs & placement, Accounts & keys, Engines, Watchdog & recovery,
Catalog & downloads, Integrations, Peers, Logs & metrics, Config, Diagnostics, About & updates.

## Screenshots

Glacier theme, from a scratch rig (`scripts/panel_demo.py`: fake GPUs and invented models, nothing loads on a real
GPU), captured by `scripts/capture_panel.py`. Both refuse any port outside 18080-18199.

| | |
|---|---|
| ![Status](screenshots/readme/panel-status.png) | ![Models](screenshots/readme/panel-models.png) |
| **Status**: health, GPUs, loaded models | **Models**: one table, chips, per-row state |
| ![Model drawer](screenshots/readme/panel-model-drawer.png) | ![Downloads](screenshots/readme/panel-downloads.png) |
| **The details drawer**: recommended placement pre-filled, per-GPU fit bars before Load | **Downloads chip**: add by repo id or link, search, queue |
| ![Test Chat](screenshots/readme/panel-test-chat.png) | ![Advanced](screenshots/readme/panel-advanced.png) |
| **Test Chat**: TTFT, tok/s, Copy as curl, Load options | **Advanced**: left menu (Integrations page shown) |

## The Models page

* The table: friendly name (the exact id under it), quant badge, size, family, "Fits on" (the catalog's recommended
  placement for the model, in the planner's words: `2x RTX 5090 · 128k`), and state. Click a column header or use
  the sort menu; the search box matches the id, the friendly name, quant, family and capabilities.
* State is one of idle, loading (with how long it has been going), loaded (busy or idle, tok/s) or error (the
  catalog's error code and its plain sentence, e.g. `[insufficient_vram] ...`), plus `pinned` and `leased` badges.
* The drawer (click a row, or `?section=Models&model=<id>`): **Load** is pre-filled in this order, field by field:
  what is running now (for a reload), the model's saved settings, the planner's recommended `load_args`, the
  server's defaults. The per-GPU bars and the verdict come from `manager.plan_preview` and update as you change the
  GPUs, context, KV type or slots; **Load** is withheld while the check says it cannot fit (a reload of a running model
  is never withheld, because the dry run still counts the running copy). **Restart as it is** is the old "Reload
  this model" (`manager.load(force=True)`, no arguments). Manage holds Pin, Test, Benchmark and Delete; Sampler
  defaults, Virtual models & presets and Aliases follow. "All settings..." opens the original three-tier dialog.
* Every box-changing control (pin, delete, save defaults, restart) is disabled with the reason for a remote viewer on
  an open install (the D32 rule), exactly like the old table's. Loading and unloading stay open, as before.

## How it is built

* `src/studioforge/gui/nav.py` is the single registry: `TOP_TABS` (four entries), `ADVANCED_PAGES`
  (the sub-pages), `LEGACY_ROUTES` (every old tab and section name) and `resolve_location()`.
* `src/studioforge/gui/app.py` builds the shell. Each tab and each Advanced page is built the first
  time it is opened, so a closed page costs no timers.
* Settings are still rendered from the single settings schema (`settings_schema.py`, through
  `gui/state.py` `config_field_specs`) by the same `_Fields` helper as before.
* Every action calls the same backend function as before (`manager.load`, `manager.unload`,
  `manager.set_pinned`, `registry.save_settings`, `registry.delete_model`, `downloader.*`,
  `apply_config_updates`, ...). The D32 rule for remote viewers on an open install is applied by the
  same helpers (`require_local_admin`, `admin_control`).
* No new HTTP route and no new API. The only new server behaviour is an HTTP 307 from `/` for an
  old-style link (below).

## Links and old names

`/?section=<Status|Models|Test Chat|Advanced>` opens a tab; `/?section=Advanced&page=<slug>` opens an
Advanced page; `/?section=Models&filter=<loaded|downloads>&q=<text>&model=<id>&repo=<owner/repo>` opens
the Models page filtered, with a model's drawer or the add-model dialog already open.

Every name the 2.0.x panel accepted (`?tab=` values and `?section=` values) is redirected (307) to the
canonical form above, keeping `repo`, `quant` and `model`. The protocol handler's
`?tab=download&repo=owner/repo` and `?tab=models&model=<id>` links keep working unchanged.
`tests/unit/test_panel_four_tabs.py` enumerates every old name.

| Old (2.0.x) | New |
|---|---|
| `?tab=dashboard`, `?section=Overview` | Status |
| `?tab=setup` | Status (the "Finish setup" banner) |
| `?tab=models`, `?section=Models` (Library) | Models |
| `?tab=download`, `?tab=downloads` (Models > Download) | Models, Downloads chip (+ add-model dialog when `repo` is given) |
| `?tab=chat` (Models > Chat) | Test Chat |
| `?tab=server`, `?tab=settings`, `?section=Integrations` | Advanced > Integrations |
| `?tab=logs`, `?tab=log`, `?section=Logs & Metrics` | Advanced > Logs & metrics |
| `?section=Serving` | Advanced > Serving & network |
| `?section=Engines` | Advanced > Engines |
| `?section=Placement` | Advanced > GPUs & placement |
| `?section=Users & Keys` (also `users`, `keys`) | Advanced > Accounts & keys |
| `?section=Updates` | Advanced > About & updates |
| `?section=Diagnostics` | Advanced > Diagnostics |
| `?section=Advanced` (no `page`: the 2.0.x page of every setting) | Advanced > Config |
| `/setup` (first-run guide) | unchanged |

## Old control to new place

### Overview (now Status)

| Old control | New place |
|---|---|
| "Finish setup" banner and link to the setup guide | Status banner |
| Base URL card with copy button | Status header strip (also Advanced > Serving & network) |
| GPUs panel (VRAM bar, utilisation, temperature) | Status, one card per GPU, plus what is on it and its leases |
| VRAM holders and **Reclaim** (orphaned llama-server) | Advanced > GPUs & placement |
| Loaded models cards (busy/idle, slots, tok/s, active requests, engine drift, tier) | Status, loaded models list (same detail lines) |
| Loaded card **Unload** | Status (Unload), Models drawer |
| Loaded card **Pin** | Models drawer, Models row state |
| Loaded card **Reload** (restart model) | Models drawer > "Restart as it is" (the same `manager.load(force=True)` with no arguments) |
| **Unload all** | Advanced > Watchdog & recovery |
| **Restart engines** | Advanced > Engines |
| **Restart server** | Advanced > Watchdog & recovery |
| Setup checklist (nine items, fix buttons) | Advanced > Diagnostics (Readiness checklist) |

### Models (Library, Download, Chat sub-tabs)

| Old control | New place |
|---|---|
| Library table, search box, sort by column header | Models table, search, same sort headers |
| Row badge: status (loaded/loading/unloaded/failed) | Row state chip (idle, loading with progress, loaded, error with code) plus pinned and leased badges |
| Row: copy model id | Drawer header |
| Row: Load / edit settings (three-tier dialog, optimal settings per GPU set, Load here, Measure parallel, Load at exactly N tokens) | Drawer > Load (recommended placement pre-filled, fit bars); Advanced options expander; "All settings..." opens the original three-tier dialog unchanged |
| Row: Unload | Drawer (and Status, for a loaded model) |
| Row: Pin / unpin | Drawer |
| Row: Test (one short completion) and A/B the draft model | Drawer > Manage |
| Row: Benchmark | Drawer > Manage |
| Row: Delete (registration or files; virtual model removes only the registration) | Drawer > Manage |
| Row: Edit virtual model / persona | Drawer > Virtual models & presets |
| Rescan | Models toolbar |
| LoRA adapters, virtual models, personas panel (Rescan adapters, New virtual model) | Models page, "Virtual models & adapters" section under the table |
| D66 "Unsupported arch" badge and the load refusal text | Row badge and drawer (Load disabled with the reason) |
| Download tab: "From Hugging Face" quant picker for `?repo=` | Models > Add model dialog |
| Download tab: Find a model (search, sort, period, MTP filter, quant picker, queue) | Models > Add model dialog |
| Download tab: Downloads queue (pause, resume, cancel, show finished, disk line) | Models > Downloads chip |
| Setup cards: Where models live, Defaults for every load, Downloads & Hugging Face | Advanced > Catalog & downloads |
| Chat sub-tab (model picker, load/unload, details, conversation, attach image, quick tests, request settings, thinking, copy all, clear) | Test Chat (unchanged) plus copy as curl and a link to the Models drawer |

### Other sections

| Old section | New page (all under Advanced) |
|---|---|
| Serving (Listening, HTTPS, idle unload, CORS, Network & access, credentials) | Serving & network |
| Engines (engine controls, activate, smoke test, install, Backend & what you can run, feature support) | Engines |
| Placement (GPU planner settings, GPU leases, Tenants + warm, Peers placeholder) | GPUs & placement (+ Peers) |
| Integrations (snippets, create key, Point OpenClaw, lmstudio:// protocol) | Integrations |
| Users & Keys (posture, accounts, API keys, audit log) | Accounts & keys |
| Updates | About & updates |
| Logs & Metrics (live log, Prometheus text) | Logs & metrics |
| Diagnostics (Run checks, support bundle, Health) | Diagnostics |
| Advanced (Start with Windows, every remaining setting, raw configuration, planner quant affinity, Unload all models) | Config |
| Watchdog port and bind address (inside Serving and the settings catch-all) | Watchdog & recovery shows the same schema fields, plus restart/unload-all |

New in 2.2.0 (presentation only): Status page, Models search/filter chips/drawer/fit bars,
Test Chat copy-as-curl, the Advanced sub-navigation, Watchdog & recovery and Peers pages.
