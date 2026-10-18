# Changelog

StudioForge is calendar-versioned: a major, then the release date (`1.26-09-23`; `1.26.9.23` in
the wheel metadata, see [`docs/RELEASING.md`](docs/RELEASING.md)). Every change of substance is a
numbered decision in [`DECISIONS.md`](DECISIONS.md), with the measurement behind it, so this file
lists the releases and the decisions each one carries rather than repeating them.

To see what a running server includes, ask it: `GET /api/capabilities` lists every decision number
the build implements, and the `build` field on `/health` names the exact commit.

## Unreleased (on `main`)

The version string still reads `1.26-09-23`; the `build` field tells these commits apart.

- D71: a MoE's compute term is sized from the weights that run.
- D72: two slots over one KV pool; context checkpoints become settings.
- D73: host-resident tensors are not charged to a GPU.
- D74: an MTP draft head can be attached by file.
- D75: one auto-unload timer for every model whose duration nobody stated.
- D76: `load-recommended` honours the saved placement; a refusal walks down to a window that fits.
- D77 to D79: the VRAM Tenant Protocol (`vt.1`), the tenant registry, `devices: "auto"`, and the
  `studioforge-tenant` SDK ([`docs/TENANTS.md`](docs/TENANTS.md)). StudioForge never starts or
  stops a tenant.
- LM Studio compatibility ([`docs/API.md`](docs/API.md)): the native v1 REST API under `/api/v1`
  (`GET /models`, `POST /models/load` and `/unload`, `POST /models/download` with
  `GET /models/download/status`; `POST /chat` answers `501 not_implemented`), Anthropic-compatible
  `POST /v1/messages` (streaming, tools, images; `x-api-key` and bearer both work), and a stateless
  OpenAI `POST /v1/responses`. Per-request `ttl` and `priority` behave identically on all of them.
  New error codes: `not_implemented`, `unsupported_parameter`, `instance_not_found`,
  `download_not_found`. Wire fixtures in `tests/fixtures/lmstudio/` (derived from LM Studio's docs,
  not captured), an offline suite in `tests/unit/compat/`, and an opt-in `tests/compat/` suite for a
  running test instance.

- GUI theme: ForgeUI 1.2.0 (the ThemeForge bundle; was a ThemeForge-labelled UnifyingTheme 1.0.0
  copy). The panel now offers the new Slate, High Contrast and High Contrast Light themes beside
  the existing seven (Night Red stays, the other opt-in themes stay out). `gui/theme/` is vendored
  whole by `python -m forgeui vendor` from a per-repo manifest, `forgeui.json`, which is now the
  single place for the picker's themes, the default (`glacier`) and the storage key
  (`studioforge.theme`); the wheel ships a copy beside `gui/app.py`. The `sf-*` helper classes that
  used to be appended into `adapters/quasar.css` moved to an app-owned overlay,
  `gui/overlays/studioforge-extra.css`, vendored as `adapters/studioforge-extra.css` and loaded
  after the Quasar adapter, so `files.json` proves the bundle again. `tests/unit/test_forgeui_vendor.py`
  and a CI step run `forgeui vendor --check`, so the bundle cannot drift. The owner-machine
  registry (`sync_theme.py install studioforge`) is retired; see
  [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md#gui-theming). The tailwind adapters are no longer
  shipped (the panel uses Quasar).

Security, MCP hardening, CLI and first-run wizard (refactor workstream W3-SF; the security release
that 2.0.0 carries). Read **BREAKING** first.

### BREAKING

- **All three services bind `127.0.0.1` by default** (`server.host`, `gui.host`, `watchdog.host`; they
  were `0.0.0.0`). A `config.yaml` that already names a host keeps it: `save()` has always written
  every key, so an existing install is unchanged by the new default.
- **A network bind needs a credential.** If any enabled service listens on a non-loopback address and
  no owner account, admin API key or `server.api_key` exists, `studioforge serve` refuses to start
  with `auth_required_for_remote_bind` and exit code 4 (the tray shows "needs setup" and does not
  respawn). With a credential in place every request needs one (`/health` stays public).
- **`server.cors_origins` defaults to `[]`** (was `["*"]`); credentials stay off. A saved `["*"]` is
  kept (and warned about at start), but it no longer lets a web page reach the MCP endpoints, which
  now check `Origin` against loopback plus `cors_origins` and answer `403 origin_not_allowed`.
- **Scopes are enforced.** `read` < `inference` < `admin` per route, by prefix (`/v1/`, `/api/v0/`,
  `/api/v1/` need `inference` to write; the old D32 admin prefix list and every unclassified mutating
  route need `admin`). MCP tools are annotated, and destructive tools need an `admin` key
  (`permission_denied` otherwise).
- **Deprecated, still working:** `server.api_key` (mirrored into the key table as the admin key
  `legacy`; `keys revoke legacy` disables it) and `mcp.pin` (admin on the MCP endpoints only, with a
  logged warning; once any scoped key exists it no longer opens the plain HTTP routes). Both warn in
  `studioforge doctor`. Removal is planned no earlier than the next major.

Migration for an existing user, exactly:

1. Upgrade. Nothing changes if `server.host` was already `127.0.0.1`.
2. If your `config.yaml` has `host: 0.0.0.0` for the gateway, panel or watchdog, the server stops
   with the message above. Either run `studioforge setup` (creates the owner account; choose "my
   local network" to get HTTPS and an admin key), or `studioforge keys create --name admin --scope
   admin`, or set the three hosts to `127.0.0.1` to stay local.
3. If you set `server.api_key`, you can leave everything as it is. When ready, create one key per
   client (`studioforge keys create --name openclaw --scope inference`), switch the clients over, then
   clear `server.api_key` and `mcp.pin`.
4. Clients that call the API without a credential from another machine now get `401`. Give them a key.
5. The database migrates to schema 9 on start (`studioforge migrate --dry-run` previews it; `migrate`
   backs the database up first).

### Added

- Accounts and keys: owner account (argon2id; the hash lives in the secret store, not the database),
  API keys `sf-<r|i|a>-...` (hashed at rest, optional expiry, per-key requests/min, last-used,
  revocation), migration `009_accounts_keys.sql`, `studioforge keys|users ...`, `GET|POST /api/keys`,
  `DELETE /api/keys/{key_id}`, `GET|POST /api/users`, `POST /api/users/password`,
  `GET /api/auth/status`. New error codes `permission_denied`, `api_key_revoked`, `api_key_expired`,
  `key_rate_limited`, `origin_not_allowed`, `auth_required_for_remote_bind`, `invalid_tls_config`.
- TLS: `server.tls.mode` (`off` | `self_signed` | `cert`) with `cert_file` and `key_file`, for all three
  listeners; the self-signed certificate's SHA-256 fingerprint is in `GET /api/status` and the panel.
  New dependency `cryptography` (declared; it was already installed transitively).
- Secrets at rest: `config.yaml` and its `.bak` are `0600` (owner-only ACL on Windows); `hf.token` and
  the owner password hash use the OS keyring when the optional `keyring` extra is installed, else
  owner-only files. New dependency `argon2-cffi`.
- Audit log `logs/audit.jsonl` (append-only JSONL, rotated; `logging.audit_file_max_mb`,
  `audit_file_backups`), `GET /api/audit`.
- MCP: `readOnlyHint` / `destructiveHint` / `idempotentHint` on every tool, per-tool scopes, Origin
  validation on `/mcp` and the watchdog, audit of every mutating call, both protocol revisions
  (`2025-11-25` handshake and `2026-07-28` stateless) covered by tests.
- Operations: `studioforge doctor [--json] [--strict]` and `GET /api/doctor`; `studioforge
  support-bundle` and `POST /api/support-bundle` (credentials removed, grep-tested); Prometheus
  `GET /metrics` (gateway counters plus every loaded model's own metrics); `studioforge migrate
  [--dry-run]`; `POST /api/server/stop`.
- CLI: `setup`, `ls`, `ps`, `load [--ttl --context-length --gpu --preset]`, `unload [--all]`, `get`,
  `server start|stop|status`, `log stream`, `keys`, `users`, `update check|apply|rollback`
  (reference: [`docs/CLI.md`](docs/CLI.md), generated).
- First-run wizard: `studioforge setup` (interactive or fully flag-driven for CI) and the panel's
  `/setup` page: account, hardware, engine, model libraries (offers LM Studio's), starter model, test
  chat, snippets, exposure.
- Panel: 11 sections (Overview, Models, Serving, Engines, Placement, Integrations, Users & Keys,
  Updates, Logs & Metrics, Diagnostics, Advanced), owner-password sign-in, settings labels read from
  the settings schema.
- Docs: [`docs/SECURITY.md`](docs/SECURITY.md) (threat model, bind rules, keys, TLS, MCP),
  [`SECURITY.md`](SECURITY.md) (disclosure contact placeholder), the capability registry gains a `cli`
  column.

Contracts foundation (refactor workstream W0-SF). All additive; no serving behaviour changes.

- feat(errors): `studioforge.error_catalog` lists every `error.code` with its HTTP status, message
  template, fix and docs slug (`errors/<code>`). The error envelope gains `fix` and `docs` for
  catalogued codes; `message`, `type`, `code` and `param` are unchanged and no code was renamed. A
  unit test scans `src/` and fails on an uncatalogued or dead code.
- feat(config): `studioforge.settings_schema` walks the config model into a registry (type,
  default, description, range or choices, basic or advanced, `SF_` env name, restart required,
  secret). `config.yaml` gains `schema: 1`; files without it load unchanged.
  `GET /api/settings/schema` serves the registry read-only.
- feat(contracts): `studioforge.capability_registry` maps each capability to its REST route, MCP
  tool, `sfctl` command and panel control (`null` where a surface does not exist). A parity test
  checks every declared surface exists and reports the gaps.
- feat(contracts): `docs/contracts/` freezes `openapi.json`, `mcp-tools.json`, `errors.json`,
  `settings-schema.json` and `capabilities.json`; `docs/reference/settings.md` and
  `docs/reference/errors.md` are generated. `scripts/gen_docs.py` rewrites them, and
  `scripts/gen_docs.py --check` (a CI step, and `tests/unit/test_contracts_drift.py`) fails when
  they are stale.
- feat(api): `GET /api/version` also returns `product` and `api_versions`
  (`openai: v1`, `lmstudio_v0: v0`, `studioforge_api: v1`); `version` and `build` are unchanged.
- chore(agents): `.claude/agents/` holds review subagents for contracts, security, migrations,
  docs drift and accessibility.

Prosumer refactor, W1 (engines and hardware detection) -- see [`docs/ENGINES.md`](docs/ENGINES.md):

- Hardware detection beyond NVIDIA: `RocmGpuProbe` (amd-smi / rocm-smi / rocminfo),
  `VulkanGpuProbe` (vulkaninfo), `MetalGpuProbe` (system_profiler) and an `AutoGpuProbe`
  (NVML, ROCm, Vulkan, Metal) that records why each skipped backend failed. New config
  `gpu.backend` (default `auto`; NVIDIA boxes still get NVML). `GET /api/gpus` gains `probe`; GPU
  records gain `backend`, `vendor`, `gfx_target` and `unified_memory`.
- The engine manager is backend-aware: CUDA, `rocm`/`hip`, `vulkan` and Metal archives are chosen
  from the detected backend, and source builds take per-backend CMake flags (`-DGGML_HIP=ON
  -DAMDGPU_TARGETS=<detected gfx>` etc.).
- Engine archives are checksum-verified (GitHub digest, `.sha256` sidecar or `SHA256SUMS`); with
  no published checksum the install is recorded `verified: false` and warned about, and
  `engine.require_verified` refuses it. `engine.json` now records backend, verification, smoke
  result and warnings.
- `studioforge engine list|install|update|select|benchmark`; the single-command flags still work.
- `engine.strict_gpu` (default true) and a per-model, off-by-default `allow_partial_cpu_offload`:
  only with both set are `--n-cpu-moe` / `--no-kv-offload` legal. `-ngl 999 --fit off` are always
  passed.
- `engine.preference` (`balanced`, `fastest_replies`, `fastest_long_prompts`) and a cached
  `pp512`/`tg128` backend benchmark (`engine benchmark --yes` is the only thing that runs it).
- New error codes: `engine_checksum_mismatch`, `engine_unverified_refused`,
  `engine_benchmark_unconfirmed`.

## 1.26-09-23 (2026-09-23)

- D58: the `loaded` model alias.
- D59: `devices` places, `allowed_devices` chooses.
- D60 to D65: review rounds: same-generation pools, the GPU-only policy enforced, leases that
  survive a restart, the engine's stable channel, the planner's prediction error measured,
  on-demand loading as the default.
- D66 to D69: models the engine cannot load are refused up front; the quant picker shows MTP
  heads; the Chat tab as an ops bench; log hygiene.
- D70: identity and attribution, and the `build` field beside `version`.

## 1.26-09-04-3 (2026-09-04)

- D57: the proxy forwards the server's own guidance, bounded. Relative lease states, undeliverable
  vacate requests, holder-waiver proof.

## 1.26-09-04-2 (2026-09-04)

- D55: hardening round; an open install is open for inference, not for changing the box.
- D56: the vacate protocol, a better priority class may ask a tenant to leave.

## 1.26-09-04 (2026-09-04)

- D53: a lease refusal names the lease (`gpu_leased`, lease holder family, `context_exceeded`).
- D54: the effective launch flags of each child, and prompt-cache counters.

## 1.26-08-31 (2026-08-31)

- D52: the local-model gate (`check_loaded_model`, `/api/model-gate`).

## 1.26-08-30 (2026-08-30)

- D49 to D51: engine truth and idempotent activation; planning from observed footprints.

## 1.26-08-28 (2026-08-28)

- D44 to D48: PIN rate limiting, the three load-priority tiers, and a tier that survives a restart.

## 1.26-08-23 (2026-08-23)

- The first distributable package: release workflow, updater, and the `sfctl` companion carrying
  the server's version. Decisions D1 to D43.
