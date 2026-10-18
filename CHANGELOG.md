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
