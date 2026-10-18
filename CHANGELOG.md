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
