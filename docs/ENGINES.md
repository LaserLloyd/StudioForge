# Engines and hardware detection

StudioForge serves models through llama.cpp's `llama-server`. This page is about the two things
that decide *which* `llama-server` runs and *where*: the **engine manager** (download or build,
verify, smoke-test, version) and **hardware detection** (which GPUs there are, through which API).
For what a given engine build can do once it runs, see [ENGINE-FEATURES.md](ENGINE-FEATURES.md).

## Backends

An engine is built for exactly one GPU API:

| Backend | Hardware | Prebuilt archive | Source build flags |
| --- | --- | --- | --- |
| `cuda` | NVIDIA | `win-cuda-<ver>` (+ the `cudart` bundle). No Linux archive upstream, so Linux + NVIDIA is always a source build | `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<sm list>` |
| `rocm` | AMD | `rocm-<ver>` (Linux tarball, Windows zip), `hip-radeon` (Windows) | `-DGGML_HIP=ON -DAMDGPU_TARGETS=<gfx list>` |
| `vulkan` | any Vulkan GPU (AMD, Intel, NVIDIA) | `vulkan` | `-DGGML_VULKAN=ON` |
| `metal` | Apple silicon | the macOS arm64 archive (no backend token in its name; labelled `metal` here) | `-DGGML_METAL=ON` |

CPU archives are never selected: StudioForge is GPU-only (see STRICT_GPU below).

The architecture lists (`sm`, `gfx`) come from the hardware probe, never from a table in the code:
a mixed 5090 + 3090 box builds for `86;120`, a Strix Halo for whatever `amd-smi`/`rocminfo` report
(`gfx1151` is just one value it can be). If the probe reports no target, a CUDA or ROCm source
build refuses to start rather than guessing.

Which backend is *selected for a box* is whatever the probe detected (below). The archive chosen is
then the one for that backend; with no matching archive the manager falls back to a source build
(`engine.allow_source_build`), whose toolchain it names when building is off.

Limits of this release, stated so they are not discovered the hard way:

- **One backend per process.** A box with an NVIDIA card and an AMD card uses whichever the probe
  finds first.
- **One engine directory per tag** (`engines/<tag>/`). Holding a HIP *and* a Vulkan build of the
  same tag side by side is not supported yet, which is also why the backend benchmark below
  compares results you gathered, rather than switching engines on its own.
- Source builds on Windows for HIP/Vulkan, and any macOS source build, are not exercised.

## Hardware detection

`gpu.backend` (default `auto`) chooses the probe; `SF_GPU_PROBE` is the environment equivalent and
applies only while the config value is `auto`.

| `gpu.backend` | Probe | Tool it runs |
| --- | --- | --- |
| `auto` | tries the four below in order, first one that lists a GPU wins | |
| `nvml` / `cuda` | `NvmlGpuProbe` | NVML (`nvidia-ml-py`), no process spawn |
| `rocm` | `RocmGpuProbe` | `amd-smi static --json` + `amd-smi metric --json`; else `rocm-smi --showmeminfo vram --showproductname --json`; else `rocminfo` |
| `vulkan` | `VulkanGpuProbe` | `vulkaninfo --summary` for the devices, full `vulkaninfo` for the memory heaps, `vulkaninfo --json` as a fallback |
| `metal` | `MetalGpuProbe` | `system_profiler SPDisplaysDataType -json` (macOS only) |
| `null`, `fake` | tests and GPU-less boxes | none |

**Detection order and what `auto` records.** NVML first, so a working NVIDIA driver always gets
NVML — identical to the behaviour before `auto` existed. Vulkan only comes after ROCm, so an
NVIDIA or AMD card is never driven through Vulkan merely because the Vulkan runtime is installed.
`GET /api/gpus` returns `probe: {selected, source, attempts: [{backend, ok, reason, gpu_count}],
notes}`: which backend answered, which tool it used, and *why each earlier one was skipped*
(`"NVML unavailable: ..."`, `"none of amd-smi, rocm-smi, rocminfo is installed"`). When no GPU is
found at all, `/health` says `no GPU detected` and points at that endpoint. Software Vulkan
devices (llvmpipe) are dropped, since CPU inference is exactly what STRICT_GPU forbids.

Every GPU record carries `backend`, `vendor`, `gfx_target` (AMD, from the driver) and
`unified_memory`.

### Unified memory (Strix Halo, Apple silicon)

The budget formula is the same on every backend and has **no heuristic bonus**:

```
total  = what the driver / tool reports for the device
free   = what the driver / tool reports as still available
budget = free        (the planner then applies planner.headroom_fraction as usual)
```

For a unified-memory device (`unified_memory: true`) that means:

- **ROCm APU** (`amd-smi` reports a `DDR`/`LPDDR`/`UMA` VRAM type): the driver's VRAM figure, which
  is the BIOS/driver carve-out; `free = total - used`. GTT (system RAM the GPU could map) is *not*
  added. If you want a larger budget, raise the carve-out in the BIOS/driver.
- **Vulkan integrated GPU**: the largest `DEVICE_LOCAL` heap; `free` = its `VK_EXT_memory_budget`
  budget, capped at the heap size.
- **Metal (Apple silicon)**: `system_profiler` reports no VRAM, so the pool is physical RAM and
  `free` is the OS's available memory. Metal itself caps a process below physical RAM, so on Metal
  `free` is an **upper bound** and the planner's headroom is what absorbs the difference.
- `rocminfo` is identification-only (no live counters): its pool size is reported as both total and
  free, and a note says so.

### `--device` names

The planner addresses GPUs by index; the supervisor turns that into llama.cpp's device names per
backend: `CUDA<i>`, `ROCm<i>`, `Vulkan<i>`, `MTL<i>`. The index is the position in the *probe's*
list, which is not guaranteed to equal llama.cpp's own numbering on Vulkan; check
`llama-server --list-devices` (`EngineManager.list_devices`) on a new box.

## Checksum verification

Every downloaded engine archive is checked against a published SHA-256 before it is extracted,
looking in this order:

1. GitHub's per-asset `digest` field (`sha256:...`) from the release API — `github-digest`;
2. a `<archive>.sha256` / `.sha256sum` sidecar asset — `sha256-sidecar`;
3. a release-wide `SHA256SUMS` / `SHA256SUMS.txt` / `checksums.txt` asset — `sha256sums`.

- **Match** → recorded `verified: true`, with the source and the digest.
- **Mismatch** → the archive is deleted and the install fails with `engine_checksum_mismatch`.
- **Nothing published** → the install proceeds, recorded `verified: false` with the reason, logged
  as a warning, returned in `warnings`, and shown by `engine list`. Never silent. Set
  `engine.require_verified: true` to refuse such installs (`engine_unverified_refused`).
- The `cudart` bundle is checked the same way when it has a digest, and warned about when it does
  not.
- A local source build has no archive; it is recorded `verified: false`, `verified_by:
  source-build ...`, with no warning, and listed as `source`.

What this is and is not: the digest comes from the same release as the archive, so it catches a
corrupt, truncated or swapped download. It is **not** a signature and does not defend against a
compromised upstream release.

`engines/<tag>/engine.json` records, per installed engine: `backend`, `variant`, `verified`,
`verified_by`, `archive_sha256`, `smoke_result` (`passed` / `partial` / `failed` / `not_run`),
`smoke_tested_at`, `installed_at`, `warnings`. Engines installed before this existed read as
`verified: null` ("unknown"), not as unverified.

```
studioforge engine list [--json]      # read-only: backend, verified, smoke, install date
studioforge engine install b10425 [--activate]
studioforge engine update [--check]   # install newest on engine.update_channel, smoke-test, then activate
studioforge engine select             # cached backend recommendation (read-only)
studioforge engine benchmark --model m.gguf --yes
```

The older single-command flags (`engine --update`, `--check`, `--install`, `--activate`,
`--smoke-test`, `--list`) still work. `GET /api/engine` returns the same fields under `installed`.

## STRICT_GPU

The product default: every model sits entirely in GPU memory, or the load is refused with the
arithmetic. Concretely:

- `--n-gpu-layers 999` and `--fit off` are passed on **every** spawn, as the last tokens of the
  argv (after the user's extra flags, so llama.cpp's last-wins rule cannot undo them).
  `--fit` defaults to *on* upstream and would silently shrink the plan.
- These are refused, at save time and again at launch: `--cpu-moe`, `--n-cpu-moe`, `--n-cpu-ffn`,
  `--no-kv-offload`, `--override-tensor`, `--rpc`, `--no-host`, `--device none`, a changed `-ngl`
  or `--fit`, and the other families in `engine.POLICY_FAMILIES`.
- `LLAMA_ARG_*` is stripped from the child's environment.

`engine.strict_gpu` (default `true`) is the single switch. Turning it off changes **nothing by
itself**. It only makes a model's own, off-by-default `allow_partial_cpu_offload` setting
effective, and even then only `--n-cpu-moe` and `--no-kv-offload` (and their short spellings)
become legal in that model's `extra_flags`. Everything else stays refused and `-ngl 999 --fit off`
are still always passed. The planner still budgets the model as fully resident, so a partial
offload is a way to fit a model the planner refuses, at the speed cost the operator accepted —
not something the planner reasons about. There is no panel control for the per-model flag yet; set
it through `PUT /api/models/{id}/settings`.

## Engine preference and the backend benchmark

When a GPU can run more than one backend, which is better depends on the workload. On Strix Halo
the HIP build has been measured ahead on prompt processing and the Vulkan build ahead on
generation, so StudioForge exposes the trade rather than hiding it:

| `engine.preference` | Picks the backend with |
| --- | --- |
| `balanced` (default) | the best geometric mean of speed relative to the fastest backend on each axis |
| `fastest_replies` | the highest `tg128` (token generation) |
| `fastest_long_prompts` | the highest `pp512` (prompt processing) |

Ties go to the vendor-native API (CUDA, ROCm, Metal, then Vulkan). The numbers are
`llama-bench`'s `pp512` and `tg128` in tokens per second, cached per
`(backend, GPU, engine tag, model class)` in `engines/backend-bench.json`; a new engine tag
invalidates a result by omission. `core.backend_select.select_backend(results, preference)` is a
pure function over those results.

**Nothing benchmarks on its own.** `studioforge engine benchmark --model m.gguf --yes` is the only
thing that runs `llama-bench` (always with `-ngl 999`): it loads the model onto the GPU for up to
several minutes, so it needs `--yes` and the library function refuses without `confirm=True`. A
source-built engine has no `llama-bench` (only `llama-server` is compiled). Nothing consumes the
recommendation automatically in this release: `engine select` reports it.

## What is verified, and what is not

Everything above is covered by unit tests with fakes, and **none of it has run on real AMD, Intel,
Vulkan or Apple hardware**. The probe fixtures in `tests/unit/fixtures/hw/` are hand-written from
each tool's documented output format and labelled synthetic; the real tools' output can differ by
version. The macOS arm64 archive being a Metal build, the `llama-bench` JSON shape, and
`llama-bench --device` are likewise taken from upstream documentation and need one confirmation on
a real machine.
