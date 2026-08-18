# Known limitations

Honest list. Anything here is a deliberate boundary or a known rough edge, not an oversight.

## GPU-only, by design

A model that does not fit entirely in VRAM is **rejected**, never partially offloaded. There is no
`--n-gpu-layers` value other than `999` anywhere in the codebase, and every launch passes
`--fit off` so the engine cannot shrink the plan behind the planner's back. If you want CPU
offload, this is the wrong server.

Consequences you will actually hit:

* A 123B model at Q4 will not load on a 32 GB card, and the rejection tells you which quant would.
* Free VRAM is read live, so a load that succeeded an hour ago can be rejected now. That is
  correct behaviour, not a regression.

## The VRAM estimate is an estimate

Weights come from summed GGUF tensor bytes (verified within 2% of file size across the whole test
library) and the KV cache is computed exactly from GQA head counts. But the compute/graph buffer is
a calibrated fraction, not a derivation, so on an unusual architecture the first load may be
mis-predicted.

Every load records predicted-vs-actual, and at **startup** the planner tunes the factor from that
history via `suggest_overhead_fraction()`, clamped to **0.03–0.15** and requiring at least 5 clean
observations. It is applied in memory only — `config.yaml` is never rewritten, so a bad calibration
is undone by a restart. It is *not* re-tuned per load: a planner whose arithmetic shifts under a
running server is harder to reason about than one that is wrong in a fixed way.

The `headroom_fraction` guard (default 10% of total VRAM) exists to absorb the remaining error.

### Calibration history before 2026-08-17 is contaminated

Until this date `actual_bytes` was the **whole device's** `used_bytes` summed across the plan's
devices, not our own child's. On a workstation that also runs a desktop, a browser and ComfyUI, that
number is mostly other processes. Across 540 recorded rows the median actual/predicted ratio is
**2.97** and p90 is **12.0** — earlier versions of this document claimed 0.81–1.23, which was never
what the column measured.

Measurement is now per-pid, and rows recorded that way carry `note = "per_pid"` in
`load_observations`. **Calibration ignores every row without that marker**, so the old rows are
inert rather than harmful — but do not read them as evidence about the estimate, and do not expect
a tuned factor until enough post-fix loads have accumulated.

Where per-process VRAM cannot be enumerated at all (containers, WSL, MIG) no observation is recorded
rather than a device total being substituted, so on those platforms the factor never self-tunes.

## Concurrency estimates are arithmetic, not benchmarks

`models.default_parallel: auto` sizes the slot count from a VRAM bound and a bandwidth-knee bound
(DECISIONS.md D17, revised by D22). The VRAM half is now **exact rather than analytic**: the planner
walks down from the cap asking its own `estimate()` whether N slots fit, which is the same function a
real load asks, so the two cannot disagree. (The old quotient divided by a *uniform* KV cost, which
for an iSWA model was 20x the real figure — every Gemma-4 row read `max_parallel: 1 (vram)` with
34 GB free.) The knee half is still a model:

* `CTX_FILL_FRACTION = 0.5` assumes slots sit half full. Real workloads vary enormously.
* The MoE derate of 0.5 is an approximation of experts fanning out with batch size, not a
  measurement. No benchmark on this rig has located a real knee yet.
* `--ctx-checkpoints` defaults to **32 per slot** and is **not modelled**, so predicted-vs-actual
  should be expected to drift upward at high slot counts. Watch it before trusting 8 slots.

Both approximations err toward *fewer* slots, which is the direction that cannot cause an OOM.

The per-layer KV geometry the walk stands on is itself derived from GGUF metadata and llama.cpp's
source, **not from measured allocations**. It reproduces the observed iSWA figures byte for byte, and
the hybrid recurrent-state term (~157 MB per slot for a Qwen3.5) is derived from the SSM dimensions;
but a model whose metadata omits the geometry reports `attention_kind: "unknown"`, and every KV
number for it should be distrusted rather than read as the cheap case.

`--kv-unified` is exposed per model and defaults **off**. Its semantics have not been verified
against a real load here; do not turn it on for a production model without checking `/props` and a
long single request first.

## Catalog speed numbers are a model, and the constants are approximations

`/api/catalog` and the MCP `list_models` report estimated tokens/second per loading option. The
arithmetic is derived rather than guessed — decode is bandwidth-bound, prefill is FLOP-bound
(DECISIONS.md D20, corrected by D22) — but every number it divides by is either a vendor peak figure
or a constant chosen to fit two measurements.

**The GPU table is nominal, not measured**: RTX 5090 1792 GB/s / 209 TFLOPS, RTX 4090 1008 / 165,
RTX 3090 936 / 71. No kernel reaches peak, and the gap is not a constant.

**Four of the estimator's constants are approximations with a stated direction of error** (the full
table with its evidence is in D22): a 1.5 ms per-token latency floor, a MoE decode derate of 0.45, a
doubled per-extra-device penalty for MoE, and a MoE prefill derate of 0.4. The three MoE constants
all push MoE rows *slower*; that is deliberate, because calibration can correct a number that is in
the right neighbourhood and D20 named over-promising as the failure to avoid. After them the two
measured anchors land at 39.4 measured / 36.1 estimated (dense Gemma-4 31B on 2x5090) and 37.3 / 47.4
(Qwen3.5-122B-A10B on all four). **Two anchors, both on this rig, both at one slot** — nothing here
has been validated at 4 or 8 slots, on another GPU generation, or on a dense model above 31B.

`est_gen_tps` is quoted at **8192 tokens of context in the window** (`REFERENCE_FILL_TOKENS`), not at
the row's own `ctx_per_slot`. That is one ordinary turn; a longer conversation is slower, which is
what `est_gen_tps_full_ctx` reports. Neither is a promise about a specific request — a real
conversation moves between them as it grows.

A MoE's *active* parameter count is only partly derivable from GGUF metadata. The estimator charges
the dense trunk (attention + output embedding, derived from the architecture dimensions) in full and
the routed share only to the experts, which for the 122B gives 7.0B against the ~10B its name claims.
The remainder is a shared expert and dense FFN layers that the metadata does not describe. The error
is in the safe direction (slower, not faster), but a MoE's uncalibrated estimate is systematically
less trustworthy than a dense model's.

Calibration closes the rest — the median of `measured / estimated`, most specific tier first — but it
needs data:

* Nothing is recorded until a model has been **resident and busy for two minutes**, and only while
  it is generating. An idle window records nothing rather than a zero.
* At least **two** matching observations are needed before a factor is applied; one is noise.
* Factors outside **[0.1, 3.0]** are discarded, so a mis-parsed metric cannot make the catalog
  confidently wrong.
* Only rows stamped with the **current `estimator_version`** contribute. A formula change retires
  every previous ratio (they corrected a formula that no longer exists), so **this rig is
  uncalibrated again after each such change** until fresh samples accumulate. The measurements
  themselves survive and still populate `measured_gen_tps`.
* A model nobody has run yet falls back to the `peers` tier: *other* models of the **same density**
  (dense vs MoE) on the same GPU class and the same device count, as a median of per-model medians.
  On a library with only one measured MoE and no measured dense model, that tier is empty and the
  answer is `basis: "none"` — correctly, since a dense model's efficiency does not predict a MoE's.

So a freshly scanned library shows `"estimated"` everywhere and stays that way until models are
actually used. Read `confidence` before trusting the number: `"measured"` means this exact placement
and context really were observed.

Three further caveats:

* `fits` is true of the instant the catalog was built (cached 20s). Another client can take the VRAM
  a second later. The load itself is still planned against live VRAM, so the worst case is a
  refusal with numbers, never a degraded load.
* `max_parallel` inherits the concurrency approximations above (`CTX_FILL_FRACTION`, the MoE knee
  derate). Both err toward fewer slots.
* `recommended` optimises for **this server's** definition of enough context (`models.default_ctx`,
  raised to `models.thinking_default_ctx` for a thinking model). A client whose real need is a
  60-token classification prompt is being handed a far larger window than it wants; read the whole
  `options` table rather than the recommendation when your workload is not a conversation.

## Reserving GPUs for other software

`planner.excluded_devices` and `planner.reserved_mb` exist for a box that shares its GPUs (ComfyUI,
a training run). Both default empty. Two caveats:

* They constrain **our** planner only. Nothing enforces the reservation against the other program,
  and nothing stops it taking the memory first.
* A per-model `device_override` outranks `excluded_devices` (with a warning). `reserved_mb` applies
  even to a forced placement.

## Who is holding the VRAM: what can and cannot be answered

`GET /api/vram/holders`, the Dashboard's "VRAM holders" panel and `/api/status.vram_processes` name
every process holding GPU memory and classify it (`ours`, `child-of-live-process`, `orphan`,
`foreign`). Four honest limits (DECISIONS.md D23):

* **Per-process byte figures are platform-dependent.** NVML reports 0 for every process on Windows
  under the WDDM driver model, so the numbers come from the Windows performance counter
  `\GPU Process Memory(*)\Dedicated Usage` — the same one Task Manager shows. That figure is a
  **per-process total across adapters**: there is no sound way to map a PDH instance to a CUDA
  ordinal (NVML exposes no adapter LUID), so a model split over two GPUs reports its full total on
  each of its per-GPU rows. `used_bytes_source` says which source a number came from; do not sum
  `pdh` rows across GPUs. `/api/vram/holders` already aggregates per pid for this reason.
* **Where neither source answers** (containers, WSL, MIG) holders are listed with size unknown. The
  payload says so rather than reporting 0 as though it were a measurement.
* **Only orphans are ever killed.** `POST /api/vram/reclaim` and the startup sweep touch a process
  only when its executable is under *our* `engines/` directory **and** its parent process is gone.
  A llama-server belonging to a live process — another StudioForge instance, a test run, a
  hand-started server — is reported with its parent named and never killed automatically, however
  much it holds. Use the watchdog's `nuke_all_models` if you really want those gone.
* **An engine launched through a wrapper** (a profiler, a CUDA shim) whose process *name* is not
  `llama-server` is not matched by the sweep. The watchdog's argv-based matcher still finds it.

## One instance per data directory

A StudioForge process that starts background work takes an exclusive OS lock on
`<data_dir>/.instance.lock`. A second process pointed at the same data directory still starts and
still serves reads, but runs **no background work at all**: no download resume, no TTL eviction, no
auto-load, no orphan sweep. It logs one ERROR naming the holder's pid at startup and reports
`"instance": "secondary"` plus `instance_holder_pid` from `/health` — check that field before
concluding that a healthy-looking server is doing nothing.

This exists because two instances sharing a data directory corrupted a 19 GB download on 2026-08-18
(DECISIONS.md D24). Limits:

* **The lock covers a data directory, not a model library.** Two instances with *different*
  `data_dir` values pointing at one `models.dir` are still two writers. The per-file `.part` lock
  turns that into a clean refusal rather than a corrupt file, but nothing prevents the setup.
* **A crashed holder is taken over automatically** — the kernel releases the lock however the
  process dies — so there is no stale-lock cleanup step and no `--force` flag to remember.
* **On a filesystem with no working locks** (some network shares) acquisition fails soft: a warning,
  and the old unguarded behaviour.
* **On POSIX the lock is advisory.** It binds cooperating processes; a third-party tool writing into
  our files is not stopped by it.

## Downloads: what "verified" actually means

A `.part` file is owned by exactly one transfer, under an exclusive OS lock held from the first byte
through verification to the rename. A second writer is refused with a message naming the cause,
never joined. Before publishing, the file is fsynced and its **on-disk size** is compared against
both the streamed byte count and the size the repository declared; the streamed sha256 is checked on
top when HuggingFace published one.

* **A published sha256 is not always available.** Many GGUF repos publish none, in which case the
  guarantee is size-only and the log says `length_verified_only` rather than implying more.
* **A file already at the destination is adopted on size, and on checksum only below 2 GiB.**
  Re-reading a 20 GB file on every enqueue of a model you already have costs minutes of disk
  bandwidth for a check the size comparison almost always subsumes. Above that ceiling the log line
  is `adopted_by_size_only`.
* **A file at the destination that fails those checks is renamed to `<name>.corrupt-<timestamp>`,
  never deleted.** It leaves the registry's view (the scanner only looks at `.gguf`) and stays on
  disk for you to inspect or remove. If a loaded model has the file open, the download is refused
  instead — unload it first.
* **Transient failures are retried, definite ones are not.** Five attempts, 2 s to 60 s with jitter,
  each resuming from the partial file: transport errors, 5xx, 429 (honouring `Retry-After`) and a
  momentary OS-level block on the `.part`. A 404, 401/403, a size or checksum mismatch, a full disk
  and a locked `.part` fail immediately. The queue shows `retrying in Ns (attempt k/5)` while it
  waits, and on failure says whether Resume continues from the partial or starts over.

## Pre-download fit estimates are cruder still

Before a file is on disk there is no GGUF metadata, so the KV portion of a fit verdict is a bounded
allowance rather than a calculation. The verdict says so (`approximate: true`). If another quant of
the same model is already in the registry, its metadata is used and the estimate becomes exact.

## Multi-GPU splitting is proportional, not measured

`tensor_split` is derived from usable free VRAM per device. It does not model interconnect
bandwidth, and on this class of hardware (PCIe, no NVLink) a split model is meaningfully slower
than a single-GPU one — which is why the planner exhausts every single-GPU option first.

Mixing GPU generations in one split works but runs at the slower card's pace.

## LoRA

* Adapters must be **GGUF** adapters. PEFT `.bin`/`.safetensors` adapters are not loadable by
  llama.cpp and are ignored by the scanner.
* Adapter/base compatibility is checked by architecture metadata and warned on, not enforced —
  llama.cpp will happily load a mismatched adapter and produce nonsense.
* Two virtual models over the same base share one `llama-server` instance **only when they differ
  purely in request-time fields** (system prompt / sampler-default presets). Any launch-time delta
  — an adapter set, a ctx/kv override — still costs a dedicated instance, because it changes the
  child's argv.
* Quality of a LoRA over a heavily quantized base (Q4 and below) is often poor. That is a llama.cpp
  property, not a bug here.

## Speculative decoding

* Target and draft must share a tokenizer. Vocab size is checked and a mismatch is refused; a
  matching vocab with a different tokenizer only warns.
* Drafting can be **slower** than not drafting. Use the Test action's A/B, which reports the
  acceptance rate — below roughly 60% the draft is usually costing more than it saves.
* `llama-server` reports drafting only via the per-slot `speculative` flag and a completion's
  `draft_n`/`draft_n_accepted`. `/props` reports `speculative.types: "none"` even while actively
  drafting; do not trust that field.

## Reasoning / thinking models

Default is `--reasoning-format none`, which keeps thoughts inline in `content` as `<think>` tags.
That guarantees a non-empty reply, but it means:

* **Thoughts are part of `content`.** If your client does not strip `<think>` blocks, they will be
  visible. Set `reasoning_format: deepseek` per model if you would rather have them separated —
  but then `content` can be empty, and the gateway's merge safety net will fold reasoning back in.
* **Stop sequences match thinking text too.** A reasoning model that narrates a format inside its
  thoughts will trip a stop sequence on it. Verified previously: `]]` as a stop sequence produced
  an empty completion on gemma4-31b. Prefer `max_tokens` over exotic stop strings with these models.

## Vision

* Images are re-encoded to PNG and downscaled to `gateway.max_image_dim` (default 2048) before
  being forwarded. Very large images therefore lose detail.
* `--cache-reuse` is **disabled by llama.cpp itself** for multimodal models — it logs
  `cache_reuse is not supported by multimodal, it will be disabled`. Vision models get no
  prompt-cache benefit.
* Image token cost is read from mmproj metadata where present, otherwise a documented default of
  1024 tokens/image. That is a context-budget estimate, not a VRAM one.

## Engine

* Pinned to llama.cpp `b10425`. Newer releases rename flags (that release removed the entire
  `--draft*` family), so upgrading is deliberate, per-model pinnable, and gated on a smoke test.
* `engines/active.json` is what every load actually uses; `engine.pinned_tag` in `config.yaml` is
  only a *preference* consulted when active.json is missing or names an uninstalled engine. When the
  two disagree a WARNING is logged at startup — they disagreed silently for weeks here, so config
  said `b10441` while every child ran `b10425`.
* `--defrag-thold` is deprecated upstream and is no longer emitted. The per-model `defrag_thold`
  setting still loads and saves, but it does nothing.
* The source-build fallback is implemented and its command construction is tested, but a full CUDA
  compile has not been exercised end-to-end here — the official binary supports every GPU on the
  reference machine, so the fallback has never been needed.

## Platform

* Linux is primary, Windows is fully supported and is what the reference rig runs. macOS is not
  supported (no CUDA).
* On Windows, `current` is a `current.txt` pointer file rather than a symlink, because symlinks
  need administrator rights or Developer Mode.
* **"Children die with the server" is a Windows guarantee only.** Each `llama-server` child is put
  in a job object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, so the kernel kills it whatever
  happens to us. On Linux a `kill -9` of the gateway leaves the children running (the `atexit`
  handler does not run) until the next startup sweep reclaims them. If the job cannot be created or
  a child cannot be assigned to it — no pywin32, or an existing job that refuses nesting — a WARNING
  is logged once and loads continue *without* the net.
* Registering the `lmstudio://` handler rebinds `HKCU\Software\Classes\lmstudio`. If LM Studio
  updates or re-registers itself it may silently take the scheme back. Nothing re-asserts it; the
  Server tab shows the true current state.

## Ports

The default gateway port is `1234`, which LM Studio also uses. Both can share the same model
library on disk, but not the same port — quit one or change `server.port`.

## Not implemented

* No Ollama-compatible (`/api/generate`, `/api/tags`) or KoboldCpp-compatible endpoints. See
  [`COMPARISON.md`](COMPARISON.md) for what was considered and why.
* No embedded model-quantization tooling — download the quant you want.
* No multi-node / distributed inference (llama.cpp RPC exists but is not wired up).
* No auth beyond a single shared API key. No per-user accounts, no rate limiting.
