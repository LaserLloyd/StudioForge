# Architectural decisions

Running log of decisions made while building StudioForge, with the reasoning that produced them.
Newest sections appended as the build progresses.

---

## D1. Inference is llama.cpp's `llama-server`, one child process per loaded model

**Decision:** never implement inference. Each loaded model is a supervised `llama-server` child on
an internal port (`18100–18200`); the gateway reverse-proxies to it.

**Why:** llama.cpp already implements OpenAI-compatible chat completions, SSE streaming, tool
calling, `json_schema` grammars, multimodal projectors, LoRA hot-scaling, and speculative
decoding. Reimplementing any of that would be strictly worse. One process per model (rather than
one process hosting many) gives per-model crash isolation, per-model engine version pinning, and
`SIGKILL` as a guaranteed VRAM-reclaim path — all three are hard requirements here.

---

## D2. Engine pinned to llama.cpp `b10425`, verified against the real binary

**Decision:** pin `b10425` and download the official `llama-b10425-bin-win-cuda-13.3-x64` build
(Linux: the matching `ubuntu-cuda` asset). Update path documented in [D3](#d3).

**Why:** the box has 2×RTX 5090 (Blackwell, sm_120) + 2×RTX 3090 (Ampere, sm_86). The prebuilt
CUDA 13.3 binary was verified to enumerate all four devices:

```
CUDA0: NVIDIA GeForce RTX 5090 (32606 MiB, 30991 MiB free)
CUDA1: NVIDIA GeForce RTX 5090 (32606 MiB, 30991 MiB free)
CUDA2: NVIDIA GeForce RTX 3090 (24575 MiB, 23332 MiB free)
CUDA3: NVIDIA GeForce RTX 3090 (24575 MiB, 23332 MiB free)
```

so sm_120 is covered by the official binary and the source-build fallback is genuinely a
*fallback*, not the primary path. It is still implemented, because a future arch (or a driver too
old for the newest CUDA build) will need it.

**Critical finding — flag names must be read from the binary, not from memory.** `b10425` renamed
the entire speculative-decoding surface:

| Old spelling | `b10425` | Status |
| --- | --- | --- |
| `--draft` / `--draft-n` / `--draft-max` | `--spec-draft-n-max` | **removed** — accepted then ignored |
| `--draft-min` / `--draft-n-min` | `--spec-draft-n-min` | **removed** — accepted then ignored |
| `--cache-type-k-draft` | `--spec-draft-type-k` | still a live alias |
| `--n-gpu-layers-draft` | `--spec-draft-ngl` | still a live alias |
| `--model-draft` / `-md` | `--spec-draft-model` | still a live alias |
| *(n/a)* | `--spec-type {none,draft-simple,draft-eagle3,ngram-*,...}` | **required** to enable drafting at all |

Only the `--draft*` spellings are genuinely gone; the rest are current aliases, so the flag
validator deliberately accepts them — rejecting a live alias would be a false positive.

Passing the old names does not fail loudly; `llama-server` prints "the argument has been removed"
and continues, which would look like speculative decoding silently doing nothing. Two consequences
for the design:

1. The expert-tier "extra flags" box is validated against the **pinned engine's own `--help`
   output** at save time, so an unknown or removed flag is a save-time error rather than a mystery
   at load time.
2. `--flash-attn` now takes a value (`on|off|auto`, default `auto`), and `--jinja` is enabled by
   default. Both are modelled as tri-state/explicit rather than bare booleans.

---

## D3. Engines are versioned artifacts under `engines/<tag>/`, never "whatever is on PATH"

**Decision:** each engine build lives in `<data_dir>/engines/<tag>/`, is smoke-tested before being
marked active, and keeps `engine.keep_versions` previous versions. There is a global default tag
plus an optional **per-model pin**.

**Why:** new llama.cpp releases periodically break older GGUFs or rename flags (see D2 —
`b10425` did exactly this). Without a per-model pin, one engine update can break one model and the
only recovery is downgrading globally. Updating the engine deliberately does **not** touch running
instances; new loads pick up the new engine and the GUI offers an explicit "reload on new engine".

**Update procedure:** bump `engine.pinned_tag` in `config.yaml` (or use the Server tab / `sfctl
update engine`), which fetches, extracts, smoke-tests, and activates. Rollback = set the tag back;
the previous directory is still there.

---

## D4. `--ctx-size` is the **total** KV budget across slots, not per-slot

**Decision:** the planner treats `ctx_size` as the per-slot context the user asked for and
launches `llama-server` with `--ctx-size (ctx_size * parallel)`, sizing the KV estimate off that
total. `models.default_parallel` is **1**.

**Why:** verified against the running binary. A load with `--ctx-size 4096` and no `--parallel`
reported `total_slots: 4` and `n_ctx: 4096` in `/props` — i.e. llama.cpp divides the context among
slots, giving 1024 tokens each. Two bugs follow if this is modelled wrong: a planner that
multiplies KV by slots over-estimates by 4× and rejects loads that fit, and a user who asks for
"8192 context" silently gets 2048 per conversation. Defaulting `parallel` to 1 means the number the
user typed is the context they get; raising it is an explicit advanced choice that visibly raises
the VRAM estimate.

---

## D5. Errors are OpenAI-shaped, with diagnostics in an additive namespace

**Decision:** every API error renders as `{"error": {"message", "type", "code", "param"}}`. Extra
diagnostics (planner numbers, suggestions, `llama-server` stderr tail) go under
`error.studioforge`.

**Why:** the `openai` client parses that envelope to build its exception hierarchy, so the four
keys must stay exactly where it looks for them; anything extra is ignored by the client but
available to our own GUI/CLI. A failed load therefore returns a clear `502` whose message contains
the stderr tail — never a hang.

Status code choices: `404` unknown model, `400` bad request (including "image sent to a non-vision
model"), `502` load failure / upstream death, `503` model busy or evict-refused, **`507`
insufficient VRAM** (closest standard code to "no room", and distinguishable from a transient
`503`).

---

## D6. Secret redaction is a log processor, not a call-site discipline

**Decision:** the API key and HF token are registered as secret *values* at config load; a
structlog processor scrubs them from every event, including inside strings and nested dicts.

**Why:** config objects and `llama-server` launch command lines get logged wholesale in several
places. A missed call site is a leaked token; a missed processor is impossible.

---

## D7. Data directory lives outside the code tree

**Decision:** `config.yaml`, `registry.sqlite3`, `engines/`, `logs/`, `downloads/` all live in
`<data_dir>` (`%LOCALAPPDATA%\studioforge` on Windows, `~/.local/share/studioforge` on Linux),
overridable via `SF_DATA_DIR`.

**Why:** app self-update installs into `releases/vX.Y.Z/` and flips a `current` symlink. Anything
inside the release directory is destroyed by an update. Config and registry must survive it, so
they are never under the code tree.

---

## D8. Gateway defaults to port 1234

**Decision:** the OpenAI-compatible API listens on `1234` by default; the GUI is a separate port
(`8080`), the watchdog another (`1235`).

**Why:** `1234` is LM Studio's default, so migrating OpenClaw is a *host* change rather than a
host-and-port change — which is the stated acceptance criterion ("works by only changing its base
URL"). Config validation rejects collisions between these three ports and the child port range,
because that class of mistake otherwise surfaces as a confusing bind error at startup.

---

## D9. FP4 quants *prefer* Blackwell; the 3090s are never excluded

**Question raised:** should NVFP4 models be restricted to the Blackwell cards, possibly via a
separate backend?

**Measured answer — no restriction is warranted.** NVFP4 loads and generates correctly on an RTX
3090 (sm_86): verified by loading the real 17.99 GiB `gemma-4-Ortenzya-...-NVFP4.gguf` on `CUDA2`,
healthy in 12s, coherent output. llama.cpp's CUDA backend handles the format on Ampere; it is not
a Blackwell-only file format, and no separate backend is needed.

It is, however, worth *steering*. `llama-bench` (b10425, gemma-4-31B, `-p 512 -n 128 -r 2`):

| Quant | pp512 sm_120 (5090) | pp512 sm_86 (3090) | Blackwell speedup |
| --- | --- | --- | --- |
| **NVFP4** | **5991.6 tok/s** | 987.4 tok/s | **6.07x** |
| Q4_0 | 4108.8 tok/s | 1367.2 tok/s | 3.01x |

| Quant | tg128 sm_120 | tg128 sm_86 |
| --- | --- | --- |
| NVFP4 | 68.8 tok/s | 35.1 tok/s |
| Q4_0 | 74.3 tok/s | 40.3 tok/s |

Two things fall out. NVFP4 gets roughly **double** the relative Blackwell benefit that a plain Q4_0
does (6.07x vs 3.01x), which is native FP4 tensor-core acceleration showing up. And on Ampere,
NVFP4 is **~28% slower than Q4_0** at prompt processing (987 vs 1367) — so an FP4 file on a 3090 is
a worse trade than an ordinary Q4 would have been. Token generation is bandwidth-bound and tracks
file size instead (NVFP4 is the larger file here), so the FP4 win is a prefill win, not a
generation win.

**Decision:** `planner.quant_affinity` maps a quant family to a minimum compute capability plus a
mode. Defaults ship as `NVFP4: {min_compute_capability: "12.0", mode: prefer}` and the same for
`MXFP4`.

* `prefer` (default) — capable GPUs are tried first; if none has room the model still loads on the
  3090s, with a note on the plan explaining the expected slowdown. **The 3090s are fully usable for
  every model, FP4 included.**
* `require` — only capable GPUs are offered, and a rejection explains which knob to change.

A per-model `device_override` outranks affinity entirely: an explicit choice is the user's.

This lives in the planner rather than the engine because it is a *placement* question. A separate
backend would only be justified if the format did not run at all on Ampere, and it does.

---

## D10. `json_object` is upgraded to a grammar-backed schema

**Decision:** a request with `response_format: {"type": "json_object"}` is rewritten to
`{"type": "json_schema", "json_schema": {"name": "json_object", "schema": {"type": "object"}}}`
before it reaches llama-server.

**Why:** llama.cpp only compiles a grammar for `json_schema`; it treats `json_object` as a hint.
The contract test caught the consequence — `Qwen2.5-0.5B` answered with a markdown-fenced
```` ```json ```` block, which `json.loads` rejects. OpenAI's contract is that `json_object`
*guarantees* parseable JSON, and clients rely on that, so the gateway closes the gap rather than
passing an unenforced hint through. Callers see the OpenAI-standard field on the wire either way.

---

## D11. `--fit off` on every launch: the engine must not re-plan around us

**Decision:** every `llama-server` launch passes `--fit off`, and `--fit` / `--fit-target` /
`--fit-ctx` / `--n-gpu-layers` are all manager-owned flags that the expert "extra flags" box
refuses.

**Why:** `b10425` introduced `-fit, --fit [on|off]` — *"whether to adjust unset arguments to fit in
device memory"* — **defaulting to on**, alongside `--n-gpu-layers` defaulting to `auto`. Because
StudioForge always pins `--n-gpu-layers 999` and an explicit `--ctx-size`, there is nothing left
for `--fit` to adjust today. But the combination `-ngl auto` + `--fit on` is precisely a silent
partial-offload path, and this project's central promise is that a model either runs fully in VRAM
or is rejected with numbers. Leaving an engine-side autofit enabled would mean a future flag change,
or one stray entry in the extra-flags box, could quietly turn a rejection into a degraded load.

Turning it off also keeps the planner authoritative: predicted-vs-actual VRAM logging is only
meaningful if the engine did not silently change context or offload behind our back. A genuine
over-commit now fails loudly, which is the correct outcome — the planner should have caught it, and
if it did not, we want to see that in the calibration data rather than hide it.

Verified accepted by the real binary (`--fit off` + `-ngl 999` + `--ctx-size 2048` loads healthy,
`/props` reports `n_ctx: 2048`).

---

## D12. `--reasoning-format none` by default: thinking models must not return empty replies

**Decision:** every launch passes `--reasoning-format none` unless overridden per model
(`models.default_reasoning_format`, plus per-model `reasoning_format` / `reasoning` /
`reasoning_budget` in the advanced settings tier).

**Why — measured, not assumed.** llama.cpp defaults `--reasoning-format` to `auto`, which for a
reasoning model puts the thoughts in `message.reasoning_content` and leaves `message.content`
**empty**. Verified against the real `DeepSeek-R1-0528-Qwen3-8B-Q4_K_M` on b10425, same prompt,
same everything except the flag:

| `--reasoning-format` | `content` length | `reasoning_content` length | client sees |
| --- | --- | --- | --- |
| *(default `auto`)* | **0** | 316 | **an empty reply** |
| `none` | 323 | 0 | the text |

`message.reasoning_content` is not part of the OpenAI chat-completions schema, so a standard
client — OpenClaw included — reads `choices[0].message.content`, finds `""`, and concludes the
model said nothing. On a reasoning model with a modest `max_tokens` that is the *normal* case, not
an edge case, because the budget is spent thinking.

`none` also keeps streaming honest: thoughts arrive as ordinary content deltas, so the SSE stream
still passes through byte-for-byte (D1) instead of needing the gateway to rewrite chunks.

Clients that *want* structured reasoning can opt back in per model by setting `reasoning_format`
to `deepseek`. The default favours "never silently returns nothing", which is the behaviour the
compatibility requirement actually depends on.

---

## D13. Virtual-model presets are applied per request, never per launch

**Decision:** a virtual model may carry a *preset* — a system prompt plus sampler defaults
(`temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `max_tokens`). The gateway applies the
preset to each request payload; none of it ever reaches the `llama-server` argv. A virtual model
whose settings are otherwise default therefore **shares its base model's instance** — the manager
routes its requests to the base's child. Any launch-time delta (an adapter set, a `ctx_size` /
`kv_cache_type` override, anything in `ModelSettings`) still costs a dedicated instance, because it
changes the child's command line.

**Why per-request rather than per-launch:** `llama-server` *does* accept `--temp`/`--top-p`/... as
launch flags, and per-model settings already use them. But baking a preset into the launch would
make every persona a separate multi-GiB load — ten personas over one 30B base would need ten
children — which is exactly backwards for the OpenClaw use case (many named personas, one resident
model). Applied per request, N presets cost zero extra VRAM. Verified against the real engine: two
preset virtual models over `Qwen2.5-0.5B` chat successfully while `/api/status` shows exactly one
resident instance, keyed by the base id.

**Precedence rules, chosen to never surprise a client:**

* The preset's system prompt is **prepended**; a client's own system message is kept after it.
  Silently replacing a client's instructions would make identical requests behave differently
  against base vs. preset in a way the client cannot see or debug.
* Sampler defaults fill only *absent* request fields, with the alias spellings
  (`repetition_penalty`, `max_completion_tokens`) counting as present — an explicit client value
  always wins. Verified on the wire: a preset with `max_tokens: 2` yields
  `usage.completion_tokens == 2` when the request omits the field and `12` when the request says
  `max_tokens: 12`.

**Persistence:** inside the existing `virtual_models.adapters_json` column (a
`{"adapters": [...], "preset": {...}}` object when a preset exists, the legacy plain list
otherwise), so no schema migration and a rollback build still reads its own rows.

## D14 — Context is a ladder, not a constant

**Problem.** OpenClaw ran out of context. Every load without an explicit
`ctx_size` got `models.default_ctx` (8192 on this rig, chosen from the smallest
GPU); only *thinking* models were tried at a larger size, falling straight back
to 8192 if it did not fit. Agent workloads carry long tool transcripts and 8k is
nowhere near enough.

**Decision.** `models.target_ctx` (default 131072) is what every model *aims*
for. The planner walks a halving ladder from the aim down to `models.default_ctx`
and takes the largest rung that fits, clamped to the model's trained window
(`n_ctx_train`) because going past it needs RoPE scaling and silently degrades
quality.

**Invariants kept.**
- An explicit `ctx_size` — per-model or per-request — is honoured exactly. Asking
  for 4096 gives 4096, never a helpful upgrade.
- Every rung above the floor is tried with eviction DISABLED: a roomier window is
  a nicety and must never be the reason someone else's model is unloaded. Only
  the floor may evict, and only if policy allows.
- Therefore aiming high can never turn a load that would have worked into a
  rejection.

**Measured on this rig** (2× 5090 + 2× 3090): the 26B MoE and small models reach
131072; a 31B dense placed on one card steps down to 65536. With other models
resident the ladder steps down further, which is the point.

`models.default_ctx` is now the FLOOR that hardware tuning sets, not the target.

## D15 — KV cache is sized per layer, because half the library is iSWA

**Problem.** The planner sized every layer at full context. Gemma 3/4 use interleaved
sliding-window attention: only every 6th layer holds the full context; the rest keep a
1024-token window and use half the head dimension. The estimate therefore claimed
`gemma-4-31B` needed **480 GiB** of KV at 262144 and capped it at 65536. The server's
own calibration log had been recording the discrepancy for weeks
(`predicted_mb=95615 actual_mb=40037`).

**Decision.** `estimate_kv_bytes_iswa` sums per layer, reading `attention.sliding_window`,
`sliding_window_pattern`, `key_length_swa`/`value_length_swa` and the per-layer
`head_count_kv` array from the GGUF. Non-iSWA models keep the uniform estimate.

**The SWA depth must mirror llama.cpp exactly**, not approximate it:
`GGML_PAD(min(full_ctx, n_swa * n_seq_max + n_ubatch), 256)`. A flat 1.25 multiplier was
wrong in the dangerous direction — 1280 cells against a real 1536 at one slot, and 3.6x
under at four. Over-estimating merely refuses a context that would have fit;
under-estimating is an OOM at load.

**Measured.** `gemma-4-31B-it-QAT` at 262144: predicted vs real 38 GiB across two 5090s,
engine reporting `n_ctx=262144`, 3090s idle. 65536 -> 262144, a 4x unlock, and the whole
Gemma-4 fleet with it.

**Related:** the auto KV ladder is `f16 -> q8_0 -> q4_0` only. `q5_1`/`q5_0`/`q4_1` are
excluded because the prebuilt CUDA engines are not built with `GGML_CUDA_FA_ALL_QUANTS`;
with flash attention on, llama.cpp would silently run attention on the CPU — breaking the
GPU-only promise without failing loudly.

## D16 — Once eviction is decided, re-plan the whole ladder

**Problem.** D14 made context a ladder and gave it one hard invariant: rungs above the
floor are tried with eviction disabled, so a roomier window can never be the reason
someone else's model is unloaded. The implementation drew the wrong conclusion from that.
It walked the ladder without eviction, then allowed eviction **at the floor** — so the
decision to evict and the choice of context were made by the same rung.

Measured, 12:03 on this rig. Available 79,832 MB, reclaimable 19,423 MB from one idle
model. Together that is 99,255 MB, which fits:

| Context / KV | Needs | Verdict |
| --- | --- | --- |
| 262144 / q4_0 | 96,004 MB | fits |
| 65536 / f16 | 95,236 MB | fits |
| **8192 / f16** | **89,860 MB** | **what it actually loaded** |

The model paid the full price of the eviction and got the smallest window on the ladder.
Twenty-six minutes earlier the same code did the mirror image: a 27B landed at 8192 on one
card after 92.7 GB had been freed.

**Decision.** Split the two questions the floor rung was answering at once.

1. **Pass 1** walks the whole ladder — floor included — with eviction disabled. If
   anything fits, it loads and nothing is evicted. D14's invariant is *strengthened* here:
   previously the floor rung could evict during the first walk; now no rung can.
2. Eviction is decided only when pass 1 fails at every rung, i.e. not even the floor fits
   in free VRAM. That is the same trigger as before.
3. **Pass 2** re-walks the same ladder against `available + reclaimable` and takes the
   highest rung that fits. Having already paid for the eviction, taking 8192 when 65536
   costs exactly the same is simply worse.

Logged at INFO as `evicting X frees N MB -> re-planned ctx=... kv=...`.

**Invariants kept.** An explicit `ctx_size` is a one-rung ladder, so both passes honour it
verbatim. Aiming high still cannot turn a load that would have worked into a rejection —
pass 1 is a strict superset of what the old first walk tried. Nothing is evicted that
pass 1 could have avoided.

**Cost.** A failing load walks the ladder twice. The rung rejections are pure arithmetic;
the expensive part — the NVML per-process enumeration behind `vram_holders` — is computed
only for the one refusal that reaches the caller. That is also what fixed the
`load rejected` INFO spam: fifteen lines per ordinary walk, none of them the outcome.
There is now exactly one `load planned` (or one `load rejected`) INFO line per plan,
carrying the rungs tried and the choice made.

## D17 — Parallel slots are estimated, not fixed at 1

**Problem.** D4 established that `--ctx-size` is the total across slots and set
`models.default_parallel: 1`, which was right at the time: it made "the number you typed
is the context you get" true. It also froze concurrency at one conversation for every
model on a 113 GB rig. llama.cpp's own default is `-np -1` (auto), which D4 measured as 4
slots. OpenClaw runs several agents at once, and each queued behind the others.

Nothing in the memory model needed to change — `estimate()` already sizes KV off
`ctx_size * parallel`, `max_ctx_for_budget` already divides by slots, and the iSWA path is
already parallel-aware. What was missing was a *policy* for choosing the number.

**Decision.** `models.default_parallel` accepts `"auto"` (the new code default), and the
planner sizes the slot count per model **and per candidate device set**. Two independent
bounds; the smaller wins:

```
kv_bytes_per_token = n_layer * n_head_kv * (head_dim_k*bpe(kv_k) + head_dim_v*bpe(kv_v))

by_vram  = kv_budget_bytes // (ctx_per_slot * kv_bytes_per_token)

ctx_used = ctx_per_slot * CTX_FILL_FRACTION            # 0.5
by_knee  = active_weight_bytes / (ctx_used * kv_bytes_per_token)
           * (0.5 if MoE else 1.0)                     # MOE_KNEE_DERATE

max_parallel = clamp(min(by_vram, round(by_knee)), 1, MAX_PARALLEL_CAP=8)
```

`by_vram` is capacity: the KV cache is the one term that scales with slots. `by_knee` is
bandwidth: decode reads the active weights once per step whatever the batch size, plus
every busy slot's KV. Once KV traffic matches weight traffic, another slot buys no
throughput and costs VRAM and latency. `active_weight_bytes` is all the weights for a
dense model and `weights * n_expert_used / n_expert` for a MoE; the MoE knee is derated by
half because experts fan out with batch size instead of staying flat. Both the fill
fraction and the derate are deliberate approximations in the safe direction (fewer slots),
not measurements — the real curves want a benchmark this rig has not run.

Worked, on the reference rig:

| Model | KV/token (f16) | Placement | Slots | Bound |
| --- | --- | --- | --- | --- |
| Qwen3.5-122B-A10B (48L, 2 KV heads, head_dim 256) | 96 KiB | 4 GPUs, 32k/q8_0 | 4 | knee |
| Qwen3.8-27B (64L, 8 KV heads, head_dim 128) | 256 KiB | one 5090, 32k/f16 | 1 | vram |
| Qwen3.8-27B | 256 KiB | two 5090s, 32k/f16 | 4 | knee |
| 8B dense (36L, 8 KV heads, head_dim 128) | 144 KiB | one 5090, 32k/f16 | 4 | vram |

**Explicit always wins, verbatim.** A `parallel` from the request, from per-model
settings, or an integer `default_parallel` switches the estimator off entirely and is
never capped — D14's "explicit value is honoured" invariant, extended to concurrency. The
estimator also *starts* at one slot and verifies its own answer by re-estimating at the
chosen count and stepping down until it really fits (the analytic bound assumes a uniform
per-token cost, which an iSWA model does not obey). So auto can never turn a load that one
slot would have allowed into a rejection.

**Placement interacts with concurrency.** The estimator runs inside the per-device-set
planning, so a split can be judged on the slots it buys. `prefer_single_gpu` still holds
except when all three of these are true: the single-GPU placement is starved at exactly
one slot, the split at least doubles it, and every added device is at least as capable as
the chosen one. The 27B row above is that case — 8 GiB of KV per 32k slot means one 5090
serves one conversation. The capability condition matters because a split runs at its
slowest member's pace, so buying slots by dragging a 5090-resident model onto a 3090
trades latency everyone feels for throughput only a batch would notice.

**Launch flags that follow from the slot count** — emitted from the plan, not from stored
settings, so they track whatever was decided:

* `--batch-size 4096` above 4 slots. The default 2048 logical batch is *shared*, so many
  slots ingesting prompts serialise behind it. Only when no explicit `batch_size` is set:
  llama.cpp takes the last occurrence of a flag, so appending would otherwise silently
  overrule the user.
* `--slot-prompt-similarity 0.3` above 1 slot. The 0.10 default scatters an agent's
  near-identical prompts across slots and loses the `--cache-reuse 256` prefix each time.
* `--ubatch-size` stays at its 512 default: that one is a VRAM term the planner models.

**`--kv-unified` is off, and stays off until it is measured.** llama.cpp enables it only
when the slot count is auto, and we always pass an explicit `--parallel`, so it is off in
practice today. Turning it on costs no extra VRAM and lets one request use the whole KV
pool instead of its slice — plausibly better for a mixed agent workload. It is exposed as
a per-model `kv_unified` opt-in and **is not recommended anywhere yet**: the standard in
this document is that a flag's behaviour is verified against a real load before it is
advised, and `-kvu` has not been. Confirm with `/props` and a long single request against
a multi-slot load before changing that.

**Also dropped:** `--defrag-thold` is deprecated in b10425 and is no longer emitted. The
`defrag_thold` setting still loads — stored rows and the GUI settings form both read it —
but is inert. Passing a deprecated flag to make a saved value look honoured is how a
setting quietly stops meaning anything.

**Not changed:** `data/config.yaml` still says `default_parallel: 1`. Flipping a live
service that is serving OpenClaw from one slot to auto is a behaviour change that wants a
deliberate moment and a look at predicted-vs-actual afterwards (`--ctx-checkpoints` is 32
*per slot* and is not modelled). The code default is `"auto"`; changing that one yaml key
is the switch.

## D18 — Calibration measures our own child, and the loop is closed

**Problem.** Two halves of the calibration story were broken, and each hid the other.

`_record_actual_vram` summed **whole-device** `used_bytes` across the plan's devices. On a
box that also runs a desktop, a browser and ComfyUI, that number is mostly other people's
memory. Over 540 recorded rows the median actual/predicted ratio is **2.97** and p90 is
**12.0**, against the 0.81–1.23 this project's own LIMITATIONS.md claimed.

And `suggest_overhead_fraction()` — the function that turns that history into a tuned
factor — was never called from `src/` at all, only from tests. The documentation said the
factor self-tunes. It did not.

Wiring the second half to the first would have been actively harmful: contaminated rows
fed to the calibrator peg `compute_overhead_fraction` at its ceiling and start refusing
loads that fit.

**Decision.** Fix the measurement, mark the fixed rows, and only then close the loop.

* `actual_bytes` is now **our child's own VRAM, attributed per pid** via the existing
  `vram_processes` helper, summed over the plan's devices. When NVML cannot enumerate
  per-process usage (containers, WSL, MIG) the observation is *skipped* rather than
  falling back to the device total — no data beats data that means something else.
* Rows measured this way carry `note = "per_pid"`. That marker is the only thing
  separating them from the contaminated history, so calibration reads nothing else.
* At startup (once, not per load) the manager reads the last 200 observations, keeps the
  clean ones, and applies `suggest_overhead_fraction()` **clamped to [0.03, 0.15]**, with
  a minimum of 5 clean rows and an INFO line naming the old and new values. Below the
  floor the compute term stops covering real graph buffers; above the ceiling it eats
  enough VRAM to refuse loads that fit. A calibration loop with no clamp is a way to make
  the planner slowly wrong with nobody noticing, so the clamp is the point.
* The tuned value lives in memory only; `config.yaml` is never rewritten. A calibration
  that turns out badly is undone by a restart rather than by editing a file. Applying it
  once at startup rather than after every load also keeps the arithmetic stable for the
  lifetime of a process — a planner whose numbers shift under a running server is much
  harder to reason about than one that is wrong in a fixed way.

**Consequence.** `load_observations` rows written before this change are contaminated and
are ignored by the calibrator. See docs/LIMITATIONS.md.

## D19 — VRAM can be reserved for the neighbours

**Decision.** `planner.excluded_devices` (a list of CUDA indices the planner may never
place on) and `planner.reserved_mb` (index → MiB held back, subtracted inside
`usable_bytes()` alongside the headroom). Both default **empty**: reserving hardware is a
policy decision about a specific box, and a shipped default must not make one.

**Why.** This rig runs ComfyUI on the same four GPUs. `headroom_fraction` could not
express "leave CUDA3 alone" — it is a percentage of *every* card, so reserving enough for
a neighbour on one card starved all four. Without a knob, the only lever was
`device_override` on every model, which is a per-model answer to a per-machine question.

**Precedence.** A per-model `device_override` naming an excluded device **wins**, with a
WARNING and a plan note. Exclusion is our placement policy; the user naming a device is
the user deciding, and silently ignoring either half would hide a real contradiction.
`reserved_mb` is different and applies even to a forced placement: it describes memory the
*neighbour* needs, which does not stop being true because someone forced a placement.

Rejections name whichever of the two is holding memory back, so a refusal caused by our
own config says so instead of looking like missing hardware.

**Removed in passing:** `models.max_loaded`. Declared, never read by anything — a config
key that silently does nothing is worse than no key, because it reads as a working cap
while VRAM is the only real limit. An existing `config.yaml` carrying it still loads.

## D20 -- The catalog is a table of loadable rows, and its speed column is calibrated

**Problem.** An agent choosing a model had to answer four questions -- which model, what context,
how many slots, will it fit -- from four different places, and could not answer the fourth at all
without trying. `/v1/models` gave identity, `/profiles` gave a per-hardware-mode fit, D17 gave a
slot count nothing exposed, and speed was not modelled anywhere. Every one of those is a number the
agent must *have* before it calls `load_model`, so it guessed, and a guess that does not fit comes
back as a 507 several seconds later.

**Decision.** `core/catalog.py` builds one table: every model, **sorted by download date descending**
(the newest mtime across its GGUF shards -- a multi-part download finishes on its last file), each
with one row per context tier from `{16k ... 1M}` capped at `n_ctx_train`, plus any pinned
`ctx_size`. Each row carries `fits`, `devices`, `kv_cache_type`, `vram_mb`, `max_parallel` +
`parallel_limited_by`, estimated and measured tokens/second, a `confidence` word, and **`load_args`
-- the exact argument object `load_model` accepts**. An agent that has chosen a row is done
choosing.

Three properties the shape depends on:

* **The planner is the only authority on placement.** A row is one
  `plan_load(ctx_size=tier, allow_evict=False)` call, so exclusions (D19), reservations, quant
  affinity (D9), `device_override` and live free VRAM are respected by construction rather than
  re-implemented. `allow_evict=False` is what makes `fits` mean "loads now without disturbing
  anything".
* **One VRAM snapshot per build.** A catalog built against a live probe would query NVML once per
  model per tier, and an early row could disagree with a late one about the same card -- a table no
  single load matches. The snapshot also caps the cost: the per-process enumeration behind a
  rejection is expensive, and a catalog produces many rejections.
* **`if_gpus_idle` beside every row**, computed against total-minus-headroom. It separates
  "impossible on this hardware" from "possible once you unload something", which is the difference
  between an agent giving up and an agent calling `unload_model`.

`max_parallel` is recomputed here rather than read off `plan.max_parallel`, because the plan only
carries an estimated count when `models.default_parallel` is `"auto"`. This rig runs an explicit
`1`, under which every plan reports one slot and the concurrency column would collapse. "What could
this placement sustain" is well defined whatever the current policy is.

**The speed estimator.** `core/throughput.py`. Decode is bandwidth-bound and a layer split is
sequential, so the per-device times add:

```
t_token         = SUM_dev (active_bytes_dev / BW_dev) + SUM_dev (kv_bytes_dev / BW_dev)
gen_tps         = eff / t_token,    eff = 0.75 - 0.05 * (n_devices - 1)
t_step(N)       = t_weights + N * t_kv
gen_tps_batched = eff * N / t_step(N)          # saturates at D17's knee
prompt_tps      = 0.35 * SUM_dev (FLOPS_dev * share_dev) / (2 * active_params)
```

`active_bytes` is `planner.active_weight_bytes` -- the same definition D17 uses, so the bandwidth
term and the FLOP term cannot disagree about what "active" means.

**The GPU constants are NOMINAL VENDOR FIGURES, NOT MEASUREMENTS.** RTX 5090 `{1792 GB/s, 209
TFLOPS fp16}`, RTX 4090 `{1008, 165}`, RTX 3090 `{936, 71}`. They sit in one table with a
deliberately pessimistic unknown-GPU fallback. They are peak numbers no kernel reaches, and the gap
is not a constant -- it depends on the model's shape, the quantization and the split.

**Measured, against the resident model, and it found a bug worth naming.** The check was run
against the real `mradermacher/Qwen3.5-122B-A10B-heretic-v2.i1-Q5_K_M` on all four GPUs at ctx 8192
/ parallel 1, whose child reports 37.05 tok/s generation and 1149 tok/s prompt over 254,420 tokens.
The first estimate came back at **249 tok/s** -- 6.7x optimistic -- and the cause was not the GPU
table.

Its GGUF says `n_expert=256, n_expert_used=8`. `planner.active_weight_bytes` applies that 3.1%
routed share to the **whole file**, including attention and the output embedding, which are not
routed. So it charged 2.7 GB of active weights for a model whose own name says ~10B parameters are
active out of 122B:

| Active weights per token | Bytes | Estimate |
| --- | --- | --- |
| flat routed share (`planner.active_weight_bytes`) | 2.7 GB | 249 tok/s |
| dense trunk + routed experts (`throughput.active_params`) | 7.0 GB | 143 tok/s |
| what the model's name implies (~A10B) | ~7.2 GB | ~100 tok/s |
| **measured** | | **37.0 tok/s** |

**So the speed path does not reuse `planner.active_weight_bytes`.** It derives the dense trunk from
metadata the GGUF already carries -- `n_layer * n_embd * (n_head*head_k + n_head_kv*head_k +
n_head_kv*head_v + n_head*head_v)` for attention, plus `n_vocab * n_embd` for the output projection
-- charges that in full, and applies the routed share only to the remainder. The input embedding is
deliberately excluded: it is a row lookup, not a matmul, so it costs one row of bandwidth per token.

The planner's version is not wrong *for the planner*. It feeds D17's knee, where under-counting
active weights means proposing **fewer** slots, which cannot cause an OOM. In a speed estimate the
same under-count means advertising a model as 2.6x faster than it is, which is the unsafe direction.
Two callers, two correct answers; the difference is now documented at both call sites.

**What remains is real, and calibration owns it.** After the trunk fix:

| | Measured (`/metrics`) | Estimate | Ratio |
| --- | --- | --- | --- |
| Generation | **37.05 tok/s** | 143.5 tok/s | 0.26 |
| Prompt | **1149 tok/s** | 3752 tok/s | 0.31 |

Still ~4x optimistic, and that part *is* the nominal numbers: peak memory bandwidth is not reached
by a batch-1 MoE decode that gathers eight scattered expert tensors per layer, and a four-way
mixed-generation layer split pays PCIe synchronisation on every token. No constant is being invented
to paper over it -- the ratio is one data point on one model, and inventing a general derate from it
is exactly the mistake this document exists to prevent. Calibration learns it per model, per
placement, from measurement. `tests/unit/test_throughput.py` pins the ratio band so a formula change
that moves it fails visibly.

**Calibration policy.** Every child already launches with `--metrics`. The existing TTL sweeper
scrapes each ready child and records tokens/second **between two scrapes** into a new
`throughput_observations` table, at most once every two minutes per model.

* **Counters, not the `*_seconds` gauges.** A gauge averages over the child's whole lifetime and
  cannot distinguish "fast now" from "was fast an hour ago". Two samples of a counter give the rate
  between them.
* **The estimate is stored beside the measurement.** Calibration needs `measured / estimated`, and
  re-deriving the estimate later would need the per-device byte split the plan had at the time --
  gone once the model is unloaded. Storing both makes a row self-contained.
* **`gpu_class` is recorded, not derived from device indices**, because CUDA ordinals are not stable
  across driver updates or a card being moved.
* Factor = **median** over three tiers: this model on these devices -> this hardware class -> 1.0.
  The median, because one sample taken while ComfyUI was hammering the same card is an outlier and a
  mean would let it move every model on the box. Minimum two rows; one is noise.
* **Clamped to [0.1, 3.0].** A ratio outside that means the two numbers describe different things (a
  mis-parsed metric, a model swapped underneath its id), and applying it would make the catalog
  confidently wrong rather than roughly right. Same reasoning as D18's clamp.
* A window with no decodes, or one where a counter went backwards (the child restarted), records
  **nothing**. A zero would poison the median.

`confidence` is `"measured"` only when this exact placement and context were observed, so an agent
can read the word literally; `"calibrated"` when a factor was applied; `"estimated"` otherwise.

**Collection runs on the timer, not on request completion.** Hooking completion was the first choice
and was rejected: responses stream, so "complete" is several code paths (normal return, client
disconnect, upstream error), and each would pay an HTTP round-trip to the child *inside* the request
path to read counters that only mean something averaged over a window. On the sweeper it is one
localhost GET per loaded model per sweep and cannot slow a request down.

**The recommendation rule.** Exactly one row per model, so "pick the recommended one" always has an
answer: for a chat-class model the highest context that fits **and sustains at least 2 slots**,
otherwise the highest context that fits, otherwise the highest reachable with every GPU idle. One
slot means every concurrent request queues behind the one before it, which on an agent host is worse
than a smaller window.

**Also.** `Planner(log_plans=False)` -- the catalog runs one plan per model per tier per hardware
state, which at INFO would be several hundred lines describing loads nobody requested. D16 removed
exactly this class of spam and a new surface must not reintroduce it; the lines still go to DEBUG.
`/v1/models` gains vendor `ctx_per_slot` / `max_parallel` for loaded models (`--ctx-size` is the
total across slots, so `loaded_context_length` alone is ambiguous above one slot), and `/api/status`
gains `requests_deferred` per loaded model -- llama.cpp queues an overflow request rather than
refusing it (D17), so a server that looks healthy and feels slow shows it only in that counter.

---

## D21 -- One watchdog, adopted across restarts; the main process is the only thing that dies

**Problem.** `POST /api/restart/server` answered `{"restarting": true, "via": "watchdog"}` and did
not restart the server. Both of its paths were broken, and each hid the other.

*The handoff was refused.* `_ask_watchdog_to_restart` sent `Authorization: Bearer
<server.api_key>` and nothing else. The watchdog enforces auth when **either** `server.api_key` or
the MCP pairing PIN is set, and accepts either -- so on the default install (key null, PIN set,
which is this box) the app sent no credential at all and was answered `401` by the watchdog's ASGI
wrapper. That reply is generated *before any watchdog code runs*, which is why `watchdog.log` was
empty at the time and the failure read as silent. The `ERROR` line existed in `studioforge.log`, one
second before the drain that followed it.

*The fallback could not work, by construction.* Refused, the app fell back to `_self_restart`:
drain, `Updater._respawn_detached()`, wait, exit. The replacement runs `_preflight_ports` while the
old process still holds `server.port` and `gui.port`, and the old process's **watchdog child** still
holds `watchdog.port`. So the replacement exited `rc 3` -- `startup port conflict ports=[1234, 8080,
1235]` -- every single time, and the parent, finding its replacement already dead, correctly stayed
up. A restart-by-self-respawn could never succeed while the process it was replacing was alive.

*And the drain flag stuck.* `manager.stop()` latches `_draining` and cancels the TTL sweeper,
because it is written for "this process is going away". After a restart that did not happen,
`/health` reported `draining: true` on a server that went on serving and loading models for hours,
and nothing was ever evicted again.

**Decision.** There is **exactly one watchdog per config file, and it outlives the main process.**
The watchdog is the supervisor; supervisors do not get restarted by the thing they supervise.

* **The watchdog is adopted, never duplicated.** `ports.inspect_running_watchdog` asks whatever
  holds `watchdog.port` for `GET /health` and adopts it when the body is watchdog-shaped and its
  `config_path` is ours. `_preflight_ports` then drops that conflict and `_spawn_watchdog` returns
  `None` instead of starting a second one.
* **Identity comes from `/health`, not from `psutil`.** `/health` needs no credential and no
  elevation, and answers while the server it watches is down -- during a restart it reports `503`
  `down`, which is a *healthy watchdog* and must still be adoptable. Process inspection is the
  fallback for naming the pid in a log line, never the gate.
* **`kill_process_tree` takes an `exclude` set, and it excludes those pids only -- not their
  descendants.** The watchdog is normally a child of the main server, so `restart_server` kills a
  tree it is standing in; without the exclusion it killed itself mid-restart. Excluding descendants
  too would be the opposite bug: after one watchdog-driven restart the new server *is* the
  watchdog's child, and the next restart must still be able to kill it.
* **The self-respawn fallback gets a handshake instead of a race.** `_respawn_detached` passes
  `SF_RESPAWN_PARENT_PID` (and a wait budget derived from `server.drain_timeout_s`). A process that
  finds that variable set **waits** for its conflicting ports instead of refusing to start, and the
  parent sets `state.handing_over` so its exit path leaves the watchdog running for the replacement
  to adopt. The wait deliberately does not check *who* holds each port: identifying a holder needs
  `psutil.net_connections`, which is exactly the call that fails without elevation on this platform,
  and gating on it would turn "wait for my parent" into "fail because I am not an administrator".
  Without the variable, a busy port still fails instantly -- an operator's mistake must not sit in a
  45-second wait.

**Every branch logs, and the failure is readable from the API.** `POST /restart/server` has to
answer *before* it acts, so its 200 is a statement of intent. `GET /api/restart/status` says how the
last attempt went, and a `failed` outcome is echoed into `/health` as `restart_failed` -- a
non-`never` answer from a live server is itself the diagnosis, because a restart that worked takes
the process that would have answered with it. And a failed restart calls `manager.resume()`: a drain
flag that cannot come back down is worse than no flag, since it is the signal an operator reads to
decide whether killing the process is safe.

**Cost, accepted.** An adopted watchdog keeps running the code it started with, so an update that
changes the watchdog itself needs a full stop/start rather than a restart. `SF_ADOPT_WATCHDOG=0` is
the escape hatch: it makes a running watchdog a hard conflict again.

**Also.** The startup banner writes through `_console`, which swallows a dead stream. `print` raises
`ValueError: I/O operation on closed file` on a detached or closed console and took the whole
process with it, after startup had otherwise finished -- the last log line was "management MCP
mounted" and there was no traceback anywhere. (`sys.stdout is None` under `pythonw.exe` is currently
survivable -- CPython's `print` and click's `echo` both no-op -- but nothing should depend on that.)

---

## D23 -- VRAM cannot outlive its owner, and every holder is named

**Incident (2026-08-18).** ~10 GiB on GPU0 and ~15.6 GiB on GPU1 were unavailable with
"everything stopped". The holders were three `data\engines\b10425\llama-server.exe` processes --
Qwen2.5-VL-7B on port 18101, Qwen3-Embedding-8B on 18102, DeepSeek-R1-8B on 18103 -- children of a
`python -m pytest tests -q` run started by a coding agent. StudioForge's own `/api/status` listed
them as `llama-server.exe` with `is_ours: false` and `used_bytes: 0`, alongside seventeen desktop
processes reporting the same zero. Nothing in the product could say who launched them, how much
each held, or whether killing them was safe. They exited when pytest finished; had pytest been
killed hard, they would have survived, because the only cleanup was an `atexit` hook.

Four independent failures had to line up, and each is fixed separately.

**1. Nothing enforced "a child dies with its parent".** `core/supervisor.py` registered an `atexit`
handler that tree-kills tracked pids. `atexit` runs on a *clean* interpreter exit and on nothing
else -- not `SIGKILL`, not Task Manager's "End task", not a segfault, not a hard kill of the
interpreter. On a GPU-only server every one of those leaks VRAM that nothing on the box knows how
to attribute.

*Decision: a Windows job object per Supervisor, with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`.* Every
child is assigned to it, so the kernel terminates them when our last handle closes -- which happens
however the process dies. The job is anonymous (a named job could be opened and closed by anyone
who guessed the name) and nesting is fine on Windows 8+, which matters because the serve process
routinely already lives in the tray launcher's or the terminal's job.

Children are spawned `CREATE_SUSPENDED` and resumed after assignment, so the window between "the
process exists" and "the process is in the job" contains no executed instruction. Assigning
straight after `Popen` would leave a sub-millisecond gap; it is avoidable, and the failure mode of
the chosen path is strictly milder (a child that is never resumed has allocated nothing, and it is
killed and reported as a load failure rather than left suspended).

**The safety net is never load-bearing.** `AssignProcessToJobObject` returning `ERROR_ACCESS_DENIED`
(a pre-Win8 job, or one without `BREAKAWAY_OK`) logs one WARNING -- once, not once per load -- and
the load proceeds unprotected. Same for a box with no pywin32: `create_child_job()` returns `None`.
A net that refuses to be hung must not be the reason a model will not load. `atexit` and
`kill_process_tree` stay exactly as they were; this is an additional guarantee, not a replacement.
Non-Windows is a no-op (process groups plus `atexit` remain the answer there).

**2. Nothing swept up what earlier runs left behind.** A leaked child was invisible until someone
noticed the GPU was full, and the first symptom was usually an insufficient-VRAM rejection with no
cause named.

*Decision: `core/vram_holders.py`, swept once at gateway startup* (from `api/app.py`, after the
supervisor exists and before models auto-load -- the ordering matters, a leak found later has
already broken the first pinned load). `find_engine_processes()` reports every `llama-server` whose
executable lives under **our** engines directory, with its parent named and classified:

| classification | meaning | swept? |
| --- | --- | --- |
| `ours` | a child of this supervisor | never |
| `child-of-live-process` | somebody else's live llama-server | **never** |
| `orphan` | our binary, parent gone or its pid recycled | yes |

Only `orphan` is killed, and that is safe *by construction*: nothing else on the box launches
binaries out of our engines tree, and no live process is waiting on it. `child-of-live-process` is
the incident's own case -- a pytest run that was legitimately using those models -- and killing it
to recover memory nobody was short of yet would have been the second bug. It is reported, with the
parent named, and left alone.

The parent check carries a pid-reuse guard identical to `supervisor.process_is_alive`: a process
that started *after* its supposed child did not spawn it. Without it, the boxes that recycle pids
fastest are exactly the ones where a leak is filed as "somebody's live child" and survives every
sweep.

Surfaces: `GET /api/vram/holders`, `POST /api/vram/reclaim` (`{"dry_run": bool}`), a Dashboard
panel with a Reclaim button on orphan rows only, and a `reclaim_orphan_engines` watchdog MCP tool
so the sweep still works when the main app is wedged. The watchdog re-implements the rule locally
rather than importing `vram_holders`, for the reason its module docstring already gives for
`kill_process_tree` and `safe_log_name`: the recovery process must not import the stack it repairs.

**3. Per-process VRAM was blank, so every holder was anonymous.** NVML reports `usedGpuMemory` as 0
for every process on Windows under WDDM -- video memory is owned by the OS there, so NVML can
enumerate the holders but cannot size them. That is why the incident's status payload showed three
llama-servers at zero bytes.

*Decision: read the Windows performance counters.* `pdh_process_dedicated_bytes()` sums
`\GPU Process Memory(*)\Dedicated Usage` per pid -- the same counter Task Manager's "Dedicated GPU
memory" column reads. Verified read-only against this box: `dwm.exe` 0.306 GiB,
`msedgewebview2.exe` 0.279 GiB, `explorer.exe` 0.104 GiB, matching Task Manager. Cached 2 s (the
status endpoint and the Dashboard poll continuously), never raises, and latches off after one
failure with a single warning. A real NVML number is never overwritten: on Linux NVML is
authoritative *and* per-GPU, which is strictly better.

**What the PDH number is not:** it is a per-process total across adapters. The PDH instance name
carries an adapter LUID (`pid_19544_luid_0x00000000_0x0000E5F5_phys_0`), NVML exposes no LUID, and
inventing a split per GPU would be worse than saying so. Hence `used_bytes_source` on every
reported holder, and hence `holders_view()` aggregating one row per **pid** rather than per
(gpu, pid) -- summing the per-GPU rows of one pid would report a 15 GiB split model as 30 GiB.

`/api/status.vram_processes` keeps all its existing keys and gains `parent_pid`, `parent_name`,
`parent_cmdline`, `classification` and `used_bytes_source`. The ~20 rows of compositor and browser
noise collapse into `desktop_processes_count` / `desktop_processes_bytes`; a holder survives the
filter if it is ours, an engine binary, named like a GPU workload (`python`, `ComfyUI`, `ollama`,
...), or holds more than 256 MiB. Nothing is hidden -- what is dropped is counted.

**4. The contract suite took the GPUs by accident.** `pyproject.toml` had `testpaths = ["tests"]`,
so a bare `pytest` collected `tests/contract`, whose `conftest.py` starts the **real** gateway
in-process against `DEV_DATA_DIR = <repo>/../data` -- the LIVE data dir, live `registry.sqlite3`,
live engines -- and the real model library at `<models-dir>`. Its only gates were `skipif`s on the
engine and the models being present, which are always true on this box. An agent running the whole
tree therefore loaded three real models onto the real GPUs, and the tests wrote the production
registry while doing it.

*Decision: two independent gates plus a temporary data dir.* Every item under `tests/contract` is
marked `contract` by that directory's `conftest.py` (applied by hook, so a new test cannot be added
*without* the mark), `addopts = "-m 'not contract'"` deselects it by default, **and**
`SF_RUN_CONTRACT=1` must be set or every item skips with a reason. Belt and braces deliberately: a
marker is one `-m` away from being overridden, and `-m contract` alone must not be enough for a
stray CI job to reach live hardware. `just test-contract` / `make test-contract` set both.

The fixture now builds `Config(data_dir=<tmp>)` and links only `engines/<tag>` into it (a Windows
directory junction: 670 MB per engine makes copying absurd, and `shutil.rmtree` was verified not to
follow junctions, so cleaning up the temp dir cannot delete the installed engine). Where the link
cannot be made the suite **skips** rather than falling back to the live directory. The engine
manager still writes its own `engine.json` next to the binary it verified, which lands inside the
linked directory -- a knowingly accepted, idempotent write to shared state, and the only
alternative to copying 670 MB per session. A unit test asserts the `addopts` string, because
deleting it would restore the incident with no other test failing.

**What stays a limitation.**

* A live *foreign* holder -- ComfyUI, another install's llama-server, a test run -- is reported and
  never killed. Reclamation is deliberately restricted to processes nothing can still be using.
* Per-process VRAM is Windows-only via PDH and per-GPU only via NVML. Where neither answers
  (containers, WSL, MIG) holders are listed with an unknown size, which the payload says outright
  rather than reporting zero as if it were a measurement.
* The job object is a Windows guarantee. On Linux a `kill -9` of the server still leaves children
  behind until the next startup sweep finds them.
* An engine process launched through a wrapper whose process *name* is not `llama-server` is not
  matched by `find_engine_processes` (the watchdog's own argv-based matcher still finds it).

---

## D22 -- The catalog's numbers were wrong for the models the rig actually runs

(Recorded after D23; the decision it corrects is D20.)

**Problem.** D20's catalog built in 50 ms and every column in it was believable except the ones that
mattered. Five faults, each hiding the others, and all five landed hardest on the models this box
actually serves.

*The KV cache was charged as if every model were uniform.* `LoadPlan.kv_bytes_per_token` for a
Gemma-4 31B said 1,966,080 B/token while `Planner.estimate` -- iSWA-aware since D15 -- allocated
2.42 GiB at 16k and 21.17 GiB at 262k on the same model, an effective 80-155 KiB/token. Two numbers
for one quantity, and the wrong one fed both consumers. The VRAM slot bound came out as
`34.5 GB // (16384 x 1.9 MB) = 1`, so **every Gemma-4 row read `max_parallel: 1 (vram)` with 34 GB
free**; the throughput KV term charged `131072 x 1.9 MB = 258 GB` per token at 262k, so the same row
read **`est_gen_tps: 1.9`** for a 31B on two 5090s.

*Half the library is not uniform, and part of it is not even attention.* Qwen3.5/3.6/3.8 (`qwen35`,
`qwen35moe`) declare `{arch}.full_attention_interval = 4`: only layers where `(il + 1) % 4 == 0`
hold a KV cache, and the rest are Gated-DeltaNet recurrent layers with a fixed per-sequence state.
The planner charged KV for all of them -- a straight 4x -- which spread Dark-Scarlett-27B across all
four GPUs at 262k (65 GB planned against ~16 GB real) and forced the 122B onto a q4_0 cache it did
not need.

*Calibration cross-contaminated.* The `gpu_class` tier pooled every observation on the box. The
registry holds 84 rows for the 122B MoE (median measured/est 0.411) and 3 for a dense Gemma-4 (ratio
9.2, rejected by the `[0.1, 3.0]` clamp -> zero usable). So the pooled median *was* the MoE's number,
and it was applied to every dense model on the rig. With the KV term fixed, that dense 31B estimates
~41 t/s against a measured 39.4 -- the dense physics was fine all along; the MoE derate was leaking
onto it.

*Small models were fiction.* A pure bandwidth roofline has no per-token latency term, so a 0.92 GB
Qwen2.5-1.5B "should" decode in 0.5 ms and the catalog said **927 tok/s** (SmolVLM: 2063). Nothing
on this rig has ever produced those numbers.

*And the recommendation picked 16k for the 122B*, because the knee gave two slots there and one at
32k -- a knee that was itself wrong, from the same KV bug. 16k is below the D14 floor an agent host
needs. A queued second conversation is a latency problem; a window that cannot hold an OpenClaw tool
transcript is a failed task.

**Decision.**

**1. Per-layer KV geometry is the single source of truth.** `planner.kv_layers(meta)` returns one
`KvLayer` per block (`full` / `swa` / `none`) and everything folds over it: `kv_alloc_bytes` (what
llama.cpp allocates), `kv_read_bytes_per_slot` (what one slot re-reads per decode step) and
`effective_kv_bytes_per_token` (the allocation divided by the window, which is what `LoadPlan` now
carries -- a context-dependent figure, so nothing may cache one value per model and reuse it across
tiers). `estimate_kv_bytes` and `estimate_kv_bytes_iswa` survive as thin wrappers over the same
arithmetic, so the uniform, iSWA and hybrid answers cannot drift apart again -- that gap *was* the
bug. `attention_kind(meta)` reports the shape (`full` / `iswa` / `hybrid` / `unknown`) as a catalog
column, derived from the geometry rather than from `general.architecture`, which is not a reliable
predictor: `gemma4` covers both a dense iSWA model and a MoE one. **`"unknown"` means "distrust every
KV number for this model", never "assume the cheap case".**

**2. The VRAM slot bound is an exact walk, not a quotient.** `Planner.max_slots_by_vram` starts at
the cap and asks `Planner.estimate(parallel=N)` -- the same function a real load asks -- until one
fits. At most eight pure-arithmetic calls, and it cannot disagree with itself. The analytic form
survives in `max_parallel_for` for callers that want arithmetic, and its knee now divides by real
read bytes rather than `ctx_used * uniform_per_token`. The catalog stopped carrying its own copy of
that walk; `catalog.slots_for_plan` is one call into `Planner.size_slots`, because a copy is exactly
how a table starts advertising four slots where a load would settle for two.

**3. Two speed columns, because one number cannot describe a row.** `est_gen_tps` is quoted at
`REFERENCE_FILL_TOKENS = 8192` -- one ordinary turn, and below every catalog tier so the quote never
exceeds its own row's window -- and `est_gen_tps_full_ctx` at the row's whole `ctx_per_slot`.
Quoting only the first flatters a 262k row nobody experiences at 262k; quoting only the second
condemns it for a cost only its last token pays. The truth is between them, and now both ends are
visible.

**4. Four new constants, each with a stated direction of error.** They are approximations, not
measurements, and they are labelled as such in the source; calibration owns what remains.

| Constant | Value | Models | Errs |
| --- | --- | --- | --- |
| `T_TOKEN_OVERHEAD_S` | 1.5 ms | sampling, the logit/grammar pass, llama-server slot + HTTP bookkeeping, CUDA graph launch at batch 1 | small models stay slightly fast -- a larger floor would drag 20-30 ms/token models slow for no physical reason |
| `MOE_DECODE_EFFICIENCY` | 0.45 | `MUL_MAT_ID` gathering `n_expert_used` scattered expert matrices per layer: occupancy- and launch-bound, not bandwidth-bound | low = a MoE looks slower than it is, the safe direction; calibration corrects upward |
| `DECODE_EFFICIENCY_PER_EXTRA_DEVICE_MOE` | 0.10 (dense stays 0.05) | a MoE routes to experts that may not live on the device the previous layer used, so the pipeline stalls harder | slow |
| `MOE_PROMPT_EFFICIENCY` | 0.4 | routing shreds a ubatch across `n_expert`, degrading one GEMM into many skinny ones plus gather/scatter | slightly optimistic, matched to the decode side so one calibration factor is not pulled in two directions |

The efficiency divides **only** the weight term. The weight read is the scattered, kernel-bound one;
the KV re-read is a long contiguous stream that lands much closer to peak, and folding the derate
into both would make the estimate wrong in the same direction twice at long context -- which is
where the catalog is read most. Prefill also stopped averaging device *rates* and now adds device
*times*: a layer split is a pipeline, and the arithmetic mean claims a 5090+3090 pair delivers 140
TFLOPS where it delivers 106.

**5. Calibration gets four tiers, a density filter and a version gate.** `model+devices` -> `model`
-> `peers` -> `none`. **The string `gpu_class` is gone.** `peers` means *other* models on the same
hardware class with the **same density** (dense vs MoE) and the **same device count**, taken as a
**median of per-model medians**, each contributing model needing its own two rows. Never a raw pool
median: one chatty model must not outvote the rest, which is exactly how 84 MoE rows outvoted 3 dense
ones. The catalog looks calibration up **per device set**, because one model's 16k and 262k rows are
routinely placed on different cards and a four-way split's measurements say nothing about a
single-GPU row.

**Only rows stamped with the current `ESTIMATOR_VERSION` contribute a ratio** (migration 004 adds
the column; pre-existing rows are NULL and excluded by construction). A calibration factor is
`measured / estimated` -- a correction to one specific formula -- so carrying v1's ratios into v2
would teach the new estimator the difference between two dead arithmetics. The *measurement* half of
each row is still true, so the rows stay and `measured_for()` still reports them; only the ratio is
retired. The cost is that the rig is uncalibrated until fresh samples accumulate, and that is
acceptable precisely because the raw v2 estimate is now in the right neighbourhood.
`manager._sample_one` predicts at the same `REFERENCE_FILL_TOKENS` the catalog quotes, so the
learned factor is exactly "real traffic divided by what we promised" rather than a comparison of two
different questions.

**6. The recommendation has a floor, and the floor outranks the second slot.**
`floor = models.default_ctx`, raised to `models.thinking_default_ctx` for a thinking model -- the
same floor the planner's own context ladder never walks below (D14). In order: the highest ctx at or
above the floor that also sustains two slots (chat-class only), else the highest at or above the
floor, else the highest that fits with `"(below floor)"` said out loud in `recommended_basis`, else
the same three-way preference applied to the `if_gpus_idle` column.

**Anchors** (measured on the live rig, 2026-08-18, from the children's own `/metrics` counters):

| Model | Placement | Measured | v2 estimate | v1 estimate |
| --- | --- | --- | --- | --- |
| Dark-Scarlett-v2.0-31B-Q8_0 (`gemma4` dense, 30.4 GiB, 60 layers) | 2x RTX 5090, ctx 262144 f16, 1 slot | **39.4** gen / **2053** pp | 36.1 / 1190 | 1.9 gen |
| Qwen3.5-122B-A10B Q5_K_M (`qwen35moe`, 81 GiB, 48 layers, 8/256 experts) | 5090x2 + 3090x2, ctx 262144 q4_0, 1 slot | **37.3** gen / **869** pp | 47.4 / 1124 | ~143 gen |

Measured/estimated for the 122B moves from 0.26 to 0.775 on generation, and its prefill from ~0.23 to
0.77 (1124 estimated vs 869 measured -- the harmonic split plus the MoE prefill derate); the dense
Gemma-4's generation lands within 9% with no calibration at all, while its prefill is estimated at 1190
against a measured 2053 -- the 0.35 dense prefill efficiency is deliberately conservative (llama.cpp's Q8_0
path uses int8 tensor cores and beats the fp16 roofline), and the per-model prompt factor calibration learns
closes that.

**What remains, and who owns it.** The constants above are the whole remaining error budget, and
calibration owns the rest -- per model, per placement, from measurement. Three gaps are known and
deliberate:

* **Two anchors, both at one slot, both on this rig.** Nothing here is validated at 4 or 8 slots, on
  another GPU generation, or on a dense model above 31B. `--ctx-checkpoints` is still unmodelled.
* **The `peers` tier cannot help yet.** It needs a second measured model of each density on this
  hardware; with one measured MoE and no measured dense model it correctly reports `none` rather
  than guessing. That is the right failure, but it means a freshly scanned library stays
  `"estimated"` for longer than D20 implied.
* **The hybrid recurrent-state term is derived, not weighed.** `nextn_predict_layers` are excluded
  from the recurrent count -- an MTP head is neither an attention block nor a mixer, and llama.cpp
  does not run it during ordinary decoding -- and that reading reproduces both the 157 MB (27B) and
  156 MB (122B) anchors exactly. If a real load measures ~3.3 MB/slot more than predicted, that
  exclusion is the line to flip.

**What pins it.** `tests/unit/test_kv_geometry.py` (the four layer shapes, byte-for-byte equality
with both legacy KV functions, the 1 GiB / 157 MB / 156 MB anchors, the exact VRAM walk),
`tests/unit/test_throughput.py` (both measured anchors, the small-model latency floor, the harmonic
prefill, all four calibration tiers and the version gate), `tests/unit/test_db.py` (migration 004),
`tests/unit/test_catalog.py` (both speed columns, `attention_kind`, each of the floor rule's four
branches, the device-order-insensitive idle dedupe, and that `_sample_one` stamps the estimator
version and predicts at the fill the catalog quotes) and `tests/unit/test_mcp.py` (`limit`, and that
no row says the same thing twice). A formula change that moves any anchor fails visibly.

---

## D24 -- A download has one writer, and completion is proven on disk

**Incident (2026-08-18).** A 19.27 GB download of `unsloth/Qwen3.8-27B-GGUF` (Q5_K_S, main file
19,270,036,448 B plus its mmproj) was enqueued by the live server at 22:01:48 and started
transferring. At 22:04:45 and again at 22:07:09 `data/logs/studioforge.log` records
`downloader.resumed groups=2` from a **second Downloader** -- a test process that had built
`create_app` against the *live* data directory, loaded the live download queue out of
`registry.sqlite3`, and begun writing the same `Qwen3.8-27B-Q5_K_S.gguf.part`.

At 22:08:24 the live server's transfer died:

```
PermissionError: [WinError 32] The process cannot access the file because it is being
used by another process: '...\Qwen3.8-27B-Q5_K_S.gguf.part'   -> status=failed
```

The other writer then "completed" the file. What landed in the library was
**22,576,551,872 bytes where 19,270,036,448 were declared** -- 3.3 GB of interleaved chunks -- and it
passed `_verify`, because `_verify` compared the *streamed* byte counter and the *streamed* sha256:
both describe what that process sent to `write()`, neither describes what the filesystem held. The
`part.replace(dest)` that followed removed the `.part`, so when the user pressed Resume at 22:12:19
there was no partial left and the download **started again from zero**. The corrupt file kept its
`.gguf` name, so the registry scanned it and listed it as a model;
`test_gguf.py::test_real_library_tensor_bytes...` fails against it for the same reason.

Four independent failures, each fixed separately.

**1. Nothing owned the `.part`.** Two processes opened it, both succeeded (Python's `open()` on
Windows shares deny-none), and their chunks interleaved. The `WinError 32` that killed the live
transfer was not the bug -- it was the *only* symptom, arriving four minutes late and blaming the
victim.

*Decision: the `.part` is opened once per transfer, under an exclusive OS byte-range lock, and held
for the whole of it.* `_PartFile` (`core/downloader.py`) takes `msvcrt.locking(LK_NBLCK)` on Windows
and `fcntl.flock(LOCK_EX|LOCK_NB)` on POSIX. A second writer is refused **immediately** with
`PartFileLockedError` naming the cause and the fix, and that error is deliberately *not* retryable:
a held lock is another writer doing its job, not a hiccup, and backing off for eight seconds does
not improve anybody's situation.

The lock spans the retry budget, not one attempt -- a backoff pause is precisely the window a second
writer would walk into. It also spans the resume re-hash and verification, which forced two
mechanics: the locked byte sits at offset `1 << 62`, past the end of any real file, because on
Windows a locked range is unreadable to every other handle and a diagnostic reading a 40 GiB partial
must not trip over our bookkeeping; and deleting or renaming the file is driven from `_PartFile`
itself (`discard()` marks it for deletion at close, `publish()` closes then renames), because
Windows refuses both while the file is open. The single uncovered instant is between that close and
the rename -- documented in `publish()`, survivable because the bytes are already verified, and
belt-and-braces against `InstanceLock` below.

**2. Verification trusted the stream, never the disk.** This is the failure that turned a messy
collision into a corrupt library entry. The check that would have caught it -- `stat` the file --
was never made.

*Decision: fsync, then prove it on disk, then publish.* Before any rename, `_verify` compares
`part.stat().st_size` against the streamed count (catching a foreign writer, a short write, a
truncation) **and** against the declared total, and keeps the streamed sha256 check on top. A
partial that fails either is unknowable garbage -- there is no way to say which bytes are ours -- so
it is deleted rather than kept for a resume that would append to rubbish.

`_adopt_complete` got the same treatment from the other end, because the corrupt file was already
*at* the destination and the old code simply declined to adopt it and moved on, leaving a 22.58 GB
`.gguf` for the scanner to find. A file at the destination is now adopted only if its size equals
the declared total, plus its sha256 when HuggingFace published one and the file is under
`ADOPT_HASH_MAX_BYTES` (2 GiB); above that the size is checked and the checksum is not, logged as
`downloader.adopted_by_size_only` rather than reported as verification that did not happen. A file
that fails is renamed to `<dest>.corrupt-<timestamp>` -- out of the scanner's way, still on disk for
a human, **never deleted** -- and the download proceeds. Renaming is refused, loudly, when a loaded
model has the file open.

Quarantine happens when the transfer starts, not at enqueue: queueing a download must not rearrange
someone's model library before it has begun, and a stale blob listing must not cost a user their
file on the strength of a size the API got wrong.

**3. One dropped connection was one dead download.** A transfer had exactly one attempt. Over an
hour of streaming that is not a policy, it is a coin flip -- and the incident's own `PermissionError`
would have been survived by a single retry.

*Decision: five attempts, exponential backoff from 2 s to a 60 s cap, jittered ±20%, each resuming
from the `.part` through the existing `Range` logic.* Transport errors, 5xx, 429 (honouring
`Retry-After`, still capped) and momentary `OSError`/`PermissionError` on the `.part` retry;
404/401/403, size mismatch, checksum mismatch, a full or read-only disk and a locked `.part` fail
immediately. Retrying a 404 five times turns "that file does not exist" into "the download hung".

The backoff is a plain `asyncio.sleep` inside the group's task, so `pause`/`cancel` cancel it on the
spot; a polled deadline would leave the Pause button inert for up to a minute. `attempt`,
`max_attempts`, `next_retry_at`, `retry_in_s`, `last_error` and `part_bytes` are on the progress
snapshot and in the queue payload, and the Download tab now says `retrying in 8s (attempt 2/5):
<error>` instead of showing a "running" bar that has not moved in thirty seconds.

`part_bytes` answers the question this incident left unanswerable. It is captured once, in `_fail`,
and the queue says either **"Resume continues from 17.9 GiB"** or **"Resume will restart from the
beginning (no partial file)"**. On the day, the answer was the second one and nothing in the product
was willing to say so.

**4. Two app instances shared one data directory.** The root cause. The port preflight (D21) catches
a duplicate `serve`, but the second writer here never bound a port: it was an in-process embedder --
`create_app` inside a test -- and nothing else stood in its way.

*Decision: `core/instance_lock.py`.* `InstanceLock(data_dir)` takes an exclusive OS lock on
`<data_dir>/.instance.lock` and writes `{pid, create_time, started_at, data_dir}` inside it.
`create_app(start_background=True)` acquires it; when another **live** process holds it, this one
becomes `secondary` and **starts no background work at all** -- no download resume, no TTL sweeper,
no auto-load, no orphan sweep -- logs one ERROR naming the holder's pid, and reports
`"instance": "secondary"` plus `instance_holder_pid` from `/health`. A secondary still serves reads:
the failure mode to avoid is a process that is silently inert, not one that is honest about it.

The kernel is the authority, not the file. A lock is released however the holder dies -- clean exit,
`SIGKILL`, Task Manager, a bluescreen -- so a lock file left by a crash is taken over automatically.
The recorded pid and creation time exist for the error message and for `holder()`; the creation time
is the pid-reuse guard, so a stranger wearing a dead process's number cannot keep a startable server
in secondary mode. Where the lock cannot be taken at all (a network share with no working locks),
acquisition degrades to today's behaviour with a warning rather than refusing to boot.

**Deliberately, the lock is taken in `create_app`, not in `build_state`.** The stdio MCP server and
every `start_background=False` test compose the same object graph; if composing it took the lock,
running one of them would demote the real server and turn a safety feature into an outage. The lock
is about *acting* on a data directory, not about describing one.

**Also fixed: the in-use guard had never run.** `_file_in_use_check` in `api/app.py` called
`supervisor.all()`, which does not exist -- the method is `supervisor.list()`. Every invocation
raised `AttributeError`, so the one thing standing between a forced re-download (which unlinks the
destination) and the weights of a *running* model was a method name that had never been executed. It
now calls `list()`, and a supervisor or registry that raises answers **conservatively**: "I cannot
tell" must not read as "go ahead and delete it".

**What remains.**

* **`ADOPT_HASH_MAX_BYTES` is a real gap.** A >2 GiB file at the destination whose size matches but
  whose contents are wrong is adopted. The alternative -- re-reading 20 GB on every enqueue of a
  model the user already has -- is worse, and the size check alone would have caught this incident.
  The log line says which check ran.
* **The publish window.** Between `close()` and `replace()` the `.part` name is momentarily
  unlocked. Sub-millisecond, after verification, and covered by `InstanceLock` in the case that
  actually happened.
* **The lock is per data directory, not per model library.** Two instances with *different* data
  dirs pointing at one `models.dir` are still two writers; the `.part` lock catches the collision
  and refuses, which is a clean failure rather than a corrupt file, but nothing prevents the
  situation.
* **`data/registry.sqlite3` still lists the corrupt Qwen3.8 file** until the live server next
  processes it. The fix is in the adopt path, which will quarantine it to
  `Qwen3.8-27B-Q5_K_S.gguf.corrupt-<ts>` on the next enqueue of that quant; nothing was renamed by
  hand.
* **Advisory on POSIX.** `flock` binds cooperating processes. A third-party tool writing into our
  `.part` is not stopped on Linux -- but nothing else knows the name, and the on-disk size check
  catches the result regardless.

**What pins it.** `tests/unit/test_instance_lock.py` (two locks on one directory, stale-lock
takeover, pid reuse, the payload staying readable while locked, and the app path: a secondary starts
no downloader and says `"instance": "secondary"`, a primary starts one and releases the lock at
shutdown, and `start_background=False` claims nothing). `tests/unit/test_downloader.py` (a second
writer refused with the partial untouched and not one request made, the refusal not retried, the
lock released on completion; a `.part` that grows behind the stream never published and discarded --
the incident reproduced with the stream hash and byte count both correct; wrong-size and
wrong-checksum files quarantined rather than adopted, size-only adoption above the ceiling, enqueue
moving nothing; 5xx retried then succeeding, retries exhausted at exactly `DOWNLOAD_MAX_ATTEMPTS`, a
404 requested once, a retry resuming from the same offset, the retry fields reaching the payload,
pause interrupting a backoff in under a second, `part_bytes` recorded on failure; the transient/fatal
split, the backoff's growth, cap and jitter; the queue panel's two sentences; and
`_file_in_use_check` calling the method that exists and answering conservatively when it cannot).

---

## D25 -- The data directory has exactly one story: `SF_DATA_DIR`, else `<repo>/data`

**Problem.** The same question had three answers. The `.bat` launchers defaulted `SF_DATA_DIR` to
`%~dp0..\data` -- a sibling of the checkout, outside it. `justfile`/`Makefile` defaulted to
`../data`, the same place, but only when invoked through them. And
`config.default_data_dir()`, which is what `studioforge serve` actually uses when nothing sets the
variable, returned the platform directory (`%LOCALAPPDATA%\studioforge`). So a user who
double-clicked the launcher and a user who typed the command got **different installs**: different
`config.yaml`, different registry, a second copy of the engine, and a model library that appeared
empty because the config that pointed at it was the other one.

**Decision.** One rule, in this order, everywhere:

1. `SF_DATA_DIR` if it is set;
2. `<repo>/data` when the package is running from a source checkout;
3. the platform data directory, for an installed wheel.

Step 2 is what makes the launchers and the CLI agree without either of them setting the variable
behind the other's back. A checkout is recognised structurally -- `pyproject.toml` two levels above
`config.py`, alongside `src/studioforge/` -- so an editable install is a checkout and a wheel in
site-packages is not. `data/` is in `.gitignore`, so the directory the app writes into is never
something git can offer to commit.

**D7 still holds.** The data dir is out of the *release* directories, which is what self-update
destroys; a git checkout is not a release directory, and `git pull` does not delete ignored files.

**The escape hatch is a file, not a flag.** Every launcher runs `if exist "%~dp0local-env.bat" call
"%~dp0local-env.bat"` before anything else, and `local-env.bat` is gitignored. Pointing a fresh
checkout at an existing install is one line in a file that can never be committed by accident:

```bat
set "SF_DATA_DIR=D:\path\to\an\existing\data"
```

**Two instances on one data directory is still one instance.** D24's `InstanceLock` makes the
second process a *secondary*: it serves reads and starts no downloader, no TTL sweeper and no
auto-load. That is a deliberate, visible degradation (`GET /health` reports
`"instance": "secondary"` and names the holder's pid), not a failure -- but it means "point the new
checkout at the old data dir" requires stopping the old server first, and the README says so where
the hook is documented.

---

## D26 -- Setup is a tab, not a YAML file

**Problem.** Every setting existed; none of them were reachable. `config.yaml` carried 81 keys and
the GUI exposed **twelve** of them, scattered down the Server tab between a health panel and an
engine panel. The other 69 were editable only by finding the data directory, opening a YAML file
and knowing which key to change -- including the two knobs this rig actually needed
(`planner.excluded_devices` and `planner.reserved_mb`, D19), the context target that decides every
plan (`models.target_ctx`, D14), the slot estimator's inputs (D17), the CUDA-build override that is
the fix when auto-detection picks wrong (D2/D3), and the MCP PIN, which is *displayed* in three
places and could be rotated from none of them. A first-run user had to read the source to find out
what they were being asked to decide.

**Decision.** A **Setup** tab, second in the strip and the landing tab on a fresh install, holding
every user-facing setting grouped by the decision being made -- model library, GPUs and memory,
engine, network and access, downloads, startup -- with a computed first-run checklist on top. The
Server tab keeps its compact four-item readiness strip and its panels; the Setup tab is the full
surface, and both read the same helpers.

### The checklist rule: required versus optional

A check is **required** when the server cannot serve inference without it: a writable data
directory, a model library with GGUFs in it, an indexed registry, at least one GPU (this system is
GPU-only), an installed engine, and something listening on the gateway port. It is **optional**
when it changes what you can *do* rather than whether the thing works: a HuggingFace token (public
repositories download fine without one), autostart, and the MCP PIN whenever `mcp.pin_required` is
off.

Only required checks gate "Ready to serve". This is the entire point of the distinction: a
checklist that shows a warning triangle next to an unset HuggingFace token on a machine that will
never download a gated model is a checklist people learn to scroll past, and then they scroll past
the amber line that says no engine is installed. Optional items render grey and say "optional" out
loud.

Every unmet item names exactly one action and renders the button for it -- Install engine, Detect
LM Studio library, Rescan, Generate PIN, Enable autostart -- because "your model directory is not
set" without a control next to it is a diagnosis, not a fix.

The checklist is **computed, never remembered**: no "setup complete" flag is stored anywhere. A
library on a drive that later fails to mount goes amber again by itself, which a stored flag could
not do. It is also deliberately not on a timer -- each item costs a directory walk, a socket bind
or a stat, and none of them change on their own.

**Fresh installs land on Setup.** "Fresh" is measured, not remembered: no `models.dir`, or no
models indexed, or no engine installed. On such a box the Dashboard is four empty panels, and the
Setup tab is the whole answer. Once the server can serve, the landing tab reverts to the Dashboard
for good.

### The advanced-section rule: generated, with three exclusions

The long tail (`gateway.*` alone is 16 keys, plus `update.*`, `watchdog.*` and the planner's
calibration constants -- 49 in total against 30 with a purpose-built control, out of 81 keys in
`config.example.yaml`) is **generated from the pydantic model** rather than hand-listed, with
type-aware widgets
derived from the annotation: `Literal` becomes a select carrying its own choices, `bool` a
checkbox, `int`/`float` a number, `list[...]` a comma-separated field, `Path` and `str` text. A key
added to `config.py` therefore appears in the GUI with no second edit, and "every setting is
reachable" stays true instead of being true on the day it was written.

Three exclusions, each stated as a rule so it keeps holding as the config grows:

1. **Keys with a purpose-built control above.** One control per key, always. Two forms for one
   value is how a user changes a setting and watches it change back.
2. **Secrets** (`server.api_key`, `hf.token`, `mcp.pin`). They need the masked/reveal widget and
   the "did this really change" guard, because a generated field drawn with the redacted
   placeholder would post `"abcd...yz"` straight back over the real credential.
3. **Types with no honest scalar rendering** -- `planner.reserved_mb` (`dict[int, int]`) and
   `planner.quant_affinity` (`dict[str, QuantAffinity]`). A mapping rendered as a text box silently
   destroys the entries the user did not retype. Both get row widgets instead:
   `reserved_mb` as a MiB field per GPU, next to that GPU's live free VRAM.

A test asserts that the only keys the generator drops are exactly those two mappings, and that the
union of "has its own control" and "is in Advanced" is every remaining key. That is what makes the
claim checkable rather than aspirational.

### One implementation of "change a setting"

Every field on the tab -- generated or hand-built, including the per-GPU maps -- saves through
`gui.tabs.apply_config_updates`, which calls the management route `PATCH /api/config` in-process.
So the Setup tab, the Server tab, the HTTP API and the MCP `set_config` tool all run the same
`apply_overrides` validation, the same atomic write, the same live-apply of the sections that can
change without a restart, and report the same `restart_required` list. The Server tab's private
copy of that logic was deleted in the process; it had drifted -- it did not apply the `mcp` section
live, which the route had already been fixed to do.

**Secrets are never rendered in clear.** The form is drawn from `state.redacted_config()`, one
function that masks all three keys, so no surface can leak one by forgetting. The OpenClaw snippets
legitimately contain the key and the PIN -- that is what makes them paste-ready -- so they are
masked on screen behind an explicit "reveal secrets" switch while the copy buttons copy the real
text. Displaying a credential on a screen someone may be sharing and putting it on that person's
clipboard are different acts, and only the second one was asked for.

**What is still not in the GUI, and why:** `data_dir` and `source_path`. The first is derived from
the environment (D25) and a form that edits it would fight whatever set `SF_DATA_DIR`; the second
is bookkeeping. Both are shown read-only in "Where things live", together with which of D25's three
rules produced the directory in force.

---

## D27 -- Boot reuses an installed engine; it never reinstalls one

**Problem.** `EngineManager.ensure_engine` runs at every boot, inside the lifespan, *before the
API port is bound*. It ran the full smoke test on the installed engine -- `--version` plus a real
GPU micro-load of the smallest GGUF in the library -- and read a failed micro-load as "this
install is broken". It then went to GitHub for the asset list, called `install()` on the **same
tag**, which ran the same micro-load again, logged `reinstall_after_failed_smoke`, re-downloaded
the ~600 MB archive (or reused a cached one), extracted it over the working engine, ran the
micro-load a third time, failed a third time, and raised. `smoke_test_timeout_s` is 180 s, so a
micro-load that hangs rather than exits costs up to nine minutes of a server that answers
nothing on any port -- the tray says "Starting...", the watchdog's restart reports
`restart_unhealthy` at 120 s, and OpenClaw sees connection refused.

Every way that micro-load fails at boot is a condition a reinstall of the same archive cannot
change: every GPU full because ComfyUI is training on the same box; the tiny model it picked is
corrupt or half-downloaded; the driver was downgraded under a CUDA 13.3 build. The reinstall
bought minutes of dead air and a rewritten engine directory, and diagnosed nothing.

**Decision.**

* **`--version` must run; that is the whole boot check for a build that has already passed a
  micro-load on this box.** It is cheap, touches no GPU, and is exactly what fails for a genuinely
  broken install (a half-extracted archive, a missing `ggml-cuda.dll`). Only that failure sends
  boot to the network, and it is logged as `engine.ensure.broken_install` with the reason.
* **The micro-load runs at boot only for a build that has never passed one** (`smoke_tested`
  false in its `engine.json`) -- the case where nothing yet proves CUDA initialises with this
  binary. A pass is written back so the next boot skips it. A failure keeps the engine active,
  logs one WARNING with the detail and the next action (`studioforge engine --smoke-test`), and
  lets the first real load report the real error with the child's stderr tail -- which is where
  "CUDA error: out of memory" or "driver version is insufficient" is actually legible.
* **Reinstalling is an explicit act** -- the Setup tab's Install button, `engine --update`,
  `install(force=True)` -- never a boot side effect. `install()` on an already-present tag still
  re-runs the smoke test and reinstalls after a failure, because there someone asked.
* **A driver too old for the installed CUDA build is one WARNING at boot**, using the same
  `_cuda_eligible` comparison `select_asset` uses to choose a build: "engine b10425 is a cuda-13.3
  build but this driver only advertises CUDA 12.4 ... update the driver, or set
  `engine.cuda_variant` and reinstall". Before this it surfaced only as the first load's opaque
  `cuda error`.
* **Installs are serialised per tag.** The Setup tab's Install button clicked while boot was
  already installing (or clicked twice) had both callers streaming into the same
  `downloads/<asset>.zip.part` and extracting into the same directory; the second finished with a
  corrupt archive or a half-overwritten engine. Behind one `asyncio.Lock` per tag, the second
  caller waits, finds the engine present, and passes through `already_present`.

**Cost.** A driver downgrade under a build that once passed is no longer caught at boot by a
failed micro-load -- it is caught by the driver warning above and by the first load. That is
the trade: boot is fast and deterministic, and the failure moves to the place that can explain
it.

**What pins it.** `tests/unit/test_engine.py`: a verified engine boots without a micro-load and
without a network call; an unverified one is micro-loaded and the pass persisted; a failed boot
micro-load keeps the engine and never calls `install()`; a binary that cannot run `--version`
is reinstalled; the driver warning names the knob and stays silent when the driver is fine; two
concurrent installs of one tag run in turn.

---

## D28 -- Exactly one process respawns the server, and a deliberate restart is not a crash

**Problem (recorded live, WP4, 2026-08-18 23:24).** With the tray supervising the server -- the
normal Windows deployment: login -> tray -> `serve` -> watchdog -- a GUI **Restart server** went
`POST /api/restart/server` -> watchdog `POST /restart` -> the watchdog killed the process tree
and spawned a replacement. The tray saw its child exit, counted crash attempt 1 of 3, notified
"The server exited unexpectedly; restarting it", waited five seconds and spawned a second
replacement. Two servers then raced for 1234/8080; the loser's port preflight exited **3**, which
the tray counted as crash 2, waited, spawned again, exited 3, crash 3, "server keeps dying; giving
up" -- and the tray sat on **Crashed -- see the logs folder** beside a perfectly healthy server it
no longer supervised and could not stop ("Stop server" then said "stopped; VRAM released" about a
process that kept running). Every intentional restart from the GUI, from `sfctl recover
--restart` or from an agent's `restart_server` produced that.

Three things were wrong at once: two supervisors both believed they owned the respawn; the tray
had no way to tell a restart it did not initiate from a crash; and an exit on a port conflict --
which no respawn can fix -- was retried three times and then misdescribed.

**Decision.** *The process that launched the server respawns it, and nothing else does.*

* **A tray-launched server restarts by exiting.** The tray sets `SF_SUPERVISOR=tray` on the
  child it spawns. `POST /api/restart/server` in such a process neither asks the watchdog nor
  respawns itself: it drains, sets `exit_code`, and shuts both uvicorn servers down gracefully;
  `serve` then exits **75** (`EXIT_RESTART_REQUESTED`, sysexits' `EX_TEMPFAIL`: "try again"). The
  tray reads that code, respawns without spending a crash attempt, and says "Restarting the
  server". One hop, a real drain, one respawner. Measured on this box: 1.0 s from the request to
  the exit, the watchdog left running for the replacement to adopt (D21).
* **The watchdog defers to a tray it can see.** Its own `restart_server` -- the wedged-server path,
  where the app cannot cooperate -- still kills the tree, but when the killed process was the
  direct child of a live tray (`supervising_tray_pid`: parent argv is `studioforge tray`, parent
  predates the child) it does **not** spawn; it waits for `/health` and reports
  `respawned_by: "tray"`. For the whole operation its `/health` carries `restart_in_progress`, and
  the tray asks that endpoint (open, no credential) before it counts an exit as a crash.
* **The tray never respawns onto a taken port.** A child that exits `EXIT_PORT_CONFLICT` (3)
  spends no crash attempt: for up to `PORT_HOLDER_GRACE` (120 s) the tray waits for whoever holds
  the port to answer `/health` as a StudioForge server -- a replacement still scanning its library
  -- and attaches to it; otherwise the status line reads **"Port 1234 is held by another program
  (LM Studio?) -- quit it, then Start server"** instead of "see the logs folder". Before any
  crash-respawn it also checks whether a server already answers, and attaches rather than fights.
* **Attached is a state, not a lie.** `TrayApp.adopted` is set when the tray attaches to a server
  it did not launch (found at startup, or one that took over the port). "Stop server" on an adopted
  server refuses with "not started by the tray" rather than declaring VRAM released; on a server it
  launched, Stop also takes down a watchdog left over from an earlier restart (`find_watchdog_pids`,
  same `--config` only), because "stop" from the outermost owner means the whole deployment.
* **Shutdown is `should_exit`, not `os.kill`.** The restart paths ended with
  `os.kill(os.getpid(), SIGINT)`, which on Windows is not a signal: `os.kill` with anything but
  the two CTRL events is `TerminateProcess`, exit code 2, no handler, no drain, no lifespan
  shutdown -- verified on this box with a handler installed. `__main__._serve` now installs
  `state.request_shutdown`, which flips uvicorn's `should_exit` on both servers exactly as its
  own signal handler does; the signal remains only as the fallback for an embedder without
  `_serve`.

**Exit codes are now vocabulary**, in `core/ports.py`: `2` config error, `3` port conflict,
`75` restart requested. The tray reads them; nothing else may reuse them.

**Cost.** A tray-supervised deployment now depends on the tray to bring the server back after a
restart, which is what supervising means. If the tray has been quit, `SF_SUPERVISOR` is not set
on the next launch and the watchdog path applies unchanged.

**What pins it.** `tests/unit/test_tray.py` (exit-code and watchdog-driven classification, no
crash attempt spent on a requested restart, a port-conflict exit waits then adopts or names the
fix, a crash-respawn attaches to a server already up, Stop refuses an adopted server and takes a
leftover watchdog down, `SF_SUPERVISOR=tray` on the child); `tests/unit/test_watchdog.py` (a
tray-owned server is killed but not respawned and `restart_in_progress` is published and
cleared; the watchdog still spawns when no tray owns the server; tray-argv recognition);
`tests/unit/test_restart_handover.py` (the route exits for the tray instead of asking the
watchdog, `_exit_for_supervisor` drains/sets 75/hands the watchdog over, the graceful hook is
preferred and the signal remains the fallback).

---

## D29 -- One model load at a time

**Problem.** The planner decides against *live* free VRAM (D14/D16/D20: the numbers are always
the machine's, never a bookkeeping copy). A `llama-server` child that is still loading has not
yet taken the memory its plan says it will -- weights stream in over 10-90 s -- so two different
cold models requested at the same moment were each planned as if the other did not exist. Both
children launched onto the same cards; one died with `CUDA error: out of memory`; that failure
is classified transient, so its one retry evicted the LRU idle model -- which was the *other*
new model, idle for the instant between "ready" and its client's first request -- and that
client's request then landed on a dead child as a `502`. Two agents asking for two big models at
once could ping-pong like this indefinitely, each load "succeeding" and each request failing.

**Decision.** `ModelManager` serialises loads behind one `asyncio.Lock` (`_load_gate`), taken
inside the per-model lock and around the whole plan -> evict -> spawn -> healthy sequence. The
second load plans only after the first child has actually allocated, and then either fits
beside it or is refused with the numbers -- which is the documented worst case ("a refusal with
numbers, never a degraded load"). Waiting is logged once per waiter at INFO; a streaming client
sees the same keep-alive comments it sees for its own load.

**Why not account for in-flight plans in the planner instead.** Subtracting a loading child's
planned footprint from usable VRAM double-counts whatever it has already allocated (free VRAM
already reflects that), so for the duration of every load the planner would refuse loads that
fit. Charging only the unallocated remainder needs per-pid attribution that Windows cannot give
(D23). Serialising is exact, needs no arithmetic, and costs only concurrency between cold loads
-- which share one disk and one PCIe bus and were not faster in parallel anyway.

**Order, so it cannot deadlock.** Per-model lock, then the gate, on every path (`ensure_loaded`,
`load`, `load(force=True)`, the pinned auto-load, `restart_backend`). Nothing under the gate ever
starts another load: eviction stops children, the transient retry re-plans and re-spawns the
same model.

**What pins it.** `tests/unit/test_load_retry.py`: a second cold load is not planned until the
first child is up, and neither evicts the other.

---

## D30 -- A forced reload plans before it unloads, and a refusal offers the real context

**Problem (seen live, WP12).** `ModelManager.load(force=True)` -- the GUI's per-model
*Restart*, `POST /api/models/{id}/restart`, `POST /restart/backend` after an engine update,
the MCP `load_model(force=true)`, the Models tab's "save then load" -- stopped the running child
first and planned second. Both call sites carried a comment promising the opposite ("a failure
never leaves the model unloaded when it was working a moment ago"), and both were wrong: a
reload whose arguments no longer fit -- VRAM had moved to ComfyUI, a bigger context was asked
for, the engine had been swapped for one that would not start -- came back as a 507 with the
model gone. Every refused reload was an outage that the code had specifically been written to
avoid.

The eviction ladder is why it was done in the wrong order: the resident instance was, to the
planner, either a pinned obstacle ("unpin X" as the first suggestion for X's own reload) or an
LRU candidate that pass 1 (no eviction) could not touch, so the only way to make room for a
model was to remove it before asking.

**Decision.** `Planner.plan_load(..., reload_of=<model_id>)` plans *as if that child were
already gone*: its planned per-GPU footprint (`_instance_footprint`, the same figure the
eviction ladder credits for a victim) is added back to the free VRAM of the cards it sits on,
it is neither an eviction candidate nor a pinned obstacle, and its pid still counts as ours in
the VRAM-holder attribution so a refusal cannot blame the model for its own reload. The
returned plan lists it *first* in `evict_model_ids`, and the manager stops it there -- after
the plan exists, before the spawn. A refusal raises with the numbers, appends "the running
instance of X was left loaded with its previous settings; nothing was unloaded", and the child
keeps serving. If the resident died while the caller waited for the load gate (D29) the hint
is dropped and it is an ordinary load.

The credit is the estimate, not a measurement; the truth is whatever the driver releases when
the child exits, and a plan made on the estimate meets the same one-retry transient-OOM path
(`_start_with_retry`) a post-eviction plan does.

**Also: `max_ctx_that_fits` is computed on the per-layer geometry.** A refusal's "reduce
context to N" walked the uniform per-token formula, which D22 established is wrong for half the
library: an iSWA model was offered a 4k window when 65k fit, a hybrid a quarter of what it could
have. `max_ctx_for_budget_geometry` walks the context ladder against `kv_alloc_bytes` -- the
number a load is actually charged -- so the offer is one the next load accepts. The uniform
formula remains only for metadata that cannot support a per-layer answer.

**What pins it.** `tests/unit/test_planner_reload.py` (the footprint is credited back; a pinned
resident is not its own obstacle; the resident pid is ours in a refusal; a reload still evicts
other idle models when it must; a device-override reload credits that device; the geometry
walk beats the uniform figure for Gemma-4 by 8x and its offer re-plans successfully),
`tests/unit/test_load_retry.py` (a refused forced reload stops nothing; the planner is told
`reload_of`; a plain load never passes it).

---

## D31 -- `data_dir` is never inside config.yaml; a named config file lives in its data dir

**Problem.** D25 says the data directory has one story -- `SF_DATA_DIR`, else `<repo>/data` --
but `Config.save()` wrote every field, `data_dir` included, and pydantic-settings ranks
constructor arguments above the environment. So the moment Setup saved anything, the file
carried `data_dir: <wherever this process was>` and that value outranked `SF_DATA_DIR` on the
next load. Two consequences, both silent: `local-env.bat` (the documented way to point a
checkout at an existing data directory) stopped working after the first save; and copying an
old install's `config.yaml` into a new checkout re-pointed the whole install -- registry, engines,
logs, downloads -- back at the old data directory. Verified on this box with the real 0.1.0 file:
`SF_DATA_DIR=E:/should_win` lost to the file.

**Decision.**

* **`data_dir` is not persisted.** `Config.UNPERSISTED_KEYS = {source_path, data_dir}` is what
  `to_yaml_dict()` excludes, so no dump, save or `apply_overrides` round-trip carries it, and
  `apply_overrides({"data_dir": ...})` refuses with a pointer to `SF_DATA_DIR`.
* **On load, a `data_dir` key in the file is ignored** -- with one WARNING naming the value in the
  file and the directory actually used, when they differ. It got there from an older build's
  save; the warning is the whole migration.
* **The data directory of a load is `resolve_data_dir()`:** `SF_DATA_DIR` if set; else, when the
  config file was *named* (`--config`, `SF_CONFIG`), the directory the file lives in; else the
  checkout / platform default. The second rule is what keeps every spawn coherent: the tray, the
  watchdog and autostart all pass `--config <data_dir>/config.yaml`, so a child without the
  environment lands in the same data directory as its parent -- which is what the file's own
  `data_dir` key used to do, without the trap. The watchdog's own loader applies the same rule.
* **`config.yaml` therefore lives in the data directory, by construction.** A config file kept
  elsewhere and pointed at with `--config` makes *that* directory the data directory (unless
  `SF_DATA_DIR` says otherwise). The test harnesses that used to save the file beside the data
  dir now save it inside, which is the production layout.

**Also in the same commit, because a stranger's file hits them first:** every port is bounded
(`server.port: 70000` used to reach `socket.bind` as an `OverflowError` traceback from the very
preflight whose job is "a sentence instead of a traceback"), every count/timeout has its sign
checked, `logging.level` is a `Literal` (case-insensitive, `warn` accepted) instead of a string
that silently downgraded a typo to INFO, `mcp.path` is normalised to a rooted route (a bare
`mcp` made the Starlette mount assert, which the app swallowed as "MCP not mounted" -- so MCP was
absent, the PIN was never enforced and the banner advertised a 404), `models.dir` naming a file
is refused, unknown keys are ignored *loudly* (one WARNING listing them), an empty
`config.yaml` is treated as missing (WARNING + regenerated) instead of "every setting silently
default forever", and `save()` fsyncs the temp file before the rename and keeps the previous
file as `config.yaml.bak`. `update.repo` defaults to `null` and the shipped placeholder is
treated the same, so the self-update check reports "not configured" without a network call.

**What pins it.** `tests/unit/test_config_hardening.py`; `tests/unit/test_updater.py` (the
unconfigured check makes no request); `tests/unit/test_watchdog.py` (harness layout).

---

## D32 -- On an open install, routes that change the box need a local caller or the PIN

**Problem.** The shipped default is `server.host: 0.0.0.0` with no `server.api_key`, and
`check_request` returned unconditionally when no key was set. So on the default install anyone
on the LAN or tailnet could, with no credential, `PATCH /api/config` (set `server.api_key`
themselves and lock the owner out; set `hf.token`; disarm `mcp.pin_required`), delete GGUF files
(`DELETE /api/models/{id}?delete_files=true&confirm=true`), install an engine or an app update,
restart the process, queue downloads and kill processes. Meanwhile the MCP `set_config` tool --
same capability, same process -- demanded the PIN. The `confirm=true` flags were footgun guards,
not authorisation. WP17 F4 made the Setup checklist *say* this; it did not close it.

**Decision.** *Reads, inference and residency stay open; changing the box does not.* When
`server.api_key` is unset, a mutating request to `/api/config`, `/api/restart/*`,
`/api/engine/*`, `/api/update/*`, `/api/vram/reclaim`, `/api/downloads*`, or a `DELETE` under
`/api/models/`, `/api/adapters/`, `/api/virtual-models/` is accepted only from a caller on this
machine (loopback peer, or an in-process call with no peer -- the GUI invoking a handler
directly) or with the MCP PIN, sent as `X-MCP-Pin` or as the bearer token (which is how `sfctl`
already sends it). Anything else is **403 `remote_admin_requires_credential`** with a message that
names the two ways in and the setting that lifts the rule (`server.api_key`). With a key set,
nothing changes: the key is the credential everywhere, as before.

Not guarded, deliberately: `/v1/*`, every `GET`, load/unload/unload-all/pin/test/benchmark,
per-model settings, scans, virtual-model creation. JIT loading from any client on the LAN is the
product (LM Studio parity), and a per-model setting is bounded by `ModelSettings` validation and
the engine's flag allow-list.

**Limits, stated.** A peer-address check trusts whatever is on loopback, which behind a reverse
proxy is the proxy: put a proxy behind `server.api_key`. The GUI's own control channel is a
WebSocket, which the same-origin policy does not cover; the gate now refuses a browser upgrade
whose `Origin` host differs from the `Host` header (a page on any site the operator visits could
otherwise drive the panel on `http://<lan-ip>:8080`), with a log line naming the fix for a proxy
that rewrites `Host`.

**Addendum (2026-08-22): the panel applies the same rule.** The GUI calls the box-changing code
in-process (no route, no auth middleware), so on the default install every action above was one
click away for any LAN host on `:8080`. Now each such GUI action -- config saves (every path goes
through `apply_config_updates`), engine install/activate, restarts, VRAM reclaim, lease release,
pin and saved settings (per the D41 addendum), deletes, download queueing, plus the GUI-only
URL-handler and autostart registrations -- runs `require_local_admin`, which with no
`server.api_key` accepts only a loopback viewer (from NiceGUI's client peer) and otherwise refuses
with the same `remote_admin_requires_credential` text. The Setup tab withholds the PIN's reveal
and copy from that viewer too, as `/api/mcp/info` already did. Reads, chat, load/unload stay open;
with a key set, the key is the credential as before. Behind a reverse proxy the peer is the proxy
-- the limit stated above, same answer: put the proxy behind `server.api_key`.

**Also in the same commit.** Load arguments are validated once, in `ModelManager.load` (every
caller): `ctx_size` and `parallel` must be positive and bounded, `kv_cache_type` must be one of
the known types -- `0` used to mean "default" by accident and `-1` reached `--ctx-size -1`.
`ModelSettings` refuses counts of 0, negative device indices and an `engine_tag` with path
characters, and `EngineManager.engine_dir` refuses any tag that is not one plain path component
(the install route's body named a directory under `engines/`). A `messages` element that is not
an object is a 400, not a 500. The non-streamed completion decrements `active_requests` in one
`finally` (a cancellation used to leave it stuck, blocking TTL unload and eviction), and a read
timeout there is a 504 `upstream_timeout` naming `server.request_timeout_s`. `mcp.enabled: false`
really leaves the endpoint unmounted. The MCP `set_config` tool live-swaps the `mcp` section like
the HTTP route; the MCP `delete_model` uses the same "is anything serving these files" guard as
the HTTP route and lists the virtual models removed with the base. `DELETE /api/models/{id}` on
an unloaded model no longer 500s on `supervisor.all()`, a method that never existed. The
image-URL SSRF guard resolves once and connects to the vetted address with the original `Host`
and SNI (the DNS-rebinding TOCTOU, WP17 open item 2).

**What pins it.** `tests/unit/test_api_hardening.py`, `tests/unit/test_vision_fetch.py`,
`tests/unit/test_gui.py` (websocket origin), `tests/unit/test_mcp_access.py` (unchanged: the PIN
still does not reach `/v1` or `/api` reads when a key is set).

---

## D33 -- Bind first, boot after; the boot has phases and every request but liveness waits for the index

**Problem.** Uvicorn binds the port only after the lifespan's startup half returns, and that half
ran the library scan, the orphan sweep, `ensure_engine` (on a fresh box: a ~600 MB download), the
manager start and the download resume. Until all of it finished, `/health` was connection-refused:
the tray said "Starting..." with nothing to show, the watchdog's restart timed out at 120 s on a
cold library, OpenClaw retried into a closed port, and a fresh clone's very first start looked
hung for minutes with no way to ask it what it was doing (WP12 left this as the remaining piece of
"every fresh-box case ends in the Setup tab").

**Decision.**

* **The lifespan yields at once.** It creates the HTTP client, a `BootStatus`, and one background
  task (`_boot`) that runs the slow half in order -- `scanning models` -> `sweeping orphaned
  engines` -> `checking engine <tag>` (`installing engine <tag>: <step> <n>%` while it downloads)
  -> `starting model manager` -> download resume -> `ready`. Each step is individually non-fatal,
  and `finish()` always runs (a failure records `failed: <why>`; a shutdown mid-boot records that
  too), so nothing waiting on the boot can wait forever. Shutdown cancels a boot still in flight
  before the ordinary teardown.
* **`/health` carries `boot: {phase, ready, elapsed_s, error}`** and, until ready, `can_serve:
  false` with `cannot_serve_reason: "still starting (<phase>) ..."`. `status` stays `ok` -- the
  process is alive, which is what liveness pollers ask -- so the tray, the watchdog and the
  `--open` waiter attach as soon as the port answers, and can now show *what* it is doing.
* **Every request except liveness and the docs waits for the first library scan** (the auth
  middleware, bounded by `SCAN_WAIT_S` = 60 s), so a client that connects the moment the port
  answers is not told the library is empty or that its model does not exist. `ModelManager.load` /
  `ensure_loaded` additionally wait for the *whole* boot (bounded by `gateway.load_timeout_s`):
  a JIT load during an engine install waits for the engine rather than failing with "no engine",
  and the streaming path's keep-alives cover the wait. After the bound the ordinary errors apply.
* **The Setup tab shows the phase.** While the boot is installing the engine, the checklist's
  engine row reads the live progress and offers no second Install button (which would have raced
  the first behind the per-tag lock).

**Also in the same commit.** A dead stderr can no longer take the process down through a log
line: the root stream handler detaches on failure instead of printing a traceback to the stream
that just failed (which is how `ValueError: I/O operation on closed file` escaped from
`log.info`), and under `pythonw` no stream handler is installed at all; both uvicorn servers use
`log_config=None` so their own loggers go through the same handlers. The GUI's uvicorn server has
a `timeout_graceful_shutdown` (its signal handler ran first on Ctrl+C and waited without bound
for every NiceGUI websocket, so one wedged browser tab held the whole process). `SF_SUPERVISOR`
is honoured only when a live tray is this process's direct parent, and it is never inherited by
the watchdog or by the replacements the watchdog spawns (a watchdog-spawned server used to exit 75
into the void on its next restart). The API's watchdog-restart client outlives the watchdog's own
120 s budget, so a slow cold start no longer produces two replacements. On Linux, `llama-server`
children are launched through a tiny `PR_SET_PDEATHSIG` shim (a separate single-threaded
interpreter that sets the flag and execs, not `preexec_fn`), the POSIX half of D23. The orphan
sweep re-verifies a pid's identity right before killing it, and the crash watcher tears down a
child that `stop()` overtook while `create_subprocess_exec` was in flight. A model root that is not
a directory right now (a dropped drive) keeps its models in the index as `stale` instead of
removing them, and the TTL sweeper unloads an instance only when its file is *removed* -- its root
was walked and it was not there -- never when it is merely unreachable (WP17 R3).

**What pins it.** `tests/unit/test_lifecycle_hardening.py` (health during boot, a listing waits
for the scan, shutdown mid-boot, the manager waits for the gate, the safe handler, the pdeathsig
prefix, the reclaim recheck), `tests/unit/test_restart_handover.py` (stale supervisor variable,
children never inherit it), `tests/unit/test_registry_sticky.py` (unreachable vs removed; the
sweeper), `tests/unit/test_gui_setup_tab.py` (install phase, no second button),
`tests/unit/test_startup_resilience.py` (readiness after the boot).

---

## D34 -- Process identity is read through venv launcher stubs

**Problem (seen live, WP13, 2026-08-19 04:46).** On Windows a virtual environment's
`Scripts/python.exe` -- the one `uv venv` and `python -m venv` both install, the one every
launcher, the tray and `sys.executable` name -- is a 270 KB *redirector*: it starts the real
interpreter as its child with the same arguments, waits, and the two are tied by a job object, so
killing the stub kills the interpreter. Two D21/D28 rules were written for a tree without the stub:

* The watchdog kills the server's process tree with only its **own** pid excluded. Under the stub
  the watchdog is the server's *grandchild*; the tree kill took its stub parent, the stub's job took
  the watchdog, and `POST /restart` ended with the server dead, the watchdog dead and nothing
  spawned -- the exact outcome D21 was written to prevent, one process further up. Reproduced live
  on 1252/1253: the watchdog log ends at `killing pid N`.
* "The tray launched this server" was decided by looking at the server's *direct parent*, which
  under the stub is a redirector whose argv is the server's own. So the watchdog would not defer to
  the tray, and the new `supervising_tray_is_alive` gate would never see the tray -- both halves of
  D28 quietly off on the deployment they were made for.

**Decision.** *A process's launcher is the nearest ancestor whose argv (past the executable)
differs from its own.* `watchdog.server.launch_parent` and `core.ports.supervising_tray_is_alive`
walk up through same-argv ancestors (bounded, with the pid-reuse guard at every hop) before asking
"is this a tray". And the watchdog's tree kill protects `own_launcher_chain(root)`: its own pid
plus every ancestor of its own between it and the target root -- never the root itself, and never
its own descendants (the next server it spawns is one, and the next restart must be able to kill
it). Verified live after the fix: kill pid 26116 -> `spawned the replacement server (pid 25032)`
-> `POST /restart finished: ok=True`; one server on 1252/8098, the watchdog still on 1253, the old
child gone.

**What pins it.** `tests/unit/test_restart_handover.py::TestKillProcessTreeExclusion` (the chain
protects the direct parent and never the root; `launch_parent` sees through a same-argv stub and
refuses a recycled pid), and the live record above.

---

## D35 -- "Is anything listening?" is answered by connecting, and a port conflict names its port

**Problem.** The first `POST /api/restart/server` after switching the rig to V2 left the rig
without a server for two minutes and then blamed LM Studio. Serve drained, exited 75 and (D28/D34)
deliberately left the watchdog running on `0.0.0.0:1235` for the replacement to adopt. The
replacement's `inspect_running_watchdog()` first asked "is anything on 1235?" by trying to **bind
`127.0.0.1:1235`** -- and on Windows that bind *succeeds* beside a wildcard listener unless the
listener set `SO_EXCLUSIVEADDRUSE`. Verdict: "nothing is listening on port 1235"; the watchdog was
never asked `/health`; adoption was skipped; the wildcard-bind conflict check then correctly said
1235 was in use, and serve exited 3. The tray, which rightly never crash-respawns a port conflict,
assumed the conflict was on **1234**, waited its 120 s grace watching a port that was free, and
reported "Port 1234 is held by another program (LM Studio?)".

**Decision.**

* `ports.port_has_listener(port, host)` **connects**; `inspect_running_watchdog` uses it for the
  "nothing there" gate. Binding answers "can *I* take this address", connecting answers "is someone
  *serving* here" -- different questions, and only the second one is the adoption question. Pinned by
  a test that starts a real `0.0.0.0` listener and asserts the probe gets past the gate.
* The tray's port-conflict path no longer assumes the server port. While the grace runs, if the
  server port is bindable the child is respawned after `PORT_CONFLICT_RETRY_DELAY` (5 s), bounded by
  `MAX_PORT_CONFLICT_RESPAWNS` (3; the counter resets once the server is RUNNING) so a port that is
  genuinely held elsewhere still ends in a report, not a spawn loop. At the deadline the report is
  built from a **fresh probe of all three ports** and names the port that is really taken and its
  holder ("Port 1235 (watchdog) is held by python.exe (pid 25684)"); the "LM Studio?" line is only
  the fallback when nothing can be named.

**Why this was not caught earlier.** The scratch-data-dir restart tests ran serve and watchdog fresh
in the same process tree; the failing case needs a watchdog that *survives* the previous serve --
which is exactly what D28/D34 made happen for the first time on the rig itself.

Tests: `tests/unit/test_ports_preflight.py` (wildcard listener), `tests/unit/test_tray.py`
(retry, bound, report names the real port).

---

## D36 -- The catalog answers "which GPUs, at what quality", and a load never interrupts a stream

**Problem.** Four faults, all visible on the live rig on 2026-08-19, and each of them made the
others harder to see. What the user wanted instead was one sentence: *"these should be the optimal
settings to run on either the 5090s, the 3090s, a single 5090, or all"* -- and, on what to assume
about the machine, *"quality is more important than speed; when possible get max quality on dual
5090s (assume you can fill them both) and dual 3090s (assume both free)"*.

*A loaded model's own VRAM was counted against it.* `HauhauCS/Gemma4-31B-QAT-...-Q4_K_M` (17.4 GB,
`gemma4` iSWA) was resident at ctx 262144 f16 on `[1, 0, 2]`, with the four GPUs at 5.6 / 7.4 / 6.5
/ 22.9 GiB free. Its own catalog rows were computed with `loaded=<everything including itself>`, so
the model that *was* loaded was told it fitted only at 262144 with a **q4_0** cache spread over
three cards:

```
16384  dev=[1,3]   f16  np=1  gen 34.3        131072 dev=[1,3,2] q8_0 np=1
262144 dev=[1,3,2] q4_0 np=1  gen 32.8  <- RECOMMENDED
```

A reload frees its allocation before the replacement takes any, so those rows were judged against
37 GB the row itself would release.

*The per-hardware view was hard-coded, live, and used a second rule.*
`ModelManager.PLACEMENT_MODES` listed `(0, 1)`, `(2, 3)` and "all" as literals -- no single-GPU
mode, and false labels on any other box. Each was planned against **live** free VRAM, so "what can
this model do on the two 5090s" was answered as "...given whatever is on them this second". And it
asked the planner for the largest context that fits, which is not the rule `/api/catalog` used --
so the two surfaces could recommend different loads for the same model on the same hardware.

*No surface looked at KV cache quality.* The auto ladder walked `f16 -> q8_0 -> q4_0` over matched
K/V pairs and the recommendation took the largest window, so **262144 on a q4_0 cache outranked
131072 on f16**. That is backwards for this box.

*And nothing said who was doing what.* Between 21:11 and 21:14 an external client loaded a 27B at
32768 on `[1, 0]`, then the 31B at 262144 on `[1, 0, 2]`, then fetched `/profiles` for every model
in the library (three planner walks each, at INFO). Not one log line named a requester, and nothing
stopped a `test_model` call from doing a full planner-sized load, on whatever cards were free,
while other agents were mid-conversation.

**Decision.**

**1. A resident model's rows are credited with its own footprint.** `catalog.CreditedProbe` adds
`Planner.instance_footprint(instance)` back to each GPU's free bytes and the entry is planned with
itself removed from `loaded` -- exactly the view D30's `reload_of` takes when the same reload is
really performed, so the table and the load cannot drift apart. The credit reaches
`slots_for_plan` too, which measures capacity off the planner's own probe: a credited fit with an
uncredited capacity would have advertised one slot. `_instance_footprint` became public
`instance_footprint` because three surfaces now credit the same figure.

**2. Hardware modes are derived, idle, and ordered by what is asked for first.** New
`core/placements.py`. `hardware_modes(gpus)` reads the inventory: `dual_<best class>`,
`dual_<second class>`, `all_gpus`, `single_<best class>` -- on this rig `dual_5090` `[0,1]`,
`dual_3090` `[2,3]`, `all_gpus` `[0,1,2,3]`, `single_5090` `[0]`, deduplicated on the device set so
a two-card box gets one dual mode rather than two names for it. Each mode is planned against **its
own cards, idle** ("assume you can fill them both"); `headroom_fraction`, `reserved_mb` and
`excluded_devices` still apply, because those describe memory that is never ours. What stands in
the way *right now* is reported separately -- `fits_now` (planned at exactly the slots `load_args`
asks for), `would_evict`, `fits_now_ctx` -- plus a `ranking` of `fastest` / `largest_context` /
`cheapest`.

`placements[0]` becomes the entry's `recommended`: a complete load recipe naming the GPUs, with
`recommended_basis` reading `2x RTX 5090: f16 KV, highest ctx 131072, 1 slot`. The per-context
table stays as the drill-down and its flag is renamed `recommended` -> `best_now`, which is the
narrower claim it was always making: "this row loads on the machine exactly as it stands".

Measured, reference rig, 17.4 GB Gemma-4 31B: `dual_5090` 131072/f16, `dual_3090` 65536/f16,
`all_gpus` 262144/f16, `single_5090` 32768/f16 -- no quantized cache anywhere, where the live
catalog had recommended 262144/q4_0.

**3. Quality first, and the KV ladder stops at a cache that works.** One function,
`catalog.choose_row`, serves the per-context table and every placement. Order: **the best KV cache
quality that reaches the context floor** at one slot or more, then the highest context at that
quality, then whatever slots that placement sustains -- reported, never bought.

The auto ladder is now `f16/f16 -> q8_0/q8_0 -> q8_0 K + q4_0 V`. **Symmetric `q4_0` is gone from
every automatic path.** K and V are not equally sensitive: llama.cpp discussion #23470 measures a
q4_0 **K** cache alone dropping Qwen2.5-7B to **11.7%** token agreement with its f16 self, a q4_0
**V** cache alone as nearly free, and the matched `q8_0/q8_0` pair at KL 0.0018. So fanning out
over matched pairs skipped the one useful cheap rung and offered the one that ruins the model --
and because a q4_0 cache reaches the biggest window, that was the rung the catalog recommended. A
q4_0 K cache remains reachable by asking for it explicitly, like every other explicit value.

Families differ by a factor of ten, and `core/kv_sensitivity.py` carries the measurements (KL over
top-40 logprobs, ~250k tokens, against a BF16 GGUF with an f16 KV cache;
`localbench.substack.com/p/kv-cache-quantization-benchmark`):

| Family | q8_0 KV | q4_0 KV | Verdict |
| --- | --- | --- | --- |
| Gemma-4 31B dense (`gemma4`) | 0.108 | 0.524 | sensitive |
| Gemma-4 26B-A4B MoE (`gemma4`) | 0.377 | 1.088 | sensitive |
| Qwen 3.6 (`qwen35` / `qwen35moe`) | 0.024 | 0.039 | tolerant |

A tolerant family may take `q8_0` when it buys a **full doubling** of the window; a sensitive one
never does, and **every unmeasured architecture is treated as sensitive**. Three measurements on
two families do not describe a library of forty models, and guessing "tolerant" is the guess whose
failure mode is a server that quietly answers worse. Gemma's iSWA layout is the plausible reason it
minds: only every sixth layer holds the full window, so each retained KV element carries more of
the model's memory of the conversation.

D20's rule survives as `planner.preference: "throughput"` -- the largest window at or above the
floor, preferring one that also sustains two slots. It is right for a host serving many short
conversations. The default is `"quality"`, and the Setup tab exposes the switch.

A record that *pins* `kv_cache_type` (the two Gemma-4 QAT records on this rig pin `q8_0`) gets a
`quality_notes` entry naming what the pin costs, and every placement carries an `if_unpinned` block
showing what the same cards reach at f16. Nothing here rewrites a saved setting: an explicit value
is honoured verbatim, and the only correct action is to show the size of the choice.

**4. `devices` is a load argument.** `load_model(..., devices=[0, 1])`, `POST
/api/models/{id}/load` and `ModelManager.load` accept a one-shot placement, applied as a
`device_override` on a *copy* of the record so the persisted settings are never touched. A CUDA
index this box does not have is a `400` naming the parameter, validated before the planner or the
supervisor is reached. `kv_cache_type_v` rides the same path, for the ladder's asymmetric rung; it
appears in `load_args` only when it differs from K, because the tool defaults V to K.

**5. Busy-aware loading and testing.** Three parts:

* **Every load carries a `source`** -- `"mcp:load_model"`, `"jit:/v1/chat/completions"`, `"gui"`,
  `"autoload"`, `"benchmark"`, `"restart-resume"`... It is stamped on `InstanceInfo.loaded_by` (so
  `/api/status`, `server_status` and the dashboard show it) and carried into the `load planned`,
  `model_spawn` and `model_ready` log lines. "A 262144-token model appeared on three GPUs" is not a
  diagnosable event without a requester.
* **The eviction ladder never kills a busy model.** It already skipped `active_requests > 0`; what
  was missing is that the *refusal* now says so. A rejection blocked by a serving model carries
  `busy_models` (id and in-flight count), `retry_after_s = 15`, and a suggestion naming it with
  "pass force=true to evict it anyway". `force=true` is the only override and a JIT load can never
  set it: an inference request arriving mid-stream for somebody else must queue or be refused,
  never win by killing the stream. A refusal that is *not* about a busy model carries no
  `retry_after_s`, because "try again later" is bad advice when nothing is going to change. An
  instance that is still `loading` is not an eviction candidate either -- it has not taken the VRAM
  its plan promises, so evicting it frees a figure that does not exist yet.
* **`test_model` is one-at-a-time, idle-only, small, and leaves the rig as found.** A second
  concurrent call is refused rather than queued (a queued smoke test answers about a server that
  has since moved). Any instance serving a request, any load in flight, or a running benchmark
  refuses the call with `retry_after_s`. A cold model is loaded at
  `min(models.default_ctx, n_ctx_train)` with **one** slot -- a canned one-sentence prompt proves
  nothing extra at 262144 -- and unloaded again unless `keep_loaded=True` or it was already
  resident. The result says what it did: `loaded_for_test`, `unloaded_after`, `ctx_size_used`. The
  docstring says out loud that this is a smoke test and `load_model` is how you pre-warm.

`/health` and `server_status` gain `busy: {active_requests, busy_models, loading, testing}`, built
from in-memory state only -- the watchdog polls `/health` constantly, so it must not touch NVML or
a child. The MCP instructions tell an agent to read it before loading or testing on a shared box.

**Also.** `placement_profiles` and the placement report run a `Planner(log_plans=False)`. Planning
is not loading (D16/D20 made that rule for the catalog); one `/profiles` sweep of the library was
several hundred INFO lines describing loads nobody requested.

**Cost, measured.** A 33-model catalog against four hardware modes builds in **806 ms** (was ~150
ms), behind the existing 20-second cache and off the event loop, with a new INFO line above
`SLOW_BUILD_MS = 1000`. The compact `list_models` payload roughly doubles per model. It is held
down deliberately: the non-recommended modes carry settings and devices but **no `load_args`**
(repeating one recipe per mode costs ~40% of a compact entry to describe four loads of which at
most one happens), `would_evict` collapses to a count, and the default `limit=25` stands. The
budget test is re-anchored with that reasoning written into it.

**What pins it.** `tests/unit/test_placements.py` (the four modes and their dedupe on other
inventories, each mode against its own idle cards, `fits_now` / `would_evict`, the rankings, the
pinned-KV pair, and that no placement ever writes back to a record),
`tests/unit/test_catalog.py` (the credit and that another model does not get it, the quality rule's
branches, both KV-sensitivity verdicts, the throughput preference, `best_now`),
`tests/unit/test_busy_loads.py` (the requester on every load, a busy model is never evicted, the
refusal names it, `force` is the one override, `busy_snapshot`, and every `test_model` rule),
`tests/unit/test_load_retry.py` (`devices` is one-shot and never persisted; a bogus index is a
structured 400), `tests/unit/test_gui_models_tab.py` (the "Optimal settings" lines).

**What remains.**

* **The KV numbers are three measurements on two families**, from one published benchmark, and this
  rig has reproduced none of them itself. The conservative default (unmeasured = sensitive) is what
  makes that acceptable; measuring a third family is the next thing that would improve it.
* **`fits_now` is an estimate of an estimate.** It plans against live free VRAM at build time and
  the catalog is cached for twenty seconds, so a `true` can go stale exactly as `fits` always
  could. The refusal path is what makes that safe, and it now carries `retry_after_s` when the
  cause is transient.
* **The per-context table still shows the planner's own minimal placement**, so "8 slots at 16384
  on two 5090s" is visible through `placements` but not in `options`. Giving every context tier a
  per-mode row would multiply the payload by four to answer a question `placements` already
  answers.
* **`busy` cannot see a request that has not reached the supervisor yet.** A load and a request
  racing inside the same millisecond can still interleave; D29's gate bounds the damage, and
  nothing here promises more than "the box was idle when we looked".

---

## D37 -- How many slots are worth running, and loading at exactly the context you asked for

**Problem.** Two things the server could not say, and a user's sentence that named both:

> *"do research and implement recommended parallel for processing (example benchmark); also include
> a Load Recommended option, for 64k, 128, 256, 512k. So it can simply specify the model and context
> needed, and the server works the rest, or returns an error if it can't load the requested context
> for some reason."*

*The slot count was arithmetic pretending to be advice.* D17 gave every placement a `max_parallel`
-- the smaller of an exact VRAM bound and a **knee** derived from memory bandwidth -- and said so
out loud: `CTX_FILL_FRACTION = 0.5` and the MoE derate of 0.5 are "deliberate approximations in the
safe direction, not measurements -- the real curves want a benchmark this rig has not run". That
number then went into `load_args.parallel`, so every catalog row was quietly advising a slot count
nobody had ever checked. `scripts/bench_parallel.py` existed to check it and had to be driven by
hand against an already-loaded model.

*And loading meant knowing the hardware.* Every path into a load was shaped around "which cards, at
what settings". An agent whose actual problem is "my transcript is 200k tokens long" had to read a
table, pick a placement, and hope. Worse, D14's context ladder is a *halving* ladder: ask for a
window that does not fit anywhere and you silently get half of it, which for a transcript-sized
request is a failure discovered mid-conversation.

### The research

Batch-1 decode is memory-bandwidth bound. Each step streams the active weights through the memory
system **once** regardless of how many sequences are in the batch, then reads each busy slot's KV
cache on top. So while weight traffic dominates, N busy slots amortise one weight read across N
tokens: aggregate throughput rises close to linearly and per-slot throughput falls slowly. Once the
N slots' KV reads match the weight read, another slot buys nothing and costs VRAM and latency --
D17's knee, `active_weights / (ctx_fill * kv_bytes_per_token)`.

Sources, all of which say the useful slot count is workload- and hardware-specific:

* <https://github.com/ggml-org/llama.cpp/discussions/4130> -- llama.cpp's own explanation of why
  continuous batching scales the aggregate.
* <https://markaicode.com/architecture/llamacpp-system-design-architecture-1158/> -- a system-design
  writeup measuring roughly **3.8x aggregate** over sequential decode from slot batching on a single
  T4.
* <https://manpages.debian.org/testing/llama.cpp-tools/llama-server.1.en.html> -- `--parallel`,
  `--cont-batching`, `/slots`, and the `/metrics` counter `llamacpp:n_busy_slots_per_decode`.
* <https://www.promptsicle.com/tips/boosting-llama-server-performance-with-batch-settings/> --
  the `--batch-size` / `--ubatch-size` dimension that sits underneath all of it.

MoE complicates it: expert fan-out grows with batch (at batch N roughly
`min(N * n_expert_used, n_expert)` experts are touched), so weight traffic stops being flat and the
knee arrives sooner. D17 derates it by half. That is still a guess, and a guess a measurement
replaces.

**Quality is not involved.** A slot count changes nothing about the answer a model gives; what each
conversation gets is `ctx_per_slot`, and that is per slot by construction (D4). "Recommended
parallel" is a throughput-and-latency decision only, which is why it can be benchmarked and why it
never touches the quality-first KV choice (D36).

### Decision

**1. `recommended_parallel` beside `max_parallel`, everywhere.** `max_parallel` is how many slots
**fit**; `recommended_parallel` is how many are **worth running**, and it is the number
`load_args.parallel` now asks for -- on every `options` row, every `placements[*].optimal` and the
entry's `recommended`. `fits_now` is planned at that same number, so the table still describes
exactly the call it hands the caller. `recommended_parallel_basis` is `"measured"` or `"estimated"`,
and `"measured"` is meant literally: a sweep on other cards, or at another context, is not used,
because the knee is set by the KV bytes a busy slot reads per step.

**The rule** (`core/parallel.py`): the largest level in 1 / 2 / 4 / 8 whose per-stream rate is still
at least **65%** of the solo rate **and** whose aggregate is at least **1.15x** the level below,
walked upward and stopping at the first failure. Stopping is the substantive part -- past the knee
the aggregate flattens, so a higher level can still beat its own depressed predecessor by 15%, and
promoting it would mean the run arguing against itself. The two thresholds are a stated *priority*,
not a measurement: aggregate can always be bought with per-stream latency, and this server's answer
is that a conversation at under two-thirds of its solo speed is one the user notices.

**With no measurement, `recommended_parallel` equals `max_parallel`,** because D17's knee is already
folded into `max_parallel`. That is written down rather than hidden: the field earns its keep the
moment a run exists, and until then the compact catalog view drops both keys so they cost nothing.

**2. The parallel benchmark** (`core/parallel_bench.py`), productising the D17 harness. It loads the
model **once** at the placement being measured with as many slots as it holds, then fires N
concurrent non-streamed completions for each N and records per-stream t/s (median of the streams'
own rates, not the aggregate over N -- one straggler would otherwise read as a knee), aggregate t/s,
p50/p95 latency, and a row per level in the new `parallel_observations` table (migration 005).

Three method decisions:

* **One load for the sweep.** Reloading between levels measures four cold caches and spends most of
  the run on loading. The one load asks for the *maximum* slots, and levels above what was launched
  are reported as **not measured**: N requests against fewer than N slots measure a queue.
* **Unique prompts per stream.** llama.cpp routes by prompt similarity (`--slot-prompt-similarity`,
  which D17 raises above one slot), so identical prompts would land several streams on one slot's
  prefix cache and the level would report the cost of a cache hit.
* **The engine's decode counter is the control.** A client that sends 8 requests to a 1-slot server
  still gets 8 answers, just serialized, and no throughput table can tell those runs apart. Each
  level takes the delta in `llamacpp:n_decode_total` and reports `completion_tokens / decode_steps`
  as `achieved_batch`. The `n_busy_slots_per_decode` gauge is recorded too but is *not* the control:
  it is a cumulative average over the child's whole life, so every earlier level drags it down. A
  run whose batch never rose says so in its notes.

It refuses while anything is serving, loading, testing or benchmarking (D36's rule, and here the
contention *is* the measurement), shares the placement benchmarker's one lock through a new
`Benchmarker.exclusive()`, is cancelable between levels, and leaves the rig as it found it: a model
it loaded is unloaded, a model that was resident is reloaded with the plan it had.

**3. `load_recommended(model, ctx)`.** Name the model and the window. The manager walks the hardware
modes in headline order (`dual_5090` -> `dual_3090` -> `all_gpus` -> `single_5090`, or
`prefer_modes`), asks the planner for **exactly** that context per slot under the quality-first KV
ladder with `parallel = recommended_parallel`, and loads the first placement that fits. **Every mode
is tried with eviction off before any is tried with it on** -- D14's "a roomier window must never be
the reason somebody else's model is unloaded", extended from context rungs to hardware modes -- and
a busy model is never a candidate either way.

**This is the one load path that is strict about context**, and that is the whole point of it. A
window that does not fit is a `507` listing, per mode, the largest context that *would* work and
what is in the way, with `retry_after_s` only when a model that is serving right now is the cause.
Above `n_ctx_train` it is a `400` carrying the number that would be accepted. `kv_min` is the other
half of strict: "give me 262144, but not at the cost of the cache" refuses a placement that only
reaches the window by quantizing and walks on to one that can afford f16.

Surfaces: MCP `load_recommended` / `benchmark_parallel`, `POST
/api/models/{id}/load-recommended`, `POST /api/models/{id}/benchmark-parallel` (a background job on
the existing `/api/benchmark/jobs` machinery), `GET /api/models/{id}/parallel-observations`, and in
the Models tab a slot line per mode, a "Measure parallel" button per mode, and 64k / 128k / 256k /
512k buttons plus a free-form field. The MCP benchmark tool is **synchronous**: there was no
existing MCP benchmark job/poll pair to mirror, and adding two tools so an agent can poll something
it runs once per model is worse than one tool that answers.

### The example run

Measured 2026-08-19 on the reference rig, on a scratch instance (ports 1256/8102/1257, its own data
directory, engine `b10425` copied read-only) so the live server was never touched. Both 5090s were
busy serving the live rig's embedding models, so this used the free 3090s.
`Qwen2.5-1.5B-Instruct-Q4_K_M`, **one RTX 3090**, 8192 tokens/slot, f16 KV, launched with 8 slots,
512-token prompts, 192 generated tokens per request:

| N | per-stream t/s | aggregate t/s | p50 (s) | p95 (s) | achieved batch |
| --- | --- | --- | --- | --- | --- |
| 1 | 302.8 | 302.8 | 0.41 | 0.41 | 1.00 |
| 2 | 225.3 | 425.3 | 0.46 | 0.49 | 1.84 |
| 4 | 134.5 | 436.0 | 0.83 | 1.00 | 3.46 |
| 8 | 83.3 | 576.9 | 1.57 | 1.77 | 6.03 |

**`recommended_parallel: 2` (measured)** -- *"at 4 each stream drops to 44% of its solo speed (floor
65%)"*. **The estimate for this placement was 8.** Reproduced across three runs within 2%.

This is the whole argument for the work package in one table. `achieved_batch` rising 1.00 -> 6.03
proves the requests really shared decode steps. And the aggregate **never stops climbing** -- 8
slots move 1.9x the tokens 1 slot does -- while a single conversation collapses to 27% of its solo
speed. A rule that maximised aggregate would have picked 8 and every user would have experienced a
model three times slower than the card can run it. D17's knee, which is a bandwidth crossover,
lands at 8; the knee in the *useful* sense is at 2. Only a measurement could have found that gap.

The same model on **two RTX 3090s** at that placement's own optimal (32768/slot, f16) holds two
slots, so the sweep is two rows (301.7 -> 230.5 per stream, 301.7 -> 433.8 aggregate) and the answer
is `2 (measured)`, *"2 is the most this placement can hold"*, with levels 4 and 8 reported as not
measured. Afterwards `/profiles` showed `dual_3090` at basis `measured` and every other mode still
`estimated` -- the strictness above, working on live data.

**An incidental confirmation.** Before run B, `/profiles` predicted `dual_3090` at
`est_gen_tps 308.0` for this model at 32768/f16. Run B measured **301.7 t/s** at one stream on that
exact placement -- 2% out. D22's throughput estimator, with its 1.5 ms per-token floor, is doing
well on a case (a 1.5B, where the roofline alone claimed 927 t/s) that used to be its worst.

Live checks of the load path on the same instance: `{"ctx_size": 32768}` loaded at exactly 32768 on
`[0, 1]` with 2 slots and f16; `{"ctx_size": 65536}` against a 32768-token model returned the `400`;
`{"ctx_size": 12345, "prefer_mode": "dual_3090"}` loaded at exactly 12345 on `[2, 3]` with 6 slots.

**Two bugs the live run found that no unit test would have.** The busy re-check *inside* the
exclusive lock read the `benchmarking` marker that lock had just set for this very run, so every
sweep refused itself with a message about itself -- the first real run failed on it. And the
observation rows recorded a null `engine_tag`, because `InstanceInfo.engine_tag` is only set when a
record *pins* an engine, which is almost never; migration 005 stores that column precisely so a run
can be retired when the llama.cpp build changes, and the code was not keeping the promise. Both are
now pinned by tests that name the incident.

**`scripts/bench_parallel.py` is kept as-is**, not rewritten as a thin CLI. It is a standalone
single-file harness that talks to a *running* gateway over HTTP and needs nothing from this
codebase, which is exactly what makes it useful against a server this build does not control -- an
older install, a different machine, a hand-launched `llama-server`. Its header now points at the
in-server benchmark as the thing to use here.

### What pins it

`tests/unit/test_parallel.py` (the rule's branches, the walk-stopping-at-the-knee case, the
thresholds, the strict definition of "measured"), `tests/unit/test_parallel_bench.py` (the busy
refusal, the shared lock, the decode-counter control, leave-as-found in all three of its shapes, the
two live bugs), `tests/unit/test_load_recommended.py` (the mode walk, fits-now before eviction,
never a busy model, the trained-window 400, the refusal's per-mode detail, `kv_min` both ways),
`tests/unit/test_db.py` (the new table and its device filter), `tests/unit/test_catalog.py` and
`tests/unit/test_placements.py` (the column, that a sweep on other cards does not steer it, that
`load_args` follows the recommendation), `tests/unit/test_catalog_routes.py` and
`tests/unit/test_mcp.py` (the surfaces), `tests/unit/test_gui_models_tab.py` (the slot line and the
context buttons).

### What remains

* **One model measured, on one card.** A 1.5B is the *easy* case: its weights are small, so weight
  traffic stops dominating early and the knee is near. A 30B dense and a big MoE would test the
  interesting half of the curve, and the MoE derate of 0.5 remains entirely unmeasured.
* **The thresholds are a policy with no knob.** 65% and 1.15x are hard-coded. A host that genuinely
  wants maximum aggregate throughput (D20's `planner.preference: "throughput"` audience) should
  arguably get different ones, and does not.
* **A measurement never expires.** Rows carry `engine_tag` and `gpu_class` so a stale run *can* be
  retired, but nothing retires one yet: swap the engine for a build with different batching and the
  old sweep goes on steering the recommendation. The throughput table solved the same problem with
  `estimator_version` (migration 004) and this one should follow.
* **`load_recommended` re-plans between deciding and loading.** The mode walk plans, then
  `manager.load` plans again with the winner's arguments, so a VRAM change in that gap turns into a
  refusal rather than a wrong load. D29's gate bounds it; it is still two decisions where one would
  be better.
* **`ctx_size` is not remembered.** Asking for 262144 loads at 262144, and the next unqualified load
  goes back to the planner's ladder. That is deliberate -- this path never writes settings -- but a
  user who wants 262144 *every* time still has to pin `ctx_size` in the model's settings.

---

## D38 -- The engine's newer features, each one measured before it was switched on

**Problem.** `b10425` shipped a pile of things StudioForge was not using: MTP and n-gram
speculative decoding (`--spec-type` grew from two useful values to eleven), tensor parallelism
(`-sm tensor`), a host-RAM prompt cache (`-cram`), GPU-side sampling (`-bs`), an explicit unified-KV
switch. Two of them are large wins on this rig. One of them is a loss. Nothing in the codebase
could tell them apart, and worse, nothing could tell whether a flag it passed was even *read* --
D2 recorded `b10425` accepting the renamed `--draft*` family and silently ignoring it, which looks
exactly like a feature that does not help.

Everything below was measured on a **scratch** instance: the engine copied read-only out of the
live data directory, port 1258, the 3090 pair (which `nvidia-smi` showed idle), and every child
killed and verified gone afterwards. The live rig was never touched.

### 1. A flag is passed only when the active engine advertises it

`EngineFeatures` parses each build's own `--help` into `engines/<tag>/features.json`: the flag list
**and the value lists** -- `--split-mode {none,layer,row,tensor}`, `--spec-type`'s eleven types,
`--flash-attn [on|off|auto]` -- plus the defaults that decide what we must *not* re-emit
(`--spec-draft-n-max 3`, `--cache-ram 8192`, `--ctx-checkpoints 32`). Warmed at boot by
`ensure_engine`, readable synchronously next to the binary so the supervisor needs no dependency on
the engine manager, and readable *without spawning anything* by the capabilities report.

The important case is the one that is easy to get wrong: **"the help could not be read" is not "the
engine has nothing".** An unknown build advertises nothing and the launch falls back to the flag
surface that predates this gating -- a draft model still drafts, no new flag appears. Guessing
either way is how a feature silently stops working.

### 2. `spec_type: auto` -- MTP where it exists, n-grams where they pay, nothing otherwise

Speculative decoding is distribution-preserving: the draft proposes, the full model verifies,
rejected tokens are resampled from the true distribution. It is the rare feature that is pure speed
with no quality cost, so `auto` is allowed to turn it on by itself. The ladder:

1. `draft-mtp` when the GGUF carries `nextn_predict_layers >= 1` -- the model's own head, no second
   model, no extra VRAM;
2. `draft-simple` when a `draft_model_id` is attached (the previous behaviour);
3. `ngram-mod` for thinking and MoE models -- draftless, ~16 MiB of host state;
4. `none`.

Measured on **Qwen3.8-27B Q5_K_S** (`qwen35`, `nextn_predict_layers=1`), one RTX 3090, 8k context,
four *distinct* 256-token prose prompts, greedy, `cache_prompt: false`:

| `--spec-type` | tok/s | vs none | acceptance (`draft_n_accepted/draft_n`) |
| --- | --- | --- | --- |
| `none` | 37.75 | -- | -- |
| `draft-mtp`, n_max 3 | **50.70** | **+34.3%** | 0.528 |
| `draft-mtp`, n_max 4 | 47.48 | +25.8% | 0.446 |
| `ngram-mod` | 37.91 | +0.4% | no drafts emitted at all |
| `draft-mtp,ngram-mod`, n_max 4 | 47.05 | +24.6% | 0.446 (draft counts identical to mtp alone) |

On a code-rewrite turn the same model measured +95% with `draft-mtp` and +114% with `ngram-mod`.
That inversion is the reason for a ladder rather than one answer: the strategies are good at
different things, and `ngram-mod` is free when it is wrong. Combining them added nothing
measurable, so `auto` picks exactly one.

**`ngram-mod` clears the "does not slow a non-repetitive request by ~10%" bar with room to spare**
(+0.4%, and `timings.draft_n` is null on unseen prose -- it does not even try), so it ships on for
thinking/MoE models. On the 1.5B the same check gave parity on the first, cold request.

**Benchmarking trap worth writing down.** `ngram-mod` learns from what it has already generated.
Sending the *same* prompt three times measured **+751%** on the 27B and +20% on the 1.5B; four
distinct prompts measured +0.4%. Every ngram number in this section is from distinct prompts.

**`--spec-draft-n-max` is 3, not 16.** The constant here was 16 under a comment claiming that was
the b10425 default; the help text says `(default: 3)`. Depth is not free: at n_max 4 acceptance
fell from 0.528 to 0.446 and throughput with it, because every extra rejected token was verified
for nothing. The flag is emitted only for `draft-*` types -- the n-gram types read
`--spec-ngram-*-n-max` instead, and emitting it there would be a flag that looks like it does
something.

**Reading it.** `/props` is still not to be trusted (it reports `speculative.types: "none"` while
drafting). `/slots[].speculative` came back `true` for `ngram-mod` too, so it means "configured",
not "working"; only `timings.draft_n` / `draft_n_accepted` mean working. Both signals were read off
the scratch loads above; `/props` was not relied on for any of them.

**Amendment (2026-08-23): `auto` is off above four slots.** Every measurement above is a *single
stream* (`--parallel 1`), where decode is memory-bound: the weights are read to produce one token
regardless, so the drafted tokens are verified almost for free and the win is real. That reasoning
inverts under concurrency. A gauntlet run loaded Dark-Scarlett-27B at `--parallel 8` and `auto`
still chose `draft-mtp` (it saw the MTP head, not the slot count) -- but eight concurrent streams
already saturate the GPU, so the drafted-then-rejected tokens are pure extra compute that slows
*every* request. `resolve_spec_type` now takes the launch's slot count and returns `none` from
`auto` above `SPEC_AUTO_MAX_SLOTS = 4` (the same "many slots" line the batch size uses). An
explicit `spec_type` is still honoured at any concurrency -- a benchmark that wants to measure
speculation at eight slots sets it and means it. The crossover is model- and hardware-specific;
four is a deliberately conservative default, not a measured knee, and is the natural thing to
calibrate per model later.

### 3. Tensor split is opt-in, gated, and was *slower* here

Qwen2.5-1.5B Q4_K_M, two RTX 3090s, 8k context, three runs each, median:

| placement | generation | prompt |
| --- | --- | --- |
| one 3090 | 352.5 tok/s | 2803.6 tok/s |
| two, `-sm layer` | 344.4 tok/s | 2722.5 tok/s |
| two, `-sm tensor` | **294.3 tok/s** | **1182.0 tok/s** |
| two, `-sm row` | fails to load | -- |

A small model is tensor mode's worst case -- the per-layer all-reduce is fixed while the halved
weight read shrinks with the model -- so a 31B may well win. That is precisely why this is a
*benchmark dimension* (every multi-GPU mode now has a `-tensor` variant for an eligible model) and
not a default. `layer` stays the default and `split_mode: auto` resolves to `layer` unless every
gate passes.

`throughput.estimate` grew a `split_mode`: layer sums the per-device times (a pipeline), tensor
takes their `max` and adds `n_layer * T_TENSOR_SYNC_S` with `T_TENSOR_SYNC_S = 60e-6` -- two PCIe
all-reduces per layer, an approximation calibrated to reproduce the crossover above, not a
measurement of this bus. Prefill is deliberately modelled *identically* for both modes: a parallel
roofline would advertise a prefill win and the rig measured a 57% prefill loss, so claiming either
would be inventing a constant.

**The gates**, all checked before the child is spawned, because llama.cpp enforces one of them by
exiting -- `-fa off -sm tensor` dies with `SPLIT_MODE_TENSOR requires flash_attn to be enabled`:
at least two devices; the engine lists `tensor`; flash attention on; an unquantized KV cache; a
dense, non-hybrid model. An explicit `tensor` that fails a gate is **refused with the reasons**;
`auto` downgrades to `layer` and records why on the plan. Someone who typed "tensor" and quietly
got "layer" would go on to benchmark the wrong thing.

Two findings the gates are *not* derived from a crash: `-sm tensor` with a `q8_0` KV cache
**started and answered correctly** on b10425 (upstream documents quantized KV as unimplemented
there, so declining it is StudioForge policy and is the gate most likely to be relaxed), and
`--backend-sampling` under tensor mode logs `backend sampling not supported with
SPLIT_MODE_TENSOR; using CPU` and carries on.

`-sm row` is dead on CUDA: `error loading model: device CUDA2 does not support split buffers`. It
stays in the `SplitMode` type only because the engine still lists it; it is documented in
LIMITATIONS.

### 4. Unified KV: verified, and deliberately turned *off* -- explicitly

D17 left this "unverified, and not recommended anywhere until it is". Measured with the 0.5B,
`--parallel 2 --ctx-size 16384`:

| | engine log | `n_ctx` per slot | VRAM | one 12k-token request | two concurrent 12k requests |
| --- | --- | --- | --- | --- | --- |
| nothing passed | `kv_unified = 'false'` | 8192 | 997 MiB | **400** up front, naming the limit | **400** each, up front |
| `--kv-unified` | `kv_unified = 'true'` | 16384 | 1005 MiB | **accepted** | **500 "Context size has been exceeded"**, mid-generation |
| `--no-kv-unified` | `kv_unified = 'false'` | 8192 | -- | 400 | -- |

So the working assumption going in was wrong in a useful way: StudioForge passing nothing does
**not** leave unified on. The engine's help says "default: enabled if number of slots is auto", and
StudioForge always passes an explicit `--parallel`, so the partitioned pool was already what
happened -- by accident of another setting.

**Decision: keep the partitioned pool, and say so.** Multi-slot launches now pass
`--no-kv-unified` explicitly. That makes the catalog's `ctx_per_slot` literally true rather than
true-by-coincidence, and it survives an engine release changing its default. The quality-first
argument decides the tie between the two things worth wanting: unified buys a lone conversation the
whole pool at the same VRAM, but it buys it by over-committing, and the way an over-commit surfaces
is a **500 during generation** on an agent that had already spent a minute thinking. A 400 before
any work starts is the better failure. `kv_unified: true` remains a per-model opt-in for a
single-user long-context model, now documented with the number above instead of a shrug.

Same VRAM either way is now measured, not assumed: 997 vs 1005 MiB.

### 5. `--cache-ram` on by default; `-ub` measured and left alone

`--cache-ram` keeps evicted prompt prefixes in *system* memory. It is the other half of
`--cache-reuse` -- reuse recovers a prefix still in the slot, this recovers one that left it, which
is exactly what happens to an OpenClaw agent's system prompt while another model borrows the slot.
Measured VRAM at `-cram 8192` and `-cram 32768`: **identical, 1492 MiB**. No token can change. On
by default at `min(32 GiB, 25% of system RAM)`.

`-ub` was measured and deliberately **not** turned on (1.5B, one 3090, 5166-token prompt):

| `-ub` | prompt processing | VRAM |
| --- | --- | --- |
| 512 (default) | 15232 tok/s | 1492 MiB |
| 1024 | 17307 tok/s (+13.6%) | 1562 MiB (+70) |
| 2048 (with `-b 2048`) | 18061 tok/s (+18.6%) | 1702 MiB (+210) |

The compute buffer grows with `-ub` and `planner.compute_overhead_fraction` does not model it.
Raising `-ub` globally would make every VRAM estimate optimistic by roughly that much, scaled to
the model -- on a GPU-only server that turns a fit into an OOM. So `engine.ubatch_size` exists,
defaults to unset, and the benchmark can measure it (`ubatch_sizes=(1024, 2048)`); making the
planner ubatch-aware is left for whoever owns `planner.py` next.

**Amendment (2026-08-23): the automatic many-slots raise, now that the planner is aware.** D40
made `Planner.estimate` charge `ubatch_scratch_bytes` for the micro-batch, which removed the OOM
objection above ("the error direction is now a refused context, not an OOM"). The one piece still
missing was an *automatic* raise -- `engine.ubatch_size` had to be set by hand. A gauntlet run at
`--parallel 8` made the case: eight cold slots re-prefilling a shared prompt is exactly the large
combined prefill a bigger micro-batch speeds up, and it was running at the engine's 512. So
`engine.ubatch_many_slots` (default **1024**, the measured +13.6% rung) now applies above
`UBATCH_MANY_SLOTS_THRESHOLD = 4` slots, through one policy (`planner.effective_ubatch`) that both
`Planner.ubatch_for(record, slots)` and `Supervisor.ubatch_for(record, slots)` call -- so the
micro-batch the estimate charges is the one the child launches with, at the slot count it commits
to (`max_slots_by_vram` re-estimates per candidate, so the raise cannot under-charge). A
single-stream load is byte-identical; an explicit per-model or `engine.ubatch_size` still wins;
`null` turns it off. 2048 (+18.6%) is available and worth a per-rig measurement before adopting as
the default -- 1024 is the conservative starting point, not a measured knee for this hardware.

`--backend-sampling` stays off: b10425 labels it experimental, and under the quality-first rule
"experimental with no measured quality claim" is itself the reason.

`--sleep-idle-seconds` is never passed -- StudioForge owns model lifetime through TTL, and a second
idle timer inside the child would unload state the supervisor believes is resident. `--fit off`
stays (D11). `--flash-attn on` stays, and tensor mode now depends on it.

**What pins it.** `tests/unit/test_engine_features.py` parses a *verbatim* trimmed excerpt of
b10425's help (`tests/unit/data/b10425_help_excerpt.txt`) -- a parser tested only against a fixture
its author invented would pass and then drop every flag in production.
`tests/unit/test_supervisor_features.py` covers the whole spec-type matrix, the refusal for a type
the engine lacks, the tensor gates one by one, refuse-vs-downgrade, and that no new flag reaches an
engine that does not advertise it. `test_throughput.py` pins the tensor arithmetic including the
crossover; `test_benchmark.py` pins that the default mode list did not change length.

**What remains.**

* **Calibration does not know about split mode.** Throughput observations key on model, device set
  and GPU class. A model benchmarked in tensor mode teaches the layer-mode estimate a correction it
  should not have. Rare enough that medians absorb it; a `split_mode` column would fix it.
* **The planner's compute-buffer term is not ubatch-aware** (above).
* **MTP was measured at one slot only.** `draft-mtp` with `--parallel > 1` is untested here; there
  is no reason to expect trouble and no evidence either.
* **`help.txt` is written with translated newlines on Windows**, so the cached copy is
  double-spaced. Harmless -- `parse_help_entries` skips blank lines -- but it is why the fixture
  generator had to filter them.
* **The catalog, placements and the Setup tab do not show any of this yet.** `LoadPlan` and
  `InstanceInfo` carry `speculative` and the resolved `split_mode`, and
  `capabilities.engine_feature_rows` renders the Engine card, but the surfaces themselves belong to
  work packages running alongside this one.

---

## D39 -- Which GPU the memory is on: LUID -> PCI bus -> CUDA ordinal

**Problem (2026-08-19).** The Dashboard's holders panel said this:

```
llama-server.exe (pid 32188) · 30.44 GiB · CUDA0,1,2,3 · ours · alias Qwen3.8-27B-ABLITERATED-…
llama-server.exe (pid 31140) · 21.43 GiB · CUDA0,1,2,3 · ours · alias Gemma4-31B-QAT-…
llama-server.exe (pid 27376) · 18.95 GiB · CUDA0,1,2,3 · foreign
```

Every size was right and every device list was wrong. Measured on the same box at the same moment:
pid 32188 held 15.52 GiB on CUDA0 and 14.48 GiB on CUDA1 and **nothing** on the two 3090s; pid
31140 10.89 + 10.10 on CUDA0/1; the "foreign" one held 18.97 GiB on CUDA2 alone. The device column
came from NVML's `nvmlDeviceGetComputeRunningProcesses`, which enumerates the devices a process has
a CUDA **context** on -- and llama.cpp opens one on every visible device at startup, whether or not
a byte of the model lands there. So three models on two cards read as three models on four cards,
which is the opposite of the answer this panel exists to give: it made the 3090s look occupied when
they were free, and hid the fact that the "foreign" 19 GiB was sitting on exactly one of them.

D23 had already looked at this and concluded it could not be done: PDH knows the bytes and the
adapter **LUID**, NVML knows the CUDA ordinals and no LUID, and "inventing a split per GPU would be
worse than saying so". That reasoning was right about NVML and wrong about the conclusion, because
the two sides can be joined through a third identifier neither of them is named after.

**Decision: join on the PCI address.**

| Step | Call | Gives |
| --- | --- | --- |
| PDH instance name | `pid_32188_luid_0x00000000_0x00013C35_phys_0` | pid + adapter LUID + bytes |
| LUID -> handle | `D3DKMTOpenAdapterFromLuid` (gdi32) | a kernel adapter handle |
| handle -> address | `D3DKMTQueryAdapterInfo(KMTQAITYPE_ADAPTERADDRESS=6)` | `BusNumber`/`DeviceNumber`/`FunctionNumber` |
| ordinal -> address | `nvmlDeviceGetPciInfo(h).bus` | the same bus, per CUDA ordinal |

Measured on the reference rig: LUID `0x13C35` -> bus `0x01` -> CUDA0 (RTX 5090), `0x155BF` ->
`0x42` -> CUDA1, `0x1671A` -> `0xC1` -> CUDA2 (RTX 3090), `0x175FB` -> `0xC2` -> CUDA3. A fifth
LUID, `0x1852C`, is the Microsoft Basic Render adapter; it opens but answers `0xFFFFFFFF` for its
address, so it resolves to nothing and its bytes land under device `-1` rather than being guessed
onto a card.

Three details are load-bearing:

* **The bus in `busId` is hex.** `"00000000:42:00.0"` is bus 66, not 42. Reading it as decimal
  silently attributes one card's memory to whatever sits at bus 42, which is the worst possible
  failure here -- a confident wrong answer, indistinguishable from a right one.
* **The map is rebuilt, never persisted.** `CUDA_VISIBLE_DEVICES`, a driver reset and a hot-plugged
  eGPU all renumber CUDA ordinals, and the whole value of this is that the number matches what
  `--device CUDA<n>` means *to this process, now*. Cached 30 s, and rebuilt immediately when a LUID
  turns up that the current map has never seen. An adapter that failed to resolve is remembered as
  tried, so the Basic Render adapter does not force a rebuild on every poll.
* **`phys_<n>` is not part of the key.** It is the physical adapter *within* a LUID (linked display
  adapters); a linked pair is one CUDA device and its instances fold into one bucket.

**What the payload says now.** Every holder carries `per_gpu_bytes`
(`{"0": 16664092672, "1": 15550488576, "2": 234725376, "3": 234721280}`) whose values always add
back up to `used_bytes` -- nothing is dropped or spread -- and `gpu_indices` lists only the devices
holding at least `HOLDER_MIN_BYTES` (256 MiB), which is what removes llama.cpp's ~0.22 GiB
per-device context and leaves the cards the weights are on. `gpu_indices_source` says which
question was answered: `pdh` is a measurement, `nvml-context` means only NVML could answer and the
list is contexts, not placement. `used_bytes` keeps its D23 meaning exactly -- a per-process total,
still not summable across per-GPU rows. `/api/status` entries additionally carry `device_bytes`,
what that row's pid holds on **that** row's `gpu_index`, which is the one figure in the payload
that may be summed across rows.

The Dashboard row therefore reads
`llama-server.exe (pid 32188) · 30.44 GiB · CUDA0 15.5 GiB, CUDA1 14.5 GiB · ours · alias …`,
verified against the live rig on 2026-08-19.

**And a "foreign" holder that knew who it was.** The third row above was a `llama-server` child of a
*scratch* StudioForge -- `--alias scratch --port 1258`, the binary copied to a temp directory by a
measurement run -- and its own argv said so the whole time. `foreign` is true and useless. A
`llama-server` binary that is **not** under our engines directory is now classified
`other-instance`, with `detail` naming its alias, port and directory. This changes nothing about
safety: `find_engine_processes` still scopes kill candidates to our own engines tree exactly as D23
requires, and an `other-instance` is never a kill candidate. `foreign` now means what it says --
ComfyUI, a browser, a game.

**What pins it.** `tests/unit/test_vram_holders.py`: the LUID parse (including `phys_1` folding and
the unparseable cases), the four-card map built over the real ctypes structures against a fake
gdi32 and a fake NVML, the hex `busId` trap, an adapter with no CUDA ordinal, the cache and its
rebuild-on-unseen-LUID, the live per-adapter aggregation (two LUIDs -> two ordinals, unmapped ->
-1, and that the split still sums to the D23 total), the four branches of the device column, the
`other-instance` classification from a fake cmdline, `device_bytes` on status rows, and the rendered
Dashboard line.

**What remains.**

* **Windows only.** D3DKMT is a Windows kernel-mode thunk and PDH is a Windows counter. On Linux
  NVML reports real per-process, per-GPU bytes and needs none of this; in a container, in WSL or
  under MIG neither side answers and the column falls back to `nvml-context`, which the payload and
  the panel both say out loud.
* **The per-device split is not yet fed back into calibration.** `process_gpu_bytes(pid)` exists for
  exactly that and nothing calls it: `load_observations` records the plan's intended
  `per_gpu_bytes` and could now record the achieved one beside it (D18, per device). It matters --
  the load above planned `--device CUDA1,CUDA0 --tensor-split 0.5079,0.4921`, i.e. *more* on CUDA1,
  and CUDA0 ended up holding 15.52 against CUDA1's 14.48, because llama.cpp puts the output layer
  and its scratch on one end of the device list. A tight card can OOM on exactly that delta, and
  today nothing learns it. Left out here because `planner.py`/`manager.py` are WP18's.
* **A holder measured only on unresolvable adapters shows no device at all.** It reports
  `gpu_indices: []` with source `nvml-context` and its bytes under `-1`; that is honest and it is
  also the least useful row in the payload.

---

## D40 -- Where a load's bytes land: observed per device, the output layer charged to the last card, `-ub` modelled

**Problem (WP22 review, 2026-08-20).** Three things that were one thing, found while closing the
follow-ups D38 and D39 left open.

*D18's calibration had been reading a number that was wrong by the device count.*
`ModelManager._record_actual_vram` summed `vram_processes` rows over the plan's devices. On Windows
every row of a pid carries the **same** PDH per-process total (D23's back-fill, one total written
onto each NVML `(gpu, pid)` row), so a two-GPU load was recorded at twice its footprint and a
four-GPU one at four times. The live registry held 29 `per_pid` rows on 2026-08-19 and every
multi-device load sat at a ratio of its device count -- a 17.4 GB model "measuring" 134 GB across
three cards, 104 GB across four -- and even a single-card 1.5B read 1.30 because the per-process
total includes the ~0.22 GiB CUDA context the child opens on every *other* card. `calibrate()`
takes the worst shortfall over weights, so on every boot of the reference rig
`compute_overhead_fraction` was pegged at its 0.15 ceiling: ~9% of every model's weights silently
subtracted from every estimate, which is the direction that refuses loads that fit.

*Nothing knew which card the bytes were on.* D39 left `process_gpu_bytes(pid)` with no caller, and
its evidence standing: the live 27B planned at `--device CUDA1,CUDA0 --tensor-split 0.5079,0.4921`
landed 15.52 GiB on CUDA0 and 14.48 on CUDA1 -- 0.76 GiB *more* on the card the split gave less to.

*And `-ub` could not be raised safely* (D38): the compute term was calibrated at the engine's 512
and did not move with the micro-batch.

**Measured** on a scratch `llama-server` (engine `b10425` copied read-only, port 1260, the 3090
pair, per-GPU bytes through D39's PDH LUID->CUDA join; every child killed and verified gone):

| model | `--device` / `--tensor-split` | `-ub` | first card | last card |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B Q8_0 | `CUDA3,CUDA2` / `0.6,0.4` | 512 | 576 MiB | **608 MiB** |
| Qwen2.5-0.5B Q8_0 | `CUDA2,CUDA3` / `0.6,0.4` | 512 | 576 MiB | **608 MiB** |
| Qwen2.5-0.5B Q8_0 | `CUDA2,CUDA3` / `0.5,0.5` | 512 | 542 MiB | **646 MiB** (+104) |
| Qwen2.5-0.5B Q8_0 | `CUDA2,CUDA3` / `0.5,0.5` | 2048 | 712 MiB | **816 MiB** (+104) |
| Qwen2.5-1.5B Q4_K_M | `CUDA2,CUDA3` / `0.5,0.5` | 512 | 820 MiB | **930 MiB** (+110) |
| Qwen2.5-1.5B Q4_K_M | `CUDA2,CUDA3` / `0.5,0.5` | 2048 | 1090 MiB | **1200 MiB** (+110) |

Two facts fall out. **The last device of the list holds more**, whichever physical card it is, and
the card given the *smaller* fraction still ends up heavier: llama.cpp assigns the output layer to
`dev_output`, the device holding layer `n_layer`, which the split arithmetic always makes the last
one. **The delta does not move with `-ub`** (+104 at both 512 and 2048), so it is weights, not the
logits scratch -- 0.75x (0.5B) and 0.6x (1.5B) of the tied-embedding tensor; the 27B's 0.76 GiB
sits between its Q6_K and Q8_0 output-tensor sizes. Meanwhile the micro-batch grows *both* cards:
+170 MiB each for +1536 tokens on the 0.5B (113 B/token/`n_embd`), +270 MiB each on the 1.5B
(180 B/token/`n_embd`), against +114 / +198 MiB on a single card (76-94 and 132 B/token/`n_embd`).

**Decision.**

1. **`measure_child_vram(probe, pid, devices)`** answers `(total, {device: bytes} | None)` in this
   order: PDH per adapter joined to CUDA ordinals (Windows, D39), restricted to the plan's devices
   so a context on a card the model is not on is left out exactly as the plan leaves it out; NVML's
   genuine per-process, per-GPU rows (Linux) -- recognised by *not* all equalling the PDH total,
   which is how the back-fill betrays itself; the PDH total **once**, with no split claimed. The
   observation carries `per_gpu_planned` and `per_gpu_actual` (migration 006, JSON text per CUDA
   index) beside the totals, and is marked `note = "per_pid_v2"`. **Calibration reads only
   `per_pid_v2` rows**, as D18 did with `per_pid` against the device-total rows before it; the old
   rows stay for the record and are inert.
2. **A device holding more than 1.15x its planned share is a WARNING** naming the card and both
   numbers (`planner.per_device_overruns`). 15%: under the 10% headroom plus the per-card CUDA
   context charge a smaller overrun is noise; above it the card was genuinely planned too tight.
3. **The planner charges the output layer to the last device** of a multi-GPU plan.
   `output_layer_bytes(meta)` is `n_vocab * n_embd` at the file's average bytes per weight (from
   `general.parameter_count`; 6.5 bpw when undeclared -- quantizers keep the embedding tensors at
   Q6_K/Q8_0 in a Q4 file). The split fractions are taken from the cards' capacities with that
   charge removed from the last one, so `--tensor-split` leans a little toward the first card and
   `per_gpu_bytes[last]` carries its block share plus the layer. It is an approximation in the
   safe direction for the card that OOMs (the first card's share is under-stated by at most the
   same figure, well inside headroom), and the per-device observation now measures it.
4. **`Planner.estimate` is micro-batch aware.** `ubatch_for(record)` resolves per-model, then
   `engine.ubatch_size`, then 512 (the supervisor's precedence); `kv_alloc_bytes` gets the real
   `-ub` for the iSWA cell formula it always took as a parameter, and `ubatch_scratch_bytes`
   charges `128 B x (ub - 512) x n_embd` **per device** -- covering the two-card figure with a
   little room. `engine.ubatch_size` is therefore safe to raise; it stays unset because it is a
   VRAM-for-prefill trade the operator should make knowingly. The supervisor raises the automatic
   `--batch-size` to at least the micro-batch, because llama.cpp clamps `n_ubatch` to `n_batch`
   silently and `-ub 4096` against the default `-b 2048` would be a flag that looks like it did
   something; an explicit `batch_size` is still the user's.

**What pins it.** `tests/unit/test_planner_parallel.py` (the Windows total counted once, the PDH
split restricted to plan devices, Linux rows summed per device, the overrun bar, the stored split,
`per_pid` rows no longer trusted), `tests/unit/test_planner.py` (the output-layer charge and the
tilted split, a last card too tight for its layer, `ubatch_scratch_bytes`, the estimate's
precedence), `tests/unit/test_supervisor_features.py` (`--batch-size` follows `-ub`),
`tests/unit/test_db.py` (migration 006 round-trip).

**What remains.**

* The output-layer charge is derived from `n_vocab * n_embd * bpw`, not read from the tensor
  table; the scanner could record the real `output.weight` / `token_embd.weight` byte count and
  make it exact. The observation rows now say how far off it is per model.
* The existing `per_pid` rows are not re-interpretable: a single-card row is only inflated by the
  other cards' contexts, a multi-card row by its device count, and the row does not say which.
  They are ignored rather than corrected.
* `UBATCH_SCRATCH_BYTES_PER_TOKEN_PER_EMBD = 128` is two models on two cards; a 30B at `-ub 2048`
  on a 4-way split is unmeasured. The direction of error is refusal, not OOM.

**Also, from the same scratch run.** D39's device column named the two idle RTX 5090s as holding
a model that lived on the 3090s: llama.cpp's per-device CUDA context is ~0.22 GiB on a 3090 but
**~0.43 GiB on a 5090**, above the 256 MiB floor. The placement column now has its own floor,
`DEVICE_PLACEMENT_MIN_BYTES = 512 MiB` (the listing floor `HOLDER_MIN_BYTES` stays 256 MiB, a
different question), which clears the Blackwell context and is still under the smallest real
placement seen here (568 MiB, the lighter card of a 0.5B pair at 4k).

## D41 -- A pin is a desired state, and a reconciler enforces it

**Problem.** `pinned` promised "keep this model loaded" and delivered only half of it. What it
did: effective TTL 0 (never idle-unloaded), excluded from every eviction ladder, warmed once at
startup by `_autoload_pinned`. What it did not: nothing ever *re*-loaded a pinned model. A child
that crash-looped past `gateway.max_restarts` sat at `state="failed"` holding nothing; a pin set
while the model was unloaded did nothing until the next boot; an autoload that failed was one log
line and gone. Worse, two paths silently *broke* an existing pin: LM Studio's request-level `ttl`
was written straight onto `instance.ttl_s` -- and `ttl_s == 0` is the wire representation of
pinned everywhere, so any client sending `{"ttl": 60}` unpinned the model the owner pinned. And
the GUI toggle saved the setting without `refresh_ttl`, so a pin clicked on a resident model was
not seen by the sweeper or the planner until the next load -- exactly when it was not needed.

**Decision.** Pinned is a *desired state*: "resident at all times", reconciled, not just
exempted.

1. **A reconcile pass rides the TTL sweep** (every `gateway.ttl_sweep_interval_s`). A pinned
   model with no `ready`/`loading` instance -- never loaded, crashed out, failed at boot -- is
   reloaded, `source="pin-reconcile"`. The pass runs as its own task so a multi-minute cold load
   never stalls the sweeper, at most one pass is in flight, and the loads carry no special
   licence: idle unpinned models may be evicted per policy, a busy model never (D36).
2. **A deliberate unload outranks the pin.** `manager.unload` / `unload_all` mark the stopped id
   suppressed; the reconciler leaves it down. Anything else is a 15-second illusion of an unload
   and a panic button that frees nothing. Any successful load of the model, or pinning it again,
   lifts the mark. The pin *setting* is never silently changed -- suppression is in-memory and
   dies with the process, so a restart restores "pinned means resident".
3. **Failures back off per model**: 60 s doubling to a 900 s ceiling, seeded also by a failed
   boot autoload. A pinned model that no longer fits is a WARNING per attempt and a longer wait,
   not a sweep-rate retry storm; the backoff resets on success or re-pin.
4. **A request-level `ttl` cannot unpin.** `_apply_ttl_override` leaves an instance whose
   `ttl_s == 0` alone. The request itself still works; only its idle-timer wish is ignored.
5. **One implementation of "set the pin"** (the D26 rule): `manager.set_pinned` does
   save-settings + `refresh_ttl` + re-arm-reconciler, and the HTTP route, the GUI toggles (Models
   row, Dashboard card) and the new MCP `pin_model` tool all call it. The GUI's own
   save-without-refresh copy is gone.
6. **Pin and saved settings join the D32 gate.** On an open install, `POST /api/models/{id}/pin`
   and `PUT /api/models/{id}/settings` need a local caller or the PIN: both outlive the instance
   -- a pin drives the boot autoload and the reconciler; saved settings shape every future load
   -- so they are box changes, not residency. Load/unload stay open (LM Studio parity).

`models.auto_load_pinned` now gates the whole promise: startup warm-up *and* the reconciler. Off,
a pin degrades to what it used to be -- no idle TTL, never evicted.

**Invariants kept.** The wire representation stays `ttl_s == 0`, nothing new travels on the
instance. Reconcile loads serialise behind the load gate (D29) and plan like any other load (D14/
D16); an explicit unload still frees VRAM immediately and stays freed (the GUI's unload-all
dialog now says so). Several models may be pinned at once; the planner's refusal text already
names pins as the obstacle when they crowd the cards.

**Tests.** `tests/unit/test_gateway_lifecycle.py` (reconciler wants/skips/backs off, suppression
round-trip, ttl-override guard, `set_pinned`), `tests/unit/test_mcp.py` (`pin_model` tool,
surface pinned exactly), `tests/unit/test_api_hardening.py` (D32 gate moves),
`tests/unit/test_catalog_routes.py` (`GET /v1/models/{id}` now carries the same runtime fields as
the list -- the two shapes had diverged), `tests/unit/test_gui.py` (filtering on "pinned").

## D42 -- A placement is a moment's answer, and the sweep re-asks the question

**Problem.** Placements accrete. Measured on this rig, 2026-08-20: at 13:42 a 27B was planned onto
`[1, 3]` -- a 5090/3090 cross-tier split sharing GPU1 -- because the 31B then held `[1, 0, 2]` and
ClawForge held 7.5 GiB of GPU2, so that genuinely was the best fit at that instant. At 13:53 the
31B was reloaded with a q8_0 cache and shrank to `[0, 1]`. From that moment `[2, 3]` sat free and
strictly better -- no shared card, no cross-tier hop -- and the 27B stayed on `[1, 3]` anyway,
contending with the 31B on GPU1 for every request either of them served. Every plan was optimal
for the instant it was made; nothing ever revisited one after the world changed. (Prior art has
the same shape: ollama's scheduler evicts and retries on pressure but never re-places a resident;
the datacenter answer is live KV migration, whose home-rig equivalent is exactly "reload an idle
model while nobody is watching".)

**Decision.** A rebalance pass rides the TTL sweep, beside D41's pin reconciler and built the
same way: single-flight task, so a multi-minute reload never stalls the sweeper.

1. **Two triggers, both standing costs, neither a throughput guess.** An idle resident is moved
   when a no-evict plan at its exact current settings (per-slot context, KV types, slot count --
   a one-rung ladder, D14) lands it (a) off every card it shares with another resident, or
   (b) on strictly fewer cards. Estimated tok/s never justifies a move: chasing estimates is
   churn, removing a standing misplacement is not.
2. **Quiet box only, and really idle.** The pass runs only when nothing is serving, loading,
   testing or benchmarking (`_busy_reason`), and only for a model idle >= 300 s -- because a
   relocation is a reload, and a reload drops the child's prompt cache. On this rig's RP
   workload that cache was 93% of a 98k-token prompt; moving a model between turns would trade
   a permanent contention win for a multi-minute reprocess loss. Five minutes idle means the
   conversation has plausibly gone away.
3. **The preview is the D30 machinery verbatim**: `plan_load(reload_of=self, allow_evict=False)`
   credits the model's own footprint back and may evict nobody. The move itself is a forced
   reload onto the previewed devices (`force=True, evict_busy=False` -- the reload licence
   without the interrupt licence, D36), so a refusal at the gate leaves the resident serving.
4. **Hysteresis**: one move (or one `suggest` log line) per model per 30 min, and a persisted
   `device_override` -- the user said "run it here" -- is never second-guessed.
5. **`planner.rebalance: auto | suggest | off`**, default `auto`. `suggest` logs the opportunity
   and acts on nothing; the MCP instructions tell agents a quietly changed placement is
   housekeeping, not something they did.

**What this is not.** Not joint planning: two models loaded in sequence still each plan against
the other's residency, and a globally-optimal *pair* placement (bin-packing both at once) remains
future work -- the rebalancer converges the layout after the fact instead. Not an eviction
change: nothing new may evict anything, in either pass.

**Tests.** `tests/unit/test_gateway_lifecycle.py` (the measured `[1,3]`-beside-`[0,1]` layout
moves to `[2,3]`; suggest/off modes; idle, cooldown, busy-box and device_override gates; a
refusal or an evicting candidate is ignored; same-placement answer means no move).


### D42, amended the same day: the rebalancer looks once a minute, and only when the world changed

The first cut asked the planner every 15-second sweep for every idle resident -- an NVML read
plus estimate arithmetic per model, for an answer that cannot change while nothing moves. Now:
the pass runs at most once per `REBALANCE_CHECK_S` (60 s); a model alone on one card is never
asked about (no move could improve it); and each model's answer is keyed to a *layout
fingerprint* -- who sits where, plus the standing leases -- so the planner is consulted again
only when that fingerprint changes, or after `REBALANCE_RECHECK_S` (600 s) for the slow drift
that free VRAM from outside processes represents. Steady state: zero planner calls. Everything
else on the sweep (idle TTL, lease expiry, the pin reconciler's wants-list) is in-memory
bookkeeping over a handful of instances; the one network touch, the per-child `/metrics` scrape,
predates this work and is one localhost GET per loaded model (D22).

## D43 -- A card can belong to one model: GPU leases

**Problem.** Two things on this rig wanted a card to themselves and had no way to say so.
Benchmarks: every mode's number is supposed to be the *card's*, but a JIT load from another
client could land beside the subject mid-measurement, and the only defence was "refuse to start
while busy". And the agent's workhorse: "give this model the two 5090s and keep everything else
off them" was expressible only as a persisted `device_override` -- which steers *that* model and
says nothing to the others. `planner.excluded_devices` is the static cousin (D19): a config
edit, for every model, forever.

**Decision.** A **lease** is a runtime claim on specific CUDA devices, kept in an in-memory book
shared by the planner and the manager.

1. **Planner.** A card leased to someone other than the model being planned is *absent* from
   its GPU view -- not ranked last, not an option (the D19 rule, per model and per moment). A
   `device_override` naming such a card is refused with the lease named; a plan or a refusal
   that lost cards to a lease says so in its notes / suggestions, with the lease id and how it
   ends. A lease with no models holds its cards for something outside this server entirely.
2. **The owner gets the measured-fastest shape.** A load of a leased model is forced onto the
   lease's devices; its slot count is sized by the estimator even when
   `models.default_parallel` is an integer (`parallel_auto` -- the cards are its alone, the
   rig-wide caution does not apply); and its split mode and micro-batch come from the model's
   latest benchmark **on exactly those devices** when one exists. Tensor split is never
   assumed: it measured *slower* than layer split here (D38 S3) and is unmeasured for large
   models, so only a measurement may choose it. Explicit per-model settings still win.
3. **Granting one.** Idle residents on the cards that are not the new owner are unloaded --
   that is what "the card is yours" means. A resident mid-request is never interrupted (D36):
   the grant refuses with `retry_after_s`. A pinned idle resident refuses too unless
   `force=true`, because a pin is a standing wish; forced, it is evicted and the D41 reconciler
   brings it back elsewhere if it fits, or on these cards once the lease ends. A card already
   leased is a conflict (409), never a takeover. A load in flight refuses the grant: it may be
   about to land on those cards.
4. **Ending one.** Explicitly (`DELETE /api/leases/{id}`, `release_gpus`, the Dashboard), or
   by the sweep once idle for `idle_ttl_s` (default 3600 s; `null` = until released). The
   owner model's own requests count as activity; an outside holder touches the lease
   (`POST /api/leases/{id}/touch`). In-memory on purpose: a lease describes a live situation
   and a restart is a clean slate; the idle TTL is what keeps a crashed benchmark or a
   forgotten reservation from holding a card forever.
5. **Benchmarks are the first client.** The placement benchmark takes a lease on each mode's
   devices for that mode; the parallel benchmark takes one on its run's devices. A neighbour
   that lands on the cards between modes is unloaded by the next grant; one that is busy fails
   that mode by name instead of contaminating its number.
6. **Surfaces.** `GET/POST /api/leases`, `DELETE /api/leases/{id}`, `POST .../touch` (the
   mutations under the D32 gate); MCP `reserve_gpus` / `release_gpus`, `server_status.leases`;
   `ServerStatus.leases`; a Dashboard panel with a release button.

**What this is not.** Not an eviction licence: a grant unloads only *idle* residents, and the
planner evicts nothing it could not evict before. Not persistence: pin the model too if it
must also come back after a restart.

**Tests.** `tests/unit/test_leases.py` (the book; the planner's refusal and the owner's right,
auto slots under an integer default; the grant's idle/busy/pinned/in-flight rules; expiry
with owner activity; the performance profile incl. measured split mode on exactly those
cards), `tests/unit/test_mcp.py` (`reserve_gpus`/`release_gpus` through `server_status`),
`tests/unit/test_catalog_routes.py` (the routes), `tests/unit/test_api_hardening.py` (D32).

### D32 and D41, amended 2026-08-22: the browser is a loopback peer, the MCP plane is a box, a request ttl cannot pin

Corrections from review; none is a new rule. **D32, browser origin.** The operator's own
browser is a loopback peer, and with `server.cors_origins: ["*"]` any page it shows can
preflight and send `PATCH /api/config` to `http://127.0.0.1:1234` looking local. So on an open
install the admin gate and the PIN reveal (`/api/mcp/info`, `/api/openclaw-setup`) apply the
websocket gate's rule: a request whose `Origin` is not this server's own origin -- host *and*
port, because another local web app is also a loopback peer and is not this server; `Origin:
null` counts as foreign -- is not "this machine" and gets the same 403 / withheld PIN. CORS
itself is unchanged: it governs what a page may *read*, never whom the server trusts.
**D32, MCP plane.** With no key, a mutating request on the MCP path from off the box needs the
PIN even when `mcp.pin_required` is off: every streamable-HTTP call is a POST and the tools
behind it (`set_config`, `delete_model`, `download_model`, `reserve_gpus`) are the box changes
the routes gate. The toggle relaxes same-machine callers only, and `/api/mcp/info` reports the
`pin_required` a caller will actually be held to. **D41 item 4, mirrored.** A request-level
`ttl` cannot pin either: `0` (and anything `int` rounds to it) is the wire form of pinned, so it
is treated as no override rather than written onto the instance. **Also:** `/api/mcp/info` and
the 401 text no longer advertise `?pin=` (still parsed, for URL-only connectors; a URL lands in
proxy logs and shell history), and the vision SSRF guard refuses CGNAT space (`100.64.0.0/10`,
the tailnet it names in its own reason for existing -- `is_private` never covered it).

### 2026-08-22 review amendments (D2, D41, D42, D43) -- code brought in line with the decisions

* **D2.** Upstream publishes no Linux CUDA archive at any tag (the ubuntu set is cpu/vulkan/rocm/
  sycl/openvino, all `.tar.gz`), so "(Linux: the matching `ubuntu-cuda` asset)" never existed.
  Linux+NVIDIA is the source build, and `install()` -- `engine --update`, the Setup tab, the MCP
  install -- now takes the same `allow_source_build` fallback `ensure_engine` always took, reuses
  an existing `<tag>-local` (D27), and names the prerequisites (git, cmake, a CUDA toolkit with
  nvcc matching the driver) when building is off. The asset parser accepts `.tar.gz` so the ROCm
  tarball serves AMD Linux. The source build bakes `$ORIGIN` into the RUNPATH (D3's
  self-contained engine dir: the copied binary used to resolve its `.so` files through the
  CMake build tree, and through a *newer* tag's once the shared vendor checkout was rebuilt).
* **D41.** Housekeeping unloads -- the placement benchmark's fresh process per mode and its
  teardown, the parallel benchmark's leave-as-found, `test_model` -- pass `deliberate=False` and
  do not suppress the pin; a pinned model benchmarked or tested is brought back by the
  reconciler within a sweep. The reconciler leaves a model under test or benchmark alone until
  the run ends, so it cannot race the run's own unload -> lease -> load swap.
* **D42.** Item 3 is now enforced rather than described: the busy check is repeated at the gate
  (after the possibly minutes-long wait behind another load), the real move carries the
  preview's `allow_evict=False`, and `require_resident` makes it a relocation only -- a resident
  the TTL sweep unloaded meanwhile is not cold-loaded. A lease owner (D43) is never a candidate:
  its placement is forced on a copy of the record, so the persisted-override gate could not see
  it, and the cards are its alone. A one-shot `load(devices=...)` placement remains eligible,
  per item 5.
* **D43.** The grant writes the book entry *before* the awaited evictions (and releases it if one
  fails): the planner reads the book, each `supervisor.stop` yields for a child teardown, and a
  load planning in that window saw the cards free. Two overlapping grants now conflict at the
  book before either unloads anything, and a victim that picked up a request between the scan
  and its stop refuses the grant (D36) instead of being torn down.
