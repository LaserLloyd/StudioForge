# The model catalog

One call that tells an agent what it can run, at what context, how fast, and how many
conversations at once — with the exact arguments to load each choice.

Before the catalog, choosing a model meant a `/v1/models` call, a `/profiles` call per model,
arithmetic about KV caches, and a guess about speed. The catalog answers all of it per model, as a
table of rows the agent hands straight back.

---

## For an LLM: the whole workflow

```
# I know the window I need, and not much else:
load_recommended(model_id, ctx_size=262144)   -> loaded at exactly that, or told why not

# I want to choose the hardware:
list_models()                                 -> catalog, newest download first
load_model(**model["recommended"]["load_args"])  -> loaded, serving
server_status()                               -> confirm; read busy before you load again
```

`recommended` is the **default load**: this model's optimal settings on the rig's best pair of
GPUs, computed as if those cards were free, with `load_args` that already name the `devices`.
`placements` is the same answer for every other set of cards on the box, and `options` is the
per-context drill-down. Reach for `model_options(model_id)` when you need a different context
window, more concurrency, or the `load_args` of a `placements` mode other than the recommended one.

**Rules that make this reliable:**

- Take `recommended` unless your task needs a different window or a different set of cards.
- Pass `load_args` **verbatim**. Do not add fields, convert units or recompute anything.
- `fits: false` means it will not load *right now*. Check that row's `if_gpus_idle`: if that says
  `fits: true`, the VRAM exists and something else is holding it — `unload_model` on another model
  makes the row available.
- Match your client concurrency to `recommended_parallel`, not to `max_parallel`. The first is how
  many slots are worth running; the second is how many fit. Beyond `max_parallel`, llama.cpp queues
  rather than refusing, so extra streams show up as latency, not errors (watch `requests_deferred`
  in `/api/status`).
- If all you know is "I need a 256k window", skip the table: `load_recommended(model_id, ctx_size)`
  picks the cards, the KV cache and the slot count for you and loads at **exactly** that context,
  or refuses with the largest context each set of cards would take.
- A row's `parallel` is the most its placement sustained *when the row was built*. If VRAM has
  moved by the time you call `load_model`, the `507` names the slot count that fits now at the
  same window (`max_parallel_that_fits`, and the first suggestion) -- the window outranks the
  second slot, so reach for fewer slots before a smaller context.

---

## Surfaces

| Surface | Use |
| --- | --- |
| `list_models` (MCP) | The catalog. Compact by default — one row per model. `full=true` for every context tier. |
| `model_options(model_id)` (MCP) | Every context tier for one model. |
| `GET /api/catalog` | The same data over HTTP. `?compact=1`, `?model=<id>`, `?refresh=1`. |
| `GET /api/models/{id}/profiles` | The *hardware-mode* cut instead of the context cut: the same `placements` list, in full, with every mode's `load_args`. |
| `load_recommended(model_id, ctx_size)` (MCP) | Name the model and the window; the server picks the rest and loads at **exactly** that context, or refuses with numbers. |
| `POST /api/models/{id}/load-recommended` | The same over HTTP: `{"ctx_size": 262144, "prefer_mode": "dual_3090"}`. |
| `benchmark_parallel(model_id)` (MCP) | Measure the slot knee, so `recommended_parallel` stops being an estimate. |
| `POST /api/models/{id}/benchmark-parallel` | The same as a background job on `/api/benchmark/jobs/{id}`. |
| `GET /api/models/{id}/parallel-observations` | The measured rows behind `recommended_parallel`. |

Both catalog surfaces are built by `core/catalog.py:build_catalog` and cached for 20 seconds:
`fits` is a claim about free VRAM at an instant, and a stale yes sends an agent into a load that
gets refused. `refresh` bypasses it — worth doing right after loading or unloading something.

---

## A row, in full

```json
{
  "ctx_per_slot": 65536,
  "fits": true,
  "devices": [0],
  "kv_cache_type": "f16",
  "vram_mb": 27416,
  "max_parallel": 2,
  "parallel_limited_by": "vram",
  "recommended_parallel": 2,
  "recommended_parallel_basis": "estimated",
  "est_prompt_tps": 1190.4,
  "est_gen_tps": 100.1,
  "est_gen_tps_full_ctx": 71.8,
  "est_gen_tps_batched": 147.3,
  "measured_gen_tps": null,
  "measured_prompt_tps": null,
  "confidence": "estimated",
  "if_gpus_idle": { "fits": true, "devices": [0], "max_parallel": 2 },
  "load_args": {
    "model_id": "pub/dense-8b",
    "ctx_size": 65536,
    "parallel": 2,
    "kv_cache_type": "f16"
  },
  "best_now": true
}
```

`best_now` marks the one row of this table that would load on the machine exactly as it stands.
It is not the same claim as the entry's `recommended`, which is a **placement** computed with its
cards idle — when you get to choose the hardware, `recommended` is the better answer.

`kv_cache_type_v` appears only when it differs from `kv_cache_type` (and likewise inside
`load_args`): the tool defaults V to K, so an equal pair would cost every caller tokens to be told
the same thing twice.

`ctx_per_slot` is the context **each conversation** gets. `--ctx-size` is the *total* across slots
(DECISIONS.md D4), so the engine is launched with `ctx_per_slot * parallel` and every slot really
gets the number in the row.

`vram_mb` is sized at this row's own `max_parallel`, so it describes the widest load this
placement could carry — not whatever slot count the planner happened to pick while checking the fit.

`max_parallel` is how many slots **fit**; `recommended_parallel` is how many are **worth running**,
and it is the one `load_args.parallel` asks for. Until a parallel benchmark has run they are the
same number and the basis reads `estimated`
— see [recommended_parallel](#recommended_parallel--how-many-are-worth-running-d37).

**Two generation speeds, because one number cannot describe the row.** Every decode step re-reads
the KV cache, so generation slows as the window fills. `est_gen_tps` is one stream with about 8k
tokens in the window — an ordinary turn, and the number worth comparing against one you have seen
elsewhere. `est_gen_tps_full_ctx` is the same stream with the window nearly full: the pessimistic
end. A real conversation moves between them as it grows, and how far apart they sit is the true
price of choosing a wide row.

The model entry (not the row) carries `attention_kind`, which explains why one model's 262k row is
cheap and another's does not exist — see [Attention kinds](#attention-kinds) below.

---

## How each column is computed

### Ordering — newest download first

`downloaded_at` is the newest mtime across the model's GGUF shards. A multi-part download finishes
on its *last* file, so keying off shard 1 would rank a just-downloaded model by when its first part
landed. This is the one ordering the catalog guarantees, because it is how the library is actually
used: from the last thing added.

### Context tiers

`{16384, 32768, 65536, 131072, 262144, 524288, 1048576}` capped at the model's `n_ctx_train`, plus
the model's own pinned `ctx_size` if it has one. Tiers past the trained window are dropped: serving
beyond it needs RoPE scaling and quietly degrades quality (D14), so offering it would be offering a
trap. A model whose window is below the smallest tier gets one row at its own window.

### fits / devices / kv_cache_type — the planner, not a copy of it

Each tier is one `Planner.plan_load(ctx_size=tier, allow_evict=False)` call. Exclusions
(`planner.excluded_devices`), reservations (`planner.reserved_mb`), quant affinity (D9), per-model
`device_override` and live free VRAM are therefore all respected by construction. `allow_evict=False`
is what makes `fits` mean "loads right now without disturbing anything".

`if_gpus_idle` is the same question asked of a machine where every GPU is empty — free VRAM becomes
total VRAM, while headroom and reservations still apply, because those describe memory that is never
ours regardless of what is loaded. That is what separates "impossible on this hardware" from
"possible once you unload something".

The whole catalog is built from **one VRAM snapshot**, so every row describes the same instant.

### Attention kinds

The model entry's `attention_kind` is derived from the GGUF's per-layer geometry
(`planner.kv_layers`), never from the architecture string — `gemma4` covers both a dense iSWA model
and a MoE one, and a new hybrid architecture lands upstream every few weeks.

| kind | What the GGUF says | What llama.cpp allocates |
| --- | --- | --- |
| `full` | nothing special | every layer holds `ctx_total` cells |
| `iswa` | `attention.sliding_window_pattern` (Gemma 3/4: five window layers per full one), `sliding_window`, `key_length_swa`/`value_length_swa` | window layers hold `GGML_PAD(min(ctx_total, window*parallel + ubatch), 256)` cells at the *swa* head dims; full layers hold `ctx_total` |
| `hybrid` | `{arch}.full_attention_interval` (qwen35/qwen35moe: 4) plus the `ssm.*` keys | layer `il` has a KV cache only when `(il+1) % interval == 0`; the rest are Gated-DeltaNet recurrent layers with **no KV** and a fixed per-sequence state of `((d_conv-1)*(d_inner + 2*n_group*d_state) + d_state*d_inner) * 4` bytes |
| `unknown` | not enough metadata for a per-layer answer | — treat every KV number for that model as unreliable, **not** as the cheap case |

Why it earns a column: a 31B Gemma-4 keeps ~85 KiB/token of effective KV at 262k where the uniform
figure says 1920 KiB — a factor of 22 — and a Qwen3.5-27B keeps a cache on 16 of its 65 blocks. That
is the whole reason those models offer wide, multi-slot rows and a same-sized full-attention model
does not. The two shapes that change the price (`iswa`, `hybrid`) also appear as a tag in the
model's one-line `summary`, so a forty-model list still shows them.

### max_parallel — two bounds, smaller wins

Still D17's pair, but both halves were rebuilt in D22.

**The VRAM bound is now exact.** `Planner.max_slots_by_vram` walks down from the cap (8) asking
`Planner.estimate(parallel=N)` — the same function a real load asks — and takes the first N that
fits the placement's usable bytes. At most eight pure-arithmetic calls, and it cannot disagree with
itself. The quotient it replaced (`kv_budget // (ctx_per_slot * kv_bytes_per_token)`) assumed KV
scales linearly with slots, which no interesting model obeys: an iSWA model's window layers grow
with `n_swa * n_seq_max` and its global layers not at all, and a hybrid model's recurrent state is
flat per slot. With a *uniform* per-token cost in that denominator, every Gemma-4 row read
`max_parallel: 1 (vram)` with 34 GB free.

**The knee still compares traffic**, but the KV side is now the bytes a slot really reads:

```
by_knee = active_weight_bytes / kv_read_bytes_per_slot(ctx_fill = ctx_per_slot * 0.5) * (0.5 if MoE else 1)
max_parallel = clamp(min(by_vram, by_knee), 1, 8)
```

`kv_read_bytes_per_slot` charges `ctx_fill` tokens on a full layer, `min(ctx_fill, window)` on a
sliding-window one, and a single fixed state read on a recurrent one. `parallel_limited_by` reports
`vram`, `knee`, `cap` or `unknown` (the last meaning the geometry could not be derived, in which case
one slot is reported rather than a guess).

The catalog computes this itself rather than reading `plan.max_parallel`, because the plan only
carries an estimated slot count when `models.default_parallel` is `"auto"`. With an explicit integer
configured — a legitimate choice, and what this rig runs — every plan reports one slot and the whole
concurrency column would collapse to 1. "What could this placement sustain" is well defined
regardless of current policy.

### recommended_parallel — how many are *worth* running (D37)

`max_parallel` is how many slots **fit**. `recommended_parallel` is how many are **worth running**,
and it is the number `load_args.parallel` asks for. `recommended_parallel_basis` says where it came
from:

| basis | meaning |
| --- | --- |
| `measured` | a parallel benchmark swept this model on **these devices at this context**, and the rule below picked the level |
| `estimated` | nobody has measured it; this is D17's bandwidth knee, clamped to `max_parallel` |

**The rule, over a measured sweep:** the largest level in 1 / 2 / 4 / 8 whose per-stream rate is
still at least **65%** of the solo rate *and* whose aggregate is at least **1.15x** the level below
it — walked upward, stopping at the first level that fails. Stopping matters: past the knee the
aggregate flattens, so a higher level can still beat its (depressed) predecessor by 15% and would
promote a slot count the run has already shown to be useless.

The two thresholds encode a priority, not a measurement. Aggregate throughput can always be bought
with per-stream latency; this server's answer is that a conversation running at under two-thirds of
its solo speed is a conversation the user notices, and a level that adds under 15% is the plateau.

`"measured"` is meant literally: a sweep on other cards, or at another context, is **not** used. The
knee is set by how many KV bytes each busy slot reads per decode step, so a run at 8192 says nothing
about the same model at 131072, and a run on one card says nothing about two.

**With no measurement, `recommended_parallel` equals `max_parallel`.** That is not redundancy, it is
the honest statement that D17's knee is already folded into `max_parallel` and there is nothing more
to say until somebody measures. The compact catalog view drops both keys while they are equal, so
they cost nothing until they mean something.

### Recommended parallel — an example run

Measured on the reference rig, 2026-08-19, on a scratch instance (ports 1256/8102/1257, its own data
directory, engine `b10425` copied read-only) so the live server was never touched.
`Qwen2.5-1.5B-Instruct-Q4_K_M`, **one RTX 3090**, 8192 tokens per slot, f16 KV, launched with 8
slots, 512-token prompts and 192 generated tokens per request:

| N | per-stream t/s | aggregate t/s | p50 (s) | p95 (s) | achieved batch |
| --- | --- | --- | --- | --- | --- |
| 1 | 302.8 | 302.8 | 0.41 | 0.41 | 1.00 |
| 2 | 225.3 | 425.3 | 0.46 | 0.49 | 1.84 |
| 4 | 134.5 | 436.0 | 0.83 | 1.00 | 3.46 |
| 8 | 83.3 | 576.9 | 1.57 | 1.77 | 6.03 |

-> **`recommended_parallel: 2` (measured)** — *"at 4 each stream drops to 44% of its solo speed
(floor 65%)"*. The estimate for this placement was **8**.

Read the last column first: `achieved batch` rising 1.00 -> 1.84 -> 3.46 -> 6.03 is the proof that
the requests really were batched into shared decode steps rather than queued. Without it the table
would be indistinguishable from eight requests served one after another.

Then read the two throughput columns against each other. This is exactly the shape D17 predicted and
exactly why the knee needed measuring: **the aggregate never stops climbing** — 8 slots move 1.9x
the tokens 1 slot does — while a single conversation collapses to 27% of its solo speed. A rule that
maximised aggregate would pick 8, and every user would experience a model three times slower than
the card can run it. The knee in the *useful* sense arrives at 2.

The same model on **two RTX 3090s** at that placement's own optimal (32768 per slot, f16) holds only
two slots, so the sweep is two rows and the recommendation is capped rather than measured at a knee:

| N | per-stream t/s | aggregate t/s | p95 (s) | achieved batch |
| --- | --- | --- | --- | --- |
| 1 | 301.7 | 301.7 | 0.41 | 1.00 |
| 2 | 230.5 | 433.8 | 0.48 | 1.84 |

-> `recommended_parallel: 2` (measured), *"2 is the most this placement can hold"*, with levels 4 and
8 reported as **not measured**: N requests against fewer than N slots measure a queue, not batching.

Afterwards `GET /api/models/{id}/profiles` showed `dual_3090` at basis `measured` and every other
mode still `estimated` — which is the strictness above, working.

**How to run it:** `benchmark_parallel(model_id)` (MCP), `POST /api/models/{id}/benchmark-parallel`
(a background job on `/api/benchmark/jobs/{id}`), or "Measure parallel" in the Models tab. It loads
the model once, sweeps, records a row per level, and leaves the rig as it found it. It refuses
outright while anything is serving, loading, testing or benchmarking — on a busy server the numbers
are the contention, not the model.

### Speed — `core/throughput.py`

**Decode is memory-bandwidth bound, plus a latency floor.** One token reads the active weights once
plus the busy slot's KV, and then pays a fixed cost no roofline can see: sampling, the grammar/logit
pass, llama-server's slot and HTTP bookkeeping, a CUDA graph launch at batch 1. A `--split-mode
layer` split runs devices in sequence, so the per-device times *add*:

```
t_weights = Σ_dev (active_bytes * share_dev / BW_dev) / eff_decode
t_kv      = Σ_dev (kv_read_bytes_per_slot * share_dev / BW_dev)
t_token   = t_weights + t_kv + 1.5 ms
gen_tps   = calibration / t_token
```

`eff_decode` is `0.75 - per_extra * (n_devices - 1)` with `per_extra` 0.05 dense / **0.10 MoE**,
then multiplied by **0.45 for a MoE** — a MoE decode is `MUL_MAT_ID` gathering `n_expert_used` small
scattered matrices per layer, which is occupancy- and launch-bound rather than bandwidth-bound. It
divides *only* the weight term: the weight read is the scattered, kernel-bound one, while the KV
re-read is a long contiguous stream that lands much closer to peak.

The **1.5 ms floor** is why small models are no longer absurd. A 0.92 GB model read at 1792 GB/s
"should" decode in 0.5 ms, so the pure roofline claimed 927 tok/s for a Qwen2.5-1.5B and 2063 for
SmolVLM. Nothing on this rig has ever produced those numbers; llama-server's fixed per-token cost is
1–2 ms.

**`kv_read_bytes_per_slot` is the input, not `ctx_fill × kv_bytes_per_token`.** That product is only
right for a uniform-attention model. Charging it to a Gemma-4 31B at 262k meant 258 GB of KV traffic
per token and an `est_gen_tps` of 1.9 for a model that measures 39.4.

Two fills are reported per row: `est_gen_tps` at `REFERENCE_FILL_TOKENS` (8192, or the row's own
window when that is smaller) and `est_gen_tps_full_ctx` at the whole `ctx_per_slot`.

Active bytes are all the weights for a dense model. For a **MoE they are the dense trunk in full
plus the routed share of the experts** — deliberately *not* `planner.active_weight_bytes`, which
applies `n_expert_used / n_expert` to the whole file including attention and embeddings. On the
reference rig's 122B (8 of 256 experts) the flat share says 2.7 GB active where the trunk model says
7.0 GB and the model's name implies ~7.2 GB. The planner's version is right for the planner —
under-counting there means proposing fewer slots, which cannot cause an OOM — but in a speed
estimate it would advertise the model as 2.6x faster than it is. The trunk is derived from metadata:

```
attn    = n_layer * n_embd * (n_head*head_k + n_head_kv*head_k + n_head_kv*head_v + n_head*head_v)
lm_head = n_vocab * n_embd
active  = attn + lm_head + (total - attn - lm_head) * n_expert_used / n_expert
```

Bytes are distributed across devices in proportion to the plan's `per_gpu_bytes`.

**Batched decode.** At N slots the weights are still read once per step but N slots' KV is read, and
the fixed overhead is paid once per *step* rather than once per token — which is most of why
batching helps at all:

```
t_step(N) = t_weights + N * t_kv + 1.5 ms
gen_tps_batched = calibration * N / t_step(N)
```

which is `N × gen_tps` while KV traffic is small and flattens as it catches up — the same crossover
`max_parallel_for` calls the knee. N is clamped to the knee so the catalog never advertises
throughput past the point the planner would stop adding slots.

**Prefill is FLOP bound** — it ingests many tokens per pass, so the weights amortise. The per-device
*times* add, exactly as decode's do, because a layer split is a pipeline and one prompt pass walks
the devices in turn:

```
t_prompt   = Σ_dev (2 * active_params * share_dev / FLOPS_dev) / eff_prompt
prompt_tps = calibration / t_prompt
eff_prompt = 0.35 * (0.4 if MoE else 1)
```

That is the harmonic (share-weighted) mean of the devices' FLOPS, not the arithmetic one. The
arithmetic mean says a 5090 paired with a 3090 runs at 140 TFLOPS; the pipeline runs at 106, because
the fast card waits while the slow one works. The **0.4 MoE prefill derate** exists because routing
shreds a ubatch across `n_expert`, degrading one GEMM into many skinny ones plus gather/scatter: the
122B measures 869 tok/s where the plain roofline says ~2800.

`active_params` comes from `general.parameter_count` when the GGUF carries it, else from the stored
bytes and a bits-per-weight table.

**Where it lands.** Measured on this rig, 2026-08-18:

| Placement | Measured | Estimated (uncalibrated) |
| --- | --- | --- |
| Dark-Scarlett-v2.0-31B-Q8_0 (gemma4 dense) on 2× RTX 5090, ctx 262144 f16, 1 slot | 39.4 gen / 2053 pp | 36.1 / 1190 |
| Qwen3.5-122B-A10B Q5_K_M (qwen35moe) on 5090×2 + 3090×2, ctx 262144 q4_0, 1 slot | 37.3 gen / 869 pp | 47.4 / 1124 |

### The GPU table is nominal

| GPU | Memory bandwidth | fp16 TFLOPS |
| --- | --- | --- |
| RTX 5090 | 1792 GB/s | 209 |
| RTX 4090 | 1008 GB/s | 165 |
| RTX 3090 | 936 GB/s | 71 |

**These are vendor peak figures. None was measured on this rig.** They all live in one table in
`core/throughput.py`, matched by name fragment with a VRAM-size fallback and a deliberately
pessimistic unknown-GPU profile. No real kernel reaches peak, and the gap is not a constant — it
depends on the model's shape, the quantization, the split and llama.cpp's kernels. Calibration is
what makes the numbers right; treat an uncalibrated estimate as an order of magnitude.

---

## Calibration, and what `confidence` means

Every child is launched with `--metrics`. The TTL sweeper scrapes each ready child's
`/metrics` and, once every two minutes per model, records the tokens/second **between two scrapes**
into `throughput_observations` — along with the estimator's prediction at that same moment, so a row
is self-contained and calibration never has to reconstruct a placement that has since been unloaded.

Counters (`llamacpp:tokens_predicted_total` ÷ `llamacpp:tokens_predicted_seconds_total`) rather than
the `*_seconds` gauges, because a gauge averages over the child's whole lifetime and cannot tell
"fast now" from "was fast an hour ago". A window with no decodes, or one where a counter went
backwards (the child restarted), records nothing — a zero would poison the median.

The prediction stored beside the measurement is taken at `REFERENCE_FILL_TOKENS` — **the same fill
`est_gen_tps` is quoted at**. That makes the learned factor exactly "what real traffic does ÷ what we
would have promised", so applying it corrects the number a user is actually shown. Predicting at some
other fill and correcting a differently-quoted column with the result would leave a systematic
offset no amount of data removes.

The learned factor is `measured / estimated`, taken as a **median** over four tiers, most specific
first:

1. `model+devices` — this model on this exact device set.
2. `model` — this model anywhere. A placement change moves the number, but far less than a *model*
   change does, so this outranks the neighbours.
3. `peers` — *other* models on this class of hardware (`"RTX 5090x2+RTX 3090x2"` — a label, not
   device indices, which are not stable across driver updates) with the **same density** (dense vs
   MoE) and the **same device count**, as a **median of per-model medians**, each contributing model
   needing its own two rows. Never a raw pool median: this rig holds 84 rows for one MoE and 3 for
   one dense model, so a pooled median *is* the MoE's number — and applying a sparse-MoE derate to
   every dense model on the box is exactly the bug this tier was rebuilt to stop.
4. `none` — 1.0, no correction.

Ratios outside `[0.1, 3.0]` are discarded rather than applied: a ratio that extreme means the
estimate and the measurement are describing different things, and applying it would make the catalog
confidently wrong rather than roughly right. Two matching rows are the minimum — one is noise.

**Only rows stamped with the current `estimator_version` contribute a ratio.** A ratio is a
correction to one specific formula; carrying one across a formula change teaches the estimator the
difference between two dead arithmetics. So a change to `throughput.estimate` bumps the version and
the rig is uncalibrated until fresh samples accumulate. The *measurements* survive — `measured_for`
still reads every row, whatever version wrote it, so `measured_gen_tps` and `confidence: "measured"`
are unaffected.

Calibration is looked up **per device set**, not once per model: one model's 16k row and its 262k row
are routinely placed on different cards, and a four-way split's measurements say nothing about a
single-GPU row. The model-level `calibration` block reports the factor the *recommended* row was
quoted with, so it always names a number some row in the table actually used.

| `confidence` | Meaning |
| --- | --- |
| `measured` | This exact placement and context were observed. `measured_gen_tps` is a real number. |
| `calibrated` | The estimate was corrected by a factor learned from observations. |
| `estimated` | Pure arithmetic off nominal hardware numbers. |

`measured` is reserved for an exact match, so an agent can trust the word literally.

---

## Quality first: how one load is chosen (D36)

Every surface that names *one* load — the entry's `recommended`, each `placements[].optimal`, the
`best_now` flag on the per-context table — goes through one function,
`catalog.choose_row`. Two implementations is how `/profiles` came to recommend 262144 tokens on a
q4_0 cache while `/api/catalog` recommended something else for the same model on the same hardware.

**The order is quality, then context, then slots.**

1. **The best KV cache quality that reaches the floor**, at one slot or more. Quantizing to reach a
   *bigger* window is not a trade this rule makes; quantizing to reach the floor at all is, because
   the alternative is not serving the model.
2. **The highest context at that quality** (already capped at `n_ctx_train`).
3. **Whatever slots that placement sustains** — reported, never bought. A slot count is a latency
   property; a KV cache type is a correctness one.

`recommended_basis` reads like `2x RTX 5090: f16 KV, highest ctx 131072, 1 slot`.

**There is a floor.** `models.default_ctx`, raised to `models.thinking_default_ctx` for a
thinking-capable model — the same floor the planner's context ladder never walks below (D14). Below
it the same order applies to whatever does fit, with `(below floor)` said out loud in the basis. If
nothing fits right now, `best_now` falls back to the `if_gpus_idle` column and the basis says
`if_gpus_idle`: "unload something" is actionable, "impossible" is not.

### Why quality outranks a doubled window

The KV cache ladder is `f16/f16 -> q8_0/q8_0 -> q8_0 K + q4_0 V`. **Symmetric `q4_0` is gone from
every automatic path.** K and V are not equally sensitive: with a q4_0 **K** cache and an f16 V
cache, Qwen2.5-7B reproduces only **11.7%** of its f16 tokens, while a q4_0 **V** cache alone is
nearly free, and the matched `q8_0/q8_0` pair sits at a KL divergence of 0.0018 (llama.cpp
discussion #23470). A q4_0 K cache is still reachable by asking for it explicitly.

Families differ by a factor of ten, measured as KL divergence over top-40 logprobs across ~250k
tokens against a BF16 GGUF with an f16 KV cache
(`localbench.substack.com/p/kv-cache-quantization-benchmark`):

| Family | q8_0 KV | q4_0 KV | Verdict |
| --- | --- | --- | --- |
| Gemma-4 31B dense (`gemma4`) | 0.108 | 0.524 | sensitive |
| Gemma-4 26B-A4B MoE (`gemma4`) | 0.377 | 1.088 | sensitive |
| Qwen 3.6 (`qwen35` / `qwen35moe`) | 0.024 | 0.039 | tolerant |

A **tolerant** family may take `q8_0` when it buys at least a full doubling of the window — 0.024 is
inside sampler noise, and 2x the context is a real capability. A **sensitive** family never does,
and **every unmeasured architecture is treated as sensitive**: three measurements on two families do
not describe a library of forty models, and guessing "tolerant" is the guess whose failure mode is a
server that quietly answers worse. The table lives in `core/kv_sensitivity.py`.

### The other rule, if you want it

`planner.preference: "throughput"` restores D20's original rule — the largest window at or above the
floor, preferring one that also sustains two slots. It is the right answer for a host serving many
short conversations, where a window nobody fills is worth less than a second slot. The default is
`"quality"`.

### A pinned KV type

A model whose saved settings pin `kv_cache_type` gets a `quality_notes` entry naming what the pin
costs on a KV-sensitive family, and every placement carries an `if_unpinned` block showing what the
same cards would reach at f16. Nothing here ever rewrites a saved setting: an explicit value is
honoured verbatim, and the only correct action is to show the size of the choice.

---

## Loading at an exact context: `load_recommended` (D37)

```
load_recommended(model_id, ctx_size=262144)                    # MCP
POST /api/models/{id}/load-recommended {"ctx_size": 262144}    # HTTP
```

Name the model and the window. The server walks the hardware modes in headline order (`dual_5090` ->
`dual_3090` -> `all_gpus` -> `single_5090`, or `prefer_mode`), asks the planner for **exactly** that
context per slot under the quality-first KV rule with `parallel = recommended_parallel`, and loads
the first placement that fits. Every mode is tried with eviction off before any is tried with it on,
and a model that is serving is never a candidate either way.

**This is the one load path that is strict about context.** Everywhere else a window that does not
fit steps down the halving ladder (D14), because a roomier window is a nicety. Here the window is the
request: an agent that asked for 262144 because its transcript is 200k long is not helped by silently
getting 131072 and finding out mid-conversation. So:

| Situation | Answer |
| --- | --- |
| Above `n_ctx_train` | `400`, `param: "ctx_size"`, naming the number that would be accepted |
| No placement reaches it | `507` with a `modes` list — per mode, the largest context that *would* fit and what is in the way |
| A serving model is what is in the way | the same `507`, plus `busy_models` and `retry_after_s` |
| Nothing transient is in the way | the same `507` with `retry_after_s: null` — "try again later" is bad advice when nothing will change |

`kv_min` ("give me 262144, but not at the cost of the cache") refuses a placement that only reaches
the window by quantizing, and walks on to one that can afford it.

Measured live on the scratch instance, 2026-08-19: `{"ctx_size": 32768}` loaded at exactly 32768 on
`[0, 1]` with 2 slots and an f16 cache; `{"ctx_size": 65536}` against a 32768-token model returned
the `400` above; `{"ctx_size": 12345, "prefer_mode": "dual_3090"}` loaded at exactly 12345 on
`[2, 3]` with 6 slots.

---

## Placements: which GPUs should this model use

`hardware_modes()` derives the modes from the inventory rather than hard-coding CUDA indices, so a
different box gets honest labels. On the reference rig, in this order:

| Mode | Devices | Label |
| --- | --- | --- |
| `dual_5090` | `[0, 1]` | 2x RTX 5090 |
| `dual_3090` | `[2, 3]` | 2x RTX 3090 |
| `all_gpus` | `[0, 1, 2, 3]` | all 4 GPUs (2x RTX 5090 + 2x RTX 3090) |
| `single_5090` | `[0]` | 1x RTX 5090 |

The best pair leads because it is the fastest placement that still leaves the rest of the box free,
and `placements[0]` is what the entry promotes to `recommended`.

**Each mode is computed against its own cards, idle.** "What can this model do on the two 5090s" is
a question about the hardware, not about the last ten seconds, so `headroom_fraction`,
`reserved_mb` and `excluded_devices` still apply but whatever is loaded does not. What stands in
the way *right now* travels beside it:

| Field | Meaning |
| --- | --- |
| `fits_now` | Would `optimal.load_args` load this second, without disturbing anything? Planned at exactly the slot count `load_args` asks for. |
| `would_evict` | The model ids this placement would have to stop if it were allowed to. May be empty beside `fits_now: false` — something the planner may not touch (a pinned model, a busy one) can be what is in the way. |
| `fits_now_ctx` | The largest context tier that does fit on that mode as things stand, or `null`. |
| `optimal.recommended_parallel` | Slots worth running on this mode, with `recommended_parallel_basis` (`measured` / `estimated`). This is what `optimal.load_args.parallel` asks for. |
| `ranking` | Any of `fastest`, `largest_context`, `cheapest` (fewest cards reaching the largest context). |
| `if_unpinned` | Present only when saved settings pin a KV type: what the same cards reach without the pin. |

Worked, on the reference rig, for a 17.4 GB Gemma-4 31B (iSWA, 60 layers):

| Mode | Optimal |
| --- | --- |
| `dual_5090` | 131072 ctx, f16 — 262144 needs 80 GB of f16 KV against 58 GB usable |
| `dual_3090` | 65536 ctx, f16 |
| `all_gpus` | 262144 ctx, f16 |
| `single_5090` | 32768 ctx, f16 |

In the **compact** list (`list_models` default) each mode gives its settings and `devices` but only
`recommended` carries a `load_args` object; call `model_options` for the mode you settle on. That
is a deliberate token trade — repeating one recipe per mode costs about 40% of a compact entry to
describe four loads of which at most one happens, on a payload asked for twenty-five models at a
time.

**`devices` is a load argument.** `load_model(..., devices=[0, 1])` and
`POST /api/models/{id}/load` with `"devices": [0, 1]` place one load on named cards without
touching the model's saved settings, so the next load without it goes back to the planner's choice.
A CUDA index this box does not have is a `400` naming the parameter. `kv_cache_type_v` is accepted
the same way, for the ladder's asymmetric rung; omit it and V follows K.

---

## A loaded model is credited with its own VRAM

A resident model's rows describe **reloading** it, and a reload frees the allocation it currently
holds before the replacement takes any. Judged against a machine that still contained the model,
the live 17.4 GB Gemma-4 31B — running at 262144 on `[1, 0, 2]` with the GPUs at 5.6/7.4/6.5/22.9
GiB free — was told by its own catalog that it fitted only at 262144 with a q4_0 cache spread over
three cards, against 37 GB the row itself would release.

`catalog.CreditedProbe` hands that footprint back as free VRAM, using
`Planner.instance_footprint` — exactly the figure D30's `reload_of` credits when the same reload is
really performed, and the same one the eviction ladder credits for a victim. The credit reaches
`slots_for_plan` too: it measures capacity off the planner's own probe, so a credited fit with an
uncredited capacity would have advertised one slot.

---

## Example: the full sequence

```jsonc
// 1. What is there?
list_models()
{
  "catalog_hint": "Each model lists loading options, one row per context size...",
  "models": [
    {
      "id": "trohrbaugh/Qwen3.5-122B-A10B-heretic-v2/...-Q5_K_M",
      "summary": "qwen35moe | 122B-A9.5B MoE | Q5_K_M | hybrid | tools+thinking | 82.9 GB | 262144 ctx train",
      "attention_kind": "hybrid",
      "downloaded_at": "2026-08-16T09:14:02Z",
      "state": "loaded",
      "loaded_plan": { "ctx_per_slot": 8192, "parallel": 1, "devices": [0, 1, 2, 3] },
      "recommended": {
        "mode": "dual_5090", "label": "2x RTX 5090", "devices": [0, 1],
        "ctx_per_slot": 65536, "kv_cache_type": "f16", "max_parallel": 4,
        "est_gen_tps": 37.0, "est_gen_tps_full_ctx": 31.2,
        "fits_now": true, "would_evict": [],
        "load_args": { "model_id": "trohrbaugh/...-Q5_K_M", "ctx_size": 65536,
                       "parallel": 4, "kv_cache_type": "f16", "devices": [0, 1] }
      },
      "recommended_basis": "2x RTX 5090: f16 KV, highest ctx 65536, 4 slots",
      "placements": [
        { "mode": "dual_5090", "label": "2x RTX 5090", "devices": [0, 1],
          "fits_now": true, "would_evict": 0,
          "ranking": ["fastest", "cheapest"],
          "optimal": { "ctx_per_slot": 65536, "kv_cache_type": "f16",
                       "max_parallel": 4, "est_gen_tps": 37.0,
                       "est_gen_tps_full_ctx": 31.2, "vram_mb": 58112 } }
      ],
      "options": [
        {
          "ctx_per_slot": 65536, "fits": true, "devices": [0, 1, 2, 3],
          "max_parallel": 4, "parallel_limited_by": "knee",
          "est_gen_tps": 37.0, "est_gen_tps_full_ctx": 31.2,
          "confidence": "calibrated", "best_now": true,
          "load_args": { "model_id": "trohrbaugh/...-Q5_K_M", "ctx_size": 65536,
                         "parallel": 4, "kv_cache_type": "f16" }
        }
      ]
    }
  ]
}

// 2. Need a different window, or another mode's load_args? Get the whole thing.
model_options(model_id="trohrbaugh/...-Q5_K_M")

// 3. Load what you chose, verbatim -- devices included.
load_model(model_id="trohrbaugh/...-Q5_K_M", ctx_size=65536,
           parallel=4, kv_cache_type="f16", devices=[0, 1])

// 4. Confirm -- and read `busy` before the next load.
server_status()
```

Working from the newest download? `list_models(limit=5)` returns the five most recent after the
other filters, which is usually what "the model I just got" means.

---

## Caveats

- Speed is an estimate until it is measured. See the nominal-numbers warning above and
  [`LIMITATIONS.md`](LIMITATIONS.md).
- `est_gen_tps` is quoted at 8192 tokens of context, not at the row's own window. The row's
  `est_gen_tps_full_ctx` is the other end; a real conversation moves between them.
- `fits` is true of one instant. Another client can load something a second later; the load is still
  planned against live VRAM, so the worst case is a refusal with numbers, never a degraded load.
- The concurrency bound's VRAM half is exact, but the knee inherits D17's approximations:
  `CTX_FILL_FRACTION = 0.5` and the MoE knee derate of 0.5 are deliberate, unmeasured, and err
  toward *fewer* slots. `recommended_parallel_basis: "measured"` is the signal that a row has escaped
  that estimate; one 1.5B model on one 3090 has, and nothing else on this rig yet has.
- The slot rule's two thresholds (65% of solo per-stream, +15% aggregate per doubling) are a stated
  priority, not a measurement. A host that genuinely wants maximum aggregate throughput would set
  them differently, and there is no config knob for that today.
- A model whose per-layer KV geometry cannot be derived reports `attention_kind: "unknown"` and one
  slot. Read that as "distrust these numbers", not as "this model is cheap".
- A model whose GGUF metadata could not be parsed at all is listed with an empty `options` list and
  an `unavailable` note rather than hidden — the catalog is the only place a user would find out the
  file is broken.
