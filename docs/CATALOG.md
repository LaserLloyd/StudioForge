# The model catalog

One call that tells an agent what it can run, at what context, how fast, and how many
conversations at once — with the exact arguments to load each choice.

Before the catalog, choosing a model meant a `/v1/models` call, a `/profiles` call per model,
arithmetic about KV caches, and a guess about speed. The catalog answers all of it per model, as a
table of rows the agent hands straight back.

---

## For an LLM: the whole workflow

```
list_models()                      -> catalog, newest download first
   pick the row with recommended: true
load_model(**row["load_args"])     -> loaded, serving
server_status()                    -> confirm, see VRAM and queue depth
```

Only reach for `model_options(model_id)` when the recommended row is not what you need — a bigger
context window, or more concurrent conversations than its `max_parallel`.

**Rules that make this reliable:**

- Take the `recommended` row unless your task needs more context than it offers.
- Pass `load_args` **verbatim**. Do not add fields, convert units or recompute anything.
- `fits: false` means it will not load *right now*. Check that row's `if_gpus_idle`: if that says
  `fits: true`, the VRAM exists and something else is holding it — `unload_model` on another model
  makes the row available.
- Match your client concurrency to `max_parallel`. Beyond it, llama.cpp queues rather than
  refusing, so extra streams show up as latency, not errors (watch `requests_deferred` in
  `/api/status`).
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
| `GET /api/models/{id}/profiles` | The *hardware-mode* cut instead of the context cut: best achievable on the 5090 pair, the 3090 pair, and the whole rig. Same columns. |

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
  "recommended": true
}
```

`ctx_per_slot` is the context **each conversation** gets. `--ctx-size` is the *total* across slots
(DECISIONS.md D4), so the engine is launched with `ctx_per_slot * parallel` and every slot really
gets the number in the row.

`vram_mb` is sized at this row's own `max_parallel`, so it describes the load `load_args` produces
— not whatever slot count the planner happened to pick while checking the fit.

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

## The recommendation rule

Exactly one row per model is marked `recommended`, so "pick the recommended one" always has an
answer.

**There is a floor.** `models.default_ctx`, raised to `models.thinking_default_ctx` for a
thinking-capable model. It is the same floor the planner's context ladder never walks below (D14),
and the catalog must agree with it: the two surfaces disagreeing about what "enough context" means is
how an agent ends up loading a window the server itself would have refused to settle for.

**The floor outranks the second slot.** This is the one thing the rule got wrong before: it would
take 16384 tokens with two slots over 32768 with one, and 16k is where an OpenClaw agent's tool
transcript stops fitting. A queued second conversation is a latency problem; a window that cannot
hold the task is a failed task. (It picked exactly that for the resident 122B, off a knee that was
itself wrong — D22.)

In order:

1. **Chat-class models** — the highest context **at or above the floor** that also sustains at least
   two slots. One slot means every concurrent request queues behind the one before it, so *above the
   floor* the second conversation is worth a context doubling.
   `recommended_basis: "highest ctx >= floor with max_parallel >= 2"`.
2. Otherwise (embeddings, rerankers, or when no row above the floor reaches two slots) — the highest
   context at or above the floor. `"highest ctx that fits >= floor"`.
3. If nothing reaches the floor — the highest context that fits, said out loud:
   `"highest ctx that fits (below floor)"`. A small window beats no recommendation, but the basis
   admits what happened.
4. If nothing fits right now — the same three-way preference applied to the `if_gpus_idle` column,
   and `recommended_basis` says `if_gpus_idle`. "Unload something" is actionable; "impossible" is
   not.

`recommended_basis` on each model names which rule fired.

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
      "recommended_basis": "highest ctx >= floor with max_parallel >= 2",
      "options": [
        {
          "ctx_per_slot": 65536, "fits": true, "devices": [0, 1, 2, 3],
          "max_parallel": 4, "parallel_limited_by": "knee",
          "est_gen_tps": 37.0, "est_gen_tps_full_ctx": 31.2,
          "confidence": "calibrated", "recommended": true,
          "load_args": { "model_id": "trohrbaugh/...-Q5_K_M", "ctx_size": 65536,
                         "parallel": 4, "kv_cache_type": "f16" }
        }
      ]
    }
  ]
}

// 2. Need more room than the recommended row? Get the whole table.
model_options(model_id="trohrbaugh/...-Q5_K_M")

// 3. Load the row you chose, verbatim.
load_model(model_id="trohrbaugh/...-Q5_K_M", ctx_size=65536,
           parallel=4, kv_cache_type="f16")

// 4. Confirm.
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
  toward *fewer* slots.
- A model whose per-layer KV geometry cannot be derived reports `attention_kind: "unknown"` and one
  slot. Read that as "distrust these numbers", not as "this model is cheap".
- A model whose GGUF metadata could not be parsed at all is listed with an empty `options` list and
  an `unavailable` note rather than hidden — the catalog is the only place a user would find out the
  file is broken.
