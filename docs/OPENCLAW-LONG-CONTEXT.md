# Long context on StudioForge — what OpenClaw needs to know

**Reference rig:** 2× RTX 5090 (32 GiB) + 2× RTX 3090 (24 GiB), Windows 11 · numbers below measured
2026-08-18 unless dated otherwise. `<rig>` in the `curl` examples is a placeholder — substitute
your own host or address.

This page answers one question: *which model do I ask for, and what context will I get?*

---

## The short version

**Ask the catalog, do not read a table.** `list_models` gives every model one row per context tier,
each with `fits`, `devices`, `max_parallel` and two speed columns, and marks exactly one
`recommended: true`. That answer is computed against the VRAM free *at that instant*, which a page
like this one cannot be.

You do **not** need to send `ctx_size` on an inference request. If you do, it is honoured exactly
and never adjusted.

Three things are worth understanding before reading a row, because they are why the numbers look
the way they do.

---

## 1. The floor: 32768, and it outranks concurrency

The recommended row is the highest context **at or above `models.default_ctx`** (raised to
`models.thinking_default_ctx` for a thinking model) that also sustains two conversations; if
nothing above the floor sustains two, the floor wins and the second slot is dropped. On this rig
both values are **32768**.

That ordering is deliberate (D14, D22). A queued second conversation is a latency problem. A window
that cannot hold an OpenClaw tool transcript is a failed task. The old rule took 16k-with-two-slots
over 32k-with-one and picked exactly that for the resident 122B; it does not any more.

A model whose own trained window is below the floor — SmolVLM at 8192, Gemma-2-2B at 8192 — is
still recommended, with `recommended_basis: "highest ctx that fits (below floor)"`. The basis says
what happened rather than pretending.

## 2. Attention kind: why some 256k windows are nearly free

The model entry carries `attention_kind`, derived from the GGUF's per-layer geometry rather than
from its architecture string.

| kind | What is cached | Consequence |
| --- | --- | --- |
| `full` | every layer holds the whole window | context is expensive and the two speed columns diverge fast |
| `iswa` | Gemma 3/4: five sliding-window layers per full one | a 31B keeps ~85 KiB/token of *effective* KV at 262k where the uniform figure says 1920 KiB |
| `hybrid` | Qwen3.5/3.6/3.8: only every 4th layer holds KV, the rest are Gated-DeltaNet recurrent layers with a fixed per-sequence state | a 27B keeps a cache on 16 of its 65 blocks |
| `unknown` | the geometry could not be derived | treat every KV number for that model as unreliable — **not** as the cheap case |

This is the whole reason an iSWA or hybrid model offers wide rows that *also* keep several slots,
while a same-sized full-attention model does not. Until 2026-08-18 the planner charged all of them
the uniform price, which is why every Gemma-4 row used to read `max_parallel: 1 (vram)` with 34 GB
free, and why a Qwen3.5-27B at 262k was spread across all four GPUs on a KV estimate of 65 GB
against a real ~16 GB.

## 3. Two speed columns, and generation slows as the window fills

`est_gen_tps` is one stream with about 8k tokens in the window — an ordinary turn.
`est_gen_tps_full_ctx` is the same stream with the window nearly full. Every decode step re-reads
the KV cache, so the second is always the smaller, and **the truth is between them**. How far apart
they sit is the real price of choosing a wide row — and on a hybrid or iSWA model they sit close
together, which is the point of the previous section.

`confidence` says how much to trust either: `measured` (this exact placement and context were
observed), `calibrated` (corrected by a factor learned from real traffic), `estimated` (nominal
vendor bandwidth and FLOPS — an order of magnitude, not a promise).

---

## Three real models, whole tables

Straight out of `/api/catalog` on 2026-08-18, on a rig with one small model already resident.
`gen` is `est_gen_tps` / `est_gen_tps_full_ctx`.

### Qwen3.5-122B-A10B Q5_K_M — `qwen35moe`, hybrid, 81 GiB, 8 of 256 experts

| ctx/slot | devices | KV | VRAM | slots | gen | |
| ---: | --- | --- | ---: | --- | --- | --- |
| 16384 | all 4 | f16 | 91.2 GB | 4 (knee) | 47.5 / 47.1 | |
| **32768** | all 4 | f16 | 90.9 GB | **2 (knee)** | 47.5 / 46.4 | **recommended** |
| 65536 | all 4 | f16 | 90.8 GB | 1 (knee) | 47.5 / 45.1 | |
| 131072 | all 4 | f16 | 92.3 GB | 1 (knee) | 47.5 / 42.7 | |
| 262144 | all 4 | f16 | 95.4 GB | 1 (knee) | 47.5 / 38.6 | `confidence: measured` |

Every row fits; the choice is entirely context versus slots. The recommendation lands on the floor
because that is the widest row still serving two conversations. Measured on this placement:
**36.7** tok/s generation — so the uncalibrated estimate is ~1.3× optimistic, and the 262144 row is
reported as `measured` rather than estimated. Before 2026-08-18 this model was recommended at
**16384** (below the floor) off a knee that was itself wrong, quoted at ~143 tok/s.

Note the cache: **f16 at 262k on a 122B**. It used to be forced onto q4_0, because the planner
charged KV for all 48 layers of a model that caches every fourth one.

### Dark-Scarlett-27B v2.0 Q5_K_M — `qwen35`, hybrid, 21 GiB

| ctx/slot | devices | KV | VRAM | slots | gen | |
| ---: | --- | --- | ---: | --- | --- | --- |
| 16384 | 1× 5090 | f16 | 28.2 GB | 3 (vram) | 53.3 / 52.4 | |
| 32768 | 1× 5090 | f16 | 26.8 GB | 1 (vram) | 53.3 / 50.8 | |
| 65536 | 1× 5090 | f16 | 28.9 GB | 1 (vram) | 53.3 / 47.9 | |
| **131072** | 2× 5090 | f16 | 50.0 GB | **3 (vram)** | 50.1 / 40.9 | **recommended** |
| 262144 | 2× 5090 | f16 | 41.5 GB | 1 (vram) | 50.1 / 34.2 | |

The 262144 row now fits on **two** GPUs at full f16 quality, 41 GB total. It used to be planned at
65 GB across all four. Note also that VRAM is not monotonic down the column — each row is sized at
its *own* `max_parallel`, so the 3-slot 131k row costs more than the 1-slot 262k one.

### Gemma-4 Ortenzya 31B NVFP4 — `gemma4`, iSWA, 18.0 GB, on the 5090 pair

| ctx/slot | slots | gen |
| ---: | --- | --- |
| 16384 | 8 (cap) | 56.4 / 55.2 |
| 32768 | 8 (cap) | 56.4 / 53.1 |
| 65536 | 5 (vram) | 56.4 / 49.1 |
| **131072** | **3 (vram)** | 56.4 / 42.8 |
| 262144 | 1 (vram) | 56.4 / 34.1 |

An iSWA model reaching the **slot cap** at 32k is exactly what the per-layer KV rebuild bought:
every one of these rows used to read `max_parallel: 1 (vram)`, and the 262k row used to be quoted
at **1.9 tok/s**. Note `est_gen_tps` is flat down the column (that column is always quoted at ~8k
of fill) while `est_gen_tps_full_ctx` falls by 40% — that gap *is* the cost of the wider window.

The quantization still decides how many slots you get. Dark-Scarlett-v2.0-31B **Q8_0** is the same
architecture and the same iSWA geometry at **30.4 GB** of weights against Ortenzya's 18.0, and its
table runs 8 slots at 16k → 6 at 32k → 3 at 65k → 2 at 131k (recommended) → 1 at 262k. It is also
the rig's best-anchored model: **38.8** tok/s measured on the 5090 pair against an uncalibrated
estimate of 35.3, which is why its 262144 row reads `confidence: measured`.

---

## Before you download: `repo_details`

`repo_details(repo_id)` reads the model's GGUF header remotely and answers, per quant, which
context each GPU placement reaches. `unsloth/Qwen3.8-27B-GGUF` on this rig:

| quant | 1× 5090 | 2× 5090 | all 4 |
| --- | --- | --- | --- |
| BF16 (51.8 GiB) | — weights alone do not fit | 32k at q8_0 | 256k |
| Q8_0 (27.9 GiB) | — | 256k | 256k |
| Q5_K_M (19.3 GiB) | 128k at q8_0 | 256k | 256k |
| IQ2_M (10.5 GiB) | 256k | 256k | 256k |

`max_ctx` in that payload is the largest window at a full-quality **f16** cache; `max_ctx_q8`
appears only when a q8_0 cache reaches further. Every cell is computed by the same planner a real
load uses, against **idle** capacity — "what would this model get?" must not change because
something happened to be loaded when you asked.

---

## Picking a model programmatically

Prefer the catalog, which is the context cut:

```
list_models(limit=5)          -> options rows, one per context tier, one recommended
model_options(model_id)       -> the full table for one model
```

`GET /api/models/<url-encoded-id>/profiles` is the *hardware-mode* cut of the same data — the best
achievable on the 5090 pair, on the 3090 pair, and on the whole rig — and it now carries
`est_gen_tps_full_ctx` too:

```bash
curl -s "http://<rig>:1234/api/models/<url-encoded-id>/profiles"
```

- `dual_5090` — fastest, and leaves both 3090s free for a second model.
- `dual_3090` — slower, keeps the 5090s free.
- `all_gpus` — largest capacity; use when a model does not fit on a pair.

Nothing is loaded by asking. Both surfaces are pure arithmetic over one VRAM snapshot.

---

## New models work automatically

Nothing on this page is per-model configuration. Every number is derived at scan time from the
GGUF's own metadata — `n_ctx_train`, layer counts, KV head counts, the sliding-window keys and
`full_attention_interval` — then combined with live VRAM. A model downloaded tomorrow is planned
the same way:

1. Download it (`repo_details` → `download_model`, the GUI **Download** tab, or `sfctl download`).
   A finished download triggers a rescan automatically.
2. Ask for it by name, or read its catalog row first if you want to choose deliberately.

The one thing that overrides all of this is a **pinned per-model `ctx_size`** — an explicit pin
always wins, and appears as an extra row in the model's table. If a model seems stuck below what
its options say, check for a pin:

```bash
curl -s "http://<rig>:1234/api/models/<url-encoded-id>/settings"
```

---

## Practical notes for long-context requests

- **First request on a cold model pays the load.** 262k models take tens of seconds to load, and a
  genuinely long prompt then takes minutes to process. That is prompt processing, not a hang.
- **Streaming is strongly preferred above ~64k of prompt.** Non-streaming requests are bounded by
  `server.request_timeout_s` (900s), which a very large prefill can exceed.
- **Keep the model resident.** Idle TTL is 900s here; after an unload the next request re-processes
  the whole transcript. For a primary agent model, pin it (`ttl_s: 0`) so it is never evicted.
- **Match your concurrency to `max_parallel`.** Beyond it llama.cpp queues rather than refusing, so
  extra streams show up as latency, not errors — visible only in `requests_deferred` on
  `/api/status`.
- **A rejection is always actionable.** The error names the binding constraint and the largest
  context that would fit, so retrying at the suggested size succeeds rather than guessing.
- **RoPE scaling is never applied.** A tier above the model's trained window is not offered at all;
  serving beyond it degrades quality quietly, which is worse than a smaller window (D14).
