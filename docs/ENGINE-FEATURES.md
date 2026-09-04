# Engine features

llama.cpp grows knobs faster than anything can track by hand, and `b10425` proved the expensive
failure mode: it renamed the whole speculative-decoding surface and *accepted the old spellings
while ignoring them*. A flag that is silently ignored looks exactly like a feature that does not
help.

So StudioForge never passes an optional flag on faith. Every build's `--help` is parsed once into
`engines/<tag>/features.json` — the flag list **and** the value lists (`--split-mode
{none,layer,row,tensor}`, `--spec-type`'s eleven drafting types, `--flash-attn [on|off|auto]`) —
and a flag reaches a child only when the active engine declares it. If the help cannot be read at
all, the build "advertises nothing" and the launch falls back to the flag surface that predates
this gating, rather than guessing.

See it for yourself:

```
studioforge capabilities          # the ENGINE FEATURES block
GET /api/capabilities             # engine.features / engine.feature_rows
```

The rest of this page is one section per feature: what it is, what it defaults to, whether it can
cost quality, how to change it, and how to tell whether it did anything. Every number quoted here
was measured on the reference rig (2× RTX 5090 + 2× RTX 3090, PCIe, no NVLink) against `b10425`;
the full runs are in DECISIONS.md **D38**.

---

## The rule

> Only lossless features are on by default. Anything with a quality cost, and anything upstream
> calls experimental, is opt-in and labelled.

| Feature | Flag | Default | Quality cost |
| --- | --- | --- | --- |
| Speculative decoding | `--spec-type` | **auto** (on where it pays) | none — distribution-preserving |
| Host-RAM prompt cache | `--cache-ram` | **on**, a shared pool of 25% of RAM capped at 32 GiB | none |
| Prompt-prefix reuse | `--cache-reuse 256`, `--slot-prompt-similarity 0.3` | **on** | none |
| Flash attention | `--flash-attn on` | **on** | none |
| Partitioned KV pool | `--no-kv-unified` | **on** above one slot | none (a capacity trade, not a quality one) |
| Engine auto-fit | `--fit off` | **off** | n/a — it would break the GPU-only policy (D11) |
| Larger micro-batch | `-ub` | off (engine's 512) | none, costs VRAM (modelled per device, D40) |
| Tensor parallelism | `--split-mode tensor` | off | none, but EXPERIMENTAL upstream |
| GPU sampling | `--backend-sampling` | off | none claimed, but EXPERIMENTAL upstream |
| Engine idle sleep | `--sleep-idle-seconds` | **never passed** | StudioForge owns TTL; two idle timers would fight |

---

## Speculative decoding

**What it is.** A cheap predictor proposes several tokens, the real model verifies them in one
batched pass, and rejected tokens are resampled from the true distribution. The output
distribution is unchanged — this is speed, not a trade. What varies is how much speed, which
depends entirely on how often the proposal is right.

**Default.** `spec_type: auto`, resolved at launch in this order:

1. **`draft-mtp`** when the GGUF carries `nextn_predict_layers >= 1`. The model's own multi-token
   prediction head: no second model, no extra VRAM.
2. **`draft-simple`** when the model has a `draft_model_id` attached. The classic draft-model path.
3. **`ngram-mod`** for thinking models and MoE models. Draftless: it drafts from n-grams in the
   text already generated, which is what a reasoning model re-treading its own thoughts, or a
   code-iteration turn, actually looks like. ~16 MiB of host state.
4. **`none`** otherwise.

Any explicit value wins verbatim, including a comma list (`draft-mtp,ngram-mod`). A value the
active engine does not offer is **refused at load with the list of what it does offer** — never
passed and ignored.

**Measured** (Qwen3.8-27B Q5_K_S, `qwen35`, `nextn_predict_layers=1`, one RTX 3090, 8k context,
four *distinct* 256-token prose prompts):

| `--spec-type` | tok/s | vs none | draft acceptance |
| --- | --- | --- | --- |
| `none` | 37.8 | — | — |
| `draft-mtp` (n_max 3) | **50.7** | **+34%** | 0.53 |
| `draft-mtp` (n_max 4) | 47.5 | +26% | 0.45 |
| `ngram-mod` | 37.9 | +0.4% | no drafts emitted at all |
| `draft-mtp,ngram-mod` | 47.1 | +25% | 0.45 (identical draft counts — the combo added nothing) |

On repetitive work the picture inverts: the same model on a code-rewrite turn measured **+95% with
`draft-mtp`** and **+114% with `ngram-mod`**. That is the whole point of the `auto` ladder — the
two strategies are good at different things and `ngram-mod` costs nothing when it is wrong.

**Draft depth.** `--spec-draft-n-max` defaults to **3**, which is the engine's own default. It was
16 here, under a comment claiming *that* was the engine default. Deeper is not better: at n_max 4
acceptance fell from 0.53 to 0.45 and throughput with it, because every rejected token was verified
for nothing. Override per model with `spec_draft_n_max`.

**How to read its effect.** Not from `/props` — it reports `speculative.types: "none"` while
actively drafting. The truthful signals are:

* `/slots[].speculative` — `true` means drafting is *configured* (it is `true` for `ngram-mod`
  even when no draft is ever emitted).
* a completion's `timings.draft_n` and `timings.draft_n_accepted` — this is the only pair that
  says drafting is *working*. Acceptance below ~0.6 usually means the draft is costing more than
  it saves.

**Benchmarking caveat.** `ngram-mod` learns from what it has already produced. Sending the same
prompt three times measured **+751%** on a 27B; four distinct prompts measured **+0.4%**. Vary the
prompt or you are benchmarking a cache.

**Turning it off.** Set the model's `spec_type` to `none`.

---

## Prompt-prefix reuse (`--cache-reuse`, `--slot-prompt-similarity`)

**What it is.** Three mechanisms, all on by default, all quality-neutral (a cache of computed KV,
not an approximation of it):

- **Prefix reuse.** With prompt caching on (the engine default, `--cache-prompt`), a request that
  lands on a slot keeps the longest common prefix of the slot's previous prompt and skips its
  prefill. No flag is needed for this; it is what recovers a shared story bible.
- **Chunk reuse** (`--cache-reuse 256`). *After* the point of divergence, any run of at least 256
  identical tokens still in the slot is shifted into place instead of recomputed. Second-order:
  it matters for an edit in the middle of a prompt, not for the shared prefix.
- **Slot routing** (`--slot-prompt-similarity 0.3` above one slot). A request is sent to the idle
  slot whose previous prompt matches it best, so an agent's near-identical prompts land where the
  prefix already is; the engine's 0.10 default scatters them.

**Two limits that no setting removes.** A slot only reuses what *it* holds — there is no sharing
across slots, so at `parallel: N` the first N concurrent requests each prefill the whole prompt.
And a hybrid or recurrent model (`attention_kind: hybrid`) reuses back to the nearest *context
checkpoint* (`--ctx-checkpoints 32`, `--checkpoint-min-step 8192`), which sit at user-message
starts; the part of a prompt that differs must begin a user message. The arithmetic and the
recipe are in [OPENCLAW-LONG-CONTEXT.md §4](OPENCLAW-LONG-CONTEXT.md#4-concurrent-requests-that-share-a-prefix).

**What a child is really running with.** A per-model setting of `null` means *inherit*, and
inherit is not off. Every instance view carries `effective` (D54) — `cache_prompt`,
`cache_reuse`, `cache_ram_mib`, `cache_idle_slots`, `cont_batching`, `kv_unified`,
`slot_prompt_similarity`, `parallel`, `ctx_per_slot`, batch sizes, checkpoints, `spec_type`,
`flash_attn` — parsed from the final argv (last occurrence wins, so `extra_flags` are in the
answer) with the engine's own defaults filled in and a `sources` map saying which is which.
`inert` names a saved setting the child could not see; `summary` is the one-line version. It is on
`/api/status loaded[]` (with the redacted `launch_args`), the `GET /api/models` row, `/introspect`,
and, as a compact subset, on every MCP instance row. `cont_batching: false` now emits
`--no-cont-batching` where the engine has it and logs `setting_inert` where it does not; before
D54 it emitted nothing.

**How to read its effect.** `timings.cache_n` (reused) against `timings.prompt_n` (processed) on
the final response of every completion, streamed or not; `usage.prompt_tokens_details.cached_tokens`
on non-streaming responses (streams need `stream_options.include_usage`); and per child the
`prompt_cache` block on `/api/status loaded[]`, from `llamacpp:prompt_tokens_cached_total`.
`usage.prompt_tokens` is the prompt's size and cannot show a hit.

**Measuring it.** `scripts/measure_prefix_cache.py --model <id>` on an idle rig (it refuses to run
under a lease or beside active requests, and never loads a model).

---

## Host-RAM prompt cache (`--cache-ram`)

**What it is.** When a slot's prompt prefix is evicted, the engine can keep it in *system* memory
instead of throwing it away, and restore it on the next request that carries the same prefix. It
is the other half of `--cache-reuse`: reuse recovers a prefix still in the slot, this one recovers
a prefix that left it. Exactly the OpenClaw pattern — a long, near-identical agent prompt arriving
again after another model borrowed the slot.

**Default.** On. `engine.cache_ram_mb: auto` = 25% of system RAM capped at 32 GiB (32 GiB on this
128 GiB box). Set an integer for MiB, `0` to disable, `-1` for the engine's "no limit".

The two settings have **different scopes** (D50). `auto` is a machine-wide **pool**: each model is
granted what the other loaded models are not already holding, floored at 4 GiB so the last one in
still has a cache at all, and the grant it got is `cache_ram_mib` on the instance — on
`/api/status loaded[]`, on `/introspect` → `instance`, and inside `effective` on the
`GET /api/models` row (D54). An explicit integer is **per child, verbatim** — you named a number, every model
gets that number, and four residents can then hold four times it. Before D50 `auto` behaved that
way too, which is how a cap documented as unable to make the box swap came to promise 128 GiB of a
128 GiB machine. If the floor pushes the total past the pool, the launch logs
`cache_ram_pool_oversubscribed` with the numbers — lower the setting on a box where host RAM is
tight.

**Quality cost.** None: it is a cache of computed KV, not an approximation of it. **VRAM cost:
none** — measured identical VRAM (1492 MiB) at `--cache-ram 8192` and at `32768`.

**How to read its effect.** A hit shows up as `timings.cache_n` rising on the final response —
`prompt_n` is what was actually processed, and `usage.prompt_tokens` is the prompt's size and
never moves. Per child, `prompt_cache` on `/api/status loaded[]` carries the lifetime
`cached_total` / `processed_total` / `hit_ratio` from the engine's own counters (see the
prefix-reuse section above).

---

## KV pool shape (`--kv-unified` / `--no-kv-unified`)

**What it is.** `--ctx-size` is the *total* KV budget across slots (D4). Partitioned, each slot
gets an equal, guaranteed slice. Unified, all slots share one pool, so a lone conversation can use
all of it — at the same total VRAM.

**Default.** Partitioned, and now **explicitly** so: multi-slot launches pass `--no-kv-unified`.
The engine's own default is "enabled if the number of slots is auto", and StudioForge always passes
an explicit slot count — so partitioned was already what happened, but only by accident of another
setting. A guarantee that depends on a flag you happen not to pass is not a guarantee.

**Measured** (0.5B, `--parallel 2 --ctx-size 16384`):

| | `n_ctx` per slot | VRAM | one 12k request | two concurrent 12k requests |
| --- | --- | --- | --- | --- |
| partitioned (default) | 8192 | 997 MiB | **400** up front, naming the limit | **400** each, up front |
| `--kv-unified` | 16384 | 1005 MiB | **accepted** | **500 "Context size has been exceeded"**, mid-generation |

**Which to want.** For an agent host, the partitioned pool: `ctx_per_slot` in the catalog is then
literally true, and an over-long request is refused before any work starts. For a single-user
long-context model, `kv_unified: true` per model doubles (or `parallel`-times) the window a lone
conversation can reach for free. The failure mode of unified is a 500 *during* generation, which is
the worst moment to find out.

---

## Tensor parallelism (`--split-mode tensor`)

**What it is.** Instead of dealing whole layers out to devices and running them in a pipeline
(`layer`), every device holds a slice of every weight matrix and of the KV, and they work at the
same time — with two cross-device all-reduces per layer.

**Default.** Off. `layer` remains the default and `split_mode: auto` still resolves to `layer`
unless every gate passes.

**Measured** (Qwen2.5-1.5B Q4_K_M, two RTX 3090s, 8k context):

| mode | generation | prompt |
| --- | --- | --- |
| one 3090 | 353 tok/s | 2804 tok/s |
| two, `layer` | 344 tok/s | 2722 tok/s |
| two, `tensor` | **294 tok/s** | **1182 tok/s** |
| two, `row` | fails: `device CUDA2 does not support split buffers` | |

A small model is the worst case for tensor mode — the per-layer sync is fixed while the halved
weight read shrinks with the model — so a 31B may well win. **Measure it**: the benchmark now
offers a tensor variant of every multi-GPU mode for an eligible model.

**Prerequisites**, all checked before the child is launched:

* at least two devices in the placement;
* the engine lists `tensor` in `--split-mode`;
* flash attention on — llama.cpp exits with `SPLIT_MODE_TENSOR requires flash_attn to be enabled`;
* an unquantized KV cache (f16/bf16/f32). Upstream documents quantized KV as unimplemented here.
  b10425 does *not* enforce it — a scratch load with `q8_0` started and answered correctly — so
  this one is StudioForge policy, and it is the gate most likely to be relaxed later;
* a dense, non-hybrid model: MoE and state-space/recurrent architectures are refused upstream.

Set `split_mode: tensor` and an ineligible model is **refused with the reasons**. Set
`split_mode: auto` and it silently falls back to `layer` with the reason recorded on the plan. The
difference is deliberate: someone who typed "tensor" and quietly got "layer" would go on to
benchmark the wrong thing.

**Interaction.** `--backend-sampling` is downgraded to CPU sampling under tensor mode
(`backend sampling not supported with SPLIT_MODE_TENSOR; using CPU`), with a log line, not an error.

---

## Micro-batch size (`-ub` / `--ubatch-size`)

**What it is.** How many tokens the engine processes per forward pass while ingesting a prompt.
Bigger means fewer, fatter passes — faster prefill, larger compute buffers.

**Default.** Unset, i.e. the engine's 512. `engine.ubatch_size` sets a global default and a
per-model `ubatch_size` overrides it.

**Measured** (1.5B, one RTX 3090, 5166-token prompt):

| `-ub` | prompt processing | VRAM |
| --- | --- | --- |
| 512 (default) | 15232 tok/s | 1492 MiB |
| 1024 | 17307 tok/s (+14%) | 1562 MiB (+70) |
| 2048 | 18061 tok/s (+19%) | 1702 MiB (+210) |

**Quality cost.** None. **Why it is off:** it is a VRAM-for-prefill trade, and a default
should not make it for you. It is *safe* to raise (D40): the planner charges the growth explicitly
-- 128 bytes per extra token per `n_embd`, **on every device** of the placement, which covers the
per-card growth measured on a two-way split (113-126 B/token/`n_embd`) with a little room -- so a
larger micro-batch costs context rather than risking an out-of-memory, and the supervisor raises
`--batch-size` to match (llama.cpp silently clamps `n_ubatch` to `n_batch`). Set
`engine.ubatch_size` globally or `ubatch_size` per model, and benchmark it with
`ubatch_sizes=(1024, 2048)`.

---

## GPU sampling (`--backend-sampling`)

Runs token sampling on the GPU instead of the CPU. `b10425` labels it *experimental* and it is
silently unavailable under tensor split. `engine.backend_sampling: false` by default; no quality
claim either way has been measured here, which under the quality-first rule is itself the reason it
is off.

---

## Flags StudioForge will never pass

* **`--sleep-idle-seconds`** — the engine's own idle sleep. StudioForge owns model lifetime through
  TTL and the sweeper; a second idle timer inside the child would unload state the supervisor still
  believes is resident.
* **`--fit on`** (and `--fit-target` / `--fit-ctx`) — `--fit off` on every launch. The planner
  already decided placement and context against live free VRAM; letting the engine re-plan around
  us is a silent partial-offload path, and the GPU-only policy exists to make an over-commit fail
  loudly instead (D11).
* **`--n-gpu-layers` anything but 999**, and anything in the `--cpu-moe` / `--override-tensor`
  family — same reason.
* **The `--draft*` family** — removed in b10425, accepted and ignored. `--spec-*` only.
