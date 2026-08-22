# Benchmarking a model — the playbook

Written for the agent (OpenClaw) as much as for a person: every step names the exact call. Give
this file to the agent as a skill or knowledge document when it is asked to "benchmark", "find the
fastest placement", or "work out how many slots" a model is worth.

There are **two benchmarks**, and they answer different questions:

| Benchmark | Question it answers | How to start it |
| --- | --- | --- |
| **Placement** | Which cards, and which split mode, run this model fastest? | `POST /api/models/{id}/benchmark` (REST) |
| **Parallel** | How many concurrent conversations is this placement worth? | `benchmark_parallel` (MCP tool) or `POST /api/models/{id}/benchmark-parallel` |

Run them in that order: placement first, then the slot sweep on the winning placement.

---

## 0. Before you start — the three rules

1. **The box must be quiet.** Call `server_status()` and check `busy`: `active_requests` 0,
   `loading` empty, `testing` null, and no benchmark running. A benchmark refuses otherwise and
   returns `retry_after_s` — wait, do not retry in a loop.
2. **Never benchmark a model someone is mid-conversation with.** The benchmark unloads the model
   and reloads it fresh for every mode, which drops its prompt cache. On an RP session with a
   100k-token prompt that is minutes of reprocessing on the next turn. Benchmark when the session
   is over, or benchmark a *different* copy/quant. (A *pinned* model is safe to benchmark: the
   run's unloads are housekeeping, not a deliberate unload, so the reconciler brings it back
   within a sweep of the run ending — D41.)
3. **Benchmark at the context you will actually run.** Speed and slot count both depend on
   `ctx_size`. Take it from the `list_models` row you intend to use (`ctx_per_slot`), or from the
   user. Do not accept the default silently.

Each mode runs under its own **GPU lease** (the cards belong to the benchmark for that mode):
idle models on those cards are unloaded, nothing else can load there, and a model that is
*busy* on those cards fails that one mode by name instead of polluting its number. You do not
have to reserve anything yourself. `server_status().leases` shows the lease while it runs.

---

## 1. Placement benchmark

**Which modes exist for this model** (they differ per model and per rig):

```
GET /api/models/{id}/benchmark/modes
```

Each entry has `key`, `devices`, `split_mode`, `applicable` and, when not applicable, the reason
(typically "does not fit on one card"). On the reference rig (2× RTX 5090 + 2× RTX 3090) the keys
are `rtx-5090-x1`, `rtx-5090-x2`, `rtx-5090-x2-tensor`, `rtx-3090-x1`, `rtx-3090-x2`,
`rtx-3090-x2-tensor`, `all`, `all-tensor`. The `-tensor` variants exist only for models that
pass the tensor-split gates (dense, non-hybrid, flash attention on, unquantized KV).

**Start it** — a background job, because a four-mode run on a 30 GB model is minutes:

```
POST /api/models/{id}/benchmark
{"modes": ["rtx-5090-x2", "rtx-5090-x2-tensor", "rtx-3090-x2"], "ctx_size": 32768, "max_tokens": 128}
```

→ `202 {"job_id": "...", "model_id": "...", "modes": [...]}`. Omit `modes` to run every
applicable one. `max_tokens` 128 or more gives a stable `generation_tps`; 32 measures start-up
noise. A `503` with code `benchmark_busy` means one is already running — benchmarks are
serialised on purpose.

**Poll it:**

```
GET /api/benchmark/jobs/{job_id}
```

→ `{"state": "running"|..., "progress": {"mode", "phase", "fraction", "completed", "total"},
"report": {...}, "error": null}`. Poll every 10–15 s until `state` is no longer `running`.
`DELETE /api/benchmark/jobs/{job_id}` cancels between modes and restores the model's settings.

**Read the report.** `report.results[]` has one row per mode:

| Field | Meaning |
| --- | --- |
| `generation_tps` | Tokens per second while generating — **what the user feels per token**. The headline number. |
| `prompt_tps` | Prompt-processing speed. Matters as much as generation for long RP prompts and agent tool transcripts. |
| `ttft_s` | Time to first token at this prompt length. |
| `load_time_s` | Cold load onto these cards. |
| `vram_used_bytes` | What the placement actually took. |
| `applicable` / `skipped_reason` | Mode was not run (does not fit) — not a failure. |
| `error` | Mode failed. If it names a lease or "serving", a neighbour was busy: rerun that mode later. |

`report.best_generation_mode` and `report.best_prompt_mode` name the winners. They usually
agree; when they do not, pick by the workload (long prompts → prompt winner).

**On tensor split:** do not assume it is faster. On the reference rig it measured *slower* than
layer split for a small model (D38). Compare `rtx-5090-x2` against `rtx-5090-x2-tensor` for the
model in hand and trust the measurement, not the intuition.

History: `GET /api/models/{id}/benchmarks` (newest first). The latest report is also what a
lease reads (step 3).

---

## 2. Parallel benchmark — how many slots

Once the placement is known, measure how many concurrent conversations it sustains:

```
benchmark_parallel(model_id="<id>", mode="dual_5090", ctx_size=32768, max_tokens=128)
```

`mode` is a **hardware-mode** name from `list_models()["placements"]` — `single_5090`,
`dual_5090`, `dual_3090`, `all_gpus` on the reference rig (note: not the placement-benchmark
keys). It loads the model at that placement with every slot that fits, sends 1, 2, 4… requests
at once, and records real throughput per level. Afterwards the catalog's `recommended_parallel`
for that model/cards/context says `recommended_parallel_basis: "measured"` instead of
`"estimated"`, and `load_args` carries the measured count.

---

## 3. Use the result — lock it in

To run the model at the measured-fastest settings with nothing beside it:

```
reserve_gpus(devices=[0, 1], model_id="<id>")     # the winning mode's devices
pin_model(model_id="<id>")                         # optional: survive restarts too
```

The lease loads the model onto exactly those cards, applies the **split mode and micro-batch
its latest benchmark measured fastest on those very devices**, and sizes slots automatically
(even when the server's `default_parallel` is 1). Nothing else will load there until you
`release_gpus(lease_id=...)` or the model sits idle for `idle_ttl_s` (default 60 min), and the
rebalancer (`planner.rebalance`, D42) never moves a leased model off its cards. A grant is
refused, not forced, if a resident on those cards picks up a request before it is unloaded.

If you only want the catalog updated — not a lock — stop after step 2: `list_models` now shows
`confidence: "measured"` for that placement and the numbers are used in every recommendation.

---

## Minimal sequence (copy this)

```
server_status()                                                    # busy empty? leases?
GET  /api/models/{id}/benchmark/modes                              # which keys are applicable
POST /api/models/{id}/benchmark  {"ctx_size": 32768, "max_tokens": 128}   # all applicable modes
GET  /api/benchmark/jobs/{job_id}        ...until state != "running"
     -> report.best_generation_mode, results[].generation_tps / prompt_tps
benchmark_parallel(model_id, mode="dual_5090", ctx_size=32768)     # on the winner's hardware mode
reserve_gpus(devices=[...winner devices...], model_id=...)         # optional: lock it in
```

## What can go wrong

| Symptom | Cause | Do |
| --- | --- | --- |
| `503 benchmark_busy` | another benchmark running | wait; `server_status().busy` |
| `503` with `retry_after_s` | a model is serving / loading / a test is running | wait that long, check again |
| one mode has `error` naming a lease or "serving" | a neighbour was busy on those cards | rerun that mode later |
| `applicable: false` | the model does not fit that placement | not an error; skip |
| numbers wildly below `list_models` estimates | `ctx_size` mismatch, or `max_tokens` too small | match the catalog row's `ctx_per_slot`; use ≥128 tokens |
| user's RP model reprocessed its whole prompt afterwards | you benchmarked it mid-session (rule 2) | don't |
