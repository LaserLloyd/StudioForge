# bench/ — Phase 0 benchmark harness

A stand-alone, repeatable measurement of one fixed workload against the running StudioForge
gateway, so that every change to the serving path is judged against a number rather than a
feeling. It talks **only** to the gateway's REST API on `http://127.0.0.1:1234` — never to a
`llama-server` child, and it launches nothing. Python 3.12, stdlib + `httpx` (already in the
repo venv). Not part of the package; run it from the repo root with the repo's interpreter.

## The workload

| | |
| --- | --- |
| Model | `ReadyArt/Dark-Scarlett-27B-v2.0-GGUF/Dark-Scarlett-27B-v2.0.i1-Q5_K_M_hb16` (27B, Q5_K_M) |
| Context | 262144 per slot, `parallel` 1 |
| Devices | CUDA 0,1 — the RTX 5090 pair |
| Load tier | `priority: 2` (D46 *agent*): may displace idle background residents, never a chat-tier model; `force` is never sent |
| Prompt lengths | 1024, 32768, 131072, 262144 tokens, each clamped to `ctx − max_tokens − 64` |
| Runs | 6 per length; run 0 is the warm-up and is discarded |
| Reply | 256 tokens, `temperature 0`, `top_k 1`, fixed `seed`, `ignore_eos` (so every run decodes the same count) |
| Cache | `cache_prompt: false` **and** a unique first line per run, so no run can reuse a slot's prefix |

Batch/ubatch, flash-attention, KV cache dtypes, split mode, tensor split and the engine build
are **recorded as launched**, not forced: they come from `GET /api/status loaded[].plan` (the
accepted placement) and `loaded[].effective` (D54: parsed from the child's real argv), plus
`GET /api/models/{id}/introspect` (`actual.build_info`, `n_ctx`, `total_slots`).

## Metrics and where each one comes from

Throughput is llama-server's own `timings` object, which the gateway forwards untouched — in the
final SSE chunk of a streamed reply (D54; the same source `core/benchmark.py` uses). The
stopwatch is only used for TTFT and wall time.

| Metric | Source | Notes |
| --- | --- | --- |
| prefill tok/s | `timings.prompt_per_second` | prompt processing, measured inside the engine |
| decode tok/s | `timings.predicted_per_second` | **the headline number** at 256k |
| prompt_n / predicted_n | `timings.prompt_n`, `timings.predicted_n` | tokens actually processed / generated |
| TTFT s | wall clock, request start → first chunk with content | includes gateway overhead |
| wall s | wall clock, request start → `[DONE]` | |
| peak VRAM per device | max of `GET /api/gpus gpus[].used_bytes`, sampled every 0.5 s in a thread during the run | whole-card usage, not per process |
| load wall s | `POST /api/models/{id}/load` call → return (the route returns when the child is ready) | |
| `cache_hit_suspected` | `prompt_n < 98%` of the fitted prompt size, or `timings.cache_n > 0` | the run must have evaluated the whole prompt |
| `timings_missing` | no `timings` chunk arrived; rates fall back to the stopwatch (marked) | should never happen on this engine |

Statistics per prompt length, over runs 1..N: **median** and **p95 (nearest-rank:** the value at
rank ⌈0.95·n⌉ of the sorted sample**)**.

## Running

All commands from the repo root. `SF_API_KEY` (or `--api-key`) is sent as a bearer token when set;
a loopback caller needs nothing else (D32/D55: load, unload and inference are open on the box,
and `/api/vram/holders` gives a loopback caller the unredacted view).

```
# Dry run: print the plan and every request it would make. Contacts nothing.
.venv\Scripts\python.exe bench\run_bench.py --dry-run

# Smoke: uses whatever instance of the model is ALREADY resident (no load, no restore),
# 1024-token prompt, 2 runs, 64 tokens. Writes results/<sha>-smoke.json tagged "smoke": true.
.venv\Scripts\python.exe bench\run_bench.py --smoke

# Baseline: the full protocol below. Budget ~30-45 min (four lengths x six runs at 256k).
.venv\Scripts\python.exe bench\run_bench.py --label before-kv-change

# Compare two results files (markdown table + verdict). Exit 0 always.
.venv\Scripts\python.exe bench\compare.py bench\results\<new>.json [bench\results\<old>.json]
```

Options: `--server`, `--model`, `--ctx`, `--devices 0,1`, `--parallel`, `--prompt-lengths`,
`--runs`, `--max-tokens`, `--seed`, `--results-dir`, `--label`, `--no-restore`, `--api-key`.
Exit codes: **0** done and valid · **1** harness/HTTP error · **2** the server refused (its
message and `error.studioforge` details are printed verbatim) · **3** completed but invalid.

### The baseline protocol

1. **Preflight** — `GET /api/health` must be `ok` and not draining, `busy.active_requests` 0, no
   load in progress, no smoke test, no priority hold; `GET /api/status` must show no running
   benchmark, an empty load queue and **no GPU lease on the target devices** other than one that
   **names the model under test** (a loopback caller would be waived past the D55 lease guard on
   unload, so the harness refuses any other lease; a lease taken for the bench model is recorded
   as `own_lease` and is the quietest rig a baseline can have). The
   resident instances are snapshotted (model, `plan`, `priority`) for the restore.
2. **Validity, pre-load** — `GET /api/vram/holders` (D23) and `GET /api/gpus` are recorded.
3. **Load** — if the model is resident at exactly the requested shape it is reused; if resident
   at another shape it is unloaded first (never `force`); then
   `POST /api/models/{id}/load {"ctx_size", "parallel", "devices", "priority": 2}`. A refusal
   (507 `insufficient_vram` / `gpu_leased`, 409 `lease_conflict`, 503 `priority_hold` /
   `model_busy`) is printed verbatim and the run exits 2 after restoring.
4. **Record the launch** — `effective`, `plan`, `launch_args`, `resolved_engine_tag`, introspect.
5. **Validity, post-load**, then **prompt fitting**: deterministic prose from a fixed word list,
   calibrated with `POST /v1/tokenize` to land within [clamp − 3%, clamp] (never above). Without
   that route, a one-token completion's `timings.prompt_n` calibrates instead.
6. **Runs** — per length, `--runs` streamed `POST /v1/completions` with the VRAM sampler running.
7. **Validity, post-run.**
8. **Restore** (unless `--no-restore`) — unload the bench instance, then reload every model that
   was resident at preflight with its previous `plan` tuple (ctx, parallel, devices, both KV
   types) and its previous tier. Runs even after a refusal or Ctrl-C.
9. **Write** `results/<sha12>[-<label>][-smoke].json` (`dirty: true` when the tree has
   uncommitted changes; a second run at the same sha gets `-2`, `-3`, …), print the summary,
   and — for a baseline — the comparison against the newest previous non-smoke file.

## The invalidity rule

A results file is **valid only if every compute holder on the target devices during the runs was
the StudioForge child serving the model under test.** The decision is taken on the two
`/api/vram/holders` snapshots that bracket the measurement (post-load and post-run); the
pre-load snapshot is recorded for context (an idle background model the tier-2 load displaced
never shared the cards with a run). "On a device" means ≥ 512 MiB there — `per_gpu_bytes`
(PDH, D39) when the server has it, else `gpu_indices` together with a ≥ 512 MiB total; a holder
with no attribution at all cannot be proven off the cards and counts as on them. Desktop/WDDM
processes are already collapsed by the server (< 256 MiB) and are recorded as a count, never
decisive. An invalid file still carries every number; `compare.py` labels its verdict advisory.

Since 2026-09-10 the sampler thread also reads `/api/vram/holders` every 10 s **during** each run;
any compute holder on the target devices that is not the model under test marks that run
(`foreign_holders_seen`) and the file invalid (`validity.runs_with_foreign_holders`), because the
bracketing snapshots cannot see a co-tenant that arrives after the post-load check and leaves
before the post-run one — which is what the first baseline of that day suffered (a JIT load split
across CUDA 0, 1 and 3 beside the bench instance for three of its four lengths).

## The acceptance rule (as amended by the owner)

**Improvements found along the way are kept regardless of size.** The ≥ 10% bar — median decode
tok/s at 256k, *or* median prefill tok/s at ≥ 128k (the worst delta over every bucket ≥ 128k),
with no > 2% regression in the other — gates **only major restructuring** such as adding a new
backend. `compare.py` prints one of: `KEEP` (any improvement, below the bar), `MAJOR-RESTRUCTURING
BAR MET`, `NEUTRAL` (within ± 2%, no improvement), `REGRESSION` (> 2% loss in a headline), or
`none` when a file lacks the 256k / ≥ 128k buckets (a smoke run). Rejected changes go in
`rejected.md`.

## Gotchas

- Run it when the box is quiet. The D46 restore can bring a *recently active* (< 300 s)
  background model back beside the bench instance after the load; the post-run check would
  then mark the file invalid. Give the rig five idle minutes first.
- `server.request_timeout_s` (default 900 s) bounds one streamed request at the gateway; a very
  slow 256k prefill would surface as a 504 — raise the setting rather than the harness timeout.
- `/v1/tokenize` JIT-loads an absent model, which is why calibration runs only after the load.
- The prompt is word-salad prose; it measures the engine, not the model's prose quality.
