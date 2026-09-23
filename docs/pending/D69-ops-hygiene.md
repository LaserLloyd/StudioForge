## D69 — Ops hygiene from the 2026-09-22 review: log lines that say what happened, log files that stop growing, a restart that does not wait for nothing

**Status.** Pending, lane `lane/ops-hygiene` (based on `b73de7e`). One commit per item, numbered as
in the 2026-09-22 review's prioritized list. `tests/unit` green at each commit; ruff, `ruff format
--check` and mypy (the gated packages) clean.

**Context.** The 2026-09-22 review read the live logs from 2026-09-13 on. Most of the warning volume
was not a fault. It was the same per-state fact repeated on every load, preview planners logging at
INFO as if they were loads, a miss the D51 safety band already absorbs, and exception text that
came out empty. The real faults it found were NiceGUI tracebacks, a 30 s restart drain with nothing
in flight, and three log files with no rotation. The sections below give the evidence, the change
and the tests for each item.

### §7 — Upstream failures name their exception

**Evidence.** `str()` of several httpx exceptions is empty. `openai_routes._stream_upstream` logged
`stream failed error=` six times on 2026-09-20 20:48:17, the only record of six cut streams.
`_forward` raised `llama-server for '…' failed: . Recent llama-server output: …`, four ERROR lines
on 09-16/17.

**Change.** `openai_routes._exc_text(exc)` returns `ClassName: message`, or the class name alone when
the message is empty. It is the shape `supervisor._watch` has used since D60. Both sites use it: the
`stream failed` warning, the mid-stream SSE error frame and the non-streamed `UpstreamError`.
`_consume_load_exception` was left alone: its exceptions are StudioForge errors with messages.

**Tests.** `tests/unit/test_upstream_error_text.py`: the helper, a stream cut by `ReadError('')`
(warning field, frame text, `[DONE]` still sent, request slot released), and a non-streamed
`RemoteProtocolError('')`.

### §8 — Preview planners log at DEBUG

**Evidence.** `_log_plan` and the terminal refusal honour `Planner._log_plans`, but two lines did not.
`re-planned after eviction` was logged at INFO from every catalog, placements and fit preview.
On 2026-09-20 that was 560 of 1585 log lines (154 of 670 on 09-22), each one naming an eviction of
the embedding or judge model "for" K2 that never happened. The review's first reading took them for
real K2 evictions. `load rejected: device leased to another holder` had the same gap: a preview
planning a model whose saved `device_override` names a leased card logged it at INFO. Almost all of
its 340 live lines came from real refused loads (332 sit beside a request refusal); those still log
at INFO, and only the preview copies go.

**Change.** Both now use `log.info if self._log_plans else log.debug`, the same rule as `load
planned`. A real load (`log_plans=True`) logs exactly as before. The server-chosen candidate case of
the lease line stays at DEBUG, as it was.

**Tests.** `tests/unit/test_preview_planner_logging.py`: both lines at DEBUG with `log_plans=False` and
at INFO with `log_plans=True` (both fail against the old code).

### §14 — `auto` is sized as f16, silently; an unknown KV type warns once

**Evidence.** 54 `unknown kv cache type, assuming f16` warnings since 09-13, all with
`kv_cache_type=auto`, in bursts of 3-5 within 20 ms. `models.default_kv_cache_type: auto` is a
request for the quality ladder, which `Planner._kv_options` fans out. The download-fit preview
(`downloader.py`, the "does this remote file fit" sizing) passes the config default straight into
`kv_alloc_bytes`, and so into `kv_bytes_per_element`.

**Change.** `kv_bytes_per_element("auto")` returns the f16 width without logging. f16 is the
ladder's first and most expensive rung, so the answer can only err toward refusing. A genuinely
unknown type still gets f16 and a WARNING, once per type per process; repeats go to DEBUG. The
once-per-process rule is a small shared helper, `studioforge.logging.first_time(*key)`: a bounded
process-wide set (4,096 keys, then it starts over) with `reset_first_time()` for tests. §16 uses it
too.

**Tests.** `tests/unit/test_kv_type_auto.py`: `auto` equals f16 with no log line, both directly and
through `kv_alloc_bytes`. An unknown type warns once and then logs at DEBUG, and a second unknown type
warns on its own. `first_time` returns True once per key and resets at its cap.

### §16 — Per-state facts are said once

**Evidence.** Between 09-13 and 09-22, `registry.alias_collision` logged 92 times: three lines per
rescan for the same Dark-Scarlett IQ4_XS/Q5_K_M pair, one per shared short alias.
`thinking model loads with no reasoning_format` logged 136 times, once per load of every thinking
model, although it describes a per-model setting.

**Change.** `Registry._rebuild_aliases` collects the collisions per `(kept, dropped)` pair and logs one
WARNING per pair, with `aliases=[...]` (sorted) in place of the old single `alias=` field. It warns
only when that `(kept, dropped, aliases)` state is new to the process. A rescan that finds the same
collision logs it at DEBUG, and a collision that goes away and comes back is warned again. The
thinking-format warning in the manager's load path is a WARNING the first time per model per process
(`first_time("reasoning_format", model_id)`, §14) and DEBUG after. The manager edit is the call site
and the import only.

**Tests.** `tests/unit/test_registry.py::test_an_alias_collision_is_warned_once_per_state` (one line
per pair, DEBUG on rescans, re-warned after the collision comes back) and the existing collision test
reading `aliases`. `tests/unit/test_thinking_format_nag.py` covers three loads of one model, then a
second model: WARNING for each model once, DEBUG for the repeats, nothing for a model with a
`reasoning_format`. Both fail against the old code.

### §6 — The D63 dead zone: an over-estimate D51 cannot correct is not warned about

**Evidence.** `observed_correction` returns `None` when the factor is within 0.005 of 1, and the
factor is `ratio × OBS_SAFETY` (1.10). So for every over-estimate between 8.6 % and 9.5 % (ratio
0.9045 to 0.9136) D51 makes no correction. The plan stays on the formula, `observe` treats it as
uncorrected and applies the 5 % bar, and the load is warned. Every repeat does the same. The registry
has two Hy-MT2 rows 13 s apart, same configuration, both ratio 0.911 and both `corrected=False`.
There were 52 `vram prediction error exceeds the bar` warnings since 09-13: 28 were Hy-MT2 at −8.9 %,
and the rest were Precog-123B −9.2 %, Dark-Scarlett Q5_K_M −9.1 %, Orion-26B −9.4 %, Qwen3-VL-Emb-8B
−9.9 % and Hy-MT2 −9.1/−11.4 %.

**Choice.** The bar is asymmetric, not "skip the warning when D51 would make no correction". An
over-estimate costs headroom, never an OOM. Up to `1 − 1/OBS_SAFETY` (9.1 %) of headroom is what D51
reserves on purpose on every corrected plan, so a formula that far over already sits where a measured
plan would put the load. It is not a defect worth a WARNING, on the first load or on any later one.
Skipping only the uncorrectable band would still warn at −7 % on first loads and then correct the
plan *upward*. An under-estimate is the OOM direction and keeps the 5 % bar, so it stays loud.

**Change.** `PREDICTION_OVER_ESTIMATE_WARN_PCT = 10.0` sits beside `PREDICTION_ERROR_WARN_PCT = 5.0`
in `core/planner.py`, and the literal `0.005` became `OBS_NOOP_TOLERANCE`. `observe()` holds a
negative error (the child holds less than the formula said) to the 10 % bar and a positive one to
5 %. The warning's `bar_pct` and `last_observation()`'s `bar_pct` name the bar that applied, and
`last_observation()` also carries `over_bar_pct` (an additive field; `model_info.vram_prediction` passes
it through). The corrected-plan warning ("measured footprint exceeds the corrected estimate") is
unchanged: it is the under-estimate direction. Nothing about the correction itself changed.

**Tests.** `tests/unit/test_planner_prediction_error.py`:
- the over-estimate bar is pinned at or past the far edge of the uncorrectable band, computed from
  `OBS_SAFETY` and `OBS_NOOP_TOLERANCE` and walked ratio by ratio;
- the four live ratios (0.911, 0.908, 0.906, 0.901) are not warnings;
- a −20 % miss still is, with `bar_pct` 10;
- the old "7 % off either way warns" case is now split: +7 % warns at bar 5, −7 % does not.
