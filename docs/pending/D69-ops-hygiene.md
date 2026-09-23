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
