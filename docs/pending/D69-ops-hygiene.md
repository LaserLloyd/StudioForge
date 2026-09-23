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
real K2 evictions. `load rejected: device leased to another holder` (340 lines) came from the same
previews refusing a saved `device_override` on a leased card at every catalog build.

**Change.** Both now use `log.info if self._log_plans else log.debug`, the same rule as `load
planned`. A real load (`log_plans=True`) logs exactly as before. The server-chosen candidate case of
the lease line stays at DEBUG, as it was.

**Tests.** `tests/unit/test_preview_planner_logging.py`: both lines at DEBUG with `log_plans=False` and
at INFO with `log_plans=True` (both fail against the old code).
