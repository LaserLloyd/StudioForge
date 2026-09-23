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
