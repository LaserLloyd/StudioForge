"""Per-model benchmark dialog: which GPU placement is actually fastest.

The question this answers is not "how fast is this model" but "where should it
run" -- on the reference rig a model can go on one 5090, both 5090s, one 3090,
both 3090s or all four, and the answer differs per model *and* differs between
prompt processing and generation. So the dialog runs the modes the user picks
and puts the numbers side by side, marking the two winners separately.

**The subsystem is optional.** It is developed independently of this panel, so
everything here is probed at call time and its absence renders as a plain
explanation rather than an error: the Models tab must work identically on a
build that has no benchmarking at all. All of the derivation lives in
:mod:`studioforge.gui.state` and is tested against the documented JSON shapes,
so only the thin adapter below depends on the subsystem existing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from nicegui import ui

from studioforge.gui import state as st
from studioforge.gui.tabs import GuiContext, api_request, notify_error, panel_guard

#: Poll cadence for a running job. A benchmark reloads the model once per mode,
#: so sub-second polling would only add noise.
POLL_INTERVAL_S = 1.0


class BenchmarkUnavailable(RuntimeError):
    """The benchmark subsystem is not present in this build."""


class _Bridge:
    """In-process access to the benchmark route handlers.

    The handlers are called directly rather than over HTTP because they *are*
    the subsystem's front door: the job table, the "one run at a time" lock and
    the mode validation all live behind them, and reimplementing any of that
    here would be a second, drifting copy. Calling them in-process also keeps
    the GUI free of any self-referential URL.

    Every method is resolved lazily by name, so a build without the subsystem
    (or with a differently named handler) degrades to "not available" instead of
    failing at import time.
    """

    def __init__(self, routes: Any, ctx: GuiContext) -> None:
        self._routes = routes
        self._ctx = ctx

    def _handler(self, name: str) -> Any:
        handler = getattr(self._routes, name, None)
        if not callable(handler):
            raise BenchmarkUnavailable(f"this build has no '{name}' benchmark endpoint")
        return handler

    async def modes(self, model_id: str, *, ctx_size: int) -> Any:
        return await self._handler("model_benchmark_modes")(
            model_id, api_request(self._ctx), ctx_size=ctx_size
        )

    async def start(
        self, model_id: str, *, modes: Sequence[str], ctx_size: int, max_tokens: int
    ) -> dict[str, Any]:
        response = await self._handler("start_benchmark")(
            model_id,
            api_request(self._ctx),
            payload={
                "modes": list(modes),
                "ctx_size": ctx_size,
                "max_tokens": max_tokens,
            },
        )
        return _payload_of(response)

    async def job(self, job_id: str) -> Any:
        return await self._handler("benchmark_job")(job_id, api_request(self._ctx))

    async def cancel(self, job_id: str) -> Any:
        return await self._handler("cancel_benchmark_job")(job_id, api_request(self._ctx))

    async def history(self, model_id: str, *, limit: int) -> Any:
        return await self._handler("list_model_benchmarks")(
            model_id, api_request(self._ctx), limit=limit
        )


def _payload_of(response: Any) -> dict[str, Any]:
    """Body of a handler that returns a ``JSONResponse`` (start returns 202)."""
    body = getattr(response, "body", None)
    if body is None:
        return dict(response) if isinstance(response, dict) else {}
    decoded = json.loads(body)
    return decoded if isinstance(decoded, dict) else {}


def bridge(ctx: GuiContext) -> _Bridge | None:
    """The benchmark bridge, or ``None`` when this build has no benchmarking.

    The import is deliberately inside the function: the subsystem is developed
    separately, and an import error here must cost one dialog rather than the
    whole Models tab.
    """
    try:
        from studioforge.api import mgmt_routes
    except Exception:  # noqa: BLE001 - absence is a supported state
        return None
    if not hasattr(mgmt_routes, "benchmark_job"):
        return None
    return _Bridge(mgmt_routes, ctx)


def is_available(ctx: GuiContext) -> bool:
    return bridge(ctx) is not None


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


def open_dialog(ctx: GuiContext, record: Any) -> None:
    """Open the benchmark dialog for one model."""
    with (
        ui.dialog() as dialog,
        ui.card().classes("min-w-[48rem] max-w-[95vw]"),
        panel_guard("The benchmark panel"),
    ):
        _body(ctx, record, dialog)
    dialog.open()


def _body(ctx: GuiContext, record: Any, dialog: Any) -> None:  # noqa: C901 - one flow
    ui.label(f"Benchmark — {record.id}").classes("font-medium")
    service = bridge(ctx)
    if service is None:
        ui.label(st.BENCHMARK_UNAVAILABLE_NOTE).classes("text-sm opacity-70")
        ui.button("Close", on_click=dialog.close).props("flat")
        return

    ui.label(
        "Runs the same short prompt on each GPU placement you tick, reloading the model "
        "between modes. Prompt processing and generation are reported separately because "
        "the fastest placement for one is often not the fastest for the other."
    ).classes("text-xs opacity-70")

    state: dict[str, Any] = {
        "modes": [],
        "selected": [],
        "job_id": None,
        "job": None,
        "rows": [],
        "report": None,
        "history": [],
    }

    with ui.row().classes("w-full gap-3 flex-wrap items-center"):
        ctx_size = ui.number("ctx size", value=4096, precision=0)
        ctx_size.props("dense outlined").classes("w-32")
        max_tokens = ui.number("tokens to generate", value=128, precision=0)
        max_tokens.props("dense outlined").classes("w-44")

    modes_box = ui.column().classes("w-full gap-0")
    progress_box = ui.column().classes("w-full gap-1")
    results_box = ui.column().classes("w-full gap-1")
    history_box = ui.column().classes("w-full gap-1")

    def paint_modes() -> None:
        modes_box.clear()
        with modes_box:
            rows: list[st.BenchmarkModeRow] = state["modes"]
            if not rows:
                ui.label("No GPU modes reported.").classes("text-xs opacity-60")
                return
            ui.label("Modes").classes("text-sm font-medium mt-1")
            for row in rows:
                with ui.row().classes("items-center gap-2 no-wrap"):
                    box = ui.checkbox(row.label, value=row.key in state["selected"]).props("dense")
                    if not row.applicable:
                        # Greyed with the reason attached: a checkbox that cannot
                        # be ticked and does not say why reads as a bug.
                        box.props("disable")
                    box.tooltip(row.tooltip)
                    box.on_value_change(
                        lambda event, key=row.key: _toggle(state, key, bool(event.value))
                    )
                    ui.label(row.detail).classes("text-xs opacity-60 font-mono")
                    if not row.applicable and row.skipped_reason:
                        ui.label(row.skipped_reason).classes("text-xs text-warning")

    def paint_results() -> None:
        results_box.clear()
        with results_box:
            rows: list[st.BenchmarkResultRow] = state["rows"]
            if not rows:
                return
            _results_table(rows, state["report"])

    def paint_history() -> None:
        history_box.clear()
        with history_box:
            entries: list[Any] = state["history"]
            if not entries:
                return
            with ui.expansion(f"Previous runs ({len(entries)})", icon="history").classes("w-full"):
                for entry in entries:
                    ui.label(st.benchmark_history_label(entry)).classes("text-xs font-medium mt-1")
                    report = (entry or {}).get("report")
                    _results_table(st.benchmark_result_rows(report), report)

    async def load_modes() -> None:
        try:
            payload = await service.modes(record.id, ctx_size=int(ctx_size.value or 4096))
        except BenchmarkUnavailable as exc:
            modes_box.clear()
            with modes_box:
                ui.label(f"{st.BENCHMARK_UNAVAILABLE_NOTE} ({exc})").classes("text-sm opacity-70")
            return
        except Exception as exc:  # noqa: BLE001
            modes_box.clear()
            with modes_box:
                ui.label(f"benchmark modes unavailable: {exc}").classes("text-sm opacity-70")
            return
        state["modes"] = st.benchmark_modes(payload)
        state["selected"] = st.default_selected_modes(state["modes"])
        paint_modes()

    async def load_history() -> None:
        try:
            payload = await service.history(record.id, limit=5)
        except Exception:  # noqa: BLE001 - history is a bonus, never a blocker
            return
        entries = (payload or {}).get("benchmarks") if isinstance(payload, dict) else payload
        state["history"] = list(entries or [])
        paint_history()

    async def start() -> None:
        reason = st.benchmark_start_disabled_reason(state["selected"])
        if reason:
            ui.notify(reason, type="warning")
            return
        try:
            payload = await service.start(
                record.id,
                modes=list(state["selected"]),
                ctx_size=int(ctx_size.value or 4096),
                max_tokens=int(max_tokens.value or 128),
            )
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="start benchmark")
            return
        state["job_id"] = (payload or {}).get("job_id") if isinstance(payload, dict) else payload
        state["rows"] = []
        state["report"] = None
        paint_results()
        paint_progress()

    async def cancel() -> None:
        if not state["job_id"]:
            return
        try:
            await service.cancel(str(state["job_id"]))
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="cancel benchmark")
            return
        ui.notify("benchmark cancelled", type="warning")

    def paint_progress() -> None:
        progress_box.clear()
        if not state["job_id"]:
            return
        with progress_box:
            bar = ui.linear_progress(value=0.0, show_value=False, size="12px").props("rounded")
            text = ui.label("starting…").classes("text-xs font-mono opacity-80")
            cancel_button = ui.button("Cancel", icon="cancel", on_click=cancel).props("flat dense")

        async def tick() -> None:
            if not state["job_id"]:
                return
            try:
                job = await service.job(str(state["job_id"]))
            except Exception as exc:  # noqa: BLE001
                text.set_text(st.poll_failure_note(exc))
                return
            state["job"] = job
            bar.set_value(st.benchmark_progress_fraction(job))
            text.set_text(st.benchmark_progress_text(job))
            if st.benchmark_job_finished(job):
                timer.deactivate()
                # There is nothing left to cancel; a Cancel button that stays
                # after the job ended reads as a control that does nothing.
                cancel_button.set_visibility(False)
                state["job_id"] = None
                error = (job or {}).get("error")
                if error:
                    text.set_text(f"benchmark failed: {error}")
                    return
                state["report"] = (job or {}).get("report")
                state["rows"] = st.benchmark_result_rows(state["report"])
                text.set_text(f"finished · {len(state['rows'])} mode(s)")
                paint_results()
                await load_history()

        timer = ui.timer(POLL_INTERVAL_S, tick)

    with ui.row().classes("w-full justify-end gap-2 mt-2"):
        ui.button("Close", on_click=dialog.close).props("flat")
        ui.button("Run benchmark", icon="speed", on_click=start).props("color=primary")

    ui.timer(0.05, load_modes, once=True)
    ui.timer(0.05, load_history, once=True)


def _toggle(state: dict[str, Any], key: str, checked: bool) -> None:
    selected: list[str] = list(state["selected"])
    if checked and key not in selected:
        selected.append(key)
    if not checked and key in selected:
        selected.remove(key)
    state["selected"] = selected


def _results_table(rows: Sequence[st.BenchmarkResultRow], report: Any = None) -> None:
    """Comparison table, with the two winners marked separately."""
    if not rows:
        return
    fastest_generation, fastest_prompt = st.report_best_modes(
        report if isinstance(report, dict) else None, rows
    )
    columns = [
        {"name": "label", "label": "Mode", "field": "label", "align": "left"},
        {"name": "load", "label": "Load (s)", "field": "load", "align": "right"},
        {"name": "ttft", "label": "TTFT (s)", "field": "ttft", "align": "right"},
        {"name": "prompt", "label": "Prompt tok/s", "field": "prompt", "align": "right"},
        {"name": "generation", "label": "Generation tok/s", "field": "generation"},
        {"name": "note", "label": "", "field": "note", "align": "left"},
    ]
    data = []
    for row in rows:
        marks = []
        if row.mode == fastest_generation:
            marks.append("fastest generation")
        if row.mode == fastest_prompt:
            marks.append("fastest prompt")
        data.append(
            {
                "label": row.label,
                "load": _round(row.load_time_s),
                "ttft": _round(row.ttft_s, 2),
                "prompt": _round(row.prompt_tps),
                "generation": _round(row.generation_tps),
                "note": row.status_text or " · ".join(marks),
            }
        )
    ui.table(columns=columns, rows=data, row_key="label").props("dense flat").classes("w-full")
    speedup = st.benchmark_speedup_text(rows)
    if speedup:
        ui.label(speedup).classes("text-xs text-positive")
    for note in st.benchmark_report_notes(report if isinstance(report, dict) else None):
        ui.label(note).classes("text-xs opacity-70")


def _round(value: float | None, precision: int = 1) -> str:
    if value is None:
        return st.UNKNOWN
    return f"{value:.{precision}f}"


__all__ = ["BenchmarkUnavailable", "bridge", "is_available", "open_dialog"]
