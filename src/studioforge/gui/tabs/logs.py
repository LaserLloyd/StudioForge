"""Logs tab: the server's own ring buffer, and each child's stderr.

Two sources, because they answer different questions. The ring buffer is what
StudioForge did; a model's log is what ``llama-server`` said -- and a load
failure explains itself there and nowhere else, which is why per-model logs are
reachable even for a model that is no longer running.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from studioforge.gui import state as st
from studioforge.gui.tabs import GuiContext, run_blocking

SERVER_SOURCE = "StudioForge server"
LEVELS = ("ALL", "DEBUG", "INFO", "WARNING", "ERROR")
LINE_CHOICES = (100, 200, 500, 1000, 2000)


def render(ctx: GuiContext) -> None:
    model_ids: list[str] = []
    try:
        if ctx.registry is not None:
            model_ids = [r.id for r in ctx.registry.all() if not r.is_virtual]
    except Exception:  # noqa: BLE001
        model_ids = []

    with ui.column().classes("w-full gap-2 p-2"):
        with ui.row().classes("w-full items-center gap-2 flex-wrap"):
            source = ui.select(
                [SERVER_SOURCE, *model_ids], value=SERVER_SOURCE, label="Source", with_input=True
            )
            source.props("dense outlined").classes("w-96")
            level = ui.select(list(LEVELS), value="INFO", label="Level")
            level.props("dense outlined").classes("w-32")
            count = ui.select(list(LINE_CHOICES), value=200, label="Lines")
            count.props("dense outlined").classes("w-28")
            follow = ui.checkbox("Follow", value=True)
            ui.button("Refresh", icon="refresh", on_click=lambda: refresh()).props("outline dense")
        path_label = ui.label("").classes("text-xs font-mono opacity-60")
        stale = ui.label("").classes("text-xs text-warning")
        body = ui.label("").classes(
            "w-full font-mono text-xs whitespace-pre-wrap bg-black/10 dark:bg-white/5 "
            "p-2 rounded overflow-auto max-h-[60vh]"
        )

    async def refresh() -> None:
        # Driven by the Follow timer, so a panel_guard card would stack one per
        # tick forever. Report staleness in place and keep the last good body.
        try:
            await _refresh_once()
        except Exception as exc:  # noqa: BLE001 - a poll must never kill the tab
            stale.set_text(f"log view is stale: {exc}")
            return
        stale.set_text("")

    async def _refresh_once() -> None:
        if True:
            lines = int(count.value or 200)
            if source.value == SERVER_SOURCE:
                from studioforge.logging import RING_BUFFER

                wanted = None if level.value == "ALL" else str(level.value)
                entries = RING_BUFFER.tail(lines, wanted)
                path_label.set_text(f"in-memory ring buffer · {len(entries)} line(s)")
                body.set_text("\n".join(st.log_line_text(entry) for entry in entries) or "(empty)")
                return
            model_id = str(source.value)
            supervisor = ctx.supervisor
            if supervisor is None:
                body.set_text("(supervisor unavailable)")
                return
            # File I/O off the event loop: a 100k-line log would otherwise stall
            # every other viewer of the panel.
            text_lines: list[str] = await run_blocking(supervisor.tail_log, model_id, lines)
            path: Any = await run_blocking(supervisor.log_path, model_id)
            path_label.set_text(str(path) if path else "(no log file yet)")
            body.set_text("\n".join(text_lines) or "(empty)")

    async def tick() -> None:
        if follow.value:
            await refresh()

    for widget in (source, level, count):
        widget.on_value_change(lambda _: refresh())

    ui.timer(0.1, refresh, once=True)
    ui.timer(max(2.0, ctx.refresh_interval), tick)
