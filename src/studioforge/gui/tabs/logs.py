"""Logs tab: the server's own ring buffer, and each child's stderr.

Two sources, because they answer different questions. The ring buffer is what
StudioForge did; a model's log is what ``llama-server`` said -- and a load
failure explains itself there and nowhere else, which is why per-model logs are
reachable even for a model that is no longer running.

**Local viewers only (D55).** Both sources are prose written for the operator:
absolute paths, the data-dir layout, the model library's on-disk names, and
whatever a child wrote to stderr. That is the operator's own business and not a
remote viewer's, so this tab follows the D32 rule the box-changing controls
follow -- a browser on this machine, or an install with ``server.api_key`` set
(where reaching the panel at all took the key). The refusal is rendered in
place, and the refresh path checks again: a disabled widget is one websocket
frame from enabled, and this tab's whole content is the thing being withheld.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from studioforge.gui import state as st
from studioforge.gui.tabs import (
    REMOTE_VIEWER_NOTE,
    GuiContext,
    element_alive,
    require_local_admin,
    run_blocking,
    viewer_may_change_box,
)

SERVER_SOURCE = "StudioForge server"
LEVELS = ("ALL", "DEBUG", "INFO", "WARNING", "ERROR")
LINE_CHOICES = (100, 200, 500, 1000, 2000)


def render(ctx: GuiContext) -> None:
    if not viewer_may_change_box(ctx):
        with ui.column().classes("w-full gap-2 p-2"):
            ui.label("Logs are not shown to a remote viewer").classes("text-base")
            ui.label(
                "The server log and each model's llama-server output carry absolute "
                "paths and this box's layout, so " + REMOTE_VIEWER_NOTE
            ).classes("text-sm opacity-70 max-w-3xl")
        return

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
            "w-full font-mono text-xs whitespace-pre-wrap sf-well "
            "p-2 rounded overflow-auto max-h-[60vh]"
        )

    async def refresh() -> None:
        # Driven by the Follow timer, so a panel_guard card would stack one per
        # tick forever. Report staleness in place and keep the last good body.
        #
        # D50: the staleness label belongs to the same panel as the body, so the
        # liveness check goes here rather than only inside -- writing "log view
        # is stale" into a page that no longer exists helps nobody.
        if not element_alive(body):
            return
        try:
            await _refresh_once()
        except Exception as exc:  # noqa: BLE001 - a poll must never kill the tab
            stale.set_text(f"log view is stale: {exc}")
            return
        stale.set_text("")

    async def _refresh_once() -> None:
        # Checked again here, not only in ``render``: the controls above live in
        # a page a websocket can re-enable, and this is the function that
        # actually reads the files.
        require_local_admin(ctx, "reading the server and model logs")
        # D50: a timer body that awaits file I/O, so the page can be rebuilt
        # underneath it mid-tick. Checked on entry (cheap, and this is also
        # reached from the source/level pickers) and again after the awaits,
        # because that is the window that actually opens.
        if not element_alive(body):
            return
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
        if not element_alive(body):
            return
        path_label.set_text(str(path) if path else "(no log file yet)")
        body.set_text("\n".join(text_lines) or "(empty)")

    async def tick() -> None:
        if follow.value:
            await refresh()

    for widget in (source, level, count):
        widget.on_value_change(lambda _: refresh())

    ui.timer(0.1, refresh, once=True)
    ui.timer(max(2.0, ctx.refresh_interval), tick)
