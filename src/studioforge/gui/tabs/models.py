"""Models tab: the library table, and the three-tier settings dialog.

The table is a real table -- fixed columns with clickable headers -- rather than
a list of cards, because with thirty models the questions are comparative ("which
did I download last", "which of these is the big one", "which can see images")
and those are answered by sorting a column, not by reading thirty paragraphs. It
opens sorted by download date, newest first: the model someone is looking for is
overwhelmingly the one that just arrived. Capabilities are coloured icons with
tooltips, so the feature column stays one line wide per model.

The dialog is the heart of this tab. It shows a **live fit verdict** from
``manager.plan_preview()`` that updates while the user is still choosing a
context size, so a load that cannot work is visible *before* the button is
pressed rather than as a rejection afterwards. Expert flags are validated
against the engine's own ``--help`` at save time and the engine's error strings
are shown verbatim -- they already name the replacement for a removed flag, and
paraphrasing them would throw that away.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from nicegui import ui

from studioforge.gui import state as st
from studioforge.gui.tabs import GuiContext, busy, notify_error, panel_guard, run_blocking

_KV_TYPES = ("", "f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0")
_FLASH_ATTN = ("", "on", "off", "auto")
_SPLIT_MODES = ("", "none", "layer", "row", "tensor")
_REASONING_FORMATS = ("", "none", "deepseek", "deepseek-legacy")
_REASONING = ("", "on", "off", "auto")
_TRI = {"auto": "Auto", "on": "On", "off": "Off"}


def _tri_to_value(text: str | None) -> bool | None:
    return {"on": True, "off": False}.get(str(text or "auto"))


def _tri_from_value(value: bool | None) -> str:
    if value is None:
        return "auto"
    return "on" if value else "off"


# ---------------------------------------------------------------------------
# Tab
# ---------------------------------------------------------------------------


#: Column widths, shared by the header and every row so the two line up. The
#: table scrolls horizontally rather than wrapping, because a header that has
#: wrapped onto a second line no longer heads anything.
#: Sort key -> the CSS custom property that carries that column's width. The
#: widths live on ``:root`` rather than on the table element so that a drag
#: survives ``table.refresh()``, which throws away every node the table owns.
_COLUMN_VARS: Final[dict[str, str]] = {
    "name": "name",
    "date": "date",
    "size": "size",
    "quant": "quant",
    "architecture": "arch",
    "type": "features",
    "recent": "used",
    "loaded": "status",
}

#: Starting widths, in grid order. "actions" has no sort key of its own but is
#: resizable like the rest, so it lives here too.
_DEFAULT_WIDTHS: Final[tuple[tuple[str, str], ...]] = (
    ("name", "22rem"),
    ("date", "7rem"),
    ("size", "6rem"),
    ("quant", "7rem"),
    ("arch", "7rem"),
    ("features", "8rem"),
    ("used", "7rem"),
    ("status", "7rem"),
    ("actions", "17rem"),
)

_GRID_TEMPLATE = " ".join(f"var(--sfm-{key})" for key, _ in _DEFAULT_WIDTHS)
_ROOT_VARS = "".join(f"--sfm-{key}:{width};" for key, width in _DEFAULT_WIDTHS)

#: The table is a CSS grid rather than a row of fixed-width flex children.
#: Flex items default to ``min-width:auto``, which is why long model ids used to
#: shove every column to their right off the screen instead of ellipsing: the
#: ``truncate`` class cannot shrink an item that refuses to go below its content
#: width. Grid tracks are sized by the template, and every cell clips.
_TABLE_CSS = f"""
:root{{{_ROOT_VARS}}}
.sfm-row{{
  display:grid;
  grid-template-columns:{_GRID_TEMPLATE};
  column-gap:.5rem;align-items:center;width:max-content;min-width:100%;
}}
.sfm-row>*{{min-width:0;overflow:hidden;}}
.sfm-col{{display:flex;flex-direction:column;min-width:0;overflow:hidden;}}
/* Anything that should ellipse rather than push its neighbour aside. */
.sfm-txt{{
  display:block;min-width:0;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap;
}}
/* Header cells host the drag handle, so they must not clip it. */
.sfm-head>*{{overflow:visible;position:relative;}}
.sfm-grip{{
  position:absolute;top:0;right:-.3rem;width:.6rem;height:100%;
  cursor:col-resize;z-index:15;border-radius:2px;
}}
.sfm-grip:hover,.sfm-grip.sfm-dragging{{background:currentColor;opacity:.4;}}
body.sfm-resizing{{cursor:col-resize!important;user-select:none!important;}}
"""

#: Installed once per page. Delegated listeners on ``document`` keep working
#: across refreshes, and the widths are remembered per browser.
_TABLE_JS = """
(() => {
  if (window.__sfmResizeReady) return;
  window.__sfmResizeReady = true;
  const KEY = 'sf.models.colwidths';
  const MIN = 40;
  const root = document.documentElement;
  let widths = {};
  try { widths = JSON.parse(localStorage.getItem(KEY) || '{}') || {}; } catch (e) { widths = {}; }
  const applyAll = () => {
    for (const [k, v] of Object.entries(widths)) {
      if (typeof v === 'number' && v >= MIN) root.style.setProperty('--sfm-' + k, v + 'px');
    }
  };
  const save = () => { try { localStorage.setItem(KEY, JSON.stringify(widths)); } catch (e) {} };
  applyAll();

  let drag = null;
  document.addEventListener('pointerdown', (ev) => {
    const grip = ev.target && ev.target.closest && ev.target.closest('.sfm-grip');
    if (!grip) return;
    const cell = grip.parentElement;
    if (!cell) return;
    drag = {
      col: grip.dataset.col,
      startX: ev.clientX,
      startW: cell.getBoundingClientRect().width,
      grip: grip,
    };
    grip.classList.add('sfm-dragging');
    document.body.classList.add('sfm-resizing');
    try { grip.setPointerCapture(ev.pointerId); } catch (e) {}
    ev.preventDefault();
    ev.stopPropagation();
  }, true);

  document.addEventListener('pointermove', (ev) => {
    if (!drag) return;
    const w = Math.max(MIN, Math.round(drag.startW + (ev.clientX - drag.startX)));
    root.style.setProperty('--sfm-' + drag.col, w + 'px');
    widths[drag.col] = w;
    ev.preventDefault();
  });

  const stop = () => {
    if (!drag) return;
    drag.grip.classList.remove('sfm-dragging');
    document.body.classList.remove('sfm-resizing');
    drag = null;
    save();
  };
  document.addEventListener('pointerup', stop);
  document.addEventListener('pointercancel', stop);

  // Double-click a handle to put that one column back to its default.
  document.addEventListener('dblclick', (ev) => {
    const grip = ev.target && ev.target.closest && ev.target.closest('.sfm-grip');
    if (!grip) return;
    root.style.removeProperty('--sfm-' + grip.dataset.col);
    delete widths[grip.dataset.col];
    save();
    ev.preventDefault();
    ev.stopPropagation();
  }, true);
})();
"""

_GRIP_TIP = "drag to resize this column · double-click to reset it"


def _install_table_chrome() -> None:
    """Add the grid stylesheet and the column-resize handlers to this page."""
    ui.add_css(_TABLE_CSS)
    ui.add_body_html(f"<script>{_TABLE_JS}</script>")


def _grip(column_key: str) -> None:
    """The draggable right-hand edge of a header cell."""
    ui.element("div").classes("sfm-grip").props(f"data-col={column_key}").tooltip(_GRIP_TIP)


def _stored_sort() -> tuple[str, bool]:
    """Last-used sort column and direction for this browser.

    ``app.storage.user`` needs a page context and a storage secret; both exist
    in production, but the fallback keeps the table rendering anywhere they do
    not (tests, odd embedding contexts). A browser that has never sorted gets
    newest-downloaded first, which is what people are almost always looking for.
    """
    try:
        from nicegui import app as nicegui_app

        stored = nicegui_app.storage.user
        key = st.stored_sort_key(stored.get("models_sort"))
        return (key, st.stored_sort_descending(stored.get("models_sort_desc"), key))
    except Exception:  # noqa: BLE001 - a lost preference must not lose the table
        key = st.DEFAULT_SORT_KEY
        return (key, st.stored_sort_descending(None, key))


def _store_sort(key: str, descending: bool) -> None:
    try:
        from nicegui import app as nicegui_app

        nicegui_app.storage.user["models_sort"] = key
        nicegui_app.storage.user["models_sort_desc"] = descending
    except Exception:  # noqa: BLE001
        pass


def render(ctx: GuiContext, params: Mapping[str, Any] | None = None) -> None:
    # A ``?tab=models&model=<id>`` deep link lands here filtered to that model,
    # which is the closest thing to "scroll to it" that survives a rescan.
    focus = str((params or {}).get("model") or "")
    sort_key, descending = _stored_sort()
    view: dict[str, Any] = {"needle": focus, "sort": sort_key, "desc": descending}
    _install_table_chrome()
    with ui.column().classes("w-full gap-3 p-2"):
        table = _model_table(ctx, view)
        with ui.row().classes("items-center gap-2 w-full"):
            ui.label("Model library").classes("text-lg font-medium")
            ui.label(
                "click a column header to sort · click again to reverse · "
                "drag a header edge to resize"
            ).classes("text-xs opacity-60")
            ui.space()
            search = ui.input(placeholder="filter id, quant, arch, capability…", value=focus)
            search.props("dense outlined clearable")
            search.on_value_change(lambda _: (view.update(needle=search.value), table.refresh()))
            ui.button("Rescan", icon="refresh", on_click=lambda: _scan(ctx, table)).props(
                "outline dense"
            )
        if focus:
            with ui.row().classes("items-center gap-2"):
                ui.label(f"Showing only models matching '{focus}'.").classes("text-xs text-warning")

                def clear_focus() -> None:
                    search.set_value("")
                    view["needle"] = ""
                    table.refresh()

                ui.button("show all", on_click=clear_focus).props("flat dense")
        table()
        _adapters_panel(ctx, table)


async def _scan(ctx: GuiContext, table: Any) -> None:
    with busy(message="Scanning the model library…"):
        try:
            result = await run_blocking(ctx.registry.scan)
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="scan")
            return
    ui.notify(
        f"scan: {len(result.added)} added, {len(result.removed)} removed, "
        f"{result.unchanged} unchanged, {len(result.errors)} errors",
        type="positive",
    )
    table.refresh()


def _model_table(ctx: GuiContext, view: dict[str, Any]) -> Any:
    """The refreshable library table.

    ``view`` (filter + sort) lives outside the refreshable so a refresh after
    an action (unload, pin, save) keeps the user's filter instead of resetting
    the table under them.
    """

    @ui.refreshable
    def table() -> None:
        with panel_guard("Model library"):
            records = list(ctx.registry.all()) if ctx.registry is not None else []
            loaded = (
                {i.model_id: i for i in ctx.supervisor.list()} if ctx.supervisor is not None else {}
            )
            total = len(records)
            records = st.filter_models(records, view.get("needle"))
            records = st.sort_models(
                records,
                str(view.get("sort") or ""),
                bool(view.get("desc")),
                loaded_ids=set(loaded),
            )
            if not records:
                if total:
                    ui.label(
                        f"No match for '{view.get('needle')}' among {total} model(s)."
                    ).classes("text-sm opacity-70")
                else:
                    ui.label(
                        "No models found. Check models.dir on the Server tab, then Rescan."
                    ).classes("text-sm opacity-70")
                return
            shown = f"{len(records)} of {total}" if len(records) != total else f"{total}"
            ui.label(f"{shown} model(s)").classes("text-xs opacity-60")
            with (
                ui.column().classes("w-full gap-0 overflow-x-auto"),
                ui.column().classes("gap-0 w-max min-w-full"),
            ):
                _header_row(view, table)
                for record in records:
                    _model_row(ctx, record, loaded.get(_serving_id(ctx, record)), table)

    return table


def _header_row(view: dict[str, Any], table: Any) -> None:
    """Clickable column headers, with an arrow on the active one."""
    active = str(view.get("sort") or "")
    descending = bool(view.get("desc"))

    def click(key: str) -> None:
        new_key, new_descending = st.next_sort(active, descending, key)
        view["sort"], view["desc"] = new_key, new_descending
        _store_sort(new_key, new_descending)
        table.refresh()

    with ui.element("div").classes(
        "sfm-row sfm-head border-b border-white/20 pb-1 mb-1 sticky top-0 bg-inherit z-10"
    ):
        for column in st.MODEL_COLUMNS:
            arrow = st.sort_indicator(column.key, active, descending)
            is_active = column.key == active
            with ui.element("div"):
                button = ui.button(
                    column.label, on_click=lambda _, key=column.key: click(key)
                ).classes("w-full justify-start px-1")
                props = "flat dense no-caps size=sm align=left"
                props += " color=primary" if is_active else " color=grey"
                if arrow:
                    props += f" icon-right={arrow}"
                button.props(props)
                if is_active:
                    direction = st.sort_direction_text(column.key, descending)
                    button.tooltip(
                        f"{column.tooltip} Sorting by this column, {direction}; "
                        "click again to reverse."
                    )
                else:
                    first = st.sort_direction_text(column.key, column.descending_first)
                    button.tooltip(f"{column.tooltip} Click to sort, {first}.")
                _grip(_COLUMN_VARS[column.key])
        with ui.element("div"):
            ui.label("Actions").classes("sfm-txt text-xs opacity-60 pl-2")
            _grip("actions")


def _serving_id(ctx: GuiContext, record: Any) -> str:
    """Which instance answers for this record (a persona rides its base, D13)."""
    try:
        if ctx.manager is not None:
            return str(ctx.manager.serving_record(record).id)
    except Exception:  # noqa: BLE001 - status display only; fall back to itself
        pass
    return str(record.id)


def _copy_text(text: str, *, what: str = "copied") -> None:
    ui.clipboard.write(text)
    ui.notify(what, type="positive")


def _model_row(ctx: GuiContext, record: Any, instance: Any, table: Any) -> None:
    status = st.model_status_label(instance)
    shared = st.shares_base_instance(record)
    added_at = st.model_added_at(record)
    with ui.element("div").classes(
        "sfm-row border-b border-white/10 py-1 hover:bg-white/5 rounded"
    ):
        with ui.element("div").classes("sfm-col"):
            with ui.element("div").classes("flex items-center gap-1 w-full min-w-0"):
                # The id gets whatever room is left after the copy button, and
                # ellipses inside it -- the full id is in the tooltip and one
                # click away on the clipboard.
                ui.label(record.id).classes("sfm-txt font-medium text-sm flex-1").tooltip(record.id)
                ui.button(
                    icon="content_copy",
                    on_click=lambda r=record: _copy_text(r.id, what=f"copied '{r.id}'"),
                ).props("flat dense size=sm").classes("shrink-0").tooltip(
                    "Copy model id (what clients ask for)"
                )
            base_line = st.virtual_base_line(record)
            if base_line:
                ui.label(base_line).classes("sfm-txt text-xs opacity-70").tooltip(base_line)
            for line in st.preset_summary_lines(record.preset):
                ui.label(line).classes("sfm-txt text-xs opacity-60").tooltip(line)
            virtual_badges = [
                text for text in st.capability_badges(record) if text in _VIRTUAL_BADGES
            ]
            if virtual_badges or record.settings.pinned:
                with ui.row().classes("gap-1 flex-wrap"):
                    for text in virtual_badges:
                        ui.badge(text, color="secondary").classes("text-xs")
                    if record.settings.pinned:
                        ui.badge("pinned", color="accent").classes("text-xs")
        ui.label(st.format_when(added_at)).classes("sfm-txt text-xs opacity-80 font-mono").tooltip(
            f"file timestamp: {st.format_datetime(added_at)}"
        )
        ui.label(st.format_gib(record.size_bytes)).classes("sfm-txt text-xs font-mono")
        ui.label(record.quant).classes("sfm-txt text-xs font-mono").tooltip(record.quant)
        ui.label(record.architecture).classes("sfm-txt text-xs font-mono").tooltip(
            record.architecture
        )
        with ui.element("div").classes("sfm-col"):
            ui.label(record.kind).classes("sfm-txt text-xs opacity-60")
            with ui.element("div").classes("flex gap-1 items-center flex-nowrap"):
                for item in st.capability_icons(record):
                    ui.icon(item.icon, color=item.colour, size="1.15rem").tooltip(item.tooltip)
        ui.label(st.format_when(record.last_used_at)).classes(
            "sfm-txt text-xs opacity-70 font-mono"
        ).tooltip(
            f"last request: {st.format_datetime(record.last_used_at)}"
            if record.last_used_at
            else "never used through this server"
        )
        with ui.element("div").classes("flex items-center min-w-0"):
            ui.badge(status, color=st.status_colour(status)).classes("text-xs")
        with ui.element("div").classes("flex items-center flex-nowrap min-w-0"):
            if record.is_virtual:
                ui.button(
                    icon="edit", on_click=lambda r=record: _persona_dialog(ctx, table, record=r)
                ).props("flat dense").tooltip("Edit this virtual model (persona / adapters)")
            ui.button(icon="tune", on_click=lambda r=record: _settings_dialog(ctx, r, table)).props(
                "flat dense"
            ).tooltip("Load / edit settings")
            if instance is not None and instance.state != "stopped":
                if shared:
                    # The instance belongs to the base; unloading it from a
                    # persona row would silently kill every sibling persona.
                    ui.button(icon="stop_circle").props("flat dense disable").tooltip(
                        f"Serving via {record.base_model_id}'s instance — unload it from "
                        "the base model's row"
                    )
                else:
                    ui.button(
                        icon="stop_circle", on_click=lambda r=record: _unload(ctx, r, table)
                    ).props("flat dense color=negative").tooltip("Unload")
            else:
                ui.button(
                    icon="play_arrow", on_click=lambda r=record: _settings_dialog(ctx, r, table)
                ).props("flat dense color=positive").tooltip("Load")
            ui.button(icon="push_pin", on_click=lambda r=record: _toggle_pin(ctx, r, table)).props(
                "flat dense"
            ).tooltip("Pin / unpin")
            ui.button(icon="science", on_click=lambda r=record: _test(ctx, r)).props(
                "flat dense"
            ).tooltip("Test: one short completion, with latency and tok/s")
            ui.button(icon="speed", on_click=lambda r=record: _benchmark(ctx, r)).props(
                "flat dense"
            ).tooltip(
                "Benchmark: compare load time, prompt and generation speed across GPU placements"
            )
            ui.button(icon="delete", on_click=lambda r=record: _delete_dialog(ctx, r, table)).props(
                "flat dense color=negative"
            ).tooltip("Delete")


#: The badges that describe what a *virtual* model is, as opposed to what a
#: model can do. They stay as words: "persona" and "LoRA" carry a VRAM story
#: (D13) that no icon conveys, and there are only ever a couple of them.
_VIRTUAL_BADGES = frozenset({"persona", "LoRA", "virtual"})


def _benchmark(ctx: GuiContext, record: Any) -> None:
    """Open the benchmark dialog, importing the subsystem lazily.

    Lazy because benchmarking is an optional subsystem developed separately: an
    import error here must cost one dialog, not the whole Models tab.
    """
    try:
        from studioforge.gui.tabs import benchmark
    except Exception as exc:  # noqa: BLE001
        ui.notify(f"{st.BENCHMARK_UNAVAILABLE_NOTE} ({exc})", type="warning", multi_line=True)
        return
    benchmark.open_dialog(ctx, record)


# ---------------------------------------------------------------------------
# Row actions
# ---------------------------------------------------------------------------


async def _unload(ctx: GuiContext, record: Any, table: Any) -> None:
    with busy(message=f"Unloading {record.id}…"):
        try:
            await ctx.manager.unload(record.id)
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="unload")
            return
    ui.notify(f"{record.id} unloaded", type="positive")
    table.refresh()


async def _toggle_pin(ctx: GuiContext, record: Any, table: Any) -> None:
    settings = record.settings.model_copy(update={"pinned": not record.settings.pinned})
    try:
        updated = await run_blocking(ctx.registry.save_settings, record.id, settings)
    except Exception as exc:  # noqa: BLE001
        notify_error(exc, what="pin")
        return
    ui.notify(f"{record.id} {'pinned' if updated.settings.pinned else 'unpinned'}", type="positive")
    table.refresh()


async def _test(ctx: GuiContext, record: Any) -> None:
    with busy(message=f"Testing {record.id} (loads it if needed)…"):
        try:
            result = await ctx.manager.test_model(record.id, None)
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="test")
            return
    with ui.dialog() as dialog, ui.card().classes("min-w-[30rem]"):
        ui.label(f"Test — {record.id}").classes("font-medium")
        for line in st.test_result_lines(result):
            ui.label(line).classes("text-sm font-mono")
        if result.get("embedding_dims") is None:
            ui.label(str(result.get("text") or "(no text returned)")).classes(
                "text-xs whitespace-pre-wrap opacity-80"
            )
        with ui.row().classes("justify-end w-full gap-2"):
            if record.settings.draft_model_id:
                ui.button(
                    "A/B the draft model",
                    on_click=lambda: _draft_ab(ctx, record),
                ).props("outline dense").tooltip(
                    "Runs the same test with and without speculative decoding. "
                    "This reloads the model twice."
                )
            ui.button("Close", on_click=dialog.close).props("flat")
    dialog.open()


async def _draft_ab(ctx: GuiContext, record: Any) -> None:
    """Measure the draft model's real effect, then restore the original settings.

    Worth the two reloads: a poorly matched draft model can make generation
    slower while still looking like it is working, and the only way to know is
    to run the same prompt both ways and compare tok/s alongside the acceptance
    rate.
    """
    original = record.settings.model_copy(deep=True)
    without = original.model_copy(update={"draft_model_id": None})
    try:
        with busy(message="A/B: measuring with the draft model…"):
            with_draft = await ctx.manager.test_model(record.id, None)
        with busy(message="A/B: reloading without the draft model…"):
            await run_blocking(ctx.registry.save_settings, record.id, without)
            await ctx.manager.load(record.id, force=True, source="gui")
            without_draft = await ctx.manager.test_model(record.id, None)
    except Exception as exc:  # noqa: BLE001
        notify_error(exc, what="A/B comparison")
        return
    finally:
        # Always put the model back the way the user had it, even on failure.
        try:
            await run_blocking(ctx.registry.save_settings, record.id, original)
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="restore settings after A/B")

    with ui.dialog() as dialog, ui.card().classes("min-w-[30rem]"):
        ui.label(f"Speculative decoding A/B — {record.id}").classes("font-medium")
        ui.label(f"draft model: {original.draft_model_id}").classes("text-xs opacity-70 font-mono")
        for line in st.ab_comparison_lines(with_draft, without_draft):
            ui.label(line).classes("text-sm font-mono")
        ui.label(
            "Your settings have been restored; reload the model to put the draft back in play."
        ).classes("text-xs opacity-70")
        ui.button("Close", on_click=dialog.close).props("flat")
    dialog.open()


def _delete_dialog(ctx: GuiContext, record: Any, table: Any) -> None:
    loaded = ctx.supervisor is not None and ctx.supervisor.get(record.id) is not None
    with ui.dialog() as dialog, ui.card().classes("min-w-[26rem]"):
        ui.label(f"Delete {record.id}?").classes("font-medium")
        if loaded:
            # Same refusal the management API gives, in the same words.
            ui.label(f"model '{record.id}' is loaded; unload it before deleting").classes(
                "text-negative text-sm"
            )
            ui.button("Close", on_click=dialog.close).props("flat")
            dialog.open()
            return
        if record.is_virtual:
            # A virtual model is only a registration; its base's files are
            # never touched, and offering a files checkbox here would imply
            # otherwise.
            ui.label(
                f"This removes only the virtual model. The base model "
                f"({record.base_model_id or st.UNKNOWN}) and its files are untouched."
            ).classes("text-xs opacity-70")
            files = None
        else:
            ui.label(
                "Removing it from the registry is always safe. Deleting the files is not."
            ).classes("text-xs opacity-70")
            files = ui.checkbox("also delete the GGUF files from disk", value=False)

        async def confirm() -> None:
            delete_files = bool(files.value) if files is not None else False
            try:
                removed = await run_blocking(
                    ctx.registry.delete_model, record.id, delete_files=delete_files
                )
            except Exception as exc:  # noqa: BLE001
                notify_error(exc, what="delete")
                return
            dialog.close()
            if record.is_virtual:
                ui.notify(f"{record.id} removed", type="positive")
            else:
                ui.notify(
                    f"{record.id} removed ({len(removed)} file(s) "
                    f"{'deleted' if delete_files else 'left on disk'})",
                    type="positive",
                )
            table.refresh()

        with ui.row():
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("Delete", on_click=confirm).props("color=negative")
    dialog.open()


# ---------------------------------------------------------------------------
# The three-tier settings dialog
# ---------------------------------------------------------------------------


def _settings_dialog(ctx: GuiContext, record: Any, table: Any) -> None:  # noqa: C901
    form = st.form_from_settings(record.settings)
    attached: list[dict[str, Any]] = list(form["adapters"])
    all_records = list(ctx.registry.all()) if ctx.registry is not None else []
    drafts = st.plausible_draft_models(all_records, record)
    draft_options = [""] + [r.id for r in drafts]

    with ui.dialog() as dialog, ui.card().classes("min-w-[46rem] max-w-[95vw]"):
        ui.label(f"Settings — {record.id}").classes("font-medium")
        ui.label(
            f"{record.quant} · {st.format_gib(record.size_bytes)} · trained ctx "
            f"{record.meta.n_ctx_train if record.meta else st.UNKNOWN}"
        ).classes("text-xs opacity-70 font-mono")

        # --- optimal settings per hardware mode (D36) --------------------
        # Filled by a one-shot timer rather than inline: it plans this model at
        # every context tier on every hardware mode, which is arithmetic but
        # enough of it to make opening the dialog feel slow on a big library.
        placements_card = ui.card().classes("w-full bg-black/5 dark:bg-white/5")
        with placements_card:
            ui.label("Optimal settings — computing…").classes("text-xs opacity-60")
        ui.timer(
            0.05,
            lambda: _render_placements(ctx, record, placements_card, dialog, table),
            once=True,
        )

        # --- fit verdict -------------------------------------------------
        verdict_card = ui.card().classes("w-full bg-black/5 dark:bg-white/5")

        # --- Tier 1: basic ----------------------------------------------
        ui.label("Basic").classes("text-sm font-medium mt-2")
        with ui.row().classes("w-full gap-3 flex-wrap"):
            ctx_input = (
                ui.number("Context length (per conversation)", value=form["ctx_size"], precision=0)
                .props("dense outlined clearable")
                .classes("w-56")
            )
            kv_k = ui.select(
                list(_KV_TYPES), value=form["kv_cache_type"] or "", label="KV type (K)"
            )
            kv_k.props("dense outlined").classes("w-32")
            kv_v = ui.select(
                list(_KV_TYPES), value=form["kv_cache_type_v"] or "", label="KV type (V)"
            )
            kv_v.props("dense outlined").classes("w-32")
            ttl = ui.number("TTL seconds (0 = never)", value=form["ttl_s"], precision=0)
            ttl.props("dense outlined clearable").classes("w-44")
            pinned = ui.checkbox("Pinned", value=bool(form["pinned"]))
        with ui.row().classes("w-full gap-3 flex-wrap items-center"):
            draft = ui.select(
                draft_options, value=form["draft_model_id"] or "", label="Draft model"
            )
            draft.props("dense outlined").classes("w-80")
            device = (
                ui.input("Device override (e.g. 0,1)", value=form["device_override"])
                .props("dense outlined")
                .classes("w-52")
            )
        draft_note = ui.label("").classes("text-xs text-warning")
        if not drafts:
            ui.label(
                "No plausible draft model in the library: speculative decoding needs a smaller "
                "model with the same tokenizer (matching vocab size or architecture family)."
            ).classes("text-xs opacity-60")

        # --- adapters ----------------------------------------------------
        adapters_box = ui.column().classes("w-full gap-1")
        _render_attached(ctx, adapters_box, attached)

        # --- Tier 2: advanced -------------------------------------------
        with ui.expansion("Advanced", icon="settings").classes("w-full"):
            with ui.row().classes("w-full gap-3 flex-wrap"):
                batch = _num("Batch size", form["batch_size"])
                ubatch = _num("Ubatch size", form["ubatch_size"])
                threads = _num("Threads", form["threads"])
                threads_batch = _num("Threads (batch)", form["threads_batch"])
            with ui.row().classes("w-full gap-3 flex-wrap items-center"):
                parallel = _num("Parallel slots", form["parallel"])
                cont = ui.toggle(_TRI, value=_tri_from_value(form["cont_batching"]))
                cont.props("dense").tooltip("Continuous batching")
                flash = ui.select(
                    list(_FLASH_ATTN),
                    value=form["flash_attn"] or "",
                    label="flash-attn (takes a value)",
                )
                flash.props("dense outlined").classes("w-48")
            slot_hint = ui.label("").classes("text-xs text-warning")
            with ui.row().classes("w-full gap-3 flex-wrap items-center"):
                split = ui.select(
                    list(_SPLIT_MODES), value=form["split_mode"] or "", label="split-mode"
                )
                split.props("dense outlined").classes("w-40")
                main_gpu = _num("main-gpu", form["main_gpu"])
                mlock = ui.checkbox("mlock", value=bool(form["mlock"]))
                no_mmap = ui.toggle(_TRI, value=_tri_from_value(form["no_mmap"]))
                no_mmap.props("dense").tooltip("no-mmap")
            with ui.row().classes("w-full gap-3 flex-wrap"):
                rope_base = _num("rope-freq-base", form["rope_freq_base"], precision=None)
                rope_scale = _num("rope-freq-scale", form["rope_freq_scale"], precision=None)
                rope_scaling = ui.input("rope-scaling", value=form["rope_scaling"])
                rope_scaling.props("dense outlined").classes("w-40")
            with ui.row().classes("w-full gap-3 flex-wrap items-center"):
                cache_reuse = _num("cache-reuse", form["cache_reuse"])
                no_shift = ui.toggle(_TRI, value=_tri_from_value(form["no_context_shift"]))
                no_shift.props("dense").tooltip("no-context-shift")
                defrag = _num("defrag-thold", form["defrag_thold"], precision=None)
            reuse_hint = ui.label("").classes("text-xs opacity-80")
            ui.label("Reasoning / thinking models").classes("text-xs font-medium mt-1")
            with ui.row().classes("w-full gap-3 flex-wrap items-center"):
                reasoning_format = ui.select(
                    list(_REASONING_FORMATS),
                    value=form["reasoning_format"] or "",
                    label="reasoning-format",
                )
                reasoning_format.props("dense outlined").classes("w-44")
                reasoning_format.tooltip(st.REASONING_FORMAT_HELP)
                reasoning = ui.select(
                    list(_REASONING), value=form["reasoning"] or "", label="reasoning"
                )
                reasoning.props("dense outlined").classes("w-32")
                reasoning_budget = _num("reasoning-budget", form["reasoning_budget"])
            ui.label(st.REASONING_FORMAT_HELP).classes("text-xs opacity-70")
            reasoning_hint = ui.label("").classes("text-xs text-warning")
            ui.label("Default sampler parameters").classes("text-xs font-medium mt-1")
            with ui.row().classes("w-full gap-3 flex-wrap"):
                temperature = _num("temperature", form["temperature"], precision=None)
                top_p = _num("top-p", form["top_p"], precision=None)
                top_k = _num("top-k", form["top_k"])
                min_p = _num("min-p", form["min_p"], precision=None)
                repeat_penalty = _num("repeat-penalty", form["repeat_penalty"], precision=None)

        # --- Tier 3: expert ---------------------------------------------
        with ui.expansion("Expert", icon="warning").classes("w-full"):
            ui.label(
                "Passed straight to llama-server, after our own flags so a deliberate "
                "override wins. Validated against this engine's --help when you save."
            ).classes("text-xs opacity-70")
            extra = ui.textarea("Extra flags", value=form["extra_flags"])
            extra.props("dense outlined autogrow").classes("w-full font-mono")
            extra_errors = ui.column().classes("w-full gap-0")
            engine_tag = ui.input("Engine tag override", value=form["engine_tag"])
            engine_tag.props("dense outlined").classes("w-48")

        # --- live derivations -------------------------------------------
        def collect() -> dict[str, Any]:
            data = dict(form)
            data.update(
                {
                    "ctx_size": ctx_input.value,
                    "kv_cache_type": kv_k.value,
                    "kv_cache_type_v": kv_v.value,
                    "ttl_s": ttl.value,
                    "pinned": pinned.value,
                    "draft_model_id": draft.value,
                    "device_override": device.value,
                    "batch_size": batch.value,
                    "ubatch_size": ubatch.value,
                    "threads": threads.value,
                    "threads_batch": threads_batch.value,
                    "parallel": parallel.value,
                    "cont_batching": _tri_to_value(cont.value),
                    "flash_attn": flash.value,
                    "split_mode": split.value,
                    "main_gpu": main_gpu.value,
                    "mlock": mlock.value,
                    "no_mmap": _tri_to_value(no_mmap.value),
                    "rope_freq_base": rope_base.value,
                    "rope_freq_scale": rope_scale.value,
                    "rope_scaling": rope_scaling.value,
                    "cache_reuse": cache_reuse.value,
                    "no_context_shift": _tri_to_value(no_shift.value),
                    "defrag_thold": defrag.value,
                    "temperature": temperature.value,
                    "top_p": top_p.value,
                    "top_k": top_k.value,
                    "min_p": min_p.value,
                    "repeat_penalty": repeat_penalty.value,
                    "reasoning_format": reasoning_format.value,
                    "reasoning": reasoning.value,
                    "reasoning_budget": reasoning_budget.value,
                    "extra_flags": extra.value,
                    "engine_tag": engine_tag.value,
                    "adapters": attached,
                }
            )
            return data

        def refresh_derived() -> None:
            slot_hint.set_text(
                st.per_slot_ctx_hint(
                    _as_int(ctx_input.value),
                    _as_int(parallel.value),
                    default_ctx=ctx.config.models.default_ctx,
                )
            )
            reuse_hint.set_text(
                st.cache_reuse_hint(
                    _as_int(cache_reuse.value), ctx.config.models.default_cache_reuse
                )
            )
            reasoning_hint.set_text(
                st.reasoning_format_hint(
                    str(reasoning_format.value or "") or None,
                    ctx.config.models.default_reasoning_format,
                )
            )
            chosen = next((r for r in drafts if r.id == draft.value), None)
            draft_note.set_text(st.draft_uncertainty_note(record, chosen) or "")
            _refresh_verdict(ctx, record, verdict_card, ctx_input, kv_k, parallel)

        for widget in (ctx_input, kv_k, kv_v, parallel, cache_reuse, draft, reasoning_format):
            widget.on_value_change(lambda _: refresh_derived())
        refresh_derived()

        # --- actions -----------------------------------------------------
        async def save(*, then_load: bool) -> None:
            extra_errors.clear()
            try:
                settings = st.settings_from_form(collect())
            except Exception as exc:  # noqa: BLE001 - pydantic validation message
                notify_error(exc, what="settings")
                return
            if settings.extra_flags.strip():
                tag = settings.engine_tag or ctx.config.engine.pinned_tag
                try:
                    errors = await ctx.engine_manager.validate_extra_flags(
                        tag, settings.extra_flags
                    )
                except Exception as exc:  # noqa: BLE001
                    errors = [f"cannot validate flags: {exc}"]
                if errors:
                    with extra_errors:
                        for message in errors:
                            # Verbatim: these already name the replacement flag.
                            ui.label(message).classes("text-negative text-xs")
                    ui.notify(
                        "Extra flags rejected — see the Expert section",
                        type="negative",
                        multi_line=True,
                    )
                    return
            try:
                await run_blocking(ctx.registry.save_settings, record.id, settings)
            except Exception as exc:  # noqa: BLE001
                notify_error(exc, what="save settings")
                return
            ui.notify(f"settings saved for {record.id}", type="positive")
            table.refresh()
            if not then_load:
                dialog.close()
                return
            with busy(message=f"Loading {record.id}…"):
                try:
                    instance = await ctx.manager.load(record.id, force=True, source="gui")
                except Exception as exc:  # noqa: BLE001
                    notify_error(exc, what="load")
                    return
            dialog.close()
            ui.notify(f"{record.id} ready on port {instance.port}", type="positive")
            table.refresh()

        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("Save", on_click=lambda: save(then_load=False)).props("outline")
            ui.button("Save and load", on_click=lambda: save(then_load=True)).props("color=primary")
    dialog.open()


def _num(label: str, value: Any, *, precision: int | None = 0) -> Any:
    widget = ui.number(label, value=value, precision=precision)
    return widget.props("dense outlined clearable").classes("w-40")


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _render_placements(ctx: GuiContext, record: Any, card: Any, dialog: Any, table: Any) -> None:
    """The "Optimal settings" block: one line per hardware mode, with a Load button.

    Answers the question the settings form cannot: not "does *this* combination
    fit" but "what is the best this model can do on the 5090s, on the 3090s, on
    one card, on everything" -- each computed as if those cards were free, with
    what stands in the way right now said separately (D36).
    """
    card.clear()
    with card:
        try:
            profiles = ctx.manager.placement_profiles(record.id)
        except Exception as exc:  # noqa: BLE001 - never block the dialog on this
            ui.label(f"optimal settings unavailable: {exc}").classes("text-xs opacity-70")
            return
        ui.label("Optimal settings, per set of GPUs (as if those cards were free)").classes(
            "text-sm font-medium"
        )
        for note in profiles.get("quality_notes") or []:
            ui.label(note).classes("text-xs text-warning")
        lines = st.placement_lines(profiles)
        if not lines:
            ui.label("No GPU on this box can hold this model.").classes("text-xs opacity-70")
            return
        for index, line in enumerate(lines):
            with ui.row().classes("w-full items-center gap-2 flex-nowrap"):
                ui.badge(line.label, color="primary" if index == 0 else "secondary").classes(
                    "text-xs shrink-0"
                )
                ui.label(line.summary).classes("text-xs font-mono flex-1 min-w-0")
                ui.label(line.availability).classes(f"text-xs text-{line.colour} shrink-0")
                if line.load_args:
                    recipe = ", ".join(f"{k}={v}" for k, v in line.load_args.items())
                    ui.button(
                        "Load here",
                        on_click=lambda ln=line: _load_placement(ctx, ln, dialog, table),
                    ).props("flat dense size=sm").tooltip(f"load_model({recipe})")


async def _load_placement(ctx: GuiContext, line: Any, dialog: Any, table: Any) -> None:
    """Load one placement, confirming first when it would stop something else."""
    if line.would_evict:
        confirmed = await _confirm(
            f"Loading {line.label} needs {len(line.would_evict)} model(s) unloaded: "
            f"{', '.join(line.would_evict)}. Continue?"
        )
        if not confirmed:
            return
    args = dict(line.load_args)
    model_id = args.pop("model_id", None)
    with busy(message=f"Loading on {line.label}…"):
        try:
            instance = await ctx.manager.load(model_id, **args, force=True, source="gui")
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="load")
            return
    dialog.close()
    ui.notify(f"{model_id} ready on port {instance.port} ({line.label})", type="positive")
    table.refresh()


async def _confirm(question: str) -> bool:
    """A yes/no dialog, awaited. Used before anything that stops another model."""
    with ui.dialog() as confirm, ui.card():
        ui.label(question).classes("text-sm")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=lambda: confirm.submit(False)).props("flat")
            ui.button("Continue", on_click=lambda: confirm.submit(True)).props("color=warning")
    return bool(await confirm)


def _refresh_verdict(
    ctx: GuiContext, record: Any, card: Any, ctx_input: Any, kv_widget: Any, parallel_widget: Any
) -> None:
    """Re-run ``plan_preview`` and repaint the verdict card."""
    card.clear()
    with card:
        try:
            preview = ctx.manager.plan_preview(
                record.id,
                ctx_size=_as_int(ctx_input.value),
                kv_cache_type=(kv_widget.value or None),
                parallel=_as_int(parallel_widget.value),
            )
        except Exception as exc:  # noqa: BLE001 - a preview failure must not block the dialog
            ui.label(f"fit check unavailable: {exc}").classes("text-xs opacity-70")
            return
        verdict = st.fit_verdict(preview)
        with ui.row().classes("items-center gap-2"):
            ui.icon(
                "check_circle" if verdict.fits else "error",
                color=verdict.colour,
            )
            ui.label(verdict.headline).classes(f"text-{verdict.colour} font-medium text-sm")
        for line in verdict.detail_lines:
            ui.label(line).classes("text-xs font-mono opacity-80")
        for line in st.per_gpu_projection_lines(verdict):
            ui.label(line).classes("text-xs font-mono opacity-70")
        for note in verdict.notes:
            colour = "text-warning" if note == verdict.fp4_warning else "opacity-70"
            ui.label(f"note: {note}").classes(f"text-xs {colour}")
        for suggestion in verdict.suggestions:
            ui.label(f"try: {suggestion}").classes("text-xs text-warning")


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def _render_attached(
    ctx: GuiContext, box: Any, attached: list[dict[str, Any]], on_change: Any = None
) -> None:
    """Attached-adapter rows with a scale slider each.

    ``on_change`` fires after an attach/detach so the persona dialog can keep
    its instance-sharing indicator honest while the user edits the set.
    """
    available = list(ctx.registry.adapters()) if ctx.registry is not None else []
    box.clear()

    def changed() -> None:
        if on_change is not None:
            on_change()

    with box:
        ui.label("Attached LoRA adapters").classes("text-sm font-medium mt-2")
        if not available:
            ui.label("No GGUF LoRA adapters found in the model directories.").classes(
                "text-xs opacity-60"
            )
            return
        for item in list(attached):
            with ui.row().classes("items-center gap-2 w-full"):
                ui.label(str(item.get("adapter_id"))).classes("text-xs font-mono grow truncate")
                slider = ui.slider(
                    min=-2.0, max=2.0, step=0.05, value=float(item.get("scale", 1.0))
                )
                slider.classes("w-40")
                slider.on_value_change(
                    lambda event, row=item: row.update({"scale": float(event.value)})
                )
                ui.label().bind_text_from(slider, "value", lambda v: f"{float(v):.2f}").classes(
                    "text-xs font-mono w-12"
                )

                def detach(row: dict[str, Any] = item) -> None:
                    attached.remove(row)
                    _render_attached(ctx, box, attached, on_change)
                    changed()

                ui.button(icon="link_off", on_click=detach).props("flat dense")
        options = [a.id for a in available if a.id not in {i["adapter_id"] for i in attached}]
        if options:
            with ui.row().classes("items-center gap-2"):
                picker = ui.select(options, value=options[0], label="attach")
                picker.props("dense outlined").classes("w-72")

                def attach() -> None:
                    attached.append({"adapter_id": picker.value, "scale": 1.0})
                    _render_attached(ctx, box, attached, on_change)
                    changed()

                ui.button("Attach", icon="link", on_click=attach).props("outline dense")


def _adapters_panel(ctx: GuiContext, table: Any) -> None:
    with ui.expansion("LoRA adapters, virtual models and personas", icon="extension").classes(
        "w-full"
    ):
        ui.label(
            "A virtual model is a base under its own name, optionally carrying LoRA adapters "
            "and/or a persona (a system prompt plus sampler defaults, applied per request). "
            "That name is how clients pick all of it through the OpenAI API — the protocol "
            "has no LoRA or system-prompt parameter. A persona-only virtual model shares its "
            "base's running instance, so any number of personas cost no extra VRAM."
        ).classes("text-xs opacity-70")
        listing = ui.column().classes("w-full gap-1")

        def refresh_list() -> None:
            listing.clear()
            with listing, panel_guard("Adapters"):
                adapters = list(ctx.registry.adapters()) if ctx.registry is not None else []
                if not adapters:
                    ui.label("none found").classes("text-xs opacity-60")
                    return
                for adapter in adapters:
                    ui.label(
                        f"{adapter.id} · {st.format_bytes(adapter.size_bytes)}"
                        f" · rank {adapter.rank or st.UNKNOWN}"
                        f" · base {adapter.base_architecture or st.UNKNOWN}"
                    ).classes("text-xs font-mono opacity-80")

        refresh_list()
        with ui.row().classes("gap-2"):
            ui.button(
                "Rescan adapters",
                icon="refresh",
                on_click=lambda: _scan_adapters(ctx, refresh_list),
            ).props("outline dense")
            ui.button(
                "New virtual model / persona",
                icon="add",
                on_click=lambda: _persona_dialog(ctx, table),
            ).props("outline dense")


async def _scan_adapters(ctx: GuiContext, refresh_list: Any) -> None:
    with busy(message="Scanning for adapters…"):
        try:
            await run_blocking(ctx.registry.scan_adapters)
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="adapter scan")
            return
    refresh_list()


def _persona_dialog(ctx: GuiContext, table: Any, record: Any = None) -> None:  # noqa: C901
    """Create or edit a virtual model: base + adapters + persona (D13).

    The instance-sharing indicator is the load-bearing part: it tells the user
    *before* they commit whether this persona rides the base's instance for
    free or (because of adapters / setting overrides) will cost the VRAM of a
    dedicated llama-server child.
    """
    bases = [r for r in ctx.registry.all() if not r.is_virtual] if ctx.registry else []
    adapters = list(ctx.registry.adapters()) if ctx.registry is not None else []
    editing = record is not None
    chosen: list[dict[str, Any]] = (
        [a.model_dump(mode="python") for a in record.settings.adapters] if editing else []
    )
    preset_form = st.form_from_preset(record.preset if editing else None)
    has_overrides = editing and st.has_launch_overrides(record.settings)

    with ui.dialog() as dialog, ui.card().classes("min-w-[38rem] max-w-[95vw]"):
        ui.label(f"Edit {record.id}" if editing else "New virtual model / persona").classes(
            "font-medium"
        )
        name = ui.input(
            "Model id (what clients will ask for)", value=record.id if editing else ""
        ).props("dense outlined")
        name.classes("w-full")
        if editing:
            name.props("disable")
        base = ui.select(
            [r.id for r in bases],
            value=(record.base_model_id if editing else (bases[0].id if bases else None)),
            label="Base model",
        )
        base.props("dense outlined").classes("w-full")
        if editing:
            base.props("disable")
        if not bases:
            ui.label("No base models in the library — scan or download one first.").classes(
                "text-xs text-warning"
            )

        # --- instance sharing / VRAM cost -------------------------------
        with ui.row().classes("items-center gap-2 w-full"):
            share_icon = ui.icon("share", size="1.2rem")
            share_note = ui.label("").classes("text-xs grow")

        def refresh_sharing() -> None:
            shares, text = st.virtual_instance_note(
                str(base.value or "the base"),
                has_adapters=bool(chosen),
                has_overrides=bool(has_overrides),
            )
            share_icon.props(
                f"name={'share' if shares else 'memory'} "
                f"color={'positive' if shares else 'warning'}"
            )
            share_note.set_text(text)

        base.on_value_change(lambda _: refresh_sharing())
        if has_overrides:
            ui.label(
                "This virtual model has launch-time setting overrides (from its settings "
                "dialog); clearing those there would let it share the base's instance."
            ).classes("text-xs opacity-60")

        # --- persona: system prompt + sampler defaults -------------------
        ui.label("Persona").classes("text-sm font-medium mt-2")
        ui.label(
            "Applied to each request by the gateway. The system prompt is prepended before "
            "the client's own; sampler defaults fill only fields the client left unset."
        ).classes("text-xs opacity-70")
        system_prompt = ui.textarea("System prompt", value=str(preset_form["system_prompt"]))
        system_prompt.props("dense outlined autogrow").classes("w-full")
        with ui.row().classes("w-full gap-3 flex-wrap"):
            temperature = _num("temperature", preset_form["temperature"], precision=None)
            top_p = _num("top-p", preset_form["top_p"], precision=None)
            top_k = _num("top-k", preset_form["top_k"])
            min_p = _num("min-p", preset_form["min_p"], precision=None)
            repeat_penalty = _num("repeat-penalty", preset_form["repeat_penalty"], precision=None)
            max_tokens = _num("max-tokens", preset_form["max_tokens"])

        # --- adapters -----------------------------------------------------
        box = ui.column().classes("w-full gap-1")
        if not adapters:
            ui.label(
                "No LoRA adapters in the library — persona-only is fine, and costs no VRAM."
            ).classes("text-xs opacity-60")
        _render_attached(ctx, box, chosen, on_change=refresh_sharing)
        refresh_sharing()

        async def save() -> None:
            from studioforge.types import AdapterAttachment

            try:
                preset = st.preset_from_form(
                    {
                        "system_prompt": system_prompt.value,
                        "temperature": temperature.value,
                        "top_p": top_p.value,
                        "top_k": top_k.value,
                        "min_p": min_p.value,
                        "repeat_penalty": repeat_penalty.value,
                        "max_tokens": max_tokens.value,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - pydantic validation message
                notify_error(exc, what="persona")
                return
            try:
                saved = await run_blocking(
                    lambda: ctx.registry.create_virtual_model(
                        id=str(name.value or "").strip(),
                        base_model_id=str(base.value or ""),
                        name=None,
                        adapters=[AdapterAttachment.model_validate(a) for a in chosen],
                        preset=preset,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                notify_error(exc, what="save virtual model")
                return
            dialog.close()
            ui.notify(f"{'updated' if editing else 'created'} {saved.id}", type="positive")
            table.refresh()

        with ui.row().classes("justify-end gap-2 w-full"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("Save" if editing else "Create", on_click=save).props("color=primary")
    dialog.open()
