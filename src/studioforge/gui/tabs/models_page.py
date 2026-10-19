"""Models: one page for the whole library.

Replaces the Library / Download / Chat sub-tabs and the cards that used to sit under the
table. A single searchable table (friendly name, quant badge, size, family, "fits on" hint,
state) with filter chips (All, Loaded, Downloads), and a details drawer per model that holds
everything you can do to one. The drawer is also what Test Chat opens to load a model.

Nothing here decides anything. Every action calls the function the old tab called:

* load -> ``manager.load`` (with the catalog's recommended ``load_args`` pre-filled) after a
  ``manager.plan_preview`` that is drawn as per-GPU bars *before* the button is pressed;
* unload -> ``manager.unload``; pin -> ``manager.set_pinned``; delete ->
  ``registry.delete_model``; sampler defaults -> ``registry.save_settings``;
* downloads -> the Download tab's own panels, unchanged;
* everything else (full settings dialog, test, benchmark, personas) is the old dialog.

Failures show in the row with the catalog's error code and the plain sentence it carries.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from nicegui import ui

from studioforge.gui import models_view as mv
from studioforge.gui import nav
from studioforge.gui import state as st
from studioforge.gui.tabs import (
    GuiContext,
    admin_control,
    notify_error,
    panel_guard,
    require_local_admin,
    run_blocking,
    single_flight,
    viewer_may_change_box,
)
from studioforge.gui.tabs import models as legacy
from studioforge.logging import get_logger

log = get_logger(__name__)

#: Loads this process has started and not yet seen return, ``{model id: started at}``.
#: Process-wide on purpose (like ``single_flight``): a second browser looking at the same
#: rig should see the same row spinning.
_PENDING: dict[str, float] = {}

#: The last failed load per model, ``{model id: LoadError}``. Cleared by a successful load
#: or an unload; shown only while the model is not running.
_ERRORS: dict[str, mv.LoadError] = {}

_QUICK_CTX = (
    ("8k", 8192),
    ("16k", 16384),
    ("32k", 32768),
    ("64k", 65536),
    ("128k", 131072),
    ("256k", 262144),
)

_KV_TYPES = ("", "f16", "q8_0", "q4_0", "bf16", "f32", "q5_1", "q5_0", "q4_1")

_PAGE_CSS = """
.sfp-row{display:grid;
  grid-template-columns:minmax(14rem,3.2fr) 6.5rem 6rem 7rem minmax(10rem,2fr) minmax(9rem,1.6fr);
  column-gap:.75rem;align-items:center;width:100%;}
.sfp-row>*{min-width:0;}
.sfp-body{cursor:pointer;border-radius:6px;padding:.4rem .25rem;}
.sfp-body:hover,.sfp-body:focus-visible{background:rgba(127,127,127,.12);outline:none;}
.sfp-body:focus-visible{box-shadow:0 0 0 2px var(--q-primary);}
.sfp-txt{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.sfp-narrow{display:none;}
.sfp-bar{position:relative;height:12px;border-radius:6px;background:rgba(127,127,127,.25);overflow:hidden;}
.sfp-bar>.sfp-used{position:absolute;left:0;top:0;bottom:0;background:rgba(127,127,127,.65);}
.sfp-bar>.sfp-add{position:absolute;top:0;bottom:0;background:var(--q-primary);}
.sfp-bar.sfp-over>.sfp-add{background:var(--q-negative);}
@media (max-width:820px){
  .sfp-row{grid-template-columns:1fr auto;}
  .sfp-wide{display:none!important;}
  .sfp-narrow{display:block;}
}
"""


class _Refresher:
    """Anything with ``.refresh()``, fanned out: the legacy dialogs take a ``table``."""

    def __init__(self, *targets: Callable[[], Any]) -> None:
        self._targets = targets

    def refresh(self) -> None:
        for target in self._targets:
            try:
                target()
            except Exception as exc:  # noqa: BLE001 - a stale target must not break the action
                log.debug("refresh target failed", error=str(exc))


# ---------------------------------------------------------------------------
# Shared reads
# ---------------------------------------------------------------------------


def _records(ctx: GuiContext) -> list[Any]:
    try:
        return list(ctx.registry.all()) if ctx.registry is not None else []
    except Exception:  # noqa: BLE001
        return []


def _instances(ctx: GuiContext) -> dict[str, Any]:
    try:
        return {i.model_id: i for i in ctx.supervisor.list()} if ctx.supervisor else {}
    except Exception:  # noqa: BLE001
        return {}


def _gpus(ctx: GuiContext) -> list[Any]:
    try:
        return list(ctx.probe.list_gpus()) if ctx.probe is not None else []
    except Exception:  # noqa: BLE001
        return []


def _leases(ctx: GuiContext) -> list[Any]:
    try:
        return list(ctx.manager.leases.all()) if ctx.manager is not None else []
    except Exception:  # noqa: BLE001
        return []


def _download_count(ctx: GuiContext) -> int:
    try:
        return len(list(ctx.downloader.active())) if ctx.downloader is not None else 0
    except Exception:  # noqa: BLE001
        return 0


def aliases_of(ctx: GuiContext, record: Any) -> list[str]:
    """The extra names a client may use for this model, beyond its id.

    The registry derives short aliases (file stem, ``publisher/stem``, repo) and resolves
    them in ``resolve``; there is no public listing, so the candidates come from the same
    method the registry builds them with and each is checked through ``resolve`` -- only
    names that really answer for this model are shown.
    """
    registry = ctx.registry
    candidates = getattr(registry, "_alias_candidates", None)
    if registry is None or candidates is None:
        return []
    try:
        names = list(candidates(record))
    except Exception:  # noqa: BLE001
        return []
    out: list[str] = []
    for name in names:
        if name == record.id.lower() or name in {o.lower() for o in out}:
            continue
        try:
            hit = registry.resolve(name)
        except Exception:  # noqa: BLE001
            hit = None
        if hit is not None and hit.id == record.id:
            out.append(name)
    return out


def error_of(exc: BaseException) -> mv.LoadError:
    """The catalog code and the plain sentence for a failed action."""
    code = str(getattr(exc, "code", "") or type(exc).__name__)
    message = getattr(exc, "message", None) or str(exc) or code
    return mv.LoadError(code=code, message=str(message), at=time.time())


async def run_load(
    ctx: GuiContext, model_id: str, kwargs: Mapping[str, Any], *, force: bool
) -> Any:
    """The one call every Load button makes: ``manager.load`` with the chosen arguments.

    Marks the model as loading for the row, and records the failure (code and sentence) for
    the row to show; the error is re-raised so the caller can toast it too. ``kwargs`` is
    exactly what the catalog's ``load_args`` carry (``ctx_size``, ``kv_cache_type``,
    ``parallel``, ``devices``), so the planner stays the only decider.
    """
    _PENDING[model_id] = time.time()
    _ERRORS.pop(model_id, None)
    try:
        return await ctx.manager.load(model_id, **dict(kwargs), force=force, source="gui")
    except Exception as exc:
        _ERRORS[model_id] = error_of(exc)
        raise
    finally:
        _PENDING.pop(model_id, None)


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


class ModelsPage:
    """The Models tab for one browser page. ``apply`` moves it to a link's intent."""

    def __init__(self, ctx: GuiContext, go: Callable[[nav.Location], None]) -> None:
        self.ctx = ctx
        self.go = go
        sort_key, descending = legacy._stored_sort()
        self.view: dict[str, Any] = {
            "chip": nav.FILTER_ALL,
            "needle": "",
            "sort": sort_key,
            "desc": descending,
            "fits": {},
            "fits_at": 0.0,
            "sig": None,
        }
        self.drawer: ModelDrawer | None = None
        self.search: Any = None
        self.body: Any = None
        self.chips: Any = None
        self.link_params: dict[str, Any] | None = None

    # -- building ----------------------------------------------------------------

    def build(self, location: nav.Location, params: Mapping[str, Any] | None = None) -> None:
        ctx = self.ctx
        ui.add_css(_PAGE_CSS)
        if location.repo or (params or {}).get("error"):
            self.link_params = dict(params or {"repo": location.repo, "quant": location.quant})
        self.view["chip"] = location.filter or nav.FILTER_ALL
        self.view["needle"] = location.q or ""
        if location.model and not self._knows(location.model):
            # A link to a model this library does not have: show the empty result saying so.
            self.view["needle"] = location.model
        refresh = _Refresher(lambda: self.refresh_all())
        self.drawer = ModelDrawer(ctx, self.go, on_change=refresh.refresh)

        @ui.refreshable
        def body() -> None:
            self._body()

        self.body = body
        with ui.column().classes("w-full gap-3 p-2"):
            self._toolbar()
            body()
            with (
                ui.column()
                .classes("w-full")
                .bind_visibility_from(
                    self.view, "chip", backward=lambda c: c != nav.FILTER_DOWNLOADS
                ),
                panel_guard("Virtual models & adapters"),
            ):
                legacy._adapters_panel(ctx, _Refresher(self.refresh_all))

        ui.timer(0.05, self._load_fits, once=True)
        ui.timer(max(2.0, ctx.refresh_interval), self._tick)
        self.apply(location, first=True)

    def apply(self, location: nav.Location, *, first: bool = False) -> None:
        """Move to what a link or another tab asked for (filter, search, drawer, add dialog)."""
        if not first:
            self.view["chip"] = location.filter or self.view["chip"]
            if location.q:
                self.view["needle"] = location.q
                if self.search is not None:
                    self.search.set_value(location.q)
            self.refresh_all()
        if location.model and self.drawer is not None:
            model_id = location.model
            if self._knows(model_id):
                ui.timer(0.1, lambda: self.drawer.open(model_id), once=True)
            elif not first:
                self.view["needle"] = model_id
                self.refresh_all()
        if location.repo and not first:
            self.show_add(st.deep_link_params({"repo": location.repo, "quant": location.quant}))

    def _knows(self, model_id: str) -> bool:
        try:
            return self.ctx.registry is not None and self.ctx.registry.resolve(model_id) is not None
        except Exception:  # noqa: BLE001
            return False

    # -- refresh -----------------------------------------------------------------

    def refresh_all(self) -> None:
        self.view["sig"] = None
        if self.chips is not None:
            self.chips.refresh()
        if self.body is not None:
            self.body.refresh()

    def _signature(self) -> tuple[Any, ...]:
        instances = _instances(self.ctx)
        rows = tuple(
            (
                r.id,
                bool(r.settings.pinned),
                str(getattr(instances.get(self._serving_id(r)), "state", "")),
                int(getattr(instances.get(self._serving_id(r)), "active_requests", 0) or 0),
            )
            for r in _records(self.ctx)
        )
        pending = tuple(sorted((k, int((time.time() - v) // 2)) for k, v in _PENDING.items()))
        errors = tuple(sorted((k, e.code) for k, e in _ERRORS.items()))
        return (
            rows,
            pending,
            errors,
            _download_count(self.ctx),
            self.view["chip"],
            len(self.view["fits"]),
            tuple(sorted(mv.leased_model_ids(_leases(self.ctx), instances.values()))),
        )

    async def _tick(self) -> None:
        if self.body is None:
            return
        signature = self._signature()
        if signature != self.view["sig"]:
            self.view["sig"] = signature
            self.chips.refresh()
            # The Downloads view keeps its own search box and polls its own queue; rebuilding
            # it here would throw away what is being typed.
            if self.view["chip"] != nav.FILTER_DOWNLOADS:
                self.body.refresh()
        if time.time() - float(self.view["fits_at"]) > 20.0:
            await self._load_fits()

    async def _load_fits(self) -> None:
        manager = self.ctx.manager
        if manager is None:
            return
        try:
            payload = await run_blocking(manager.catalog, compact=True)
        except Exception as exc:  # noqa: BLE001 - the hint is a nicety
            log.debug("catalog unavailable for the fits-on column", error=str(exc))
            self.view["fits_at"] = time.time()
            return
        self.view["fits"] = mv.catalog_by_id(payload)
        self.view["fits_at"] = time.time()
        self.view["sig"] = None
        if self.body is not None:
            self.body.refresh()

    def _serving_id(self, record: Any) -> str:
        return legacy._serving_id(self.ctx, record)

    # -- toolbar -----------------------------------------------------------------

    def _toolbar(self) -> None:
        with ui.row().classes("w-full items-center gap-2 flex-wrap"):
            self.search = (
                ui.input(
                    placeholder="Search models, quant, family, capability…",
                    value=self.view["needle"],
                )
                .props('dense outlined clearable aria-label="Search models"')
                .classes("grow min-w-[14rem] max-w-[34rem]")
            )
            self.search.on_value_change(self._on_search)

            @ui.refreshable
            def chips() -> None:
                self._chips()

            self.chips = chips
            chips()
            ui.space()
            sort = ui.select(
                {c.key: f"Sort: {c.label}" for c in st.MODEL_COLUMNS},
                value=self.view["sort"],
                on_change=lambda e: self._set_sort(e.value),
            ).props('dense outlined options-dense aria-label="Sort models"')
            sort.classes("w-44")
            ui.button("Add model", icon="add", on_click=lambda: self.show_add()).props(
                "color=primary dense no-caps"
            ).tooltip("Download from Hugging Face: paste a repo id or a link, or search")
            ui.button("Rescan", icon="refresh", on_click=self._rescan).props(
                "outline dense no-caps"
            ).tooltip("Look for new or removed model files")

    def _chips(self) -> None:
        ctx = self.ctx
        records = _records(ctx)
        instances = _instances(ctx)
        loaded = len({r.id for r in records if self._serving_id(r) in instances})
        with ui.row().classes("items-center gap-1"):
            for chip, label, count in (
                (nav.FILTER_ALL, "All", len(records)),
                (nav.FILTER_LOADED, "Loaded", loaded),
                (nav.FILTER_DOWNLOADS, "Downloads", _download_count(ctx)),
            ):
                active = self.view["chip"] == chip
                button = ui.button(
                    f"{label} {count}", on_click=lambda _=None, c=chip: self._set_chip(c)
                )
                button.props(
                    "rounded no-caps "
                    + ("unelevated color=primary" if active else "outline")
                    + f' aria-pressed="{"true" if active else "false"}"'
                )

    def _on_search(self, event: Any) -> None:
        self.view["needle"] = str(event.value or "")
        if self.body is not None:
            self.body.refresh()

    def _set_chip(self, chip: str) -> None:
        self.view["chip"] = chip
        self.refresh_all()

    def _set_sort(self, key: str) -> None:
        if key == self.view["sort"]:
            return
        self.view["sort"] = key
        self.view["desc"] = st.MODEL_COLUMN_BY_KEY[key].descending_first
        legacy._store_sort(self.view["sort"], self.view["desc"])
        if self.body is not None:
            self.body.refresh()

    async def _rescan(self) -> None:
        await legacy._scan(self.ctx, _Refresher(self.refresh_all))

    # -- body --------------------------------------------------------------------

    def _body(self) -> None:
        if self.view["chip"] == nav.FILTER_DOWNLOADS:
            self._downloads_view()
            return
        with panel_guard("The model table"):
            self._table()

    def _table(self) -> None:
        ctx = self.ctx
        records = _records(ctx)
        instances = _instances(ctx)
        leased = mv.leased_model_ids(_leases(ctx), instances.values())
        total = len(records)
        loaded_ids = {r.id for r in records if self._serving_id(r) in instances}
        rows = mv.filter_rows(
            records, needle=self.view["needle"], chip=self.view["chip"], loaded_ids=loaded_ids
        )
        rows = st.sort_models(rows, self.view["sort"], self.view["desc"], loaded_ids=loaded_ids)
        if not rows:
            self._empty(total)
            return
        shown = f"{len(rows)} of {total}" if len(rows) != total else f"{total}"
        ui.label(f"{shown} model(s)").classes("text-xs opacity-60")
        arch = legacy._arch_fields(ctx, rows)
        with ui.column().classes("w-full gap-0"):
            self._header()
            for record in rows:
                self._row(record, instances, leased, arch.get(record.id))

    def _empty(self, total: int) -> None:
        with ui.column().classes("w-full items-start gap-2 p-4"):
            if total:
                ui.label(
                    f"No model matches '{self.view['needle']}'"
                    + (" among the loaded ones." if self.view["chip"] == nav.FILTER_LOADED else ".")
                ).classes("text-sm opacity-80")
                return
            ui.label("No models yet.").classes("text-base font-medium")
            ui.label(
                "Add one from Hugging Face, or point StudioForge at a folder that already has "
                "GGUF files (Advanced > Catalog & downloads) and Rescan."
            ).classes("text-sm opacity-80")
            with ui.row().classes("gap-2"):
                ui.button("Add model", icon="add", on_click=lambda: self.show_add()).props(
                    "color=primary no-caps"
                )
                ui.button(
                    "Catalog sources",
                    icon="folder_open",
                    on_click=lambda: self.go(nav.Location(nav.ADVANCED, page="catalog")),
                ).props("outline no-caps")

    def _header(self) -> None:
        active, descending = self.view["sort"], bool(self.view["desc"])

        def cell(label: str, key: str | None, classes: str = "") -> None:
            with ui.element("div").classes(classes):
                if key is None:
                    ui.label(label).classes("text-xs opacity-60 px-1")
                    return
                arrow = st.sort_indicator(key, active, descending)
                props = "flat dense no-caps size=sm align=left"
                props += " color=primary" if key == active else " color=grey"
                if arrow:
                    props += f" icon-right={arrow}"
                ui.button(label, on_click=lambda _=None, k=key: self._sort_click(k)).props(props)

        with ui.element("div").classes("sfp-row sfp-wide border-b border-white/20 pb-1 mb-1"):
            cell("Model", "name")
            cell("Quant", "quant")
            cell("Size", "size")
            cell("Family", "architecture")
            cell("Fits on", None)
            cell("State", "loaded")

    def _sort_click(self, key: str) -> None:
        self.view["sort"], self.view["desc"] = st.next_sort(
            self.view["sort"], bool(self.view["desc"]), key
        )
        legacy._store_sort(self.view["sort"], self.view["desc"])
        self.refresh_all()

    def _row(
        self,
        record: Any,
        instances: Mapping[str, Any],
        leased: set[str],
        arch: Mapping[str, Any] | None,
    ) -> None:
        serving_id = self._serving_id(record)
        instance = instances.get(serving_id)
        refusal = st.arch_load_refusal(arch)
        state = mv.row_state(
            record,
            instance,
            pinned=bool(record.settings.pinned),
            leased=record.id in leased or serving_id in leased,
            error=_ERRORS.get(record.id),
            pending_since=_PENDING.get(record.id) or _PENDING.get(serving_id),
            arch_refusal=refusal,
        )
        hint = mv.fits_hint(self.view["fits"].get(record.id))
        name = mv.friendly_name(record)
        row = ui.element("div").classes("sfp-row sfp-body border-b border-white/10")
        row.props(f'role="button" tabindex="0" aria-label="Open details for {_attr(name)}"')
        row.on("click", lambda _=None, r=record.id: self._open(r))
        row.on("keydown.enter", lambda _=None, r=record.id: self._open(r))
        row.on("keydown.space.prevent", lambda _=None, r=record.id: self._open(r))
        with row:
            with ui.element("div").classes("min-w-0"):
                ui.label(name).classes("sfp-txt font-medium text-sm").tooltip(record.id)
                added = f"added {st.format_when(st.model_added_at(record))}"
                ui.label(f"{record.id} · {added}" if record.id != name else added).classes(
                    "sfp-txt text-xs opacity-60 font-mono"
                )
                base_line = st.virtual_base_line(record)
                if base_line:
                    ui.label(base_line).classes("sfp-txt text-xs opacity-70").tooltip(base_line)
                for line in st.preset_summary_lines(record.preset):
                    ui.label(line).classes("sfp-txt text-xs opacity-60").tooltip(line)
                virtual_badges = [
                    text for text in st.capability_badges(record) if text in legacy._VIRTUAL_BADGES
                ]
                if virtual_badges:
                    with ui.row().classes("gap-1 flex-wrap"):
                        for text in virtual_badges:
                            ui.badge(text, color="secondary").classes("text-xs")
                ui.label(
                    f"{record.quant} · {st.format_gib(record.size_bytes, precision=1)} · "
                    f"{mv.family_label(record)}"
                ).classes("sfp-narrow sfp-txt text-xs opacity-80")
            ui.badge(record.quant, color=mv.quant_colour(record.quant)).classes(
                "sfp-wide text-xs justify-self-start"
            )
            ui.label(st.format_gib(record.size_bytes, precision=1)).classes(
                "sfp-wide text-xs font-mono"
            )
            ui.label(mv.family_label(record)).classes("sfp-wide sfp-txt text-xs font-mono")
            ui.label(hint.text).classes(f"sfp-wide sfp-txt text-xs text-{hint.colour}").tooltip(
                hint.tooltip or hint.text
            )
            with ui.element("div").classes("min-w-0"):
                self._state_cell(record, state)

    def _state_cell(self, record: Any, state: mv.RowState) -> None:
        with ui.row().classes("items-center gap-1 flex-wrap"):
            if state.primary == mv.STATE_LOADING:
                ui.spinner(size="xs").props(f'aria-label="Loading {_attr(record.id)}"')
            ui.badge(state.primary, color=state.colour).classes("text-xs")
            for text, colour, tooltip in state.badges:
                ui.badge(text, color=colour).classes("text-xs").tooltip(tooltip)
        if state.primary == mv.STATE_LOADING and state.progress_s is not None:
            ui.label(f"loading · {st.format_duration(state.progress_s)}").classes(
                "text-xs opacity-70"
            )
        elif state.primary == mv.STATE_LOADED and state.detail:
            ui.label(state.detail).classes("text-xs opacity-70 sfp-txt")
        elif state.error is not None:
            ui.label(f"[{state.error.code}] {state.error.message}").classes(
                "text-xs text-negative whitespace-normal break-words"
            ).props('role="alert"')

    def _open(self, model_id: str) -> None:
        if self.drawer is not None:
            self.drawer.open(model_id)

    # -- add a model -------------------------------------------------------------

    def show_add(self, params: Mapping[str, Any] | None = None) -> None:
        """Switch to the Downloads chip, with the add-a-model box ready (and a repo looked up)."""
        self.link_params = dict(params) if params else None
        self.view["chip"] = nav.FILTER_DOWNLOADS
        self.refresh_all()

    def _downloads_view(self) -> None:
        """Add by repo id or link, search Hugging Face, and the queue: the old Download tab."""
        from studioforge.gui.tabs import download

        ctx = self.ctx
        link, self.link_params = self.link_params, None
        with panel_guard("Add a model"), ui.card().classes("w-full"):
            ui.label("Add a model").classes("text-lg font-medium")
            ui.label(
                "Paste a Hugging Face repo id (owner/repo) or a link to the model page, or "
                "search below. Picked files are queued in the list underneath."
            ).classes("text-xs opacity-70")
            with ui.row().classes("w-full items-start gap-2 no-wrap"):
                box = (
                    ui.input(placeholder="owner/repo or https://huggingface.co/owner/repo")
                    .props('dense outlined clearable aria-label="Hugging Face repo id or link"')
                    .classes("grow")
                )
                look = ui.button("Look up", icon="search").props("color=primary dense no-caps")
            picker = ui.column().classes("w-full gap-1")

            def lookup(raw: str | None = None, pick_quant: str | None = None) -> None:
                picker.clear()
                repo_id, error = mv.parse_hf_reference(raw if raw is not None else box.value)
                with picker:
                    if error or not repo_id:
                        ui.label(error or "").classes("text-xs text-warning")
                        return
                    download._deep_link_panel(ctx, {"repo": repo_id, "quant": pick_quant})

            look.on_click(lambda: lookup())
            box.on("keydown.enter", lambda: lookup())
            if link:
                if link.get("repo"):
                    box.set_value(str(link["repo"]))
                with picker:
                    download._deep_link_panel(ctx, link)
        with (
            panel_guard("Search"),
            ui.expansion("Search Hugging Face", icon="search", value=not link).classes("w-full"),
        ):
            download._search_panel(ctx)
        with panel_guard("Downloads"):
            download._queue_panel(ctx)


def _attr(text: str) -> str:
    """An id made safe inside a double-quoted HTML/Quasar attribute."""
    return str(text).replace('"', "'")


# ---------------------------------------------------------------------------
# The drawer
# ---------------------------------------------------------------------------


class ModelDrawer:
    """The per-model details drawer, shared by the Models page and Test Chat."""

    def __init__(
        self,
        ctx: GuiContext,
        go: Callable[[nav.Location], None],
        *,
        on_change: Callable[[], Any] | None = None,
    ) -> None:
        self.ctx = ctx
        self.go = go
        self.on_change = on_change or (lambda: None)
        self.model_id: str | None = None
        self.dialog = ui.dialog().props("position=right full-height")
        with (
            self.dialog,
            ui.card()
            .classes("w-[40rem] max-w-[100vw] h-full overflow-auto")
            .style("border-radius:0"),
        ):
            self.content = ui.column().classes("w-full gap-3")
        ui.timer(max(2.0, ctx.refresh_interval), self._tick)
        self._sig: Any = None

    # -- open/close --------------------------------------------------------------

    def open(self, model_id: str) -> None:
        self.model_id = model_id
        self._build()
        self.dialog.open()

    def close(self) -> None:
        self.dialog.close()

    def changed(self) -> None:
        """An action finished: repaint the drawer and tell the page."""
        if self.model_id is not None and self.dialog.value:
            self._build()
        self.on_change()

    def _record(self) -> Any | None:
        if self.model_id is None or self.ctx.registry is None:
            return None
        try:
            return self.ctx.registry.resolve(self.model_id)
        except Exception:  # noqa: BLE001
            return None

    def _tick(self) -> None:
        # Repaint only when this model's state changed, so form edits are not wiped.
        if self.model_id is None or not self.dialog.value:
            return
        record = self._record()
        if record is None:
            return
        instance = _instances(self.ctx).get(legacy._serving_id(self.ctx, record))
        sig = (
            str(getattr(instance, "state", "")),
            int(getattr(instance, "active_requests", 0) or 0),
            record.id in _PENDING,
            record.id in _ERRORS,
            bool(record.settings.pinned),
        )
        if sig != self._sig:
            self._sig = sig
            state_box = getattr(self, "_state_box", None)
            if state_box is not None:
                state_box.refresh()

    # -- content -----------------------------------------------------------------

    def _build(self) -> None:
        self.content.clear()
        record = self._record()
        with self.content:
            if record is None:
                ui.label("That model is no longer in the library.").classes("text-sm")
                ui.button("Close", on_click=self.close).props("flat no-caps")
                return
            with panel_guard("The model details"):
                self._header(record)

                @ui.refreshable
                def state_box() -> None:
                    fresh = self._record() or record
                    self._state(fresh)

                self._state_box = state_box
                state_box()
                self._load_section(record)
                self._manage_section(record)
                self._sampler_section(record)
                self._presets_section(record)
                self._aliases_section(record)

    def _header(self, record: Any) -> None:
        with ui.row().classes("w-full items-start no-wrap gap-2"):
            with ui.column().classes("gap-0 grow min-w-0"):
                ui.label(mv.friendly_name(record)).classes("text-lg font-medium break-words")
                with ui.row().classes("items-center gap-1 no-wrap"):
                    ui.label(record.id).classes("text-xs font-mono opacity-70 sfp-txt")
                    ui.button(
                        icon="content_copy",
                        on_click=lambda: legacy._copy_text(record.id, what=f"copied '{record.id}'"),
                    ).props('flat dense size=sm aria-label="Copy the model id"').tooltip(
                        "Copy the model id (what clients ask for)"
                    )
            ui.button(icon="close", on_click=self.close).props(
                'flat round dense aria-label="Close the details"'
            )
        with ui.row().classes("items-center gap-2 flex-wrap"):
            ui.badge(record.quant, color=mv.quant_colour(record.quant)).classes("text-xs")
            ui.label(st.format_gib(record.size_bytes)).classes("text-xs font-mono")
            ui.label(mv.family_label(record)).classes("text-xs font-mono")
            ui.label(record.kind).classes("text-xs opacity-70")
            for item in st.capability_icons(record):
                ui.icon(item.icon, color=item.colour, size="1.15rem").tooltip(item.tooltip)
        base_line = st.virtual_base_line(record)
        if base_line:
            ui.label(base_line).classes("text-xs opacity-70")

    # -- state -------------------------------------------------------------------

    def _state(self, record: Any) -> None:
        ctx = self.ctx
        serving_id = legacy._serving_id(ctx, record)
        instance = _instances(ctx).get(serving_id)
        arch = legacy._arch_fields(ctx, [record]).get(record.id)
        leased = mv.leased_model_ids(_leases(ctx), _instances(ctx).values())
        state = mv.row_state(
            record,
            instance,
            pinned=bool(record.settings.pinned),
            leased=record.id in leased or serving_id in leased,
            error=_ERRORS.get(record.id),
            pending_since=_PENDING.get(record.id) or _PENDING.get(serving_id),
            arch_refusal=st.arch_load_refusal(arch),
        )
        with ui.card().classes("w-full sf-well gap-1"):
            with ui.row().classes("items-center gap-2"):
                if state.primary == mv.STATE_LOADING:
                    ui.spinner(size="sm")
                ui.badge(state.primary, color=state.colour).classes("text-sm")
                for text, colour, tooltip in state.badges:
                    ui.badge(text, color=colour).classes("text-xs").tooltip(tooltip)
                if state.detail and state.primary != mv.STATE_ERROR:
                    ui.label(state.detail).classes("text-xs opacity-70")
            if instance is not None and instance.plan is not None and state.primary == "loaded":
                plan = instance.plan
                ui.label(
                    f"{st.device_text(instance)} · context {plan.ctx_size} x {plan.parallel} "
                    f"slot(s) · KV {plan.kv_cache_type} · port {instance.port}"
                ).classes("text-xs font-mono opacity-80")
            if state.error is not None:
                ui.label(f"[{state.error.code}]").classes("text-xs font-mono text-negative")
                ui.label(state.error.message).classes(
                    "text-sm text-negative whitespace-pre-wrap"
                ).props('role="alert"')

    # -- load --------------------------------------------------------------------

    def _load_section(self, record: Any) -> None:  # noqa: C901, PLR0915
        ctx = self.ctx
        serving_id = legacy._serving_id(ctx, record)
        gpus = _gpus(ctx)
        refusal = legacy._arch_refusal_text(ctx, record)
        loaded = serving_id in _instances(ctx)
        shared = st.shares_base_instance(record)
        entry = self._catalog_entry(record)
        rec = entry.get("recommended") if isinstance(entry, Mapping) else None
        args: dict[str, Any] = dict((rec or {}).get("load_args") or {})

        with ui.expansion(
            "Load" if not loaded else "Reload or unload", icon="play_arrow", value=True
        ).classes("w-full"):
            if refusal:
                with ui.row().classes("items-start gap-2 no-wrap"):
                    ui.badge(st.UNSUPPORTED_ARCH_BADGE, color="negative").classes("text-xs")
                    ui.label(refusal).classes("text-xs text-negative")
            if rec:
                hint = mv.fits_hint(entry)
                ui.label(
                    f"Recommended: {rec.get('label')} · {mv.fmt_ctx(rec.get('ctx_per_slot'))} "
                    f"context · {rec.get('kv_cache_type', 'f16')} KV"
                ).classes("text-sm font-medium")
                ui.label(hint.tooltip).classes(f"text-xs text-{hint.colour}")
            else:
                ui.label(
                    "No recommendation: the planner could not place this model with the "
                    "defaults. Adjust the fields below and the fit check will say why."
                    if entry
                    else "The recommendation is still being computed; the fields below use "
                    "your defaults."
                ).classes("text-xs opacity-70")

            ui.label("GPUs").classes("text-xs font-medium mt-1")
            chosen = {int(d) for d in (args.get("devices") or [])}
            gpu_boxes: dict[int, Any] = {}
            with ui.row().classes("gap-3 flex-wrap"):
                for gpu in gpus:
                    gpu_boxes[gpu.index] = ui.checkbox(
                        f"GPU{gpu.index} {gpu.name} · "
                        f"{st.format_gib(gpu.total_bytes, precision=0)}",
                        value=gpu.index in chosen,
                    ).props("dense")
            if gpus:
                ui.label("Leave all unchecked to let the planner choose.").classes(
                    "text-xs opacity-60"
                )
            with ui.row().classes("w-full gap-3 flex-wrap items-end"):
                ctx_input = (
                    ui.number(
                        "Context (tokens per conversation)",
                        value=args.get("ctx_size") or ctx.config.models.default_ctx,
                        precision=0,
                    )
                    .props("dense outlined")
                    .classes("w-64")
                )
                kv = (
                    ui.select(
                        list(_KV_TYPES),
                        value=str(args.get("kv_cache_type") or ""),
                        label="KV cache type (K)",
                    )
                    .props("dense outlined")
                    .classes("w-40")
                )
            trained = record.meta.n_ctx_train if record.meta else 0
            quick = [
                (label, tokens) for label, tokens in _QUICK_CTX if not trained or tokens <= trained
            ]
            if quick:
                with ui.row().classes("gap-1 flex-wrap items-center"):
                    ui.label("Quick:").classes("text-xs opacity-60")
                    for label, tokens in quick:
                        ui.button(
                            label, on_click=lambda _=None, t=tokens: ctx_input.set_value(t)
                        ).props("flat dense no-caps size=sm")
            with ui.expansion("Advanced options", icon="settings").classes("w-full"):
                with ui.row().classes("w-full gap-3 flex-wrap items-end"):
                    parallel = (
                        ui.number("Parallel slots", value=args.get("parallel"), precision=0)
                        .props("dense outlined clearable")
                        .classes("w-40")
                    )
                    kv_v = (
                        ui.select(
                            list(_KV_TYPES),
                            value=str(args.get("kv_cache_type_v") or ""),
                            label="KV cache type (V)",
                        )
                        .props("dense outlined")
                        .classes("w-40")
                    )
                ui.label(
                    "Everything else (batch sizes, threads, flash-attn, split mode, reasoning, "
                    "extra flags, draft model, adapters, TTL, load priority) is in the full "
                    "settings dialog."
                ).classes("text-xs opacity-70")
                ui.button(
                    "All settings…",
                    icon="tune",
                    on_click=lambda: legacy._settings_dialog(
                        ctx, self._record() or record, _Refresher(self.changed)
                    ),
                ).props("outline dense no-caps")

            fit_box = ui.column().classes("w-full gap-1 mt-1")
            preview_cell: dict[str, Any] = {"preview": None}

            def selected_devices() -> list[int] | None:
                picked = sorted(i for i, box in gpu_boxes.items() if box.value)
                return picked or None

            def load_kwargs() -> dict[str, Any]:
                out: dict[str, Any] = {}
                if ctx_input.value:
                    out["ctx_size"] = int(ctx_input.value)
                if kv.value:
                    out["kv_cache_type"] = str(kv.value)
                if kv_v.value:
                    out["kv_cache_type_v"] = str(kv_v.value)
                if parallel.value:
                    out["parallel"] = int(parallel.value)
                devices = selected_devices()
                if devices:
                    out["devices"] = devices
                return out

            def recompute() -> None:
                fit_box.clear()
                with fit_box:
                    try:
                        preview = ctx.manager.plan_preview(record.id, **load_kwargs())
                    except Exception as exc:  # noqa: BLE001 - a bad combination is an answer
                        preview_cell["preview"] = None
                        ui.label(f"Fit check: {getattr(exc, 'message', None) or exc}").classes(
                            "text-xs text-warning"
                        )
                        return
                    preview_cell["preview"] = preview
                    verdict = st.fit_verdict(preview)
                    with ui.row().classes("items-center gap-2"):
                        ui.icon(
                            "check_circle" if verdict.fits else "error", color=verdict.colour
                        ).props('aria-hidden="true"')
                        ui.label(verdict.headline).classes(
                            f"text-{verdict.colour} font-medium text-sm"
                        ).props('role="status"')
                    own = _instances(ctx).get(serving_id)
                    footprint = dict(getattr(getattr(own, "plan", None), "per_gpu_bytes", {}) or {})
                    for bar in mv.fit_bars(preview, gpus, credit=footprint):
                        _draw_fit_bar(bar)
                    for line in verdict.detail_lines:
                        ui.label(line).classes("text-xs font-mono opacity-80")
                    for note in verdict.notes:
                        ui.label(f"note: {note}").classes("text-xs opacity-70")
                    for suggestion in verdict.suggestions:
                        ui.label(f"try: {suggestion}").classes("text-xs text-warning")

            for widget in (ctx_input, kv, kv_v, parallel, *gpu_boxes.values()):
                widget.on_value_change(lambda _: recompute())
            if ctx.manager is not None:
                recompute()

            async def do_load() -> None:
                await self._load(record, load_kwargs(), preview_cell["preview"], force=loaded)

            with ui.row().classes("w-full gap-2 mt-1"):
                load_button = ui.button(
                    "Reload with these settings" if loaded else "Load",
                    icon="restart_alt" if loaded else "play_arrow",
                    on_click=do_load,
                ).props("color=primary no-caps")
                if refusal:
                    load_button.disable()
                    load_button.tooltip(f"Cannot load: {refusal}")
                if loaded and not shared:
                    ui.button(
                        "Unload",
                        icon="stop_circle",
                        on_click=lambda: legacy._unload(
                            ctx, self._record() or record, _Refresher(self.changed)
                        ),
                    ).props("outline color=negative no-caps")
                elif loaded and shared:
                    base = self._record_of(record.base_model_id)
                    if base is not None:
                        ui.button(
                            f"Unload {mv.friendly_name(base)}",
                            icon="stop_circle",
                            on_click=lambda: legacy._unload(ctx, base, _Refresher(self.changed)),
                        ).props("outline color=negative no-caps").tooltip(
                            "This persona runs on its base model's instance; unloading stops "
                            "it for every persona built on that base."
                        )

    def _record_of(self, model_id: str | None) -> Any | None:
        if not model_id or self.ctx.registry is None:
            return None
        try:
            return self.ctx.registry.resolve(model_id)
        except Exception:  # noqa: BLE001
            return None

    def _catalog_entry(self, record: Any) -> Mapping[str, Any] | None:
        """The catalog row for ``record``: from the table's cache, else the manager's cache."""
        manager = self.ctx.manager
        if manager is None:
            return None
        try:
            payload = manager.catalog(model=record.id, compact=True)
        except Exception as exc:  # noqa: BLE001 - the recommendation is a nicety
            log.debug("catalog entry unavailable", model_id=record.id, error=str(exc))
            return None
        models = payload.get("models") or [] if isinstance(payload, Mapping) else []
        return models[0] if models else None

    async def _load(
        self, record: Any, kwargs: dict[str, Any], preview: Mapping[str, Any] | None, *, force: bool
    ) -> None:
        from studioforge.gui.tabs.dashboard import _Page

        ctx = self.ctx
        page = _Page.capture()
        evict = [str(m) for m in ((preview or {}).get("evict_model_ids") or [])]
        if evict and not await legacy._confirm(
            f"Loading {record.id} needs {len(evict)} model(s) unloaded: {', '.join(evict)}. "
            "Continue?"
        ):
            return
        with page, single_flight(f"models.load:{record.id}", f"load of {record.id}") as claimed:
            if not claimed:
                return
            self.on_change()  # the row starts showing "loading" now
            started = time.time()
            try:
                instance = await run_load(ctx, record.id, kwargs, force=force)
            except Exception as exc:  # noqa: BLE001 - the refusal is the useful part
                page.error(exc, what="load")
                return
            finally:
                self.changed()
            plan = instance.plan
            where = ", ".join(f"GPU{d}" for d in (plan.devices if plan else []))
            page.notify(
                f"{record.id} ready{f' on {where}' if where else ''} in "
                f"{st.format_latency(time.time() - started)}",
                type="positive",
            )

    # -- manage ------------------------------------------------------------------

    def _manage_section(self, record: Any) -> None:
        ctx = self.ctx
        may_change = viewer_may_change_box(ctx)
        table = _Refresher(self.changed)
        with ui.expansion("Manage", icon="build").classes("w-full"):  # noqa: SIM117
            with ui.row().classes("gap-2 flex-wrap"):
                pin = ui.button(
                    "Unpin" if record.settings.pinned else "Pin",
                    icon="push_pin",
                    on_click=lambda: legacy._toggle_pin(ctx, self._record() or record, table),
                ).props(
                    "outline dense no-caps" + (" color=accent" if record.settings.pinned else "")
                )
                admin_control(
                    pin,
                    may_change=may_change,
                    what="pin",
                    tooltip=st.pin_tooltip(
                        record.settings.pinned, auto_load_pinned=ctx.config.models.auto_load_pinned
                    ),
                )
                ui.button(
                    "Test",
                    icon="science",
                    on_click=lambda: legacy._test(ctx, self._record() or record),
                ).props("outline dense no-caps").tooltip(
                    "One short completion, with latency and tok/s. Loads and unloads the model "
                    "if it is cold."
                )
                ui.button(
                    "Benchmark",
                    icon="speed",
                    on_click=lambda: legacy._benchmark(ctx, self._record() or record),
                ).props("outline dense no-caps").tooltip(
                    "Compare load time, prompt and generation speed across GPU placements"
                )
                delete = ui.button(
                    "Delete…",
                    icon="delete",
                    on_click=lambda: self._delete(record),
                ).props("outline dense no-caps color=negative")
                admin_control(delete, may_change=may_change, what="delete")

    def _delete(self, record: Any) -> None:
        def after() -> None:
            self.dialog.close()
            self.model_id = None
            self.on_change()

        legacy._delete_dialog(self.ctx, self._record() or record, _Refresher(after))

    # -- sampler defaults --------------------------------------------------------

    def _sampler_section(self, record: Any) -> None:
        ctx = self.ctx
        form = st.form_from_settings(record.settings)
        with ui.expansion("Sampler defaults", icon="tune").classes("w-full"):
            ui.label(
                "Used when a request does not set the value itself. Blank means the model's own "
                "recommendation."
            ).classes("text-xs opacity-70")
            widgets: dict[str, Any] = {}
            with ui.row().classes("w-full gap-3 flex-wrap"):
                for key, label, precision in (
                    ("temperature", "temperature", None),
                    ("top_p", "top-p", None),
                    ("top_k", "top-k", 0),
                    ("min_p", "min-p", None),
                    ("repeat_penalty", "repeat-penalty", None),
                ):
                    widgets[key] = legacy._num(label, form[key], precision=precision)

            async def save() -> None:
                fresh = self._record() or record
                data = st.form_from_settings(fresh.settings)
                data.update({key: widget.value for key, widget in widgets.items()})
                try:
                    settings = st.settings_from_form(data)
                except Exception as exc:  # noqa: BLE001 - the validation message is the point
                    notify_error(exc, what="sampler defaults")
                    return
                try:
                    require_local_admin(ctx, "save settings")
                    await run_blocking(ctx.registry.save_settings, fresh.id, settings)
                except Exception as exc:  # noqa: BLE001
                    notify_error(exc, what="save settings")
                    return
                ui.notify(f"sampler defaults saved for {fresh.id}", type="positive")
                self.on_change()

            button = ui.button("Save defaults", icon="save", on_click=save).props(
                "outline dense no-caps"
            )
            admin_control(button, may_change=viewer_may_change_box(ctx), what="save settings")

    # -- presets / virtual models ------------------------------------------------

    def _presets_section(self, record: Any) -> None:
        ctx = self.ctx
        records = _records(ctx)
        built_on = [r for r in records if r.is_virtual and r.base_model_id == record.id]
        table = _Refresher(self.changed)
        title = f"Virtual models & presets ({len(built_on)})" if not record.is_virtual else "Preset"
        with ui.expansion(title, icon="extension").classes("w-full"):
            if record.is_virtual:
                ui.label(
                    f"A persona on {record.base_model_id or st.UNKNOWN}. "
                    + "".join(st.preset_summary_lines(record.preset))
                ).classes("text-xs opacity-80")
                ui.button(
                    "Edit this virtual model",
                    icon="edit",
                    on_click=lambda: legacy._persona_dialog(
                        ctx, table, record=self._record() or record
                    ),
                ).props("outline dense no-caps")
                return
            if not built_on:
                ui.label(
                    "None yet. A virtual model is this model under its own name, with a system "
                    "prompt and sampler defaults and/or LoRA adapters."
                ).classes("text-xs opacity-70")
            for virtual in built_on:
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    ui.label(virtual.id).classes("text-xs font-mono grow truncate")
                    ui.button(
                        icon="edit",
                        on_click=lambda v=virtual: legacy._persona_dialog(ctx, table, record=v),
                    ).props('flat dense aria-label="Edit this virtual model"').tooltip("Edit")
            ui.button(
                "New virtual model / preset",
                icon="add",
                on_click=lambda: legacy._persona_dialog(ctx, table, base_id=record.id),
            ).props("outline dense no-caps")

    # -- aliases -----------------------------------------------------------------

    def _aliases_section(self, record: Any) -> None:
        names = aliases_of(self.ctx, record)
        with ui.expansion(f"Aliases ({len(names)})", icon="label").classes("w-full"):
            ui.label(
                "Clients can ask for the model by any of these names; they all reach the same file."
            ).classes("text-xs opacity-70")
            for name in [record.id, *names]:
                with ui.row().classes("items-center gap-1 no-wrap"):
                    ui.label(name).classes("text-xs font-mono")
                    ui.button(
                        icon="content_copy",
                        on_click=lambda n=name: legacy._copy_text(n, what=f"copied '{n}'"),
                    ).props('flat dense size=sm aria-label="Copy this name"')


def _draw_fit_bar(bar: mv.FitBar) -> None:
    """One GPU: used (grey) + this load (blue, red when it does not fit)."""
    with ui.column().classes("w-full gap-0"):
        with ui.row().classes("w-full items-center justify-between no-wrap"):
            ui.label(f"GPU{bar.index} · {bar.name}").classes(
                "text-xs " + ("font-medium" if bar.targeted else "opacity-70")
            )
            ui.label(bar.text).classes("text-xs font-mono opacity-80")
        used = round(bar.used_fraction * 100, 1)
        add = round(bar.add_fraction * 100, 1)
        host = ui.element("div").classes("sfp-bar w-full" + (" sfp-over" if bar.over else ""))
        host.props(
            f'role="img" aria-label="GPU{bar.index}: {used:.0f} percent used'
            + (f", this load adds {add:.0f} percent" if bar.add_bytes else "")
            + '"'
        )
        with host:
            ui.element("div").classes("sfp-used").style(f"width:{used}%")
            if bar.add_bytes:
                ui.element("div").classes("sfp-add").style(f"left:{used}%;width:{max(add, 0.8)}%")


# ---------------------------------------------------------------------------
# Entry point used by the shell
# ---------------------------------------------------------------------------


def render(
    ctx: GuiContext,
    location: nav.Location,
    go: Callable[[nav.Location], None],
    params: Mapping[str, Any] | None = None,
) -> ModelsPage:
    page = ModelsPage(ctx, go)
    page.build(location, params)
    return page


__all__ = ["ModelDrawer", "ModelsPage", "aliases_of", "error_of", "render", "run_load"]
