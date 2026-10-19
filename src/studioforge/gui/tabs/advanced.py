"""Advanced: everything that is not Status, Models or Test Chat.

A left sub-navigation (:data:`studioforge.gui.nav.ADVANCED_PAGES`) over the pages that used
to be top-level sections. Each page is built the first time it is opened, so a closed page
costs no timers. Nothing here is new behaviour: a page is the old section's cards, composed
exactly as before, plus the controls that moved here from the Overview (restart, unload
everything) and the two small pages that had no home (Watchdog, Peers).

Settings are drawn from the settings schema by the same ``_Fields`` helper as before.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from nicegui import ui

from studioforge import __version__
from studioforge.gui import nav
from studioforge.gui import state as st
from studioforge.gui.tabs import (
    GuiContext,
    admin_control,
    panel_guard,
    viewer_may_change_box,
)
from studioforge.gui.tabs.widgets import copy_button

Go = Callable[[nav.Location], None]

_ADV_CSS = """
.sf-adv{flex-wrap:nowrap;gap:.5rem;align-items:flex-start;}
.sf-adv-nav{min-width:13.5rem;max-width:13.5rem;}
.sf-adv-nav .q-tab{justify-content:flex-start;}
.sf-adv-pick{display:none!important;width:100%;}
@media (max-width:820px){
  .sf-adv{flex-wrap:wrap;}
  .sf-adv-navwrap{display:none!important;}
  .sf-adv-pick{display:inline-flex!important;}
}
"""


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def _column() -> Any:
    return ui.column().classes("w-full gap-4 p-2")


def serving_page(ctx: GuiContext) -> None:
    from studioforge.gui.tabs import serving

    serving.render(ctx)


def placement_page(ctx: GuiContext) -> None:
    from studioforge.gui.tabs import dashboard, setup

    with _column():
        setup.placement_cards(ctx)
        dashboard._vram_holders_panel(ctx)
        dashboard.placement_panels(ctx)


def accounts_page(ctx: GuiContext) -> None:
    from studioforge.gui.tabs import users_keys

    users_keys.render(ctx)


def engines_page(ctx: GuiContext) -> None:
    from studioforge.gui.tabs import dashboard, server, setup

    async def nothing() -> None:
        return None

    with _column():
        with ui.card().classes("w-full"):
            ui.label("Restart engines").classes("text-lg font-medium")
            ui.label(st.RESTART_ENGINES_HELP).classes("text-xs opacity-70")
            admin_control(
                ui.button(
                    "Restart engines",
                    icon="autorenew",
                    on_click=lambda: dashboard._restart_engines(ctx, nothing),
                ).props("outline dense no-caps"),
                may_change=viewer_may_change_box(ctx),
                what="restart engines",
                tooltip=st.RESTART_ENGINES_HELP,
            )
        setup.engine_cards(ctx)
        server.engine_panels(ctx)


def watchdog_page(ctx: GuiContext) -> None:
    """The recovery sidecar's settings, plus the controls that take the server down or up."""
    from studioforge.gui.tabs import dashboard, setup

    with _column():
        with ui.card().classes("w-full"):
            ui.label("Watchdog").classes("text-lg font-medium")
            ui.label(
                "A small separate process that restarts the server if it stops answering and "
                "can kill stuck engines. It keeps its own port. A change takes effect after a "
                "restart."
            ).classes("text-xs opacity-70")
            with panel_guard("Watchdog settings"):
                fields = setup._Fields(ctx, setup._payload(ctx))
                for key, label in (
                    ("watchdog.enabled", "Run the watchdog"),
                    ("watchdog.host", "Watchdog bind address"),
                    ("watchdog.port", "Watchdog port"),
                    ("watchdog.poll_interval_s", "Check the server every (seconds)"),
                    ("watchdog.health_timeout_s", "Health check timeout (seconds)"),
                    (
                        "watchdog.wedged_after_failures",
                        "Treat the server as stuck after (failed checks)",
                    ),
                ):
                    fields.row(key, label=label)
                setup._save_button(ctx, fields, None)
        with ui.card().classes("w-full"):
            ui.label("Restart and unload").classes("text-lg font-medium")
            ui.label(
                "The controls for when something is stuck. Unload all frees every GPU; Restart "
                "server takes the gateway down for a few seconds and every client sees a "
                "connection error."
            ).classes("text-xs opacity-70")
            banner = ui.label("").classes("text-sm text-warning")

            async def nothing() -> None:
                return None

            def unload_all() -> None:
                loaded = [i.model_id for i in ctx.supervisor.list()] if ctx.supervisor else []
                dashboard._unload_all_dialog(ctx, loaded, nothing)

            with ui.row().classes("gap-2"):
                ui.button("Unload all models", icon="layers_clear", on_click=unload_all).props(
                    "outline dense no-caps color=negative"
                ).tooltip("Unload every resident model and free all VRAM.")
                ui.button(
                    "Restart server",
                    icon="power_settings_new",
                    on_click=lambda: dashboard._restart_server_dialog(ctx, banner),
                ).props("outline dense no-caps color=warning").tooltip(st.RESTART_SERVER_WARNING)


def catalog_page(ctx: GuiContext) -> None:
    from studioforge.gui.tabs import setup

    with _column():
        ui.label(
            "Where the model files live, what every load starts from, and how downloads reach "
            "Hugging Face. Adding a model is on the Models tab."
        ).classes("text-xs opacity-70")
        setup.library_cards(ctx)


def integrations_page(ctx: GuiContext) -> None:
    from studioforge.gui.tabs import integrations

    integrations.render(ctx)


def peers_page(ctx: GuiContext) -> None:
    with _column(), ui.card().classes("w-full"):
        ui.label("Peers").classes("text-lg font-medium")
        ui.label(
            "Peers and the placement decision log arrive with multi-host support (milestone M3)."
        ).classes("text-sm opacity-80")
        ui.label(
            "When it does, other StudioForge machines on your network are listed here and a "
            "model can be placed on any of them. Nothing is configured or sent today."
        ).classes("text-xs opacity-70")


def logs_page(ctx: GuiContext) -> None:
    from studioforge.gui.tabs import logs

    logs.render(ctx)
    with ui.column().classes("w-full gap-2 p-2"):
        logs.metrics_panel(ctx)


def config_page(ctx: GuiContext, go: Go | None = None) -> None:
    from studioforge.gui.tabs import server, setup

    with _column():
        if go is not None:
            with ui.row().classes("items-center gap-2"):
                ui.label("Check the result of a change:").classes("text-xs opacity-70")
                ui.button(
                    "Run checks",
                    icon="troubleshoot",
                    on_click=lambda: go(nav.Location(nav.ADVANCED, page="diagnostics")),
                ).props("flat dense no-caps")
        setup.advanced_cards(ctx)
        server.config_panels(ctx)
        ui.label(
            "Raw engine flags for one model are set in its settings: Models, open the model, "
            "Advanced options, All settings, then Expert."
        ).classes("text-xs opacity-70")


def diagnostics_page(ctx: GuiContext) -> None:
    from studioforge.gui.tabs import diagnostics, setup

    with _column(), panel_guard("The readiness checklist"):
        setup.checklist_cards(ctx)
    diagnostics.render(ctx)


def about_page(ctx: GuiContext) -> None:
    from studioforge.gui.tabs import status, updates

    with _column(), ui.card().classes("w-full"):
        ui.label("About StudioForge").classes("text-lg font-medium")
        try:
            from studioforge.build import build_id

            build = build_id()
        except Exception:  # noqa: BLE001
            build = st.UNKNOWN
        with ui.row().classes("w-full gap-x-8 gap-y-1 flex-wrap"):
            for label, value in (
                ("Version", f"v{__version__}"),
                ("Build", build),
                ("Uptime", status.uptime_text(ctx)),
                ("Config file", str(ctx.config.config_path)),
            ):
                with ui.column().classes("gap-0"):
                    ui.label(label).classes("text-xs opacity-60")
                    with ui.row().classes("items-center gap-1 no-wrap"):
                        ui.label(value).classes("text-sm font-mono break-all")
                        if label != "Uptime":
                            copy_button(value, label=f"Copy the {label.lower()}")
    updates.render(ctx)


#: Builders by slug. A page that needs to move the viewer elsewhere takes ``go``.
_BUILDERS: dict[str, Callable[..., None]] = {
    "serving": serving_page,
    "placement": placement_page,
    "accounts": accounts_page,
    "engines": engines_page,
    "watchdog": watchdog_page,
    "catalog": catalog_page,
    "integrations": integrations_page,
    "peers": peers_page,
    "logs": logs_page,
    "config": config_page,
    "diagnostics": diagnostics_page,
    "about": about_page,
}

#: The slugs whose builder takes ``go`` as a second argument.
_TAKES_GO = frozenset({"config"})


def builder_slugs() -> tuple[str, ...]:
    return tuple(_BUILDERS)


# ---------------------------------------------------------------------------
# The hub
# ---------------------------------------------------------------------------


class AdvancedHub:
    """The sub-navigation and its lazily built pages for one browser page."""

    def __init__(self, ctx: GuiContext, go: Go) -> None:
        self.ctx = ctx
        self.go = go
        self.boxes: dict[str, Any] = {}
        self.built: set[str] = set()
        self.tabs: Any = None
        self.pick: Any = None
        self.ready = False

    def build(self, location: nav.Location) -> None:
        ui.add_css(_ADV_CSS)
        initial = location.page or nav.DEFAULT_ADVANCED_PAGE
        with ui.column().classes("w-full gap-1 p-0"):
            # The same choice as the left navigation, as a menu: twelve stacked tabs would
            # push the page a screen down on a phone, so CSS swaps one for the other.
            self.pick = (
                ui.select(
                    {page.slug: page.label for page in nav.ADVANCED_PAGES},
                    value=initial,
                    label="Advanced page",
                    on_change=lambda e: self.select(str(e.value)),
                )
                .props('dense outlined options-dense aria-label="Advanced page"')
                .classes("sf-adv-pick")
            )
        with ui.row().classes("w-full sf-adv"):
            # The wrapper is what the phone CSS hides: Quasar's own tab styles win a fight
            # over the tabs element's display.
            with ui.element("div").classes("sf-adv-navwrap"):
                self.tabs = (
                    ui.tabs(value=initial, on_change=lambda e: self._on_tab(str(e.value)))
                    .props('vertical inline-label aria-label="Advanced pages"')
                    .classes("sf-adv-nav")
                )
                with self.tabs:
                    for page in nav.ADVANCED_PAGES:
                        ui.tab(page.slug, label=page.label, icon=page.icon).tooltip(page.summary)
            with ui.tab_panels(self.tabs, value=initial).props("vertical").classes("grow min-w-0"):
                for page in nav.ADVANCED_PAGES:
                    with ui.tab_panel(page.slug).classes("p-0"):
                        self.boxes[page.slug] = ui.column().classes("w-full gap-0")
        self.ensure(initial)
        self.ready = True

    def _on_tab(self, slug: str) -> None:
        """A left-navigation click: keep the phone menu in step, then show the page."""
        if self.pick is not None and self.pick.value != slug:
            self.pick.set_value(slug)
        self.ensure(slug)

    def ensure(self, slug: str) -> None:
        """Build ``slug`` the first time it is shown; then keep the address bar in step."""
        if slug in self.built or slug not in self.boxes:
            self._sync_url(slug)
            return
        self.built.add(slug)
        with self.boxes[slug], panel_guard(f"The {_label(slug)} page"):
            builder = _BUILDERS[slug]
            if slug in _TAKES_GO:
                builder(self.ctx, self.go)
            else:
                builder(self.ctx)
        self._sync_url(slug)

    def select(self, slug: str) -> None:
        if slug not in self.boxes:
            return
        if self.pick is not None and self.pick.value != slug:
            self.pick.set_value(slug)  # no-op loop: the handler re-enters with the same slug
        if self.tabs.value != slug:
            self.tabs.set_value(slug)  # the change handler builds it
        else:
            self.ensure(slug)

    def sync_url(self) -> None:
        """Put the page being shown in the address bar (the shell calls this when Advanced is
        opened, including the first time, when the hub was only just built)."""
        self._sync_url(str(self.tabs.value), force=True)

    def apply(self, location: nav.Location) -> None:
        self.select(location.page or nav.DEFAULT_ADVANCED_PAGE)

    def _sync_url(self, slug: str, *, force: bool = False) -> None:
        if not self.ready and not force:
            return  # still building: the address already says where we are
        url = nav.canonical_url(nav.Location(nav.ADVANCED, page=slug))
        try:
            ui.run_javascript(f"history.replaceState(null, '', {json.dumps(url)})")
        except Exception:  # noqa: BLE001 - no client (a test), nothing to rewrite
            return


def _label(slug: str) -> str:
    page = nav.advanced_page(slug)
    return page.label if page else slug


def render(ctx: GuiContext, location: nav.Location, go: Go) -> AdvancedHub:
    hub = AdvancedHub(ctx, go)
    hub.build(location)
    return hub


__all__ = ["AdvancedHub", "builder_slugs", "render"]
