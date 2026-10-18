"""Overview: what the server is doing now, how to reach it, and what is left to set up.

Composition only. The GPU, VRAM-holder and loaded-model panels are the
Dashboard's; the checklist is the Setup tab's. What is new here is the Base URL
card (the one address every client needs) and the "Finish setup" banner that
sends a fresh install to the first-run wizard.
"""

from __future__ import annotations

from nicegui import ui

from studioforge.gui.tabs import GuiContext, panel_guard
from studioforge.gui.tabs.widgets import copy_button, is_remote_without_admin
from studioforge.logging import get_logger

log = get_logger(__name__)

#: Where the first-run wizard lives (the same constant the app registers).
SETUP_PATH = "/setup"


def needs_setup(ctx: GuiContext) -> bool:
    """Whether this install still has first-run work to do.

    "Fresh" is measured, not remembered: no model library, no models indexed,
    or no engine installed. A missing owner account is *not* a reason on its own
    -- a loopback-only install needs none -- but a network bind with no owner
    account or admin key is (the server refuses to start in that state).

    Any failure answers "no": a banner is not worth a broken page.
    """
    try:
        if not ctx.config.models.dir:
            return True
        if ctx.registry is not None and not ctx.registry.all():
            return True
        if ctx.engine_manager is not None and ctx.engine_manager.active() is None:
            return True
        return is_remote_without_admin(ctx)
    except Exception:  # noqa: BLE001
        return False


def base_urls(ctx: GuiContext) -> list[dict[str, str]]:
    """Every address a client can use, as ``{label, url}`` where ``url`` ends in ``/v1``.

    The scheme follows ``server.tls.mode``; a specific bind address is the only
    one listed (the server answers nowhere else).
    """
    from studioforge.core.netinfo import reachable_urls
    from studioforge.tlsutil import scheme_for

    scheme = scheme_for(ctx.config)
    try:
        found = reachable_urls(ctx.config.server.port, host=ctx.config.server.host, scheme=scheme)
    except Exception as exc:  # noqa: BLE001 - interface enumeration can fail
        log.debug("could not list reachable addresses", error=str(exc))
        found = []
    rows = [
        {"label": str(row.get("label") or row.get("kind") or ""), "url": f"{row['url']}/v1"}
        for row in found
    ]
    return rows or [
        {"label": "This machine", "url": f"{scheme}://127.0.0.1:{ctx.config.server.port}/v1"}
    ]


def primary_base_url(ctx: GuiContext) -> str:
    """The one base URL to hand out: the first (best) reachable address, ``.../v1``."""
    return base_urls(ctx)[0]["url"]


def server_root(ctx: GuiContext) -> str:
    """``scheme://host:port`` of the best address, without ``/v1``."""
    url = primary_base_url(ctx)
    return url[: -len("/v1")] if url.endswith("/v1") else url


def _setup_banner(ctx: GuiContext) -> None:
    if not needs_setup(ctx):
        return
    with (
        ui.card().classes("w-full").props('role="status"'),
        ui.row().classes("w-full items-center gap-3 no-wrap"),
    ):
        ui.icon("rocket_launch", color="warning", size="1.8rem").props('aria-hidden="true"')
        with ui.column().classes("gap-0 grow"):
            ui.label("Finish setup").classes("text-lg font-medium")
            ui.label(
                "StudioForge is not ready to answer yet. The setup guide takes you through "
                "the engine, a model library and a first model in a few minutes."
            ).classes("text-sm opacity-80")
        ui.button(
            "Open the setup guide",
            icon="arrow_forward",
            on_click=lambda: ui.navigate.to(SETUP_PATH),
        ).props("color=primary")


def base_url_card(ctx: GuiContext) -> None:
    """The address clients put in their "OpenAI base URL" box, with a copy button."""
    with ui.card().classes("w-full"):
        ui.label("Base URL").classes("text-lg font-medium")
        ui.label(
            "Paste this into any program that asks for an OpenAI-compatible server address."
        ).classes("text-xs opacity-70")
        rows = base_urls(ctx)
        for row in rows:
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                ui.label(row["url"]).classes("font-mono text-sm")
                copy_button(row["url"], label=f"Copy {row['url']}", what="Copied the base URL")
                if row["label"]:
                    ui.label(row["label"]).classes("text-xs opacity-60")


def render(ctx: GuiContext) -> None:
    from studioforge.gui.tabs import dashboard, setup

    with ui.column().classes("w-full gap-4 p-2"):
        with panel_guard("The setup banner"):
            _setup_banner(ctx)
        with panel_guard("The Base URL card"):
            base_url_card(ctx)
        dashboard.overview_panels(ctx)
        setup.checklist_cards(ctx)


__all__ = [
    "SETUP_PATH",
    "base_url_card",
    "base_urls",
    "needs_setup",
    "primary_base_url",
    "render",
    "server_root",
]
