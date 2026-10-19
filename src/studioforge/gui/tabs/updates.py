"""Updates: which version is running, is there a newer one, install it or go back.

A thin page over the same handlers ``GET /api/update`` and ``POST
/api/update/install|rollback`` use, called in-process. Installing and rolling
back restart the server, so each sits behind a confirmation and the usual
local-admin rule, and each leaves an audit line.

The update source is the ``update.repo`` setting (``owner/name`` on GitHub). With
it unset there is nothing to ask, and the page says so instead of failing.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from studioforge import __version__
from studioforge.gui.tabs import (
    GuiContext,
    admin_control,
    api_request,
    audit_panel,
    busy,
    panel_guard,
    remote_viewer_banner,
    require_local_admin,
    viewer_may_change_box,
)
from studioforge.gui.tabs.dashboard import _Page
from studioforge.gui.tabs.widgets import confirm, fmt_ts
from studioforge.logging import get_logger

log = get_logger(__name__)

NOT_CONFIGURED = (
    "No update source is set, so StudioForge does not look for new versions. Set update.repo "
    "(the GitHub repository that publishes releases, as owner/name) under Advanced > "
    "Config to turn this on."
)


def local_status(ctx: GuiContext) -> dict[str, Any]:
    """The status without asking the network (what ``GET /api/update`` returns by default)."""
    updater = getattr(ctx.api_state, "updater", None)
    if updater is None:
        return {
            "current_version": __version__,
            "configured": False,
            "note": "Updates are not available on this server build.",
            "releases_installed": [],
        }
    return dict(updater.status_sync().to_dict())


async def check_for_updates(ctx: GuiContext) -> dict[str, Any]:
    """Ask the update source (``GET /api/update?check=true``)."""
    from studioforge.api.mgmt_routes import update_status

    return dict(await update_status(api_request(ctx), check=True))


async def install_update(ctx: GuiContext, tag: str | None) -> dict[str, Any]:
    require_local_admin(ctx, "install an update")
    from studioforge.api.mgmt_routes import update_install

    audit_panel(ctx, "panel.update.install", tag or "latest", result="started")
    result = dict(await update_install(api_request(ctx), tag=tag, confirm=True, restart=True))
    audit_panel(ctx, "panel.update.install", tag or "latest")
    return result


async def roll_back(ctx: GuiContext) -> dict[str, Any]:
    require_local_admin(ctx, "roll back")
    from studioforge.api.mgmt_routes import update_rollback

    audit_panel(ctx, "panel.update.rollback", "", result="started")
    result = dict(await update_rollback(api_request(ctx), confirm=True))
    audit_panel(ctx, "panel.update.rollback")
    return result


def summary_lines(status: dict[str, Any]) -> list[str]:
    """Plain sentences for a status dict."""
    lines = [f"Running version {status.get('current_version', __version__)}."]
    if status.get("current_release"):
        lines.append(f"Active release: {status['current_release']}.")
    if not status.get("configured", True):
        lines.append(str(status.get("note") or NOT_CONFIGURED))
        return lines
    if status.get("error"):
        lines.append(f"The last check failed: {status['error']}")
    elif status.get("checked_at"):
        when = fmt_ts(status["checked_at"])
        if status.get("update_available"):
            lines.append(f"Version {status.get('latest_version')} is available (checked {when}).")
        else:
            lines.append(f"You are up to date (checked {when}).")
    else:
        lines.append("Not checked yet in this session.")
    return lines


def render(ctx: GuiContext) -> None:
    holder = ui.column().classes("w-full gap-4 p-2")
    state: dict[str, Any] = {"status": local_status(ctx)}

    def draw() -> None:
        holder.clear()
        status = state["status"]
        configured = bool(status.get("configured", True))
        installed = list(status.get("releases_installed") or [])
        may_change = viewer_may_change_box(ctx)
        with holder, panel_guard("Updates"):
            with ui.card().classes("w-full"):
                ui.label("Updates").classes("text-lg font-medium")
                if not may_change:
                    remote_viewer_banner()
                for line in summary_lines(status):
                    ui.label(line).classes("text-sm")
                if not configured:
                    ui.link("Open Advanced > Config", "/?section=Advanced&page=config").classes(
                        "text-sm"
                    )
                with ui.row().classes("items-center gap-2"):
                    ui.button("Check for updates", icon="system_update_alt", on_click=check).props(
                        "outline dense"
                    ).set_enabled(configured)
                    if configured and status.get("update_available"):
                        admin_control(
                            ui.button(
                                f"Install {status.get('latest_version') or 'the update'}",
                                icon="download",
                                on_click=lambda: install(status.get("latest_tag")),
                            ).props("color=primary dense"),
                            may_change=may_change,
                            what="Install the update",
                        )

            with ui.card().classes("w-full"):
                ui.label("Installed releases").classes("text-lg font-medium")
                if installed:
                    current = status.get("current_release")
                    for name in installed:
                        marker = " (running)" if name == current else ""
                        ui.label(f"{name}{marker}").classes("text-sm font-mono")
                else:
                    ui.label(
                        "No separate releases are installed: this copy was not installed by the "
                        "updater."
                    ).classes("text-sm opacity-70")
                previous = status.get("previous_release")
                if previous:
                    ui.label(f"Previous release: {previous}").classes("text-xs opacity-70")
                rollback = admin_control(
                    ui.button(
                        "Roll back to the previous release", icon="undo", on_click=back
                    ).props("outline dense color=warning"),
                    may_change=may_change,
                    what="Roll back",
                )
                if not previous:
                    rollback.disable()
                ui.label(
                    "Installing or rolling back restarts the server. Models unload and clients "
                    "reconnect when it is back."
                ).classes("text-xs opacity-70")

    async def check() -> None:
        page = _Page.capture()
        with page, busy(message="Checking for updates…"):
            try:
                state["status"] = await check_for_updates(ctx)
            except Exception as exc:  # noqa: BLE001
                page.error(exc, what="check for updates")
                return
        draw()

    async def install(tag: str | None) -> None:
        page = _Page.capture()
        ok = await confirm(
            "Install this update?",
            "StudioForge downloads the release, restarts, and goes back to the old version by "
            "itself if the new one does not start. Models unload during the restart.",
            action="Install and restart",
            colour="primary",
        )
        if not ok:
            return
        with page, busy(message="Installing the update…"):
            try:
                result = await install_update(ctx, tag)
            except Exception as exc:  # noqa: BLE001
                page.error(exc, what="install the update")
                return
        page.notify(
            str(result.get("message") or "Update installed. The server is restarting."),
            type="positive",
            multi_line=True,
        )

    async def back() -> None:
        page = _Page.capture()
        ok = await confirm(
            "Go back to the previous release?",
            "The server restarts on the release it was running before the last update.",
            action="Roll back and restart",
        )
        if not ok:
            return
        with page, busy(message="Rolling back…"):
            try:
                result = await roll_back(ctx)
            except Exception as exc:  # noqa: BLE001
                page.error(exc, what="roll back")
                return
        page.notify(
            str(result.get("message") or "Rolled back. The server is restarting."),
            type="positive",
            multi_line=True,
        )

    draw()
