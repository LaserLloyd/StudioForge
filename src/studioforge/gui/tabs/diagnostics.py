"""Diagnostics: is this install healthy, and a file to send when it is not.

"Run checks" is the doctor (:mod:`studioforge.core.doctor`): ports, GPUs, the
engine, disk space, file permissions, HTTPS, database migrations and the
sign-in posture, each as one finding with a plain message and, for a warning or
a failure, what to do about it. "Download support bundle" zips the redacted
configuration, the doctor report, versions and log tails for a bug report; the
bundle builder removes credentials in four layers, and the page says so.

Both are admin actions (they name paths and ports), so they follow the same
local-admin rule as the rest of the box-changing controls, and the download
leaves an audit line.
"""

from __future__ import annotations

import time
from typing import Any

from nicegui import ui

from studioforge.core.doctor import SEVERITIES, run_doctor
from studioforge.gui.tabs import (
    GuiContext,
    admin_control,
    audit_panel,
    busy,
    panel_guard,
    remote_viewer_banner,
    require_local_admin,
    run_blocking,
    viewer_may_change_box,
)
from studioforge.gui.tabs.dashboard import _Page
from studioforge.gui.tabs.widgets import status_row
from studioforge.logging import get_logger

log = get_logger(__name__)

#: severity -> (icon, colour)
SEVERITY_STYLE: dict[str, tuple[str, str]] = {
    "ok": ("check_circle", "positive"),
    "info": ("info", "info"),
    "warn": ("warning", "warning"),
    "fail": ("error", "negative"),
}

BUNDLE_NOTE = (
    "The bundle holds the settings with every key, token, PIN and password removed, the check "
    "results, version numbers, your GPU list and the tail of the logs. No model files and no "
    "chat content. Read it before you share it."
)


def summary_text(report: dict[str, Any]) -> str:
    """One line: how many of each, and the verdict."""
    counts = report.get("counts") or {}
    parts = [f"{counts.get(s, 0)} {label}" for s, label in _LABELS if counts.get(s, 0)]
    verdict = {
        "ok": "Everything checked is fine.",
        "info": "Everything checked is fine; some notes below.",
        "warn": "Some things deserve a look.",
        "fail": "Something is broken. Fix the items marked as failed first.",
    }.get(str(report.get("worst", "ok")), "")
    return (", ".join(parts) or "No findings") + ". " + verdict


_LABELS = (("ok", "ok"), ("info", "note(s)"), ("warn", "warning(s)"), ("fail", "failed"))


def ordered_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Worst first, then in the order the checks ran."""
    rank = {s: i for i, s in enumerate(SEVERITIES)}
    rows = list(report.get("findings") or [])
    return sorted(rows, key=lambda r: -rank.get(str(r.get("severity")), 0))


async def run_checks(ctx: GuiContext) -> dict[str, Any]:
    """Run the doctor off the event loop and return its ``to_dict()``."""
    require_local_admin(ctx, "run the diagnostic checks")
    report = await run_blocking(
        run_doctor,
        ctx.config,
        probe=ctx.probe,
        engine_manager=ctx.engine_manager,
        accounts=getattr(ctx.api_state, "accounts", None),
    )
    return dict(report.to_dict())


async def make_support_bundle(ctx: GuiContext) -> tuple[str, bytes]:
    """``(filename, zip bytes)``: a fresh doctor run plus the redacted bundle, off the loop."""
    require_local_admin(ctx, "build the support bundle")
    from studioforge.core.support_bundle import build_bundle

    def build() -> bytes:
        report = run_doctor(
            ctx.config,
            probe=ctx.probe,
            engine_manager=ctx.engine_manager,
            accounts=getattr(ctx.api_state, "accounts", None),
        ).to_dict()
        return build_bundle(ctx.config, doctor=report, probe=ctx.probe)

    payload = await run_blocking(build)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    audit_panel(ctx, "panel.support_bundle", f"{len(payload)} bytes")
    return f"studioforge-support-{stamp}.zip", payload


def render(ctx: GuiContext) -> None:
    from studioforge.gui.tabs import server

    may_change = viewer_may_change_box(ctx)
    with ui.column().classes("w-full gap-4 p-2"):
        with panel_guard("The checks"), ui.card().classes("w-full"):
            ui.label("Checks").classes("text-lg font-medium")
            ui.label(
                "Looks at the ports, GPUs, engine, disk space, file permissions, HTTPS, database "
                "and sign-in setup, and tells you what to fix."
            ).classes("text-xs opacity-70")
            if not may_change:
                remote_viewer_banner()
            summary = ui.label("").classes("text-sm")
            results = ui.column().classes("w-full gap-2")

            async def run() -> None:
                page = _Page.capture()
                with page, busy(message="Running the checks…"):
                    try:
                        report = await run_checks(ctx)
                    except Exception as exc:  # noqa: BLE001
                        page.error(exc, what="run the checks")
                        return
                results.clear()
                summary.set_text(summary_text(report))
                with results:
                    for finding in ordered_findings(report):
                        icon, colour = SEVERITY_STYLE.get(
                            str(finding.get("severity")), SEVERITY_STYLE["info"]
                        )
                        wants_fix = finding.get("severity") in ("warn", "fail")
                        status_row(
                            icon,
                            colour,
                            str(finding.get("title", "")),
                            str(finding.get("message", "")),
                            str(finding.get("fix", "")) if wants_fix else "",
                        )

            admin_control(
                ui.button("Run checks", icon="health_and_safety", on_click=run).props(
                    "color=primary"
                ),
                may_change=may_change,
                what="Run checks",
            )

        with panel_guard("The support bundle"), ui.card().classes("w-full"):
            ui.label("Support bundle").classes("text-lg font-medium")
            ui.label(BUNDLE_NOTE).classes("text-xs opacity-70")

            async def bundle() -> None:
                page = _Page.capture()
                with page, busy(message="Building the support bundle…"):
                    try:
                        name, payload = await make_support_bundle(ctx)
                    except Exception as exc:  # noqa: BLE001
                        page.error(exc, what="build the support bundle")
                        return
                    ui.download(payload, name, "application/zip")
                page.notify("Support bundle ready. Credentials were removed.", type="positive")

            admin_control(
                ui.button("Download support bundle", icon="download", on_click=bundle).props(
                    "outline"
                ),
                may_change=may_change,
                what="Download the support bundle",
            )
        server.health_panels(ctx)
