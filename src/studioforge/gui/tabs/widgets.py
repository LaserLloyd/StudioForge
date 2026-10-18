"""Small controls the newer pages share: code blocks, confirmations, one-time secrets.

Kept separate from :mod:`studioforge.gui.tabs` (the plumbing) because these draw
elements. Every control here carries an accessible name, because an icon-only
button with no label is invisible to a screen reader.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any
from urllib.parse import quote

from nicegui import ui

from studioforge.gui.tabs import GuiContext

NEVER = "never"
DASH = "—"

#: Query parameter that opens a panel section directly (``/?section=Updates``).
SECTION_PARAM = "section"


def section_url(name: str) -> str:
    """A relative link to one of the panel's sections (never an absolute URL)."""
    return f"/?{SECTION_PARAM}={quote(name)}"


def fmt_ts(value: float | None, *, empty: str = DASH) -> str:
    """A local ``YYYY-MM-DD HH:MM`` for a unix time, ``empty`` for none."""
    if not value:
        return empty
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(value)))
    except (OverflowError, OSError, ValueError):
        return empty


def copy_to_clipboard(text: str, *, what: str = "Copied") -> None:
    """Put ``text`` on the viewer's clipboard and say so."""
    ui.clipboard.write(text)
    ui.notify(what, type="positive")


def copy_button(text: str, *, label: str = "Copy", what: str = "Copied") -> Any:
    """An icon button that copies ``text``; the accessible name says what for."""
    button = ui.button(icon="content_copy", on_click=lambda: copy_to_clipboard(text, what=what))
    button.props(f'flat dense aria-label="{label}"').tooltip(label)
    return button


def code_block(title: str, text: str, *, note: str = "") -> None:
    """A titled, monospace, copyable block (a snippet, a URL, a fingerprint)."""
    with ui.column().classes("w-full gap-0"):
        with ui.row().classes("items-center gap-2"):
            ui.label(title).classes("text-xs font-medium")
            copy_button(text, label=f"Copy {title}", what=f"Copied {title}")
        if note:
            ui.label(note).classes("text-xs opacity-70")
        ui.label(text).classes("w-full font-mono text-xs whitespace-pre-wrap sf-well p-2 rounded")


def status_row(icon: str, colour: str, title: str, message: str = "", fix: str = "") -> None:
    """One finding: icon and colour, a title, what it means, and (if any) what to do."""
    with ui.row().classes("w-full items-start gap-2 no-wrap"):
        ui.icon(icon, color=colour, size="1.2rem").props('aria-hidden="true"')
        with ui.column().classes("gap-0 grow"):
            ui.label(title).classes("text-sm font-medium")
            if message:
                ui.label(message).classes("text-xs opacity-80 whitespace-pre-wrap")
            if fix:
                ui.label(f"Fix: {fix}").classes("text-xs text-warning whitespace-pre-wrap")


async def confirm(
    title: str, body: str, *, action: str, colour: str = "negative", cancel: str = "Cancel"
) -> bool:
    """A yes/no dialog, awaited. ``action`` is the verb on the confirming button."""
    with ui.dialog() as dialog, ui.card().classes("min-w-[24rem] max-w-[34rem]"):
        ui.label(title).classes("font-medium")
        ui.label(body).classes("text-sm opacity-80 whitespace-pre-wrap")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button(cancel, on_click=lambda: dialog.submit(False)).props("flat")
            ui.button(action, on_click=lambda: dialog.submit(True)).props(f"color={colour}")
    result = await dialog
    dialog.clear()
    return bool(result)


def one_time_secret_dialog(title: str, secret: str, *, hint: str) -> Any:
    """Show a secret exactly once, with a copy button.

    The value is held only by this dialog's elements: it is not written to the
    page state, a log, a notification or the audit trail, and closing the
    dialog deletes it. ``hint`` says where it goes next.

    Built at page level (not inside the button that asked for it) so a repaint
    of the card behind it cannot delete it before it has been read.
    """
    try:
        page: Any = ui.context.client
    except Exception:  # noqa: BLE001 - no page context (a direct call from a test)
        page = nullcontext()
    with (
        page,
        ui.dialog().props("persistent") as dialog,
        ui.card().classes("min-w-[28rem] max-w-[40rem]"),
    ):
        ui.label(title).classes("font-medium")
        ui.label(hint).classes("text-sm opacity-80")
        ui.label(secret).classes("w-full font-mono text-xs break-all sf-well p-2 rounded").props(
            'aria-label="The new secret, shown once"'
        )
        ui.label(
            "Copy it now. It is not stored and cannot be shown again; if you lose it, make a "
            "new one and revoke this one."
        ).classes("text-xs text-warning")
        with ui.row().classes("w-full justify-end gap-2"):
            copy_button(secret, label="Copy the secret", what="Copied. It will not be shown again")

            def dismiss() -> None:
                dialog.close()
                dialog.delete()

            ui.button("I have copied it", on_click=dismiss).props("color=primary")
    dialog.open()
    return dialog


def is_remote_without_admin(ctx: GuiContext) -> bool:
    """A network bind that nobody has to sign in to, which the server refuses to start in."""
    from studioforge.api.auth import has_admin_credential, remote_services

    accounts = getattr(ctx.api_state, "accounts", None)
    return bool(remote_services(ctx.config)) and not has_admin_credential(ctx.config, accounts)
