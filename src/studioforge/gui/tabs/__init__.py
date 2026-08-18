"""Shared plumbing for the GUI tabs.

The tabs are presentation only. Anything that derives or formats a value lives
in :mod:`studioforge.gui.state`; anything that decides something lives in
``studioforge.core``. What is left here is the context object the tabs read the
system through, plus the two things every tab needs: a way to run a
possibly-slow call without freezing the event loop, and a way to fail visibly
instead of blanking the page.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from nicegui import ui

from studioforge.config import Config
from studioforge.errors import StudioForgeError
from studioforge.logging import get_logger

log = get_logger(__name__)


@dataclass
class GuiContext:
    """Everything a tab is allowed to touch.

    The GUI runs in the *same process* as the gateway and shares its object
    graph by reference, so there is no HTTP-to-self anywhere: a tab calls
    ``ctx.manager.load(...)`` directly. That is what keeps the panel working
    unchanged behind ``tailscale serve``, on a plain-HTTP tailnet, or under any
    reverse proxy -- there is no self-referential URL to get wrong.
    """

    config: Config
    api_state: Any

    def _part(self, name: str) -> Any:
        return getattr(self.api_state, name, None)

    @property
    def manager(self) -> Any:
        return self._part("manager")

    @property
    def registry(self) -> Any:
        return self._part("registry")

    @property
    def supervisor(self) -> Any:
        return self._part("supervisor")

    @property
    def planner(self) -> Any:
        return self._part("planner")

    @property
    def probe(self) -> Any:
        return self._part("probe")

    @property
    def engine_manager(self) -> Any:
        return self._part("engine_manager")

    @property
    def downloader(self) -> Any:
        """May be ``None``: downloads are an optional subsystem."""
        return self._part("downloader")

    @property
    def db(self) -> Any:
        return self._part("db")

    @property
    def refresh_interval(self) -> float:
        return max(0.5, float(self.config.gui.refresh_interval_s))


class _InProcessRequest:
    """The only thing the management routes read off a ``Request``.

    Some controls (restart, benchmark) are implemented *as route handlers* and
    nowhere else: the handler is where the confirmation rule, the job table and
    the "one benchmark at a time" lock live. The GUI shares the gateway's object
    graph in the same process, so it calls those handlers directly with this
    stand-in rather than making an HTTP request back to itself -- which would
    mean the panel had to know its own externally-visible URL, the one thing
    this GUI deliberately never knows (it is what keeps it working unchanged
    behind ``tailscale serve``, a proxy prefix, or plain HTTP on a tailnet).

    Both route modules access exactly ``request.app.state`` and nothing else, so
    this is the whole surface. It is asserted by a test.
    """

    def __init__(self, api_state: Any) -> None:
        self.app = SimpleNamespace(state=api_state)


def api_request(ctx: GuiContext) -> Any:
    """A stand-in ``Request`` for calling a route handler in-process."""
    return _InProcessRequest(ctx.api_state)


def error_text(exc: BaseException) -> str:
    """User-facing message for an exception, preferring our own error text."""
    if isinstance(exc, StudioForgeError):
        return exc.message
    return f"{type(exc).__name__}: {exc}"


def notify_error(exc: BaseException, *, what: str = "") -> None:
    prefix = f"{what}: " if what else ""
    ui.notify(f"{prefix}{error_text(exc)}", type="negative", multi_line=True, close_button=True)
    log.warning("gui action failed", what=what or None, error=str(exc))


@contextmanager
def panel_guard(what: str) -> Iterator[None]:
    """Render an error card instead of losing the whole page to a traceback.

    A control panel that shows an empty screen when one subsystem is missing is
    worse than useless -- the operator cannot tell the difference between "no
    models loaded" and "the GUI crashed".
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - the point is to contain everything
        log.exception("gui panel failed", panel=what, error=str(exc))
        with ui.card().classes("w-full bg-red-50 dark:bg-red-950"):
            ui.label(f"{what} could not be rendered").classes("text-negative font-medium")
            ui.label(error_text(exc)).classes("text-xs font-mono whitespace-pre-wrap")


async def run_blocking[T](fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a synchronous call off the event loop.

    Every viewer of the panel shares one event loop, so a multi-second
    synchronous call (a library scan, an engine install) would freeze the UI for
    everyone, not just the person who clicked.
    """
    return await asyncio.to_thread(lambda: fn(*args, **kwargs))


@contextmanager
def busy(button: Any = None, *, message: str = "") -> Iterator[None]:
    """Disable a control and show a spinner while an action runs."""
    notification = ui.notification(message, spinner=True, timeout=None) if message else None
    if button is not None:
        button.disable()
    try:
        yield
    finally:
        if button is not None:
            button.enable()
        if notification is not None:
            notification.dismiss()


def badge(text: str, *, colour: str = "primary") -> Any:
    return ui.badge(text, color=colour).classes("text-xs")


def section(title: str, subtitle: str = "") -> None:
    ui.label(title).classes("text-lg font-medium mt-2")
    if subtitle:
        ui.label(subtitle).classes("text-xs opacity-70")


def mono(text: str, *, classes: str = "") -> Any:
    return ui.label(text).classes(f"font-mono text-xs {classes}".strip())


__all__ = [
    "GuiContext",
    "api_request",
    "badge",
    "busy",
    "error_text",
    "mono",
    "notify_error",
    "panel_guard",
    "run_blocking",
    "section",
]
