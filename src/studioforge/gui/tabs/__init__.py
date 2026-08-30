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
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from nicegui import ui

from studioforge.config import Config
from studioforge.errors import StudioForgeError
from studioforge.gui import state as st
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


# ---------------------------------------------------------------------------
# D32 for the panel: on an open install, changing the box needs a local viewer
# ---------------------------------------------------------------------------


def viewer_host() -> str:
    """Peer address of the browser driving the current page or event.

    NiceGUI runs every page build, timer and event handler inside that page's
    client context, so this is the socket peer of the viewer who clicked --
    which behind a reverse proxy is the proxy, the same stated limit as the
    API's peer check. ``""`` when there is no client context at all: a direct
    in-process call (a test driving an action function), which never crossed
    a network and is trusted the way :func:`is_local_request` trusts it.
    """
    try:
        return str(ui.context.client.ip or "")
    except (RuntimeError, AttributeError):
        return ""


def viewer_may_change_box(ctx: GuiContext) -> bool:
    """Whether the current viewer passes the D32 rule.

    With ``server.api_key`` set, reaching the panel at all took the key (the
    gate exchanged it for the session cookie), so the key *is* the credential
    and nothing more is asked -- exactly as the API behaves. Without one, only
    a viewer on this machine may change the box.
    """
    if ctx.config.server.api_key:
        return True
    # The API's own loopback predicate, so the panel and ``check_request``
    # cannot disagree on what "this machine" means.
    from studioforge.api.auth import _is_loopback

    host = viewer_host()
    return not host or _is_loopback(host)


#: What a remote viewer on an open install is told. Names the two ways in, in
#: the same words the API uses for the same refusal.
REMOTE_VIEWER_NOTE = (
    "this changes the server itself and server.api_key is not set, so the panel only "
    "accepts it from a browser on this machine. Open the panel on the box, or set "
    "server.api_key (Setup tab, on the box) and sign in with it to manage the server "
    "remotely with a real credential."
)


def require_local_admin(ctx: GuiContext, what: str) -> None:
    """Refuse a box-changing action from a remote viewer on an open install.

    The API closed this in D32: with no ``server.api_key``, editing config,
    deleting files, restarting, installing engines, queueing downloads and
    killing processes take a caller on this machine or the PIN. The panel
    calls the same code in-process -- no route, no auth middleware -- so it
    has to apply the rule itself, from the viewer's peer, or every one of those
    actions is one click away for anyone on the LAN at ``:8080``. Reads, chat,
    load and unload stay open, as D32 keeps them on the API.
    """
    if viewer_may_change_box(ctx):
        return
    log.warning("gui action refused: remote viewer on an open install", what=what)
    raise StudioForgeError(
        REMOTE_VIEWER_NOTE, code="remote_admin_requires_credential", status_code=403
    )


def admin_control(control: Any, *, may_change: bool, what: str, tooltip: str = "") -> Any:
    """Disable a box-changing control the current viewer may not use (D49-9).

    Before this, every engine button rendered enabled for a remote viewer on an
    open install and answered the click with a red toast -- which teaches the
    operator that the panel is broken rather than that they are on the wrong
    side of D32. A disabled button carrying the reason names the two ways in.

    Courtesy, not enforcement: :func:`require_local_admin` still runs inside
    the action, because a disabled button is one WebSocket message away from
    being enabled again.

    Exactly one tooltip is attached either way -- NiceGUI appends a tooltip
    element per call, so calling it twice would render two.
    """
    if not may_change:
        control.disable()
        control.tooltip(f"{what}: {REMOTE_VIEWER_NOTE}")
    elif tooltip:
        control.tooltip(tooltip)
    return control


def remote_viewer_banner() -> None:
    """One line at the top of a card whose controls are disabled (D49-9).

    Per card, not per button: a row of greyed-out buttons with eight identical
    tooltips is noise, and the answer -- open the panel on the box, or set an
    API key -- is the same one for all of them.
    """
    with ui.row().classes("w-full items-center gap-2 no-wrap"):
        ui.icon("lock", color="warning", size="1rem")
        ui.label(REMOTE_VIEWER_NOTE).classes("text-xs text-warning")


async def apply_config_updates(ctx: GuiContext, updates: dict[str, Any]) -> dict[str, Any]:
    """The GUI's one and only "change a setting".

    Calls the management route in-process rather than reimplementing it, so the
    Setup tab, the Server tab, ``PATCH /api/config`` and the MCP ``set_config``
    tool cannot drift on what saving a setting means. The handler validates
    through :func:`studioforge.config.apply_overrides`, writes ``config.yaml``
    atomically, mutates the shared config object where a change can take effect
    live, and reports which keys still need a restart.

    Returns the route's payload: ``{"updated": [...], "restart_required": [...]}``.
    """
    if not updates:
        return {"updated": [], "restart_required": []}
    # Checked here, not only in the tabs: this is the single path every
    # setting goes through, so a remote viewer on an open install cannot set
    # server.api_key and lock the owner out whichever button they found.
    require_local_admin(ctx, "change a setting")
    from studioforge.api.mgmt_routes import set_config

    payload = await set_config(api_request(ctx), updates)
    return dict(payload)


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


class BusySpinner:
    """Handle yielded by :func:`busy` so a long action can retitle its spinner.

    The engine install is the case that needs it: several minutes of download
    behind a fixed "Installing…" reads as a hang (D49-10). Retitling is a
    silent no-op when the caller asked for no message, so no caller has to
    check, and a failed cosmetic update never fails the action it decorates.
    """

    def __init__(self, notification: Any = None) -> None:
        self._notification = notification

    def set_message(self, text: str) -> None:
        if self._notification is None or not text:
            return
        with suppress(Exception):  # cosmetic: a dismissed notification is not an error
            self._notification.message = text
            self._notification.update()


# ---------------------------------------------------------------------------
# One long action at a time (D50)
# ---------------------------------------------------------------------------

#: Keys of the long actions currently in flight, process-wide.
#:
#: Process-wide, not per-client, deliberately: the thing being protected is the
#: rig, not the page. Two browsers pointed at the same box double-loading every
#: resident model is the same accident as one browser double-clicking, and the
#: models are shared by both.
_IN_FLIGHT: set[str] = set()


@contextmanager
def single_flight(key: str, what: str) -> Iterator[bool]:
    """Claim a long action, or yield ``False`` because it is already running.

    On 2026-08-30 the Server tab's "activate + reload" fired twice 118 ms apart
    -- an ordinary double-click -- and both runs went through: every resident
    model was force-reloaded twice, ~31 s of churn, and a 37 GB model was killed
    3.6 seconds after it reported ready.

    The guard lives at the *action*, not at the button, for two reasons. First,
    :func:`busy` only disables a control when it is handed one, and most of
    these handlers are ``lambda``s wired straight into ``on_click`` with the
    button nowhere in scope -- ``activate_engine`` was exactly that. Second, a
    disabled button is one WebSocket message and one browser tab away from being
    enabled again, so it was never the thing that made this safe; it is a
    courtesy, the same courtesy :func:`admin_control` provides.

    First line of defence, not the only one: ``ModelManager`` folds a forced
    reload that queued behind an identical one, so an unguarded double-click no
    longer thrashes either. Both exist because they fail differently -- this one
    is instant and says so, that one is correct even for two clicks that never
    met a GUI.

    Released in a ``finally``, so an action that raises does not wedge its key
    for the life of the process.

    Usage::

        with single_flight("engine.activate", "engine activation") as claimed:
            if not claimed:
                return
            ...
    """
    if key in _IN_FLIGHT:
        log.info("gui action already in flight", key=key)
        ui.notify(st.already_running_note(what), type="warning", multi_line=True)
        yield False
        return
    _IN_FLIGHT.add(key)
    try:
        yield True
    finally:
        _IN_FLIGHT.discard(key)


def element_alive(element: Any) -> bool:
    """Whether it is still safe to repaint into ``element`` (D50).

    A panel's ``refresh()`` is held by a timer, by its own action handlers and
    by the buttons on the cards it drew, and every one of those calls it *after*
    an await. If the page was rebuilt in the meantime -- a reconnect, a tab
    switch, the repaint that follows a reload -- the container those closures
    captured is gone, and building an element inside it raises ``The parent
    element this slot belongs to has been deleted`` from NiceGUI's own
    ``Slot.parent``, which is what appeared at 04:20:33 on 2026-08-30 as the
    reloads finished.

    That is not an error to report, it is a repaint with nowhere to go: the page
    it belonged to no longer exists, and a fresh panel with its own timer is
    already drawing the same thing. So the callers bail out silently.

    NiceGUI 3.16 exposes ``Element.is_deleted``, and tears a client down by
    marking every one of its elements deleted *before* the weakrefs can die
    (``Client._handle_delete`` -> ``remove_all_elements``), so this one flag
    covers both the rebuilt-panel and the disconnected-client cases. The
    ``except`` is for the ordering nobody promised us: reaching a dead client
    through a live element raises, and a liveness check that can itself raise is
    not a liveness check.
    """
    if element is None:
        return False
    try:
        if bool(element.is_deleted):
            return False
        return not bool(element.client.is_deleted)
    except Exception:  # noqa: BLE001 - a check that cannot answer means "gone"
        return False


@contextmanager
def busy(button: Any = None, *, message: str = "") -> Iterator[BusySpinner]:
    """Disable a control and show a spinner while an action runs.

    Yields the spinner: an action that knows how far along it is can keep the
    message current; every other caller ignores the value.
    """
    notification = ui.notification(message, spinner=True, timeout=None) if message else None
    if button is not None:
        button.disable()
    try:
        yield BusySpinner(notification)
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
    "REMOTE_VIEWER_NOTE",
    "BusySpinner",
    "GuiContext",
    "admin_control",
    "api_request",
    "apply_config_updates",
    "badge",
    "busy",
    "element_alive",
    "error_text",
    "mono",
    "notify_error",
    "panel_guard",
    "remote_viewer_banner",
    "require_local_admin",
    "run_blocking",
    "section",
    "single_flight",
    "viewer_host",
    "viewer_may_change_box",
]
