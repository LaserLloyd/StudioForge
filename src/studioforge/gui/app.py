"""The web GUI: a second uvicorn app in the *same* process as the gateway.

Three architectural facts shape every line of this module.

**It is a web app, and only a web app.** There is no desktop/native surface
anywhere. The panel is reached over the tailnet, so it must behave like an
ordinary web page behind an ordinary proxy.

**It shares the gateway's object graph by reference.** ``create_gui_app`` is
handed the API app's ``state``, so a tab calls ``manager.load(...)`` directly
instead of making an HTTP request back to ourselves. That removes the entire
class of bug where the GUI needs to know its own externally-visible URL --
there are no absolute URLs here at all, which is exactly what makes the panel
work identically on plain HTTP over a tailnet and behind ``tailscale serve``'s
HTTPS front end.

**Auth mirrors the gateway's.** When ``server.api_key`` is set, the panel is
gated too: a browser gets a small login page that exchanges the key for a
signed cookie, and an API client can present the same ``Authorization: Bearer``
header the gateway accepts. When no key is configured the page itself is open,
which matches the gateway's own LAN/tailnet-trust default -- reads, chat, load
and unload work for anyone who can reach it -- but the buttons that change the
*box* (config, files, engines, restarts, downloads) follow the API's D32 rule
from the viewer's peer address: they work from a browser on this machine and
refuse a remote one (:func:`studioforge.gui.tabs.require_local_admin`). The
cookie carries a *derived* token, never the key, and ``secure`` is deliberately
not set because the common deployment is plain HTTP on a tailnet.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from nicegui import app as nicegui_app
from nicegui import ui
from starlette.datastructures import Headers

from studioforge import __version__
from studioforge.config import Config
from studioforge.gui import state as st
from studioforge.gui.state import UNKNOWN
from studioforge.gui.tabs import GuiContext, panel_guard
from studioforge.logging import get_logger

log = get_logger(__name__)

#: Session cookie holding a token derived from the API key. Not the key itself.
COOKIE_NAME = "sf_gui_session"

#: Relative paths only -- never an absolute URL, so any proxy prefix works.
LOGIN_PATH = "/login"
LOGOUT_PATH = "/logout"

#: Reachable without the key: the login form itself, its POST target, and the
#: liveness probe. Everything else is gated when a key is configured. The
#: vendored theme assets (see _THEME_URL_PREFIX below) are exempted the same
#: way, but by prefix rather than exact path.
_OPEN_PATHS = frozenset({LOGIN_PATH, LOGOUT_PATH, "/favicon.ico", "/gui-health"})

#: Public, non-sensitive static assets (CSS/JS/JSON -- no user data, no
#: control surface) so the login page itself can be themed before the visitor
#: has a session. Prefix, not an exact-path member of _OPEN_PATHS, because it
#: covers a whole directory; "/sf-theme" with no trailing slash and anything
#: not under it (e.g. "/sf-themex") still goes through the gate below.
_THEME_URL_PREFIX = "/sf-theme/"

#: NiceGUI's element tree and page routes are process-global singletons, so the
#: pages are registered exactly once even if the app factory is called again
#: (tests do; a reload could).
_PAGES_REGISTERED = False
_NICEGUI_MOUNTED = False

#: The live context the (global) page functions read. Rebound on every
#: ``create_gui_app`` call so the newest wiring wins.
_CONTEXT: GuiContext | None = None

#: Setup sits second, right after the Dashboard, because it is read once and
#: then rarely -- but a *fresh* install opens on it (see :func:`_default_tab`),
#: since on a box with no engine and no library the Dashboard has nothing to
#: show and the checklist has everything.
TAB_NAMES = ("Dashboard", "Setup", "Models", "Download", "Chat", "Server", "Logs")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def session_token(api_key: str) -> str:
    """Cookie value for ``api_key``: an HMAC of a fixed label under the key.

    Derived rather than stored so the cookie is useless as a credential
    anywhere else, and so a key rotation invalidates every existing session
    without any server-side session table.
    """
    return hmac.new(
        api_key.encode("utf-8"), b"studioforge-gui-session-v1", hashlib.sha256
    ).hexdigest()


def _presented_key(headers: Headers) -> str | None:
    """The API key from the request headers, the way clients send it."""
    authorization = headers.get("authorization")
    if authorization:
        prefix, _, value = authorization.partition(" ")
        if prefix.lower() == "bearer" and value.strip():
            return value.strip()
    api_key = headers.get("x-api-key")
    return api_key.strip() if api_key else None


def _cookie_value(headers: Headers) -> str | None:
    raw = headers.get("cookie")
    if not raw:
        return None
    for chunk in raw.split(";"):
        name, _, value = chunk.strip().partition("=")
        if name == COOKIE_NAME:
            return value
    return None


def _safe_eq(presented: str, expected: str) -> bool:
    """Constant-time compare that cannot 500 on odd header bytes.

    Starlette decodes headers as latin-1, so any byte >= 0x80 yields a
    non-ASCII ``str`` -- and ``hmac.compare_digest`` raises ``TypeError`` for
    those. Uncaught in the auth gate that became an unhandled 500 (and a dead
    websocket) for anything as ordinary as a stale non-ASCII cookie from
    another app on the same host. Comparing bytes keeps the constant-time
    property and turns such a probe into an ordinary 401.
    """
    return hmac.compare_digest(
        presented.encode("utf-8", "surrogateescape"),
        expected.encode("utf-8", "surrogateescape"),
    )


def _is_authorized(headers: Headers, expected: str) -> bool:
    presented = _presented_key(headers)
    if presented and _safe_eq(presented, expected):
        return True
    cookie = _cookie_value(headers)
    if not cookie:
        return False
    return _safe_eq(cookie, session_token(expected))


def _host_only(value: str) -> str:
    """``host[:port]`` -> ``host``, IPv6 brackets kept, lower-cased."""
    text = (value or "").strip().lower()
    if text.startswith("["):
        return text.split("]", 1)[0] + "]"
    return text.rsplit(":", 1)[0] if ":" in text else text


#: Where NiceGUI mounts its socket.io control channel. It is **not** only a
#: WebSocket: socket.io opens with HTTP long-polling and upgrades afterwards,
#: so the same control channel is reachable as ordinary ``GET``/``POST``
#: requests under this prefix -- which is how the D32 origin gate was
#: bypassable by transport selection until D55 (a cross-site page that simply
#: never upgraded drove the panel over polling, and NiceGUI's own
#: ``cors_allowed_origins='*'`` echoed the attacker's Origin back with
#: credentials allowed).
NICEGUI_SOCKET_PREFIX = "/_nicegui_ws/"

#: Sent on every GUI response. The panel is never a legitimate frame: with the
#: viewer's peer address deciding what they may do (D32), an iframe of
#: ``http://127.0.0.1:8080/`` on any page the operator visits renders a panel
#: whose viewer IP is loopback -- so every admin control, up to and including
#: the PIN reveal, is one clickjacked click away. ``frame-ancestors`` is the
#: modern spelling and ``X-Frame-Options`` the one older browsers obey; both
#: are sent because they disagree about nothing here.
#:
#: Deliberately *only* ``frame-ancestors``. A ``default-src``/``script-src``
#: policy would also be worth having, but NiceGUI's page is built from inline
#: bootstrap script, inline styles and Quasar's own bundles, and a CSP that
#: breaks the panel is worse than no CSP: it would take the operator's only
#: recovery surface down at exactly the moment they need it. Adding one is a
#: browser-verified change, not a header-set change.
_FRAME_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"x-frame-options", b"DENY"),
    (b"content-security-policy", b"frame-ancestors 'none'"),
)


def _same_origin_scope(scope: Any) -> bool:
    """False only for a browser request whose Origin names a different host.

    Non-browser clients send no Origin and pass; a browser always sends one on
    a WebSocket handshake and on any cross-origin fetch. Ports are deliberately
    not compared: the panel is reached through the same host:port it was served
    from, so the host alone settles same-site, and a proxy that only rewrites
    the port stays usable.
    """
    headers = Headers(scope=scope)
    origin = headers.get("origin")
    host = headers.get("host")
    if not origin or not host:
        return True
    from urllib.parse import urlsplit

    try:
        origin_host = (urlsplit(origin).hostname or "").lower()
    except ValueError:
        return False
    if not origin_host:
        return False
    if ":" in origin_host and not origin_host.startswith("["):
        origin_host = f"[{origin_host}]"
    return origin_host == _host_only(host)


def _with_frame_headers(send: Any) -> Any:
    """Wrap an ASGI ``send`` so every response start carries :data:`_FRAME_HEADERS`."""

    async def wrapped(message: Any) -> None:
        if message.get("type") == "http.response.start":
            existing = {name.lower() for name, _ in message.get("headers") or ()}
            message = dict(message)
            message["headers"] = [
                *(message.get("headers") or ()),
                *((name, value) for name, value in _FRAME_HEADERS if name not in existing),
            ]
        await send(message)

    return wrapped


def _cross_site_refusal() -> Any:
    """The 403 for a cross-site request to the control channel.

    A plain ASGI response with **no** ``Access-Control-Allow-Origin`` header:
    the refusal must not be readable by the page that made it, or the answer
    itself becomes a probe. NiceGUI's socket.io would have echoed the caller's
    Origin here, which is the half of the bypass that made it useful.
    """
    return JSONResponse(
        {
            "error": {
                "message": (
                    "Refused: this is the control panel's own control channel and the "
                    "request's Origin is not the host it was served from. If a reverse "
                    "proxy rewrites Host, forward the original one "
                    "(proxy_set_header Host $host)."
                ),
                "type": "invalid_request_error",
                "code": "cross_site_control_channel",
                "param": None,
            }
        },
        status_code=403,
    )


class GuiAuthGate:
    """Raw ASGI gate so websockets are covered, not just page loads.

    A ``BaseHTTPMiddleware`` would let NiceGUI's websocket through untouched,
    and that socket *is* the control channel -- every button press travels on
    it. So this is plain ASGI and closes an unauthenticated upgrade.
    """

    def __init__(self, app: Any, config: Config) -> None:
        self.app = app
        self.config = config

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        kind = scope.get("type")
        path = scope.get("path", "/") or "/"
        control_channel = kind == "websocket" or (
            kind == "http" and path.startswith(NICEGUI_SOCKET_PREFIX)
        )
        if control_channel and not _same_origin_scope(scope):
            # A WebSocket handshake is not subject to the same-origin policy,
            # and NiceGUI's socket.io accepts any Origin -- so a page on any
            # website the operator visits could open the control channel to
            # http://<lan-ip>:8080 and press its buttons. With no API key set
            # there is no cookie to be missing, so this check is the only
            # thing between "someone on my LAN" and "any site I browse".
            #
            # D55: socket.io's *first* transport is HTTP long-polling, not a
            # WebSocket, so keying this on ``type == "websocket"`` left the
            # whole channel open to a cross-site page that simply never
            # upgraded -- verified against the live panel, which answered
            # ``GET /_nicegui_ws/socket.io/?transport=polling`` with
            # ``access-control-allow-origin: https://evil.example`` and a
            # session cookie. The gate is now the path, not the transport.
            log.warning(
                "gui control channel refused: Origin does not match Host (cross-site)",
                origin=Headers(scope=scope).get("origin"),
                host=Headers(scope=scope).get("host"),
                transport=kind,
                hint=(
                    "a reverse proxy that rewrites the Host header must forward the "
                    "original one (proxy_set_header Host $host)"
                ),
            )
            if kind == "websocket":
                await receive()
                await send({"type": "websocket.close", "code": 1008})
                return
            await _cross_site_refusal()(scope, receive, send)
            return
        if kind == "http":
            # Every GUI response carries the frame headers, including NiceGUI's
            # own -- which is why this wraps ``send`` rather than living in a
            # response-returning middleware: the page, its assets and the
            # polling transport all come out of NiceGUI's mount, below us.
            send = _with_frame_headers(send)
        expected = self.config.server.api_key
        if not expected or kind not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        if (
            path in _OPEN_PATHS
            or path.startswith(_THEME_URL_PREFIX)
            or scope.get("method") == "OPTIONS"
        ):
            await self.app(scope, receive, send)
            return
        if _is_authorized(headers, expected):
            await self.app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            await receive()  # consume websocket.connect before refusing
            await send({"type": "websocket.close", "code": 1008})
            return

        accept = headers.get("accept", "")
        if "text/html" in accept:
            response: Response = RedirectResponse(LOGIN_PATH, status_code=303)
        else:
            response = JSONResponse(
                {
                    "error": {
                        "message": (
                            "This panel requires the server API key. Open "
                            f"{LOGIN_PATH} in a browser or send it as "
                            "'Authorization: Bearer <key>'."
                        ),
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                        "param": None,
                    }
                },
                status_code=401,
            )
        await response(scope, receive, send)


# ---------------------------------------------------------------------------
# Theme assets
# ---------------------------------------------------------------------------
#
# Vendored by tools/sync_theme.py from the unifyingTheme package (V26-09-16)
# -- see docs/DEVELOPMENT.md's "GUI theming" section. Never hand-edited: a
# change here belongs in that package's src/ or adapters/, followed by
# `python tools/sync_theme.py app studioforge`.

#: Where the four vendored files (ui-theme.js/.css/-base.css/.json) live.
_THEME_DIR = Path(__file__).parent / "theme"


def _asset_version(name: str) -> str:
    """First 10 hex chars of a vendored asset's sha256, for cache-busting.

    Computed once at import so every page load and the login page agree on
    the same query string without re-hashing the file per request. Falls
    back to a fixed placeholder if the vendored copy is somehow missing --
    the asset itself 404s in that case, which is the more useful signal.
    """
    try:
        digest = hashlib.sha256((_THEME_DIR / name).read_bytes()).hexdigest()
    except OSError:  # pragma: no cover - only if the vendored copy is missing
        return "0" * 10
    return digest[:10]


_THEME_JS_VERSION = _asset_version("ui-theme.js")
_THEME_BASE_CSS_VERSION = _asset_version("ui-theme-base.css")
_THEME_CSS_VERSION = _asset_version("ui-theme.css")


def _theme_script_tag(name: str, version: str) -> str:
    return f'<script src="{_THEME_URL_PREFIX}{name}?v={version}"></script>'


def _theme_link_tag(name: str, version: str) -> str:
    return f'<link rel="stylesheet" href="{_THEME_URL_PREFIX}{name}?v={version}">'


#: Injected once into every page's <head> (see _register_theme_assets and
#: _login_html), in this order: the runtime script -- blocking, so the
#: palette is on the page before first paint -- then the element layer
#: (loaded before the app's own CSS so its rules keep priority), then the
#: contract tokens (loaded last so they beat any same-named legacy value).
_THEME_HEAD_HTML = (
    _theme_script_tag("ui-theme.js", _THEME_JS_VERSION)
    + _theme_link_tag("ui-theme-base.css", _THEME_BASE_CSS_VERSION)
    + _theme_link_tag("ui-theme.css", _THEME_CSS_VERSION)
)


def _theme_manifest() -> Mapping[str, Any]:
    """The vendored picker manifest: enabled themes in picker order."""
    try:
        text = (_THEME_DIR / "ui-theme.json").read_text(encoding="utf-8")
        return json.loads(text)
    except (OSError, ValueError):  # pragma: no cover - only if the copy is missing/corrupt
        return {"manifest": {"default": "glacier"}, "themes": []}


_THEME_MANIFEST = _theme_manifest()


def _theme_options() -> dict[str, str]:
    """``{slug: name}`` for the header picker, already in picker order."""
    return {theme["slug"]: theme["name"] for theme in _THEME_MANIFEST.get("themes", [])}


def _theme_default() -> str:
    return str(_THEME_MANIFEST.get("manifest", {}).get("default", "glacier"))


#: Page-level event the header picker listens on; the browser emits it from
#: UITheme.onChange so the select follows switches made elsewhere.
_THEME_CHANGE_EVENT = "sf_theme_change"


#: Guards the one-time /sf-theme static mount and the one-time <head>
#: injection. Both are process-global (``nicegui_app`` is a singleton, and
#: ``ui.add_head_html(shared=True)`` appends to a process-global list on
#: every call) -- mirrors _PAGES_REGISTERED/_NICEGUI_MOUNTED just below,
#: since tests call create_gui_app more than once.
_THEME_ASSETS_REGISTERED = False


def _register_theme_assets() -> None:
    global _THEME_ASSETS_REGISTERED
    if _THEME_ASSETS_REGISTERED:
        return
    _THEME_ASSETS_REGISTERED = True
    nicegui_app.add_static_files(_THEME_URL_PREFIX.rstrip("/"), _THEME_DIR)
    ui.add_head_html(_THEME_HEAD_HTML, shared=True)


_LOGIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StudioForge — sign in</title>
{theme_head}
<style>
 :root {{ color-scheme: light dark; }}
 body {{ font-family: var(--font-sans); display: grid; place-items: center;
        min-height: 100vh; margin: 0; background: var(--surface-0); color: var(--text-primary); }}
 form {{ background: var(--surface-1); padding: 2rem; border-radius: var(--radius-lg);
        width: min(24rem, 90vw); box-shadow: var(--shadow-3);
        border: 1px solid var(--border-subtle); }}
 h1 {{ font-size: 1.1rem; margin: 0 0 .25rem; }}
 p {{ font-size: .8rem; color: var(--text-secondary); margin: 0 0 1.25rem; }}
 input {{ width: 100%; padding: .6rem .7rem; font-size: 1rem; border-radius: var(--radius-sm);
         border: 1px solid var(--border-strong); background: var(--surface-sunken);
         color: var(--text-primary); box-sizing: border-box; }}
 input:focus-visible {{ outline: var(--focus-ring-width) solid var(--focus-ring);
         outline-offset: var(--focus-ring-offset); }}
 button {{ margin-top: 1rem; width: 100%; padding: .6rem; font-size: 1rem; border: 0;
          border-radius: var(--radius-sm); background: var(--accent); color: var(--on-accent);
          cursor: pointer; }}
 button:hover {{ background: var(--accent-hover); }}
 button:focus-visible {{ outline: var(--focus-ring-width) solid var(--focus-ring);
         outline-offset: var(--focus-ring-offset); }}
 .error {{ color: var(--danger-text); font-size: .8rem; margin-top: .75rem; }}
</style></head>
<body><form method="post" action="{login_path}">
 <h1>StudioForge</h1>
 <p>This panel is protected by the server API key.</p>
 <input type="password" name="api_key" placeholder="API key" autofocus
        autocomplete="current-password">
 <button type="submit">Sign in</button>
 {error}
</form></body></html>
"""


def _login_html(error: str = "") -> str:
    block = f'<div class="error">{error}</div>' if error else ""
    return _LOGIN_PAGE.format(login_path=LOGIN_PATH, error=block, theme_head=_THEME_HEAD_HTML)


def _install_auth_routes(app: FastAPI, config: Config) -> None:
    """Login/logout plus the gate. Registered before NiceGUI's catch-all mount."""

    @app.get(LOGIN_PATH, include_in_schema=False)
    async def login_form() -> HTMLResponse:
        if not config.server.api_key:
            return HTMLResponse(_login_html(), status_code=200)
        return HTMLResponse(_login_html())

    @app.post(LOGIN_PATH, include_in_schema=False)
    async def login_submit(api_key: str = Form("")) -> Response:
        expected = config.server.api_key
        if not expected:
            return RedirectResponse("/", status_code=303)
        if not api_key or not _safe_eq(api_key.strip(), expected):
            # Never log the submitted value, correct or not.
            log.info("gui login rejected")
            return HTMLResponse(_login_html("That key was not accepted."), status_code=401)
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(
            COOKIE_NAME,
            session_token(expected),
            httponly=True,
            samesite="lax",
            path="/",
            max_age=30 * 24 * 3600,
        )
        log.info("gui login accepted")
        return response

    @app.get(LOGOUT_PATH, include_in_schema=False)
    async def logout() -> Response:
        response = RedirectResponse(LOGIN_PATH, status_code=303)
        response.delete_cookie(COOKIE_NAME, path="/")
        return response

    @app.get("/gui-health", include_in_schema=False)
    async def gui_health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "gui": True}

    app.add_middleware(GuiAuthGate, config=config)


# ---------------------------------------------------------------------------
# Page shell
# ---------------------------------------------------------------------------


def _context() -> GuiContext:
    if _CONTEXT is None:  # pragma: no cover - create_gui_app always sets it
        raise RuntimeError("GUI context not initialised")
    return _CONTEXT


def _theme_picker() -> None:
    """Compact Quasar select that drives ``window.UITheme`` (adapters/quasar.js).

    The page is already painted in the visitor's stored theme before this
    element even mounts (ui-theme.js runs synchronously in <head>, before
    first paint), so this is purely a control, not the source of truth for
    what's on screen. Its initial value is the manifest default; the moment
    the client connects, ``sync_initial`` corrects it to whatever
    ``UITheme.current()`` actually applied and subscribes to
    ``UITheme.onChange``, so a switch made in another tab (the runtime's
    storage listener) or from script keeps the select in step. The
    ``mirroring`` guard stops those corrections from round-tripping through
    ``on_change`` -- NiceGUI fires the change handler on a programmatic
    ``select.value =`` assignment exactly the same as on a user pick.
    """
    options = _theme_options()
    mirroring = False

    select = (
        ui.select(options, value=_theme_default())
        .props('dense outlined options-dense aria-label="Theme"')
        .classes("sf-theme-picker")
    )
    # An icon rather than a tooltip: a tooltip stays up over the open menu.
    with select.add_slot("prepend"):
        ui.icon("palette", size="xs")

    def mirror(slug: Any) -> None:
        nonlocal mirroring
        if slug not in options or slug == select.value:
            return
        mirroring = True
        try:
            select.value = slug
        finally:
            mirroring = False

    def on_change(event: Any) -> None:
        if not mirroring:
            ui.run_javascript(f"UITheme.set({json.dumps(event.value)})")

    select.on_value_change(on_change)
    ui.on(_THEME_CHANGE_EVENT, lambda event: mirror(event.args))

    async def sync_initial() -> None:
        mirror(await ui.run_javascript("UITheme.current()"))
        ui.run_javascript(
            f"UITheme.onChange((d) => emitEvent({json.dumps(_THEME_CHANGE_EVENT)}, d.slug))"
        )

    ui.timer(0.0, sync_initial, once=True)


def _header(ctx: GuiContext) -> Any:
    with ui.header().classes("items-center justify-between px-4 py-2"):
        with ui.row().classes("items-center gap-3"):
            ui.icon("memory", size="1.6rem")
            ui.label("StudioForge").classes("text-lg font-semibold")
            ui.label(f"v{__version__}").classes("text-xs opacity-70")
        with ui.row().classes("items-center gap-2"):
            status = ui.label("").classes("text-xs opacity-80 font-mono")
            _theme_picker()
            if ctx.config.server.api_key:
                ui.link("sign out", LOGOUT_PATH).classes("text-xs")
    return status


def _status_line(ctx: GuiContext) -> str:
    """Compact header summary; degrades to a dash rather than raising."""
    try:
        loaded = ctx.supervisor.list() if ctx.supervisor is not None else []
        gpus = ctx.probe.list_gpus() if ctx.probe is not None else []
    except Exception:  # noqa: BLE001 - header must never break the page
        return UNKNOWN
    free = sum(g.free_bytes for g in gpus)
    total = sum(g.total_bytes for g in gpus)
    gpu_text = (
        f"{len(gpus)} GPU · {free / 1024**3:.1f}/{total / 1024**3:.1f} GiB free"
        if gpus
        else "no GPU"
    )
    # D50: a child mid-spawn counted as loaded here, so the header claimed
    # residency the moment a load started and said nothing for the ~30 s it
    # took to arrive. ``loaded_count_text`` breaks the two apart.
    return f"{st.loaded_count_text(loaded)} · {gpu_text}"


def _default_tab(ctx: GuiContext) -> str:
    """Where the page lands with no deep link: Setup on a fresh install.

    "Fresh" is measured, not remembered: no model library, or no models
    indexed, or no engine installed. On such a box the Dashboard is four empty
    panels and the Setup tab is the whole answer, so opening there saves the
    new user from having to guess which tab is the one that matters. Once the
    server can actually serve, this reverts to the Dashboard for good.

    Any failure lands on the Dashboard: the landing tab is not worth a broken
    page.
    """
    try:
        config = ctx.config
        if not config.models.dir:
            return "Setup"
        if ctx.registry is not None and not ctx.registry.all():
            return "Setup"
        if ctx.engine_manager is not None and ctx.engine_manager.active() is None:
            return "Setup"
    except Exception:  # noqa: BLE001 - never break the page over a landing tab
        return "Dashboard"
    return "Dashboard"


def _render_index(query: Mapping[str, Any] | None = None) -> None:
    """Build the single-page shell: header, tab strip, six panels.

    ``query`` carries the deep-link intent (``?tab=download&repo=owner/repo``)
    that the protocol handler produces from HuggingFace's download button. With
    no query string this behaves exactly as before and opens the Dashboard.
    """
    ctx = _context()
    ui.page_title("StudioForge")
    status = _header(ctx)
    params = st.deep_link_params(query)

    from studioforge.gui.tabs import chat, dashboard, download, logs, models, server, setup

    renderers: dict[str, Callable[[], None]] = {
        "Dashboard": lambda: dashboard.render(ctx),
        "Setup": lambda: setup.render(ctx),
        "Models": lambda: models.render(ctx, params),
        "Download": lambda: download.render(ctx, params),
        "Chat": lambda: chat.render(ctx),
        "Server": lambda: server.render(ctx),
        "Logs": lambda: logs.render(ctx),
    }

    with ui.tabs().classes("w-full") as tabs:
        for name in TAB_NAMES:
            ui.tab(name)
    with ui.tab_panels(tabs, value=st.initial_tab(params, default=_default_tab(ctx))).classes(
        "w-full"
    ):
        for name in TAB_NAMES:
            with ui.tab_panel(name), panel_guard(f"The {name} tab"):
                renderers[name]()

    def tick() -> None:
        status.set_text(_status_line(ctx))

    tick()
    ui.timer(max(2.0, ctx.refresh_interval), tick)


def _register_pages() -> None:
    global _PAGES_REGISTERED
    if _PAGES_REGISTERED:
        return
    _PAGES_REGISTERED = True

    @ui.page("/")
    def index(request: Request) -> None:
        # NiceGUI injects the Starlette request when the page declares it,
        # which is how the deep-link query string reaches the tabs.
        _render_index(dict(request.query_params))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


#: Where the panel's session-signing key lives, under the data dir.
GUI_SECRET_FILE = "gui_secret"


def _storage_secret(config: Config) -> str:
    """The key Starlette signs the panel's session cookie with (D55).

    It used to be ``sha256("studioforge-gui::" + data_dir)`` -- derived from a
    value the watchdog publishes: un-credentialed ``:1235/health`` returns
    ``config_path``, whose parent *is* the data dir, so anyone who could reach
    the recovery sidecar could compute the signing key and forge a session
    cookie. Today that cookie carries only tab preferences, which is why this
    was a latent finding rather than a live one -- but a signing key nobody can
    keep secret is not a signing key, and every future use of
    ``app.storage.user`` would inherit the flaw.

    So: 32 random bytes, written once to ``<data_dir>/gui_secret`` with
    owner-only permissions where the platform has them, and read back on every
    later start so sessions survive a restart. Any failure to read or write
    falls back to a per-process random secret -- sessions do not survive that
    restart, which is a cosmetic loss, and the key is still not guessable.
    """
    import os
    import secrets

    path = config.data_dir / GUI_SECRET_FILE
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if len(existing) >= 32:
            return existing
    except OSError:
        pass
    value = secrets.token_hex(32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        if os.name != "nt":
            path.chmod(0o600)
    except OSError as exc:
        log.warning(
            "could not persist the GUI session secret; panel sessions will not survive a restart",
            path=str(path),
            error=str(exc),
        )
    return value


def create_gui_app(config: Config, *, api_state: Any) -> FastAPI:
    """Build the GUI's FastAPI app. Never starts a server.

    ``ui.run_with`` mounts NiceGUI into *our* app so the caller owns the
    uvicorn lifecycle; ``ui.run()`` would start its own server and block, which
    would break the single-process/two-ports design in ``__main__``.
    """
    global _CONTEXT, _NICEGUI_MOUNTED
    _CONTEXT = GuiContext(config=config, api_state=api_state)

    app = FastAPI(
        title="StudioForge Control Panel",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    _install_auth_routes(app, config)
    _register_pages()
    _register_theme_assets()

    if not _NICEGUI_MOUNTED:
        ui.run_with(
            app,
            title="StudioForge",
            favicon="🔥",
            dark=True,
            reconnect_timeout=10.0,
            show_welcome_message=False,
            # Storage is keyed per browser session; the secret is random and
            # persisted so it survives a restart (see _storage_secret).
            storage_secret=_storage_secret(config),
        )
        _NICEGUI_MOUNTED = True
    else:
        # NiceGUI's own app is a process-global singleton and its middleware
        # stack is already built, so a second factory call re-uses the mount
        # rather than reconfiguring it. Production calls this once; tests and
        # reloads must not blow up on the second call.
        app.mount("/", nicegui_app)

    log.info("gui app created", port=config.gui.port, auth=bool(config.server.api_key))
    return app


__all__ = ["COOKIE_NAME", "LOGIN_PATH", "LOGOUT_PATH", "create_gui_app", "session_token"]
