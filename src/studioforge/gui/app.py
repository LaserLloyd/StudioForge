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
from collections.abc import Callable, Mapping
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
#: liveness probe. Everything else is gated when a key is configured.
_OPEN_PATHS = frozenset({LOGIN_PATH, LOGOUT_PATH, "/favicon.ico", "/gui-health"})

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


def _same_origin_websocket(scope: Any) -> bool:
    """False only for a browser upgrade whose Origin names a different host.

    Non-browser clients send no Origin and pass; a browser always sends one on
    a WebSocket handshake. Ports are deliberately not compared: the panel is
    reached through the same host:port it was served from, so the host alone
    settles same-site, and a proxy that only rewrites the port stays usable.
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
        if scope.get("type") == "websocket" and not _same_origin_websocket(scope):
            # A WebSocket handshake is not subject to the same-origin policy,
            # and NiceGUI's socket.io accepts any Origin -- so a page on any
            # website the operator visits could open the control channel to
            # http://<lan-ip>:8080 and press its buttons. With no API key set
            # there is no cookie to be missing, so this check is the only
            # thing between "someone on my LAN" and "any site I browse".
            log.warning(
                "gui websocket refused: Origin does not match Host (cross-site upgrade)",
                origin=Headers(scope=scope).get("origin"),
                host=Headers(scope=scope).get("host"),
                hint=(
                    "a reverse proxy that rewrites the Host header must forward the "
                    "original one (proxy_set_header Host $host)"
                ),
            )
            await receive()
            await send({"type": "websocket.close", "code": 1008})
            return
        expected = self.config.server.api_key
        if not expected or scope.get("type") not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "/")
        headers = Headers(scope=scope)
        if path in _OPEN_PATHS or scope.get("method") == "OPTIONS":
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


_LOGIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StudioForge — sign in</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ font-family: system-ui, sans-serif; display: grid; place-items: center;
        min-height: 100vh; margin: 0; background: #11151c; color: #e8eaed; }}
 form {{ background: #1b212b; padding: 2rem; border-radius: 12px; width: min(24rem, 90vw);
        box-shadow: 0 10px 40px rgba(0,0,0,.4); }}
 h1 {{ font-size: 1.1rem; margin: 0 0 .25rem; }}
 p {{ font-size: .8rem; opacity: .7; margin: 0 0 1.25rem; }}
 input {{ width: 100%; padding: .6rem .7rem; font-size: 1rem; border-radius: 8px;
         border: 1px solid #333c4a; background: #11151c; color: inherit; box-sizing: border-box; }}
 button {{ margin-top: 1rem; width: 100%; padding: .6rem; font-size: 1rem; border: 0;
          border-radius: 8px; background: #4f7cff; color: #fff; cursor: pointer; }}
 .error {{ color: #ff8a80; font-size: .8rem; margin-top: .75rem; }}
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
    return _LOGIN_PAGE.format(login_path=LOGIN_PATH, error=block)


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


def _header(ctx: GuiContext) -> Any:
    with ui.header().classes("items-center justify-between px-4 py-2"):
        with ui.row().classes("items-center gap-3"):
            ui.icon("memory", size="1.6rem")
            ui.label("StudioForge").classes("text-lg font-semibold")
            ui.label(f"v{__version__}").classes("text-xs opacity-70")
        with ui.row().classes("items-center gap-2"):
            status = ui.label("").classes("text-xs opacity-80 font-mono")
            dark = ui.dark_mode(value=True)
            ui.button(icon="dark_mode", on_click=dark.toggle).props("flat round dense")
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
    return f"{len(loaded)} loaded · {gpu_text}"


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

    if not _NICEGUI_MOUNTED:
        ui.run_with(
            app,
            title="StudioForge",
            favicon="🔥",
            dark=True,
            reconnect_timeout=10.0,
            show_welcome_message=False,
            # Storage is keyed per browser session; the secret is derived from
            # the data dir so it survives a restart without being guessable
            # from the outside.
            storage_secret=hashlib.sha256(
                f"studioforge-gui::{config.data_dir}".encode()
            ).hexdigest(),
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
