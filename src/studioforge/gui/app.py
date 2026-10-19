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

**Auth mirrors the gateway's.** Whenever the server requires a credential
(:func:`studioforge.api.auth.auth_enforced`: ``server.api_key`` is set, any API
key exists, or a network bind has an owner account) the panel is gated too. A
browser gets a small sign-in page that takes the owner's username and password,
or an *admin* API key, and exchanges it for a signed cookie; an API client can
present the same ``Authorization: Bearer`` header the gateway accepts. On an open
install (loopback, no credentials) the page itself is open, which matches the
gateway's own default -- reads, chat, load and unload work for anyone who can
reach it -- but the buttons that change the *box* (config, files, engines,
restarts, downloads) follow the API's D32 rule from the viewer's peer address:
they work from a browser on this machine and refuse a remote one
(:func:`studioforge.gui.tabs.require_local_admin`). The cookie carries a
*derived*, signed token, never a key or a password, and ``secure`` is set only
when the panel is served over HTTPS (``server.tls.mode``).
"""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from nicegui import app as nicegui_app
from nicegui import ui
from starlette.datastructures import Headers

from studioforge import __version__
from studioforge.api.auth import GUARD, auth_enforced
from studioforge.authkeys import looks_like_key, scope_at_least
from studioforge.config import Config
from studioforge.credential_guard import client_key
from studioforge.errors import AuthError
from studioforge.gui import nav
from studioforge.gui import state as st
from studioforge.gui.state import UNKNOWN
from studioforge.gui.tabs import GuiContext, panel_guard
from studioforge.logging import get_logger
from studioforge.secure_files import write_private
from studioforge.tlsutil import scheme_for

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

#: The panel's top line: exactly four tabs (see :mod:`studioforge.gui.nav`, which also
#: holds the Advanced sub-navigation and every old tab/section name that still redirects).
TAB_NAMES = nav.TOP_TABS

#: Where the first-run wizard lives.
SETUP_PATH = "/setup"


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


#: How long an owner or key sign-in lasts.
SESSION_DAYS = 30
_SESSION_LABEL = "studioforge-gui-session-v2"


def _session_signature(secret: str, kind: str, ident: str, expiry: int, fingerprint: str) -> str:
    """HMAC-SHA256 over kind, identity, expiry and a fingerprint of the credential.

    The fingerprint is what ties a session to the credential it came from: the
    owner's changes whenever the password does, a key's is the head of its
    stored hash. Changing a password, or revoking or replacing a key, therefore
    ends every session opened with the old one -- with no server-side table.
    """
    message = "|".join((_SESSION_LABEL, kind, quote(ident, safe=""), str(expiry), fingerprint))
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def make_session_cookie(
    secret: str, kind: str, ident: str, fingerprint: str, *, now: float | None = None
) -> str:
    """``kind.ident.expiry.signature`` for an owner (``owner``) or key (``key``) sign-in."""
    expiry = int((time.time() if now is None else now) + SESSION_DAYS * 86400)
    quoted = quote(ident, safe="")
    return (
        f"{kind}.{quoted}.{expiry}.{_session_signature(secret, kind, ident, expiry, fingerprint)}"
    )


def _split_session_cookie(value: str) -> tuple[str, str, int, str] | None:
    """``(kind, ident, expiry, signature)`` or ``None`` for anything malformed."""
    try:
        head, signature = value.rsplit(".", 1)
        head, expiry_text = head.rsplit(".", 1)
        kind, quoted = head.split(".", 1)
        expiry = int(expiry_text)
    except ValueError:
        return None
    if kind not in {"owner", "key"} or not quoted or not signature:
        return None
    return kind, unquote(quoted), expiry, signature


def _key_record(accounts: Any, key_id: str) -> Any:
    for record in accounts.list_keys():
        if record.id == key_id:
            return record
    return None


def _key_fingerprint(record: Any) -> str:
    return str(record.key_hash)[:16]


def _session_actor(
    value: str, secret: str, accounts: Any, *, now: float | None = None
) -> str | None:
    """The actor a signed session cookie proves, or ``None``.

    Tolerates garbage (a stale cookie from another app on the same host is an
    ordinary "not signed in"), compares in constant time, and re-checks the
    credential behind the session on every call, so a disabled account, a
    revoked key or a downgraded scope ends the session at once.
    """
    parts = _split_session_cookie(value)
    if parts is None or accounts is None:
        return None
    kind, ident, expiry, signature = parts
    if expiry <= (time.time() if now is None else now):
        return None
    try:
        if kind == "owner":
            account = accounts.get_account(ident)
            if account is None or account.disabled:
                return None
            fingerprint = accounts.password_fingerprint(account.username)
            ident = account.username
            actor = f"owner:{account.username}"
        else:
            record = _key_record(accounts, ident)
            if record is None or not record.active() or record.scope != "admin":
                return None
            fingerprint = _key_fingerprint(record)
            actor = f"key:{record.id}"
    except Exception:  # noqa: BLE001 - a broken store means "not signed in", not a 500
        return None
    expected = _session_signature(secret, kind, ident, expiry, fingerprint)
    return actor if _safe_eq(signature, expected) else None


def _legacy_still_valid(accounts: Any, api_key: str) -> bool:
    """False when the operator revoked the ``legacy`` mirror of ``server.api_key``."""
    if accounts is None:
        return True
    try:
        return accounts.authenticate_legacy(api_key) is not None
    except AuthError:
        return False
    except Exception:  # noqa: BLE001 - bookkeeping trouble must not lock the owner out
        return True


def _admin_key_actor(accounts: Any, token: str) -> tuple[str | None, str | None]:
    """``(actor, problem)`` for a presented scoped key.

    ``actor`` when it is an active *admin* key; otherwise ``problem`` says why
    not ("scope", or the revoked/expired message), or both are ``None`` when the
    token is simply not one of ours.
    """
    if accounts is None or not looks_like_key(token):
        return None, None
    try:
        principal = accounts.authenticate(token, count=False)
    except AuthError as exc:
        return None, exc.message
    except Exception:  # noqa: BLE001
        return None, None
    if principal is None:
        return None, None
    if not scope_at_least(principal.scope, "admin"):
        return None, "scope"
    return principal.actor, None


def authorize(
    headers: Headers,
    config: Config,
    accounts: Any,
    secret_for: Callable[[], str],
) -> str | None:
    """The actor a request is signed in as (``legacy``, ``owner:<name>``, ``key:<id>``).

    ``None`` means not signed in. Only admin credentials open the panel: it can
    change the box, so a read or inference key is not enough.
    """
    legacy = config.server.api_key
    presented = _presented_key(headers)
    if presented:
        if legacy and _safe_eq(presented, legacy) and _legacy_still_valid(accounts, legacy):
            return "legacy"
        actor, _problem = _admin_key_actor(accounts, presented)
        if actor:
            return actor
    cookie = _cookie_value(headers)
    if not cookie:
        return None
    if legacy and _safe_eq(cookie, session_token(legacy)) and _legacy_still_valid(accounts, legacy):
        return "legacy"
    if "." in cookie:
        return _session_actor(cookie, secret_for(), accounts)
    return None


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

    def __init__(self, app: Any, config: Config, api_state: Any = None) -> None:
        self.app = app
        self.config = config
        self.api_state = api_state
        self._secret: str | None = None

    @property
    def accounts(self) -> Any:
        """The live account store; re-read per request (it may be attached late)."""
        return getattr(self.api_state, "accounts", None)

    def secret(self) -> str:
        """The panel's session-signing secret, read once on first need."""
        if self._secret is None:
            self._secret = _storage_secret(self.config)
        return self._secret

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
        if kind not in ("http", "websocket") or not auth_enforced(self.config, self.accounts):
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
        actor = authorize(headers, self.config, self.accounts, self.secret)
        if actor is not None:
            # Page code reads this back (``viewer_actor``) for the audit log.
            scope.setdefault("state", {})["panel_actor"] = actor
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
                            "This panel requires a sign-in. Open "
                            f"{LOGIN_PATH} in a browser, or send an admin API key as "
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
# gui/theme/ is the ForgeUI (ThemeForge) drop-in bundle, vendored whole by
# `python -m forgeui vendor` from the manifest at the repository root
# (forgeui.json; see docs/DEVELOPMENT.md's "GUI theming" section). Never
# hand-edited: StudioForge's own CSS is the overlay gui/overlays/
# studioforge-extra.css, which vendor copies to adapters/studioforge-extra.css.
# What is StudioForge's own is THEME_SETTINGS below, and it is read from
# forgeui.json so the theme choice lives in exactly one place.

#: The drop-in bundle: ui-theme.js/.css/-base.css, adapters/, themes.json.
_THEME_DIR = Path(__file__).parent / "theme"


def _load_forgeui_manifest() -> Mapping[str, Any]:
    """forgeui.json: beside this module in an installed wheel (hatch force-include),
    at the repository root in a source checkout or editable install."""
    here = Path(__file__).resolve().parent
    for candidate in (here / "forgeui.json", here.parents[2] / "forgeui.json"):
        if candidate.is_file():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    raise RuntimeError(
        "forgeui.json (the theme manifest) was not found next to the GUI package or at the "
        "repository root; the wheel must ship it (pyproject.toml force-include)"
    )


_FORGEUI_MANIFEST = _load_forgeui_manifest()

#: This app's own theme settings, read from forgeui.json ``settings``. The bundle
#: carries none: they ride as data-* attributes on the ui-theme.js <script> tag
#: (see _THEME_HEAD_HTML), and the header picker lists the same themes in the
#: same order.
THEME_SETTINGS: Mapping[str, Any] = {
    **_FORGEUI_MANIFEST["settings"],
    "themes": tuple(_FORGEUI_MANIFEST["settings"]["themes"]),
}

#: App-owned CSS overlays the manifest lists (extra_files), served from the bundle
#: folder after the Quasar adapter so they win source-order ties.
_THEME_OVERLAY_CSS: tuple[str, ...] = tuple(
    target for target in _FORGEUI_MANIFEST.get("extra_files", {}) if target.endswith(".css")
)


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
_QUASAR_JS_VERSION = _asset_version("adapters/quasar.js")
_QUASAR_CSS_VERSION = _asset_version("adapters/quasar.css")


def _theme_script_tag(name: str, version: str, attrs: str = "") -> str:
    return f'<script src="{_THEME_URL_PREFIX}{name}?v={version}"{attrs}></script>'


def _theme_settings_attrs() -> str:
    """THEME_SETTINGS as the runtime's data-* configuration."""
    attrs = (
        f' data-themes="{",".join(THEME_SETTINGS["themes"])}"'
        f' data-default="{THEME_SETTINGS["default"]}"'
        f' data-storage-key="{THEME_SETTINGS["storage_key"]}"'
    )
    for key in ("default_dark", "default_light"):
        if THEME_SETTINGS.get(key):
            attrs += f' data-{key.replace("_", "-")}="{THEME_SETTINGS[key]}"'
    return attrs


def _theme_link_tag(name: str, version: str) -> str:
    return f'<link rel="stylesheet" href="{_THEME_URL_PREFIX}{name}?v={version}">'


#: Injected once into every page's <head> (see _register_theme_assets and
#: _login_html), in this order: the runtime script -- blocking, so the
#: palette is on the page before first paint, and carrying THEME_SETTINGS --
#: then the Quasar adapter's script (it needs the runtime), the element
#: layer (loaded before the app's own CSS so its rules keep priority), the
#: contract tokens (loaded after it so they beat any same-named legacy
#: value), then the Quasar adapter's CSS, which maps the tokens onto
#: Quasar's components, and last the app-owned overlay(s) from forgeui.json.
_THEME_HEAD_HTML = (
    _theme_script_tag("ui-theme.js", _THEME_JS_VERSION, _theme_settings_attrs())
    + _theme_script_tag("adapters/quasar.js", _QUASAR_JS_VERSION)
    + _theme_link_tag("ui-theme-base.css", _THEME_BASE_CSS_VERSION)
    + _theme_link_tag("ui-theme.css", _THEME_CSS_VERSION)
    + _theme_link_tag("adapters/quasar.css", _QUASAR_CSS_VERSION)
    + "".join(_theme_link_tag(rel, _asset_version(rel)) for rel in _THEME_OVERLAY_CSS)
)


def _theme_registry() -> Mapping[str, Mapping[str, Any]]:
    """The bundle's theme registry (themes.json), keyed by slug."""
    try:
        data = json.loads((_THEME_DIR / "themes.json").read_text(encoding="utf-8"))
        return {theme["slug"]: theme for theme in data["themes"]}
    except (OSError, ValueError, KeyError, TypeError):  # pragma: no cover
        # Only if the vendored copy is missing or corrupt.
        return {}


_THEME_REGISTRY = _theme_registry()


def _theme_options() -> dict[str, str]:
    """``{slug: name}`` for the header picker, in THEME_SETTINGS order. A slug
    the bundle does not know is left out rather than shown without a name."""
    return {
        slug: str(_THEME_REGISTRY[slug]["name"])
        for slug in THEME_SETTINGS["themes"]
        if slug in _THEME_REGISTRY
    }


def _theme_default() -> str:
    return str(THEME_SETTINGS["default"])


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
    # R18: max_cache_age=0 -- ?v= hashes are computed once at import (see
    # _asset_version above), so a re-vendor with no restart would otherwise
    # keep this process serving the new bytes under the old query string for
    # up to an hour under NiceGUI's default max-age=3600. A restart is what
    # actually recomputes the hash (see docs/DEVELOPMENT.md); this just stops
    # the browser from compounding that with its own long cache meanwhile.
    nicegui_app.add_static_files(_THEME_URL_PREFIX.rstrip("/"), _THEME_DIR, max_cache_age=0)
    ui.add_head_html(_THEME_HEAD_HTML, shared=True)


_LOGIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StudioForge — sign in</title>
{theme_head}
<style>
 body {{ font-family: var(--font-sans); display: grid; place-items: center;
        min-height: 100vh; margin: 0; background: var(--surface-0); color: var(--text-primary); }}
 form {{ background: var(--surface-1); padding: 2rem; border-radius: var(--radius-lg);
        width: min(24rem, 90vw); box-shadow: var(--shadow-3);
        border: 1px solid var(--border-subtle); }}
 h1 {{ font-size: 1.1rem; margin: 0 0 .25rem; }}
 p {{ font-size: .8rem; color: var(--text-secondary); margin: 0 0 1.25rem; }}
 label {{ display: block; font-size: .8rem; margin: .9rem 0 .25rem;
         color: var(--text-secondary); }}
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
 <h1>Sign in to StudioForge</h1>
 <p>Use the owner username and password, or leave the username empty and paste an admin
    API key.</p>
 <label for="username">Username (leave empty to use an API key)</label>
 <input id="username" type="text" name="username" autocomplete="username"
        autocapitalize="none" spellcheck="false">
 <label for="credential">Password or API key</label>
 <input id="credential" type="password" name="credential" autofocus
        autocomplete="current-password" required>
 <button type="submit">Sign in</button>
 <div class="error" role="alert">{error}</div>
</form></body></html>
"""


def _login_html(error: str = "") -> str:
    return _LOGIN_PAGE.format(
        login_path=LOGIN_PATH, error=html.escape(error), theme_head=_THEME_HEAD_HTML
    )


#: Login failure texts. None of them echoes what was submitted.
_NOT_ACCEPTED = "That username and password, or that API key, was not accepted."
_NEEDS_ADMIN = "That key cannot open the control panel (it needs the admin scope)."


def _install_auth_routes(app: FastAPI, config: Config, api_state: Any = None) -> None:
    """Login/logout plus the gate. Registered before NiceGUI's catch-all mount."""

    def accounts() -> Any:
        return getattr(api_state, "accounts", None)

    def session_cookie(response: Response, value: str) -> Response:
        response.set_cookie(
            COOKIE_NAME,
            value,
            httponly=True,
            samesite="lax",
            secure=scheme_for(config) == "https",
            path="/",
            max_age=SESSION_DAYS * 24 * 3600,
        )
        return response

    def audit(actor: str, result: str, client: str | None) -> None:
        trail = getattr(api_state, "audit", None)
        if trail is None:
            return
        try:
            trail.record(
                actor=actor,
                action="panel.login",
                target="",
                result=result,
                surface="panel",
                peer=client,
            )
        except Exception:  # noqa: BLE001 - never fail a sign-in over bookkeeping
            return

    @app.get(LOGIN_PATH, include_in_schema=False)
    async def login_form() -> HTMLResponse:
        return HTMLResponse(_login_html())

    @app.post(LOGIN_PATH, include_in_schema=False)
    async def login_submit(
        request: Request,
        api_key: str = Form(""),
        credential: str = Form(""),
        username: str = Form(""),
    ) -> Response:
        if not _same_origin_scope(request.scope):
            # A cross-site page must not be able to sign a visitor in to an
            # account of its choosing (login CSRF), or probe credentials.
            return HTMLResponse(
                _login_html("Refused: the form was not submitted from this page."), status_code=403
            )
        store = accounts()
        if not auth_enforced(config, store):
            return RedirectResponse("/", status_code=303)
        client = client_key(request.client.host if request.client else None)
        wait = GUARD.retry_after(client)
        if wait > 0:
            seconds = max(1, math.ceil(wait))
            return HTMLResponse(
                _login_html(f"Too many incorrect attempts. Wait {seconds} seconds and try again."),
                status_code=429,
                headers={"Retry-After": str(seconds)},
            )
        who = username.strip()
        secret_text = credential or api_key
        problem = _NOT_ACCEPTED
        if who and store is not None:
            from fastapi.concurrency import run_in_threadpool

            account = await run_in_threadpool(store.verify_password, who, secret_text)
            if account is not None:
                GUARD.record_success(client)
                audit(f"owner:{account.username}", "ok", client)
                log.info("gui login accepted", kind="owner")
                value = make_session_cookie(
                    _storage_secret(config),
                    "owner",
                    account.username,
                    store.password_fingerprint(account.username),
                )
                return session_cookie(RedirectResponse("/", status_code=303), value)
        elif secret_text.strip():
            presented = secret_text.strip()
            legacy = config.server.api_key
            if legacy and _safe_eq(presented, legacy) and _legacy_still_valid(store, legacy):
                GUARD.record_success(client)
                audit("legacy", "ok", client)
                log.info("gui login accepted", kind="legacy")
                return session_cookie(RedirectResponse("/", status_code=303), session_token(legacy))
            actor, why = _admin_key_actor(store, presented)
            record = _key_record(store, actor.split(":", 1)[1]) if actor else None
            if actor and record is not None:
                GUARD.record_success(client)
                audit(actor, "ok", client)
                log.info("gui login accepted", kind="key")
                value = make_session_cookie(
                    _storage_secret(config), "key", record.id, _key_fingerprint(record)
                )
                return session_cookie(RedirectResponse("/", status_code=303), value)
            if why == "scope":
                # The key is genuine, not a guess, so it does not feed the lockout.
                return HTMLResponse(_login_html(_NEEDS_ADMIN), status_code=403)
            if why:
                problem = why
        # Never log the submitted value, correct or not.
        GUARD.record_failure(client)
        audit(f"owner:{who}" if who else "anonymous", "denied", client)
        log.info("gui login rejected")
        return HTMLResponse(_login_html(problem), status_code=401)

    @app.get(LOGOUT_PATH, include_in_schema=False)
    async def logout() -> Response:
        response = RedirectResponse(LOGIN_PATH, status_code=303)
        response.delete_cookie(COOKIE_NAME, path="/")
        return response

    @app.get("/gui-health", include_in_schema=False)
    async def gui_health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "gui": True}

    app.add_middleware(GuiAuthGate, config=config, api_state=api_state)


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
    the client connects, ``sync_initial`` installs the ``UITheme.onChange``
    bridge and has it immediately emit the current slug, in one
    fire-and-forget call -- so a slow or blocked round-trip can no longer
    leave the bridge uninstalled the way an awaited ``UITheme.current()``
    lookup first, subscribe second sequence could (R21). Both the initial
    correction and any later switch made in another tab (the runtime's
    storage listener) or from script arrive at ``mirror`` the same way. The
    ``mirroring`` guard stops those corrections from round-tripping through
    ``on_change`` -- NiceGUI fires the change handler on a programmatic
    ``select.value =`` assignment exactly the same as on a user pick.
    """
    options = _theme_options()
    mirroring = False

    select = (
        ui.select(options, value=_theme_default(), label="Theme")
        .props("dense outlined options-dense")
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

    def sync_initial() -> None:
        # R21: install the bridge and ask it to emit the current slug in one
        # fire-and-forget call -- nothing here awaits a JS round-trip, so a
        # slow or blocked client can no longer leave onChange unsubscribed.
        ui.run_javascript(
            "UITheme.onChange((d) => emitEvent("
            f"{json.dumps(_THEME_CHANGE_EVENT)}, d.slug)); "
            f"emitEvent({json.dumps(_THEME_CHANGE_EVENT)}, UITheme.current());"
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
            ui.link("setup guide", SETUP_PATH).classes("text-xs")
            if auth_enforced(ctx.config, getattr(ctx.api_state, "accounts", None)):
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


class Shell:
    """The four tabs for one browser page, each built the first time it is opened.

    A tab nobody opens costs nothing: no widgets, no polling timers. ``go`` is what the
    buttons inside a tab call to send the viewer somewhere else in the panel ("Open in
    Models", "Open logs") without a page reload.
    """

    def __init__(self, ctx: GuiContext, location: nav.Location, params: Mapping[str, Any]) -> None:
        self.ctx = ctx
        self.location = location
        self.params = params
        self.tabs: Any = None
        self.boxes: dict[str, Any] = {}
        self.handles: dict[str, Any] = {}
        self.built: set[str] = set()
        self.pending: dict[str, nav.Location] = {}
        self.ready = False

    def build(self) -> None:
        with (
            ui.tabs(on_change=lambda e: self._on_change(str(e.value)))
            .props("outside-arrows mobile-arrows")
            .classes("w-full") as tabs
        ):
            for name in nav.TOP_TABS:
                ui.tab(name)
        self.tabs = tabs
        with ui.tab_panels(tabs, value=self.location.tab).classes("w-full"):
            for name in nav.TOP_TABS:
                with ui.tab_panel(name).classes("p-0"):
                    self.boxes[name] = ui.column().classes("w-full gap-0")
        self.ensure(self.location.tab, self.pending.pop(self.location.tab, self.location))
        self.ready = True

    def go(self, location: nav.Location) -> None:
        """Open ``location`` (a tab, plus a page or filter inside it)."""
        if self.tabs.value == location.tab:
            self.ensure(location.tab, location)
        else:
            self.pending[location.tab] = location
            self.tabs.set_value(location.tab)

    def _on_change(self, tab: str) -> None:
        self.ensure(tab, self.pending.pop(tab, None))
        self._sync_url(tab)

    def ensure(self, tab: str, location: nav.Location | None = None) -> None:
        if tab not in self.boxes:
            return
        if tab in self.built:
            handle = self.handles.get(tab)
            if location is not None and hasattr(handle, "apply"):
                handle.apply(location)
            return
        self.built.add(tab)
        where = location or nav.Location(tab)
        with self.boxes[tab], panel_guard(f"The {tab} page"):
            self.handles[tab] = self._build_tab(tab, where)

    def _build_tab(self, tab: str, where: nav.Location) -> Any:
        ctx = self.ctx
        if tab == nav.STATUS:
            from studioforge.gui.tabs import status

            return status.render(ctx, self.go)
        if tab == nav.MODELS:
            from studioforge.gui.tabs import models_page

            return models_page.render(ctx, where, self.go, self.params)
        if tab == nav.TEST_CHAT:
            from studioforge.gui.tabs import chat, models_page

            drawer = models_page.ModelDrawer(ctx, self.go)
            chat.render(ctx, open_drawer=drawer.open, go=self.go)
            return drawer
        from studioforge.gui.tabs import advanced

        return advanced.render(ctx, where, self.go)

    def _sync_url(self, tab: str) -> None:
        if not self.ready or tab == nav.ADVANCED:
            return  # the hub rewrites the address with its page
        url = nav.canonical_url(nav.Location(tab))
        try:
            ui.run_javascript(f"history.replaceState(null, '', {json.dumps(url)})")
        except Exception:  # noqa: BLE001 - no client (a test): nothing to rewrite
            return


def _render_index(query: Mapping[str, Any] | None = None) -> None:
    """Build the single-page shell: header, the four tabs.

    ``query`` carries the deep-link intent: ``?section=Models&model=<id>``, the old
    ``?tab=download&repo=owner/repo`` the protocol handler produces, or a bare ``/``
    (Status). Old names are redirected before this runs (see :func:`_redirect_for`).
    """
    ctx = _context()
    ui.page_title("StudioForge")
    status = _header(ctx)
    params = st.deep_link_params(query)
    shell = Shell(ctx, nav.resolve_location(query), params)
    shell.build()

    def tick() -> None:
        status.set_text(_status_line(ctx))

    tick()
    ui.timer(max(2.0, ctx.refresh_interval), tick)


def _redirect_for(query: Mapping[str, Any]) -> Response | None:
    """A 307 to the canonical form when the link uses a name from an older panel."""
    if not nav.needs_redirect(query):
        return None
    return RedirectResponse(nav.canonical_url(nav.resolve_location(query)), status_code=307)


def _register_pages() -> None:
    global _PAGES_REGISTERED
    if _PAGES_REGISTERED:
        return
    _PAGES_REGISTERED = True

    @ui.page("/")
    def index(request: Request) -> Response | None:
        # NiceGUI injects the Starlette request when the page declares it,
        # which is how the deep-link query string reaches the tabs.
        query = dict(request.query_params)
        redirect = _redirect_for(query)
        if redirect is not None:
            return redirect
        _render_index(query)
        return None

    @ui.page(SETUP_PATH)
    def setup_page() -> None:
        # Behind the same gate as the panel itself (see GuiAuthGate).
        from studioforge.gui.tabs import setup_wizard_page

        setup_wizard_page.render(_context())


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
        # Owner-only on POSIX (0600) and on Windows (an owner-only ACL).
        write_private(path, value)
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
    _install_auth_routes(app, config, api_state)
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

    log.info(
        "gui app created",
        port=config.gui.port,
        auth=auth_enforced(config, getattr(api_state, "accounts", None)),
    )
    return app


__all__ = [
    "COOKIE_NAME",
    "LOGIN_PATH",
    "LOGOUT_PATH",
    "SETUP_PATH",
    "TAB_NAMES",
    "GuiAuthGate",
    "authorize",
    "create_gui_app",
    "make_session_cookie",
    "session_token",
]
