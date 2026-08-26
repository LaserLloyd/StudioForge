"""Optional bearer-token auth, shared by the API, the GUI and the watchdog."""

from __future__ import annotations

import hmac
import math
from typing import Any

from fastapi import Request

from studioforge.config import SECRET_CONFIG_PATHS, Config, redact, redact_config_dict
from studioforge.credential_guard import CredentialGuard, client_key

# Re-exported: these live in ``studioforge.config`` (a leaf the watchdog may import)
# but every API/MCP surface has always taken them from here.
__all__ = ["SECRET_CONFIG_PATHS", "redact", "redact_config_dict"]
from studioforge.errors import AuthError

# Endpoints reachable without a key: liveness probes and CORS preflight. Keeping
# /health open lets the watchdog and load balancers check the server without
# being handed a credential.
PUBLIC_PATHS = frozenset({"/health", "/healthz", "/api/health", "/favicon.ico"})

#: Query values meaning "false" for the /health ``deep`` flag. Anything else
#: (including junk) fails closed: ``/health?deep=true`` runs a real completion
#: against every loaded model, which must never be reachable without the key.
_FALSY = frozenset({"", "0", "false", "no", "off"})

#: Process-wide lockout for repeated wrong credentials. One instance for the
#: whole API: the thing being protected is a single eight-digit PIN, so a
#: per-router or per-request counter would count nothing.
GUARD = CredentialGuard()


def _client_of(request: Any) -> str | None:
    client = getattr(request, "client", None)
    return client_key(getattr(client, "host", None) if client else None)


def _throttled(client: str | None) -> AuthError | None:
    """The 429 to raise if this client is locked out, else ``None``."""
    wait = GUARD.retry_after(client)
    if wait <= 0:
        return None
    seconds = max(1, math.ceil(wait))
    return AuthError(
        f"Too many incorrect credentials from this address. Wait {seconds}s and try again. "
        "This lockout exists because the MCP pairing PIN is only eight digits: without it the "
        "whole keyspace is reachable by one machine in hours. Set server.api_key for a real "
        "credential, or run from the server's own machine.",
        code="too_many_credential_attempts",
        status_code=429,
        details={"retry_after_s": seconds},
    )


def _matches(provided: str, expected: str) -> bool:
    """Constant-time comparison that cannot 500 on odd header bytes.

    Starlette decodes headers as latin-1, so a byte >= 0x80 yields a non-ASCII
    ``str`` -- and ``hmac.compare_digest`` raises ``TypeError`` for those,
    which would escape the auth middleware as a 500. Comparing bytes keeps the
    constant-time property and turns any such probe into an ordinary 401.
    """
    return hmac.compare_digest(
        provided.encode("utf-8", "surrogateescape"),
        expected.encode("utf-8", "surrogateescape"),
    )


def extract_key(request: Request) -> str | None:
    """Accept the key the way every OpenAI-compatible client sends it."""
    header = request.headers.get("authorization")
    if header:
        prefix, _, value = header.partition(" ")
        if prefix.lower() == "bearer" and value:
            return value.strip()
        if header.strip() and not _:
            return header.strip()
    api_key = request.headers.get("x-api-key")
    if api_key:
        return api_key.strip()
    return None


#: Rejected, not merely unadvertised. ``?pin=`` used to be accepted so that an
#: MCP connector configurable only by URL could pair; the cost was that the
#: credential then sat in reverse-proxy access logs, browser history, shell
#: history, referrer headers and any crash report -- places nothing ever
#: expires from, for a secret with no expiry either. A header costs a
#: connector one configuration field. This constant exists so the 401 can say
#: *why* a URL that used to work now does not, instead of looking like a wrong
#: PIN.
PIN_IN_QUERY_NOTE = (
    "The MCP pairing PIN is no longer accepted as a '?pin=' query parameter: a URL is "
    "written to access logs, shell history and browser history, which is no place for a "
    "credential. Send it as the 'X-MCP-Pin' header or as 'Authorization: Bearer <pin>'."
)


def pin_in_query(request: Any) -> bool:
    """Whether the caller tried the retired ``?pin=`` form."""
    params = getattr(request, "query_params", None)
    if params is None:
        return False
    value = params.get("pin")
    return bool(value and value.strip())


def extract_pin(request: Request) -> str | None:
    """The MCP pairing PIN, from a header or the bearer slot -- never a URL.

    See :data:`PIN_IN_QUERY_NOTE` for why the query form is refused rather
    than quietly deprecated.
    """
    pin = request.headers.get("x-mcp-pin") or request.headers.get("x-studioforge-pin")
    if pin and pin.strip():
        return pin.strip()
    return None


def is_mcp_path(path: str, config: Config) -> bool:
    mcp_path = getattr(getattr(config, "mcp", None), "path", "/mcp") or "/mcp"
    return path == mcp_path or path.startswith(mcp_path.rstrip("/") + "/")


#: Management routes that change the box -- its config, its files, its
#: processes -- rather than its models' residency. On an install with no
#: ``server.api_key`` these need a caller on this machine or the MCP PIN
#: (D32); everything else stays open for LM Studio parity (JIT loading from any
#: client on the LAN is the product). Prefix-matched on the path, and only for
#: mutating methods.
_ADMIN_MUTATION_PREFIXES: tuple[str, ...] = (
    "/api/config",
    "/api/restart/",
    "/api/engine/",
    "/api/update/",
    "/api/vram/reclaim",
    "/api/downloads",
    "/api/leases",
)
#: Deletions of things on disk / in the registry: same rule, DELETE only.
_ADMIN_DELETE_PREFIXES: tuple[str, ...] = (
    "/api/models/",
    "/api/adapters/",
    "/api/virtual-models/",
)
#: Per-model writes that persist across restarts: saved settings and the pin.
#: Residency stays open (load/unload from the LAN is the product), but these
#: two outlive the instance -- a pin drives the boot autoload and the
#: reconciler (D41), and saved settings shape every future load -- so they are
#: box changes under D32, matched as ``(verb, path suffix)`` under /api/models/.
_ADMIN_SETTINGS_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("POST", "/pin"),
    ("PUT", "/settings"),
    ("PATCH", "/settings"),
)
_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def is_admin_mutation(method: str, path: str) -> bool:
    """Whether ``method path`` is one of the routes D32 guards on an open install."""
    verb = (method or "").upper()
    if verb not in _MUTATING_METHODS:
        return False
    if any(path == p.rstrip("/") or path.startswith(p) for p in _ADMIN_MUTATION_PREFIXES):
        return True
    if path.startswith("/api/models/") and any(
        verb == m and path.endswith(s) for m, s in _ADMIN_SETTINGS_SUFFIXES
    ):
        return True
    return verb == "DELETE" and any(path.startswith(p) for p in _ADMIN_DELETE_PREFIXES)


def is_local_request(request: Any) -> bool:
    """A caller on this machine, or an in-process call (no peer at all).

    The in-process case is the GUI invoking a route handler directly and the
    tests' ``TestClient`` shims; both are trusted the way ``may_reveal_pin``
    trusts them. Behind a reverse proxy the peer is the proxy -- a documented
    limit of any peer-address check; put the proxy behind ``server.api_key``.
    """
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    if not host:
        return True
    return _is_loopback(str(host))


_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def _origin_key(value: str, scheme: str) -> tuple[str, int] | None:
    """``(hostname, port)`` for an origin or a ``Host`` header, defaults filled in.

    Both are parsed with ``urlsplit`` so IPv6 brackets, case and an implicit
    port are handled one way. ``None`` means unparseable, which the caller
    treats as foreign.
    """
    from urllib.parse import urlsplit

    text = (value or "").strip()
    if not text:
        return None
    try:
        parts = urlsplit(text if "//" in text else "//" + text)
        hostname = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        return None
    if not hostname:
        return None
    if port is None:
        port = _DEFAULT_PORTS.get((parts.scheme or scheme or "http").lower(), 80)
    return hostname, port


def cross_site_browser_request(request: Any) -> bool:
    """True for a browser request whose ``Origin`` is not this server's own origin.

    The D32 gate trusts a loopback peer -- and the operator's browser *is* a
    loopback peer. With the shipped ``cors_origins: ["*"]`` any page the
    operator visits can preflight and send ``PATCH /api/config`` to
    ``http://127.0.0.1:1234`` and arrive looking local, so a peer-address
    check alone leaves every box-changing route (and the PIN reveal) one
    malicious tab away. The GUI's websocket gate already refuses a cross-site
    upgrade for exactly this reason; this is the same rule for the API.

    The comparison includes the port, unlike the websocket gate: a page served
    by another local web app (a ComfyUI custom node, a dev server) is also a
    loopback peer, and it is not this server. Non-browser clients send no
    ``Origin`` and pass; so does an in-process call with no headers at all.
    ``Origin: null`` (sandboxed frames, some redirects) is foreign.
    """
    headers = getattr(request, "headers", None)
    if headers is None:
        return False
    origin = headers.get("origin")
    host = headers.get("host")
    if not origin or not host:
        return False
    if origin.strip().lower() == "null":
        return True
    scheme = str(getattr(getattr(request, "url", None), "scheme", "") or "http")
    origin_key = _origin_key(origin, scheme)
    host_key = _origin_key(host, scheme)
    if origin_key is None or host_key is None:
        return True
    return origin_key != host_key


REMOTE_ADMIN_NOTE = (
    "this management route changes the server itself and server.api_key is not set, so it "
    "is only accepted from this machine or with the MCP pairing PIN (header 'X-MCP-Pin', "
    "or as the bearer token -- sfctl sends it that way). A browser request from another "
    "origin is not 'this machine', even on loopback. Set server.api_key on the Setup "
    "tab to manage the server remotely with a real credential."
)


def check_request(request: Request, config: Config) -> None:
    """Raise :class:`AuthError` unless the request is authorized.

    Two credentials are accepted, and which one works depends on the path:

    * ``server.api_key`` -- the full credential, valid everywhere.
    * ``mcp.pin`` -- a short pairing code valid **only on the MCP path**. It
      exists so a client can be paired by reading eight digits off the startup
      banner instead of copying a long key around, and it is deliberately
      scoped: it grants the management tools, never inference or the rest of
      the API.

    When ``server.api_key`` is unset the server is open (LAN/Tailscale trust),
    which matches how LM Studio behaves out of the box -- but an MCP PIN is
    still enforced when one is configured, so the control plane is never
    accidentally the *least* protected surface. And on that open install the
    routes that change the *box* -- edit config, delete files, restart,
    install an engine or an update, queue downloads, kill processes -- need a
    caller on this machine or the PIN (D32): the MCP ``set_config`` tool
    demanded the PIN while ``PATCH /api/config`` -- same capability, same
    process -- demanded nothing, and anyone on the LAN could set
    ``server.api_key`` and lock the owner out.
    """
    path = request.url.path
    if request.method == "OPTIONS":
        return
    if path in PUBLIC_PATHS:
        # The liveness form stays credential-free for watchdogs and load
        # balancers -- but /health?deep=true is genuine inference against
        # every loaded model, so it needs the key like any other request.
        deep = (request.query_params.get("deep") or "").strip().lower()
        if deep in _FALSY:
            return

    mcp_config = getattr(config, "mcp", None)
    on_mcp = is_mcp_path(path, config)
    pin = getattr(mcp_config, "pin", None) if mcp_config else None
    pin_required = bool(getattr(mcp_config, "pin_required", False)) and bool(pin)

    expected = config.server.api_key
    provided = extract_key(request)
    header_pin = extract_pin(request)

    # Only credentialed traffic is throttled. An open install serves plenty of
    # requests that carry nothing at all (that is the LM Studio parity the
    # product depends on), and counting those would let a chatty poller lock
    # the operator's own address out of a server neither of them was attacking.
    client = _client_of(request) if (provided or header_pin) else None
    if client is not None:
        throttled = _throttled(client)
        if throttled is not None:
            # Refused before any comparison, so a correct guess that lands
            # during a lockout still buys nothing.
            raise throttled

    if expected and provided and _matches(provided, expected):
        GUARD.record_success(client)
        return

    if on_mcp and pin:
        candidate = header_pin or provided
        if candidate and _matches(candidate, str(pin)):
            GUARD.record_success(client)
            return
        if pin_required:
            if candidate:
                GUARD.record_failure(client)
            note = " " + PIN_IN_QUERY_NOTE if pin_in_query(request) else ""
            raise AuthError(
                "This MCP endpoint needs the pairing PIN. Send it as "
                "'X-MCP-Pin: <pin>' or as a bearer token. Read it ON THE SERVER: "
                "the startup banner, `studioforge config`, or the control panel "
                "(Setup -> Network & access). GET /api/mcp/info returns 'pin': null "
                "to a caller in your position -- it only reveals the PIN to someone "
                "who is already on the box or already authenticated, which is the "
                "point of it being a credential." + note,
                code="invalid_mcp_pin",
            )

    if not expected:
        # D32 covers the MCP plane too. Every streamable-HTTP JSON-RPC call is
        # a POST, and the tools behind it (set_config, delete_model,
        # download_model, reserve_gpus) are the same box changes the HTTP
        # routes gate -- so with no key, a remote caller needs the PIN here
        # even when `mcp.pin_required` is off. That toggle used to be the one
        # way to make the control plane the *least* protected surface: on an
        # open install it opened set_config to the LAN with no credential at
        # all. It now relaxes same-machine callers only. The GET side (the SSE
        # stream) stays open, like every other read.
        gated = is_admin_mutation(request.method, path) or (
            on_mcp and (request.method or "").upper() in _MUTATING_METHODS
        )
        if gated and (not is_local_request(request) or cross_site_browser_request(request)):
            candidate = header_pin or provided
            if pin and candidate and _matches(candidate, str(pin)):
                GUARD.record_success(client)
                return
            if candidate:
                GUARD.record_failure(client)
            note = " " + PIN_IN_QUERY_NOTE if pin_in_query(request) else ""
            raise AuthError(
                REMOTE_ADMIN_NOTE + note,
                code="remote_admin_requires_credential",
                status_code=403,
            )
        return
    if provided:
        GUARD.record_failure(client)
    raise AuthError(
        "Incorrect API key provided. Set the same key the server has in "
        "server.api_key, or send it as 'Authorization: Bearer <key>'."
    )


def _is_loopback(host: str) -> bool:
    import ipaddress

    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def may_reveal_pin(request: Any, config: Config) -> bool:
    """Whether this caller may be handed the MCP pairing PIN in full.

    ``/api/mcp/info`` and ``/api/openclaw-setup`` return ``mcp.pin`` verbatim so
    that pairing a client is one request. That reasoning ("the caller is already
    authenticated to reach this route") holds only when ``server.api_key`` is
    set -- and the shipped default leaves it unset, which is precisely the
    install where the PIN is the *only* credential guarding the MCP control
    plane. On such a box, bound to ``0.0.0.0`` by default, anything on the LAN
    could read the PIN off an open endpoint and then use it to load, unload,
    delete and download models. The PIN was theatre exactly when it mattered.

    So: reveal when a credential was actually required to get here, and
    otherwise only to a caller on this machine. A request with no peer at all is
    an in-process call (the GUI renders these panels by invoking the route
    handler directly) and is trusted -- it never crossed a network. A browser
    request from another origin is refused even on loopback: with
    ``Access-Control-Allow-Origin: *`` the response body is readable
    cross-origin, so a page the operator visits could otherwise read the PIN
    off ``http://127.0.0.1:1234/api/mcp/info`` and then use it.
    """
    if config.server.api_key:
        return True
    if cross_site_browser_request(request):
        return False
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    if not host:
        return True
    return _is_loopback(str(host))


#: What to say instead of the PIN when it is withheld. Actionable on purpose:
#: the two ways to read it locally, and the one setting that makes this endpoint
#: safe to ask remotely.
PIN_WITHHELD_NOTE = (
    "withheld: server.api_key is not set, so this endpoint requires no "
    "credential and returning the PIN to a remote caller would defeat it. Read "
    "the PIN from the server's startup banner or `studioforge config` on the "
    "host itself, or set server.api_key and ask again."
)


def auth_dependency(config: Config) -> Any:
    async def _dependency(request: Request) -> None:
        check_request(request, config)

    return _dependency
