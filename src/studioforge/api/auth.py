"""Optional bearer-token auth, shared by the API, the GUI and the watchdog."""

from __future__ import annotations

import hmac
from typing import Any

from fastapi import Request

from studioforge.config import SECRET_CONFIG_PATHS, Config, redact, redact_config_dict

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


def extract_pin(request: Request) -> str | None:
    """The MCP pairing PIN, however the client chose to send it."""
    pin = request.headers.get("x-mcp-pin") or request.headers.get("x-studioforge-pin")
    if pin and pin.strip():
        return pin.strip()
    query = request.query_params.get("pin")
    return query.strip() if query and query.strip() else None


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


REMOTE_ADMIN_NOTE = (
    "this management route changes the server itself and server.api_key is not set, so it "
    "is only accepted from this machine or with the MCP pairing PIN (header 'X-MCP-Pin', "
    "or as the bearer token -- sfctl sends it that way). Set server.api_key on the Setup "
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

    if expected and provided and _matches(provided, expected):
        return

    if on_mcp and pin:
        candidate = extract_pin(request) or provided
        if candidate and _matches(candidate, str(pin)):
            return
        if pin_required:
            raise AuthError(
                "This MCP endpoint needs the pairing PIN. Send it as "
                "'X-MCP-Pin: <pin>', as a bearer token, or as ?pin=<pin>. The "
                "PIN is printed in the server's startup banner and available "
                "from GET /api/mcp/info.",
                code="invalid_mcp_pin",
            )

    if not expected:
        if is_admin_mutation(request.method, path) and not is_local_request(request):
            candidate = extract_pin(request) or provided
            if pin and candidate and _matches(candidate, str(pin)):
                return
            raise AuthError(
                REMOTE_ADMIN_NOTE, code="remote_admin_requires_credential", status_code=403
            )
        return
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
    handler directly) and is trusted -- it never crossed a network.
    """
    if config.server.api_key:
        return True
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
