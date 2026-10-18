"""Authentication and authorization for the API, the control panel and MCP.

Credentials, in the order they are tried:

1. ``server.api_key`` -- the legacy single shared key (**deprecated**). It is an
   ``admin`` credential, mirrored into the key table as the key named ``legacy``.
2. A scoped API key (``sf-<r|i|a>-...``), stored hashed: ``read`` < ``inference``
   < ``admin``, optional expiry and per-minute limit (``core/accounts.py``).
3. ``mcp.pin`` -- the eight-digit pairing code (**deprecated**). Accepted on the
   MCP path only, as an ``admin`` credential, with a logged warning.

Whether a credential is *required* is a posture, not a flag: it is when
``server.api_key`` is set, when any key exists, or when any of the three services
binds a non-loopback address (:func:`auth_enforced`). A loopback-only install
with no keys stays open, exactly as before, so a first run needs no setup. And
a server that would listen on the network with no owner account or admin key
refuses to start (:func:`check_bind_rule`).

When a credential *is* presented its scope is enforced whatever the posture:
:func:`required_scope` maps every route to ``read``, ``inference`` or ``admin``.
"""

from __future__ import annotations

import hmac
import math
import time
from typing import Any

from fastapi import Request

from studioforge.authkeys import (
    LOCAL_ADMIN,
    Principal,
    is_loopback_host,
    looks_like_key,
    origin_allowed,
)
from studioforge.config import SECRET_CONFIG_PATHS, Config, redact, redact_config_dict
from studioforge.credential_guard import CredentialGuard, client_key

# Re-exported: these live in ``studioforge.config`` (a leaf the watchdog may import)
# but every API/MCP surface has always taken them from here.
__all__ = [
    "LOCAL_ADMIN",
    "SECRET_CONFIG_PATHS",
    "origin_allowed",
    "redact",
    "redact_config_dict",
]
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
    # LM Studio's native twin of POST /api/downloads: same disk write, same gate.
    "/api/v1/models/download",
    "/api/leases",
    # W3: accounts, keys, the support bundle and the first-run wizard decide who
    # may use the server -- the most box-changing routes there are.
    "/api/keys",
    "/api/users",
    "/api/support-bundle",
    "/api/setup",
    # D77: registering, deregistering or warming a tenant is the same class of
    # change as taking a lease -- it names URLs this server will then call.
    "/api/tenants",
    # D55: creating one persists a registry row carrying a ``system_prompt``
    # that the gateway then applies to every request routed to that id, and the
    # id appears in ``/v1/models``. Deleting one was already gated by
    # ``_ADMIN_DELETE_PREFIXES``; planting one was not, so an open install could
    # be given a restart-surviving system-prompt backdoor over any base model by
    # anyone who could reach the port. Creation is the box change here.
    "/api/virtual-models",
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

#: Exact paths that are POSTs by shape but reads by effect, and so stay open
#: even under an admin prefix. Checked FIRST, ahead of the prefix match,
#: because the prefix rule is deliberately body-blind and path-only: that is
#: the right coarseness for ``/api/engine/install`` (600 MB onto this disk) and
#: the wrong one for ``/api/engine/validate-flags``, which execs the engine's
#: own ``--help`` and compares strings. Nothing on the box changes, so gating it
#: only stopped a remote operator checking expert flags *before* saving them --
#: while the save itself stays gated by ``_ADMIN_SETTINGS_SUFFIXES``, which is
#: where the box change actually happens. D49.
_OPEN_POST_PATHS = frozenset({"/api/engine/validate-flags"})


def is_admin_mutation(method: str, path: str) -> bool:
    """Whether ``method path`` is one of the routes D32 guards on an open install.

    ``_OPEN_POST_PATHS`` wins over the prefix match: a POST that only reads is
    not a box change, whatever prefix it happens to live under (D49).
    """
    verb = (method or "").upper()
    if verb not in _MUTATING_METHODS:
        return False
    if path in _OPEN_POST_PATHS:
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


#: The D32 refusal for a body-gated action rather than a whole route. Same
#: three ways out as :data:`REMOTE_ADMIN_NOTE`, worded for a request whose
#: other fields are perfectly acceptable: the caller is told the route itself
#: stays open, so dropping the one field is a working retry.
REMOTE_ADMIN_ACTION_NOTE = (
    "server.api_key is not set, so a field that changes the server's own saved state is "
    "only accepted from this machine or with the MCP pairing PIN (header 'X-MCP-Pin', or "
    "as the bearer token -- sfctl sends it that way). A browser request from another origin "
    "is not 'this machine', even on loopback. Set server.api_key on the Setup tab to manage "
    "the server remotely with a real credential. The rest of this route stays open: the same "
    "request without that field is accepted."
)


def is_admin_caller(request: Any, config: Config) -> bool:
    """D32's "may change the box" test, as a boolean rather than a refusal (D55).

    Same three acceptances as :func:`require_admin_action`, in the same order:
    ``server.api_key`` configured (the middleware already authenticated this
    request), a caller on this machine that is not a cross-site browser page,
    or the MCP pairing PIN.

    Used where the answer *shapes* a response rather than refusing it -- how
    much of a log or a process listing this caller may see, whether a
    lease-held model may be unloaded. Deliberately records nothing in the
    credential lockout: these callers are not attempting a credential, they are
    being measured, and counting a read against the operator's own address
    would lock them out of a server nobody was attacking.
    """
    established = principal_of(request)
    if established is not None and established.kind in _CREDENTIAL_KINDS:
        # A scoped key is only as powerful as its scope: an inference key on
        # this machine is not an admin because it is on this machine.
        return established.allows("admin")
    if config.server.api_key:
        return True
    if is_local_request(request) and not cross_site_browser_request(request):
        return True
    mcp_config = getattr(config, "mcp", None)
    pin = getattr(mcp_config, "pin", None) if mcp_config else None
    if not pin:
        return False
    candidate = extract_pin(request) or extract_key(request)
    return bool(candidate and _matches(candidate, str(pin)))


def require_admin_action(request: Request, config: Config, action: str) -> None:
    """D32 for an admin action the path-based middleware cannot see (D48).

    :func:`is_admin_mutation` decides from ``method`` + ``path`` alone, which is
    everything it can know before the body is read. Some routes are deliberately
    open as a whole and carry ONE optional flag that writes settings --
    ``POST /api/models/{id}/load-recommended`` with ``persist: true`` is the
    first: residency stays open for LM Studio parity, but the persisted profile
    outlives the instance exactly like the ``PUT /settings`` that
    ``_ADMIN_SETTINGS_SUFFIXES`` guards. A handler calls this once it has parsed
    the body and knows the box change is really being asked for.

    Same three acceptances as the middleware's D32 branch, in the same order:
    ``server.api_key`` configured means :func:`check_request` already
    authenticated this request and there is nothing left to prove; otherwise a
    caller on this machine that is not a cross-site browser page; otherwise the
    MCP pairing PIN. Anything else is the existing 403
    ``remote_admin_requires_credential``, with ``action`` naming the thing being
    refused so the message is about the flag, not the route.

    Args:
        request: the live request -- its peer, ``Origin`` and headers are read.
        config: the server config, for ``server.api_key`` and ``mcp.pin``.
        action: what is being refused, as a noun phrase for the message
            ("persisting the resolved profile to this model's settings").

    Raises:
        AuthError: 403 ``remote_admin_requires_credential``.
    """
    established = principal_of(request)
    if established is not None and established.kind in _CREDENTIAL_KINDS:
        if established.allows("admin"):
            return
        raise deny_scope(established, "admin", action)
    if config.server.api_key:
        return
    if is_local_request(request) and not cross_site_browser_request(request):
        return

    mcp_config = getattr(config, "mcp", None)
    pin = getattr(mcp_config, "pin", None) if mcp_config else None
    candidate = extract_pin(request) or extract_key(request)
    client = _client_of(request) if candidate else None
    if pin and candidate and _matches(candidate, str(pin)):
        GUARD.record_success(client)
        return
    # Fed into the same process-wide lockout the middleware consults on the way
    # in, so a wrong PIN costs the same here as it does on a gated path.
    if candidate:
        GUARD.record_failure(client)
    note = " " + PIN_IN_QUERY_NOTE if pin_in_query(request) else ""
    raise AuthError(
        f"Refusing {action}: " + REMOTE_ADMIN_ACTION_NOTE + note,
        code="remote_admin_requires_credential",
        status_code=403,
    )


# ---------------------------------------------------------------------------
# Scopes, posture and the bind rule
# ---------------------------------------------------------------------------

#: Paths that expose or change who may log in. Admin even to read.
_ADMIN_READ_PREFIXES: tuple[str, ...] = (
    "/api/keys",
    "/api/users",
    "/api/audit",
    "/api/doctor",
    "/api/support-bundle",
)
#: Per-model writes an ``inference`` key may make: loading is what a first
#: request does anyway (JIT), and ``test`` is one short completion.
_INFERENCE_MODEL_SUFFIXES: tuple[str, ...] = ("/load", "/load-recommended", "/unload", "/test")
#: Browser-reachable inference surfaces (prefix-matched, so routes added later
#: -- LM Studio's ``/api/v1``, Anthropic's ``/v1/messages`` -- are covered
#: without an edit here).
_INFERENCE_PREFIXES: tuple[str, ...] = ("/v1/", "/api/v0/", "/api/v1/")


def required_scope(
    method: str, path: str, config: Config | None = None, *, deep: bool = False
) -> str:
    """The weakest scope that may call ``method path``: ``read``, ``inference`` or ``admin``.

    * ``/mcp``: ``read`` here -- each tool then demands its own scope
      (:mod:`studioforge.mcp_policy`).
    * Reads (GET/HEAD) are ``read``, except the account/audit/doctor pages.
    * ``/v1/*``, ``/api/v0/*``, ``/api/v1/*`` POSTs (chat, completions, embeddings,
      native load/unload) are ``inference``; the native model download is ``admin``.
    * Every route the D32 prefix list guards is ``admin``, and so is every
      other mutating ``/api`` route, except loading/unloading/testing a model.
    * ``/health?deep=true`` runs a real completion per model: ``inference``.
    """
    verb = (method or "").upper()
    if config is not None and is_mcp_path(path, config):
        return "read"
    if path in PUBLIC_PATHS:
        return "inference" if deep else "read"
    if verb in {"GET", "HEAD", "OPTIONS"}:
        return "admin" if path.startswith(_ADMIN_READ_PREFIXES) else "read"
    if is_admin_mutation(verb, path):
        return "admin"
    if path.startswith("/api/v1/models/download"):
        return "admin"
    if path.startswith(_INFERENCE_PREFIXES):
        return "inference"
    if (
        path.startswith("/api/models/")
        and verb == "POST"
        and path.endswith(_INFERENCE_MODEL_SUFFIXES)
    ):
        return "inference"
    return "admin"


def remote_services(config: Config) -> list[tuple[str, str]]:
    """``(service, host)`` for every enabled listener that is not loopback-only."""
    found: list[tuple[str, str]] = []
    if not is_loopback_host(config.server.host):
        found.append(("the gateway", config.server.host))
    if config.gui.enabled and not is_loopback_host(config.gui.host):
        found.append(("the control panel", config.gui.host))
    if config.watchdog.enabled and not is_loopback_host(config.watchdog.host):
        found.append(("the watchdog", config.watchdog.host))
    return found


def auth_enforced(config: Config, accounts: Any = None) -> bool:
    """Whether a request must carry a valid credential.

    True when ``server.api_key`` is set, when any active API key exists, or when
    a listener is reachable from the network *and* an owner account exists or
    a key ever did (revoking the last key at runtime must not reopen a
    network-bound server: the bind rule is only checked at start).
    False for a loopback-only install with no credentials (the first-run state)
    -- and for a network-bound one that never had any, a state
    ``studioforge serve`` refuses to start in (:func:`check_bind_rule`), so it
    is only reachable by embedding the app, where the legacy open rules (D32)
    still apply.
    """
    if config.server.api_key:
        return True
    if accounts is None:
        return False
    if accounts.has_active_keys_cached():
        return True
    if not remote_services(config):
        return False
    had_keys = getattr(accounts, "had_keys_cached", None)
    return bool(accounts.has_owner_cached() or (callable(had_keys) and had_keys()))


def has_admin_credential(config: Config, accounts: Any = None) -> bool:
    """An owner account, an active admin key, or the (deprecated) ``server.api_key`` exists."""
    if config.server.api_key:
        return True
    return bool(accounts is not None and accounts.has_admin_credential())


def check_bind_rule(config: Config, accounts: Any = None) -> None:
    """Refuse to start a network-reachable server that nobody has to log in to.

    Raises ``auth_required_for_remote_bind`` (catalogued) naming the exposed
    services and the two commands that fix it. Loopback-only binds always pass.
    """
    exposed = remote_services(config)
    if not exposed or has_admin_credential(config, accounts):
        return
    services = ", ".join(name for name, _host in exposed)
    hosts = ", ".join(host for _name, host in exposed)
    raise AuthError(
        f"Refusing to start: {services} would listen on a non-loopback address ({hosts}) but "
        "no owner account or admin API key exists, so anyone on the network could drive this "
        "server. Fix: run `studioforge setup` (creates the owner account) or "
        "`studioforge keys create --name admin --scope admin`, then start again. To stay "
        "local-only instead, set server.host, gui.host and watchdog.host to 127.0.0.1 "
        "(config.yaml).",
        code="auth_required_for_remote_bind",
        status_code=503,
        details={"exposed": [{"service": n, "host": h} for n, h in exposed]},
    )


def accounts_of(request: Any) -> Any:
    """The :class:`~studioforge.core.accounts.AccountStore` of the app serving ``request``."""
    try:
        app = request.app
    except (AttributeError, KeyError):  # a bare scope (tests, in-process calls) has no app
        return None
    return getattr(getattr(app, "state", None), "accounts", None)


def principal_of(request: Any) -> Principal | None:
    """The principal :func:`check_request` established for ``request``, if any."""
    state = getattr(request, "state", None)
    found = getattr(state, "principal", None) if state is not None else None
    return found if isinstance(found, Principal) else None


#: Principal kinds that came from a presented credential (not "no credential at all").
_CREDENTIAL_KINDS = frozenset({"key", "legacy", "owner", "pin"})

_PIN_WARN_INTERVAL_S = 600.0
_last_pin_warning = 0.0


def _warn_pin_deprecated() -> None:
    """One WARNING per ten minutes when the deprecated PIN authenticates a call."""
    global _last_pin_warning
    now = time.monotonic()
    if _last_pin_warning and now - _last_pin_warning < _PIN_WARN_INTERVAL_S:
        return
    _last_pin_warning = now
    from studioforge.logging import get_logger

    get_logger(__name__).warning(
        "a client authenticated with the deprecated mcp.pin; switch it to a scoped API key "
        "(`studioforge keys create --scope admin`) -- the PIN will stop being minted in a "
        "future release"
    )


def resolve_principal(
    provided: str | None,
    header_pin: str | None,
    config: Config,
    accounts: Any = None,
    *,
    on_mcp: bool = False,
    count: bool = True,
) -> Principal | None:
    """Who a presented credential belongs to, or ``None`` if it is nobody's.

    Order: ``server.api_key`` -> scoped key -> ``mcp.pin`` (MCP path only).
    Raises :class:`AuthError` for a stored key that cannot be used (revoked,
    expired, over its rate limit). ``count=False`` re-checks a credential the
    middleware already admitted without spending the key's rate-limit budget.
    """
    expected = config.server.api_key
    if expected and provided and _matches(provided, expected):
        if accounts is not None:
            return accounts.authenticate_legacy(expected)
        return Principal("legacy", "", "legacy", "admin")
    if provided and accounts is not None and looks_like_key(provided):
        found = accounts.authenticate(provided, count=count)
        if found is not None:
            return found
    if on_mcp:
        pin = getattr(getattr(config, "mcp", None), "pin", None)
        candidate = header_pin or provided
        if pin and candidate and _matches(candidate, str(pin)):
            if count:
                _warn_pin_deprecated()
            return Principal("pin", "", "mcp-pin", "admin")
    return None


def deny_scope(principal: Principal, need: str, action: str) -> AuthError:
    return AuthError(
        f"This credential's scope ({principal.scope}) is not enough for {action}; it needs "
        f"{need}. Use a key with the scope the message names (read < inference < admin): "
        "`studioforge keys create --scope admin`, or the panel's Users & Keys page.",
        code="permission_denied",
        status_code=403,
        details={"scope": principal.scope, "needed": need},
    )


def _remember(request: Any, principal: Principal) -> None:
    """Attach the established principal to the request for handlers and the audit log."""
    try:
        request.state.principal = principal
    except Exception:  # noqa: BLE001 - a stand-in request without .state still authenticates
        return


def check_request(request: Request, config: Config, accounts: Any = None) -> None:
    """Raise :class:`AuthError` unless the request is authorized.

    Three credentials are accepted, tried in this order (see the module
    docstring): ``server.api_key`` (deprecated, admin), a scoped API key, and
    ``mcp.pin`` (deprecated, admin, **MCP path only** -- a short pairing code
    that grants the management tools, never inference or the rest of the API).

    A valid credential establishes a :class:`~studioforge.authkeys.Principal`
    (stored on ``request.state``) whose scope must cover the route
    (:func:`required_scope`): a read-only key cannot POST, an inference key
    cannot change configuration. That holds even on an install that otherwise
    needs no credential.

    When no credential matches: if auth is not enforced (:func:`auth_enforced`:
    loopback-only, no keys, no ``server.api_key``) the server is open, which
    matches how LM Studio behaves out of the box -- but an MCP PIN is still
    enforced when one is configured, so the control plane is never accidentally
    the *least* protected surface. And on that open install the routes that
    change the *box* -- edit config, delete files, restart, install an engine or
    an update, queue downloads, kill processes -- need a caller on this machine
    or the PIN (D32): the MCP ``set_config`` tool demanded the PIN while
    ``PATCH /api/config`` -- same capability, same process -- demanded nothing,
    and anyone on the LAN could set ``server.api_key`` and lock the owner out.
    """
    path = request.url.path
    if request.method == "OPTIONS":
        return
    deep = False
    if path in PUBLIC_PATHS:
        # The liveness form stays credential-free for watchdogs and load
        # balancers -- but /health?deep=true is genuine inference against
        # every loaded model, so it needs the key like any other request.
        raw_deep = (request.query_params.get("deep") or "").strip().lower()
        if raw_deep in _FALSY:
            return
        deep = True

    accounts = accounts if accounts is not None else accounts_of(request)
    mcp_config = getattr(config, "mcp", None)
    on_mcp = is_mcp_path(path, config)
    pin = getattr(mcp_config, "pin", None) if mcp_config else None
    pin_required = bool(getattr(mcp_config, "pin_required", False)) and bool(pin)

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

    try:
        principal = resolve_principal(
            provided, header_pin, config, accounts, on_mcp=on_mcp, count=True
        )
    except AuthError as exc:
        # A revoked or expired key is a wrong credential for lockout purposes;
        # a rate-limited one is a right credential used too fast.
        if exc.code != "key_rate_limited":
            GUARD.record_failure(client)
        raise
    if principal is not None:
        if principal.allows("admin"):
            # Only a credential as strong as the ones being guessed (the PIN,
            # the owner password: both admin) forgives earlier failures. A
            # read or inference key clearing the record would let its holder
            # interleave one good request between every three PIN guesses and
            # walk the 8-digit space with no lockout at all -- a privilege
            # escalation from the weakest key to admin.
            GUARD.record_success(client)
        _remember(request, principal)
        need = required_scope(request.method, path, config, deep=deep)
        if not principal.allows(need):
            raise deny_scope(principal, need, f"{request.method} {path}")
        return

    enforced = auth_enforced(config, accounts)

    if on_mcp and pin and pin_required:
        candidate = header_pin or provided
        if candidate:
            GUARD.record_failure(client)
        note = " " + PIN_IN_QUERY_NOTE if pin_in_query(request) else ""
        raise AuthError(
            "This MCP endpoint needs a credential: an API key (preferred) or the pairing PIN. "
            "Send it as 'Authorization: Bearer <key-or-pin>' or 'X-MCP-Pin: <pin>'. The PIN is "
            "on the server: the startup banner, `studioforge config`, or the control panel "
            "(Setup -> Network & access). GET /api/mcp/info returns 'pin': null "
            "to a caller in your position -- it only reveals the PIN to someone "
            "who is already on the box or already authenticated, which is the "
            "point of it being a credential." + note,
            code="invalid_mcp_pin",
        )

    if not enforced:
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
        local = is_local_request(request) and not cross_site_browser_request(request)
        if gated and not local:
            candidate = header_pin or provided
            if pin and candidate and _matches(candidate, str(pin)):
                # D32: on an open install the PIN also opens the box-changing
                # HTTP routes, not only /mcp (deprecated, like the PIN itself).
                GUARD.record_success(client)
                _warn_pin_deprecated()
                _remember(request, Principal("pin", "", "mcp-pin", "admin"))
                return
            if candidate:
                GUARD.record_failure(client)
            note = " " + PIN_IN_QUERY_NOTE if pin_in_query(request) else ""
            raise AuthError(
                REMOTE_ADMIN_NOTE + note,
                code="remote_admin_requires_credential",
                status_code=403,
            )
        _remember(
            request,
            Principal("local", "", "unauthenticated", "admin")
            if local
            else Principal("anonymous", "", "unauthenticated", "inference"),
        )
        return
    if provided or header_pin:
        # A wrong X-MCP-Pin header is a failed credential too: without this a
        # caller could walk the 8-digit PIN space through the header without
        # ever tripping the per-address lockout (only the bearer form counted).
        GUARD.record_failure(client)
    if provided and looks_like_key(provided):
        message = "Incorrect API key provided."
    else:
        message = "An API key is required."
    raise AuthError(
        message + " Send it as 'Authorization: Bearer <key>' (or 'X-API-Key'). Create one with "
        "`studioforge keys create`, or in the panel under Users & Keys."
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
    established = principal_of(request)
    if established is not None and established.kind in _CREDENTIAL_KINDS:
        return established.allows("admin")
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
