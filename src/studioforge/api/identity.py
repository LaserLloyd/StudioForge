"""Who answered: response identity headers, request ids, deprecated-alias markers (D90).

A client that talks to "an OpenAI-compatible server on :1234" cannot tell
StudioForge from stock llama-server, LM Studio or vLLM by the body alone, and
cannot say afterwards which model and engine produced an answer. Four headers
close that gap on every response:

``Server: StudioForge/<version>``
    The HTTP-standard product token.
``X-Request-Id``
    The client's own id echoed back (when it is a sane token), else a fresh one.
    The same id is bound into the structured log for the duration of the request.
``X-SF-Model``
    The id that actually served the request (``X@b`` for a copy, the base for a
    preset-only virtual model). Only on responses a model produced or was
    resolved for.
``X-SF-Engine``
    The llama.cpp build tag behind it.

``server.identity_headers: false`` turns all four off. This is a pure ASGI
middleware (not ``BaseHTTPMiddleware``) so a streamed body is never buffered
and the context variable the routes write into is the one this reads.

The same module owns the two other cross-cutting response markers:

* ``Deprecation`` (draft-ietf-httpapi-deprecation-header) plus a ``Link`` with
  ``rel="deprecation"`` on the old StudioForge-specific aliases of a surface
  that now has a standard spelling (``DEPRECATED_ALIASES``). No ``Sunset``
  until the bench team reports each surface migrated (lead decision F-10).
* ``enforce_autoload``: llama-server's per-request ``?autoload=false``.
"""

from __future__ import annotations

import contextvars
import re
import uuid
from typing import Any

import structlog
from starlette.datastructures import Headers, MutableHeaders

from studioforge import __version__
from studioforge.core.leases import ALIASES_DEPRECATED_AT, MIGRATION_DOC
from studioforge.errors import StudioForgeError

__all__ = [
    "DEPRECATED_ALIASES",
    "DEPRECATION_EPOCH",
    "DEPRECATION_LINK",
    "IdentityMiddleware",
    "clean_request_id",
    "deprecation_headers",
    "enforce_autoload",
    "is_deprecated_alias",
    "note_serving",
    "register_deprecated_alias",
    "request_id_of",
]

#: 2026-11-09T00:00:00Z -- the release that introduced the standard spellings.
#: One date and one document for every deprecated surface: the lease routes
#: (D87) set the same pair, so the two can never disagree.
DEPRECATION_EPOCH = ALIASES_DEPRECATED_AT
DEPRECATION_LINK = f'<{MIGRATION_DOC}>; rel="deprecation"'

#: Request ids a client may supply: one short token, no whitespace or control
#: characters (they land in headers and log lines).
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:/=+\-]{1,128}$")

#: The mutable slot a route writes ``note_serving`` into; the middleware reads it
#: when the response starts. One dict per request, shared by reference with every
#: task the request spawns (``BaseHTTPMiddleware`` copies the context, not the dict).
_serving: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "sf_serving", default=None
)


def clean_request_id(raw: str | None) -> str | None:
    """The client's id if it is a sane token, else ``None`` (a fresh one is made)."""
    if raw is None:
        return None
    value = raw.strip()
    return value if _REQUEST_ID_RE.match(value) else None


def request_id_of(request: Any) -> str:
    """The id of this request (set by :class:`IdentityMiddleware`; a fresh one in-process)."""
    state = getattr(request, "scope", {}).get("state") or {}
    found = state.get("request_id") if isinstance(state, dict) else None
    return str(found) if found else uuid.uuid4().hex


def note_serving(model_id: str | None) -> None:
    """Record which instance serves this request, for ``X-SF-Model`` / ``X-SF-Engine``.

    Called by the route once it knows (before the first response byte, streaming
    included). A no-op outside an HTTP request.
    """
    slot = _serving.get()
    if slot is not None and model_id:
        slot["model"] = str(model_id)


# ---------------------------------------------------------------------------
# Deprecated aliases (F-10)
# ---------------------------------------------------------------------------

#: ``(METHOD, compiled full-match path pattern)`` of every route that is now an
#: alias of a standard surface. Other work packages append theirs with
#: :func:`register_deprecated_alias` (lease field routes, ``/api/models/{id}/plan``...).
DEPRECATED_ALIASES: list[tuple[str, re.Pattern[str]]] = [
    ("POST", re.compile(r"/api/models/.+/(?:load|load-recommended|unload)")),
    ("GET", re.compile(r"/api/models/.+/(?:plan|plan-recommended)")),
    ("POST", re.compile(r"/api/v1/models/(?:load|unload)")),
    # D87: the REST touch is the alias of PATCH /api/leases/{id} {"renew_time"}.
    ("POST", re.compile(r"/api/leases/[^/]+/touch/?")),
]


def register_deprecated_alias(method: str, pattern: str) -> None:
    """Mark ``method`` + the full-match regex ``pattern`` as a deprecated alias."""
    DEPRECATED_ALIASES.append((method.upper(), re.compile(pattern)))


def is_deprecated_alias(method: str, path: str) -> bool:
    verb = (method or "").upper()
    return any(verb == m and p.fullmatch(path) for m, p in DEPRECATED_ALIASES)


def deprecation_headers() -> list[tuple[str, str]]:
    """The two headers every deprecated surface carries (structured-field date)."""
    return [("Deprecation", f"@{DEPRECATION_EPOCH}"), ("Link", DEPRECATION_LINK)]


# ---------------------------------------------------------------------------
# ?autoload=false
# ---------------------------------------------------------------------------

_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def autoload_requested(request: Any, *, default: bool = True) -> bool:
    """The per-request ``?autoload=true|false`` (llama-server router vocabulary)."""
    raw = ""
    params = getattr(request, "query_params", None)
    if params is not None:
        raw = (params.get("autoload") or "").strip().lower()
    if raw in _FALSE_VALUES:
        return False
    if raw in _TRUE_VALUES:
        return True
    return default


def enforce_autoload(request: Any, state: Any, serving_id: str, *, default: bool = True) -> None:
    """Refuse a just-in-time load when the caller sent ``?autoload=false``.

    404 ``model_not_loaded`` instead of starting a load, exactly what
    ``llama-server --no-models-autoload`` does per request. A model that is
    already serving passes. ``default=False`` makes the absence of the parameter
    mean "do not load" (the GET natives, which are polls, never loads).
    """
    if autoload_requested(request, default=default):
        return
    try:
        instance = state.supervisor.get(serving_id)
    except Exception:  # noqa: BLE001 - an unreadable supervisor reads as not loaded
        instance = None
    if instance is not None and instance.state == "ready":
        return
    loading = instance is not None and instance.state == "loading"
    raise StudioForgeError(
        f"The model '{serving_id}' is {'still loading' if loading else 'not loaded'}, "
        "and this request said not to load it (autoload=false).",
        code="model_not_loaded",
        status_code=404,
        param="model",
        details={"model_id": serving_id, "loading": loading},
    )


# ---------------------------------------------------------------------------
# The middleware
# ---------------------------------------------------------------------------


def _header_safe(value: str) -> str:
    """A header value that survives latin-1 encoding and holds no control characters."""
    return "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in str(value))[:256]


def _engine_tag_for(state: Any, model_id: str) -> str | None:
    """The build serving ``model_id``: the child's own, else its pin, else the active one."""
    try:
        instance = state.supervisor.get(model_id)
    except Exception:  # noqa: BLE001
        instance = None
    if instance is not None:
        tag = getattr(instance, "resolved_engine_tag", None) or getattr(
            instance, "engine_tag", None
        )
        if tag:
            return str(tag)
    try:
        record = state.registry.get(model_id)
        pinned = getattr(getattr(record, "settings", None), "engine_tag", None)
        if pinned:
            return str(pinned)
    except Exception:  # noqa: BLE001
        pass
    try:
        active = state.engine_manager.active()
        tag = getattr(active, "tag", None)
        return str(tag) if tag else None
    except Exception:  # noqa: BLE001 - no engine installed
        return None


class IdentityMiddleware:
    """Request id, identity headers and alias deprecation markers on every HTTP response."""

    def __init__(self, app: Any, config: Any) -> None:
        self.app = app
        self.config = config

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = Headers(scope=scope).get("x-request-id")
        rid = clean_request_id(incoming) or uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = rid
        slot: dict[str, str] = {}
        token = _serving.set(slot)
        log_tokens = structlog.contextvars.bind_contextvars(request_id=rid)
        deprecated = is_deprecated_alias(scope.get("method", ""), scope.get("path", ""))

        async def send_with_identity(message: Any) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                if getattr(self.config.server, "identity_headers", True):
                    headers["Server"] = f"StudioForge/{__version__}"
                    headers["X-Request-Id"] = rid
                    model = slot.get("model")
                    if model:
                        headers["X-SF-Model"] = _header_safe(model)
                        app = scope.get("app")
                        engine = _engine_tag_for(getattr(app, "state", None), model)
                        if engine:
                            headers["X-SF-Engine"] = _header_safe(engine)
                if deprecated and "deprecation" not in headers:
                    for name, value in deprecation_headers():
                        headers.append(name, value)
            await send(message)

        try:
            await self.app(scope, receive, send_with_identity)
        finally:
            _serving.reset(token)
            structlog.contextvars.reset_contextvars(**log_tokens)
