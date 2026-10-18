"""What each MCP tool may do, and who may call it.

Both MCP servers (the gateway's ``/mcp`` and the watchdog's) share this policy.
Three things live here, all driven by one table so they cannot drift apart:

1. **Annotations.** Every tool is published with ``readOnlyHint`` /
   ``destructiveHint`` / ``idempotentHint`` (MCP ``ToolAnnotations``) derived from
   its row, so a client can show a confirmation for the dangerous ones.
2. **Authorization.** :class:`PolicyExtension` wraps ``tools/call``: a tool that
   is read-only needs a ``read`` scope, one that changes state needs
   ``inference``, and a *destructive* one (or one the table does not know) needs
   ``admin``. A key that is not enough gets ``permission_denied`` (catalogued),
   not a silent no-op. Unknown tools fail closed.
3. **Audit.** Every call that is not read-only is appended to ``audit.jsonl``
   with who called it, what it targeted and whether it worked -- including the
   denials.

A leaf module: it imports only the MCP SDK and :mod:`studioforge.authkeys` /
:mod:`studioforge.auditlog`, so the watchdog (which may not import the app
machinery) uses it unchanged.

Direct, in-process ``server.call_tool(...)`` and the stdio transport bypass this
layer on purpose: they are the local process itself, which already holds the
database and the config file.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from mcp.server.extension import Extension
from mcp_types import CallToolRequestParams, CallToolResult, TextContent, ToolAnnotations

from studioforge.auditlog import AuditLog, redact_update_keys
from studioforge.authkeys import LOCAL_ADMIN, Principal

__all__ = [
    "GATEWAY_TOOLS",
    "WATCHDOG_TOOLS",
    "PolicyExtension",
    "ToolPolicy",
    "annotations_for",
    "policy_for",
]

Kind = Literal["read", "mutate", "destructive"]


@dataclass(frozen=True)
class ToolPolicy:
    """One tool's row: what kind of effect it has and the scope that earns it."""

    kind: Kind
    scope: str
    #: Calling it twice with the same arguments leaves the same state.
    idempotent: bool = False
    #: An argument that, when true, raises the requirement to ``admin``
    #: (``load_recommended(persist=true)`` writes saved settings).
    admin_arg: str | None = None


def _read() -> ToolPolicy:
    return ToolPolicy("read", "read", idempotent=True)


def _mutate(
    *, idempotent: bool = False, scope: str = "inference", admin_arg: str | None = None
) -> ToolPolicy:
    return ToolPolicy("mutate", scope, idempotent=idempotent, admin_arg=admin_arg)


def _destructive(*, idempotent: bool = False) -> ToolPolicy:
    return ToolPolicy("destructive", "admin", idempotent=idempotent)


#: The gateway's ``/mcp`` tools. A new tool MUST get a row (a test enforces it);
#: until it does it is treated as destructive.
GATEWAY_TOOLS: dict[str, ToolPolicy] = {
    # reads
    "check_loaded_model": _read(),
    "connection_info": _read(),
    "get_config": _read(),
    "gpu_leases": _read(),
    "gpu_tenants": _read(),
    "list_models": _read(),
    "model_info": _read(),
    "model_options": _read(),
    "plan_load": _read(),
    "repo_details": _read(),
    "search_models": _read(),
    "server_status": _read(),
    # state-changing, additive: residency and benchmarking
    "benchmark_parallel": _mutate(),
    "load_model": _mutate(idempotent=True),
    "load_recommended": _mutate(idempotent=True, admin_arg="persist"),
    "release_gpus": _mutate(idempotent=True),
    "test_model": _mutate(),
    # pinning persists across restarts (the REST route is an admin route, D32)
    "pin_model": _mutate(idempotent=True, scope="admin"),
    # destructive: removes files, rewrites configuration, evicts or reserves cards
    "delete_model": _destructive(),
    "download_model": _destructive(),
    "reserve_gpus": _destructive(),
    "set_config": _destructive(idempotent=True),
    "unload_model": _destructive(idempotent=True),
}

#: The recovery watchdog's tools.
WATCHDOG_TOOLS: dict[str, ToolPolicy] = {
    "get_config": _read(),
    "gpu_status": _read(),
    "health": _read(),
    "tail_logs": _read(),
    "kill_model": _destructive(),
    "nuke_all_models": _destructive(),
    "reclaim_orphan_engines": _destructive(),
    "restart_server": _destructive(),
    "rollback_update": _destructive(),
    "set_config": _destructive(idempotent=True),
}

_TABLES: dict[str, dict[str, ToolPolicy]] = {"gateway": GATEWAY_TOOLS, "watchdog": WATCHDOG_TOOLS}

#: The row an unclassified tool gets: the most restrictive one.
_UNKNOWN = ToolPolicy("destructive", "admin")


def policy_for(server: str, tool: str) -> ToolPolicy:
    return _TABLES.get(server, {}).get(tool, _UNKNOWN)


def annotations_for(server: str, tool: str) -> ToolAnnotations:
    """The MCP annotations published for ``tool`` (derived, never hand-written per tool)."""
    row = policy_for(server, tool)
    if row.kind == "read":
        return ToolAnnotations(
            read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
        )
    return ToolAnnotations(
        read_only_hint=False,
        destructive_hint=row.kind == "destructive",
        idempotent_hint=row.idempotent,
        open_world_hint=False,
    )


# ---------------------------------------------------------------------------
# Enforcement and audit
# ---------------------------------------------------------------------------

#: ``headers -> Principal | None``. The gateway and the watchdog each supply the
#: resolver that knows their credentials.
Resolver = Callable[[Mapping[str, str]], Principal | None]


def _headers_of(ctx: Any) -> Mapping[str, str] | None:
    """The HTTP headers of the request that carries this tool call, ``None`` off HTTP."""
    request = getattr(ctx, "request", None)
    headers = getattr(request, "headers", None)
    if headers is not None:
        return headers
    session = getattr(ctx, "session", None)
    dispatch = getattr(session, "_dispatch_context", None) or getattr(session, "dctx", None)
    transport = getattr(dispatch, "transport", None)
    return getattr(transport, "headers", None)


def _target(tool: str, arguments: Mapping[str, Any]) -> str:
    """A short, secret-free description of what a call is aimed at."""
    if tool in {"set_config"}:
        updates = arguments.get("updates")
        return redact_update_keys(updates) if isinstance(updates, dict) else ""
    for key in ("model_id", "model_name", "repo_id", "lease_id"):
        if arguments.get(key):
            return str(arguments[key])
    if arguments.get("devices") is not None:
        return f"devices={arguments['devices']}"
    return ""


def _denial(tool: str, principal: Principal | None, needed: str) -> CallToolResult:
    scope = principal.scope if principal else "none"
    payload = {
        "ok": False,
        "error": {
            "message": (
                f"This credential's scope ({scope}) is not enough for {tool}; it needs {needed}. "
                "Use a key with the scope the message names (read < inference < admin)."
            ),
            "code": "permission_denied",
            "type": "invalid_request_error",
            "details": {"tool": tool, "scope": scope, "needed": needed},
        },
    }
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))],
        structured_content=payload,
        is_error=True,
    )


def _outcome(result: Any) -> str:
    if getattr(result, "is_error", False):
        return "error"
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict) and structured.get("ok") is False:
        return "error"
    return "ok"


class PolicyExtension(Extension):
    """Wraps ``tools/call`` with the scope check and the audit record."""

    identifier = "com.laserlloyd/studioforge-policy"

    def __init__(
        self,
        server: str,
        resolve: Resolver,
        audit: Callable[[], AuditLog | None] | AuditLog | None = None,
    ) -> None:
        self._server = server
        self._resolve = resolve
        self._audit = audit

    def settings(self) -> dict[str, Any]:
        return {"scopes": ["read", "inference", "admin"], "annotations": True}

    def _audit_log(self) -> AuditLog | None:
        return self._audit() if callable(self._audit) else self._audit

    async def intercept_tool_call(
        self, params: CallToolRequestParams, ctx: Any, call_next: Any
    ) -> Any:
        tool = params.name
        arguments = params.arguments or {}
        headers = _headers_of(ctx)
        # No HTTP headers = not the HTTP transport (stdio, an in-process client):
        # the local process itself.
        principal = LOCAL_ADMIN if headers is None else self._resolve(headers)
        row = policy_for(self._server, tool)
        needed = row.scope
        if row.admin_arg and arguments.get(row.admin_arg):
            needed = "admin"
        mutating = row.kind != "read"
        actor = principal.actor if principal else "anonymous"
        audit = self._audit_log()
        if principal is None or not principal.allows(needed):
            if audit is not None and mutating:
                audit.record(
                    actor=actor,
                    action=f"mcp.{tool}",
                    target=_target(tool, arguments),
                    result="denied",
                    surface=f"mcp:{self._server}",
                    detail=f"needs {needed}",
                )
            return _denial(tool, principal, needed)
        result = await call_next(ctx)
        if audit is not None and mutating:
            audit.record(
                actor=actor,
                action=f"mcp.{tool}",
                target=_target(tool, arguments),
                result=_outcome(result),
                surface=f"mcp:{self._server}",
            )
        return result
