"""``sfctl mcp``: one stdio MCP server that merges the server's TWO HTTP MCPs.

Why merge two upstreams instead of one
--------------------------------------
StudioForge splits its control plane deliberately:

* the **management** MCP lives inside the main app (``<url>/mcp``) and can do
  everything -- list/load/unload/download/delete models, read config;
* the **recovery** MCP lives in the *watchdog*, a separate process on a separate
  port (``<watchdog_url>/mcp``, default 1235), and can restart the app, kill a
  stuck model child, nuke every model, tail logs and roll back an update.

The whole point of that split is that the watchdog answers when the main app
does not. So this proxy connects to each upstream **lazily and independently**:
one being down never fails the stdio session, never removes the other's tools,
and never breaks the agent's session. With the main app wedged, the agent still
sees ``restart_server`` and can fix the box; the management tools remain listed
(so the agent knows they exist) but return a tool-result error telling it to run
``restart_server`` first.

Why the ``recovery_`` prefix
----------------------------
Both upstreams expose ``get_config``/``set_config``, and the watchdog has its own
``health``. Merging them into one namespace needs a deterministic rule, and the
rule is: **management keeps the bare name; the watchdog's colliding names get a
``recovery_`` prefix.** Management is the everyday surface, so it stays
unadorned; the watchdog's uniquely-named tools (``restart_server``,
``kill_model``, ``nuke_all_models``, ``tail_logs``, ``gpu_status``,
``rollback_update``) collide with nothing and keep their names. Anything else
new that appears on the watchdog is prefixed too -- prefixing an unknown name is
safe, shadowing a management tool is not.

Transport failures become tool-result *errors* (``is_error=True``), never
protocol errors: an MCP client that gets a protocol error may drop the whole
session, which is exactly the outcome this design exists to prevent.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

import mcp.types as types
from mcp import Client

# ``create_mcp_http_client`` is the SDK's documented way to give a transport
# custom headers (its own docstring shows the Bearer-token case), but
# ``mcp.client.streamable_http`` declares no ``__all__``, so a strict type
# checker treats the name as a non-exported re-export. The ignore is about that
# packaging detail only.
from mcp.client.streamable_http import (  # type: ignore[attr-defined]
    create_mcp_http_client,
    streamable_http_client,
)
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server

from studioforge_companion.config import ServerProfile

PROXY_NAME = "studioforge"
PROXY_VERSION = "0.1.0"

#: Prefix applied to watchdog tools whose names would otherwise collide.
RECOVERY_PREFIX = "recovery_"

#: Watchdog tools that exist nowhere else, so they keep their own names. Keeping
#: ``restart_server`` unprefixed matters: it is the name the error message for a
#: down management tool tells the agent to call.
WATCHDOG_UNPREFIXED: frozenset[str] = frozenset(
    {
        "restart_server",
        "kill_model",
        "nuke_all_models",
        "tail_logs",
        "gpu_status",
        "rollback_update",
        "reclaim_orphan_engines",
    }
)

#: How long a successful ``list_tools`` is reused. Short, because an upstream
#: coming back should show up in the agent's next tool list, not minutes later.
TOOL_CACHE_TTL_S = 30.0

#: How long a *failure* is remembered before retrying. Shorter still, but
#: non-zero so a hard-down server does not add a connect timeout to every call.
TOOL_FAILURE_TTL_S = 5.0

#: Advertised when the main app has never answered. The management tools must
#: still appear -- an agent that cannot see ``load_model`` does not know the
#: capability exists -- and invoking one returns a clear "restart first" error.
MANAGEMENT_FALLBACK_TOOLS: tuple[tuple[str, str], ...] = (
    ("list_models", "Start here. Every model, newest download first, with how to load it."),
    ("model_options", "Every loading option for ONE model: all context sizes, with speeds."),
    ("model_info", "Full detail for one model, including what it is actually running as."),
    ("load_model", "Load a model into VRAM now, returning once it is serving."),
    (
        "load_recommended",
        "Say the model and the context you need; the server picks the GPUs, the KV "
        "cache and the slot count and loads at exactly that context.",
    ),
    ("unload_model", "Unload a model, freeing its VRAM immediately."),
    (
        "pin_model",
        "Keep a model loaded at all times: no idle TTL, never evicted, loaded at "
        "startup and reloaded if it goes down. pinned=false removes the pin.",
    ),
    (
        "reserve_gpus",
        "Give specific GPUs to one model (or hold them for an outside program) until "
        "released or idle; nothing else loads there meanwhile.",
    ),
    ("release_gpus", "End a GPU reservation early by its lease id."),
    ("search_models", "Find GGUF repos on HuggingFace. One compact row per repo."),
    (
        "repo_details",
        "One HuggingFace repo in full: quant sizes, fit, and the context matrix.",
    ),
    ("download_model", "Queue a GGUF model download from HuggingFace."),
    ("delete_model", "Remove a model from the registry, optionally deleting its GGUF files."),
    (
        "server_status",
        "Live snapshot: VRAM, who is holding it, loaded models, queue, engine.",
    ),
    ("test_model", "Smoke-test a model end to end and report its speed."),
    (
        "benchmark_parallel",
        "Measure how many concurrent slots this model is worth running, so "
        "recommended_parallel stops being an estimate.",
    ),
    ("get_config", "Read the effective configuration, with secrets redacted."),
    ("set_config", "Change configuration values, validate them, and persist to disk."),
    ("connection_info", "Every address this server can be reached on, best first."),
)


#: Upper bound on a failure description. A misbehaving upstream can return a
#: multi-kilobyte validation dump, and pasting that into a tool result or a
#: terminal buries the one fact that matters.
MAX_ERROR_CHARS = 200


def describe_exception(exc: BaseException) -> str:
    """One short, readable line from a failure, flattening ``ExceptionGroup``s.

    anyio task groups wrap a refused connection in an ``ExceptionGroup`` whose
    ``str()`` is "unhandled errors in a TaskGroup (1 sub-exception)" -- true and
    useless. The user needs the cause, so the group is unwrapped and trimmed.
    """
    if isinstance(exc, BaseExceptionGroup):
        parts = [describe_exception(sub) for sub in exc.exceptions]
        unique = list(dict.fromkeys(parts))
        return _trim("; ".join(unique) or exc.__class__.__name__)
    text = " ".join(str(exc).split())
    return _trim(f"{exc.__class__.__name__}: {text}" if text else exc.__class__.__name__)


def _trim(text: str) -> str:
    return text if len(text) <= MAX_ERROR_CHARS else text[: MAX_ERROR_CHARS - 3] + "..."


def _fallback_schema() -> dict[str, Any]:
    """Permissive schema for a tool we have not been able to introspect yet."""
    return {"type": "object", "properties": {}, "additionalProperties": True}


@dataclass
class UpstreamTool:
    """One tool as re-advertised locally, plus where to send it."""

    exposed_name: str
    remote_name: str
    upstream: Upstream
    tool: types.Tool


@dataclass
class Upstream:
    """A lazily-connected HTTP MCP endpoint.

    No session is held open between calls. A held session across a server
    restart is worse than useless -- it looks alive and fails at the worst
    moment -- and a fresh handshake per call costs milliseconds on a LAN.
    """

    label: str
    url: str
    api_key: str | None = None
    timeout_s: float = 120.0
    #: ``recovery_`` handling only applies to the watchdog.
    is_watchdog: bool = False

    _tools: list[types.Tool] | None = field(default=None, init=False, repr=False)
    _fetched_at: float = field(default=0.0, init=False, repr=False)
    _last_error: str | None = field(default=None, init=False, repr=False)

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def reachable(self) -> bool:
        return self._tools is not None and self._last_error is None

    @contextlib.asynccontextmanager
    async def _session(self) -> AsyncIterator[Client]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        async with AsyncExitStack() as stack:
            http = await stack.enter_async_context(create_mcp_http_client(headers=headers))
            transport = streamable_http_client(self.url, http_client=http)
            client = await stack.enter_async_context(Client(transport))
            yield client

    async def list_tools(self, *, refresh: bool = False) -> list[types.Tool]:
        """Cached tool list. Raises on failure so the caller can substitute."""
        now = time.monotonic()
        age = now - self._fetched_at
        fresh_ttl = TOOL_FAILURE_TTL_S if self._last_error else TOOL_CACHE_TTL_S
        if not refresh and self._tools is not None and age < fresh_ttl:
            return self._tools
        if not refresh and self._last_error is not None and age < TOOL_FAILURE_TTL_S:
            raise RuntimeError(self._last_error)
        try:
            async with self._session() as client:
                result = await client.list_tools()
        except Exception as exc:
            self._fetched_at = now
            self._last_error = describe_exception(exc)
            raise RuntimeError(self._last_error) from None
        self._tools = list(result.tools)
        self._fetched_at = now
        self._last_error = None
        return self._tools

    async def call(self, name: str, arguments: dict[str, Any] | None) -> types.CallToolResult:
        """Forward one call, verbatim in and verbatim out."""
        async with self._session() as client:
            result = await client.call_tool(name, arguments or {})
        self._last_error = None
        return result


def _error_result(text: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], is_error=True)


def expose_name(remote_name: str, *, is_watchdog: bool) -> str:
    """Apply the naming rule. See the module docstring for why it is this way."""
    if not is_watchdog or remote_name in WATCHDOG_UNPREFIXED:
        return remote_name
    if remote_name.startswith(RECOVERY_PREFIX):
        return remote_name
    return f"{RECOVERY_PREFIX}{remote_name}"


def _annotate(tool: types.Tool, suffix: str) -> types.Tool:
    description = (tool.description or "").rstrip()
    return tool.model_copy(update={"description": f"{description}\n\n{suffix}".strip()})


class McpProxy:
    """Merges the management and recovery upstreams into one stdio MCP server."""

    def __init__(self, profile: ServerProfile) -> None:
        self.profile = profile
        self.management = Upstream(
            label="management",
            url=profile.mcp_url,
            api_key=profile.api_key,
            is_watchdog=False,
        )
        self.watchdog = Upstream(
            label="recovery",
            url=profile.watchdog_mcp_url,
            api_key=profile.api_key,
            is_watchdog=True,
        )

    # -- tool table --------------------------------------------------------

    async def build_table(self) -> dict[str, UpstreamTool]:
        """Current merged tool table, one entry per exposed name.

        Management is built first so a watchdog tool can never shadow it, which
        is the invariant the ``recovery_`` rule exists to guarantee.
        """
        table: dict[str, UpstreamTool] = {}

        try:
            tools = await self.management.list_tools()
        except RuntimeError:
            note = (
                "NOTE: the main StudioForge server is not answering right now "
                "(%s). Calling this tool will fail until it is back -- use "
                "`restart_server` (watchdog) first." % (self.management.last_error or "unreachable")
            )
            for name, description in MANAGEMENT_FALLBACK_TOOLS:
                tool = types.Tool(
                    name=name,
                    description=f"{description}\n\n{note}",
                    input_schema=_fallback_schema(),
                )
                table[name] = UpstreamTool(name, name, self.management, tool)
        else:
            for tool in tools:
                table[tool.name] = UpstreamTool(tool.name, tool.name, self.management, tool)

        try:
            watchdog_tools = await self.watchdog.list_tools()
        except RuntimeError:
            # Nothing to substitute: with no watchdog there is no recovery, and
            # advertising phantom recovery tools would be actively misleading.
            return table

        for tool in watchdog_tools:
            exposed = expose_name(tool.name, is_watchdog=True)
            renamed = tool.model_copy(update={"name": exposed})
            if exposed != tool.name:
                renamed = _annotate(
                    renamed,
                    f"(watchdog tool `{tool.name}`, exposed as `{exposed}` because the main "
                    f"server also has a `{tool.name}`. Works even when the main server is down.)",
                )
            else:
                renamed = _annotate(
                    renamed, "(watchdog tool: works even when the main server is down.)"
                )
            table[exposed] = UpstreamTool(exposed, tool.name, self.watchdog, renamed)
        return table

    # -- MCP handlers ------------------------------------------------------

    async def on_list_tools(
        self, _ctx: ServerRequestContext[Any], _params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        table = await self.build_table()
        return types.ListToolsResult(tools=[entry.tool for entry in table.values()])

    async def on_call_tool(
        self, _ctx: ServerRequestContext[Any], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        table = await self.build_table()
        entry = table.get(params.name)
        if entry is None:
            known = ", ".join(sorted(table)) or "(none -- no upstream is reachable)"
            return _error_result(f"unknown tool {params.name!r}. Available tools: {known}")

        upstream = entry.upstream
        try:
            return await upstream.call(entry.remote_name, params.arguments)
        except Exception as exc:
            detail = describe_exception(exc)
            upstream._last_error = detail  # noqa: SLF001 - same module, cache the failure
            upstream._fetched_at = time.monotonic()  # noqa: SLF001
            if upstream.is_watchdog:
                return _error_result(
                    f"the StudioForge watchdog at {upstream.url} is not answering "
                    f"({detail}). Recovery tools are unavailable; the watchdog process "
                    f"itself may need to be started on the host."
                )
            return _error_result(
                f"the main StudioForge server at {upstream.url} is not answering "
                f"({detail}). It may be wedged or stopped. Try the watchdog tool "
                f"`restart_server` (it runs in a separate process on "
                f"{self.profile.effective_watchdog_url} and stays available), then retry "
                f"`{entry.exposed_name}`."
            )

    # -- serving -----------------------------------------------------------

    def instructions(self) -> str:
        """Explain the merged namespace to the agent reading it."""
        recovery_prefixed = ", ".join(
            f"`{RECOVERY_PREFIX}{n}`" for n in ("health", "get_config", "set_config")
        )
        unprefixed = ", ".join(f"`{n}`" for n in sorted(WATCHDOG_UNPREFIXED))
        return (
            f"Control plane for the StudioForge LLM server at {self.profile.url}.\n\n"
            "START WITH `list_models` -- it returns the model catalog newest-download-first; "
            "every model has `options` rows (one per context size) with `fits`, `devices`, "
            "`max_parallel`, estimated and measured tokens/sec, and a `load_args` object. "
            "Pick the row marked `recommended: true` (or call `model_options` for the full table "
            "if you need a different context size or more concurrency) and pass its `load_args` "
            "verbatim to `load_model`. Inference is NOT here -- use the OpenAI-compatible HTTP "
            "API on the server (POST /v1/chat/completions); naming an unloaded model there loads "
            "it just-in-time.\n\n"
            "To get a NEW model: `search_models` for compact repo rows (no file sizes exist at "
            "search time), then `repo_details(repo_id)` for that repo's real per-quant sizes, "
            "fit verdicts and the context each GPU placement reaches, then "
            "`download_model(repo_id, quant)`. It appears in `list_models` when it lands.\n\n"
            "Two upstreams are merged into this one tool list:\n"
            "  1. MANAGEMENT (the main app): models, loading, config, status. "
            "These keep their plain names, e.g. `list_models`, `load_model`, "
            "`get_config`, `server_status`.\n"
            f"  2. RECOVERY (the watchdog, a separate process on "
            f"{self.profile.effective_watchdog_url}): {unprefixed}, plus "
            f"{recovery_prefixed}.\n\n"
            f"Naming rule: watchdog tools whose names collide with a management tool are "
            f"exposed with a `{RECOVERY_PREFIX}` prefix (so `{RECOVERY_PREFIX}get_config` is the "
            "watchdog's own config, `get_config` is the server's). Watchdog-only tools keep "
            "their names.\n\n"
            "IMPORTANT: the `recovery_*` tools and the watchdog-only tools KEEP WORKING WHEN "
            "THE MAIN SERVER IS DOWN -- that is why they are separate. If a management tool "
            "reports the main server unreachable, call `restart_server`, then retry.\n\n"
            "If VRAM is missing, `server_status` says who holds it: `vram_orphan_count` above "
            "zero means leaked llama-server processes with nothing waiting on them, and "
            "`reclaim_orphan_engines` kills exactly those (never somebody's live child)."
        )

    def build_server(self) -> Server[Any]:
        return Server(
            PROXY_NAME,
            version=PROXY_VERSION,
            instructions=self.instructions(),
            on_list_tools=self.on_list_tools,
            on_call_tool=self.on_call_tool,
        )

    async def serve_stdio(self) -> None:
        """Run the proxy on stdin/stdout, the transport OpenClaw registers."""
        from mcp import stdio_server

        server = self.build_server()
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())


async def call_watchdog_tool(
    profile: ServerProfile, name: str, arguments: dict[str, Any] | None = None
) -> types.CallToolResult:
    """One-shot call against the watchdog's MCP, used by ``sfctl recover``.

    ``recover`` deliberately does not go through the main app's HTTP API: the
    situation it exists for is the main app being unreachable.
    """
    upstream = Upstream(
        label="recovery",
        url=profile.watchdog_mcp_url,
        api_key=profile.api_key,
        is_watchdog=True,
    )
    return await upstream.call(name, arguments)


async def probe_watchdog_auth(profile: ServerProfile, *, timeout_s: float = 5.0) -> str:
    """``"ok"`` | ``"unauthorized"`` | ``"unreachable"`` -- what the watchdog says to us.

    The MCP client collapses every non-2xx into "Server returned an error
    response", which hides the one distinction the operator needs after a
    failed ``recover``: is the watchdog DOWN, or is it UP and refusing our
    credential? One plain HTTP round-trip with the profile's credential tells
    them apart; ``/health`` alone would not, because it is deliberately open.
    """
    import httpx

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if profile.api_key:
        headers["Authorization"] = f"Bearer {profile.api_key}"
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "sfctl-probe", "version": "0"},
        },
    }
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(profile.watchdog_mcp_url, json=body, headers=headers)
    except Exception:  # noqa: BLE001 - any transport failure is "unreachable"
        return "unreachable"
    if response.status_code in (401, 403):
        return "unauthorized"
    return "ok"


def result_text(result: types.CallToolResult) -> str:
    """Flatten a tool result's text blocks for terminal display."""
    parts: list[str] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
        else:
            parts.append(repr(block))
    return "\n".join(parts)
