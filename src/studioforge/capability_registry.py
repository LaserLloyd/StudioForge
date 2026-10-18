"""The capability registry: one row per thing StudioForge can do, and every surface that does it.

P11 ("headless parity") says whatever the panel can do, the API, the CLI and an
agent can do too. This module is the written-down map of that promise: for each
capability, the REST route, the MCP tool (on the gateway or the watchdog), the
``sfctl`` command and the panel control that perform it today.

**Filled from what EXISTS.** A surface that does not exist is ``None`` -- never a
guess. ``tests/unit/test_capability_parity.py`` asserts that every declared
route is a route of the FastAPI app, every declared MCP tool is a registered
tool and every declared ``sfctl`` command is a registered command; it *reports*
(without failing) the ``None`` surfaces, so later workstreams have a to-do list
and filling one in is a one-line change here plus the code it names.

Not a router: nothing dispatches through this table. It is documentation that a
test keeps true (and ``docs/contracts/capabilities.json`` freezes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

McpServer = Literal["gateway", "watchdog"]


@dataclass(frozen=True)
class McpTool:
    server: McpServer
    tool: str


@dataclass(frozen=True)
class Capability:
    id: str
    description: str
    #: ``(METHOD, "/path/{param}")`` exactly as the OpenAPI schema spells it.
    api: tuple[str, str] | None = None
    mcp: McpTool | None = None
    #: The ``sfctl`` command path, optionally followed by the flags that select
    #: the behaviour (``"recover --kill"``). The parity test checks the path.
    sfctl: str | None = None
    #: The panel control, ``"<Tab> tab: <control>"``.
    panel: str | None = None
    #: Anything a reader needs to interpret a partial surface.
    notes: str = ""

    def surfaces(self) -> dict[str, Any]:
        return {"api": self.api, "mcp": self.mcp, "sfctl": self.sfctl, "panel": self.panel}

    def missing(self) -> list[str]:
        return [name for name, value in self.surfaces().items() if value is None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "api": {"method": self.api[0], "path": self.api[1]} if self.api else None,
            "mcp": {"server": self.mcp.server, "tool": self.mcp.tool} if self.mcp else None,
            "sfctl": self.sfctl,
            "panel": self.panel,
            "notes": self.notes,
        }


def _g(tool: str) -> McpTool:
    return McpTool("gateway", tool)


def _w(tool: str) -> McpTool:
    return McpTool("watchdog", tool)


CAPABILITIES: tuple[Capability, ...] = (
    # --- models ------------------------------------------------------------
    Capability(
        "models.list",
        "List the models in the library with their load state.",
        api=("GET", "/api/models"),
        mcp=_g("list_models"),
        sfctl="models list",
        panel="Models tab: model table",
    ),
    Capability(
        "models.load",
        "Load a model onto the GPUs (planned, with a context and placement).",
        api=("POST", "/api/models/{model_id}/load"),
        mcp=_g("load_model"),
        sfctl="models load",
        panel="Models tab: Load",
    ),
    Capability(
        "models.unload",
        "Unload a model and free its VRAM.",
        api=("POST", "/api/models/{model_id}/unload"),
        mcp=_g("unload_model"),
        sfctl="models unload",
        panel="Models tab: Unload",
    ),
    Capability(
        "models.unload_all",
        "Unload every loaded model.",
        api=("POST", "/api/models/unload-all"),
        panel="Dashboard tab: Unload all",
    ),
    Capability(
        "models.delete",
        "Delete a model's files from disk (needs confirmation).",
        api=("DELETE", "/api/models/{model_id}"),
        mcp=_g("delete_model"),
        sfctl="models delete",
        panel="Models tab: Delete",
    ),
    Capability(
        "models.download",
        "Download a GGUF from HuggingFace.",
        api=("POST", "/api/downloads"),
        mcp=_g("download_model"),
        sfctl="download",
        panel="Download tab: Download",
    ),
    Capability(
        "models.test",
        "Run one short completion against a model and report latency and tokens per second.",
        api=("POST", "/api/models/{model_id}/test"),
        mcp=_g("test_model"),
        sfctl="models test",
        panel="Models tab: Test",
    ),
    Capability(
        "models.pin",
        "Pin a model so it stays loaded.",
        api=("POST", "/api/models/{model_id}/pin"),
        mcp=_g("pin_model"),
        sfctl="models pin",
        panel="Models tab: Pin",
    ),
    Capability(
        "models.scan",
        "Rescan the model directories.",
        api=("POST", "/api/models/scan"),
        sfctl="models scan",
        panel="Models tab: Rescan",
    ),
    Capability(
        "hf.search",
        "Search HuggingFace for GGUF repositories.",
        api=("GET", "/api/hf/search"),
        mcp=_g("search_models"),
        sfctl="search",
        panel="Download tab: Search",
    ),
    # --- engines -----------------------------------------------------------
    Capability(
        "engine.list",
        "Show the installed llama.cpp builds and which is active.",
        api=("GET", "/api/engine"),
        panel="Server tab: engine builds",
        notes="`sfctl update --engine TAG --check` prints the engine block but is not a "
        "dedicated command.",
    ),
    Capability(
        "engine.install",
        "Download, verify and smoke-test a llama.cpp build.",
        api=("POST", "/api/engine/install"),
        sfctl="update --engine",
        panel="Server tab: Install",
    ),
    Capability(
        "engine.activate",
        "Make an installed llama.cpp build the active engine.",
        api=("POST", "/api/engine/activate"),
        panel="Server tab: activate + reload",
    ),
    # --- configuration -----------------------------------------------------
    Capability(
        "config.get",
        "Read the effective configuration (secrets redacted).",
        api=("GET", "/api/config"),
        mcp=_g("get_config"),
        sfctl="config get",
        panel="Setup tab: settings form",
    ),
    Capability(
        "config.set",
        "Change configuration keys (validated, persisted, restart needs reported).",
        api=("PATCH", "/api/config"),
        mcp=_g("set_config"),
        sfctl="config set",
        panel="Setup tab: Save",
    ),
    Capability(
        "config.schema",
        "Describe every configuration key: type, default, range, level, env override.",
        api=("GET", "/api/settings/schema"),
    ),
    # --- server ------------------------------------------------------------
    Capability(
        "server.restart",
        "Restart the server process (drops in-flight requests and unloads every model).",
        api=("POST", "/api/restart/server"),
        mcp=_w("restart_server"),
        sfctl="recover --restart",
        panel="Dashboard tab: Restart server",
    ),
    Capability(
        "status.get",
        "VRAM per GPU, loaded models, queue depth, engine and uptime.",
        api=("GET", "/api/status"),
        mcp=_g("server_status"),
        sfctl="status",
        panel="Dashboard tab",
    ),
    Capability(
        "health.get",
        "Is the server up, and can it serve? (the watchdog diagnoses a wedged one).",
        api=("GET", "/health"),
        mcp=_w("health"),
        sfctl="recover",
        notes="`sfctl recover` with no flags prints the watchdog's health diagnosis.",
    ),
    Capability(
        "logs.read",
        "Read the server log, or one model's llama-server log.",
        api=("GET", "/api/logs"),
        mcp=_w("tail_logs"),
        sfctl="logs",
        panel="Logs tab: Refresh",
    ),
    # --- GPU sharing -------------------------------------------------------
    Capability(
        "leases.list",
        "List standing GPU leases.",
        api=("GET", "/api/leases"),
        mcp=_g("gpu_leases"),
        sfctl="leases list",
        panel="Dashboard tab: GPU leases",
    ),
    Capability(
        "leases.acquire",
        "Reserve GPUs for a holder until released or expired.",
        api=("POST", "/api/leases"),
        mcp=_g("reserve_gpus"),
        sfctl="leases add",
    ),
    Capability(
        "leases.release",
        "Release a GPU lease.",
        api=("DELETE", "/api/leases/{lease_id}"),
        mcp=_g("release_gpus"),
        sfctl="leases release",
        panel="Dashboard tab: Release lease",
    ),
    Capability(
        "tenants.list",
        "List the registry of programs sharing the GPUs.",
        api=("GET", "/api/tenants"),
        mcp=_g("gpu_tenants"),
        sfctl="tenants list",
        panel="Dashboard tab: tenants",
    ),
    # --- updates and recovery ---------------------------------------------
    Capability(
        "update.check",
        "Compare the running version with the latest release.",
        api=("GET", "/api/update"),
        sfctl="update --check",
        notes="The panel's `Check for engine update` checks llama.cpp builds, not the app.",
    ),
    Capability(
        "update.apply",
        "Install the latest release (restarts the server).",
        api=("POST", "/api/update/install"),
        sfctl="update",
    ),
    Capability(
        "update.rollback",
        "Roll back to the previous release.",
        api=("POST", "/api/update/rollback"),
        mcp=_w("rollback_update"),
    ),
    Capability(
        "recovery.kill_model",
        "Kill one model's llama-server child through the watchdog.",
        mcp=_w("kill_model"),
        sfctl="recover --kill",
    ),
    Capability(
        "recovery.nuke_all",
        "Kill every model child through the watchdog.",
        mcp=_w("nuke_all_models"),
        sfctl="recover --nuke",
    ),
    Capability(
        "recovery.reclaim_orphans",
        "Kill leaked llama-server processes that hold VRAM with no parent.",
        api=("POST", "/api/vram/reclaim"),
        mcp=_w("reclaim_orphan_engines"),
        panel="Dashboard tab: Reclaim",
    ),
)

BY_ID: dict[str, Capability] = {c.id: c for c in CAPABILITIES}

if len(BY_ID) != len(CAPABILITIES):  # pragma: no cover - import-time guard
    raise RuntimeError("duplicate capability id in capability_registry.CAPABILITIES")


def missing_surfaces() -> dict[str, list[str]]:
    """``{capability id: [surfaces that do not exist yet]}`` for every partial capability."""
    return {c.id: c.missing() for c in CAPABILITIES if c.missing()}


def export() -> dict[str, Any]:
    """The registry as plain data (``docs/contracts/capabilities.json``)."""
    return {
        "capabilities": [c.to_dict() for c in sorted(CAPABILITIES, key=lambda c: c.id)],
        "surfaces": ["api", "mcp", "sfctl", "panel"],
    }
