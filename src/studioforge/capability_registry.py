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

#: The surfaces whose absence the gap report counts. ``cli`` is reported separately.
TRACKED_SURFACES = ("api", "mcp", "sfctl", "panel")


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
    #: The panel control, ``"<Section> page: <control>"``.
    panel: str | None = None
    #: The ``studioforge`` (server) command path, optionally followed by the flags that
    #: select the behaviour (``"unload --all"``). The parity test checks the path and
    #: the flags. Tracked beside, not inside, the four surfaces above: the original
    #: gap report keeps its meaning and ``missing_cli`` reports this column.
    cli: str | None = None
    #: Anything a reader needs to interpret a partial surface.
    notes: str = ""

    def surfaces(self) -> dict[str, Any]:
        return {
            "api": self.api,
            "mcp": self.mcp,
            "sfctl": self.sfctl,
            "panel": self.panel,
            "cli": self.cli,
        }

    def missing(self) -> list[str]:
        """The tracked surfaces (api, mcp, sfctl, panel) this capability lacks."""
        return [name for name in TRACKED_SURFACES if self.surfaces()[name] is None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "api": {"method": self.api[0], "path": self.api[1]} if self.api else None,
            "mcp": {"server": self.mcp.server, "tool": self.mcp.tool} if self.mcp else None,
            "sfctl": self.sfctl,
            "panel": self.panel,
            "cli": self.cli,
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
        panel="Models tab: library",
        cli="ls",
    ),
    Capability(
        "models.load",
        "Load a model onto the GPUs (planned, with a context and placement).",
        api=("POST", "/api/models/{model_id}/load"),
        mcp=_g("load_model"),
        sfctl="models load",
        panel="Models tab: details drawer, Load",
        cli="load",
    ),
    Capability(
        "models.unload",
        "Unload a model and free its VRAM.",
        api=("POST", "/api/models/{model_id}/unload"),
        mcp=_g("unload_model"),
        sfctl="models unload",
        panel="Models tab: details drawer, Unload",
        cli="unload",
    ),
    Capability(
        "models.unload_all",
        "Unload every loaded model.",
        api=("POST", "/api/models/unload-all"),
        panel="Advanced > Watchdog & recovery: Unload all models",
        cli="unload --all",
    ),
    Capability(
        "models.delete",
        "Delete a model's files from disk (needs confirmation).",
        api=("DELETE", "/api/models/{model_id}"),
        mcp=_g("delete_model"),
        sfctl="models delete",
        panel="Models tab: details drawer, Manage, Delete",
    ),
    Capability(
        "models.download",
        "Download a GGUF from HuggingFace.",
        api=("POST", "/api/downloads"),
        mcp=_g("download_model"),
        sfctl="download",
        panel="Models tab: Add model, Downloads",
        cli="get",
    ),
    Capability(
        "models.test",
        "Run one short completion against a model and report latency and tokens per second.",
        api=("POST", "/api/models/{model_id}/test"),
        mcp=_g("test_model"),
        sfctl="models test",
        panel="Models tab: details drawer, Manage, Test",
    ),
    Capability(
        "models.pin",
        "Pin a model so it stays loaded.",
        api=("POST", "/api/models/{model_id}/pin"),
        mcp=_g("pin_model"),
        sfctl="models pin",
        panel="Models tab: details drawer, Manage, Pin",
    ),
    Capability(
        "models.scan",
        "Rescan the model directories.",
        api=("POST", "/api/models/scan"),
        cli="scan",
        sfctl="models scan",
        panel="Models tab: Rescan",
    ),
    Capability(
        "hf.search",
        "Search HuggingFace for GGUF repositories.",
        api=("GET", "/api/hf/search"),
        mcp=_g("search_models"),
        sfctl="search",
        panel="Models tab: Add model, Search Hugging Face",
    ),
    # --- engines -----------------------------------------------------------
    Capability(
        "engine.list",
        "Show the installed llama.cpp builds and which is active.",
        api=("GET", "/api/engine"),
        panel="Advanced > Engines: installed builds",
        notes="`sfctl update --engine TAG --check` prints the engine block but is not a "
        "dedicated command.",
        cli="engine list",
    ),
    Capability(
        "engine.install",
        "Download, verify and smoke-test a llama.cpp build.",
        api=("POST", "/api/engine/install"),
        sfctl="update --engine",
        panel="Advanced > Engines: Install",
        cli="engine install",
    ),
    Capability(
        "engine.activate",
        "Make an installed llama.cpp build the active engine.",
        api=("POST", "/api/engine/activate"),
        panel="Advanced > Engines: activate + reload",
        cli="engine install --activate",
    ),
    # --- configuration -----------------------------------------------------
    Capability(
        "config.get",
        "Read the effective configuration (secrets redacted).",
        api=("GET", "/api/config"),
        mcp=_g("get_config"),
        sfctl="config get",
        panel="Advanced > Serving & network: settings",
        cli="config",
    ),
    Capability(
        "config.set",
        "Change configuration keys (validated, persisted, restart needs reported).",
        api=("PATCH", "/api/config"),
        mcp=_g("set_config"),
        sfctl="config set",
        panel="Advanced > Serving & network: Save",
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
        panel="Advanced > Watchdog & recovery: Restart server",
    ),
    Capability(
        "status.get",
        "VRAM per GPU, loaded models, queue depth, engine and uptime.",
        api=("GET", "/api/status"),
        mcp=_g("server_status"),
        sfctl="status",
        panel="Status tab",
        cli="ps",
    ),
    Capability(
        "health.get",
        "Is the server up, and can it serve? (the watchdog diagnoses a wedged one).",
        api=("GET", "/health"),
        mcp=_w("health"),
        sfctl="recover",
        panel="Advanced > Diagnostics: Run checks",
        notes="`sfctl recover` with no flags prints the watchdog's health diagnosis.",
        cli="server status",
    ),
    Capability(
        "logs.read",
        "Read the server log, or one model's llama-server log.",
        api=("GET", "/api/logs"),
        mcp=_w("tail_logs"),
        sfctl="logs",
        panel="Advanced > Logs & metrics: Refresh",
        cli="log stream",
    ),
    # --- GPU sharing -------------------------------------------------------
    Capability(
        "leases.list",
        "List standing GPU leases.",
        api=("GET", "/api/leases"),
        mcp=_g("gpu_leases"),
        sfctl="leases list",
        panel="Advanced > GPUs & placement: GPU leases",
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
        panel="Advanced > GPUs & placement: Release lease",
    ),
    Capability(
        "tenants.list",
        "List the registry of programs sharing the GPUs.",
        api=("GET", "/api/tenants"),
        mcp=_g("gpu_tenants"),
        sfctl="tenants list",
        panel="Advanced > GPUs & placement: tenants",
    ),
    # --- updates and recovery ---------------------------------------------
    Capability(
        "update.check",
        "Compare the running version with the latest release.",
        api=("GET", "/api/update"),
        sfctl="update --check",
        panel="Advanced > About & updates: Check for updates",
        notes="The panel's `Check for engine update` checks llama.cpp builds, not the app.",
        cli="update check",
    ),
    Capability(
        "update.apply",
        "Install the latest release (restarts the server).",
        api=("POST", "/api/update/install"),
        sfctl="update",
        panel="Advanced > About & updates: Install",
        cli="update apply",
    ),
    Capability(
        "update.rollback",
        "Roll back to the previous release.",
        api=("POST", "/api/update/rollback"),
        mcp=_w("rollback_update"),
        panel="Advanced > About & updates: Roll back",
        cli="update rollback",
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
        panel="Advanced > GPUs & placement: Reclaim",
    ),
    # --- accounts, keys and audit (W3) -------------------------------------
    Capability(
        "auth.status",
        "Is a credential required, who exists, and which listeners are on the network?",
        api=("GET", "/api/auth/status"),
        cli="doctor",
        panel="Advanced > Accounts & keys: security posture",
        notes="`studioforge doctor` reports the same posture as findings.",
    ),
    Capability(
        "keys.list",
        "List API keys (never the tokens).",
        api=("GET", "/api/keys"),
        cli="keys list",
        panel="Advanced > Accounts & keys: keys table",
        notes="Deliberately not an MCP tool: an agent must not manage the keys that bound it.",
    ),
    Capability(
        "keys.create",
        "Create a named, scoped (read | inference | admin) key; the token is shown once.",
        api=("POST", "/api/keys"),
        cli="keys create",
        panel="Advanced > Accounts & keys: Create key",
    ),
    Capability(
        "keys.revoke",
        "Revoke a key; it stops working immediately.",
        api=("DELETE", "/api/keys/{key_id}"),
        cli="keys revoke",
        panel="Advanced > Accounts & keys: Revoke",
    ),
    Capability(
        "users.list",
        "List the accounts that can sign in to the panel.",
        api=("GET", "/api/users"),
        cli="users list",
        panel="Advanced > Accounts & keys: accounts table",
    ),
    Capability(
        "users.create",
        "Create an account (the first is the owner, which satisfies the network-bind rule).",
        api=("POST", "/api/users"),
        cli="users create",
        panel="Advanced > Accounts & keys: Create account",
    ),
    Capability(
        "users.password",
        "Change an account's password.",
        api=("POST", "/api/users/password"),
        cli="users passwd",
        panel="Advanced > Accounts & keys: Change password",
    ),
    Capability(
        "audit.read",
        "Read the append-only audit log of admin mutations.",
        api=("GET", "/api/audit"),
        panel="Advanced > Accounts & keys: audit log",
        notes="The file itself is logs/audit.jsonl.",
    ),
    # --- diagnostics and operations (W3) -----------------------------------
    Capability(
        "doctor.run",
        "Check ports, GPU, engine, disk, permissions, certificate, migrations and auth posture.",
        api=("GET", "/api/doctor"),
        cli="doctor",
        panel="Advanced > Diagnostics: Run checks",
    ),
    Capability(
        "support_bundle.create",
        "A zip for bug reports: redacted config, doctor, versions, GPUs and log tails.",
        api=("POST", "/api/support-bundle"),
        cli="support-bundle",
        panel="Advanced > Diagnostics: Download support bundle",
    ),
    Capability(
        "metrics.get",
        "Prometheus text: gateway counters, GPU memory and every loaded model's metrics.",
        api=("GET", "/metrics"),
        panel="Advanced > Logs & metrics: Metrics",
    ),
    Capability(
        "server.stop",
        "Shut the server down gracefully.",
        api=("POST", "/api/server/stop"),
        cli="server stop",
        notes="The panel has no Stop button on purpose: it would cut off the page that sent it.",
    ),
    Capability(
        "server.start",
        "Start the server in the background and wait until it answers.",
        cli="server start",
        notes="Starts a process, so it cannot be an API route of the process it starts.",
    ),
    Capability(
        "setup.wizard",
        "First run: account, hardware, engine, library, starter model, test chat, exposure.",
        cli="setup",
        panel="/setup page",
    ),
    Capability(
        "db.migrate",
        "Preview (--dry-run) or apply pending database migrations, after a backup.",
        cli="migrate",
    ),
)

BY_ID: dict[str, Capability] = {c.id: c for c in CAPABILITIES}

if len(BY_ID) != len(CAPABILITIES):  # pragma: no cover - import-time guard
    raise RuntimeError("duplicate capability id in capability_registry.CAPABILITIES")


def missing_surfaces() -> dict[str, list[str]]:
    """``{capability id: [surfaces that do not exist yet]}`` for every partial capability."""
    return {c.id: c.missing() for c in CAPABILITIES if c.missing()}


def missing_cli() -> list[str]:
    """Capability ids that have no ``studioforge`` command yet."""
    return [c.id for c in CAPABILITIES if c.cli is None]


def export() -> dict[str, Any]:
    """The registry as plain data (``docs/contracts/capabilities.json``)."""
    return {
        "capabilities": [c.to_dict() for c in sorted(CAPABILITIES, key=lambda c: c.id)],
        "surfaces": ["api", "mcp", "sfctl", "panel", "cli"],
    }
