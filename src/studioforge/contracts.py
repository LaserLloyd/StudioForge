"""Generate the frozen contracts and the generated reference docs.

Everything under ``docs/contracts/`` and ``docs/reference/`` is produced here,
from code, and diffed in CI (``scripts/gen_docs.py --check`` and
``tests/unit/test_contracts_drift.py``). A change to a route, an MCP tool, an
error code, a setting or a capability therefore shows up in review as a diff of
a committed file, and a *breaking* diff is visible without anyone having to
remember to look.

Files (repo-relative path -> content):

======================================== ==============================================
``docs/contracts/openapi.json``          the gateway's OpenAPI document (``/openapi.json``)
``docs/contracts/mcp-tools.json``        every MCP tool, gateway and watchdog
``docs/contracts/errors.json``           the error catalog
``docs/contracts/settings-schema.json``  the settings registry
``docs/contracts/capabilities.json``     the capability registry
``docs/reference/settings.md``           the settings reference page
``docs/reference/errors.md``             the error reference page
``docs/CLI.md``                          the command line reference (from the typer tree)
======================================== ==============================================

The app is built over a throwaway data directory with a null GPU probe and a
cleaned ``SF_*`` environment, so the output is the same on every machine and
nothing of the developer's install leaks in. No port is bound, no model is
loaded and no engine is started.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from studioforge import capability_registry, cli_docs, error_catalog, settings_schema

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Bump when the *shape* of a contract file changes (not its content).
CONTRACT_FORMAT = 1


@contextlib.contextmanager
def _clean_environment() -> Iterator[None]:
    """No ``SF_*`` from the caller's shell; a null GPU probe."""
    saved = {k: v for k, v in os.environ.items() if k.startswith("SF_")}
    for key in saved:
        del os.environ[key]
    os.environ["SF_GPU_PROBE"] = "null"
    try:
        yield
    finally:
        os.environ.pop("SF_GPU_PROBE", None)
        os.environ.update(saved)


def dump_json(data: Any) -> str:
    """Deterministic JSON: sorted keys, two-space indent, LF, trailing newline."""
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _tool_rows(server: str, tools: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for tool in sorted(tools, key=lambda t: t.name):
        data = tool.model_dump(mode="json", exclude_none=True)
        rows.append(
            {
                "name": tool.name,
                "server": server,
                "description": inspect.cleandoc(tool.description or ""),
                "input_schema": data.get("input_schema", {}),
                "output_schema": data.get("output_schema"),
                # Placeholder until MCP hardening adds readOnlyHint / destructiveHint.
                "annotations": data.get("annotations"),
            }
        )
    return rows


async def _collect(state: Any, config_path: Path) -> list[dict[str, Any]]:
    from studioforge.mcp.management import build_management_mcp
    from studioforge.watchdog.server import Watchdog, build_watchdog_mcp

    gateway = build_management_mcp(state)
    watchdog = build_watchdog_mcp(Watchdog(config_path))
    rows = _tool_rows("gateway", await gateway.list_tools())
    rows += _tool_rows("watchdog", await watchdog.list_tools())
    return rows


def live_surfaces() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``(openapi document, mcp tool rows)`` from an in-process app. Safe to call anywhere."""
    from studioforge.api.app import build_state, create_app
    from studioforge.config import Config, GuiConfig, LoggingConfig, ModelsConfig, WatchdogConfig

    with (
        _clean_environment(),
        tempfile.TemporaryDirectory(prefix="sf-contracts-", ignore_cleanup_errors=True) as tmp,
    ):
        root = Path(tmp)
        config = Config(
            data_dir=root / "data",
            models=ModelsConfig(dir=root / "models"),
            gui=GuiConfig(enabled=False),
            watchdog=WatchdogConfig(enabled=False),
            logging=LoggingConfig(level="ERROR"),
        )
        config.ensure_dirs()
        config_path = config.save(root / "data" / "config.yaml")
        state = build_state(config)
        try:
            app = create_app(config, state=state, start_background=False)
            openapi = json.loads(json.dumps(app.openapi()))
            tools = asyncio.run(_collect(state, config_path))
        finally:
            with contextlib.suppress(Exception):
                state.db.close()
    return openapi, tools


def render_contracts(
    openapi: dict[str, Any] | None = None, tools: list[dict[str, Any]] | None = None
) -> dict[str, str]:
    """Every generated file as ``{repo-relative path: text}``."""
    if openapi is None or tools is None:
        openapi, tools = live_surfaces()
    return {
        "docs/contracts/openapi.json": dump_json(openapi),
        "docs/contracts/mcp-tools.json": dump_json(
            {"format": CONTRACT_FORMAT, "count": len(tools), "tools": tools}
        ),
        "docs/contracts/errors.json": dump_json(
            {
                "format": CONTRACT_FORMAT,
                "envelope": ["message", "type", "code", "param", "fix", "docs"],
                "errors": error_catalog.export(),
            }
        ),
        "docs/contracts/settings-schema.json": dump_json(settings_schema.export()),
        "docs/contracts/capabilities.json": dump_json(capability_registry.export()),
        "docs/reference/settings.md": settings_schema.render_markdown(),
        "docs/reference/errors.md": error_catalog.render_markdown(),
        "docs/CLI.md": cli_docs.render_markdown(),
    }


def read_committed(path: str, root: Path = REPO_ROOT) -> str | None:
    """The committed text with line endings normalised (a Windows checkout has CRLF)."""
    target = root / path
    if not target.is_file():
        return None
    return target.read_bytes().decode("utf-8").replace("\r\n", "\n")


def stale_files(generated: dict[str, str], root: Path = REPO_ROOT) -> list[str]:
    """Paths whose committed content differs from what the code generates."""
    return [path for path, text in generated.items() if read_committed(path, root) != text]


def write_files(generated: dict[str, str], root: Path = REPO_ROOT) -> list[str]:
    """Write every file that changed, as LF bytes. Returns the paths written."""
    written = []
    for path in stale_files(generated, root):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(generated[path].encode("utf-8"))
        written.append(path)
    return written
