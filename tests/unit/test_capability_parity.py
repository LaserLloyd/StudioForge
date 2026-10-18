"""Capability parity: every surface the registry declares really exists.

``studioforge.capability_registry`` maps each capability to the REST route, MCP
tool, ``sfctl`` command and panel control that perform it. This test makes the
map honest in one direction -- a *declared* route, tool or command must exist --
and merely *reports* the other direction: a surface declared ``None`` is a gap
for a later workstream to fill (P11), not a failure. The gaps are printed (run
with ``-s``) and raised as a ``UserWarning`` so they appear in the warnings
summary of every run.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import pytest
import typer.main

from studioforge import contracts
from studioforge.capability_registry import (
    BY_ID,
    CAPABILITIES,
    export,
    missing_cli,
    missing_surfaces,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPANION_SRC = REPO_ROOT / "packages" / "studioforge-companion" / "src"
if str(COMPANION_SRC) not in sys.path:
    sys.path.insert(0, str(COMPANION_SRC))

from studioforge_companion import cli as sfctl  # noqa: E402


@pytest.fixture(scope="module")
def surfaces() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The real FastAPI document and both MCP servers' tool lists, built in-process."""
    return contracts.live_surfaces()


def _routes(openapi: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (method.upper(), path)
        for path, item in openapi["paths"].items()
        for method in item
        if method.lower() in {"get", "post", "put", "patch", "delete"}
    }


def _sfctl_commands() -> set[str]:
    """Every registered ``sfctl`` command path, e.g. ``models list`` and ``status``."""
    found: set[str] = set()

    def walk(command: Any, prefix: str) -> None:
        for name, sub in getattr(command, "commands", {}).items():
            path = f"{prefix} {name}".strip()
            if getattr(sub, "commands", None):
                walk(sub, path)
            else:
                found.add(path)

    walk(typer.main.get_command(sfctl.app), "")
    return found


def _cli_root() -> Any:
    from studioforge.__main__ import app

    return typer.main.get_command(app)


def _cli_commands() -> set[str]:
    """Every registered ``studioforge`` command path, e.g. ``ls`` and ``keys create``."""
    found: set[str] = set()

    def walk(command: Any, prefix: str) -> None:
        for name, sub in getattr(command, "commands", {}).items():
            path = f"{prefix} {name}".strip()
            if getattr(sub, "commands", None):
                walk(sub, path)
            else:
                found.add(path)

    walk(_cli_root(), "")
    return found


def _command_path(declared: str) -> str:
    """``"recover --kill"`` -> ``"recover"``: drop the flags that select a behaviour."""
    return " ".join(part for part in declared.split() if not part.startswith(("-", "<", "[")))


def test_registry_is_well_formed() -> None:
    assert len(BY_ID) == len(CAPABILITIES) >= 25
    for capability in CAPABILITIES:
        assert capability.id.count(".") == 1 and capability.id == capability.id.lower()
        assert capability.description.strip()
        assert any(capability.surfaces().values()), f"{capability.id}: no surface at all"
    required = {
        "models.list",
        "models.load",
        "models.unload",
        "models.delete",
        "models.download",
        "models.test",
        "engine.list",
        "engine.install",
        "engine.activate",
        "config.get",
        "config.set",
        "server.restart",
        "status.get",
        "leases.list",
        "health.get",
        "update.check",
        "update.apply",
        "update.rollback",
        "logs.read",
    }
    assert required <= set(BY_ID), sorted(required - set(BY_ID))


def test_every_declared_api_route_exists(
    surfaces: tuple[dict[str, Any], list[dict[str, Any]]],
) -> None:
    routes = _routes(surfaces[0])
    bad = [
        f"{c.id}: {c.api[0]} {c.api[1]}"
        for c in CAPABILITIES
        if c.api is not None and (c.api[0], c.api[1]) not in routes
    ]
    assert not bad, f"declared API routes that the FastAPI app does not serve: {bad}"


def test_every_declared_mcp_tool_is_registered(
    surfaces: tuple[dict[str, Any], list[dict[str, Any]]],
) -> None:
    registered = {(row["server"], row["name"]) for row in surfaces[1]}
    bad = [
        f"{c.id}: {c.mcp.server}:{c.mcp.tool}"
        for c in CAPABILITIES
        if c.mcp is not None and (c.mcp.server, c.mcp.tool) not in registered
    ]
    assert not bad, f"declared MCP tools that no server registers: {bad}"


def test_every_declared_sfctl_command_exists() -> None:
    commands = _sfctl_commands()
    bad = [
        f"{c.id}: {c.sfctl}"
        for c in CAPABILITIES
        if c.sfctl is not None and _command_path(c.sfctl) not in commands
    ]
    assert not bad, f"declared sfctl commands that are not registered: {bad}"


def test_sfctl_flag_selected_behaviours_are_real_options() -> None:
    """``recover --kill`` must name a flag ``recover`` really has."""
    root = typer.main.get_command(sfctl.app)
    for c in CAPABILITIES:
        if c.sfctl is None:
            continue
        flags = [p for p in c.sfctl.split() if p.startswith("--")]
        if not flags:
            continue
        command: Any = root
        for part in _command_path(c.sfctl).split():
            command = command.commands[part]
        options = {opt for param in command.params for opt in getattr(param, "opts", [])}
        assert set(flags) <= options, (c.id, flags, sorted(options))


def test_every_declared_cli_command_exists() -> None:
    commands = _cli_commands()
    bad = [
        f"{c.id}: {c.cli}"
        for c in CAPABILITIES
        if c.cli is not None and _command_path(c.cli) not in commands
    ]
    assert not bad, f"declared `studioforge` commands that are not registered: {bad}"


def test_cli_flag_selected_behaviours_are_real_options() -> None:
    """``unload --all`` must name a flag ``unload`` really has."""
    for c in CAPABILITIES:
        if c.cli is None:
            continue
        flags = [p for p in c.cli.split() if p.startswith("--")]
        if not flags:
            continue
        command: Any = _cli_root()
        for part in _command_path(c.cli).split():
            command = command.commands[part]
        options = {opt for param in command.params for opt in getattr(param, "opts", [])}
        assert set(flags) <= options, (c.id, flags, sorted(options))


def test_the_security_and_operations_capabilities_have_an_api_and_a_cli() -> None:
    for cap_id in (
        "keys.list",
        "keys.create",
        "keys.revoke",
        "users.create",
        "users.password",
        "doctor.run",
        "support_bundle.create",
        "server.stop",
    ):
        capability = BY_ID[cap_id]
        assert capability.api is not None and capability.cli is not None, cap_id


def test_no_two_capabilities_claim_the_same_route() -> None:
    seen: dict[tuple[str, str], str] = {}
    for c in CAPABILITIES:
        if c.api is None:
            continue
        assert c.api not in seen, f"{c.id} and {seen[c.api]} both claim {c.api}"
        seen[c.api] = c.id


def test_report_surfaces_that_do_not_exist_yet() -> None:
    """Never fails: lists the gaps so later workstreams can close them."""
    gaps = missing_surfaces()
    total = sum(len(v) for v in gaps.values())
    lines = [f"  {cap_id}: {', '.join(missing)}" for cap_id, missing in sorted(gaps.items())]
    no_cli = missing_cli()
    report = (
        f"{total} missing surfaces across {len(gaps)} capabilities:\n"
        + "\n".join(lines)
        + f"\n{len(no_cli)} capabilities have no `studioforge` command yet: {', '.join(no_cli)}"
    )
    print("\n" + report)  # visible with -s
    if gaps:
        warnings.warn(report, UserWarning, stacklevel=1)
    assert total == sum(len(c.missing()) for c in CAPABILITIES)  # the report is the registry's view


def test_export_lists_every_capability_once() -> None:
    rows = export()["capabilities"]
    assert [r["id"] for r in rows] == sorted(BY_ID)
