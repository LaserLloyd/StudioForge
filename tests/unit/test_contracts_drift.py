"""The committed contracts and generated reference docs equal what the code generates.

``docs/contracts/*.json`` and ``docs/reference/*.md`` are produced from code by
``scripts/gen_docs.py``. This is the P5 gate: change a route, an MCP tool, an
error code, a setting or a capability without regenerating, and this fails
with the one command that fixes it. CI runs the same check as
``scripts/gen_docs.py --check``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from studioforge import __version__, contracts

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_FILES = {
    "docs/contracts/openapi.json",
    "docs/contracts/mcp-tools.json",
    "docs/contracts/errors.json",
    "docs/contracts/settings-schema.json",
    "docs/contracts/capabilities.json",
    "docs/reference/settings.md",
    "docs/reference/errors.md",
    "docs/CLI.md",
}


@pytest.fixture(scope="module")
def generated() -> dict[str, str]:
    return contracts.render_contracts()


def test_the_expected_files_are_generated(generated: dict[str, str]) -> None:
    assert set(generated) == EXPECTED_FILES


def test_committed_contracts_equal_regenerated(generated: dict[str, str]) -> None:
    stale = contracts.stale_files(generated, REPO_ROOT)
    assert not stale, (
        f"stale generated files: {stale} -- run `uv run python scripts/gen_docs.py` "
        "and commit the result (a breaking OpenAPI/MCP change needs a version bump)"
    )


def test_generation_is_deterministic(generated: dict[str, str]) -> None:
    """Building the app a second time yields byte-identical contracts."""
    assert contracts.render_contracts() == generated


def test_openapi_describes_the_gateway(generated: dict[str, str]) -> None:
    doc: dict[str, Any] = json.loads(generated["docs/contracts/openapi.json"])
    assert doc["info"]["title"] == "StudioForge" and doc["info"]["version"] == __version__
    for path in ("/v1/chat/completions", "/api/models", "/api/settings/schema", "/health"):
        assert path in doc["paths"], path


def test_mcp_contract_lists_both_servers(generated: dict[str, str]) -> None:
    doc = json.loads(generated["docs/contracts/mcp-tools.json"])
    servers: dict[str, int] = {}
    for tool in doc["tools"]:
        servers[tool["server"]] = servers.get(tool["server"], 0) + 1
        assert tool["description"].strip() and isinstance(tool["input_schema"], dict)
        assert "annotations" in tool  # placeholder until MCP hardening fills it
    assert servers == {"gateway": 23, "watchdog": 10}
    assert doc["count"] == 33


def test_contract_json_is_canonical(generated: dict[str, str]) -> None:
    for path, text in generated.items():
        if path.endswith(".json"):
            assert "\r" not in text and text.endswith("\n")
            assert text == contracts.dump_json(json.loads(text)), path


def test_errors_and_settings_contracts_match_their_registries(generated: dict[str, str]) -> None:
    from studioforge import error_catalog, settings_schema

    errors = json.loads(generated["docs/contracts/errors.json"])
    assert [e["code"] for e in errors["errors"]] == sorted(error_catalog.CATALOG)
    settings = json.loads(generated["docs/contracts/settings-schema.json"])
    assert settings["count"] == len(settings_schema.build_registry())
