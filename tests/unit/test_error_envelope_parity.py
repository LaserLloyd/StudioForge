"""#14: fix/docs on framework 404/405, MCP and watchdog error results, SSE frames."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from fastapi.testclient import TestClient

from studioforge.api.openai_routes import _sse_error
from studioforge.error_catalog import CATALOG, with_error_extras
from studioforge.watchdog.server import _error, _needs_confirmation

ROOT = Path(__file__).resolve().parents[2]


def test_framework_codes_are_catalogued() -> None:
    assert CATALOG["not_found"].http_status == 404
    assert CATALOG["method_not_allowed"].http_status == 405


def test_with_error_extras_adds_and_never_overwrites() -> None:
    res = with_error_extras(
        {"ok": False, "error": {"code": "confirmation_required", "fix": "mine"}}
    )
    assert res["error"]["fix"] == "mine"
    assert res["error"]["docs"] == "errors/confirmation_required"
    unknown = with_error_extras({"ok": False, "error": {"code": "zzz_unknown"}})
    assert "fix" not in unknown["error"]


def test_watchdog_results_carry_fix_and_docs() -> None:
    for result in (_error("boom", "internal_error"), _needs_confirmation("do it", "Bad.")):
        assert result["error"]["fix"]
        assert result["error"]["docs"].startswith("errors/")


def test_sse_error_frame_carries_fix_and_docs() -> None:
    body = json.loads(_sse_error("x", code="internal_error").decode()[5:])
    assert body["error"]["fix"] and body["error"]["docs"] == "errors/internal_error"


def test_framework_404_and_405_envelope(tmp_path) -> None:
    from studioforge.api.app import build_state, create_app
    from tests.unit.test_mcp_hardening import make_config

    config = make_config(tmp_path)
    app = create_app(config, state=build_state(config), start_background=False)
    with TestClient(app) as client:
        r = client.get("/api/definitely-not-a-route")
        assert r.status_code == 404
        err = r.json()["error"]
        assert err["code"] == "not_found" and err["fix"] and err["docs"] == "errors/not_found"
        r = client.delete("/health")
        assert r.status_code == 405
        assert r.json()["error"]["code"] == "method_not_allowed"


def test_ast_scan_of_packages_literals_are_catalogued() -> None:
    """Error codes spelled as ``code=...`` in packages/ must be in the catalog or local."""
    seen: set[str] = set()
    for path in (ROOT / "packages").rglob("*.py"):
        if "/tests/" in path.as_posix() or ".venv" in path.as_posix():
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values, strict=True):
                    if (
                        isinstance(k, ast.Constant)
                        and k.value == "code"
                        and isinstance(v, ast.Constant)
                        and isinstance(v.value, str)
                    ):
                        seen.add(v.value)
    # Companion/tenant packages may carry client-side codes; they must at least
    # be snake_case so a future catalogue entry can adopt them.
    assert all(c.islower() and " " not in c for c in seen), seen
