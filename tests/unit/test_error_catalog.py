"""The error catalog is complete, and every code in ``src/`` has a row.

P3 ("no dead ends") needs a code, a status, a message, a fix and a docs slug for
every error a caller can see. The catalog (``studioforge.error_catalog``) holds
them; this test is what keeps it honest: it scans ``src/`` with ``ast`` for every
way the codebase spells an error code and fails when one is not catalogued (and,
the other way round, when a catalogued code is no longer raised anywhere).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from studioforge import error_catalog
from studioforge.error_catalog import CATALOG, SURFACES, envelope_extras
from studioforge.errors import (
    AuthError,
    BadRequestError,
    ModelNotFoundError,
    StudioForgeError,
)

SRC = Path(__file__).resolve().parents[2] / "src" / "studioforge"

#: Keyword arguments / assignment targets whose string value is an error code.
_CODE_NAMES = frozenset({"code", "reason_code"})


def _str(node: ast.expr | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def raised_codes() -> dict[str, list[str]]:
    """Every error-code literal in ``src/``, mapped to the ``path:line`` it appears at.

    Spellings recognised: ``f(code="x")`` / ``reason_code="x"``; a class-body or
    local ``code = "x"`` / ``.reason_code = "x"``; and an envelope dict literal
    ``{"code": "x"}``.
    """
    found: dict[str, list[str]] = {}

    def add(code: str | None, path: Path, line: int) -> None:
        if code:
            found.setdefault(code, []).append(f"{path.relative_to(SRC.parent).as_posix()}:{line}")

    for path in sorted(SRC.rglob("*.py")):
        if "gui/theme" in path.as_posix():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg in _CODE_NAMES:
                add(_str(node.value), path, node.value.lineno)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    name = (
                        target.id
                        if isinstance(target, ast.Name)
                        else target.attr
                        if isinstance(target, ast.Attribute)
                        else None
                    )
                    if name in _CODE_NAMES:
                        add(_str(node.value), path, node.lineno)
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values, strict=True):
                    if _str(key) == "code":
                        add(_str(value), path, key.lineno if key else node.lineno)
    return found


def test_every_code_in_src_is_catalogued() -> None:
    missing = {code: where for code, where in raised_codes().items() if code not in CATALOG}
    assert not missing, (
        "error codes raised in src/ but missing from studioforge/error_catalog.py "
        "(add a row with a status, message, fix and docs slug -- never rename a code): "
        + "; ".join(f"{code} ({', '.join(where[:2])})" for code, where in sorted(missing.items()))
    )


def test_no_catalogued_code_is_dead() -> None:
    raised = raised_codes()
    stale = sorted(code for code in CATALOG if code not in raised)
    assert not stale, f"catalogued but not raised anywhere in src/ (remove or fix): {stale}"


def test_the_scan_finds_the_known_spellings() -> None:
    """Guards the scanner itself: if these vanish, the scan is silently blind."""
    raised = raised_codes()
    for code in (
        "model_not_found",  # class attribute
        "confirmation_required",  # keyword argument
        "internal_error",  # envelope dict
        "allowed_devices_unavailable",  # reason_code=
        "gated_repo",  # nested details dict
    ):
        assert code in raised, f"scanner no longer sees {code}"


@pytest.mark.parametrize("code", sorted(CATALOG))
def test_every_entry_is_complete(code: str) -> None:
    entry = CATALOG[code]
    assert entry.code == code
    assert entry.message.strip(), f"{code}: empty message template"
    assert entry.fix.strip(), f"{code}: empty fix"
    assert entry.docs == f"errors/{code}"
    assert entry.error_type.strip()
    assert entry.surface in SURFACES
    if entry.surface in {"api", "panel"}:
        assert isinstance(entry.http_status, int) and 400 <= entry.http_status <= 599, code
    else:
        assert entry.http_status is None or 400 <= entry.http_status <= 599


def test_codes_are_stable_snake_case() -> None:
    for code in CATALOG:
        assert code == code.lower() and " " not in code and code.replace("_", "").isalnum(), code


def test_export_is_sorted_and_complete() -> None:
    rows = error_catalog.export()
    assert [r["code"] for r in rows] == sorted(CATALOG)
    for row in rows:
        assert {"code", "http_status", "error_type", "message", "fix", "docs", "surface"} <= set(
            row
        )


def test_class_defaults_agree_with_the_catalog() -> None:
    """A class's built-in status/type must be what the catalog documents."""
    classes: list[type[StudioForgeError]] = [
        subclass
        for subclass in _all_subclasses(StudioForgeError)
        if subclass.code is not None and subclass.code in CATALOG
    ]
    assert len(classes) >= 15
    for cls in classes:
        entry = CATALOG[str(cls.code)]
        assert entry.http_status == cls.status_code, (cls.__name__, entry.code)
        assert entry.error_type == cls.error_type, (cls.__name__, entry.code)


def _all_subclasses(cls: type[StudioForgeError]) -> set[type[StudioForgeError]]:
    # Import the modules that define subclasses, so the walk sees them.
    import studioforge.core.autostart
    import studioforge.core.downloader
    import studioforge.core.engine
    import studioforge.core.hf_meta
    import studioforge.core.updater  # noqa: F401

    out: set[type[StudioForgeError]] = set()
    for sub in cls.__subclasses__():
        out.add(sub)
        out |= _all_subclasses(sub)
    return out


def test_envelope_gains_fix_and_docs_and_keeps_the_four_keys() -> None:
    payload = ModelNotFoundError("nope").to_payload()["error"]
    assert {"message", "type", "code", "param"} <= set(payload)
    assert payload["code"] == "model_not_found"
    assert payload["type"] == "invalid_request_error"
    assert payload["fix"] == CATALOG["model_not_found"].fix
    assert payload["docs"] == "errors/model_not_found"


def test_uncatalogued_code_gets_no_extras_and_nothing_breaks() -> None:
    payload = BadRequestError("x", code="made_up_code").to_payload()["error"]
    assert "fix" not in payload and "docs" not in payload
    assert payload["code"] == "made_up_code"
    assert envelope_extras(None) == {}
    assert StudioForgeError("boom").to_payload()["error"]["code"] is None


def test_auth_error_envelope_over_http_shape() -> None:
    payload = AuthError("bad key").to_payload()["error"]
    assert payload["code"] == "invalid_api_key" and payload["docs"] == "errors/invalid_api_key"
