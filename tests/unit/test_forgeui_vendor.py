"""The vendored ForgeUI bundle (gui/theme/) must not drift from forgeui.json.

Two layers:

* ``test_vendored_copy_*`` need only the standard library: they prove the folder is the
  bundle it claims to be (every ``files.json`` hash matches, nothing hand-edited, nothing
  extra but the app-owned overlay) and that the app's theme settings come from the manifest.
* ``test_forgeui_vendor_check_passes`` runs the real ``python -m forgeui vendor --check``
  against the installed ``forgeui`` package. CI installs it and sets ``FORGEUI_REQUIRED=1``
  so a missing package fails instead of skipping; locally it skips when forgeui is absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "forgeui.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _dest() -> Path:
    return ROOT / _manifest()["dest"]


def _digest(path: Path) -> str:
    """sha256 over LF line endings, as the bundle's files.json is computed."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_vendored_copy_matches_its_files_json() -> None:
    dest = _dest()
    listed = json.loads((dest / "files.json").read_text(encoding="utf-8"))["files"]
    problems = [f"missing {rel}" for rel in listed if not (dest / rel).is_file()]
    problems += [
        f"edited {rel}: move the change into the overlay and re-run `python -m forgeui vendor`"
        for rel, want in listed.items()
        if (dest / rel).is_file() and _digest(dest / rel) != want
    ]
    owned = set(listed) | set(_manifest().get("extra_files", {})) | {"files.json"}
    present = [
        p.relative_to(dest).as_posix()
        for p in sorted(dest.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    ]
    problems += [f"unexpected {rel}" for rel in present if rel not in owned]
    assert not problems, problems


def test_vendored_copy_is_the_version_the_manifest_accepts() -> None:
    stamp = (_dest() / "VERSION").read_text(encoding="utf-8").split()
    assert stamp[0] in {"ThemeForge", "ForgeUI"}, stamp
    want = _manifest()["version"]
    assert stamp[1] == want or stamp[1].startswith(want + "."), (stamp, want)
    assert re.fullmatch(r"\d+\.\d+\.\d+", stamp[1]), stamp


def test_overlays_are_app_owned_files_not_appended_into_the_bundle() -> None:
    manifest = _manifest()
    dest = _dest()
    listed = json.loads((dest / "files.json").read_text(encoding="utf-8"))["files"]
    assert manifest["extra_files"], "StudioForge keeps its sf-* helpers in an overlay"
    for target, source in manifest["extra_files"].items():
        assert target not in listed, f"{target} is a bundle file; overlays need their own name"
        assert (dest / target).read_bytes().replace(b"\r\n", b"\n") == (
            ROOT / source
        ).read_bytes().replace(b"\r\n", b"\n"), f"{target} is stale: run `python -m forgeui vendor`"


def test_settings_name_themes_the_bundle_has() -> None:
    settings = _manifest()["settings"]
    registry = json.loads((_dest() / "themes.json").read_text(encoding="utf-8"))["themes"]
    known = {t["slug"]: t for t in registry}
    assert set(settings["themes"]) <= set(known)
    assert settings["default"] == "auto" or settings["default"] in settings["themes"]
    # ForgeUI 1.2's accessibility themes are offered; opt-in themes only if listed on purpose.
    assert {"slate", "high-contrast", "high-contrast-light"} <= set(settings["themes"])
    extra_opt_in = {s for s in settings["themes"] if known[s].get("set") == "opt-in"}
    assert extra_opt_in <= {"night-red"}, extra_opt_in


def test_app_theme_settings_have_one_source() -> None:
    from studioforge.gui.app import THEME_SETTINGS

    settings = _manifest()["settings"]
    assert list(THEME_SETTINGS["themes"]) == settings["themes"]
    assert THEME_SETTINGS["default"] == settings["default"]
    assert THEME_SETTINGS["storage_key"] == settings["storage_key"]


def test_wheel_ships_the_manifest_the_gui_reads_at_runtime() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"forgeui.json" = "studioforge/gui/forgeui.json"' in pyproject


def test_forgeui_vendor_check_passes() -> None:
    try:
        from forgeui import vendor
    except ImportError:
        if os.environ.get("FORGEUI_REQUIRED"):
            pytest.fail("FORGEUI_REQUIRED is set but the forgeui package is not installed")
        pytest.skip("forgeui is not installed (pip install it to run the drift check locally)")
    lines: list[str] = []
    code = vendor.run(MANIFEST, check=True, out=lines.append)
    assert code == 0, "\n".join(lines)
