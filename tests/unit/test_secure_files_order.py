"""Windows owner-only write ordering: restrict first, and never write when it failed."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from studioforge import secure_files


def _trace(monkeypatch, *, restrict_ok: bool, verdict: bool | None) -> list[str]:
    calls: list[str] = []

    def restrict(path: Path, *, directory: bool = False) -> bool:
        calls.append(f"restrict:{path.stat().st_size}")  # size at restrict time
        return restrict_ok

    monkeypatch.setattr(secure_files, "restrict_to_owner", restrict)
    monkeypatch.setattr(secure_files, "is_owner_only", lambda path: verdict)
    return calls


def test_existing_file_is_restricted_before_it_is_truncated(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "config.yaml"
    target.write_bytes(b"old-content")
    calls = _trace(monkeypatch, restrict_ok=True, verdict=True)
    secure_files._write_private_windows(target, b"secret")
    assert calls == ["restrict:11"]  # old bytes still there when the ACL was applied
    assert target.read_bytes() == b"secret"


def test_failed_restriction_on_an_open_file_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "new.key"
    _trace(monkeypatch, restrict_ok=False, verdict=False)
    with pytest.raises(PermissionError):
        secure_files._write_private_windows(target, b"secret")
    assert not target.exists()

    existing = tmp_path / "old.key"
    existing.write_bytes(b"old")
    with pytest.raises(PermissionError):
        secure_files._write_private_windows(existing, b"secret")
    assert existing.read_bytes() == b"old"


def test_unverifiable_filesystem_still_saves(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "fat.key"
    _trace(monkeypatch, restrict_ok=False, verdict=None)
    secure_files._write_private_windows(target, b"secret")
    assert target.read_bytes() == b"secret"


# Verified every time, not only when icacls failed (review 2026-11-09)


def _acl_trace(
    monkeypatch: pytest.MonkeyPatch, *, restrict_ok: bool, verdict: bool | None
) -> list[str]:
    calls: list[str] = []

    def restrict(path: Path, *, directory: bool = False) -> bool:
        calls.append("restrict")
        return restrict_ok

    def check(path: Path) -> bool | None:
        calls.append("verify")
        return verdict

    monkeypatch.setattr(secure_files, "restrict_to_owner", restrict)
    monkeypatch.setattr(secure_files, "is_owner_only", check)
    return calls


def test_an_explicit_foreign_ace_that_survives_restriction_blocks_the_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "config.yaml"
    existing.write_bytes(b"old")
    calls = _acl_trace(monkeypatch, restrict_ok=True, verdict=False)
    with pytest.raises(PermissionError, match="explicit"):
        secure_files._write_private_windows(existing, b"secret")
    assert calls == ["restrict", "verify"]
    assert existing.read_bytes() == b"old"

    fresh = tmp_path / "new.key"
    with pytest.raises(PermissionError):
        secure_files._write_private_windows(fresh, b"secret")
    assert not fresh.exists()


def test_a_verified_restriction_is_checked_and_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "config.yaml"
    calls = _acl_trace(monkeypatch, restrict_ok=True, verdict=True)
    secure_files._write_private_windows(target, b"secret")
    assert calls == ["restrict", "verify"] and target.read_bytes() == b"secret"


@pytest.mark.skipif(os.name != "nt", reason="icacls output parsing is Windows-only")
def test_owner_only_ignores_a_localised_summary_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import getpass

    target = tmp_path / "f.txt"
    target.write_text("x", encoding="utf-8")
    user = getpass.getuser()
    stdout = (
        f"{target} HOST\\{user}:(F)\n\n"
        "1 Dateien erfolgreich verarbeitet, bei 0 Dateien ist ein Fehler aufgetreten.\n"
    )

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(returncode=0, stdout=stdout)

    monkeypatch.setattr(secure_files.subprocess, "run", fake_run)
    assert secure_files.is_owner_only(target) is True
