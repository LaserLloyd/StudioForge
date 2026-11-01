"""Windows owner-only write ordering: restrict first, and never write when it failed."""

from __future__ import annotations

from pathlib import Path

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
