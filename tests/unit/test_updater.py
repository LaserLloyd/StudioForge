"""Tests for app self-update, including the auto-rollback guarantee."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import threading
import zipfile
from pathlib import Path
from typing import Any

import pytest

from studioforge.config import Config
from studioforge.core.updater import (
    ReleaseInfo,
    UpdateError,
    Updater,
    _extract,
    _find_checksum,
    _parse_release,
    _sha256_file,
    _version_key,
)


def make_config(tmp_path: Path, **kwargs: Any) -> Config:
    # A configured release repo, so the checks below exercise the network
    # path; an unconfigured one short-circuits (see the "not configured" tests).
    kwargs.setdefault("update", {"repo": "example-org/studioforge"})
    config = Config(data_dir=tmp_path / "data", **kwargs)
    config.ensure_dirs()
    return config


def make_release_zip(path: Path, *, top_dir: str | None = "studioforge-v0.2.0") -> bytes:
    """A release archive shaped like GitHub's: everything under one top dir."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        prefix = f"{top_dir}/" if top_dir else ""
        zf.writestr(f"{prefix}pyproject.toml", "[project]\nname='studioforge'\n")
        zf.writestr(f"{prefix}src/studioforge/__init__.py", "__version__='0.2.0'\n")
        zf.writestr(f"{prefix}README.md", "# StudioForge\n")
    data = buffer.getvalue()
    path.write_bytes(data)
    return data


# ---------------------------------------------------------------------------
# Version comparison
# ---------------------------------------------------------------------------


def test_version_key_is_numeric_not_lexical() -> None:
    """A string compare would put 0.9.0 above 0.10.0 and block real updates."""
    assert _version_key("0.10.0") > _version_key("0.9.0")
    assert _version_key("v1.0.0") > _version_key("0.99.99")
    assert _version_key("1.2") == _version_key("1.2.0")


def test_version_key_tolerates_junk() -> None:
    assert _version_key("v2.1.3-rc1") == (2, 1, 3)
    assert _version_key("garbage") == (0, 0, 0)
    assert _version_key("1.0.0+build7") == (1, 0, 0)


def test_version_key_reads_the_hyphenated_calendar_date() -> None:
    """``1.26-08-23`` and the PEP 440 ``1.26.8.23`` must be the same version.

    The display string, the tag and the wheel metadata all say the same date in
    three spellings; if they did not sort equal the updater would offer the
    running build to itself as an update, or refuse a genuinely newer one.
    """
    assert _version_key("1.26-08-23") == _version_key("1.26.8.23") == (1, 26, 8, 23)
    assert _version_key("v1.26-08-23") == _version_key("1.26-08-23")
    assert _version_key("1.26-08-23") > _version_key("0.2.0")
    assert _version_key("1.26-08-24") > _version_key("1.26-08-23")
    assert _version_key("1.27-01-04") > _version_key("1.26-12-31")


def test_version_key_still_drops_a_prerelease_suffix() -> None:
    """A non-numeric hyphen chunk must not sort the candidate above the release."""
    assert _version_key("1.26-08-23-rc1") == _version_key("1.26-08-23")
    assert _version_key("1.0.0-beta") < _version_key("1.0.1")


def test_release_newer_than_current() -> None:
    old = ReleaseInfo(
        tag="v0.0.1",
        version="0.0.1",
        name="",
        published_at=None,
        prerelease=False,
        notes="",
        asset_name=None,
        asset_url=None,
        asset_size=0,
        checksum_url=None,
    )
    new = ReleaseInfo(**{**old.__dict__, "version": "99.0.0", "tag": "v99.0.0"})
    assert not old.is_newer_than_current
    assert new.is_newer_than_current


# ---------------------------------------------------------------------------
# Pointer file
# ---------------------------------------------------------------------------


def test_pointer_roundtrip_and_previous(tmp_path: Path) -> None:
    updater = Updater(make_config(tmp_path))
    for name in ("v0.1.0", "v0.2.0", "v0.10.0"):
        (updater.releases_dir / name).mkdir(parents=True)

    assert updater.current_release() is None
    updater._write_pointer("v0.2.0")
    assert updater.current_release() == "v0.2.0"
    # Newest installed that is not active -- 0.10.0 beats 0.1.0 numerically.
    assert updater.previous_release() == "v0.10.0"

    updater._write_pointer("v0.10.0")
    assert updater.previous_release() == "v0.2.0"


def test_installed_releases_sorted_newest_first(tmp_path: Path) -> None:
    updater = Updater(make_config(tmp_path))
    for name in ("v0.9.0", "v0.10.0", "v0.2.0"):
        (updater.releases_dir / name).mkdir(parents=True)
    assert updater.installed_releases() == ["v0.10.0", "v0.9.0", "v0.2.0"]


def test_pointer_write_is_atomic_leaves_no_temp(tmp_path: Path) -> None:
    updater = Updater(make_config(tmp_path))
    updater._write_pointer("v1.0.0")
    leftovers = list(updater.config.data_dir.glob("current.tmp*"))
    assert leftovers == []
    assert updater.pointer_path.read_text(encoding="utf-8").strip() == "v1.0.0"


def test_missing_releases_dir_is_not_an_error(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path / "nothing-here")
    updater = Updater(config)
    assert updater.installed_releases() == []
    assert updater.current_release() is None
    assert updater.previous_release() is None


# ---------------------------------------------------------------------------
# Archive handling
# ---------------------------------------------------------------------------


def test_extract_flattens_single_top_level_dir(tmp_path: Path) -> None:
    """Release archives wrap in project-vX.Y.Z/; keeping it would break paths."""
    archive = tmp_path / "release.zip"
    make_release_zip(archive, top_dir="studioforge-v0.2.0")
    dest = tmp_path / "installed"
    _extract(archive, dest)
    assert (dest / "pyproject.toml").is_file()
    assert (dest / "src" / "studioforge" / "__init__.py").is_file()
    assert not (dest / "studioforge-v0.2.0").exists()


def test_extract_handles_already_flat_archive(tmp_path: Path) -> None:
    archive = tmp_path / "flat.zip"
    make_release_zip(archive, top_dir=None)
    dest = tmp_path / "installed"
    _extract(archive, dest)
    assert (dest / "pyproject.toml").is_file()


def test_extract_overwrites_existing_release_dir(tmp_path: Path) -> None:
    dest = tmp_path / "installed"
    dest.mkdir()
    (dest / "pyproject.toml").write_text("stale", encoding="utf-8")
    archive = tmp_path / "release.zip"
    make_release_zip(archive)
    _extract(archive, dest)
    assert "stale" not in (dest / "pyproject.toml").read_text(encoding="utf-8")


def test_extract_leaves_no_staging_dir(tmp_path: Path) -> None:
    archive = tmp_path / "release.zip"
    make_release_zip(archive)
    dest = tmp_path / "installed"
    _extract(archive, dest)
    assert not list(tmp_path.glob(".*staging"))


def test_zip_path_traversal_is_refused(tmp_path: Path) -> None:
    """A release archive is remote input; ../ must never escape the dest."""
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../pwned.txt", "nope")
    with pytest.raises(UpdateError, match="unsafe path"):
        _extract(archive, tmp_path / "installed")
    assert not (tmp_path.parent / "pwned.txt").exists()


def test_tar_link_entries_are_refused(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    with pytest.raises(UpdateError, match="link entry"):
        _extract(archive, tmp_path / "installed")


def test_tar_traversal_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "evil2.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        data = b"nope"
        info = tarfile.TarInfo("../../pwned2.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    with pytest.raises(UpdateError, match="unsafe path"):
        _extract(archive, tmp_path / "installed")


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------


def test_find_checksum_sha256sum_format() -> None:
    text = "abc123  other-file.zip\ndeadbeef  studioforge-v0.2.0.zip\ncafe  third.tar.gz\n"
    assert _find_checksum(text, "studioforge-v0.2.0.zip") == "deadbeef"
    assert _find_checksum(text, "missing.zip") is None


def test_find_checksum_binary_star_prefix() -> None:
    assert _find_checksum("deadbeef *release.zip\n", "release.zip") == "deadbeef"


def test_find_checksum_bare_hash() -> None:
    bare = "a" * 64
    assert _find_checksum(bare + "\n", "anything.zip") == bare


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    target = tmp_path / "blob.bin"
    payload = b"studioforge" * 5000
    target.write_bytes(payload)
    assert _sha256_file(target) == hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# GitHub payload parsing
# ---------------------------------------------------------------------------


def test_parse_release_picks_archive_and_checksum() -> None:
    payload = {
        "tag_name": "v0.3.0",
        "name": "0.3.0",
        "published_at": "2026-08-01T00:00:00Z",
        "prerelease": False,
        "body": "notes",
        "assets": [
            {"name": "notes.txt", "browser_download_url": "u1", "size": 1},
            {"name": "studioforge-0.3.0.zip", "browser_download_url": "u2", "size": 4096},
            {"name": "SHA256SUMS.txt", "browser_download_url": "u3", "size": 64},
        ],
    }
    release = _parse_release(payload)
    assert release.version == "0.3.0"
    assert release.asset_name == "studioforge-0.3.0.zip"
    assert release.asset_size == 4096
    assert release.checksum_url == "u3"
    assert not release.prerelease


def test_parse_release_without_archive() -> None:
    release = _parse_release({"tag_name": "v9.9.9", "assets": []})
    assert release.asset_url is None
    assert release.checksum_url is None


# ---------------------------------------------------------------------------
# Install / rollback against a local release server
# ---------------------------------------------------------------------------


class FakeReleaseServer:
    """Serves a release zip and its checksum over real HTTP."""

    def __init__(self, payload: bytes, *, checksum: str | None) -> None:
        import http.server
        import socketserver

        self.payload = payload
        self.checksum = checksum
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib naming
                if self.path.endswith("SHA256SUMS.txt"):
                    if outer.checksum is None:
                        self.send_response(404)
                        self.end_headers()
                        return
                    body = f"{outer.checksum}  release.zip\n".encode()
                else:
                    body = outer.payload
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: Any) -> None:
                return

        import socket

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = int(probe.getsockname()[1])
        self.httpd = socketserver.TCPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self.httpd.shutdown()


def release_for(server: FakeReleaseServer, version: str = "0.2.0") -> ReleaseInfo:
    return ReleaseInfo(
        tag=f"v{version}",
        version=version,
        name=version,
        published_at=None,
        prerelease=False,
        notes="",
        asset_name="release.zip",
        asset_url=f"{server.base}/release.zip",
        asset_size=len(server.payload),
        checksum_url=f"{server.base}/SHA256SUMS.txt",
    )


async def test_install_switches_pointer(tmp_path: Path) -> None:
    archive = tmp_path / "release.zip"
    payload = make_release_zip(archive)
    server = FakeReleaseServer(payload, checksum=hashlib.sha256(payload).hexdigest())
    try:
        updater = Updater(make_config(tmp_path))
        release = release_for(server)
        updater._resolve_release = lambda tag: _immediate(release)  # type: ignore[assignment]

        result = await updater.install(restart=False)
        assert result["installed"] == "0.2.0"
        assert result["restarted"] is False
        assert updater.current_release() == "0.2.0"
        assert (updater.releases_dir / "0.2.0" / "pyproject.toml").is_file()
    finally:
        server.stop()


async def test_install_verifies_checksum_and_rejects_mismatch(tmp_path: Path) -> None:
    archive = tmp_path / "release.zip"
    payload = make_release_zip(archive)
    server = FakeReleaseServer(payload, checksum="0" * 64)
    try:
        updater = Updater(make_config(tmp_path))
        release = release_for(server)
        updater._resolve_release = lambda tag: _immediate(release)  # type: ignore[assignment]
        with pytest.raises(UpdateError, match="checksum mismatch"):
            await updater.install(restart=False)
        # Nothing was activated, and the bad archive is gone.
        assert updater.current_release() is None
        assert not list(updater.config.downloads_dir.glob("release.zip"))
    finally:
        server.stop()


async def test_install_without_checksum_proceeds_with_warning(tmp_path: Path) -> None:
    """Refusing to update is also a failure mode; warn instead of blocking."""
    archive = tmp_path / "release.zip"
    payload = make_release_zip(archive)
    server = FakeReleaseServer(payload, checksum=None)
    try:
        updater = Updater(make_config(tmp_path))
        release = ReleaseInfo(**{**release_for(server).__dict__, "checksum_url": None})
        updater._resolve_release = lambda tag: _immediate(release)  # type: ignore[assignment]
        result = await updater.install(restart=False)
        assert result["installed"] == "0.2.0"
    finally:
        server.stop()


async def test_install_runs_drain_first(tmp_path: Path) -> None:
    archive = tmp_path / "release.zip"
    payload = make_release_zip(archive)
    server = FakeReleaseServer(payload, checksum=hashlib.sha256(payload).hexdigest())
    calls: list[str] = []

    async def drain() -> None:
        calls.append("drained")

    try:
        updater = Updater(make_config(tmp_path))
        release = release_for(server)
        updater._resolve_release = lambda tag: _immediate(release)  # type: ignore[assignment]
        await updater.install(drain=drain, restart=False)
        assert calls == ["drained"]
    finally:
        server.stop()


async def test_install_auto_rolls_back_when_health_check_fails(tmp_path: Path) -> None:
    """The rollback guarantee: an unhealthy new release is reverted."""
    archive = tmp_path / "release.zip"
    payload = make_release_zip(archive)
    server = FakeReleaseServer(payload, checksum=hashlib.sha256(payload).hexdigest())
    try:
        config = make_config(tmp_path)
        config.update.health_check_timeout_s = 1.0
        updater = Updater(config)

        # A previous release must exist for a rollback to be possible.
        (updater.releases_dir / "0.1.0").mkdir(parents=True)
        updater._write_pointer("0.1.0")

        restarts: list[str] = []
        updater.restart_service = lambda **_: bool(restarts.append("restart"))  # type: ignore[assignment]

        async def never_healthy(_timeout: float, **_kw: Any) -> bool:
            return False

        updater.wait_for_health = never_healthy  # type: ignore[assignment]
        release = release_for(server)
        updater._resolve_release = lambda tag: _immediate(release)  # type: ignore[assignment]

        result = await updater.install(restart=True)
        assert result["healthy"] is False
        assert result["rolled_back"] is True
        assert updater.current_release() == "0.1.0"
        assert len(restarts) == 2  # once forward, once back
    finally:
        server.stop()


async def test_install_keeps_new_release_when_healthy(tmp_path: Path) -> None:
    archive = tmp_path / "release.zip"
    payload = make_release_zip(archive)
    server = FakeReleaseServer(payload, checksum=hashlib.sha256(payload).hexdigest())
    try:
        updater = Updater(make_config(tmp_path))
        (updater.releases_dir / "0.1.0").mkdir(parents=True)
        updater._write_pointer("0.1.0")
        updater.restart_service = lambda **_: True  # type: ignore[assignment]

        async def healthy(_timeout: float, **_kw: Any) -> bool:
            return True

        updater.wait_for_health = healthy  # type: ignore[assignment]
        release = release_for(server)
        updater._resolve_release = lambda tag: _immediate(release)  # type: ignore[assignment]

        result = await updater.install(restart=True)
        assert result["healthy"] is True
        assert result["rolled_back"] is False
        assert updater.current_release() == "0.2.0"
    finally:
        server.stop()


async def test_rollback_without_previous_release_errors(tmp_path: Path) -> None:
    updater = Updater(make_config(tmp_path))
    (updater.releases_dir / "0.1.0").mkdir(parents=True)
    updater._write_pointer("0.1.0")
    with pytest.raises(UpdateError, match="no previous release"):
        await updater.rollback(restart=False)


async def test_rollback_switches_pointer(tmp_path: Path) -> None:
    updater = Updater(make_config(tmp_path))
    for name in ("0.1.0", "0.2.0"):
        (updater.releases_dir / name).mkdir(parents=True)
    updater._write_pointer("0.2.0")
    result = await updater.rollback(restart=False)
    assert result["rolled_back_to"] == "0.1.0"
    assert updater.current_release() == "0.1.0"


async def test_install_refuses_release_without_asset(tmp_path: Path) -> None:
    updater = Updater(make_config(tmp_path))
    release = ReleaseInfo(
        tag="v9.9.9",
        version="9.9.9",
        name="",
        published_at=None,
        prerelease=False,
        notes="",
        asset_name=None,
        asset_url=None,
        asset_size=0,
        checksum_url=None,
    )
    updater._resolve_release = lambda tag: _immediate(release)  # type: ignore[assignment]
    with pytest.raises(UpdateError, match="no installable asset"):
        await updater.install(restart=False)


# ---------------------------------------------------------------------------
# check() and health polling
# ---------------------------------------------------------------------------


async def test_check_reports_update_available(tmp_path: Path) -> None:
    updater = Updater(make_config(tmp_path))
    release = ReleaseInfo(
        tag="v99.0.0",
        version="99.0.0",
        name="",
        published_at=None,
        prerelease=False,
        notes="",
        asset_name="a.zip",
        asset_url="u",
        asset_size=1,
        checksum_url=None,
    )

    async def latest() -> ReleaseInfo:
        return release

    updater.latest_release = latest  # type: ignore[assignment]
    status = await updater.check()
    assert status.update_available is True
    assert status.latest_version == "99.0.0"
    assert status.error is None


async def test_check_surfaces_network_error_instead_of_raising(tmp_path: Path) -> None:
    """The GUI polls this; it must degrade to a message, not a traceback."""
    updater = Updater(make_config(tmp_path))

    async def boom() -> ReleaseInfo:
        raise UpdateError("could not reach GitHub: nope")

    updater.latest_release = boom  # type: ignore[assignment]
    status = await updater.check()
    assert status.update_available is False
    assert status.error is not None
    assert "GitHub" in status.error


async def test_install_reports_restarted_false_when_the_restart_failed(
    tmp_path: Path,
) -> None:
    """'restarted: true' after a failed service restart is an unverified lie."""
    archive = tmp_path / "release.zip"
    payload = make_release_zip(archive)
    server = FakeReleaseServer(payload, checksum=hashlib.sha256(payload).hexdigest())
    try:
        config = make_config(tmp_path)
        config.update.health_check_timeout_s = 1.0
        updater = Updater(config)
        (updater.releases_dir / "0.1.0").mkdir(parents=True)
        updater._write_pointer("0.1.0")
        updater.restart_service = lambda **_: False  # type: ignore[assignment]

        async def healthy(_timeout: float, **_kw: Any) -> bool:
            return True

        updater.wait_for_health = healthy  # type: ignore[assignment]
        release = release_for(server)
        updater._resolve_release = lambda tag: _immediate(release)  # type: ignore[assignment]

        result = await updater.install(restart=True)
        assert result["restarted"] is False, (
            "restart_service returned False but the result claimed a restart happened"
        )
    finally:
        server.stop()


async def test_wait_for_health_rejects_an_ok_from_the_wrong_version(tmp_path: Path) -> None:
    """The old process answering its own health poll must not pass the check.

    Deterministic on Windows: the respawned child exits at its port preflight
    because the old process still holds the port, and the old process then
    answers /health with status ok -- reporting the *old* version. An "ok"
    carrying the wrong version is proof the update did NOT take effect.
    """
    import http.server
    import socket
    import socketserver

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            body = json.dumps({"status": "ok", "version": "0.1.0"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: Any) -> None:
            return

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    httpd = socketserver.TCPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        updater = Updater(make_config(tmp_path), health_url=f"http://127.0.0.1:{port}/health")
        # The old process (0.1.0) answering is NOT the new release being healthy.
        assert await updater.wait_for_health(3.0, expect_version="0.2.0") is False
        # But it IS proof the matching version is up.
        assert await updater.wait_for_health(10.0, expect_version="0.1.0") is True
        # And without an expectation the plain liveness meaning is unchanged.
        assert await updater.wait_for_health(10.0) is True
    finally:
        httpd.shutdown()


async def test_wait_for_health_true_against_live_endpoint(tmp_path: Path) -> None:
    import http.server
    import socket
    import socketserver

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            body = json.dumps({"status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: Any) -> None:
            return

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    httpd = socketserver.TCPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        updater = Updater(make_config(tmp_path), health_url=f"http://127.0.0.1:{port}/health")
        assert await updater.wait_for_health(10.0) is True
    finally:
        httpd.shutdown()


async def test_wait_for_health_false_when_nothing_listens(tmp_path: Path) -> None:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    updater = Updater(make_config(tmp_path), health_url=f"http://127.0.0.1:{port}/health")
    assert await updater.wait_for_health(2.0) is False


def test_status_sync_needs_no_network(tmp_path: Path) -> None:
    updater = Updater(make_config(tmp_path))
    (updater.releases_dir / "0.1.0").mkdir(parents=True)
    updater._write_pointer("0.1.0")
    status = updater.status_sync()
    assert status.current_release == "0.1.0"
    assert status.releases_installed == ["0.1.0"]
    assert status.latest_version is None


def test_prune_keeps_current_and_recent(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.engine.keep_versions = 2
    updater = Updater(config)
    for name in ("0.1.0", "0.2.0", "0.3.0", "0.4.0"):
        (updater.releases_dir / name).mkdir(parents=True)
    updater._write_pointer("0.1.0")  # oldest is active
    updater._prune_releases()
    remaining = set(updater.installed_releases())
    assert "0.1.0" in remaining, "the active release must never be pruned"
    assert "0.4.0" in remaining
    assert len(remaining) <= 3


async def _immediate(value: Any) -> Any:
    return value


# ---------------------------------------------------------------------------
# no update repo configured
# ---------------------------------------------------------------------------


async def test_check_without_a_repo_is_a_local_answer(tmp_path: Path) -> None:
    """The shipped default has no public repo: no network call, no error."""
    config = Config(data_dir=tmp_path / "data")
    assert config.update.repo is None
    updater = Updater(config)

    async def never() -> Any:
        raise AssertionError("must not ask GitHub when no repo is configured")

    updater.latest_release = never  # type: ignore[assignment]
    status = await updater.check()
    assert status.configured is False
    assert status.update_available is False
    assert status.error is None
    assert "update.repo" in (status.note or "")
    assert status.to_dict()["configured"] is False


@pytest.mark.parametrize("value", [None, "", "   ", "studioforge/studioforge", "no-slash"])
def test_placeholder_or_junk_repo_counts_as_unconfigured(tmp_path: Path, value: str | None) -> None:
    config = Config(data_dir=tmp_path / "data", update={"repo": value})
    assert config.update.configured_repo is None


def test_a_real_repo_is_configured(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path / "data", update={"repo": " someone/studioforge "})
    assert config.update.configured_repo == "someone/studioforge"


async def test_list_releases_without_a_repo_raises_a_readable_error(tmp_path: Path) -> None:
    from studioforge.core.updater import UpdateError

    updater = Updater(Config(data_dir=tmp_path / "data"))
    with pytest.raises(UpdateError) as excinfo:
        await updater.list_releases()
    assert "update.repo" in excinfo.value.message
