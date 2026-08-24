"""App self-update with automatic rollback.

Layout (all inside ``data_dir``, which by design survives every update)::

    <data_dir>/releases/1.26-08-23/   unpacked release
    <data_dir>/releases/0.2.0/        previous release, kept for rollback
    <data_dir>/current.txt            the ACTIVE release directory name
    <data_dir>/current                POSIX-only convenience symlink

**Why a pointer file rather than only a symlink:** on Windows, creating a
symlink needs either Developer Mode or administrator rights, so a
symlink-only design would make self-update fail on the secondary platform for
a reason that has nothing to do with updating. ``current.txt`` is one line of
text that works identically everywhere, and it is also what the watchdog reads
-- the watchdog deliberately shares no code with this module (it must keep
working when the main app is wedged), so the two agree on a *file format*
rather than on an implementation.

The rollback guarantee: after switching, the new process must answer
``/health`` within ``update.health_check_timeout_s``. If it does not, the
pointer is put back and the previous release restarted. That check is why the
update is safe to trigger from an agent tool.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from studioforge import __version__
from studioforge.config import Config
from studioforge.core.ports import (
    DEFAULT_RESPAWN_WAIT_S,
    ENV_RESPAWN_PARENT_PID,
    ENV_RESPAWN_WAIT_S,
)
from studioforge.errors import StudioForgeError
from studioforge.logging import get_logger

log = get_logger(__name__)

CURRENT_POINTER = "current.txt"
RELEASES_DIR = "releases"


class UpdateError(StudioForgeError):
    status_code = 500
    error_type = "server_error"
    code = "update_failed"


@dataclass(frozen=True)
class ReleaseInfo:
    tag: str
    version: str
    name: str
    published_at: str | None
    prerelease: bool
    notes: str
    asset_name: str | None
    asset_url: str | None
    asset_size: int
    checksum_url: str | None

    @property
    def is_newer_than_current(self) -> bool:
        return _version_key(self.version) > _version_key(__version__)


@dataclass
class UpdateStatus:
    current_version: str = __version__
    current_release: str | None = None
    latest_version: str | None = None
    latest_tag: str | None = None
    update_available: bool = False
    checked_at: float | None = None
    releases_installed: list[str] = field(default_factory=list)
    previous_release: str | None = None
    error: str | None = None
    #: False when ``update.repo`` is unset (or still the shipped placeholder):
    #: the check made no network call and ``update_available`` means nothing.
    configured: bool = True
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_version": self.current_version,
            "current_release": self.current_release,
            "latest_version": self.latest_version,
            "latest_tag": self.latest_tag,
            "update_available": self.update_available,
            "checked_at": self.checked_at,
            "releases_installed": self.releases_installed,
            "previous_release": self.previous_release,
            "error": self.error,
            "configured": self.configured,
            "note": self.note,
        }


def _version_key(version: str) -> tuple[int, ...]:
    """Sortable tuple from a version string, tolerant of junk.

    Comparison is numeric per component so ``v0.10.0`` correctly beats
    ``v0.9.0`` -- a plain string compare gets that backwards, which would make
    the updater refuse real updates.

    A hyphen means two different things here, so it is read two different ways.
    This project's releases are calendar-versioned and spell the date with
    hyphens (``1.26-08-23``, tagged ``v1.26-08-23``), while the package metadata
    has to say the same thing in PEP 440 (``1.26.8.23``); an all-digit chunk
    after a hyphen is therefore just another numeric component, and the two
    spellings must produce the same key or the updater would think a release is
    newer or older than the build already running. A non-numeric chunk
    (``2.1.3-rc1``) is a pre-release suffix and is dropped along with everything
    after it -- keeping it would sort the release candidate *above* the release.
    """
    cleaned = version.strip().lstrip("vV").split("+", 1)[0]
    head, *suffixes = cleaned.split("-")
    date_parts: list[str] = []
    for chunk in suffixes:
        if not chunk.isdigit():
            break
        date_parts.append(chunk)
    cleaned = ".".join([head, *date_parts])
    parts: list[int] = []
    for chunk in cleaned.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


class Updater:
    """Checks for, installs, and rolls back app releases."""

    def __init__(
        self,
        config: Config,
        *,
        client: httpx.AsyncClient | None = None,
        health_url: str | None = None,
    ) -> None:
        self.config = config
        self._client = client
        self._owns_client = client is None
        #: Handle of the most recent detached respawn, so a caller that is
        #: about to exit can check the replacement actually came up.
        self._last_child: subprocess.Popen[bytes] | None = None
        self._health_url = health_url or (f"http://127.0.0.1:{config.server.port}/health")
        #: How long the replacement may wait for our ports before giving up.
        #: Must comfortably exceed uvicorn's graceful-shutdown budget, which is
        #: ``server.drain_timeout_s`` -- that is the time between us deciding to
        #: exit and the ports actually being free.
        self.respawn_wait_s = round(
            max(DEFAULT_RESPAWN_WAIT_S, config.server.drain_timeout_s + 20.0), 1
        )

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0, follow_redirects=True)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # -- paths -------------------------------------------------------------

    @property
    def releases_dir(self) -> Path:
        return self.config.data_dir / RELEASES_DIR

    @property
    def pointer_path(self) -> Path:
        return self.config.data_dir / CURRENT_POINTER

    def installed_releases(self) -> list[str]:
        """Release directory names, newest version first."""
        if not self.releases_dir.is_dir():
            return []
        names = [p.name for p in self.releases_dir.iterdir() if p.is_dir()]
        return sorted(names, key=_version_key, reverse=True)

    def current_release(self) -> str | None:
        try:
            text = self.pointer_path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return text or None

    def previous_release(self) -> str | None:
        """The newest installed release that is not the active one."""
        current = self.current_release()
        for name in self.installed_releases():
            if name != current:
                return name
        return None

    def _write_pointer(self, name: str) -> None:
        """Point at ``name`` atomically, and refresh the POSIX symlink."""
        self.pointer_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.pointer_path.with_suffix(".tmp")
        tmp.write_text(f"{name}\n", encoding="utf-8")
        tmp.replace(self.pointer_path)

        if os.name != "nt":
            link = self.config.data_dir / "current"
            target = self.releases_dir / name
            with contextlib.suppress(OSError):
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(target, target_is_directory=True)

    # -- release discovery -------------------------------------------------

    #: What ``check()`` says when there is no repo to ask. Not an error: a
    #: build without a public home is the normal state of this project today.
    NOT_CONFIGURED_NOTE = (
        "app self-update is not configured: set update.repo to the GitHub "
        "'owner/name' that publishes StudioForge releases (Setup tab or config.yaml)"
    )

    def repo(self) -> str | None:
        """The release repository, or ``None`` when self-update is unconfigured."""
        return self.config.update.configured_repo

    async def check(self) -> UpdateStatus:
        """Compare the running version against the configured GitHub repo.

        With no repo configured this is a local answer -- no network call, no
        error, ``configured: false`` and a note saying how to turn it on.
        """
        status = UpdateStatus(
            current_release=self.current_release(),
            releases_installed=self.installed_releases(),
            previous_release=self.previous_release(),
            checked_at=time.time(),
        )
        if self.repo() is None:
            status.configured = False
            status.note = self.NOT_CONFIGURED_NOTE
            return status
        try:
            release = await self.latest_release()
        except UpdateError as exc:
            status.error = exc.message
            return status
        if release is None:
            status.error = "no releases found"
            return status
        status.latest_version = release.version
        status.latest_tag = release.tag
        status.update_available = release.is_newer_than_current
        return status

    async def list_releases(self, limit: int = 20) -> list[ReleaseInfo]:
        repo = self.repo()
        if repo is None:
            raise UpdateError(self.NOT_CONFIGURED_NOTE)
        client = await self._http()
        url = f"https://api.github.com/repos/{repo}/releases"
        try:
            response = await client.get(url, params={"per_page": limit}, headers=_gh_headers())
        except httpx.HTTPError as exc:
            raise UpdateError(f"could not reach GitHub: {exc}") from exc
        if response.status_code == 404:
            raise UpdateError(f"update repo '{repo}' not found (set update.repo in config.yaml)")
        if response.status_code >= 400:
            raise UpdateError(f"GitHub returned HTTP {response.status_code} for {url}")
        payload = response.json()
        if not isinstance(payload, list):
            raise UpdateError("unexpected response from GitHub releases API")
        return [_parse_release(item) for item in payload]

    async def latest_release(self) -> ReleaseInfo | None:
        allow_pre = self.config.update.channel == "prerelease"
        releases = await self.list_releases()
        candidates = [r for r in releases if allow_pre or not r.prerelease]
        if not candidates:
            return None
        return max(candidates, key=lambda r: _version_key(r.version))

    # -- install -----------------------------------------------------------

    async def install(
        self,
        tag: str | None = None,
        *,
        drain: Any = None,
        restart: bool = True,
    ) -> dict[str, Any]:
        """Download, verify, install, switch, restart, and verify health.

        ``drain`` is an optional awaitable-returning callable that must settle
        in-flight work before the switch. An update that yanks the process out
        from under a streaming response or a half-finished model download is a
        data-loss bug, so the caller is expected to supply one.
        """
        release = await self._resolve_release(tag)
        if release.asset_url is None:
            raise UpdateError(
                f"release {release.tag} has no installable asset; expected a .zip or .tar.gz"
            )

        previous = self.current_release()
        target_name = release.version if release.version else release.tag
        dest = self.releases_dir / target_name

        archive = await self._download_asset(release)
        try:
            await self._verify_checksum(release, archive)
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            await asyncio.to_thread(_extract, archive, dest)
        finally:
            _unlink_quietly(archive)

        if drain is not None:
            log.info("draining before update switch")
            await drain()

        self._write_pointer(target_name)
        log.info("switched release pointer", release=target_name, previous=previous)

        result: dict[str, Any] = {
            "installed": target_name,
            "previous": previous,
            "restarted": False,
            "healthy": None,
            "rolled_back": False,
        }
        if not restart:
            return result

        # The return value matters: reporting "restarted": true when the
        # service control failed is exactly the kind of unverified success
        # this project exists to avoid.
        result["restarted"] = bool(self.restart_service())

        healthy = await self.wait_for_health(
            self.config.update.health_check_timeout_s,
            expect_version=release.version or None,
        )
        result["healthy"] = healthy
        if not healthy:
            log.error("new release failed its health check; rolling back")
            if previous:
                self._write_pointer(previous)
                self.restart_service()
                result["rolled_back"] = True
                result["healthy_after_rollback"] = await self.wait_for_health(
                    self.config.update.health_check_timeout_s,
                    expect_version=previous,
                )
            else:
                result["rollback_error"] = "no previous release to roll back to"
        self._prune_releases()
        return result

    async def rollback(self, *, restart: bool = True) -> dict[str, Any]:
        """Switch back to the previous release."""
        previous = self.previous_release()
        if previous is None:
            raise UpdateError("no previous release is installed to roll back to")
        current = self.current_release()
        self._write_pointer(previous)
        result: dict[str, Any] = {
            "rolled_back_to": previous,
            "from": current,
            "restarted": False,
        }
        if restart:
            result["restarted"] = bool(self.restart_service())
            result["healthy"] = await self.wait_for_health(
                self.config.update.health_check_timeout_s,
                expect_version=previous,
            )
        return result

    async def _resolve_release(self, tag: str | None) -> ReleaseInfo:
        if tag is None:
            release = await self.latest_release()
            if release is None:
                raise UpdateError("no releases available")
            if not release.is_newer_than_current:
                raise UpdateError(f"already running {__version__}; latest is {release.version}")
            return release
        for release in await self.list_releases(limit=50):
            if release.tag == tag or release.version == tag.lstrip("vV"):
                return release
        raise UpdateError(f"release '{tag}' not found in {self.repo()}")

    async def _download_asset(self, release: ReleaseInfo) -> Path:
        assert release.asset_url is not None
        client = await self._http()
        self.config.downloads_dir.mkdir(parents=True, exist_ok=True)
        dest = self.config.downloads_dir / (release.asset_name or f"{release.tag}.zip")
        try:
            async with client.stream(
                "GET", release.asset_url, headers=_gh_headers(accept_octet=True)
            ) as response:
                if response.status_code >= 400:
                    raise UpdateError(
                        f"downloading {release.asset_name} failed with HTTP {response.status_code}"
                    )
                with dest.open("wb") as handle:
                    async for chunk in response.aiter_bytes(1024 * 1024):
                        handle.write(chunk)
        except httpx.HTTPError as exc:
            # install()'s `finally` never runs when the failure happens here --
            # `archive` is not assigned yet -- so a network drop mid-stream used
            # to strand a truncated release archive in downloads_dir forever.
            _unlink_quietly(dest)
            raise UpdateError(f"downloading the release failed: {exc}") from exc
        except BaseException:
            _unlink_quietly(dest)
            raise
        return dest

    async def _verify_checksum(self, release: ReleaseInfo, archive: Path) -> None:
        """Verify against a published checksum when one exists.

        A release without a checksum file is installed anyway (with a warning)
        rather than blocked, because refusing to update is also a failure mode --
        but the absence is logged so it is visible.
        """
        if release.checksum_url is None:
            log.warning(
                "release has no checksum file; installing unverified",
                tag=release.tag,
            )
            return
        client = await self._http()
        try:
            response = await client.get(release.checksum_url, headers=_gh_headers())
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise UpdateError(f"could not fetch the release checksum: {exc}") from exc

        expected = _find_checksum(response.text, archive.name)
        if expected is None:
            log.warning("checksum file lists no entry for this asset", asset=archive.name)
            return
        actual = await asyncio.to_thread(_sha256_file, archive)
        if actual.lower() != expected.lower():
            _unlink_quietly(archive)
            raise UpdateError(
                f"checksum mismatch for {archive.name}: expected {expected}, got {actual}"
            )
        log.info("release checksum verified", asset=archive.name)

    def _prune_releases(self) -> None:
        keep = max(2, self.config.engine.keep_versions)
        current = self.current_release()
        names = self.installed_releases()
        for name in names[keep:]:
            if name == current:
                continue
            path = self.releases_dir / name
            log.info("pruning old release", release=name)
            shutil.rmtree(path, ignore_errors=True)

    # -- service control ---------------------------------------------------

    def restart_service(self, *, unit: str = "studioforge") -> bool:
        """Restart the service, per-platform.

        Linux uses systemd. Windows has no systemd, so a service restart is
        attempted via ``sc``/sc-equivalent and otherwise falls back to
        re-spawning a detached process -- the watchdog implements the same
        fallback independently for the wedged case.
        """
        if os.name != "nt":
            for command in (
                ["systemctl", "restart", unit],
                ["systemctl", "--user", "restart", unit],
            ):
                try:
                    result = subprocess.run(command, capture_output=True, timeout=60)
                    if result.returncode == 0:
                        return True
                except (OSError, subprocess.TimeoutExpired):
                    continue
            log.warning("systemctl restart failed; the supervisor must restart us")
            return False

        try:
            result = subprocess.run(["sc", "stop", unit], capture_output=True, timeout=60)
            started = subprocess.run(["sc", "start", unit], capture_output=True, timeout=60)
            if started.returncode == 0:
                return True
            log.info(
                "windows service restart unavailable, respawning detached",
                stop_rc=result.returncode,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        return self._respawn_detached()

    def _respawn_detached(self) -> bool:
        """Start a fresh process and let this one exit.

        Used on Windows where there is no service manager. The child is fully
        detached so it survives this process terminating.

        The child is told **which pid it is replacing**
        (``SF_RESPAWN_PARENT_PID``). Without it the restart could not work at
        all: we still hold the API port, the GUI port and -- through our
        watchdog child -- the watchdog port at the moment the child runs its own
        preflight, so the child exited with "startup port conflict" every single
        time, and this process then stayed up because its replacement was
        already dead. With it, the child waits for those ports instead of
        refusing to start, and we exit and hand them over. See DECISIONS.md D21.
        """
        try:
            flags = 0
            if os.name == "nt":
                flags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008  # DETACHED
            env = dict(os.environ)
            env[ENV_RESPAWN_PARENT_PID] = str(os.getpid())
            env.setdefault(ENV_RESPAWN_WAIT_S, str(self.respawn_wait_s))
            # Keep the handle: Popen succeeding says nothing about whether the
            # child SURVIVED. It still has to pass its own port preflight, and
            # callers that exit on the strength of this need a way to check.
            self._last_child = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "studioforge",
                    "serve",
                    "--config",
                    str(self.config.config_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=flags,
                close_fds=True,
                env=env,
            )
            return True
        except Exception as exc:
            log.error("could not respawn the server", error=str(exc))
            return False

    async def wait_for_health(self, timeout_s: float, *, expect_version: str | None = None) -> bool:
        """Poll ``/health`` until it answers ok, or the timeout expires.

        ``expect_version`` is what keeps this check honest after a restart: on
        Windows the respawned child can exit at its port preflight because the
        *old* process still holds the port -- and the old process then answers
        this very poll with ``status: ok``. An "ok" from the wrong version is
        the old code still running, which is exactly the case the caller needs
        to know about, so it does not count as healthy.
        """
        deadline = time.time() + timeout_s
        # A fresh process needs a moment to bind; polling instantly just burns
        # the first second on connection refusals.
        await asyncio.sleep(1.0)
        while time.time() < deadline:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(self._health_url)
                if response.status_code == 200:
                    body = response.json()
                    if body.get("status") == "ok":
                        reported = str(body.get("version") or "")
                        if expect_version is None or reported == expect_version:
                            return True
                        log.warning(
                            "health answered with the wrong version; the old process "
                            "is still running",
                            expected=expect_version,
                            reported=reported,
                        )
            except (httpx.HTTPError, ValueError):
                pass
            await asyncio.sleep(1.0)
        return False

    # -- status ------------------------------------------------------------

    def status_sync(self) -> UpdateStatus:
        """Local-only status (no network), for the GUI's first paint."""
        return UpdateStatus(
            current_release=self.current_release(),
            releases_installed=self.installed_releases(),
            previous_release=self.previous_release(),
        )


def _unlink_quietly(path: Path) -> None:
    """Best-effort delete; a leftover temp file must never fail an update."""
    with contextlib.suppress(OSError):
        path.unlink()


def _gh_headers(*, accept_octet: bool = False) -> dict[str, str]:
    headers = {
        "Accept": "application/octet-stream" if accept_octet else "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": f"studioforge/{__version__}",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _parse_release(item: dict[str, Any]) -> ReleaseInfo:
    tag = str(item.get("tag_name") or "")
    assets = item.get("assets") or []
    asset_name = asset_url = checksum_url = None
    asset_size = 0
    for asset in assets:
        name = str(asset.get("name") or "")
        lowered = name.lower()
        if lowered.endswith((".zip", ".tar.gz", ".tgz")) and asset_url is None:
            asset_name = name
            asset_url = asset.get("url") or asset.get("browser_download_url")
            asset_size = int(asset.get("size") or 0)
        elif "sha256" in lowered or lowered.endswith((".sha256", "sums.txt", "checksums.txt")):
            checksum_url = asset.get("browser_download_url") or asset.get("url")
    return ReleaseInfo(
        tag=tag,
        version=tag.lstrip("vV") or "0.0.0",
        name=str(item.get("name") or tag),
        published_at=item.get("published_at"),
        prerelease=bool(item.get("prerelease")),
        notes=str(item.get("body") or "")[:4000],
        asset_name=asset_name,
        asset_url=asset_url,
        asset_size=asset_size,
        checksum_url=checksum_url,
    )


def _find_checksum(text: str, filename: str) -> str | None:
    """Pull one hash out of a ``sha256sum``-style listing."""
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1].lstrip("*") == filename:
            return parts[0]
        if len(parts) == 1 and len(parts[0]) == 64:
            return parts[0]
    return None


def _sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _extract(archive: Path, dest: Path) -> None:
    """Extract an archive, flattening a single top-level directory.

    Release archives usually wrap everything in ``project-vX.Y.Z/``; without
    flattening, the installed tree would gain a level and every path derived
    from the release root would be wrong.
    """
    dest.mkdir(parents=True, exist_ok=True)
    staging = dest.parent / f".{dest.name}.staging"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    try:
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zf:
                _safe_extract_zip(zf, staging)
        else:
            with tarfile.open(archive) as tf:
                _safe_extract_tar(tf, staging)

        entries = list(staging.iterdir())
        root = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging
        for item in root.iterdir():
            target = dest / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract, refusing entries that escape ``dest``.

    A release archive is remote input; a ``../`` member would let an update
    write anywhere on the filesystem.
    """
    root = dest.resolve()
    for member in zf.namelist():
        if not _is_within(root, (dest / member).resolve()):
            raise UpdateError(f"release archive contains an unsafe path: {member}")
    zf.extractall(dest)


def _is_within(root: Path, target: Path) -> bool:
    """True when *target* is *root* itself or lives inside it.

    A plain ``str.startswith`` compares raw strings, so a sibling directory
    whose name merely begins with the root's name (``.x.staging`` vs
    ``.x.staging2``) passed the check and a crafted archive member could be
    written outside the staging directory.
    """
    try:
        target.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_extract_tar(tf: tarfile.TarFile, dest: Path) -> None:
    root = dest.resolve()
    for member in tf.getmembers():
        if not _is_within(root, (dest / member.name).resolve()):
            raise UpdateError(f"release archive contains an unsafe path: {member.name}")
        if member.issym() or member.islnk():
            raise UpdateError(f"release archive contains a link entry: {member.name}")
    tf.extractall(dest)


def write_release_manifest(dest: Path, version: str) -> Path:
    """Record what a release directory contains, for diagnostics."""
    manifest = dest / "release.json"
    manifest.write_text(
        json.dumps({"version": version, "installed_at": time.time()}, indent=2),
        encoding="utf-8",
    )
    return manifest
