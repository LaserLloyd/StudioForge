"""Owner-only files, on POSIX and on Windows.

``config.yaml`` carries ``server.api_key``, ``mcp.pin`` and ``hf.token``; the
key snapshot, the self-signed certificate's private key and the local CLI key
are credentials too. A file written with the default umask is readable by every
account on the machine, so every credential-bearing file goes through here:

* POSIX: ``0600`` (and ``0700`` for directories that only hold such files).
* Windows: the inherited ACL is removed and a single full-control entry for the
  current user is granted (``icacls``). Windows has no mode bits, so this is the
  honest equivalent of ``chmod 600``.

A stdlib-only leaf: the watchdog (a separate process that imports almost
nothing) writes ``config.yaml`` too, and must be able to import this.

Every function is best-effort in the sense that a filesystem that cannot hold
permissions (FAT, some network shares) must not stop the server from saving its
config; they return ``False`` and log nothing themselves -- callers that care
(``studioforge doctor``) call :func:`is_owner_only` and report.
"""

from __future__ import annotations

import contextlib
import getpass
import os
import subprocess
from pathlib import Path

__all__ = ["is_owner_only", "restrict_to_owner", "write_private"]

#: ``icacls`` ends the run with a few hundred ms of process spawn on a cold
#: Windows box; a stuck one must not hang a config save.
_ICACLS_TIMEOUT_S = 15


def restrict_to_owner(path: Path, *, directory: bool = False) -> bool:
    """Make ``path`` readable and writable by the current user only. ``True`` on success."""
    target = Path(path)
    if os.name == "nt":
        return _restrict_windows(target, directory=directory)
    try:
        target.chmod(0o700 if directory else 0o600)
    except OSError:
        return False
    return True


def _restrict_windows(path: Path, *, directory: bool) -> bool:
    user = _windows_user()
    if not user:
        return False
    grant = f"{user}:(OI)(CI)F" if directory else f"{user}:F"
    try:
        result = subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", grant],
            capture_output=True,
            text=True,
            timeout=_ICACLS_TIMEOUT_S,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _windows_user() -> str:
    domain = os.environ.get("USERDOMAIN", "").strip()
    try:
        name = getpass.getuser()
    except Exception:  # noqa: BLE001 - no usable account name: report failure instead
        return ""
    return f"{domain}\\{name}" if domain and name else name


def write_private(path: Path, data: bytes | str, *, encoding: str = "utf-8") -> None:
    """Write ``data`` to ``path`` so that no other account ever sees it.

    The file is created owner-only *before* the secret goes in (``0600`` at
    ``open(2)`` on POSIX; created empty, ACL-restricted, then filled on
    Windows), so there is no window in which it is world-readable. An existing
    file is truncated in place, keeping its (already restricted) permissions.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = data.encode(encoding) if isinstance(data, str) else data
    if os.name == "nt":
        existed = target.exists()
        if not existed:
            target.touch()
            restrict_to_owner(target)
        with target.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if existed:
            restrict_to_owner(target)
        return
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    with contextlib.suppress(OSError):
        target.chmod(0o600)


def is_owner_only(path: Path) -> bool | None:
    """Whether only the current user can open ``path``; ``None`` if it cannot be told.

    POSIX: no group/other bits set. Windows: ``icacls`` lists no principal other
    than the current user, SYSTEM and the Administrators group (the entries
    Windows keeps for itself even on a locked-down file).
    """
    target = Path(path)
    if not target.exists():
        return None
    if os.name != "nt":
        try:
            return (target.stat().st_mode & 0o077) == 0
        except OSError:
            return None
    try:
        result = subprocess.run(
            ["icacls", str(target)],
            capture_output=True,
            text=True,
            timeout=_ICACLS_TIMEOUT_S,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    user = getpass.getuser().lower()
    allowed = (user, "nt authority\\system", "builtin\\administrators")
    principals: list[str] = []
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text or text.lower().startswith("successfully processed"):
            continue
        # First line is "<path> <principal>:(perms)"; the rest are indented entries.
        entry = text[len(str(target)) :].strip() if text.startswith(str(target)) else text
        principal = entry.split(":(", 1)[0].strip().lower()
        if principal:
            principals.append(principal)
    return all(any(p == a or p.endswith("\\" + a) for a in allowed) for p in principals)
