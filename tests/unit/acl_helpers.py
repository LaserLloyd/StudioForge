"""Owner-only assertions that say why they failed (ACL listing on Windows, mode on POSIX)."""

from __future__ import annotations

import getpass
import os
import subprocess
from pathlib import Path

from studioforge.secure_files import is_owner_only


def describe_access(path: Path) -> str:
    """What the filesystem says about who can open ``path``."""
    if os.name != "nt":
        try:
            return f"mode {path.stat().st_mode & 0o777:o}"
        except OSError as exc:
            return f"stat failed: {exc}"
    try:
        listing = subprocess.run(
            ["icacls", str(path)], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"icacls failed: {exc}"
    return (
        f"user={getpass.getuser()!r} USERDOMAIN={os.environ.get('USERDOMAIN')!r}\n"
        f"icacls rc={listing.returncode}\n{listing.stdout}{listing.stderr}"
    )


def assert_owner_only(path: Path) -> None:
    assert is_owner_only(path) is True, f"{path} is not owner-only: {describe_access(path)}"
