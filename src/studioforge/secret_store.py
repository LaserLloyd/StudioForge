"""Where small secrets live at rest: the OS keyring, else an owner-only file.

Two things StudioForge stores that are not API keys: the HuggingFace token
(``hf.token``) and the owner account's password hash. Each goes to the OS
keyring (Windows Credential Manager, macOS Keychain, Secret Service) when the
optional ``keyring`` package is installed and has a real backend; otherwise to
``<data-dir>/secrets/<name>``, a file only the current user can read
(:mod:`studioforge.secure_files`). Headless Linux boxes have no keyring, which
is exactly what the file fallback is for.

``keyring`` is a soft import (``pip install studioforge[keyring]``). The choice
is made once per store:

* ``SF_SECRETS_BACKEND=file``     -- never touch a keyring (tests, containers);
* ``SF_SECRETS_BACKEND=keyring``  -- insist on one (fall back to file with a
  warning if there is none);
* unset / ``auto``                -- keyring when it is usable.

A keyring that *raises* at runtime does not lose a secret: ``set`` falls back to
the file, and ``get`` looks in both places.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any

from studioforge.secure_files import restrict_to_owner, write_private

__all__ = ["SecretStore", "keyring_available"]

_log = logging.getLogger("studioforge.secrets")

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _import_keyring() -> Any | None:
    try:
        import keyring  # type: ignore[import-not-found,unused-ignore]

        return keyring
    except Exception:  # noqa: BLE001 - absent, or a broken optional backend
        return None


def keyring_available() -> bool:
    """Whether the ``keyring`` package is installed with a backend that can store secrets."""
    module = _import_keyring()
    if module is None:
        return False
    try:
        backend = module.get_keyring()
    except Exception:  # noqa: BLE001
        return False
    name = f"{type(backend).__module__}.{type(backend).__name__}".lower()
    if "fail" in name or "null" in name:
        return False
    priority = getattr(backend, "priority", 1)
    return not (isinstance(priority, (int, float)) and priority <= 0)


class SecretStore:
    """``get`` / ``set`` / ``delete`` for named secrets, scoped to one data directory."""

    def __init__(self, data_dir: Path, *, backend: str | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.dir = self.data_dir / "secrets"
        wanted = (backend or os.environ.get("SF_SECRETS_BACKEND") or "auto").strip().lower()
        if wanted not in {"auto", "keyring", "file"}:
            wanted = "auto"
        use_keyring = wanted != "file" and keyring_available()
        if wanted == "keyring" and not use_keyring:
            _log.warning("SF_SECRETS_BACKEND=keyring but no usable keyring; using files")
        self.backend = "keyring" if use_keyring else "file"
        # One service name per data directory so two installs never share entries.
        digest = hashlib.sha256(str(self.data_dir.resolve()).encode()).hexdigest()[:10]
        self._service = f"studioforge:{digest}"

    # -- file side ---------------------------------------------------------

    def _file(self, name: str) -> Path:
        return self.dir / _SAFE.sub("_", name)

    def _file_get(self, name: str) -> str | None:
        try:
            text = self._file(name).read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return text or None

    def _file_set(self, name: str, value: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        restrict_to_owner(self.dir, directory=True)
        write_private(self._file(name), value)

    # -- public ------------------------------------------------------------

    def get(self, name: str) -> str | None:
        if self.backend == "keyring":
            module = _import_keyring()
            if module is not None:
                with contextlib.suppress(Exception):
                    value = module.get_password(self._service, name)
                    if value:
                        return str(value)
        return self._file_get(name)

    def set(self, name: str, value: str) -> str:
        """Store ``value``; returns the backend that took it (``keyring`` or ``file``)."""
        if self.backend == "keyring":
            module = _import_keyring()
            if module is not None:
                try:
                    module.set_password(self._service, name, value)
                    # A stale file copy from before the keyring existed must not outlive it.
                    with contextlib.suppress(OSError):
                        self._file(name).unlink()
                    return "keyring"
                except Exception as exc:  # noqa: BLE001 - locked/denied keyring: keep the secret
                    _log.warning("keyring refused %s (%s); storing it in a 0600 file", name, exc)
        self._file_set(name, value)
        return "file"

    def delete(self, name: str) -> None:
        if self.backend == "keyring":
            module = _import_keyring()
            if module is not None:
                with contextlib.suppress(Exception):
                    module.delete_password(self._service, name)
        with contextlib.suppress(OSError):
            self._file(name).unlink()

    def describe(self) -> dict[str, Any]:
        return {"backend": self.backend, "dir": str(self.dir) if self.backend == "file" else None}
