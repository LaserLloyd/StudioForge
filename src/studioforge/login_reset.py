"""Lockout recovery for someone who can reach the data directory but not the panel.

Drop a file named ``RESET_LOGIN`` into the data directory and start the server
(``studioforge serve``, the tray, ``docker compose up``): at that start, and only
then, the sentinel is read, consumed and applied (D81).

    RESET_LOGIN            empty   -> the owner gets a generated password, written to
                                      ``RESET_LOGIN.password`` (owner-only, never logged)
    RESET_LOGIN            alice   -> same, for the account ``alice`` (created if missing)
    RESET_LOGIN            alice:a-long-new-password
                                   -> that password (everything after the first colon)

**Why a file and not a flag or a route.** Whoever can create a file in the data
directory can already read the database, the key snapshot and the CLI key, so the
sentinel widens nothing. Nothing in ``api/``, ``gui/``, ``mcp/`` or ``watchdog/``
imports this module and nothing watches for the file while the server runs: it is
impossible to trigger over the network, which a "reset" endpoint could never promise.

**Why it cannot replay.** The file is deleted *before* anything is changed. If it
cannot be deleted (a read-only volume, an editor holding it open on Windows) the
reset is refused rather than applied, so a sentinel that survives a start never
resets anything twice. A rejected reset (password under ten characters) still
consumes the file; the log line says why and the file is simply written again.

The password never reaches a log line, a console line or an exception message.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from studioforge.logging import get_logger
from studioforge.secure_files import write_private

if TYPE_CHECKING:
    from studioforge.core.accounts import AccountStore

log = get_logger(__name__)

__all__ = [
    "PASSWORD_FILE_NAME",
    "SENTINEL_NAME",
    "LoginReset",
    "apply_sentinel",
    "sentinel_path",
]

#: The sentinel's name inside the data directory. Upper case on purpose: it is a
#: file a person creates by hand, and it should be unmistakable in a listing.
SENTINEL_NAME: Final = "RESET_LOGIN"
#: Where a generated password is written (owner-only). Read it, sign in, delete it.
PASSWORD_FILE_NAME: Final = "RESET_LOGIN.password"
#: A sentinel is a name and a password. Anything bigger is not one.
_MAX_BYTES: Final = 4096


@dataclass(frozen=True)
class LoginReset:
    """What one start did about a sentinel. Never carries a password."""

    username: str | None
    #: ``True`` when the account did not exist and was created.
    created: bool
    #: Set when the password was generated; the file holding it.
    password_file: Path | None
    #: Set when nothing was changed; why.
    error: str | None = None

    @property
    def applied(self) -> bool:
        return self.error is None

    def describe(self) -> str:
        """One line for the console and the log. No secret."""
        if self.error:
            return f"RESET_LOGIN was not applied: {self.error}"
        verb = "created" if self.created else "reset the password of"
        line = f"RESET_LOGIN consumed: {verb} the account '{self.username}'"
        if self.password_file is not None:
            return f"{line}; the generated password is in {self.password_file} (owner-only)"
        return line


def sentinel_path(data_dir: Path | str) -> Path:
    return Path(data_dir) / SENTINEL_NAME


def apply_sentinel(data_dir: Path | str, store: AccountStore) -> LoginReset | None:
    """Consume ``<data_dir>/RESET_LOGIN`` and apply it; ``None`` when there is none.

    Called once, from ``studioforge serve`` before the listeners are created.
    """
    path = sentinel_path(data_dir)
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        return _done(LoginReset(None, False, None, f"{path} is not a regular file; left alone"))
    try:
        raw = path.read_bytes()[: _MAX_BYTES + 1]
    except OSError as exc:
        return _done(LoginReset(None, False, None, f"could not read {path}: {exc.strerror or exc}"))
    # Consume FIRST. A sentinel that cannot be removed is never applied: the one
    # thing it must not do is run again at the next start.
    try:
        path.unlink()
    except OSError as exc:
        return _done(
            LoginReset(
                None,
                False,
                None,
                f"could not delete {path} ({exc.strerror or exc}), so it was not applied; "
                "make the data directory writable and start again",
            )
        )
    if len(raw) > _MAX_BYTES:
        return _done(
            LoginReset(None, False, None, f"{path.name} is larger than {_MAX_BYTES} bytes")
        )

    username, password = parse(raw)
    name = username or store.owner_username() or "owner"
    generated = not password
    if generated:
        password = secrets.token_urlsafe(18)
    try:
        account, created = store.reset_login(name, password)
    except Exception as exc:  # noqa: BLE001 - one readable line; the file is already consumed
        return _done(LoginReset(name, False, None, str(getattr(exc, "message", exc))))
    password_file: Path | None = None
    if generated:
        password_file = Path(data_dir) / PASSWORD_FILE_NAME
        try:
            write_private(password_file, password + "\n")
        except OSError as exc:
            return _done(
                LoginReset(
                    name,
                    created,
                    None,
                    f"reset the account '{account.username}' but could not write "
                    f"{password_file}: {exc.strerror or exc}; run `studioforge users passwd "
                    f"{account.username}` to set a password",
                )
            )
    return _done(LoginReset(account.username, created, password_file))


def parse(raw: bytes) -> tuple[str, str]:
    """``(username, password)`` from the sentinel's bytes; either may be empty.

    Surrounding whitespace is dropped (``echo name:pw > file`` in ``cmd.exe`` leaves a
    trailing space and Notepad a newline), so a password cannot start or end with a space
    here. Only the first line counts. The split is at the first colon: usernames cannot
    contain one, passwords can.
    """
    text = raw.decode("utf-8-sig", "replace").strip()
    line = text.splitlines()[0].strip() if text else ""
    name, _sep, password = line.partition(":")
    return name.strip(), password


def _done(result: LoginReset) -> LoginReset:
    if result.applied:
        log.warning(result.describe())
    else:
        log.error(result.describe())
    return result
