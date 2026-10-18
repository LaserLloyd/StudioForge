"""The audit log: an append-only JSONL record of every admin mutation.

``<data_dir>/logs/audit.jsonl``, one JSON object per line::

    {"ts": "2026-10-18T09:30:00.123Z", "actor": "key:3f9a...", "action": "config.set",
     "target": "models.default_ctx", "result": "ok", "surface": "api", "peer": "192.168.1.20"}

* ``actor``     -- who: ``key:<id>``, ``owner:<username>``, ``legacy``, ``pin``,
  ``local`` (no credential, this machine) or ``anonymous``.
* ``action``    -- what, as ``<area>.<verb>`` (``mcp.delete_model``, ``api.DELETE /api/models/x``).
* ``target``    -- the thing acted on; never a secret value (a config change lists
  the *keys* touched, and a secret key's value is not recorded at all).
* ``result``    -- ``ok``, ``denied`` or ``error``.

Append-only by construction: the file is only ever opened in append mode, a line
is written and flushed in one call, and rotation renames the whole file (to
``audit.<stamp>.jsonl``) rather than editing it. A stdlib-only leaf, because the
watchdog -- a separate process -- writes to the same file for the destructive
tools it carries; both open the file per line, so neither pins it against the
other's rotation.

An audit failure must never take the server down or block a request, so
:meth:`AuditLog.record` swallows I/O errors (and says so once on the logger).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from studioforge.logfiles import MB, backups_of, rotate_if_large
from studioforge.secure_files import restrict_to_owner

__all__ = ["AuditLog", "audit_path", "redact_update_keys"]

_log = logging.getLogger("studioforge.audit")

AUDIT_FILE = "audit.jsonl"

#: Dotted config paths whose *value* must never appear anywhere. Mirrors
#: ``config.SECRET_CONFIG_PATHS`` (kept literal here so this module stays a leaf).
_SECRET_KEYS = frozenset({"server.api_key", "mcp.pin", "hf.token"})


def audit_path(logs_dir: Path) -> Path:
    return Path(logs_dir) / AUDIT_FILE


def redact_update_keys(updates: dict[str, Any]) -> str:
    """``"models.default_ctx, server.api_key"`` -- the keys a config change touched, no values."""
    return ", ".join(sorted(str(k) for k in updates))


class AuditLog:
    """Append one JSON line per event. Thread-safe; safe to share between processes."""

    def __init__(
        self, logs_dir: Path, *, max_mb: int = 10, backups: int = 5, check_every: int = 50
    ) -> None:
        self.path = audit_path(logs_dir)
        self._max_bytes = max(0, int(max_mb)) * MB
        self._backups = max(1, int(backups))
        self._check_every = max(1, check_every)
        self._lock = threading.Lock()
        self._since_check = 0
        self._warned = False

    def record(
        self,
        *,
        actor: str,
        action: str,
        target: str = "",
        result: str = "ok",
        surface: str = "api",
        peer: str | None = None,
        detail: str | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
            + f".{int((time.time() % 1) * 1000):03d}Z",
            "actor": actor,
            "action": action,
            "target": target[:500],
            "result": result,
            "surface": surface,
        }
        if peer:
            entry["peer"] = peer
        if detail:
            entry["detail"] = detail[:500]
        line = json.dumps(entry, separators=(",", ":"), ensure_ascii=True) + "\n"
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                fresh = not self.path.exists()
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.flush()
                if fresh:
                    restrict_to_owner(self.path)
                self._since_check += 1
                if self._max_bytes and self._since_check >= self._check_every:
                    self._since_check = 0
                    rotate_if_large(
                        self.path, max_bytes=self._max_bytes, backup_count=self._backups
                    )
        except OSError as exc:
            if not self._warned:
                self._warned = True
                _log.warning("could not write the audit log %s: %s", self.path, exc)

    def tail(self, n: int = 100) -> list[dict[str, Any]]:
        """The last ``n`` entries, oldest first, reaching into rotated files when needed.

        Unreadable lines are skipped. Reading across a rotation matters: the file is
        renamed away the moment it is large, so a tail of only the live file would
        read empty right after the entries that most deserve a look.
        """
        wanted = max(1, n)
        lines: list[str] = []
        for path in [*reversed(backups_of(self.path)), self.path]:
            try:
                lines = path.read_text(encoding="utf-8").splitlines() + lines
            except OSError:
                continue
            if len(lines) >= wanted:
                break
        out: list[dict[str, Any]] = []
        for line in lines[-wanted:]:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out
