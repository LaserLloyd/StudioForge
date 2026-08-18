"""structlog setup with secret redaction.

Secrets (the server API key and the HuggingFace token) pass through config
objects that get logged wholesale in places, so redaction is done as a
processor rather than at each call site -- a missed call site is a leaked
token, a missed processor is impossible.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import structlog

_SECRET_KEYS = {
    "api_key",
    "apikey",
    "token",
    "hf_token",
    "authorization",
    "password",
    "secret",
}
_REDACTED = "***REDACTED***"

# Registered at runtime so log lines that embed a secret in a longer string
# (a launch command line, an error message) still get scrubbed.
_secret_values: set[str] = set()


def register_secret(value: str | None) -> None:
    """Record a secret value so it is scrubbed from all future log output."""
    if value and len(value) >= 6:
        _secret_values.add(value)


def _scrub_text(text: str) -> str:
    # Snapshot: register_secret can run from another thread (config reload)
    # while this iterates, and mutating a set mid-iteration raises.
    for secret in tuple(_secret_values):
        if secret in text:
            text = text.replace(secret, _REDACTED)
    return text


def _redact_value(value: Any) -> Any:
    """Scrub one value, whatever its shape.

    Lists matter as much as dicts here: an argv is a list of strings, and a
    launch command line is exactly the kind of value that carries a token.
    """
    if isinstance(value, str):
        return _scrub_text(value)
    if isinstance(value, dict):
        return _redact_dict(value)
    if isinstance(value, (list, tuple)):
        scrubbed = [_redact_value(item) for item in value]
        return tuple(scrubbed) if isinstance(value, tuple) else scrubbed
    return value


def _redact(
    _logger: Any, _name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    for key, value in list(event_dict.items()):
        if key.lower() in _SECRET_KEYS and value is not None:
            event_dict[key] = _REDACTED
        else:
            event_dict[key] = _redact_value(value)
    return event_dict


def _redact_dict(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key.lower() in _SECRET_KEYS and value is not None:
            out[key] = _REDACTED
        else:
            out[key] = _redact_value(value)
    return out


class RingBufferHandler(logging.Handler):
    """Keeps the last N formatted records in memory for the GUI's Logs tab."""

    def __init__(self, capacity: int = 2000) -> None:
        super().__init__()
        self.capacity = capacity
        self.records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "ts": record.created,
                "level": record.levelname,
                "logger": record.name,
                "message": _scrub_text(record.getMessage()),
            }
        except Exception:  # pragma: no cover - never let logging break the app
            return
        self.records.append(entry)
        if len(self.records) > self.capacity:
            del self.records[: len(self.records) - self.capacity]

    def tail(self, n: int = 200, level: str | None = None) -> list[dict[str, Any]]:
        records = self.records
        if level:
            wanted = level.upper()
            order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if wanted in order:
                threshold = order.index(wanted)
                records = [r for r in records if r["level"] in order[threshold:]]
        return records[-n:]


RING_BUFFER = RingBufferHandler()


def configure_logging(
    level: str = "INFO", *, json_logs: bool = False, log_dir: Path | None = None
) -> None:
    """Configure structlog + stdlib logging. Safe to call more than once."""
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # Redaction runs LAST before rendering, so the exception text that
            # format_exc_info just inserted (an httpx Request repr, a
            # subprocess error carrying an argv) is scrubbed too. Before this
            # ordering, tracebacks bypassed redaction entirely.
            _redact,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.getLevelNamesMapping().get(level.upper(), logging.INFO))

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(stream)
    root.addHandler(RING_BUFFER)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "studioforge.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root.addHandler(file_handler)

    # uvicorn's own loggers duplicate access lines; keep them but quieter.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)
