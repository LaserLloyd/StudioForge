"""``studioforge support-bundle``: everything a maintainer needs, nothing that is a secret.

A zip with the redacted configuration, the doctor report, versions, the GPU list
and the tail of the logs and the audit log. Built for attaching to a bug report
or a forum post, so the one rule is: **no credential leaves**. Redaction is
layered, because any single layer can miss something:

1. *Structure.* The configuration is dumped through ``redact_config_dict`` (the
   shared list of secret keys) and then every value whose key *name* looks like a
   credential (``key``, ``token``, ``pin``, ``password``, ``secret``, ``hash``) is
   masked, wherever it sits.
2. *Literals.* Every secret value known to the running config -- ``server.api_key``,
   ``mcp.pin``, ``hf.token`` -- is removed from all text by exact match.
3. *Shapes.* Patterns that look like a credential whatever their source: StudioForge
   keys (``sf-r|i|a-...``), ``Bearer`` tokens, ``X-MCP-Pin`` headers, ``api_key=``
   / ``token=`` / ``password=`` assignments, argon2 hashes, HuggingFace tokens.
4. *Paths.* The user's home directory becomes ``~``.

Also never included: the database, the key snapshot, the CLI key, ``secrets/``,
the TLS private key, models or any prompt/response content.
"""

from __future__ import annotations

import io
import json
import platform
import re
import sys
import time
import zipfile
from collections import deque
from pathlib import Path
from typing import Any

import yaml

from studioforge import __version__
from studioforge.config import Config, redact_config_dict

__all__ = ["build_bundle", "redact_text", "secret_values"]

#: Config keys whose VALUE is masked whatever section they are in.
_SECRET_NAME = re.compile(r"(key|token|pin|password|secret|hash|credential)", re.IGNORECASE)

#: Shapes of credentials, independent of where they came from.
_SHAPES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sf-[ria]-[0-9A-Za-z]{20,}"), "sf-?-[REDACTED]"),
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{8,}"), r"\1 [REDACTED]"),
    (
        re.compile(r"(?i)(x-(?:mcp|studioforge)-pin['\"]?\s*[:=]\s*['\"]?)[^\s'\",;]+"),
        r"\1[REDACTED]",
    ),
    (re.compile(r"(?i)(x-api-key['\"]?\s*[:=]\s*['\"]?)[^\s'\",;]+"), r"\1[REDACTED]"),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|token|password|passwd|secret|pin)(['\"]?\s*[:=]\s*['\"]?)"
            r"(?!\[REDACTED\]|\*\*\*|null\b|none\b)[^\s'\",;&}]{3,}"
        ),
        r"\1\2[REDACTED]",
    ),
    (re.compile(r"\$argon2[a-z0-9]*\$[^\s'\"]+"), "[REDACTED-HASH]"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), "hf_[REDACTED]"),
    (re.compile(r"[?&](pin|api_key|token|key)=[^&\s]+", re.IGNORECASE), r"?\1=[REDACTED]"),
)

#: Log lines kept per file, and audit lines.
DEFAULT_LOG_LINES = 2000
AUDIT_LINES = 500


def secret_values(config: Config) -> list[str]:
    """Every credential value the config holds (>= 4 chars), longest first."""
    found = [config.server.api_key, config.mcp.pin, config.hf.token]
    values = {str(v) for v in found if v and len(str(v)) >= 4}
    return sorted(values, key=len, reverse=True)


def redact_text(text: str, config: Config | None = None, *, extra: list[str] | None = None) -> str:
    """``text`` with literal secrets, credential-shaped strings and the home path removed."""
    literals = list(extra or [])
    if config is not None:
        literals += secret_values(config)
    for value in sorted(set(literals), key=len, reverse=True):
        text = text.replace(value, "[REDACTED]")
    for pattern, replacement in _SHAPES:
        text = pattern.sub(replacement, text)
    home = str(Path.home())
    if home and len(home) > 3:
        text = text.replace(home, "~").replace(home.replace("\\", "/"), "~")
    return text


def _mask_tree(node: Any, key: str = "") -> Any:
    if isinstance(node, dict):
        return {k: _mask_tree(v, str(k)) for k, v in node.items()}
    if isinstance(node, list):
        return [_mask_tree(v, key) for v in node]
    if node and isinstance(node, str) and _SECRET_NAME.search(key):
        return "[REDACTED]"
    return node


def redacted_config(config: Config) -> dict[str, Any]:
    dumped = config.to_yaml_dict()
    redact_config_dict(dumped)
    # `pin_required` and the like are flags, not secrets: only strings are masked.
    return _mask_tree(dumped)


def _tail(path: Path, lines: int) -> str | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return "".join(deque(handle, maxlen=lines))
    except OSError:
        return None


def _versions() -> dict[str, Any]:
    from studioforge.build import build_id

    info: dict[str, Any] = {
        "studioforge": __version__,
        "build": build_id(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    from importlib import metadata

    packages = {}
    for name in ("fastapi", "uvicorn", "httpx", "pydantic", "mcp", "nicegui", "psutil"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    info["packages"] = packages
    return info


def _gpus(probe: Any) -> list[dict[str, Any]]:
    if probe is None:
        return []
    try:
        return [
            {
                "index": g.index,
                "name": g.name,
                "total_bytes": g.total_bytes,
                "free_bytes": g.free_bytes,
                "compute_capability": getattr(g, "compute_capability", None),
            }
            for g in probe.list_gpus()
        ]
    except Exception as exc:  # noqa: BLE001
        return [{"error": str(exc)}]


def build_bundle(
    config: Config,
    *,
    doctor: dict[str, Any] | None = None,
    probe: Any = None,
    log_lines: int = DEFAULT_LOG_LINES,
) -> bytes:
    """The bundle as zip bytes. ``doctor`` is a ``DoctorReport.to_dict()`` (run one if None)."""
    if doctor is None:
        from studioforge.core.doctor import run_doctor

        doctor = run_doctor(config, probe=probe, skip=frozenset({"connectivity"})).to_dict()

    def clean(text: str) -> str:
        return redact_text(text, config)

    files: dict[str, str] = {
        "README.txt": (
            "StudioForge support bundle\n"
            f"created: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n\n"
            "Contents: versions.json, config.redacted.yaml, doctor.json, gpus.json, and the\n"
            "tail of the server, watchdog and audit logs.\n"
            "Credentials (API keys, the MCP PIN, tokens, password hashes) are removed, the home\n"
            "directory is shown as ~, and no database, key file, certificate key, model or\n"
            "prompt/response content is included.\n"
        ),
        "versions.json": json.dumps(_versions(), indent=2),
        "config.redacted.yaml": clean(
            yaml.safe_dump(redacted_config(config), sort_keys=False, default_flow_style=False)
        ),
        "doctor.json": clean(json.dumps(doctor, indent=2)),
        "gpus.json": clean(json.dumps(_gpus(probe), indent=2)),
    }
    for name in ("studioforge.log", "watchdog.log", "audit.jsonl"):
        limit = AUDIT_LINES if name == "audit.jsonl" else log_lines
        text = _tail(config.logs_dir / name, limit)
        if text is not None:
            files[f"logs/{name}.tail"] = clean(text)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return buffer.getvalue()
