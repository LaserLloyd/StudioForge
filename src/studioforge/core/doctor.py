"""``studioforge doctor``: is this install healthy, safe and reachable?

One pass over everything that commonly goes wrong, each result a
:class:`Finding` with a stable ``code``, a severity, a message and -- always -- a
``fix``. The same report backs the CLI (``studioforge doctor [--json]``), the
API (``GET /api/doctor``), the panel's Diagnostics page and the support bundle.

Checks (read-only: nothing here starts, stops, writes or binds):

* **ports**       -- the three configured ports are ours or free, not someone else's
* **gpu**         -- GPUs and driver as the probe sees them
* **engine**      -- a llama.cpp build is installed and its binary exists
* **disk**        -- free space on the data and model directories
* **permissions** -- the data directory is writable; ``config.yaml`` and the key
  files are owner-only
* **certificate** -- TLS mode, the certificate's expiry and fingerprint
* **connectivity**-- the three services answer on loopback (when they run)
* **migrations**  -- the database is at the version this build expects
* **auth**        -- the security posture: network bind without an admin credential,
  deprecated credentials in use, a wildcard CORS origin, plain HTTP on the network

Severity: ``ok`` and ``info`` are fine; ``warn`` is a risk or a degraded state;
``fail`` is broken or unsafe and makes the CLI exit non-zero.
"""

from __future__ import annotations

import shutil
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from studioforge.authkeys import is_loopback_host, snapshot_path
from studioforge.config import Config
from studioforge.secure_files import is_owner_only
from studioforge.tlsutil import scheme_for

__all__ = ["DoctorReport", "Finding", "run_doctor"]

SEVERITIES = ("ok", "info", "warn", "fail")

#: Free space under which the data / model directory is flagged.
_DISK_WARN_GIB = 20.0
_DISK_FAIL_GIB = 2.0


@dataclass(frozen=True)
class Finding:
    check_id: str
    severity: str
    title: str
    message: str
    fix: str = ""

    def to_dict(self) -> dict[str, str]:
        row = {
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "fix": self.fix,
        }
        row["code"] = self.check_id
        return row


@dataclass
class DoctorReport:
    findings: list[Finding] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

    def add(self, check_id: str, severity: str, title: str, message: str, fix: str = "") -> None:
        self.findings.append(Finding(check_id, severity, title, message, fix))

    @property
    def worst(self) -> str:
        order = {s: i for i, s in enumerate(SEVERITIES)}
        return max((f.severity for f in self.findings), key=lambda s: order[s], default="ok")

    @property
    def ok(self) -> bool:
        return self.worst != "fail"

    def counts(self) -> dict[str, int]:
        return {s: sum(1 for f in self.findings if f.severity == s) for s in SEVERITIES}

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "worst": self.worst,
            "counts": self.counts(),
            "generated_at": self.generated_at,
            "findings": [f.to_dict() for f in self.findings],
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_ports(report: DoctorReport, config: Config, running: dict[str, bool]) -> None:
    from studioforge.core.ports import check_startup_ports

    try:
        conflicts = check_startup_ports(config)
    except Exception as exc:  # noqa: BLE001 - a check that cannot run is a finding, not a crash
        report.add("ports.unchecked", "warn", "Ports", f"Could not check the ports: {exc}")
        return
    ours = {
        "server": running.get("gateway"),
        "gui": running.get("panel"),
        "watchdog": running.get("watchdog"),
    }
    foreign = [c for c in conflicts if not ours.get(c.role)]
    if not foreign:
        report.add(
            "ports.ok",
            "ok",
            "Ports",
            f"{config.server.port}, {config.gui.port} and {config.watchdog.port} are free or "
            "held by this install.",
        )
        return
    for conflict in foreign:
        who = conflict.holder.name or "another program"
        report.add(
            "ports.in_use",
            "fail",
            "Ports",
            f"{conflict.setting} ({conflict.port}) is held by {who}"
            + (" (LM Studio)" if conflict.holder.is_lmstudio else "")
            + ".",
            f"Stop {who}, or set {conflict.setting} to a free port (config.yaml).",
        )


def _check_gpu(report: DoctorReport, probe: Any) -> None:
    if probe is None:
        report.add("gpu.unchecked", "info", "GPU", "No GPU probe available.")
        return
    try:
        gpus = list(probe.list_gpus())
    except Exception as exc:  # noqa: BLE001
        report.add(
            "gpu.probe_failed",
            "warn",
            "GPU",
            f"The GPU probe failed: {exc}",
            "Install or update the GPU driver, then run `studioforge doctor` again.",
        )
        return
    if not gpus:
        report.add(
            "gpu.none",
            "warn",
            "GPU",
            "No GPU was detected, so StudioForge cannot load a model.",
            "Install the GPU driver (NVIDIA: `nvidia-smi` must work), then restart.",
        )
        return
    names = ", ".join(f"{g.name} ({g.total_bytes / 1024**3:.0f} GiB)" for g in gpus)
    report.add("gpu.ok", "ok", "GPU", f"{len(gpus)} GPU(s) via {probe.backend}: {names}.")


def _check_engine(report: DoctorReport, engine_manager: Any) -> None:
    if engine_manager is None:
        report.add("engine.unchecked", "info", "Engine", "No engine manager available.")
        return
    try:
        active = engine_manager.active()
    except Exception as exc:  # noqa: BLE001
        report.add("engine.error", "warn", "Engine", f"The engine inventory failed: {exc}")
        return
    if active is None:
        report.add(
            "engine.none",
            "warn",
            "Engine",
            "No llama.cpp engine is installed.",
            "Run `studioforge setup` or `studioforge engine --update` to install and "
            "smoke-test one.",
        )
        return
    if not Path(active.server_binary).is_file():
        report.add(
            "engine.binary_missing",
            "fail",
            "Engine",
            f"The active engine {active.tag} has no binary at {active.server_binary}.",
            "Reinstall it: `studioforge engine --update`.",
        )
        return
    if not active.smoke_tested:
        report.add(
            "engine.not_smoke_tested",
            "warn",
            "Engine",
            f"The active engine {active.tag} has not passed a smoke test.",
            "Run the smoke test from the panel's Engines page, or reinstall with "
            "`studioforge engine --update`.",
        )
        return
    report.add("engine.ok", "ok", "Engine", f"Engine {active.tag} ({active.variant}) is installed.")


def _free_gib(path: Path) -> float | None:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free / 1024**3
    except OSError:
        return None


def _check_disk(report: DoctorReport, config: Config) -> None:
    targets: list[tuple[str, Path]] = [("data directory", config.data_dir)]
    if config.models.dir is not None:
        targets.append(("model library", Path(config.models.dir)))
    for label, path in targets:
        free = _free_gib(path)
        if free is None:
            report.add("disk.unreadable", "info", "Disk", f"Could not read free space for {label}.")
        elif free < _DISK_FAIL_GIB:
            report.add(
                "disk.full",
                "fail",
                "Disk",
                f"The {label} ({path}) has {free:.1f} GiB free.",
                "Free some space, or move it: models.dir / SF_DATA_DIR.",
            )
        elif free < _DISK_WARN_GIB:
            report.add(
                "disk.low",
                "warn",
                "Disk",
                f"The {label} ({path}) has {free:.1f} GiB free, less than one large model.",
                "Free some space before downloading more models.",
            )
        else:
            report.add("disk.ok", "ok", "Disk", f"The {label} has {free:.0f} GiB free.")


def _check_permissions(report: DoctorReport, config: Config) -> None:
    data_dir = config.data_dir
    if not data_dir.is_dir():
        report.add(
            "permissions.no_data_dir",
            "info",
            "Permissions",
            f"The data directory {data_dir} does not exist yet; it is created on first start.",
        )
        return
    probe = data_dir / ".doctor-write-test"
    try:
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        report.add(
            "permissions.not_writable",
            "fail",
            "Permissions",
            f"The data directory {data_dir} is not writable: {exc}",
            "Run StudioForge as the user that owns this directory, or point SF_DATA_DIR at a "
            "writable one.",
        )
        return
    loose: list[str] = []
    for path in (
        config.config_path,
        config.config_path.with_suffix(config.config_path.suffix + ".bak"),
        snapshot_path(data_dir),
        data_dir / "cli.key",
    ):
        if path.exists() and is_owner_only(path) is False:
            loose.append(path.name)
    if loose:
        report.add(
            "permissions.loose",
            "warn",
            "Permissions",
            f"{', '.join(loose)} can be read by other accounts on this machine.",
            "Save the configuration once (`studioforge config`, or any change in the panel); "
            "files holding credentials are then rewritten owner-only.",
        )
    else:
        report.add(
            "permissions.ok",
            "ok",
            "Permissions",
            "The data directory is writable and credential files are owner-only.",
        )


def _check_certificate(report: DoctorReport, config: Config) -> None:
    mode = config.server.tls.mode
    if mode == "off":
        report.add("tls.off", "info", "Certificate", "TLS is off (plain HTTP).")
        return
    from studioforge.tlsutil import TlsError, certificate_fingerprint

    if mode == "cert":
        cert = Path(config.server.tls.cert_file) if config.server.tls.cert_file else None
    else:
        cert = config.data_dir / "tls" / "selfsigned.crt"
    if cert is None or not cert.is_file():
        if mode == "self_signed":
            report.add(
                "tls.pending",
                "info",
                "Certificate",
                "TLS is self_signed; the certificate is generated when the server first starts.",
            )
        else:
            report.add(
                "tls.cert_missing",
                "fail",
                "Certificate",
                f"server.tls.cert_file {cert} does not exist.",
                "Point server.tls.cert_file and key_file at readable PEM files, or set "
                "server.tls.mode to self_signed.",
            )
        return
    try:
        import datetime

        from cryptography import x509

        pem = cert.read_bytes()
        loaded = x509.load_pem_x509_certificate(pem)
        remaining = loaded.not_valid_after_utc - datetime.datetime.now(datetime.UTC)
        fingerprint = certificate_fingerprint(pem)
    except (TlsError, ValueError, OSError, ImportError) as exc:
        report.add(
            "tls.unreadable",
            "fail",
            "Certificate",
            f"The certificate {cert} could not be read: {exc}",
            "Replace it with a valid PEM certificate.",
        )
        return
    if remaining.days < 0:
        report.add(
            "tls.expired",
            "fail",
            "Certificate",
            f"The certificate expired {-remaining.days} day(s) ago.",
            "Renew it (self_signed regenerates automatically on restart).",
        )
    elif remaining.days < 30:
        report.add(
            "tls.expiring",
            "warn",
            "Certificate",
            f"The certificate expires in {remaining.days} day(s). SHA-256 {fingerprint}.",
            "Renew it (self_signed regenerates automatically on restart).",
        )
    else:
        report.add(
            "tls.ok",
            "ok",
            "Certificate",
            f"TLS ({mode}) certificate valid for {remaining.days} more days. "
            f"SHA-256 {fingerprint}.",
        )


def _loopback(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host


def _check_connectivity(
    report: DoctorReport,
    config: Config,
    fetch: Callable[[str], int | None],
) -> dict[str, bool]:
    scheme = scheme_for(config)
    targets = {
        "gateway": f"{scheme}://{_loopback(config.server.host)}:{config.server.port}/health",
        "panel": f"{scheme}://{_loopback(config.gui.host)}:{config.gui.port}/gui-health",
        "watchdog": f"{scheme}://{_loopback(config.watchdog.host)}:{config.watchdog.port}/health",
    }
    enabled = {
        "gateway": True,
        "panel": config.gui.enabled,
        "watchdog": config.watchdog.enabled,
    }
    running: dict[str, bool] = {}
    for name, url in targets.items():
        if not enabled[name]:
            continue
        status = fetch(url)
        running[name] = status is not None and status < 500
        if running[name]:
            report.add(
                "connectivity.up", "ok", f"Connectivity: {name}", f"{name} answers at {url}."
            )
        elif status is None:
            report.add(
                "connectivity.down",
                "info",
                f"Connectivity: {name}",
                f"Nothing answers at {url} (the server is not running, or this {name} is off).",
                "Start it with `studioforge serve` (or the tray / service).",
            )
        else:
            report.add(
                "connectivity.unhealthy",
                "warn",
                f"Connectivity: {name}",
                f"{url} answered HTTP {status}.",
                "Read the log: `studioforge log stream`.",
            )
    return running


def _check_migrations(report: DoctorReport, config: Config) -> None:
    from studioforge.db import SCHEMA_VERSION

    path = config.db_path
    if not path.is_file():
        report.add(
            "migrations.no_db",
            "info",
            "Migrations",
            "No database yet; it is created and migrated on first start.",
        )
        return
    try:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=2)
        try:
            row = connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        report.add(
            "migrations.unreadable",
            "warn",
            "Migrations",
            f"The database could not be read: {exc}",
            "If the server is running it holds the file; otherwise check the file's permissions. "
            "A corrupt database is moved aside and recreated on start.",
        )
        return
    version = int(row[0]) if row and row[0] is not None else 0
    if version == SCHEMA_VERSION:
        report.add(
            "migrations.ok", "ok", "Migrations", f"The database is at schema {version} (current)."
        )
    elif version < SCHEMA_VERSION:
        report.add(
            "migrations.pending",
            "info",
            "Migrations",
            f"The database is at schema {version}; this build expects {SCHEMA_VERSION}. "
            "It is migrated automatically on the next start.",
            "Preview with `studioforge migrate --dry-run`.",
        )
    else:
        report.add(
            "migrations.newer",
            "fail",
            "Migrations",
            f"The database is at schema {version}, newer than this build ({SCHEMA_VERSION}).",
            "You are running an older version against a newer database. Update StudioForge, "
            "or restore the backup taken before the upgrade.",
        )


def _check_auth(report: DoctorReport, config: Config, accounts: Any) -> None:
    from studioforge.api.auth import has_admin_credential, remote_services

    exposed = remote_services(config)
    has_admin = (
        has_admin_credential(config, accounts)
        if accounts is not None
        else bool(config.server.api_key)
    )
    if exposed and not has_admin:
        hosts = ", ".join(f"{name} on {host}" for name, host in exposed)
        report.add(
            "auth.remote_bind_without_auth",
            "fail",
            "Authentication",
            f"{hosts} would listen on the network, but no owner account or admin API key exists. "
            "The server refuses to start in this state.",
            "Run `studioforge setup` (or `studioforge keys create --scope admin`), or set "
            "server.host, gui.host and watchdog.host to 127.0.0.1.",
        )
    elif exposed:
        report.add(
            "auth.remote_bind",
            "info",
            "Authentication",
            "The server listens on the network and requires an API key or an account.",
        )
        if scheme_for(config) == "http":
            report.add(
                "auth.remote_plain_http",
                "warn",
                "Authentication",
                "The network listeners use plain HTTP, so keys and passwords cross the network "
                "in the clear.",
                "Set server.tls.mode to self_signed (or cert), or front the server with a TLS "
                "reverse proxy / `tailscale serve`.",
            )
    elif all(
        is_loopback_host(h) for h in (config.server.host, config.gui.host, config.watchdog.host)
    ):
        report.add(
            "auth.local_only",
            "ok",
            "Authentication",
            "All three services listen on this machine only.",
        )
    if config.server.api_key:
        report.add(
            "auth.legacy_api_key",
            "warn",
            "Authentication",
            "server.api_key is set. It still works (as the admin key 'legacy') but is deprecated.",
            "Create scoped keys with `studioforge keys create`, move your clients to them, then "
            "revoke 'legacy' and clear server.api_key.",
        )
    if config.mcp.pin:
        report.add(
            "auth.mcp_pin",
            "info",
            "Authentication",
            "mcp.pin is set. It is deprecated and grants admin on the MCP endpoints.",
            "Give agents a scoped key instead (`studioforge keys create --scope admin`), then "
            "clear mcp.pin.",
        )
    if any(origin.strip() == "*" for origin in config.server.cors_origins):
        report.add(
            "auth.cors_wildcard",
            "warn",
            "Authentication",
            "server.cors_origins contains '*': any website can read this API from a visitor's "
            "browser.",
            "Replace it with the exact origins that need it (the default is an empty list).",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _default_fetch(config: Config) -> Callable[[str], int | None]:
    import httpx

    def fetch(url: str) -> int | None:
        try:
            response = httpx.get(url, timeout=2.0, verify=not url.startswith("https://"))
        except Exception:  # noqa: BLE001 - not running is an answer
            return None
        return response.status_code

    return fetch


def run_doctor(
    config: Config,
    *,
    probe: Any = None,
    engine_manager: Any = None,
    accounts: Any = None,
    fetch: Callable[[str], int | None] | None = None,
    skip: frozenset[str] = frozenset(),
) -> DoctorReport:
    """Run every check. ``skip`` names checks to leave out (``ports``, ``gpu``, ...)."""
    report = DoctorReport()
    running = (
        _check_connectivity(report, config, fetch or _default_fetch(config))
        if "connectivity" not in skip
        else {}
    )
    if "ports" not in skip:
        _check_ports(report, config, running)
    if "gpu" not in skip:
        _check_gpu(report, probe)
    if "engine" not in skip:
        _check_engine(report, engine_manager)
    if "disk" not in skip:
        _check_disk(report, config)
    if "permissions" not in skip:
        _check_permissions(report, config)
    if "certificate" not in skip:
        _check_certificate(report, config)
    if "migrations" not in skip:
        _check_migrations(report, config)
    if "auth" not in skip:
        _check_auth(report, config, accounts)
    return report
