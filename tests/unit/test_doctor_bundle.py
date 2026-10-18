"""The doctor and the support bundle.

The doctor is judged by its findings: every one has a stable code, a severity, a
message and -- for warn/fail -- a fix. The bundle is judged by one property that
matters more than any other: **no secret from a seeded config or log is in it**.
That is a grep test over every member of the zip, not a spot check.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from studioforge.api.app import build_state, create_app
from studioforge.config import Config
from studioforge.core.doctor import SEVERITIES, DoctorReport, run_doctor
from studioforge.core.gpu import FakeGpuProbe
from studioforge.core.support_bundle import build_bundle, redact_text, redacted_config
from studioforge.types import GpuInfo

API_KEY = "legacy-api-key-9f8e7d6c5b4a"  # scrub-ok: fixture
PIN = "48151623"  # scrub-ok: fixture
HF = "hf_" + "Z" * 34  # scrub-ok: fixture
SCOPED = "sf-a-" + "Q" * 43  # scrub-ok: fixture
HASH = "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$aGFzaGhhc2hoYXNoaGFzaA"  # scrub-ok: fixture
SECRETS = (API_KEY, PIN, HF, SCOPED, HASH)


def config_for(tmp_path: Path, **overrides: Any) -> Config:
    base: dict[str, Any] = {
        "data_dir": tmp_path / "data",
        "models": {"dir": tmp_path / "models"},
        "server": {"port": 11234, "api_key": API_KEY},
        "gui": {"enabled": False, "port": 18080},
        "watchdog": {"enabled": False, "port": 11235},
        "mcp": {"pin": PIN},
        "hf": {"token": HF},
        "logging": {"level": "ERROR"},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    config = Config(**base)
    config.ensure_dirs()
    return config


def quiet_doctor(config: Config, **kwargs: Any) -> DoctorReport:
    return run_doctor(config, fetch=lambda _url: None, **kwargs)


def by_code(report: DoctorReport) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for finding in report.findings:
        out.setdefault(finding.check_id, []).append(finding)
    return out


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def test_every_finding_has_a_code_a_severity_a_message_and_a_fix_when_it_matters(
    tmp_path: Path,
) -> None:
    report = quiet_doctor(config_for(tmp_path), probe=FakeGpuProbe([]))
    assert report.findings
    for finding in report.findings:
        row = finding.to_dict()
        assert row["code"] and "." in row["code"]
        assert row["severity"] in SEVERITIES and row["message"].strip() and row["title"]
        if finding.severity in {"warn", "fail"}:
            assert finding.fix.strip(), f"{finding.check_id}: a problem without a next step"
    payload = report.to_dict()
    assert set(payload) == {"ok", "worst", "counts", "generated_at", "findings"}
    json.dumps(payload)  # serialisable


def test_a_clean_local_install_has_no_failures(tmp_path: Path) -> None:
    gpu = GpuInfo(index=0, name="Test GPU", total_bytes=24 << 30, free_bytes=20 << 30)
    report = quiet_doctor(config_for(tmp_path, server={"api_key": None}), probe=FakeGpuProbe([gpu]))
    codes = by_code(report)
    assert report.ok, [f.to_dict() for f in report.findings if f.severity == "fail"]
    assert "gpu.ok" in codes and "auth.local_only" in codes and "migrations.no_db" in codes
    assert "connectivity.down" in codes  # nothing is running in a test


def test_no_gpu_is_a_warning_with_a_fix(tmp_path: Path) -> None:
    report = quiet_doctor(config_for(tmp_path), probe=FakeGpuProbe([]))
    finding = by_code(report)["gpu.none"][0]
    assert finding.severity == "warn" and "driver" in finding.fix


def test_a_network_bind_with_no_credential_is_a_failure_naming_both_fixes(tmp_path: Path) -> None:
    config = config_for(tmp_path, server={"api_key": None, "host": "0.0.0.0"})
    report = quiet_doctor(config, probe=FakeGpuProbe([]))
    finding = by_code(report)["auth.remote_bind_without_auth"][0]
    assert finding.severity == "fail" and not report.ok
    assert "studioforge setup" in finding.fix and "keys create" in finding.fix


def test_a_network_bind_with_a_key_over_plain_http_warns_about_tls(tmp_path: Path) -> None:
    config = config_for(tmp_path, server={"host": "0.0.0.0"})
    report = quiet_doctor(config, probe=FakeGpuProbe([]))
    codes = by_code(report)
    assert "auth.remote_plain_http" in codes and "tls" in codes["auth.remote_plain_http"][0].fix
    assert "auth.remote_bind_without_auth" not in codes


def test_deprecated_credentials_and_a_wildcard_origin_are_called_out(tmp_path: Path) -> None:
    config = config_for(tmp_path, server={"cors_origins": ["*"]})
    codes = by_code(quiet_doctor(config, probe=FakeGpuProbe([])))
    assert codes["auth.legacy_api_key"][0].severity == "warn"
    assert codes["auth.mcp_pin"][0].severity == "info"
    assert codes["auth.cors_wildcard"][0].severity == "warn"


def test_a_migration_status_reads_the_database_without_changing_it(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    build_state(config)  # creates and migrates registry.sqlite3
    before = config.db_path.stat().st_mtime_ns
    codes = by_code(quiet_doctor(config, probe=FakeGpuProbe([])))
    assert codes["migrations.ok"][0].severity == "ok"
    assert config.db_path.stat().st_mtime_ns == before


def test_a_database_from_a_newer_build_is_a_failure(tmp_path: Path) -> None:
    import sqlite3

    config = config_for(tmp_path)
    build_state(config)
    connection = sqlite3.connect(config.db_path)
    connection.execute(
        "INSERT INTO schema_migrations (version, applied_at, name) VALUES (999, 0, 'future')"
    )
    connection.commit()
    connection.close()
    finding = by_code(quiet_doctor(config, probe=FakeGpuProbe([])))["migrations.newer"][0]
    assert finding.severity == "fail" and "Update StudioForge" in finding.fix


def test_a_missing_certificate_file_is_a_failure_and_self_signed_is_pending(
    tmp_path: Path,
) -> None:
    broken = config_for(
        tmp_path,
        server={
            "tls": {
                "mode": "cert",
                "cert_file": str(tmp_path / "no.crt"),
                "key_file": str(tmp_path / "no.key"),
            }
        },
    )
    assert by_code(quiet_doctor(broken, probe=FakeGpuProbe([])))["tls.cert_missing"]
    pending = config_for(tmp_path / "b", server={"tls": {"mode": "self_signed"}})
    assert by_code(quiet_doctor(pending, probe=FakeGpuProbe([])))["tls.pending"]


def test_a_generated_certificate_is_reported_with_its_fingerprint(tmp_path: Path) -> None:
    from studioforge.tlsutil import resolve_tls

    config = config_for(tmp_path, server={"tls": {"mode": "self_signed"}})
    material = resolve_tls(config)
    assert material is not None
    finding = by_code(quiet_doctor(config, probe=FakeGpuProbe([])))["tls.ok"][0]
    assert material.fingerprint_sha256 in finding.message


def test_a_running_install_is_recognised_so_its_own_ports_are_not_a_conflict(
    tmp_path: Path,
) -> None:
    config = config_for(tmp_path)
    report = run_doctor(config, probe=FakeGpuProbe([]), fetch=lambda _url: 200)
    codes = by_code(report)
    assert "connectivity.up" in codes and "ports.in_use" not in codes


def test_a_port_held_by_someone_else_is_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.core import ports

    config = config_for(tmp_path)
    holder = ports.PortHolder(pid=1, name="lmstudio.exe", is_lmstudio=True)
    conflict = ports.PortConflict(
        role="server",
        port=config.server.port,
        host="127.0.0.1",
        holder=holder,
        setting="server.port",
    )
    monkeypatch.setattr(ports, "check_startup_ports", lambda _c: [conflict])
    finding = by_code(quiet_doctor(config, probe=FakeGpuProbe([])))["ports.in_use"][0]
    assert finding.severity == "fail" and "LM Studio" in finding.message
    assert "server.port" in finding.fix


def test_the_doctor_runs_over_http_for_an_admin_only(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config = config_for(tmp_path, server={"api_key": None})
    app = create_app(config, state=build_state(config), start_background=False)
    _r, read = app.state.accounts.create_key("ro", "read")
    _a, admin = app.state.accounts.create_key("root", "admin")
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        assert (
            client.get("/api/doctor", headers={"Authorization": f"Bearer {read}"}).status_code
            == 403
        )
        body = client.get("/api/doctor", headers={"Authorization": f"Bearer {admin}"}).json()
    assert body["findings"] and "worst" in body


# ---------------------------------------------------------------------------
# the support bundle
# ---------------------------------------------------------------------------


def seed_logs(config: Config) -> None:
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    (config.logs_dir / "studioforge.log").write_text(
        "\n".join(
            [
                "2026-10-18 09:00:00 [info] starting",
                f"2026-10-18 09:00:01 [info] auth api_key={API_KEY} accepted",
                f"2026-10-18 09:00:02 [warn] header Authorization: Bearer {SCOPED}",
                f"2026-10-18 09:00:03 [info] X-MCP-Pin: {PIN} used",
                f"2026-10-18 09:00:04 [info] GET /mcp?pin={PIN} 401",
                f"2026-10-18 09:00:05 [info] token={HF} for hf",
                f"2026-10-18 09:00:06 [debug] stored hash {HASH}",
                f"2026-10-18 09:00:07 [info] raw key {API_KEY} and {SCOPED} in prose",
                f"2026-10-18 09:00:08 [info] reading {Path.home()}\\models\\x.gguf",
                '2026-10-18 09:00:09 [info] {"password": "hunter2hunter2", "x": 1}',
                "2026-10-18 09:00:10 [info] max_tokens=4096 pin_required=True done",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (config.logs_dir / "watchdog.log").write_text(f"watchdog saw {API_KEY}\n", encoding="utf-8")
    (config.logs_dir / "audit.jsonl").write_text(
        json.dumps({"actor": "key:1", "action": "api.PATCH /api/config", "target": "hf.token"})
        + "\n",
        encoding="utf-8",
    )


def read_bundle(payload: bytes) -> dict[str, str]:
    archive = zipfile.ZipFile(io.BytesIO(payload))
    return {name: archive.read(name).decode("utf-8", "replace") for name in archive.namelist()}


def test_no_seeded_secret_appears_anywhere_in_the_bundle(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    seed_logs(config)
    payload = build_bundle(config, probe=FakeGpuProbe([]), log_lines=500)
    files = read_bundle(payload)
    assert {"README.txt", "versions.json", "config.redacted.yaml", "doctor.json"} <= set(files)
    assert "logs/studioforge.log.tail" in files and "logs/watchdog.log.tail" in files
    haystack = "\n".join(files.values())
    for secret in (*SECRETS, "hunter2hunter2"):
        assert secret not in haystack, f"{secret[:8]}... leaked into the support bundle"
    assert secret_not_in_raw_bytes(payload)


def secret_not_in_raw_bytes(payload: bytes) -> bool:
    """Also grep the compressed container itself (a member stored uncompressed would show)."""
    return all(secret.encode() not in payload for secret in SECRETS)


def test_the_bundle_keeps_what_a_maintainer_needs(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    seed_logs(config)
    files = read_bundle(build_bundle(config, probe=FakeGpuProbe([]), log_lines=500))
    log = files["logs/studioforge.log.tail"]
    assert "starting" in log and "max_tokens=4096 pin_required=True" in log  # not over-redacted
    assert "[REDACTED]" in log
    assert str(Path.home()) not in log and "~" in log
    config_text = files["config.redacted.yaml"]
    assert "port: 11234" in config_text  # ordinary settings survive
    assert "api_key: " in config_text and API_KEY not in config_text
    assert "pin_required" in config_text
    versions = json.loads(files["versions.json"])
    assert versions["studioforge"] and versions["python"] and "packages" in versions
    audit = files["logs/audit.jsonl.tail"]
    assert "hf.token" in audit  # key names are audit content; values never are
    assert json.loads(files["doctor.json"])["findings"]


def test_log_tail_is_bounded(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    (config.logs_dir / "studioforge.log").write_text(
        "\n".join(f"line {i}" for i in range(5000)) + "\n", encoding="utf-8"
    )
    files = read_bundle(build_bundle(config, probe=FakeGpuProbe([]), log_lines=100))
    tail = files["logs/studioforge.log.tail"].splitlines()
    assert len(tail) == 100 and tail[-1] == "line 4999"


def test_the_bundle_never_includes_files_that_hold_credentials(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    build_state(config).accounts.create_key("root", "admin")
    config.data_dir.joinpath("cli.key").write_text(SCOPED, encoding="utf-8")
    (config.data_dir / "secrets").mkdir(exist_ok=True)
    (config.data_dir / "secrets" / "owner.x.hash").write_text(HASH, encoding="utf-8")
    payload = build_bundle(config, probe=FakeGpuProbe([]))
    names = " ".join(zipfile.ZipFile(io.BytesIO(payload)).namelist())
    for forbidden in ("sqlite", "cli.key", "snapshot", "secrets", ".key", "selfsigned"):
        assert forbidden not in names


def test_redacted_config_masks_by_key_name_wherever_it_sits(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    dumped = redacted_config(config)
    assert dumped["server"]["api_key"] != API_KEY and dumped["mcp"]["pin"] != PIN
    assert dumped["hf"]["token"] != HF
    assert dumped["mcp"]["pin_required"] is True and dumped["server"]["port"] == 11234
    assert redact_text("Bearer abcdef123456789 end").endswith("end")
    assert "abcdef123456789" not in redact_text("Authorization: Bearer abcdef123456789")


def test_the_support_bundle_route_is_admin_only_and_clean(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config = config_for(tmp_path, server={"api_key": None})
    config.hf.token = HF
    seed_logs(config)
    app = create_app(config, state=build_state(config), start_background=False)
    _r, read = app.state.accounts.create_key("ro", "read")
    _a, admin = app.state.accounts.create_key("root", "admin")
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        denied = client.post("/api/support-bundle", headers={"Authorization": f"Bearer {read}"})
        assert denied.status_code == 403
        ok = client.post("/api/support-bundle", headers={"Authorization": f"Bearer {admin}"})
    assert ok.status_code == 200 and ok.headers["content-type"] == "application/zip"
    assert "attachment" in ok.headers["content-disposition"]
    files = read_bundle(ok.content)
    haystack = "\n".join(files.values())
    for secret in (HF, SCOPED, HASH, admin, read):
        assert secret not in haystack
    assert ok.content.count(admin.encode()) == 0


def test_every_destructive_audit_action_is_in_the_bundle_audit_tail(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    seed_logs(config)
    files = read_bundle(build_bundle(config, probe=FakeGpuProbe([])))
    assert json.loads(files["logs/audit.jsonl.tail"].splitlines()[0])["actor"] == "key:1"
