"""Secrets at rest and TLS: owner-only files, the keyring (with a file fallback), certificates.

Nothing here touches the real OS keyring: the suite pins ``SF_SECRETS_BACKEND=file``
(``conftest.py``) and the keyring cases inject a fake ``keyring`` module.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import socket
import ssl
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from pydantic import ValidationError

from studioforge.config import HF_TOKEN_SECRET, Config, load_config
from studioforge.secret_store import SecretStore
from studioforge.secure_files import is_owner_only, restrict_to_owner, write_private
from studioforge.tlsutil import (
    TlsError,
    certificate_fingerprint,
    describe_tls,
    resolve_tls,
    scheme_for,
)

TOKEN = "hf_" + "a" * 34  # scrub-ok: fixture


# ---------------------------------------------------------------------------
# owner-only files
# ---------------------------------------------------------------------------


def test_write_private_creates_an_owner_only_file(tmp_path: Path) -> None:
    target = tmp_path / "secret.txt"
    write_private(target, "hello")
    assert target.read_text(encoding="utf-8") == "hello"
    assert is_owner_only(target) is True


def test_write_private_tightens_an_existing_loose_file(tmp_path: Path) -> None:
    target = tmp_path / "loose.txt"
    target.write_text("old", encoding="utf-8")
    if os.name != "nt":
        target.chmod(0o666)
        assert is_owner_only(target) is False
    write_private(target, "new")
    assert target.read_text(encoding="utf-8") == "new" and is_owner_only(target) is True


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
def test_write_private_never_writes_into_a_loose_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The secret goes in only after the existing file's mode is already 0600."""
    target = tmp_path / "loose.txt"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o644)
    seen: list[int] = []
    real_fdopen = os.fdopen

    def spying_fdopen(fd: int, *args: object, **kwargs: object) -> object:
        seen.append(os.fstat(fd).st_mode & 0o777)
        return real_fdopen(fd, *args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(os, "fdopen", spying_fdopen)
    write_private(target, "secret")
    assert seen == [0o600]


def test_a_missing_file_is_not_reported_either_way(tmp_path: Path) -> None:
    assert is_owner_only(tmp_path / "nope") is None


def test_restrict_to_owner_on_a_directory(tmp_path: Path) -> None:
    folder = tmp_path / "sub"
    folder.mkdir()
    assert restrict_to_owner(folder, directory=True) is True
    assert is_owner_only(folder) is True


def test_config_yaml_and_its_backup_are_written_owner_only(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path / "data")
    config.server.api_key = "a-secret-api-key"
    path = config.save(tmp_path / "data" / "config.yaml")
    config.save(path)  # the second save writes the .bak
    assert is_owner_only(path) is True
    backup = path.with_suffix(path.suffix + ".bak")
    assert backup.is_file() and is_owner_only(backup) is True
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
def test_config_yaml_mode_is_0600_on_posix(tmp_path: Path) -> None:
    path = Config(data_dir=tmp_path).save(tmp_path / "config.yaml")
    assert (path.stat().st_mode & 0o777) == 0o600


# ---------------------------------------------------------------------------
# the secret store and the keyring
# ---------------------------------------------------------------------------


class FakeKeyring(types.ModuleType):
    """Just enough of the ``keyring`` module: a dict behind get/set/delete_password."""

    class _Backend:
        priority = 5

    def __init__(self) -> None:
        super().__init__("keyring")
        self.store: dict[tuple[str, str], str] = {}
        self.fail_set = False

    def get_keyring(self) -> Any:
        return self._Backend()

    def set_password(self, service: str, name: str, value: str) -> None:
        if self.fail_set:
            raise RuntimeError("keyring is locked")
        self.store[(service, name)] = value

    def get_password(self, service: str, name: str) -> str | None:
        return self.store.get((service, name))

    def delete_password(self, service: str, name: str) -> None:
        self.store.pop((service, name), None)


@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> FakeKeyring:
    module = FakeKeyring()
    monkeypatch.setitem(sys.modules, "keyring", module)
    monkeypatch.setenv("SF_SECRETS_BACKEND", "auto")
    return module


def test_the_file_backend_stores_owner_only_under_secrets(tmp_path: Path) -> None:
    store = SecretStore(tmp_path)
    assert store.backend == "file"
    assert store.set("hf.token", TOKEN) == "file"
    assert store.get("hf.token") == TOKEN
    assert is_owner_only(tmp_path / "secrets" / "hf.token") is True
    store.delete("hf.token")
    assert store.get("hf.token") is None


def test_a_usable_keyring_is_preferred(tmp_path: Path, fake_keyring: FakeKeyring) -> None:
    store = SecretStore(tmp_path)
    assert store.backend == "keyring"
    assert store.set("hf.token", TOKEN) == "keyring"
    assert TOKEN in fake_keyring.store.values()
    assert not (tmp_path / "secrets").exists()
    assert store.get("hf.token") == TOKEN


def test_a_keyring_that_refuses_falls_back_to_the_file_without_losing_the_secret(
    tmp_path: Path, fake_keyring: FakeKeyring
) -> None:
    fake_keyring.fail_set = True
    store = SecretStore(tmp_path)
    assert store.set("owner.owner.hash", "$argon2id$x") == "file"  # scrub-ok: fixture
    assert store.get("owner.owner.hash") == "$argon2id$x"  # scrub-ok: fixture


def test_two_data_directories_do_not_share_keyring_entries(
    tmp_path: Path, fake_keyring: FakeKeyring
) -> None:
    one, two = SecretStore(tmp_path / "a"), SecretStore(tmp_path / "b")
    one.set("hf.token", "first-secret")
    assert two.get("hf.token") is None


def test_env_file_forces_the_file_backend_even_with_a_keyring(
    tmp_path: Path, fake_keyring: FakeKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SF_SECRETS_BACKEND", "file")
    assert SecretStore(tmp_path).backend == "file"


def test_hf_token_moves_to_the_keyring_and_comes_back_on_load(
    tmp_path: Path, fake_keyring: FakeKeyring, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SF_DATA_DIR", raising=False)
    config = Config(data_dir=tmp_path / "data")
    config.hf.token = TOKEN
    path = config.save(tmp_path / "data" / "config.yaml")
    on_disk = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert on_disk["hf"]["token"] is None and TOKEN not in path.read_text(encoding="utf-8")
    assert TOKEN in fake_keyring.store.values()
    loaded = load_config(path)
    assert loaded.hf.token == TOKEN
    # clearing it clears the keyring too, so it does not come back
    loaded.hf.token = None
    loaded.save(path)
    assert TOKEN not in fake_keyring.store.values()
    assert load_config(path).hf.token is None


def test_hf_token_stays_in_the_0600_file_without_a_keyring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SF_DATA_DIR", raising=False)
    config = Config(data_dir=tmp_path / "data")
    config.hf.token = TOKEN
    path = config.save(tmp_path / "data" / "config.yaml")
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["hf"]["token"] == TOKEN
    assert is_owner_only(path) is True
    assert SecretStore(tmp_path / "data").get(HF_TOKEN_SECRET) is None


# ---------------------------------------------------------------------------
# TLS
# ---------------------------------------------------------------------------


def tls_config(tmp_path: Path, mode: str = "self_signed", **extra: Any) -> Config:
    return Config(data_dir=tmp_path / "data", server={"tls": {"mode": mode, **extra}})


def test_off_is_plain_http(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path)
    assert resolve_tls(config) is None and scheme_for(config) == "http"
    assert describe_tls(config) == {"mode": "off", "fingerprint_sha256": None, "not_after": None}


def test_self_signed_is_generated_once_with_a_fingerprint(tmp_path: Path) -> None:
    config = tls_config(tmp_path)
    assert scheme_for(config) == "https"
    assert describe_tls(config)["fingerprint_sha256"] is None  # not generated yet
    material = resolve_tls(config)
    assert material is not None and material.mode == "self_signed"
    assert material.cert_file.is_file() and material.key_file.is_file()
    assert is_owner_only(material.key_file) is True
    fingerprint = material.fingerprint_sha256
    assert len(fingerprint) == 95 and fingerprint.count(":") == 31
    assert fingerprint == fingerprint.upper()
    assert certificate_fingerprint(material.cert_file) == fingerprint
    again = resolve_tls(config)
    assert again is not None and again.fingerprint_sha256 == fingerprint  # reused, not regenerated
    shown = describe_tls(config)
    assert shown["fingerprint_sha256"] == fingerprint and shown["not_after"]
    assert set(material.uvicorn_kwargs()) == {"ssl_certfile", "ssl_keyfile"}


def test_the_self_signed_certificate_covers_loopback(tmp_path: Path) -> None:
    from cryptography import x509

    material = resolve_tls(tls_config(tmp_path))
    assert material is not None
    cert = x509.load_pem_x509_certificate(material.cert_file.read_bytes())
    names = {
        str(n)
        for n in cert.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value.get_values_for_type(x509.DNSName)
    }
    ips = {
        str(i)
        for i in cert.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value.get_values_for_type(x509.IPAddress)
    }
    assert "localhost" in names and "127.0.0.1" in ips


def test_a_nearly_expired_self_signed_certificate_is_replaced(tmp_path: Path) -> None:
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    config = tls_config(tmp_path)
    material = resolve_tls(config)
    assert material is not None
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(x509.oid.NameOID.COMMON_NAME, "old")])
    now = datetime.datetime.now(datetime.UTC)
    old = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(now - datetime.timedelta(days=800))
        .not_valid_after(now + datetime.timedelta(days=3))
        .sign(key, hashes.SHA256())
    )
    material.cert_file.write_bytes(old.public_bytes(serialization.Encoding.PEM))
    renewed = resolve_tls(config)
    assert renewed is not None and renewed.fingerprint_sha256 != certificate_fingerprint(
        old.public_bytes(serialization.Encoding.PEM)
    )


def test_cert_mode_serves_the_given_files(tmp_path: Path) -> None:
    generated = resolve_tls(tls_config(tmp_path))
    assert generated is not None
    config = Config(
        data_dir=tmp_path / "other",
        server={
            "tls": {
                "mode": "cert",
                "cert_file": str(generated.cert_file),
                "key_file": str(generated.key_file),
            }
        },
    )
    material = resolve_tls(config)
    assert material is not None and material.mode == "cert"
    assert material.fingerprint_sha256 == generated.fingerprint_sha256


def test_cert_mode_needs_both_files_and_they_must_exist(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="cert_file"):
        tls_config(tmp_path, "cert")
    config = tls_config(
        tmp_path, "cert", cert_file=str(tmp_path / "no.crt"), key_file=str(tmp_path / "no.key")
    )
    with pytest.raises(TlsError, match="does not exist"):
        resolve_tls(config)


def test_a_bad_certificate_is_a_clear_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.crt"
    bad.write_text("not a certificate", encoding="utf-8")
    with pytest.raises(TlsError):
        certificate_fingerprint(bad)


def test_serve_refuses_a_missing_certificate_with_the_catalogued_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from studioforge.__main__ import app as cli

    monkeypatch.setenv("SF_DATA_DIR", str(tmp_path / "data"))
    config = Config(
        data_dir=tmp_path / "data",
        server={
            "port": 11234,
            "tls": {
                "mode": "cert",
                "cert_file": str(tmp_path / "no.crt"),
                "key_file": str(tmp_path / "no.key"),
            },
        },
        gui={"enabled": False, "port": 18080},
        watchdog={"enabled": False, "port": 11235},
    )
    config.ensure_dirs()
    path = config.save(tmp_path / "data" / "config.yaml")
    result = CliRunner().invoke(cli, ["serve", "--config", str(path), "--no-gui", "--no-watchdog"])
    assert result.exit_code == 2
    assert "TLS is misconfigured" in result.output and "server.tls.mode" in result.output


def test_uvicorn_serves_https_with_the_generated_certificate(tmp_path: Path) -> None:
    """The ssl kwargs really work: HTTPS answers, plain HTTP does not, and the
    certificate on the wire is the one whose fingerprint the panel shows."""
    import uvicorn

    material = resolve_tls(tls_config(tmp_path))
    assert material is not None

    async def app(scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, log_config=None, access_log=False, **material.uvicorn_kwargs())
    )
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started
        reply = httpx.get(f"https://127.0.0.1:{port}/", verify=False)  # noqa: S501
        assert reply.status_code == 200 and reply.text == "ok"
        with pytest.raises(httpx.HTTPError):
            httpx.get(f"http://127.0.0.1:{port}/", timeout=3)
        pem = ssl.get_server_certificate(("127.0.0.1", port))
        assert certificate_fingerprint(pem.encode()) == material.fingerprint_sha256
        der = ssl.PEM_cert_to_DER_cert(pem)
        assert hashlib.sha256(der).hexdigest().upper() == material.fingerprint_sha256.replace(
            ":", ""
        )
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        with contextlib.suppress(OSError):
            sock.close()
