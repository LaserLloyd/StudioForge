"""Self-signed TLS: limited SANs, no regeneration on address rotation, locked atomic pair."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from cryptography import x509

from studioforge import tlsutil
from studioforge.config import Config
from studioforge.tlsutil import resolve_tls


def _config(tmp_path: Path, host: str = "127.0.0.1") -> Config:
    return Config(
        data_dir=tmp_path / "data",
        server={"host": host, "tls": {"mode": "self_signed"}},
    )


def _sans(path: Path) -> set[str]:
    return tlsutil._san_set(x509.load_pem_x509_certificate(path.read_bytes()))


def test_sans_are_loopback_only_for_default_bind(tmp_path: Path) -> None:
    material = resolve_tls(_config(tmp_path))
    assert material is not None
    assert _sans(material.cert_file) == {"localhost", "127.0.0.1", "::1"}


def test_wildcard_bind_adds_no_machine_addresses(tmp_path: Path) -> None:
    config = Config(
        data_dir=tmp_path / "data",
        server={"host": "0.0.0.0", "tls": {"mode": "self_signed"}},  # noqa: S104
    )
    material = resolve_tls(config)
    assert material is not None
    assert _sans(material.cert_file) == {"localhost", "127.0.0.1", "::1"}


def test_configured_bind_host_is_covered_and_triggers_one_regeneration(tmp_path: Path) -> None:
    first = resolve_tls(_config(tmp_path))
    again = resolve_tls(_config(tmp_path))
    assert first and again and first.fingerprint_sha256 == again.fingerprint_sha256
    wider = resolve_tls(_config(tmp_path, host="192.0.2.7"))
    assert wider is not None
    assert "192.0.2.7" in _sans(wider.cert_file)
    assert wider.fingerprint_sha256 != first.fingerprint_sha256


def test_machine_address_rotation_does_not_regenerate(tmp_path: Path, monkeypatch) -> None:
    first = resolve_tls(_config(tmp_path))
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [(0, 0, 0, "", ("2001:db8::99", 0))])
    monkeypatch.setattr("socket.gethostname", lambda: "rotated-host")
    second = resolve_tls(_config(tmp_path))
    assert first and second and first.fingerprint_sha256 == second.fingerprint_sha256


def test_mismatched_pair_is_regenerated(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = resolve_tls(config)
    assert first is not None
    other = _config(tmp_path / "other")
    donor = resolve_tls(other)
    assert donor is not None
    first.key_file.write_bytes(donor.key_file.read_bytes())  # key no longer matches cert
    healed = resolve_tls(config)
    assert healed is not None
    assert healed.fingerprint_sha256 != first.fingerprint_sha256
    assert tlsutil._self_signed_is_current(
        healed.cert_file, healed.key_file, *tlsutil._san_names(config)
    )


def test_concurrent_starts_agree_on_one_pair(tmp_path: Path) -> None:
    config = _config(tmp_path)
    results: list[str] = []
    errors: list[BaseException] = []

    def start() -> None:
        try:
            material = resolve_tls(config)
            assert material is not None
            results.append(material.fingerprint_sha256)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=start) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(set(results)) == 1
    final = resolve_tls(config)
    assert final is not None and final.fingerprint_sha256 == results[0]
    assert not list((tmp_path / "data" / "tls").glob("*.new"))


# uvicorn reads the pair under the lock a regeneration takes (review 2026-11-09)


class _Listener:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.loaded = False
        self.lock_held_during_load: bool | None = None

    def load(self) -> None:
        errors: list[BaseException] = []

        def contend() -> None:
            try:
                with tlsutil._folder_lock(self.folder):
                    pass
            except tlsutil.TlsError as exc:
                errors.append(exc)

        worker = threading.Thread(target=contend)
        worker.start()
        worker.join()
        self.lock_held_during_load = bool(errors)
        self.loaded = True


def test_load_listeners_loads_under_the_tls_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tlsutil, "_LOCK_WAIT_S", 0.3)
    config = _config(tmp_path)
    assert tlsutil.resolve_tls(config) is not None
    listener = _Listener(Path(config.data_dir) / "tls")
    done = _Listener(Path(config.data_dir) / "tls")
    done.loaded = True
    assert tlsutil.load_listeners(config, [listener, done, None]) is True
    assert listener.loaded and listener.lock_held_during_load is True
    assert done.lock_held_during_load is None, "an already-loaded config is left alone"


def test_load_listeners_leaves_plain_http_and_own_certs_to_uvicorn(tmp_path: Path) -> None:
    listener = _Listener(tmp_path)
    assert tlsutil.load_listeners(Config(data_dir=tmp_path / "data"), [listener]) is False
    assert listener.loaded is False


def test_a_real_uvicorn_config_loads_the_self_signed_pair(tmp_path: Path) -> None:
    import ssl

    import uvicorn

    config = _config(tmp_path)
    material = tlsutil.resolve_tls(config)
    assert material is not None

    async def app(scope: Any, receive: Any, send: Any) -> None:  # pragma: no cover
        return None

    listener = uvicorn.Config(app, log_config=None, **material.uvicorn_kwargs())
    assert tlsutil.load_listeners(config, [listener]) is True
    assert listener.loaded and isinstance(listener.ssl, ssl.SSLContext)
