"""TLS for the three listeners: a generated self-signed certificate, or your own.

``server.tls.mode`` is one switch for the gateway, the control panel and the
watchdog:

* ``off``          -- plain HTTP (the default; fine on loopback).
* ``self_signed``  -- a certificate generated into ``<data-dir>/tls/`` the first
  time it is needed and reused afterwards. Its SHA-256 fingerprint is shown in
  ``/api/status`` and the panel so a client can pin it. Regenerated when it is
  about to expire or when this machine's addresses changed (the SANs).
* ``cert``         -- the PEM files named by ``server.tls.cert_file`` / ``key_file``.

A leaf module (``cryptography`` imported lazily, no other StudioForge imports
besides :mod:`studioforge.secure_files`): the watchdog, a separate process that
may not import the app machinery, needs the same ``ssl_certfile`` / ``ssl_keyfile``
for its own listener.

Not a trust anchor: a self-signed certificate protects against passive
eavesdropping, and against active attackers only for clients that pin the
fingerprint. For a public name, put a real certificate in ``cert`` mode or use a
reverse proxy / ``tailscale serve``.
"""

from __future__ import annotations

import datetime
import hashlib
import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from studioforge.secure_files import restrict_to_owner, write_private

__all__ = [
    "TlsError",
    "TlsMaterial",
    "certificate_fingerprint",
    "internal_verify",
    "resolve_tls",
    "scheme_for",
]

CERT_NAME = "selfsigned.crt"
KEY_NAME = "selfsigned.key"
#: Self-signed certificates are valid this long and renewed 30 days before the end.
VALIDITY_DAYS = 825
RENEW_BEFORE_DAYS = 30


class TlsError(ValueError):
    """The TLS settings cannot be honoured (missing file, unreadable PEM, no cryptography)."""


@dataclass(frozen=True)
class TlsMaterial:
    cert_file: Path
    key_file: Path
    fingerprint_sha256: str
    mode: str
    not_after: str | None = None

    def uvicorn_kwargs(self) -> dict[str, str]:
        return {"ssl_certfile": str(self.cert_file), "ssl_keyfile": str(self.key_file)}

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "fingerprint_sha256": self.fingerprint_sha256,
            "not_after": self.not_after,
            "cert_file": str(self.cert_file),
        }


def scheme_for(config: Any) -> str:
    """``https`` when ``server.tls.mode`` is not ``off``, else ``http``."""
    tls = getattr(getattr(config, "server", None), "tls", None)
    return "http" if tls is None or tls.mode == "off" else "https"


def internal_verify(config: Any) -> bool | str:
    """The ``verify=`` argument for httpx calls *this process makes to its own listeners*.

    Plain HTTP: irrelevant (``True``). TLS: the call goes to loopback and the
    certificate is ours, possibly self-signed with no name for ``127.0.0.1``, so
    verification is skipped -- the peer is this machine, there is no network to
    be attacked. Never use for a remote host.
    """
    return scheme_for(config) == "http"


def certificate_fingerprint(cert_pem: bytes | Path) -> str:
    """SHA-256 of the DER form, as colon-separated upper-case hex (the browser's format)."""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import serialization
    except ImportError as exc:  # pragma: no cover - cryptography is a declared dependency
        raise TlsError("the 'cryptography' package is required for TLS") from exc
    data = cert_pem.read_bytes() if isinstance(cert_pem, Path) else cert_pem
    try:
        cert = x509.load_pem_x509_certificate(data)
    except ValueError as exc:
        raise TlsError(f"not a PEM certificate: {exc}") from exc
    digest = hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest().upper()
    return ":".join(digest[i : i + 2] for i in range(0, len(digest), 2))


def _not_after(cert_pem: bytes) -> str | None:
    try:
        from cryptography import x509

        cert = x509.load_pem_x509_certificate(cert_pem)
        return cert.not_valid_after_utc.isoformat()
    except Exception:  # noqa: BLE001 - informational only
        return None


def _local_names() -> tuple[list[str], list[str]]:
    """``(dns names, ip addresses)`` this machine answers to."""
    dns = {"localhost"}
    ips = {"127.0.0.1", "::1"}
    try:
        host = socket.gethostname()
        if host:
            dns.add(host)
        for info in socket.getaddrinfo(host, None):
            address = str(info[4][0]).split("%", 1)[0]
            try:
                ipaddress.ip_address(address)
            except ValueError:
                continue
            ips.add(address)
    except OSError:
        pass
    return sorted(dns), sorted(ips)


def _san_set(cert: Any) -> set[str]:
    from cryptography import x509

    try:
        ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return set()
    names = {str(n) for n in ext.get_values_for_type(x509.DNSName)}
    names |= {str(n) for n in ext.get_values_for_type(x509.IPAddress)}
    return names


def _generate_self_signed(cert_path: Path, key_path: Path) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    dns, ips = _local_names()
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "StudioForge"),
            x509.NameAttribute(NameOID.COMMON_NAME, "StudioForge self-signed"),
        ]
    )
    now = datetime.datetime.now(datetime.UTC)
    alt: list[x509.GeneralName] = [x509.DNSName(d) for d in dns]
    alt += [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=VALIDITY_DAYS))
        .add_extension(x509.SubjectAlternativeName(alt), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    restrict_to_owner(cert_path.parent, directory=True)
    write_private(
        key_path,
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def _self_signed_is_current(cert_path: Path, key_path: Path) -> bool:
    """Exists, not near expiry, and covers every address this machine has now."""
    if not (cert_path.is_file() and key_path.is_file()):
        return False
    try:
        from cryptography import x509

        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        remaining = cert.not_valid_after_utc - datetime.datetime.now(datetime.UTC)
        if remaining < datetime.timedelta(days=RENEW_BEFORE_DAYS):
            return False
        dns, ips = _local_names()
        return set(dns) | set(ips) <= _san_set(cert)
    except Exception:  # noqa: BLE001 - unreadable means regenerate
        return False


def resolve_tls(config: Any) -> TlsMaterial | None:
    """The certificate and key to serve, or ``None`` for plain HTTP.

    Generates the self-signed pair on first use. Raises :class:`TlsError` with a
    sentence the operator can act on when the settings cannot be honoured.
    """
    tls = config.server.tls
    if tls.mode == "off":
        return None
    if tls.mode == "cert":
        cert_file, key_file = Path(tls.cert_file), Path(tls.key_file)
        for label, path in (("cert_file", cert_file), ("key_file", key_file)):
            if not path.is_file():
                raise TlsError(f"server.tls.{label} {path} does not exist or is not a file")
        pem = cert_file.read_bytes()
        return TlsMaterial(
            cert_file, key_file, certificate_fingerprint(pem), "cert", _not_after(pem)
        )
    folder = Path(config.data_dir) / "tls"
    cert_path, key_path = folder / CERT_NAME, folder / KEY_NAME
    if not _self_signed_is_current(cert_path, key_path):
        try:
            _generate_self_signed(cert_path, key_path)
        except ImportError as exc:  # pragma: no cover
            raise TlsError("the 'cryptography' package is required for self_signed TLS") from exc
        except OSError as exc:
            raise TlsError(
                f"could not write the self-signed certificate to {folder}: {exc}"
            ) from exc
    pem = cert_path.read_bytes()
    return TlsMaterial(
        cert_path, key_path, certificate_fingerprint(pem), "self_signed", _not_after(pem)
    )
