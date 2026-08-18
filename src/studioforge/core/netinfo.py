"""Which addresses this server is actually reachable on.

``server.host`` is a *bind* address -- almost always ``0.0.0.0`` -- which tells
a user nothing about where to point a client. This works out the concrete
addresses another machine can use, so the startup banner, the GUI and
``/api/mcp/info`` can print a URL that works instead of one that does not.

Tailscale is treated as a first-class answer, and listed first, because that is
how this server is reached from another machine: a tailnet address keeps working
across networks, while a LAN address silently breaks the moment the DHCP lease
or the network changes.
"""

from __future__ import annotations

import ipaddress
import shutil
import socket
import subprocess
from dataclasses import dataclass

from studioforge.logging import get_logger

log = get_logger(__name__)

#: Tailscale hands out addresses from the CGNAT range 100.64.0.0/10.
_TAILSCALE_NET = ipaddress.ip_network("100.64.0.0/10")

#: Link-local; useless to advertise.
_LINK_LOCAL = ipaddress.ip_network("169.254.0.0/16")


@dataclass(frozen=True)
class Address:
    ip: str
    kind: str  # "tailscale" | "lan" | "loopback"
    label: str

    @property
    def sort_key(self) -> tuple[int, str]:
        order = {"tailscale": 0, "lan": 1, "loopback": 2}
        return (order.get(self.kind, 3), self.ip)


def _classify(ip: str) -> str | None:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return None
    if addr.version != 4:
        return None
    if addr in _TAILSCALE_NET:
        return "tailscale"
    if addr.is_loopback:
        return "loopback"
    if addr in _LINK_LOCAL or addr.is_multicast or addr.is_reserved:
        return None
    if addr.is_private or addr.is_global:
        return "lan"
    return None


def tailscale_addresses() -> list[str]:
    """Ask the Tailscale CLI directly when it is installed.

    More reliable than inferring from the interface list: a tailnet address is
    in the CGNAT range, but so is anything else using that range, and Tailscale
    interfaces are not always enumerated by the usual host lookups.
    """
    binary = shutil.which("tailscale")
    if binary is None:
        return []
    try:
        result = subprocess.run([binary, "ip", "-4"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def local_addresses() -> list[Address]:
    """Every IPv4 address a client could plausibly reach this host on."""
    found: dict[str, Address] = {}

    for ip in tailscale_addresses():
        if _classify(ip) is not None:
            found[ip] = Address(ip=ip, kind="tailscale", label="Tailscale")

    # A UDP connect() to a public address does no traffic but makes the OS pick
    # the interface it would actually route from -- which is the address a LAN
    # client should use, and is more accurate than taking the first hostname.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.settimeout(0.5)
            probe.connect(("8.8.8.8", 80))
            primary = str(probe.getsockname()[0])
        kind = _classify(primary)
        if kind and primary not in found:
            found[primary] = Address(
                ip=primary, kind=kind, label="LAN" if kind == "lan" else kind.title()
            )
    except OSError:
        pass

    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            # sockaddr is (host, port) for AF_INET; the tuple is typed loosely.
            ip = str(info[4][0])
            kind = _classify(ip)
            if kind and ip not in found:
                found[ip] = Address(
                    ip=ip, kind=kind, label="LAN" if kind == "lan" else kind.title()
                )
    except (OSError, socket.gaierror):
        pass

    found.setdefault("127.0.0.1", Address(ip="127.0.0.1", kind="loopback", label="This machine"))
    return sorted(found.values(), key=lambda a: a.sort_key)


def reachable_urls(port: int, path: str = "", *, host: str | None = None) -> list[dict[str, str]]:
    """Concrete URLs for a service on ``port``.

    When the server is bound to a specific interface rather than a wildcard,
    that is the only address it answers on -- advertising others would be a
    lie, so the bind address wins.
    """
    suffix = path if path.startswith("/") or not path else f"/{path}"
    if host and host not in {"0.0.0.0", "::", ""}:
        return [
            {
                "ip": host,
                "kind": "bound",
                "label": "Configured bind address",
                "url": f"http://{host}:{port}{suffix}",
            }
        ]
    return [
        {
            "ip": address.ip,
            "kind": address.kind,
            "label": address.label,
            "url": f"http://{address.ip}:{port}{suffix}",
        }
        for address in local_addresses()
    ]


def primary_url(port: int, path: str = "", *, host: str | None = None) -> str:
    """The single best URL to hand someone: tailnet, else LAN, else loopback."""
    urls = reachable_urls(port, path, host=host)
    return urls[0]["url"] if urls else f"http://127.0.0.1:{port}{path}"
