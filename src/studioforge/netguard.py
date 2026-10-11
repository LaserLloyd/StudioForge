"""Where this server is allowed to send an outbound request on a tenant's behalf (D77).

The tenant registry stores URLs that THIS process later calls: a tenant's
``health_url`` every thirty seconds, its ``warm_url`` when an operator clicks
Warm, and -- through the lease book -- its ``vacate_url``. Every one of those
is a URL a remote caller handed us, and a server that fetches arbitrary URLs
for anyone who can reach its port is a port scanner for the LAN and a way to
reach whatever a firewall keeps off it. So the set of hosts a tenant URL may
name is closed: the loopback addresses, the tailnet (CGNAT) range
``100.64.0.0/10`` that Tailscale hands out, and any host the operator lists
in ``tenants.allowed_hosts``. Names other than ``localhost`` are never
resolved -- a DNS name that resolves somewhere else later is the classic
rebinding hole, so a name is only accepted when it appears in the allow-list
verbatim.

A leaf module: stdlib only, imported by ``config.py`` (which validates the
entries an operator wrote) and by ``core/tenants.py`` (which validates what a
tenant registers at runtime), so the two cannot disagree.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterable
from urllib.parse import urlsplit

#: Tailscale's CGNAT range: every tailnet address a node hands out lives here.
TAILNET_NETWORK = ipaddress.ip_network("100.64.0.0/10")
TENANT_URL_MAX_CHARS = 512
_SCHEMES = ("http", "https")
_DEFAULT_PORTS = {"http": 80, "https": 443}


class TenantUrlError(ValueError):
    """A tenant URL this server will not call. The text names the rule."""


def literal_ip(text: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """``text`` as an address, every spelling ``connect()`` accepts, or ``None``.

    ``ipaddress`` parses dotted-quad and IPv6 only; ``127.1``, ``0x7f000001``
    and ``2130706433`` all route to loopback and slip past it, so
    ``inet_aton`` is the fallback. An IPv4-mapped IPv6 address answers as the
    IPv4 it maps.
    """
    raw = (text or "").strip().strip("[]")
    if not raw:
        return None
    parsed: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
    try:
        parsed = ipaddress.ip_address(raw)
    except ValueError:
        try:
            packed = socket.inet_aton(raw)
        except (OSError, ValueError):
            return None
        parsed = ipaddress.ip_address(socket.inet_ntoa(packed))
    mapped = getattr(parsed, "ipv4_mapped", None)
    return mapped if mapped is not None else parsed


def host_is_loopback(host: str) -> bool:
    lowered = (host or "").strip().lower().rstrip(".")
    if lowered == "localhost" or lowered.endswith(".localhost"):
        return True
    ip = literal_ip(host)
    return ip is not None and ip.is_loopback


def host_is_tailnet(host: str) -> bool:
    ip = literal_ip(host)
    return ip is not None and ip.version == 4 and ip in TAILNET_NETWORK


def host_allowed(host: str, allowed_hosts: Iterable[str] = ()) -> bool:
    """Loopback, the tailnet range, or a host the operator listed verbatim.

    A listed entry is compared case-insensitively as written: an IP literal
    matches the same literal (and its other spellings, through
    :func:`literal_ip`), a name matches the same name. Nothing is resolved.
    """
    if host_is_loopback(host) or host_is_tailnet(host):
        return True
    lowered = (host or "").strip().lower().rstrip(".")
    ip = literal_ip(host)
    for entry in allowed_hosts:
        text = str(entry or "").strip().lower().rstrip(".")
        if not text:
            continue
        if text == lowered:
            return True
        listed = literal_ip(text)
        if ip is not None and listed is not None and ip == listed:
            return True
    return False


def validate_tenant_url(
    url: object, *, field: str, allowed_hosts: Iterable[str] = (), require_host_policy: bool = True
) -> str:
    """The URL this server may call for a tenant, or :class:`TenantUrlError`.

    http(s) only, no embedded credentials, a usable port, under
    :data:`TENANT_URL_MAX_CHARS`, no whitespace -- and, with
    ``require_host_policy`` (the default), a host :func:`host_allowed` accepts.
    The message names ``field`` so a 400 can say which key was wrong.
    """
    text = url.strip() if isinstance(url, str) else ""
    if not text:
        raise TenantUrlError(f"{field} is empty; give an http(s) URL")
    if len(text) > TENANT_URL_MAX_CHARS:
        raise TenantUrlError(f"{field} is longer than {TENANT_URL_MAX_CHARS} characters")
    if any(ch.isspace() for ch in text):
        raise TenantUrlError(f"{field} contains whitespace")
    try:
        parts = urlsplit(text)
    except ValueError as exc:
        raise TenantUrlError(f"{field} could not be parsed ({exc})") from exc
    scheme = (parts.scheme or "").lower()
    if scheme not in _SCHEMES:
        raise TenantUrlError(
            f"{field} must be an http:// or https:// URL, got "
            f"{('scheme ' + scheme) if scheme else 'no scheme'}"
        )
    host = parts.hostname or ""
    if not host:
        raise TenantUrlError(f"{field} has no host")
    try:
        parts.port  # noqa: B018 - the property raises on an unusable port
    except ValueError as exc:
        raise TenantUrlError(f"{field} has an unusable port (1-65535)") from exc
    if parts.username or parts.password:
        raise TenantUrlError(f"{field} must not embed credentials")
    if require_host_policy and not host_allowed(host, allowed_hosts):
        raise TenantUrlError(
            f"{field} names host {host!r}, which this server will not call: a tenant URL "
            f"must be loopback, a tailnet address (100.64.0.0/10) or a host listed in "
            f"tenants.allowed_hosts; DNS names other than localhost are never resolved"
        )
    return text


def url_target(url: str) -> tuple[str, int]:
    """``(hostname, port)`` of a validated URL, the scheme's default port filled in."""
    parts = urlsplit(url)
    scheme = (parts.scheme or "http").lower()
    return str(parts.hostname or ""), int(parts.port or _DEFAULT_PORTS.get(scheme, 0))
