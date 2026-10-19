"""studioforge-tenant: the tenant side of StudioForge's VRAM Tenant Protocol (vt.1).

A program that shares a GPU box with StudioForge -- an image server, a speech
server, a benchmark harness -- is a *tenant*: it leases a card before it
allocates on it, keeps the lease touched while it works, releases it when done,
and lets a better class ask it to leave (StudioForge DECISIONS.md D43, D56,
D61, D77-D79; the contract is ``docs/TENANTS.md`` in the server repo).

This package is that behaviour, once, as a thin client over the documented
HTTP routes: :class:`TenantClient` (acquire / touch / release / release_stale /
register / deregister / touch_loop) and :func:`vacate_asgi_route` (the D56
callback as an ASGI app a tenant mounts on its own server). ``httpx`` is the
only dependency, and nothing here imports ``studioforge``.

What it never does: start StudioForge, start anything else, or ``force`` a
lease. Each program stays its own repo, port, launcher and update path; the
protocol is how they share the cards.
"""

from __future__ import annotations

from studioforge_tenant.client import (
    PROTOCOL_VERSION,
    Granted,
    Outcome,
    Refused,
    TenantClient,
    Unreachable,
    Unsupported,
)
from studioforge_tenant.vacate import (
    VACATE_LEASE_HEADER,
    VACATE_TOKEN_HEADER,
    VacateRequest,
    constant_time_matcher,
    vacate_asgi_route,
)

__version__ = "2.1.0"

__all__ = [
    "PROTOCOL_VERSION",
    "VACATE_LEASE_HEADER",
    "VACATE_TOKEN_HEADER",
    "Granted",
    "Outcome",
    "Refused",
    "TenantClient",
    "Unreachable",
    "Unsupported",
    "VacateRequest",
    "__version__",
    "constant_time_matcher",
    "vacate_asgi_route",
]
