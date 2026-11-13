"""Who a request says it is, for lease ownership and instance routing (D91, 8a).

One place builds a :class:`~studioforge.core.instances.Caller` from a request:
the lease ids it names (``X-SF-Lease``, ``X-SF-Lease-Id``, a body or query
``lease_id``), its ``X-SF-Client`` label, the scoped key it authenticated with
and whether it is an admin (D32). The routes hand that to the manager, which
decides whose instance a request, a load or an unload means. Before D91 the
key alone decided, and every bench run on one client shares one key.
"""

from __future__ import annotations

from typing import Any

from studioforge.api.auth import is_admin_caller, key_owner_of
from studioforge.core.attribution import client_label
from studioforge.core.instances import LEASE_HEADERS, Caller
from studioforge.types import ModelRecord


def _lease_ids(request: Any, extra: Any = ()) -> tuple[str, ...]:
    found: list[str] = []
    headers = getattr(request, "headers", None)
    for name in LEASE_HEADERS:
        value = headers.get(name) if headers is not None else None
        if isinstance(value, str) and value.strip():
            found.append(value.strip())
    query = getattr(request, "query_params", None)
    value = query.get("lease_id") if query is not None else None
    if isinstance(value, str) and value.strip():
        found.append(value.strip())
    for value in extra or ():
        if isinstance(value, str) and value.strip():
            found.append(value.strip())
    return tuple(dict.fromkeys(found))


def caller_of(request: Any, state: Any, *, lease_ids: Any = ()) -> Caller:
    """The :class:`Caller` behind ``request`` (``lease_ids``: body values to add)."""
    headers = getattr(request, "headers", None)
    label = client_label(headers.get("x-sf-client") if headers is not None else None)
    peer = getattr(getattr(request, "client", None), "host", None)
    config = getattr(state, "config", None)
    try:
        admin = bool(config is not None and is_admin_caller(request, config))
    except Exception:  # noqa: BLE001 - an identity read never fails a request
        admin = False
    return Caller(
        lease_ids=_lease_ids(request, lease_ids),
        client=label,
        key=key_owner_of(request),
        admin=admin,
        peer=client_label(str(peer)) if peer else None,
    )


def routed_serving(
    state: Any, request: Any, record: ModelRecord, *, lease_ids: Any = ()
) -> ModelRecord:
    """The record whose child serves this request: the serving record, routed (D91, item 6)."""
    serving = state.manager.serving_record(record)
    route = getattr(state.manager, "route_for", None)
    leases = getattr(state.manager, "leases", None)
    try:
        standing = len(leases) if leases is not None else 0
    except TypeError:  # a stand-in manager without a real book
        standing = 0
    if route is None or not standing:
        # Plain mode, or no lease standing: nothing to route by (and no work).
        return serving
    routed: ModelRecord = route(serving, caller_of(request, state, lease_ids=lease_ids))
    return routed
