"""Accounts, API keys, the audit log, the doctor, the support bundle and ``/metrics``.

Everything that is about *who may use this server* and *whether it is healthy*,
as routes, so the CLI (``studioforge keys``, ``users``, ``doctor``,
``support-bundle``), the panel's Users & Keys and Diagnostics pages and an agent
all reach the same code.

Scopes (see ``api/auth.py:required_scope``): listing or changing keys, accounts,
the audit log, the doctor and the support bundle are ``admin``; ``/metrics`` and
``/api/auth/status`` are ``read``. A token is returned **once**, in the response to
the call that created it, and is never stored or logged; only its SHA-256 is.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from fastapi import APIRouter, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from studioforge.api.auth import (
    auth_enforced,
    has_admin_credential,
    remote_services,
)
from studioforge.errors import BadRequestError, StudioForgeError
from studioforge.logging import get_logger
from studioforge.tlsutil import describe_tls

log = get_logger(__name__)

router = APIRouter()
metrics_router = APIRouter()


def _state(request: Request) -> Any:
    return request.app.state


def _accounts(request: Request) -> Any:
    accounts = getattr(_state(request), "accounts", None)
    if accounts is None:
        raise StudioForgeError(
            "Accounts are not available on this server build.",
            code="internal_error",
            status_code=500,
        )
    return accounts


def _set_audit_target(request: Request, target: str) -> None:
    """Tell the audit middleware what this request acted on (never a secret)."""
    try:
        request.state.audit_target = target
    except AttributeError:  # an in-process stand-in request has no .state
        return


class CreateKeyBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    scope: Literal["read", "inference", "admin"] = "inference"
    expires_in_days: float | None = Field(default=None, gt=0, le=3650)
    rate_limit_rpm: int | None = Field(default=None, ge=1)


class CreateUserBody(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


# ---------------------------------------------------------------------------
# Posture
# ---------------------------------------------------------------------------


@router.get("/auth/status")
async def auth_status(request: Request) -> dict[str, Any]:
    """Whether credentials are required, who exists, and what is exposed.

    No secrets: counts, scopes and the fingerprint of the TLS certificate (which is
    public by nature). ``bind`` lists the listeners reachable from the network.
    """
    state = _state(request)
    config = state.config
    accounts = getattr(state, "accounts", None)
    summary = accounts.summary() if accounts is not None else {}
    return {
        "auth_required": auth_enforced(config, accounts),
        "admin_credential_exists": has_admin_credential(config, accounts),
        "legacy_api_key_set": bool(config.server.api_key),
        "mcp_pin_set": bool(config.mcp.pin),
        "bind": {
            "gateway": config.server.host,
            "panel": config.gui.host if config.gui.enabled else None,
            "watchdog": config.watchdog.host if config.watchdog.enabled else None,
            "remote": [{"service": name, "host": host} for name, host in remote_services(config)],
        },
        "tls": describe_tls(config),
        **summary,
    }


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------


@router.get("/keys")
async def list_keys(request: Request) -> dict[str, Any]:
    """Every key (revoked ones included), without hashes or tokens."""
    accounts = _accounts(request)
    return {"keys": [k.public() for k in accounts.list_keys()]}


@router.post("/keys")
async def create_key(request: Request, body: CreateKeyBody) -> dict[str, Any]:
    """Create a named, scoped key. The ``token`` in the answer is shown only now."""
    accounts = _accounts(request)
    principal = getattr(getattr(request, "state", None), "principal", None)
    record, token = accounts.create_key(
        body.name,
        body.scope,
        expires_in_days=body.expires_in_days,
        rate_limit_rpm=body.rate_limit_rpm,
        created_by=principal.actor if principal else None,
    )
    _set_audit_target(request, f"{record.name} ({record.scope}) id={record.id}")
    return {
        "key": record.public(),
        "token": token,
        "note": "Copy the token now; it is not stored and cannot be shown again.",
    }


@router.delete("/keys/{key_id}")
async def revoke_key(request: Request, key_id: str) -> dict[str, Any]:
    """Revoke a key by id (or by name when unambiguous). Idempotent."""
    accounts = _accounts(request)
    record = accounts.revoke_key(key_id)
    if record is None:
        raise BadRequestError(
            f"There is no key '{key_id}'. List them with GET /api/keys.",
            code="invalid_request",
            param="key_id",
        )
    _set_audit_target(request, f"{record.name} id={record.id}")
    return {"key": record.public()}


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------


@router.get("/users")
async def list_users(request: Request) -> dict[str, Any]:
    accounts = _accounts(request)
    return {"users": [a.public() for a in accounts.list_accounts()]}


@router.post("/users")
async def create_user(request: Request, body: CreateUserBody) -> dict[str, Any]:
    """Create an account. The first one is the owner (what the bind rule asks for)."""
    accounts = _accounts(request)
    account = await run_in_threadpool(accounts.create_account, body.username, body.password)
    _set_audit_target(request, f"{account.username} ({account.role})")
    return {"user": account.public()}


@router.post("/users/password")
async def change_password(request: Request, body: CreateUserBody) -> dict[str, Any]:
    """Set an account's password (``username`` + the new ``password``)."""
    accounts = _accounts(request)
    account = await run_in_threadpool(accounts.set_password, body.username, body.password)
    _set_audit_target(request, account.username)
    return {"user": account.public()}


# ---------------------------------------------------------------------------
# Audit, doctor, support bundle
# ---------------------------------------------------------------------------


@router.get("/audit")
async def audit_tail(request: Request, n: int = Query(100, ge=1, le=2000)) -> dict[str, Any]:
    """The last ``n`` audit entries, oldest first."""
    audit = getattr(_state(request), "audit", None)
    return {"entries": audit.tail(n) if audit is not None else []}


@router.get("/doctor")
async def doctor(request: Request) -> dict[str, Any]:
    """The doctor report: ports, GPU, engine, disk, permissions, TLS, migrations, auth."""
    from studioforge.core.doctor import run_doctor

    state = _state(request)
    report = await run_in_threadpool(
        run_doctor,
        state.config,
        probe=getattr(state, "probe", None),
        engine_manager=getattr(state, "engine_manager", None),
        accounts=getattr(state, "accounts", None),
    )
    return report.to_dict()


@router.post("/support-bundle")
async def support_bundle(request: Request) -> Response:
    """A zip for bug reports: redacted config, doctor, versions, GPUs, log tails."""
    from studioforge.core.doctor import run_doctor
    from studioforge.core.support_bundle import build_bundle

    state = _state(request)

    def build() -> bytes:
        report = run_doctor(
            state.config,
            probe=getattr(state, "probe", None),
            engine_manager=getattr(state, "engine_manager", None),
            accounts=getattr(state, "accounts", None),
        ).to_dict()
        return build_bundle(state.config, doctor=report, probe=getattr(state, "probe", None))

    payload = await run_in_threadpool(build)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    _set_audit_target(request, f"{len(payload)} bytes")
    return Response(
        payload,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="studioforge-support-{stamp}.zip"'},
    )


# ---------------------------------------------------------------------------
# Prometheus
# ---------------------------------------------------------------------------


@metrics_router.get("/metrics", include_in_schema=True, tags=["health"])
async def metrics(request: Request) -> Response:
    """Prometheus text: gateway counters, GPU memory and every loaded model's own metrics."""
    from studioforge.core.metrics import render_metrics

    text = await render_metrics(_state(request))
    return Response(text, media_type="text/plain; version=0.0.4; charset=utf-8")
