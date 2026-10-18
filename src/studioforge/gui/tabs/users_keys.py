"""Users & Keys: who may use this server, and with what.

The panel's face of :class:`studioforge.core.accounts.AccountStore` and the
``/api/keys`` and ``/api/users`` routes: the sign-in posture, the owner and
other accounts, and named, scoped API keys with the usual one-time token.

Rules that hold on every control here:

* **Admin first.** Each mutation calls :func:`require_local_admin` before it
  touches the store, exactly as the other pages do, then the same route handler
  the HTTP API uses (so validation cannot drift), then writes an audit line with
  the signed-in actor and ``surface='panel'``.
* **A token is shown once.** The new key's token goes to the one-time dialog and
  nowhere else: not a table, not a notification, not a log line, not the audit
  trail. Hashes never leave the store.
* **Passwords stay out of the audit trail and the logs.** Only the account name
  is recorded.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui
from pydantic import ValidationError

from studioforge.api.auth import auth_enforced, remote_services
from studioforge.errors import StudioForgeError
from studioforge.gui.tabs import (
    GuiContext,
    api_request,
    audit_panel,
    notify_error,
    panel_guard,
    remote_viewer_banner,
    require_local_admin,
    viewer_actor,
    viewer_may_change_box,
)
from studioforge.gui.tabs.widgets import (
    DASH,
    confirm,
    fmt_ts,
    one_time_secret_dialog,
)
from studioforge.logging import get_logger

log = get_logger(__name__)

SCOPES = ("read", "inference", "admin")

SCOPE_HELP = {
    "read": "can look at status and lists, and read metrics",
    "inference": "can also chat with models and load them",
    "admin": "can change settings, delete models and manage keys",
}

AUDIT_ROWS = 50


# ---------------------------------------------------------------------------
# Data (pure enough to test without a page)
# ---------------------------------------------------------------------------


def _accounts(ctx: GuiContext) -> Any:
    accounts = getattr(ctx.api_state, "accounts", None)
    if accounts is None:
        raise StudioForgeError(
            "Accounts are not available on this server build.",
            code="internal_error",
            status_code=500,
        )
    return accounts


def key_state(row: dict[str, Any]) -> str:
    """``revoked``, ``expired`` or ``active`` for a ``KeyRecord.public()`` row."""
    if row.get("revoked"):
        return "revoked"
    if row.get("expired"):
        return "expired"
    return "active"


def key_rows(ctx: GuiContext) -> list[dict[str, Any]]:
    """The keys table rows. Built from ``KeyRecord.public()``: no hash, no token."""
    accounts = getattr(ctx.api_state, "accounts", None)
    if accounts is None:
        return []
    rows: list[dict[str, Any]] = []
    for record in accounts.list_keys():
        public = record.public()
        rows.append(
            {
                "id": public["id"],
                "name": public["name"],
                "scope": public["scope"],
                "prefix": f"{public['prefix']}…",
                "created": fmt_ts(public["created_at"]),
                "last_used": fmt_ts(public["last_used_at"], empty="never"),
                "expires": fmt_ts(public["expires_at"], empty="never"),
                "rpm": public["rate_limit_rpm"] or "no limit",
                "state": key_state(public),
            }
        )
    return rows


def account_rows(ctx: GuiContext) -> list[dict[str, Any]]:
    accounts = getattr(ctx.api_state, "accounts", None)
    if accounts is None:
        return []
    return [
        {
            "username": a.username,
            "role": a.role,
            "created": fmt_ts(a.created_at),
            "last_login": fmt_ts(a.last_login_at, empty="never"),
            "state": "disabled" if a.disabled else "active",
        }
        for a in accounts.list_accounts()
    ]


def auth_posture(ctx: GuiContext) -> dict[str, Any]:
    """What ``GET /api/auth/status`` reports, assembled from the same pieces."""
    config = ctx.config
    accounts = getattr(ctx.api_state, "accounts", None)
    summary = accounts.summary() if accounts is not None else {}
    return {
        "auth_required": auth_enforced(config, accounts),
        "legacy_api_key_set": bool(config.server.api_key),
        "mcp_pin_set": bool(config.mcp.pin),
        "remote": [{"service": name, "host": host} for name, host in remote_services(config)],
        **summary,
    }


def posture_lines(posture: dict[str, Any]) -> list[tuple[str, str, str]]:
    """``(icon, colour, text)`` lines for the posture card."""
    lines: list[tuple[str, str, str]] = []
    if posture.get("auth_required"):
        lines.append(("lock", "positive", "Sign-in is required to use the API and this panel."))
    else:
        lines.append(
            (
                "lock_open",
                "warning",
                "No sign-in is required yet: anyone on this computer can use the server. Create an "
                "owner account or a key before you share it on your network.",
            )
        )
    owner = posture.get("owner_exists")
    lines.append(
        (
            "person" if owner else "person_off",
            "positive" if owner else "warning",
            f"Owner account: {'created' if owner else 'none yet'} "
            f"({posture.get('accounts', 0)} account(s)).",
        )
    )
    scopes = posture.get("keys_by_scope") or {}
    by_scope = ", ".join(f"{n} {s}" for s, n in scopes.items() if n) or "none"
    lines.append(
        (
            "vpn_key",
            "primary",
            f"API keys: {posture.get('keys_active', 0)} active ({by_scope}), "
            f"{posture.get('keys_revoked', 0)} revoked.",
        )
    )
    lines.append(
        (
            "key",
            "primary",
            f"Secrets are stored in: {posture.get('secret_backend', 'unknown')}.",
        )
    )
    if posture.get("legacy_api_key_set"):
        lines.append(
            (
                "history",
                "warning",
                "The old shared server key (server.api_key) is set. It still works but is "
                "deprecated: make named keys below and clear it.",
            )
        )
    if posture.get("mcp_pin_set"):
        lines.append(
            (
                "history",
                "warning",
                "The MCP pairing PIN is set. It still works but is deprecated: give agents a "
                "scoped key instead.",
            )
        )
    for item in posture.get("remote") or []:
        lines.append(
            ("public", "info", f"{item['service'].capitalize()} listens on {item['host']}.")
        )
    return lines


def _friendly(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        where = ".".join(str(x) for x in err.get("loc", ()) if x != "body")
        parts.append(f"{where}: {err.get('msg', 'is not valid')}" if where else err.get("msg", ""))
    return "That value is not allowed (" + "; ".join(parts) + ")."


# ---------------------------------------------------------------------------
# Mutations: admin check -> route handler -> audit line
# ---------------------------------------------------------------------------


async def create_key(
    ctx: GuiContext,
    name: str,
    scope: str,
    *,
    expires_in_days: float | None = None,
    rate_limit_rpm: int | None = None,
) -> tuple[dict[str, Any], str]:
    """Create a scoped key. Returns ``(public row, token)``; the token is for the dialog only."""
    require_local_admin(ctx, "create an API key")
    from studioforge.api.security_routes import CreateKeyBody
    from studioforge.api.security_routes import create_key as route

    try:
        body = CreateKeyBody(
            name=(name or "").strip(),
            scope=scope,  # type: ignore[arg-type]
            expires_in_days=expires_in_days or None,
            rate_limit_rpm=int(rate_limit_rpm) if rate_limit_rpm else None,
        )
    except ValidationError as exc:
        raise StudioForgeError(_friendly(exc), code="invalid_request", status_code=400) from exc
    payload = await route(api_request(ctx), body)
    record = dict(payload["key"])
    audit_panel(ctx, "panel.key.create", f"{record['name']} ({record['scope']}) id={record['id']}")
    return record, str(payload["token"])


async def revoke_key(ctx: GuiContext, key_id: str) -> dict[str, Any]:
    require_local_admin(ctx, "revoke an API key")
    from studioforge.api.security_routes import revoke_key as route

    payload = await route(api_request(ctx), key_id)
    record = dict(payload["key"])
    audit_panel(ctx, "panel.key.revoke", f"{record['name']} id={record['id']}")
    return record


def _check_passwords(password: str, again: str) -> None:
    if password != again:
        raise StudioForgeError(
            "The two passwords do not match. Type the same password in both boxes.",
            code="invalid_request",
            status_code=400,
        )


async def create_account(
    ctx: GuiContext, username: str, password: str, again: str
) -> dict[str, Any]:
    require_local_admin(ctx, "create an account")
    _check_passwords(password, again)
    from studioforge.api.security_routes import CreateUserBody, create_user

    try:
        body = CreateUserBody(username=(username or "").strip(), password=password)
    except ValidationError as exc:
        raise StudioForgeError(
            "Enter a username (up to 64 characters) and a password.",
            code="invalid_request",
            status_code=400,
        ) from exc
    payload = await create_user(api_request(ctx), body)
    user = dict(payload["user"])
    audit_panel(ctx, "panel.account.create", f"{user['username']} ({user['role']})")
    return user


async def change_password(
    ctx: GuiContext, username: str, password: str, again: str
) -> dict[str, Any]:
    require_local_admin(ctx, "change a password")
    _check_passwords(password, again)
    from studioforge.api.security_routes import CreateUserBody
    from studioforge.api.security_routes import change_password as route

    try:
        body = CreateUserBody(username=(username or "").strip(), password=password)
    except ValidationError as exc:
        raise StudioForgeError(
            "Choose an account and enter the new password.", code="invalid_request", status_code=400
        ) from exc
    payload = await route(api_request(ctx), body)
    user = dict(payload["user"])
    audit_panel(ctx, "panel.account.password", user["username"])
    return user


def audit_rows(ctx: GuiContext, n: int = AUDIT_ROWS) -> list[dict[str, Any]]:
    audit = getattr(ctx.api_state, "audit", None)
    if audit is None:
        return []
    try:
        return list(reversed(audit.tail(n)))
    except Exception as exc:  # noqa: BLE001
        log.debug("could not read the audit log", error=str(exc))
        return []


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def posture_card(ctx: GuiContext) -> None:
    with ui.card().classes("w-full"):
        ui.label("Who can use this server").classes("text-lg font-medium")
        for icon, colour, text in posture_lines(auth_posture(ctx)):
            with ui.row().classes("w-full items-start gap-2 no-wrap"):
                ui.icon(icon, color=colour, size="1.1rem").props('aria-hidden="true"')
                ui.label(text).classes("text-sm")
        ui.label(f"You are signed in as: {viewer_actor()}").classes("text-xs opacity-70")


def _accounts_card(ctx: GuiContext, refresh: Any) -> None:
    rows = account_rows(ctx)
    with ui.card().classes("w-full"):
        ui.label("Accounts").classes("text-lg font-medium")
        ui.label(
            "The first account is the owner. Accounts sign in to this panel with a username and "
            "password."
        ).classes("text-xs opacity-70")
        if rows:
            ui.table(
                columns=[
                    {"name": "username", "label": "Username", "field": "username", "align": "left"},
                    {"name": "role", "label": "Role", "field": "role", "align": "left"},
                    {"name": "created", "label": "Created", "field": "created", "align": "left"},
                    {
                        "name": "last_login",
                        "label": "Last sign-in",
                        "field": "last_login",
                        "align": "left",
                    },
                    {"name": "state", "label": "State", "field": "state", "align": "left"},
                ],
                rows=rows,
                row_key="username",
            ).props("dense flat").classes("w-full")
        else:
            ui.label("No accounts yet.").classes("text-sm opacity-70")

        with ui.expansion("Create an account", icon="person_add").classes("w-full"):
            username = ui.input("Username").props('dense outlined autocomplete="off"')
            password = ui.input("Password", password=True, password_toggle_button=True).props(
                'dense outlined autocomplete="new-password"'
            )
            again = ui.input("Type the password again", password=True).props(
                'dense outlined autocomplete="new-password"'
            )

            async def make() -> None:
                try:
                    user = await create_account(ctx, username.value, password.value, again.value)
                except Exception as exc:  # noqa: BLE001
                    notify_error(exc, what="create account")
                    return
                ui.notify(
                    f"Account '{user['username']}' created ({user['role']}).", type="positive"
                )
                refresh()

            ui.button("Create account", icon="person_add", on_click=make).props(
                "color=primary dense"
            )

        if rows:
            with ui.expansion("Change a password", icon="password").classes("w-full"):
                who = ui.select(
                    [row["username"] for row in rows], value=rows[0]["username"], label="Account"
                ).props("dense outlined")
                new = ui.input("New password", password=True, password_toggle_button=True).props(
                    'dense outlined autocomplete="new-password"'
                )
                repeat = ui.input("Type the new password again", password=True).props(
                    'dense outlined autocomplete="new-password"'
                )

                async def change() -> None:
                    try:
                        user = await change_password(ctx, who.value, new.value, repeat.value)
                    except Exception as exc:  # noqa: BLE001
                        notify_error(exc, what="change password")
                        return
                    ui.notify(
                        f"Password for '{user['username']}' changed. Everyone signed in with the "
                        "old password has to sign in again.",
                        type="positive",
                        multi_line=True,
                    )
                    refresh()

                ui.button("Change password", icon="password", on_click=change).props(
                    "color=primary dense"
                )


def _keys_card(ctx: GuiContext, refresh: Any) -> None:
    rows = key_rows(ctx)
    with ui.card().classes("w-full"):
        ui.label("API keys").classes("text-lg font-medium")
        ui.label(
            "A key lets a program use the server without a password. Give each program its own "
            "key with the smallest scope that works, so you can revoke one without touching the "
            "others."
        ).classes("text-xs opacity-70")
        if rows:
            table = (
                ui.table(
                    columns=[
                        {"name": "name", "label": "Name", "field": "name", "align": "left"},
                        {"name": "scope", "label": "Scope", "field": "scope", "align": "left"},
                        {
                            "name": "prefix",
                            "label": "Starts with",
                            "field": "prefix",
                            "align": "left",
                        },
                        {
                            "name": "created",
                            "label": "Created",
                            "field": "created",
                            "align": "left",
                        },
                        {
                            "name": "last_used",
                            "label": "Last used",
                            "field": "last_used",
                            "align": "left",
                        },
                        {
                            "name": "expires",
                            "label": "Expires",
                            "field": "expires",
                            "align": "left",
                        },
                        {"name": "rpm", "label": "Requests/min", "field": "rpm", "align": "left"},
                        {"name": "state", "label": "State", "field": "state", "align": "left"},
                        {"name": "actions", "label": "", "field": "id", "align": "right"},
                    ],
                    rows=rows,
                    row_key="id",
                )
                .props("dense flat")
                .classes("w-full")
            )
            table.add_slot(
                "body-cell-actions",
                """
                <q-td :props="props">
                  <q-btn v-if="props.row.state === 'active'" flat dense color="negative"
                         label="Revoke" :aria-label="'Revoke key ' + props.row.name"
                         @click="() => $parent.$emit('revoke', props.row.id)" />
                </q-td>
                """,
            )

            async def ask_revoke(event: Any) -> None:
                key_id = str(event.args)
                row = next((r for r in rows if r["id"] == key_id), None)
                if row is None:
                    return
                own = viewer_actor() == f"key:{key_id}"
                body = (
                    f"Anything using the key '{row['name']}' stops working immediately. This "
                    "cannot be undone; you would make a new key."
                )
                if own:
                    body += " You are signed in with this key, so you will be signed out."
                if not await confirm(f"Revoke '{row['name']}'?", body, action="Revoke key"):
                    return
                try:
                    await revoke_key(ctx, key_id)
                except Exception as exc:  # noqa: BLE001
                    notify_error(exc, what="revoke key")
                    return
                ui.notify(f"Key '{row['name']}' revoked.", type="positive")
                refresh()

            table.on("revoke", ask_revoke)
        else:
            ui.label("No keys yet.").classes("text-sm opacity-70")

        create_key_form(ctx, refresh)


def create_key_form(ctx: GuiContext, refresh: Any, *, default_scope: str = "inference") -> None:
    """Name, scope, optional expiry and rate limit; the token appears once, in a dialog."""
    with ui.expansion("Create a key", icon="add").classes("w-full"):
        name = ui.input("Key name (what will use it)").props('dense outlined autocomplete="off"')
        scope = ui.select(
            {s: f"{s} — {SCOPE_HELP[s]}" for s in SCOPES}, value=default_scope, label="Scope"
        ).props("dense outlined")
        expiry = ui.number("Expires after (days, empty = never)", min=0, precision=0).props(
            "dense outlined clearable"
        )
        rpm = ui.number("Requests per minute (empty = no limit)", min=0, precision=0).props(
            "dense outlined clearable"
        )

        async def make() -> None:
            try:
                record, token = await create_key(
                    ctx,
                    name.value or "",
                    scope.value,
                    expires_in_days=float(expiry.value) if expiry.value else None,
                    rate_limit_rpm=int(rpm.value) if rpm.value else None,
                )
            except Exception as exc:  # noqa: BLE001
                notify_error(exc, what="create key")
                return
            name.set_value("")
            one_time_secret_dialog(
                f"Key '{record['name']}' created",
                token,
                hint=f"Give this {record['scope']} key to the program that will use it.",
            )
            refresh()

        ui.button("Create key", icon="vpn_key", on_click=make).props("color=primary dense")


def _audit_card(ctx: GuiContext) -> None:
    with ui.expansion(f"Audit log (last {AUDIT_ROWS})", icon="history").classes("w-full"):
        rows = audit_rows(ctx)
        if not rows:
            ui.label("Nothing recorded yet.").classes("text-sm opacity-70")
            return
        ui.label(
            "Every change made through the API, the agent tools and this panel, newest first. "
            "Secrets are never recorded."
        ).classes("text-xs opacity-70")
        table_rows = [
            {
                "ts": str(r.get("ts", "")),
                "actor": str(r.get("actor", "")),
                "action": str(r.get("action", "")),
                "target": str(r.get("target", "")) or DASH,
                "result": str(r.get("result", "")),
                "surface": str(r.get("surface", "")),
            }
            for r in rows
        ]
        ui.table(
            columns=[
                {"name": k, "label": label, "field": k, "align": "left"}
                for k, label in (
                    ("ts", "When (UTC)"),
                    ("actor", "Who"),
                    ("action", "What"),
                    ("target", "On"),
                    ("result", "Result"),
                    ("surface", "From"),
                )
            ],
            rows=table_rows,
        ).props("dense flat").classes("w-full")


def render(ctx: GuiContext) -> None:
    holder = ui.column().classes("w-full gap-4 p-2")

    def draw() -> None:
        holder.clear()
        with holder:
            with panel_guard("The sign-in summary"):
                posture_card(ctx)
                if not viewer_may_change_box(ctx):
                    remote_viewer_banner()
            if getattr(ctx.api_state, "accounts", None) is None:
                ui.label("Accounts are not available on this server build.").classes(
                    "text-sm opacity-70"
                )
                return
            with panel_guard("Accounts"):
                _accounts_card(ctx, draw)
            with panel_guard("API keys"):
                _keys_card(ctx, draw)
            with panel_guard("The audit log"):
                _audit_card(ctx)

    draw()


__all__ = [
    "account_rows",
    "audit_rows",
    "auth_posture",
    "change_password",
    "create_account",
    "create_key",
    "key_rows",
    "key_state",
    "posture_lines",
    "render",
    "revoke_key",
]
