"""Serving: the address, who can reach it, how long models stay loaded.

The decisions an owner makes about *how the server is offered* to the network:
the port, which interfaces it listens on, HTTPS, how long idle models stay in
memory, and which browser pages may call it. The existing Network & access card
(credentials, reachable addresses) sits underneath. Every save goes through
:func:`studioforge.gui.tabs.apply_config_updates`, the one path every setting
takes, so validation, the atomic write, the restart notice and the audit line
cannot differ from the Advanced page.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from studioforge.gui import state as st
from studioforge.gui.tabs import GuiContext, notify_error, panel_guard
from studioforge.gui.tabs.overview import base_url_card
from studioforge.gui.tabs.widgets import copy_button, is_remote_without_admin, section_url
from studioforge.logging import get_logger

log = get_logger(__name__)

TLS_MODES: dict[str, str] = {
    "off": "Off (plain HTTP)",
    "self_signed": "Self-signed certificate (StudioForge makes one)",
    "cert": "My own certificate files",
}


def _warning_card(ctx: GuiContext) -> None:
    """Refuse-to-start warning: a network bind with nobody to sign in."""
    if not is_remote_without_admin(ctx):
        return
    from studioforge.api.auth import remote_services

    where = ", ".join(f"{name} ({host})" for name, host in remote_services(ctx.config))
    with ui.card().classes("w-full sf-danger-card").props('role="alert"'):
        with ui.row().classes("items-center gap-2 no-wrap"):
            ui.icon("gpp_maybe", color="negative", size="1.6rem").props('aria-hidden="true"')
            ui.label("The server will refuse to start").classes("text-lg font-medium text-negative")
        ui.label(
            f"{where} would listen on the network, but there is no owner account or admin key, so "
            "anyone on the network could drive this server. Create an owner account or an admin "
            "key first, or set the bind addresses back to 127.0.0.1 (this computer only)."
        ).classes("text-sm")
        ui.link("Open Accounts & keys", section_url("Users & Keys")).classes("text-sm")


def _fingerprint_line(ctx: GuiContext) -> None:
    from studioforge.tlsutil import describe_tls

    info = describe_tls(ctx.config)
    if info.get("mode") == "off":
        ui.label("HTTPS is off. Traffic on the network is not encrypted.").classes(
            "text-xs opacity-70"
        )
        return
    fingerprint = info.get("fingerprint_sha256")
    if not fingerprint:
        ui.label(
            "The certificate does not exist yet. It is made the first time the server starts "
            "with this mode."
        ).classes("text-xs opacity-70")
        return
    with ui.row().classes("items-center gap-2 no-wrap"):
        ui.label("Certificate fingerprint (SHA-256)").classes("text-xs font-medium")
        copy_button(str(fingerprint), label="Copy the certificate fingerprint")
    ui.label(str(fingerprint)).classes("font-mono text-xs break-all")
    expires = info.get("not_after")
    if expires:
        ui.label(f"Valid until {expires}").classes("text-xs opacity-70")
    ui.label(
        "Compare this with what your browser shows before trusting a self-signed certificate."
    ).classes("text-xs opacity-70")


def listening_card(ctx: GuiContext) -> None:
    """Port and bind addresses for the three listeners."""
    from studioforge.gui.tabs import setup

    card = ui.card().classes("w-full")

    def draw() -> None:
        card.clear()
        with card, panel_guard("Listening"):
            ui.label("Listening").classes("text-lg font-medium")
            ui.label(
                "The port clients connect to, and which network interfaces the server, the control "
                "panel and the recovery watchdog answer on."
            ).classes("text-xs opacity-70")
            fields = setup._Fields(ctx, setup._payload(ctx))
            fields.row("server.port", label="Server port")
            ui.label(st.port_conflict_note(ctx.config.server.port)).classes("text-xs opacity-70")
            choices = ["127.0.0.1", "0.0.0.0"]
            for key, label in (
                ("server.host", "Server bind address"),
                ("gui.host", "Control panel bind address"),
                ("watchdog.host", "Watchdog bind address"),
            ):
                current = st.spec_display_value(setup._payload(ctx), fields.specs[key])
                fields.row(key, label=label, options=sorted({*choices, str(current or "")} - {""}))
            ui.label(st.bind_note(ctx.config.server.host)).classes("text-xs opacity-70")
            with ui.row().classes("items-center gap-2"):
                setup._save_button(ctx, fields, draw)

    draw()


def tls_card(ctx: GuiContext) -> None:
    """HTTPS mode, certificate files and the certificate's fingerprint."""
    from studioforge.gui.tabs import setup

    card = ui.card().classes("w-full")

    def draw() -> None:
        card.clear()
        tls = ctx.config.server.tls
        with card, panel_guard("HTTPS"):
            ui.label("HTTPS").classes("text-lg font-medium")
            ui.label(
                "Encrypts traffic between other computers and this server. Needed before you share "
                "it on your network. A change takes effect after a restart."
            ).classes("text-xs opacity-70")
            mode = ui.select(TLS_MODES, value=tls.mode, label="HTTPS mode").props("dense outlined")
            cert = ui.input("Certificate file (PEM)", value=str(tls.cert_file or "")).props(
                "dense outlined"
            )
            key = ui.input("Private key file (PEM)", value=str(tls.key_file or "")).props(
                "dense outlined"
            )
            cert.bind_visibility_from(mode, "value", backward=lambda v: v == "cert")
            key.bind_visibility_from(mode, "value", backward=lambda v: v == "cert")
            _fingerprint_line(ctx)
            result = ui.label("").classes("text-xs")

            async def save() -> None:
                updates: dict[str, Any] = {}
                if mode.value != tls.mode:
                    updates["server.tls.mode"] = mode.value
                if mode.value == "cert":
                    if (cert.value or "") != str(tls.cert_file or ""):
                        updates["server.tls.cert_file"] = cert.value or None
                    if (key.value or "") != str(tls.key_file or ""):
                        updates["server.tls.key_file"] = key.value or None
                await setup._save(ctx, updates, result, draw)

            ui.button("Save", icon="save", on_click=save).props("color=primary dense")

    draw()


def ttl_card(ctx: GuiContext) -> None:
    """How long idle models stay in GPU memory (JIT loading and unloading)."""
    from studioforge.gui.tabs import setup

    with ui.card().classes("w-full"), panel_guard("Idle unloading"):
        ui.label("Load on demand, unload when idle").classes("text-lg font-medium")
        ui.label(
            "A model is loaded the first time something asks for it and unloaded after it has been "
            "idle this long. 0 keeps it loaded."
        ).classes("text-xs opacity-70")
        fields = setup._Fields(ctx, setup._payload(ctx))
        fields.row("models.default_ttl_s", label="Unload after idle (seconds)")
        fields.row("models.auto_unload_idle_s", label="Unload automatically when idle (seconds)")
        tiers = setup._ttl_by_priority_rows(ctx)
        result = ui.label("").classes("text-xs")

        async def save() -> None:
            updates = fields.updates()
            try:
                wanted = st.ttl_by_priority_map({tier: w.value for tier, w in tiers.items()})
            except ValueError as exc:
                notify_error(exc, what="idle unload per load priority")
                result.set_text("not saved")
                return
            current = {int(k): int(v) for k, v in ctx.config.models.ttl_by_priority.items()}
            if wanted != current:
                updates["models.ttl_by_priority"] = wanted
            await setup._save(ctx, updates, result, None)

        ui.button("Save", icon="save", on_click=save).props("color=primary dense")


def cors_card(ctx: GuiContext) -> None:
    """Which websites may call the API from a browser."""
    from studioforge.gui.tabs import setup

    with ui.card().classes("w-full"), panel_guard("Browser access"):
        ui.label("Browser access (CORS)").classes("text-lg font-medium")
        ui.label(
            "Websites listed here may call the API from a visitor's browser. Leave it empty "
            "unless a web app you run needs it; programs and scripts are not affected. A change "
            "takes effect after a restart."
        ).classes("text-xs opacity-70")
        fields = setup._Fields(ctx, setup._payload(ctx))
        fields.row("server.cors_origins", label="Allowed websites (comma separated)")
        fields.row("server.cors_allow_credentials", label="Let those websites send credentials")
        setup._save_button(ctx, fields, None)


def render(ctx: GuiContext) -> None:
    from studioforge.gui.tabs import setup

    with ui.column().classes("w-full gap-4 p-2"):
        with panel_guard("The start-up warning"):
            _warning_card(ctx)
        with panel_guard("The Base URL card"):
            base_url_card(ctx)
        listening_card(ctx)
        tls_card(ctx)
        ttl_card(ctx)
        cors_card(ctx)
        setup.serving_cards(ctx)
