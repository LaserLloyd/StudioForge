"""Server tab: configuration, health and the llama.cpp engine.

Two things here are easy to get wrong and expensive to get wrong.

**Restart-required keys.** Some config changes take effect immediately because
the config object is shared by reference; others (ports, bind addresses, the
data dir) cannot. ``RESTART_REQUIRED_KEYS`` drives a visible marker so a user
does not sit waiting for a port change that will never happen.

**Masked secrets.** The API key and HF token are displayed masked. Sending the
mask back would overwrite the real secret with the placeholder and lock every
client out, so a secret field is only included in the update when
:func:`studioforge.gui.state.masked_secret_changed` says it really changed.

**Shared with the Setup tab.** The engine actions (:func:`install_engine`,
:func:`smoke_engine`, :func:`activate_engine`) and the OpenClaw snippet
builders (:func:`openclaw_payload`, :func:`snippets`, :func:`snippet_block`)
are public because :mod:`studioforge.gui.tabs.setup` renders the same controls.
There is one implementation of each; two panels calling it is the point.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from studioforge import __version__
from studioforge.gui import state as st
from studioforge.gui.tabs import (
    GuiContext,
    apply_config_updates,
    busy,
    notify_error,
    panel_guard,
    run_blocking,
)


def render(ctx: GuiContext) -> None:
    with ui.column().classes("w-full gap-4 p-2"):
        _setup_panel(ctx)
        _protocol_panel(ctx)
        _openclaw_panel(ctx)
        _health_panel(ctx)
        _config_panel(ctx)
        _engine_panel(ctx)
        _capabilities_panel(ctx)


# ---------------------------------------------------------------------------
# First-run readiness
# ---------------------------------------------------------------------------


def _setup_panel(ctx: GuiContext) -> None:
    """A compact readiness strip, so this tab never looks fine while it is not.

    The **full** checklist -- nine items, including the optional ones, with the
    button that fixes each -- lives on the Setup tab; this is the four-item
    version that stops someone editing ports on a box with no engine installed.
    Both read :func:`studioforge.gui.state.setup_status` and
    :func:`~studioforge.gui.state.first_run_checks` respectively; neither
    decides anything itself.
    """
    card = ui.card().classes("w-full")

    def refresh() -> None:
        card.clear()
        with card, panel_guard("Setup"):
            items = _setup_items(ctx)
            ready = st.setup_is_ready(items)
            with ui.row().classes("items-center gap-2"):
                ui.icon(
                    "check_circle" if ready else "build_circle",
                    color="positive" if ready else "warning",
                )
                ui.label("Ready to serve" if ready else "Setup").classes("font-medium")
                ui.space()
                ui.label("the Setup tab has the full checklist and every setting").classes(
                    "text-xs opacity-60"
                )
            for item in items:
                with ui.row().classes("items-center gap-2 w-full no-wrap"):
                    ui.icon(item.icon, color=item.colour, size="1rem")
                    ui.label(f"{item.name}: {item.detail}").classes(
                        "text-xs font-mono grow truncate"
                    )
                    if item.action == "scan":
                        ui.button(
                            "Scan models",
                            icon="search",
                            on_click=lambda: _scan_models(ctx, refresh),
                        ).props("outline dense")
                    elif item.action == "install-engine":
                        ui.button(
                            f"Install engine {ctx.config.engine.pinned_tag}",
                            icon="download",
                            on_click=lambda: install_engine(
                                ctx, ctx.config.engine.pinned_tag, refresh
                            ),
                        ).props("outline dense")

    refresh()


def _setup_items(ctx: GuiContext) -> list[st.SetupItem]:
    model_count = 0
    gpu_count = 0
    engine_tag: str | None = None
    try:
        model_count = len(ctx.registry.all()) if ctx.registry is not None else 0
    except Exception:  # noqa: BLE001
        model_count = 0
    try:
        gpu_count = len(ctx.probe.list_gpus()) if ctx.probe is not None else 0
    except Exception:  # noqa: BLE001
        gpu_count = 0
    try:
        active = ctx.engine_manager.active() if ctx.engine_manager is not None else None
        engine_tag = active.tag if active is not None else None
    except Exception:  # noqa: BLE001
        engine_tag = None
    return st.setup_status(
        model_dir=ctx.config.models.dir,
        model_count=model_count,
        gpu_count=gpu_count,
        engine_tag=engine_tag,
        pinned_tag=ctx.config.engine.pinned_tag,
    )


async def _scan_models(ctx: GuiContext, refresh: Any) -> None:
    with busy(message="Scanning the model library…"):
        try:
            result = await run_blocking(ctx.registry.scan)
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="scan")
            return
    ui.notify(
        f"scan: {len(result.added)} added, {result.unchanged} unchanged, "
        f"{len(result.errors)} errors",
        type="positive",
    )
    refresh()


# ---------------------------------------------------------------------------
# HuggingFace download button (URL scheme handler)
# ---------------------------------------------------------------------------


def _protocol_panel(ctx: GuiContext) -> None:
    """Register the URL schemes behind HuggingFace's download button.

    Taking over ``lmstudio://`` changes the behaviour of *another installed
    application*, so it is a separate, explicitly confirmed action rather than
    part of the normal registration -- and the dialog says plainly what it does
    and that it is reversible.
    """
    with ui.expansion("HuggingFace download button", icon="download_for_offline").classes("w-full"):
        ui.label(
            "HuggingFace model pages have a download button that opens a local app through a "
            "URL scheme. Registering StudioForge lets that button open this panel's quant "
            "picker instead of (or as well as) LM Studio."
        ).classes("text-xs opacity-70")
        body = ui.column().classes("w-full gap-1")

        def refresh() -> None:
            body.clear()
            with body, panel_guard("Protocol handler"):
                from studioforge.core import protocol

                info: dict[str, Any] = dict(protocol.status(ctx.config))
                own = info.get("studioforge")
                lm = info.get("lmstudio")
                raw_backup = info.get("backup")
                backup: dict[str, Any] = raw_backup if isinstance(raw_backup, dict) else {}
                taken_over = bool(backup.get("lmstudio_command"))

                ui.label(f"platform: {info.get('platform')}").classes(
                    "text-xs font-mono opacity-60"
                )
                with ui.row().classes("items-center gap-2"):
                    ui.icon(
                        "check_circle" if own else "radio_button_unchecked",
                        color="positive" if own else "grey",
                    )
                    ui.label(f"studioforge:// {'registered' if own else 'not registered'}").classes(
                        "text-xs font-mono"
                    )
                with ui.row().classes("items-center gap-2"):
                    ui.icon(
                        "check_circle" if taken_over else "radio_button_unchecked",
                        color="warning" if taken_over else "grey",
                    )
                    ui.label(
                        "lmstudio:// handled by StudioForge"
                        if taken_over
                        else "lmstudio:// left to LM Studio"
                    ).classes("text-xs font-mono")
                if lm:
                    ui.label(f"current lmstudio:// command: {lm}").classes(
                        "text-xs font-mono opacity-60 whitespace-pre-wrap"
                    )
                if taken_over:
                    ui.label(
                        f"LM Studio's original command is backed up and will be restored: "
                        f"{backup.get('lmstudio_command')}"
                    ).classes("text-xs opacity-60 whitespace-pre-wrap")

                with ui.row().classes("gap-2 flex-wrap"):
                    ui.button(
                        "Register studioforge:// handler",
                        icon="link",
                        on_click=lambda: _register_protocol(ctx, refresh, takeover=False),
                    ).props("outline dense")
                    if not taken_over:
                        ui.button(
                            "Also handle lmstudio:// links…",
                            icon="warning",
                            on_click=lambda: _takeover_dialog(ctx, refresh),
                        ).props("outline dense color=warning")
                    else:
                        ui.button(
                            "Restore LM Studio",
                            icon="undo",
                            on_click=lambda: _restore_protocol(ctx, refresh),
                        ).props("outline dense")

        refresh()


async def _register_protocol(ctx: GuiContext, refresh: Any, *, takeover: bool) -> None:
    from studioforge.core import protocol

    with busy(message="Registering the URL handler…"):
        try:
            raw = await run_blocking(protocol.register, ctx.config, takeover_lmstudio=takeover)
            result: dict[str, Any] = dict(raw)
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="register protocol handler")
            return
    ui.notify(
        "lmstudio:// links now open StudioForge"
        if result.get("lmstudio_taken_over")
        else "studioforge:// registered",
        type="positive",
    )
    refresh()


def _takeover_dialog(ctx: GuiContext, refresh: Any) -> None:
    with ui.dialog() as dialog, ui.card().classes("min-w-[32rem]"):
        ui.label("Take over the LM Studio download button?").classes("font-medium")
        ui.label(
            "HuggingFace's button emits an lmstudio:// link. If StudioForge claims that "
            "scheme, clicking it on huggingface.co will open this panel's quant picker "
            "instead of LM Studio."
        ).classes("text-sm")
        ui.label(
            "This changes the behaviour of another application you have installed. "
            "LM Studio itself keeps working normally — only the browser button changes. "
            "The previous command is backed up first, and 'Restore LM Studio' puts it back."
        ).classes("text-xs opacity-70")
        with ui.row().classes("justify-end w-full gap-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")

            async def confirm() -> None:
                dialog.close()
                await _register_protocol(ctx, refresh, takeover=True)

            ui.button("Take over lmstudio://", on_click=confirm).props("color=warning")
    dialog.open()


async def _restore_protocol(ctx: GuiContext, refresh: Any) -> None:
    from studioforge.core import protocol

    with busy(message="Restoring LM Studio's handler…"):
        try:
            result: dict[str, Any] = dict(await run_blocking(protocol.unregister, ctx.config))
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="restore LM Studio handler")
            return
    restored = result.get("lmstudio_restored")
    ui.notify(
        f"restored LM Studio: {restored}" if restored else "handlers removed",
        type="positive",
        multi_line=True,
    )
    refresh()


# ---------------------------------------------------------------------------
# OpenClaw setup snippets
# ---------------------------------------------------------------------------


def _openclaw_panel(ctx: GuiContext) -> None:
    with ui.expansion("Point OpenClaw at this server", icon="link").classes("w-full"):
        body = ui.column().classes("w-full gap-2")

        async def load() -> None:
            body.clear()
            try:
                payload = await openclaw_payload(ctx)
            except Exception as exc:  # noqa: BLE001
                with body:
                    ui.label(f"could not build the snippets: {exc}").classes(
                        "text-negative text-xs"
                    )
                return
            with body:
                ui.label(
                    "Generated by the same endpoint sfctl and the MCP server use, so these are "
                    "the exact values this instance is serving on."
                ).classes("text-xs opacity-70")
                for title, snippet in snippets(payload):
                    snippet_block(title, snippet)

        ui.timer(0.05, load, once=True)


async def openclaw_payload(ctx: GuiContext) -> dict[str, Any]:
    """Reuse the management route rather than re-deriving the snippets.

    ``GET /api/openclaw-setup`` already composes them and is the single source
    of truth; calling the handler with a shim that exposes ``app.state`` keeps
    the GUI and the CLI from drifting apart, and avoids putting any absolute URL
    in the GUI's own sources.
    """
    import json
    from types import SimpleNamespace

    from studioforge.api.mgmt_routes import openclaw_setup

    shim = SimpleNamespace(app=SimpleNamespace(state=ctx.api_state))
    response = await openclaw_setup(shim)  # type: ignore[arg-type]
    decoded = json.loads(bytes(response.body).decode("utf-8"))
    return decoded if isinstance(decoded, dict) else {}


def snippets(payload: dict[str, Any]) -> list[tuple[str, str]]:
    import json

    out: list[tuple[str, str]] = []
    inference = payload.get("inference") or {}
    if inference:
        out.append(
            (
                "Environment (inference)",
                "\n".join(f"{key}={value}" for key, value in inference.items()),
            )
        )
    mcp = payload.get("mcp")
    if mcp:
        out.append(("MCP server (management)", json.dumps(mcp, indent=2)))
    companion = payload.get("companion_config")
    if companion:
        out.append(
            (
                "Companion config",
                "\n".join(f"{key}: {value}" for key, value in companion.items()),
            )
        )
    return out


def snippet_block(title: str, snippet: str) -> None:
    with ui.column().classes("w-full gap-0"):
        with ui.row().classes("items-center gap-2"):
            ui.label(title).classes("text-xs font-medium")
            ui.button(icon="content_copy", on_click=lambda: copy_text(snippet)).props(
                "flat dense"
            ).tooltip("Copy")
        ui.label(snippet).classes(
            "w-full font-mono text-xs whitespace-pre-wrap bg-black/10 dark:bg-white/5 p-2 rounded"
        )


def copy_text(text: str) -> None:
    ui.clipboard.write(text)
    ui.notify("copied", type="positive")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def _health_panel(ctx: GuiContext) -> None:
    ui.label("Health").classes("text-lg font-medium")
    body = ui.label("").classes("text-xs font-mono opacity-80 whitespace-pre-wrap")

    def refresh() -> None:
        try:
            loaded = ctx.supervisor.list() if ctx.supervisor is not None else []
            started = float(getattr(ctx.api_state, "started_at", 0.0) or 0.0)
            import time as _time

            uptime = st.format_duration(_time.time() - started) if started else st.UNKNOWN
            draining = getattr(ctx.manager, "draining", False)
            lines = [
                f"version {__version__} · uptime {uptime} · draining {draining}",
                f"gateway bound to {ctx.config.server.host}:{ctx.config.server.port}"
                f" · panel on port {ctx.config.gui.port}",
                f"auth {'enabled' if ctx.config.server.api_key else 'disabled (open on the LAN)'}",
                f"config file {ctx.config.config_path}",
                f"loaded: {', '.join(i.model_id for i in loaded) or 'none'}",
            ]
            body.set_text("\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            body.set_text(f"health unavailable: {exc}")

    refresh()
    ui.timer(max(2.0, ctx.refresh_interval), refresh)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _config_payload(ctx: GuiContext) -> dict[str, Any]:
    """Config as the management API renders it, with secrets already masked.

    Shared with the Setup tab so both forms are drawn from the same masked
    values, and so the "did this secret really change" guard always compares
    against the same placeholder shape.
    """
    return st.redacted_config(ctx.config)


def _config_panel(ctx: GuiContext) -> None:
    ui.label("Configuration").classes("text-lg font-medium")
    ui.label("A ↻ marker means the change is saved but only takes effect after a restart.").classes(
        "text-xs opacity-70"
    )

    with panel_guard("Configuration"):
        payload = _config_payload(ctx)
        widgets: dict[str, Any] = {}
        originals: dict[str, Any] = {}

        with ui.grid(columns=2).classes("w-full gap-3"):
            for field in st.CONFIG_FIELDS:
                value = st.config_value(payload, field.key)
                originals[field.key] = value
                with ui.row().classes("items-center gap-2 w-full no-wrap"):
                    widgets[field.key] = _config_widget(field, value)
                    if field.restart_required:
                        ui.icon("restart_alt", color="warning").tooltip(
                            "takes effect after a restart"
                        )

        affinity_table: dict[str, Any] = (payload.get("planner") or {}).get("quant_affinity") or {}
        affinity_widgets: dict[str, Any] = {}
        with ui.column().classes("w-full gap-0"):
            ui.label("Planner quant affinity (hardware-tuned defaults)").classes(
                "text-sm font-medium mt-2"
            )
            for line in st.quant_affinity_summary(affinity_table):
                ui.label(line).classes("text-xs font-mono opacity-70")
            for family, spec in sorted(affinity_table.items()):
                if not isinstance(spec, dict):
                    continue
                with ui.row().classes("items-center gap-2"):
                    ui.label(family).classes("text-xs font-mono w-20")
                    widget = ui.toggle(
                        ["prefer", "require"], value=str(spec.get("mode") or "prefer")
                    ).props("dense")
                    widget.tooltip(
                        "prefer = try the faster GPUs first but use any of them; "
                        "require = refuse to place this quant on an ineligible GPU"
                    )
                    affinity_widgets[f"planner.quant_affinity.{family}.mode"] = widget
            ui.label(st.QUANT_AFFINITY_NOTE).classes("text-xs opacity-60")

        result = ui.label("").classes("text-xs")

        async def save() -> None:
            updates: dict[str, Any] = {}
            for key, widget in affinity_widgets.items():
                family = key.split(".")[2]
                current = (affinity_table.get(family) or {}).get("mode", "prefer")
                if widget.value != current:
                    updates[key] = widget.value
            for field in st.CONFIG_FIELDS:
                raw = widgets[field.key].value
                if field.kind == "secret":
                    # The guard that stops a redacted placeholder from being
                    # written back over the real secret.
                    if not st.masked_secret_changed(originals[field.key], raw):
                        continue
                    updates[field.key] = str(raw).strip()
                    continue
                coerced = st.coerce_config_value(field.kind, raw)
                if coerced != originals[field.key]:
                    updates[field.key] = coerced
            if not updates:
                result.set_text("nothing changed")
                result.classes(replace="text-xs opacity-70")
                return
            try:
                # One implementation of "change a setting", shared with the
                # Setup tab and with PATCH /api/config: it validates through
                # apply_overrides, writes config.yaml, applies live what can be
                # applied live, and reports what still needs a restart.
                payload = await apply_config_updates(ctx, updates)
            except Exception as exc:  # noqa: BLE001
                notify_error(exc, what="save config")
                return
            needs_restart = list(payload.get("restart_required") or [])
            message = st.save_result_text(payload)
            result.set_text(message)
            result.classes(replace="text-xs text-warning" if needs_restart else "text-xs")
            ui.notify(message, type="positive", multi_line=True)

        with ui.row().classes("gap-2"):
            ui.button("Save configuration", icon="save", on_click=save).props("color=primary")
            ui.button(
                "Unload all models",
                icon="layers_clear",
                on_click=lambda: _unload_all(ctx),
            ).props("outline")


def _config_widget(field: st.ConfigField, value: Any) -> Any:
    widget: Any
    if field.kind == "bool":
        widget = ui.checkbox(field.label, value=bool(value))
    elif field.kind in ("int", "float"):
        widget = ui.number(
            field.label, value=value, precision=0 if field.kind == "int" else None
        ).props("dense outlined")
    elif field.kind == "select":
        widget = ui.select(list(field.options), value=value, label=field.label).props(
            "dense outlined"
        )
    elif field.kind == "list":
        text = ", ".join(str(v) for v in (value or []))
        widget = ui.input(field.label, value=text).props("dense outlined")
    elif field.kind == "secret":
        widget = ui.input(field.label, value=str(value or ""), password=True).props(
            "dense outlined"
        )
        widget.tooltip(
            "Shown masked. Leave it as-is to keep the current secret; type a new one to replace it."
        )
    else:
        widget = ui.input(field.label, value="" if value is None else str(value)).props(
            "dense outlined"
        )
    widget.classes("grow")
    if field.help:
        widget.tooltip(field.help)
    return widget


async def _unload_all(ctx: GuiContext) -> None:
    with busy(message="Unloading every model…"):
        try:
            unloaded = await ctx.manager.unload_all()
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="unload all")
            return
    ui.notify(f"unloaded {len(unloaded)} model(s)", type="positive")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _engine_panel(ctx: GuiContext) -> None:
    ui.label("Engine").classes("text-lg font-medium")
    body = ui.column().classes("w-full gap-1")
    releases_box = ui.column().classes("w-full gap-1")

    def refresh() -> None:
        body.clear()
        with body, panel_guard("Engine"):
            manager = ctx.engine_manager
            if manager is None:
                ui.label("engine manager unavailable").classes("text-xs opacity-60")
                return
            active = manager.active()
            if active is None:
                ui.label(
                    f"No engine installed. Pinned tag is {ctx.config.engine.pinned_tag}; "
                    "install it below."
                ).classes("text-sm text-warning")
            else:
                ui.label(f"active: {active.tag} ({active.variant})").classes("text-sm font-medium")
                ui.label(
                    f"{active.version_string or st.UNKNOWN} · smoke tested "
                    f"{active.smoke_tested} · {active.server_binary}"
                ).classes("text-xs font-mono opacity-70")
            for info in manager.installed():
                marker = "★" if info.active else "·"
                with ui.row().classes("items-center gap-2"):
                    ui.label(f"{marker} {info.tag} ({info.variant})").classes(
                        "text-xs font-mono grow"
                    )
                    ui.button("smoke test", on_click=lambda t=info.tag: smoke_engine(ctx, t)).props(
                        "flat dense"
                    )
                    ui.button(
                        "activate + reload",
                        on_click=lambda t=info.tag: activate_engine(ctx, t, refresh),
                    ).props("flat dense")

    refresh()

    with ui.row().classes("items-center gap-2"):
        tag_input = ui.input("tag to install", value=ctx.config.engine.pinned_tag)
        tag_input.props("dense outlined").classes("w-40")
        ui.button(
            "Install",
            icon="download",
            on_click=lambda: install_engine(ctx, tag_input.value, refresh),
        ).props("outline dense")
        ui.button(
            "List releases", icon="list", on_click=lambda: _releases(ctx, releases_box)
        ).props("outline dense")


async def install_engine(ctx: GuiContext, tag: Any, refresh: Any) -> None:
    tag = str(tag or "").strip()
    if not tag:
        ui.notify("enter a tag", type="warning")
        return
    with busy(message=f"Installing engine {tag} (this can take a while)…"):
        try:
            info = await ctx.engine_manager.install(tag)
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="engine install")
            return
    ui.notify(f"installed {info.tag} ({info.variant})", type="positive")
    refresh()


async def smoke_engine(ctx: GuiContext, tag: str) -> None:
    with busy(message=f"Smoke testing {tag}…"):
        try:
            ok, detail = await ctx.engine_manager.smoke_test(tag)
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="smoke test")
            return
    ui.notify(
        f"{tag}: {'PASS' if ok else 'FAIL'}\n{detail[:400]}",
        type="positive" if ok else "negative",
        multi_line=True,
        close_button=True,
    )


async def activate_engine(ctx: GuiContext, tag: str, refresh: Any) -> None:
    """Pin an engine and reload everything that is currently loaded onto it.

    A running child keeps the binary it started with, so switching engines only
    means anything once the models are restarted -- doing it in one action is
    what makes the change observable.
    """
    with busy(message=f"Activating {tag} and reloading models…"):
        try:
            await run_blocking(ctx.engine_manager.set_active, tag)
            ctx.config.engine.pinned_tag = tag
            await run_blocking(ctx.config.save)
            loaded = [i.model_id for i in ctx.supervisor.list()]
            for model_id in loaded:
                await ctx.manager.load(model_id, force=True)
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="activate engine")
            return
    ui.notify(f"{tag} active; reloaded {len(loaded)} model(s)", type="positive")
    refresh()


async def _releases(ctx: GuiContext, box: Any) -> None:
    with busy(message="Fetching llama.cpp releases…"):
        try:
            releases = await ctx.engine_manager.list_releases(20)
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="list releases")
            return
    box.clear()
    with box:
        ui.label("available releases").classes("text-xs font-medium")
        ui.label(", ".join(releases) or "none").classes("text-xs font-mono opacity-70")
        # Saying so beats someone comparing this list against GitHub's front
        # page and concluding the fetch is broken.
        ui.label("prereleases and non-bNNNN tags are hidden: they carry no engine build").classes(
            "text-xs opacity-60"
        )


# ---------------------------------------------------------------------------
# Backend capabilities: "what kinds of model can I actually run?"
# ---------------------------------------------------------------------------


def _capabilities_panel(ctx: GuiContext) -> None:
    """What this backend supports, cross-referenced against *this* library.

    Ordered by what the question actually was. The user's complaint was "I can't
    tell what types of models I can handle", so the panel opens with how many of
    their own models fit where, and the raw reach of the engine (142
    architectures, 39 quantizations) comes second -- a supported-architecture
    list answers a question nobody asked until they already know their 30 GB
    download will load.

    The report is built off the event loop and the update check is a *separate*
    refresh, because the latter talks to GitHub and this panel must paint at
    local speed.
    """
    ui.label("Backend & what you can run").classes("text-lg font-medium")
    body = ui.column().classes("w-full gap-2")
    update_row = ui.row().classes("w-full items-center gap-2")
    report_holder: dict[str, Any] = {"report": None}

    async def load() -> None:
        body.clear()
        with body:
            ui.label("Reading the engine's capabilities…").classes("text-xs opacity-60")
        try:
            report = await run_blocking(_build_capability_report, ctx)
        except Exception as exc:  # noqa: BLE001 - an optional panel, never the page
            body.clear()
            with body:
                ui.label(f"capabilities unavailable: {exc}").classes("text-sm opacity-70")
            return
        report_holder["report"] = report
        body.clear()
        with body, panel_guard("Backend capabilities"):
            _capabilities_body(ctx, report)

    async def check_update() -> None:
        update_row.clear()
        with update_row:
            ui.spinner(size="sm")
            ui.label("checking GitHub for a newer engine…").classes("text-xs opacity-60")
        payload: dict[str, Any]
        try:
            # One authority for "is there a newer engine": it verifies the tag
            # has an asset this driver can run before offering an Install
            # button, which is what the raw release list could not do.
            payload = await ctx.engine_manager.check_update(limit=5)
        except Exception as exc:  # noqa: BLE001
            payload = {"checked": True, "error": str(exc)}
        _paint_update(ctx, update_row, payload)

    _paint_update(ctx, update_row, None)
    ui.timer(0.1, load, once=True)
    with ui.row().classes("items-center gap-2"):
        ui.button("Refresh", icon="refresh", on_click=load).props("outline dense")
        ui.button("Check for engine update", icon="cloud_download", on_click=check_update).props(
            "outline dense"
        )


def _build_capability_report(ctx: GuiContext) -> dict[str, Any]:
    """Same report the ``/api/capabilities`` endpoint and the CLI return.

    Imported lazily and called in-process: there is no HTTP hop to ourselves and
    therefore no self-referential URL, which is what keeps this panel working
    behind any proxy.
    """
    from studioforge.core.capabilities import build_report

    gpus = list(ctx.probe.list_gpus()) if ctx.probe is not None else []
    records = list(ctx.registry.all()) if ctx.registry is not None else []
    report = build_report(
        ctx.config,
        gpus=gpus,
        records=records,
        engine_manager=ctx.engine_manager,
        probe=ctx.probe,
    )
    result: dict[str, Any] = report.to_dict()
    return result


def _paint_update(ctx: GuiContext, row: Any, update: dict[str, Any] | None) -> None:
    row.clear()
    with row:
        ui.label(st.engine_update_line(update)).classes("text-sm")
        if st.engine_update_available(update):
            latest = str((update or {}).get("latest") or "")
            ui.button(
                f"Install {latest}",
                icon="download",
                on_click=lambda tag=latest: install_engine(ctx, tag, lambda: None),
            ).props("outline dense color=primary").tooltip(st.ENGINE_UPDATE_NOTE)
            ui.label(st.ENGINE_UPDATE_NOTE).classes("text-xs opacity-70")


def _capabilities_body(ctx: GuiContext, report: dict[str, Any]) -> None:
    # --- 1. the answer to the actual question ---------------------------
    ui.label(st.sizing_headline(report)).classes("text-base font-medium")
    note = st.sizing_note(report)
    if note:
        # Verbatim: it says this is a weights-only estimate and points at the
        # per-model fit check. Paraphrasing it would make it sound exact.
        ui.label(note).classes("text-xs opacity-70")
    too_big = st.too_big_models(report)
    if too_big:
        ui.label("too big for this hardware: " + ", ".join(too_big)).classes(
            "text-xs text-warning font-mono"
        )

    # --- 2. the models that genuinely cannot load ------------------------
    warning = st.unsupported_warning(report)
    if warning:
        with ui.card().classes("w-full bg-orange-50 dark:bg-orange-950"):
            ui.label(warning).classes("text-warning text-sm font-medium")
            for model_id, architecture in st.unsupported_models(report):
                ui.label(f"{model_id} — {architecture}").classes("text-xs font-mono")

    # --- 3. the engine ---------------------------------------------------
    for line in st.engine_summary_lines(report):
        ui.label(line).classes("text-xs font-mono opacity-80")
    caveat = st.capability_source_caveat(report)
    if caveat:
        # Provenance matters: a bundled snapshot can disagree with the engine
        # that is actually installed, and hiding that turns this panel from an
        # answer into a guess.
        ui.label(caveat).classes("text-xs text-warning")

    # --- 4. the hardware -------------------------------------------------
    for line in st.hardware_summary_lines(report):
        ui.label(line).classes("text-xs font-mono opacity-70")

    # --- 5. your library, cross-referenced -------------------------------
    ui.label("Your library").classes("text-sm font-medium mt-2")
    _chip_row("architectures", st.architecture_chips(report))
    _chip_row("quantizations", st.quant_chips(report))
    for name, text in st.quant_hardware_notes(report):
        ui.label(f"{name}: {text}").classes("text-xs opacity-70")
    _chip_row("capabilities", st.library_capability_chips(report))

    # --- 6. what the backend can do at all -------------------------------
    with ui.expansion("Feature support", icon="checklist").classes("w-full"):
        for name, text in st.feature_rows(report):
            with ui.row().classes("items-start gap-2 no-wrap"):
                ui.label(name).classes("text-xs font-mono w-32 shrink-0")
                ui.label(text).classes("text-xs opacity-80")

    # --- 7. the long tail, behind a door ---------------------------------
    architectures = st.supported_architectures(report)
    quants = st.supported_quant_types(report)
    with ui.expansion(
        f"Supported architectures ({len(architectures)}) and quantizations ({len(quants)})",
        icon="list",
    ).classes("w-full"):
        needle = ui.input(placeholder="filter…").props("dense outlined clearable").classes("w-64")
        listing = ui.label("").classes("text-xs font-mono opacity-80 whitespace-pre-wrap")

        def repaint() -> None:
            names = st.filter_architectures(architectures, needle.value)
            listing.set_text(
                ", ".join(names) if names else f"no architecture matches {needle.value!r}"
            )

        needle.on_value_change(lambda _: repaint())
        repaint()
        ui.label("quantizations: " + ", ".join(sorted(quants))).classes(
            "text-xs font-mono opacity-70 whitespace-pre-wrap"
        )


def _chip_row(title: str, chips: list[st.Chip]) -> None:
    if not chips:
        return
    with ui.row().classes("w-full items-center gap-1 flex-wrap"):
        ui.label(title).classes("text-xs opacity-60 w-28 shrink-0")
        for chip in chips:
            badge = ui.badge(chip.text, color="secondary").classes("text-xs")
            if chip.tooltip:
                badge.tooltip(f"{chip.count} {chip.tooltip}")
