"""Setup tab: everything a new install has to be told, in one place.

The problem it solves is not that the settings were missing -- they were all in
``config.yaml`` -- but that reaching them meant knowing which of 81 keys to
edit, in a file, in a directory the user had to find first. The Server tab
exposed twelve of them. This tab exposes all of them, grouped by the decision
being made rather than by the pydantic model, with a checklist at the top that
answers the only question a fresh checkout actually has: *what is still stopping
this from serving a model?* (DECISIONS.md D26.)

Three rules hold everywhere in here:

**One implementation of "change a setting".** Every field, including the
generated Advanced ones, saves through
:func:`studioforge.gui.tabs.apply_config_updates`, which calls the same
management route ``PATCH /api/config`` uses. Validation, atomic write and
restart-required flagging therefore cannot drift between surfaces.

**Secrets are masked and only sent when they really changed.** The API key, the
HF token and the MCP PIN render behind a reveal button, and
:func:`studioforge.gui.state.masked_secret_changed` decides whether a field
holds a new secret or the placeholder it was drawn with. Posting the
placeholder back would replace a working credential with nine literal
characters.

**The tab is a renderer.** Every rule -- what counts as ready, what is required
versus optional, which keys get a generated widget -- is a pure function in
:mod:`studioforge.gui.state` with a unit test.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from nicegui import ui

from studioforge.gui import state as st
from studioforge.gui.tabs import (
    GuiContext,
    apply_config_updates,
    busy,
    notify_error,
    panel_guard,
    run_blocking,
)
from studioforge.gui.tabs.server import (
    activate_engine,
    copy_text,
    install_engine,
    openclaw_payload,
    smoke_engine,
    snippets,
)

#: Counting every ``*.gguf`` under a multi-terabyte library is slow enough to be
#: felt, and the checklist only needs "some / none / lots". Stop here and say
#: the count is a floor.
GGUF_COUNT_LIMIT = 2000

#: Keys rendered with a purpose-built control in one of the sections below.
#: They are excluded from the generated Advanced section so every key has
#: exactly one control on this tab.
COVERED_KEYS: tuple[str, ...] = (
    "models.dir",
    "models.extra_dirs",
    "models.default_ctx",
    "models.target_ctx",
    "models.thinking_default_ctx",
    "models.default_parallel",
    "models.ctx_per_slot_default",
    "models.default_kv_cache_type",
    "models.default_ttl_s",
    "models.auto_load_pinned",
    "models.default_model",
    "models.preload_default_model",
    "planner.headroom_fraction",
    "planner.preference",
    "planner.compute_overhead_fraction",
    "planner.excluded_devices",
    "engine.pinned_tag",
    "engine.cuda_variant",
    "engine.allow_source_build",
    "engine.keep_versions",
    "server.host",
    "server.port",
    "server.api_key",
    "gui.port",
    "watchdog.port",
    "mcp.pin",
    "mcp.pin_required",
    "hf.token",
    "hf.max_concurrent_downloads",
    "hf.chunk_bytes",
    "logging.level",
)


def render(ctx: GuiContext) -> None:
    with ui.column().classes("w-full gap-4 p-2"):
        _checklist_card(ctx)
        _library_card(ctx)
        _gpu_card(ctx)
        _engine_card(ctx)
        _network_card(ctx)
        _downloads_card(ctx)
        _startup_card(ctx)
        _advanced_card(ctx)


# ---------------------------------------------------------------------------
# Shared plumbing for the section forms
# ---------------------------------------------------------------------------


def _specs() -> dict[str, st.ConfigFieldSpec]:
    """Field metadata for every config key, by dotted key.

    Regenerated per render rather than cached at import: it is cheap, and a
    module-level cache would go stale the moment the config model is reloaded
    in a test.
    """
    return st.spec_by_key(st.config_field_specs())


def _payload(ctx: GuiContext) -> dict[str, Any]:
    """The config as the management API renders it, with every secret masked."""
    return st.redacted_config(ctx.config)


class _Fields:
    """One section's worth of inputs, saved together in a single update.

    Field-at-a-time saving would mean a write to ``config.yaml`` per keystroke
    and a restart notice per key; a section-at-a-time Save button matches how
    the settings are actually reasoned about ("point it at my library, then set
    the context").
    """

    def __init__(self, ctx: GuiContext, payload: dict[str, Any]) -> None:
        self.ctx = ctx
        self.payload = payload
        self.specs: dict[str, st.ConfigFieldSpec] = _specs()
        self.used: list[st.ConfigFieldSpec] = []
        self.widgets: dict[str, Any] = {}

    def row(self, key: str, *, label: str | None = None, options: list[str] | None = None) -> Any:
        """Render one config key: input, the key itself, and its explanation."""
        spec = self.specs.get(key)
        if spec is None:  # pragma: no cover - a key that no longer exists
            return None
        self.used.append(spec)
        value = st.spec_display_value(self.payload, spec)
        with ui.column().classes("w-full gap-0"):
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                widget = _widget(spec, value, label=label, options=options)
                self.widgets[key] = widget
                if spec.restart_required:
                    ui.icon("restart_alt", color="warning").tooltip(
                        "saved immediately, but only takes effect after a restart"
                    )
            ui.label(f"{spec.key} · {spec.summary}").classes("text-xs opacity-60")
        return widget

    def updates(self) -> dict[str, Any]:
        values = {key: widget.value for key, widget in self.widgets.items()}
        return st.config_updates_from_form(self.used, self.payload, values)


def _widget(
    spec: st.ConfigFieldSpec,
    value: Any,
    *,
    label: str | None = None,
    options: list[str] | None = None,
) -> Any:
    caption = label or spec.label
    widget: Any
    if options is not None:
        widget = ui.select(options, value=value, label=caption).props("dense outlined")
    elif spec.kind == "bool":
        widget = ui.checkbox(caption, value=bool(value))
    elif spec.kind in ("int", "float"):
        widget = ui.number(caption, value=value, precision=0 if spec.kind == "int" else None).props(
            "dense outlined"
        )
    elif spec.kind == "select":
        widget = ui.select(list(spec.options), value=value, label=caption).props("dense outlined")
    elif spec.kind == "secret":
        # Masked by default, revealable by the person sitting in front of it --
        # and what is drawn is the redacted placeholder, never the secret, so
        # revealing it shows "abcd...yz" until something new is typed.
        widget = ui.input(
            caption, value=str(value or ""), password=True, password_toggle_button=True
        ).props("dense outlined")
        widget.tooltip(
            "Shown masked. Leave it as-is to keep the current value; type a new one to replace it."
        )
    else:
        widget = ui.input(caption, value="" if value is None else str(value)).props(
            "dense outlined"
        )
    widget.classes("grow")
    return widget


async def _save(ctx: GuiContext, updates: dict[str, Any], result: Any, refresh: Any = None) -> None:
    """Persist a section's changes through the one shared code path."""
    if not updates:
        result.set_text("nothing changed")
        result.classes(replace="text-xs opacity-70")
        return
    try:
        payload = await apply_config_updates(ctx, updates)
    except Exception as exc:  # noqa: BLE001
        notify_error(exc, what="save settings")
        return
    message = st.save_result_text(payload)
    needs_restart = bool(payload.get("restart_required"))
    result.set_text(message)
    result.classes(replace="text-xs text-warning" if needs_restart else "text-xs")
    ui.notify(message, type="positive", multi_line=True)
    if refresh is not None:
        refresh()


def _save_button(ctx: GuiContext, fields: _Fields, refresh: Any = None) -> Any:
    result = ui.label("").classes("text-xs")

    async def save() -> None:
        await _save(ctx, fields.updates(), result, refresh)

    ui.button("Save", icon="save", on_click=save).props("color=primary dense")
    return result


def _secret_line(label: str, value: str | None, *, unset_note: str, hint: str = "") -> None:
    """``currently ***`` with a Show/Hide eye and a Copy button.

    The input above it draws only the redacted placeholder (so a stray save can
    never post a fingerprint back as the value), which left the operator with
    no way to *read* the PIN the watchdog and every MCP client demand -- the
    tray has no console for the startup banner, and the only other place was a
    collapsed expansion behind a switch. Whoever can use this panel can already
    rotate the PIN and edit every setting, so showing it on request gives away
    nothing the panel did not already give away; it just stops the hunt.
    """
    masked = st.secret_state_text(value, unset_note=unset_note)
    with ui.row().classes("items-center gap-1 no-wrap"):
        shown = ui.label(f"currently {masked}").classes("text-xs font-mono opacity-70")
        if value:
            state = {"revealed": False}

            def toggle() -> None:
                state["revealed"] = not state["revealed"]
                shown.set_text(f"currently {value if state['revealed'] else masked}")
                eye.props(f"icon={'visibility_off' if state['revealed'] else 'visibility'}")

            eye = (
                ui.button(icon="visibility", on_click=toggle)
                .props("flat dense round size=sm")
                .tooltip(f"Show / hide the {label}")
            )
            ui.button(icon="content_copy", on_click=lambda: copy_text(value)).props(
                "flat dense round size=sm"
            ).tooltip(f"Copy the {label}")
    if hint:
        ui.label(hint).classes("text-xs opacity-70")


def _section(title: str, subtitle: str = "") -> None:
    ui.label(title).classes("text-lg font-medium")
    if subtitle:
        ui.label(subtitle).classes("text-xs opacity-70")


# ---------------------------------------------------------------------------
# 1. First-run checklist
# ---------------------------------------------------------------------------


def _checklist_card(ctx: GuiContext) -> None:
    """What is still stopping this box from serving, and the button that fixes it.

    Deliberately not on a timer. Every item costs a syscall or two (a directory
    walk, a socket bind, a startup-folder stat) and none of them change on their
    own, so it is computed on the first paint and after anything that could have
    changed the answer.
    """
    card = ui.card().classes("w-full")

    def refresh() -> None:
        card.clear()
        with card, panel_guard("First-run checklist"):
            checks = _collect_checks(ctx)
            ready = st.checklist_is_ready(checks)
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                ui.icon(
                    "check_circle" if ready else "build_circle",
                    color="positive" if ready else "warning",
                    size="1.5rem",
                )
                ui.label("Ready to serve" if ready else "Setup").classes("text-lg font-medium")
                ui.space()
                ui.button("Re-check", icon="refresh", on_click=refresh).props("outline dense")
            ui.label(st.checklist_headline(checks)).classes("text-sm opacity-80")
            for check in checks:
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    ui.icon(check.icon, color=check.colour, size="1rem").tooltip(check.status_text)
                    ui.label(f"{check.name}: {check.detail}").classes(
                        "text-xs font-mono grow truncate"
                    ).tooltip(check.help)
                    if not check.ok and check.action:
                        ui.button(
                            check.action_label,
                            on_click=lambda action=check.action: _run_action(ctx, action, refresh),
                        ).props("outline dense")

    refresh()


def _collect_checks(ctx: GuiContext) -> list[st.SetupCheck]:
    """Gather the live facts the checklist is a pure function of."""
    config = ctx.config
    models_dir = config.models.dir
    exists, gguf_count = _library_facts(models_dir)
    gpus = _safe(lambda: list(ctx.probe.list_gpus()) if ctx.probe is not None else [], [])
    engine = _safe(
        lambda: ctx.engine_manager.active() if ctx.engine_manager is not None else None, None
    )
    reachable, port_detail = _port_facts(config)
    autostart_enabled, autostart_mechanism = _autostart_facts(config)
    return st.first_run_checks(
        data_dir=config.data_dir,
        data_dir_writable=_is_writable(config.data_dir),
        models_dir=models_dir,
        models_dir_exists=exists,
        gguf_count=gguf_count,
        indexed_count=_safe(lambda: len(ctx.registry.all()) if ctx.registry is not None else 0, 0),
        gpu_count=len(gpus),
        driver_version=_safe(
            lambda: ctx.probe.driver_version() if ctx.probe is not None else None, None
        ),
        cuda_driver=_safe(
            lambda: ctx.probe.cuda_driver_version() if ctx.probe is not None else None, None
        ),
        excluded_devices=config.planner.excluded_devices,
        engine_tag=getattr(engine, "tag", None),
        engine_smoke_tested=bool(getattr(engine, "smoke_tested", False)),
        pinned_tag=config.engine.pinned_tag,
        api_port=config.server.port,
        api_reachable=reachable,
        api_port_detail=port_detail,
        mcp_pin_set=bool(config.mcp.pin),
        mcp_pin_required=bool(config.mcp.pin_required),
        hf_token_set=bool(config.hf.token),
        autostart_enabled=autostart_enabled,
        autostart_mechanism=autostart_mechanism,
        bind_host=config.server.host,
        api_key_set=bool(config.server.api_key),
        gui_host=config.gui.host if config.gui.enabled else None,
        watchdog_host=config.watchdog.host if config.watchdog.enabled else None,
        boot_phase=_boot_phase(ctx),
    )


def _boot_phase(ctx: GuiContext) -> str | None:
    """The boot's current phase while it is still running, else None (D33)."""
    boot = getattr(ctx.api_state, "boot", None)
    if boot is None or getattr(boot, "ready", True):
        return None
    return str(getattr(boot, "phase", "") or "") or None


def _safe(fn: Any, fallback: Any) -> Any:
    """A missing subsystem is a display gap on this tab, never an error card."""
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return fallback


def _is_writable(path: Any) -> bool:
    """Can we actually write here? Proved by writing, not by ``os.access``.

    ``os.access(W_OK)`` lies on Windows -- it reports the read-only *attribute*
    and knows nothing about ACLs -- and the data dir is where config.yaml, the
    registry and every log go, so a wrong answer here is the least useful
    possible place for one.
    """
    try:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".sf-write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def _library_facts(models_dir: Any) -> tuple[bool, int]:
    """``(exists, gguf_count)`` for the library, with the walk capped."""
    if not models_dir:
        return (False, 0)
    try:
        root = Path(models_dir)
        if not root.is_dir():
            return (False, 0)
        count = 0
        for _ in root.rglob("*.gguf"):
            count += 1
            if count >= GGUF_COUNT_LIMIT:
                break
    except OSError:
        return (False, 0)
    return (True, count)


def _port_facts(config: Any) -> tuple[bool, str]:
    """Is the gateway port ours, someone else's, or nobody's?"""
    from studioforge.core import ports

    port = config.server.port
    host = config.server.host
    probe_host = "0.0.0.0" if host in ("", "::") else host
    try:
        free = ports.port_is_bindable(port, probe_host)
    except Exception:  # noqa: BLE001
        return (True, f"port {port} — could not be probed")
    if free:
        return (
            False,
            f"port {port} on {probe_host} is free — nothing is listening on the gateway port",
        )
    holder = _safe(lambda: ports.find_port_holder(port), None)
    if holder is not None and holder.pid == os.getpid():
        return (True, f"listening on {probe_host}:{port} (this process)")
    if holder is not None and holder.is_studioforge:
        return (
            False,
            f"port {port} is held by another StudioForge instance (pid {holder.pid})",
        )
    if holder is not None and holder.is_lmstudio:
        return (False, f"port {port} is held by LM Studio (pid {holder.pid}) — quit it or move")
    return (True, f"port {port} is in use, which is what a running gateway looks like")


def _autostart_facts(config: Any) -> tuple[bool, str]:
    try:
        from studioforge.core import autostart

        status = autostart.status(config)
    except Exception:  # noqa: BLE001
        return (False, "")
    return (bool(status.enabled), status.mechanism)


async def _run_action(ctx: GuiContext, action: str, refresh: Any) -> None:
    """Perform the one fix an unmet checklist item names."""
    if action == "scan":
        await _rescan(ctx, refresh)
    elif action == "detect-library":
        await _detect_library(ctx, refresh)
    elif action == "install-engine":
        await install_engine(ctx, ctx.config.engine.pinned_tag, refresh)
    elif action == "generate-pin":
        await _generate_pin(ctx, refresh)
    elif action == "set-api-key":
        await _generate_api_key(ctx, refresh)
    elif action == "enable-autostart":
        await _set_autostart(ctx, True, refresh)
    elif action == "open-data-dir":
        _open_path(ctx.config.data_dir)
    elif action == "reprobe":
        ui.notify("re-probed", type="positive")
        refresh()


# ---------------------------------------------------------------------------
# 2. Model library
# ---------------------------------------------------------------------------


def _library_card(ctx: GuiContext) -> None:
    card = ui.card().classes("w-full")

    def refresh() -> None:
        card.clear()
        with card, panel_guard("Model library"):
            _library_body(ctx, refresh)

    refresh()


def _library_body(ctx: GuiContext, refresh: Any) -> None:
    _section(
        "Model library",
        "Where your GGUF files are. They are indexed in place — nothing is ever copied or moved.",
    )
    fields = _Fields(ctx, _payload(ctx))
    fields.row("models.dir", label="Model directory")
    fields.row("models.extra_dirs", label="Extra model directories (comma separated)")

    exists, gguf_count = _library_facts(ctx.config.models.dir)
    disk = _safe(
        lambda: _disk_report(ctx),
        None,
    )
    ui.label(
        st.models_dir_status_line(
            ctx.config.models.dir, exists=exists, gguf_count=gguf_count, disk=disk
        )
    ).classes("text-xs font-mono opacity-80")

    detection = ui.label("").classes("text-xs opacity-70 whitespace-pre-wrap")
    with ui.row().classes("gap-2 flex-wrap"):
        ui.button(
            "Detect LM Studio library",
            icon="search",
            on_click=lambda: _detect_library(ctx, refresh),
        ).props("outline dense").tooltip(
            "Probes the downloadsFolder recorded in ~/.lmstudio/settings.json first, then "
            "LM Studio's default locations."
        )
        ui.button("Rescan now", icon="refresh", on_click=lambda: _rescan(ctx, refresh)).props(
            "outline dense"
        )
        ui.button(
            "Show what was probed",
            icon="list",
            on_click=lambda: _show_candidates(ctx, detection),
        ).props("flat dense")

    ui.separator()
    _section("Defaults for every load", "")
    fields.row("models.default_ctx", label="Context floor")
    fields.row("models.target_ctx", label="Context target")
    fields.row("models.thinking_default_ctx", label="Thinking-model floor")
    ui.label(
        "Context is a ladder, not a constant (DECISIONS D14): a load aims for the target and "
        "the planner halves down from it to the largest window that actually fits in VRAM, "
        "never below the floor. An explicit per-model or per-request size is always honoured."
    ).classes("text-xs opacity-70")
    fields.row("models.default_parallel", label="Conversation slots")
    fields.row("models.ctx_per_slot_default", label="Context per slot (estimator)")
    ui.label(
        "'auto' sizes the slot count per model and per placement (D17). llama-server's "
        "--ctx-size is the TOTAL budget shared by the slots, so StudioForge multiplies the "
        "per-conversation window by the slot count when it launches."
    ).classes("text-xs opacity-70")
    fields.row("models.default_kv_cache_type", label="KV cache type")
    fields.row("models.default_ttl_s", label="Idle unload after (s)")
    fields.row("models.auto_load_pinned", label="Load pinned models at startup")

    model_ids = _safe(lambda: sorted(r.id for r in ctx.registry.all()), [])
    if model_ids:
        fields.row("models.default_model", label="Default model", options=["", *model_ids])
    else:
        fields.row("models.default_model", label="Default model")
    fields.row("models.preload_default_model", label="Preload that default at startup")

    with ui.row().classes("items-center gap-3"):
        _save_button(ctx, fields, refresh)


def _disk_report(ctx: GuiContext) -> dict[str, Any] | None:
    from studioforge.core import diskspace

    if not ctx.config.models.dir:
        return None
    queued = 0
    downloader = ctx.downloader
    if downloader is not None:
        queued = int(_safe(downloader.queued_remaining_bytes, 0) or 0)
    return diskspace.disk_report(ctx.config.models.dir, queued)


async def _rescan(ctx: GuiContext, refresh: Any) -> None:
    if ctx.registry is None:
        ui.notify("no registry on this instance", type="warning")
        return
    with busy(message="Scanning the model library…"):
        try:
            result = await run_blocking(ctx.registry.scan)
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="scan")
            return
    ui.notify(
        f"scan: {len(result.added)} added, {result.unchanged} unchanged, "
        f"{len(result.errors)} error(s)",
        type="positive",
    )
    refresh()


async def _detect_library(ctx: GuiContext, refresh: Any) -> None:
    from studioforge.config import detect_model_dir

    with busy(message="Looking for an LM Studio library…"):
        detected = await run_blocking(detect_model_dir)
    note = st.lmstudio_detection_note(detected, ctx.config.models.dir)
    if not detected or str(detected) == str(ctx.config.models.dir or ""):
        ui.notify(note, type="info" if detected else "warning", multi_line=True)
        return
    try:
        payload = await apply_config_updates(ctx, {"models.dir": str(detected)})
    except Exception as exc:  # noqa: BLE001
        notify_error(exc, what="set models.dir")
        return
    ui.notify(f"{note} ({st.save_result_text(payload)})", type="positive", multi_line=True)
    refresh()


def _show_candidates(ctx: GuiContext, target: Any) -> None:
    from studioforge.config import lmstudio_model_dir_candidates

    rows: list[dict[str, Any]] = []
    for candidate in lmstudio_model_dir_candidates():
        exists, count = _library_facts(candidate)
        rows.append({"path": str(candidate), "exists": exists, "gguf_count": count})
    lines = st.lmstudio_candidate_lines(rows)
    target.set_text("\n".join(lines) if lines else "nothing was probed")


# ---------------------------------------------------------------------------
# 3. GPUs and memory
# ---------------------------------------------------------------------------


def _gpu_card(ctx: GuiContext) -> None:
    card = ui.card().classes("w-full")

    def refresh() -> None:
        card.clear()
        with card, panel_guard("GPUs & memory"):
            _gpu_body(ctx, refresh)

    refresh()


def _gpu_body(ctx: GuiContext, refresh: Any) -> None:
    _section("GPUs & memory", "What the planner is allowed to use, and what it must leave alone.")
    gpus = _safe(lambda: list(ctx.probe.list_gpus()) if ctx.probe is not None else [], [])
    holders = _safe(lambda: _holder_dicts(ctx), [])
    rows = st.gpu_setup_rows(
        gpus,
        excluded_devices=ctx.config.planner.excluded_devices,
        reserved_mb=ctx.config.planner.reserved_mb,
        holders=holders,
    )

    with ui.row().classes("items-center gap-2"):
        ui.label(
            f"driver {_safe(lambda: ctx.probe.driver_version(), None) or st.UNKNOWN}"
            f" · probe backend {getattr(ctx.probe, 'backend', st.UNKNOWN)}"
        ).classes("text-xs font-mono opacity-70")
        ui.space()
        ui.button("Re-probe", icon="refresh", on_click=refresh).props("outline dense")

    if not rows:
        ui.label(
            "No GPUs reported. StudioForge is GPU-only, so nothing can load until NVML sees a "
            "card — check the driver, and that this process can reach it."
        ).classes("text-sm text-warning")

    excluded_widgets: dict[int, Any] = {}
    reserved_widgets: dict[int, Any] = {}
    for row in rows:
        with ui.card().classes("w-full"), ui.column().classes("w-full gap-1"):
            ui.label(row.summary()).classes("text-xs font-mono")
            with ui.row().classes("items-center gap-4 flex-wrap"):
                excluded_widgets[row.index] = ui.checkbox(
                    "never place models here", value=row.excluded
                ).tooltip(
                    "planner.excluded_devices — the planner will not use this GPU at all. A "
                    "per-model device override still wins, with a warning."
                )
                reserved_widgets[row.index] = (
                    ui.number("reserve (MiB)", value=row.reserved_mb, precision=0)
                    .props("dense outlined")
                    .classes("w-40")
                    .tooltip(
                        "planner.reserved_mb — MiB held back on this card for a neighbour "
                        "(ComfyUI, a training job). Applies even to a forced placement."
                    )
                )

    ui.label(
        st.device_policy_note(ctx.config.planner.excluded_devices, ctx.config.planner.reserved_mb)
    ).classes("text-xs opacity-70")
    ui.label(st.DEVICE_RECOGNITION_NOTE).classes("text-xs opacity-70")

    fields = _Fields(ctx, _payload(ctx))
    fields.row("planner.headroom_fraction", label="VRAM headroom (fraction of every GPU)")
    fields.row("planner.preference", label="Optimise loads for")
    ui.label(st.PLANNER_PREFERENCE_NOTE).classes("text-xs opacity-70")
    result = ui.label("").classes("text-xs")

    async def save() -> None:
        updates = fields.updates()
        excluded = st.excluded_devices_list(
            {index: widget.value for index, widget in excluded_widgets.items()}
        )
        reserved = st.reserved_mb_map(
            {index: widget.value for index, widget in reserved_widgets.items()}
        )
        if excluded != sorted(ctx.config.planner.excluded_devices):
            updates["planner.excluded_devices"] = excluded
        if reserved != {int(k): int(v) for k, v in ctx.config.planner.reserved_mb.items()}:
            updates["planner.reserved_mb"] = reserved
        await _save(ctx, updates, result, refresh)

    ui.button("Save", icon="save", on_click=save).props("color=primary dense")

    with ui.expansion("Advanced planner arithmetic", icon="calculate").classes("w-full"):
        advanced = _Fields(ctx, _payload(ctx))
        advanced.row("planner.compute_overhead_fraction", label="Compute overhead fraction")
        ui.label(
            "Calibrated against observed loads. Raising it makes the planner more cautious and "
            "costs context; lowering it risks an out-of-memory at load time."
        ).classes("text-xs opacity-70")
        _save_button(ctx, advanced, refresh)


def _holder_dicts(ctx: GuiContext) -> list[dict[str, Any]]:
    """VRAM holders per GPU, in the shape ``gpu_setup_rows`` expects.

    One dict per (gpu, pid) row, carrying ``device_bytes`` -- what the pid
    holds on that row's card, from the PDH per-adapter split joined to CUDA
    ordinals (D39) -- so the per-GPU "N process(es) holding X" line no longer
    counts a two-card model's whole total on each of its cards. Where the split
    is unavailable ``device_bytes`` is ``None`` and the summary falls back to
    ``used_bytes`` as before.
    """
    from studioforge.core.gpu import vram_processes
    from studioforge.core.vram_holders import pdh_process_gpu_bytes

    if ctx.probe is None:
        return []
    try:
        per_gpu = pdh_process_gpu_bytes()
    except Exception:  # noqa: BLE001 - the split is a bonus, the row is not
        per_gpu = {}
    out: list[dict[str, Any]] = []
    for entry in vram_processes(ctx.probe):
        split = per_gpu.get(entry.pid) or {}
        out.append(
            {
                "pid": entry.pid,
                "name": entry.name,
                "used_bytes": entry.used_bytes,
                "device_bytes": split.get(entry.gpu_index) if split else None,
                "gpu_indices": [entry.gpu_index],
                "is_ours": entry.is_ours,
            }
        )
    return out


# ---------------------------------------------------------------------------
# 4. Engine
# ---------------------------------------------------------------------------


def _engine_card(ctx: GuiContext) -> None:
    card = ui.card().classes("w-full")

    def refresh() -> None:
        card.clear()
        with card, panel_guard("Engine"):
            _engine_body(ctx, refresh)

    refresh()


def _engine_body(ctx: GuiContext, refresh: Any) -> None:
    _section(
        "llama.cpp engine",
        "Versioned artifacts under engines/<tag>/, never whatever happens to be on PATH.",
    )
    manager = ctx.engine_manager
    if manager is None:
        ui.label("engine manager unavailable on this instance").classes("text-xs opacity-60")
        return

    active = _safe(manager.active, None)
    if active is None:
        ui.label(
            f"No engine installed. The pinned tag is {ctx.config.engine.pinned_tag} — install it."
        ).classes("text-sm text-warning")
    else:
        ui.label(f"active: {active.tag} ({active.variant})").classes("text-sm font-medium")
        ui.label(
            f"{active.version_string or st.UNKNOWN} · smoke tested {active.smoke_tested} · "
            f"{active.server_binary}"
        ).classes("text-xs font-mono opacity-70")

    installed = _safe(lambda: [info.model_dump(mode="json") for info in manager.installed()], [])
    for line, info in zip(st.engine_install_rows(installed), installed, strict=False):
        with ui.row().classes("items-center gap-2 w-full no-wrap"):
            ui.label(line).classes("text-xs font-mono grow truncate")
            tag = str(info.get("tag") or "")
            ui.button("smoke test", on_click=lambda t=tag: smoke_engine(ctx, t)).props("flat dense")
            ui.button(
                "activate + reload", on_click=lambda t=tag: activate_engine(ctx, t, refresh)
            ).props("flat dense")

    fields = _Fields(ctx, _payload(ctx))
    fields.row("engine.pinned_tag", label="Pinned tag")
    fields.row("engine.cuda_variant", label="CUDA build")
    ui.label(
        st.cuda_variant_note(
            _safe(lambda: ctx.probe.cuda_driver_version() if ctx.probe else None, None),
            ctx.config.engine.cuda_variant,
        )
    ).classes("text-xs opacity-70")
    fields.row("engine.allow_source_build", label="Build from source when no asset fits")
    fields.row("engine.keep_versions", label="Old engine versions to keep")
    _save_button(ctx, fields, refresh)

    ui.separator()
    update_row = ui.row().classes("w-full items-center gap-2")
    _paint_update(ctx, update_row, None, refresh)

    async def check() -> None:
        update_row.clear()
        with update_row:
            ui.spinner(size="sm")
            ui.label("checking GitHub for a newer engine…").classes("text-xs opacity-60")
        try:
            payload = await manager.check_update(limit=5)
        except Exception as exc:  # noqa: BLE001
            payload = {"checked": True, "error": str(exc)}
        _paint_update(ctx, update_row, payload, refresh)

    with ui.row().classes("items-center gap-2"):
        tag_input = ui.input("tag to install", value=ctx.config.engine.pinned_tag)
        tag_input.props("dense outlined").classes("w-40")
        ui.button(
            "Install",
            icon="download",
            on_click=lambda: install_engine(ctx, tag_input.value, refresh),
        ).props("outline dense")
        ui.button("Check for update", icon="cloud_download", on_click=check).props("outline dense")


def _paint_update(ctx: GuiContext, row: Any, update: dict[str, Any] | None, refresh: Any) -> None:
    row.clear()
    with row:
        ui.label(st.engine_update_line(update)).classes("text-sm")
        if st.engine_update_available(update):
            latest = str((update or {}).get("latest") or "")
            ui.button(
                f"Install {latest}",
                icon="download",
                on_click=lambda tag=latest: install_engine(ctx, tag, refresh),
            ).props("outline dense color=primary")
            ui.label(st.ENGINE_UPDATE_NOTE).classes("text-xs opacity-70")


# ---------------------------------------------------------------------------
# 5. Network and access
# ---------------------------------------------------------------------------


def _network_card(ctx: GuiContext) -> None:
    card = ui.card().classes("w-full")

    def refresh() -> None:
        card.clear()
        with card, panel_guard("Network & access"):
            _network_body(ctx, refresh)

    refresh()


def _network_body(ctx: GuiContext, refresh: Any) -> None:
    _section("Network & access", "Where this server listens, and what it takes to talk to it.")
    fields = _Fields(ctx, _payload(ctx))
    fields.row(
        "server.host",
        label="Bind address",
        options=["0.0.0.0", "127.0.0.1", ctx.config.server.host],
    )
    ui.label(st.bind_note(ctx.config.server.host)).classes("text-xs opacity-70")
    fields.row("server.port", label="Gateway port")
    ui.label(st.port_conflict_note(ctx.config.server.port)).classes("text-xs opacity-70")
    fields.row("gui.port", label="Control panel port")
    fields.row("watchdog.port", label="Watchdog port")

    ui.separator()
    _section("Credentials", "")
    fields.row("server.api_key", label="API key")
    _secret_line(
        "API key",
        ctx.config.server.api_key,
        unset_note="not set — the gateway and this panel are open to anyone who can reach them",
    )
    fields.row("mcp.pin", label="MCP pairing PIN")
    _secret_line(
        "MCP pairing PIN",
        ctx.config.mcp.pin,
        unset_note="not set",
        hint=(
            "Agents pair the MCP endpoint with it, and the RECOVERY watchdog (port "
            f"{ctx.config.watchdog.port}) requires it too when no API key is set — "
            "sfctl: `sfctl servers add rig <url> --api-key <PIN> --use`."
        ),
    )
    fields.row("mcp.pin_required", label="Require the PIN for MCP")
    with ui.row().classes("items-center gap-2 flex-wrap"):
        _save_button(ctx, fields, refresh)
        ui.button(
            "Generate new PIN", icon="password", on_click=lambda: _generate_pin(ctx, refresh)
        ).props("outline dense").tooltip(
            "Mints a new pairing code and saves it. Every already-paired agent has to be "
            "given the new one — which is exactly what you want after a leak."
        )
        ui.button(
            "Clear API key",
            icon="lock_open",
            on_click=lambda: _clear_secret(ctx, "server.api_key", refresh),
        ).props("outline dense")

    ui.separator()
    _section("Reachable at", "")
    endpoints = _safe(lambda: _endpoints(ctx), [])
    for line in st.reachable_lines(endpoints):
        ui.label(line).classes("text-xs font-mono opacity-80")
    if not endpoints:
        ui.label("could not enumerate this machine's addresses").classes("text-xs opacity-60")

    _openclaw_panel(ctx)


def _endpoints(ctx: GuiContext) -> list[dict[str, str]]:
    from studioforge.core.netinfo import reachable_urls

    return reachable_urls(ctx.config.server.port, host=ctx.config.server.host)


def _openclaw_panel(ctx: GuiContext) -> None:
    """The ready-to-paste OpenClaw configuration, secrets masked until asked.

    Built by ``GET /api/openclaw-setup`` -- the same endpoint ``sfctl`` and the
    MCP server use -- so what is shown is what this instance is really serving
    on. Copy always copies the real value; the *display* is masked, because a
    control panel on a shared screen should not put a credential in plain text
    by default.
    """
    with ui.expansion("Point OpenClaw at this server", icon="link").classes("w-full"):
        body = ui.column().classes("w-full gap-2")
        revealed = {"on": False}

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
            secrets = [ctx.config.server.api_key, ctx.config.mcp.pin, ctx.config.hf.token]
            blocks = list(snippets(payload))
            pin = payload.get("mcp_pin")
            if pin:
                url = (payload.get("companion_config") or {}).get("server.url", "")
                blocks.append(
                    (
                        "Pair the companion (sfctl)",
                        f"sfctl servers add rig {url} --api-key {pin}",
                    )
                )
            with body:
                ui.switch(
                    "reveal secrets",
                    value=revealed["on"],
                    on_change=lambda event: _toggle_reveal(revealed, event, load),
                ).props("dense")
                ui.label(
                    "Generated by the same endpoint sfctl and the MCP server use, so these are "
                    "the exact values this instance is serving on. Copy copies the real value "
                    "even while it is masked here."
                ).classes("text-xs opacity-70")
                for title, snippet in blocks:
                    shown = snippet if revealed["on"] else st.mask_secrets(snippet, secrets)
                    with ui.column().classes("w-full gap-0"):
                        with ui.row().classes("items-center gap-2"):
                            ui.label(title).classes("text-xs font-medium")
                            ui.button(
                                icon="content_copy", on_click=lambda text=snippet: copy_text(text)
                            ).props("flat dense").tooltip("Copy (with the real values)")
                        ui.label(shown).classes(
                            "w-full font-mono text-xs whitespace-pre-wrap "
                            "bg-black/10 dark:bg-white/5 p-2 rounded"
                        )

        ui.timer(0.05, load, once=True)


def _toggle_reveal(state: dict[str, bool], event: Any, reload: Any) -> None:
    state["on"] = bool(getattr(event, "value", False))
    ui.timer(0.01, reload, once=True)


async def _generate_api_key(ctx: GuiContext, refresh: Any) -> None:
    """The one-click fix for "reachable from the network with no key" (WP17 F4).

    Mints a random key and saves it as ``server.api_key``. Shown masked on the
    panel like every other secret; the user copies it from there into
    OpenClaw (``Authorization: Bearer <key>``). ``RESTART_REQUIRED_KEYS`` decides
    whether the change is live or needs a restart -- the save result says which.
    """
    import secrets

    try:
        payload = await apply_config_updates(ctx, {"server.api_key": secrets.token_urlsafe(24)})
    except Exception as exc:  # noqa: BLE001
        notify_error(exc, what="set API key")
        return
    ui.notify(
        f"an API key was generated and saved ({st.save_result_text(payload)}). "
        "Copy it from the Network & access card into your clients; the PIN is unchanged.",
        type="positive",
        multi_line=True,
    )
    refresh()


async def _generate_pin(ctx: GuiContext, refresh: Any) -> None:
    from studioforge.config import generate_pin

    try:
        payload = await apply_config_updates(ctx, {"mcp.pin": generate_pin()})
    except Exception as exc:  # noqa: BLE001
        notify_error(exc, what="generate PIN")
        return
    # The new PIN is deliberately not put in this notification: it is on the
    # panel, masked, and in the startup banner.
    ui.notify(
        f"a new MCP PIN was generated and saved ({st.save_result_text(payload)}). "
        "Re-pair any agent that used the old one.",
        type="positive",
        multi_line=True,
    )
    refresh()


async def _clear_secret(ctx: GuiContext, key: str, refresh: Any) -> None:
    try:
        payload = await apply_config_updates(ctx, {key: None})
    except Exception as exc:  # noqa: BLE001
        notify_error(exc, what=f"clear {key}")
        return
    ui.notify(st.save_result_text(payload), type="warning", multi_line=True)
    refresh()


# ---------------------------------------------------------------------------
# 6. Downloads and HuggingFace
# ---------------------------------------------------------------------------


def _downloads_card(ctx: GuiContext) -> None:
    card = ui.card().classes("w-full")

    def refresh() -> None:
        card.clear()
        with card, panel_guard("Downloads & HuggingFace"):
            _downloads_body(ctx, refresh)

    refresh()


def _downloads_body(ctx: GuiContext, refresh: Any) -> None:
    _section("Downloads & HuggingFace", "Only needed to fetch new models; nothing is sent out.")
    fields = _Fields(ctx, _payload(ctx))
    fields.row("hf.token", label="HuggingFace token")
    token_state = st.secret_state_text(
        ctx.config.hf.token,
        unset_note="not set — public repositories still download fine",
    )
    ui.label(f"currently {token_state}").classes("text-xs font-mono opacity-70")
    fields.row("hf.max_concurrent_downloads", label="Concurrent downloads")
    with ui.expansion("Advanced", icon="tune").classes("w-full"):
        fields.row("hf.chunk_bytes", label="Chunk size (bytes)")
    with ui.row().classes("items-center gap-2 flex-wrap"):
        _save_button(ctx, fields, refresh)
        ui.button(
            "Clear token",
            icon="lock_open",
            on_click=lambda: _clear_secret(ctx, "hf.token", refresh),
        ).props("outline dense")

    disk = _safe(lambda: _disk_report(ctx), None)
    line = st.disk_line(disk)
    if line:
        ui.label(line).classes(
            "text-xs font-mono " + ("text-warning" if st.disk_is_low(disk) else "opacity-80")
        )


# ---------------------------------------------------------------------------
# 7. Startup and service
# ---------------------------------------------------------------------------


def _startup_card(ctx: GuiContext) -> None:
    card = ui.card().classes("w-full")

    def refresh() -> None:
        card.clear()
        with card, panel_guard("Startup & service"):
            _startup_body(ctx, refresh)

    refresh()


def _startup_body(ctx: GuiContext, refresh: Any) -> None:
    _section("Startup & service", "Whether this comes back by itself, and where everything lives.")
    enabled, mechanism = _autostart_facts(ctx.config)
    ui.label(
        f"autostart: {'enabled' if enabled else 'not enabled'}"
        + (f" via {mechanism}" if mechanism else "")
    ).classes("text-xs font-mono opacity-80")
    with ui.row().classes("gap-2 flex-wrap"):
        ui.button(
            "Enable (tray + server)" if os.name == "nt" else "Enable",
            icon="play_circle",
            on_click=lambda: _set_autostart(ctx, True, refresh),
        ).props("outline dense")
        ui.button(
            "Disable", icon="stop_circle", on_click=lambda: _set_autostart(ctx, False, refresh)
        ).props("outline dense")
    ui.label(
        "Windows: a hidden .vbs shim in your per-user Startup folder, no admin rights needed. "
        "Linux: a systemd --user unit. The system-wide units for a headless server are in "
        "deploy/."
    ).classes("text-xs opacity-70")

    ui.separator()
    fields = _Fields(ctx, _payload(ctx))
    fields.row("logging.level", label="Log level", options=["DEBUG", "INFO", "WARNING", "ERROR"])
    _save_button(ctx, fields, refresh)

    ui.separator()
    _section("Where things live", "")
    source = st.data_dir_source(os.environ.get("SF_DATA_DIR"), _checkout_dir())
    for name, value in st.where_things_live(ctx.config, source=source):
        with ui.row().classes("w-full items-center gap-2 no-wrap"):
            ui.label(name).classes("text-xs opacity-60 w-56 shrink-0")
            ui.label(value).classes("text-xs font-mono grow truncate")
            ui.button(icon="content_copy", on_click=lambda text=value: copy_text(text)).props(
                "flat dense"
            ).tooltip("Copy")

    with ui.row().classes("gap-2 flex-wrap"):
        ui.button(
            "Open data dir", icon="folder_open", on_click=lambda: _open_path(ctx.config.data_dir)
        ).props("outline dense").tooltip(
            "Opens a file manager on the machine running StudioForge, not on the machine you "
            "are browsing from."
        )
        ui.button(
            "Open logs", icon="description", on_click=lambda: _open_path(ctx.config.logs_dir)
        ).props("outline dense")
        ui.button(
            "Restart server", icon="power_settings_new", on_click=lambda: _restart_dialog(ctx)
        ).props("outline dense color=warning").tooltip(st.RESTART_SERVER_WARNING)


def _checkout_dir() -> Any:
    from studioforge.config import _checkout_data_dir

    return _checkout_data_dir()


def _open_path(path: Any) -> None:
    """Open a directory on the machine running the server. Never raises."""
    import subprocess
    import sys

    try:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(target))  # noqa: S606 - a directory we own
        else:  # pragma: no cover - the panel is usually driven on Windows here
            subprocess.Popen(["xdg-open", str(target)])  # noqa: S603,S607
    except (OSError, AttributeError) as exc:
        ui.notify(f"could not open {path}: {exc}", type="warning", multi_line=True)
        return
    ui.notify(f"opened {path} on the server", type="positive")


async def _set_autostart(ctx: GuiContext, enable: bool, refresh: Any) -> None:
    from studioforge.core import autostart

    with busy(message="Updating the startup entry…"):
        try:
            if enable:
                status = await run_blocking(autostart.enable, ctx.config, tray=os.name == "nt")
            else:
                status = await run_blocking(autostart.disable, ctx.config)
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="autostart")
            return
    ui.notify(status.describe(), type="positive", multi_line=True)
    refresh()


def _restart_dialog(ctx: GuiContext) -> None:
    """Confirm first: this is the one control that takes the gateway down."""
    with ui.dialog() as dialog, ui.card().classes("min-w-[30rem]"):
        ui.label("Restart the StudioForge server?").classes("font-medium")
        ui.label(st.RESTART_SERVER_WARNING).classes("text-sm opacity-80")
        banner = ui.label("").classes("text-sm text-warning")

        async def confirm() -> None:
            try:
                from studioforge.api.admin_routes import restart_server
                from studioforge.gui.tabs import api_request

                payload = await restart_server(api_request(ctx), confirm=True)
            except Exception as exc:  # noqa: BLE001
                notify_error(exc, what="restart server")
                return
            banner.set_text(st.restart_server_note(payload))
            ui.notify(st.restart_server_note(payload), type="warning", multi_line=True)
            dialog.close()

        with ui.row().classes("justify-end gap-2 w-full"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("Restart server", on_click=confirm).props("color=negative")
    dialog.open()


# ---------------------------------------------------------------------------
# 8. Advanced: every remaining key, generated from the config model
# ---------------------------------------------------------------------------


def _advanced_card(ctx: GuiContext) -> None:
    card = ui.card().classes("w-full")

    def refresh() -> None:
        card.clear()
        with card, panel_guard("Advanced settings"):
            _advanced_body(ctx, refresh)

    refresh()


def _advanced_body(ctx: GuiContext, refresh: Any) -> None:
    specs = st.config_field_specs()
    remaining = st.advanced_field_specs(specs, COVERED_KEYS)
    with ui.expansion(
        f"Advanced — every other setting ({len(remaining)})", icon="settings"
    ).classes("w-full"):
        ui.label(
            "Generated from the configuration model itself, so a key added to StudioForge "
            "appears here without anyone remembering to add a form row. Secrets and the "
            "per-GPU maps are not here: they have their own controls above."
        ).classes("text-xs opacity-70")
        payload = _payload(ctx)
        fields = _Fields(ctx, payload)
        for section in st.config_sections(remaining):
            in_section = [spec for spec in remaining if spec.section == section]
            if not in_section:
                continue
            with ui.expansion(f"{section} ({len(in_section)})").classes("w-full"):
                for spec in in_section:
                    fields.row(spec.key)
        _save_button(ctx, fields, refresh)
