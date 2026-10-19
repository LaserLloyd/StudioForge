"""Status: what a person checks first, and nothing to configure.

Health and boot phase, every GPU with what is on it, the loaded models with busy or idle
and tok/s, active requests, the engine, the version and build, uptime. The only actions are
Unload (a model), Open in Models and Open logs. Every setting that used to sit on the
Overview lives under Advanced; the loaded-model pin/reload controls moved to the Models
drawer and the restart/unload-everything buttons to Advanced > Watchdog & recovery.

The numbers come from the same places as before (the supervisor, the probe, the lease book,
``BootStatus``); the words come from :mod:`studioforge.gui.status_view`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from nicegui import ui

from studioforge import __version__
from studioforge.gui import nav
from studioforge.gui import state as st
from studioforge.gui import status_view as sv
from studioforge.gui.tabs import GuiContext, element_alive, panel_guard
from studioforge.gui.tabs.widgets import copy_button

#: What a Status button calls to move to another place in the panel. The shell
#: (``app.py``) supplies the real one; tests and direct calls get a no-op.
Go = Callable[[nav.Location], None]


def _noop(_location: nav.Location) -> None:
    return None


def render(ctx: GuiContext, go: Go = _noop) -> None:
    from studioforge.gui.tabs import overview

    with ui.column().classes("w-full gap-4 p-2"):
        with panel_guard("The setup banner"):
            overview._setup_banner(ctx)
        with panel_guard("The health summary"):
            _health_card(ctx, go)
        with panel_guard("The GPU cards"):
            _gpu_section(ctx)
        with panel_guard("The loaded models"):
            _loaded_section(ctx, go)


# ---------------------------------------------------------------------------
# Reads (every one tolerates a missing subsystem)
# ---------------------------------------------------------------------------


def _gpus(ctx: GuiContext) -> list[Any]:
    try:
        return list(ctx.probe.list_gpus()) if ctx.probe is not None else []
    except Exception:  # noqa: BLE001 - a probe gap is a display gap
        return []


def _instances(ctx: GuiContext) -> list[Any]:
    try:
        return list(ctx.supervisor.list()) if ctx.supervisor is not None else []
    except Exception:  # noqa: BLE001
        return []


def _leases(ctx: GuiContext) -> list[Any]:
    try:
        return list(ctx.manager.leases.all()) if ctx.manager is not None else []
    except Exception:  # noqa: BLE001
        return []


def _boot(ctx: GuiContext) -> dict[str, Any] | None:
    boot = getattr(ctx.api_state, "boot", None)
    try:
        return dict(boot.snapshot()) if boot is not None else None
    except Exception:  # noqa: BLE001
        return None


def _engine_tag(ctx: GuiContext) -> str:
    try:
        active = ctx.engine_manager.active() if ctx.engine_manager is not None else None
    except Exception:  # noqa: BLE001
        active = None
    tag = getattr(active, "tag", None)
    return str(tag) if tag else "none installed"


def _boot_text(boot: dict[str, Any]) -> str:
    """``ready (6.1s)`` / ``scanning models (2.0s)`` / a dash when the app reports no boot."""
    if not boot:
        return st.UNKNOWN
    phase = str(boot.get("phase") or st.UNKNOWN)
    elapsed = boot.get("elapsed_s")
    return f"{phase} ({elapsed}s)" if isinstance(elapsed, int | float) else phase


def _build() -> str:
    try:
        from studioforge.build import build_id

        return build_id()
    except Exception:  # noqa: BLE001
        return st.UNKNOWN


def uptime_text(ctx: GuiContext) -> str:
    started = float(getattr(ctx.api_state, "started_at", 0.0) or 0.0)
    return st.format_duration(time.time() - started) if started else st.UNKNOWN


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def _health_card(ctx: GuiContext, go: Go) -> None:
    from studioforge.gui.tabs.overview import base_urls

    with ui.card().classes("w-full"):
        with ui.row().classes("w-full items-center gap-3 flex-wrap"):
            chip = ui.badge("", color="grey").classes("text-sm").props('role="status"')
            detail = ui.label("").classes("text-sm opacity-80 grow")
            ui.button(
                "Open logs",
                icon="article",
                on_click=lambda: go(nav.Location(nav.ADVANCED, page="logs")),
            ).props("flat dense no-caps").tooltip("Advanced > Logs & metrics")
        facts = ui.row().classes("w-full gap-x-8 gap-y-1 flex-wrap")
        with ui.row().classes("w-full items-center gap-2 no-wrap"):
            ui.label("Base URL").classes("text-xs opacity-70 shrink-0")
            url = base_urls(ctx)[0]["url"]
            ui.label(url).classes("font-mono text-sm truncate")
            copy_button(url, label=f"Copy {url}", what="Copied the base URL")

    def paint() -> None:
        if not element_alive(facts):
            return
        instances = _instances(ctx)
        gpus = _gpus(ctx)
        health = sv.health_chip(
            _boot(ctx),
            draining=bool(getattr(ctx.manager, "draining", False)),
            instances=instances,
            gpus_seen=bool(gpus) or ctx.probe is None,
            manager_present=ctx.manager is not None,
        )
        chip.set_text(health.label)
        chip.props(f"color={health.colour}")
        detail.set_text(health.detail)
        boot = _boot(ctx) or {}
        facts.clear()
        with facts:
            for label, value in (
                ("Boot", _boot_text(boot)),
                ("Loaded", st.loaded_count_text(instances)),
                ("Active requests", str(sv.total_active_requests(instances))),
                ("Engine", _engine_tag(ctx)),
                ("Version", f"v{__version__} · build {_build()}"),
                ("Uptime", uptime_text(ctx)),
            ):
                with ui.column().classes("gap-0"):
                    ui.label(label).classes("text-xs opacity-60")
                    ui.label(value).classes("text-sm font-mono")

    paint()
    ui.timer(max(2.0, ctx.refresh_interval), paint)


# ---------------------------------------------------------------------------
# GPUs
# ---------------------------------------------------------------------------


def _gpu_section(ctx: GuiContext) -> None:
    ui.label("GPUs").classes("text-lg font-medium")
    ram = ui.label("").classes("text-xs opacity-70")
    container = ui.row().classes("w-full gap-3 flex-wrap")

    def refresh() -> None:
        if not element_alive(container):
            return
        gpus = _gpus(ctx)
        instances = _instances(ctx)
        leases = _leases(ctx)
        try:
            from studioforge.core.gpu import system_ram

            total, used = system_ram()
        except Exception:  # noqa: BLE001
            total, used = (0, 0)
        ram.set_text(st.ram_text(total, used))
        container.clear()
        with container:
            if not gpus:
                backend = getattr(ctx.probe, "backend", "unknown")
                ui.label(
                    f"No GPUs reported (probe backend: {backend}). The gateway still runs, "
                    "but the planner cannot place models."
                ).classes("text-sm opacity-70")
                return
            for gpu in gpus:
                with ui.card().classes("min-w-[16rem] grow"):
                    ui.label(st.gpu_headline(gpu)).classes("font-medium text-sm")
                    bar = ui.linear_progress(
                        value=st.vram_fraction(gpu), show_value=False, size="14px"
                    ).props("rounded")
                    bar.props(
                        f'aria-label="GPU{gpu.index} memory used: '
                        f'{round(st.vram_fraction(gpu) * 100)} percent"'
                    )
                    for line in st.gpu_detail_lines(gpu):
                        ui.label(line).classes("text-xs opacity-80 font-mono")
                    ui.separator()
                    for line in sv.gpu_occupant_lines(gpu, instances, leases):
                        ui.label(line).classes("text-xs font-mono truncate")

    refresh()
    ui.timer(ctx.refresh_interval, refresh)


# ---------------------------------------------------------------------------
# Loaded models
# ---------------------------------------------------------------------------


def _loaded_section(ctx: GuiContext, go: Go) -> None:
    from studioforge.gui.tabs import dashboard

    in_flight = {"busy": False}
    ui.label("Loaded models").classes("text-lg font-medium")
    stale = ui.label("").classes("text-xs text-warning")
    container = ui.column().classes("w-full gap-2")

    def paint(entries: list[Any]) -> None:
        container.clear()
        with container:
            if not entries:
                with ui.row().classes("items-center gap-2"):
                    ui.label("Nothing loaded. A model loads on first use, or from Models.").classes(
                        "text-sm opacity-70"
                    )
                    ui.button(
                        "Open Models",
                        icon="view_list",
                        on_click=lambda: go(nav.Location(nav.MODELS)),
                    ).props("flat dense no-caps")
                return
            # Once per repaint, not once per card: ``active()`` reads disk.
            active_engine = dashboard._active_engine_tag(ctx)
            for instance, introspection, pinned in entries:
                _loaded_row(ctx, instance, introspection, pinned, refresh, go, active_engine)

    async def refresh() -> None:
        if not element_alive(container) or in_flight["busy"]:
            return
        in_flight["busy"] = True
        try:
            entries = await dashboard._loaded_state(ctx)
        except Exception as exc:  # noqa: BLE001 - keep the last good rows, say so once
            stale.set_text(st.poll_failure_note(exc))
            return
        finally:
            in_flight["busy"] = False
        if not element_alive(container):
            return
        stale.set_text("")
        paint(entries)

    # First paint from the supervisor alone (no round trip to each child), so the page is
    # never blank; the timer then adds what only the child can say (slots, build, context).
    paint([(instance, None, _is_pinned(ctx, instance)) for instance in _instances(ctx)])
    ui.timer(ctx.refresh_interval, refresh)
    ui.timer(0.1, refresh, once=True)


def _is_pinned(ctx: GuiContext, instance: Any) -> bool:
    try:
        record = ctx.registry.resolve(instance.model_id) if ctx.registry is not None else None
        return bool(record.settings.pinned) if record is not None else False
    except Exception:  # noqa: BLE001
        return False


def _loaded_row(
    ctx: GuiContext,
    instance: Any,
    introspection: dict[str, Any] | None,
    pinned: bool,
    refresh: Any,
    go: Go,
    active_engine: str | None = None,
) -> None:
    from studioforge.gui.tabs import dashboard

    label, colour = sv.busy_label(instance)
    with ui.card().classes("w-full"), ui.column().classes("w-full gap-1"):
        with ui.row().classes("w-full items-center gap-2 no-wrap"):
            ui.label(instance.model_id).classes("font-medium text-sm grow truncate")
            fp4_note = st.fp4_plan_note(instance)
            if fp4_note:
                # Informative, never alarming (D9): the model runs correctly here.
                ui.badge("slower prefill", color="info").classes("text-xs").tooltip(
                    f"{fp4_note}. Generation speed is unaffected; only prompt processing is "
                    "slower than a Blackwell card would be."
                )
            if st.is_speculative(introspection):
                ui.badge("draft", color="accent").classes("text-xs").tooltip(
                    "Speculative decoding is armed on this instance"
                )
            if pinned:
                ui.badge("pinned", color="accent").classes("text-xs")
            tier_badge = st.priority_badge_label(instance.priority)
            if tier_badge:
                ui.badge(tier_badge, color="secondary").classes("text-xs").tooltip(
                    st.priority_tooltip(instance.priority)
                )
            drift = st.instance_engine_drift(
                getattr(instance, "resolved_engine_tag", None), active_engine
            )
            if drift:
                ui.badge(drift[0], color="warning").classes("text-xs").tooltip(drift[1])
            ui.badge(label, color=colour).classes("text-xs")
            speed = getattr(instance, "last_tokens_per_second", None)
            ui.label(f"{speed} tok/s" if speed else "— tok/s").classes("text-xs font-mono")
            ui.button(
                "Open in Models",
                icon="open_in_new",
                on_click=lambda model_id=instance.model_id: go(
                    nav.Location(nav.MODELS, model=model_id)
                ),
            ).props("flat dense no-caps")
            ui.button(
                "Unload",
                icon="stop_circle",
                on_click=lambda model_id=instance.model_id: dashboard._unload_one(
                    ctx, model_id, refresh
                ),
            ).props("flat dense no-caps color=negative").tooltip(
                "Unload this model and free its VRAM. It reloads on its next request."
            )
        ui.label(
            f"port {instance.port if instance.port is not None else st.UNKNOWN}"
            f" · ctx {_ctx_text(instance, introspection)}"
            f" · {_slots_text(instance, introspection)}"
            f" · {st.device_text(instance)}"
            f" · TTL {st.instance_ttl_text(instance, pinned=pinned)}"
            f" · tier {st.priority_label(instance.priority)}"
        ).classes("text-xs font-mono opacity-80")
        ui.label(
            f"active requests {instance.active_requests} · total {instance.total_requests}"
            f" · generated {st.tokens_generated(introspection)} tokens"
            f" · modalities {st.modalities_text(introspection)}"
            f" · build {_build_text(instance, introspection)}"
        ).classes("text-xs font-mono opacity-60")
        for slot in st.activity_slot_rows(introspection) or []:
            ui.label(st.slot_line(slot)).classes("text-xs font-mono opacity-70 pl-2")
        if instance.last_error:
            ui.label(str(instance.last_error)[:400]).classes(
                "text-xs text-negative whitespace-pre-wrap"
            )


def _ctx_text(instance: Any, introspection: dict[str, Any] | None) -> str:
    """The context the child really runs with; the plan's figure when it cannot be asked."""
    actual = st.actual_ctx_text(introspection)
    plan = getattr(instance, "plan", None)
    if actual == st.UNKNOWN and plan is not None and getattr(plan, "ctx_size", None):
        return f"{plan.ctx_size} (planned)"
    return actual


def _slots_text(instance: Any, introspection: dict[str, Any] | None) -> str:
    text = st.activity_slots_text(introspection)
    plan = getattr(instance, "plan", None)
    if introspection is None and plan is not None and getattr(plan, "parallel", None):
        return f"{plan.parallel} slot(s)"
    return text


def _build_text(instance: Any, introspection: dict[str, Any] | None) -> str:
    text = st.build_info_text(introspection)
    tag = getattr(instance, "resolved_engine_tag", None)
    return str(tag) if text == st.UNKNOWN and tag else text


__all__ = ["render", "uptime_text"]
