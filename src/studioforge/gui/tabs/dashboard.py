"""Dashboard: what the GPUs and the loaded models are doing right now.

Everything here is polled on ``gui.refresh_interval_s`` and every reading is
optional. NVML may be absent, ``/slots`` may not answer, nothing may be loaded
-- each of those is a normal state that must render as an empty panel or a dash,
never as a traceback. The numbers shown are the *actual* ones llama-server
reports (context, slots), not the ones we asked for, because the gap between
the two is exactly what an operator needs to see.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from studioforge.gui import state as st
from studioforge.gui.tabs import GuiContext, api_request, busy, notify_error

LOG_TAIL_LINES = 40


def _safe_gpus(ctx: GuiContext) -> list[Any]:
    try:
        return list(ctx.probe.list_gpus()) if ctx.probe is not None else []
    except Exception:  # noqa: BLE001 - a probe failure is a display gap, not an error
        return []


def _safe_ram() -> tuple[int, int]:
    try:
        from studioforge.core.gpu import system_ram

        return system_ram()
    except Exception:  # noqa: BLE001
        return (0, 0)


def render(ctx: GuiContext) -> None:
    with ui.column().classes("w-full gap-4 p-2"):
        _gpu_panel(ctx)
        _vram_holders_panel(ctx)
        _loaded_panel(ctx)
        _log_panel(ctx)


# ---------------------------------------------------------------------------
# GPUs
# ---------------------------------------------------------------------------


def _gpu_panel(ctx: GuiContext) -> None:
    ui.label("GPUs").classes("text-lg font-medium")
    ram = ui.label("").classes("text-xs opacity-70")
    container = ui.row().classes("w-full gap-3 flex-wrap")

    def refresh() -> None:
        gpus = _safe_gpus(ctx)
        total_ram, used_ram = _safe_ram()
        ram.set_text(st.ram_text(total_ram, used_ram))
        container.clear()
        with container:
            if not gpus:
                backend = getattr(ctx.probe, "backend", "unknown")
                ui.label(
                    f"No GPUs reported (probe backend: {backend}). "
                    "NVML may be unavailable — the gateway still runs, but the "
                    "planner cannot place models."
                ).classes("text-sm opacity-70")
                return
            for gpu in gpus:
                with ui.card().classes("min-w-[16rem] grow"):
                    ui.label(st.gpu_headline(gpu)).classes("font-medium text-sm")
                    ui.linear_progress(
                        value=st.vram_fraction(gpu), show_value=False, size="14px"
                    ).props("rounded")
                    for line in st.gpu_detail_lines(gpu):
                        ui.label(line).classes("text-xs opacity-80 font-mono")

    refresh()
    ui.timer(ctx.refresh_interval, refresh)


# ---------------------------------------------------------------------------
# VRAM holders
# ---------------------------------------------------------------------------


def _vram_holders_panel(ctx: GuiContext) -> None:
    """ "Who has my VRAM", including the answers that are not us.

    The GPU gauges above show that memory is gone; this shows who took it. It
    exists because on 2026-08-18 ~25 GiB was held by three llama-server
    children of a stray ``pytest`` run and no surface in the product could name
    them -- ``/api/status`` said ``llama-server.exe``, ``0 bytes``,
    ``is_ours: false``, which is three facts and no answer.

    Reclaim is offered for orphans only (parent dead: pure leak). A holder that
    belongs to a live foreign process is named, not killed -- something is
    still using it.
    """
    in_flight = {"busy": False}

    with ui.row().classes("w-full items-center gap-2"):
        ui.label("VRAM holders").classes("text-lg font-medium")
        ui.space()
        note = ui.label("").classes("text-xs opacity-70")
    container = ui.column().classes("w-full gap-1")

    async def refresh() -> None:
        if in_flight["busy"]:
            return
        in_flight["busy"] = True
        try:
            from studioforge.api.mgmt_routes import vram_holders as holders_route

            view = await holders_route(api_request(ctx))
        except Exception as exc:  # noqa: BLE001 - a probe gap is not an error card
            note.set_text(st.poll_failure_note(exc))
            return
        finally:
            in_flight["busy"] = False

        note.set_text(st.vram_holders_note(view))
        container.clear()
        with container:
            for holder in view.get("holders") or []:
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    ui.label(st.vram_holder_line(holder)).classes(
                        "text-xs font-mono opacity-80 grow truncate"
                    )
                    if st.vram_holder_is_reclaimable(holder):
                        ui.button(
                            "Reclaim",
                            icon="cleaning_services",
                            on_click=lambda: _reclaim_orphans(ctx, refresh),
                        ).props("outline dense color=negative").tooltip(
                            "Kill every orphaned llama-server (parent process gone) and "
                            "free its VRAM. Holders belonging to a live process are "
                            "never touched."
                        )

    ui.timer(max(2.0, ctx.refresh_interval), refresh)
    ui.timer(0.2, refresh, once=True)


async def _reclaim_orphans(ctx: GuiContext, refresh: Any) -> None:
    with busy(message="Reclaiming leaked VRAM…"):
        try:
            from studioforge.api.mgmt_routes import vram_reclaim

            payload = await vram_reclaim(api_request(ctx), dry_run=False)
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="reclaim VRAM")
            return
    ui.notify(
        f"killed {payload.get('killed', 0)} of {payload.get('orphans_found', 0)} orphan(s)",
        type="positive",
    )
    await refresh()


# ---------------------------------------------------------------------------
# Loaded models
# ---------------------------------------------------------------------------


async def _loaded_state(ctx: GuiContext) -> list[tuple[Any, dict[str, Any] | None, bool]]:
    """``(instance, introspection, pinned)`` for every running child."""
    supervisor = ctx.supervisor
    manager = ctx.manager
    if supervisor is None or manager is None:
        return []
    out: list[tuple[Any, dict[str, Any] | None, bool]] = []
    for instance in supervisor.list():
        introspection: dict[str, Any] | None
        try:
            introspection = await manager.introspect(instance.model_id)
        except Exception:  # noqa: BLE001 - a child that will not answer is still listed
            introspection = None
        pinned = False
        try:
            record = ctx.registry.resolve(instance.model_id) if ctx.registry else None
            pinned = bool(record.settings.pinned) if record is not None else False
        except Exception:  # noqa: BLE001
            pinned = False
        out.append((instance, introspection, pinned))
    return out


def _loaded_panel(ctx: GuiContext) -> None:
    """Per-model live status, with the same detail LM Studio shows.

    Rendered as cards rather than a flat table because each model has a
    variable number of slots and each slot has its own activity line, prompt
    cache hit rate and drafting flag -- that does not fit a fixed grid.

    Poll discipline: a failed tick sets a stale marker and keeps the last good
    cards (an error card per tick would stack forever), and a tick that is
    still in flight is never doubled -- a slow child answering ``/slots``
    must not pile up concurrent refreshes.
    """
    loaded_ids: list[str] = []
    in_flight = {"busy": False}

    with ui.row().classes("w-full items-center gap-2 flex-wrap"):
        ui.label("Loaded models").classes("text-lg font-medium")
        ui.space()
        ui.button(
            "Unload all",
            icon="layers_clear",
            on_click=lambda: _unload_all_dialog(ctx, loaded_ids, refresh),
        ).props("outline dense color=negative").tooltip(
            "Unload every resident model and free all VRAM."
        )
        ui.button(
            "Restart engines",
            icon="autorenew",
            on_click=lambda: _restart_engines(ctx, refresh),
        ).props("outline dense").tooltip(st.RESTART_ENGINES_HELP)
        ui.button(
            "Restart server",
            icon="power_settings_new",
            on_click=lambda: _restart_server_dialog(ctx, banner),
        ).props("outline dense color=warning").tooltip(st.RESTART_SERVER_WARNING)
    # Annotated because the "Restart server" handler above closes over it
    # before this line runs -- fine at runtime (the click cannot happen until
    # the page is built) but mypy needs the type stated.
    banner: ui.label = ui.label("").classes("text-sm text-warning")
    stale = ui.label("").classes("text-xs text-warning")
    container = ui.column().classes("w-full gap-2")

    async def refresh() -> None:
        if in_flight["busy"]:
            return
        in_flight["busy"] = True
        try:
            entries = await _loaded_state(ctx)
        except Exception as exc:  # noqa: BLE001 - degrade to stale, never stack
            stale.set_text(st.poll_failure_note(exc))
            return
        finally:
            in_flight["busy"] = False
        stale.set_text("")
        loaded_ids[:] = [instance.model_id for instance, _, _ in entries]
        container.clear()
        with container:
            if not entries:
                ui.label(
                    "Nothing loaded. A model loads on first use, or from the Models tab."
                ).classes("text-sm opacity-70")
                return
            for instance, introspection, pinned in entries:
                _loaded_card(ctx, instance, introspection, pinned, refresh)

    ui.timer(ctx.refresh_interval, refresh)
    ui.timer(0.1, refresh, once=True)


def _loaded_card(
    ctx: GuiContext,
    instance: Any,
    introspection: dict[str, Any] | None,
    pinned: bool,
    refresh: Any,
) -> None:
    with ui.card().classes("w-full"), ui.column().classes("w-full gap-1"):
        with ui.row().classes("w-full items-center gap-2 no-wrap"):
            ui.label(instance.model_id).classes("font-medium text-sm grow truncate")
            fp4_note = st.fp4_plan_note(instance)
            if fp4_note:
                # Informative, never alarming (DECISIONS D9): the model runs
                # correctly here, prompt processing is just faster on a card
                # with native FP4 acceleration.
                ui.badge("slower prefill", color="info").classes("text-xs").tooltip(
                    f"{fp4_note}. The model is fully supported on this GPU — generation "
                    "speed is unaffected; only prompt processing is slower than a "
                    "Blackwell card would be."
                )
            if st.is_speculative(introspection):
                ui.badge("draft", color="accent").classes("text-xs").tooltip(
                    "Speculative decoding is armed on this instance"
                )
            if pinned:
                ui.badge("pinned", color="accent").classes("text-xs")
            ui.badge(
                st.activity_label(introspection),
                color=st.activity_colour(introspection),
            ).classes("text-xs")
            ui.button(
                icon="restart_alt",
                on_click=lambda model_id=instance.model_id: _restart_model(ctx, model_id, refresh),
            ).props("flat dense").tooltip(
                "Reload this model's llama-server child with its current settings. "
                "The planner runs again against current free VRAM."
            )
            ui.button(
                icon="stop_circle",
                on_click=lambda model_id=instance.model_id: _unload_one(ctx, model_id, refresh),
            ).props("flat dense color=negative").tooltip(
                "Unload this model and free its VRAM. It reloads on its next request."
            )
        ui.label(
            f"port {instance.port if instance.port is not None else st.UNKNOWN}"
            f" · actual ctx {st.actual_ctx_text(introspection)}"
            f" · {st.activity_slots_text(introspection)}"
            f" · {st.device_text(instance)}"
            f" · TTL {st.instance_ttl_text(instance, pinned=pinned)}"
        ).classes("text-xs font-mono opacity-80")
        ui.label(
            f"active requests {instance.active_requests}"
            f" · total {instance.total_requests}"
            f" · last {instance.last_tokens_per_second or st.UNKNOWN} tok/s"
            f" · generated {st.tokens_generated(introspection)} tokens"
            f" · modalities {st.modalities_text(introspection)}"
            f" · build {st.build_info_text(introspection)}"
        ).classes("text-xs font-mono opacity-60")
        slots = st.activity_slot_rows(introspection)
        if slots:
            for slot in slots:
                ui.label(st.slot_line(slot)).classes("text-xs font-mono opacity-70 pl-2")
        if instance.last_error:
            ui.label(instance.last_error[:400]).classes("text-xs text-negative whitespace-pre-wrap")


# ---------------------------------------------------------------------------
# Unload / restart actions
#
# All of these are in-process calls on the shared object graph, and all of them
# are awaited with a spinner rather than run on the event loop: an unload waits
# for a child to exit and a reload waits for a 17 GiB model to come back, and
# either would freeze the panel for every viewer if it blocked.
# ---------------------------------------------------------------------------


async def _unload_one(ctx: GuiContext, model_id: str, refresh: Any) -> None:
    with busy(message=f"Unloading {model_id}…"):
        try:
            await ctx.manager.unload(model_id)
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="unload")
            return
    ui.notify(f"{model_id} unloaded", type="positive")
    await refresh()


async def _restart_model(ctx: GuiContext, model_id: str, refresh: Any) -> None:
    with busy(message=f"Restarting {model_id}…"):
        try:
            # A force-load rather than unload-then-load: the planner runs once
            # against current free VRAM, and a failure never leaves the model
            # unloaded when it was working a moment ago.
            instance = await ctx.manager.load(model_id, force=True)
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="restart model")
            return
    ui.notify(f"{model_id} restarted on port {instance.port}", type="positive")
    await refresh()


def _unload_all_dialog(ctx: GuiContext, loaded_ids: list[str], refresh: Any) -> None:
    """Confirm first: this drops every resident model, pinned ones included."""
    with ui.dialog() as dialog, ui.card().classes("min-w-[28rem]"):
        ui.label("Unload all models?").classes("font-medium")
        ui.label(st.unload_all_prompt(loaded_ids)).classes("text-sm whitespace-pre-wrap opacity-80")

        async def confirm() -> None:
            dialog.close()
            with busy(message="Unloading every model…"):
                try:
                    unloaded = await ctx.manager.unload_all()
                except Exception as exc:  # noqa: BLE001
                    notify_error(exc, what="unload all")
                    return
            ui.notify(f"unloaded {len(unloaded)} model(s)", type="positive")
            await refresh()

        with ui.row().classes("justify-end gap-2 w-full"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("Unload all", on_click=confirm).props("color=negative")
    dialog.open()


async def _restart_engines(ctx: GuiContext, refresh: Any) -> None:
    """Reload every loaded model's engine child, keeping the API up.

    Calls the management handler in-process rather than reimplementing its loop,
    so the GUI and the API cannot drift on what "restart the backend" means.
    """
    with busy(message="Restarting the inference engines…"):
        try:
            from studioforge.api.admin_routes import restart_backend

            payload = await restart_backend(api_request(ctx))
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="restart engines")
            return
    ui.notify(st.restart_backend_note(payload), type="positive", multi_line=True)
    await refresh()


def _restart_server_dialog(ctx: GuiContext, banner: Any) -> None:
    """Confirm, then hand the process off to the watchdog (or a respawn).

    Deliberately not a plain button: this is the one control on the panel that
    takes the gateway down, and every client talking to it sees a connection
    error. The dialog says so in those words.
    """
    with ui.dialog() as dialog, ui.card().classes("min-w-[30rem]"):
        ui.label("Restart the StudioForge server?").classes("font-medium")
        ui.label(st.RESTART_SERVER_WARNING).classes("text-sm opacity-80")

        async def confirm() -> None:
            dialog.close()
            banner.set_text("Restarting the server… this page will reconnect by itself.")
            try:
                from studioforge.api.admin_routes import restart_server

                # confirm=True is required by the endpoint; the dialog above is
                # what that confirmation means here.
                payload = await restart_server(api_request(ctx), confirm=True)
            except Exception as exc:  # noqa: BLE001
                banner.set_text("")
                notify_error(exc, what="restart server")
                return
            banner.set_text(st.restart_server_note(payload))
            ui.notify(st.restart_server_note(payload), type="warning", multi_line=True)

        with ui.row().classes("justify-end gap-2 w-full"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            ui.button("Restart server", on_click=confirm).props("color=negative")
    dialog.open()


# ---------------------------------------------------------------------------
# Live request log
# ---------------------------------------------------------------------------


def _log_panel(ctx: GuiContext) -> None:
    with ui.row().classes("items-center gap-3"):
        ui.label("Live log").classes("text-lg font-medium")
        level = ui.select(["ALL", "DEBUG", "INFO", "WARNING", "ERROR"], value="INFO").props(
            "dense outlined"
        )
    body = ui.label("").classes(
        "w-full font-mono text-xs whitespace-pre-wrap bg-black/10 dark:bg-white/5 p-2 rounded"
    )

    def refresh() -> None:
        from studioforge.logging import RING_BUFFER

        wanted = None if level.value == "ALL" else str(level.value)
        entries = RING_BUFFER.tail(LOG_TAIL_LINES, wanted)
        if not entries:
            body.set_text("(no log lines yet)")
            return
        body.set_text("\n".join(st.log_line_text(entry) for entry in entries))

    refresh()
    ui.timer(max(1.0, ctx.refresh_interval), refresh)
