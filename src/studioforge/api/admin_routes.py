"""Restart and bulk-unload controls.

Kept out of ``mgmt_routes`` because these are the operations that deliberately
take the server down, and it is worth them being obvious in one small file.

Two different meanings of "restart" are exposed, because they solve different
problems:

* **Restart the inference backend** -- stop and reload the ``llama-server``
  children while this process keeps running. That is what you want after an
  engine update, or when a model is misbehaving. Cheap, no downtime for the API.
* **Restart the server** -- the whole StudioForge process. Delegated to the
  watchdog sidecar when it is reachable, because a process cannot reliably
  restart itself: it has to die, and something that outlives it must bring it
  back. Falling back to a detached self-respawn covers the no-watchdog case.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from fastapi import APIRouter, Body, Request

from studioforge.api.auth import PIN_WITHHELD_NOTE, may_reveal_pin
from studioforge.core.ports import EXIT_RESTART_REQUESTED, supervised_by
from studioforge.errors import BadRequestError, ModelNotFoundError
from studioforge.logging import get_logger

log = get_logger(__name__)

router = APIRouter()


def _state(request: Request) -> Any:
    return request.app.state


#: Strong references to fire-and-forget restart tasks. The event loop keeps
#: only weak references to tasks, so a restart handed off after the HTTP reply
#: could otherwise be garbage-collected before it runs -- the caller would get
#: "restarting": true and nothing would restart.
_RESTART_TASKS: set[asyncio.Task[Any]] = set()


def _restart_task_done(task: asyncio.Task[Any]) -> None:
    """Make a restart task's death audible.

    The reply has already gone out saying ``{"restarting": true}``, so this task
    is the only thing that can still tell anyone the truth. Discarding it
    without inspecting it -- which is what a bare ``add_done_callback(set.discard)``
    does -- means an exception on the handoff path is retrievable by nobody and
    logged nowhere: the API promised a restart, the process stayed up and not a
    single line was written.
    """
    _RESTART_TASKS.discard(task)
    if task.cancelled():
        log.error("the restart task was cancelled; nothing restarted")
        return
    exc = task.exception()
    if exc is not None:
        log.error(
            "the restart task crashed; nothing restarted",
            error=f"{exc.__class__.__name__}: {exc}",
        )


def _spawn_restart_task(coro: Any) -> asyncio.Task[Any]:
    task = asyncio.create_task(coro)
    _RESTART_TASKS.add(task)
    task.add_done_callback(_restart_task_done)
    return task


def _record(state: Any, outcome: str, detail: str, **extra: Any) -> dict[str, Any]:
    """Remember how the last restart attempt went, for ``GET /api/restart/status``.

    ``POST /restart/server`` has to answer *before* the restart happens -- the
    process is about to die, so a reply sent afterwards would never arrive.
    That makes its 200 a statement of intent, and a failure after it has to land
    somewhere the operator can read. Previously the only trace of "the
    replacement died, so we stayed up" was one ERROR line in a log nobody was
    tailing, while the API kept answering as though the restart had happened.
    """
    status = {"outcome": outcome, "detail": detail, "at": time.time(), **extra}
    state.restart_status = status
    return status


def _watchdog_credential(config: Any) -> tuple[str | None, str]:
    """The bearer token the watchdog will accept from us, and what it is called.

    **This is the bug that made every watchdog restart fall through.** The
    watchdog enforces auth when *either* ``server.api_key`` or the MCP pairing
    PIN is set, and accepts either one. This call sent only ``server.api_key``,
    so on the default install -- no API key, PIN required, which is exactly this
    box -- it sent no credential at all, got a 401 from the watchdog's ASGI
    wrapper before any watchdog code ran (hence nothing in watchdog.log), and
    fell back to the self-respawn that cannot work while we hold the ports.
    """
    key = getattr(config.server, "api_key", None)
    if key:
        return key, "api_key"
    mcp = getattr(config, "mcp", None)
    pin = getattr(mcp, "pin", None) if mcp is not None else None
    if pin:
        return str(pin), "mcp_pin"
    return None, "none"


@router.post("/models/unload-all")
async def unload_all_models(request: Request) -> dict[str, Any]:
    """Unload every resident model, freeing all VRAM."""
    state = _state(request)
    unloaded = await state.manager.unload_all()
    log.info("unloaded all models", count=len(unloaded))
    return {"unloaded": unloaded, "count": len(unloaded)}


@router.post("/models/{model_id:path}/restart")
async def restart_model(model_id: str, request: Request) -> dict[str, Any]:
    """Reload one model's ``llama-server`` child with its current settings.

    A force-load rather than unload-then-load so the planner runs once against
    current free VRAM and the model is never left unloaded on a failure path.
    """
    state = _state(request)
    record = state.registry.resolve(model_id)
    if record is None:
        raise ModelNotFoundError(model_id, known=state.registry.known_ids())
    instance = await state.manager.load(record.id, force=True)
    return {
        "model_id": record.id,
        "restarted": True,
        "pid": instance.pid,
        "port": instance.port,
        "engine_tag": instance.engine_tag,
    }


@router.post("/restart/backend")
async def restart_backend(request: Request) -> dict[str, Any]:
    """Restart every loaded model's engine process, keeping the API up.

    Used after an engine update: running children keep the engine they were
    launched with, so a reload is what actually moves them onto the new build.
    """
    state = _state(request)
    loaded = [i.model_id for i in state.supervisor.list() if i.state == "ready"]
    restarted: list[str] = []
    failed: list[dict[str, str]] = []
    for model_id in loaded:
        try:
            await state.manager.load(model_id, force=True)
            restarted.append(model_id)
        except Exception as exc:
            log.warning("could not restart model", model_id=model_id, error=str(exc))
            failed.append({"model_id": model_id, "error": str(exc)})
    return {"restarted": restarted, "failed": failed, "count": len(restarted)}


@router.post("/restart/server")
async def restart_server(
    request: Request, confirm: bool = Body(False, embed=True)
) -> dict[str, Any]:
    """Restart the whole StudioForge process.

    Prefers the watchdog: a process cannot reliably restart itself, and the
    watchdog is a separate always-on sidecar that exists precisely to outlive
    this one. Without it, fall back to a detached respawn.
    """
    state = _state(request)
    if not confirm:
        raise BadRequestError(
            "restarting drops in-flight requests and unloads every model; pass confirm=true",
            param="confirm",
            code="confirmation_required",
        )

    config = state.config
    watchdog_url = f"http://127.0.0.1:{config.watchdog.port}"
    supervisor = supervised_by()
    log.info(
        "restart requested",
        watchdog_enabled=bool(config.watchdog.enabled),
        supervised_by=supervisor,
    )

    if supervisor == "tray":
        # The tray launched us and respawns a child that exits (D28). Exiting
        # with EXIT_RESTART_REQUESTED is the whole restart: no watchdog round
        # trip, a graceful drain, and exactly one process bringing us back.
        # Asking the watchdog instead would have it kill us and then defer to
        # the tray anyway; respawning ourselves would race the tray's respawn.
        _record(state, "supervisor-respawn", "exiting so the tray that launched us respawns us")
        _spawn_restart_task(_exit_for_supervisor(state))
        return {
            "restarting": True,
            "via": "tray",
            "note": "the API will be unavailable for a few seconds",
            "verify": (
                "GET /health for the new uptime_s; if this process is still here, "
                "GET /api/restart/status says why"
            ),
        }

    if config.watchdog.enabled:
        reachable, detail = await _watchdog_is_reachable(watchdog_url)
        if reachable:
            credential, kind = _watchdog_credential(config)
            log.info(
                "handing the restart to the watchdog",
                watchdog_url=watchdog_url,
                credential=kind,
            )
            if credential is None:
                # Not fatal: a watchdog with neither credential configured is
                # open by design. Worth saying out loud, because the reverse --
                # a credential the watchdog wants and we do not have -- is the
                # failure this whole path was silently hitting.
                log.info("no api_key or MCP pin configured; posting without a credential")
            # Hand off, then return before we are killed so the caller gets
            # a response rather than a dropped connection.
            _record(state, "handing-off", f"asking the watchdog at {watchdog_url} to restart us")
            _spawn_restart_task(_ask_watchdog_to_restart(state, watchdog_url))
            return {
                "restarting": True,
                "via": "watchdog",
                "watchdog_url": watchdog_url,
                "credential": kind,
                "note": "the API will be unavailable for a few seconds",
                "verify": (
                    "GET /health for the new uptime_s; if this process is still here, "
                    "GET /api/restart/status says why"
                ),
            }
        log.warning("watchdog unreachable, falling back to self-respawn", error=detail)

    reason = (
        "the watchdog is disabled"
        if not config.watchdog.enabled
        else "the watchdog did not answer"
    )
    _record(state, "self-respawn", f"{reason}; respawning this process")
    _spawn_restart_task(_self_restart(state))
    return {
        "restarting": True,
        "via": "self-respawn",
        "note": f"{reason}; restarting this process directly",
        "verify": (
            "GET /health for the new uptime_s; if this process is still here, "
            "GET /api/restart/status says why"
        ),
    }


@router.get("/restart/status")
async def restart_status(request: Request) -> dict[str, Any]:
    """How the last restart attempt on *this* process went.

    Exists because ``POST /restart/server`` must answer before it acts. When the
    restart works this route is unreachable -- the process it belongs to is
    gone, and a fresh one reports ``never``. So a non-``never`` answer from a
    live server is itself the diagnosis: the restart did not happen, and
    ``detail`` says why.
    """
    state = _state(request)
    status = getattr(state, "restart_status", None)
    if status is None:
        return {"outcome": "never", "detail": "no restart has been requested since this start"}
    return dict(status)


async def _watchdog_is_reachable(watchdog_url: str) -> tuple[bool, str]:
    """Whether the watchdog answers ``/health``. Never raises.

    A 5xx means the watchdog is there but unwell; anything else it says (200
    ``up``, or 503 because the main server it watches looks down) is a watchdog
    that can take the handoff.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            health = await client.get(f"{watchdog_url}/health")
    except httpx.HTTPError as exc:
        return False, f"{exc.__class__.__name__}: {exc}"
    if health.status_code >= 500 and health.status_code != 503:
        return False, f"the watchdog answered /health with {health.status_code}"
    return True, f"the watchdog answered /health with {health.status_code}"


async def _ask_watchdog_to_restart(state: Any, watchdog_url: str) -> None:
    """Ask the watchdog to restart us, after letting our reply flush.

    Uses the watchdog's plain ``POST /restart`` rather than its MCP tool. A
    JSON-RPC ``tools/call`` over streamable-HTTP is only valid after an
    ``initialize`` handshake that establishes a session id; posting one cold
    gets a protocol error back, which this used to discard -- so the endpoint
    answered ``{"restarting": true}`` while nothing restarted.

    The credential comes from :func:`_watchdog_credential`, which falls back to
    the MCP pairing PIN. Sending only ``server.api_key`` meant that on a default
    install this request was answered ``401`` by the watchdog's ASGI wrapper --
    before any watchdog code ran, which is why ``watchdog.log`` showed nothing
    at all -- and every restart fell through to the self-respawn path.

    If the handoff fails for any reason we fall back to respawning ourselves,
    because the caller has already been told the process is going down.
    """
    await asyncio.sleep(0.5)
    credential, kind = _watchdog_credential(state.config)
    headers = {"Authorization": f"Bearer {credential}"} if credential else {}
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{watchdog_url}/restart", headers=headers)
        if response.status_code < 400:
            log.info("watchdog accepted the restart handoff", status=response.status_code)
            _record(state, "handed-off", "the watchdog accepted the restart and is killing us")
            return
        hint = (
            f" (we sent {kind}; the watchdog wants server.api_key or the MCP pin)"
            if response.status_code == 401
            else ""
        )
        log.error(
            "watchdog refused the restart handoff; respawning instead",
            status=response.status_code,
            credential=kind,
            body=response.text[:300],
        )
        _record(
            state,
            "watchdog-refused",
            f"the watchdog answered {response.status_code}{hint}; falling back to self-respawn",
            status=response.status_code,
            credential=kind,
        )
    except Exception as exc:  # noqa: BLE001 - any failure must reach the fallback
        log.error(
            "watchdog restart call failed; respawning instead",
            error=f"{exc.__class__.__name__}: {exc}",
        )
        _record(
            state,
            "watchdog-unreachable",
            f"the handoff failed ({exc.__class__.__name__}: {exc}); falling back to self-respawn",
        )
    await _self_restart(state)


#: Long enough to catch a replacement that dies on import or config, short
#: enough that we hand the ports over before its wait budget runs out. The
#: replacement no longer fails on a busy port -- it waits for ours (D21).
_RESPAWN_SETTLE_S = 3.0


async def _self_restart(state: Any) -> None:
    """Drain, respawn detached, then exit this process.

    The replacement is spawned with ``SF_RESPAWN_PARENT_PID`` set to our pid, so
    its port preflight *waits* for the ports we are about to release instead of
    refusing to start on them. Before that, this could never succeed: we hold
    the API and GUI ports (and our watchdog child holds its own) at the exact
    moment the replacement checks them, so it exited rc 3 every time and we
    stayed up -- after the API had already answered ``{"restarting": true}``.
    """
    from studioforge.core.updater import Updater

    await asyncio.sleep(0.5)
    try:
        await state.manager.stop()
    except Exception as exc:  # pragma: no cover - best effort
        log.warning("drain before restart failed", error=str(exc))

    updater = Updater(state.config)
    if not updater._respawn_detached():
        log.error("could not respawn; not exiting, the server stays up")
        _record(state, "failed", "could not start a replacement process; this server stays up")
        await _undrain(state)
        return

    # `_respawn_detached` only proves Popen succeeded. Give the child a moment
    # to fail on something we cannot fix by exiting -- a bad config, a broken
    # import -- before we make ourselves unrecoverable.
    await asyncio.sleep(_RESPAWN_SETTLE_S)
    child = getattr(updater, "_last_child", None)
    if child is not None and child.poll() is not None:
        log.error(
            "the replacement process exited immediately; staying up",
            returncode=child.returncode,
        )
        _record(
            state,
            "failed",
            (
                f"restart did not happen: the replacement process exited immediately "
                f"(rc={child.returncode}). This server stayed up. Check the log for "
                f"'startup port conflict' and for a config error."
            ),
            returncode=child.returncode,
        )
        await _undrain(state)
        return
    log.info(
        "respawned; this process is exiting",
        child_pid=getattr(child, "pid", None),
        wait_budget_s=updater.respawn_wait_s,
    )
    _record(state, "exiting", "a replacement was started and this process is exiting")
    # Leave the watchdog for the replacement to adopt: killing it here would
    # take the supervisor down with us and free its port a moment too late (D21).
    state.handing_over = True
    _shutdown_this_process(state)


async def _exit_for_supervisor(state: Any) -> None:
    """Drain, then exit with ``EXIT_RESTART_REQUESTED`` for the tray to respawn us.

    The tray launched this process and respawns a child that exits (D28); the
    exit code tells it this exit was asked for. The watchdog is left running
    for the replacement to adopt (D21), exactly as in a self-respawn.
    """
    await asyncio.sleep(0.5)
    try:
        await state.manager.stop()
    except Exception as exc:  # pragma: no cover - best effort
        log.warning("drain before restart failed", error=str(exc))
    state.handing_over = True
    state.exit_code = EXIT_RESTART_REQUESTED
    _record(state, "exiting", "drained; exiting for the tray to respawn this process")
    _shutdown_this_process(state)


def _shutdown_this_process(state: Any) -> None:
    """Stop the servers gracefully, the way Ctrl+C would.

    ``state.request_shutdown`` is installed by ``__main__._serve`` and flips
    uvicorn's ``should_exit`` on both servers, which runs the drain and the
    lifespan shutdown on every platform. The signal it replaces was only a
    signal on POSIX: on Windows ``os.kill(pid, SIGINT)`` is ``TerminateProcess``
    -- verified on this box: a handler installed for SIGINT never runs and
    the process exits 2 -- so the restart paths skipped every ``finally`` and
    exited with a code the tray counted as a crash. The signal stays as the
    fallback for an embedder that composed the app without ``_serve``.
    """
    import os
    import signal

    request_shutdown = getattr(state, "request_shutdown", None)
    if callable(request_shutdown):
        request_shutdown()
        return
    os.kill(os.getpid(), signal.SIGTERM if os.name != "nt" else signal.SIGINT)


async def _undrain(state: Any) -> None:
    """Put the drain flag back down after a restart that did not happen.

    ``manager.stop()`` latches ``draining`` for a shutdown that is now not
    coming. Left set, ``/health`` reports ``draining: true`` forever on a server
    that is still taking requests and loading models -- the flag an operator
    reads to decide whether killing the process is safe, permanently lying.
    """
    resume = getattr(state.manager, "resume", None)
    if resume is None:  # pragma: no cover - older manager
        return
    try:
        await resume()
    except Exception as exc:  # pragma: no cover - best effort
        log.warning("could not clear the drain flag", error=str(exc))


@router.get("/capabilities")
async def capabilities(request: Request, check_update: bool = False) -> dict[str, Any]:
    """What this backend supports, what the hardware allows, and what is newer.

    Answers "what kinds of model can I actually run?" in one call, for the GUI,
    the CLI and MCP alike.
    """
    from studioforge.core.capabilities import build_report

    state = _state(request)
    report = build_report(
        state.config,
        gpus=state.probe.list_gpus(),
        records=state.registry.all(),
        engine_manager=state.engine_manager,
        probe=state.probe,
    )
    payload = report.to_dict()

    payload["update"] = {"checked": False}
    if check_update:
        # Network call, so opt-in: the GUI paints the panel first and refreshes
        # this separately rather than blocking the whole page on GitHub.
        try:
            # EngineManager.check_update owns the whole question: it filters
            # prereleases and non-build tags out of the release list, confirms
            # the tag it names has an asset this driver can run, and compares
            # build numbers rather than strings. Reconstructing any of that here
            # is how the three copies of this check drifted apart.
            payload["update"] = await state.engine_manager.check_update(limit=5)
        except Exception as exc:
            log.warning("engine release check failed", error=str(exc))
            payload["update"] = {"checked": True, "error": str(exc)}
    return payload


@router.get("/mcp/info")
async def mcp_info(request: Request) -> dict[str, Any]:
    """Where to reach the MCP endpoint, and the PIN to pair with it.

    Advertises the **Tailscale** address first because that is the one that
    keeps working: a tailnet IP survives a network change, a DHCP renewal and a
    move between sites, where a LAN address silently stops resolving. The LAN
    addresses are returned alongside so a client on the same network can take
    the shorter, faster direct hop when it wants one.
    """
    from studioforge.core.netinfo import reachable_urls

    state = _state(request)
    config = state.config
    mcp = config.mcp

    endpoints = reachable_urls(config.server.port, mcp.path, host=config.server.host)
    tailscale = [e for e in endpoints if e["kind"] == "tailscale"]
    lan = [e for e in endpoints if e["kind"] in {"lan", "bound"}]
    loopback = [e for e in endpoints if e["kind"] == "loopback"]

    # Returning the PIN in full is what makes "pair a new client" a single
    # request -- but only when reaching this route actually cost a credential.
    # With server.api_key unset (the shipped default) it costs nothing, and the
    # PIN is then the only thing standing in front of the MCP control plane.
    reveal = may_reveal_pin(request, config)
    payload: dict[str, Any] = {
        "enabled": mcp.enabled,
        "path": mcp.path,
        "transport": "streamable-http",
        "pin": (mcp.pin if mcp.pin_required else None) if reveal else None,
        "pin_required": bool(mcp.pin_required and mcp.pin),
        "auth": {
            "header": "X-MCP-Pin",
            "alternatives": ["Authorization: Bearer <pin>", "?pin=<pin>"],
            "api_key_also_accepted": bool(config.server.api_key),
        },
        "recommended": (tailscale or lan or loopback or [{}])[0].get("url"),
        "endpoints": endpoints,
        "tailscale": tailscale,
        "lan": lan,
        "loopback": loopback,
        "watchdog": {
            "enabled": config.watchdog.enabled,
            "endpoints": reachable_urls(config.watchdog.port, "/mcp", host=config.watchdog.host),
        },
    }
    if not reveal and mcp.pin_required and mcp.pin:
        payload["pin_note"] = PIN_WITHHELD_NOTE
    return payload
