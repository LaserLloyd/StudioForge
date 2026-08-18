"""FastAPI application factory and wiring.

The app object carries the composed system on ``app.state`` (config, db,
registry, planner, supervisor, manager, engine manager, downloader, http
client) so route handlers stay thin and tests can build the whole stack with
substituted parts.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from studioforge import __version__
from studioforge.api import admin_routes, health_routes, mgmt_routes, openai_routes
from studioforge.api.auth import check_request
from studioforge.config import Config, load_config
from studioforge.core.downloader import Downloader
from studioforge.core.engine import EngineManager
from studioforge.core.gpu import get_probe
from studioforge.core.health import PROBE_MAX_TOKENS, PROBE_TIMEOUT_S, deep_health
from studioforge.core.manager import ModelManager
from studioforge.core.planner import Planner
from studioforge.core.registry import Registry
from studioforge.core.supervisor import Supervisor
from studioforge.core.updater import Updater
from studioforge.db import Database
from studioforge.errors import StudioForgeError
from studioforge.logging import configure_logging, get_logger

log = get_logger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Applies the optional bearer key to every route in one place."""

    def __init__(self, app: Any, config: Config) -> None:
        super().__init__(app)
        self.config = config

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        try:
            check_request(request, self.config)
        except StudioForgeError as exc:
            return JSONResponse(exc.to_payload(), status_code=exc.status_code)
        return await call_next(request)


#: Strong references to in-flight post-download scan tasks, keyed by download
#: group. Two jobs in one: the event loop holds only weak references to tasks,
#: so without this a scan scheduled by the completion callback could be
#: garbage-collected mid-flight and simply never run -- the download would say
#: "completed" and the model would stay invisible. And keying by group
#: deduplicates: re-enqueueing an already-complete multi-shard group emits one
#: "completed" event per file, which would otherwise queue one full library
#: scan per shard.
_RESCAN_TASKS: dict[str, asyncio.Task[Any]] = {}


def _forget_rescan_task(task: asyncio.Task[Any], *, group_id: str) -> None:
    if _RESCAN_TASKS.get(group_id) is task:
        _RESCAN_TASKS.pop(group_id, None)


def _file_in_use_check(supervisor: Any, registry: Any) -> Any:
    """"Does a loaded model have this file open?" for the downloader.

    A forced re-download unlinks the destination first, and the quarantine path
    renames it; doing either to a file llama-server has mmapped is worse than
    refusing the download.

    **This guard did not work.** It called ``supervisor.all()``, which does not
    exist -- the method is ``supervisor.list()``. Every call therefore raised
    ``AttributeError``, and at the one call site that mattered
    (``Downloader.enqueue`` with ``force=True``) the exception either propagated
    as an opaque 500 or, worse, the guard was simply never reached, so a forced
    download could unlink the weights of a *running* model. Nothing failed
    loudly enough to be noticed, because the guard only fires on a path nobody
    exercises until the day it matters.

    A raising predicate must not decide "not in use" by accident either, so the
    supervisor and registry calls are wrapped: an unreadable registry answers
    *conservatively*. Returning False on an error would mean "go ahead, delete
    it", which is the wrong default for a question about live weights.
    """

    def check(path: Any) -> bool:
        try:
            target = Path(path).resolve()
        except OSError:  # pragma: no cover - unresolvable path
            return False
        try:
            instances = supervisor.list()
        except Exception as exc:  # noqa: BLE001 - see the docstring: be conservative
            log.warning("could not list loaded models; treating the file as in use", error=str(exc))
            return True
        for instance in instances:
            try:
                record = registry.resolve(instance.model_id)
            except Exception:  # noqa: BLE001 - one unresolvable model is not an answer
                continue
            if record is None:
                continue
            for candidate in [*getattr(record, "shards", []), record.mmproj_path]:
                if candidate is None:
                    continue
                try:
                    if Path(candidate).resolve() == target:
                        return True
                except OSError:  # pragma: no cover
                    continue
        return False

    return check


def rescan_when_group_completes(downloader: Any, registry: Any) -> Any:
    """Progress callback that rescans the registry when a download group finishes.

    Without this, a completed download sits on disk invisible to ``/v1/models``,
    the Models tab and the MCP ``list_models`` tool until someone happens to run
    a manual scan -- while the queue says "completed" and the ``download_model``
    tool's own docstring promises the model will appear. The scan runs off the
    event loop (it stats and parses GGUF headers) and only when the *whole*
    group is complete: a model missing shard 3 of 5 is not a model.

    The callback must never raise: the downloader drops a subscriber that
    throws, which would silently disable this for every later download.
    """
    from studioforge.core.downloader import DownloadProgress

    def _on_progress(progress: DownloadProgress) -> None:
        if progress.status != "completed":
            return
        try:
            if downloader.group_status(progress.group_id) != "completed":
                return
        except Exception:  # noqa: BLE001 - a status hiccup must not kill the subscriber
            return

        def _scan() -> None:
            try:
                result = registry.scan()
                log.info(
                    "registry rescanned after download",
                    group_id=progress.group_id,
                    added=len(result.added),
                )
            except Exception as exc:  # noqa: BLE001 - never propagate into the transfer task
                log.warning(
                    "post-download registry scan failed; run a manual scan",
                    group_id=progress.group_id,
                    error=str(exc),
                )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _scan()
        else:
            pending = _RESCAN_TASKS.get(progress.group_id)
            if pending is not None and not pending.done():
                return
            task = loop.create_task(asyncio.to_thread(_scan))
            _RESCAN_TASKS[progress.group_id] = task
            task.add_done_callback(
                functools.partial(_forget_rescan_task, group_id=progress.group_id)
            )

    return _on_progress


async def _reclaim_orphaned_engines(state: Any) -> None:
    """Kill leaked ``llama-server`` processes before this server does anything.

    Runs after the supervisor exists and **before** models are auto-loaded, for
    the obvious reason: a leaked child holding 15 GiB makes the first pinned
    load fail with an insufficient-VRAM rejection that names no cause.

    Only orphans are killed -- our engine binary, our engines directory, and a
    dead parent. That combination cannot be anyone else's process, so the sweep
    is safe by construction; a llama-server belonging to a *live* process
    (another instance, a test run) is left alone and shows up in
    ``/api/vram/holders`` instead. See DECISIONS.md D23.

    Never fatal. A boot that stops because the safety sweep failed would be a
    worse bug than the leak it was cleaning up.
    """
    from studioforge.core.vram_holders import reclaim_orphans

    try:
        actions = await asyncio.to_thread(
            reclaim_orphans,
            state.config.engines_dir,
            own_pids=state.supervisor.child_pids(),
        )
    except Exception as exc:  # noqa: BLE001 - a failed sweep must not stop startup
        log.warning("orphan engine sweep failed", error=str(exc))
        return
    if actions:
        log.warning(
            "reclaimed orphaned llama-server processes left over from a previous run",
            count=len(actions),
            killed=sum(1 for action in actions if action.get("killed")),
            pids=[action["pid"] for action in actions],
        )


def build_state(config: Config, *, version: str = __version__) -> Any:
    """Compose the object graph. Separated from the app so tests can reuse it."""

    class State:
        pass

    state = State()
    config.ensure_dirs()

    db = Database(config.db_path)
    # Recovery, not plain migrate(): a corrupt registry.sqlite3 must degrade
    # the server (fresh DB, corrupt file kept aside), never stop it booting.
    db.migrate_with_recovery()

    probe = get_probe()
    engine_manager = EngineManager(config, probe=probe)
    registry = Registry(config, db)
    planner = Planner(
        config,
        probe,
        observation_sink=lambda row: db.record_load_observation(**row),
    )
    # The probe goes in so an unload can log the VRAM it actually reclaimed
    # rather than asserting that it did.
    supervisor = Supervisor(config, resolve_binary=engine_manager.server_binary, probe=probe)
    manager = ModelManager(
        config,
        registry=registry,
        planner=planner,
        supervisor=supervisor,
        db=db,
        version=version,
    )

    state.config = config  # type: ignore[attr-defined]
    state.db = db  # type: ignore[attr-defined]
    state.probe = probe  # type: ignore[attr-defined]
    state.engine_manager = engine_manager  # type: ignore[attr-defined]
    state.registry = registry  # type: ignore[attr-defined]
    state.planner = planner  # type: ignore[attr-defined]
    state.supervisor = supervisor  # type: ignore[attr-defined]
    state.manager = manager  # type: ignore[attr-defined]
    downloader = Downloader(config, db)
    # A finished download must show up in /v1/models without a manual scan.
    downloader.subscribe(rescan_when_group_completes(downloader, registry))
    downloader.set_in_use_check(_file_in_use_check(supervisor, registry))
    state.downloader = downloader  # type: ignore[attr-defined]
    state.updater = Updater(config)  # type: ignore[attr-defined]
    state.started_at = time.time()  # type: ignore[attr-defined]
    state.version = version  # type: ignore[attr-defined]
    # Defaults, so every composed state answers the question even when nobody
    # asked for the lock. `create_app` overwrites these when it is the process
    # that will actually run the background workers -- see _claim_data_dir.
    state.instance_lock = None  # type: ignore[attr-defined]
    state.instance_role = "primary"  # type: ignore[attr-defined]
    return state


def _claim_data_dir(config: Config) -> tuple[Any, str]:
    """Take exclusive ownership of the data directory, or report that we cannot.

    Returns ``(lock_or_None, role)`` where role is ``"primary"`` or
    ``"secondary"``.

    **Why.** On 2026-08-18 a test process built a second ``create_app`` against
    the live data directory. Its Downloader read the live queue out of
    ``registry.sqlite3`` and started writing the same ``.part`` the live server
    was streaming into; the live transfer died with ``WinError 32`` and the
    other writer published an interleaved 22.58 GB file. Every background worker
    in this process has the same shape of hazard -- the TTL sweeper evicts
    models it does not own, the orphan sweep inspects children it did not spawn,
    auto-load starts engines on GPUs another instance is planning around.

    So a second instance still *builds*: it serves reads, and its ``/health``
    says plainly what it is. What it does not do is act. The port preflight
    normally catches a duplicate ``serve`` long before this, which is exactly
    why this matters -- the case it misses is the in-process embedder, a test or
    a script that never binds a port at all.

    Acquisition is never fatal: a data directory on a filesystem with no working
    locks (some network shares) must degrade to today's behaviour, loudly,
    rather than refuse to start.
    """
    from studioforge.core.instance_lock import InstanceLock

    try:
        lock = InstanceLock(config.data_dir)
        if lock.acquire():
            return lock, "primary"
    except OSError as exc:
        log.warning(
            "could not take the instance lock; continuing without it",
            data_dir=str(config.data_dir),
            error=str(exc),
        )
        return None, "primary"
    holder = lock.holder() or {}
    log.error(
        "another StudioForge instance owns this data directory: background workers "
        "(download resume, TTL sweeper, auto-load, orphan sweep) will NOT start in this "
        "process. Point this one at a different data_dir, or stop the other instance.",
        data_dir=str(config.data_dir),
        holder_pid=holder.get("pid"),
        holder_started_at=holder.get("started_at"),
        lock_path=str(lock.path),
    )
    return lock, "secondary"


def create_app(
    config: Config | None = None,
    *,
    state: Any = None,
    start_background: bool = True,
) -> FastAPI:
    config = config or load_config(create=True)
    configure_logging(
        config.logging.level, json_logs=config.logging.json_logs, log_dir=config.logs_dir
    )

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.server.request_timeout_s, connect=10.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )
        # Off the event loop (a cold scan of a large library stats and parses
        # every GGUF header), and never fatal: a failing scan means an empty
        # model list and a loud log line, not a server that will not boot.
        try:
            scan = await asyncio.to_thread(app.state.registry.scan)
        except Exception as exc:
            log.error(
                "model scan failed at startup; serving with an empty model list",
                error=str(exc),
            )
        else:
            log.info(
                "model scan complete",
                added=len(scan.added),
                removed=len(scan.removed),
                unchanged=scan.unchanged,
                errors=len(scan.errors),
                duration_s=round(scan.duration_s, 2),
            )
            for path, error in scan.errors[:10]:
                log.warning("model scan error", path=path, error=error)

        # A secondary instance does nothing in the background at all -- no
        # orphan sweep, no auto-load, no TTL sweeper, and above all no download
        # resume. See _claim_data_dir for the incident this is guarding.
        secondary = getattr(app.state, "instance_role", "primary") != "primary"
        if start_background and secondary:
            log.warning(
                "background workers are disabled in this process because another "
                "instance owns the data directory",
                holder_pid=(getattr(app.state, "instance_holder", None) or {}).get("pid"),
            )
        if start_background and not secondary:
            await _reclaim_orphaned_engines(app.state)
            try:
                engine = await app.state.engine_manager.ensure_engine()
                log.info("engine ready", tag=engine.tag, variant=engine.variant)
            except Exception as exc:
                log.error("engine not ready", error=str(exc))
            await app.state.manager.start()
            # Resumes anything a crash left half-downloaded.
            with contextlib.suppress(Exception):
                await app.state.downloader.start()

        try:
            yield
        finally:
            with contextlib.suppress(Exception):
                await app.state.downloader.stop()
            with contextlib.suppress(Exception):
                await app.state.manager.stop()
            with contextlib.suppress(Exception):
                await app.state.client.aclose()
            # The engine manager and updater lazily open their own httpx
            # clients (GitHub release checks); without these they leak and
            # warn "Unclosed client session" at every shutdown.
            with contextlib.suppress(Exception):
                await app.state.engine_manager.aclose()
            with contextlib.suppress(Exception):
                await app.state.updater.aclose()
            with contextlib.suppress(Exception):
                app.state.db.close()
            # Last, and after the DB: the lock is the statement "this process is
            # still using this directory", and it must outlive everything that
            # is still using it.
            if getattr(app.state, "instance_lock", None) is not None:
                with contextlib.suppress(Exception):
                    app.state.instance_lock.release()

    app = FastAPI(
        title="StudioForge",
        version=__version__,
        description="GPU-only OpenAI-compatible LLM gateway over llama.cpp llama-server",
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    composed = state if state is not None else build_state(config)
    for key, value in vars(composed).items():
        setattr(app.state, key, value)
    app.state.client = None

    # Claimed here rather than in build_state because the lock is about *acting*
    # on a data directory, not about composing an object graph: the stdio MCP
    # server and every `start_background=False` test build the same graph and
    # must not be able to lock the live server out of its own downloads.
    app.state.instance_holder = None
    if start_background:
        lock, role = _claim_data_dir(config)
        app.state.instance_lock = lock if role == "primary" else None
        app.state.instance_role = role
        if role != "primary" and lock is not None:
            app.state.instance_holder = lock.holder()

    app.add_middleware(AuthMiddleware, config=config)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=config.server.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(openai_routes.router, tags=["openai"])
    app.include_router(mgmt_routes.router, prefix="/api", tags=["management"])
    app.include_router(admin_routes.router, prefix="/api", tags=["admin"])
    app.include_router(health_routes.router, prefix="/api", tags=["health"])

    # Management-plane MCP over streamable HTTP, behind the same API key.
    # Failure here must not stop the server from serving inference.
    try:
        from studioforge.mcp.management import mount_management_mcp

        # Follow the CONFIGURED path. Hardcoding "/mcp" while auth
        # (`is_mcp_path`) and `/api/mcp/info` both honoured `mcp.path` meant a
        # customised path mounted MCP where the PIN was not accepted, and
        # advertised a URL that 404s -- defeating the PIN entirely.
        mount_management_mcp(app, app.state, path=config.mcp.path or "/mcp")
    except Exception as exc:
        log.error("management MCP not mounted", error=str(exc))

    _install_error_handlers(app)

    @app.get("/health", tags=["health"])
    async def health(
        deep: bool = Query(
            False,
            description=(
                "Run a real streamed completion against every loaded model "
                "instead of only reporting that this process is up."
            ),
        ),
        timeout_s: float | None = Query(None, gt=0, le=300),
    ) -> dict[str, Any]:
        """Liveness by default; genuine generation when ``deep=true``.

        The shallow answer stays the default because the watchdog and load
        balancers poll it constantly, and it must stay free. But shallow is
        also exactly the check that lied for hours during the incident behind
        the deep mode -- a 200 here proves this process is alive and nothing
        more. ``?deep=true`` is the one that proves models generate.
        """
        loaded = app.state.supervisor.list()
        payload: dict[str, Any] = {
            "status": "ok",
            "version": __version__,
            "uptime_s": round(time.time() - app.state.started_at, 1),
            "loaded_models": [i.model_id for i in loaded],
            "draining": app.state.manager.draining,
            # "secondary" means another process owns this data directory, so
            # this one runs no background work: downloads do not resume, models
            # are not evicted on TTL and nothing is auto-loaded. A poller that
            # cannot see this reads a perfectly healthy 200 from a process that
            # is deliberately inert, which is precisely the confusion the
            # 2026-08-18 incident produced.
            "instance": getattr(app.state, "instance_role", "primary"),
        }
        holder = getattr(app.state, "instance_holder", None)
        if payload["instance"] != "primary" and holder:
            payload["instance_holder_pid"] = holder.get("pid")
        # A restart that did not happen is invisible otherwise: the API answered
        # {"restarting": true} and this process is still the one replying. Say
        # so here, where the watchdog and every poller will see it.
        restart = getattr(app.state, "restart_status", None)
        if restart is not None and restart.get("outcome") == "failed":
            payload["restart_failed"] = restart
        if not deep:
            return payload
        gateway = app.state.config.gateway
        result = await deep_health(
            app.state.supervisor,
            app.state.client,
            registry=app.state.registry,
            timeout_s=timeout_s
            if timeout_s is not None
            else getattr(gateway, "deep_probe_timeout_s", PROBE_TIMEOUT_S),
            max_tokens=getattr(gateway, "deep_probe_max_tokens", PROBE_MAX_TOKENS),
        )
        payload["status"] = result.status
        payload["probe"] = result.model_dump(mode="json")
        return payload

    return app


def _install_error_handlers(app: FastAPI) -> None:
    """Every error leaves as an OpenAI-shaped envelope.

    The ``openai`` client builds its exception hierarchy by parsing this
    envelope, so a FastAPI/Starlette default error body would surface to users
    as an unhelpful generic APIError.
    """

    @app.exception_handler(StudioForgeError)
    async def _studioforge_error(_request: Request, exc: StudioForgeError) -> JSONResponse:
        if exc.status_code >= 500:
            log.error("request failed", error=exc.message, code=exc.code, type=exc.error_type)
        else:
            log.info("request rejected", error=exc.message, code=exc.code)
        headers: dict[str, str] = {}
        # A 503 here means "busy / transient", so tell the client how long to
        # wait instead of leaving it to guess a backoff (and hammer us).
        if exc.status_code == 503:
            headers["Retry-After"] = str(exc.details.get("retry_after_s", 5))
        return JSONResponse(exc.to_payload(), status_code=exc.status_code, headers=headers)

    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        error_type = "invalid_request_error" if exc.status_code < 500 else "server_error"
        return JSONResponse(
            {
                "error": {
                    "message": str(exc.detail),
                    "type": error_type,
                    "code": None,
                    "param": None,
                }
            },
            status_code=exc.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            {
                "error": {
                    "message": f"Invalid request: {exc.errors()}",
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                    "param": None,
                }
            },
            status_code=400,
        )

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error", error=str(exc))
        return JSONResponse(
            {
                "error": {
                    "message": f"Internal server error: {exc}",
                    "type": "server_error",
                    "code": "internal_error",
                    "param": None,
                }
            },
            status_code=500,
        )
