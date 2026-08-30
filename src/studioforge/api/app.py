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
from studioforge.core.leases import LeaseBook
from studioforge.core.manager import ModelManager
from studioforge.core.planner import Planner
from studioforge.core.registry import Registry
from studioforge.core.supervisor import Supervisor
from studioforge.core.updater import Updater
from studioforge.db import Database
from studioforge.errors import StudioForgeError
from studioforge.logging import configure_logging, get_logger

log = get_logger(__name__)


#: Paths that answer during the boot without waiting for the first library
#: scan: liveness for the watchdog/tray/load balancers, and the API docs.
_NO_BOOT_WAIT_PATHS = frozenset({"/health", "/healthz", "/api/health", "/docs", "/openapi.json"})

#: How long a request waits for the boot's first library scan (D33). A cold
#: scan of a large library is tens of seconds; past this the request proceeds
#: against whatever is indexed so far, and /health says why.
SCAN_WAIT_S = 60.0


class AuthMiddleware(BaseHTTPMiddleware):
    """Applies the optional bearer key to every route in one place -- and holds
    every request but liveness until the boot's first library scan is in (D33),
    so a client that connects the moment the port answers is not told the
    library is empty or a model does not exist."""

    def __init__(self, app: Any, config: Config) -> None:
        super().__init__(app)
        self.config = config

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        try:
            check_request(request, self.config)
        except StudioForgeError as exc:
            headers: dict[str, str] = {}
            if exc.status_code == 429 and exc.details.get("retry_after_s"):
                # The middleware short-circuits before the exception handler
                # below ever runs, so the header has to be set here too or a
                # locked-out client gets the wait only in the JSON body.
                headers["Retry-After"] = str(exc.details["retry_after_s"])
            return JSONResponse(exc.to_payload(), status_code=exc.status_code, headers=headers)
        if request.url.path not in _NO_BOOT_WAIT_PATHS:
            await wait_for_boot(request.app.state, timeout_s=SCAN_WAIT_S, scan_only=True)
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
    """ "Does a loaded model have this file open?" for the downloader.

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


def _tag_in_use_check(supervisor: Any, engine_manager: Any) -> Any:
    """ "Which loaded models are running this engine build?" for the installer.

    A reinstall over an already-present tag re-extracts the zip over a directory
    whose ``llama-server.exe`` a running child holds open: on Windows that is a
    ``WinError 32`` partway through the extraction, which is how a live engine
    directory ends up half-overwritten (D49-7). The installer refuses instead,
    naming the models that would have to be unloaded first, and this is where it
    gets the names.

    **The attribution is exact since D50**, and it was not before. An instance
    used to record only the tag it was *asked* for, and the overwhelmingly
    common load asks for nothing -- ``engine_tag`` is ``None``, "whatever was
    active at spawn" -- with no record of what that turned out to be, so a
    ``None`` was attributed to whatever is active *now*. Right up until somebody
    activated a different build: a child launched from b1 and still running it
    was then counted against b2, and reinstalling b1 over its own open
    ``llama-server.exe`` was not refused. The supervisor now stamps
    :attr:`~studioforge.types.InstanceInfo.resolved_engine_tag` at spawn, so
    that guess is only the fallback for an instance predating the stamp (or one
    whose engine could not be identified), where the old approximation stands.

    Failure is answered conservatively, like :func:`_file_in_use_check`: a
    supervisor that cannot be listed reports a placeholder holder rather than
    "nothing is using it", because the wrong answer here overwrites a running
    binary.
    """

    def in_use(tag: str) -> list[str]:
        wanted = str(tag)
        try:
            instances = supervisor.list()
        except Exception as exc:  # noqa: BLE001 - see the docstring: be conservative
            log.warning(
                "could not list loaded models; treating the engine as in use",
                engine_tag=wanted,
                error=str(exc),
            )
            return ["(could not list loaded models)"]
        try:
            active = engine_manager.active()
            active_tag = active.tag if active is not None else None
        except Exception as exc:  # noqa: BLE001 - an unreadable active.json is not fatal
            log.warning("could not read the active engine", error=str(exc))
            active_tag = None
        holders: list[str] = []
        for instance in instances:
            launched_with = (
                getattr(instance, "resolved_engine_tag", None)
                or getattr(instance, "engine_tag", None)
                or active_tag
            )
            if launched_with == wanted:
                holders.append(instance.model_id)
        return holders

    return in_use


def _engine_tag_resolver(engine_manager: Any) -> Any:
    """ "Which build does a request for this tag actually land on?" (D50).

    The supervisor stamps the answer on every child it starts, so "activate +
    reload" can skip the residents that are already current instead of
    restarting all of them -- twice, when the button is double-clicked.

    ``None`` in means "whatever is active", which is what nearly every load
    asks for; a pinned tag is looked up so that a pin naming a build that is
    not installed comes back as itself rather than as the active one -- the
    spawn is going to fail on that pin a moment later, and the recorded tag
    should say what it tried, not what it would have used instead. ``None`` out
    means the engine could not be identified at all, which every reader treats
    as "cannot prove this child is current".
    """

    def resolve(tag: str | None) -> str | None:
        try:
            info = engine_manager.get(tag) if tag else engine_manager.active()
        except Exception as exc:  # noqa: BLE001 - naming a build is never fatal
            log.warning("could not resolve the engine tag", engine_tag=tag, error=str(exc))
            return tag
        if info is None:
            return tag
        return str(info.tag)

    return resolve


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


class BootStatus:
    """Where startup is, for /health and for the routes that wait on it.

    ``phase`` is a short human string (``"scanning models"``, ``"installing
    engine b10425"``, ``"ready"``); ``scanned`` fires when the registry has its
    first index and ``done`` when the whole boot has run -- successfully or
    not, so nothing that waits on it can wait forever.
    """

    def __init__(self) -> None:
        self.phase = "starting"
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.error: str | None = None
        self.scanned = asyncio.Event()
        self.done = asyncio.Event()

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        log.info("boot phase", phase=phase, elapsed_s=round(time.time() - self.started_at, 1))

    def set_progress(self, phase: str) -> None:
        """Update the phase text without a log line (per-tick progress)."""
        if self.phase != phase:
            self.phase = phase

    def finish(self, error: str | None = None) -> None:
        self.error = error
        self.finished_at = time.time()
        self.phase = "ready" if error is None else f"failed: {error}"
        self.scanned.set()
        self.done.set()

    @property
    def ready(self) -> bool:
        return self.done.is_set()

    def snapshot(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "ready": self.ready,
            "elapsed_s": round((self.finished_at or time.time()) - self.started_at, 1),
            "error": self.error,
        }


async def wait_for_boot(state: Any, *, timeout_s: float, scan_only: bool = False) -> bool:
    """Block until the boot has scanned (or finished), or ``timeout_s`` passes.

    Returns True when the awaited phase is reached. Callers proceed either way:
    a scan that is still running after the wait yields the ordinary "unknown
    model" answer, which is honest -- and the wait is what makes it rare.
    """
    boot = getattr(state, "boot", None)
    if boot is None:
        return True
    event = boot.scanned if scan_only else boot.done
    if event.is_set():
        return True
    try:
        await asyncio.wait_for(event.wait(), timeout=max(0.0, timeout_s))
    except TimeoutError:
        return False
    return True


async def _boot(state: Any, *, start_background: bool) -> None:
    """The slow half of startup, after the port is bound (D33).

    Every step is individually non-fatal and the phase is published as it
    goes; ``BootStatus.finish`` always runs, so ``/health`` never sits on a
    stale phase and no waiter is stranded.
    """
    boot: BootStatus = state.boot
    try:
        boot.set_phase("scanning models")
        # Off the event loop (a cold scan of a large library stats and parses
        # every GGUF header), and never fatal: a failing scan means an empty
        # model list and a loud log line, not a server that will not boot.
        try:
            scan = await asyncio.to_thread(state.registry.scan)
        except Exception as exc:  # noqa: BLE001
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
        boot.scanned.set()

        # A secondary instance does nothing in the background at all -- no
        # orphan sweep, no auto-load, no TTL sweeper, and above all no download
        # resume. See _claim_data_dir for the incident this is guarding.
        secondary = getattr(state, "instance_role", "primary") != "primary"
        if start_background and secondary:
            log.warning(
                "background workers are disabled in this process because another "
                "instance owns the data directory",
                holder_pid=(getattr(state, "instance_holder", None) or {}).get("pid"),
            )
            # A secondary may show the queue but must never write into the
            # shared model directory (D24): enqueue/resume from the API, the
            # MCP tool and the GUI all refuse with the holder's pid.
            holder_pid = (getattr(state, "instance_holder", None) or {}).get("pid")
            state.downloader.disable_transfers(
                f"another StudioForge instance (pid {holder_pid}) owns this data directory; "
                "queue the download there, or stop it and restart this one"
            )
        if start_background and not secondary:
            boot.set_phase("sweeping orphaned engines")
            await _reclaim_orphaned_engines(state)
            pinned = getattr(state.config.engine, "pinned_tag", "")
            boot.set_phase(f"checking engine {pinned}".rstrip())
            try:
                # An install on a fresh box streams ~600 MB; its progress is the
                # phase, updated in place (no log line per tick).
                engine = await state.engine_manager.ensure_engine(
                    progress=lambda step, fraction: boot.set_progress(
                        f"installing engine {pinned}: {step} {fraction:.0%}"
                    )
                )
                log.info("engine ready", tag=engine.tag, variant=engine.variant)
                state.engine_status = {
                    "ok": True,
                    "tag": engine.tag,
                    "variant": engine.variant,
                    "smoke_tested": engine.smoke_tested,
                }
            except Exception as exc:  # noqa: BLE001
                log.error("engine not ready", error=str(exc))
                state.engine_status = {"ok": False, "tag": None, "error": str(exc)}
            boot.set_phase("starting model manager")
            await state.manager.start()
            # Resumes anything a crash left half-downloaded. Never fatal --
            # but never silent either: a queue that could not be read stays
            # "running" with no task behind it, and the log is the only place
            # that says why.
            try:
                await state.downloader.start()
            except Exception as exc:  # noqa: BLE001 - see above
                log.error("download queue did not start; downloads will not resume", error=str(exc))
    except asyncio.CancelledError:
        boot.finish("shut down before boot completed")
        raise
    except Exception as exc:  # noqa: BLE001 - the boot must always finish, and say how
        log.exception("boot failed", error=str(exc))
        boot.finish(str(exc))
        return
    boot.finish()
    log.info("boot complete", elapsed_s=boot.snapshot()["elapsed_s"])


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
    # One lease book shared by the planner (which honours leases) and the
    # manager (which grants and expires them) -- D43.
    leases = LeaseBook()
    planner = Planner(
        config,
        probe,
        observation_sink=lambda row: db.record_load_observation(**row),
        # The read side of the same table (D51): the sink has been writing
        # predicted-vs-actual since D18 and nothing read it back at plan time.
        observation_lookup=db.matching_observation,
        leases=leases,
    )
    # The probe goes in so an unload can log the VRAM it actually reclaimed
    # rather than asserting that it did. `resolve_engine_tag` is the same shape
    # of injection as `resolve_binary` -- a callable, so core keeps knowing
    # nothing about the engine manager -- and answers the question a Path
    # cannot: which build a request for `tag` (None = the active one) lands on,
    # taken from the EngineInfo rather than parsed off a directory name (D50).
    supervisor = Supervisor(
        config,
        resolve_binary=engine_manager.server_binary,
        resolve_engine_tag=_engine_tag_resolver(engine_manager),
        probe=probe,
    )
    # Late-bound rather than a constructor argument: the supervisor already
    # depends on the engine manager for `resolve_binary`, and the reverse
    # dependency (D49-7's "is this build in use?") would otherwise be a cycle.
    engine_manager.tag_in_use = _tag_in_use_check(supervisor, engine_manager)
    manager = ModelManager(
        config,
        registry=registry,
        planner=planner,
        supervisor=supervisor,
        db=db,
        version=version,
        leases=leases,
    )

    state.config = config  # type: ignore[attr-defined]
    state.db = db  # type: ignore[attr-defined]
    state.probe = probe  # type: ignore[attr-defined]
    state.engine_manager = engine_manager  # type: ignore[attr-defined]
    state.registry = registry  # type: ignore[attr-defined]
    state.planner = planner  # type: ignore[attr-defined]
    state.supervisor = supervisor  # type: ignore[attr-defined]
    state.manager = manager  # type: ignore[attr-defined]
    state.leases = leases  # type: ignore[attr-defined]
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
        # Bind first, boot after (D33). Everything slow -- the library scan,
        # the orphan sweep, an engine install on a fresh box (~600 MB), the
        # pinned auto-loads -- runs in a background task, and uvicorn starts
        # answering as soon as this yields. /health reports the phase; the
        # routes that need the registry or the engine wait for the phase they
        # need (bounded), so an early caller sees a slow answer instead of a
        # wrong one.
        app.state.boot = BootStatus()
        app.state.manager.boot_gate = app.state.boot.done
        app.state.boot_task = asyncio.create_task(
            _boot(app.state, start_background=start_background), name="studioforge-boot"
        )
        try:
            yield
        finally:
            boot_task = getattr(app.state, "boot_task", None)
            if boot_task is not None and not boot_task.done():
                boot_task.cancel()
                with contextlib.suppress(BaseException):
                    await boot_task
            with contextlib.suppress(Exception):
                await app.state.downloader.stop()
            with contextlib.suppress(Exception):
                await app.state.manager.stop()
            # After the manager has stopped every model: the supervisor's own
            # close is what releases the Windows job object (D23) -- and a
            # child that somehow survived stop_all dies with it. It was never
            # called before (WP17 F11), so the safety net only ever fired by
            # accident, at interpreter exit.
            with contextlib.suppress(Exception):
                await app.state.supervisor.aclose()
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
    if config.mcp.enabled:
        try:
            from studioforge.mcp.management import mount_management_mcp

            # Follow the CONFIGURED path. Hardcoding "/mcp" while auth
            # (`is_mcp_path`) and `/api/mcp/info` both honoured `mcp.path` meant
            # a customised path mounted MCP where the PIN was not accepted, and
            # advertised a URL that 404s -- defeating the PIN entirely.
            mount_management_mcp(app, app.state, path=config.mcp.path or "/mcp")
        except Exception as exc:
            log.error("management MCP not mounted", error=str(exc))
    else:
        # `mcp.enabled: false` used to be inert: the endpoint stayed mounted
        # and accepted the PIN while /api/mcp/info reported it disabled.
        log.info("management MCP disabled by config (mcp.enabled: false)")

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
            # What this server is in the middle of (D36). Cheap by
            # construction -- in-memory state only, no NVML and no HTTP to a
            # child -- because the watchdog polls this endpoint constantly. A
            # caller about to load or smoke-test a model on a shared box reads
            # this first: a load that would have to evict a model mid-request
            # is refused, and test_model refuses outright while anything here
            # is non-zero.
            "busy": app.state.manager.busy_snapshot(),
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
        # "Up" and "able to serve a model" are different claims, and a poller
        # that only sees the first reads a healthy 200 from a box with no engine
        # and no GPU. Say which it is, and why not, in the same payload.
        boot = getattr(app.state, "boot", None)
        if boot is not None:
            payload["boot"] = boot.snapshot()
        payload.update(_serving_readiness(app.state))
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


def _serving_readiness(state: Any) -> dict[str, Any]:
    """The part of ``/health`` that says whether a model could be served.

    ``engine`` is what the lifespan's ``ensure_engine`` concluded (``ok``,
    ``tag``, ``variant``, or ``ok: false`` with the error), ``gpu_count`` what
    the probe sees now, ``models_indexed`` the registry size. ``can_serve`` is
    the conjunction that matters -- engine present, at least one GPU -- and
    ``cannot_serve_reason`` names the first thing missing and its fix, so a
    watchdog, the tray or an agent reading ``/health`` learns the next action
    without opening the GUI. Never raises: a probe that fails reads as zero
    GPUs, which is the honest answer for the purpose.
    """
    engine = getattr(state, "engine_status", None)
    if engine is None:
        # A secondary instance, or a composed app that never ran ensure_engine:
        # report what is installed rather than nothing.
        try:
            info = state.engine_manager.active()
        except Exception:  # noqa: BLE001 - readiness must never fail /health
            info = None
        if info is not None:
            engine = {
                "ok": True,
                "tag": info.tag,
                "variant": info.variant,
                "smoke_tested": info.smoke_tested,
            }
        else:
            engine = {"ok": False, "tag": None, "error": "no llama-server engine is installed"}
    try:
        gpu_count = len(state.planner.probe.list_gpus())
    except Exception:  # noqa: BLE001
        gpu_count = 0
    try:
        models_indexed = len(state.registry.all())
    except Exception:  # noqa: BLE001
        models_indexed = 0

    reason: str | None = None
    boot = getattr(state, "boot", None)
    if boot is not None and not boot.ready:
        reason = f"still starting ({boot.phase}); the port is up, the rest is on its way"
    elif not engine.get("ok"):
        reason = (
            "no usable llama-server engine: install one from the Setup tab or run "
            "`studioforge engine --update`"
            + (f" ({engine.get('error')})" if engine.get("error") else "")
        )
    elif gpu_count == 0:
        reason = (
            "no NVIDIA GPU detected (this server is GPU-only): check the driver and "
            "nvidia-smi, then re-probe from the Setup tab"
        )
    elif models_indexed == 0:
        reason = (
            "no models indexed: point models.dir at a GGUF library on the Setup tab "
            "(or download one), then rescan"
        )
    payload: dict[str, Any] = {
        "engine": engine,
        "gpu_count": gpu_count,
        "models_indexed": models_indexed,
        "can_serve": reason is None,
    }
    if reason is not None:
        payload["cannot_serve_reason"] = reason
    return payload


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
        # 429 is the credential lockout. Same reasoning: the wait is a fact the
        # server knows, so it goes in the header every HTTP client already
        # understands, not only in the JSON body.
        elif exc.status_code == 429 and exc.details.get("retry_after_s"):
            headers["Retry-After"] = str(exc.details["retry_after_s"])
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
