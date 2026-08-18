"""Model lifecycle orchestration: JIT loading, TTL unloading, eviction.

This is the glue between the registry (what models exist), the planner (where
a model can go) and the supervisor (running llama-server children). It owns the
rule that makes OpenClaw work unchanged: naming an unloaded model in a request
loads it, and every request that arrives *during* that load waits for the same
load rather than starting a second one or erroring.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Sequence
from typing import Any

from studioforge.config import Config, KvCacheType
from studioforge.core.gpu import vram_processes
from studioforge.core.planner import OBSERVATION_NOTE_PER_PID, Planner
from studioforge.core.registry import Registry
from studioforge.core.supervisor import Supervisor
from studioforge.db import Database
from studioforge.errors import (
    BadRequestError,
    InsufficientVramError,
    ModelLoadError,
    ModelNotFoundError,
    StudioForgeError,
)
from studioforge.logging import get_logger
from studioforge.types import (
    AdapterRecord,
    InstanceInfo,
    LoadPlan,
    LoadRejected,
    ModelRecord,
    ModelSettings,
    ServerStatus,
)

log = get_logger(__name__)

#: Reference value for "this virtual model changes nothing at launch time".
#: Compared by value (pydantic ``__eq__``), so any future ModelSettings field
#: automatically counts as a launch-time delta until proven request-time.
_DEFAULT_SETTINGS = ModelSettings()

#: Substrings in a failed launch's stderr tail that mean "the machine was out
#: of memory *at that moment*". Taken from the real messages llama.cpp/CUDA
#: emit, not invented: a model swap that overlaps the previous model's teardown
#: hits these, and the identical load succeeds seconds later. Matched
#: case-insensitively.
TRANSIENT_LOAD_MARKERS: tuple[str, ...] = (
    "out of memory",
    "cuda error",
    "failed to allocate",
    "cudamalloc",
    "ggml_assert",
)

#: Substrings that mean the *configuration* is wrong. These never retry: the
#: second attempt would fail identically, just slower, and would bury the real
#: message under a second copy of itself.
CONFIG_ERROR_MARKERS: tuple[str, ...] = (
    "unknown argument",
    "invalid argument",
    "unrecognized argument",
    "error while handling argument",
    "unsupported model architecture",
    "unknown model architecture",
    "unknown architecture",
    "failed to open gguf file",
    "no such file or directory",
    "does not exist",
)


#: Sanity ceilings for per-request load arguments. Not policy -- the planner
#: refuses anything VRAM cannot hold with the numbers -- but a bound that keeps
#: ``--ctx-size -100`` and ``--parallel 0`` out of a child's argv, where they
#: die as an opaque llama-server exit instead of a 400 naming the parameter.
MAX_REQUEST_CTX = 16_777_216
MAX_REQUEST_PARALLEL = 256
KV_CACHE_TYPES: tuple[str, ...] = tuple(KvCacheType.__args__)  # type: ignore[attr-defined]


def validate_load_args(
    *,
    ctx_size: int | None,
    parallel: int | None,
    kv_cache_type: Any,
) -> None:
    """Raise :class:`BadRequestError` for a load argument no load could use.

    Called by :meth:`ModelManager.load` (so the HTTP route, the MCP tool, the
    GUI and the benchmark all get it) rather than by each caller. ``0``/negative
    values used to fall through the planner's ``x or default`` idiom -- ``0``
    silently meant "default", ``-1`` was honoured verbatim into the argv.
    """
    if ctx_size is not None and not (1 <= int(ctx_size) <= MAX_REQUEST_CTX):
        raise BadRequestError(
            f"ctx_size must be between 1 and {MAX_REQUEST_CTX} tokens (got {ctx_size}); "
            "omit it to let the planner choose",
            param="ctx_size",
        )
    if parallel is not None and not (1 <= int(parallel) <= MAX_REQUEST_PARALLEL):
        raise BadRequestError(
            f"parallel must be between 1 and {MAX_REQUEST_PARALLEL} slots (got {parallel}); "
            "omit it to let the planner choose",
            param="parallel",
        )
    if kv_cache_type is not None and str(kv_cache_type) not in KV_CACHE_TYPES:
        raise BadRequestError(
            f"kv_cache_type must be one of {', '.join(KV_CACHE_TYPES)} (got {kv_cache_type!r})",
            param="kv_cache_type",
        )


def classify_load_failure(stderr_tail: Sequence[str]) -> str:
    """``"transient"`` / ``"config"`` / ``"unknown"`` for a failed launch.

    Configuration wins over transience when both match: a bad flag can make
    llama-server abort in a way that also trips ``GGML_ASSERT``, and retrying a
    bad flag is pure waste.
    """
    blob = "\n".join(stderr_tail).lower()
    if any(marker in blob for marker in CONFIG_ERROR_MARKERS):
        return "config"
    if any(marker in blob for marker in TRANSIENT_LOAD_MARKERS):
        return "transient"
    return "unknown"


class ModelManager:
    """Owns "which models are loaded, and why"."""

    def __init__(
        self,
        config: Config,
        *,
        registry: Registry,
        planner: Planner,
        supervisor: Supervisor,
        db: Database,
        version: str = "0.1.0",
    ) -> None:
        self.config = config
        self.registry = registry
        self.planner = planner
        self.supervisor = supervisor
        self.db = db
        self.version = version
        self._started_at = time.time()
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()
        #: One load at a time, machine-wide. See :meth:`_load_locked` (D29):
        #: two cold loads planned side by side each see the VRAM the other has
        #: not allocated yet, both launch, one OOMs, the retry evicts the
        #: other, and the first client's request lands on a dead child.
        self._load_gate = asyncio.Lock()
        self._ttl_task: asyncio.Task[None] | None = None
        self._autoload_task: asyncio.Task[None] | None = None
        self._draining = False
        #: Set by the app to the boot's "done" event (D33): a JIT load that
        #: arrives while the library is still being scanned or the engine is
        #: still installing waits for that instead of answering "unknown
        #: model" / "no engine" from a half-built state. None means "no boot
        #: to wait for" (tests, the stdio MCP server).
        self.boot_gate: asyncio.Event | None = None
        self._load_waiters: dict[str, int] = {}
        #: model_id -> (ts, counters) the next throughput delta is measured from.
        self._throughput_baseline: dict[str, tuple[float, dict[str, float]]] = {}
        #: ``(built_at, full_catalog)``; see :meth:`catalog`.
        self._catalog_cache: tuple[float, dict[str, Any]] | None = None
        #: model_id -> the newest raw gauge scrape, for /api/status. Kept
        #: separate from the baseline because status wants "right now" while
        #: calibration wants "averaged over a long enough window to mean
        #: something".
        self._throughput_gauges: dict[str, dict[str, Any]] = {}

    # -- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        """Start background work: the TTL sweeper and any pinned auto-loads."""
        self._calibrate_from_history()
        self._ttl_task = asyncio.create_task(self._ttl_loop(), name="studioforge-ttl-sweep")
        if self.config.models.auto_load_pinned or self.config.models.preload_default_model:
            # Held by reference: the event loop keeps only weak refs to tasks,
            # and stop() must be able to cancel a preload still in flight --
            # otherwise a slow cold load continues after shutdown and can
            # spawn llama-server children with nobody left to stop them.
            self._autoload_task = asyncio.create_task(
                self._autoload_pinned(), name="studioforge-autoload"
            )

    async def stop(self, *, drain_timeout_s: float | None = None) -> None:
        """Drain in-flight requests, then stop every child."""
        self._draining = True
        if self._ttl_task is not None:
            self._ttl_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ttl_task
            self._ttl_task = None
        if self._autoload_task is not None:
            # A preload mid-flight must not keep starting children while (or
            # after) we drain and stop everything.
            self._autoload_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._autoload_task
            self._autoload_task = None

        deadline = time.time() + (
            drain_timeout_s if drain_timeout_s is not None else self.config.server.drain_timeout_s
        )
        while time.time() < deadline:
            in_flight = sum(i.active_requests for i in self.supervisor.list())
            if in_flight <= 0:
                break
            log.info("draining in-flight requests", in_flight=in_flight)
            await asyncio.sleep(0.25)
        else:
            remaining = sum(i.active_requests for i in self.supervisor.list())
            if remaining:
                log.warning("drain timeout expired, stopping anyway", in_flight=remaining)

        await self.supervisor.stop_all()

    async def resume(self) -> None:
        """Undo a drain that did not end in a shutdown.

        ``stop()`` is written for "this process is going away", so it latches
        ``_draining`` and cancels the TTL sweeper. A restart that *fails* --
        the replacement process could not start, so we stay up -- left both in
        that state forever: ``/health`` reported ``draining: true`` on a server
        that went on happily serving and loading models for hours, and nothing
        was ever evicted again because the sweeper was dead. A drain flag that
        cannot come back down is worse than no flag, because it is the one
        signal an operator uses to decide whether a process is safe to kill.

        Children are not restarted here: ``stop()`` already unloaded them and
        they reload on demand, which is the normal cold-start behaviour.
        """
        was_draining = self._draining
        self._draining = False
        if self._ttl_task is None or self._ttl_task.done():
            self._ttl_task = asyncio.create_task(self._ttl_loop(), name="studioforge-ttl-sweep")
        if was_draining:
            log.info("drain cancelled; the server stays up")

    @property
    def draining(self) -> bool:
        return self._draining

    # -- JIT loading ------------------------------------------------------

    def serving_record(self, record: ModelRecord) -> ModelRecord:
        """The record whose ``llama-server`` instance serves this model.

        A virtual model whose only differences from its base are *request-time*
        (system prompt, sampler defaults -- see
        :class:`~studioforge.types.VirtualPreset`) shares the base's instance:
        that sharing is the entire efficiency point of presets, and it is what
        keeps ten personas over one 30B base from costing ten loads. Any
        launch-time delta -- adapters, a ctx/kv override, anything at all in
        :class:`ModelSettings` -- means a different child argv, so the virtual
        model keeps its own dedicated instance exactly as before.
        """
        if not record.is_virtual or record.base_model_id is None:
            return record
        if record.settings != _DEFAULT_SETTINGS:
            return record
        base = self.registry.get(record.base_model_id)
        return base if base is not None else record

    async def _await_boot(self) -> None:
        """Wait (bounded by the load timeout) for the app's boot to finish.

        A caller that arrives mid-boot -- OpenClaw reconnecting the moment the
        port answers, an agent's first ``load_model`` -- would otherwise be told
        the model does not exist because the scan has not run yet. Bounded so
        a boot that hangs cannot hang every request with it: after the wait
        the ordinary errors apply.
        """
        gate = self.boot_gate
        if gate is None or gate.is_set():
            return
        log.info("waiting for startup to finish before loading", model="pending")
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(gate.wait(), timeout=float(self.config.gateway.load_timeout_s))

    async def _lock_for(self, model_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(model_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[model_id] = lock
            return lock

    def _prune_lock(self, model_id: str) -> None:
        """Drop a per-model lock nobody is holding or waiting on.

        Without this the lock table grows by one entry per model id ever
        requested and never shrinks -- harmless for a static library, a slow
        leak once virtual models can be created and deleted through the API.
        Only called when the waiter count for ``model_id`` has hit zero; a
        caller that raced past that check simply creates a fresh lock, which is
        safe because nobody can hold the discarded one.
        """
        lock = self._locks.get(model_id)
        if lock is not None and not lock.locked():
            self._locks.pop(model_id, None)

    async def ensure_loaded(self, name: str) -> tuple[ModelRecord, InstanceInfo]:
        """Resolve a model name and make sure it is serving, loading if needed.

        Concurrent callers for the same model all wait on one lock, so a burst
        of requests against a cold model produces exactly one llama-server
        start and every request is served once it is up -- queuing, never
        erroring, which is what LM Studio's JIT behaviour looks like to a
        client.

        The returned record is the one the caller *named* (a virtual model
        keeps its identity); the returned instance may belong to its base --
        use ``instance.model_id`` for anything that talks to the supervisor.
        """
        await self._await_boot()
        record = self.registry.resolve(name)
        if record is None:
            raise ModelNotFoundError(name, known=self.registry.known_ids())
        serving = self.serving_record(record)

        instance = self.supervisor.get(serving.id)
        if instance is not None and instance.state == "ready":
            self.registry.touch(record.id)
            return record, instance

        lock = await self._lock_for(serving.id)
        self._load_waiters[serving.id] = self._load_waiters.get(serving.id, 0) + 1
        try:
            async with lock:
                # Re-check: another waiter may have completed the load.
                instance = self.supervisor.get(serving.id)
                if instance is not None and instance.state == "ready":
                    self.registry.touch(record.id)
                    return record, instance
                instance = await self._load_locked(serving)
                self.registry.touch(record.id)
                return record, instance
        finally:
            remaining = self._load_waiters.get(serving.id, 1) - 1
            if remaining <= 0:
                self._load_waiters.pop(serving.id, None)
                self._prune_lock(serving.id)
            else:
                self._load_waiters[serving.id] = remaining

    async def load(
        self,
        name: str,
        *,
        ctx_size: int | None = None,
        kv_cache_type: Any = None,
        parallel: int | None = None,
        force: bool = False,
    ) -> InstanceInfo:
        """Explicit load (GUI/CLI/MCP). ``force`` reloads an already-running model.

        A preset-only virtual model loads its *base* instance -- launching a
        second copy of the same weights because the user clicked Load on the
        persona rather than the base would defeat the sharing.

        A forced reload **plans before it unloads** (D30). The planner is told
        which instance is being replaced and credits its footprint back, so a
        reload that fits is spawned after the resident child is stopped, and a
        reload that is refused -- the args no longer fit, VRAM moved to
        ComfyUI, the engine is gone -- raises with the numbers while the model
        that was serving a moment ago goes on serving. The old order (stop,
        then plan) turned every refused reload into an outage.
        """
        validate_load_args(ctx_size=ctx_size, parallel=parallel, kv_cache_type=kv_cache_type)
        await self._await_boot()
        record = self.registry.resolve(name)
        if record is None:
            raise ModelNotFoundError(name, known=self.registry.known_ids())
        record = self.serving_record(record)
        lock = await self._lock_for(record.id)
        async with lock:
            existing = self.supervisor.get(record.id)
            reload_of: str | None = None
            if existing is not None and existing.state == "ready":
                if not force:
                    return existing
                reload_of = record.id
            return await self._load_locked(
                record,
                ctx_size=ctx_size,
                kv_cache_type=kv_cache_type,
                parallel=parallel,
                reload_of=reload_of,
            )

    async def _load_locked(
        self,
        record: ModelRecord,
        *,
        ctx_size: int | None = None,
        kv_cache_type: Any = None,
        parallel: int | None = None,
        reload_of: str | None = None,
    ) -> InstanceInfo:
        """Plan, evict if needed, launch. Caller must hold the per-model lock.

        Loads are also serialised machine-wide behind ``_load_gate`` (D29).
        The planner decides against *live* free VRAM, and a child that is
        still loading has not yet taken the memory its plan says it will:
        two different cold models requested at once were each planned as if
        the other did not exist, both children launched onto the same cards,
        one died with ``CUDA error: out of memory``, its one transient retry
        evicted the other -- idle for the instant between "ready" and its
        client's first request -- and that client's request then hit a dead
        child. Behind the gate the second load plans after the first has
        actually allocated, and either fits beside it or is refused with the
        numbers. The per-model lock is taken first, then the gate, everywhere,
        so the two cannot deadlock.
        """
        if self._load_gate.locked():
            log.info(
                "waiting for another model load to finish before planning",
                model_id=record.id,
                queued=sum(self._load_waiters.values()),
            )
        async with self._load_gate:
            return await self._load_gated(
                record,
                ctx_size=ctx_size,
                kv_cache_type=kv_cache_type,
                parallel=parallel,
                reload_of=reload_of,
            )

    async def _load_gated(
        self,
        record: ModelRecord,
        *,
        ctx_size: int | None,
        kv_cache_type: Any,
        parallel: int | None,
        reload_of: str | None = None,
    ) -> InstanceInfo:
        existing = self.supervisor.get(record.id)
        if reload_of is not None and (existing is None or existing.state != "ready"):
            # The resident child went away while we waited for the gate (a
            # crash, a TTL unload): there is nothing to replace, so this is an
            # ordinary load.
            reload_of = None
        if existing is not None and existing.state == "loading" and reload_of is None:
            # The supervisor is already bringing this child up -- its crash
            # watcher relaunching it after an exit. Planning again would launch
            # nothing (start() returns the in-flight instance) but could evict
            # bystanders for a placement that is already decided, and the
            # caller would then be handed a "loading" instance and forward a
            # request to a port nobody is listening on yet. Wait for that
            # start to settle instead; only a start that fails falls through
            # to a fresh plan.
            settled = await self._await_settled(record.id)
            if settled is not None:
                return settled

        draft = self._draft_for(record)
        adapters = self._adapters_for(record)

        plan_result = self.planner.plan_load(
            record,
            ctx_size=ctx_size,
            kv_cache_type=kv_cache_type,
            parallel=parallel,
            loaded=self.supervisor.list(),
            draft=draft,
            adapters=[a for a, _ in adapters],
            reload_of=reload_of,
        )

        if isinstance(plan_result, LoadRejected):
            if reload_of is not None:
                log.warning(
                    "forced reload refused; the running instance is kept",
                    model_id=record.id,
                    reason=plan_result.reason,
                )
                plan_result.suggestions.append(
                    f"the running instance of {reload_of} was left loaded with its "
                    f"previous settings; nothing was unloaded"
                )
            raise self._vram_error(plan_result)

        plan: LoadPlan = plan_result
        victims = list(plan.evict_model_ids)
        if reload_of is not None and reload_of not in victims:
            victims.insert(0, reload_of)
        for victim in victims:
            if victim == reload_of:
                log.info("stopping the running instance for a forced reload", model_id=victim)
            else:
                log.info("evicting to make room", victim=victim, for_model=record.id)
            await self.supervisor.stop(victim)

        engine_tag = record.settings.engine_tag
        try:
            instance = await self._start_with_retry(
                record,
                plan,
                engine_tag=engine_tag,
                draft=draft,
                adapters=adapters,
                ctx_size=ctx_size,
                kv_cache_type=kv_cache_type,
                parallel=parallel,
            )
        except StudioForgeError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ModelLoadError(f"failed to start '{record.id}': {exc}") from exc

        # The supervisor only knows the raw per-model setting, which is usually
        # None. The EFFECTIVE ttl -- global default folded in, and 0 for pinned --
        # lives here, and both the TTL sweeper and the planner's pinned check read
        # it off the instance. Without this, models.default_ttl_s never applied
        # (nothing idle-unloaded) and a pinned model was not recognised as
        # pinned, so the planner would happily evict it.
        self.apply_effective_ttl(record, instance)
        self.registry.touch(record.id)
        # The instance carries the pid, and the pid is what makes the
        # observation about *this* child rather than about the whole card.
        self._record_actual_vram(record, instance.plan or plan, instance)
        return instance

    #: How often a caller waiting on an in-flight start re-checks it.
    SETTLE_POLL_S = 0.25

    async def _await_settled(self, model_id: str) -> InstanceInfo | None:
        """Wait for an in-flight start of ``model_id`` to reach a terminal state.

        Returns the instance once it is ``ready``; ``None`` when it failed,
        vanished, or is still loading after the configured load timeout (in
        which case the caller plans a fresh load and the supervisor's own
        limits decide what happens to the straggler).
        """
        deadline = time.monotonic() + float(self.config.gateway.load_timeout_s) + 5.0
        while time.monotonic() < deadline:
            instance = self.supervisor.get(model_id)
            if instance is None or instance.state != "loading":
                return instance if instance is not None and instance.state == "ready" else None
            await asyncio.sleep(self.SETTLE_POLL_S)
        log.warning(
            "an in-flight start never settled; planning a fresh load",
            model_id=model_id,
            waited_s=round(float(self.config.gateway.load_timeout_s) + 5.0),
        )
        return None

    def _vram_error(self, rejected: LoadRejected) -> InsufficientVramError:
        """Turn a planner refusal into the 507, keeping every number with it."""
        return InsufficientVramError(
            rejected.message(),
            details={
                "required_bytes": rejected.required_bytes,
                "available_bytes": rejected.available_bytes,
                "per_gpu_free": rejected.per_gpu_free,
                "max_ctx_that_fits": rejected.max_ctx_that_fits,
                "max_parallel_that_fits": rejected.max_parallel_that_fits,
                "suggestions": rejected.suggestions,
                "notes": rejected.notes,
                # Who is holding the VRAM: on a shared GPU box this is usually
                # the actual answer to "why did this stop working".
                "vram_holders": [h.model_dump() for h in rejected.vram_holders],
                "estimate_mb": rejected.estimate.breakdown_mb(),
            },
        )

    async def _start_with_retry(
        self,
        record: ModelRecord,
        plan: LoadPlan,
        *,
        engine_tag: str | None,
        draft: ModelRecord | None,
        adapters: Sequence[tuple[AdapterRecord, float]],
        ctx_size: int | None = None,
        kv_cache_type: Any = None,
        parallel: int | None = None,
    ) -> InstanceInfo:
        """Launch the child, retrying **once** after a transient OOM.

        Models intermittently die at launch with an allocation failure -- most
        often during a swap, while the outgoing model's VRAM is still being
        released by the driver -- and the identical load succeeds moments
        later. That is worth one retry.

        The retry only happens after evicting the LRU unpinned model, and only
        when there is one to evict. Retrying without changing anything is
        pointless: the same argv against the same hardware fails the same way,
        so the second attempt costs a load timeout and produces the same error.
        Freeing memory is what makes the second attempt a *different* attempt.

        A configuration error (bad flag, missing file, unsupported
        architecture) is never retried -- see :func:`classify_load_failure`.
        """
        try:
            return await self.supervisor.start(
                record, plan, engine_tag=engine_tag, draft=draft, adapters=adapters
            )
        except ModelLoadError as exc:
            stderr = exc.details.get("stderr")
            tail = [str(line) for line in stderr] if isinstance(stderr, list) else []
            kind = classify_load_failure(tail)
            if kind != "transient":
                raise
            victim = self._eviction_candidate(exclude=record.id)
            if victim is None:
                log.warning(
                    "load failed transiently but nothing can be evicted; not retrying",
                    model_id=record.id,
                    reason="a retry with identical conditions would fail identically",
                )
                raise
            log.warning(
                "transient load failure; evicting and retrying once",
                model_id=record.id,
                victim=victim,
            )
            await self.supervisor.stop(victim)

        # Re-plan: free VRAM has changed, so placement and context may too.
        # The caller's explicit overrides (ctx_size/kv/parallel) must survive
        # the retry -- replanning without them would silently load the model
        # with a different context than the one the user asked for.
        replanned = self.planner.plan_load(
            record,
            ctx_size=ctx_size,
            kv_cache_type=kv_cache_type,
            parallel=parallel,
            loaded=self.supervisor.list(),
            draft=draft,
            adapters=[a for a, _ in adapters],
        )
        if isinstance(replanned, LoadRejected):
            raise self._vram_error(replanned)
        instance = await self.supervisor.start(
            record, replanned, engine_tag=engine_tag, draft=draft, adapters=adapters
        )
        log.info("load succeeded on retry", model_id=record.id)
        return instance

    def _eviction_candidate(self, *, exclude: str) -> str | None:
        """LRU unpinned idle instance to evict, or None if there is nothing."""
        for instance in self.planner._evictable(self.supervisor.list()):
            if instance.model_id != exclude:
                return instance.model_id
        return None

    def apply_effective_ttl(self, record: ModelRecord, instance: InstanceInfo) -> None:
        """Stamp the effective TTL onto a running instance (0 == pinned)."""
        instance.ttl_s = self.ttl_for(record)

    def refresh_ttl(self, model_id: str) -> int | None:
        """Re-apply the effective TTL after settings change while loaded.

        Toggling `pinned` on a resident model has to take effect immediately;
        otherwise the pin would only apply on the next load, which is exactly
        when the user does not need it.
        """
        record = self.registry.resolve(model_id)
        instance = self.supervisor.get(record.id) if record else None
        if record is None or instance is None:
            return None
        self.apply_effective_ttl(record, instance)
        return instance.ttl_s

    #: Observations read at startup to tune the planner's overhead fraction.
    #: Newest-first, so a factor that drifted a year ago cannot outvote how the
    #: box behaves today.
    CALIBRATION_WINDOW = 200

    def _calibrate_from_history(self) -> None:
        """Tune ``planner.compute_overhead_fraction`` from clean load history.

        The calibration loop was open: every load recorded predicted-vs-actual
        and ``suggest_overhead_fraction()`` existed, but nothing ever called it
        outside the tests, so the documentation's claim that the factor
        self-tunes was simply untrue. Closing it at startup (rather than after
        every load) keeps the number stable for the lifetime of a process --
        a planner whose arithmetic shifts under a running server is far harder
        to reason about than one that is wrong in a fixed way.
        """
        try:
            observations = self.db.load_observations(limit=self.CALIBRATION_WINDOW)
            self.planner.calibrate(observations)
        except Exception as exc:  # pragma: no cover - calibration must never stop a boot
            log.debug("could not calibrate overhead fraction", error=str(exc))

    def _record_actual_vram(
        self, record: ModelRecord, plan: LoadPlan, instance: InstanceInfo
    ) -> None:
        """Log predicted-vs-actual so the planner's fudge factor can be tuned.

        Measures **our child's own** VRAM, per pid, not the devices' total
        ``used_bytes``. The device total is whatever else happens to be on the
        card -- the desktop compositor, ComfyUI, another model of ours -- so it
        answered a question nobody asked and answered it differently every
        time. Over 540 historical rows it produced a median actual/predicted
        ratio of 2.97 against a documented 0.81-1.23.

        NVML cannot enumerate per-process VRAM everywhere (containers, WSL,
        MIG). When it cannot, the observation is skipped rather than falling
        back to the device total: no data beats data that means something else.
        """
        try:
            pid = instance.pid
            if pid is None:
                return
            devices = set(plan.devices)
            holders = vram_processes(self.planner.probe, own_pids=[pid])
            actual = sum(
                h.used_bytes
                for h in holders
                if h.pid == pid and (not devices or h.gpu_index in devices)
            )
            if actual > 0:
                self.planner.observe(
                    model_id=record.id,
                    plan=plan,
                    actual_bytes=actual,
                    note=OBSERVATION_NOTE_PER_PID,
                )
            else:
                log.debug(
                    "no per-pid VRAM attribution available; skipping observation",
                    model_id=record.id,
                    pid=pid,
                )
        except Exception as exc:  # pragma: no cover - telemetry must never break a load
            log.debug("could not record vram observation", error=str(exc))

    # -- unloading --------------------------------------------------------

    async def unload(self, name: str) -> bool:
        record = self.registry.resolve(name)
        model_id = record.id if record is not None else name
        if self.supervisor.get(model_id) is None:
            return False
        await self.supervisor.stop(model_id)
        return True

    async def unload_all(self) -> list[str]:
        ids = [i.model_id for i in self.supervisor.list()]
        await self.supervisor.stop_all()
        return ids

    # -- TTL --------------------------------------------------------------

    def ttl_for(self, record: ModelRecord) -> int:
        """Effective TTL in seconds; 0 means pinned (never idle-unload)."""
        if record.settings.pinned:
            return 0
        if record.settings.ttl_s is not None:
            return record.settings.ttl_s
        return self.config.models.default_ttl_s

    async def _ttl_loop(self) -> None:
        interval = self.config.gateway.ttl_sweep_interval_s
        while True:
            try:
                await asyncio.sleep(interval)
                await self._sweep_ttl()
                await self._sample_throughput()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - the sweeper must never die
                log.warning("ttl sweep failed", error=str(exc))

    async def _sweep_ttl(self) -> None:
        now = time.time()
        for instance in self.supervisor.list():
            if instance.state != "ready" or instance.active_requests > 0:
                continue
            if self._model_was_removed(instance.model_id):
                # The file is gone (its directory was walked and it was not
                # there): on Linux the child keeps serving from the unlinked
                # inode while /v1/models 404s it and /api/status lists it, and a
                # pinned one holds its VRAM until restart. A model that is
                # merely *unreachable* (its drive dropped) is kept stale by the
                # registry and is not touched here (WP17 R3).
                log.warning(
                    "unloading a model whose file was removed",
                    model_id=instance.model_id,
                    pid=instance.pid,
                    port=instance.port,
                )
                await self.supervisor.stop(instance.model_id)
                continue
            ttl = instance.ttl_s
            if not ttl:  # None or 0 -> pinned / no TTL
                continue
            last = instance.last_activity_at or instance.started_at or now
            idle = now - last
            if idle >= ttl:
                log.info(
                    "unloading idle model",
                    model_id=instance.model_id,
                    idle_s=round(idle),
                    ttl_s=ttl,
                )
                await self.supervisor.stop(instance.model_id)

    def _model_was_removed(self, model_id: str) -> bool:
        """True only when the registry has scanned and no longer knows ``model_id``.

        Never true before the first scan has completed (a cold registry knows
        nothing), and never true for a stale record (unreachable is not
        removed).
        """
        try:
            if getattr(self.registry, "last_scan_at", None) is None:
                return False
            return self.registry.get(model_id) is None
        except Exception:  # noqa: BLE001 - a registry hiccup is not a removal
            return False

    # -- measured throughput ----------------------------------------------

    #: Minimum span between two recorded throughput observations for one
    #: model. The TTL sweeper runs every ~15s, which is far too often: a busy
    #: model would write thousands of rows a day, and a 15-second window on an
    #: agent workload measures whatever one request happened to be doing. Two
    #: minutes is long enough to average over several requests and short enough
    #: that a model unloaded after ten minutes still contributes data.
    THROUGHPUT_RECORD_MIN_S = 120.0

    #: Rows kept per model. Calibration reads a median over a recent window, so
    #: older rows cost storage and buy nothing.
    THROUGHPUT_KEEP_PER_MODEL = 200

    async def _sample_throughput(self) -> None:
        """Scrape ``/metrics`` from every ready child; record real tokens/sec.

        Runs on the existing TTL sweeper rather than on the request-completion
        path. Hooking completion was the first choice and was rejected: the
        gateway streams responses, so "complete" is several places (normal
        return, client disconnect, upstream error), and every one of them would
        pay an HTTP round-trip to the child *inside* the request path to read
        counters that only mean something averaged over a window anyway. On the
        timer, the same numbers cost one localhost GET per loaded model per
        sweep and cannot slow a single request down.

        Nothing here may raise: it is a background timer, and a metrics
        endpoint that has changed shape, a child mid-restart or a locked
        database must all degrade to "no observation this round".
        """
        now = time.time()
        for instance in self.supervisor.list():
            if instance.state != "ready":
                self._throughput_baseline.pop(instance.model_id, None)
                self._throughput_gauges.pop(instance.model_id, None)
                continue
            try:
                await self._sample_one(instance, now)
            except Exception as exc:  # noqa: BLE001 - telemetry, never a failure
                log.debug("throughput sample failed", model_id=instance.model_id, error=str(exc))

    async def _sample_one(self, instance: InstanceInfo, now: float) -> None:
        """One model's window: measure, predict at the same reference, record both.

        The prediction stored beside the measurement is deliberately taken at
        :data:`throughput.REFERENCE_FILL_TOKENS` -- **the same fill the
        catalog's ``est_gen_tps`` column is quoted at** -- rather than at this
        child's actual KV occupancy, which we cannot see from ``/metrics``
        anyway. That makes the learned factor exactly "what real traffic does
        divided by what we would have promised", so applying it to the catalog
        column corrects the number a user is actually shown. Predicting at some
        other fill and correcting a differently-quoted column with the result
        would leave a systematic offset that no amount of data removes.

        The row is stamped with :data:`throughput.ESTIMATOR_VERSION`, because a
        calibration factor is a correction to one specific formula: carrying
        ratios across a formula change teaches the estimator the difference
        between two dead arithmetics (D22). An unstamped row is never used for
        calibration -- so this stamp is not optional decoration, it is what
        keeps calibration alive across this and every future estimator change.
        """
        from studioforge.core import throughput
        from studioforge.core.planner import kv_read_bytes_per_slot

        model_id = instance.model_id
        text = await self.supervisor.metrics(model_id)
        counters = throughput.parse_metrics(text)
        if not counters:
            log.debug("no metrics from child", model_id=model_id)
            return

        self._throughput_gauges[model_id] = {
            "sampled_at": now,
            **{name: counters[name] for name in throughput.GAUGE_METRICS if name in counters},
        }

        baseline = self._throughput_baseline.get(model_id)
        if baseline is None:
            self._throughput_baseline[model_id] = (now, counters)
            return
        elapsed = now - baseline[0]
        if elapsed < self.THROUGHPUT_RECORD_MIN_S:
            return

        sample = throughput.sample_between(baseline[1], counters, elapsed_s=elapsed)
        # Either way the window is spent: keep measuring forward from here so a
        # long idle stretch cannot dilute the next average.
        self._throughput_baseline[model_id] = (now, counters)
        if sample is None:
            return

        plan = instance.plan
        record = self.registry.get(model_id)
        if plan is None or record is None or record.meta is None:
            return

        gpus = {g.index: g for g in self.planner.probe.list_gpus()}
        predicted = throughput.estimate(
            record.meta,
            plan.estimate.weights_bytes,
            plan.per_gpu_bytes or {},
            kv_read_bytes_per_slot=kv_read_bytes_per_slot(
                record.meta,
                kv_k=plan.kv_cache_type,
                kv_v=plan.kv_cache_type_v,
                ctx_fill=min(int(plan.ctx_size), throughput.REFERENCE_FILL_TOKENS),
            ),
            parallel=plan.parallel,
            gpus=gpus,
        )
        self.db.record_throughput_observation(
            model_id=model_id,
            ts=now,
            devices=",".join(str(d) for d in sorted(plan.devices)),
            gpu_class=throughput.gpu_class([gpus[d] for d in plan.devices if d in gpus]),
            ctx_size=plan.ctx_size,
            parallel=plan.parallel,
            kv_cache_type=plan.kv_cache_type,
            prompt_tps=sample["prompt_tps"],
            gen_tps=sample["gen_tps"],
            est_prompt_tps=predicted["prompt_tps"],
            est_gen_tps=predicted["gen_tps"],
            estimator_version=throughput.ESTIMATOR_VERSION,
            n_busy_slots=sample["n_busy_slots"],
            requests_deferred=sample["requests_deferred"],
            sample_s=sample["sample_s"],
        )
        log.debug(
            "throughput observed",
            model_id=model_id,
            gen_tps=sample["gen_tps"],
            est_gen_tps=predicted["gen_tps"],
            window_s=sample["sample_s"],
        )
        with contextlib.suppress(Exception):
            self.db.prune_throughput_observations(keep_per_model=self.THROUGHPUT_KEEP_PER_MODEL)

    def metrics_snapshot(self) -> dict[str, dict[str, Any]]:
        """Newest ``/metrics`` gauges per loaded model, for ``/api/status``.

        Read from the collector's cache rather than scraped on demand: status
        is polled continuously by the GUI, and adding a fan-out of HTTP calls
        to every child on every poll would make the dashboard the heaviest
        client on the box. A gauge up to one sweep old is the right trade.
        """
        return dict(self._throughput_gauges)

    async def _autoload_pinned(self) -> None:
        """Warm pinned models, and optionally the configured default.

        Preloading the default turns the very first client request from a
        multi-minute cold load into a normal one -- the difference between
        "the server is broken" and "the server is fast" on first contact.
        """
        wanted: list[str] = []
        if self.config.models.auto_load_pinned:
            wanted.extend(r.id for r in self.registry.all() if r.settings.pinned)
        default = self.config.models.default_model
        if self.config.models.preload_default_model and default:
            record = self.registry.resolve(default)
            if record is None:
                log.warning("default model not found, cannot preload", default_model=default)
            elif record.id not in wanted:
                wanted.append(record.id)

        for model_id in wanted:
            try:
                await self.load(model_id)
                log.info("preloaded model", model_id=model_id)
            except Exception as exc:
                log.warning("failed to preload model", model_id=model_id, error=str(exc))

    # -- helpers ----------------------------------------------------------

    def _draft_for(self, record: ModelRecord) -> ModelRecord | None:
        draft_id = record.settings.draft_model_id
        if not draft_id:
            return None
        draft = self.registry.resolve(draft_id)
        if draft is None:
            log.warning(
                "draft model not found, loading without speculative decoding",
                model_id=record.id,
                draft_model_id=draft_id,
            )
        return draft

    def _adapters_for(self, record: ModelRecord) -> list[tuple[AdapterRecord, float]]:
        out: list[tuple[AdapterRecord, float]] = []
        for attachment in record.settings.adapters:
            adapter = self.registry.get_adapter(attachment.adapter_id)
            if adapter is None:
                log.warning(
                    "attached adapter missing from registry, skipping",
                    model_id=record.id,
                    adapter_id=attachment.adapter_id,
                )
                continue
            out.append((adapter, attachment.scale))
        return out

    # -- status -----------------------------------------------------------

    def status(self, *, engine: Any = None, active_downloads: int = 0) -> ServerStatus:
        from studioforge.core.gpu import system_ram

        total_ram, used_ram = system_ram()
        loaded = self.supervisor.list()
        return ServerStatus(
            version=self.version,
            uptime_s=time.time() - self._started_at,
            gpus=self.planner.probe.list_gpus(),
            system_ram_total_bytes=total_ram,
            system_ram_used_bytes=used_ram,
            loaded=loaded,
            model_count=len(self.registry.all()),
            # Who holds VRAM, ours and foreign. Free-VRAM numbers alone cannot
            # explain a load that started failing because something else on the
            # box (ComfyUI, a training run) took the memory.
            vram_processes=vram_processes(
                self.planner.probe, own_pids=[i.pid for i in loaded if i.pid is not None]
            ),
            engine=engine,
            queue_depth=sum(self._load_waiters.values()),
            active_downloads=active_downloads,
            draining=self._draining,
        )

    #: Hardware modes OpenClaw can choose between. 5090s first: they are the
    #: fastest cards and leaving the 3090s free lets a second model run beside
    #: this one. "all" is the fallback for anything that will not fit on a pair.
    PLACEMENT_MODES: tuple[tuple[str, tuple[int, ...] | None, str], ...] = (
        ("dual_5090", (0, 1), "Both RTX 5090s. Fastest, and leaves the 3090s free."),
        ("dual_3090", (2, 3), "Both RTX 3090s. Slower, but keeps the 5090s free."),
        ("all_gpus", None, "All four GPUs. Largest capacity; uses the whole rig."),
    )

    def placement_profiles(self, name: str) -> dict[str, Any]:
        """Best achievable load for this model in each hardware mode.

        Answers "which model should I run, and how?" in one call, so an agent
        does not have to probe context sizes by trial and error. Each mode
        reports the largest context that fits and the best-quality KV cache
        that reaches it -- the same decision a real load would make.

        Carries the concurrency and speed columns too (``max_parallel``,
        ``parallel_limited_by``, ``ctx_per_slot``, ``kv_bytes_per_token``,
        ``est_gen_tps`` at the reference fill and ``est_gen_tps_full_ctx`` at
        the whole window, ``est_prompt_tps``), so this endpoint and
        ``/api/catalog`` never disagree about the same placement. The
        difference between them is the question asked: ``/profiles`` asks "what
        is the best this model can do on each *hardware mode*", the catalog
        asks "what does this model do at each *context size*".
        """
        from studioforge.core import catalog as catalog_mod

        record = self.registry.resolve(name)
        if record is None:
            raise ModelNotFoundError(name, known=self.registry.known_ids())

        trained = record.meta.n_ctx_train if record.meta else None
        profiles: list[dict[str, Any]] = []
        original = record.settings
        try:
            for key, devices, description in self.PLACEMENT_MODES:
                record.settings = original.model_copy(
                    update={"device_override": list(devices) if devices else None}
                )
                result = self.planner.plan_load(record, loaded=())
                entry: dict[str, Any] = {
                    "mode": key,
                    "description": description,
                    "gpus": list(devices) if devices else "all",
                }
                if isinstance(result, LoadPlan):
                    slots, bound, vram = catalog_mod.slots_for_plan(self.planner, record, result)
                    speed = catalog_mod.estimate_speed(self.planner, record, result, slots, {})
                    entry.update(
                        fits=True,
                        ctx_size=result.ctx_size,
                        ctx_per_slot=result.ctx_size,
                        kv_cache_type=result.kv_cache_type,
                        kv_bytes_per_token=result.kv_bytes_per_token,
                        devices=list(result.devices),
                        max_parallel=slots,
                        parallel_limited_by=bound,
                        est_gen_tps=speed["gen_tps"],
                        est_gen_tps_full_ctx=speed["gen_tps_full_ctx"],
                        est_prompt_tps=speed["prompt_tps"],
                        est_gen_tps_batched=speed["gen_tps_batched"],
                        load_args={
                            "model_id": record.id,
                            "ctx_size": result.ctx_size,
                            "parallel": slots,
                            "kv_cache_type": result.kv_cache_type,
                        },
                        # Sized at max_parallel, matching load_args, not at
                        # whatever slot count the plan happened to carry.
                        vram_gib=round(vram / (1024**3), 2),
                    )
                else:
                    entry.update(
                        fits=False,
                        reason=result.reason,
                        max_ctx_that_fits=result.max_ctx_that_fits,
                    )
                profiles.append(entry)
        finally:
            record.settings = original

        usable = [p for p in profiles if p["fits"]]
        best = max(usable, key=lambda p: p["ctx_size"], default=None)
        # Cheapest = fewest/fastest cards that still reach the best context, so
        # an agent can leave the rest of the rig free for something else.
        cheapest = next(
            (p for p in profiles if p["fits"] and best and p["ctx_size"] == best["ctx_size"]),
            None,
        )
        return {
            "model_id": record.id,
            "n_ctx_train": trained,
            "size_gib": round(record.size_bytes / (1024**3), 2),
            "profiles": profiles,
            "best_mode": best["mode"] if best else None,
            "max_ctx": best["ctx_size"] if best else None,
            "recommended_mode": cheapest["mode"] if cheapest else None,
        }

    # -- catalog ----------------------------------------------------------

    def catalog(
        self,
        *,
        model: str | None = None,
        compact: bool = False,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """The model catalog, cached for a few seconds.

        The full table is built once and the ``model``/``compact`` views are
        *derived* from it rather than rebuilt. Building costs a planner walk
        per model per context tier; filtering costs a list comprehension. An
        agent that calls ``list_models`` and then ``model_options`` for three
        candidates should pay for one build, not four.

        The cache is deliberately short (:data:`catalog.CACHE_TTL_S`): ``fits``
        is a claim about free VRAM at this instant, and a stale yes sends an
        agent into a load that will be refused.
        """
        from studioforge.core.catalog import CACHE_TTL_S, build_catalog, compact_entry

        now = time.time()
        cached = self._catalog_cache
        if refresh or cached is None or now - cached[0] > CACHE_TTL_S:
            full = build_catalog(
                registry=self.registry,
                planner=self.planner,
                supervisor=self.supervisor,
                db=self.db,
            )
            self._catalog_cache = (now, full)
        else:
            full = cached[1]

        entries = full["models"]
        if model:
            record = self.registry.resolve(model)
            if record is None:
                raise ModelNotFoundError(model, known=self.registry.known_ids())
            entries = [e for e in entries if e["id"] == record.id]
        if compact:
            entries = [compact_entry(e) for e in entries]
        return {**full, "models": entries, "count": len(entries), "compact": compact}

    def plan_preview(
        self,
        name: str,
        *,
        ctx_size: int | None = None,
        kv_cache_type: Any = None,
        parallel: int | None = None,
    ) -> dict[str, Any]:
        """Fit verdict without loading anything — powers the GUI's live check."""
        record = self.registry.resolve(name)
        if record is None:
            raise ModelNotFoundError(name, known=self.registry.known_ids())
        result = self.planner.plan_load(
            record,
            ctx_size=ctx_size,
            kv_cache_type=kv_cache_type,
            parallel=parallel,
            loaded=self.supervisor.list(),
            draft=self._draft_for(record),
            adapters=[a for a, _ in self._adapters_for(record)],
        )
        if isinstance(result, LoadRejected):
            return {
                "fits": False,
                "model_id": record.id,
                "reason": result.reason,
                "message": result.message(),
                "required_bytes": result.required_bytes,
                "available_bytes": result.available_bytes,
                "per_gpu_free": result.per_gpu_free,
                "max_ctx_that_fits": result.max_ctx_that_fits,
                "max_parallel_that_fits": result.max_parallel_that_fits,
                "suggestions": result.suggestions,
                # Notes matter as much on a refusal as on a plan: "you asked
                # for more context than this model was trained for" is often
                # the reason the fit preview says no.
                "notes": result.notes,
                "vram_holders": [h.model_dump() for h in result.vram_holders],
                "estimate_mb": result.estimate.breakdown_mb(),
            }
        return {
            "fits": True,
            "model_id": record.id,
            "devices": result.devices,
            "tensor_split": result.tensor_split,
            "split_mode": result.split_mode,
            "ctx_size": result.ctx_size,
            "parallel": result.parallel,
            "kv_cache_type": result.kv_cache_type,
            "flash_attn": result.flash_attn,
            "per_gpu_bytes": result.per_gpu_bytes,
            "evict_model_ids": result.evict_model_ids,
            "notes": result.notes,
            "estimate_mb": result.estimate.breakdown_mb(),
            "single_gpu": result.fits_single_gpu,
        }

    async def test_model(self, name: str, prompt: str | None = None) -> dict[str, Any]:
        """Canned one-sentence completion; reports latency and tokens/sec."""
        import httpx

        record, instance = await self.ensure_loaded(name)
        # A preset-only virtual model serves from its base's instance, so the
        # URL and request accounting key off instance.model_id, not record.id.
        serving_id = instance.model_id
        base = self.supervisor.base_url(serving_id)
        if base is None:
            raise ModelLoadError(f"model '{record.id}' is not serving")

        started = time.perf_counter()
        if record.kind == "embedding":
            payload: dict[str, Any] = {"input": prompt or "StudioForge test embedding."}
            url = f"{base}/v1/embeddings"
        else:
            payload = {
                "model": record.id,
                "messages": [
                    {"role": "user", "content": prompt or "Reply with one short sentence."}
                ],
                "max_tokens": 64,
                "stream": False,
            }
            url = f"{base}/v1/chat/completions"

        self.supervisor.mark_request_start(serving_id)
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
        finally:
            self.supervisor.mark_request_end(serving_id)

        elapsed = time.perf_counter() - started
        if record.kind == "embedding":
            vectors = data.get("data", [])
            dims = len(vectors[0]["embedding"]) if vectors else 0
            return {
                "model_id": record.id,
                "ok": dims > 0,
                "latency_s": round(elapsed, 3),
                "embedding_dims": dims,
            }

        usage = data.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        text = ""
        choices = data.get("choices") or []
        if choices:
            text = (choices[0].get("message") or {}).get("content") or ""
        result: dict[str, Any] = {
            "model_id": record.id,
            "ok": bool(text),
            "latency_s": round(elapsed, 3),
            "completion_tokens": completion_tokens,
            "tokens_per_second": round(completion_tokens / elapsed, 2) if elapsed > 0 else None,
            "text": text[:400],
        }
        result.update(draft_stats(data.get("timings")))
        return result

    async def introspect(self, model_id: str) -> dict[str, Any]:
        """Actual running settings as llama-server reports them, plus slot state."""
        record = self.registry.resolve(model_id)
        if record is not None:
            # A preset-only virtual model runs inside its base's instance.
            model_id = self.serving_record(record).id
        instance = self.supervisor.get(model_id)
        if instance is None:
            return {"loaded": False}
        props = await self.supervisor.props(model_id)
        slots = await self.supervisor.slots(model_id)
        actual: dict[str, Any] = {}
        if props:
            defaults = props.get("default_generation_settings") or {}
            actual = {
                "n_ctx": defaults.get("n_ctx"),
                "total_slots": props.get("total_slots"),
                "model_path": props.get("model_path"),
                "model_alias": props.get("model_alias"),
                "chat_format": (defaults.get("params") or {}).get("chat_format"),
                "lora": (defaults.get("params") or {}).get("lora"),
                "modalities": props.get("modalities"),
                "build_info": props.get("build_info"),
                # Whether speculative decoding is actually armed comes from
                # /slots, NOT /props: verified against b10425, where
                # default_generation_settings.params["speculative.types"] stays
                # "none" even with a draft model loaded and drafting (it
                # describes per-request defaults, not the server's config). The
                # per-slot "speculative" flag is the truthful signal.
                "speculative": _slots_speculative(slots),
            }
        return {
            "loaded": True,
            "instance": instance.model_dump(mode="json"),
            "requested": instance.plan.model_dump(mode="json") if instance.plan else None,
            "actual": actual,
            "activity": slot_activity(slots),
            "props": props,
            "slots": slots,
        }

    def evictable_ids(self) -> Sequence[str]:
        return [i.model_id for i in self.planner._evictable(self.supervisor.list())]


def _slots_speculative(slots: list[dict[str, Any]] | None) -> bool | None:
    """Whether speculative decoding is armed, per llama-server's own report.

    Read from ``/slots``, not ``/props``. Verified against b10425: with a draft
    model loaded and demonstrably drafting (completion ``timings`` showed
    ``draft_n: 64, draft_n_accepted: 64``), ``/props`` still reported
    ``speculative.types: "none"`` -- that field describes per-request sampling
    defaults, not the server's speculative configuration. The per-slot
    ``speculative`` boolean is the only truthful signal the engine exposes.
    """
    if not slots:
        return None
    return any(bool(slot.get("speculative")) for slot in slots)


def slot_activity(slots: list[dict[str, Any]] | None) -> dict[str, Any]:
    """LM Studio-style live activity, derived from llama-server's slot state.

    Gives the Dashboard the same detail LM Studio surfaces -- whether a slot is
    ingesting the prompt (with progress) or generating (with a running token
    count) -- rather than a bare busy/idle flag. Field names differ across
    llama.cpp releases, so every read is defensive: a renamed field degrades one
    number to ``None`` instead of breaking the whole panel.
    """
    if not slots:
        return {"slots": [], "busy": 0, "idle": 0, "state": "idle", "tokens_generated": 0}

    entries: list[dict[str, Any]] = []
    busy = 0
    total_generated = 0
    for slot in slots:
        processing = bool(slot.get("is_processing"))
        prompt_total = _as_int(slot.get("n_prompt_tokens"))
        prompt_done = _as_int(slot.get("n_prompt_tokens_processed"))
        cached = _as_int(slot.get("n_prompt_tokens_cache"))
        generated = _as_int(slot.get("n_decoded")) or _as_int(slot.get("n_tokens_predicted"))

        if not processing:
            state = "idle"
        elif prompt_total and prompt_done is not None and prompt_done < prompt_total:
            state = "processing_prompt"
        else:
            state = "generating"

        if processing:
            busy += 1
        total_generated += generated or 0

        entries.append(
            {
                "id": slot.get("id"),
                "state": state,
                "label": _activity_label(state, prompt_done, prompt_total, generated),
                "n_ctx": slot.get("n_ctx"),
                "prompt_tokens": prompt_total,
                "prompt_tokens_processed": prompt_done,
                "prompt_tokens_cached": cached,
                "tokens_generated": generated,
                "speculative": slot.get("speculative"),
                "task_id": slot.get("id_task"),
            }
        )

    overall = "idle"
    if any(e["state"] == "generating" for e in entries):
        overall = "generating"
    elif any(e["state"] == "processing_prompt" for e in entries):
        overall = "processing_prompt"

    return {
        "slots": entries,
        "busy": busy,
        "idle": len(entries) - busy,
        "state": overall,
        "label": next(
            (e["label"] for e in entries if e["state"] != "idle"),
            "Idle",
        ),
        "tokens_generated": total_generated,
    }


def _activity_label(
    state: str, prompt_done: int | None, prompt_total: int | None, generated: int | None
) -> str:
    if state == "processing_prompt":
        if prompt_total:
            pct = int(100 * (prompt_done or 0) / prompt_total)
            return f"Processing prompt {prompt_done or 0}/{prompt_total} ({pct}%)"
        return "Processing prompt"
    if state == "generating":
        if generated:
            return f"Generating - {generated} tokens"
        return "Generating"
    return "Idle"


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def draft_stats(timings: Any) -> dict[str, Any]:
    """Speculative-decoding acceptance stats from a completion's ``timings``.

    ``draft_n`` / ``draft_n_accepted`` are the only place llama-server reports
    whether drafting actually happened for a request, and their ratio is the
    acceptance rate that decides whether a draft pairing is worth keeping -- a
    poorly matched draft can make generation *slower*, so surfacing this is what
    makes a bad pairing visible instead of mysterious.
    """
    if not isinstance(timings, dict):
        return {}
    drafted = timings.get("draft_n")
    accepted = timings.get("draft_n_accepted")
    if not isinstance(drafted, int) or drafted <= 0:
        return {"speculative_used": False}
    rate = None
    if isinstance(accepted, int):
        rate = round(accepted / drafted, 4)
    return {
        "speculative_used": True,
        "draft_tokens": drafted,
        "draft_tokens_accepted": accepted,
        "draft_acceptance_rate": rate,
        "engine_predicted_per_second": timings.get("predicted_per_second"),
        "engine_prompt_per_second": timings.get("prompt_per_second"),
    }
