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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from studioforge.config import Config, KvCacheType
from studioforge.core.gpu import vram_processes
from studioforge.core.leases import DEFAULT_IDLE_TTL_S, LeaseBook
from studioforge.core.planner import BUSY_RETRY_AFTER_S, OBSERVATION_NOTE_PER_PID_DEVICE, Planner
from studioforge.core.priority import PRIORITY_BACKGROUND, PriorityLock, normalise_priority
from studioforge.core.registry import Registry
from studioforge.core.supervisor import Supervisor
from studioforge.db import Database
from studioforge.errors import (
    BadRequestError,
    InsufficientVramError,
    LeaseConflictError,
    ModelBusyError,
    ModelLoadError,
    ModelNotFoundError,
    StudioForgeError,
)
from studioforge.logging import get_logger
from studioforge.types import (
    MB,
    AdapterRecord,
    GpuLease,
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


#: How recently a displaced model must have served to count as *active* and be
#: reloaded after a priority load lands (D46). The figure is D42's idleness
#: threshold for the same reason it is D42's: five minutes without a request
#: means the conversation has plausibly gone away, so the VRAM is better left
#: to whoever the priority load was for.
RESTORE_ACTIVE_WINDOW_S = 300.0


def reload_settings(plan: LoadPlan) -> dict[str, Any]:
    """The settings tuple a faithful reload of ``plan``'s instance needs.

    ``ctx_per_slot or ctx_size`` is the load-bearing subtlety: ``ctx_size`` on
    a plan is already per slot, but older plans carried only the total, and
    reloading a 4-slot model at its *total* context quadruples the ask. One
    implementation, because the D42 rebalancer and the D46 restore must never
    drift on it.
    """
    return {
        "ctx_size": int(plan.ctx_per_slot or plan.ctx_size),
        "kv_cache_type": plan.kv_cache_type,
        "kv_cache_type_v": plan.kv_cache_type_v,
        "parallel": int(plan.parallel),
    }


@dataclass
class _RestoreEntry:
    """What it takes to bring a displaced model back exactly as it was (D46).

    The five-field settings tuple is the D42 rebalancer's: everything else a
    faithful reload needs, the planner re-derives. ``devices`` is deliberately
    absent -- the old cards now belong to the priority model, so the restore
    lets the planner place it fresh, with no eviction licence at all.
    """

    model_id: str
    priority: int
    ctx_size: int
    kv_cache_type: Any
    kv_cache_type_v: Any
    parallel: int
    evicted_at: float


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
    kv_cache_type_v: Any = None,
    devices: Sequence[int] | None = None,
    known_devices: Sequence[int] | None = None,
) -> None:
    """Raise :class:`BadRequestError` for a load argument no load could use.

    Called by :meth:`ModelManager.load` (so the HTTP route, the MCP tool, the
    GUI and the benchmark all get it) rather than by each caller. ``0``/negative
    values used to fall through the planner's ``x or default`` idiom -- ``0``
    silently meant "default", ``-1`` was honoured verbatim into the argv.

    ``devices`` is checked against ``known_devices`` -- the CUDA indices the
    probe actually reports -- so ``devices: [7]`` on a four-card box is a 400
    naming the parameter rather than a planner refusal several frames deeper
    that reads like a VRAM problem.
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
    for param, value in (("kv_cache_type", kv_cache_type), ("kv_cache_type_v", kv_cache_type_v)):
        if value is not None and str(value) not in KV_CACHE_TYPES:
            raise BadRequestError(
                f"{param} must be one of {', '.join(KV_CACHE_TYPES)} (got {value!r})",
                param=param,
            )
    if devices is None:
        return
    indices = list(devices)
    if not indices:
        raise BadRequestError(
            "devices must name at least one CUDA index; omit it to let the planner choose",
            param="devices",
        )
    if len(set(indices)) != len(indices):
        raise BadRequestError(
            f"devices must not repeat a CUDA index (got {indices})",
            param="devices",
        )
    if known_devices is not None:
        unknown = sorted(set(indices) - set(known_devices))
        if unknown:
            raise BadRequestError(
                f"devices names CUDA index/indices {unknown}, which this machine does not "
                f"have (it has {sorted(known_devices)})",
                param="devices",
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


class ModelNotResidentError(ModelLoadError):
    """A load that existed only to relocate a resident found none to move (D42).

    Raised at the gate, not before it: the wait behind another load is the
    window in which a TTL unload can take the model away, and a relocation
    that proceeded anyway would cold-load a model nobody asked for.
    """


def _kv_rank(kv_k: str, kv_v: str) -> int:
    from studioforge.core.kv_sensitivity import kv_quality_rank

    return int(kv_quality_rank(kv_k, kv_v))


def measure_child_vram(
    probe: Any, pid: int, devices: Sequence[int]
) -> tuple[int, dict[int, int] | None]:
    """Our child's VRAM on ``devices``: ``(total_bytes, {device: bytes} | None)``.

    Three answers, tried in order (D40):

    1. **PDH per adapter, joined to CUDA ordinals** (Windows, D39) -- the only
       per-device figure Windows has. Summed over the plan's devices; a CUDA
       context the child opened on a card it was not placed on is left out,
       exactly as the plan leaves it out.
    2. **NVML per process per GPU** (Linux) -- ``vram_processes`` rows whose
       ``used_bytes`` NVML itself measured. Detected by their NOT all equalling
       the PDH per-process total: on Windows :func:`gpu._fill_missing_used_bytes`
       writes that one total onto every row of the pid, which is what made the
       old sum wrong, and which is why this path never sums rows that match it.
    3. **The PDH per-process total, once**, with no per-device split -- when
       the LUID map is unavailable. Still our child's bytes, still not counted
       per card.

    ``(0, None)`` when nothing can attribute, so the caller skips the row.
    """
    from studioforge.core.vram_holders import (
        OTHER_ADAPTER,
        pdh_process_dedicated_bytes,
        process_gpu_bytes,
    )

    wanted = [int(d) for d in devices]
    # A fresh sample: the child just became ready, and a cached counter read
    # from before its weights landed would say it holds nothing.
    split = process_gpu_bytes(pid, ttl_s=0.0)
    measured = {d: int(b) for d, b in split.items() if d != OTHER_ADAPTER and b > 0}
    if measured:
        per_device = {d: measured.get(d, 0) for d in wanted} if wanted else dict(measured)
        total = sum(per_device.values()) if wanted else sum(measured.values())
        if total > 0:
            return total, per_device
    pdh_total = int(pdh_process_dedicated_bytes(ttl_s=0.0).get(pid, 0) or 0)
    rows = [h for h in vram_processes(probe, own_pids=[pid]) if h.pid == pid and h.used_bytes > 0]
    rows = [h for h in rows if not wanted or h.gpu_index in wanted]
    if rows and not (pdh_total > 0 and all(h.used_bytes == pdh_total for h in rows)):
        # Genuine per-GPU figures (NVML on Linux): one row per device, summable.
        per_device = {}
        for row in rows:
            per_device[row.gpu_index] = per_device.get(row.gpu_index, 0) + int(row.used_bytes)
        return sum(per_device.values()), per_device
    if pdh_total > 0:
        return pdh_total, None
    return 0, None


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
        leases: LeaseBook | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.planner = planner
        self.supervisor = supervisor
        self.db = db
        self.version = version
        #: GPU leases (D43). Shared with the planner by the app so a lease
        #: granted here is honoured there; a manager built without one (tests)
        #: keeps a private, empty book.
        self.leases = leases if leases is not None else LeaseBook()
        self._started_at = time.time()
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()
        #: One load at a time, machine-wide. See :meth:`_load_locked` (D29):
        #: two cold loads planned side by side each see the VRAM the other has
        #: not allocated yet, both launch, one OOMs, the retry evicts the
        #: other, and the first client's request lands on a dead child.
        #: Priority-aware since D46: waiters are served best tier first, FIFO
        #: within a tier, so a chat load never queues behind background ones.
        self._load_gate = PriorityLock()
        #: model ids whose load is in flight behind the gate, for /health.busy.
        #: The supervisor's "loading" state only appears once a child exists;
        #: this covers the planning window too, which is where two callers used
        #: to collide.
        self._loading: set[str] = set()
        #: One smoke test at a time, and never while the server is busy (D36).
        #: `test_model` loads a model to poke it and unloads it afterwards; two
        #: of those at once, or one during somebody's stream, is the opposite of
        #: a health check.
        self._test_gate = asyncio.Lock()
        self._testing: str | None = None
        #: Set by :class:`~studioforge.core.benchmark.Benchmarker` when one is
        #: constructed, so `busy` can see a run in progress. The benchmarker
        #: lives on the app state, which this object cannot reach.
        self.benchmarker: Any = None
        self._ttl_task: asyncio.Task[None] | None = None
        self._autoload_task: asyncio.Task[None] | None = None
        #: The reconcile pass keeping pinned models resident (D41). Spawned by
        #: the TTL loop so a multi-minute cold load never stalls the sweeper,
        #: and held by reference for the same reason as the autoload task.
        self._reconcile_task: asyncio.Task[None] | None = None
        #: Serving ids a caller *explicitly* unloaded. A pin means "keep this
        #: loaded", but a deliberate unload outranks it until the model is
        #: loaded or pinned again -- otherwise "Unload" on a pinned model is a
        #: 15-second illusion and the panic button frees nothing (D41).
        self._pin_suppressed: set[str] = set()
        #: serving id -> (next_attempt_ts, current_delay_s): exponential
        #: backoff for pinned reloads that keep failing, so a model that no
        #: longer fits is retried politely instead of every sweep.
        self._pin_retry: dict[str, tuple[float, float]] = {}
        #: The placement-rebalance pass (D42): single-flight, like the pin
        #: reconciler, and for the same reason -- a reload takes minutes and
        #: the sweeper must keep sweeping underneath it.
        self._rebalance_task: asyncio.Task[None] | None = None
        #: model id -> when it was last rebalanced (or last suggested), so one
        #: model is never shuffled -- or logged about -- more than once per
        #: cooldown window.
        self._rebalance_last: dict[str, float] = {}
        #: When the rebalancer last scanned, and per model the layout it
        #: scanned against -- so the planner is asked once per changed world,
        #: not once per sweep (D42, amended).
        self._rebalance_checked_at = 0.0
        self._rebalance_seen: dict[str, tuple[Any, float]] = {}
        #: model id -> the tier its last explicit load asked for (D46), so a
        #: JIT reload of the chat model keeps being the chat model. In-memory
        #: on purpose: tiers describe what the clients are doing right now,
        #: and a restart is a clean slate the next explicit load repopulates.
        self._model_priority: dict[str, int] = {}
        #: model id -> tier, for chat/agent loads currently in flight (gate
        #: wait included). While any hold stands, new inference and new loads
        #: for models of a *worse* tier are refused with a 503 + Retry-After,
        #: so they drain off the cards instead of competing with the load.
        self._priority_holds: dict[str, int] = {}
        #: model id -> what it takes to bring back a recently-active model a
        #: priority load displaced (D46); consumed by the single-flight
        #: restore task right after the displacing load settles.
        self._restore_entries: dict[str, _RestoreEntry] = {}
        self._restore_task: asyncio.Task[None] | None = None
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
        if self._reconcile_task is not None:
            # Same reason: a pinned reload in flight must not race the drain.
            self._reconcile_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reconcile_task
            self._reconcile_task = None
        if self._rebalance_task is not None:
            self._rebalance_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._rebalance_task
            self._rebalance_task = None
        if self._restore_task is not None:
            # A restore reload in flight must not spawn children into the drain.
            self._restore_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._restore_task
            self._restore_task = None
        self._restore_entries.clear()

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

    def _effective_priority(self, model_id: str) -> int:
        """The tier a model is serving (or would load) at (D46).

        A running instance carries the truth; an unloaded model falls back to
        the tier its last explicit load asked for, so the chat model stays the
        chat model across a TTL unload and its JIT reload; a model nobody ever
        tiered is background.
        """
        instance = self.supervisor.get(model_id)
        if instance is not None:
            return instance.priority
        return self._model_priority.get(model_id, PRIORITY_BACKGROUND)

    def _resolve_tier(self, model_id: str, priority: int | None) -> int:
        """The tier this load runs at: the validated explicit ask, else memory."""
        tier = normalise_priority(priority)
        if tier is None:
            return self._model_priority.get(model_id, PRIORITY_BACKGROUND)
        return tier

    def _current_hold(self) -> tuple[str, int] | None:
        """The winning hold -- best tier, model id as the tiebreak -- or None.

        One implementation on purpose: the 503's named holder and
        ``busy.priority_hold`` must be the same answer.
        """
        holds = self._priority_holds
        if not holds:
            return None
        holder = min(holds, key=lambda mid: (holds[mid], mid))
        return holder, holds[holder]

    def admission_check(self, model_id: str) -> None:
        """Raise the D46 hold refusal this model's traffic would get right now.

        For the API layer: the streaming endpoints call this *before* the 200
        and the SSE stream begin, so a held request is a real HTTP 503 with
        ``Retry-After`` instead of an in-band error frame most clients treat
        as fatal. ``ensure_loaded`` runs the same check as the backstop.
        """
        self._refuse_if_held(model_id, priority=self._effective_priority(model_id))

    def _refuse_if_held(self, model_id: str, *, priority: int) -> None:
        """503 anything a standing chat/agent load outranks (D46).

        While a tier-1 or tier-2 load is in flight -- gate wait included --
        new inference for worse-tier models and new worse-tier loads are
        refused so those models drain, become evictable, and stop competing
        with the load for bandwidth. Traffic for the loading model itself
        queues as always (LM Studio parity), and equal-or-better tiers pass.
        """
        if model_id in self._priority_holds:
            return
        hold = self._current_hold()
        if hold is None:
            return
        holder, hold_p = hold
        if priority <= hold_p:
            return
        raise ModelBusyError(
            f"a priority {hold_p} model ('{holder}') is loading; requests and "
            f"loads for lower-priority models are held off until it is serving "
            f"so it gets the cards and the bandwidth. Retry shortly.",
            details={
                "busy": self.busy_snapshot(),
                "retry_after_s": BUSY_RETRY_AFTER_S,
                "priority_hold": {"model_id": holder, "priority": hold_p},
            },
        )

    async def ensure_loaded(
        self, name: str, *, source: str = "jit"
    ) -> tuple[ModelRecord, InstanceInfo]:
        """Resolve a model name and make sure it is serving, loading if needed.

        Concurrent callers for the same model all wait on one lock, so a burst
        of requests against a cold model produces exactly one llama-server
        start and every request is served once it is up -- queuing, never
        erroring, which is what LM Studio's JIT behaviour looks like to a
        client.

        The returned record is the one the caller *named* (a virtual model
        keeps its identity); the returned instance may belong to its base --
        use ``instance.model_id`` for anything that talks to the supervisor.

        A JIT load inherits the tier of the model's last explicit load (D46),
        so the chat model reloading mid-conversation gets the chat tier's
        rights; and while a better-tier load is in flight, requests for
        worse-tier models are refused here with a 503 + Retry-After rather
        than served -- they must drain, not compete.
        """
        await self._await_boot()
        record = self.registry.resolve(name)
        if record is None:
            raise ModelNotFoundError(name, known=self.registry.known_ids())
        serving = self.serving_record(record)
        priority = self._effective_priority(serving.id)
        self._refuse_if_held(serving.id, priority=priority)

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
                # Re-resolve the tier under the lock: an explicit load that
                # tiered this model while we queued must not be undone by a
                # JIT reload cold-loading it at the stale background tier.
                priority = self._effective_priority(serving.id)
                instance = await self._load_locked(
                    serving,
                    source=source,
                    priority=priority,
                    hold=priority < PRIORITY_BACKGROUND,
                )
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
        kv_cache_type_v: Any = None,
        parallel: int | None = None,
        devices: Sequence[int] | None = None,
        force: bool = False,
        source: str = "api",
        evict_busy: bool | None = None,
        allow_evict: bool | None = None,
        require_resident: bool = False,
        priority: int | None = None,
        hold_traffic: bool = True,
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

        ``devices`` is a **one-shot placement for this load only** (D36): the
        catalog's per-hardware-mode rows carry one in their ``load_args``, so
        "run this on the two 5090s" is a load argument rather than a settings
        edit. It is applied as a ``device_override`` on a *copy* of the record
        -- the persisted settings are never touched, so the next load without
        it goes back to letting the planner choose. ``kv_cache_type_v``
        likewise, for the ladder's asymmetric rung (q8_0 K with a q4_0 V).

        ``evict_busy`` is whether the eviction ladder may stop a model that is
        serving a request (D36). ``None`` (the default) follows ``force``: a
        caller who can see what they are interrupting and says ``force=true``
        means it. The two are separable because ``force`` also means "reload
        the running instance", and four callers need the reload without the
        licence to interrupt a stream -- ``load_recommended``, the parallel
        benchmark, the placement benchmark and the D42 rebalancer all pass
        ``evict_busy=False``. That licence is re-checked *at the gate*, not
        only here: the wait behind another load can be minutes, and a resident
        that picked up a request meanwhile is refused with ``retry_after_s``
        rather than stopped mid-stream.

        ``allow_evict`` is the planner's eviction policy for this one load;
        ``None`` keeps ``planner.on_insufficient``. The rebalancer passes
        ``False`` so the real move carries the same no-evict rule as its
        preview (D42). ``require_resident`` makes the load a pure relocation:
        if the instance it was meant to move is no longer ``ready`` when the
        gate opens (a TTL unload, a crash), nothing is loaded -- a move must
        never resurrect a model nobody asked for.

        ``priority`` is the load's tier (D46): 1 the active chat model, 2 a
        dispatched agent, 3 (or ``None``) background. An explicit tier is
        remembered per model, so the next load of the same model -- a JIT
        reload included -- keeps it. A tier-1 or tier-2 load plans as if idle
        lower-tier residents were gone (they are displaced from the cards it
        picks, and the recently-active ones reloaded afterwards where they
        fit), jumps the load queue, and -- unless ``hold_traffic`` is
        ``False``, the internal callers' setting -- holds new worse-tier
        traffic off with a 503 while it is in flight.
        """
        validate_load_args(
            ctx_size=ctx_size,
            parallel=parallel,
            kv_cache_type=kv_cache_type,
            kv_cache_type_v=kv_cache_type_v,
            devices=list(devices) if devices is not None else None,
            known_devices=self._known_devices(),
        )
        await self._await_boot()
        record = self.registry.resolve(name)
        if record is None:
            raise ModelNotFoundError(name, known=self.registry.known_ids())
        record = self.serving_record(record)
        if devices is not None:
            record = record.model_copy(
                update={
                    "settings": record.settings.model_copy(
                        update={"device_override": [int(d) for d in devices]}
                    )
                }
            )
        tier = self._resolve_tier(record.id, priority)
        if hold_traffic:
            self._refuse_if_held(record.id, priority=tier)
        if normalise_priority(priority) is not None:
            # An explicit tier is the caller saying what this model *is*;
            # remember it so a TTL unload and JIT reload keep it (D46).
            # Written only past the admission gate: a load the hold just
            # refused must not re-tier the model as a side effect.
            self._model_priority[record.id] = tier
        hold = tier < PRIORITY_BACKGROUND and hold_traffic
        lock = await self._lock_for(record.id)
        self._load_waiters[record.id] = self._load_waiters.get(record.id, 0) + 1
        try:
            async with lock:
                existing = self.supervisor.get(record.id)
                reload_of: str | None = None
                if existing is not None and existing.state == "ready":
                    if not force:
                        if normalise_priority(priority) is not None:
                            # Re-tiering a running model must reach the live
                            # instance -- the planner's candidacy and
                            # _effective_priority read it there, and a memo
                            # the instance contradicts is a silent no-op.
                            existing.priority = tier
                        return existing
                    reload_of = record.id
                return await self._load_locked(
                    record,
                    ctx_size=ctx_size,
                    kv_cache_type=kv_cache_type,
                    kv_cache_type_v=kv_cache_type_v,
                    parallel=parallel,
                    reload_of=reload_of,
                    force=force,
                    source=source,
                    evict_busy=force if evict_busy is None else evict_busy,
                    allow_evict=allow_evict,
                    require_resident=require_resident,
                    priority=tier,
                    hold=hold,
                )
        finally:
            # The same waiter bookkeeping ensure_loaded keeps: without it,
            # _prune_lock can discard a lock this call is queued on, and two
            # tasks end up inside _load_locked for one model -- the first
            # finisher then pops the second's hold mid-load.
            remaining = self._load_waiters.get(record.id, 1) - 1
            if remaining <= 0:
                self._load_waiters.pop(record.id, None)
                self._prune_lock(record.id)
            else:
                self._load_waiters[record.id] = remaining

    #: Contexts the "Load recommended" buttons offer, and the ones the MCP
    #: docstring names. Not a limit -- any integer up to the trained window is
    #: accepted -- but these four are what an agent asks for in practice, and
    #: naming them turns "how much context can I have" into a button.
    RECOMMENDED_CTX_TIERS: tuple[int, ...] = (65536, 131072, 262144, 524288)

    async def load_recommended(
        self,
        name: str,
        ctx_size: int,
        *,
        prefer_modes: Sequence[str] | None = None,
        kv_min: str | None = None,
        source: str = "api",
        priority: int | None = None,
    ) -> InstanceInfo:
        """Load at **exactly** ``ctx_size`` per slot, letting the server pick the rest.

        The user's sentence this exists for: *"so it can simply specify the
        model and context needed, and the server works the rest, or returns an
        error if it can't load the requested context for some reason"*.

        So the caller names two things and nothing else. This walks the hardware
        modes in headline order (``dual_5090`` -> ``dual_3090`` -> ``all_gpus``
        -> ``single_5090`` on this rig, or ``prefer_modes``) and for each asks
        the planner for that exact context per slot under the quality-first KV
        ladder (D36: f16 -> q8_0/q8_0 -> q8_0 K + q4_0 V, never a q4_0 K) with
        ``parallel = recommended_parallel``. The first mode that fits **now**
        wins; if none does, the same walk runs with eviction of **idle** models
        allowed -- never a busy one (D36) -- and only then does it refuse.

        **This is the one load path that is strict about context.** Everywhere
        else, a context that does not fit steps down a ladder (D14) because a
        roomier window is a nicety. Here the window is the request: an agent
        that asked for 262144 because its transcript is 200k long is not helped
        by silently getting 131072 and discovering it mid-conversation. So the
        refusal is structured and says, per mode, the largest context that
        *would* work and what is in the way.

        Args:
            name: model id or alias.
            ctx_size: tokens per slot, exactly. Above the model's
                ``n_ctx_train`` this is a 400 rather than an attempt -- serving
                past the trained window needs RoPE scaling and quietly degrades
                quality (D14), so it is refused with the number that would work.
            prefer_modes: hardware-mode keys to try, in this order, instead of
                the headline order. An unknown key is a 400 naming the ones this
                box has.
            kv_min: the lowest KV cache quality to accept -- ``"f16"`` means "do
                not quantize the cache to reach this window at all". Omitted,
                the full quality-first ladder is available.
            source: who asked, stamped on the instance and the log lines (D36).
            priority: the load's tier (D46): 1 the active chat model, 2 a
                dispatched agent, 3 (or omitted) background. A tier-1/2 load
                walks the modes with eviction of worse-tier idle models allowed
                from the first round, so the best mode wins even when
                background models must be displaced for it; the recently
                active ones are reloaded afterwards where they fit.
        """
        from studioforge.core import catalog as catalog_mod
        from studioforge.core import placements as placements_mod

        validate_load_args(ctx_size=ctx_size, parallel=None, kv_cache_type=None)
        await self._await_boot()
        record = self.registry.resolve(name)
        if record is None:
            raise ModelNotFoundError(name, known=self.registry.known_ids())
        record = self.serving_record(record)
        tier = self._resolve_tier(record.id, priority)
        self._refuse_if_held(record.id, priority=tier)
        if normalise_priority(priority) is not None:
            # Same rule as load(): an explicit tier is remembered, but only
            # past the admission gate (D46).
            self._model_priority[record.id] = tier
        if record.meta is None:
            raise ModelLoadError(
                f"'{record.id}' has no readable GGUF metadata, so its trained context "
                f"window and KV geometry are unknown and a load at an exact context "
                f"cannot be planned. Rescan the library; if the file is damaged, "
                f"re-download it.",
                details={"model_id": record.id, "stale_reason": record.stale_reason},
            )

        trained = int(getattr(record.meta, "n_ctx_train", 0) or 0)
        if trained > 0 and int(ctx_size) > trained:
            raise BadRequestError(
                f"'{record.id}' is trained to {trained} tokens; ask for {trained} or "
                f"fewer. Serving past the trained window needs RoPE scaling and "
                f"degrades quality, so this server will not do it silently.",
                param="ctx_size",
                details={"n_ctx_train": trained, "requested_ctx": int(ctx_size)},
            )

        modes = self._modes_for_recommendation(prefer_modes)
        observations = self.parallel_observations(record.id)

        # A model that is already resident is about to be RELOADED at the new
        # context (the load below is a forced reload), so the walk must see the
        # machine the way that reload will: with this instance's own footprint
        # credited back (D30's reload_of credit, D36's CreditedProbe -- the same
        # figure). Without the credit the walk planned against VRAM the model
        # itself was holding, refused "not enough VRAM" for a window the reload
        # one line later would have fitted, and under-reported the slot count.
        # And a resident model that is mid-conversation is not reloaded at all:
        # a load never interrupts a stream (D36), this one included.
        resident = self.supervisor.get(record.id)
        if resident is not None and resident.state != "ready":
            resident = None
        if resident is not None:
            if resident.active_requests > 0:
                raise ModelBusyError(
                    f"'{record.id}' is serving {resident.active_requests} request(s); "
                    f"loading it at {int(ctx_size)} tokens would interrupt them. Wait for "
                    f"the stream(s) to finish, or pass force=true to load_model to do it "
                    f"anyway.",
                    details={
                        "busy": self.busy_snapshot(),
                        "retry_after_s": BUSY_RETRY_AFTER_S,
                        "loaded_by": resident.loaded_by,
                    },
                )
            plan_now = resident.plan
            if (
                plan_now is not None
                and int(plan_now.ctx_per_slot or plan_now.ctx_size) == int(ctx_size)
                and (
                    kv_min is None
                    or _kv_rank(plan_now.kv_cache_type, plan_now.kv_cache_type_v)
                    <= _kv_rank(kv_min, kv_min)
                )
            ):
                # Already exactly that. Reloading would cost a cold start and
                # a window of "loading" for every client, to arrive where we are.
                log.info(
                    "already loaded at the requested context",
                    model_id=record.id,
                    ctx_size=int(ctx_size),
                    devices=plan_now.devices,
                    source=source,
                )
                return resident

        probe: Any = self.planner.probe
        if resident is not None:
            footprint = Planner.instance_footprint(resident)
            if footprint:
                probe = catalog_mod.CreditedProbe(self.planner.probe, footprint)
        planner = Planner(self.config, probe, log_plans=False)
        loaded = [i for i in self.supervisor.list() if i.model_id != record.id]

        # A chat/agent load (D46) walks with eviction of worse-tier idle
        # models allowed from the FIRST round: the headline mode that fits by
        # displacing background models beats a lesser mode that fits beside
        # them, which is the whole point of the tier. Background keeps the
        # gentler two-round walk -- fit beside everyone first.
        rounds: tuple[bool, ...] = (True,) if tier < PRIORITY_BACKGROUND else (False, True)
        attempts: list[dict[str, Any]] = []
        hold = tier < PRIORITY_BACKGROUND
        if hold:
            # D46: the hold must span the mode walk too, not only the final
            # load -- otherwise a worse-tier load can take the D29 gate while
            # this one is still choosing its placement. The inner self.load()
            # re-registers the same key and pops it; the outer pop is the
            # backstop for every refusal path out of the walk.
            self._priority_holds[record.id] = tier
        try:
            for allow_evict in rounds:
                attempts = [
                    self._mode_attempt(
                        record,
                        mode,
                        ctx_size=int(ctx_size),
                        kv_min=kv_min,
                        planner=planner,
                        loaded=loaded,
                        observations=observations,
                        allow_evict=allow_evict,
                        catalog_mod=catalog_mod,
                        placements_mod=placements_mod,
                        priority=tier,
                    )
                    for mode in modes
                ]
                winner = next((a for a in attempts if a["fits"]), None)
                if winner is None:
                    continue
                log.info(
                    "loading at the requested context",
                    model_id=record.id,
                    ctx_size=int(ctx_size),
                    mode=winner["mode"],
                    devices=winner["devices"],
                    kv_cache_type=winner["kv_cache_type"],
                    parallel=winner["parallel"],
                    evicting=winner["would_evict"],
                    source=source,
                )
                # force=True is the RELOAD half of force (the model may be
                # resident at another context); evict_busy=False keeps D36's
                # rule -- a model that is serving is never a candidate, and
                # this path has no override for that by design.
                return await self.load(
                    record.id,
                    ctx_size=int(ctx_size),
                    kv_cache_type=winner["kv_cache_type"],
                    kv_cache_type_v=winner["kv_cache_type_v"],
                    parallel=winner["parallel"],
                    devices=winner["devices"],
                    force=True,
                    source=source,
                    evict_busy=False,
                    priority=tier,
                )

            raise self._recommendation_refused(record, int(ctx_size), attempts, trained=trained)
        finally:
            if hold:
                self._priority_holds.pop(record.id, None)

    def _modes_for_recommendation(self, prefer_modes: Sequence[str] | None) -> list[Any]:
        """The hardware modes to walk, in the order to walk them."""
        from studioforge.core import placements as placements_mod

        modes = placements_mod.hardware_modes(
            self.planner.probe.list_gpus(), excluded=self.config.planner.excluded_devices
        )
        if not modes:
            raise InsufficientVramError(
                "no usable GPU was found, and this server is GPU-only",
                details={
                    "suggestions": [
                        "check the driver and `nvidia-smi`",
                        "planner.excluded_devices may be reserving every card",
                    ]
                },
            )
        if prefer_modes is None:
            return modes
        by_key = {m.key: m for m in modes}
        unknown = [key for key in prefer_modes if key not in by_key]
        if unknown:
            raise BadRequestError(
                f"unknown hardware mode(s): {', '.join(unknown)}; this box has: "
                + ", ".join(by_key),
                param="prefer_modes",
            )
        return [by_key[key] for key in prefer_modes]

    def _mode_attempt(
        self,
        record: ModelRecord,
        mode: Any,
        *,
        ctx_size: int,
        kv_min: str | None,
        planner: Planner,
        loaded: Sequence[InstanceInfo],
        observations: Sequence[Mapping[str, Any]],
        allow_evict: bool,
        catalog_mod: Any,
        placements_mod: Any,
        priority: int = PRIORITY_BACKGROUND,
    ) -> dict[str, Any]:
        """Can this mode hold ``ctx_size`` per slot, and if not, what is in the way?

        One planner call at one slot decides the fit and the KV rung (the ladder
        is the planner's, so this cannot drift from what an ordinary load would
        choose), then the slot count is worked out and the plan re-checked at it.
        Re-checked rather than assumed: the slot count comes from a capacity
        figure, and a placement that fits at one slot and not at four must come
        back as four-does-not-fit rather than as a load that fails.
        """
        from studioforge.core.kv_sensitivity import kv_quality_label, kv_quality_rank

        pinned = placements_mod.forced_onto(record, mode.devices)
        attempt: dict[str, Any] = {
            "mode": mode.key,
            "label": mode.label,
            "devices": list(mode.devices),
            "fits": False,
            "would_evict": [],
        }
        result = planner.plan_load(
            pinned,
            ctx_size=ctx_size,
            parallel=1,
            loaded=loaded,
            draft=self._draft_for(record),
            adapters=[a for a, _ in self._adapters_for(record)],
            allow_evict=allow_evict,
            priority=priority,
        )
        if not isinstance(result, LoadPlan):
            attempt["reason"] = result.reason
            attempt["max_ctx_that_fits"] = result.max_ctx_that_fits
            attempt["busy_models"] = list(result.busy_models)
            attempt["vram_holders"] = [h.model_dump() for h in result.vram_holders]
            attempt["rejected"] = result
            return attempt

        if kv_min is not None and kv_quality_rank(
            result.kv_cache_type, result.kv_cache_type_v
        ) > kv_quality_rank(kv_min, kv_min):
            attempt["reason"] = (
                f"only reaches {ctx_size} tokens with a "
                f"{kv_quality_label(result.kv_cache_type, result.kv_cache_type_v)} KV "
                f"cache, below the {kv_min} minimum this call asked for"
            )
            attempt["max_ctx_that_fits"] = None
            attempt["busy_models"] = []
            attempt["vram_holders"] = []
            return attempt

        slots, _bound, _vram = catalog_mod.slots_for_plan(planner, pinned, result)
        wanted = catalog_mod.recommended_slots(record, result, slots, observations=observations)
        plan = result
        parallel = int(wanted["value"])
        while parallel > 1:
            candidate = planner.plan_load(
                pinned,
                ctx_size=ctx_size,
                kv_cache_type=result.kv_cache_type,
                kv_cache_type_v=result.kv_cache_type_v,
                parallel=parallel,
                loaded=loaded,
                draft=self._draft_for(record),
                adapters=[a for a, _ in self._adapters_for(record)],
                allow_evict=allow_evict,
                # Same tier as the fit decision above: the slot count must be
                # judged in the same world (preemption credit included), or a
                # tier load's re-check silently walks down to one slot (D46).
                priority=priority,
            )
            if isinstance(candidate, LoadPlan):
                plan = candidate
                break
            parallel -= 1

        attempt.update(
            fits=True,
            ctx_size=ctx_size,
            kv_cache_type=plan.kv_cache_type,
            kv_cache_type_v=plan.kv_cache_type_v,
            parallel=parallel,
            recommended_parallel_basis=wanted["basis"],
            devices=list(plan.devices),
            would_evict=list(plan.evict_model_ids),
            vram_mb=round(plan.estimate.total_bytes / MB),
        )
        return attempt

    def _recommendation_refused(
        self,
        record: ModelRecord,
        ctx_size: int,
        attempts: Sequence[Mapping[str, Any]],
        *,
        trained: int,
    ) -> InsufficientVramError:
        """The structured "no", with the largest context each mode *would* take.

        A refusal here is terminal for the request (there is no CPU fallback and
        this path will not quietly shrink the window), so it has to be
        actionable in one read: per mode, the largest context that fits and what
        stands in the way; a ``retry_after_s`` only when something transient --
        a model that is serving right now -- is the cause, because "try again
        later" is bad advice when nothing is going to change (D36).
        """
        busy: list[dict[str, Any]] = []
        for attempt in attempts:
            for entry in attempt.get("busy_models") or []:
                if entry not in busy:
                    busy.append(dict(entry))
        best_ctx = max(
            (int(a.get("max_ctx_that_fits") or 0) for a in attempts),
            default=0,
        )
        modes = [
            {
                "mode": a["mode"],
                "label": a["label"],
                "devices": list(a["devices"]),
                "largest_ctx_that_fits": a.get("max_ctx_that_fits"),
                "reason": a.get("reason"),
                "busy_models": list(a.get("busy_models") or []),
            }
            for a in attempts
        ]
        suggestions: list[str] = []
        if best_ctx:
            suggestions.append(
                f"ask for {best_ctx} tokens instead -- that is the largest context "
                f"that fits on any of these placements right now"
            )
        if busy:
            names = ", ".join(f"{b['model_id']} ({b['active_requests']} in flight)" for b in busy)
            suggestions.append(
                f"or wait for {names} to finish; a load never interrupts a stream, "
                f"so this context becomes available when they do"
            )
        elif not best_ctx:
            suggestions.append(
                "unload something, or ask for a smaller context: no placement on this "
                "box reaches the requested window even with eviction allowed"
            )
        if trained:
            suggestions.append(f"this model's trained window is {trained} tokens")

        rejected = next(
            (a.get("rejected") for a in attempts if a.get("rejected") is not None), None
        )
        details: dict[str, Any] = {
            "requested_ctx": ctx_size,
            "n_ctx_train": trained or None,
            "modes": modes,
            "largest_ctx_that_fits": best_ctx or None,
            "busy_models": busy,
            "retry_after_s": BUSY_RETRY_AFTER_S if busy else None,
            "suggestions": suggestions,
        }
        if rejected is not None:
            details.update(
                required_bytes=rejected.required_bytes,
                available_bytes=rejected.available_bytes,
                per_gpu_free=rejected.per_gpu_free,
                vram_holders=[h.model_dump() for h in rejected.vram_holders],
                estimate_mb=rejected.estimate.breakdown_mb(),
            )
        return InsufficientVramError(
            f"Cannot load '{record.id}' at exactly {ctx_size} tokens per slot on any "
            f"placement of this box. " + " ".join(suggestions),
            details=details,
        )

    def _known_devices(self) -> list[int] | None:
        """CUDA indices the probe reports, or ``None`` when it cannot be asked.

        ``None`` rather than ``[]``: a probe that failed says nothing about what
        the box has, and refusing every device because NVML was unavailable
        would turn a diagnostic problem into a serving outage.
        """
        try:
            return [g.index for g in self.planner.probe.list_gpus()]
        except Exception:  # noqa: BLE001 - validation must not fail on a sick probe
            return None

    async def _load_locked(
        self,
        record: ModelRecord,
        *,
        ctx_size: int | None = None,
        kv_cache_type: Any = None,
        kv_cache_type_v: Any = None,
        parallel: int | None = None,
        reload_of: str | None = None,
        force: bool = False,
        source: str = "api",
        evict_busy: bool | None = None,
        allow_evict: bool | None = None,
        require_resident: bool = False,
        priority: int = PRIORITY_BACKGROUND,
        hold: bool = False,
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
                priority=priority,
            )
        # Visible in /health.busy.loading from the moment the load is asked
        # for, gate wait included: a load queued behind another one is still a
        # load in flight, and "the box was idle when we looked" must not be
        # true only between the gate and the spawn.
        self._loading.add(record.id)
        # The hold spans the same window (D46): from the moment a chat/agent
        # load is asked for -- its gate wait very much included -- until it
        # settles either way, worse-tier traffic is refused at admission.
        if hold:
            self._priority_holds[record.id] = priority
        try:
            await self._load_gate.acquire(priority)
            try:
                return await self._load_gated(
                    record,
                    ctx_size=ctx_size,
                    kv_cache_type=kv_cache_type,
                    kv_cache_type_v=kv_cache_type_v,
                    parallel=parallel,
                    reload_of=reload_of,
                    force=force,
                    source=source,
                    evict_busy=force if evict_busy is None else evict_busy,
                    allow_evict=allow_evict,
                    require_resident=require_resident,
                    priority=priority,
                )
            finally:
                self._load_gate.release()
        finally:
            self._loading.discard(record.id)
            if hold:
                self._priority_holds.pop(record.id, None)
            # Bring back what this load displaced, where it still fits --
            # also on failure: the victims are already gone either way (D46).
            self._kick_restore()

    async def _load_gated(
        self,
        record: ModelRecord,
        *,
        ctx_size: int | None,
        kv_cache_type: Any,
        kv_cache_type_v: Any = None,
        parallel: int | None,
        reload_of: str | None = None,
        force: bool = False,
        source: str = "api",
        evict_busy: bool | None = None,
        allow_evict: bool | None = None,
        require_resident: bool = False,
        priority: int = PRIORITY_BACKGROUND,
    ) -> InstanceInfo:
        existing = self.supervisor.get(record.id)
        if require_resident and (existing is None or existing.state != "ready"):
            # A relocation of a model that is no longer resident is not a
            # load: the rebalancer previewed a move for an idle child, and if
            # the TTL sweep unloaded it while the move waited at the gate,
            # the right outcome is "nothing" -- a cold load here would
            # resurrect a model nobody asked for, onto cards it was never
            # planned against (D42). Checked here rather than in the caller
            # because the gate wait *is* the window.
            raise ModelNotResidentError(
                f"'{record.id}' is no longer resident; there is nothing to relocate"
            )
        if reload_of is not None and (existing is None or existing.state != "ready"):
            # The resident child went away while we waited for the gate (a
            # crash, a TTL unload): there is nothing to replace, so this is an
            # ordinary load.
            reload_of = None
        interrupt_ok = force if evict_busy is None else evict_busy
        if (
            reload_of is not None
            and not interrupt_ok
            and existing is not None
            and existing.active_requests > 0
        ):
            # The box was idle when the caller looked, but that look was
            # before the per-model lock and the D29 gate -- a wait behind
            # another load is minutes, and ensure_loaded hands a ready
            # resident to requests with no lock at all. The reload licence
            # without the interrupt licence (D36/D42) has to be re-checked
            # here, where nothing between this line and supervisor.stop()
            # yields to the event loop; evict_busy alone cannot do it,
            # because the planner never sees the resident it is replacing.
            raise ModelBusyError(
                f"'{record.id}' is serving {existing.active_requests} request(s); "
                f"reloading it now would interrupt them. Wait for the stream(s) to "
                f"finish, or pass force=true to load_model to do it anyway.",
                details={
                    "busy": self.busy_snapshot(),
                    "retry_after_s": BUSY_RETRY_AFTER_S,
                    "loaded_by": existing.loaded_by,
                },
            )
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

        record, parallel_auto = self._apply_lease_profile(record)
        draft = self._draft_for(record)
        adapters = self._adapters_for(record)

        plan_result = self.planner.plan_load(
            record,
            ctx_size=ctx_size,
            kv_cache_type=kv_cache_type,
            kv_cache_type_v=kv_cache_type_v,
            parallel=parallel,
            loaded=self.supervisor.list(),
            draft=draft,
            adapters=[a for a, _ in adapters],
            reload_of=reload_of,
            parallel_auto=parallel_auto,
            # A model serving a request is never evicted to make room for
            # another one (D36); only an explicit force=true from a caller who
            # can see what they are interrupting overrides that, and a JIT load
            # can never set it. A caller that needs force's RELOAD half without
            # that licence passes evict_busy=False explicitly.
            evict_busy=interrupt_ok,
            # None defers to planner.on_insufficient; the rebalancer's move
            # carries its preview's allow_evict=False so that a world that
            # changed between preview and gate is refused, not evicted (D42).
            allow_evict=allow_evict,
            source=source,
            priority=priority,
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
                # The plan chose this victim while it was idle, but each stop
                # above yields for a child teardown -- it may be serving by
                # now, and D36 says a stream is never killed to make room
                # without the explicit licence. Refusing here leaves earlier
                # victims stopped; their restore entries stand, and the
                # _kick_restore in _load_locked's finally brings the active
                # ones back.
                inst = self.supervisor.get(victim)
                if inst is not None and inst.active_requests > 0 and not interrupt_ok:
                    raise ModelBusyError(
                        f"'{victim}' picked up {inst.active_requests} request(s) after "
                        f"the plan chose it for eviction; loading '{record.id}' now "
                        f"would interrupt them. Retry shortly.",
                        details={
                            "busy": self.busy_snapshot(),
                            "retry_after_s": BUSY_RETRY_AFTER_S,
                        },
                    )
                self._note_displaced(record, victim, priority=priority)
                log.info(
                    "evicting to make room", victim=victim, for_model=record.id, priority=priority
                )
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
                kv_cache_type_v=kv_cache_type_v,
                parallel=parallel,
                source=source,
                parallel_auto=parallel_auto,
                allow_evict=allow_evict,
                priority=priority,
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
        # Any successful load -- JIT, GUI, MCP, the reconciler itself -- means
        # the model is wanted up again: an earlier explicit unload no longer
        # holds the pin back, and a failing reload's backoff starts over (D41).
        self._pin_suppressed.discard(record.id)
        self._pin_retry.pop(record.id, None)
        # A fresh load of this model supersedes any pending restore of it.
        self._restore_entries.pop(record.id, None)
        self.registry.touch(record.id)
        # The instance carries the pid, and the pid is what makes the
        # observation about *this* child rather than about the whole card.
        self._record_actual_vram(record, instance.plan or plan, instance)
        return instance

    def _note_displaced(self, for_record: ModelRecord, victim: str, *, priority: int) -> None:
        """Remember a recently-active victim of a chat/agent load, to bring back (D46).

        Only tier-1/2 loads restore their victims: background-vs-background
        eviction is the pre-D46 LRU world, where a reload would be churn. A
        victim that was not recently active stays unloaded -- that is the
        deal: the user's chat takes the cards, the idle background model has
        no standing claim on coming back.
        """
        if priority >= PRIORITY_BACKGROUND:
            return
        instance = self.supervisor.get(victim)
        if instance is None or instance.plan is None:
            return
        now = time.time()
        last = instance.last_activity_at or instance.started_at or 0.0
        active = instance.active_requests > 0 or (now - last) <= RESTORE_ACTIVE_WINDOW_S
        if not active:
            log.info(
                "displaced model was idle and stays unloaded",
                model_id=victim,
                idle_s=round(now - last),
                for_model=for_record.id,
            )
            return
        self._restore_entries[victim] = _RestoreEntry(
            model_id=victim,
            priority=instance.priority,
            evicted_at=now,
            **reload_settings(instance.plan),
        )
        log.info(
            "displaced model was recently active; it is reloaded after the "
            "priority load where it fits",
            model_id=victim,
            for_model=for_record.id,
        )

    def _kick_restore(self) -> None:
        """Start the single-flight restore pass when displaced models wait on it."""
        if not self._restore_entries or self._draining:
            # A load that was parked at the gate when stop() ran can finish
            # during the drain; its kick must not spawn a task into shutdown.
            return
        if self._restore_task is not None and not self._restore_task.done():
            return
        self._restore_task = asyncio.create_task(
            self._restore_evicted(), name="sf-priority-restore"
        )

    async def _restore_evicted(self) -> None:
        """Reload the recently-active models a priority load displaced (D46).

        One attempt per entry, best tier first and earliest eviction within a
        tier, each with the exact settings tuple it ran with before (the D42
        five-field reload) on whatever cards the planner now picks -- with no
        eviction licence at all, so a restore can never displace anything in
        turn. An entry that no longer fits is dropped, not retried: "if it
        fits, reload it" was the promise, and a wants-list that lingers is the
        pin reconciler's job, not this pass's.
        """
        while self._restore_entries:
            if self._draining:
                # A shutdown mid-window: do not spawn children into the drain.
                self._restore_entries.clear()
                return
            model_id = min(
                self._restore_entries,
                key=lambda mid: (
                    self._restore_entries[mid].priority,
                    self._restore_entries[mid].evicted_at,
                ),
            )
            entry = self._restore_entries.pop(model_id)
            if self.supervisor.get(model_id) is not None:
                continue  # already back -- a JIT request beat this pass to it
            if model_id in self._pin_suppressed:
                continue  # explicitly unloaded meanwhile; down stays down (D41)
            try:
                await self.load(
                    model_id,
                    ctx_size=entry.ctx_size,
                    kv_cache_type=entry.kv_cache_type,
                    kv_cache_type_v=entry.kv_cache_type_v,
                    parallel=entry.parallel,
                    source="priority-restore",
                    evict_busy=False,
                    allow_evict=False,
                    priority=entry.priority,
                    hold_traffic=False,
                )
                log.info(
                    "restored after a priority eviction",
                    model_id=model_id,
                    priority=entry.priority,
                    ctx_size=entry.ctx_size,
                )
            except (InsufficientVramError, ModelBusyError) as exc:
                log.info(
                    "displaced model stays unloaded: no room left after the priority load",
                    model_id=model_id,
                    reason=getattr(exc, "message", str(exc)),
                )
            except StudioForgeError as exc:
                log.warning(
                    "restore after a priority eviction failed",
                    model_id=model_id,
                    error=getattr(exc, "message", str(exc)),
                )
            except Exception:  # pragma: no cover - defensive
                log.exception("restore after a priority eviction crashed", model_id=model_id)

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
                # A box that is BUSY rather than full: these models would have
                # freed the VRAM but are serving right now, so the refusal is
                # worth retrying and says how long to wait (D36).
                "busy_models": rejected.busy_models,
                "retry_after_s": rejected.retry_after_s,
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
        kv_cache_type_v: Any = None,
        parallel: int | None = None,
        source: str = "api",
        parallel_auto: bool = False,
        allow_evict: bool | None = None,
        priority: int = PRIORITY_BACKGROUND,
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

        A load planned with ``allow_evict=False`` (the D42 relocation) does
        not evict to retry either: the retry's whole point is to change the
        conditions by freeing a bystander, and that is exactly the licence
        such a load does not have. The trade-off is honest -- the resident
        was already stopped for the reload (D30 plans before it unloads, but
        the launch itself failed), so the model stays down until the next JIT
        request or the pin reconciler, which is what any forced reload that
        dies at launch leaves behind today.
        """
        try:
            return await self.supervisor.start(
                record,
                plan,
                engine_tag=engine_tag,
                draft=draft,
                adapters=adapters,
                source=source,
                priority=priority,
            )
        except ModelLoadError as exc:
            stderr = exc.details.get("stderr")
            tail = [str(line) for line in stderr] if isinstance(stderr, list) else []
            kind = classify_load_failure(tail)
            if kind != "transient":
                raise
            if allow_evict is False:
                log.warning(
                    "transient load failure; a no-evict load does not evict to retry",
                    model_id=record.id,
                )
                raise
            victim = self._eviction_candidate(exclude=record.id, for_priority=priority)
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
            # Same restore promise as a planner-chosen victim (D46): a
            # recently-active model displaced by the retry comes back too.
            self._note_displaced(record, victim, priority=priority)
            await self.supervisor.stop(victim)

        # Re-plan: free VRAM has changed, so placement and context may too.
        # The caller's explicit overrides (ctx_size/kv/parallel) must survive
        # the retry -- replanning without them would silently load the model
        # with a different context than the one the user asked for.
        replanned = self.planner.plan_load(
            record,
            ctx_size=ctx_size,
            kv_cache_type=kv_cache_type,
            kv_cache_type_v=kv_cache_type_v,
            parallel=parallel,
            loaded=self.supervisor.list(),
            draft=draft,
            adapters=[a for a, _ in adapters],
            parallel_auto=parallel_auto,
            allow_evict=allow_evict,
            priority=priority,
        )
        if isinstance(replanned, LoadRejected):
            raise self._vram_error(replanned)
        # The replan may carry victims of its own -- routinely so for a
        # tier-1/2 load, whose preemption credit lists the lower-tier
        # residents on its chosen cards (D46). Launching without stopping
        # them spawns the child into VRAM the plan only imagined free.
        for victim in replanned.evict_model_ids:
            if self.supervisor.get(victim) is None:
                continue
            self._note_displaced(record, victim, priority=priority)
            log.info("evicting to make room on the retry", victim=victim, for_model=record.id)
            await self.supervisor.stop(victim)
        instance = await self.supervisor.start(
            record,
            replanned,
            engine_tag=engine_tag,
            draft=draft,
            adapters=adapters,
            source=source,
            priority=priority,
        )
        log.info("load succeeded on retry", model_id=record.id)
        return instance

    def _eviction_candidate(self, *, exclude: str, for_priority: int | None = None) -> str | None:
        """Best eviction victim -- worst tier, LRU within it -- or None.

        ``for_priority`` keeps the D46 rule on the transient-OOM retry path
        too: a background load's retry may not free the chat model.
        """
        for instance in self.planner._evictable(self.supervisor.list(), for_priority=for_priority):
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

    def set_pinned(self, model_id: str, pinned: bool) -> tuple[ModelRecord, int]:
        """Persist the pin and make it bite now. One implementation for every surface.

        The HTTP route, the GUI toggle and the MCP tool all used to compose
        ``save_settings`` + ``refresh_ttl`` themselves, and the GUI forgot the
        second half -- a pin that only applied on the next load. Returns the
        updated record and the effective TTL (0 == pinned).

        Pinning also re-arms the reconciler (D41): an explicit unload's
        suppression and any retry backoff are cleared, so a freshly pinned
        model is brought up by the next sweep rather than whenever the old
        bookkeeping expires.
        """
        record = self.registry.resolve(model_id)
        if record is None:
            raise ModelNotFoundError(model_id, known=self.registry.known_ids())
        settings = record.settings.model_copy(update={"pinned": pinned})
        updated = self.registry.save_settings(record.id, settings)
        effective = self.refresh_ttl(updated.id)
        if pinned:
            serving_id = self.serving_record(updated).id
            self._pin_suppressed.discard(serving_id)
            self._pin_retry.pop(serving_id, None)
        return updated, effective if effective is not None else self.ttl_for(updated)

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

        Measures **our child's own** VRAM, per pid **and per device**, not the
        devices' total ``used_bytes``. The device total is whatever else happens
        to be on the card -- the desktop compositor, ComfyUI, another model of
        ours -- so it answered a question nobody asked and answered it
        differently every time. Over 540 historical rows it produced a median
        actual/predicted ratio of 2.97 against a documented 0.81-1.23.

        The per-device split matters as much as the pid (D40). The previous
        version summed :func:`vram_processes` rows over the plan's devices, and
        on Windows every one of those rows carries the *same* PDH per-process
        total -- so a two-GPU load was recorded at twice its footprint and a
        four-GPU load at four times, and D18's calibration pegged the overhead
        fraction at its ceiling on every boot. :func:`measure_child_vram`
        reads the per-adapter figure instead, and the observation carries both
        the plan's share and the measured bytes per card.

        Neither PDH nor NVML can attribute per process everywhere (containers,
        WSL, MIG). When nothing can, the observation is skipped rather than
        falling back to the device total: no data beats data that means
        something else.
        """
        try:
            pid = instance.pid
            if pid is None:
                return
            actual, per_device = measure_child_vram(self.planner.probe, pid, plan.devices)
            if actual > 0:
                self.planner.observe(
                    model_id=record.id,
                    plan=plan,
                    actual_bytes=actual,
                    note=OBSERVATION_NOTE_PER_PID_DEVICE,
                    per_gpu_actual=per_device,
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

    async def unload(self, name: str, *, deliberate: bool = True) -> bool:
        """Stop the child serving ``name``; a deliberate unload sticks (D41).

        The stopped id is marked so the pin reconciler leaves it down: whoever
        called this could see the model was pinned and asked anyway. Loading
        or re-pinning the model lifts the mark; the pin *setting* is untouched.

        ``deliberate=False`` is for housekeeping teardown -- the benchmarks'
        fresh-process-per-mode and leave-as-found unloads, ``test_model``
        putting the rig back -- where nobody chose to take the model down.
        Those stop without the mark, like the lease eviction and the TTL sweep
        do, so a pinned model that was benchmarked or tested is brought back
        by the reconciler instead of staying down until a restart.
        """
        record = self.registry.resolve(name)
        model_id = record.id if record is not None else name
        if deliberate:
            # An explicit unload also cancels a pending priority restore:
            # whoever called this wants the model DOWN, not down-for-a-moment.
            self._restore_entries.pop(model_id, None)
        if self.supervisor.get(model_id) is None:
            return False
        if deliberate:
            self._pin_suppressed.add(model_id)
        await self.supervisor.stop(model_id)
        return True

    async def unload_all(self) -> list[str]:
        """Stop every child. Pinned ones stay down until loaded or re-pinned (D41)."""
        ids = [i.model_id for i in self.supervisor.list()]
        self._pin_suppressed.update(ids)
        self._restore_entries.clear()
        await self.supervisor.stop_all()
        return ids

    # -- GPU leases (D43) --------------------------------------------------

    async def acquire_lease(
        self,
        devices: Sequence[int],
        *,
        holder: str,
        model_ids: Sequence[str] = (),
        reason: str = "",
        idle_ttl_s: float | None = DEFAULT_IDLE_TTL_S,
        force: bool = False,
    ) -> GpuLease:
        """Give ``devices`` to ``model_ids`` (or to nobody) until released or idle.

        Idle residents on those cards that are not the new owners are unloaded
        -- that is what "the card is yours" means. A resident mid-request is
        never interrupted: the call refuses with ``retry_after_s`` (D36). A
        *pinned* idle resident refuses too unless ``force`` is set, because a
        pin is a standing wish; with ``force`` it is evicted and the pin
        reconciler brings it back on other cards if they fit, or on these once
        the lease ends. A chat- or agent-tier idle resident (D46) refuses the
        same way: a lease grant does not outrank the tiers without ``force``.
        A card already leased is a conflict, never a takeover.
        """
        wanted = sorted({int(d) for d in devices})
        if not wanted:
            raise BadRequestError("a lease must name at least one CUDA device", param="devices")
        known = self._known_devices()
        if known is not None:
            unknown = [d for d in wanted if d not in known]
            if unknown:
                raise BadRequestError(
                    f"unknown CUDA device(s) {unknown}; this box has {known}", param="devices"
                )
        owners: list[str] = []
        for name in model_ids:
            record = self.registry.resolve(name)
            if record is None:
                raise ModelNotFoundError(name, known=self.registry.known_ids())
            serving_id = self.serving_record(record).id
            if serving_id not in owners:
                owners.append(serving_id)
        if self.leases.conflicts(wanted):
            # Let the book phrase the conflict: it names the lease in the way.
            self.leases.acquire(wanted, holder=holder, model_ids=owners)
        if self._loading:
            raise ModelBusyError(
                f"a model load is in flight ({', '.join(sorted(self._loading))}) and may "
                f"land on these cards; retry once it settles",
                details={"retry_after_s": BUSY_RETRY_AFTER_S, "loading": sorted(self._loading)},
            )
        busy: list[InstanceInfo] = []
        pinned: list[InstanceInfo] = []
        tiered: list[InstanceInfo] = []
        victims: list[InstanceInfo] = []
        for instance in self.supervisor.list():
            if instance.model_id in owners or instance.plan is None:
                continue
            if not (set(instance.plan.devices) & set(wanted)):
                continue
            if instance.state == "loading" or instance.active_requests > 0:
                busy.append(instance)
            elif instance.ttl_s == 0 and not force:
                pinned.append(instance)
            elif instance.priority < PRIORITY_BACKGROUND and not force:
                # A chat/agent-tier resident is a standing claim the way a pin
                # is (D46): "background can never displace it" must hold for
                # the lease plane too, or reserve_gpus is an end-run around
                # the tier rule. force works exactly as it does for a pin.
                tiered.append(instance)
            else:
                victims.append(instance)
        if busy:
            names = ", ".join(f"{i.model_id} ({i.active_requests} in flight)" for i in busy)
            raise ModelBusyError(
                f"{names} is serving on CUDA {wanted}; a lease never interrupts a stream "
                f"(D36). Retry when it is idle.",
                details={
                    "retry_after_s": BUSY_RETRY_AFTER_S,
                    "busy_models": [
                        {"model_id": i.model_id, "active_requests": i.active_requests} for i in busy
                    ],
                },
            )
        if pinned:
            names = ", ".join(i.model_id for i in pinned)
            raise LeaseConflictError(
                f"pinned model(s) {names} are resident on CUDA {wanted}; pass force=true to "
                f"evict them -- the pin reconciler brings them back on other cards if they "
                f"fit, or on these once the lease ends",
                param="force",
                details={"pinned": [i.model_id for i in pinned]},
            )
        if tiered:
            names = ", ".join(f"{i.model_id} (priority {i.priority})" for i in tiered)
            raise LeaseConflictError(
                f"higher-priority model(s) {names} are resident on CUDA {wanted}; a lease "
                f"grant does not outrank the chat or agent tier (D46). Pass force=true to "
                f"evict them anyway, or lease other cards",
                param="force",
                details={"priority_models": [i.model_id for i in tiered]},
            )
        # Book first, evict second. The planner reads the book, and every
        # supervisor.stop() below yields to the event loop for the length of a
        # child teardown: with the entry written afterwards, a JIT load (or
        # the pin reconciler, or the rebalancer) planning during those awaits
        # saw the cards unblocked and landed on the VRAM the victim had just
        # freed, and two overlapping grants both passed conflicts() and both
        # evicted before one of them lost at the book. Nothing between the
        # checks above and this line awaits, so the scan and the entry are one
        # atomic step; a failed eviction takes the entry back out because every
        # caller only ever releases a lease it was handed.
        lease = self.leases.acquire(
            wanted, holder=holder, model_ids=owners, reason=reason, idle_ttl_s=idle_ttl_s
        )
        try:
            for victim in victims:
                current = self.supervisor.get(victim.model_id)
                if current is None:
                    continue  # gone on its own while an earlier victim was torn down
                if current.active_requests > 0:
                    # Idle at the scan, busy now: the stop has no busy guard of
                    # its own, and D36 holds across the whole grant.
                    raise ModelBusyError(
                        f"{current.model_id} picked up {current.active_requests} request(s) "
                        f"on CUDA {wanted} while the lease was being granted; a lease never "
                        f"interrupts a stream (D36). Retry when it is idle.",
                        details={
                            "retry_after_s": BUSY_RETRY_AFTER_S,
                            "busy_models": [
                                {
                                    "model_id": current.model_id,
                                    "active_requests": current.active_requests,
                                }
                            ],
                        },
                    )
                log.info(
                    "evicting for a gpu lease",
                    victim=victim.model_id,
                    devices=wanted,
                    holder=holder,
                    for_models=owners,
                )
                await self.supervisor.stop(victim.model_id)
        except BaseException:
            # CancelledError included: a cancelled benchmark must not leave a
            # lease standing that nobody holds a handle to.
            self.leases.release(lease.id)
            raise
        log.info(
            "gpu lease acquired",
            lease_id=lease.id,
            devices=wanted,
            holder=holder,
            model_ids=owners,
            idle_ttl_s=idle_ttl_s,
            reason=reason or None,
        )
        return lease

    def release_lease(self, lease_id: str) -> GpuLease:
        lease = self.leases.release(lease_id)
        log.info(
            "gpu lease released",
            lease_id=lease.id,
            devices=list(lease.devices),
            holder=lease.holder,
        )
        return lease

    def touch_lease(self, lease_id: str) -> GpuLease:
        return self.leases.touch(lease_id)

    def _expire_leases(self) -> None:
        """Release leases idle past their ttl; an owner's own activity counts as a touch.

        In-memory only: a handful of dict lookups per sweep.
        """
        if not len(self.leases):
            return
        for lease in self.leases.all():
            for model_id in lease.model_ids:
                instance = self.supervisor.get(model_id)
                if instance is None:
                    continue
                if instance.state == "loading":
                    self.leases.touch(lease.id)
                    continue
                last = instance.last_activity_at or instance.started_at
                if last:
                    self.leases.touch(lease.id, at=last)
        for lease in self.leases.expired():
            self.leases.release(lease.id)
            log.info(
                "gpu lease released by the system: idle past its ttl",
                lease_id=lease.id,
                holder=lease.holder,
                devices=list(lease.devices),
                model_ids=list(lease.model_ids),
                idle_ttl_s=lease.idle_ttl_s,
            )

    def _apply_lease_profile(self, record: ModelRecord) -> tuple[ModelRecord, bool]:
        """Steer a leased model onto its cards in the measured-fastest shape (D43).

        Returns ``(record, parallel_auto)``. A lease naming this model means
        the cards are its alone, so: the placement is forced onto the lease's
        devices; the split mode (and micro-batch) come from the model's latest
        benchmark *on exactly those devices* when one exists -- tensor split
        measured slower than layer on this rig for a small model and is
        unmeasured for large ones (D38), so only a measurement may pick it;
        and the slot count is sized by the estimator even when
        ``models.default_parallel`` is an integer. An explicit per-model
        ``device_override``, ``split_mode`` or ``parallel`` still wins.
        """
        lease = self.leases.for_model(record.id)
        if lease is None:
            return record, False
        settings = record.settings
        updates: dict[str, Any] = {}
        basis = "default"
        if settings.device_override is None:
            updates["device_override"] = list(lease.devices)
        if settings.split_mode is None:
            best = self._measured_best_mode(record.id, lease.devices)
            if best is not None:
                updates["split_mode"] = best.get("split_mode") or "layer"
                if best.get("ubatch") and settings.ubatch_size is None:
                    updates["ubatch_size"] = int(best["ubatch"])
                basis = f"measured: {best.get('mode')}"
        parallel_auto = settings.parallel is None
        if updates:
            record = record.model_copy(update={"settings": settings.model_copy(update=updates)})
        log.info(
            "lease performance profile",
            model_id=record.id,
            lease_id=lease.id,
            devices=list(lease.devices),
            split_mode=record.settings.split_mode or "layer",
            split_basis=basis,
            parallel="auto" if parallel_auto else record.settings.parallel,
        )
        return record, parallel_auto

    def _measured_best_mode(self, model_id: str, devices: Sequence[int]) -> dict[str, Any] | None:
        """The winning generation mode of the latest benchmark on exactly ``devices``."""
        reader = getattr(self.db, "latest_benchmark", None)
        if reader is None:
            return None
        try:
            row = reader(model_id)
        except Exception:  # noqa: BLE001 - a missing table is not a load failure
            return None
        report = row.get("report") if isinstance(row, dict) else None
        if not isinstance(report, dict):
            return None
        best = report.get("best_generation_mode")
        for result in report.get("results") or []:
            if not isinstance(result, dict) or result.get("mode") != best:
                continue
            if sorted(int(d) for d in result.get("devices") or []) == sorted(devices):
                return result
        return None

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
                self._expire_leases()
                self._maybe_reconcile_pinned()
                self._maybe_rebalance()
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

    # -- pinned-model reconciler (D41) --------------------------------------

    #: First retry delay after a pinned reload fails, and its ceiling. The
    #: sweep runs every ~15s; retrying a load that just failed on the very
    #: next sweep would mostly re-measure the same full rig.
    PIN_RETRY_BASE_S = 60.0
    PIN_RETRY_CAP_S = 900.0

    def _maybe_reconcile_pinned(self) -> None:
        """Spawn one reconcile pass when a pinned model is down and wanted up.

        A task rather than an await: a cold load takes minutes, and the TTL
        sweeper must keep sweeping while it runs. At most one pass is in
        flight; the pass itself serialises behind the load gate like any
        other caller (D29).
        """
        if self._draining or not self.config.models.auto_load_pinned:
            return
        if self._autoload_task is not None and not self._autoload_task.done():
            return  # boot warm-up still owns the job
        if self._reconcile_task is not None and not self._reconcile_task.done():
            return
        wanted = self._pinned_needing_load()
        if not wanted:
            return
        self._reconcile_task = asyncio.create_task(
            self._reconcile_pinned(wanted), name="studioforge-pin-reconcile"
        )

    def _pinned_needing_load(self) -> list[str]:
        """Serving ids of pinned models that should be resident but are not.

        Skips a model that is already up or coming up, one a caller
        deliberately unloaded (:meth:`unload` marks those), one whose load is
        in flight, one inside its failure backoff window, and one under a
        smoke test or a benchmark -- those runs unload and reload the model
        themselves (without the deliberate-unload mark, D41), and a reconcile
        load spawned in their unload->lease->load gap would race the run and
        fail its next grant with "a model load is in flight".
        """
        now = time.time()
        under_test = {self._testing, getattr(self.benchmarker, "benchmarking", None)}
        wanted: list[str] = []
        for record in self.registry.all():
            if not record.settings.pinned:
                continue
            serving_id = self.serving_record(record).id
            if serving_id in wanted or serving_id in self._loading or serving_id in under_test:
                continue
            instance = self.supervisor.get(serving_id)
            if instance is not None and instance.state in ("ready", "loading"):
                continue
            if serving_id in self._pin_suppressed:
                continue
            next_at, _delay = self._pin_retry.get(serving_id, (0.0, 0.0))
            if now < next_at:
                continue
            wanted.append(serving_id)
        return wanted

    async def _reconcile_pinned(self, wanted: list[str]) -> None:
        """Reload each missing pinned model, backing off per model on failure.

        Loads carry no special licence: the planner treats them like the boot
        autoload (idle unpinned models may be evicted per policy, a busy model
        never -- D36), and a refusal is a WARNING plus a longer wait, not a
        crash of the sweep.
        """
        for serving_id in wanted:
            if self._draining:
                return
            try:
                await self.load(serving_id, source="pin-reconcile", hold_traffic=False)
                log.info("reloaded pinned model", model_id=serving_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one model must not block the rest
                _next, delay = self._pin_retry.get(serving_id, (0.0, 0.0))
                delay = min(max(delay * 2, self.PIN_RETRY_BASE_S), self.PIN_RETRY_CAP_S)
                self._pin_retry[serving_id] = (time.time() + delay, delay)
                log.warning(
                    "pinned model failed to reload; backing off",
                    model_id=serving_id,
                    retry_in_s=round(delay),
                    error=str(exc),
                )

    # -- placement rebalancer (D42) ------------------------------------------

    #: How long a model must have been idle before it may be relocated. A
    #: relocation is a reload, and a reload drops the child's prompt cache --
    #: on an agent/RP workload that cache is a ~100k-token reprocess, so a
    #: model is only moved when its conversation has plausibly gone away.
    REBALANCE_MIN_IDLE_S = 300.0
    #: Per-model floor between relocations (and between "suggest" log lines).
    #: The win is removing a *standing* misplacement; anything that needs to
    #: move more often than this is churn, not optimisation.
    REBALANCE_COOLDOWN_S = 1800.0
    #: How often the rebalancer *looks* at all, and how long an unchanged
    #: layout's answer is trusted before the planner is asked again. The
    #: sweep runs every 15 s; a planner preview is NVML plus estimate
    #: arithmetic, and the steady state must cost nothing.
    REBALANCE_CHECK_S = 60.0
    REBALANCE_RECHECK_S = 600.0

    def _maybe_rebalance(self) -> None:
        """Spawn one relocation when a resident model has a strictly better home.

        Placements accrete: each plan was optimal for the instant it was made,
        and nothing used to revisit it when the world changed. Measured on this
        rig -- a 27B placed on [1, 3] because a 31B held [1, 0, 2], still
        sharing GPU1 long after the 31B's reload shrank it to [0, 1], while
        both 3090s sat free. This pass re-asks the planner's question for idle
        residents, on a quiet box, and acts only on a strict improvement.
        """
        mode = getattr(self.config.planner, "rebalance", "off")
        if mode not in ("suggest", "auto") or self._draining:
            return
        if self._rebalance_task is not None and not self._rebalance_task.done():
            return
        if self._reconcile_task is not None and not self._reconcile_task.done():
            return
        if self._busy_reason() is not None:
            # Any in-flight request, load, test or benchmark: a relocation
            # would compete with it for PCIe and CPU, and the layout it was
            # computed against is about to change anyway.
            return
        now = time.time()
        if now - self._rebalance_checked_at < self.REBALANCE_CHECK_S:
            return
        self._rebalance_checked_at = now
        found = self._rebalance_opportunity()
        if found is None:
            return
        model_id, plan, reason = found
        self._rebalance_last[model_id] = time.time()
        if mode == "suggest":
            log.info(
                "rebalance available (planner.rebalance=suggest; not acting)",
                model_id=model_id,
                to_devices=list(plan.devices),
                reason=reason,
            )
            return
        self._rebalance_task = asyncio.create_task(
            self._rebalance(model_id, plan, reason), name="studioforge-rebalance"
        )

    def _rebalance_opportunity(self) -> tuple[str, LoadPlan, str] | None:
        """The first idle resident with a strictly better no-evict placement.

        "Strictly better" is deliberately narrow -- two triggers, both about
        standing costs, neither about chasing throughput estimates:

        * the model shares a card with another resident and a placement with
          no sharing exists at the same context, KV type and slot count;
        * a placement on strictly fewer cards exists at the same settings
          (every extra card in a layer split is a synchronisation hop).

        The candidate plan must evict nothing (its own footprint is credited
        back, D30), the model must have been idle for REBALANCE_MIN_IDLE_S,
        and a persisted ``device_override`` -- the user said "run it here" --
        is never second-guessed. Neither is a lease owner (D43): it sits on
        exactly its lease's cards by construction, so there is nothing to
        re-ask.
        """
        now = time.time()
        loaded = self.supervisor.list()
        layout = self._layout_fingerprint(loaded)
        for instance in loaded:
            if instance.state != "ready" or instance.active_requests > 0:
                continue
            plan = instance.plan
            if plan is None or not plan.devices:
                continue
            last = instance.last_activity_at or instance.started_at or now
            if now - last < self.REBALANCE_MIN_IDLE_S:
                continue
            if now - self._rebalance_last.get(instance.model_id, 0.0) < self.REBALANCE_COOLDOWN_S:
                continue
            record = self.registry.get(instance.model_id)
            if record is None or record.settings.device_override is not None:
                continue
            if self.leases.for_model(instance.model_id) is not None:
                # A lease owner's placement is forced onto the lease's cards on
                # a COPY of the record at load time (_apply_lease_profile), so
                # the persisted-override gate above cannot see it. Previewing
                # the raw record offers "[0] alone suffices" for a model leased
                # onto [2, 3], and the reload's one-shot devices would then
                # outrank the lease in _apply_lease_profile: the owner moved
                # off the cards that were reserved for it. Never a candidate.
                continue
            neighbours: set[int] = set()
            for other in loaded:
                if other.model_id != instance.model_id and other.plan is not None:
                    neighbours.update(other.plan.devices)
            overlap = set(plan.devices) & neighbours
            if not overlap and len(plan.devices) <= 1:
                continue  # alone on one card: no move could improve it, no planner call
            seen = self._rebalance_seen.get(instance.model_id)
            if seen is not None and seen[0] == layout and now - seen[1] < self.REBALANCE_RECHECK_S:
                continue  # the same world as last time; the answer has not changed
            self._rebalance_seen[instance.model_id] = (layout, now)
            candidate = self.planner.plan_load(
                record,
                **reload_settings(plan),
                loaded=loaded,
                draft=self._draft_for(record),
                adapters=[a for a, _ in self._adapters_for(record)],
                reload_of=instance.model_id,
                allow_evict=False,
                evict_busy=False,
                source="rebalance-preview",
            )
            if not isinstance(candidate, LoadPlan):
                continue
            if set(candidate.evict_model_ids) - {instance.model_id}:
                continue  # defensive: a no-evict preview must not name victims
            new_devices = set(candidate.devices)
            new_overlap = new_devices & neighbours
            if overlap and not new_overlap:
                return (
                    instance.model_id,
                    candidate,
                    f"shares GPU{sorted(overlap)} with another resident; "
                    f"{sorted(new_devices)} is free of it",
                )
            if not (new_overlap - overlap) and len(candidate.devices) < len(plan.devices):
                return (
                    instance.model_id,
                    candidate,
                    f"spread over {len(plan.devices)} cards; "
                    f"{len(candidate.devices)} suffice at the same settings",
                )
        return None

    def _layout_fingerprint(self, loaded: Sequence[InstanceInfo]) -> tuple[Any, ...]:
        """What the rebalance answer depends on: who sits where, and what is leased."""
        residents = tuple(
            sorted((i.model_id, tuple(i.plan.devices)) for i in loaded if i.plan is not None)
        )
        leases = tuple(sorted((lease.id, tuple(lease.devices)) for lease in self.leases.all()))
        return (residents, leases)

    async def _rebalance(self, model_id: str, plan: LoadPlan, reason: str) -> None:
        """Force-reload one model onto the previewed placement (D30/D36 rules).

        ``force=True`` for the reload half only; ``evict_busy=False`` keeps the
        no-interrupt rule, and the gate re-checks it after the wait, so a
        resident that picked up a request meanwhile is refused, not stopped.
        ``allow_evict=False`` carries the preview's no-evict rule into the
        real plan, and ``require_resident`` keeps the move a move: a refusal
        -- the world changed between preview and gate -- leaves the resident
        child serving exactly as a refused forced reload always has, and a
        resident that vanished meanwhile is not reloaded cold.
        """
        if self.leases.for_model(model_id) is not None:
            # Granted between the preview and now: the cards are its alone.
            log.info(
                "rebalance skipped: the model was leased cards after the preview", model_id=model_id
            )
            return
        try:
            instance = await self.load(
                model_id,
                **reload_settings(plan),
                devices=plan.devices,
                force=True,
                evict_busy=False,
                allow_evict=False,
                require_resident=True,
                source="rebalance",
                hold_traffic=False,
            )
            log.info(
                "rebalanced model",
                model_id=model_id,
                devices=list(instance.plan.devices) if instance.plan else list(plan.devices),
                reason=reason,
            )
        except asyncio.CancelledError:
            raise
        except ModelBusyError as exc:
            # A move that never happened should not burn the 30-minute
            # cooldown: the next quiet look may find the model idle again.
            self._rebalance_last.pop(model_id, None)
            log.info(
                "rebalance deferred; the model became busy while the move waited",
                model_id=model_id,
                error=str(exc),
            )
        except ModelNotResidentError as exc:
            log.info(
                "rebalance skipped; the model is no longer resident",
                model_id=model_id,
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 - the model keeps its old placement
            log.warning(
                "rebalance failed; the model keeps its placement",
                model_id=model_id,
                error=str(exc),
            )

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
                await self.load(model_id, source="autoload", hold_traffic=False)
                log.info("preloaded model", model_id=model_id)
            except Exception as exc:
                # Seed the reconciler's backoff (D41): the very next sweep
                # re-attempting a load that failed seconds ago would only
                # repeat the refusal.
                self._pin_retry[model_id] = (
                    time.time() + self.PIN_RETRY_BASE_S,
                    self.PIN_RETRY_BASE_S,
                )
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
            leases=self.leases.all(),
        )

    def parallel_observations(self, model_id: str, *, limit: int = 64) -> list[dict[str, Any]]:
        """Measured slot sweeps for one model, or ``[]``.

        Wrapped rather than called inline because four callers want it (the
        catalog, the placements, ``load_recommended`` and the observations
        route) and none of them should fail when the table is missing (an older
        data directory, a manager built with ``db=None`` in a test): a missing
        measurement means ``recommended_parallel`` falls back to the estimate
        and says so, which is a worse answer, not a broken surface.
        """
        if self.db is None:
            return []
        try:
            return list(self.db.parallel_observations(model_id, limit=limit))
        except Exception as exc:  # noqa: BLE001 - measurements are a bonus
            log.debug("parallel observations unavailable", model_id=model_id, error=str(exc))
            return []

    def placement_profiles(self, name: str) -> dict[str, Any]:
        """Optimal settings for this model on every hardware mode of this box.

        Answers "which GPUs should I give this model, and what do I get" in one
        call. Each mode is judged **with its own cards idle** -- the user's
        "assume you can fill them both" -- and what stands in the way right now
        travels beside it as ``fits_now`` / ``would_evict``.

        Three things this used to get wrong, all fixed by delegating to
        :func:`studioforge.core.placements.placement_report` (D36):

        * the modes were the literals ``(0, 1)``, ``(2, 3)`` and "all", so a box
          with other hardware was described wrongly and the single-best-card
          mode the user asks about did not exist;
        * each mode was planned against **live** free VRAM, which answers a
          different question from the one the endpoint's own name asks;
        * it asked the planner for the largest context that fits, a *second*
          recommendation rule, so this endpoint and ``/api/catalog`` could
          recommend different loads for the same model on the same hardware.
          Both now call :func:`studioforge.core.catalog.choose_row`.

        The pre-D36 keys (``mode``, ``gpus``, ``fits``, ``ctx_size``,
        ``load_args``) are kept inside each entry so an existing caller is not
        broken by the richer shape; ``fits`` now means "fits on this mode with
        its cards idle", which is what the mode was always meant to describe.
        """
        from studioforge.core import catalog as catalog_mod
        from studioforge.core import placements as placements_mod

        record = self.registry.resolve(name)
        if record is None:
            raise ModelNotFoundError(name, known=self.registry.known_ids())

        # Planning is not loading (D16/D20), and /profiles is pure planning: at
        # INFO one call would log a line per model per mode per rung, which is
        # exactly the flood the 2026-08-19 log review found when an external
        # client fetched profiles for the whole library.
        planner = Planner(self.config, self.planner.probe, log_plans=False)
        instance = self.supervisor.get(record.id)
        loaded = [i for i in self.supervisor.list() if i.model_id != record.id]
        live = planner
        if instance is not None:
            footprint = Planner.instance_footprint(instance)
            if footprint:
                live = Planner(
                    self.config,
                    catalog_mod.CreditedProbe(self.planner.probe, footprint),
                    log_plans=False,
                )

        entries = placements_mod.placement_report(
            record,
            planner=planner,
            live_planner=live,
            loaded=loaded,
            floor=catalog_mod.recommendation_floor(self.config, record),
            preference=str(getattr(self.config.planner, "preference", "quality")),
            parallel_observations=self.parallel_observations(record.id),
        )
        for entry in entries:
            optimal = entry.get("optimal")
            entry["gpus"] = list(entry["devices"])
            entry["fits"] = optimal is not None
            if optimal is not None:
                entry["ctx_size"] = optimal["ctx_per_slot"]
                entry["load_args"] = optimal["load_args"]

        usable = [e for e in entries if e.get("optimal")]
        best = max(usable, key=lambda e: int(e["optimal"]["ctx_per_slot"]), default=None)
        cheapest = next((e for e in usable if "cheapest" in e.get("ranking", [])), None)
        return {
            "model_id": record.id,
            "n_ctx_train": record.meta.n_ctx_train if record.meta else None,
            "size_gib": round(record.size_bytes / (1024**3), 2),
            "floor": catalog_mod.recommendation_floor(self.config, record),
            "modes": entries,
            # The pre-D36 name for the same list.
            "profiles": entries,
            "best_mode": usable[0]["mode"] if usable else None,
            "cheapest_mode": cheapest["mode"] if cheapest else None,
            "max_ctx": int(best["optimal"]["ctx_per_slot"]) if best else None,
            "recommended_mode": usable[0]["mode"] if usable else None,
            "quality_notes": catalog_mod.quality_notes(record),
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
            # The preview must answer for the load the Load button would run,
            # and that load runs at the model's remembered tier (D46) -- a
            # tier-blind preview says "does not fit" for a chat model whose
            # real load would displace background residents and land fine.
            priority=self._effective_priority(record.id),
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

    #: What a caller is told to wait when the server is too busy to smoke-test.
    #: One agent turn, roughly; short enough that a poll is not a stall.
    TEST_RETRY_AFTER_S = 15.0

    def busy_snapshot(self) -> dict[str, Any]:
        """What this server is in the middle of, cheaply (D36).

        ``/health`` is polled constantly by the watchdog, so this reads only
        in-memory state: no NVML, no HTTP to a child, no planner. Three
        independent kinds of busy, because they have different remedies -- an
        in-flight request clears itself, a load in flight means the VRAM answer
        is about to change, and a test in flight means another caller has taken
        the one-at-a-time smoke-test slot.
        """
        loaded = self.supervisor.list()
        current = self._current_hold()
        hold = None if current is None else {"model_id": current[0], "priority": current[1]}
        return {
            "active_requests": sum(i.active_requests for i in loaded),
            "busy_models": [
                {"model_id": i.model_id, "active_requests": i.active_requests}
                for i in loaded
                if i.active_requests > 0
            ],
            "loading": sorted(self._loading | {i.model_id for i in loaded if i.state == "loading"}),
            "testing": self._testing,
            # The chat/agent load currently holding worse-tier traffic off
            # (D46), or None -- the reason behind any 503 a client just got.
            "priority_hold": hold,
        }

    def _busy_reason(self) -> str | None:
        """Why a smoke test must not start right now, or ``None``.

        No exemption for the model being tested. An earlier draft ignored its
        own instance on the theory that "its own idle instance is not a reason
        to refuse a test of itself" -- but an *idle* instance never appears here
        (it has no in-flight requests), so the exemption could only ever fire
        for a model that was **busy**, which is precisely the case that must be
        refused. Testing a model that is mid-conversation measures the queue.
        """
        busy = self.busy_snapshot()
        if busy["busy_models"]:
            names = ", ".join(
                f"{b['model_id']} ({b['active_requests']} in flight)" for b in busy["busy_models"]
            )
            return f"{names} is serving requests"
        if busy["loading"]:
            return f"a model load is in flight ({', '.join(busy['loading'])})"
        benchmarking = getattr(self.benchmarker, "benchmarking", None)
        if benchmarking:
            return f"a benchmark of {benchmarking} is running"
        return None

    async def test_model(
        self,
        name: str,
        prompt: str | None = None,
        *,
        keep_loaded: bool = False,
    ) -> dict[str, Any]:
        """Smoke-test one model on an otherwise idle server, and leave it as found.

        This is a **health check**, not a way to pre-warm a model, and D36 made
        it behave like one. Live evidence, 2026-08-19: an external client walked
        the library testing models, and each test was an ordinary load -- at the
        planner's full target context, on whatever cards were free, evicting
        whatever was in the way, concurrently with everything else. A tool whose
        job is to answer "does this model work" must not be able to rearrange a
        server that several agents are using.

        So:

        * **One at a time.** A second concurrent ``test_model`` is refused
          rather than queued: a queued smoke test is a smoke test whose answer
          arrives after the state it described has changed.
        * **Idle only.** Any instance serving a request, any load in flight, or
          a running benchmark refuses the call with ``retry_after_s``. It also
          takes the D29 load gate for the load itself, so it cannot race one.
        * **Small.** If the model is not already loaded it is loaded at
          ``min(models.default_ctx, n_ctx_train)`` with **one** slot: a canned
          one-sentence prompt needs no 262144-token window, and sizing one costs
          VRAM the rest of the box wanted.
        * **Left as found.** A model this call loaded is unloaded again, unless
          ``keep_loaded=True``. A model that was already resident is never
          touched.

        The result says what it did (``loaded_for_test``, ``unloaded_after``,
        ``ctx_size_used``) so a caller can tell a test of a warm model from a
        test that cost a cold load.
        """
        import httpx

        record = self.registry.resolve(name)
        if record is None:
            raise ModelNotFoundError(name, known=self.registry.known_ids())
        serving = self.serving_record(record)

        if self._testing is not None:
            raise ModelBusyError(
                f"a test of '{self._testing}' is already running; test_model is "
                f"one-at-a-time so its answer describes a server that is not moving",
                details={"testing": self._testing, "retry_after_s": self.TEST_RETRY_AFTER_S},
            )

        async with self._test_gate:
            resident = self.supervisor.get(serving.id)
            was_loaded = resident is not None and resident.state == "ready"
            reason = self._busy_reason()
            if reason is not None:
                raise ModelBusyError(
                    f"the server is busy ({reason}); a smoke test must run on an idle "
                    f"server or it measures the queue instead of the model",
                    details={
                        "busy": self.busy_snapshot(),
                        "retry_after_s": self.TEST_RETRY_AFTER_S,
                    },
                )
            self._testing = record.id
            try:
                return await self._run_test(
                    record,
                    serving,
                    prompt,
                    was_loaded=was_loaded,
                    keep_loaded=keep_loaded,
                    httpx=httpx,
                )
            finally:
                self._testing = None

    async def _run_test(
        self,
        record: ModelRecord,
        serving: ModelRecord,
        prompt: str | None,
        *,
        was_loaded: bool,
        keep_loaded: bool,
        httpx: Any,
    ) -> dict[str, Any]:
        ctx_used: int | None = None
        if not was_loaded:
            ctx_used = self.smoke_test_ctx(serving)
            await self.load(
                serving.id,
                ctx_size=ctx_used,
                parallel=1,
                source="mcp:test_model",
                hold_traffic=False,
            )
        instance = self.supervisor.get(serving.id)
        if instance is None or instance.state != "ready":
            raise ModelLoadError(f"model '{record.id}' is not serving")
        # A preset-only virtual model serves from its base's instance, so the
        # URL and request accounting key off instance.model_id, not record.id.
        serving_id = instance.model_id
        base = self.supervisor.base_url(serving_id)
        if base is None:
            raise ModelLoadError(f"model '{record.id}' is not serving")
        if was_loaded and instance.plan is not None:
            ctx_used = instance.plan.ctx_per_slot or instance.plan.ctx_size

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
        # Leave the rig as found. Only ever unloads what this call loaded, and
        # only when the caller did not ask to keep it.
        unloaded = False
        if not was_loaded and not keep_loaded:
            # Housekeeping, not a choice to take the model down: a pinned
            # model found unloaded (reconcile backoff, a fresh pin) must stay
            # wanted by the reconciler after the test (D41).
            unloaded = await self.unload(serving.id, deliberate=False)

        result: dict[str, Any] = {
            "model_id": record.id,
            "latency_s": round(elapsed, 3),
            "loaded_for_test": not was_loaded,
            "unloaded_after": unloaded,
            "ctx_size_used": ctx_used,
        }
        if record.kind == "embedding":
            vectors = data.get("data", [])
            dims = len(vectors[0]["embedding"]) if vectors else 0
            result["ok"] = dims > 0
            result["embedding_dims"] = dims
            return result

        usage = data.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        text = ""
        choices = data.get("choices") or []
        if choices:
            text = (choices[0].get("message") or {}).get("content") or ""
        result.update(
            ok=bool(text),
            completion_tokens=completion_tokens,
            tokens_per_second=round(completion_tokens / elapsed, 2) if elapsed > 0 else None,
            text=text[:400],
        )
        result.update(draft_stats(data.get("timings")))
        return result

    def smoke_test_ctx(self, record: ModelRecord) -> int:
        """Context a smoke test loads at: the server floor, capped at the window.

        Deliberately not the planner's target: a canned one-sentence prompt
        proves the model generates at 4096 tokens exactly as well as at 262144,
        and the difference is tens of gigabytes of KV cache taken from whatever
        else the box is doing. An explicit per-model ``ctx_size`` still wins,
        because an explicit value always does.
        """
        pinned = record.settings.ctx_size
        if pinned:
            return int(pinned)
        floor = int(self.config.models.default_ctx)
        trained = int(getattr(record.meta, "n_ctx_train", 0) or 0) if record.meta else 0
        return min(floor, trained) if trained > 0 else floor

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
