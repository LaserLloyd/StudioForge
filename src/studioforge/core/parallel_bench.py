"""Measuring the slot knee: load once, sweep the concurrency, record the curve.

:mod:`studioforge.core.parallel` states the *rule* for turning a curve into a
recommended slot count. This module produces the curve, and it is the reason
``recommended_parallel_basis`` can ever say ``"measured"``.

What it does, in order: refuse if the server is busy or another benchmark is
running; load the model **once** at the placement being measured, with as many
slots as the placement can hold; then for each N in 1, 2, 4, 8 fire N concurrent
non-streamed completions and record per-stream tokens/second, aggregate
tokens/second and p95 latency. Afterwards it puts the rig back the way it found
it.

Three things about the method are load-bearing.

**One load for the whole sweep.** Reloading between levels would measure a cold
KV cache and a cold page cache four times over, and each load is tens of seconds
of the run. The one load asks for the *maximum* slots so that every level is
served by the same child -- a level of 8 against a 4-slot launch does not measure
batching, it measures a queue.

**The engine's own slot accounting is the control.** A client that sends 8
requests to a 1-slot server still gets 8 answers, just serialized, and the
throughput table alone cannot tell that run from a batched one. So each level
takes the delta in ``llamacpp:n_decode_total`` from ``/metrics`` and reports
``completion_tokens / decode_steps`` as ``achieved_batch``: ~N means batching
happened, ~1.0 with N > 1 means it did not, and the recommendation from a run
whose batch never rose is worth nothing. This is the point of the D17 harness
(``scripts/bench_parallel.py``) that this productises.

**Leave the rig as found.** The server must be idle for the run to start at all,
so nothing is interrupted -- but a model that was resident when the run began is
resident with its old plan when it ends, and a model that was not is unloaded.
A benchmark that quietly leaves a 8-slot child holding VRAM has changed the
thing it was measuring.
"""

from __future__ import annotations

import asyncio
import re
import statistics
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import httpx

from studioforge.core import parallel as parallel_mod
from studioforge.core import placements as placements_mod
from studioforge.core import throughput
from studioforge.errors import BadRequestError, ModelBusyError, ModelLoadError
from studioforge.logging import get_logger
from studioforge.types import InstanceInfo, KvCacheType, LoadPlan, ModelRecord

if TYPE_CHECKING:
    from studioforge.core.benchmark import Benchmarker
    from studioforge.core.manager import ModelManager

log = get_logger(__name__)

#: The sweep. Same levels the rule reads (:data:`parallel.PARALLEL_LEVELS`),
#: because a level that is measured but cannot be recommended is a minute of
#: GPU time spent on nothing.
DEFAULT_STREAMS: tuple[int, ...] = parallel_mod.PARALLEL_LEVELS

#: Roughly how many tokens of prompt each request carries. Long enough that the
#: prefill is real work (a one-line prompt makes every level look like pure
#: decode and hides the shared logical batch that ``--batch-size`` governs),
#: short enough that eight of them do not fill a small window.
DEFAULT_PROMPT_TOKENS = 512

#: Tokens generated per request. The decode phase is what the knee is about, so
#: this wants to dominate the prefill without making the 8-stream level minutes
#: long on a slow placement.
DEFAULT_MAX_TOKENS = 128

#: What a caller is told to wait when the server is too busy to measure. Same
#: figure ``test_model`` uses: a benchmark and a smoke test are refused for the
#: same reason and there is no sense in two different answers to "how long".
BUSY_RETRY_AFTER_S = 15.0

#: Per-request ceiling. An 8-stream level on a big model at a wide context is
#: legitimately slow; a hung child is not, and the difference is minutes.
REQUEST_TIMEOUT_S = 600.0

_FILLER = (
    "The gateway keeps one inference engine process per loaded model, "
    "reverse-proxies OpenAI-compatible requests to it, and refuses any load "
    "that would not fit entirely in video memory. It plans placement from a "
    "memory estimate covering weights, the key/value cache sized on the total "
    "context across slots, scratch compute buffers, an optional vision "
    "projector, low-rank adapters and a fixed per-device driver charge. "
)


def build_prompt(target_tokens: int, *, stream_index: int = 0) -> str:
    """A filler prompt of roughly ``target_tokens``, unique per stream.

    Unique because identical prompts are the one thing that makes a concurrency
    measurement lie: llama.cpp routes by prompt similarity
    (``--slot-prompt-similarity``, which the planner raises above one slot --
    D17) and would land several streams on the same slot's prefix cache, so the
    level would report the cost of a cache hit rather than of N conversations.
    """
    words_per_repeat = len(_FILLER.split())
    repeats = max(1, int(target_tokens / max(1, words_per_repeat)) + 1)
    return (
        f"Request {stream_index}. Summarise the following design note in one "
        f"paragraph, and be specific.\n\n" + _FILLER * repeats
    )


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------

_METRIC_RE = re.compile(r"^llamacpp:(\w+)\s+([\d.eE+-]+)\s*$")


def busy_reason_from(snapshot: Mapping[str, Any]) -> str | None:
    """The busy sentence for a snapshot, minus the "a benchmark is running" arm.

    ``ModelManager._busy_reason`` builds the same sentence and adds that arm on
    the end. Inside :meth:`Benchmarker.exclusive` the running benchmark is this
    call, so the extra arm would refuse every sweep at its own second check --
    which is exactly what the first live run did.
    """
    busy = list(snapshot.get("busy_models") or [])
    if busy:
        names = ", ".join(f"{b['model_id']} ({b['active_requests']} in flight)" for b in busy)
        return f"{names} is serving requests"
    loading = list(snapshot.get("loading") or [])
    if loading:
        return f"a model load is in flight ({', '.join(loading)})"
    testing = snapshot.get("testing")
    if testing:
        return f"a smoke test of {testing} is running"
    return None


def parse_metrics(text: str) -> dict[str, float]:
    """llama.cpp's Prometheus text format as ``{name: value}``.

    Only the unlabelled ``llamacpp:*`` samples, which is all this child emits.
    A metrics endpoint that is off (``--metrics`` not passed) or unreachable
    yields an empty mapping and the run continues without the control figure,
    saying so in its notes -- the throughput numbers are still the measurement.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        match = _METRIC_RE.match(line.strip())
        if match is None:
            continue
        try:
            out[match.group(1)] = float(match.group(2))
        except ValueError:  # pragma: no cover - the regex already constrains it
            continue
    return out


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class StreamResult:
    """One request of one level."""

    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None

    @property
    def tokens_per_second(self) -> float | None:
        if self.error is not None or self.latency_s <= 0 or self.completion_tokens <= 0:
            return None
        return self.completion_tokens / self.latency_s


@dataclass
class LevelResult:
    """One concurrency level: the row that goes into the DB and the table."""

    n_streams: int
    wall_s: float = 0.0
    per_stream_tps: float | None = None
    aggregate_tps: float | None = None
    p50_latency_s: float | None = None
    p95_latency_s: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: ``completion_tokens / decode steps`` across the level. ~N proves the
    #: slots really batched; ~1.0 at N > 1 means they serialized.
    achieved_batch: float | None = None
    #: llama.cpp's own ``n_busy_slots_per_decode`` gauge, reported as it was
    #: read. A cumulative average over the child's whole life, so it lags the
    #: level badly -- kept for continuity with the D17 harness' output, never
    #: used as the control.
    busy_slots_gauge: float | None = None
    requests_deferred: float | None = None
    failed: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_streams": self.n_streams,
            "wall_s": round(self.wall_s, 3),
            "per_stream_tps": _round(self.per_stream_tps),
            "aggregate_tps": _round(self.aggregate_tps),
            "p50_latency_s": _round(self.p50_latency_s, 3),
            "p95_latency_s": _round(self.p95_latency_s, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "achieved_batch": _round(self.achieved_batch),
            "busy_slots_gauge": _round(self.busy_slots_gauge),
            "requests_deferred": _round(self.requests_deferred),
            "failed": self.failed,
            "error": self.error,
        }


@dataclass
class ParallelReport:
    """The whole sweep, ready to persist and to render."""

    model_id: str
    mode: str | None
    devices: list[int]
    ctx_per_slot: int
    kv_cache_type: str
    kv_cache_type_v: str
    parallel_launched: int
    max_parallel: int
    prompt_tokens: int
    max_tokens: int
    gpu_class: str | None = None
    engine_tag: str | None = None
    run_id: str = ""
    started_at: float = 0.0
    finished_at: float | None = None
    levels: list[LevelResult] = field(default_factory=list)
    recommended_parallel: int = 1
    recommended_parallel_basis: str = "estimated"
    recommended_parallel_detail: str = ""
    loaded_for_benchmark: bool = False
    unloaded_after: bool = False
    restored: bool | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "run_id": self.run_id,
            "mode": self.mode,
            "devices": list(self.devices),
            "ctx_per_slot": self.ctx_per_slot,
            "kv_cache_type": self.kv_cache_type,
            "kv_cache_type_v": self.kv_cache_type_v,
            "parallel_launched": self.parallel_launched,
            "max_parallel": self.max_parallel,
            "prompt_tokens": self.prompt_tokens,
            "max_tokens": self.max_tokens,
            "gpu_class": self.gpu_class,
            "engine_tag": self.engine_tag,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "levels": [level.to_dict() for level in self.levels],
            "recommended_parallel": self.recommended_parallel,
            "recommended_parallel_basis": self.recommended_parallel_basis,
            "recommended_parallel_detail": self.recommended_parallel_detail,
            "loaded_for_benchmark": self.loaded_for_benchmark,
            "unloaded_after": self.unloaded_after,
            "restored": self.restored,
            "notes": list(self.notes),
        }

    def observation_rows(self) -> list[dict[str, Any]]:
        """The measured levels as ``db.record_parallel_observation`` keywords.

        A level that produced no usable rate is left out rather than written as
        a null: the rule reads these rows back and a row of nulls is not a
        measurement, it is a gap that would silently lower the knee.
        """
        rows: list[dict[str, Any]] = []
        for level in self.levels:
            if level.per_stream_tps is None or level.aggregate_tps is None:
                continue
            rows.append(
                {
                    "model_id": self.model_id,
                    "ts": time.time(),
                    "run_id": self.run_id,
                    "devices": ",".join(str(d) for d in sorted(self.devices)),
                    "gpu_class": self.gpu_class,
                    "ctx_per_slot": self.ctx_per_slot,
                    "kv_cache_type": self.kv_cache_type,
                    "kv_cache_type_v": self.kv_cache_type_v,
                    "n_streams": level.n_streams,
                    "per_stream_tps": level.per_stream_tps,
                    "aggregate_tps": level.aggregate_tps,
                    "p95_latency_s": level.p95_latency_s,
                    "prompt_tokens": level.prompt_tokens,
                    "completion_tokens": level.completion_tokens,
                    "n_busy_slots": level.achieved_batch,
                    "engine_tag": self.engine_tag,
                }
            )
        return rows


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(float(value), digits)


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


class ParallelBenchmarker:
    """Runs one concurrency sweep at a time, sharing the benchmark lock."""

    def __init__(
        self,
        manager: ModelManager,
        benchmarker: Benchmarker,
        *,
        db: Any = None,
        probe: Any = None,
        engine_manager: Any = None,
    ) -> None:
        self.manager = manager
        self.benchmarker = benchmarker
        self.db = db if db is not None else getattr(manager, "db", None)
        self._probe = probe if probe is not None else manager.planner.probe
        # Only ever asked for the ACTIVE engine tag. The supervisor deliberately
        # knows nothing about the engine manager, so `InstanceInfo.engine_tag`
        # is None unless the record pins one -- which is almost never, and which
        # is why the first live run wrote every observation with a null tag.
        self._engine_manager = engine_manager

    # -- placement --------------------------------------------------------

    def resolve_devices(self, mode: str | None) -> list[int] | None:
        """CUDA indices for a hardware-mode key, or ``None`` for "the planner's".

        The keys are the catalog's own (:func:`placements.hardware_modes`), so
        "measure the 3090s" is the same word here as everywhere else. An unknown
        key is a 400 naming the ones this box has rather than a silent fallback
        to whatever the planner would have chosen -- the caller asked about
        specific hardware and would otherwise be handed numbers for other cards.
        """
        if mode is None:
            return None
        modes = placements_mod.hardware_modes(
            self._probe.list_gpus(), excluded=self.manager.config.planner.excluded_devices
        )
        found = next((m for m in modes if m.key == mode), None)
        if found is None:
            raise BadRequestError(
                f"unknown hardware mode {mode!r}; this box has: "
                + (", ".join(m.key for m in modes) or "none"),
                param="mode",
            )
        return list(found.devices)

    def refuse_if_busy(self) -> None:
        """The up-front half of the busy rule, for a route that answers before it runs."""
        self._refuse_if_busy()

    def _refuse_if_busy(self, *, ignore_benchmark: bool = False) -> None:
        """Refuse while anything else is using the rig (the D36 rule).

        ``ignore_benchmark`` drops the "a benchmark is running" arm, and exists
        because of a bug the first live run found: the re-check *inside*
        :meth:`Benchmarker.exclusive` saw the marker that context manager had
        just set for this very run and refused itself, so no sweep could ever
        get past its own second check. Inside the lock, "a benchmark is running"
        is a statement about us; the arms that still matter are a model serving,
        a load in flight and a smoke test, and those come straight off
        ``busy_snapshot``.
        """
        reason = (
            busy_reason_from(self.manager.busy_snapshot())
            if ignore_benchmark
            else self.manager._busy_reason()
        )
        if reason is None:
            return
        raise ModelBusyError(
            f"the server is busy ({reason}); a parallel benchmark measures how "
            f"many slots are worth running, which is meaningless while somebody "
            f"else's requests are in the same slots",
            details={
                "busy": self.manager.busy_snapshot(),
                "retry_after_s": BUSY_RETRY_AFTER_S,
            },
        )

    # -- the run ----------------------------------------------------------

    async def run(
        self,
        record: ModelRecord,
        *,
        mode: str | None = None,
        devices: Sequence[int] | None = None,
        ctx_size: int | None = None,
        kv_cache_type: str | None = None,
        kv_cache_type_v: str | None = None,
        streams: Sequence[int] = DEFAULT_STREAMS,
        prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        cancel_event: asyncio.Event | None = None,
        on_progress: Any = None,
    ) -> ParallelReport:
        """Sweep the concurrency for one model on one placement.

        ``mode`` names a hardware mode (``dual_5090``...); ``devices`` names the
        cards directly; neither leaves the placement to the planner. ``ctx_size``
        defaults to whatever the placement's optimal row would load at, so the
        measurement describes the load the catalog actually recommends.

        Refuses with :class:`~studioforge.errors.ModelBusyError` while anything
        is serving, loading, testing or benchmarking -- the WP18/D36 rule, for
        the same reason it applies to ``test_model`` and more strongly: the
        numbers here *are* the contention.
        """
        from studioforge.core.manager import validate_load_args

        # The same validation an ordinary load gets, and before anything is
        # planned or locked: a bad kv_cache_type here would otherwise surface as
        # a planner refusal several frames deeper that reads like a VRAM problem.
        validate_load_args(
            ctx_size=ctx_size,
            parallel=None,
            kv_cache_type=kv_cache_type,
            kv_cache_type_v=kv_cache_type_v,
            devices=list(devices) if devices is not None else None,
            known_devices=[g.index for g in self._probe.list_gpus()],
        )
        self._refuse_if_busy()
        wanted = list(devices) if devices is not None else self.resolve_devices(mode)
        levels = sorted({int(n) for n in streams if int(n) >= 1})
        if not levels:
            raise BadRequestError("'streams' must contain at least one level", param="streams")

        async with self.benchmarker.exclusive(record.id):
            # Re-checked inside the lock: the gap between the first check and
            # taking the slot is exactly where a JIT request lands. The
            # benchmark arm is skipped because inside the lock that arm is us.
            self._refuse_if_busy(ignore_benchmark=True)
            return await self._run_locked(
                record,
                mode=mode,
                devices=wanted,
                ctx_size=ctx_size,
                kv_cache_type=kv_cache_type,
                kv_cache_type_v=kv_cache_type_v,
                levels=levels,
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                cancel_event=cancel_event,
                on_progress=on_progress,
            )

    async def _run_locked(
        self,
        record: ModelRecord,
        *,
        mode: str | None,
        devices: Sequence[int] | None,
        ctx_size: int | None,
        kv_cache_type: str | None,
        kv_cache_type_v: str | None,
        levels: Sequence[int],
        prompt_tokens: int,
        max_tokens: int,
        cancel_event: asyncio.Event | None,
        on_progress: Any,
    ) -> ParallelReport:
        resident = self.manager.supervisor.get(record.id)
        was_loaded = resident is not None and resident.state == "ready"
        previous = _plan_args(resident) if was_loaded and resident is not None else None

        target = self._target_plan(
            record,
            devices=devices,
            ctx_size=ctx_size,
            kv_cache_type=kv_cache_type,
            kv_cache_type_v=kv_cache_type_v,
            levels=levels,
        )
        report = ParallelReport(
            model_id=record.id,
            mode=mode,
            devices=list(target["devices"]),
            ctx_per_slot=int(target["ctx_size"]),
            kv_cache_type=str(target["kv_cache_type"]),
            kv_cache_type_v=str(target["kv_cache_type_v"]),
            parallel_launched=int(target["parallel"]),
            max_parallel=int(target["max_parallel"]),
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            gpu_class=self._gpu_class(),
            engine_tag=record.settings.engine_tag,
            run_id=uuid.uuid4().hex[:16],
            started_at=time.time(),
        )
        wanted_levels = [n for n in levels if n <= report.parallel_launched]
        dropped = [n for n in levels if n > report.parallel_launched]
        if dropped:
            report.notes.append(
                f"levels {', '.join(str(n) for n in dropped)} were not measured: this "
                f"placement holds {report.parallel_launched} slots, and N requests "
                f"against fewer than N slots measure a queue rather than batching"
            )

        reuse = was_loaded and _plan_serves(resident, report)
        if reuse:
            report.notes.append(
                "the resident instance already had the placement and slot count "
                "this run needs, so nothing was reloaded"
            )
        else:
            await self.manager.load(
                record.id,
                ctx_size=report.ctx_per_slot,
                kv_cache_type=report.kv_cache_type,
                kv_cache_type_v=report.kv_cache_type_v,
                parallel=report.parallel_launched,
                devices=report.devices,
                force=True,
                # force here is the reload half only: the busy check above is
                # what guards a serving model, and the load must not be able
                # to evict one that started in the gap (D36).
                evict_busy=False,
                source="benchmark:parallel",
            )
            report.loaded_for_benchmark = True

        try:
            await self._sweep(
                record,
                report,
                levels=wanted_levels,
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                cancel_event=cancel_event,
                on_progress=on_progress,
            )
        finally:
            await self._leave_as_found(record, report, was_loaded=was_loaded, previous=previous)

        self._finish(record, report)
        return report

    # -- placement arithmetic ---------------------------------------------

    def _target_plan(
        self,
        record: ModelRecord,
        *,
        devices: Sequence[int] | None,
        ctx_size: int | None,
        kv_cache_type: str | None,
        kv_cache_type_v: str | None,
        levels: Sequence[int],
    ) -> dict[str, Any]:
        """What to launch: the placement's own optimal, at as many slots as fit.

        Deliberately the catalog's answer rather than a second opinion. A sweep
        that measured 8192/q8_0 while the catalog recommends 65536/f16 would
        produce a number that is then applied to a load it never described --
        and the KV bytes per slot, which is what sets the knee, is precisely
        what differs between them.
        """
        from studioforge.core import catalog as catalog_mod

        pinned = record if devices is None else placements_mod.forced_onto(record, devices)
        plan = self.manager.planner.plan_load(
            pinned,
            ctx_size=ctx_size,
            # Already checked against KV_CACHE_TYPES by validate_load_args in
            # run(); the planner's parameter is a Literal and this is the cast
            # that says so once rather than narrowing the public signature.
            kv_cache_type=cast("KvCacheType | None", kv_cache_type),
            kv_cache_type_v=cast("KvCacheType | None", kv_cache_type_v),
            parallel=1,
            loaded=[i for i in self.manager.supervisor.list() if i.model_id != record.id],
            allow_evict=False,
            reload_of=record.id if self.manager.supervisor.get(record.id) else None,
        )
        if not isinstance(plan, LoadPlan):
            raise self.manager._vram_error(plan)
        slots, _bound, _vram = catalog_mod.slots_for_plan(self.manager.planner, pinned, plan)
        launch = max(1, min(int(slots), max(int(n) for n in levels)))
        return {
            "devices": list(plan.devices),
            "ctx_size": int(plan.ctx_size),
            "kv_cache_type": plan.kv_cache_type,
            "kv_cache_type_v": plan.kv_cache_type_v,
            "parallel": launch,
            "max_parallel": int(slots),
        }

    def _active_engine_tag(self) -> str | None:
        """The engine this server is running, or ``None`` if it cannot be asked."""
        if self._engine_manager is None:
            return None
        try:
            active = self._engine_manager.active()
        except Exception:  # noqa: BLE001 - a label is not worth failing a run for
            return None
        return getattr(active, "tag", None)

    def _gpu_class(self) -> str | None:
        try:
            return throughput.gpu_class(self._probe.list_gpus())
        except Exception:  # noqa: BLE001 - a label is not worth failing a run for
            return None

    # -- the sweep --------------------------------------------------------

    async def _sweep(
        self,
        record: ModelRecord,
        report: ParallelReport,
        *,
        levels: Sequence[int],
        prompt_tokens: int,
        max_tokens: int,
        cancel_event: asyncio.Event | None,
        on_progress: Any,
    ) -> None:
        instance = self.manager.supervisor.get(record.id)
        if instance is None or instance.state != "ready":
            raise ModelLoadError(f"model '{record.id}' is not serving after load")
        base = self.manager.supervisor.base_url(instance.model_id)
        if base is None:
            raise ModelLoadError(f"model '{record.id}' is not serving after load")
        # The engine that ACTUALLY served the sweep, not the one the record
        # pins -- which is normally nothing, so the first live run recorded
        # every row with a null tag. A llama.cpp build can change the batching
        # behaviour being measured, so an observation that cannot name its
        # engine cannot be retired when that engine is replaced (005's comment
        # says exactly this, and the code was not keeping the promise).
        report.engine_tag = (
            instance.engine_tag or record.settings.engine_tag or self._active_engine_tag()
        )

        total = len(levels) or 1
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(REQUEST_TIMEOUT_S, connect=10.0)
        ) as client:
            # One warm-up so the first level is not paying for the child's very
            # first graph build and page faults, which would flatter every
            # subsequent level by comparison.
            await self._one_request(client, base, record, prompt_tokens, max_tokens, stream_index=0)
            for position, n_streams in enumerate(levels):
                if cancel_event is not None and cancel_event.is_set():
                    report.notes.append(
                        f"canceled after {position} of {len(levels)} concurrency levels"
                    )
                    break
                _emit(on_progress, n_streams, "measuring", position, total)
                report.levels.append(
                    await self._level(
                        client,
                        base,
                        record,
                        instance,
                        n_streams=n_streams,
                        prompt_tokens=prompt_tokens,
                        max_tokens=max_tokens,
                    )
                )
                _emit(on_progress, n_streams, "done", position, total)

    async def _level(
        self,
        client: httpx.AsyncClient,
        base: str,
        record: ModelRecord,
        instance: InstanceInfo,
        *,
        n_streams: int,
        prompt_tokens: int,
        max_tokens: int,
    ) -> LevelResult:
        result = LevelResult(n_streams=n_streams)
        before = await self._metrics(base)

        self.manager.supervisor.mark_request_start(instance.model_id)
        started = time.perf_counter()
        try:
            results = await asyncio.gather(
                *[
                    self._one_request(
                        client, base, record, prompt_tokens, max_tokens, stream_index=index
                    )
                    for index in range(n_streams)
                ]
            )
        finally:
            self.manager.supervisor.mark_request_end(instance.model_id)
        result.wall_s = time.perf_counter() - started
        after = await self._metrics(base)

        ok = [r for r in results if r.error is None and r.completion_tokens > 0]
        result.failed = len(results) - len(ok)
        if not ok:
            result.error = next(
                (r.error for r in results if r.error), "every request returned no tokens"
            )
            return result

        rates = [r.tokens_per_second for r in ok]
        latencies = sorted(r.latency_s for r in ok)
        result.completion_tokens = sum(r.completion_tokens for r in ok)
        result.prompt_tokens = sum(r.prompt_tokens for r in ok)
        # Per stream is the MEDIAN of the streams' own rates, not the aggregate
        # divided by N: one straggler that finished after the others would drag
        # a mean down and would be read as a knee that is really a scheduling
        # artefact. The p95 below is where a straggler belongs.
        result.per_stream_tps = statistics.median([r for r in rates if r is not None])
        result.aggregate_tps = (
            result.completion_tokens / result.wall_s if result.wall_s > 0 else None
        )
        result.p50_latency_s = statistics.median(latencies)
        result.p95_latency_s = _percentile(latencies, 0.95)

        decodes = after.get("n_decode_total", 0.0) - before.get("n_decode_total", 0.0)
        if decodes > 0:
            result.achieved_batch = result.completion_tokens / decodes
        result.busy_slots_gauge = after.get("n_busy_slots_per_decode")
        if "requests_deferred" in after:
            result.requests_deferred = after.get("requests_deferred", 0.0) - before.get(
                "requests_deferred", 0.0
            )
        return result

    async def _one_request(
        self,
        client: httpx.AsyncClient,
        base: str,
        record: ModelRecord,
        prompt_tokens: int,
        max_tokens: int,
        *,
        stream_index: int,
    ) -> StreamResult:
        """One non-streamed completion. Never raises; a failure is a datum."""
        payload = {
            "model": record.id,
            "messages": [
                {"role": "user", "content": build_prompt(prompt_tokens, stream_index=stream_index)}
            ],
            "max_tokens": max_tokens,
            # Greedy and unseeded-identical so two runs are comparable, and the
            # prompt cache off so every request really ingests its prompt --
            # otherwise level 8 would measure seven cache hits.
            "temperature": 0,
            "top_k": 1,
            "seed": 0,
            "cache_prompt": False,
            "stream": False,
        }
        started = time.perf_counter()
        try:
            response = await client.post(f"{base}/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001 - one bad stream is a number, not a crash
            return StreamResult(error=str(exc))
        usage = data.get("usage") or {}
        return StreamResult(
            latency_s=time.perf_counter() - started,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )

    async def _metrics(self, base: str) -> dict[str, float]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{base}/metrics")
                response.raise_for_status()
                return parse_metrics(response.text)
        except Exception as exc:  # noqa: BLE001 - the control figure is a bonus
            log.debug("parallel benchmark metrics unavailable", error=str(exc))
            return {}

    # -- teardown ---------------------------------------------------------

    async def _leave_as_found(
        self,
        record: ModelRecord,
        report: ParallelReport,
        *,
        was_loaded: bool,
        previous: dict[str, Any] | None,
    ) -> None:
        """Unload what this run loaded, or put back what it displaced."""
        if not report.loaded_for_benchmark:
            return
        if not was_loaded:
            report.unloaded_after = await self.manager.unload(record.id)
            return
        if previous is None:  # pragma: no cover - a ready instance always has a plan
            report.restored = False
            return
        try:
            await self.manager.load(
                record.id,
                **previous,
                force=True,
                evict_busy=False,
                source="benchmark:parallel-restore",
            )
            report.restored = True
        except Exception as exc:  # noqa: BLE001 - a failed restore must be reported, not raised
            report.restored = False
            report.notes.append(
                f"the instance that was resident before this run could not be "
                f"restored ({exc}); it is loaded at the benchmark's settings"
            )
            log.warning("parallel benchmark could not restore", model_id=record.id, error=str(exc))

    def _finish(self, record: ModelRecord, report: ParallelReport) -> None:
        """Record the rows and apply the rule to them."""
        rows = report.observation_rows()
        if self.db is not None and rows:
            for row in rows:
                try:
                    self.db.record_parallel_observation(**row)
                except Exception as exc:  # noqa: BLE001 - a run is worth reporting either way
                    log.warning(
                        "could not record a parallel observation",
                        model_id=record.id,
                        error=str(exc),
                    )
                    break

        recommended = parallel_mod.recommended_parallel(
            record.meta,
            weights_bytes=int(record.size_bytes),
            ctx_per_slot=report.ctx_per_slot,
            kv_cache_type=report.kv_cache_type,
            kv_cache_type_v=report.kv_cache_type_v,
            max_parallel=report.max_parallel,
            observations=rows,
        )
        report.recommended_parallel = int(recommended["value"])
        report.recommended_parallel_basis = str(recommended["basis"])
        report.recommended_parallel_detail = str(recommended["detail"])
        report.finished_at = time.time()

        batched = [level for level in report.levels if level.n_streams > 1]
        if batched and all((level.achieved_batch or 0.0) < 1.5 for level in batched):
            # The one result that invalidates the rest: the numbers describe a
            # queue, not concurrency, and a recommendation from them would say
            # "one slot" about a placement that was never given more than one.
            report.notes.append(
                "the engine's decode counter never showed more than ~1 sequence per "
                "step, so these levels serialized rather than batched -- check that "
                "the child really launched with --parallel > 1 (/props total_slots) "
                "before trusting the recommendation"
            )
        log.info(
            "parallel benchmark complete",
            model_id=record.id,
            devices=report.devices,
            ctx_per_slot=report.ctx_per_slot,
            levels=len(report.levels),
            recommended_parallel=report.recommended_parallel,
        )


def for_state(state: Any) -> ParallelBenchmarker:
    """The process-wide runner, created on first use and cached on ``state``.

    One factory because two surfaces reach for it -- the HTTP route and the MCP
    tool -- and the object owns (through the placement benchmarker) the "one run
    at a time" lock. Two instances would be two locks, which is no lock.
    """
    from studioforge.core.benchmark import Benchmarker

    runner = getattr(state, "parallel_benchmarker", None)
    if runner is not None:
        return runner
    benchmarker = getattr(state, "benchmarker", None)
    if benchmarker is None:
        benchmarker = Benchmarker(state.manager, probe=state.probe)
        state.benchmarker = benchmarker
    runner = ParallelBenchmarker(
        state.manager,
        benchmarker,
        db=getattr(state, "db", None),
        probe=state.probe,
        engine_manager=getattr(state, "engine_manager", None),
    )
    state.parallel_benchmarker = runner
    return runner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plan_args(instance: InstanceInfo) -> dict[str, Any] | None:
    """The ``manager.load`` keywords that would reproduce this instance."""
    plan = instance.plan
    if plan is None:
        return None
    return {
        "ctx_size": int(plan.ctx_per_slot or plan.ctx_size),
        "parallel": int(plan.parallel),
        "kv_cache_type": plan.kv_cache_type,
        "kv_cache_type_v": plan.kv_cache_type_v,
        "devices": list(plan.devices),
    }


def _plan_serves(instance: InstanceInfo | None, report: ParallelReport) -> bool:
    """Whether the resident child can host this sweep without a reload.

    Same cards, same context per slot, same KV types, and at least as many slots
    as the top level asks for. Anything less and the top levels would queue
    behind slots that do not exist, which is the one thing this measurement must
    not silently do.
    """
    plan = instance.plan if instance is not None else None
    if plan is None:
        return False
    return (
        sorted(plan.devices) == sorted(report.devices)
        and int(plan.ctx_per_slot or plan.ctx_size) == report.ctx_per_slot
        and plan.kv_cache_type == report.kv_cache_type
        and plan.kv_cache_type_v == report.kv_cache_type_v
        and int(plan.parallel) >= report.parallel_launched
    )


def _percentile(sorted_values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile of an already-sorted list.

    Nearest-rank rather than interpolated: with four samples an interpolated p95
    is a weighted average of the two slowest, which is a number no request
    experienced. The point of p95 here is "how bad did the unlucky stream get".
    """
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, max(0, int(round(fraction * len(sorted_values))) - 1))
    return float(sorted_values[index])


def _emit(on_progress: Any, level: int | None, phase: str, position: int, total: int) -> None:
    """Report progress; a broken callback must never fail a benchmark."""
    if on_progress is None:
        return
    fraction = (position + (1.0 if phase == "done" else 0.3)) / max(1, total)
    try:
        on_progress(str(level) if level is not None else None, phase, min(1.0, max(0.0, fraction)))
    except Exception:  # noqa: BLE001 - progress is cosmetic
        log.debug("parallel benchmark progress callback failed")


def rows_from(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The measured levels of a persisted report, for re-applying the rule.

    Used by the GUI and by tests, which hold the JSON rather than the dataclass.
    """
    return [
        {
            "n_streams": level.get("n_streams"),
            "per_stream_tps": level.get("per_stream_tps"),
            "aggregate_tps": level.get("aggregate_tps"),
            "ts": float(level.get("n_streams") or 0),
            "run_id": report.get("run_id"),
        }
        for level in report.get("levels") or []
    ]
