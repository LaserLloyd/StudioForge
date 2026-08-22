"""Model benchmarking: load time and throughput per GPU placement mode.

"Which cards should this model run on?" is a question the planner answers for
*fit*, not for *speed*. A 14 GiB model fits on a single RTX 3090 and on a
single RTX 5090 and on two of either; only measurement says which is fastest,
and the answer differs per model (a small model is bandwidth-bound and hates
being split; a big one has no choice). This module runs the same deterministic
completion under each placement the hardware actually offers and reports load
time, time-to-first-token, prompt throughput and generation throughput.

Three decisions are worth explaining, because each has a tempting wrong answer.

**Modes are derived from the live GPU list, never hardcoded.** GPUs are grouped
by ``(name, compute capability)`` in the planner's own fastest-first order, and
each group contributes a 1-device mode plus one mode per additional device in
the group (so a 4x3090 box gets x1/x2/x3/x4, and a single-GPU laptop gets
exactly one mode). An ``"all"`` mode is added only when it is not already the
same device set as a group mode.

**Applicability comes from the planner, not from guesswork.** Whether a model
can run on "1x RTX 3090" is a VRAM question with a lot of terms in it -- KV
cache sizing, the mmproj, adapters, a draft model, per-device CUDA context
overhead -- and :mod:`studioforge.core.planner` is the one place that knows all
of them. Re-deriving a cheaper approximation here would eventually disagree
with the planner, and the mode that "should" work would fail at load. So each
mode is planned with a temporary ``device_override`` and a rejection is
reported verbatim: the user sees *why* their 32 GiB model cannot do "1x RTX
3090" instead of the mode silently vanishing.

**Throughput numbers come from llama-server's own ``timings``, not our stopwatch.**
The engine reports ``prompt_n``/``prompt_per_second`` and
``predicted_n``/``predicted_per_second`` measured inside the inference loop, so
they exclude HTTP framing, SSE parsing and our proxy overhead -- they are the
same numbers ``llama-bench`` prints and therefore the ones a user can compare
against anything else. Wall clock is only a fallback, and when it is used the
report says so in ``notes``.

**Runs are serialized.** Two benchmarks at once would compete for the very
resource being measured; the numbers would be meaningless and the loads might
not even fit. A second :meth:`Benchmarker.run` therefore fails fast rather than
queueing behind a multi-minute job.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any

import httpx

from studioforge.core.gpu import fastest_gpu_order
from studioforge.core.planner import BUSY_RETRY_AFTER_S
from studioforge.core.supervisor import SPLIT_MODE_TENSOR, tensor_split_model_blockers
from studioforge.errors import BadRequestError, ModelBusyError, ModelLoadError
from studioforge.logging import get_logger
from studioforge.types import GpuInfo, LoadRejected, ModelRecord, ModelSettings

if TYPE_CHECKING:
    from studioforge.core.gpu import GpuProbe
    from studioforge.core.manager import ModelManager

log = get_logger(__name__)

DEFAULT_CTX_SIZE = 4096
DEFAULT_MAX_TOKENS = 128

#: Fixed prompt so two runs are comparable. Long enough (a few hundred tokens)
#: that prompt processing is actually measurable -- a one-line prompt gives a
#: ``prompt_per_second`` figure dominated by fixed per-request cost.
DEFAULT_PROMPT = (
    "You are reviewing the design of a small self-hosted inference gateway. "
    "The gateway keeps one inference engine process per loaded model, "
    "reverse-proxies OpenAI-compatible requests to it, and refuses any load "
    "that would not fit entirely in video memory. It plans placement from a "
    "memory estimate that accounts for model weights, the key/value cache "
    "sized on the total context across slots, scratch compute buffers, an "
    "optional vision projector, low-rank adapters, an optional draft model "
    "for speculative decoding, and a fixed per-device driver context charge. "
    "Loads record predicted-versus-actual memory so the overhead factor can "
    "be tuned from real observations rather than guesswork. Idle models are "
    "unloaded after a configurable time to live, pinned models are never "
    "evicted, and a load that cannot be satisfied returns a structured error "
    "carrying the numbers and a concrete suggestion such as a smaller "
    "context, a quantized cache, or a different placement. Explain, in a "
    "short paragraph, the trade-off between running one model across several "
    "devices and keeping it on a single device, and say which situations "
    "make each choice the right one. Be specific about memory bandwidth, "
    "cross-device transfer, and the fixed cost of an extra device context."
)

#: Vendor/brand words dropped when turning a device name into a mode label.
#: "NVIDIA GeForce RTX 5090" reads better -- and slugs better -- as "RTX 5090".
_NOISE_WORDS = frozenset({"nvidia", "geforce", "corporation", "corp", "inc", "ltd", "co"})

#: How far through a mode each phase is, for the overall progress fraction.
_PHASE_WEIGHT: dict[str, float] = {
    "planning": 0.0,
    "loading": 0.15,
    "generating": 0.6,
    "done": 1.0,
}

ProgressFn = Callable[[str | None, str, float], None]


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkMode:
    """One launch configuration to benchmark: a placement, and how it is run.

    ``key`` is a stable, slug-safe identifier: it travels in URLs and is stored
    inside persisted reports, so it must not change shape between releases for
    the same hardware. The extra dimensions append a suffix rather than
    reshaping the key, so ``rtx-5090-x2`` still means what it always meant.
    """

    key: str
    label: str
    devices: list[int]
    gpu_name: str | None
    #: ``--split-mode`` for this mode. Only ever something other than ``layer``
    #: on a multi-device mode, and only when the model is eligible: tensor mode
    #: is experimental upstream and measured slower than layer on this rig, so
    #: it is offered as something to *measure*, never as a default.
    split_mode: str = "layer"
    #: ``-ub/--ubatch-size`` for this mode, or ``None`` for the engine's 512.
    #: Trades compute-buffer VRAM for prompt-processing speed, so it is a
    #: prefill dimension and is off unless the caller asks for it.
    ubatch: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "devices": list(self.devices),
            "gpu_name": self.gpu_name,
            "split_mode": self.split_mode,
            "ubatch": self.ubatch,
        }


def _display_name(name: str) -> str:
    """ "NVIDIA GeForce RTX 5090" -> "RTX 5090"; unknown names pass through."""
    tokens = [token for token in re.split(r"\s+", name.strip()) if token]
    kept = [token for token in tokens if token.lower().strip(".,") not in _NOISE_WORDS]
    return " ".join(kept) or name.strip() or "GPU"


def _slug(text: str) -> str:
    """Lowercase, hyphen-separated, URL- and filename-safe."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "gpu"


def _expand(
    mode: BenchmarkMode, split_modes: Sequence[str], ubatch_sizes: Sequence[int]
) -> list[BenchmarkMode]:
    """One placement -> its split-mode and ubatch variants.

    Ordered so the plain mode always comes first and keeps its original key:
    the variants are additive, and a report from before they existed still
    lines up row for row with one from after.
    """
    variants: list[BenchmarkMode] = []
    modes = list(split_modes) or ["layer"]
    if len(mode.devices) < 2:
        # Split mode is meaningless on one device (the supervisor emits
        # ``--split-mode none``), so a "tensor" variant there would benchmark
        # the identical launch twice.
        modes = ["layer"]
    for split in modes:
        base = (
            mode
            if split == "layer"
            else replace(
                mode,
                key=f"{mode.key}-{split}",
                label=f"{mode.label}, {split} split",
                split_mode=split,
            )
        )
        variants.append(base)
        for ubatch in ubatch_sizes:
            variants.append(
                replace(
                    base,
                    key=f"{base.key}-ub{ubatch}",
                    label=f"{base.label}, ubatch {ubatch}",
                    ubatch=ubatch,
                )
            )
    return variants


def available_modes(
    gpus: Sequence[GpuInfo],
    *,
    split_modes: Sequence[str] = ("layer",),
    ubatch_sizes: Sequence[int] = (),
) -> list[BenchmarkMode]:
    """Configurations worth benchmarking on this machine, fastest-first.

    Grouping is by ``(name, compute capability)`` so two identical cards form
    one family; ordering within and between groups follows
    :func:`studioforge.core.gpu.fastest_gpu_order`, which is the same ranking
    the planner uses when it picks "the best single GPU". Each group yields
    ``x1 .. xN`` modes; a final ``"all"`` mode appears only when there is more
    than one GPU and it is not already identical to a group mode (on a box with
    four identical cards, "all" *is* ``x4``, and offering both would benchmark
    the same placement twice).

    ``split_modes`` and ``ubatch_sizes`` add dimensions on top of the placement.
    Both default to "just the placement", so every existing caller gets exactly
    the list it always got: a benchmark suite that silently doubled in length
    would turn a two-minute job into a four-minute one for people who never
    asked about tensor parallelism.
    """
    if not gpus:
        return []

    by_index = {gpu.index: gpu for gpu in gpus}
    order = [index for index in fastest_gpu_order(gpus) if index in by_index]

    # Groups in first-appearance order, which is fastest-family-first.
    groups: list[tuple[str, list[int]]] = []
    buckets: dict[tuple[str, tuple[int, int] | None], list[int]] = {}
    for index in order:
        gpu = by_index[index]
        key = (gpu.name, gpu.compute_capability)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = []
            buckets[key] = bucket
            groups.append((gpu.name, bucket))
        bucket.append(index)

    modes: list[BenchmarkMode] = []
    seen_devices: set[tuple[int, ...]] = set()
    seen_prefixes: set[str] = set()
    for name, indices in groups:
        display = _display_name(name)
        prefix = _slug(display)
        # Two families can share a name but differ in compute capability
        # (a rebadged card, a mixed driver view). Keys must stay unique.
        candidate, counter = prefix, 2
        while candidate in seen_prefixes:
            candidate = f"{prefix}-{counter}"
            counter += 1
        seen_prefixes.add(candidate)

        for width in range(1, len(indices) + 1):
            devices = indices[:width]
            fingerprint = tuple(devices)
            if fingerprint in seen_devices:
                continue
            seen_devices.add(fingerprint)
            modes.extend(
                _expand(
                    BenchmarkMode(
                        key=f"{candidate}-x{width}",
                        label=f"{width}x {display}",
                        devices=list(devices),
                        gpu_name=name,
                    ),
                    split_modes,
                    ubatch_sizes,
                )
            )

    all_devices = tuple(order)
    if len(order) > 1 and all_devices not in seen_devices:
        seen_devices.add(all_devices)
        modes.extend(
            _expand(
                BenchmarkMode(
                    key="all",
                    label=f"All {len(order)} GPUs",
                    devices=list(all_devices),
                    gpu_name=None,
                ),
                split_modes,
                ubatch_sizes,
            )
        )
    return modes


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """One mode's measurement, or the reason there isn't one."""

    mode: str
    label: str
    devices: list[int]
    #: How the mode was launched, so a persisted report stays interpretable
    #: after the mode list changes shape. ``layer`` and ``None`` are the
    #: defaults every pre-WP20 report implicitly used.
    split_mode: str = "layer"
    ubatch: int | None = None
    applicable: bool = True
    skipped_reason: str | None = None
    load_time_s: float | None = None
    ttft_s: float | None = None
    prompt_tokens: int | None = None
    prompt_tps: float | None = None
    generated_tokens: int | None = None
    generation_tps: float | None = None
    vram_used_bytes: int | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.applicable and self.error is None


@dataclass
class BenchmarkReport:
    """Every mode's result plus the winners, ready to persist as JSON."""

    model_id: str
    started_at: float
    finished_at: float | None
    ctx_size: int
    max_tokens: int
    prompt_chars: int
    results: list[BenchmarkResult]
    best_generation_mode: str | None = None
    best_prompt_mode: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def recompute_best(self) -> None:
        """Pick the fastest successful mode for each metric.

        Ties keep the first (fastest-first) mode, which prefers the simpler
        placement when two are indistinguishable.
        """
        self.best_generation_mode = _best(self.results, "generation_tps")
        self.best_prompt_mode = _best(self.results, "prompt_tps")


def _best(results: Sequence[BenchmarkResult], attribute: str) -> str | None:
    best_key: str | None = None
    best_value = float("-inf")
    for result in results:
        if not result.succeeded:
            continue
        value = getattr(result, attribute)
        if value is None:
            continue
        if float(value) > best_value:
            best_value = float(value)
            best_key = result.mode
    return best_key


# ---------------------------------------------------------------------------
# Benchmarker
# ---------------------------------------------------------------------------


#: Strong refs for unloads that outlive a cancelled benchmark.
_DETACHED_UNLOADS: set[asyncio.Task[Any]] = set()


class Benchmarker:
    """Runs the benchmark suite for one model, one mode at a time."""

    def __init__(self, manager: ModelManager, *, probe: GpuProbe | None = None) -> None:
        self.manager = manager
        self._probe: GpuProbe = probe if probe is not None else manager.planner.probe
        # Serializes the whole suite. See the module docstring: concurrent runs
        # would compete for the VRAM they are measuring.
        self._lock = asyncio.Lock()
        self._benchmarking: str | None = None
        # Back-reference so the manager can see a run in progress. A benchmark
        # rewrites a record's settings and loads the model once per mode, which
        # is exactly the state in which a smoke test must not start (D36) --
        # and the manager cannot reach `app.state`, where this object lives.
        manager.benchmarker = self

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    @property
    def benchmarking(self) -> str | None:
        """Model id currently under benchmark, if any.

        A run temporarily rewrites the registry record's settings to steer
        placement per mode, and restores them at the end. A settings save
        landing inside that window is written to SQLite and then silently
        overwritten in memory by the restore, so the user's change vanishes
        until a restart. Callers use this to refuse the write instead.
        """
        return self._benchmarking

    @contextlib.asynccontextmanager
    async def exclusive(self, model_id: str) -> AsyncIterator[None]:
        """Hold the one benchmark slot for the duration of the block.

        Added so :mod:`studioforge.core.parallel_bench` shares this object's
        lock rather than owning a second one. Two measurement runs at once
        compete for exactly the resource each is measuring, and a parallel
        sweep is no less a benchmark than a placement sweep -- giving it its own
        lock would have let the two interleave and produce two sets of numbers
        that describe neither run.

        Setting ``_benchmarking`` here is the other half: it is what
        ``ModelManager._busy_reason`` reads, so a smoke test refuses while
        either kind of run is in flight (D36).

        Raises :class:`~studioforge.errors.ModelBusyError` rather than queueing.
        A measurement that waits several minutes for its turn describes a
        machine that has since moved.
        """
        if self._lock.locked():
            raise ModelBusyError(
                f"a benchmark of '{self._benchmarking or 'another model'}' is already "
                f"running; benchmarks are serialized because concurrent runs would "
                f"compete for the VRAM being measured",
                code="benchmark_busy",
            )
        async with self._lock:
            self._benchmarking = model_id
            try:
                yield
            finally:
                self._benchmarking = None

    def _refuse_if_busy(self) -> None:
        """Refuse while anything is serving, loading or smoke-testing (D36).

        The same rule the parallel benchmark applies, for the same two reasons:
        the numbers would measure the contention rather than the model, and
        each mode's load is a forced reload that must not be the thing that
        evicts somebody's mid-stream model. The benchmark arm of the manager's
        own ``_busy_reason`` is deliberately not consulted -- inside this
        object the running benchmark is us, and the lock above already refuses
        a second one. A manager without ``busy_snapshot`` (a test stub) is
        treated as idle.
        """
        snapshot_fn = getattr(self.manager, "busy_snapshot", None)
        if snapshot_fn is None:
            return
        from studioforge.core.parallel_bench import busy_reason_from

        snapshot = snapshot_fn()
        reason = busy_reason_from(snapshot)
        if reason is None:
            return
        raise ModelBusyError(
            f"the server is busy ({reason}); a benchmark loads the model once per "
            f"placement and measures it alone, so it waits for an idle server",
            details={"busy": snapshot, "retry_after_s": BUSY_RETRY_AFTER_S},
        )

    # -- planning ---------------------------------------------------------

    def split_modes_for(self, record: ModelRecord) -> list[str]:
        """Split modes worth offering for ``record``: ``layer``, maybe ``tensor``.

        Model-only gating, deliberately: whether the *engine* offers tensor mode
        and whether *this plan's* KV cache and flash-attn setting allow it are
        decided at launch by the supervisor, which is the one place that knows
        both. Offering a mode the launch then refuses costs one clearly-worded
        error; hiding a mode the launch would have accepted costs a measurement
        the user asked for.
        """
        if tensor_split_model_blockers(record):
            return ["layer"]
        return ["layer", SPLIT_MODE_TENSOR]

    def modes_for(
        self,
        record: ModelRecord,
        *,
        ctx_size: int,
        ubatch_sizes: Sequence[int] = (),
    ) -> list[tuple[BenchmarkMode, bool, str | None]]:
        """Every mode with the planner's verdict on whether it can run.

        Inapplicable modes are *returned*, not dropped, so the caller can show
        the rejection reason next to the mode the user was hoping for.
        """
        modes = available_modes(
            self._probe.list_gpus(),
            split_modes=self.split_modes_for(record),
            ubatch_sizes=ubatch_sizes,
        )
        return [(mode, *self._applicability(record, mode, ctx_size=ctx_size)) for mode in modes]

    @staticmethod
    def _settings_for(record: ModelRecord, mode: BenchmarkMode) -> ModelSettings:
        """The record's settings, steered at one mode. One place, two callers.

        ``_applicability`` and ``_run_mode`` have to agree exactly: a mode
        planned with a layer split and then *run* with a tensor one would report
        the wrong verdict for the wrong launch.
        """
        update: dict[str, Any] = {
            "device_override": list(mode.devices),
            "split_mode": mode.split_mode,
        }
        if mode.ubatch is not None:
            update["ubatch_size"] = mode.ubatch
        return record.settings.model_copy(update=update)

    def _applicability(
        self, record: ModelRecord, mode: BenchmarkMode, *, ctx_size: int
    ) -> tuple[bool, str | None]:
        candidate = record.model_copy(update={"settings": self._settings_for(record, mode)})
        try:
            result = self.manager.planner.plan_load(
                candidate,
                ctx_size=ctx_size,
                loaded=self.manager.supervisor.list(),
                draft=self.manager._draft_for(record),
                adapters=[adapter for adapter, _ in self.manager._adapters_for(record)],
                # The benchmark unloads this model before every mode, so VRAM it
                # currently holds is genuinely available; without allowing the
                # eviction the planner would reject a mode that will in fact fit.
                allow_evict=True,
            )
        except Exception as exc:  # pragma: no cover - planner errors are rare
            log.warning("benchmark.plan_failed", model_id=record.id, mode=mode.key, error=str(exc))
            return False, f"could not plan this placement: {exc}"
        if isinstance(result, LoadRejected):
            return False, result.reason
        return True, None

    # -- running ----------------------------------------------------------

    async def run(
        self,
        record: ModelRecord,
        *,
        modes: Sequence[str] | None = None,
        ctx_size: int = DEFAULT_CTX_SIZE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        prompt: str | None = None,
        on_progress: ProgressFn | None = None,
        cancel_event: asyncio.Event | None = None,
        ubatch_sizes: Sequence[int] = (),
    ) -> BenchmarkReport:
        """Benchmark ``record`` across ``modes`` (default: every applicable one).

        Raises :class:`~studioforge.errors.ModelBusyError` when another run is
        in flight. ``cancel_event`` is checked between modes, so a cancel stops
        cleanly at a mode boundary rather than mid-load; task cancellation also
        works and restores state the same way.

        ``ubatch_sizes`` adds a prompt-processing dimension (e.g. ``(1024,
        2048)``): each placement is measured again at that ``-ub``. Empty by
        default -- it multiplies the run length, and it only moves prefill.
        """
        if self._lock.locked():
            raise ModelBusyError(
                "a benchmark is already running; benchmarks are serialized because "
                "concurrent runs would compete for the VRAM being measured",
                code="benchmark_busy",
            )
        self._refuse_if_busy()
        async with self._lock:
            # Re-checked inside the lock: the gap between the first check and
            # taking the slot is exactly where a JIT request lands.
            self._refuse_if_busy()
            return await self._run_locked(
                record,
                modes=modes,
                ctx_size=ctx_size,
                max_tokens=max_tokens,
                prompt=prompt,
                on_progress=on_progress,
                cancel_event=cancel_event,
                ubatch_sizes=ubatch_sizes,
            )

    async def _run_locked(
        self,
        record: ModelRecord,
        *,
        modes: Sequence[str] | None,
        ctx_size: int,
        max_tokens: int,
        prompt: str | None,
        on_progress: ProgressFn | None,
        cancel_event: asyncio.Event | None,
        ubatch_sizes: Sequence[int] = (),
    ) -> BenchmarkReport:
        text = DEFAULT_PROMPT if prompt is None else prompt
        report = BenchmarkReport(
            model_id=record.id,
            started_at=time.time(),
            finished_at=None,
            ctx_size=ctx_size,
            max_tokens=max_tokens,
            prompt_chars=len(text),
            results=[],
            notes=[
                "throughput is llama-server's own `timings` (prompt_per_second / "
                "predicted_per_second), which excludes gateway overhead"
            ],
        )

        _emit(on_progress, None, "planning", 0, 1)
        planned = self.modes_for(record, ctx_size=ctx_size, ubatch_sizes=ubatch_sizes)
        if modes is not None:
            wanted = list(dict.fromkeys(modes))
            known = {mode.key for mode, _, _ in planned}
            unknown = [key for key in wanted if key not in known]
            if unknown:
                raise BadRequestError(
                    f"unknown benchmark mode(s): {', '.join(unknown)}; "
                    f"available: {', '.join(sorted(known)) or 'none'}",
                    param="modes",
                )
            # Kept in fastest-first order rather than the caller's order, so
            # two runs of the same set are directly comparable.
            selected = set(wanted)
            planned = [entry for entry in planned if entry[0].key in selected]

        total = len(planned) or 1
        # Restoring this exact object is the whole safety story: a benchmark
        # that left a model pinned to one GPU would silently degrade every
        # later load. It happens in `finally`, so an error, a rejection or a
        # cancellation all land on the same path.
        original_settings = record.settings
        self._benchmarking = record.id
        try:
            for position, (mode, applicable, reason) in enumerate(planned):
                if cancel_event is not None and cancel_event.is_set():
                    report.notes.append(f"canceled after {position} of {len(planned)} modes")
                    break
                if not applicable:
                    report.results.append(
                        BenchmarkResult(
                            mode=mode.key,
                            label=mode.label,
                            devices=list(mode.devices),
                            split_mode=mode.split_mode,
                            ubatch=mode.ubatch,
                            applicable=False,
                            skipped_reason=reason,
                        )
                    )
                    _emit(on_progress, mode.key, "done", position, total)
                    continue
                report.results.append(
                    await self._run_mode(
                        record,
                        mode,
                        base_settings=original_settings,
                        ctx_size=ctx_size,
                        max_tokens=max_tokens,
                        prompt=text,
                        report=report,
                        position=position,
                        total=total,
                        on_progress=on_progress,
                    )
                )
        finally:
            record.settings = original_settings
            self._benchmarking = None
            await self._restore_unloaded(record.id)

        report.finished_at = time.time()
        report.recompute_best()
        _emit(on_progress, None, "done", total, total)
        log.info(
            "benchmark.complete",
            model_id=record.id,
            modes=len(report.results),
            best_generation_mode=report.best_generation_mode,
        )
        return report

    async def _restore_unloaded(self, model_id: str) -> None:
        """Unload the model, even when the run is being cancelled.

        Shielded: on cancellation the await returns immediately but the unload
        still completes in the background, so a cancelled benchmark never
        leaves a child process holding VRAM.
        """
        # The event loop keeps only WEAK references to tasks, so a detached
        # unload can be garbage-collected mid-flight -- leaving a llama-server
        # child holding VRAM, which is the exact failure this shield exists to
        # prevent. Same guard the restart and rescan paths already use.
        task = asyncio.ensure_future(self.manager.unload(model_id))
        _DETACHED_UNLOADS.add(task)
        task.add_done_callback(_DETACHED_UNLOADS.discard)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            log.info("benchmark.unload_detached", model_id=model_id)
        except Exception as exc:  # pragma: no cover - teardown is best-effort
            log.warning("benchmark.unload_failed", model_id=model_id, error=str(exc))

    async def _run_mode(
        self,
        record: ModelRecord,
        mode: BenchmarkMode,
        *,
        base_settings: ModelSettings,
        ctx_size: int,
        max_tokens: int,
        prompt: str,
        report: BenchmarkReport,
        position: int,
        total: int,
        on_progress: ProgressFn | None,
    ) -> BenchmarkResult:
        """Measure one placement. Never raises for engine/model failures."""
        result = BenchmarkResult(
            mode=mode.key,
            label=mode.label,
            devices=list(mode.devices),
            split_mode=mode.split_mode,
            ubatch=mode.ubatch,
        )
        lease = None
        try:
            # A fresh process per mode: reusing a running instance would measure
            # a warm cache on whichever devices happened to be selected first.
            await self.manager.unload(record.id)
            # The mode's cards are this benchmark's alone until the mode is
            # done (D43): an idle resident on them is unloaded, a busy one
            # fails the mode by name, and nothing else may load there meanwhile
            # -- so the number measured is the card's, not the neighbours'.
            lease = await self.manager.acquire_lease(
                mode.devices,
                holder="benchmark",
                model_ids=[record.id],
                reason=f"benchmark {mode.key}",
            )
            steered = record.model_copy(update={"settings": base_settings})
            record.settings = self._settings_for(steered, mode)

            free_before = self._free_by_device()
            _emit(on_progress, mode.key, "loading", position, total)
            started = time.perf_counter()
            # force reloads (a fresh child per mode); evict_busy=False keeps
            # D36's rule that a load never interrupts a stream.
            instance = await self.manager.load(
                record.id, ctx_size=ctx_size, force=True, evict_busy=False, source="benchmark"
            )
            result.load_time_s = round(time.perf_counter() - started, 3)
            result.vram_used_bytes = self._vram_delta(free_before, mode.devices)

            _emit(on_progress, mode.key, "generating", position, total)
            await self._measure(
                record,
                instance.model_id,
                result,
                prompt=prompt,
                max_tokens=max_tokens,
                report=report,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # One bad mode must not cost the user the other three.
            result.error = str(exc)
            log.warning("benchmark.mode_failed", model_id=record.id, mode=mode.key, error=str(exc))
        finally:
            if lease is not None:
                with contextlib.suppress(Exception):
                    self.manager.release_lease(lease.id)
        _emit(on_progress, mode.key, "done", position, total)
        return result

    async def _measure(
        self,
        record: ModelRecord,
        serving_id: str,
        result: BenchmarkResult,
        *,
        prompt: str,
        max_tokens: int,
        report: BenchmarkReport,
    ) -> None:
        """One streaming completion; fills the throughput fields on ``result``.

        Streaming is what makes time-to-first-token observable at all -- a
        non-streamed request only reveals total latency. The token rates still
        come from the engine's ``timings``; the stopwatch here is only for TTFT
        and for the fallback path.
        """
        base = self.manager.supervisor.base_url(serving_id)
        if base is None:
            raise ModelLoadError(f"model '{record.id}' is not serving after load")

        payload: dict[str, Any] = {
            "model": record.id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            # Deterministic: greedy sampling makes two runs comparable, and a
            # reused prompt cache would make prompt processing look free.
            "temperature": 0,
            "top_k": 1,
            "seed": 0,
            "cache_prompt": False,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        timings: dict[str, Any] | None = None
        usage: dict[str, Any] | None = None
        ttft: float | None = None

        self.manager.supervisor.mark_request_start(serving_id)
        started = time.perf_counter()
        try:
            async with (
                httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client,
                client.stream("POST", f"{base}/v1/chat/completions", json=payload) as response,
            ):
                response.raise_for_status()
                async for line in response.aiter_lines():
                    chunk = _sse_chunk(line)
                    if chunk is None:
                        continue
                    if isinstance(chunk.get("timings"), dict):
                        timings = chunk["timings"]
                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]
                    if ttft is None and _has_content(chunk):
                        ttft = time.perf_counter() - started
        finally:
            self.manager.supervisor.mark_request_end(serving_id)
        wall = time.perf_counter() - started

        result.ttft_s = round(ttft, 4) if ttft is not None else None
        if timings is not None:
            result.prompt_tokens = _as_int(timings.get("prompt_n"))
            result.prompt_tps = _as_float(timings.get("prompt_per_second"))
            result.generated_tokens = _as_int(timings.get("predicted_n"))
            result.generation_tps = _as_float(timings.get("predicted_per_second"))

        if result.generation_tps is None or result.generated_tokens is None:
            self._wall_clock_fallback(result, usage=usage, wall=wall, ttft=ttft, report=report)

    def _wall_clock_fallback(
        self,
        result: BenchmarkResult,
        *,
        usage: dict[str, Any] | None,
        wall: float,
        ttft: float | None,
        report: BenchmarkReport,
    ) -> None:
        """Derive rates from our own clock when the engine reported no timings.

        Noted in the report because these numbers include HTTP and SSE overhead
        and are therefore not comparable with ``llama-bench``.
        """
        if result.generated_tokens is None and usage is not None:
            result.generated_tokens = _as_int(usage.get("completion_tokens"))
        if result.prompt_tokens is None and usage is not None:
            result.prompt_tokens = _as_int(usage.get("prompt_tokens"))

        generation_window = max(1e-6, wall - (ttft or 0.0))
        if result.generation_tps is None and result.generated_tokens:
            result.generation_tps = round(result.generated_tokens / generation_window, 2)
        if result.prompt_tps is None and result.prompt_tokens and ttft:
            result.prompt_tps = round(result.prompt_tokens / max(1e-6, ttft), 2)

        note = (
            f"{result.mode}: llama-server reported no `timings`; rates for this mode "
            f"were computed from wall clock and include gateway overhead"
        )
        if note not in report.notes:
            report.notes.append(note)

    # -- VRAM -------------------------------------------------------------

    def _free_by_device(self) -> dict[int, int]:
        return {gpu.index: gpu.free_bytes for gpu in self._probe.list_gpus()}

    def _vram_delta(self, before: dict[int, int], devices: Sequence[int]) -> int | None:
        """VRAM the load actually consumed, as a free-memory delta.

        A delta rather than ``used_bytes``: other processes (a desktop, another
        model) already hold memory on these cards, and attributing that to this
        load would inflate every number on a busy box.
        """
        after = self._free_by_device()
        total = 0
        for device in devices:
            if device not in before or device not in after:
                return None
            total += max(0, before[device] - after[device])
        return total or None


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


def _sse_chunk(line: str) -> dict[str, Any] | None:
    """Parse one ``data:`` SSE line into a JSON object, or None."""
    if not line.startswith("data:"):
        return None
    body = line[5:].strip()
    if not body or body == "[DONE]":
        return None
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _has_content(chunk: dict[str, Any]) -> bool:
    """Whether this chunk carries the first visible generated text."""
    for choice in chunk.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        for key in ("content", "reasoning_content"):
            if delta.get(key):
                return True
    return False


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return round(number, 3) if number > 0 else None


def _emit(
    on_progress: ProgressFn | None, mode: str | None, phase: str, position: int, total: int
) -> None:
    """Report progress; a broken callback must never fail the benchmark."""
    if on_progress is None:
        return
    span = max(1, total)
    fraction = (position + _PHASE_WEIGHT.get(phase, 0.0)) / span
    with contextlib.suppress(Exception):
        on_progress(mode, phase, min(1.0, max(0.0, fraction)))


# ---------------------------------------------------------------------------
# Background jobs
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkJob:
    """In-memory state of one background benchmark run.

    A full suite on a large model takes minutes, which no synchronous HTTP
    request survives; the API starts a task and clients poll this.
    """

    job_id: str
    model_id: str
    modes: list[str]
    state: str = "running"
    phase: str = "planning"
    mode: str | None = None
    fraction: float = 0.0
    completed: int = 0
    total: int = 0
    report: dict[str, Any] | None = None
    error: str | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None

    def on_progress(self, mode: str | None, phase: str, fraction: float) -> None:
        self.mode = mode
        self.phase = phase
        self.fraction = round(fraction, 4)
        if phase == "done" and mode is not None:
            self.completed += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "state": self.state,
            "progress": {
                "mode": self.mode,
                "phase": self.phase,
                "fraction": self.fraction,
                "completed": self.completed,
                "total": self.total,
            },
            "report": self.report,
            "error": self.error,
            "cancel_requested": self.cancel_event.is_set(),
        }


class BenchmarkJobs:
    """Bounded, insertion-ordered job table.

    Bounded because it lives for the process lifetime: without a cap, every
    benchmark ever started would be retained. Completed reports are in SQLite,
    so evicting an old job loses nothing durable.
    """

    def __init__(self, *, limit: int = 20) -> None:
        self._limit = max(1, limit)
        self._jobs: OrderedDict[str, BenchmarkJob] = OrderedDict()

    def create(self, model_id: str, modes: Sequence[str]) -> BenchmarkJob:
        job = BenchmarkJob(
            job_id=uuid.uuid4().hex[:16],
            model_id=model_id,
            modes=list(modes),
            total=len(modes),
        )
        self._jobs[job.job_id] = job
        while len(self._jobs) > self._limit:
            # Oldest first, but never evict a live job -- a client is polling it.
            victim = next(
                (key for key, value in self._jobs.items() if value.state != "running"), None
            )
            if victim is None:
                break
            del self._jobs[victim]
        return job

    def get(self, job_id: str) -> BenchmarkJob | None:
        return self._jobs.get(job_id)

    def all(self) -> list[BenchmarkJob]:
        return list(self._jobs.values())

    def running(self) -> BenchmarkJob | None:
        return next((job for job in self._jobs.values() if job.state == "running"), None)
