"""Choosing between engine backends from cached micro-benchmark results.

A box can often run the same model through more than one llama.cpp backend --
a Strix Halo through HIP *and* Vulkan, an NVIDIA card through CUDA *and*
Vulkan. They are not equally fast, and not in the same way: on Strix Halo the
HIP build has been measured ahead on prompt processing and the Vulkan build
ahead on generation. So the choice is a user preference applied to measured
numbers, not a rule:

* ``fastest_replies``       -> highest token generation (``tg128``)
* ``fastest_long_prompts``  -> highest prompt processing (``pp512``)
* ``balanced`` (default)    -> the best geometric mean of each backend's
  speed relative to the fastest one on that axis

This module is the pure half: the result record, the on-disk cache
(``<data_dir>/engines/backend-bench.json``, deliberately a JSON file rather
than a SQLite table so it needs no migration and survives a registry reset) and
:func:`select_backend`. **Nothing here runs a benchmark.** The function that
does, :func:`studioforge.core.backend_bench.run_backend_benchmark`, spawns
``llama-bench`` on the GPU and is only reachable from an explicit
``studioforge engine benchmark`` call.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PREFERENCES: tuple[str, ...] = ("balanced", "fastest_replies", "fastest_long_prompts")

#: Tie-break when two backends score identically: the vendor-native API first.
_BACKEND_RANK = {"cuda": 0, "rocm": 1, "metal": 2, "vulkan": 3}

CACHE_FILE = "backend-bench.json"
CACHE_VERSION = 1


@dataclass(frozen=True)
class BackendBenchResult:
    """One measured (backend, GPU, model class) point.

    ``pp512`` / ``tg128`` are llama-bench's tokens per second for a 512-token
    prompt and a 128-token generation. ``gpu`` is a stable key for the device
    (its name, or its gfx target); ``model_class`` is whatever label the caller
    benchmarked with (a model id, or a coarse class such as ``"dense-8b-q4"``)
    -- numbers are only comparable within one model class.
    """

    backend: str
    gpu: str
    engine_tag: str
    pp512: float
    tg128: float
    model_class: str = ""
    measured_at: float = 0.0
    source: str = "llama-bench"

    @property
    def usable(self) -> bool:
        return all(math.isfinite(v) and v > 0 for v in (self.pp512, self.tg128))

    def key(self) -> tuple[str, str, str, str]:
        return (self.backend, self.gpu, self.engine_tag, self.model_class)


@dataclass(frozen=True)
class BackendChoice:
    backend: str
    result: BackendBenchResult
    preference: str
    #: preference score per candidate backend (higher is better; unitless).
    scores: dict[str, float] = field(default_factory=dict)
    reason: str = ""


def _normalise(preference: str) -> str:
    value = (preference or "balanced").strip().lower()
    if value not in PREFERENCES:
        raise ValueError(f"unknown engine preference {preference!r}; expected one of {PREFERENCES}")
    return value


def select_backend(
    results: Sequence[BackendBenchResult], preference: str = "balanced"
) -> BackendChoice | None:
    """The backend to use for ONE (GPU, model class), or ``None`` without data.

    ``results`` should all describe the same device and model class (use
    :func:`select_backend_per_gpu` to split a mixed list). If a backend has
    several results (different engine tags) the fastest per axis is used for
    that backend. Unusable results (zero, negative, NaN) are ignored. Ties are
    broken in favour of the vendor-native API (CUDA, ROCm, Metal, Vulkan).
    Raises :class:`ValueError` for an unknown preference.
    """
    preference = _normalise(preference)
    best: dict[str, BackendBenchResult] = {}
    for item in results:
        if not item.usable:
            continue
        current = best.get(item.backend)
        if current is None:
            best[item.backend] = item
        else:  # one point per backend: the best of each axis, from the newest tag measured
            best[item.backend] = BackendBenchResult(
                backend=item.backend,
                gpu=item.gpu,
                engine_tag=item.engine_tag
                if item.measured_at >= current.measured_at
                else current.engine_tag,
                pp512=max(item.pp512, current.pp512),
                tg128=max(item.tg128, current.tg128),
                model_class=item.model_class,
                measured_at=max(item.measured_at, current.measured_at),
                source=item.source,
            )
    if not best:
        return None
    top_pp = max(r.pp512 for r in best.values())
    top_tg = max(r.tg128 for r in best.values())

    def score(r: BackendBenchResult) -> float:
        if preference == "fastest_replies":
            return r.tg128
        if preference == "fastest_long_prompts":
            return r.pp512
        return math.sqrt((r.pp512 / top_pp) * (r.tg128 / top_tg))

    scores = {backend: score(r) for backend, r in best.items()}
    ordered = sorted(
        best,
        key=lambda b: (
            -scores[b],
            # secondary: the other axis, then the vendor-native tie-break
            -(best[b].pp512 if preference == "fastest_replies" else best[b].tg128),
            _BACKEND_RANK.get(b, 99),
            b,
        ),
    )
    winner = ordered[0]
    chosen = best[winner]
    return BackendChoice(
        backend=winner,
        result=chosen,
        preference=preference,
        scores=scores,
        reason=(
            f"{preference}: {winner} at pp512 {chosen.pp512:.1f} t/s, tg128 {chosen.tg128:.1f} t/s"
            + (f" (of {', '.join(sorted(best))})" if len(best) > 1 else " (only backend measured)")
        ),
    )


def select_backend_per_gpu(
    results: Iterable[BackendBenchResult], preference: str = "balanced", *, model_class: str = ""
) -> dict[str, BackendChoice]:
    """:func:`select_backend` for every GPU key present (``{gpu: choice}``)."""
    by_gpu: dict[str, list[BackendBenchResult]] = {}
    for item in results:
        if item.model_class == model_class:
            by_gpu.setdefault(item.gpu, []).append(item)
    out: dict[str, BackendChoice] = {}
    for gpu, items in sorted(by_gpu.items()):
        choice = select_backend(items, preference)
        if choice is not None:
            out[gpu] = choice
    return out


class BackendBenchCache:
    """Persisted benchmark results, one per (backend, gpu, engine tag, model class).

    Plain JSON, rewritten atomically. A corrupt or unreadable file reads as
    empty (a cache must never block a server start); a result for an engine tag
    other than the one asked about is simply not returned by :meth:`results`
    when ``engine_tag`` is given, so installing a new engine invalidates the
    numbers without deleting them.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._items: dict[tuple[str, str, str, str], BackendBenchResult] = {}
        self.load()

    @classmethod
    def for_engines_dir(cls, engines_dir: Path) -> BackendBenchCache:
        return cls(Path(engines_dir) / CACHE_FILE)

    def load(self) -> None:
        self._items = {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        entries = raw.get("results") if isinstance(raw, dict) else None
        if not isinstance(entries, list):
            return
        for entry in entries:
            result = _result_from_dict(entry)
            if result is not None:
                self._items[result.key()] = result

    def save(self) -> None:
        payload = {
            "version": CACHE_VERSION,
            "results": [asdict(r) for r in sorted(self._items.values(), key=lambda r: r.key())],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            raise

    def put(self, result: BackendBenchResult, *, save: bool = True) -> None:
        if result.measured_at <= 0:
            result = BackendBenchResult(**{**asdict(result), "measured_at": time.time()})
        self._items[result.key()] = result
        if save:
            self.save()

    def results(
        self,
        *,
        gpu: str | None = None,
        model_class: str | None = None,
        engine_tag: str | None = None,
        max_age_s: float | None = None,
    ) -> list[BackendBenchResult]:
        now = time.time()
        return [
            r
            for r in self._items.values()
            if (gpu is None or r.gpu == gpu)
            and (model_class is None or r.model_class == model_class)
            and (engine_tag is None or r.engine_tag == engine_tag)
            and (max_age_s is None or now - r.measured_at <= max_age_s)
        ]

    def choose(
        self,
        gpu: str,
        preference: str = "balanced",
        *,
        model_class: str = "",
        engine_tag: str | None = None,
        max_age_s: float | None = None,
    ) -> BackendChoice | None:
        """The cached recommendation for one GPU, or ``None`` if nothing is cached."""
        return select_backend(
            self.results(
                gpu=gpu, model_class=model_class, engine_tag=engine_tag, max_age_s=max_age_s
            ),
            preference,
        )

    def __len__(self) -> int:
        return len(self._items)


def _result_from_dict(entry: Any) -> BackendBenchResult | None:
    if not isinstance(entry, dict):
        return None
    try:
        return BackendBenchResult(
            backend=str(entry["backend"]),
            gpu=str(entry["gpu"]),
            engine_tag=str(entry.get("engine_tag", "")),
            pp512=float(entry["pp512"]),
            tg128=float(entry["tg128"]),
            model_class=str(entry.get("model_class", "")),
            measured_at=float(entry.get("measured_at", 0.0)),
            source=str(entry.get("source", "llama-bench")),
        )
    except (KeyError, TypeError, ValueError):
        return None
