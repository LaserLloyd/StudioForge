"""Pure helpers shared by run_bench.py and compare.py.

No network, no imports from the StudioForge package: everything here is
testable with a plain pytest and stays importable from either script.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from typing import Any

#: D46 tiers (core/priority.py). The harness loads at the AGENT tier: it may
#: displace idle background residents from the cards it takes, never a chat one.
TIER_CHAT, TIER_AGENT, TIER_BACKGROUND = 1, 2, 3

#: Per-device floor below which a holder's bytes are a CUDA context, not a
#: placement (core/vram_holders.py DEVICE_PLACEMENT_MIN_BYTES, D39/D40).
DEVICE_PLACEMENT_MIN_BYTES = 512 * 1024 * 1024

#: (key, label, higher_is_better). Keys are the per-run row fields.
METRICS: tuple[tuple[str, str, bool], ...] = (
    ("prefill_tps", "prefill tok/s", True),
    ("decode_tps", "decode tok/s", True),
    ("ttft_s", "TTFT s", False),
    ("wall_s", "wall s", False),
)

#: The amended acceptance rule (README.md): the bar gates major restructuring only.
MAJOR_BAR_PCT = 10.0
REGRESSION_TOLERANCE_PCT = 2.0
DECODE_GATE_LENGTH = 262144
PREFILL_GATE_MIN_LENGTH = 131072


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def median(values: Iterable[float]) -> float | None:
    data = sorted(float(v) for v in values)
    if not data:
        return None
    mid = len(data) // 2
    if len(data) % 2:
        return data[mid]
    return (data[mid - 1] + data[mid]) / 2.0


def p95(values: Iterable[float]) -> float | None:
    """Nearest-rank 95th percentile: the value at rank ceil(0.95 * n), 1-based."""
    data = sorted(float(v) for v in values)
    if not data:
        return None
    rank = max(1, math.ceil(0.95 * len(data)))
    return data[rank - 1]


def delta_pct(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return (float(new) - float(old)) / abs(float(old)) * 100.0


def summarize_runs(runs: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per prompt length: median/p95/n per metric, peak VRAM and cache flags.

    Warm-up rows and rows carrying an ``error`` are excluded; a metric a run
    could not report (``None``) is skipped for that run only.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in runs:
        buckets.setdefault(str(row["prompt_length"]), []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for key, rows in buckets.items():
        kept = [r for r in rows if not r.get("warmup") and not r.get("error")]
        entry: dict[str, Any] = {"runs_measured": len(kept), "runs_total": len(rows)}
        for metric, _label, _hib in METRICS:
            vals = [r[metric] for r in kept if r.get(metric) is not None]
            entry[metric] = {"median": median(vals), "p95": p95(vals), "n": len(vals)}
        peak: dict[str, int] = {}
        for r in kept:
            for dev, used in (r.get("peak_vram_bytes") or {}).items():
                peak[str(dev)] = max(peak.get(str(dev), 0), int(used))
        entry["peak_vram_bytes"] = peak
        entry["prompt_n_median"] = median([r["prompt_n"] for r in kept if r.get("prompt_n")])
        entry["cache_hit_suspected_runs"] = sum(1 for r in kept if r.get("cache_hit_suspected"))
        entry["timings_missing_runs"] = sum(1 for r in kept if r.get("timings_missing"))
        entry["foreign_holder_runs"] = sum(1 for r in kept if r.get("foreign_holders_seen"))
        summary[key] = entry
    return summary


# --------------------------------------------------------------------------
# validity
# --------------------------------------------------------------------------


def holder_on_devices(holder: dict[str, Any], devices: Sequence[int]) -> tuple[bool, str]:
    """Whether a ``/api/vram/holders`` row holds a placement on any of ``devices``.

    ``per_gpu_bytes`` (PDH, D39) is the measurement; ``gpu_indices`` is the
    fallback (``nvml-context`` names every card the process has a context on,
    so it needs a real total too); no attribution at all counts as on them.
    """
    per_gpu = holder.get("per_gpu_bytes") or {}
    if per_gpu:
        hit = any(int(per_gpu.get(str(d)) or 0) >= DEVICE_PLACEMENT_MIN_BYTES for d in devices)
        return hit, "per_gpu_bytes"
    used = int(holder.get("used_bytes") or 0)
    indices = holder.get("gpu_indices") or []
    if indices:
        hit = used >= DEVICE_PLACEMENT_MIN_BYTES and any(d in indices for d in devices)
        return hit, str(holder.get("gpu_indices_source") or "gpu_indices")
    return used >= DEVICE_PLACEMENT_MIN_BYTES, "unattributed"


def classify_validity(
    payload: dict[str, Any], devices: Sequence[int], own_pids: Iterable[int]
) -> dict[str, Any]:
    """The invalidity rule on one holders snapshot: valid only when every compute
    holder on the target devices is one of ``own_pids``. Desktop processes are
    already collapsed by the server; their count is recorded, never decisive."""
    mine = {int(p) for p in own_pids}
    offending: list[dict[str, Any]] = []
    own: list[dict[str, Any]] = []
    for holder in payload.get("holders") or []:
        hit, how = holder_on_devices(holder, devices)
        if not hit:
            continue
        row = {
            "pid": holder.get("pid"),
            "name": holder.get("name"),
            "classification": holder.get("classification"),
            "alias": holder.get("alias"),
            "port": holder.get("port"),
            "used_bytes": holder.get("used_bytes"),
            "per_gpu_bytes": holder.get("per_gpu_bytes"),
            "gpu_indices": holder.get("gpu_indices"),
            "attribution": how,
        }
        (own if holder.get("pid") in mine else offending).append(row)
    return {
        "valid": not offending,
        "offending": offending,
        "own": own,
        "holder_count": len(payload.get("holders") or []),
        "desktop_processes_count": payload.get("desktop_processes_count"),
        "desktop_processes_bytes": payload.get("desktop_processes_bytes"),
        "per_gpu_bytes_source": payload.get("per_gpu_bytes_source"),
    }


# --------------------------------------------------------------------------
# verdict (the amended rule)
# --------------------------------------------------------------------------


def _headline(summary: dict[str, Any], metric: str, key: str) -> float | None:
    entry = summary.get(key)
    if not entry:
        return None
    return (entry.get(metric) or {}).get("median")


def verdict(
    new: dict[str, Any],
    old: dict[str, Any],
    *,
    new_valid: bool = True,
    old_valid: bool = True,
) -> dict[str, Any]:
    """Grade ``new`` against ``old`` (both ``summarize_runs`` outputs) by the amended
    rule. Headlines: median decode tok/s at 256k, and median prefill tok/s at
    >= 128k as the WORST delta over every such bucket present in both files."""
    decode_key = str(DECODE_GATE_LENGTH)
    decode = delta_pct(
        _headline(new, "decode_tps", decode_key), _headline(old, "decode_tps", decode_key)
    )
    prefill_deltas: dict[str, float] = {}
    for key in sorted((k for k in new if k in old and int(k) >= PREFILL_GATE_MIN_LENGTH), key=int):
        delta = delta_pct(_headline(new, "prefill_tps", key), _headline(old, "prefill_tps", key))
        if delta is not None:
            prefill_deltas[key] = delta
    prefill = min(prefill_deltas.values()) if prefill_deltas else None
    prefill_key = (
        f">={PREFILL_GATE_MIN_LENGTH} (min over {','.join(prefill_deltas)})"
        if prefill_deltas
        else None
    )
    result: dict[str, Any] = {
        "decode_delta_pct": decode,
        "prefill_delta_pct": prefill,
        "prefill_deltas_pct": prefill_deltas,
        "decode_bucket": decode_key,
        "prefill_bucket": prefill_key,
    }
    if decode is None or prefill is None:
        result["verdict"] = "no-verdict"
        result["line"] = (
            "VERDICT: none -- both files need the 256k decode bucket and a >=128k prefill "
            "bucket (a smoke run has neither)"
        )
        return result
    tol = -REGRESSION_TOLERANCE_PCT
    detail = f"decode {decode:+.1f}% at {decode_key}, prefill {prefill:+.1f}% at {prefill_key}"
    if decode < tol or prefill < tol:
        result["verdict"] = "regression"
        line = (
            f"VERDICT: REGRESSION ({detail}) -- exceeds the "
            f"{REGRESSION_TOLERANCE_PCT:.0f}% tolerance"
        )
    elif (decode >= MAJOR_BAR_PCT and prefill >= tol) or (
        prefill >= MAJOR_BAR_PCT and decode >= tol
    ):
        result["verdict"] = "major-bar-met"
        line = (
            f"VERDICT: MAJOR-RESTRUCTURING BAR MET ({detail}; >= {MAJOR_BAR_PCT:.0f}% "
            f"with no > {REGRESSION_TOLERANCE_PCT:.0f}% regression)"
        )
    elif decode > 0 or prefill > 0:
        result["verdict"] = "keep"
        line = (
            f"VERDICT: KEEP ({detail}) -- an improvement is kept regardless of size; "
            f"below the {MAJOR_BAR_PCT:.0f}% bar that gates major restructuring"
        )
    else:
        result["verdict"] = "neutral"
        line = f"VERDICT: NEUTRAL ({detail}) -- within tolerance, no improvement"
    if not (new_valid and old_valid):
        which = " and ".join(n for n, ok in (("new", new_valid), ("old", old_valid)) if not ok)
        line = f"INVALID ({which} results file failed the validity rule) -- advisory only: {line}"
        result["verdict"] = "invalid:" + result["verdict"]
    result["line"] = line
    return result


# --------------------------------------------------------------------------
# prompts and small utilities
# --------------------------------------------------------------------------

#: The vocabulary as a prose block rather than a 160-element literal: the
#: words are the data, the split is how they are read.
_WORDS_TEXT = """\
the of and to in that is was for with as his on be at by this had not are but from or have
    an they which one you were her all she there would their we him been has when who will more no
    if out so said what up its about into than them can only other new some could time these two
    may then do first any my now such like our over man me even most made after also did many
    before must through back years where much your way well down should because each just those
    people how too little state good very make world still own see men work long get here between
    both life being under never day same another know while last might us great old year off come
    since against go came right used take three house river mountain garden window letter morning
    evening silver copper stone bread water light shadow voice story music paper travel market
    harbour castle forest winter summer autumn spring engine signal answer question memory village
    teacher doctor farmer sailor painter carpenter merchant walked carried opened closed watched
    listened remembered forgot wrote read built broke found lost quiet bright heavy narrow ancient
    gentle sudden careful curious distant"""

WORDS: tuple[str, ...] = tuple(_WORDS_TEXT.split())


def build_prompt(n_words: int, seed: int, header: str = "") -> str:
    """Deterministic English-like prose of exactly ``n_words`` words. Prefix-stable:
    a larger ``n_words`` at the same seed extends the text (monotone calibration).
    ``header`` is prepended verbatim (the per-run marker that defeats prefix reuse)."""
    rng = random.Random(seed)
    words: list[str] = []
    sentences_in_para = 0
    while len(words) < n_words:
        length = rng.randint(5, 15)
        sentence = [rng.choice(WORDS) for _ in range(length)]
        sentence[0] = sentence[0].capitalize()
        if length > 8 and rng.random() < 0.5:
            cut = rng.randint(3, length - 3)
            sentence[cut - 1] += ","
        sentence[-1] += rng.choice(".....?!")
        sentences_in_para += 1
        if sentences_in_para >= rng.randint(4, 9):
            sentence[-1] += "\n\n"
            sentences_in_para = 0
        words.extend(sentence)
    body = " ".join(words[:n_words]).replace("\n\n ", "\n\n").strip()
    return f"{header}{body}" if header else body


def estimate_words(target_tokens: int, tokens_per_word: float) -> int:
    return max(1, math.ceil(target_tokens / max(0.05, tokens_per_word)))


def clamp_prompt_length(requested: int, ctx: int, max_tokens: int, margin: int = 64) -> int:
    """The largest prompt (tokens) that leaves room for the reply inside ``ctx``."""
    return max(1, min(int(requested), int(ctx) - int(max_tokens) - int(margin)))


def parse_int_list(text: str) -> list[int]:
    values = [int(part) for part in str(text).replace(";", ",").split(",") if part.strip()]
    if not values:
        raise ValueError(f"expected a comma-separated list of integers, got {text!r}")
    return values


def result_filename(sha: str, label: str | None, smoke: bool) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in (label or "")).strip("-")
    name = sha[:12] if sha else "nogit"
    if safe:
        name += f"-{safe}"
    if smoke:
        name += "-smoke"
    return name + ".json"


def plan_tuple(plan: dict[str, Any] | None) -> tuple[Any, ...] | None:
    """The D42 five-field reload tuple of an instance's plan, for equality checks."""
    if not plan:
        return None
    return (
        plan.get("ctx_size"),
        plan.get("parallel"),
        tuple(sorted(int(d) for d in (plan.get("devices") or []))),
        plan.get("kv_cache_type"),
        plan.get("kv_cache_type_v"),
    )
