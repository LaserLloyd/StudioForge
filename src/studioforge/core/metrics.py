"""Prometheus ``/metrics``: the gateway's own counters plus the children's.

Two sources, one page:

* **The gateway's counters.** Requests by route and status class, request
  duration (a sum and a count, so ``rate(sum)/rate(count)`` is the mean), in-flight
  requests, uptime, loaded models, per-GPU memory and the active-key count. Kept in
  memory by :class:`GatewayMetrics` -- a few dicts behind one lock, observed once
  per request by the auth middleware.
* **Every loaded model's ``llama-server --metrics``.** Each child is launched with
  ``--metrics`` (``core/supervisor.py``), so its own Prometheus text (prompt
  tokens, generation tokens, slots, KV usage) is fetched, given a ``model`` label
  and appended -- one scrape target for the whole box. A child that does not
  answer is skipped, never an error: ``/metrics`` must come back while a model is
  mid-load.

Route labels use the route *template* (``/api/models/{model_id}/load``), not the
URL, so label cardinality is bounded by the code, not by the traffic.
"""

from __future__ import annotations

import re
import threading
import time
from collections import defaultdict
from typing import Any

__all__ = ["GatewayMetrics", "label_child_metrics", "render_metrics"]

_ESCAPES = str.maketrans({"\\": "\\\\", '"': '\\"', "\n": "\\n"})


def _esc(value: Any) -> str:
    return str(value).translate(_ESCAPES)


class GatewayMetrics:
    """Request counters. Thread-safe; ``observe`` is cheap enough to call per request."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, str, str], int] = defaultdict(int)
        self._duration_sum: dict[tuple[str, str], float] = defaultdict(float)
        self._duration_count: dict[tuple[str, str], int] = defaultdict(int)
        self.in_flight = 0

    def begin(self) -> None:
        with self._lock:
            self.in_flight += 1

    def observe(self, method: str, route: str, status: int, seconds: float) -> None:
        klass = f"{status // 100}xx"
        with self._lock:
            self.in_flight = max(0, self.in_flight - 1)
            self._requests[(method, route, klass)] += 1
            self._duration_sum[(method, route)] += max(0.0, seconds)
            self._duration_count[(method, route)] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests": dict(self._requests),
                "duration_sum": dict(self._duration_sum),
                "duration_count": dict(self._duration_count),
                "in_flight": self.in_flight,
            }


_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(.+)$")


def label_child_metrics(text: str, model: str) -> list[str]:
    """Add ``model="<id>"`` to every sample line of a child's Prometheus text.

    ``# HELP`` / ``# TYPE`` lines pass through unchanged (a Prometheus parser
    tolerates repeats of the same metadata across scrape sections of one page
    only when they agree, so they are de-duplicated by the caller).
    """
    out: list[str] = []
    label = f'model="{_esc(model)}"'
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.startswith("#"):
            out.append(line)
            continue
        match = _SAMPLE.match(line)
        if not match:
            continue
        name, labels, value = match.groups()
        if labels:
            inner = labels[1:-1].strip()
            labels = "{" + (inner + "," if inner else "") + label + "}"
        else:
            labels = "{" + label + "}"
        out.append(f"{name}{labels} {value}")
    return out


def _gauge(lines: list[str], name: str, help_text: str, rows: list[tuple[str, float]]) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} gauge")
    for labels, value in rows:
        lines.append(f"{name}{labels} {value:g}")


async def render_metrics(state: Any, *, include_children: bool = True) -> str:
    """The full ``/metrics`` page for a composed app ``state``."""
    from studioforge import __version__
    from studioforge.build import build_id

    lines: list[str] = []
    metrics: GatewayMetrics | None = getattr(state, "metrics", None)
    if metrics is not None:
        snap = metrics.snapshot()
        lines += [
            "# HELP studioforge_http_requests_total Requests handled, by route and status class.",
            "# TYPE studioforge_http_requests_total counter",
        ]
        for (method, route, klass), count in sorted(snap["requests"].items()):
            lines.append(
                f'studioforge_http_requests_total{{method="{method}",route="{_esc(route)}",'
                f'status="{klass}"}} {count}'
            )
        lines += [
            "# HELP studioforge_http_request_duration_seconds Request duration (sum and count).",
            "# TYPE studioforge_http_request_duration_seconds summary",
        ]
        for (method, route), total in sorted(snap["duration_sum"].items()):
            labels = f'method="{method}",route="{_esc(route)}"'
            lines.append(f"studioforge_http_request_duration_seconds_sum{{{labels}}} {total:g}")
            count = snap["duration_count"][(method, route)]
            lines.append(f"studioforge_http_request_duration_seconds_count{{{labels}}} {count}")
        _gauge(
            lines,
            "studioforge_http_requests_in_flight",
            "Requests being handled right now.",
            [("", float(snap["in_flight"]))],
        )

    _gauge(
        lines,
        "studioforge_build_info",
        "Constant 1; the version and build are the labels.",
        [(f'{{version="{_esc(__version__)}",build="{_esc(build_id())}"}}', 1.0)],
    )
    started = getattr(state, "started_at", None)
    if started:
        _gauge(
            lines,
            "studioforge_uptime_seconds",
            "Seconds since this server started.",
            [("", max(0.0, time.time() - float(started)))],
        )

    loaded: list[Any] = []
    try:
        loaded = list(state.supervisor.list())
    except Exception:  # noqa: BLE001 - metrics must answer when a subsystem cannot
        loaded = []
    _gauge(
        lines,
        "studioforge_models_loaded",
        "Models currently resident.",
        [("", float(len(loaded)))],
    )
    try:
        gpus = list(state.probe.list_gpus())
    except Exception:  # noqa: BLE001
        gpus = []
    if gpus:
        _gauge(
            lines,
            "studioforge_gpu_memory_total_bytes",
            "Total memory per GPU.",
            [(f'{{gpu="{g.index}",name="{_esc(g.name)}"}}', float(g.total_bytes)) for g in gpus],
        )
        _gauge(
            lines,
            "studioforge_gpu_memory_free_bytes",
            "Free memory per GPU.",
            [(f'{{gpu="{g.index}",name="{_esc(g.name)}"}}', float(g.free_bytes)) for g in gpus],
        )
    accounts = getattr(state, "accounts", None)
    if accounts is not None:
        try:
            summary = accounts.summary()
            _gauge(
                lines,
                "studioforge_api_keys_active",
                "Active API keys, by scope.",
                [
                    (f'{{scope="{scope}"}}', float(count))
                    for scope, count in summary["keys_by_scope"].items()
                ],
            )
        except Exception:  # noqa: BLE001
            pass

    if include_children and loaded:
        seen_meta: set[str] = set()
        for instance in loaded:
            try:
                text = await state.supervisor.metrics(instance.model_id)
            except Exception:  # noqa: BLE001 - one unreachable child is not a failed scrape
                text = None
            if not text:
                continue
            for line in label_child_metrics(text, instance.model_id):
                if line.startswith("#"):
                    if line in seen_meta:
                        continue
                    seen_meta.add(line)
                lines.append(line)
    return "\n".join(line for line in lines if line is not None) + "\n"
