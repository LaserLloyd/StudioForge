"""The ROCm / Vulkan / Metal probes: cached, off the caller's thread, bounded.

``amd-smi`` / ``vulkaninfo`` / ``system_profiler`` take 50-500 ms (``vulkaninfo``
sometimes seconds) and the status page, ``/metrics`` and the planner ask about
once a second. These tests drive the probes with fake tools -- nothing is
spawned -- and prove: one tool run per TTL window however many callers there
are, the tool never runs on the calling thread, and a tool that hangs costs a
caller at most ``timeout_s``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from studioforge.config import Config, GpuConfig
from studioforge.core.gpu import (
    AutoGpuProbe,
    RocmGpuProbe,
    VulkanGpuProbe,
    probe_for_config,
    reset_probe,
)
from studioforge.core.metrics import GatewayMetrics, render_metrics
from tests.unit.test_gpu_probes import (
    ROCM_SMI,
    VK_FULL,
    VK_SUMMARY,
    FakeTools,
    fx,
)


@pytest.fixture(autouse=True)
def _fresh_probe(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.delenv("SF_GPU_PROBE", raising=False)
    reset_probe()
    yield
    reset_probe()


class GatedTools(FakeTools):
    """A FakeTools whose every run waits on ``gate`` (set it to let tools finish)."""

    def __init__(self, outputs: dict[tuple[str, ...], tuple[int, str] | str]) -> None:
        super().__init__(outputs)
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.threads: set[int] = set()
        self.timeouts: list[float] = []

    def runner(self, argv: Sequence[str], timeout: float) -> tuple[int, str]:
        self.threads.add(threading.get_ident())
        self.timeouts.append(timeout)
        self.entered.set()
        self.gate.wait(10.0)  # a safety net: a test that forgets to release still ends
        return super().runner(argv, timeout)


def _rocm(tools: FakeTools, **kw: Any) -> RocmGpuProbe:
    return RocmGpuProbe(runner=tools.runner, which=tools.which, **kw)


def _wait(predicate: Any, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------------------
# One subprocess per TTL window
# ---------------------------------------------------------------------------


def test_many_concurrent_callers_share_one_tool_run() -> None:
    tools = GatedTools({ROCM_SMI: fx("rocm_smi_2gpu.json")})
    probe = _rocm(tools, ttl_s=60.0)
    results: list[int] = []

    def ask() -> None:
        results.append(len(probe.list_gpus()))

    callers = [threading.Thread(target=ask) for _ in range(8)]
    for t in callers:
        t.start()
    assert tools.entered.wait(5.0)
    time.sleep(0.1)  # let every caller reach the wait
    tools.gate.set()
    for t in callers:
        t.join(5.0)

    assert results == [2] * 8
    assert tools.calls.count(ROCM_SMI) == 1
    probe.list_gpus()  # inside the TTL window: still the same single run
    assert tools.calls.count(ROCM_SMI) == 1


def test_one_run_per_ttl_window_and_a_fresh_run_after_it() -> None:
    now = [100.0]
    tools = FakeTools({ROCM_SMI: fx("rocm_smi_2gpu.json")})
    probe = _rocm(tools, ttl_s=2.0, clock=lambda: now[0])
    for _ in range(5):
        probe.list_gpus()
        probe.driver_version()
        probe.available()
    assert tools.calls.count(ROCM_SMI) == 1
    now[0] += 2.5
    probe.list_gpus()
    assert tools.calls.count(ROCM_SMI) == 2


def test_the_tool_never_runs_on_the_calling_thread() -> None:
    tools = FakeTools({ROCM_SMI: fx("rocm_smi_2gpu.json")})
    seen: list[int] = []
    inner = tools.runner

    def runner(argv: Sequence[str], timeout: float) -> tuple[int, str]:
        seen.append(threading.get_ident())
        return inner(argv, timeout)

    RocmGpuProbe(runner=runner, which=tools.which, ttl_s=0.0).list_gpus()
    assert seen and threading.get_ident() not in seen


def test_each_tool_run_is_clamped_to_the_probe_budget() -> None:
    tools = FakeTools(
        {VK_SUMMARY: fx("vulkaninfo_summary_mixed.txt"), VK_FULL: fx("vulkaninfo_full_mixed.txt")}
    )
    seen: list[float] = []
    inner = tools.runner

    def runner(argv: Sequence[str], timeout: float) -> tuple[int, str]:
        seen.append(timeout)
        return inner(argv, timeout)

    probe = VulkanGpuProbe(runner=runner, which=tools.which, ttl_s=0.0, timeout_s=5.0)
    assert len(probe.list_gpus()) == 1
    # `vulkaninfo` (full) asks for 60 s on its own; the 5 s budget wins.
    assert seen and max(seen) <= 5.0


# ---------------------------------------------------------------------------
# A hung tool
# ---------------------------------------------------------------------------


def test_a_hung_tool_does_not_block_beyond_the_timeout() -> None:
    tools = GatedTools({ROCM_SMI: fx("rocm_smi_2gpu.json")})
    probe = _rocm(tools, ttl_s=0.0, timeout_s=0.3)

    started = time.monotonic()
    assert probe.list_gpus() == []
    assert time.monotonic() - started < 3.0  # ~0.8 s, never the tool's own runtime
    assert "timed out" in (probe.failure_reason or "")

    # The tool is still hung: later callers get an answer at once instead of
    # each paying the timeout again, and no second tool run is started.
    started = time.monotonic()
    for _ in range(20):
        assert probe.list_gpus() == []
    assert time.monotonic() - started < 0.5
    assert len(tools.calls) == 0  # nothing finished yet, and nothing new began
    assert len(tools.threads) == 1

    # It finishes (the budget is long gone, so that run publishes "nothing");
    # the next call starts a fresh run and sees the GPUs.
    tools.gate.set()
    assert _wait(lambda: probe._flight is None)
    assert len(probe.list_gpus()) == 2


def test_a_hung_refresh_serves_the_last_good_answer_while_it_is_recent() -> None:
    now = [0.0]
    tools = GatedTools({ROCM_SMI: fx("rocm_smi_2gpu.json")})
    tools.gate.set()
    probe = _rocm(tools, ttl_s=1.0, clock=lambda: now[0], timeout_s=0.2)
    assert len(probe.list_gpus()) == 2

    tools.gate.clear()  # the tool hangs from now on
    now[0] = 5.0
    started = time.monotonic()
    assert len(probe.list_gpus()) == 2  # last good answer, 5 s old
    assert time.monotonic() - started < 3.0

    now[0] = 100.0  # far too old to vouch for
    assert probe.list_gpus() == []
    tools.gate.set()
    assert _wait(lambda: probe._flight is None)


def test_shutdown_while_a_refresh_runs_is_not_undone_by_its_late_result() -> None:
    tools = GatedTools({ROCM_SMI: fx("rocm_smi_2gpu.json")})
    probe = _rocm(tools, ttl_s=60.0, timeout_s=5.0)
    caller = threading.Thread(target=probe.list_gpus)
    caller.start()
    assert tools.entered.wait(5.0)
    probe.shutdown()
    tools.gate.set()
    caller.join(5.0)
    assert _wait(lambda: len(tools.calls) >= 1)
    time.sleep(0.1)
    assert probe._stamp is None and probe._gpus == []  # the old run did not repopulate it


def test_a_tool_that_raises_still_reports_no_gpus_with_the_reason() -> None:
    def boom(argv: Sequence[str], timeout: float) -> tuple[int, str]:
        raise RuntimeError("tool exploded")

    probe = RocmGpuProbe(runner=boom, which=lambda n: "/bin/x", ttl_s=0.0)
    assert probe.list_gpus() == []
    assert "tool exploded" in (probe.failure_reason or "")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_gpu_config_defaults() -> None:
    cfg = GpuConfig()
    assert cfg.probe_cache_s == 2.0 and cfg.probe_timeout_s == 15.0


def test_probe_for_config_applies_the_cache_and_timeout(tmp_path: Path) -> None:
    config = Config(
        data_dir=tmp_path / "data",
        gpu={"backend": "rocm", "probe_cache_s": 7.5, "probe_timeout_s": 3.0},
    )
    probe = probe_for_config(config)
    assert isinstance(probe, RocmGpuProbe)
    assert probe._ttl == 7.5 and probe._timeout == 3.0


def test_auto_probe_passes_the_cache_and_timeout_to_its_cli_candidates() -> None:
    auto = AutoGpuProbe(cache_s=4.0, timeout_s=6.0)
    clis = [p for p in auto._candidates if hasattr(p, "_ttl")]
    assert len(clis) == 3
    assert all(p._ttl == 4.0 and p._timeout == 6.0 for p in clis)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# The event loop
# ---------------------------------------------------------------------------


async def test_metrics_keeps_the_event_loop_free_while_the_probe_waits() -> None:
    def slow_list() -> list[Any]:
        time.sleep(0.6)  # a vendor tool taking its time
        return []

    state = SimpleNamespace(
        metrics=GatewayMetrics(),
        started_at=0.0,
        supervisor=SimpleNamespace(list=lambda: []),
        probe=SimpleNamespace(list_gpus=slow_list),
        accounts=None,
    )
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        await render_metrics(state)
    finally:
        task.cancel()
    # A blocked loop would have managed ~0 ticks across the 0.6 s probe.
    assert ticks >= 10
