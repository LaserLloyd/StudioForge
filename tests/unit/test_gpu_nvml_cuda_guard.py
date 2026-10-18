"""A CUDA engine with a failing NVML must not be pointed at a Vulkan device list.

Vulkan lists the very same NVIDIA cards, so the auto-selector used to fall
through to it when NVML failed (boot race, driver update) and the CUDA build
was then launched with ``--device Vulkan0``. With a CUDA engine active the
probe now stops at NVML, reports no GPUs, and says why and how to fix it.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.api.app import build_state, create_app
from studioforge.config import Config
from studioforge.core import gpu as gpu_mod
from studioforge.core.doctor import DoctorReport, _check_gpu
from studioforge.core.engine import EngineManager
from studioforge.core.gpu import (
    AutoGpuProbe,
    MetalGpuProbe,
    NvmlGpuProbe,
    RocmGpuProbe,
    VulkanGpuProbe,
    reset_probe,
)
from studioforge.types import EngineInfo
from tests.unit.test_gpu_probes import VK_FULL, VK_SUMMARY, FakeTools, fx


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for key in list(os.environ):
        if key.startswith("SF_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SF_GPU_PROBE", "null")
    reset_probe()
    yield
    reset_probe()


def _failing_nvml(monkeypatch: pytest.MonkeyPatch) -> NvmlGpuProbe:
    def boom() -> object:
        raise ImportError("Driver Not Loaded")

    monkeypatch.setattr(gpu_mod, "_load_nvml", boom)
    return NvmlGpuProbe()


class Rig:
    """Failing NVML + a Vulkan runtime that lists a card; records Vulkan runs."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, **auto_kw: Any) -> None:
        self.vulkan_tools = FakeTools(
            {
                VK_SUMMARY: fx("vulkaninfo_summary_mixed.txt"),
                VK_FULL: fx("vulkaninfo_full_mixed.txt"),
            }
        )
        none = FakeTools({})
        kw = {"ttl_s": 0.0}
        self.auto = AutoGpuProbe(
            [
                _failing_nvml(monkeypatch),
                RocmGpuProbe(runner=none.runner, which=none.which, **kw),
                VulkanGpuProbe(
                    runner=self.vulkan_tools.runner, which=self.vulkan_tools.which, **kw
                ),
                MetalGpuProbe(system=lambda: "Linux", runner=none.runner, which=none.which, **kw),
            ],
            **auto_kw,
        )


def test_cuda_engine_with_failing_nvml_does_not_fall_through_to_vulkan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = Rig(monkeypatch, engine_backend=lambda: "cuda")
    assert rig.auto.list_gpus() == []
    assert rig.auto.backend == "null"  # never "vulkan": llama_device_name stays out of it
    assert rig.auto.available() is False
    assert rig.vulkan_tools.calls == []  # vulkaninfo was not even run

    reason = rig.auto.blocked_reason or ""
    assert "NVML unavailable" in reason and "Driver Not Loaded" in reason
    assert "CUDA build" in reason and "nvidia-smi" in reason  # the fix text

    info = rig.auto.diagnostics()
    assert info["selected"] is None and info["blocked"] == reason
    assert [a["backend"] for a in info["attempts"]] == ["cuda"]  # nothing else was tried


@pytest.mark.parametrize("engine", ["vulkan", "rocm", None, ""])
def test_other_engines_keep_the_fall_through(monkeypatch: pytest.MonkeyPatch, engine: Any) -> None:
    rig = Rig(monkeypatch, engine_backend=lambda: engine)
    assert rig.auto.backend == "vulkan"
    assert len(rig.auto.list_gpus()) == 1
    assert rig.auto.blocked_reason is None
    assert rig.auto.diagnostics()["blocked"] is None


def test_no_engine_binding_keeps_the_fall_through(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Rig(monkeypatch).auto.backend == "vulkan"


def test_an_engine_inventory_that_raises_does_not_break_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken() -> str:
        raise OSError("engines dir unreadable")

    assert Rig(monkeypatch, engine_backend=broken).auto.backend == "vulkan"


def test_shutdown_forgets_the_block_so_a_changed_engine_is_re_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = ["cuda"]
    rig = Rig(monkeypatch, engine_backend=lambda: engine[0])
    assert rig.auto.backend == "null" and rig.auto.blocked_reason
    engine[0] = "vulkan"
    assert rig.auto.backend == "null"  # sticky, like every other resolution
    rig.auto.shutdown()
    assert rig.auto.blocked_reason is None
    assert rig.auto.backend == "vulkan"


def _manager(tmp_path: Path, probe: AutoGpuProbe) -> EngineManager:
    config = Config(data_dir=tmp_path / "data")
    config.ensure_dirs()
    return EngineManager(config, probe=probe)


@pytest.mark.parametrize(("backend", "variant"), [("cuda", "cuda-13.3"), ("", "cuda-12.4")])
def test_the_engine_manager_tells_the_probe_which_backend_is_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend: str, variant: str
) -> None:
    rig = Rig(monkeypatch)
    manager = _manager(tmp_path, rig.auto)
    info = EngineInfo(
        tag="b10425",
        path=tmp_path,
        server_binary=tmp_path / "llama-server",
        variant=variant,
        backend=backend,
    )
    monkeypatch.setattr(manager, "active", lambda: info)
    assert manager.active_backend() == "cuda"
    assert rig.auto.list_gpus() == []
    assert "NVML unavailable" in (rig.auto.blocked_reason or "")


def test_no_installed_engine_means_no_block(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rig = Rig(monkeypatch)
    manager = _manager(tmp_path, rig.auto)
    assert manager.active_backend() is None
    assert rig.auto.backend == "vulkan"  # a box with no engine yet installs the Vulkan one


def test_health_reports_nvml_unavailable_and_the_fix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = Config(
        data_dir=tmp_path / "data",
        server={"host": "127.0.0.1", "port": 1234},
        models={"dir": tmp_path / "models"},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
    )
    state = build_state(config)
    state.engine_status = {"ok": True, "tag": "b10425", "variant": "cuda-13.3"}
    rig = Rig(monkeypatch, engine_backend=lambda: "cuda")
    state.planner.probe = rig.auto
    app = create_app(config, state=state, start_background=False)
    with TestClient(app) as client:
        end = time.monotonic() + 30
        while not client.app.state.boot.ready and time.monotonic() < end:  # type: ignore[attr-defined]
            time.sleep(0.02)
        body = client.get("/health").json()
    assert body["can_serve"] is False
    assert body["gpu_count"] == 0
    assert "NVML unavailable" in body["cannot_serve_reason"]
    assert "nvidia-smi" in body["cannot_serve_reason"]
    assert rig.vulkan_tools.calls == []


def test_doctor_names_nvml_not_a_generic_missing_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = Rig(monkeypatch, engine_backend=lambda: "cuda")
    report = DoctorReport()
    _check_gpu(report, rig.auto)
    (finding,) = report.findings
    assert finding.check_id == "gpu.nvml_unavailable"
    assert "NVML unavailable" in finding.message
