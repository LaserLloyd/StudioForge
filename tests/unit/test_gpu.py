"""Unit tests for the GPU probe (studioforge.core.gpu).

Everything except the explicitly-marked live test runs without touching real
hardware: NVML is replaced by a fake module object via ``_load_nvml``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from studioforge.core import gpu as gpu_mod
from studioforge.core.gpu import (
    FakeGpuProbe,
    NullGpuProbe,
    NvmlGpuProbe,
    fastest_gpu_order,
    get_probe,
    reset_probe,
    system_ram,
    total_free_vram,
)
from studioforge.errors import ConfigError
from studioforge.types import GpuInfo

MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024


def make_gpu(
    index: int,
    *,
    name: str = "Fake GPU",
    total: int = 24 * GIB,
    free: int | None = None,
    cc: tuple[int, int] | None = (8, 6),
) -> GpuInfo:
    free_bytes = total if free is None else free
    return GpuInfo(
        index=index,
        name=name,
        total_bytes=total,
        free_bytes=free_bytes,
        used_bytes=total - free_bytes,
        compute_capability=cc,
    )


@pytest.fixture(autouse=True)
def _clean_probe_cache() -> Any:
    reset_probe()
    yield
    reset_probe()


# ---------------------------------------------------------------------------
# Fake NVML module used to exercise NvmlGpuProbe without hardware
# ---------------------------------------------------------------------------


class FakeNvml:
    """Duck-typed stand-in for the ``pynvml`` module."""

    NVML_TEMPERATURE_GPU = 0

    def __init__(
        self,
        count: int = 2,
        *,
        cuda_version: int = 13000,
        bytes_strings: bool = True,
    ) -> None:
        self.count = count
        self.cuda_version = cuda_version
        self.bytes_strings = bytes_strings
        self.init_calls = 0
        self.shutdown_calls = 0

    def nvmlInit(self) -> None:
        self.init_calls += 1

    def nvmlShutdown(self) -> None:
        self.shutdown_calls += 1

    def nvmlDeviceGetCount(self) -> int:
        return self.count

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def nvmlDeviceGetName(self, handle: int) -> bytes | str:
        name = f"NVIDIA GeForce RTX 5090 #{handle}"
        return name.encode() if self.bytes_strings else name

    def nvmlDeviceGetMemoryInfo(self, handle: int) -> Any:
        return SimpleNamespace(total=32 * GIB, free=30 * GIB, used=2 * GIB)

    def nvmlDeviceGetUtilizationRates(self, handle: int) -> Any:
        return SimpleNamespace(gpu=17, memory=5)

    def nvmlDeviceGetTemperature(self, handle: int, sensor: int) -> int:
        return 41

    def nvmlDeviceGetCudaComputeCapability(self, handle: int) -> tuple[int, int]:
        return (12, 0)

    def nvmlSystemGetDriverVersion(self) -> bytes | str:
        return b"610.88" if self.bytes_strings else "610.88"

    def nvmlSystemGetCudaDriverVersion_v2(self) -> int:
        return self.cuda_version


@pytest.fixture()
def fake_nvml(monkeypatch: pytest.MonkeyPatch) -> FakeNvml:
    nvml = FakeNvml()
    monkeypatch.setattr(gpu_mod, "_load_nvml", lambda: nvml)
    return nvml


# ---------------------------------------------------------------------------
# FakeGpuProbe behaviour
# ---------------------------------------------------------------------------


class TestFakeGpuProbe:
    def test_listing_sorted_and_complete(self) -> None:
        probe = FakeGpuProbe([make_gpu(2), make_gpu(0), make_gpu(1)])
        gpus = probe.list_gpus()
        assert [g.index for g in gpus] == [0, 1, 2]
        assert probe.backend == "fake"
        assert probe.available() is True

    def test_get_gpu_missing_index_is_none(self) -> None:
        probe = FakeGpuProbe([make_gpu(0)])
        assert probe.get_gpu(0) is not None
        assert probe.get_gpu(5) is None

    def test_set_free_updates_free_and_used(self) -> None:
        probe = FakeGpuProbe([make_gpu(0, total=24 * GIB)])
        probe.set_free(0, 10 * GIB)
        info = probe.get_gpu(0)
        assert info is not None
        assert info.free_bytes == 10 * GIB
        assert info.used_bytes == 14 * GIB

    def test_set_free_rejects_out_of_range(self) -> None:
        probe = FakeGpuProbe([make_gpu(0, total=24 * GIB)])
        with pytest.raises(ValueError):
            probe.set_free(0, 25 * GIB)
        with pytest.raises(KeyError):
            probe.set_free(9, 0)

    def test_consume_simulates_allocation(self) -> None:
        probe = FakeGpuProbe([make_gpu(0, total=24 * GIB)])
        probe.consume(0, 6 * GIB)
        probe.consume(0, 6 * GIB)
        info = probe.get_gpu(0)
        assert info is not None
        assert info.free_bytes == 12 * GIB
        with pytest.raises(ValueError):
            probe.consume(0, 13 * GIB)

    def test_constructor_copies_input(self) -> None:
        original = make_gpu(0, total=24 * GIB)
        probe = FakeGpuProbe([original])
        probe.consume(0, 4 * GIB)
        assert original.free_bytes == 24 * GIB  # caller's object untouched

    def test_total_free_vram(self) -> None:
        probe = FakeGpuProbe(
            [make_gpu(0, total=32 * GIB, free=30 * GIB), make_gpu(1, total=24 * GIB, free=4 * GIB)]
        )
        assert total_free_vram(probe) == 34 * GIB
        assert total_free_vram(NullGpuProbe()) == 0


# ---------------------------------------------------------------------------
# fastest_gpu_order
# ---------------------------------------------------------------------------


class TestFastestGpuOrder:
    def test_reference_rig_ordering(self) -> None:
        gpus = [
            make_gpu(0, name="RTX 5090", total=32607 * MIB, cc=(12, 0)),
            make_gpu(1, name="RTX 5090", total=32607 * MIB, cc=(12, 0)),
            make_gpu(2, name="RTX 3090", total=24576 * MIB, cc=(8, 6)),
            make_gpu(3, name="RTX 3090", total=24576 * MIB, cc=(8, 6)),
        ]
        assert fastest_gpu_order(gpus) == [0, 1, 2, 3]
        # Order of the input must not matter.
        assert fastest_gpu_order(list(reversed(gpus))) == [0, 1, 2, 3]

    def test_cc_beats_vram(self) -> None:
        gpus = [
            make_gpu(0, total=48 * GIB, cc=(8, 6)),
            make_gpu(1, total=16 * GIB, cc=(12, 0)),
        ]
        assert fastest_gpu_order(gpus) == [1, 0]

    def test_vram_breaks_cc_tie_then_index(self) -> None:
        gpus = [
            make_gpu(0, total=16 * GIB, cc=(8, 6)),
            make_gpu(1, total=24 * GIB, cc=(8, 6)),
            make_gpu(2, total=24 * GIB, cc=(8, 6)),  # full tie with 1 -> lower index first
        ]
        assert fastest_gpu_order(gpus) == [1, 2, 0]

    def test_minor_version_matters(self) -> None:
        gpus = [make_gpu(0, cc=(8, 0)), make_gpu(1, cc=(8, 9))]
        assert fastest_gpu_order(gpus) == [1, 0]

    def test_unknown_capability_ranks_last(self) -> None:
        gpus = [make_gpu(0, cc=None), make_gpu(1, cc=(8, 6))]
        assert fastest_gpu_order(gpus) == [1, 0]

    def test_empty(self) -> None:
        assert fastest_gpu_order([]) == []


# ---------------------------------------------------------------------------
# NvmlGpuProbe against the fake NVML module
# ---------------------------------------------------------------------------


class TestNvmlProbeFaked:
    @pytest.mark.parametrize(
        ("packed", "expected"),
        [(13000, (13, 0)), (12040, (12, 4)), (11080, (11, 8))],
    )
    def test_cuda_driver_version_arithmetic(
        self, monkeypatch: pytest.MonkeyPatch, packed: int, expected: tuple[int, int]
    ) -> None:
        nvml = FakeNvml(cuda_version=packed)
        monkeypatch.setattr(gpu_mod, "_load_nvml", lambda: nvml)
        probe = NvmlGpuProbe()
        assert probe.cuda_driver_version() == expected

    def test_list_gpus_fills_all_fields(self, fake_nvml: FakeNvml) -> None:
        probe = NvmlGpuProbe()
        gpus = probe.list_gpus()
        assert len(gpus) == 2
        gpu = gpus[0]
        assert gpu.index == 0
        assert gpu.name == "NVIDIA GeForce RTX 5090 #0"  # bytes decoded to str
        assert gpu.total_bytes == 32 * GIB
        assert gpu.free_bytes == 30 * GIB
        assert gpu.used_bytes == 2 * GIB
        assert gpu.utilization_pct == 17.0
        assert gpu.temperature_c == 41.0
        assert gpu.compute_capability == (12, 0)
        assert probe.driver_version() == "610.88"
        assert fake_nvml.init_calls == 1  # init once, cached

    def test_str_names_handled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        nvml = FakeNvml(bytes_strings=False)
        monkeypatch.setattr(gpu_mod, "_load_nvml", lambda: nvml)
        probe = NvmlGpuProbe()
        assert probe.list_gpus()[0].name == "NVIDIA GeForce RTX 5090 #0"
        assert probe.driver_version() == "610.88"

    def test_get_gpu_bounds(self, fake_nvml: FakeNvml) -> None:
        probe = NvmlGpuProbe()
        assert probe.get_gpu(1) is not None
        assert probe.get_gpu(2) is None
        assert probe.get_gpu(-1) is None

    def test_nvml_missing_degrades_to_null_behaviour(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom() -> Any:
            raise ImportError("no pynvml on this box")

        monkeypatch.setattr(gpu_mod, "_load_nvml", boom)
        probe = NvmlGpuProbe()
        assert probe.available() is False
        assert probe.list_gpus() == []
        assert probe.get_gpu(0) is None
        assert probe.driver_version() is None
        assert probe.cuda_driver_version() is None
        probe.shutdown()  # still idempotent / no raise

    def test_nvml_init_failure_degrades(self, monkeypatch: pytest.MonkeyPatch) -> None:
        nvml = FakeNvml()

        def failing_init() -> None:
            raise RuntimeError("NVML: Driver Not Loaded")

        nvml.nvmlInit = failing_init  # type: ignore[method-assign]
        monkeypatch.setattr(gpu_mod, "_load_nvml", lambda: nvml)
        probe = NvmlGpuProbe()
        assert probe.available() is False
        assert probe.list_gpus() == []

    def test_per_field_degradation_utilization(self, fake_nvml: FakeNvml) -> None:
        def broken(handle: int) -> Any:
            raise RuntimeError("NVML: Not Supported")

        fake_nvml.nvmlDeviceGetUtilizationRates = broken  # type: ignore[method-assign]
        probe = NvmlGpuProbe()
        gpus = probe.list_gpus()
        assert len(gpus) == 2
        assert gpus[0].utilization_pct is None
        # The other fields survived.
        assert gpus[0].total_bytes == 32 * GIB
        assert gpus[0].compute_capability == (12, 0)

    def test_shutdown_idempotent(self, fake_nvml: FakeNvml) -> None:
        probe = NvmlGpuProbe()
        assert probe.available() is True
        probe.shutdown()
        probe.shutdown()
        assert fake_nvml.shutdown_calls == 1
        # Probe re-initialises on next use.
        assert probe.available() is True
        assert fake_nvml.init_calls == 2


# ---------------------------------------------------------------------------
# get_probe factory
# ---------------------------------------------------------------------------


class TestGetProbe:
    def test_force_fake_reads_env_spec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        spec = [
            {
                "index": 0,
                "name": "Fake RTX 5090",
                "total_bytes": 32607 * MIB,
                "free_bytes": 30000 * MIB,
                "compute_capability": [12, 0],
            },
            {
                "index": 1,
                "name": "Fake RTX 3090",
                "total_bytes": 24576 * MIB,
                "compute_capability": [8, 6],
            },
        ]
        monkeypatch.setenv("SF_FAKE_GPUS", json.dumps(spec))
        probe = get_probe(force="fake")
        assert probe.backend == "fake"
        gpus = probe.list_gpus()
        assert len(gpus) == 2
        assert gpus[0].name == "Fake RTX 5090"
        assert gpus[0].free_bytes == 30000 * MIB
        assert gpus[0].compute_capability == (12, 0)
        # free_bytes defaults to total_bytes when omitted.
        assert gpus[1].free_bytes == 24576 * MIB
        assert fastest_gpu_order(gpus) == [0, 1]

    def test_env_var_selects_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SF_GPU_PROBE", "null")
        probe = get_probe()
        assert probe.backend == "null"
        assert probe.available() is False

    def test_cache_and_reset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SF_FAKE_GPUS", "[]")
        first = get_probe(force="null")
        assert get_probe() is first  # cached
        assert get_probe(force="null") is first  # same backend forced -> cached
        second = get_probe(force="fake")  # different backend -> rebuilt
        assert second is not first
        assert second.backend == "fake"
        reset_probe()
        third = get_probe(force="null")
        assert third is not first

    def test_unknown_backend_rejected(self) -> None:
        with pytest.raises(ConfigError):
            get_probe(force="rocm")

    def test_bad_fake_spec_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SF_FAKE_GPUS", "not json")
        with pytest.raises(ConfigError):
            get_probe(force="fake")
        monkeypatch.setenv("SF_FAKE_GPUS", '{"index": 0}')
        with pytest.raises(ConfigError):
            get_probe(force="fake")


# ---------------------------------------------------------------------------
# system_ram
# ---------------------------------------------------------------------------


def test_system_ram_plausible() -> None:
    total, used = system_ram()
    assert total > used > 0
    assert total > 1 * GIB  # any real box has more than 1 GiB


# ---------------------------------------------------------------------------
# Live hardware test (skipped on boxes without NVML/GPUs)
# ---------------------------------------------------------------------------

_live_probe = NvmlGpuProbe()


@pytest.mark.skipif(not _live_probe.available(), reason="no NVML/GPUs on this machine")
class TestLiveHardware:
    def test_reference_rig(self) -> None:
        probe = _live_probe
        gpus = probe.list_gpus()

        print("\nindex  name                          total MiB  free MiB   cc     util  temp")
        for g in gpus:
            print(
                f"{g.index:>5}  {g.name:<28}  {g.total_bytes // MIB:>9}  "
                f"{g.free_bytes // MIB:>8}  {g.cc_str:<5}  "
                f"{g.utilization_pct!s:>4}  {g.temperature_c!s:>4}"
            )
        print(f"driver={probe.driver_version()}  cuda={probe.cuda_driver_version()}")

        assert len(gpus) == 4
        assert [g.index for g in gpus] == [0, 1, 2, 3]
        for g in gpus:
            assert "RTX" in g.name
            assert 0 <= g.free_bytes <= g.total_bytes
            assert g.used_bytes >= 0
        for g in gpus[:2]:  # RTX 5090s
            assert abs(g.total_bytes // MIB - 32607) < 512
            assert g.compute_capability == (12, 0)
        for g in gpus[2:]:  # RTX 3090s
            assert abs(g.total_bytes // MIB - 24576) < 512
            assert g.compute_capability == (8, 6)

        order = fastest_gpu_order(gpus)
        assert order[0] in (0, 1)  # a 5090 first
        assert "5090" in gpus[order[0]].name

        cuda = probe.cuda_driver_version()
        assert cuda is not None
        assert cuda[0] >= 12

        driver = probe.driver_version()
        assert driver is not None and driver[0].isdigit()

        assert total_free_vram(probe) > 0
