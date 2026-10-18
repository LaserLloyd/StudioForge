"""ROCm / Vulkan / Metal probes and the auto-selector, driven by a fake runner.

No process is spawned and no GPU is touched: every probe takes an injected
``runner`` (argv, timeout -> (returncode, stdout)) and ``which``. The tool
output comes from the SYNTHETIC fixtures in ``tests/unit/fixtures/hw/``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from studioforge.core import gpu as gpu_mod
from studioforge.core.gpu import (
    AutoGpuProbe,
    FakeGpuProbe,
    MetalGpuProbe,
    NvmlGpuProbe,
    RocmGpuProbe,
    VulkanGpuProbe,
    get_probe,
    llama_device_name,
    probe_diagnostics,
    reset_probe,
    run_command,
)
from studioforge.errors import ConfigError
from studioforge.types import GpuInfo

FIXTURES = Path(__file__).parent / "fixtures" / "hw"
MIB = 1024**2
GIB = 1024**3


def fx(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class FakeTools:
    """Maps an argv (as a tuple) to (returncode, stdout) and records every call."""

    def __init__(self, outputs: dict[tuple[str, ...], tuple[int, str] | str]) -> None:
        self.outputs = {k: (v if isinstance(v, tuple) else (0, v)) for k, v in outputs.items()}
        self.calls: list[tuple[str, ...]] = []
        self.installed = {argv[0] for argv in outputs}

    def runner(self, argv: Sequence[str], timeout: float) -> tuple[int, str]:
        key = tuple(argv)
        self.calls.append(key)
        return self.outputs.get(key, (1, ""))

    def which(self, name: str) -> str | None:
        return f"/usr/bin/{name}" if name in self.installed else None

    def kwargs(self) -> dict[str, object]:
        return {"runner": self.runner, "which": self.which, "ttl_s": 0.0}


AMD_SMI_STATIC = ("amd-smi", "static", "--json")
AMD_SMI_METRIC = ("amd-smi", "metric", "--json")
ROCM_SMI = ("rocm-smi", "--showmeminfo", "vram", "--showproductname", "--json")
ROCMINFO = ("rocminfo",)
VK_SUMMARY = ("vulkaninfo", "--summary")
VK_FULL = ("vulkaninfo",)
VK_JSON = ("vulkaninfo", "--json")
SYSPROF = ("system_profiler", "SPDisplaysDataType", "-json")


# ---------------------------------------------------------------------------
# ROCm
# ---------------------------------------------------------------------------


def test_rocm_probe_prefers_amd_smi() -> None:
    tools = FakeTools(
        {
            AMD_SMI_STATIC: fx("amd_smi_static_discrete.json"),
            AMD_SMI_METRIC: fx("amd_smi_metric_discrete.json"),
            ROCM_SMI: fx("rocm_smi_2gpu.json"),
        }
    )
    probe = RocmGpuProbe(**tools.kwargs())
    assert probe.backend == "rocm"
    (gpu,) = probe.list_gpus()
    assert gpu.name == "AMD Radeon RX 7900 XTX" and gpu.gfx_target == "gfx1100"
    assert probe.source == "amd-smi"
    assert probe.driver_version() == "6.8.5"
    assert probe.cuda_driver_version() is None
    assert probe.compute_processes() == []
    assert probe.get_gpu(0) is not None and probe.get_gpu(7) is None
    assert ROCM_SMI not in tools.calls  # the older tool is not needed


def test_rocm_probe_falls_back_to_rocm_smi_and_fills_gfx_from_rocminfo() -> None:
    tools = FakeTools({ROCM_SMI: fx("rocm_smi_2gpu.json"), ROCMINFO: fx("rocminfo_gfx1100.txt")})
    probe = RocmGpuProbe(**tools.kwargs())
    first, second = probe.list_gpus()
    assert probe.source == "rocm-smi"
    assert first.gfx_target == "gfx1100"  # from rocm-smi's own "GFX Version"
    # rocminfo has one GPU agent, so only the first slot could be filled and the
    # second (which rocm-smi named but gave no gfx for) stays honestly unknown.
    assert second.gfx_target is None
    assert probe.driver_version() == "6.8.5"


def test_rocm_probe_rocminfo_only_notes_unknown_free_memory() -> None:
    tools = FakeTools({ROCMINFO: fx("rocminfo_gfx1151.txt")})
    probe = RocmGpuProbe(**tools.kwargs())
    (gpu,) = probe.list_gpus()
    assert probe.source == "rocminfo"
    assert gpu.gfx_target == "gfx1151"
    assert gpu.free_bytes == gpu.total_bytes
    assert any("no live counters" in n for n in probe.notes)


def test_rocm_probe_strix_halo_unified_memory_through_amd_smi() -> None:
    tools = FakeTools(
        {
            AMD_SMI_STATIC: fx("amd_smi_static_strix_halo.json"),
            AMD_SMI_METRIC: fx("amd_smi_metric_strix_halo.json"),
        }
    )
    (gpu,) = RocmGpuProbe(**tools.kwargs()).list_gpus()
    assert gpu.unified_memory is True and gpu.gfx_target == "gfx1151"
    assert gpu.free_bytes == 97692 * MIB


def test_rocm_probe_without_metric_says_so() -> None:
    tools = FakeTools({AMD_SMI_STATIC: fx("amd_smi_static_discrete.json")})
    probe = RocmGpuProbe(**tools.kwargs())
    (gpu,) = probe.list_gpus()
    assert gpu.free_bytes == gpu.total_bytes
    assert any("metric unavailable" in n for n in probe.notes)


def test_rocm_probe_reports_why_nothing_was_found() -> None:
    absent = RocmGpuProbe(**FakeTools({}).kwargs())
    assert absent.available() is False and absent.list_gpus() == []
    assert "none of amd-smi, rocm-smi, rocminfo is installed" in (absent.failure_reason or "")

    present = RocmGpuProbe(**FakeTools({ROCM_SMI: fx("rocm_smi_empty.json")}).kwargs())
    assert present.available() is False
    assert "rocm-smi found but listed no devices" in (present.failure_reason or "")


def test_rocm_probe_caches_for_ttl_and_refreshes_after() -> None:
    now = [100.0]
    tools = FakeTools({ROCM_SMI: fx("rocm_smi_2gpu.json")})
    probe = RocmGpuProbe(runner=tools.runner, which=tools.which, ttl_s=5.0, clock=lambda: now[0])
    probe.list_gpus()
    probe.list_gpus()
    assert tools.calls.count(ROCM_SMI) == 1
    now[0] += 6.0
    probe.list_gpus()
    assert tools.calls.count(ROCM_SMI) == 2
    probe.shutdown()
    probe.list_gpus()
    assert tools.calls.count(ROCM_SMI) == 3


def test_probe_swallows_unexpected_errors() -> None:
    def boom(argv: Sequence[str], timeout: float) -> tuple[int, str]:
        raise RuntimeError("tool exploded")

    probe = RocmGpuProbe(runner=boom, which=lambda name: "/bin/x", ttl_s=0.0)
    assert probe.list_gpus() == []
    assert "tool exploded" in (probe.failure_reason or "")


def test_run_command_never_raises_for_a_missing_binary() -> None:
    assert run_command(["definitely-not-a-real-binary-xyz", "--version"], 2.0) == (127, "")


# ---------------------------------------------------------------------------
# Vulkan
# ---------------------------------------------------------------------------


def test_vulkan_probe_combines_summary_and_full_output() -> None:
    tools = FakeTools(
        {VK_SUMMARY: fx("vulkaninfo_summary_mixed.txt"), VK_FULL: fx("vulkaninfo_full_mixed.txt")}
    )
    probe = VulkanGpuProbe(**tools.kwargs())
    assert probe.backend == "vulkan"
    (gpu,) = probe.list_gpus()
    assert gpu.name.startswith("AMD Radeon RX 7900 XTX")
    assert gpu.total_bytes == 25753026560 and gpu.free_bytes == 25501368320
    assert probe.driver_version() == "Mesa 24.1.2"
    assert any("llvmpipe" in n for n in probe.notes)
    assert probe.source == "vulkaninfo"


def test_vulkan_probe_json_fallback_for_a_single_device() -> None:
    summary = (
        "Devices:\n========\nGPU0:\n\tvendorID = 0x1002\n"
        "\tdeviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\n\tdeviceName = AMD Radeon RX 7900 XTX\n"
    )
    tools = FakeTools({VK_SUMMARY: summary, VK_JSON: fx("vulkaninfo_single.json")})
    (gpu,) = VulkanGpuProbe(**tools.kwargs()).list_gpus()
    assert gpu.total_bytes == 25753026560
    assert gpu.free_bytes == gpu.total_bytes  # no budget in the profile document


def test_vulkan_probe_strix_halo_integrated() -> None:
    tools = FakeTools(
        {
            VK_SUMMARY: fx("vulkaninfo_summary_strix_halo.txt"),
            VK_FULL: fx("vulkaninfo_full_strix_halo.txt"),
        }
    )
    (gpu,) = VulkanGpuProbe(**tools.kwargs()).list_gpus()
    assert gpu.unified_memory is True and gpu.vendor == "amd"
    assert gpu.total_bytes == 96 * GIB


def test_vulkan_probe_failure_reasons() -> None:
    missing = VulkanGpuProbe(**FakeTools({}).kwargs())
    assert missing.available() is False
    assert "vulkaninfo is not installed" in (missing.failure_reason or "")

    only_cpu = (
        "GPU0:\n\tvendorID = 0x10005\n\tdeviceType = PHYSICAL_DEVICE_TYPE_CPU\n"
        "\tdeviceName = llvmpipe\n"
    )
    cpu_only = VulkanGpuProbe(**FakeTools({VK_SUMMARY: only_cpu}).kwargs())
    assert cpu_only.list_gpus() == []
    assert "no usable GPU" in (cpu_only.failure_reason or "")


# ---------------------------------------------------------------------------
# Metal
# ---------------------------------------------------------------------------


def test_metal_probe_apple_silicon() -> None:
    tools = FakeTools({SYSPROF: fx("system_profiler_apple_m2_max.json")})
    probe = MetalGpuProbe(
        system=lambda: "Darwin", memory=lambda: (64 * GIB, 40 * GIB), **tools.kwargs()
    )
    (gpu,) = probe.list_gpus()
    assert probe.backend == "metal"
    assert gpu.unified_memory and gpu.total_bytes == 64 * GIB and gpu.free_bytes == 40 * GIB
    assert any("upper bound" in n for n in probe.notes)


def test_metal_probe_refuses_other_platforms_without_running_anything() -> None:
    tools = FakeTools({SYSPROF: fx("system_profiler_apple_m2_max.json")})
    probe = MetalGpuProbe(system=lambda: "Windows", **tools.kwargs())
    assert probe.list_gpus() == []
    assert "not macOS" in (probe.failure_reason or "")
    assert tools.calls == []


# ---------------------------------------------------------------------------
# Auto
# ---------------------------------------------------------------------------


def _nvml_that_fails(monkeypatch: pytest.MonkeyPatch) -> NvmlGpuProbe:
    def boom() -> object:
        raise ImportError("no NVIDIA driver on this box")

    monkeypatch.setattr(gpu_mod, "_load_nvml", boom)
    return NvmlGpuProbe()


def test_auto_records_why_each_earlier_backend_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    rocm = RocmGpuProbe(**FakeTools({}).kwargs())
    vulkan = VulkanGpuProbe(
        **FakeTools(
            {
                VK_SUMMARY: fx("vulkaninfo_summary_mixed.txt"),
                VK_FULL: fx("vulkaninfo_full_mixed.txt"),
            }
        ).kwargs()
    )
    metal = MetalGpuProbe(system=lambda: "Linux", **FakeTools({}).kwargs())
    auto = AutoGpuProbe([_nvml_that_fails(monkeypatch), rocm, vulkan, metal])

    assert auto.backend == "vulkan"
    assert len(auto.list_gpus()) == 1
    info = auto.diagnostics()
    assert info["selected"] == "vulkan" and info["source"] == "vulkaninfo"
    attempts = {a["backend"]: a for a in info["attempts"]}
    assert list(attempts) == ["cuda", "rocm", "vulkan"]  # metal never tried
    assert attempts["cuda"]["ok"] is False and "NVML unavailable" in attempts["cuda"]["reason"]
    assert "none of amd-smi" in attempts["rocm"]["reason"]
    assert attempts["vulkan"]["ok"] is True and attempts["vulkan"]["gpu_count"] == 1
    assert any("llvmpipe" in n for n in info["notes"])


def test_auto_picks_nvml_first_and_never_consults_the_others() -> None:
    nvidia = FakeGpuProbe([GpuInfo(index=0, name="NVIDIA RTX", total_bytes=1, free_bytes=1)])
    never = RocmGpuProbe(
        runner=lambda a, t: (_ for _ in ()).throw(AssertionError("must not run")),
        which=lambda n: "/x",
    )
    auto = AutoGpuProbe([nvidia, never])
    assert auto.backend == "fake"  # the first probe that lists a GPU, whatever it is called
    assert [g.name for g in auto.list_gpus()] == ["NVIDIA RTX"]
    assert auto.available()


def test_auto_with_nothing_available_is_an_empty_null_probe() -> None:
    auto = AutoGpuProbe([FakeGpuProbe([]), FakeGpuProbe([])])
    assert auto.backend == "null"
    assert auto.available() is False and auto.list_gpus() == []
    assert auto.get_gpu(0) is None and auto.driver_version() is None
    assert auto.cuda_driver_version() is None and auto.compute_processes() == []
    assert auto.diagnostics()["selected"] is None
    assert len(auto.diagnostics()["attempts"]) == 2


def test_auto_resolves_again_after_shutdown() -> None:
    first = FakeGpuProbe([])
    second = FakeGpuProbe([])
    auto = AutoGpuProbe([first, second])
    assert auto.backend == "null"
    second._gpus[0] = GpuInfo(index=0, name="late GPU", total_bytes=1, free_bytes=1)
    assert auto.backend == "null"  # sticky until shutdown
    auto.shutdown()
    assert auto.backend == "fake"


def test_auto_default_candidates_are_nvml_rocm_vulkan_metal() -> None:
    auto = AutoGpuProbe()
    assert [type(p).__name__ for p in auto._candidates] == [
        "NvmlGpuProbe",
        "RocmGpuProbe",
        "VulkanGpuProbe",
        "MetalGpuProbe",
    ]


def test_probe_diagnostics_for_a_plain_probe() -> None:
    info = probe_diagnostics(FakeGpuProbe([]))
    assert info["selected"] == "fake" and info["attempts"] == []


# ---------------------------------------------------------------------------
# get_probe choices + device names
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_probe(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.delenv("SF_GPU_PROBE", raising=False)
    reset_probe()
    yield
    reset_probe()


@pytest.mark.parametrize(
    ("choice", "cls"),
    [
        ("auto", AutoGpuProbe),
        ("nvml", NvmlGpuProbe),
        ("cuda", NvmlGpuProbe),
        ("rocm", RocmGpuProbe),
        ("vulkan", VulkanGpuProbe),
        ("metal", MetalGpuProbe),
    ],
)
def test_get_probe_builds_the_requested_family(choice: str, cls: type) -> None:
    assert type(get_probe(force=choice)) is cls


def test_get_probe_default_is_auto() -> None:
    assert isinstance(get_probe(), AutoGpuProbe)


def test_get_probe_caches_per_family() -> None:
    first = get_probe(force="rocm")
    assert get_probe() is first
    assert get_probe(force="rocm") is first
    other = get_probe(force="vulkan")
    assert other is not first and isinstance(other, VulkanGpuProbe)
    # cuda and nvml are one family
    nv = get_probe(force="nvml")
    assert get_probe(force="cuda") is nv


def test_get_probe_env_selects_a_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SF_GPU_PROBE", "vulkan")
    assert isinstance(get_probe(), VulkanGpuProbe)


def test_get_probe_rejects_unknown_choice() -> None:
    with pytest.raises(ConfigError):
        get_probe(force="directml")


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("cuda", "CUDA2"),
        ("rocm", "ROCm2"),
        ("vulkan", "Vulkan2"),
        ("metal", "MTL2"),
        ("fake", "CUDA2"),
        (None, "CUDA2"),
    ],
)
def test_llama_device_name(backend: str | None, expected: str) -> None:
    assert llama_device_name(backend, 2) == expected
