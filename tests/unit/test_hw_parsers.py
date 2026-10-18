"""Fixture-driven tests for ``core/hw_parsers.py``.

The fixtures under ``tests/unit/fixtures/hw/`` are SYNTHETIC: written by hand
from each tool's documented output format, not captured from hardware (see the
README beside them). These tests pin what the parsers do with those formats
and with malformed input; they cannot prove a real tool prints exactly this.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from studioforge.core.hw_parsers import (
    find_gfx,
    gpus_from_rocminfo,
    gpus_from_vulkan,
    is_system_memory_type,
    merge_amd_smi,
    parse_amd_smi_metric,
    parse_amd_smi_static,
    parse_int,
    parse_rocm_smi_json,
    parse_rocminfo,
    parse_system_profiler_displays,
    parse_vulkaninfo_json,
    parse_vulkaninfo_text,
    to_bytes,
)

FIXTURES = Path(__file__).parent / "fixtures" / "hw"
MIB = 1024**2
GIB = 1024**3


def fx(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# scalar helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (1024, None, 1024),
        ("2048", None, 2048),
        ("8 GB", None, 8 * GIB),
        ("1536 MB", None, 1536 * MIB),
        ({"value": 24560, "unit": "MB"}, None, 24560 * MIB),
        (25149440, "KB", 25149440 * 1024),
        ("N/A", None, 0),
        (None, None, 0),
        (True, None, 0),
        ("12 parsecs", None, 0),
    ],
)
def test_to_bytes(value: object, unit: str | None, expected: int) -> None:
    assert to_bytes(value, unit) == expected


def test_parse_int_and_gfx_and_memory_type() -> None:
    assert parse_int("0x1002") == 0x1002
    assert parse_int(4098) == 4098
    assert parse_int("4098") == 4098
    assert parse_int("nope") is None
    assert parse_int(None) is None
    assert find_gfx("amdgcn-amd-amdhsa--gfx1151:sramecc+:xnack-") == "gfx1151"
    assert find_gfx("gfx90a") == "gfx90a"
    assert find_gfx("AMD Radeon RX 7900 XTX") is None
    assert find_gfx(None) is None
    assert is_system_memory_type("LPDDR5X") and is_system_memory_type("UMA")
    assert not is_system_memory_type("GDDR6") and not is_system_memory_type("HBM3")
    assert not is_system_memory_type(None)


# ---------------------------------------------------------------------------
# rocm-smi
# ---------------------------------------------------------------------------


def test_rocm_smi_two_cards() -> None:
    result = parse_rocm_smi_json(fx("rocm_smi_2gpu.json"))
    assert result.driver == "6.8.5"
    first, second = result.gpus
    assert (first.index, first.name) == (0, "AMD Radeon RX 7900 XTX")
    assert first.total_bytes == 25753026560
    assert first.used_bytes == 1253376000
    assert first.free_bytes == 25753026560 - 1253376000
    assert first.backend == "rocm" and first.vendor == "amd"
    assert first.gfx_target == "gfx1100"
    assert first.unified_memory is False
    assert second.name == "AMD Radeon RX 7800 XT"
    assert second.gfx_target is None  # not in the output: the probe asks rocminfo
    assert second.free_bytes == second.total_bytes == 17163091968


def test_rocm_smi_empty_and_garbage() -> None:
    assert parse_rocm_smi_json(fx("rocm_smi_empty.json")).gpus == []
    assert parse_rocm_smi_json("").gpus == []
    assert parse_rocm_smi_json("not json at all").gpus == []
    assert parse_rocm_smi_json("[1, 2]").gpus == []


def test_rocm_smi_tolerates_log_noise_before_json_and_lowercase_keys() -> None:
    text = (
        "WARNING: kernel too old\n"
        '{"card0": {"card series": "Radeon X", "VRAM Total Memory (B)": "100", '
        '"VRAM Total Used Memory (B)": "40"}}'
    )
    (gpu,) = parse_rocm_smi_json(text).gpus
    assert gpu.name == "Radeon X"
    assert (gpu.total_bytes, gpu.used_bytes, gpu.free_bytes) == (100, 40, 60)


# ---------------------------------------------------------------------------
# amd-smi
# ---------------------------------------------------------------------------


def test_amd_smi_discrete_card() -> None:
    static = parse_amd_smi_static(fx("amd_smi_static_discrete.json"))
    metric = parse_amd_smi_metric(fx("amd_smi_metric_discrete.json"))
    gpus, driver = merge_amd_smi(static, metric)
    (gpu,) = gpus
    assert driver == "6.8.5"
    assert gpu.name == "AMD Radeon RX 7900 XTX"
    assert gpu.gfx_target == "gfx1100"
    assert gpu.total_bytes == 24560 * MIB
    assert gpu.used_bytes == 1195 * MIB
    assert gpu.free_bytes == 23365 * MIB
    assert gpu.utilization_pct == 7.0 and gpu.temperature_c == 41.0
    assert gpu.unified_memory is False  # GDDR6
    assert gpu.vendor == "amd" and gpu.backend == "rocm"


def test_amd_smi_strix_halo_is_unified_memory_and_wrapped_form() -> None:
    static = parse_amd_smi_static(fx("amd_smi_static_strix_halo.json"))
    metric = parse_amd_smi_metric(fx("amd_smi_metric_strix_halo.json"))
    (gpu,), _driver = merge_amd_smi(static, metric)
    assert gpu.gfx_target == "gfx1151"
    assert gpu.unified_memory is True  # LPDDR5X "VRAM"
    # Budget = what the driver reports, no GTT/system-RAM bonus.
    assert gpu.total_bytes == 98304 * MIB
    assert gpu.free_bytes == 97692 * MIB
    # "N/A" readings degrade to None, not an exception.
    assert gpu.utilization_pct is None and gpu.temperature_c is None


def test_amd_smi_without_metric_falls_back_to_static_size() -> None:
    static = parse_amd_smi_static(fx("amd_smi_static_discrete.json"))
    (gpu,), _ = merge_amd_smi(static, {})
    assert gpu.total_bytes == gpu.free_bytes == 24560 * MIB
    assert gpu.used_bytes == 0


def test_amd_smi_garbage() -> None:
    assert parse_amd_smi_static("") == []
    assert parse_amd_smi_static("{}") == []
    assert parse_amd_smi_static("42") == []
    assert parse_amd_smi_metric("[]") == {}


# ---------------------------------------------------------------------------
# rocminfo
# ---------------------------------------------------------------------------


def test_rocminfo_skips_cpu_agent_and_reads_gfx_and_pool() -> None:
    (agent,) = parse_rocminfo(fx("rocminfo_gfx1100.txt"))
    assert agent.name == "gfx1100"
    assert agent.marketing_name == "AMD Radeon RX 7900 XTX"
    assert agent.gfx_target == "gfx1100"
    assert agent.pool_bytes == 25149440 * 1024  # first GLOBAL COARSE GRAINED pool


def test_rocminfo_strix_halo_identification_only() -> None:
    gpus = gpus_from_rocminfo(parse_rocminfo(fx("rocminfo_gfx1151.txt")))
    (gpu,) = gpus
    assert gpu.gfx_target == "gfx1151"
    assert gpu.total_bytes == 100663296 * 1024
    # No live counters in rocminfo: free is reported equal to total.
    assert gpu.free_bytes == gpu.total_bytes


def test_rocminfo_garbage() -> None:
    assert parse_rocminfo("") == []
    assert parse_rocminfo("no agents here") == []


# ---------------------------------------------------------------------------
# Vulkan
# ---------------------------------------------------------------------------


def test_vulkan_summary_lists_devices_without_heaps() -> None:
    amd, cpu = parse_vulkaninfo_text(fx("vulkaninfo_summary_mixed.txt"))
    assert amd.name == "AMD Radeon RX 7900 XTX (RADV NAVI31)"
    assert amd.vendor_id == 0x1002
    assert amd.device_type == "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU"
    assert amd.driver_name == "radv" and amd.driver_info == "Mesa 24.1.2"
    assert amd.heaps == []
    assert cpu.is_cpu and not amd.is_cpu


def test_vulkan_full_output_has_heaps() -> None:
    amd, cpu = parse_vulkaninfo_text(fx("vulkaninfo_full_mixed.txt"))
    assert [h.size_bytes for h in amd.heaps] == [25753026560, 67062644736]
    assert [h.device_local for h in amd.heaps] == [True, False]
    assert amd.heaps[0].budget_bytes == 25501368320
    assert amd.local_heap is amd.heaps[0]
    assert cpu.is_cpu and cpu.heaps[0].size_bytes == 67062644736


def test_vulkan_gpus_drop_software_devices_and_use_budget_as_free() -> None:
    devices = parse_vulkaninfo_text(fx("vulkaninfo_full_mixed.txt"))
    gpus, notes = gpus_from_vulkan(devices)
    (gpu,) = gpus
    assert gpu.backend == "vulkan" and gpu.vendor == "amd"
    assert gpu.total_bytes == 25753026560
    assert gpu.free_bytes == 25501368320
    assert gpu.used_bytes == 25753026560 - 25501368320
    assert gpu.unified_memory is False
    assert gpu.index == 0
    assert any("llvmpipe" in n and "STRICT_GPU" in n for n in notes)


def test_vulkan_integrated_gpu_is_unified_memory() -> None:
    summary = fx("vulkaninfo_summary_strix_halo.txt")
    full = fx("vulkaninfo_full_strix_halo.txt")
    (device,) = parse_vulkaninfo_text(summary + "\n" + full)
    assert device.is_integrated and len(device.heaps) == 2
    (gpu,), _ = gpus_from_vulkan([device])
    assert gpu.unified_memory is True
    # Largest DEVICE_LOCAL heap is the budget; the 15 GiB host heap is ignored.
    assert gpu.total_bytes == 96 * GIB
    assert gpu.free_bytes == 102000000000


def test_vulkan_device_without_local_heap_is_skipped_with_a_note() -> None:
    (device,) = parse_vulkaninfo_text(fx("vulkaninfo_summary_strix_halo.txt"))
    gpus, notes = gpus_from_vulkan([device])
    assert gpus == []
    assert "no DEVICE_LOCAL" in notes[0]


def test_vulkan_json_profile() -> None:
    (device,) = parse_vulkaninfo_json(fx("vulkaninfo_single.json"))
    assert device.name.startswith("AMD Radeon RX 7900 XTX")
    assert device.vendor_id == 4098
    assert [h.device_local for h in device.heaps] == [True, False]
    assert device.local_heap is not None and device.local_heap.size_bytes == 25753026560


def test_vulkan_garbage() -> None:
    assert parse_vulkaninfo_text("") == []
    assert parse_vulkaninfo_text("Vulkan Instance Version: 1.3\nnothing else") == []
    assert parse_vulkaninfo_json("") == []
    assert parse_vulkaninfo_json("{}") == []
    assert parse_vulkaninfo_json("[1,2,3]") == []


# ---------------------------------------------------------------------------
# macOS
# ---------------------------------------------------------------------------


def test_apple_silicon_is_unified_memory_with_ram_as_the_pool() -> None:
    (gpu,) = parse_system_profiler_displays(
        fx("system_profiler_apple_m2_max.json"),
        memory_bytes=64 * GIB,
        available_bytes=41 * GIB,
    )
    assert gpu.name == "Apple M2 Max"
    assert gpu.backend == "metal" and gpu.vendor == "apple"
    assert gpu.unified_memory is True
    assert gpu.total_bytes == 64 * GIB
    assert gpu.free_bytes == 41 * GIB
    assert gpu.used_bytes == 23 * GIB


def test_intel_mac_discrete_card_uses_reported_vram() -> None:
    amd, intel = parse_system_profiler_displays(
        fx("system_profiler_intel_mac_amd.json"), memory_bytes=32 * GIB, available_bytes=10 * GIB
    )
    assert amd.vendor == "amd" and amd.unified_memory is False
    assert amd.total_bytes == 8 * GIB and amd.free_bytes == 8 * GIB
    assert intel.vendor == "intel" and intel.unified_memory is True
    assert intel.total_bytes == 1536 * MIB


def test_system_profiler_garbage() -> None:
    assert parse_system_profiler_displays("") == []
    assert parse_system_profiler_displays("{}") == []
    assert parse_system_profiler_displays('{"SPDisplaysDataType": "x"}') == []
