"""WP5b: card facts in NVIDIA GFD vocabulary, and card tables keyed by class.

Fake NVML only; nothing here touches a real GPU.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from studioforge.core import gpu as gpu_mod
from studioforge.core import throughput
from studioforge.core.gpu import NvmlGpuProbe
from studioforge.core.metrics import render_metrics
from studioforge.core.placements import hardware_modes
from studioforge.types import GpuInfo

GIB = 1024**3
MIB = 1024**2


class CardNvml:
    """Duck-typed pynvml with per-card facts; ``missing`` names calls to omit."""

    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_MEM = 2

    def __init__(self, cards: list[dict[str, Any]], *, missing: set[str] | None = None) -> None:
        self.cards = cards
        self.missing = missing or set()

    def nvmlInit(self) -> None:
        return None

    def nvmlShutdown(self) -> None:
        return None

    def nvmlDeviceGetCount(self) -> int:
        return len(self.cards)

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def nvmlDeviceGetName(self, handle: int) -> bytes:
        return self.cards[handle]["name"].encode()

    def nvmlDeviceGetMemoryInfo(self, handle: int) -> Any:
        total = self.cards[handle]["total"]
        return SimpleNamespace(total=total, free=total - GIB, used=GIB)

    def nvmlDeviceGetUtilizationRates(self, handle: int) -> Any:
        return SimpleNamespace(gpu=3, memory=1)

    def nvmlDeviceGetTemperature(self, handle: int, sensor: int) -> int:
        return 40

    def nvmlDeviceGetCudaComputeCapability(self, handle: int) -> tuple[int, int]:
        return self.cards[handle]["cc"]

    def nvmlSystemGetDriverVersion(self) -> bytes:
        return b"610.88"

    def nvmlSystemGetCudaDriverVersion_v2(self) -> int:
        return 13000

    def nvmlDeviceGetUUID(self, handle: int) -> bytes:
        if "uuid" in self.missing:
            raise RuntimeError("no uuid")
        return f"GPU-0000-{handle}".encode()

    def nvmlDeviceGetPciInfo(self, handle: int) -> Any:
        if "pci" in self.missing:
            raise RuntimeError("no pci")
        return SimpleNamespace(busId=f"00000000:0{handle + 1}:00.0".encode())

    def nvmlDeviceGetMemoryBusWidth(self, handle: int) -> int:
        if "bandwidth" in self.missing:
            raise RuntimeError("no bus width")
        return self.cards[handle]["bus"]

    def nvmlDeviceGetMaxClockInfo(self, handle: int, clock: int) -> int:
        assert clock == self.NVML_CLOCK_MEM
        return self.cards[handle]["clock"]


RTX5090 = {
    "name": "NVIDIA GeForce RTX 5090",
    "total": 32607 * MIB,
    "cc": (12, 0),
    "bus": 512,
    "clock": 14001,
}
RTX3090 = {
    "name": "NVIDIA GeForce RTX 3090",
    "total": 24576 * MIB,
    "cc": (8, 6),
    "bus": 384,
    "clock": 9751,
}
RTX6000_ADA = {
    "name": "NVIDIA RTX 6000 Ada Generation",
    "total": 49140 * MIB,
    "cc": (8, 9),
    "bus": 384,
    "clock": 9001,
}


def probe_for(
    monkeypatch: pytest.MonkeyPatch, cards: list[dict[str, Any]], **kw: Any
) -> NvmlGpuProbe:
    nvml = CardNvml(cards, **kw)
    monkeypatch.setattr(gpu_mod, "_load_nvml", lambda: nvml)
    return NvmlGpuProbe()


def test_nvml_card_facts_use_gfd_vocabulary(monkeypatch: pytest.MonkeyPatch) -> None:
    gpus = probe_for(monkeypatch, [RTX5090, RTX3090]).list_gpus()
    five, three = gpus
    assert five.uuid == "GPU-0000-0"
    assert five.product == "NVIDIA GeForce RTX 5090"
    assert five.memory_mib == 32607 and five.memory_free_mib == 32607 - 1024
    assert (five.compute_major, five.compute_minor) == (12, 0)
    assert five.arch == "sm_120" and five.family == "blackwell"
    assert five.pci_bus_id == "00000000:01:00.0"
    assert five.driver_version == "610.88"
    assert five.memory_bandwidth_gbps == pytest.approx(1792.1, abs=0.2)
    assert five.class_key == "sm_120-32g"
    assert five.class_label == "RTX 5090"
    assert three.class_key == "sm_86-24g" and three.family == "ampere"
    assert three.memory_bandwidth_gbps == pytest.approx(936.1, abs=0.2)
    # Legacy fields are untouched.
    assert five.name == "NVIDIA GeForce RTX 5090"
    assert five.total_bytes == 32607 * MIB and five.compute_capability == (12, 0)


def test_serialised_card_carries_canonical_and_legacy_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dumped = probe_for(monkeypatch, [RTX5090]).list_gpus()[0].model_dump(mode="json")
    for key in (
        "index",
        "name",
        "total_bytes",
        "free_bytes",
        "used_bytes",
        "compute_capability",
        "utilization_pct",
        "temperature_c",
        "uuid",
        "product",
        "memory_mib",
        "memory_free_mib",
        "compute_major",
        "compute_minor",
        "family",
        "pci_bus_id",
        "driver_version",
    ):
        assert key in dumped, key
    assert set(dumped["studioforge"]) == {"class_key", "memory_bandwidth_gbps", "class_label"}
    assert dumped["studioforge"]["class_key"] == dumped["class_key"] == "sm_120-32g"


def test_every_optional_nvml_call_degrades_to_null(monkeypatch: pytest.MonkeyPatch) -> None:
    card = probe_for(monkeypatch, [RTX5090], missing={"uuid", "pci", "bandwidth"}).list_gpus()[0]
    assert card.uuid is None and card.pci_bus_id is None and card.memory_bandwidth_gbps is None
    # The rest of the card is intact and the class key needs none of them.
    assert card.total_bytes == 32607 * MIB and card.class_key == "sm_120-32g"


def test_a_card_without_compute_capability_has_no_arch_but_a_stable_key() -> None:
    card = GpuInfo(index=0, name="Mystery", total_bytes=16 * GIB, free_bytes=16 * GIB)
    assert card.arch is None and card.family is None
    assert card.compute_major is None
    assert card.class_key == "unknown-16g"
    assert card.studioforge["memory_bandwidth_gbps"] is None


def test_class_key_is_stable_across_reads_and_ignores_the_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = probe_for(monkeypatch, [RTX3090])
    first = probe.list_gpus()[0]
    second = probe.list_gpus()[0]
    assert first.class_key == second.class_key == "sm_86-24g"
    renamed = first.model_copy(update={"name": "Totally Different Name"})
    assert renamed.class_key == first.class_key


def test_amd_arch_comes_from_gfx_target() -> None:
    card = GpuInfo(
        index=0, name="AMD Radeon RX 7900 XTX", total_bytes=24 * GIB, free_bytes=24 * GIB,
        gfx_target="gfx1100", vendor="amd",
    )  # fmt: skip
    assert card.arch == "gfx1100" and card.class_key == "gfx1100-24g"
    assert card.class_label == "Radeon RX 7900 XTX"


def test_a_48gb_workstation_card_is_not_a_5090() -> None:
    card = probe_card(RTX6000_ADA)
    perf = throughput.gpu_perf_of(card)
    assert perf.label != "RTX 5090"
    assert perf.bw_bytes_per_s == pytest.approx(card.memory_bandwidth_gbps * 1e9)
    # Without bandwidth (probe could not read it) a 48 GiB card still is not one.
    blind = card.model_copy(update={"memory_bandwidth_gbps": None})
    unknown = throughput.gpu_perf_of(blind)
    assert unknown.label == blind.class_label
    assert unknown.bw_bytes_per_s == throughput.GPU_PERF_UNKNOWN.bw_bytes_per_s


def probe_card(spec: dict[str, Any]) -> GpuInfo:
    return GpuInfo(
        index=0,
        name=spec["name"],
        total_bytes=spec["total"],
        free_bytes=spec["total"],
        compute_capability=spec["cc"],
        memory_bandwidth_gbps=round(spec["bus"] / 8 * spec["clock"] * 2 / 1000, 1),
    )


def test_named_and_vram_only_cards_keep_their_old_profiles() -> None:
    assert throughput.gpu_perf_for("NVIDIA GeForce RTX 5090").label == "RTX 5090"
    assert throughput.gpu_perf_for("Graphics Device", 32 * GIB).label == "RTX 5090"
    assert throughput.gpu_perf_for("Graphics Device", 24 * GIB).label == "RTX 3090"
    assert throughput.gpu_perf_for("Graphics Device", 48 * GIB) is throughput.GPU_PERF_UNKNOWN
    assert throughput.gpu_perf_for("X", 1, class_key="sm_89-24g").label == "RTX 4090"


def test_measured_bandwidth_beats_the_vram_guess() -> None:
    perf = throughput.gpu_perf_for("Graphics Device", 32 * GIB, bandwidth_gbps=864.0)
    assert perf.bw_bytes_per_s == 864e9 and perf.label == "unknown GPU"
    assert perf.flops_fp16 > throughput.GPU_PERF_UNKNOWN.flops_fp16


def test_gpu_class_label_separates_a_48gb_card() -> None:
    cards = [probe_card(RTX6000_ADA)]
    assert throughput.gpu_class(cards) == "RTX 6000 Ada Generationx1"


# -- placements ---------------------------------------------------------------


def card(index: int, spec: dict[str, Any]) -> GpuInfo:
    return probe_card(spec).model_copy(update={"index": index})


def test_a_48gb_card_is_its_own_class_not_the_5090s() -> None:
    gpus = [card(0, RTX5090), card(1, RTX6000_ADA), card(2, RTX6000_ADA)]
    modes = hardware_modes(gpus)
    keys = [m.key for m in modes]
    assert "dual_sm_89_48g" in keys  # pair of the second class
    assert "dual_5090" not in keys  # only one 5090
    assert [m for m in modes if m.key == "dual_sm_89_48g"][0].devices == (1, 2)
    assert "single_5090" in keys


def test_second_class_pair_and_a_three_card_mode() -> None:
    gpus = [card(0, RTX5090), card(1, RTX5090), card(2, RTX5090)] + [
        card(3, RTX3090),
        card(4, RTX3090),
    ]
    by_key = {m.key: m for m in hardware_modes(gpus)}
    assert by_key["dual_5090"].devices == (0, 1)
    assert by_key["dual_3090"].devices == (3, 4)
    assert by_key["triple_5090"].devices == (0, 1, 2)
    assert by_key["all_gpus"].devices == (0, 1, 2, 3, 4)
    assert "triple_3090" not in by_key


def test_a_lone_card_of_each_class_gets_a_single_each() -> None:
    gpus = [card(0, RTX5090), card(1, RTX3090)]
    assert [m.key for m in hardware_modes(gpus)] == ["all_gpus", "single_5090", "single_3090"]


def test_single_gpu_and_unknown_cards_work() -> None:
    assert [m.key for m in hardware_modes([card(0, RTX3090)])] == ["single_3090"]
    only = card(0, RTX6000_ADA)
    assert [m.key for m in hardware_modes([only])] == ["single_sm_89_48g"]
    assert hardware_modes([only])[0].label == "1x RTX 6000 Ada Generation"


def test_the_reference_rig_modes_are_unchanged() -> None:
    gpus = [card(0, RTX5090), card(1, RTX5090), card(2, RTX3090), card(3, RTX3090)]
    assert [m.key for m in hardware_modes(gpus)] == [
        "dual_5090",
        "dual_3090",
        "all_gpus",
        "single_5090",
    ]


# -- /metrics -------------------------------------------------------------------


async def test_metrics_expose_otel_hw_gpu_gauges() -> None:
    five = card(0, RTX5090).model_copy(update={"uuid": "GPU-abc", "free_bytes": 8 * GIB})
    state = SimpleNamespace(
        metrics=None,
        started_at=0.0,
        supervisor=SimpleNamespace(list=lambda: []),
        probe=SimpleNamespace(list_gpus=lambda: [five]),
        accounts=None,
    )
    text = await render_metrics(state)
    labels = 'hw_id="GPU-abc",hw_name="NVIDIA GeForce RTX 5090",gpu="0"'
    total = five.total_bytes
    assert f"hw_gpu_memory_limit_bytes{{{labels}}} " in text
    assert "# TYPE hw_gpu_memory_usage_bytes gauge" in text
    assert "hw_gpu_memory_utilization_ratio{" in text
    ratio = (total - 8 * GIB) / total
    line = next(ln for ln in text.splitlines() if ln.startswith("hw_gpu_memory_utilization_ratio{"))
    assert float(line.rsplit(" ", 1)[1]) == pytest.approx(ratio, rel=1e-4)
    # The older gauges stay.
    assert "studioforge_gpu_memory_total_bytes{" in text


def test_mcp_compact_gpu_and_gpu_summary_add_the_new_names() -> None:
    from studioforge.api.mgmt_routes import _gpu_summary
    from studioforge.mcp.management import _compact_gpu

    five = card(0, RTX5090).model_copy(update={"uuid": "GPU-x"})
    compact = _compact_gpu(five)
    assert compact["name"] == "NVIDIA GeForce RTX 5090" and compact["total_gib"] > 31
    assert compact["uuid"] == "GPU-x" and compact["studioforge"]["class_key"] == "sm_120-32g"
    summary = _gpu_summary(five)
    assert summary["class_key"] == "sm_120-32g" and summary["memory_mib"] == 32607
