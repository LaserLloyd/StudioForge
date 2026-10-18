"""Pure parsers for vendor GPU tools (ROCm, Vulkan, macOS).

Everything here takes the *text* a tool printed and returns data; nothing
spawns a process or touches the machine, so each parser is driven by fixture
files in ``tests/unit/fixtures/hw/``. The probe classes in
:mod:`studioforge.core.gpu` own the subprocess side.

Formats parsed (all from the tools' documented / commonly observed output;
the fixtures are hand-written SYNTHETIC samples, not captures -- see
``tests/unit/fixtures/hw/README.md``):

* ``rocm-smi --showmeminfo vram --showproductname --json``
* ``amd-smi static --json`` and ``amd-smi metric --json``
* ``rocminfo`` (agent blocks: marketing name, ``gfx`` target, memory pools)
* ``vulkaninfo --summary``, full ``vulkaninfo`` text and ``vulkaninfo --json``
* ``system_profiler SPDisplaysDataType -json``

Every parser is tolerant by construction: a field it cannot read degrades to
``None``/``0`` for that field, an unreadable document yields ``[]``. None of
them raises on malformed input.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from studioforge.types import GpuInfo

# PCI vendor ids -> a short vendor name. Used by the Vulkan and Metal parsers.
VENDOR_BY_PCI_ID: dict[int, str] = {
    0x10DE: "nvidia",
    0x1002: "amd",
    0x1022: "amd",
    0x8086: "intel",
    0x106B: "apple",
}

_GFX_RE = re.compile(r"gfx[0-9a-f]{3,5}", re.IGNORECASE)
_UNITS = {
    "b": 1,
    "kb": 1024,
    "kib": 1024,
    "mb": 1024**2,
    "mib": 1024**2,
    "gb": 1024**3,
    "gib": 1024**3,
    "tb": 1024**4,
    "tib": 1024**4,
}
#: ``vram.type`` values amd-smi reports for memory that is system RAM rather
#: than board memory. Discrete GPUs report GDDR*/HBM*; an APU's "VRAM" is a
#: carve-out of DDR/LPDDR, which is what makes it unified memory.
_SYSTEM_MEMORY_TYPES = ("uma", "unified", "ddr", "lpddr")


def to_bytes(value: Any, unit: str | None = None) -> int:
    """``value`` (+ optional unit such as ``MB``/``KiB``) as bytes; 0 when unreadable.

    Accepts ints, numeric strings, ``"8 GB"``-style strings and the
    ``{"value": 24560, "unit": "MB"}`` objects amd-smi emits. Binary
    multipliers throughout: the tools say ``MB`` and mean MiB.
    """
    if isinstance(value, dict):
        return to_bytes(value.get("value"), value.get("unit") if unit is None else unit)
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int | float):
        number = float(value)
    else:
        match = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]*)", str(value))
        if match is None:
            return 0
        number = float(match.group(1))
        unit = unit or (match.group(2) or None)
    multiplier = _UNITS.get((unit or "b").strip().lower())
    if multiplier is None:
        return 0
    return int(number * multiplier)


def parse_int(value: Any) -> int | None:
    """``0x1002`` / ``4098`` / ``"4098"`` -> int; ``None`` when not a number."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    try:
        return int(text, 16) if text.lower().startswith("0x") else int(float(text))
    except ValueError:
        return None


def is_system_memory_type(vram_type: str | None) -> bool:
    """Whether an amd-smi ``vram.type`` names system RAM (unified memory)."""
    return bool(vram_type) and str(vram_type).strip().lower().startswith(_SYSTEM_MEMORY_TYPES)


def find_gfx(text: str | None) -> str | None:
    """The first ``gfxNNNN`` token in ``text`` (lower-cased), else ``None``."""
    if not text:
        return None
    match = _GFX_RE.search(str(text))
    return match.group(0).lower() if match else None


def _loads(text: str) -> Any:
    """``json.loads`` that tolerates leading log noise before the document."""
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        pass
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return None
    try:
        return json.loads(text[min(starts) :])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# rocm-smi
# ---------------------------------------------------------------------------


@dataclass
class RocmSmiResult:
    gpus: list[GpuInfo] = field(default_factory=list)
    driver: str | None = None


def _ci_get(card: dict[str, Any], *names: str) -> Any:
    """Case-insensitive key lookup: rocm-smi changed 'Card series' -> 'Card Series'."""
    lowered = {str(k).lower(): v for k, v in card.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def parse_rocm_smi_json(text: str) -> RocmSmiResult:
    """``rocm-smi --showmeminfo vram --showproductname --json``.

    Output is ``{"card0": {...}, "card1": {...}, "system": {...}}`` with every
    value a string. Memory keys are ``VRAM Total Memory (B)`` and
    ``VRAM Total Used Memory (B)``. ``GFX Version`` is present only when the
    caller also asked for hardware info; otherwise the gfx target is left
    ``None`` for the probe to fill from ``rocminfo``.
    """
    data = _loads(text)
    result = RocmSmiResult()
    if not isinstance(data, dict):
        return result
    system = data.get("system")
    if isinstance(system, dict):
        driver = _ci_get(system, "Driver version", "Driver Version")
        result.driver = str(driver) if driver else None
    cards: list[tuple[int, dict[str, Any]]] = []
    for key, card in data.items():
        match = re.fullmatch(r"card(\d+)", str(key))
        if match and isinstance(card, dict):
            cards.append((int(match.group(1)), card))
    for ordinal, (card_index, card) in enumerate(sorted(cards, key=lambda c: c[0])):
        total = to_bytes(_ci_get(card, "VRAM Total Memory (B)"))
        used = to_bytes(_ci_get(card, "VRAM Total Used Memory (B)"))
        name = _ci_get(card, "Card Series", "Card Model", "Card SKU") or f"AMD GPU {card_index}"
        vendor_text = str(_ci_get(card, "Card Vendor") or "")
        result.gpus.append(
            GpuInfo(
                index=ordinal,
                name=str(name),
                total_bytes=total,
                free_bytes=max(0, total - used),
                used_bytes=min(used, total) if total else used,
                backend="rocm",
                vendor="amd" if ("amd" in vendor_text.lower() or not vendor_text) else None,
                gfx_target=find_gfx(str(_ci_get(card, "GFX Version") or "")),
            )
        )
    return result


# ---------------------------------------------------------------------------
# amd-smi
# ---------------------------------------------------------------------------


@dataclass
class AmdSmiStatic:
    index: int
    name: str
    gfx_target: str | None = None
    vendor_id: int | None = None
    vram_total_bytes: int = 0
    vram_type: str | None = None
    driver: str | None = None


def _amd_smi_records(text: str) -> list[dict[str, Any]]:
    data = _loads(text)
    if isinstance(data, dict):
        for key in ("gpu_data", "gpus", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data] if "gpu" in data else []
    if not isinstance(data, list):
        return []
    return [r for r in data if isinstance(r, dict)]


def parse_amd_smi_static(text: str) -> list[AmdSmiStatic]:
    """``amd-smi static --json`` (a list, or ``{"gpu_data": [...]}`` on newer ROCm)."""
    out: list[AmdSmiStatic] = []
    for position, record in enumerate(_amd_smi_records(text)):
        asic = record.get("asic") if isinstance(record.get("asic"), dict) else {}
        vram = record.get("vram") if isinstance(record.get("vram"), dict) else {}
        driver = record.get("driver") if isinstance(record.get("driver"), dict) else {}
        index = parse_int(record.get("gpu"))
        name = asic.get("market_name") or asic.get("asic_name") or "AMD GPU"
        out.append(
            AmdSmiStatic(
                index=position if index is None else index,
                name=str(name),
                gfx_target=find_gfx(str(asic.get("target_graphics_version") or "")),
                vendor_id=parse_int(asic.get("vendor_id")),
                vram_total_bytes=to_bytes(vram.get("size")),
                vram_type=str(vram["type"]) if vram.get("type") else None,
                driver=str(driver["version"]) if driver.get("version") else None,
            )
        )
    return out


@dataclass
class AmdSmiMetric:
    total_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0
    utilization_pct: float | None = None
    temperature_c: float | None = None


def _number(node: Any) -> float | None:
    raw = node.get("value") if isinstance(node, dict) else node
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None  # amd-smi prints "N/A"


def parse_amd_smi_metric(text: str) -> dict[int, AmdSmiMetric]:
    """``amd-smi metric --json`` -> ``{gpu index: memory/util/temperature}``."""
    out: dict[int, AmdSmiMetric] = {}
    for position, record in enumerate(_amd_smi_records(text)):
        index = parse_int(record.get("gpu"))
        mem = record.get("mem_usage") if isinstance(record.get("mem_usage"), dict) else {}
        total = to_bytes(mem.get("total_vram"))
        used = to_bytes(mem.get("used_vram"))
        free = to_bytes(mem.get("free_vram")) or max(0, total - used)
        usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
        temp = record.get("temperature") if isinstance(record.get("temperature"), dict) else {}
        out[position if index is None else index] = AmdSmiMetric(
            total_bytes=total,
            used_bytes=used,
            free_bytes=free,
            utilization_pct=_number(usage.get("gfx_activity")),
            temperature_c=_number(temp.get("edge")),
        )
    return out


def merge_amd_smi(
    static: list[AmdSmiStatic], metric: dict[int, AmdSmiMetric]
) -> tuple[list[GpuInfo], str | None]:
    """Join static identity with live memory into :class:`GpuInfo` records.

    Total comes from the live ``total_vram`` when present (it is what the
    driver will actually allow), else from the static size. ``unified_memory``
    is set from the VRAM type (see :data:`_SYSTEM_MEMORY_TYPES`).
    """
    gpus: list[GpuInfo] = []
    driver: str | None = None
    for ordinal, entry in enumerate(sorted(static, key=lambda s: s.index)):
        live = metric.get(entry.index)
        total = (live.total_bytes if live and live.total_bytes else 0) or entry.vram_total_bytes
        used = live.used_bytes if live else 0
        free = live.free_bytes if live and live.free_bytes else max(0, total - used)
        gpus.append(
            GpuInfo(
                index=ordinal,
                name=entry.name,
                total_bytes=total,
                free_bytes=min(free, total) if total else free,
                used_bytes=used,
                utilization_pct=live.utilization_pct if live else None,
                temperature_c=live.temperature_c if live else None,
                backend="rocm",
                vendor=VENDOR_BY_PCI_ID.get(entry.vendor_id or 0, "amd"),
                gfx_target=entry.gfx_target,
                unified_memory=is_system_memory_type(entry.vram_type),
            )
        )
        driver = driver or entry.driver
    return gpus, driver


# ---------------------------------------------------------------------------
# rocminfo
# ---------------------------------------------------------------------------


@dataclass
class RocmAgent:
    name: str  # the gfx target for a GPU agent
    marketing_name: str
    gfx_target: str | None
    pool_bytes: int  # the coarse-grained GLOBAL pool: device-allocatable memory


def parse_rocminfo(text: str) -> list[RocmAgent]:
    """GPU agents out of ``rocminfo``, in agent order (= HIP device order).

    CPU agents are skipped. ``pool_bytes`` is the first GLOBAL
    ``COARSE GRAINED`` pool (what ``hipMalloc`` draws from); rocminfo has no
    live counters, so it can say how big the memory is but not how much is
    free.
    """
    agents: list[RocmAgent] = []
    for block in re.split(r"(?m)^\s*Agent\s+\d+\s*$", text)[1:]:
        fields: dict[str, str] = {}
        pool_bytes = 0
        segment_ok = False
        for raw in block.splitlines():
            line = raw.strip()
            if line.startswith("Segment:"):
                segment_ok = "GLOBAL" in line and "COARSE GRAINED" in line
                continue
            if line.startswith("Size:") and segment_ok and not pool_bytes:
                match = re.match(r"Size:\s*(\d+)\s*\(0x[0-9a-fA-F]+\)\s*([A-Za-z]*)", line)
                if match:
                    pool_bytes = to_bytes(match.group(1), match.group(2) or "KB")
                segment_ok = False
                continue
            key, sep, value = line.partition(":")
            if sep and key not in fields:
                fields[key.strip()] = value.strip()
        if fields.get("Device Type", "").upper() != "GPU":
            continue
        name = fields.get("Name", "")
        agents.append(
            RocmAgent(
                name=name,
                marketing_name=fields.get("Marketing Name", "") or name,
                gfx_target=find_gfx(name) or find_gfx(block),
                pool_bytes=pool_bytes,
            )
        )
    return agents


def gpus_from_rocminfo(agents: list[RocmAgent]) -> list[GpuInfo]:
    """Identification-only fallback: free memory is unknown, so it equals total."""
    return [
        GpuInfo(
            index=i,
            name=a.marketing_name,
            total_bytes=a.pool_bytes,
            free_bytes=a.pool_bytes,
            backend="rocm",
            vendor="amd",
            gfx_target=a.gfx_target,
        )
        for i, a in enumerate(agents)
    ]


# ---------------------------------------------------------------------------
# Vulkan
# ---------------------------------------------------------------------------


@dataclass
class VulkanHeap:
    size_bytes: int
    budget_bytes: int = 0
    device_local: bool = False


@dataclass
class VulkanDevice:
    index: int
    name: str
    vendor_id: int | None = None
    device_type: str = ""
    driver_name: str = ""
    driver_info: str = ""
    api_version: str = ""
    heaps: list[VulkanHeap] = field(default_factory=list)

    @property
    def is_cpu(self) -> bool:
        kind = self.device_type.upper()
        return kind.endswith("_CPU") or kind == "CPU"

    @property
    def is_integrated(self) -> bool:
        return "INTEGRATED" in self.device_type.upper()

    @property
    def local_heap(self) -> VulkanHeap | None:
        """The largest DEVICE_LOCAL heap (the one weights would live in)."""
        local = [h for h in self.heaps if h.device_local and h.size_bytes > 0]
        return max(local, key=lambda h: h.size_bytes) if local else None


_GPU_HEADER = re.compile(r"(?m)^GPU(\d+):\s*$")
_KV = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")
_HEAP_START = re.compile(r"^\s*memoryHeaps\[(\d+)\]:\s*$")
_HEAP_END_PREFIXES = ("memoryType", "VkPhysical", "Format", "Device Extensions")
_HEAP_SIZE = re.compile(r"^\s*(size|budget|usage)\s*=\s*(\d+)")


def parse_vulkaninfo_text(text: str) -> list[VulkanDevice]:
    """``vulkaninfo --summary`` and/or full ``vulkaninfo`` text output.

    The summary lists ``GPUn:`` blocks of ``key = value`` lines (device name,
    vendor id, type, driver). The full output adds, inside each GPU section,
    ``VkPhysicalDeviceMemoryProperties`` with ``memoryHeaps[i]`` ``size`` /
    ``budget`` / ``usage`` and a ``MEMORY_HEAP_DEVICE_LOCAL_BIT`` flag. Both
    are parsed by the same walker; a section without heaps simply has none.
    Sections with the same ``GPUn`` number are merged, so the summary and the
    full output may be concatenated.
    """
    devices: dict[int, VulkanDevice] = {}
    matches = list(_GPU_HEADER.finditer(text))
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        index = int(match.group(1))
        device = devices.setdefault(index, VulkanDevice(index=index, name=""))
        heap: dict[str, int] | None = None
        heap_flags_local = False
        heap_open = False

        def close_heap(
            heap_ref: dict[str, int] | None, local: bool, dev: VulkanDevice = device
        ) -> None:
            if heap_ref and heap_ref.get("size"):
                dev.heaps.append(
                    VulkanHeap(
                        size_bytes=heap_ref["size"],
                        budget_bytes=heap_ref.get("budget", 0),
                        device_local=local,
                    )
                )

        for line in text[match.end() : end].splitlines():
            start = _HEAP_START.match(line)
            if start:
                if heap_open:
                    close_heap(heap, heap_flags_local)
                heap, heap_flags_local, heap_open = {}, False, True
                continue
            if heap_open:
                size = _HEAP_SIZE.match(line)
                if size and heap is not None:
                    heap[size.group(1)] = int(size.group(2))
                    continue
                if "MEMORY_HEAP_DEVICE_LOCAL_BIT" in line:
                    heap_flags_local = True
                    continue
                stripped = line.strip()
                if stripped.startswith(_HEAP_END_PREFIXES) or (line and not line[0].isspace()):
                    close_heap(heap, heap_flags_local)
                    heap, heap_open = None, False
                    # fall through: this line may still be a key = value line
            pair = _KV.match(line)
            if pair and not heap_open:
                key, value = pair.group(1), pair.group(2)
                if key == "deviceName" and not device.name:
                    device.name = value
                elif key == "vendorID" and device.vendor_id is None:
                    device.vendor_id = parse_int(value.split()[0])
                elif key == "deviceType" and not device.device_type:
                    device.device_type = value
                elif key == "driverName" and not device.driver_name:
                    device.driver_name = value
                elif key == "driverInfo" and not device.driver_info:
                    device.driver_info = value
                elif key == "apiVersion" and not device.api_version:
                    device.api_version = value.split()[0]
        if heap_open:
            close_heap(heap, heap_flags_local)
    return [devices[i] for i in sorted(devices) if devices[i].name]


def _walk(node: Any):
    yield node
    if isinstance(node, dict):
        for child in node.values():
            yield from _walk(child)
    elif isinstance(node, list):
        for child in node:
            yield from _walk(child)


def _heap_is_device_local(flags: Any) -> bool:
    """``flags`` is a bitmask int (bit 0 = DEVICE_LOCAL) or a list of flag names."""
    if isinstance(flags, list):
        return any("DEVICE_LOCAL" in str(f) for f in flags)
    mask = parse_int(flags)
    return mask is not None and bool(mask & 1)


def parse_vulkaninfo_json(text: str) -> list[VulkanDevice]:
    """``vulkaninfo --json[=N]`` (a Vulkan profile document), searched tolerantly.

    The document nests device properties and memory properties at depths that
    differ between SDK versions, so this finds the first object carrying
    ``deviceName`` and the first carrying ``memoryHeaps`` rather than
    following a fixed path. ``--json`` describes ONE device per run, so the
    result has at most one entry (index 0 unless the caller renumbers).
    """
    data = _loads(text)
    if data is None:
        return []
    props = next((n for n in _walk(data) if isinstance(n, dict) and "deviceName" in n), None)
    if props is None:
        return []
    heaps: list[VulkanHeap] = []
    mem = next((n for n in _walk(data) if isinstance(n, dict) and "memoryHeaps" in n), None)
    if mem is not None and isinstance(mem["memoryHeaps"], list):
        for item in mem["memoryHeaps"]:
            if isinstance(item, dict):
                heaps.append(
                    VulkanHeap(
                        size_bytes=to_bytes(item.get("size")),
                        device_local=_heap_is_device_local(item.get("flags")),
                    )
                )
    return [
        VulkanDevice(
            index=0,
            name=str(props["deviceName"]),
            vendor_id=parse_int(props.get("vendorID")),
            device_type=str(props.get("deviceType", "")),
            driver_name=str(props.get("driverName", "")),
            api_version=str(props.get("apiVersion", "")),
            heaps=heaps,
        )
    ]


def gpus_from_vulkan(devices: list[VulkanDevice]) -> tuple[list[GpuInfo], list[str]]:
    """Real GPUs only (software rasterisers are dropped) + notes on what was skipped.

    Device-local heap = total. Free = the heap's ``budget`` when the driver
    exposes ``VK_EXT_memory_budget`` (capped at size), else the whole heap:
    ``vulkaninfo`` is a fresh process, so its own ``usage`` is always 0 and
    other processes' usage is only visible through the budget.
    """
    gpus: list[GpuInfo] = []
    notes: list[str] = []
    for device in devices:
        if device.is_cpu:
            notes.append(f"skipped '{device.name}': software/CPU Vulkan device (STRICT_GPU)")
            continue
        heap = device.local_heap
        if heap is None:
            notes.append(f"skipped '{device.name}': no DEVICE_LOCAL memory heap reported")
            continue
        free = min(heap.budget_bytes, heap.size_bytes) if heap.budget_bytes else heap.size_bytes
        gpus.append(
            GpuInfo(
                index=len(gpus),
                name=device.name,
                total_bytes=heap.size_bytes,
                free_bytes=free,
                used_bytes=max(0, heap.size_bytes - free),
                backend="vulkan",
                vendor=VENDOR_BY_PCI_ID.get(device.vendor_id or 0),
                unified_memory=device.is_integrated,
            )
        )
    return gpus, notes


# ---------------------------------------------------------------------------
# macOS
# ---------------------------------------------------------------------------


def parse_system_profiler_displays(
    text: str, *, memory_bytes: int = 0, available_bytes: int = 0
) -> list[GpuInfo]:
    """``system_profiler SPDisplaysDataType -json`` -> GPUs.

    Apple silicon entries carry no VRAM figure: the GPU shares system memory,
    so ``unified_memory`` is set and the caller-supplied ``memory_bytes``
    (physical RAM) is the reported pool, ``available_bytes`` the free part.
    An Intel Mac with a discrete AMD card does print ``spdisplays_vram`` (e.g.
    ``"8 GB"``); that figure is used, and free is unknown so it equals total.
    """
    data = _loads(text)
    items = data.get("SPDisplaysDataType") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    gpus: list[GpuInfo] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        name = entry.get("sppci_model") or entry.get("_name") or "GPU"
        vendor_text = str(entry.get("spdisplays_vendor") or "").lower()
        vendor = next((v for v in ("apple", "amd", "nvidia", "intel") if v in vendor_text), None)
        if vendor is None and str(name).lower().startswith("apple"):
            vendor = "apple"
        vram = entry.get("spdisplays_vram") or entry.get("spdisplays_vram_shared")
        vram_bytes = to_bytes(vram)
        unified = (
            vendor == "apple"
            or ("shared" in str(vram).lower())
            or ("spdisplays_vram_shared" in entry and "spdisplays_vram" not in entry)
        )
        if unified and not vram_bytes:
            total = memory_bytes
            free = min(available_bytes, total) if available_bytes else total
        else:
            total = vram_bytes
            free = vram_bytes
        gpus.append(
            GpuInfo(
                index=len(gpus),
                name=str(name),
                total_bytes=total,
                free_bytes=free,
                used_bytes=max(0, total - free),
                backend="metal",
                vendor=vendor,
                unified_memory=unified,
            )
        )
    return gpus
