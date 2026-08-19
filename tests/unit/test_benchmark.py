"""Unit tests for the model benchmarking subsystem.

No GPU, no engine, no HTTP: the probe, the planner, the manager and
``httpx.AsyncClient`` are all substituted, so these tests pin the benchmarker's
*behaviour* (mode derivation, applicability reporting, state restoration,
metric selection, serialization) rather than any hardware.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from studioforge.core import benchmark as benchmark_module
from studioforge.core.benchmark import (
    Benchmarker,
    BenchmarkJobs,
    BenchmarkMode,
    BenchmarkResult,
    available_modes,
)
from studioforge.db import SCHEMA_VERSION, Database
from studioforge.errors import ModelBusyError
from studioforge.types import (
    GB,
    GgufMeta,
    GpuInfo,
    InstanceInfo,
    LoadPlan,
    LoadRejected,
    ModelRecord,
    ModelSettings,
)

# ---------------------------------------------------------------------------
# Hardware fixtures
# ---------------------------------------------------------------------------


def gpu(index: int, name: str, total_gib: float, cc: tuple[int, int] | None) -> GpuInfo:
    total = int(total_gib * GB)
    return GpuInfo(
        index=index,
        name=name,
        total_bytes=total,
        free_bytes=total,
        used_bytes=0,
        compute_capability=cc,
    )


def reference_rig() -> list[GpuInfo]:
    """The 2x RTX 5090 + 2x RTX 3090 development box."""
    return [
        gpu(0, "NVIDIA GeForce RTX 5090", 32, (12, 0)),
        gpu(1, "NVIDIA GeForce RTX 5090", 32, (12, 0)),
        gpu(2, "NVIDIA GeForce RTX 3090", 24, (8, 6)),
        gpu(3, "NVIDIA GeForce RTX 3090", 24, (8, 6)),
    ]


class StubProbe:
    """Fixed GPU list; the only probe method the benchmarker uses is list_gpus."""

    backend = "fake"

    def __init__(self, gpus: list[GpuInfo]) -> None:
        self._gpus = {g.index: g for g in gpus}

    def available(self) -> bool:
        return bool(self._gpus)

    def list_gpus(self) -> list[GpuInfo]:
        return [self._gpus[i].model_copy(deep=True) for i in sorted(self._gpus)]

    def get_gpu(self, index: int) -> GpuInfo | None:
        found = self._gpus.get(index)
        return found.model_copy(deep=True) if found else None

    def driver_version(self) -> str | None:
        return "610.88"

    def cuda_driver_version(self) -> tuple[int, int] | None:
        return (13, 3)

    def shutdown(self) -> None:
        return None

    def consume(self, index: int, nbytes: int) -> None:
        found = self._gpus[index]
        found.free_bytes = max(0, found.free_bytes - nbytes)
        found.used_bytes = found.total_bytes - found.free_bytes


# ---------------------------------------------------------------------------
# Manager fixtures
# ---------------------------------------------------------------------------


class StubPlanner:
    def __init__(self, probe: StubProbe, verdict: Any = None) -> None:
        self.probe = probe
        self._verdict = verdict
        self.calls: list[list[int]] = []

    def plan_load(self, record: ModelRecord, **kwargs: Any) -> Any:
        devices = list(record.settings.device_override or [])
        self.calls.append(devices)
        if self._verdict is not None:
            outcome = self._verdict(devices)
            if outcome is not None:
                return outcome
        return LoadPlan(model_id=record.id, devices=devices, ctx_size=kwargs.get("ctx_size") or 0)


class StubSupervisor:
    def __init__(self) -> None:
        self.instances: dict[str, InstanceInfo] = {}
        self.request_marks: list[str] = []

    def list(self) -> list[InstanceInfo]:
        return list(self.instances.values())

    def base_url(self, model_id: str) -> str | None:
        return "http://engine.test" if model_id in self.instances else None

    def mark_request_start(self, model_id: str) -> None:
        self.request_marks.append(f"start:{model_id}")

    def mark_request_end(self, model_id: str, *, tokens_per_second: float | None = None) -> None:
        self.request_marks.append(f"end:{model_id}")


class StubManager:
    """Just enough ModelManager for the benchmarker."""

    def __init__(self, record: ModelRecord, probe: StubProbe, *, verdict: Any = None) -> None:
        self.record = record
        self.planner = StubPlanner(probe, verdict)
        self.supervisor = StubSupervisor()
        self.loads: list[tuple[int, ...]] = []
        self.unloads: list[str] = []
        #: device tuples whose load should blow up
        self.fail_devices: set[tuple[int, ...]] = set()
        #: awaited inside load(), so a test can hold a run open
        self.load_gate: asyncio.Event | None = None
        self.load_started: asyncio.Event | None = None

    def _draft_for(self, record: ModelRecord) -> ModelRecord | None:
        return None

    def _adapters_for(self, record: ModelRecord) -> list[tuple[Any, float]]:
        return []

    async def load(
        self, name: str, *, ctx_size: int | None = None, force: bool = False, **_: Any
    ) -> InstanceInfo:
        devices = tuple(self.record.settings.device_override or [])
        self.loads.append(devices)
        if self.load_started is not None:
            self.load_started.set()
        if self.load_gate is not None:
            await self.load_gate.wait()
        if devices in self.fail_devices:
            raise RuntimeError(f"simulated load failure on {list(devices)}")
        info = InstanceInfo(model_id=self.record.id, state="ready", port=18100)
        self.supervisor.instances[self.record.id] = info
        return info

    async def unload(self, name: str) -> bool:
        self.unloads.append(name)
        return self.supervisor.instances.pop(self.record.id, None) is not None


# ---------------------------------------------------------------------------
# Fake streaming engine
# ---------------------------------------------------------------------------


def sse(*chunks: dict[str, Any]) -> list[str]:
    return [f"data: {json.dumps(chunk)}" for chunk in chunks] + ["data: [DONE]"]


def content_chunk(text: str = "hello") -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


def timings_chunk(*, prompt_tps: float, generation_tps: float) -> dict[str, Any]:
    return {
        "choices": [],
        "usage": {"prompt_tokens": 300, "completion_tokens": 64},
        "timings": {
            "prompt_n": 300,
            "prompt_per_second": prompt_tps,
            "predicted_n": 64,
            "predicted_per_second": generation_tps,
        },
    }


class FakeEngine:
    """Serves scripted SSE responses in place of ``httpx.AsyncClient``."""

    def __init__(self) -> None:
        self.scripts: list[list[str]] = []
        self.payloads: list[dict[str, Any]] = []
        self.default: list[str] = sse(
            content_chunk(), timings_chunk(prompt_tps=1, generation_tps=1)
        )

    def next_lines(self) -> list[str]:
        return self.scripts.pop(0) if self.scripts else self.default

    def client(self, **_: Any) -> Any:
        engine = self

        class _Response:
            def __init__(self, lines: list[str]) -> None:
                self._lines = lines

            async def __aenter__(self) -> Any:
                return self

            async def __aexit__(self, *_exc: Any) -> None:
                return None

            def raise_for_status(self) -> None:
                return None

            async def aiter_lines(self) -> Any:
                for line in self._lines:
                    yield line

        class _Client:
            async def __aenter__(self) -> Any:
                return self

            async def __aexit__(self, *_exc: Any) -> None:
                return None

            def stream(self, _method: str, _url: str, json: Any = None) -> Any:
                engine.payloads.append(json)
                return _Response(engine.next_lines())

        return _Client()


@pytest.fixture()
def engine(monkeypatch: pytest.MonkeyPatch) -> FakeEngine:
    fake = FakeEngine()
    monkeypatch.setattr(benchmark_module.httpx, "AsyncClient", fake.client)
    return fake


def make_record(**settings: Any) -> ModelRecord:
    return ModelRecord(
        id="acme/model-Q4_K_M",
        name="model",
        path=Path("E:/models/model-Q4_K_M.gguf"),
        size_bytes=8 * GB,
        settings=ModelSettings(**settings),
    )


# ---------------------------------------------------------------------------
# available_modes
# ---------------------------------------------------------------------------


def test_available_modes_on_reference_rig() -> None:
    modes = available_modes(reference_rig())
    assert [m.key for m in modes] == [
        "rtx-5090-x1",
        "rtx-5090-x2",
        "rtx-3090-x1",
        "rtx-3090-x2",
        "all",
    ]
    assert [m.devices for m in modes] == [[0], [0, 1], [2], [2, 3], [0, 1, 2, 3]]
    assert [m.label for m in modes] == [
        "1x RTX 5090",
        "2x RTX 5090",
        "1x RTX 3090",
        "2x RTX 3090",
        "All 4 GPUs",
    ]
    assert modes[0].gpu_name == "NVIDIA GeForce RTX 5090"
    assert modes[-1].gpu_name is None


def test_the_default_mode_list_is_unchanged_by_the_new_dimensions() -> None:
    """Split mode and ubatch are opt-in dimensions. A suite that silently
    doubled in length would turn a two-minute job into a four-minute one for
    someone who never asked about tensor parallelism."""
    assert [m.key for m in available_modes(reference_rig())] == [
        m.key for m in available_modes(reference_rig(), split_modes=("layer",), ubatch_sizes=())
    ]
    assert all(
        m.split_mode == "layer" and m.ubatch is None for m in available_modes(reference_rig())
    )


def test_tensor_variants_are_added_only_to_multi_device_modes() -> None:
    """`--split-mode` is meaningless on one GPU (the supervisor emits
    ``--split-mode none`` there), so a tensor variant would benchmark the
    identical launch twice."""
    modes = available_modes(reference_rig(), split_modes=("layer", "tensor"))
    assert [m.key for m in modes] == [
        "rtx-5090-x1",
        "rtx-5090-x2",
        "rtx-5090-x2-tensor",
        "rtx-3090-x1",
        "rtx-3090-x2",
        "rtx-3090-x2-tensor",
        "all",
        "all-tensor",
    ]
    tensor = next(m for m in modes if m.key == "rtx-5090-x2-tensor")
    assert tensor.split_mode == "tensor"
    assert tensor.devices == [0, 1]
    assert "tensor split" in tensor.label


def test_ubatch_variants_multiply_every_placement() -> None:
    modes = available_modes(
        [gpu(0, "NVIDIA GeForce RTX 4090", 24, (8, 9))], ubatch_sizes=(1024, 2048)
    )
    assert [(m.key, m.ubatch) for m in modes] == [
        ("rtx-4090-x1", None),
        ("rtx-4090-x1-ub1024", 1024),
        ("rtx-4090-x1-ub2048", 2048),
    ]


def test_available_modes_orders_fastest_family_first_regardless_of_index() -> None:
    """The 3090s occupy the low CUDA indices; the 5090 family must still lead."""
    gpus = [
        gpu(0, "NVIDIA GeForce RTX 3090", 24, (8, 6)),
        gpu(1, "NVIDIA GeForce RTX 5090", 32, (12, 0)),
        gpu(2, "NVIDIA GeForce RTX 3090", 24, (8, 6)),
    ]
    modes = available_modes(gpus)
    assert [m.key for m in modes] == ["rtx-5090-x1", "rtx-3090-x1", "rtx-3090-x2", "all"]
    assert [m.devices for m in modes] == [[1], [0], [0, 2], [1, 0, 2]]


def test_available_modes_single_gpu_has_no_all_mode() -> None:
    modes = available_modes([gpu(0, "NVIDIA GeForce RTX 4090", 24, (8, 9))])
    assert [m.key for m in modes] == ["rtx-4090-x1"]
    assert modes[0].devices == [0]


def test_available_modes_no_gpus() -> None:
    assert available_modes([]) == []


def test_available_modes_four_identical_gpus_dedupes_all() -> None:
    gpus = [gpu(i, "NVIDIA GeForce RTX 3090", 24, (8, 6)) for i in range(4)]
    modes = available_modes(gpus)
    assert [m.key for m in modes] == [
        "rtx-3090-x1",
        "rtx-3090-x2",
        "rtx-3090-x3",
        "rtx-3090-x4",
    ]
    assert [m.devices for m in modes] == [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
    # "all" would be exactly x4, so offering it would benchmark the same thing twice.
    assert all(m.key != "all" for m in modes)


def test_available_modes_keys_are_slug_safe_and_unique() -> None:
    gpus = [
        gpu(0, "NVIDIA RTX A6000 (Ada) / 48GB", 48, (8, 9)),
        gpu(1, "NVIDIA RTX A6000 (Ada) / 48GB", 48, (8, 6)),  # same name, different cc
        gpu(2, "NVIDIA", 8, (7, 5)),  # nothing but a vendor word
        gpu(3, "???", 8, (7, 0)),  # nothing sluggable at all
    ]
    modes = available_modes(gpus)
    keys = [m.key for m in modes]
    assert len(keys) == len(set(keys))
    for key in keys:
        assert key == key.lower()
        assert all(char.isalnum() or char == "-" for char in key), key
    # The two same-named families get distinct prefixes rather than colliding.
    assert "rtx-a6000-ada-48gb-x1" in keys
    assert "rtx-a6000-ada-48gb-2-x1" in keys
    # A name that is nothing but a vendor word keeps the vendor word rather
    # than collapsing to an empty key.
    assert "nvidia-x1" in keys
    # A name with no sluggable characters still yields a usable key.
    assert "gpu-x1" in keys


def test_available_modes_unknown_compute_capability_still_grouped() -> None:
    gpus = [gpu(0, "Mystery Accelerator", 16, None), gpu(1, "Mystery Accelerator", 16, None)]
    modes = available_modes(gpus)
    assert [m.key for m in modes] == ["mystery-accelerator-x1", "mystery-accelerator-x2"]


# ---------------------------------------------------------------------------
# Applicability
# ---------------------------------------------------------------------------


def _reject_small_cards(devices: list[int]) -> Any:
    """A model that only fits on the 32 GB cards."""
    if any(index >= 2 for index in devices):
        return LoadRejected(
            model_id="acme/model-Q4_K_M",
            reason="needs 30.10 GiB; the selected GPUs offer 22.40 GiB",
            required_bytes=32_000_000_000,
            available_bytes=24_000_000_000,
        )
    return None


def dense_record() -> ModelRecord:
    """A record whose GGUF metadata proves it dense and full-attention, which
    is what makes it eligible for a tensor-split measurement."""
    return ModelRecord(
        id="acme/dense-Q4_K_M",
        name="dense",
        path=Path("E:/models/dense-Q4_K_M.gguf"),
        size_bytes=8 * GB,
        meta=GgufMeta(architecture="qwen2", n_layer=32, n_embd=4096, n_head=32, n_head_kv=8),
        settings=ModelSettings(),
    )


def test_tensor_modes_are_offered_only_for_an_eligible_model() -> None:
    """Model-only gating here; the engine's feature list and the plan's KV type
    are the supervisor's business, and it refuses with a sentence."""
    probe = StubProbe(reference_rig())
    bench = Benchmarker(StubManager(dense_record(), probe), probe=probe)

    assert bench.split_modes_for(dense_record()) == ["layer", "tensor"]

    moe = dense_record()
    moe.meta = GgufMeta(
        architecture="qwen35moe",
        n_layer=32,
        n_embd=4096,
        n_head=32,
        n_head_kv=8,
        n_expert=128,
        n_expert_used=8,
    )
    assert bench.split_modes_for(moe) == ["layer"]

    hybrid = dense_record()
    hybrid.meta = GgufMeta(
        architecture="qwen35",
        n_layer=64,
        n_embd=5120,
        n_head=24,
        n_head_kv=4,
        extra={"full_attention_interval": 4},
    )
    assert bench.split_modes_for(hybrid) == ["layer"]


def test_a_tensor_mode_is_planned_and_run_with_the_same_settings() -> None:
    """A mode planned with a layer split and then RUN with a tensor one would
    report the wrong verdict for the wrong launch."""
    probe = StubProbe(reference_rig())
    record = dense_record()
    bench = Benchmarker(StubManager(record, probe), probe=probe)
    mode = next(
        m
        for m in available_modes(reference_rig(), split_modes=("layer", "tensor"))
        if m.key == "rtx-5090-x2-tensor"
    )
    settings = bench._settings_for(record, mode)
    assert settings.split_mode == "tensor"
    assert settings.device_override == [0, 1]
    assert settings.ubatch_size is None

    with_ub = bench._settings_for(record, replace(mode, ubatch=1024))
    assert with_ub.ubatch_size == 1024


def test_modes_for_reports_inapplicable_modes_with_the_planner_reason() -> None:
    probe = StubProbe(reference_rig())
    record = make_record()
    bench = Benchmarker(StubManager(record, probe, verdict=_reject_small_cards), probe=probe)

    entries = bench.modes_for(record, ctx_size=4096)
    verdicts = {mode.key: (ok, reason) for mode, ok, reason in entries}

    # Nothing is dropped: the user sees every mode and why it is unavailable.
    assert list(verdicts) == ["rtx-5090-x1", "rtx-5090-x2", "rtx-3090-x1", "rtx-3090-x2", "all"]
    assert verdicts["rtx-5090-x1"] == (True, None)
    assert verdicts["rtx-5090-x2"] == (True, None)
    assert verdicts["rtx-3090-x1"][0] is False
    assert "22.40 GiB" in (verdicts["rtx-3090-x1"][1] or "")
    assert verdicts["all"][0] is False


def test_modes_for_plans_with_a_temporary_override_not_a_saved_one() -> None:
    probe = StubProbe(reference_rig())
    record = make_record(device_override=[3])
    manager = StubManager(record, probe)
    bench = Benchmarker(manager, probe=probe)

    bench.modes_for(record, ctx_size=8192)

    assert manager.planner.calls == [[0], [0, 1], [2], [2, 3], [0, 1, 2, 3]]
    # The record itself was never touched.
    assert record.settings.device_override == [3]


def test_modes_for_treats_a_planner_explosion_as_inapplicable() -> None:
    probe = StubProbe(reference_rig())
    record = make_record()

    def boom(devices: list[int]) -> Any:
        raise RuntimeError("no GGUF metadata")

    manager = StubManager(record, probe, verdict=boom)
    bench = Benchmarker(manager, probe=probe)
    entries = bench.modes_for(record, ctx_size=4096)
    assert all(not applicable for _, applicable, _ in entries)
    assert "no GGUF metadata" in (entries[0][2] or "")


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


async def test_run_measures_every_applicable_mode_and_restores_state(
    engine: FakeEngine,
) -> None:
    probe = StubProbe(reference_rig())
    record = make_record(device_override=[3], ctx_size=2048)
    manager = StubManager(record, probe)
    bench = Benchmarker(manager, probe=probe)

    engine.scripts = [
        sse(content_chunk(), timings_chunk(prompt_tps=2000.0, generation_tps=120.0)),
        sse(content_chunk(), timings_chunk(prompt_tps=1500.0, generation_tps=90.0)),
    ]
    progress: list[tuple[str | None, str, float]] = []

    report = await bench.run(
        record,
        modes=["rtx-5090-x1", "rtx-3090-x1"],
        ctx_size=1024,
        max_tokens=32,
        on_progress=lambda mode, phase, fraction: progress.append((mode, phase, fraction)),
    )

    assert [r.mode for r in report.results] == ["rtx-5090-x1", "rtx-3090-x1"]
    assert manager.loads == [(0,), (2,)]
    assert report.results[0].generation_tps == 120.0
    assert report.results[0].prompt_tps == 2000.0
    assert report.results[0].prompt_tokens == 300
    assert report.results[0].generated_tokens == 64
    assert report.results[0].load_time_s is not None and report.results[0].load_time_s >= 0
    assert report.results[0].ttft_s is not None
    assert report.best_generation_mode == "rtx-5090-x1"
    assert report.best_prompt_mode == "rtx-5090-x1"
    assert report.ctx_size == 1024
    assert report.max_tokens == 32
    assert report.finished_at is not None

    # State restoration, the whole point of the finally block.
    assert record.settings.device_override == [3]
    assert record.settings.ctx_size == 2048
    assert manager.supervisor.instances == {}

    # The request is deterministic and does not reuse a prompt cache.
    payload = engine.payloads[0]
    assert payload["temperature"] == 0
    assert payload["cache_prompt"] is False
    assert payload["stream"] is True
    assert payload["max_tokens"] == 32

    phases = [phase for _, phase, _ in progress]
    assert "planning" in phases
    assert "loading" in phases
    assert "generating" in phases
    assert progress[-1] == (None, "done", 1.0)


async def test_run_defaults_to_a_long_deterministic_prompt(engine: FakeEngine) -> None:
    probe = StubProbe([gpu(0, "NVIDIA GeForce RTX 5090", 32, (12, 0))])
    record = make_record()
    bench = Benchmarker(StubManager(record, probe), probe=probe)

    report = await bench.run(record)

    assert report.prompt_chars == len(benchmark_module.DEFAULT_PROMPT)
    # "a few hundred tokens" -- long enough for prompt processing to be measurable.
    assert report.prompt_chars > 800
    assert engine.payloads[0]["messages"][0]["content"] == benchmark_module.DEFAULT_PROMPT


async def test_run_accepts_a_prompt_override(engine: FakeEngine) -> None:
    probe = StubProbe([gpu(0, "NVIDIA GeForce RTX 5090", 32, (12, 0))])
    record = make_record()
    bench = Benchmarker(StubManager(record, probe), probe=probe)

    report = await bench.run(record, prompt="hi")

    assert report.prompt_chars == 2
    assert engine.payloads[0]["messages"][0]["content"] == "hi"


async def test_run_reports_inapplicable_modes_without_loading_them(engine: FakeEngine) -> None:
    probe = StubProbe(reference_rig())
    record = make_record()
    manager = StubManager(record, probe, verdict=_reject_small_cards)
    bench = Benchmarker(manager, probe=probe)

    report = await bench.run(record, max_tokens=8)

    by_mode = {r.mode: r for r in report.results}
    assert len(report.results) == 5
    assert by_mode["rtx-3090-x1"].applicable is False
    assert by_mode["rtx-3090-x1"].skipped_reason is not None
    assert by_mode["rtx-3090-x1"].load_time_s is None
    # Only the two applicable modes were ever loaded.
    assert manager.loads == [(0,), (0, 1)]


async def test_one_failing_mode_does_not_abort_the_rest(engine: FakeEngine) -> None:
    probe = StubProbe(reference_rig())
    record = make_record(device_override=[1])
    manager = StubManager(record, probe)
    manager.fail_devices = {(0,)}
    bench = Benchmarker(manager, probe=probe)

    engine.scripts = [
        sse(content_chunk(), timings_chunk(prompt_tps=1000.0, generation_tps=55.0)),
        sse(content_chunk(), timings_chunk(prompt_tps=900.0, generation_tps=44.0)),
        sse(content_chunk(), timings_chunk(prompt_tps=800.0, generation_tps=33.0)),
        sse(content_chunk(), timings_chunk(prompt_tps=700.0, generation_tps=22.0)),
    ]

    report = await bench.run(record)

    by_mode = {r.mode: r for r in report.results}
    assert by_mode["rtx-5090-x1"].error is not None
    assert "simulated load failure" in by_mode["rtx-5090-x1"].error
    assert by_mode["rtx-5090-x1"].generation_tps is None
    # Every other mode still ran.
    assert by_mode["rtx-5090-x2"].generation_tps == 55.0
    assert by_mode["all"].generation_tps == 22.0
    assert report.best_generation_mode == "rtx-5090-x2"
    assert record.settings.device_override == [1]


async def test_device_override_is_restored_when_every_mode_fails(engine: FakeEngine) -> None:
    probe = StubProbe(reference_rig())
    record = make_record(device_override=[2, 3])
    manager = StubManager(record, probe)
    manager.fail_devices = {(0,), (0, 1), (2,), (2, 3), (0, 1, 2, 3)}
    bench = Benchmarker(manager, probe=probe)

    report = await bench.run(record)

    assert all(r.error is not None for r in report.results)
    assert report.best_generation_mode is None
    assert report.best_prompt_mode is None
    assert record.settings.device_override == [2, 3]


async def test_device_override_is_restored_after_cancellation(engine: FakeEngine) -> None:
    probe = StubProbe(reference_rig())
    record = make_record(device_override=[3])
    manager = StubManager(record, probe)
    manager.load_gate = asyncio.Event()
    manager.load_started = asyncio.Event()
    bench = Benchmarker(manager, probe=probe)

    task = asyncio.create_task(bench.run(record))
    await asyncio.wait_for(manager.load_started.wait(), timeout=5)
    # Mid-load: the harshest moment to be cancelled at.
    assert record.settings.device_override == [0]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert record.settings.device_override == [3]
    # The lock is released, so a later benchmark can still run.
    assert bench.busy is False


async def test_cancel_event_stops_between_modes(engine: FakeEngine) -> None:
    probe = StubProbe(reference_rig())
    record = make_record()
    manager = StubManager(record, probe)
    bench = Benchmarker(manager, probe=probe)
    cancel = asyncio.Event()

    def on_progress(mode: str | None, phase: str, fraction: float) -> None:
        if phase == "done" and mode is not None:
            cancel.set()

    report = await bench.run(record, on_progress=on_progress, cancel_event=cancel)

    assert len(report.results) == 1
    assert manager.loads == [(0,)]
    assert any("canceled after" in note for note in report.notes)
    assert record.settings.device_override is None


async def test_second_run_while_one_is_in_flight_is_rejected(engine: FakeEngine) -> None:
    probe = StubProbe(reference_rig())
    record = make_record()
    manager = StubManager(record, probe)
    manager.load_gate = asyncio.Event()
    manager.load_started = asyncio.Event()
    bench = Benchmarker(manager, probe=probe)

    task = asyncio.create_task(bench.run(record))
    await asyncio.wait_for(manager.load_started.wait(), timeout=5)

    assert bench.busy is True
    with pytest.raises(ModelBusyError) as excinfo:
        await bench.run(record)
    assert "already running" in str(excinfo.value)
    assert excinfo.value.status_code == 503

    manager.load_gate.set()
    await task
    assert bench.busy is False


async def test_unknown_mode_key_is_a_bad_request(engine: FakeEngine) -> None:
    from studioforge.errors import BadRequestError

    probe = StubProbe(reference_rig())
    record = make_record()
    bench = Benchmarker(StubManager(record, probe), probe=probe)

    with pytest.raises(BadRequestError):
        await bench.run(record, modes=["rtx-9090-x1"])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


async def test_engine_timings_win_over_wall_clock(engine: FakeEngine) -> None:
    """The stopwatch would report a wildly different number; timings must win."""
    probe = StubProbe([gpu(0, "NVIDIA GeForce RTX 5090", 32, (12, 0))])
    record = make_record()
    bench = Benchmarker(StubManager(record, probe), probe=probe)
    engine.scripts = [
        sse(content_chunk(), timings_chunk(prompt_tps=1924.76, generation_tps=651.26))
    ]

    report = await bench.run(record)

    result = report.results[0]
    assert result.prompt_tps == 1924.76
    assert result.generation_tps == 651.26
    assert result.prompt_tokens == 300
    assert result.generated_tokens == 64
    # A fake stream returns in microseconds, so a wall-clock rate would be
    # orders of magnitude higher -- proof the engine numbers were used.
    assert not any("wall clock" in note for note in report.notes)


async def test_wall_clock_fallback_is_used_and_noted(engine: FakeEngine) -> None:
    probe = StubProbe([gpu(0, "NVIDIA GeForce RTX 5090", 32, (12, 0))])
    record = make_record()
    bench = Benchmarker(StubManager(record, probe), probe=probe)
    engine.scripts = [
        sse(
            content_chunk(),
            {"choices": [], "usage": {"prompt_tokens": 300, "completion_tokens": 64}},
        )
    ]

    report = await bench.run(record)

    result = report.results[0]
    assert result.generated_tokens == 64
    assert result.prompt_tokens == 300
    assert result.generation_tps is not None and result.generation_tps > 0
    notes = [note for note in report.notes if "wall clock" in note]
    assert len(notes) == 1
    assert result.mode in notes[0]


async def test_best_modes_when_only_one_succeeds(engine: FakeEngine) -> None:
    probe = StubProbe(reference_rig())
    record = make_record()
    manager = StubManager(record, probe, verdict=_reject_small_cards)
    manager.fail_devices = {(0, 1)}
    bench = Benchmarker(manager, probe=probe)
    engine.scripts = [sse(content_chunk(), timings_chunk(prompt_tps=10.0, generation_tps=5.0))]

    report = await bench.run(record)

    assert report.best_generation_mode == "rtx-5090-x1"
    assert report.best_prompt_mode == "rtx-5090-x1"


def test_best_ignores_skipped_and_errored_results() -> None:
    from studioforge.core.benchmark import BenchmarkReport

    report = BenchmarkReport(
        model_id="m",
        started_at=0.0,
        finished_at=1.0,
        ctx_size=4096,
        max_tokens=128,
        prompt_chars=10,
        results=[
            BenchmarkResult(
                mode="a", label="A", devices=[0], applicable=False, skipped_reason="no room"
            ),
            BenchmarkResult(mode="b", label="B", devices=[1], generation_tps=999.0, error="boom"),
            BenchmarkResult(mode="c", label="C", devices=[2], generation_tps=10.0, prompt_tps=1.0),
        ],
    )
    report.recompute_best()
    assert report.best_generation_mode == "c"
    assert report.best_prompt_mode == "c"


async def test_vram_used_is_a_free_memory_delta(engine: FakeEngine) -> None:
    probe = StubProbe(reference_rig())
    record = make_record()
    manager = StubManager(record, probe)

    original_load = manager.load

    async def load_and_consume(name: str, **kwargs: Any) -> InstanceInfo:
        info = await original_load(name, **kwargs)
        probe.consume(0, 6 * GB)
        return info

    manager.load = load_and_consume  # type: ignore[method-assign]
    bench = Benchmarker(manager, probe=probe)

    report = await bench.run(record, modes=["rtx-5090-x1"])

    assert report.results[0].vram_used_bytes == 6 * GB


async def test_report_is_json_serializable(engine: FakeEngine) -> None:
    probe = StubProbe(reference_rig())
    record = make_record()
    bench = Benchmarker(StubManager(record, probe), probe=probe)

    report = await bench.run(record, modes=["rtx-5090-x1"])
    payload = report.to_dict()

    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["model_id"] == record.id
    assert round_tripped["results"][0]["mode"] == "rtx-5090-x1"
    assert round_tripped["results"][0]["devices"] == [0]


def test_mode_to_dict_shape() -> None:
    """The launch dimensions travel with the mode, so a persisted report stays
    interpretable when the mode list grows another one."""
    mode = BenchmarkMode(key="k", label="L", devices=[0, 1], gpu_name="G")
    assert mode.to_dict() == {
        "key": "k",
        "label": "L",
        "devices": [0, 1],
        "gpu_name": "G",
        "split_mode": "layer",
        "ubatch": None,
    }


# ---------------------------------------------------------------------------
# Job table
# ---------------------------------------------------------------------------


def test_job_table_is_bounded_but_never_drops_a_running_job() -> None:
    jobs = BenchmarkJobs(limit=3)
    first = jobs.create("m0", ["a"])
    for index in range(1, 5):
        jobs.create(f"m{index}", ["a"])
        # Everything after the first is immediately terminal.
        for job in jobs.all():
            if job is not first and job.state == "running":
                job.state = "completed"

    assert len(jobs.all()) == 3
    # The live job survived eviction even though it is the oldest.
    assert jobs.get(first.job_id) is first
    assert jobs.running() is first


def test_job_progress_and_payload_shape() -> None:
    jobs = BenchmarkJobs(limit=5)
    job = jobs.create("m", ["a", "b"])
    job.on_progress(None, "planning", 0.0)
    job.on_progress("a", "loading", 0.1)
    job.on_progress("a", "done", 0.5)

    payload = job.to_dict()
    assert payload["state"] == "running"
    assert payload["model_id"] == "m"
    assert payload["progress"] == {
        "mode": "a",
        "phase": "done",
        "fraction": 0.5,
        "completed": 1,
        "total": 2,
    }
    assert payload["report"] is None
    assert payload["error"] is None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path) -> Any:
    database = Database(tmp_path / "registry.sqlite3")
    database.migrate()
    yield database
    database.close()


def test_benchmarks_migration_is_idempotent(tmp_path: Path) -> None:
    database = Database(tmp_path / "registry.sqlite3")
    database.migrate()
    database.migrate()
    assert database.schema_version() == SCHEMA_VERSION
    assert SCHEMA_VERSION >= 2

    names = [
        row["name"]
        for row in database.connect().execute("SELECT name FROM schema_migrations ORDER BY version")
    ]
    # Membership, not position: migrations are appended over time, and this
    # test is about 002 having run, not about it being the newest thing.
    assert "002_benchmarks.sql" in names
    indexes = [
        row["name"]
        for row in database.connect().execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'benchmarks'"
        )
    ]
    assert "idx_benchmarks_model_ts" in indexes
    database.close()


def test_benchmark_round_trip(db: Database) -> None:
    report = {
        "model_id": "acme/model",
        "started_at": 1000.0,
        "finished_at": 1100.0,
        "ctx_size": 4096,
        "max_tokens": 128,
        "prompt_chars": 42,
        "results": [{"mode": "rtx-5090-x1", "devices": [0], "generation_tps": 120.5}],
        "best_generation_mode": "rtx-5090-x1",
        "notes": [],
    }
    row_id = db.save_benchmark("acme/model", report)
    assert row_id > 0

    rows = db.list_benchmarks("acme/model")
    assert len(rows) == 1
    assert rows[0]["id"] == row_id
    assert rows[0]["ts"] == 1000.0
    assert rows[0]["ctx_size"] == 4096
    assert rows[0]["max_tokens"] == 128
    # The _json suffix never leaks; nested structures survive.
    assert "report_json" not in rows[0]
    assert rows[0]["report"] == report
    assert rows[0]["report"]["results"][0]["generation_tps"] == 120.5


def test_benchmarks_are_newest_first_and_filtered_by_model(db: Database) -> None:
    for index, ts in enumerate([100.0, 300.0, 200.0]):
        db.save_benchmark("acme/a", {"started_at": ts, "ctx_size": index, "max_tokens": 8})
    db.save_benchmark("acme/b", {"started_at": 999.0})

    mine = db.list_benchmarks("acme/a")
    assert [row["ts"] for row in mine] == [300.0, 200.0, 100.0]
    assert {row["model_id"] for row in mine} == {"acme/a"}

    assert len(db.list_benchmarks()) == 4
    assert len(db.list_benchmarks("acme/a", limit=2)) == 2

    latest = db.latest_benchmark("acme/a")
    assert latest is not None and latest["ts"] == 300.0
    assert db.latest_benchmark("acme/missing") is None


def test_save_benchmark_without_started_at_uses_now(db: Database) -> None:
    import time as _time

    before = _time.time()
    db.save_benchmark("acme/a", {"results": []})
    row = db.latest_benchmark("acme/a")
    assert row is not None
    assert row["ts"] >= before
    assert row["ctx_size"] is None
