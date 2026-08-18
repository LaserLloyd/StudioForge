"""Planner hardening driven by three production incidents.

* **Reasoning models need a bigger default context.** *"the 8192 default
  truncated chain-of-thought on reasoning models; we raised it to 32768 for
  those specifically."* A thinking model spends its budget thinking, so the
  ordinary default cuts the answer off before it starts.
* **A context beyond the trained window must not be silent.** *"we set 256K on
  a 128K model and silently got 128K."*
* **VRAM contention must name a culprit.** *"ComfyUI runs on the same GPU box;
  loads failed with 'model failed to load' and no visibility."*
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from studioforge.config import Config
from studioforge.core import gpu as gpu_module
from studioforge.core.manager import ModelManager
from studioforge.core.planner import THINKING_DEFAULT_CTX, Planner
from studioforge.types import (
    GB,
    GgufMeta,
    GpuInfo,
    InstanceInfo,
    LoadPlan,
    LoadRejected,
    ModelCapabilities,
    ModelRecord,
    ModelSettings,
    VramProcess,
)


class StubProbe:
    """Minimal GpuProbe with optional VRAM-holder reporting."""

    backend = "fake"

    def __init__(self, gpus: list[GpuInfo], processes: list[VramProcess] | None = None) -> None:
        self._gpus = {g.index: g.model_copy(deep=True) for g in gpus}
        self._processes = processes or []

    def available(self) -> bool:
        return bool(self._gpus)

    def list_gpus(self) -> list[GpuInfo]:
        return [self._gpus[i].model_copy(deep=True) for i in sorted(self._gpus)]

    def get_gpu(self, index: int) -> GpuInfo | None:
        return self._gpus.get(index)

    def driver_version(self) -> str | None:
        return "610.88"

    def cuda_driver_version(self) -> tuple[int, int] | None:
        return (13, 3)

    def compute_processes(self) -> list[VramProcess]:
        return [p.model_copy(deep=True) for p in self._processes]

    def shutdown(self) -> None:
        return None


class LegacyProbe(StubProbe):
    """A probe from before process enumeration existed."""

    compute_processes = None  # type: ignore[assignment]


def gpu(index: int, total_gib: float, free_gib: float) -> GpuInfo:
    total, free = int(total_gib * GB), int(free_gib * GB)
    return GpuInfo(
        index=index,
        name=f"FakeGPU{index}",
        total_bytes=total,
        free_bytes=free,
        used_bytes=total - free,
        compute_capability=(12, 0),
    )


def make_record(
    *,
    thinking: bool = False,
    n_ctx_train: int = 131072,
    tensor_bytes: int = 4 * GB,
    settings: ModelSettings | None = None,
    model_id: str = "test/model",
) -> ModelRecord:
    return ModelRecord(
        id=model_id,
        name=model_id,
        path="/models/test.gguf",
        size_bytes=tensor_bytes,
        meta=GgufMeta(
            architecture="qwen3",
            n_layer=32,
            n_head=32,
            n_head_kv=8,
            n_embd=4096,
            n_ctx_train=n_ctx_train,
            tensor_bytes=tensor_bytes,
            quant_label="Q4_K_M",
        ),
        capabilities=ModelCapabilities(thinking=thinking),
        settings=settings or ModelSettings(),
    )


def make_planner(probe: Any, **planner_overrides: Any) -> Planner:
    config = Config(data_dir="/tmp/sf-incidents")
    for key, value in planner_overrides.items():
        setattr(config.planner, key, value)
    return Planner(config, probe)


def notes_text(result: LoadPlan | LoadRejected) -> str:
    if isinstance(result, LoadPlan):
        return " ".join(result.notes)
    return " ".join([*result.notes, *result.suggestions])


# ---------------------------------------------------------------------------
# Incident 6: reasoning models need a bigger default context
# ---------------------------------------------------------------------------


def test_thinking_model_gets_the_larger_default_context() -> None:
    """"the 8192 default truncated chain-of-thought on reasoning models"."""
    planner = make_planner(StubProbe([gpu(0, 32, 31)]))

    plan = planner.plan_load(make_record(thinking=True))

    assert isinstance(plan, LoadPlan)
    # The planner aims at models.target_ctx and steps down to what fits; with a
    # small model on a 32 GiB card the aim is reached outright. The incident is
    # that the window must never be the cramped default, so assert the floor it
    # must clear rather than one exact value.
    assert plan.ctx_size >= THINKING_DEFAULT_CTX
    assert "thinking model" in notes_text(plan)


def test_a_plain_model_also_gets_a_roomy_context() -> None:
    """Agent workloads ran out of room at 8192 on ordinary models too.

    OpenClaw carries long tool transcripts, so the aim applies to every model,
    not just reasoning ones -- the step-down is what keeps big models loadable.
    """
    planner = make_planner(StubProbe([gpu(0, 32, 31)]))

    plan = planner.plan_load(make_record(thinking=False))

    assert isinstance(plan, LoadPlan)
    assert plan.ctx_size > 8192


def test_an_explicit_context_always_wins() -> None:
    """The boost is a default, so anything explicit must override it."""
    planner = make_planner(StubProbe([gpu(0, 32, 31)]))
    record = make_record(thinking=True, settings=ModelSettings(ctx_size=4096))

    plan = planner.plan_load(record)
    per_request = planner.plan_load(make_record(thinking=True), ctx_size=16384)

    assert isinstance(plan, LoadPlan) and plan.ctx_size == 4096
    assert isinstance(per_request, LoadPlan) and per_request.ctx_size == 16384


def test_the_boost_is_clamped_to_the_trained_window() -> None:
    """Never default past what the model was trained for."""
    planner = make_planner(StubProbe([gpu(0, 32, 31)]))

    plan = planner.plan_load(make_record(thinking=True, n_ctx_train=16384))

    assert isinstance(plan, LoadPlan)
    assert plan.ctx_size == 16384


def test_a_small_trained_window_clamps_below_the_global_default() -> None:
    """Never launch past what the model was trained for.

    The floor is a floor for the LADDER, not a licence to exceed a model's
    trained window: going beyond n_ctx_train needs RoPE scaling and silently
    degrades quality. This previously ran a 4096-trained model at 8192.
    """
    planner = make_planner(StubProbe([gpu(0, 32, 31)]))

    plan = planner.plan_load(make_record(thinking=True, n_ctx_train=4096))

    assert isinstance(plan, LoadPlan)
    assert plan.ctx_size == 4096


def test_the_boost_never_turns_a_working_load_into_a_rejection() -> None:
    """"Do NOT exceed what the planner says fits" -- back off, do not fail.

    24 GiB of weights on a 32 GiB card leaves room for an 8k KV cache but not
    a 32k one, so the model must still load, at the smaller context.
    """
    planner = make_planner(StubProbe([gpu(0, 32, 31)]))

    plan = planner.plan_load(make_record(thinking=True, tensor_bytes=24 * GB))

    assert isinstance(plan, LoadPlan), "the elevated default made a fitting model unloadable"
    # Stepped down from the aim to something that genuinely fits, and never
    # below the floor.
    assert 8192 <= plan.ctx_size < 131072


def test_the_boost_never_causes_an_eviction() -> None:
    """A nicer default context is not worth unloading somebody else's model."""
    planner = make_planner(StubProbe([gpu(0, 32, 31)]))
    resident = InstanceInfo(
        model_id="other/model",
        state="ready",
        ttl_s=1800,
        started_at=1.0,
        last_activity_at=1.0,
        plan=LoadPlan(model_id="other/model", devices=[0], per_gpu_bytes={0: 20 * GB}),
    )

    plan = planner.plan_load(make_record(thinking=True, tensor_bytes=24 * GB), loaded=[resident])

    assert isinstance(plan, LoadPlan)
    assert plan.evict_model_ids == []
    assert 8192 <= plan.ctx_size < 131072


# ---------------------------------------------------------------------------
# Incident 11: context beyond the trained window
# ---------------------------------------------------------------------------


def test_context_beyond_the_trained_window_is_flagged() -> None:
    """"we set 256K on a 128K model and silently got 128K"."""
    planner = make_planner(StubProbe([gpu(0, 80, 79)]))
    record = make_record(n_ctx_train=131072, tensor_bytes=2 * GB)

    plan = planner.plan_load(record, ctx_size=262144)

    assert isinstance(plan, LoadPlan)
    text = notes_text(plan)
    assert "131072" in text
    assert "262144" in text
    assert "rope" in text.lower()


def test_no_note_when_the_context_is_within_the_trained_window() -> None:
    planner = make_planner(StubProbe([gpu(0, 32, 31)]))

    plan = planner.plan_load(make_record(n_ctx_train=131072), ctx_size=8192)

    assert isinstance(plan, LoadPlan)
    assert "trained context window" not in notes_text(plan)


def test_the_context_is_not_silently_clamped() -> None:
    """RoPE scaling is a legitimate choice: warn, do not overrule."""
    planner = make_planner(StubProbe([gpu(0, 80, 79)]))

    plan = planner.plan_load(make_record(n_ctx_train=131072, tensor_bytes=2 * GB), ctx_size=262144)

    assert isinstance(plan, LoadPlan)
    assert plan.ctx_size == 262144


def test_the_warning_survives_a_rejection() -> None:
    """A refusal is where the user is reading, so the note has to be there too."""
    planner = make_planner(StubProbe([gpu(0, 8, 7)]))

    rejected = planner.plan_load(make_record(n_ctx_train=8192, tensor_bytes=6 * GB), ctx_size=65536)

    assert isinstance(rejected, LoadRejected)
    assert any("trained context window" in note for note in rejected.notes)


# ---------------------------------------------------------------------------
# Incident 7: foreign VRAM holders
# ---------------------------------------------------------------------------


def test_a_rejection_names_the_process_holding_the_vram() -> None:
    """"ComfyUI on the same box, 'model failed to load', no visibility"."""
    probe = StubProbe(
        [gpu(0, 24, 2)],
        processes=[VramProcess(gpu_index=0, pid=1234, name="python.exe", used_bytes=21 * GB)],
    )
    planner = make_planner(probe)

    rejected = planner.plan_load(make_record(tensor_bytes=16 * GB))

    assert isinstance(rejected, LoadRejected)
    assert any("python.exe" in s and "1234" in s for s in rejected.suggestions)
    assert rejected.vram_holders[0].pid == 1234
    assert "21.00 GiB held by python.exe (pid 1234) on CUDA0" in rejected.message()


def test_our_own_children_are_not_reported_as_foreign() -> None:
    """Blaming our own resident models would be noise, not attribution."""
    probe = StubProbe(
        [gpu(0, 24, 2)],
        processes=[VramProcess(gpu_index=0, pid=777, name="llama-server.exe", used_bytes=21 * GB)],
    )
    planner = make_planner(probe)
    ours = InstanceInfo(model_id="other/model", state="ready", pid=777, ttl_s=0)

    rejected = planner.plan_load(make_record(tensor_bytes=16 * GB), loaded=[ours])

    assert isinstance(rejected, LoadRejected)
    assert rejected.vram_holders[0].is_ours is True
    assert not any("llama-server.exe" in s for s in rejected.suggestions)


def test_a_probe_that_cannot_enumerate_degrades_silently() -> None:
    """NVML often cannot list processes in containers/WSL: no holders, no crash."""
    planner = make_planner(LegacyProbe([gpu(0, 24, 2)]))

    rejected = planner.plan_load(make_record(tensor_bytes=16 * GB))

    assert isinstance(rejected, LoadRejected)
    assert rejected.vram_holders == []
    assert rejected.suggestions, "a rejection must still be actionable"


def test_nvml_enumeration_reads_pid_name_and_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The NVML path itself, without hardware."""

    class FakeNvml:
        def nvmlInit(self) -> None:
            return None

        def nvmlDeviceGetCount(self) -> int:
            return 1

        def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
            return index

        def nvmlDeviceGetComputeRunningProcesses_v3(self, handle: int) -> list[Any]:
            return [SimpleNamespace(pid=4321, usedGpuMemory=3 * GB)]

        def nvmlSystemGetProcessName(self, pid: int) -> bytes:
            return rb"C:\ComfyUI\python.exe"

    monkeypatch.setattr(gpu_module, "_load_nvml", lambda: FakeNvml())
    probe = gpu_module.NvmlGpuProbe()

    holders = gpu_module.vram_processes(probe, own_pids=[999])

    assert len(holders) == 1
    assert holders[0].pid == 4321
    assert holders[0].used_bytes == 3 * GB
    assert holders[0].is_ours is False
    assert holders[0].name  # psutil name or the NVML fallback, never blank


def test_nvml_failure_yields_no_holders(monkeypatch: pytest.MonkeyPatch) -> None:
    """"Degrade silently if NVML cannot enumerate" -- one empty list, no raise."""

    class BrokenNvml:
        def nvmlInit(self) -> None:
            return None

        def nvmlDeviceGetCount(self) -> int:
            return 1

        def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
            return index

        def nvmlDeviceGetComputeRunningProcesses(self, handle: int) -> list[Any]:
            raise RuntimeError("NVML_ERROR_NOT_SUPPORTED")

    monkeypatch.setattr(gpu_module, "_load_nvml", lambda: BrokenNvml())

    assert gpu_module.vram_processes(gpu_module.NvmlGpuProbe()) == []


def test_server_status_reports_vram_holders() -> None:
    """Contention has to be visible *before* a load is refused, too."""
    probe = StubProbe(
        [gpu(0, 24, 2)],
        processes=[VramProcess(gpu_index=0, pid=1234, name="python.exe", used_bytes=21 * GB)],
    )
    manager = ModelManager(
        Config(data_dir="/tmp/sf-incidents"),
        registry=_StatusRegistry(),  # type: ignore[arg-type]
        planner=make_planner(probe),
        supervisor=_StatusSupervisor(),  # type: ignore[arg-type]
        db=None,  # type: ignore[arg-type]
    )

    status = manager.status()

    assert [p.pid for p in status.vram_processes] == [1234]
    assert status.vram_processes[0].name == "python.exe"


class _StatusRegistry:
    def all(self) -> list[ModelRecord]:
        return []


class _StatusSupervisor:
    def list(self) -> list[InstanceInfo]:
        return []


# ---------------------------------------------------------------------------
# Incident 14: OpenClaw ran out of context because every load defaulted to 8192
# ---------------------------------------------------------------------------


def test_the_ladder_steps_down_to_the_largest_window_that_fits() -> None:
    """Aim high, then halve until it fits -- never reject for wanting room."""
    roomy = make_planner(StubProbe([gpu(0, 80, 79)]))
    tight = make_planner(StubProbe([gpu(0, 32, 31)]))

    big = roomy.plan_load(make_record(thinking=False))
    small = tight.plan_load(make_record(thinking=False, tensor_bytes=24 * GB))

    assert isinstance(big, LoadPlan) and isinstance(small, LoadPlan)
    # Same model settings, different VRAM: the roomier card gets the roomier
    # window, and both load.
    assert big.ctx_size > small.ctx_size
    assert small.ctx_size >= 8192


def test_an_explicit_request_is_never_silently_upgraded() -> None:
    """Asking for 4096 must give 4096, not a helpful surprise."""
    planner = make_planner(StubProbe([gpu(0, 80, 79)]))

    per_model = planner.plan_load(make_record(settings=ModelSettings(ctx_size=4096)))
    per_request = planner.plan_load(make_record(), ctx_size=4096)

    assert isinstance(per_model, LoadPlan) and per_model.ctx_size == 4096
    assert isinstance(per_request, LoadPlan) and per_request.ctx_size == 4096


def test_the_ladder_never_exceeds_the_trained_window() -> None:
    """Going past n_ctx_train needs RoPE scaling and degrades quality."""
    planner = make_planner(StubProbe([gpu(0, 80, 79)]))

    plan = planner.plan_load(make_record(thinking=False, n_ctx_train=16384))

    assert isinstance(plan, LoadPlan)
    assert plan.ctx_size <= 16384
