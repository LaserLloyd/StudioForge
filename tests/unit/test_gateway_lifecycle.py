"""Regression tests for two defects that silently pinned VRAM forever.

Both were found by review, not by the suite, and both are invisible in normal
use -- the server keeps answering while models accumulate in VRAM until a
restart. They are pinned here because nothing else would catch a reintroduction.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from studioforge.api import openai_routes
from studioforge.config import Config
from studioforge.core.manager import ModelManager
from studioforge.types import (
    GB,
    GgufMeta,
    InstanceInfo,
    LoadPlan,
    ModelRecord,
    ModelSettings,
)


class CountingSupervisor:
    """Records request-slot bookkeeping so leaks are observable."""

    def __init__(self) -> None:
        self.active = 0
        self.starts = 0
        self.ends = 0
        self.instances: dict[str, InstanceInfo] = {}

    def mark_request_start(self, model_id: str) -> None:
        self.starts += 1
        self.active += 1
        instance = self.instances.get(model_id)
        if instance is not None:
            instance.active_requests += 1

    def mark_request_end(self, model_id: str, *, tokens_per_second: float | None = None) -> None:
        self.ends += 1
        self.active -= 1
        instance = self.instances.get(model_id)
        if instance is not None:
            instance.active_requests = max(0, instance.active_requests - 1)

    def base_url(self, model_id: str) -> str:
        return "http://127.0.0.1:1"

    def get(self, model_id: str) -> InstanceInfo | None:
        return self.instances.get(model_id)

    def list(self) -> list[InstanceInfo]:
        return list(self.instances.values())

    async def stop(self, model_id: str, **_kwargs: Any) -> None:
        self.instances.pop(model_id, None)

    async def stop_all(self, **_kwargs: Any) -> None:
        self.instances.clear()

    def tail_log(self, model_id: str, n: int = 200) -> list[str]:
        return []


class FakeStreamResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self.status_code = 200
        self._chunks = chunks

    async def __aenter__(self) -> FakeStreamResponse:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def aiter_raw(self) -> Any:
        for chunk in self._chunks:
            yield chunk
            await asyncio.sleep(0)

    async def aread(self) -> bytes:
        return b""


class FakeClient:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def stream(self, *_args: object, **_kwargs: object) -> FakeStreamResponse:
        return FakeStreamResponse(self._chunks)


class FakeState:
    def __init__(self, supervisor: CountingSupervisor, chunks: list[bytes]) -> None:
        self.supervisor = supervisor
        self.client = FakeClient(chunks)
        self.config = Config(data_dir="/tmp/sf-lifecycle")


def make_record(model_id: str = "test/model", **settings: Any) -> ModelRecord:
    return ModelRecord(
        id=model_id,
        name=model_id,
        path="/models/test.gguf",
        size_bytes=GB,
        meta=GgufMeta(architecture="llama", n_layer=8, n_head=8, n_head_kv=8, n_embd=512),
        settings=ModelSettings(**settings),
    )


# ---------------------------------------------------------------------------
# Regression 1: client disconnect must not leak the request slot
# ---------------------------------------------------------------------------


async def test_request_slot_released_on_normal_completion() -> None:
    supervisor = CountingSupervisor()
    state = FakeState(supervisor, [b'data: {"delta":1}\n\n', b"data: [DONE]\n\n"])
    record = make_record()

    chunks = [
        chunk
        async for chunk in openai_routes._stream_upstream(
            state, record, "http://x/v1/chat/completions", {}, 0.0
        )
    ]
    assert b"[DONE]" in b"".join(chunks)
    assert supervisor.starts == 1
    assert supervisor.ends == 1
    assert supervisor.active == 0


async def test_request_slot_released_when_client_disconnects_midstream() -> None:
    """The bug: yielding [DONE] before the decrement swallowed the decrement.

    Closing an async generator raises GeneratorExit at the paused yield; a
    ``yield`` during that unwind raises RuntimeError and skips everything after
    it. With the decrement placed after the sentinel, every disconnect leaked a
    slot -- and a non-zero ``active_requests`` permanently blocks both TTL
    unload and eviction for that model.
    """
    supervisor = CountingSupervisor()
    # Enough chunks that we can stop consuming partway through.
    state = FakeState(supervisor, [b'data: {"delta":%d}\n\n' % i for i in range(50)])
    record = make_record()

    generator = openai_routes._stream_upstream(
        state, record, "http://x/v1/chat/completions", {}, 0.0
    )
    await generator.__anext__()
    await generator.__anext__()
    assert supervisor.active == 1

    # Simulate the client going away mid-stream. This must not raise: yielding
    # during the GeneratorExit unwind would surface as RuntimeError here.
    await generator.aclose()

    assert supervisor.ends == 1, "mark_request_end never ran on disconnect"
    assert supervisor.active == 0, "active_requests leaked; VRAM would be pinned forever"


async def test_request_slot_not_taken_when_the_stream_is_never_iterated() -> None:
    """A disconnect before the first read must not leave a dangling slot.

    ``StreamingResponse`` does not start the generator until the body is first
    iterated, so incrementing outside it leaked a slot with no matching
    decrement whenever a client hung up that early.
    """
    supervisor = CountingSupervisor()
    state = FakeState(supervisor, [b"data: [DONE]\n\n"])
    record = make_record()

    generator = openai_routes._stream_upstream(
        state, record, "http://x/v1/chat/completions", {}, 0.0
    )
    await generator.aclose()  # never iterated

    assert supervisor.starts == 0
    assert supervisor.active == 0


# ---------------------------------------------------------------------------
# Regression 2: the effective TTL must reach the running instance
# ---------------------------------------------------------------------------


class StubRegistry:
    def __init__(self, records: list[ModelRecord]) -> None:
        self._records = {r.id: r for r in records}

    def all(self) -> list[ModelRecord]:
        return list(self._records.values())

    def resolve(self, name: str) -> ModelRecord | None:
        return self._records.get(name)

    def get(self, model_id: str) -> ModelRecord | None:
        return self._records.get(model_id)

    def known_ids(self) -> list[str]:
        return list(self._records)

    def save_settings(self, model_id: str, settings: ModelSettings) -> ModelRecord:
        record = self._records[model_id]
        record.settings = settings
        return record

    def touch(self, model_id: str) -> None:
        return None


def make_manager(records: list[ModelRecord], **models_cfg: Any) -> tuple[ModelManager, Any]:
    config = Config(data_dir="/tmp/sf-ttl")
    for key, value in models_cfg.items():
        setattr(config.models, key, value)
    supervisor = CountingSupervisor()
    manager = ModelManager(
        config,
        registry=StubRegistry(records),  # type: ignore[arg-type]
        planner=None,  # type: ignore[arg-type]
        supervisor=supervisor,  # type: ignore[arg-type]
        db=None,  # type: ignore[arg-type]
    )
    return manager, supervisor


def loaded(model_id: str, ttl_s: int | None) -> InstanceInfo:
    return InstanceInfo(
        model_id=model_id,
        state="ready",
        ttl_s=ttl_s,
        plan=LoadPlan(model_id=model_id, devices=[0]),
    )


def test_effective_ttl_folds_in_the_global_default() -> None:
    """The supervisor only knows the raw setting, which is usually None."""
    record = make_record()
    manager, supervisor = make_manager([record], default_ttl_s=1800)
    instance = loaded(record.id, None)  # what the supervisor produces

    manager.apply_effective_ttl(record, instance)

    assert instance.ttl_s == 1800, "default_ttl_s never reached the instance -> nothing unloads"


def test_effective_ttl_marks_pinned_models_with_zero() -> None:
    """0 is the wire representation of 'pinned' that the planner checks."""
    record = make_record(pinned=True)
    manager, _ = make_manager([record], default_ttl_s=1800)
    instance = loaded(record.id, None)

    manager.apply_effective_ttl(record, instance)

    assert instance.ttl_s == 0, "a pinned model not marked 0 would be evictable"


def test_pin_beats_an_explicit_ttl() -> None:
    record = make_record(pinned=True, ttl_s=60)
    manager, _ = make_manager([record])
    instance = loaded(record.id, 60)
    manager.apply_effective_ttl(record, instance)
    assert instance.ttl_s == 0


def test_per_model_ttl_overrides_the_default() -> None:
    record = make_record(ttl_s=42)
    manager, _ = make_manager([record], default_ttl_s=1800)
    instance = loaded(record.id, 42)
    manager.apply_effective_ttl(record, instance)
    assert instance.ttl_s == 42


def test_refresh_ttl_applies_a_pin_to_an_already_loaded_model() -> None:
    """Toggling a pin must bite immediately, not at the next load."""
    record = make_record()
    manager, supervisor = make_manager([record], default_ttl_s=1800)
    instance = loaded(record.id, 1800)
    supervisor.instances[record.id] = instance

    record.settings = ModelSettings(pinned=True)
    assert manager.refresh_ttl(record.id) == 0
    assert instance.ttl_s == 0


def test_refresh_ttl_is_a_noop_for_unloaded_models() -> None:
    record = make_record()
    manager, _ = make_manager([record])
    assert manager.refresh_ttl(record.id) is None
    assert manager.refresh_ttl("no/such-model") is None


@pytest.mark.parametrize(
    ("pinned", "ttl_s", "default", "expected"),
    [
        (False, None, 1800, 1800),
        (False, 0, 1800, 0),
        (False, 900, 1800, 900),
        (True, None, 1800, 0),
        (True, 900, 1800, 0),
        (False, None, 0, 0),
    ],
)
def test_ttl_for_matrix(pinned: bool, ttl_s: int | None, default: int, expected: int) -> None:
    record = make_record(pinned=pinned, ttl_s=ttl_s)
    manager, _ = make_manager([record], default_ttl_s=default)
    assert manager.ttl_for(record) == expected


# ---------------------------------------------------------------------------
# set_pinned: the one implementation every surface shares
# ---------------------------------------------------------------------------


def test_set_pinned_persists_and_bites_the_resident_instance() -> None:
    record = make_record()
    manager, supervisor = make_manager([record], default_ttl_s=1800)
    instance = loaded(record.id, 1800)
    supervisor.instances[record.id] = instance

    updated, effective = manager.set_pinned(record.id, True)

    assert updated.settings.pinned is True
    assert effective == 0
    assert instance.ttl_s == 0, "the pin must bite immediately, not at the next load"

    updated, effective = manager.set_pinned(record.id, False)
    assert updated.settings.pinned is False
    assert effective == 1800
    assert instance.ttl_s == 1800


def test_set_pinned_answers_the_effective_ttl_for_unloaded_models() -> None:
    record = make_record()
    manager, _ = make_manager([record], default_ttl_s=1800)
    _, effective = manager.set_pinned(record.id, True)
    assert effective == 0
    _, effective = manager.set_pinned(record.id, False)
    assert effective == 1800


def test_set_pinned_rejects_an_unknown_model() -> None:
    from studioforge.errors import ModelNotFoundError

    manager, _ = make_manager([make_record()])
    with pytest.raises(ModelNotFoundError):
        manager.set_pinned("no/such-model", True)


# ---------------------------------------------------------------------------
# The pinned reconciler (D41): a pin means "keep loaded at all times"
# ---------------------------------------------------------------------------


def test_reconciler_wants_only_missing_pinned_models() -> None:
    pinned_record = make_record("pinned/model", pinned=True)
    plain = make_record("plain/model")
    manager, supervisor = make_manager([pinned_record, plain])

    assert manager._pinned_needing_load() == ["pinned/model"]

    supervisor.instances["pinned/model"] = loaded("pinned/model", 0)
    assert manager._pinned_needing_load() == []


def test_reconciler_wants_a_pinned_model_whose_child_crashed_out() -> None:
    """State 'failed' (restart limit reached) is down, not resident."""
    record = make_record("pinned/model", pinned=True)
    manager, supervisor = make_manager([record])
    instance = loaded(record.id, 0)
    instance.state = "failed"
    supervisor.instances[record.id] = instance

    assert manager._pinned_needing_load() == [record.id]


async def test_explicit_unload_suppresses_the_reconciler_until_repin() -> None:
    """A deliberate unload outranks the pin; pinning again re-arms it."""
    record = make_record("pinned/model", pinned=True)
    manager, supervisor = make_manager([record])
    supervisor.instances[record.id] = loaded(record.id, 0)

    assert await manager.unload(record.id) is True
    assert manager._pinned_needing_load() == [], "unload must stick, not last 15 seconds"

    manager.set_pinned(record.id, True)
    assert manager._pinned_needing_load() == [record.id]


async def test_unload_all_suppresses_every_pin() -> None:
    records = [make_record("a/model", pinned=True), make_record("b/model", pinned=True)]
    manager, supervisor = make_manager(records)
    for r in records:
        supervisor.instances[r.id] = loaded(r.id, 0)

    await manager.unload_all()

    assert manager._pinned_needing_load() == [], "the panic button must actually free VRAM"


async def test_reconciler_loads_with_its_own_source() -> None:
    record = make_record("pinned/model", pinned=True)
    manager, supervisor = make_manager([record])
    calls: list[tuple[str, str]] = []

    async def fake_load(model_id: str, **kwargs: Any) -> InstanceInfo:
        calls.append((model_id, kwargs.get("source", "")))
        instance = loaded(model_id, 0)
        supervisor.instances[model_id] = instance
        return instance

    manager.load = fake_load  # type: ignore[method-assign]
    await manager._reconcile_pinned([record.id])

    assert calls == [(record.id, "pin-reconcile")]


async def test_reconciler_backs_off_after_a_failed_reload() -> None:
    record = make_record("pinned/model", pinned=True)
    manager, _ = make_manager([record])

    async def failing_load(model_id: str, **kwargs: Any) -> InstanceInfo:
        raise RuntimeError("no room")

    manager.load = failing_load  # type: ignore[method-assign]
    await manager._reconcile_pinned([record.id])

    _next_at, delay = manager._pin_retry[record.id]
    assert delay == manager.PIN_RETRY_BASE_S
    assert manager._pinned_needing_load() == [], "inside the backoff window: not retried"

    manager._pin_retry[record.id] = (0.0, delay)  # window elapsed
    assert manager._pinned_needing_load() == [record.id]
    await manager._reconcile_pinned([record.id])
    _next_at, delay = manager._pin_retry[record.id]
    assert delay == manager.PIN_RETRY_BASE_S * 2, "each failure doubles the wait"


def test_reconciler_is_off_when_auto_load_pinned_is_off() -> None:
    record = make_record("pinned/model", pinned=True)
    manager, _ = make_manager([record], auto_load_pinned=False)

    manager._maybe_reconcile_pinned()

    assert manager._reconcile_task is None, "the pin should mean only 'no TTL, no eviction'"


# ---------------------------------------------------------------------------
# A request-level ttl must not unpin a pinned model
# ---------------------------------------------------------------------------


def test_request_ttl_override_is_ignored_on_a_pinned_instance() -> None:
    """ttl_s == 0 is the wire form of pinned; a client ttl must not overwrite it."""
    supervisor = CountingSupervisor()
    state = FakeState(supervisor, chunks=[])
    instance = loaded("pinned/model", 0)
    supervisor.instances[instance.model_id] = instance

    openai_routes._apply_ttl_override(state, instance.model_id, 60)

    assert instance.ttl_s == 0


def test_request_ttl_override_still_applies_to_unpinned_instances() -> None:
    supervisor = CountingSupervisor()
    state = FakeState(supervisor, chunks=[])
    instance = loaded("plain/model", 1800)
    supervisor.instances[instance.model_id] = instance

    openai_routes._apply_ttl_override(state, instance.model_id, 60)

    assert instance.ttl_s == 60


# ---------------------------------------------------------------------------
# Regression 3: an in-flight stream must protect its model from eviction
# ---------------------------------------------------------------------------
#
# This wires the two mechanisms together end to end: the request slot taken by
# the real streaming generator is the exact signal the planner's eviction rule
# reads. If either side regresses (the generator stops counting, or the planner
# stops checking), a busy model gets SIGKILLed mid-stream to make room for a
# new load.


def _planner_with_one_gpu() -> Any:
    from studioforge.config import PlannerConfig
    from studioforge.core.gpu import FakeGpuProbe
    from studioforge.core.planner import Planner
    from studioforge.types import GpuInfo

    config = Config(data_dir="/tmp/sf-evict")
    config.planner = PlannerConfig(headroom_fraction=0.0, on_insufficient="evict")
    probe = FakeGpuProbe(
        [
            GpuInfo(
                index=0,
                name="Fake 5090",
                total_bytes=24 * GB,
                free_bytes=4 * GB,
                used_bytes=20 * GB,
                compute_capability=(12, 0),
            )
        ]
    )
    return Planner(config, probe)


def _resident_instance(model_id: str, held_bytes: int) -> InstanceInfo:
    return InstanceInfo(
        model_id=model_id,
        state="ready",
        ttl_s=300,  # unpinned: eviction is allowed once it goes idle
        plan=LoadPlan(model_id=model_id, devices=[0], per_gpu_bytes={0: held_bytes}),
        last_activity_at=0.0,
    )


async def test_a_streaming_model_is_never_evicted_until_the_stream_ends() -> None:
    planner = _planner_with_one_gpu()
    supervisor = CountingSupervisor()
    resident = _resident_instance("resident/model", held_bytes=18 * GB)
    supervisor.instances[resident.model_id] = resident
    state = FakeState(supervisor, [b'data: {"delta":%d}\n\n' % i for i in range(50)])

    # The incoming model needs ~7 GiB; only 4 GiB is free, so loading it
    # requires evicting the resident model.
    incoming = make_record("incoming/model")
    assert incoming.meta is not None
    incoming.meta.tensor_bytes = 6 * GB
    incoming.size_bytes = 6 * GB

    # Open a real gateway stream against the resident model and hold it open.
    generator = openai_routes._stream_upstream(
        state, make_record(resident.model_id), "http://x/v1/chat/completions", {}, 0.0
    )
    await generator.__anext__()
    assert resident.active_requests == 1

    from studioforge.types import LoadRejected

    while_streaming = planner.plan_load(incoming, ctx_size=2048, loaded=supervisor.list())
    assert isinstance(while_streaming, LoadRejected), (
        "the only way to fit is evicting a model with an in-flight stream; that must be refused"
    )

    # Client hangs up; the slot is released; NOW eviction is fair game.
    await generator.aclose()
    assert resident.active_requests == 0

    after_stream = planner.plan_load(incoming, ctx_size=2048, loaded=supervisor.list())
    assert isinstance(after_stream, LoadPlan)
    assert after_stream.evict_model_ids == [resident.model_id]


# ---------------------------------------------------------------------------
# Shutdown: draining is bounded by drain_timeout_s
# ---------------------------------------------------------------------------


class StoppableSupervisor(CountingSupervisor):
    def __init__(self) -> None:
        super().__init__()
        self.stopped_all = False

    async def stop_all(self) -> None:
        self.stopped_all = True


async def test_stop_cannot_hang_past_the_drain_timeout() -> None:
    """A stuck request counter must delay shutdown, never prevent it."""
    import time as time_module

    record = make_record()
    manager, _ = make_manager([record])
    supervisor = StoppableSupervisor()
    manager.supervisor = supervisor  # type: ignore[assignment]
    stuck = loaded(record.id, 300)
    stuck.active_requests = 1  # never drains
    supervisor.instances[record.id] = stuck

    started = time_module.monotonic()
    await manager.stop(drain_timeout_s=0.5)
    elapsed = time_module.monotonic() - started

    assert supervisor.stopped_all, "children must be stopped even when draining times out"
    assert elapsed < 5.0, f"stop() took {elapsed:.1f}s; the drain deadline did not bind"


async def test_stop_returns_promptly_when_nothing_is_in_flight() -> None:
    record = make_record()
    manager, _ = make_manager([record])
    supervisor = StoppableSupervisor()
    manager.supervisor = supervisor  # type: ignore[assignment]
    supervisor.instances[record.id] = loaded(record.id, 300)

    import time as time_module

    started = time_module.monotonic()
    await manager.stop(drain_timeout_s=30.0)
    assert time_module.monotonic() - started < 5.0
    assert supervisor.stopped_all


async def test_stop_cancels_a_preload_still_in_flight() -> None:
    """A slow autoload must not keep starting children after shutdown began.

    ``start()`` spawns the pinned-model preload as a background task; ``stop()``
    must hold a reference to it and cancel it. Without that, a multi-minute
    cold load finishes *after* ``stop_all()`` has run and leaves a llama-server
    child running with nobody left to supervise or stop it.
    """
    record_a = make_record("pinned/a", pinned=True)
    record_b = make_record("pinned/b", pinned=True)
    manager, _ = make_manager([record_a, record_b])
    manager.config.models.auto_load_pinned = True
    supervisor = StoppableSupervisor()
    manager.supervisor = supervisor  # type: ignore[assignment]

    release = asyncio.Event()
    started_loads: list[str] = []

    async def slow_load(model_id: str, **kwargs: Any) -> InstanceInfo:
        started_loads.append(model_id)
        await release.wait()
        return loaded(model_id, 0)

    manager.load = slow_load  # type: ignore[method-assign]

    await manager.start()
    for _ in range(200):
        if started_loads:
            break
        await asyncio.sleep(0.01)
    assert started_loads == ["pinned/a"], "the preload task never started"

    await manager.stop(drain_timeout_s=0.1)
    release.set()
    await asyncio.sleep(0.05)

    assert started_loads == ["pinned/a"], (
        "the preload kept loading models after stop(): a load that completes "
        "after shutdown leaves an unsupervised llama-server child running"
    )


# ---------------------------------------------------------------------------
# Per-model lock table must not grow forever
# ---------------------------------------------------------------------------


async def test_failed_load_does_not_leak_a_lock_entry() -> None:
    """Every ensure_loaded ever attempted used to leave a lock behind.

    Bounded for a static library, but virtual models can be created and
    deleted through the API, so the table must shrink when the last waiter
    leaves.
    """
    from studioforge.errors import StudioForgeError

    record = make_record()
    manager, supervisor = make_manager([record])
    manager.planner = _planner_with_one_gpu()

    # 22 GiB into 4 GiB free with nothing evictable: guaranteed rejection.
    assert record.meta is not None
    record.meta.tensor_bytes = 22 * GB
    record.size_bytes = 22 * GB

    for _ in range(3):
        with pytest.raises(StudioForgeError):
            await manager.ensure_loaded(record.id)

    assert manager._locks == {}, "lock entries must be pruned when the last waiter leaves"
    assert manager._load_waiters == {}


# ---------------------------------------------------------------------------
# Regression 3: model recency must be stable and reflect the whole model
# ---------------------------------------------------------------------------


def test_newest_mtime_uses_the_latest_shard(tmp_path) -> None:
    """A multi-part download finishes on its LAST part.

    Keying recency off shard 1 would rank a just-downloaded 2-part model by
    when its first part landed, which can be hours earlier.
    """
    import os

    from studioforge.core.registry import _newest_mtime

    first = tmp_path / "model-00001-of-00002.gguf"
    second = tmp_path / "model-00002-of-00002.gguf"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    os.utime(first, (1_000_000, 1_000_000))
    os.utime(second, (2_000_000, 2_000_000))

    assert _newest_mtime([first, second], fallback=1_000_000) == 2_000_000
    # Order of the shard list must not matter.
    assert _newest_mtime([second, first], fallback=1_000_000) == 2_000_000


def test_newest_mtime_survives_a_missing_file(tmp_path) -> None:
    from studioforge.core.registry import _newest_mtime

    present = tmp_path / "there.gguf"
    present.write_bytes(b"x")
    import os

    os.utime(present, (5_000, 5_000))
    missing = tmp_path / "gone.gguf"
    assert _newest_mtime([present, missing], fallback=1.0) == 5_000


def test_newest_mtime_falls_back_when_nothing_readable(tmp_path) -> None:
    from studioforge.core.registry import _newest_mtime

    assert _newest_mtime([tmp_path / "nope.gguf"], fallback=42.0) == 42.0
