"""One retry on a transient load failure.

Incident (production, the system this replaces): *"models intermittently failed
with 'llama-server died before healthy' (SIGABRT / exit 1) -- often a transient
OOM during a model swap -- and simply retrying later worked."* A terminal
failure there is a load the user has to notice and repeat by hand.

The retry is deliberately narrow: exactly one attempt, only for allocation-shaped
stderr, and only after evicting the LRU unpinned model. Retrying without
changing anything would fail identically, just slower.
"""

from __future__ import annotations

from typing import Any

import pytest

from studioforge.config import Config
from studioforge.core.manager import ModelManager, classify_load_failure
from studioforge.errors import InsufficientVramError, ModelLoadError
from studioforge.types import (
    GB,
    GgufMeta,
    InstanceInfo,
    LoadPlan,
    LoadRejected,
    ModelRecord,
    ModelSettings,
)

OOM_STDERR = [
    "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 12000.00 MiB on device 0 failed",
    "CUDA error: out of memory",
    "llama_model_load: error loading model: failed to allocate buffer",
]

CONFIG_STDERR = [
    'error while handling argument "--bogus-flag": unknown argument',
]


def make_record(model_id: str = "test/model") -> ModelRecord:
    return ModelRecord(
        id=model_id,
        name=model_id,
        path="/models/test.gguf",
        size_bytes=8 * GB,
        meta=GgufMeta(architecture="llama", n_layer=32, n_head=32, n_head_kv=8, n_embd=4096),
        settings=ModelSettings(),
    )


class StubRegistry:
    def __init__(self, records: dict[str, ModelRecord]) -> None:
        self._records = records

    def resolve(self, name: str) -> ModelRecord | None:
        return self._records.get(name)

    def get(self, model_id: str) -> ModelRecord | None:
        return self._records.get(model_id)

    def get_adapter(self, adapter_id: str) -> None:
        return None

    def known_ids(self) -> list[str]:
        return sorted(self._records)

    def all(self) -> list[ModelRecord]:
        return list(self._records.values())

    def touch(self, model_id: str) -> None:
        return None


class StubPlanner:
    """Always plans successfully; records how often it was asked."""

    def __init__(self, result: Any = None) -> None:
        self.calls = 0
        self.result = result

    def plan_load(self, record: ModelRecord, **kwargs: Any) -> Any:
        self.calls += 1
        if self.result is not None:
            return self.result
        return LoadPlan(model_id=record.id, devices=[0], ctx_size=8192)

    def _evictable(self, loaded: list[InstanceInfo]) -> list[InstanceInfo]:
        candidates = [i for i in loaded if i.ttl_s != 0 and i.active_requests == 0]
        return sorted(candidates, key=lambda i: i.last_activity_at or 0.0)

    @property
    def probe(self) -> Any:  # pragma: no cover - only used by status()
        raise AssertionError("not used")


class StubSupervisor:
    """Fails ``fail_times`` starts with ``stderr``, then succeeds."""

    def __init__(self, *, fail_times: int = 0, stderr: list[str] | None = None) -> None:
        self.fail_times = fail_times
        self.stderr = stderr or []
        self.starts = 0
        self.stopped: list[str] = []
        self.instances: dict[str, InstanceInfo] = {}

    async def start(self, record: ModelRecord, plan: LoadPlan, **kwargs: Any) -> InstanceInfo:
        self.starts += 1
        if self.starts <= self.fail_times:
            raise ModelLoadError(
                f"llama-server for '{record.id}' exited with code 1 during startup.",
                details={"stderr": self.stderr, "argv": ["llama-server"]},
            )
        info = InstanceInfo(model_id=record.id, state="ready", port=18100, plan=plan)
        self.instances[record.id] = info
        return info

    async def stop(self, model_id: str, **kwargs: Any) -> None:
        self.stopped.append(model_id)
        self.instances.pop(model_id, None)

    def get(self, model_id: str) -> InstanceInfo | None:
        return self.instances.get(model_id)

    def list(self) -> list[InstanceInfo]:
        return list(self.instances.values())


def make_manager(supervisor: StubSupervisor, planner: StubPlanner) -> ModelManager:
    record = make_record()
    return ModelManager(
        Config(data_dir="/tmp/sf-retry"),
        registry=StubRegistry({record.id: record}),  # type: ignore[arg-type]
        planner=planner,  # type: ignore[arg-type]
        supervisor=supervisor,  # type: ignore[arg-type]
        db=None,  # type: ignore[arg-type]
    )


def resident(model_id: str, *, ttl_s: int | None = 1800, last_used: float = 1.0) -> InstanceInfo:
    return InstanceInfo(
        model_id=model_id,
        state="ready",
        port=18101,
        ttl_s=ttl_s,
        started_at=last_used,
        last_activity_at=last_used,
        plan=LoadPlan(model_id=model_id, devices=[0]),
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "CUDA error: out of memory",
        "ggml_backend_cuda_buffer_type_alloc_buffer: failed to allocate 4096 MiB",
        "cudaMalloc failed: out of memory",
        'GGML_ASSERT: ggml-cuda.cu:1234: !"CUDA error"',
    ],
)
def test_allocation_failures_are_transient(line: str) -> None:
    assert classify_load_failure([line]) == "transient"


@pytest.mark.parametrize(
    "line",
    [
        'error while handling argument "--bogus": unknown argument',
        "llama_model_load: unknown model architecture: 'nonsense'",
        "failed to open GGUF file: no such file or directory",
    ],
)
def test_configuration_failures_are_never_transient(line: str) -> None:
    assert classify_load_failure([line]) == "config"


def test_a_config_error_wins_over_an_allocation_message() -> None:
    """A bad flag that also trips an assert must not look retryable."""
    assert classify_load_failure(["unknown argument: --x", "GGML_ASSERT: boom"]) == "config"


def test_an_unrecognised_failure_is_not_retried_blindly() -> None:
    assert classify_load_failure(["segmentation fault"]) == "unknown"


# ---------------------------------------------------------------------------
# The retry itself
# ---------------------------------------------------------------------------


async def test_a_transient_oom_evicts_and_retries_once() -> None:
    """ "a transient OOM during a model swap -- retrying later worked"."""
    supervisor = StubSupervisor(fail_times=1, stderr=OOM_STDERR)
    supervisor.instances["victim/model"] = resident("victim/model")
    planner = StubPlanner()
    manager = make_manager(supervisor, planner)

    instance = await manager.load("test/model")

    assert instance.state == "ready"
    assert supervisor.starts == 2, "the transient failure was not retried"
    assert supervisor.stopped == ["victim/model"], "the retry did not change the conditions"
    assert planner.calls == 2, "the retry must re-plan against the freed VRAM"


async def test_the_retry_happens_exactly_once() -> None:
    """One retry, not a loop: a genuinely broken model must fail fast."""
    supervisor = StubSupervisor(fail_times=5, stderr=OOM_STDERR)
    supervisor.instances["victim/model"] = resident("victim/model")
    manager = make_manager(supervisor, StubPlanner())

    with pytest.raises(ModelLoadError):
        await manager.load("test/model")

    assert supervisor.starts == 2


async def test_a_configuration_error_is_never_retried() -> None:
    """A bad flag fails identically the second time; retrying just hides it."""
    supervisor = StubSupervisor(fail_times=1, stderr=CONFIG_STDERR)
    supervisor.instances["victim/model"] = resident("victim/model")
    manager = make_manager(supervisor, StubPlanner())

    with pytest.raises(ModelLoadError) as excinfo:
        await manager.load("test/model")

    assert supervisor.starts == 1
    assert supervisor.stopped == [], "a config error must not cost another model its VRAM"
    assert "exited with code 1" in excinfo.value.message


async def test_no_retry_when_there_is_nothing_to_evict() -> None:
    """Retrying without changing anything is pointless, so it does not happen."""
    supervisor = StubSupervisor(fail_times=1, stderr=OOM_STDERR)
    manager = make_manager(supervisor, StubPlanner())

    with pytest.raises(ModelLoadError):
        await manager.load("test/model")

    assert supervisor.starts == 1


async def test_pinned_models_are_never_the_victim() -> None:
    """A pinned model (ttl 0) is not evictable, so there is nothing to free."""
    supervisor = StubSupervisor(fail_times=1, stderr=OOM_STDERR)
    supervisor.instances["pinned/model"] = resident("pinned/model", ttl_s=0)
    manager = make_manager(supervisor, StubPlanner())

    with pytest.raises(ModelLoadError):
        await manager.load("test/model")

    assert supervisor.stopped == []
    assert supervisor.starts == 1


async def test_the_retry_keeps_the_callers_explicit_overrides() -> None:
    """An explicit ctx_size/kv/parallel must survive the retry's re-plan.

    Without this, a ``load(ctx_size=16384)`` that hit a transient OOM was
    re-planned at the defaults: the load "succeeds", and the 16k context the
    user asked for silently becomes the 8k default -- invisible until a long
    prompt truncates.
    """
    supervisor = StubSupervisor(fail_times=1, stderr=OOM_STDERR)
    supervisor.instances["victim/model"] = resident("victim/model")
    planner = StubPlanner()
    seen: list[dict[str, Any]] = []
    original = planner.plan_load

    def recording(record: ModelRecord, **kwargs: Any) -> Any:
        seen.append(dict(kwargs))
        return original(record, **kwargs)

    planner.plan_load = recording  # type: ignore[method-assign]
    manager = make_manager(supervisor, planner)

    instance = await manager.load("test/model", ctx_size=16384, kv_cache_type="q8_0", parallel=2)

    assert instance.state == "ready"
    assert len(seen) == 2, "the transient failure must trigger exactly one re-plan"
    for index, kwargs in enumerate(seen):
        assert kwargs.get("ctx_size") == 16384, f"plan #{index} dropped the explicit ctx_size"
        assert kwargs.get("kv_cache_type") == "q8_0", f"plan #{index} dropped the kv override"
        assert kwargs.get("parallel") == 2, f"plan #{index} dropped the parallel override"


async def test_a_rejection_after_eviction_surfaces_as_insufficient_vram() -> None:
    """If the re-plan says it still does not fit, say that -- not "load failed"."""
    supervisor = StubSupervisor(fail_times=1, stderr=OOM_STDERR)
    supervisor.instances["victim/model"] = resident("victim/model")
    planner = StubPlanner()
    manager = make_manager(supervisor, planner)

    def reject_second(record: ModelRecord, **kwargs: Any) -> Any:
        planner.calls += 1
        if planner.calls == 1:
            return LoadPlan(model_id=record.id, devices=[0], ctx_size=8192)
        return LoadRejected(model_id=record.id, reason="still no room", required_bytes=1)

    planner.plan_load = reject_second  # type: ignore[method-assign]

    with pytest.raises(InsufficientVramError):
        await manager.load("test/model")

    assert supervisor.starts == 1


# ---------------------------------------------------------------------------
# D29: one load at a time. Two cold loads planned side by side each see the
# VRAM the other has not allocated yet -- the second must plan only after
# the first child has actually come up.
# ---------------------------------------------------------------------------


async def test_a_second_cold_load_plans_only_after_the_first_has_started() -> None:
    import asyncio

    a, b = make_record("test/a"), make_record("test/b")
    order: list[str] = []
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    class GatedSupervisor(StubSupervisor):
        async def start(self, record: ModelRecord, plan: LoadPlan, **kwargs: Any) -> InstanceInfo:
            order.append(f"start:{record.id}")
            if record.id == "test/a":
                first_started.set()
                await release_first.wait()  # a slow cold load
            order.append(f"ready:{record.id}")
            return await super().start(record, plan, **kwargs)

    class OrderedPlanner(StubPlanner):
        def plan_load(self, record: ModelRecord, **kwargs: Any) -> Any:
            order.append(f"plan:{record.id}")
            return super().plan_load(record, **kwargs)

    supervisor = GatedSupervisor()
    manager = ModelManager(
        Config(data_dir="/tmp/sf-gate"),
        registry=StubRegistry({a.id: a, b.id: b}),  # type: ignore[arg-type]
        planner=OrderedPlanner(),  # type: ignore[arg-type]
        supervisor=supervisor,  # type: ignore[arg-type]
        db=None,  # type: ignore[arg-type]
    )

    load_a = asyncio.create_task(manager.ensure_loaded("test/a"))
    await first_started.wait()
    load_b = asyncio.create_task(manager.ensure_loaded("test/b"))
    await asyncio.sleep(0.05)
    assert "plan:test/b" not in order, "b must not be planned while a is still allocating"

    release_first.set()
    await asyncio.gather(load_a, load_b)
    assert order == [
        "plan:test/a",
        "start:test/a",
        "ready:test/a",
        "plan:test/b",
        "start:test/b",
        "ready:test/b",
    ]
    assert supervisor.stopped == [], "neither load evicted the other"


async def test_a_request_during_a_watcher_relaunch_waits_instead_of_replanning() -> None:
    """The supervisor's crash watcher is bringing the child back (state
    "loading"). Planning again launches nothing -- start() hands back the
    in-flight instance -- but could evict bystanders, and the caller would
    forward to a port nobody listens on yet. Wait for it to settle."""
    import asyncio

    supervisor = StubSupervisor()
    planner = StubPlanner()
    manager = make_manager(supervisor, planner)
    manager.SETTLE_POLL_S = 0.01  # type: ignore[misc]
    relaunching = InstanceInfo(
        model_id="test/model",
        state="loading",
        port=18100,
        plan=LoadPlan(model_id="test/model", devices=[0]),
    )
    supervisor.instances["test/model"] = relaunching

    async def come_up() -> None:
        await asyncio.sleep(0.05)
        relaunching.state = "ready"

    asyncio.create_task(come_up())
    record, instance = await manager.ensure_loaded("test/model")
    assert instance.state == "ready"
    assert planner.calls == 0, "an in-flight start is waited for, not re-planned"
    assert supervisor.starts == 0
    assert supervisor.stopped == []


async def test_a_watcher_relaunch_that_fails_falls_through_to_a_fresh_plan() -> None:
    import asyncio

    supervisor = StubSupervisor()
    planner = StubPlanner()
    manager = make_manager(supervisor, planner)
    manager.SETTLE_POLL_S = 0.01  # type: ignore[misc]
    dying = InstanceInfo(model_id="test/model", state="loading", port=18100)
    supervisor.instances["test/model"] = dying

    async def give_up() -> None:
        await asyncio.sleep(0.03)
        dying.state = "failed"
        supervisor.instances.pop("test/model", None)

    asyncio.create_task(give_up())
    _, instance = await manager.ensure_loaded("test/model")
    assert instance.state == "ready"
    assert planner.calls == 1 and supervisor.starts == 1


# ---------------------------------------------------------------------------
# D30: a forced reload plans before it unloads. A refused reload must leave
# the model that was serving a moment ago still serving.
# ---------------------------------------------------------------------------


async def test_a_refused_forced_reload_keeps_the_running_instance() -> None:
    supervisor = StubSupervisor()
    supervisor.instances["test/model"] = resident("test/model")
    planner = StubPlanner(
        result=LoadRejected(model_id="test/model", reason="no room", required_bytes=1)
    )
    manager = make_manager(supervisor, planner)

    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load("test/model", force=True)

    assert supervisor.stopped == [], "the resident child must not be stopped before planning"
    assert supervisor.get("test/model") is not None
    assert supervisor.starts == 0
    assert any("left loaded" in s for s in excinfo.value.details["suggestions"])


async def test_a_forced_reload_tells_the_planner_which_instance_it_replaces() -> None:
    supervisor = StubSupervisor()
    supervisor.instances["test/model"] = resident("test/model")
    planner = StubPlanner()
    seen: dict[str, Any] = {}

    def capture(record: ModelRecord, **kwargs: Any) -> Any:
        planner.calls += 1
        seen.update(kwargs)
        return LoadPlan(model_id=record.id, devices=[0], ctx_size=8192)

    planner.plan_load = capture  # type: ignore[method-assign]
    manager = make_manager(supervisor, planner)

    instance = await manager.load("test/model", force=True)

    assert seen.get("reload_of") == "test/model"
    # Planned first, then the resident stopped, then the replacement started.
    assert supervisor.stopped == ["test/model"]
    assert supervisor.starts == 1
    assert instance.state == "ready"


async def test_a_plain_load_never_passes_reload_of() -> None:
    supervisor = StubSupervisor()
    planner = StubPlanner()
    seen: dict[str, Any] = {}

    def capture(record: ModelRecord, **kwargs: Any) -> Any:
        planner.calls += 1
        seen.update(kwargs)
        return LoadPlan(model_id=record.id, devices=[0], ctx_size=8192)

    planner.plan_load = capture  # type: ignore[method-assign]
    manager = make_manager(supervisor, planner)
    await manager.load("test/model")
    assert seen.get("reload_of") is None
    assert supervisor.stopped == []
