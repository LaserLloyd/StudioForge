"""A load never interrupts a stream, and a smoke test never rearranges the rig.

Live evidence, 2026-08-19 21:11-21:14 on the reference rig. An external client
loaded a 27B at 32768 on [1, 0], then a 31B at **262144 on [1, 0, 2]**, then
fetched ``/profiles`` for every model in the library. Nothing in the log said
who had asked for any of it, and nothing stopped a smoke test from doing a full
planner-sized load, on whatever cards were free, while other agents were mid
conversation.

D36 answers that in three parts, and this file pins each of them:

* every load carries a ``source``, onto the instance and into the log lines;
* the eviction ladder skips a model that is serving, and the refusal says so
  with ``retry_after_s`` -- only an explicit ``force=true`` overrides it;
* ``test_model`` is one-at-a-time, refuses a busy server, loads small, and
  unloads what it loaded.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from studioforge.core.manager import ModelManager
from studioforge.core.planner import BUSY_RETRY_AFTER_S, Planner
from studioforge.errors import InsufficientVramError, ModelBusyError
from studioforge.types import GB, InstanceInfo, LoadPlan
from tests.unit.test_load_retry import (
    StubPlanner,
    StubProbe,
    StubRegistry,
    StubSupervisor,
    make_manager,
    make_record,
    resident,
)
from tests.unit.test_planner import make_config, rig_5090x2_3090x2

# ---------------------------------------------------------------------------
# Who asked
# ---------------------------------------------------------------------------


async def test_a_load_records_its_requester_on_the_instance() -> None:
    supervisor, planner = StubSupervisor(), StubPlanner(probe=StubProbe())
    manager = make_manager(supervisor, planner)
    instance = await manager.load("test/model", source="mcp:load_model")
    assert instance.loaded_by == "mcp:load_model"


async def test_a_jit_load_is_labelled_as_one() -> None:
    supervisor, planner = StubSupervisor(), StubPlanner(probe=StubProbe())
    manager = make_manager(supervisor, planner)
    _record, instance = await manager.ensure_loaded("test/model", source="jit:/v1/chat/completions")
    assert instance.loaded_by == "jit:/v1/chat/completions"


async def test_the_planner_is_told_who_asked() -> None:
    """So the one `load planned` INFO line per load names a requester."""
    supervisor, planner = StubSupervisor(), StubPlanner(probe=StubProbe())
    manager = make_manager(supervisor, planner)
    await manager.load("test/model", source="gui")
    assert planner.kwargs[-1]["source"] == "gui"


# ---------------------------------------------------------------------------
# Eviction never kills a busy model
# ---------------------------------------------------------------------------


def busy_planner(free_gib: float = 4.0) -> Planner:
    return Planner(make_config(), rig_5090x2_3090x2(free_gib), log_plans=False)


def serving(model_id: str, *, requests: int = 1) -> InstanceInfo:
    return InstanceInfo(
        model_id=model_id,
        state="ready",
        port=18101,
        ttl_s=1800,
        started_at=1.0,
        last_activity_at=1.0,
        active_requests=requests,
        plan=LoadPlan(
            model_id=model_id,
            devices=[0, 1],
            ctx_size=32768,
            per_gpu_bytes={0: int(28 * GB), 1: int(28 * GB)},
        ),
    )


def test_a_busy_model_is_not_an_eviction_candidate() -> None:
    planner = busy_planner()
    assert planner._evictable([serving("pub/busy")]) == []


def test_a_loading_model_is_not_an_eviction_candidate() -> None:
    """It has not taken the VRAM its plan promises, so evicting it frees nothing."""
    planner = busy_planner()
    loading = serving("pub/loading", requests=0).model_copy(update={"state": "loading"})
    assert planner._evictable([loading]) == []


def test_force_is_the_one_override() -> None:
    planner = busy_planner()
    victims = planner._evictable([serving("pub/busy")], include_busy=True)
    assert [i.model_id for i in victims] == ["pub/busy"]


def test_a_refusal_blocked_by_a_busy_model_names_it_and_says_how_long() -> None:
    planner = busy_planner(free_gib=4.0)
    result = planner.plan_load(make_record("pub/wanted"), loaded=[serving("pub/busy", requests=3)])
    assert not isinstance(result, LoadPlan)
    assert result.busy_models == [{"model_id": "pub/busy", "active_requests": 3}]
    assert result.retry_after_s == BUSY_RETRY_AFTER_S
    assert "force=true" in result.suggestions[0]
    assert "3 in flight" in result.suggestions[0]


def test_an_ordinary_vram_refusal_carries_no_retry_advice() -> None:
    """ "Try again later" is bad advice when nothing is going to change."""
    planner = busy_planner(free_gib=1.0)
    result = planner.plan_load(make_record("pub/wanted"), loaded=[])
    assert not isinstance(result, LoadPlan)
    assert result.busy_models == []
    assert result.retry_after_s is None


def test_the_manager_carries_the_busy_detail_into_its_507() -> None:
    planner = busy_planner(free_gib=4.0)
    rejected = planner.plan_load(make_record("pub/wanted"), loaded=[serving("pub/busy")])
    manager = ModelManager(
        make_config(),
        registry=StubRegistry({}),  # type: ignore[arg-type]
        planner=planner,
        supervisor=StubSupervisor(),  # type: ignore[arg-type]
        db=None,  # type: ignore[arg-type]
    )
    error = manager._vram_error(rejected)  # type: ignore[arg-type]
    assert isinstance(error, InsufficientVramError)
    assert error.details["busy_models"][0]["model_id"] == "pub/busy"
    assert error.details["retry_after_s"] == BUSY_RETRY_AFTER_S


def test_force_lets_the_plan_evict_the_busy_model() -> None:
    planner = busy_planner(free_gib=4.0)
    plan = planner.plan_load(
        make_record("pub/wanted"), loaded=[serving("pub/busy")], evict_busy=True
    )
    assert isinstance(plan, LoadPlan)
    assert plan.evict_model_ids == ["pub/busy"]


# ---------------------------------------------------------------------------
# busy_snapshot
# ---------------------------------------------------------------------------


def test_busy_snapshot_separates_the_three_kinds_of_busy() -> None:
    supervisor = StubSupervisor()
    supervisor.instances["pub/serving"] = serving("pub/serving", requests=2)
    supervisor.instances["pub/coming"] = serving("pub/coming", requests=0).model_copy(
        update={"state": "loading"}
    )
    manager = make_manager(supervisor, StubPlanner(probe=StubProbe()))
    snapshot = manager.busy_snapshot()
    assert snapshot["active_requests"] == 2
    assert snapshot["busy_models"] == [{"model_id": "pub/serving", "active_requests": 2}]
    assert snapshot["loading"] == ["pub/coming"]
    assert snapshot["testing"] is None


async def test_a_load_queued_behind_the_gate_is_already_visible_as_loading() -> None:
    """``/health.busy.loading`` from the moment a load is asked for (WP22).

    ``_loading`` used to be set only once the D29 gate was taken, so a second
    load waiting on the first was invisible -- and "the box was idle when we
    looked" could be true for a caller who looked in exactly that window.
    """
    supervisor, planner = StubSupervisor(), StubPlanner(probe=StubProbe())
    manager = make_manager(supervisor, planner)
    await manager._load_gate.acquire()  # somebody else is mid-load
    try:
        task = asyncio.create_task(manager.load("test/model", source="mcp:load_model"))
        for _ in range(50):
            await asyncio.sleep(0)
            if "test/model" in manager.busy_snapshot()["loading"]:
                break
        assert manager.busy_snapshot()["loading"] == ["test/model"]
        assert not task.done()
    finally:
        manager._load_gate.release()
    await task
    assert manager.busy_snapshot()["loading"] == []


def test_an_idle_server_is_reported_as_idle() -> None:
    manager = make_manager(StubSupervisor(), StubPlanner(probe=StubProbe()))
    assert manager.busy_snapshot() == {
        "active_requests": 0,
        "busy_models": [],
        "loading": [],
        "testing": None,
    }


# ---------------------------------------------------------------------------
# test_model
# ---------------------------------------------------------------------------


class SmokeSupervisor(StubSupervisor):
    """A supervisor whose children answer the canned smoke-test request."""

    def base_url(self, model_id: str) -> str | None:
        return f"http://127.0.0.1:18100/{model_id}"

    def mark_request_start(self, model_id: str) -> None:
        return None

    def mark_request_end(self, model_id: str) -> None:
        return None


class FakeHttpx:
    """Stands in for the ``httpx`` module ``_run_test`` imports."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    def AsyncClient(self, **_kwargs: Any) -> Any:  # noqa: N802 - mirrors httpx
        outer = self

        class _Client:
            async def __aenter__(self) -> Any:
                return self

            async def __aexit__(self, *_exc: Any) -> None:
                return None

            async def post(self, _url: str, json: Any = None) -> Any:
                outer.calls += 1

                class _Response:
                    @staticmethod
                    def raise_for_status() -> None:
                        return None

                    @staticmethod
                    def json() -> dict[str, Any]:
                        return outer.payload

                return _Response()

        return _Client()


CHAT_REPLY = {
    "choices": [{"message": {"content": "Hello."}}],
    "usage": {"completion_tokens": 2},
}


def smoke_manager() -> tuple[ModelManager, SmokeSupervisor]:
    supervisor = SmokeSupervisor()
    manager = make_manager(supervisor, StubPlanner(probe=StubProbe()))  # type: ignore[arg-type]
    return manager, supervisor


async def run_test_model(manager: ModelManager, **kwargs: Any) -> dict[str, Any]:
    fake = FakeHttpx(CHAT_REPLY)
    original = manager._run_test

    async def patched(*args: Any, **inner: Any) -> dict[str, Any]:
        inner["httpx"] = fake
        return await original(*args, **inner)

    manager._run_test = patched  # type: ignore[method-assign]
    try:
        return await manager.test_model("test/model", **kwargs)
    finally:
        manager._run_test = original  # type: ignore[method-assign]


async def test_a_smoke_test_loads_small_and_unloads_afterwards() -> None:
    """ "Leave the rig as found": a health check must not change what is loaded."""
    manager, supervisor = smoke_manager()
    result = await run_test_model(manager)
    assert result["ok"] is True
    assert result["loaded_for_test"] is True
    assert result["unloaded_after"] is True
    assert result["ctx_size_used"] == manager.config.models.default_ctx
    assert supervisor.list() == []


async def test_keep_loaded_leaves_it_resident() -> None:
    manager, supervisor = smoke_manager()
    result = await run_test_model(manager, keep_loaded=True)
    assert result["unloaded_after"] is False
    assert [i.model_id for i in supervisor.list()] == ["test/model"]


async def test_an_already_loaded_model_is_neither_reloaded_nor_unloaded() -> None:
    manager, supervisor = smoke_manager()
    supervisor.instances["test/model"] = resident("test/model")
    result = await run_test_model(manager)
    assert result["loaded_for_test"] is False
    assert result["unloaded_after"] is False
    assert [i.model_id for i in supervisor.list()] == ["test/model"]


async def test_a_smoke_test_is_refused_while_another_model_is_serving() -> None:
    manager, supervisor = smoke_manager()
    supervisor.instances["pub/other"] = serving("pub/other", requests=1)
    with pytest.raises(ModelBusyError) as excinfo:
        await run_test_model(manager)
    assert "pub/other" in excinfo.value.message
    assert excinfo.value.details["retry_after_s"] == manager.TEST_RETRY_AFTER_S


async def test_testing_a_model_that_is_itself_mid_conversation_is_refused() -> None:
    """Otherwise the smoke test measures the queue instead of the model."""
    manager, supervisor = smoke_manager()
    supervisor.instances["test/model"] = serving("test/model", requests=2)
    with pytest.raises(ModelBusyError) as excinfo:
        await run_test_model(manager)
    assert "test/model" in excinfo.value.message


async def test_a_second_concurrent_test_is_refused_not_queued() -> None:
    """A queued smoke test answers about a server that has since moved."""
    manager, _supervisor = smoke_manager()
    manager._testing = "pub/other"
    with pytest.raises(ModelBusyError) as excinfo:
        await run_test_model(manager)
    assert "already running" in excinfo.value.message


async def test_the_test_is_visible_in_busy_while_it_runs() -> None:
    manager, _supervisor = smoke_manager()
    seen: list[Any] = []
    original = manager._run_test

    async def patched(*args: Any, **inner: Any) -> dict[str, Any]:
        seen.append(manager.busy_snapshot()["testing"])
        inner["httpx"] = FakeHttpx(CHAT_REPLY)
        return await original(*args, **inner)

    manager._run_test = patched  # type: ignore[method-assign]
    await manager.test_model("test/model")
    assert seen == ["test/model"]
    assert manager.busy_snapshot()["testing"] is None


async def test_the_smoke_test_context_never_exceeds_the_trained_window() -> None:
    manager, _supervisor = smoke_manager()
    record = make_record("tiny/model")
    record = record.model_copy(
        update={"meta": record.meta.model_copy(update={"n_ctx_train": 2048})}
    )
    assert manager.smoke_test_ctx(record) == 2048


async def test_an_explicit_pinned_ctx_still_wins() -> None:
    manager, _supervisor = smoke_manager()
    record = make_record("pinned/model")
    record = record.model_copy(
        update={"settings": record.settings.model_copy(update={"ctx_size": 65536})}
    )
    assert manager.smoke_test_ctx(record) == 65536


async def test_the_test_slot_is_released_even_when_the_test_fails() -> None:
    manager, _supervisor = smoke_manager()

    async def boom(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("child died")

    manager._run_test = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await manager.test_model("test/model")
    assert manager.busy_snapshot()["testing"] is None


async def test_a_running_benchmark_refuses_a_smoke_test() -> None:
    """A benchmark loads the model once per mode and rewrites its settings; a
    smoke test landing inside that window measures the benchmark."""

    class Running:
        benchmarking = "pub/under-test"

    manager, _supervisor = smoke_manager()
    manager.benchmarker = Running()
    with pytest.raises(ModelBusyError) as excinfo:
        await run_test_model(manager)
    assert "benchmark" in excinfo.value.message


async def test_the_load_gate_is_held_for_the_whole_test_load() -> None:
    """D29's gate plus its own lock: a test cannot race a concurrent load."""
    manager, _supervisor = smoke_manager()
    assert isinstance(manager._test_gate, asyncio.Lock)
    assert isinstance(manager._load_gate, asyncio.Lock)
