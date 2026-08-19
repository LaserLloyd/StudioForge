"""The concurrency sweep: refuses a busy rig, proves batching, leaves it as found.

No GPU, no engine, no HTTP -- the planner is real (so the placement arithmetic
is the real arithmetic) and everything that touches hardware is substituted.

What these pin, beyond "it runs":

* **it will not measure a busy server**, because on a busy server the numbers
  are the contention rather than the model (the D36 rule, applied where it
  matters most);
* **it leaves the rig as found** -- a model it loaded is unloaded, a model that
  was resident is put back with the plan it had. A benchmark that leaves an
  8-slot child holding VRAM has changed the thing it measured;
* **the engine's decode counter is the control.** N requests to a 1-slot server
  still return N answers, and only the counter tells that run from a batched
  one.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from studioforge.core import parallel_bench
from studioforge.core.benchmark import Benchmarker
from studioforge.core.parallel_bench import ParallelBenchmarker, parse_metrics
from studioforge.errors import BadRequestError, ModelBusyError
from studioforge.types import GB, InstanceInfo, LoadPlan
from tests.unit.test_catalog import NOW, dense_meta, record
from tests.unit.test_planner import make_config, rig_5090x2_3090x2

# ---------------------------------------------------------------------------
# A fake engine over httpx
# ---------------------------------------------------------------------------


METRICS_TEMPLATE = """\
# HELP llamacpp:n_decode_total Total decode calls
# TYPE llamacpp:n_decode_total counter
llamacpp:n_decode_total {decodes}
llamacpp:n_busy_slots_per_decode {busy}
llamacpp:requests_deferred 0
"""


class FakeResponse:
    def __init__(self, *, payload: Any = None, text: str = "") -> None:
        self._payload = payload
        self.text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class FakeEngine:
    """Answers completions and /metrics; counts concurrency as it goes."""

    def __init__(self, *, completion_tokens: int = 64, batch: float = 4.0) -> None:
        self.completion_tokens = completion_tokens
        self.batch = batch
        self.decodes = 0.0
        self.in_flight = 0
        self.peak_in_flight = 0
        self.posts: list[dict[str, Any]] = []
        self.metrics_reads = 0

    async def post(self, url: str, json: dict[str, Any] | None = None) -> FakeResponse:
        self.posts.append(json or {})
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            # A real level takes wall time; without any, the aggregate divides
            # by ~0 and the numbers stop being comparable between levels.
            await asyncio.sleep(0.01)
        finally:
            self.in_flight -= 1
        # Each request contributes its tokens; the engine's decode counter
        # advances by tokens/batch, which is what makes achieved_batch == batch.
        self.decodes += self.completion_tokens / self.batch
        return FakeResponse(
            payload={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {
                    "prompt_tokens": 512,
                    "completion_tokens": self.completion_tokens,
                },
            }
        )

    async def get(self, url: str) -> FakeResponse:
        self.metrics_reads += 1
        return FakeResponse(text=METRICS_TEMPLATE.format(decodes=self.decodes, busy=self.batch))


class FakeClient:
    def __init__(self, engine: FakeEngine) -> None:
        self._engine = engine

    async def __aenter__(self) -> FakeClient:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    async def post(self, url: str, json: dict[str, Any] | None = None) -> FakeResponse:
        return await self._engine.post(url, json=json)

    async def get(self, url: str) -> FakeResponse:
        return await self._engine.get(url)


class FakeHttpx:
    def __init__(self, engine: FakeEngine) -> None:
        self._engine = engine

    def AsyncClient(self, *_args: Any, **_kwargs: Any) -> FakeClient:  # noqa: N802
        return FakeClient(self._engine)

    def Timeout(self, *_args: Any, **_kwargs: Any) -> object:  # noqa: N802
        return object()


# ---------------------------------------------------------------------------
# A fake manager over a real planner
# ---------------------------------------------------------------------------


class StubSupervisor:
    def __init__(self) -> None:
        self.instances: dict[str, InstanceInfo] = {}
        self.marks: list[str] = []

    def get(self, model_id: str) -> InstanceInfo | None:
        return self.instances.get(model_id)

    def list(self) -> list[InstanceInfo]:
        return list(self.instances.values())

    def base_url(self, model_id: str) -> str | None:
        return "http://engine.test" if model_id in self.instances else None

    def mark_request_start(self, model_id: str) -> None:
        self.marks.append(f"start:{model_id}")

    def mark_request_end(self, model_id: str, **_: Any) -> None:
        self.marks.append(f"end:{model_id}")


class StubDb:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record_parallel_observation(self, **fields: Any) -> None:
        self.rows.append(dict(fields))


class StubManager:
    """Enough ModelManager for a sweep, with a REAL planner underneath."""

    def __init__(self, rec: Any, *, free_gib: float = 31.0) -> None:
        from studioforge.core.planner import Planner

        self.config = make_config()
        self.planner = Planner(self.config, rig_5090x2_3090x2(free_gib), log_plans=False)
        self.supervisor = StubSupervisor()
        self.db = StubDb()
        self.record = rec
        self.loads: list[dict[str, Any]] = []
        self.unloads: list[str] = []
        self.busy: str | None = None
        self.benchmarker: Any = None

    # -- what the sweep calls -----------------------------------------
    def busy_snapshot(self) -> dict[str, Any]:
        return {"active_requests": 0, "busy_models": [], "loading": [], "testing": None}

    def _busy_reason(self) -> str | None:
        return self.busy

    def _vram_error(self, rejected: Any) -> Exception:
        from studioforge.errors import InsufficientVramError

        return InsufficientVramError(rejected.message())

    async def load(self, name: str, **kwargs: Any) -> InstanceInfo:
        self.loads.append(dict(kwargs))
        plan = LoadPlan(
            model_id=self.record.id,
            devices=list(kwargs.get("devices") or [0, 1]),
            ctx_size=int(kwargs.get("ctx_size") or 8192),
            ctx_per_slot=int(kwargs.get("ctx_size") or 8192),
            parallel=int(kwargs.get("parallel") or 1),
            kv_cache_type=kwargs.get("kv_cache_type") or "f16",
            kv_cache_type_v=kwargs.get("kv_cache_type_v") or "f16",
        )
        info = InstanceInfo(model_id=self.record.id, state="ready", port=18100, plan=plan)
        self.supervisor.instances[self.record.id] = info
        return info

    async def unload(self, name: str) -> bool:
        self.unloads.append(name)
        return self.supervisor.instances.pop(self.record.id, None) is not None


def make_runner(
    *, free_gib: float = 31.0, engine: FakeEngine | None = None, monkeypatch: Any = None
) -> tuple[ParallelBenchmarker, StubManager, Any, FakeEngine]:
    rec = record("pub/dense-8b", dense_meta(32768), mtime=NOW, size_bytes=8 * GB)
    manager = StubManager(rec, free_gib=free_gib)
    benchmarker = Benchmarker(manager, probe=manager.planner.probe)  # type: ignore[arg-type]
    runner = ParallelBenchmarker(
        manager,  # type: ignore[arg-type]
        benchmarker,
        db=manager.db,
        probe=manager.planner.probe,
    )
    fake_engine = engine or FakeEngine()
    if monkeypatch is not None:
        monkeypatch.setattr(parallel_bench, "httpx", FakeHttpx(fake_engine))
    return runner, manager, rec, fake_engine


# ---------------------------------------------------------------------------
# /metrics parsing
# ---------------------------------------------------------------------------


def test_metrics_parsing_takes_the_llamacpp_samples_and_ignores_the_comments() -> None:
    parsed = parse_metrics(METRICS_TEMPLATE.format(decodes=1234.0, busy=3.9))
    assert parsed["n_decode_total"] == 1234.0
    assert parsed["n_busy_slots_per_decode"] == 3.9


def test_an_unreachable_metrics_endpoint_is_not_an_error() -> None:
    """The throughput numbers are the measurement; the counter is the control."""
    assert parse_metrics("") == {}
    assert parse_metrics("some_other_exporter_metric 1") == {}


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


async def test_a_busy_server_is_refused_with_something_to_wait_for() -> None:
    runner, manager, rec, _engine = make_runner()
    manager.busy = "pub/other (2 in flight) is serving requests"
    with pytest.raises(ModelBusyError) as excinfo:
        await runner.run(rec)
    assert excinfo.value.details["retry_after_s"] == parallel_bench.BUSY_RETRY_AFTER_S
    assert "busy" in excinfo.value.details
    assert manager.loads == []


async def test_a_second_sweep_is_refused_rather_than_queued(monkeypatch: Any) -> None:
    """A measurement that waits its turn describes a machine that has moved."""
    runner, _manager, rec, _engine = make_runner(monkeypatch=monkeypatch)
    async with runner.benchmarker.exclusive(rec.id):
        with pytest.raises(ModelBusyError) as excinfo:
            await runner.run(rec)
    assert excinfo.value.code == "benchmark_busy"


async def test_a_placement_benchmark_and_a_slot_sweep_share_one_lock(monkeypatch: Any) -> None:
    """Two measurement runs at once compete for the resource each is measuring."""
    runner, _manager, rec, _engine = make_runner(monkeypatch=monkeypatch)
    async with runner.benchmarker.exclusive(rec.id):
        assert runner.benchmarker.busy is True
        # ...and a smoke test sees it, because _busy_reason reads `benchmarking`.
        assert runner.benchmarker.benchmarking == rec.id


async def test_an_unknown_hardware_mode_names_the_ones_this_box_has() -> None:
    runner, _manager, rec, _engine = make_runner()
    with pytest.raises(BadRequestError) as excinfo:
        await runner.run(rec, mode="dual_4090")
    assert excinfo.value.param == "mode"
    assert "dual_5090" in excinfo.value.message


async def test_a_bad_kv_cache_type_is_a_400_not_a_planner_refusal() -> None:
    runner, _manager, rec, _engine = make_runner()
    with pytest.raises(BadRequestError) as excinfo:
        await runner.run(rec, kv_cache_type="q3_k")
    assert excinfo.value.param == "kv_cache_type"


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


async def test_the_sweep_loads_once_and_measures_every_level(monkeypatch: Any) -> None:
    runner, manager, rec, engine = make_runner(monkeypatch=monkeypatch)
    report = await runner.run(rec, streams=(1, 2, 4), max_tokens=8)

    assert len(manager.loads) == 1, "one load for the whole sweep, not one per level"
    assert manager.loads[0]["parallel"] == report.parallel_launched
    assert [level.n_streams for level in report.levels] == [1, 2, 4]
    assert engine.peak_in_flight == 4
    for level in report.levels:
        assert level.per_stream_tps and level.per_stream_tps > 0
        assert level.aggregate_tps and level.aggregate_tps > 0
        assert level.p95_latency_s is not None


async def test_the_decode_counter_is_read_as_the_batch_that_really_happened(
    monkeypatch: Any,
) -> None:
    """~N proves batching; ~1.0 at N > 1 means the requests serialized."""
    runner, _manager, rec, _engine = make_runner(
        engine=FakeEngine(batch=4.0), monkeypatch=monkeypatch
    )
    report = await runner.run(rec, streams=(4,), max_tokens=8)
    level = report.levels[0]
    assert level.achieved_batch == pytest.approx(4.0, rel=0.01)


async def test_a_run_that_never_batched_says_so_out_loud(monkeypatch: Any) -> None:
    """Otherwise the recommendation would be "one slot" about a 1-slot launch."""
    runner, _manager, rec, _engine = make_runner(
        engine=FakeEngine(batch=1.0), monkeypatch=monkeypatch
    )
    report = await runner.run(rec, streams=(1, 2), max_tokens=8)
    assert any("serialized rather than batched" in note for note in report.notes)


async def test_levels_above_the_launched_slot_count_are_dropped_with_a_reason(
    monkeypatch: Any,
) -> None:
    """N requests against fewer than N slots measure a queue, not batching."""
    runner, _manager, rec, _engine = make_runner(monkeypatch=monkeypatch)
    report = await runner.run(rec, streams=(1, 2, 4, 8), max_tokens=8)
    measured = {level.n_streams for level in report.levels}
    assert max(measured) <= report.parallel_launched
    if report.parallel_launched < 8:
        assert any("measure a queue" in note for note in report.notes)


async def test_every_measured_level_is_recorded_for_the_catalog_to_read(
    monkeypatch: Any,
) -> None:
    runner, manager, rec, _engine = make_runner(monkeypatch=monkeypatch)
    report = await runner.run(rec, streams=(1, 2), max_tokens=8)
    assert len(manager.db.rows) == len(report.levels)
    row = manager.db.rows[0]
    assert row["model_id"] == rec.id
    assert row["run_id"] == report.run_id
    assert row["devices"] == ",".join(str(d) for d in sorted(report.devices))
    assert row["ctx_per_slot"] == report.ctx_per_slot
    assert row["engine_tag"] == rec.settings.engine_tag


async def test_the_run_applies_the_rule_to_its_own_numbers(monkeypatch: Any) -> None:
    runner, _manager, rec, _engine = make_runner(monkeypatch=monkeypatch)
    report = await runner.run(rec, streams=(1, 2), max_tokens=8)
    assert report.recommended_parallel_basis == "measured"
    assert 1 <= report.recommended_parallel <= report.max_parallel
    assert report.recommended_parallel_detail


async def test_a_cancel_between_levels_stops_the_sweep(monkeypatch: Any) -> None:
    runner, _manager, rec, _engine = make_runner(monkeypatch=monkeypatch)
    cancel = asyncio.Event()
    cancel.set()
    report = await runner.run(rec, streams=(1, 2, 4), max_tokens=8, cancel_event=cancel)
    assert report.levels == []
    assert any("canceled after" in note for note in report.notes)


# ---------------------------------------------------------------------------
# Leave the rig as found
# ---------------------------------------------------------------------------


async def test_a_model_this_run_loaded_is_unloaded_again(monkeypatch: Any) -> None:
    runner, manager, rec, _engine = make_runner(monkeypatch=monkeypatch)
    report = await runner.run(rec, streams=(1,), max_tokens=8)
    assert report.loaded_for_benchmark is True
    assert report.unloaded_after is True
    assert manager.unloads == [rec.id]
    assert manager.supervisor.instances == {}


async def test_a_model_that_was_resident_is_put_back_with_the_plan_it_had(
    monkeypatch: Any,
) -> None:
    """The run displaced it to get its own slot count; it does not get to keep it."""
    runner, manager, rec, _engine = make_runner(monkeypatch=monkeypatch)
    before = LoadPlan(
        model_id=rec.id,
        devices=[2, 3],
        ctx_size=16384,
        ctx_per_slot=16384,
        parallel=1,
        kv_cache_type="q8_0",
        kv_cache_type_v="q8_0",
    )
    manager.supervisor.instances[rec.id] = InstanceInfo(
        model_id=rec.id, state="ready", port=18100, plan=before
    )

    report = await runner.run(rec, streams=(1, 2), max_tokens=8)
    assert report.loaded_for_benchmark is True
    assert report.restored is True
    assert manager.unloads == []
    restore = manager.loads[-1]
    assert restore["devices"] == [2, 3]
    assert restore["ctx_size"] == 16384
    assert restore["parallel"] == 1
    assert restore["kv_cache_type"] == "q8_0"
    assert restore["source"] == "benchmark:parallel-restore"


async def test_a_resident_instance_that_already_serves_the_sweep_is_not_reloaded(
    monkeypatch: Any,
) -> None:
    """Same cards, same context, enough slots: reloading would only cost a minute."""
    runner, manager, rec, _engine = make_runner(monkeypatch=monkeypatch)
    # Ask for a specific placement so the expected plan is knowable, then make
    # the resident instance exactly that.
    probe_report = await runner.run(rec, streams=(1,), max_tokens=8)
    manager.loads.clear()
    manager.unloads.clear()
    manager.supervisor.instances[rec.id] = InstanceInfo(
        model_id=rec.id,
        state="ready",
        port=18100,
        plan=LoadPlan(
            model_id=rec.id,
            devices=list(probe_report.devices),
            ctx_size=probe_report.ctx_per_slot,
            ctx_per_slot=probe_report.ctx_per_slot,
            parallel=probe_report.parallel_launched,
            kv_cache_type=probe_report.kv_cache_type,
            kv_cache_type_v=probe_report.kv_cache_type_v,
        ),
    )
    report = await runner.run(rec, streams=(1,), max_tokens=8)
    assert report.loaded_for_benchmark is False
    assert manager.loads == []
    assert manager.unloads == []
    assert any("nothing was reloaded" in note for note in report.notes)
    assert manager.supervisor.instances[rec.id].plan is not None


async def test_a_failed_sweep_still_puts_the_rig_back(monkeypatch: Any) -> None:
    """The teardown is in a finally, so an engine that dies mid-sweep is not a leak."""

    class Exploding(FakeEngine):
        async def post(self, url: str, json: dict[str, Any] | None = None) -> FakeResponse:
            raise RuntimeError("boom")

    runner, manager, rec, _engine = make_runner(engine=Exploding(), monkeypatch=monkeypatch)
    report = await runner.run(rec, streams=(1,), max_tokens=8)
    assert report.levels[0].error is not None
    assert manager.unloads == [rec.id]
