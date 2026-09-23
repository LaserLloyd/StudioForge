"""Every load path refuses an unloadable architecture before it does anything (D66).

The live sequence this replaces, 2026-09-20..22 (three times): CrucibleForge
leased CUDA [0, 1] for a K2-Horizon benchmark -- a lease grant unloads the idle
residents on those cards -- then ``load-recommended`` walked the modes, planned,
composed the argv, spawned, and the child died 0.3 s into startup with
``unknown model architecture: 'k2-horizon'``. The client got ``502
model_load_failed``, which reads like a fault to report, and nothing remembered
the answer.

What these pin:

* **one helper, first**: ``load``, ``ensure_loaded``, ``load_recommended``
  (and its dry run), ``plan_preview``, ``lease_check``, ``acquire_lease`` and
  ``_load_locked`` refuse with ``400 unsupported_architecture`` before any hold,
  queue, plan, lease, eviction or spawn -- and a ready resident handed back
  untouched is never refused;
* **the explicit tier is remembered only after a load succeeds** (review item
  19): a refused or failed load no longer re-tiers the model;
* **the background passes skip it** -- pin reconciler, rebalancer, boot
  autoload -- with one WARNING, not a backoff that spawns forever;
* **the runtime memo**: a startup death with either marker becomes the typed
  400 and is remembered per (file, mtime, build) until the file, the build or
  the scan changes; any other death is not;
* **every surface says so**: ``/api/models``, ``/v1/models``, the catalog,
  MCP ``model_info`` and ``plan_load``, ``/api/capabilities``, the GUI
  helpers, the Chat tab's ``unsupported_reason`` -- and a streaming request is
  a real 400 before any SSE byte.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.api import app as app_module
from studioforge.api.app import build_state, create_app
from studioforge.config import Config
from studioforge.core import manager as manager_mod
from studioforge.core.engine import ARCH_TABLE_CANARIES, ArchitectureTable
from studioforge.core.manager import ModelManager
from studioforge.errors import (
    InsufficientVramError,
    ModelLoadError,
    UnsupportedArchitectureError,
)
from studioforge.gui import state as st
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
from tests.unit.test_arch_probe import make_engine
from tests.unit.test_catalog import catalog_for
from tests.unit.test_catalog_routes import FakeProbe, FakeRegistry
from tests.unit.test_catalog_routes import make_record as route_record

K2 = "infini/K2-Horizon-Q6_K"
OK = "pub/Qwen3.5-27B-Q5_K_M"
ACTIVE = "b11037"

K2_TAIL = [
    "0.00.129.726 I srv    load_model: loading model 'K2-Horizon-Q6_K.gguf'",
    "0.00.323.191 E llama_model_load: error loading model: unknown model architecture: "
    "'k2-horizon'",
    "0.00.324.614 E srv  llama_server: exiting due to model loading error",
]
PRE_TOKENIZER_TAIL = [
    "llama_model_load: error loading model: error loading model vocabulary: unknown "
    "pre-tokenizer type: 'moonshot-v9'",
]
MISSING_FILE_TAIL = [
    "gguf_init_from_file: failed to open GGUF file 'x.gguf' (No such file or directory)",
]
CONFIG_TAIL = ['error while handling argument "--bogus": unknown argument']


def table(*names: str, signature: tuple[int, int] = (1, 100)) -> ArchitectureTable:
    blob = b"".join(name.encode() + b"\x00" for name in (*ARCH_TABLE_CANARIES, *names))
    return ArchitectureTable.from_bytes(blob, library="llama.dll", signature=signature)


#: What the live b11037 library says, for the two model families these tests use.
LIVE_TABLE = table("qwen35", "qwen35moe", "gemma4")


def make_record(
    model_id: str,
    architecture: str,
    *,
    path: Path | str = "/models/model.gguf",
    settings: ModelSettings | None = None,
) -> ModelRecord:
    return ModelRecord(
        id=model_id,
        name=model_id,
        path=Path(path),
        size_bytes=8 * GB,
        architecture=architecture,
        meta=GgufMeta(architecture=architecture, n_layer=32, n_head=32, n_head_kv=8, n_embd=4096),
        settings=settings or ModelSettings(),
    )


class StubRegistry:
    def __init__(self, records: Sequence[ModelRecord]) -> None:
        self._records = {r.id: r for r in records}
        self.last_scan_at: float | None = 5.0

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


class StubProbe:
    def list_gpus(self) -> list[GpuInfo]:
        return [
            GpuInfo(index=i, name=f"GPU{i}", total_bytes=32 * GB, free_bytes=30 * GB)
            for i in (0, 1)
        ]


class StubPlanner:
    """Plans everything onto CUDA 0 (or refuses with ``result``); counts every ask."""

    def __init__(self, result: Any = None) -> None:
        self.calls = 0
        self.result = result
        self._probe = StubProbe()

    def plan_load(self, record: ModelRecord, **kwargs: Any) -> Any:
        self.calls += 1
        if self.result is not None:
            return self.result
        return LoadPlan(model_id=record.id, devices=[0], ctx_size=8192)

    def _evictable(
        self,
        loaded: list[InstanceInfo],
        *,
        include_busy: bool = False,
        for_priority: int | None = None,
    ) -> list[InstanceInfo]:
        return [i for i in loaded if i.ttl_s != 0 and i.active_requests == 0]

    @property
    def probe(self) -> Any:
        return self._probe


class StubSupervisor:
    """A supervisor that knows which build a launch lands on and what it can load.

    ``tables`` maps a requested tag (``None`` = the active build) to that
    build's architecture table; a start fails ``fail_times`` times with
    ``stderr`` as its log tail.
    """

    def __init__(
        self,
        tables: dict[str | None, ArchitectureTable | None] | None = None,
        *,
        fail_times: int = 0,
        stderr: list[str] | None = None,
    ) -> None:
        self.tables = {None: LIVE_TABLE} if tables is None else tables
        self.fail_times = fail_times
        self.stderr = stderr or []
        self.starts = 0
        self.stopped: list[str] = []
        self.instances: dict[str, InstanceInfo] = {}

    def architecture_table(self, tag: str | None) -> ArchitectureTable | None:
        return self.tables.get(tag)

    def resolved_engine_tag(self, tag: str | None) -> str | None:
        return tag or ACTIVE

    async def start(self, record: ModelRecord, plan: LoadPlan, **kwargs: Any) -> InstanceInfo:
        self.starts += 1
        if self.starts <= self.fail_times:
            raise ModelLoadError(
                f"llama-server for '{record.id}' exited with code 1 during startup. "
                "Last output:\n" + "\n".join(self.stderr),
                details={"exit_code": 1, "stderr": list(self.stderr), "argv": ["llama-server"]},
            )
        info = InstanceInfo(
            model_id=record.id,
            state="ready",
            port=18100,
            plan=plan,
            priority=kwargs.get("priority", 3),
            loaded_by=kwargs.get("source"),
        )
        self.instances[record.id] = info
        return info

    async def stop(self, model_id: str, **kwargs: Any) -> None:
        self.stopped.append(model_id)
        self.instances.pop(model_id, None)

    def get(self, model_id: str) -> InstanceInfo | None:
        return self.instances.get(model_id)

    def list(self) -> list[InstanceInfo]:
        return list(self.instances.values())


class RecordingLog:
    """Stands in for a module's structlog logger (see test_rejection_log_levels)."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str, dict[str, Any]]] = []

    def _record(self, level: str) -> Any:
        return lambda event, **fields: self.lines.append((level, event, fields))

    def __getattr__(self, name: str) -> Any:
        if name in {"debug", "info", "warning", "error", "exception"}:
            return self._record(name)
        raise AttributeError(name)

    def events(self, level: str, event: str) -> list[dict[str, Any]]:
        return [fields for lvl, ev, fields in self.lines if lvl == level and ev == event]


@pytest.fixture()
def manager_log(monkeypatch: pytest.MonkeyPatch) -> RecordingLog:
    recorder = RecordingLog()
    monkeypatch.setattr(manager_mod, "log", recorder)
    return recorder


def make_manager(
    records: Sequence[ModelRecord],
    supervisor: StubSupervisor | None = None,
    planner: StubPlanner | None = None,
) -> tuple[ModelManager, StubSupervisor, StubPlanner]:
    supervisor = supervisor or StubSupervisor()
    planner = planner or StubPlanner()
    manager = ModelManager(
        Config(data_dir="/tmp/sf-arch"),
        registry=StubRegistry(records),  # type: ignore[arg-type]
        planner=planner,  # type: ignore[arg-type]
        supervisor=supervisor,  # type: ignore[arg-type]
        db=None,  # type: ignore[arg-type]
    )
    return manager, supervisor, planner


def k2(**kwargs: Any) -> ModelRecord:
    return make_record(K2, "k2-horizon", **kwargs)


def qwen(**kwargs: Any) -> ModelRecord:
    return make_record(OK, "qwen35", **kwargs)


def resident(model_id: str, *, devices: Sequence[int] = (0,), last: float = 1.0) -> InstanceInfo:
    return InstanceInfo(
        model_id=model_id,
        state="ready",
        port=18101,
        ttl_s=1800,
        started_at=last,
        last_activity_at=last,
        plan=LoadPlan(model_id=model_id, devices=list(devices), ctx_size=8192),
    )


def assert_nothing_happened(manager: ModelManager, sup: StubSupervisor, plan: StubPlanner) -> None:
    assert plan.calls == 0, "the planner was asked"
    assert sup.starts == 0, "a child was spawned"
    assert sup.stopped == [], "a resident was stopped"
    assert manager._loading == set(), "the load was marked in flight"
    assert manager._priority_holds == {}, "worse-tier traffic was held off"
    assert manager.leases.all() == [], "a lease was granted"


# ---------------------------------------------------------------------------
# Every entry refuses first
# ---------------------------------------------------------------------------


async def test_a_jit_request_is_refused_before_anything_happens() -> None:
    manager, sup, plan = make_manager([k2()])
    with pytest.raises(UnsupportedArchitectureError) as caught:
        await manager.ensure_loaded(K2, priority=1)
    assert caught.value.status_code == 400
    assert caught.value.param == "model"
    assert caught.value.details["architecture"] == "k2-horizon"
    assert caught.value.details["engine_tag"] == ACTIVE
    assert caught.value.details["source"] == "binary"
    assert_nothing_happened(manager, sup, plan)


async def test_an_explicit_load_is_refused_and_does_not_retier_the_model() -> None:
    manager, sup, plan = make_manager([k2()])
    with pytest.raises(UnsupportedArchitectureError):
        await manager.load(K2, priority=1, force=True)
    assert manager._model_priority == {}, "a refused load re-tiered the model"
    assert_nothing_happened(manager, sup, plan)


async def test_a_ready_resident_is_handed_back_but_never_reloaded_onto_a_build_without_it() -> None:
    """A resident runs on the build it was launched from. Handing it back
    launches nothing; a forced reload launches on today's build, so it is
    refused -- and the running child keeps serving."""
    manager, sup, plan = make_manager([k2()])
    sup.instances[K2] = resident(K2)

    assert await manager.load(K2) is sup.instances[K2]
    record, instance = await manager.ensure_loaded(K2)
    assert instance is sup.instances[K2]
    with pytest.raises(UnsupportedArchitectureError):
        await manager.load(K2, force=True)
    assert sup.stopped == [], "the resident was stopped for a reload that could not launch"
    assert plan.calls == 0


async def test_load_recommended_and_its_dry_run_refuse_before_the_walk() -> None:
    manager, sup, plan = make_manager([k2()])
    with pytest.raises(UnsupportedArchitectureError):
        await manager.load_recommended(K2, 32768, priority=1)
    with pytest.raises(UnsupportedArchitectureError):
        await manager.plan_recommended(K2, 32768)
    assert manager._model_priority == {}
    assert_nothing_happened(manager, sup, plan)


def test_the_plan_preview_reports_the_refusal_instead_of_a_fit() -> None:
    manager, sup, plan = make_manager([k2()])
    preview = manager.plan_preview(K2)
    assert preview["fits"] is False
    assert preview["dry_run"] is True
    assert preview["reason_code"] == "unsupported_architecture"
    assert preview["status_code"] == 400
    assert preview["error"]["code"] == "unsupported_architecture"
    assert preview["architecture"] == "k2-horizon"
    assert preview["engine_tag"] == ACTIVE
    assert plan.calls == 0

    verdict = st.fit_verdict(preview)
    assert verdict.fits is False
    assert verdict.headline == "Cannot load: unsupported architecture"
    assert verdict.detail_lines == [preview["message"]]
    assert verdict.per_gpu == [] and verdict.suggestions == []


def test_the_pre_stream_checks_refuse_it() -> None:
    manager, sup, plan = make_manager([k2()])
    with pytest.raises(UnsupportedArchitectureError):
        manager.lease_check(K2)
    with pytest.raises(UnsupportedArchitectureError):
        manager.arch_check(K2)
    manager.arch_check("nobody/unknown-model")  # the caller's own resolution says 404


async def test_a_lease_for_a_model_that_cannot_load_evicts_nobody() -> None:
    manager, sup, plan = make_manager([k2(), qwen(), make_record("pub/embed", "qwen2")])
    sup.instances["pub/embed"] = resident("pub/embed", devices=(0,))

    with pytest.raises(UnsupportedArchitectureError):
        await manager.acquire_lease([0], holder="crucibleforge", model_ids=[K2])
    assert sup.stopped == [], "the embedding model was evicted for a model that cannot load"
    assert manager.leases.all() == []

    # The control: the same lease for a loadable model does evict the idle resident.
    lease = await manager.acquire_lease([0], holder="crucibleforge", model_ids=[OK])
    assert sup.stopped == ["pub/embed"]
    manager.release_lease(lease.id)


async def test_load_locked_refuses_before_the_load_is_in_flight() -> None:
    manager, sup, plan = make_manager([k2()])
    with pytest.raises(UnsupportedArchitectureError):
        await manager._load_locked(k2(), priority=1, hold=True)
    assert_nothing_happened(manager, sup, plan)


async def test_benchmark_runners_refuse_before_their_first_lease() -> None:
    from studioforge.core.benchmark import Benchmarker
    from studioforge.core.parallel_bench import ParallelBenchmarker

    manager, sup, plan = make_manager([k2()])
    benchmarker = Benchmarker(manager)
    with pytest.raises(UnsupportedArchitectureError):
        await benchmarker.run(k2())
    with pytest.raises(UnsupportedArchitectureError):
        await ParallelBenchmarker(manager, benchmarker=benchmarker).run(k2())
    assert_nothing_happened(manager, sup, plan)


async def test_cannot_tell_never_refuses() -> None:
    """No library, an unreadable one, a stand-in without the accessor: load."""
    manager, sup, plan = make_manager([k2()], StubSupervisor({None: None}))
    instance = await manager.load(K2)
    assert instance.state == "ready"
    assert manager.arch_verdict(k2()).supported is None
    assert manager.unsupported_reason(k2()) is None


async def test_a_pinned_build_without_it_names_the_fix_when_the_active_build_has_it() -> None:
    pinned = k2(settings=ModelSettings(engine_tag="b10425"))
    sup = StubSupervisor({"b10425": table("qwen35"), None: table("qwen35", "k2-horizon")})
    manager, _, plan = make_manager([pinned], sup)
    with pytest.raises(UnsupportedArchitectureError) as caught:
        await manager.load(K2)
    details = caught.value.details
    assert details["engine_tag"] == "b10425"
    assert details["engine_tag_pinned"] is True
    assert details["active_engine_tag"] == ACTIVE
    assert details["active_engine_supports"] is True
    assert "clear the model's engine_tag" in caught.value.message
    assert plan.calls == 0


# ---------------------------------------------------------------------------
# The explicit tier is remembered only after a load succeeds (review item 19)
# ---------------------------------------------------------------------------


async def test_the_tier_is_remembered_only_once_the_load_has_succeeded() -> None:
    manager, sup, plan = make_manager([qwen()], StubSupervisor(fail_times=1, stderr=CONFIG_TAIL))
    with pytest.raises(ModelLoadError):
        await manager.load(OK, priority=1)
    assert OK not in manager._model_priority, "a load that died re-tiered the model"

    instance = await manager.load(OK, priority=1)
    assert instance.priority == 1
    assert manager._model_priority[OK] == 1


async def test_a_planner_refusal_does_not_retier_the_model_either() -> None:
    refusal = LoadRejected(model_id=OK, reason="does not fit", required_bytes=1, available_bytes=0)
    manager, sup, plan = make_manager([qwen()], planner=StubPlanner(result=refusal))
    with pytest.raises(InsufficientVramError):
        await manager.load(OK, priority=1)
    assert manager._model_priority == {}


async def test_re_tiering_a_ready_resident_is_still_remembered() -> None:
    manager, sup, plan = make_manager([qwen()])
    sup.instances[OK] = resident(OK)
    assert (await manager.load(OK, priority=2)).priority == 2
    assert manager._model_priority[OK] == 2


# ---------------------------------------------------------------------------
# The background passes skip it, and say so once
# ---------------------------------------------------------------------------


def test_the_pin_reconciler_skips_it_and_warns_once(manager_log: RecordingLog) -> None:
    pinned = ModelSettings(pinned=True)
    manager, sup, plan = make_manager([k2(settings=pinned), qwen(settings=pinned)])
    assert manager._pinned_needing_load() == [OK]
    assert manager._pinned_needing_load() == [OK]
    warnings = manager_log.events("warning", "unsupported architecture: not loading")
    assert len(warnings) == 1, "a standing refusal is one WARNING, not one per sweep"
    assert warnings[0]["model_id"] == K2
    assert warnings[0]["by"] == "pin reconciler"
    assert manager._pin_retry == {}, "skipped, not backed off into ~96 doomed spawns a day"


def test_the_rebalancer_leaves_it_where_it_is() -> None:
    """A move is a reload on today's build: a resident that build cannot
    load is not a candidate, however badly placed it is."""
    manager, sup, plan = make_manager([k2(), qwen()])
    sup.instances[K2] = resident(K2, devices=(0, 1))
    assert manager._rebalance_opportunity() is None
    assert plan.calls == 0

    sup.instances.clear()
    sup.instances[OK] = resident(OK, devices=(0, 1))
    found = manager._rebalance_opportunity()
    assert found is not None and found[0] == OK, "the control: a loadable resident is moved"


async def test_boot_autoload_skips_it_without_seeding_a_backoff(manager_log: RecordingLog) -> None:
    manager, sup, plan = make_manager([k2(settings=ModelSettings(pinned=True))])
    await manager._autoload_pinned()
    assert sup.starts == 0 and plan.calls == 0
    assert manager._pin_retry == {}
    assert len(manager_log.events("warning", "unsupported architecture: not loading")) == 1
    assert manager_log.events("warning", "failed to preload model") == []


# ---------------------------------------------------------------------------
# The runtime memo
# ---------------------------------------------------------------------------


def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "K2-Horizon-Q6_K.gguf"
    path.write_bytes(b"GGUF" + b"\x00" * 60)
    return path


def bump_mtime(path: Path) -> None:
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 9_000_000_000))


async def test_a_startup_death_of_unknown_architecture_is_typed_and_remembered(
    tmp_path: Path,
) -> None:
    record = k2(path=model_file(tmp_path))
    # The library cannot tell (None), so the first launch really happens.
    sup = StubSupervisor({None: None}, fail_times=5, stderr=K2_TAIL)
    manager, _, plan = make_manager([record], sup)

    with pytest.raises(UnsupportedArchitectureError) as first:
        await manager.load(K2)
    assert sup.starts == 1 and plan.calls == 1
    details = first.value.details
    assert details["source"] == "runtime"
    assert isinstance(details["first_failed_at"], float)
    assert details["stderr"] == K2_TAIL
    assert "rejected at startup ('unknown model architecture')" in first.value.message

    # Remembered: no second plan, no second spawn -- for any path.
    with pytest.raises(UnsupportedArchitectureError) as again:
        await manager.ensure_loaded(K2)
    with pytest.raises(UnsupportedArchitectureError):
        await manager.load_recommended(K2, 32768)
    assert sup.starts == 1 and plan.calls == 1
    assert again.value.details["first_failed_at"] == details["first_failed_at"]

    fields = manager.load_support_fields(record)
    assert fields["arch_supported"] is False
    assert fields["engine_supported"] is False
    assert "rejected at startup" in fields["unsupported_reason"]
    assert fields["last_load_failure"]["code"] == "unsupported_architecture"
    assert fields["last_load_failure"]["engine_tag"] == ACTIVE
    assert manager.unsupported_reason(record) == (
        f"llama.cpp build {ACTIVE} does not include the 'k2-horizon' architecture"
    )


async def test_the_memo_lapses_when_the_model_file_changes(tmp_path: Path) -> None:
    path = model_file(tmp_path)
    sup = StubSupervisor({None: None}, fail_times=1, stderr=K2_TAIL)
    manager, _, _ = make_manager([k2(path=path)], sup)
    with pytest.raises(UnsupportedArchitectureError):
        await manager.load(K2)
    bump_mtime(path)  # a re-download
    instance = await manager.load(K2)
    assert sup.starts == 2 and instance.state == "ready"
    assert manager.load_support_fields(k2(path=path)).get("last_load_failure") is None


async def test_a_rescan_or_an_engine_change_forgets_it(tmp_path: Path) -> None:
    record = k2(path=model_file(tmp_path))
    sup = StubSupervisor({None: None}, fail_times=2, stderr=K2_TAIL)
    manager, _, _ = make_manager([record], sup)

    with pytest.raises(UnsupportedArchitectureError):
        await manager.load(K2)
    manager.registry.last_scan_at = 6.0  # type: ignore[attr-defined]
    with pytest.raises(UnsupportedArchitectureError):
        await manager.load(K2)
    assert sup.starts == 2, "a rescan forgets the rejection: the next load launches again"

    assert manager.forget_arch_rejections("b11100", "activate") == 1
    assert manager.arch_verdict(record).supported is None


async def test_a_reinstalled_build_forgets_it(tmp_path: Path) -> None:
    """The library DOES name the architecture here -- the build rejects the
    model's pre-tokenizer, which no library can be asked about -- so only the
    runtime memo refuses, until the library's signature moves."""
    record = k2(path=model_file(tmp_path))
    sup = StubSupervisor(
        {None: table("k2-horizon", signature=(1, 100))},
        fail_times=1,
        stderr=PRE_TOKENIZER_TAIL,
    )
    manager, _, _ = make_manager([record], sup)
    with pytest.raises(UnsupportedArchitectureError) as caught:
        await manager.load(K2)
    assert caught.value.details["rejected"] == {"kind": "pre_tokenizer", "name": "moonshot-v9"}
    assert "the pre-tokenizer 'moonshot-v9'" in caught.value.message
    with pytest.raises(UnsupportedArchitectureError):
        await manager.load(K2)
    assert sup.starts == 1

    sup.tables[None] = table("k2-horizon", signature=(2, 100))
    assert (await manager.load(K2)).state == "ready"
    assert sup.starts == 2


async def test_a_missing_file_is_not_an_unsupported_model(tmp_path: Path) -> None:
    record = k2(path=model_file(tmp_path))
    sup = StubSupervisor({None: None}, fail_times=1, stderr=MISSING_FILE_TAIL)
    manager, _, _ = make_manager([record], sup)
    with pytest.raises(ModelLoadError) as caught:
        await manager.load(K2)
    assert not isinstance(caught.value, UnsupportedArchitectureError)
    failure = manager.load_support_fields(record)["last_load_failure"]
    assert failure["code"] == "model_load_failed"
    assert failure["message"].startswith(f"llama-server for '{K2}' exited with code 1")
    assert manager.arch_verdict(record).supported is None

    await manager.load(K2)
    assert sup.starts == 2, "nothing was remembered, so the next load launched"
    assert "last_load_failure" not in manager.load_support_fields(record)


def test_the_engine_change_hook_is_wired_to_the_manager(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path / "data")
    state = build_state(config)
    assert state.engine_manager.on_engine_change == state.manager.forget_arch_rejections


# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------


def test_the_catalog_has_nothing_to_recommend_and_says_fits_now_false() -> None:
    from tests.unit.test_catalog import library

    records = library()
    unsupported_id = records[0].id

    def support(record: ModelRecord) -> dict[str, Any]:
        if record.id == unsupported_id:
            return {
                "arch_supported": False,
                "arch_note": "llama.cpp build b11037 does not include the 'x' architecture",
                "engine_supported": False,
                "unsupported_reason": "full refusal",
            }
        return {"arch_supported": True}

    full = {m["id"]: m for m in catalog_for(records, load_support=support)["models"]}
    entry = full[unsupported_id]
    assert entry["arch_supported"] is False
    assert entry["engine_supported"] is False
    assert entry["unsupported_reason"] == "full refusal"
    assert entry["fits_now"] is False
    assert entry["fits_now_basis"].startswith("unsupported_architecture: ")
    assert entry["recommended"] is None
    assert entry["recommended_basis"].startswith("not recommended: ")
    assert entry["placements"] == [] and entry["options"] == []
    assert entry["unavailable"].startswith("cannot be loaded: ")
    others = [m for mid, m in full.items() if mid != unsupported_id]
    assert all(m["arch_supported"] is True and m["recommended"] for m in others)

    compact = {
        m["id"]: m for m in catalog_for(records, load_support=support, compact=True)["models"]
    }
    assert compact[unsupported_id]["arch_supported"] is False
    assert all("arch_supported" not in m for mid, m in compact.items() if mid != unsupported_id), (
        "a loadable row pays nothing in the compact view"
    )


def test_the_gui_badge_and_load_refusal() -> None:
    assert st.arch_badge(None) is None
    assert st.arch_badge({"arch_supported": True}) is None
    assert st.arch_badge({"arch_supported": None}) is None, "cannot tell earns no badge"
    assert st.arch_badge({"arch_supported": False, "arch_note": "short"}) == (
        st.UNSUPPORTED_ARCH_BADGE,
        "short",
    )
    full = {"arch_supported": False, "arch_note": "short", "arch_message": "the whole reason"}
    assert st.arch_badge(full) == ("Unsupported arch", "the whole reason")
    assert st.arch_load_refusal(full) == "the whole reason"
    assert st.arch_load_refusal({"arch_supported": True}) is None


def test_the_chat_tabs_question_has_a_short_answer() -> None:
    manager, _, _ = make_manager([k2(), qwen()])
    assert manager.unsupported_reason(k2()) == (
        f"llama.cpp build {ACTIVE} does not include the 'k2-horizon' architecture"
    )
    assert manager.unsupported_reason(qwen()) is None
    verdicts = manager.arch_verdicts([k2(), qwen()])
    assert verdicts[K2].supported is False and verdicts[OK].supported is True


# -- through the real app, with a fake engine on disk -------------------------

ROUTE_OK = "vendor/thing-Q4_K_M"  # architecture "llama"
ROUTE_K2 = "infini/K2-Horizon-GGUF/K2-Horizon-Q6_K"
MESSAGES = [{"role": "user", "content": "hello"}]


class Registry(FakeRegistry):
    def touch(self, model_id: str) -> None:
        return None


def _route_k2() -> ModelRecord:
    base = route_record(ROUTE_K2)
    assert base.meta is not None
    return base.model_copy(
        update={
            "architecture": "k2-horizon",
            "meta": base.meta.model_copy(update={"architecture": "k2-horizon"}),
        }
    )


@pytest.fixture()
def arch_app(tmp_path: Path) -> Any:
    config = Config(
        data_dir=tmp_path / "data",
        server={"host": "127.0.0.1", "port": 1234},
        models={"dir": tmp_path / "models"},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
    )
    state = build_state(config)
    # A build whose llama.dll names llama (and the canaries) but not k2-horizon.
    make_engine(config.engines_dir, "b99999", "qwen35")
    state.engine_manager.set_active("b99999")
    built = create_app(config, state=state, start_background=False)
    built.state.registry = Registry([route_record(ROUTE_OK), _route_k2()])
    built.state.probe = FakeProbe()
    built.state.planner.probe = FakeProbe()
    built.state.manager.registry = built.state.registry
    return built


def _error(response: Any) -> dict[str, Any]:
    assert response.status_code == 400, response.text
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()["error"]
    assert body["code"] == "unsupported_architecture"
    assert body["param"] == "model"
    assert body["studioforge"]["architecture"] == "k2-horizon"
    assert body["studioforge"]["engine_tag"] == "b99999"
    assert body["studioforge"]["source"] == "binary"
    return body


def test_a_streaming_chat_is_a_400_before_any_sse_byte(arch_app: Any) -> None:
    with TestClient(arch_app) as http:
        streamed = http.post(
            "/v1/chat/completions",
            json={"model": ROUTE_K2, "messages": MESSAGES, "stream": True},
        )
        plain = http.post("/v1/chat/completions", json={"model": ROUTE_K2, "messages": MESSAGES})
        legacy = http.post("/v1/completions", json={"model": ROUTE_K2, "prompt": "hi"})
    body = _error(streamed)
    assert "b99999 does not include" in body["message"]
    assert "data:" not in streamed.text
    _error(plain)
    _error(legacy)


def test_the_management_routes_refuse_it(arch_app: Any) -> None:
    # Loopback: a lease is a box change behind the D32 admin gate.
    with TestClient(arch_app, client=("127.0.0.1", 50000)) as http:
        _error(http.post(f"/api/models/{ROUTE_K2}/load", json={}))
        _error(http.post(f"/api/models/{ROUTE_K2}/load-recommended", json={"ctx_size": 32768}))
        _error(http.get(f"/api/models/{ROUTE_K2}/plan-recommended", params={"ctx_size": 32768}))
        _error(http.post("/api/leases", json={"devices": [0], "model_ids": [ROUTE_K2]}))
        _error(http.post(f"/api/models/{ROUTE_K2}/benchmark", json={}))
        _error(http.post(f"/api/models/{ROUTE_K2}/benchmark-parallel", json={}))
        plan = http.get(f"/api/models/{ROUTE_K2}/plan").json()
    assert plan["fits"] is False and plan["reason_code"] == "unsupported_architecture"
    assert arch_app.state.manager.leases.all() == []


def test_the_refusal_is_logged_as_a_warning(arch_app: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = RecordingLog()
    monkeypatch.setattr(app_module, "log", recorder)
    with TestClient(arch_app) as http:
        http.post("/v1/chat/completions", json={"model": ROUTE_K2, "messages": MESSAGES})
    rejected = recorder.events("warning", "request rejected")
    assert rejected and rejected[0]["code"] == "unsupported_architecture"
    assert "unsupported_architecture" in app_module.WARNING_REJECTION_CODES


def test_the_listings_say_which_models_cannot_load(arch_app: Any) -> None:
    with TestClient(arch_app) as http:
        models = {m["id"]: m for m in http.get("/api/models").json()["models"]}
        openai = {m["id"]: m for m in http.get("/v1/models").json()["data"]}
        one = http.get(f"/v1/models/{ROUTE_K2}").json()
        catalog = {m["id"]: m for m in http.get("/api/catalog").json()["models"]}

    assert models[ROUTE_K2]["arch_supported"] is False
    assert (
        models[ROUTE_K2]["arch_note"]
        == "llama.cpp build b99999 does not include the 'k2-horizon' architecture"
    )
    assert models[ROUTE_K2]["engine_supported"] is False
    assert "No StudioForge setting changes that" in models[ROUTE_K2]["unsupported_reason"]
    assert models[ROUTE_OK]["arch_supported"] is True
    assert "arch_note" not in models[ROUTE_OK]

    assert openai[ROUTE_K2]["studioforge"]["arch_supported"] is False
    assert openai[ROUTE_K2]["studioforge"]["arch_note"].endswith("'k2-horizon' architecture")
    assert openai[ROUTE_OK]["studioforge"]["arch_supported"] is True
    assert one["studioforge"]["arch_supported"] is False

    assert catalog[ROUTE_K2]["engine_supported"] is False
    assert catalog[ROUTE_K2]["fits_now"] is False
    assert catalog[ROUTE_K2]["recommended"] is None
    assert catalog[ROUTE_OK]["arch_supported"] is True
    assert catalog[ROUTE_OK]["recommended"] is not None


def test_capabilities_judge_each_model_from_the_library(arch_app: Any) -> None:
    with TestClient(arch_app) as http:
        body = http.get("/api/capabilities").json()
    engine = body["engine"]
    assert engine["tag"] == "b99999"
    assert engine["capability_source"] == "binary"
    assert engine["capability_describes_engine"] is True
    assert engine["architecture_library"] == "llama.dll"
    assert "llama" in engine["architectures"]
    assert "k2-horizon" not in engine["architectures"]
    library = body["library"]
    assert library["unsupported_by_engine"] == [
        {
            "model_id": ROUTE_K2,
            "architecture": "k2-horizon",
            "engine_tag": "b99999",
            "source": "binary",
        }
    ]
    assert library["unknown_to_architecture_list"] == []
    assert library["architecture_verdict_source"] == "binary"
    assert library["architecture_verdicts_from_binary"] == 2


async def test_mcp_model_info_and_plan_load_carry_the_verdict(arch_app: Any) -> None:
    from studioforge.mcp.management import build_management_mcp

    server = build_management_mcp(arch_app.state)

    async def call(name: str, **arguments: Any) -> dict[str, Any]:
        result = await server.call_tool(name, arguments)
        payload = json.loads(result.content[0].text)
        assert isinstance(payload, dict)
        return payload

    info = await call("model_info", model_id=ROUTE_K2)
    assert info["arch_supported"] is False
    assert info["engine_supported"] is False
    assert "k2-horizon" in info["unsupported_reason"]
    assert (await call("model_info", model_id=ROUTE_OK))["arch_supported"] is True

    plan = await call("plan_load", model_id=ROUTE_K2)
    assert plan["plan"]["fits"] is False
    assert plan["plan"]["reason_code"] == "unsupported_architecture"

    refused = await call("load_model", model_id=ROUTE_K2)
    assert refused["ok"] is False
    assert refused["error"]["code"] == "unsupported_architecture"
