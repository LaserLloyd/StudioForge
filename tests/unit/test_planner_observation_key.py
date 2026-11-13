"""The D51 match key names WHERE a measurement was taken (D88, A-12, owner decision F-6).

Until D88 an observation row was matched by ``(model, ctx, slots, kv_k, kv_v,
device COUNT, kv_unified)``. Three things followed, all confirmed in the
2026-11-09 review:

* a correction learned on two 32 GiB Blackwell cards was spent on two 24 GiB
  Ampere cards of the same count -- and inside ONE planning pass the memo, keyed
  the same way, handed the first pair's answer to the second;
* a tensor split and a layer split shared one row;
* a GGUF replaced under the same id inherited the old file's correction.

Migration 011 adds ``class_key``, ``split_mode`` and ``gguf_size``. Rows from
before it are legacy matches: still used, but only when no row naming the exact
values exists. These tests pin the key, the legacy rule, the planner's use of it
and the virtual-copy fallback.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from studioforge.core.planner import (
    CORRECTION_NOTE_PREFIX,
    OBSERVATION_NOTE_PER_PID_DEVICE,
    Planner,
    device_class_key,
    placement_class_key,
)
from studioforge.db import SCHEMA_VERSION, Database
from studioforge.types import GB, AdapterAttachment, GpuInfo, LoadPlan, ModelSettings
from tests.unit.test_planner import (
    StubProbe,
    gpu,
    make_config,
    make_meta,
    make_record,
    rig_5090x2_3090x2,
)

BLACKWELL_PAIR = "sm_120-32g+sm_120-32g"
AMPERE_PAIR = "sm_86-24g+sm_86-24g"
SIZE = 8 * GB


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "registry.sqlite3")
    database.migrate()
    yield database
    database.close()


def row(**overrides: Any) -> dict[str, Any]:
    """A clean, matchable D88 observation row (a 3090 pair, layer split)."""
    base: dict[str, Any] = {
        "model_id": "test/model",
        "ctx_size": 8192,
        "parallel": 1,
        "kv_cache_type": "f16",
        "kv_cache_type_v": "f16",
        "devices": "2,3",
        "predicted_bytes": 10 * GB,
        "actual_bytes": 11 * GB,
        "weights_bytes": 8 * GB,
        "ok": True,
        "note": OBSERVATION_NOTE_PER_PID_DEVICE,
        "class_key": AMPERE_PAIR,
        "split_mode": "layer",
        "gguf_size": SIZE,
    }
    base.update(overrides)
    return base


def ask(**overrides: Any) -> dict[str, Any]:
    key: dict[str, Any] = {
        "ctx_size": 8192,
        "parallel": 1,
        "kv_cache_type": "f16",
        "kv_cache_type_v": "f16",
        "device_count": 2,
        "class_key": AMPERE_PAIR,
        "split_mode": "layer",
        "gguf_size": SIZE,
    }
    key.update(overrides)
    return key


def forced(devices: list[int], **settings: Any) -> Any:
    return make_record(settings=ModelSettings(device_override=devices, **settings))


def plan_on(planner: Planner, record: Any) -> LoadPlan:
    plan = planner.plan_load(record, ctx_size=8192, parallel=1)
    assert isinstance(plan, LoadPlan), plan
    return plan


def corrected(plan: LoadPlan) -> bool:
    return any(n.startswith(CORRECTION_NOTE_PREFIX) for n in plan.notes)


def db_planner(db: Database, probe: Any = None) -> Planner:
    return Planner(
        make_config(),
        probe if probe is not None else rig_5090x2_3090x2(),
        observation_sink=lambda r: db.record_load_observation(**r),
        observation_lookup=db.matching_observation,
        log_plans=False,
    )


# ---------------------------------------------------------------------------
# The class key
# ---------------------------------------------------------------------------


def test_a_card_class_is_its_arch_and_its_size_never_its_name() -> None:
    assert device_class_key(gpu(2, 24.0, 20.0, (8, 6))) == "sm_86-24g"
    assert device_class_key(gpu(0, 31.84, 20.0, (12, 0))) == "sm_120-32g"
    # A 48 GB workstation card of the same arch is its own class.
    assert device_class_key(gpu(4, 47.99, 40.0, (12, 0))) == "sm_120-48g"
    amd = GpuInfo(index=0, name="Radeon", total_bytes=24 * GB, free_bytes=0, gfx_target="gfx1100")
    assert device_class_key(amd) == "gfx1100-24g"
    unknown = GpuInfo(index=0, name="Mystery", total_bytes=16 * GB, free_bytes=0)
    assert device_class_key(unknown) == "unknown-16g"


def test_a_class_the_probe_already_supplies_is_trusted() -> None:
    """The card-facts work (B.3) puts ``class_key`` on the card; both must agree."""
    card = SimpleNamespace(
        class_key="sm_86-24g", compute_capability=(9, 9), gfx_target=None, total_bytes=1
    )
    assert device_class_key(card) == "sm_86-24g"  # type: ignore[arg-type]


def test_a_placement_class_is_order_blind() -> None:
    a, b = gpu(3, 24.0, 1.0, (8, 6)), gpu(0, 31.84, 1.0, (12, 0))
    assert placement_class_key([a, b]) == placement_class_key([b, a]) == "sm_120-32g+sm_86-24g"
    assert placement_class_key([]) is None


# ---------------------------------------------------------------------------
# The match rule (Database.matching_observation)
# ---------------------------------------------------------------------------


def test_migration_011_adds_the_three_columns(db: Database) -> None:
    assert SCHEMA_VERSION >= 11
    columns = {r["name"] for r in db.query("PRAGMA table_info(load_observations)")}
    assert {"class_key", "split_mode", "gguf_size"} <= columns


def test_migration_011_upgrades_a_schema_10_database_and_keeps_its_rows(tmp_path: Path) -> None:
    """A row written at schema 10 survives the upgrade as a legacy match."""
    from studioforge.db import _migration_sources, _split_statements

    path = tmp_path / "registry.sqlite3"
    raw = sqlite3.connect(path)
    raw.execute(
        "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at REAL, name TEXT)"
    )
    for version, name, sql in _migration_sources():
        if version > 10:
            break
        for statement in _split_statements(sql):
            raw.execute(statement)
        raw.execute("INSERT INTO schema_migrations VALUES (?, 0, ?)", (version, name))
    raw.execute(
        "INSERT INTO load_observations (model_id, ts, ctx_size, parallel, kv_cache_type, "
        "kv_cache_type_v, devices, predicted_bytes, actual_bytes, weights_bytes, ok, note) "
        "VALUES ('test/model', 1.0, 8192, 1, 'f16', 'f16', '0,1', 1, ?, 1, 1, ?)",
        (11 * GB, OBSERVATION_NOTE_PER_PID_DEVICE),
    )
    raw.commit()
    raw.close()

    upgraded = Database(path)
    try:
        upgraded.migrate()
        assert upgraded.schema_version() == SCHEMA_VERSION
        stored = upgraded.load_observations("test/model")[0]
        assert stored["class_key"] is None and stored["split_mode"] is None
        assert stored["gguf_size"] is None
        found = upgraded.matching_observation("test/model", **ask())
        assert found is not None and found["legacy_match"] is True
    finally:
        upgraded.close()


def test_the_exact_row_matches(db: Database) -> None:
    db.record_load_observation(**row())
    found = db.matching_observation("test/model", **ask())
    assert found is not None and found["legacy_match"] is False


@pytest.mark.parametrize(
    "asked",
    [
        pytest.param({"class_key": BLACKWELL_PAIR}, id="another card class"),
        pytest.param({"split_mode": "tensor"}, id="another split"),
        pytest.param({"gguf_size": SIZE + 4096}, id="a replaced file"),
    ],
)
def test_a_row_of_another_class_split_or_file_is_never_spent(
    db: Database, asked: dict[str, Any]
) -> None:
    db.record_load_observation(**row())
    assert db.matching_observation("test/model", **ask(**asked)) is None


def test_a_pre_011_row_is_a_legacy_match(db: Database) -> None:
    """F-6: old rows stay usable, and say so."""
    db.record_load_observation(**row(class_key=None, split_mode=None, gguf_size=None))
    found = db.matching_observation("test/model", **ask())
    assert found is not None and found["legacy_match"] is True
    # ...for any class: it cannot say which it measured.
    assert db.matching_observation("test/model", **ask(class_key=BLACKWELL_PAIR)) is not None


def test_an_exact_row_beats_a_newer_legacy_row(db: Database) -> None:
    db.record_load_observation(**row(ts=100.0, actual_bytes=11 * GB))
    db.record_load_observation(
        **row(ts=200.0, actual_bytes=13 * GB, class_key=None, split_mode=None, gguf_size=None)
    )
    found = db.matching_observation("test/model", **ask())
    assert found is not None and found["actual_bytes"] == 11 * GB
    assert found["legacy_match"] is False


def test_a_caller_that_cannot_say_does_not_filter(db: Database) -> None:
    """The pre-D88 call shape still finds rows of any class (and is not legacy-flagged)."""
    db.record_load_observation(**row())
    key = {k: v for k, v in ask().items() if k not in {"class_key", "split_mode", "gguf_size"}}
    found = db.matching_observation("test/model", **key)
    assert found is not None and found["legacy_match"] is False


# ---------------------------------------------------------------------------
# The planner: the class decides which correction is spent
# ---------------------------------------------------------------------------


def test_a_correction_learned_on_one_class_is_not_spent_on_another(db: Database) -> None:
    """The A-12 case: measured on the 5090 pair, planned on the 3090 pair."""
    planner = db_planner(db)
    learned = plan_on(planner, forced([0, 1]))
    assert not corrected(learned)
    planner.observe(
        model_id=learned.model_id,
        plan=learned,
        actual_bytes=int(learned.estimate.total_bytes * 1.2),
        note=OBSERVATION_NOTE_PER_PID_DEVICE,
        gguf_size=SIZE,
    )
    stored = db.load_observations("test/model")[0]
    assert stored["class_key"] == BLACKWELL_PAIR
    assert stored["split_mode"] == "layer"
    assert stored["gguf_size"] == SIZE

    assert corrected(plan_on(planner, forced([0, 1]))), "the same class: spent"
    assert corrected(plan_on(planner, forced([1, 0]))), "order does not matter"
    assert not corrected(plan_on(planner, forced([2, 3]))), "another class: never"


def test_a_legacy_row_is_still_spent_and_the_note_says_what_it_is(db: Database) -> None:
    planner = db_planner(db)
    base = plan_on(planner, forced([2, 3]))
    db.record_load_observation(
        **row(
            actual_bytes=int(base.estimate.total_bytes * 1.2),
            class_key=None,
            split_mode=None,
            gguf_size=None,
        )
    )
    plan = plan_on(planner, forced([2, 3]))
    note = next(n for n in plan.notes if n.startswith(CORRECTION_NOTE_PREFIX))
    assert "before D88" in note
    applied = planner.applied_correction_for(plan.model_id, plan)
    assert applied is not None and applied.correction.legacy is True


def test_a_replaced_file_starts_its_own_history(db: Database) -> None:
    planner = db_planner(db)
    base = plan_on(planner, forced([2, 3]))
    db.record_load_observation(**row(actual_bytes=int(base.estimate.total_bytes * 1.2)))
    assert corrected(plan_on(planner, forced([2, 3])))
    bigger = forced([2, 3]).model_copy(update={"size_bytes": SIZE + 1})
    assert not corrected(plan_on(planner, bigger))


def test_the_split_mode_is_part_of_the_key_and_auto_does_not_filter(db: Database) -> None:
    planner = db_planner(db)
    base = plan_on(planner, forced([2, 3]))
    db.record_load_observation(**row(actual_bytes=int(base.estimate.total_bytes * 1.2)))
    assert not corrected(plan_on(planner, forced([2, 3], split_mode="tensor")))
    # "auto" resolves at launch; the row stores what it resolved to.
    assert corrected(plan_on(planner, forced([2, 3], split_mode="auto")))


def test_one_card_is_split_mode_none(db: Database) -> None:
    planner = db_planner(db)
    plan = plan_on(planner, forced([2]))
    planner.observe(
        model_id=plan.model_id,
        plan=plan,
        actual_bytes=int(plan.estimate.total_bytes),
        note=OBSERVATION_NOTE_PER_PID_DEVICE,
        gguf_size=SIZE,
    )
    stored = db.load_observations("test/model")[0]
    assert stored["class_key"] == "sm_86-24g" and stored["split_mode"] == "none"


def test_one_pass_over_two_same_count_pairs_asks_about_each_class() -> None:
    """The memo collision: keyed by count, [0, 1]'s answer was reused for [2, 3]."""
    asked: list[tuple[Any, ...]] = []

    def lookup(model_id: str, **key: Any) -> dict[str, Any] | None:
        asked.append((key["device_count"], key.get("class_key")))
        if key.get("class_key") == BLACKWELL_PAIR:
            # Only the 5090 pair was ever measured, and it measured big.
            return {"actual_bytes": 60 * GB, "ok": True, "note": OBSERVATION_NOTE_PER_PID_DEVICE}
        return None

    # The 5090 pair offers ~41.6 GiB usable: the formula (~32.5 GiB) fits it,
    # the measurement (clamped to 1.3x the formula) does not.
    probe = StubProbe(
        [
            gpu(0, 31.84, 24.0, (12, 0)),
            gpu(1, 31.84, 24.0, (12, 0)),
            gpu(2, 24.0, 23.5, (8, 6)),
            gpu(3, 24.0, 23.5, (8, 6)),
        ]
    )
    planner = Planner(make_config(), probe, observation_lookup=lookup, log_plans=False)
    big = make_record(meta=make_meta(tensor_bytes=30 * GB), size_bytes=30 * GB)
    plan = planner.plan_load(big, ctx_size=8192, parallel=1)
    assert isinstance(plan, LoadPlan)
    assert (2, BLACKWELL_PAIR) in asked and (2, AMPERE_PAIR) in asked
    # The 5090 pair's (clamped) correction refused it there; the 3090 pair,
    # never measured, planned on the formula.
    assert sorted(plan.devices) == [2, 3]
    assert not corrected(plan)


# ---------------------------------------------------------------------------
# Virtual copies borrow their base's history (B.1's ephemeral instances too)
# ---------------------------------------------------------------------------


def copy_of_base(**settings: Any) -> Any:
    return make_record(
        "test/model@b",
        settings=ModelSettings(device_override=[2, 3], **settings),
    ).model_copy(update={"is_virtual": True, "base_model_id": "test/model"})


def test_a_copy_with_no_history_uses_its_bases_rows(db: Database) -> None:
    planner = db_planner(db)
    base = plan_on(planner, forced([2, 3]))
    db.record_load_observation(**row(actual_bytes=int(base.estimate.total_bytes * 1.2)))
    plan = plan_on(planner, copy_of_base())
    note = next(n for n in plan.notes if n.startswith(CORRECTION_NOTE_PREFIX))
    assert "measured as 'test/model'" in note


def test_a_copy_with_adapters_or_a_draft_keeps_its_own_history(db: Database) -> None:
    planner = db_planner(db)
    base = plan_on(planner, forced([2, 3]))
    db.record_load_observation(**row(actual_bytes=int(base.estimate.total_bytes * 1.2)))
    with_adapter = copy_of_base(adapters=[AdapterAttachment(adapter_id="lora-1")])
    assert planner._history_fallback_id(with_adapter) is None
    with_draft = copy_of_base(draft_model_id="pub/tiny-draft")
    assert planner._history_fallback_id(with_draft) is None
    assert planner._history_fallback_id(copy_of_base()) == "test/model"


def test_a_copys_own_row_wins_over_its_bases(db: Database) -> None:
    planner = db_planner(db)
    base = plan_on(planner, forced([2, 3]))
    db.record_load_observation(**row(actual_bytes=int(base.estimate.total_bytes * 1.2)))
    db.record_load_observation(
        **row(model_id="test/model@b", actual_bytes=int(base.estimate.total_bytes * 0.9))
    )
    plan = plan_on(planner, copy_of_base())
    note = next(n for n in plan.notes if n.startswith(CORRECTION_NOTE_PREFIX))
    assert "measured as" not in note
    assert plan.estimate.total_bytes < base.estimate.total_bytes
