"""A transient scan error must not make a model disappear.

Incident (production, the system this replaces): *"a model vanished from the
catalogue, so a load failed with 'Model not found' -- and it reappeared minutes
later."* The file was there the whole time; one scan could not read it. Anything
that indexes a live directory hits this: a file being written, an antivirus
lock, a sync client, flaky I/O over a network mount.

The registry's old behaviour was to put the parse error in ``ScanResult.errors``
and drop the model, which turns a blip into a client-visible 404 for a model id
that the client has saved in its own config.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import pytest

from studioforge.config import Config, ModelsConfig
from studioforge.core import gguf
from studioforge.core.registry import Registry
from studioforge.db import Database
from studioforge.types import GgufMeta

MODEL_REL = "publisher/repo-GGUF/keeper-Q4_K_M.gguf"
MODEL_ID = "publisher/repo-GGUF/keeper-Q4_K_M"
OTHER_REL = "publisher/repo-GGUF/other-Q4_K_M.gguf"
OTHER_ID = "publisher/repo-GGUF/other-Q4_K_M"


class FlakyMetaReader:
    """Parses fine until a filename is added to :attr:`fail`."""

    def __init__(self) -> None:
        self.fail: set[str] = set()

    def __call__(self, path: Path, shard_paths: Sequence[Path] | None = None) -> GgufMeta:
        if path.name in self.fail:
            raise gguf.GgufError(f"could not read {path.name}: the file is locked")
        return GgufMeta(
            architecture="qwen3",
            n_layer=32,
            n_embd=4096,
            n_head=32,
            n_head_kv=8,
            n_ctx_train=131072,
            quant_label="Q4_K_M",
            tensor_bytes=4096,
        )


@pytest.fixture(autouse=True)
def _no_sf_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("SF_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture()
def library(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    for rel in (MODEL_REL, OTHER_REL):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0" * 4096)
    return root


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "data" / "registry.sqlite3")
    database.migrate()
    yield database
    database.close()


@pytest.fixture()
def reader() -> FlakyMetaReader:
    return FlakyMetaReader()


@pytest.fixture()
def reg(library: Path, tmp_path: Path, db: Database, reader: FlakyMetaReader) -> Registry:
    config = Config(data_dir=tmp_path / "data", models=ModelsConfig(dir=library))
    return Registry(config, db, meta_reader=reader)


def test_transient_parse_error_keeps_the_model(reg: Registry, reader: FlakyMetaReader) -> None:
    """The incident: a model that fails one scan must not vanish from the catalogue.

    The file is still on disk, so the previously indexed record is kept and
    flagged stale rather than removed -- a client that saved this model id
    keeps resolving it.
    """
    first = reg.scan()
    assert MODEL_ID in first.added

    reader.fail.add("keeper-Q4_K_M.gguf")
    second = reg.scan(force=True)  # force=True re-parses, so the failure is real

    assert second.stale == [MODEL_ID]
    assert MODEL_ID not in second.removed
    record = reg.get(MODEL_ID)
    assert record is not None, "a transient read error removed a model that is still on disk"
    assert record.stale is True
    assert record.stale_reason is not None and "locked" in record.stale_reason
    # The error is still reported: sticky is not silent.
    assert any(model_id == MODEL_ID for model_id, _ in second.errors)
    # The healthy neighbour is untouched.
    assert reg.get(OTHER_ID) is not None
    assert reg.get(OTHER_ID).stale is False  # type: ignore[union-attr]


def test_stale_clears_when_the_file_parses_again(reg: Registry, reader: FlakyMetaReader) -> None:
    """ "...and it reappeared minutes later" -- the recovery must clear the flag."""
    reg.scan()
    reader.fail.add("keeper-Q4_K_M.gguf")
    reg.scan(force=True)
    assert reg.get(MODEL_ID).stale is True  # type: ignore[union-attr]

    reader.fail.clear()
    third = reg.scan(force=True)

    assert third.stale == []
    record = reg.get(MODEL_ID)
    assert record is not None
    assert record.stale is False
    assert record.stale_reason is None


def test_a_deleted_file_is_still_removed(reg: Registry, library: Path) -> None:
    """Stickiness is about unreadable files, not missing ones.

    A model whose file is genuinely gone must still leave the registry,
    otherwise the catalogue slowly fills with models that cannot ever load.
    """
    reg.scan()
    (library / MODEL_REL).unlink()

    result = reg.scan()

    assert MODEL_ID in result.removed
    assert result.stale == []
    assert reg.get(MODEL_ID) is None


def test_an_unknown_model_that_fails_to_parse_is_not_invented(
    reg: Registry, reader: FlakyMetaReader
) -> None:
    """Nothing to carry over means nothing is carried over: still just an error."""
    reader.fail.add("keeper-Q4_K_M.gguf")

    result = reg.scan()

    assert reg.get(MODEL_ID) is None
    assert result.stale == []
    assert any(model_id == MODEL_ID for model_id, _ in result.errors)


def test_stale_flag_is_visible_on_the_wire(reg: Registry, reader: FlakyMetaReader) -> None:
    """A stale model must be identifiable by a client, not silently degraded."""
    reg.scan()
    reader.fail.add("keeper-Q4_K_M.gguf")
    reg.scan(force=True)

    entry = next(e for e in reg.openai_list() if e["id"] == MODEL_ID)

    assert entry["studioforge"]["stale"] is True


# ---------------------------------------------------------------------------
# Unreachable is not removed: a root that is not there right now (a dropped
# drive, an unmounted share) keeps every model under it as stale; a file that
# is gone from a root that WAS walked is removed. And the manager unloads only
# the latter (WP17 R3).
# ---------------------------------------------------------------------------


def test_a_missing_model_root_keeps_its_models_as_stale(
    tmp_path: Path, db: Database, reader: FlakyMetaReader
) -> None:
    import shutil

    primary = tmp_path / "models"
    second = tmp_path / "drive2"
    for root, rel in ((primary, MODEL_REL), (second, OTHER_REL)):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0" * 4096)
    config = Config(
        data_dir=tmp_path / "data", models=ModelsConfig(dir=primary, extra_dirs=[second])
    )
    reg = Registry(config, db, meta_reader=reader)
    first = reg.scan()
    assert set(first.added) == {MODEL_ID, OTHER_ID}

    # The second drive drops.
    shutil.rmtree(second)
    result = reg.scan()
    assert result.removed == [], "unreachable is not removed"
    assert OTHER_ID in result.stale
    kept = reg.get(OTHER_ID)
    assert kept is not None and kept.stale is True
    assert "not available" in (kept.stale_reason or "")
    assert reg.get(MODEL_ID) is not None and reg.get(MODEL_ID).stale is False

    # It comes back: the record is fresh again.
    (second / OTHER_REL).parent.mkdir(parents=True, exist_ok=True)
    (second / OTHER_REL).write_bytes(b"\0" * 4096)
    back = reg.scan()
    assert OTHER_ID not in back.stale and OTHER_ID not in back.removed
    assert reg.get(OTHER_ID).stale is False


def test_a_file_gone_from_a_walked_root_is_removed(reg: Registry, library: Path) -> None:
    reg.scan()
    (library / OTHER_REL).unlink()
    result = reg.scan()
    assert result.removed == [OTHER_ID]
    assert reg.get(OTHER_ID) is None


async def test_the_sweeper_unloads_a_removed_model_but_not_an_unreachable_one() -> None:
    from studioforge.core.manager import ModelManager
    from studioforge.types import InstanceInfo, LoadPlan
    from tests.unit.test_load_retry import StubPlanner, StubRegistry, StubSupervisor, make_record

    class ScannedRegistry(StubRegistry):
        last_scan_at: float | None = 1.0

    removed_id, unreachable_id, fine_id = "gone/model", "dropped/model", "fine/model"
    unreachable = make_record(unreachable_id)
    unreachable.stale = True
    unreachable.stale_reason = "model directory D:/drive2 is not available right now"
    registry = ScannedRegistry({unreachable_id: unreachable, fine_id: make_record(fine_id)})
    supervisor = StubSupervisor()
    for model_id in (removed_id, unreachable_id, fine_id):
        supervisor.instances[model_id] = InstanceInfo(
            model_id=model_id,
            state="ready",
            port=18100,
            ttl_s=0,  # pinned: TTL alone would never touch these
            plan=LoadPlan(model_id=model_id, devices=[0]),
        )
    manager = ModelManager(
        Config(data_dir="/tmp/sf-sweep"),
        registry=registry,  # type: ignore[arg-type]
        planner=StubPlanner(),  # type: ignore[arg-type]
        supervisor=supervisor,  # type: ignore[arg-type]
        db=None,  # type: ignore[arg-type]
    )
    await manager._sweep_ttl()
    assert supervisor.stopped == [removed_id]

    # Before the first scan nothing is "removed", whatever the registry lacks.
    registry.last_scan_at = None
    supervisor.instances[removed_id] = InstanceInfo(
        model_id=removed_id, state="ready", port=18101, ttl_s=0
    )
    supervisor.stopped.clear()
    await manager._sweep_ttl()
    assert supervisor.stopped == []
