"""#16: no dead config keys, ranges on the numeric knobs, a label on every key."""

from __future__ import annotations

import pytest

from studioforge import settings_schema
from studioforge.config import Config, HfConfig
from studioforge.core import hf_meta


def test_update_dead_keys_are_gone() -> None:
    keys = {s.key for s in settings_schema.build_registry()}
    assert "update.auto_check" not in keys and "update.check_interval_h" not in keys


def test_old_config_files_with_removed_keys_still_load() -> None:
    cfg = Config.model_validate({"update": {"auto_check": False, "check_interval_h": 1}})
    assert not hasattr(cfg.update, "auto_check")


def test_ranges_are_published() -> None:
    by_key = {s.key: s for s in settings_schema.build_registry()}
    head = by_key["planner.headroom_fraction"].constraints
    assert head["ge"] == 0.0 and head["lt"] == 0.9
    prio = by_key["benchmark.lease_priority"].constraints
    assert prio["ge"] == 1 and prio["le"] == 3
    with pytest.raises(ValueError):
        Config.model_validate({"planner": {"headroom_fraction": 0.95}})
    with pytest.raises(ValueError):
        Config.model_validate({"benchmark": {"lease_priority": 4}})


def test_every_setting_has_a_label() -> None:
    assert all(s.label.strip() for s in settings_schema.build_registry())
    assert settings_schema.label_for("models.default_ttl_s") == "Default TTL s"


def test_hf_cache_dir_is_wired(tmp_path) -> None:
    cfg = Config(data_dir=tmp_path / "d")
    assert hf_meta.cache_dir(cfg) == tmp_path / "d" / "cache" / "hf_meta"
    cfg2 = Config(data_dir=tmp_path / "d", hf=HfConfig(cache_dir=tmp_path / "hfc"))
    assert hf_meta.cache_dir(cfg2) == tmp_path / "hfc"
