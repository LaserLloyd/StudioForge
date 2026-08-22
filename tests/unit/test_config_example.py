"""``config.example.yaml`` is the generated default config, pinned to the model.

The file's own header and the README promise "every key the app understands,
with the shipped default". Nothing enforced it: ``engine.cache_ram_mb``,
``engine.ubatch_size``, ``engine.backend_sampling`` and ``planner.preference``
were added to ``config.py`` across several work packages and the example was
never regenerated (docs/DEVELOPMENT.md, "Adding a config key", step 3), so the
promise drifted without a failing test. The oracle is ``Config.to_yaml_dict()``
-- the exact dump ``save()`` writes, which is also how the example is produced
-- built against a scratch data dir with every ``SF_*`` override cleared, so
the comparison is defaults against defaults and a developer's shell cannot
fail it. ``data_dir`` is excluded by ``to_yaml_dict`` itself (D31), which is
why the example deliberately has no such key.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from studioforge.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "config.example.yaml"


def _flatten(tree: Any, prefix: str = "") -> dict[str, Any]:
    """``{"a": {"b": 1}}`` -> ``{"a.b": 1}``; an empty mapping is a leaf, so a
    key whose default is ``{}`` (``planner.reserved_mb``) is still compared."""
    if not isinstance(tree, dict) or not tree:
        return {prefix: tree}
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        flat.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    return flat


@pytest.fixture
def shipped_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    for name in list(os.environ):
        if name.startswith("SF_"):
            monkeypatch.delenv(name)
    return _flatten(Config(data_dir=tmp_path).to_yaml_dict())


def test_example_lists_every_key_with_its_shipped_default(
    shipped_defaults: dict[str, Any],
) -> None:
    example = _flatten(yaml.safe_load(EXAMPLE.read_text(encoding="utf-8")))
    missing = sorted(set(shipped_defaults) - set(example))
    stale = sorted(set(example) - set(shipped_defaults))
    assert not missing and not stale, (
        f"config.example.yaml drifted from config.py -- regenerate it "
        f"(docs/DEVELOPMENT.md, 'Adding a config key'). "
        f"missing from example: {missing}; no longer a config key: {stale}"
    )
    differs = {
        key: {"example": example[key], "shipped": value}
        for key, value in shipped_defaults.items()
        if example[key] != value
    }
    assert not differs, f"config.example.yaml shows a default config.py no longer ships: {differs}"
