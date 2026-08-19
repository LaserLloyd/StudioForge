"""Engine feature detection: what a build advertises, read from its own --help.

The whole point of :class:`~studioforge.core.engine.EngineFeatures` is that
StudioForge never passes a flag on faith (DECISIONS.md D2/D38), so these tests
are pinned to a *verbatim* excerpt of the pinned engine's help text rather than
to a hand-written imitation of it: a parser that only works against the fixture
its author invented would pass here and drop every flag in production.
"""

from __future__ import annotations

import json
from pathlib import Path

from studioforge.core.engine import (
    FEATURES_FILE,
    EngineFeatures,
    parse_engine_features,
    probe_engine_features,
    read_features_file,
    write_features_file,
)

HELP_EXCERPT = (Path(__file__).parent / "data" / "b10425_help_excerpt.txt").read_text(
    encoding="utf-8"
)


def features() -> EngineFeatures:
    return parse_engine_features(HELP_EXCERPT, "b10425")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_split_modes_come_from_the_value_placeholder() -> None:
    """``-sm, --split-mode {none,layer,row,tensor}`` -- the braces are the list."""
    assert features().split_modes == ("none", "layer", "row", "tensor")
    assert features().supports_split("tensor")
    assert not features().supports_split("pipeline")


def test_spec_types_are_read_not_hardcoded() -> None:
    """Upstream adds speculative types faster than this file could track them."""
    spec = features().spec_types
    for expected in ("none", "draft-simple", "draft-mtp", "ngram-mod", "ngram-cache"):
        assert expected in spec, f"{expected} missing from {spec}"
    # The description that follows the enumeration must not leak into it.
    assert "comma-separated" not in " ".join(spec)


def test_supports_spec_requires_every_member_of_a_combo() -> None:
    caps = features()
    assert caps.supports_spec("draft-mtp,ngram-mod")
    assert not caps.supports_spec("draft-mtp,ngram-invented")
    assert not caps.supports_spec("")


def test_flash_attn_values_and_the_boolean_features() -> None:
    caps = features()
    assert caps.flash_attn_values == ("on", "off", "auto")
    assert caps.backend_sampling is True
    assert caps.cache_ram is True
    assert caps.ctx_checkpoints is True
    assert caps.fit is True
    assert caps.reasoning_budget is True
    assert caps.kv_unified is True


def test_engine_defaults_are_captured_from_the_help_text() -> None:
    """The numbers we refuse to guess at: they decide what we do NOT emit."""
    caps = features()
    assert caps.spec_draft_n_max_default == 3
    assert caps.cache_ram_default_mib == 8192
    assert caps.ctx_checkpoints_default == 32
    # Unified KV is NOT unconditionally on -- it depends on the slot count, and
    # StudioForge always passes an explicit one. Verified live in D38.
    assert caps.kv_unified_default == "enabled if number of slots is auto"


def test_removed_flags_are_not_advertised() -> None:
    """b10425 accepts ``--draft`` and ignores it; treating it as live is the
    exact failure D2 recorded."""
    caps = features()
    assert "--spec-draft-n-max" in caps.flags
    for removed in ("--draft", "--draft-n", "--draft-max"):
        assert removed not in caps.flags
        assert not caps.has(removed)


def test_unknown_engine_advertises_nothing() -> None:
    caps = EngineFeatures.unknown("b99999")
    assert caps.known is False
    assert not caps.has("--cache-ram")
    assert not caps.supports_spec("draft-mtp")
    assert not caps.supports_split("layer")


def test_empty_help_is_unknown_not_empty_support() -> None:
    """Nothing parsed must never read as "the engine offers nothing"."""
    assert parse_engine_features("", "b10425").known is False


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_features_round_trip_through_the_cache_file(tmp_path: Path) -> None:
    original = features()
    write_features_file(tmp_path, original)
    loaded = read_features_file(tmp_path, "b10425")
    assert loaded is not None
    assert loaded.to_dict() == original.to_dict()


def test_cache_file_for_another_tag_is_ignored(tmp_path: Path) -> None:
    """An engine directory that was reused for a different build must not serve
    the old build's answers -- that is how a renamed flag comes back to life."""
    write_features_file(tmp_path, features())
    assert read_features_file(tmp_path, "b11000") is None


def test_unparseable_cache_file_is_ignored(tmp_path: Path) -> None:
    (tmp_path / FEATURES_FILE).write_text("{not json", encoding="utf-8")
    assert read_features_file(tmp_path) is None
    (tmp_path / FEATURES_FILE).write_text(json.dumps({"known": False}), encoding="utf-8")
    assert read_features_file(tmp_path) is None


def test_probe_falls_back_to_help_txt_and_writes_the_cache(tmp_path: Path) -> None:
    """The supervisor's synchronous path: features.json -> help.txt -> --help.

    Reaching help.txt matters because that file is already written by the
    flag-validation path, so a box that has ever validated extra flags never
    pays for a subprocess here.
    """
    (tmp_path / "help.txt").write_text(HELP_EXCERPT, encoding="utf-8")
    binary = tmp_path / "llama-server.exe"
    binary.write_text("", encoding="utf-8")

    caps = probe_engine_features(binary, "b10425")
    assert caps.known is True
    assert caps.supports_spec("draft-mtp")
    assert (tmp_path / FEATURES_FILE).is_file(), "the probe must leave the cache behind"


def test_probe_without_a_binary_degrades_to_unknown(tmp_path: Path) -> None:
    """A model must still load on a box whose engine help cannot be read."""
    caps = probe_engine_features(tmp_path / "does-not-exist.exe", "b10425")
    assert caps.known is False


def test_probe_never_executes_something_that_is_not_the_engine(tmp_path: Path) -> None:
    """Building a command line must not run whatever ``resolve_binary`` returned.

    A ``resolve_binary`` hook can legitimately return a wrapper or a test stub.
    Launching it with ``--help`` to see what it supports is a side effect nobody
    asked for -- it cost two flaky failures here, where the supervisor's fake
    child was started with ``--help`` and bound the port the next test wanted.
    """
    from studioforge.core import engine as engine_module

    stub = tmp_path / "fake_llama_server.py"
    stub.write_text("raise SystemExit('this must never run')", encoding="utf-8")
    ran: list[object] = []
    monkeypatched = engine_module.subprocess.run

    def _spy(*args: object, **kwargs: object) -> object:  # pragma: no cover - must not fire
        ran.append(args)
        return monkeypatched(*args, **kwargs)  # type: ignore[arg-type]

    engine_module.subprocess.run = _spy  # type: ignore[assignment]
    try:
        assert probe_engine_features(stub, "b10425").known is False
    finally:
        engine_module.subprocess.run = monkeypatched  # type: ignore[assignment]
    assert ran == []


# ---------------------------------------------------------------------------
# The capabilities report
# ---------------------------------------------------------------------------


def test_the_report_carries_the_engines_own_feature_surface(tmp_path: Path) -> None:
    """The architecture/quant lists say what llama.cpp supports at the pinned
    tag; this says what the binary on this disk will accept. They are different
    questions and the report has to answer both."""
    from studioforge.config import Config
    from studioforge.core.capabilities import engine_capabilities

    config = Config(data_dir=tmp_path)
    engine_dir = config.engines_dir / config.engine.pinned_tag
    engine_dir.mkdir(parents=True, exist_ok=True)
    (engine_dir / "help.txt").write_text(HELP_EXCERPT, encoding="utf-8")

    caps = engine_capabilities(config)
    assert caps.features["known"] is True
    assert "tensor" in caps.features["split_modes"]
    assert "draft-mtp" in caps.features["spec_types"]


def test_feature_rows_distinguish_absent_from_off() -> None:
    """ "Not advertised" and "off" are different facts; conflating them makes a
    missing feature read as a disabled one."""
    from studioforge.core.capabilities import engine_feature_rows

    rows = {row["name"]: row["value"] for row in engine_feature_rows(features().to_dict())}
    assert rows["Split modes"] == "none, layer, row, tensor"
    assert rows["GPU sampling"] == "yes"

    unknown = {row["name"]: row["value"] for row in engine_feature_rows({"known": False})}
    assert set(unknown.values()) == {"unknown"}
