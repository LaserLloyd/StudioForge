"""``engine.stub``: the no-GPU staging engine (D85).

What must hold: it is off by default, refused unless the GPU probe is the fake
one (it can never run against real cards), satisfied without any download or
install, boot-checked through the stub's own ``--version``, labelled ``+stub``
in the build string, and the supervisor wiring (launch prefix, feature
detection) comes from the same manager.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from studioforge import build
from studioforge.api.app import build_state
from studioforge.config import RESTART_REQUIRED_KEYS, Config
from studioforge.core import gpu as gpu_module
from studioforge.core.engine import STUB_TAG, EngineError, EngineManager
from studioforge.core.gpu import FakeGpuProbe, NullGpuProbe
from studioforge.errors import ConfigError
from studioforge.testing import stub_llama_server
from studioforge.types import GB, GpuInfo

FAKE_GPUS = (
    '[{"index":0,"name":"RTX 5090","total_bytes":34359738368,"compute_capability":[12,0]},'
    '{"index":1,"name":"RTX 3090","total_bytes":25769803776,"compute_capability":[8,6]}]'
)


@pytest.fixture(autouse=True)
def _reset_stub_label() -> Iterator[None]:
    """The ``+stub`` build label is process-wide; no test may leak it."""
    build.mark_stub_engine(False)
    build.build_id.cache_clear()  # type: ignore[attr-defined]
    yield
    build.mark_stub_engine(False)
    build.build_id.cache_clear()  # type: ignore[attr-defined]


def stub_config(tmp_path: Path, *, backend: str = "fake") -> Config:
    cfg = Config(data_dir=tmp_path / "data")
    cfg.engine.stub = True
    cfg.gpu.backend = backend  # type: ignore[assignment]
    cfg.ensure_dirs()
    return cfg


def fake_probe() -> FakeGpuProbe:
    return FakeGpuProbe(
        [GpuInfo(index=0, name="RTX 5090", total_bytes=32 * GB, free_bytes=32 * GB)]
    )


# ---------------------------------------------------------------------------
# Config and the refusal
# ---------------------------------------------------------------------------


def test_stub_is_off_by_default(tmp_path: Path) -> None:
    cfg = Config(data_dir=tmp_path)
    assert cfg.engine.stub is False
    manager = EngineManager(cfg, probe=NullGpuProbe())
    assert manager.stub_mode is False
    assert manager.launch_prefix() == []


def test_stub_can_be_set_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SF_ENGINE__STUB", "1")
    assert Config(data_dir=tmp_path).engine.stub is True


def test_stub_needs_a_restart_to_change() -> None:
    assert "engine.stub" in RESTART_REQUIRED_KEYS


@pytest.mark.parametrize("probe", [NullGpuProbe(), None])
def test_stub_is_refused_on_a_real_or_unset_backend(tmp_path: Path, probe: Any) -> None:
    cfg = stub_config(tmp_path, backend="auto")
    with pytest.raises(ConfigError) as excinfo:
        EngineManager(cfg, probe=probe)
    message = str(excinfo.value)
    assert "engine.stub" in message
    assert "fake" in message


def test_stub_is_refused_when_the_probe_is_not_fake_even_if_the_config_says_fake(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigError):
        EngineManager(stub_config(tmp_path, backend="fake"), probe=NullGpuProbe())


def test_stub_is_allowed_with_a_fake_probe_or_a_fake_backend_and_no_probe(
    tmp_path: Path,
) -> None:
    assert EngineManager(stub_config(tmp_path), probe=fake_probe()).stub_mode is True
    assert EngineManager(stub_config(tmp_path, backend="fake")).stub_mode is True


def test_a_loosely_typed_config_can_never_switch_it_on() -> None:
    from unittest.mock import MagicMock

    # A mock's attributes are truthy; only a real `True` may enable the stub.
    assert EngineManager(MagicMock(), probe=NullGpuProbe()).stub_mode is False


# ---------------------------------------------------------------------------
# The engine surface
# ---------------------------------------------------------------------------


def test_the_stub_engine_info_has_no_files(tmp_path: Path) -> None:
    manager = EngineManager(stub_config(tmp_path), probe=fake_probe())
    info = manager.active()
    assert info is not None
    assert info.tag == STUB_TAG
    assert info.server_binary == Path(sys.executable)
    assert info.active and info.smoke_tested and info.smoke_result == "passed"
    assert info.variant == "stub"
    assert manager.get(STUB_TAG) is not None
    assert manager.get("b10425") is None
    assert [i.tag for i in manager.installed()] == [STUB_TAG]
    assert manager.server_binary() == Path(sys.executable)
    assert manager.server_binary(STUB_TAG) == Path(sys.executable)
    assert not (tmp_path / "data" / "engines" / "active.json").exists()


def test_a_pinned_real_tag_is_still_not_installed(tmp_path: Path) -> None:
    manager = EngineManager(stub_config(tmp_path), probe=fake_probe())
    with pytest.raises(EngineError):
        manager.server_binary("b10425")


def test_launch_prefix_runs_the_stub_module_by_path(tmp_path: Path) -> None:
    manager = EngineManager(stub_config(tmp_path), probe=fake_probe())
    prefix = manager.launch_prefix()
    assert prefix[0] == sys.executable
    assert Path(prefix[-1]) == Path(stub_llama_server.__file__).resolve()
    assert Path(prefix[-1]).is_file()


async def test_ensure_engine_passes_the_boot_check_without_any_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    manager = EngineManager(stub_config(tmp_path), probe=fake_probe())

    async def no_network(*_a: object, **_k: object) -> None:
        raise AssertionError("the stub engine must not touch the network")

    for name in ("list_assets", "install", "build_from_source", "list_releases"):
        monkeypatch.setattr(manager, name, no_network)
    info = await manager.ensure_engine()
    assert info.tag == STUB_TAG
    assert info.version_string == "version: 0 (stub)"
    assert manager.active() is not None
    assert (await manager.version_string(STUB_TAG)) == "version: 0 (stub)"
    assert not (tmp_path / "data" / "engines" / "active.json").exists()
    assert not [
        p for p in (tmp_path / "data" / "engines").glob("*") if p.is_dir() and p.name != STUB_TAG
    ]


async def test_ensure_engine_logs_the_stub_mode_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock

    from studioforge.core import engine as engine_module

    fake_log = MagicMock()
    monkeypatch.setattr(engine_module, "log", fake_log)
    manager = EngineManager(stub_config(tmp_path), probe=fake_probe())
    await manager.ensure_engine()
    events = [call.args[0] for call in fake_log.warning.call_args_list]
    assert events == ["engine_stub_mode"]


async def test_list_devices_and_help_come_from_the_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SF_FAKE_GPUS", FAKE_GPUS)
    manager = EngineManager(stub_config(tmp_path), probe=fake_probe())
    devices = await manager.list_devices(STUB_TAG)
    assert devices == [
        "CUDA0: RTX 5090 (32768 MiB, 32768 MiB free)",
        "CUDA1: RTX 3090 (24576 MiB, 24576 MiB free)",
    ]
    assert "--ctx-size" in await manager.help_text(STUB_TAG)
    assert "--tensor-split" in await manager.supported_flags(STUB_TAG)


async def test_engine_features_are_the_stubs(tmp_path: Path) -> None:
    manager = EngineManager(stub_config(tmp_path), probe=fake_probe())
    features = await manager.engine_features()
    assert features.known and features.tag == STUB_TAG
    assert features.cache_ram and features.has("--tensor-split")
    assert manager.stub_features(None) is features
    assert manager.stub_features("b10425").known is False
    # Nothing was written under engines/ for it.
    assert not (tmp_path / "data" / "engines" / STUB_TAG / "features.json").exists()


async def test_smoke_test_of_the_stub_needs_no_model(tmp_path: Path) -> None:
    manager = EngineManager(stub_config(tmp_path), probe=fake_probe())
    ok, detail = await manager.smoke_test(STUB_TAG)
    assert ok
    assert "stub engine" in detail


def test_architecture_table_is_not_read_for_the_stub(tmp_path: Path) -> None:
    manager = EngineManager(stub_config(tmp_path), probe=fake_probe())
    assert manager.architecture_table(None) is None
    assert manager.knows_architecture(None, "llama") is None


# ---------------------------------------------------------------------------
# The build label
# ---------------------------------------------------------------------------


def test_the_build_reads_stub_only_in_stub_mode(tmp_path: Path) -> None:
    plain = build.build_id()
    assert not plain.endswith("+stub")
    EngineManager(Config(data_dir=tmp_path / "plain"), probe=NullGpuProbe())
    assert build.build_id() == plain
    EngineManager(stub_config(tmp_path), probe=fake_probe())
    assert build.build_id() == f"{plain}+stub"
    build.mark_stub_engine(False)
    assert build.build_id() == plain


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_build_state_wires_the_supervisor_to_the_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SF_FAKE_GPUS", FAKE_GPUS)
    gpu_module.reset_probe()
    try:
        state = build_state(stub_config(tmp_path))
        sup = state.supervisor
        assert sup._launch_prefix == state.engine_manager.launch_prefix() != []  # noqa: SLF001
        assert sup._stub_children is True  # noqa: SLF001
        assert state.probe.backend == "fake"
        assert [g.index for g in state.probe.list_gpus()] == [0, 1]
    finally:
        gpu_module.reset_probe()


def test_build_state_without_the_stub_is_unchanged(tmp_path: Path) -> None:
    gpu_module.reset_probe()
    try:
        cfg = Config(data_dir=tmp_path / "data")
        cfg.gpu.backend = "null"
        state = build_state(cfg)
        assert state.supervisor._launch_prefix == []  # noqa: SLF001
        assert state.supervisor._stub_children is False  # noqa: SLF001
    finally:
        gpu_module.reset_probe()


def test_build_state_refuses_the_stub_on_a_real_probe(tmp_path: Path) -> None:
    gpu_module.reset_probe()
    try:
        cfg = stub_config(tmp_path, backend="null")
        with pytest.raises(ConfigError):
            build_state(cfg)
    finally:
        gpu_module.reset_probe()
