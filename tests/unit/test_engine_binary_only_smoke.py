"""A fresh install has no GGUF: the engine smoke test must not fail it (2.0.0 README install test).

``studioforge engine --update`` / ``engine install`` smoke-tested the new build
with a real micro-load, which needs a model. On an empty library the result was
"no model available" -> not ok -> exit 1 and the engine never activated, on the
one machine state every first install starts in. With no model the smoke test is
now ``--version`` plus ``--list-devices`` only, recorded as
``binary-only (no model available)``; the install succeeds and activates, and the
next check with a model in the library runs the full micro-load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from studioforge.core.engine import (
    BIN_NAME,
    SMOKE_BINARY_ONLY,
    EngineManager,
    _SmokeResult,
    engine_record,
)
from tests.unit.test_engine import (  # noqa: F401
    MIXED_GPUS,
    TAG,
    _active_tag,
    _cli,
    _EngineCliManager,
    _stub_prebuilt,
    tmp_config,
)
from tests.unit.test_engine_backends import manager_for

VERSION_TEXT = "version: 10425 (abcdef0)\nbuilt with cc for x86_64\n"
DEVICES_TEXT = "Available devices:\n  CUDA0: NVIDIA GeForce RTX 5090 (32109 MiB, 31000 MiB free)\n"


def _install_binary(mgr: EngineManager) -> Path:
    directory = mgr.engines_dir / TAG
    directory.mkdir(parents=True)
    (directory / BIN_NAME).write_bytes(b"x")
    return directory


def _stub_capture(mgr: EngineManager, monkeypatch: pytest.MonkeyPatch, *, version_code: int = 0):
    calls: list[list[str]] = []

    async def fake_capture(binary: Path, args: Any) -> tuple[int, str]:
        calls.append(list(args))
        if list(args) == ["--version"]:
            return version_code, VERSION_TEXT if version_code == 0 else "libllama.so: not found"
        if list(args) == ["--list-devices"]:
            return 0, DEVICES_TEXT
        raise AssertionError(f"unexpected engine invocation: {args}")

    monkeypatch.setattr(mgr, "_capture", fake_capture)
    return calls


@pytest.mark.asyncio
async def test_smoke_without_a_model_runs_version_and_list_devices_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = manager_for(tmp_path, MIXED_GPUS, os_token="win")
    _install_binary(mgr)
    calls = _stub_capture(mgr, monkeypatch)
    monkeypatch.setattr(mgr, "find_tiny_model", lambda: None)

    async def no_micro_load(*_a: Any, **_k: Any) -> Any:  # pragma: no cover - must not run
        raise AssertionError("a micro-load needs a model; there is none")

    monkeypatch.setattr(mgr, "_micro_load", no_micro_load)

    ok, result, detail = await mgr.smoke_check(TAG)

    assert ok is True
    assert result == SMOKE_BINARY_ONLY == "binary-only (no model available)"
    assert calls == [["--version"], ["--list-devices"]]
    assert "1 device(s) listed" in detail and "not a full pass" in detail
    info = mgr.get(TAG)
    assert info is not None
    assert info.smoke_result == SMOKE_BINARY_ONLY
    assert info.smoke_tested is False, "binary-only is not a pass: the boot path must re-check"
    assert engine_record(info)["smoke_result"] == SMOKE_BINARY_ONLY
    # The strict API stays strict: smoke_test() still says "no full pass".
    assert (await mgr.smoke_test(TAG))[0] is False


@pytest.mark.asyncio
async def test_a_binary_that_does_not_run_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = manager_for(tmp_path, MIXED_GPUS, os_token="win")
    _install_binary(mgr)
    _stub_capture(mgr, monkeypatch, version_code=127)
    monkeypatch.setattr(mgr, "find_tiny_model", lambda: None)

    ok, result, detail = await mgr.smoke_check(TAG)

    assert (ok, result) == (False, "failed")
    assert "libllama.so" in detail


@pytest.mark.asyncio
async def test_with_a_model_the_full_micro_load_still_decides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = manager_for(tmp_path, MIXED_GPUS, os_token="win")
    _install_binary(mgr)
    _stub_capture(mgr, monkeypatch)
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setattr(mgr, "find_tiny_model", lambda: model)

    async def load_fails(binary: Path, path: Path) -> tuple[bool, str]:
        return False, "never reported /health ok"

    monkeypatch.setattr(mgr, "_micro_load", load_fails)
    ok, result, _ = await mgr.smoke_check(TAG)
    assert (ok, result) == (False, "failed"), "a model that will not load is a real failure"

    async def load_ok(binary: Path, path: Path) -> tuple[bool, str]:
        return True, "ok"

    monkeypatch.setattr(mgr, "_micro_load", load_ok)
    ok, result, _ = await mgr.smoke_check(TAG)
    assert (ok, result) == (True, "passed")
    info = mgr.get(TAG)
    assert info is not None and info.smoke_tested is True and info.smoke_result == "passed"


@pytest.mark.asyncio
async def test_install_on_an_empty_library_records_binary_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = manager_for(tmp_path, MIXED_GPUS, os_token="win")
    _stub_prebuilt(mgr, monkeypatch)

    async def binary_only(tag: str, tiny_model: Any) -> _SmokeResult:
        return _SmokeResult(False, "x", version_ok=True, version_string="b", no_model=True)

    monkeypatch.setattr(mgr, "_smoke", binary_only)

    info = await mgr.install(TAG, activate=True)

    assert info.smoke_result == SMOKE_BINARY_ONLY and info.smoke_tested is False
    assert info.active is True
    meta = json.loads((mgr.engine_dir(TAG) / "engine.json").read_text(encoding="utf-8"))
    assert meta["smoke_result"] == SMOKE_BINARY_ONLY


def test_the_pre_2_0_spelling_reads_as_binary_only(tmp_path: Path) -> None:
    mgr = manager_for(tmp_path, [])
    directory = _install_binary(mgr)
    (directory / "engine.json").write_text(
        json.dumps({"tag": TAG, "variant": "cuda-13.3", "smoke_result": "partial"}),
        encoding="utf-8",
    )
    info = mgr.get(TAG)
    assert info is not None and info.smoke_result == SMOKE_BINARY_ONLY


# --- the CLI: exit code and activation -----------------------------------------


class _EmptyLibraryCliManager(_EngineCliManager):
    """The CLI double, with a smoke check that reports the binary-only outcome."""

    async def smoke_check(self, tag: str, **_kwargs: Any) -> tuple[bool, str, str]:
        type(self).calls.append(f"smoke_test({tag})")
        return True, SMOKE_BINARY_ONLY, f"engine {tag}: no GGUF for the micro-load"


def test_cli_engine_update_succeeds_and_activates_on_an_empty_library(
    monkeypatch: pytest.MonkeyPatch,
    tmp_config: Any,  # noqa: F811
) -> None:
    result = _cli(
        monkeypatch, tmp_config, "engine", "--update", manager_cls=_EmptyLibraryCliManager
    )

    assert result.exit_code == 0, result.output
    assert "activate(b10488)" in _EmptyLibraryCliManager.calls
    assert _active_tag(tmp_config) == "b10488"
    assert tmp_config.engine.pinned_tag == "b10488"
    assert "binary-only (no model available)" in result.output


def test_cli_engine_update_subcommand_also_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_config: Any,  # noqa: F811
) -> None:
    result = _cli(monkeypatch, tmp_config, "engine", "update", manager_cls=_EmptyLibraryCliManager)
    assert result.exit_code == 0, result.output
    assert _active_tag(tmp_config) == "b10488"


def test_cli_engine_update_still_exits_1_when_the_binary_itself_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_config: Any,  # noqa: F811
) -> None:
    class Broken(_EngineCliManager):
        async def smoke_check(self, tag: str, **_kwargs: Any) -> tuple[bool, str, str]:
            type(self).calls.append(f"smoke_test({tag})")
            return False, "failed", "libllama.so: cannot open shared object file"

    result = _cli(monkeypatch, tmp_config, "engine", "--update", manager_cls=Broken)

    assert result.exit_code == 1
    assert "smoke test FAILED" in result.output
    assert _active_tag(tmp_config) is None
