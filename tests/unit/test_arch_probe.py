"""The architecture probe and its vocabulary (D66).

2026-09-20..22: K2-Horizon (``general.architecture = 'k2-horizon'``, in no
llama.cpp release) was leased for, planned and spawned three times, and each
child died 0.3 s into startup with ``unknown model architecture``. The answer
was on disk the whole time: the build's ``llama`` library carries every
architecture name it can load as a NUL-terminated literal.

What these pin, below the manager:

* the table: *absent* is ``False`` (the one certain answer), *present* --
  exactly, or only as the tail of a longer literal, which is what a
  tail-merging linker leaves -- is ``True``, and anything that is not an
  architecture-shaped name is ``None``;
* the library is found beside the binary (or in ``lib/``), read once per
  (path, mtime, size), re-read when it changes, and distrusted -- answered
  ``None``, never ``False`` -- when the canary names are missing;
* the engine manager and the supervisor answer for a tag, and an install or
  activation is announced;
* the verdict's words, details and memo, and the 400 it becomes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from studioforge.config import Config
from studioforge.core import engine as engine_mod
from studioforge.core.arch_support import (
    ArchVerdict,
    StartupRejection,
    StartupRejectionMemo,
    file_signature,
    startup_rejection,
)
from studioforge.core.engine import (
    ARCH_TABLE_CANARIES,
    ArchitectureTable,
    EngineManager,
    architecture_table,
    find_llama_library,
    knows_architecture,
)
from studioforge.core.supervisor import Supervisor
from studioforge.errors import UnsupportedArchitectureError

#: The child log's own last words, from the live K2-Horizon attempts.
K2_TAIL = [
    "0.00.129.726 I srv    load_model: loading model "
    "'K2-Horizon-MoVA-36B-A4B-uncensored-Q6_K.gguf'",
    "0.00.323.191 E llama_model_load: error loading model: unknown model architecture: "
    "'k2-horizon'",
    "0.00.323.200 E llama_model_load_from_file_impl: failed to load model",
    "0.00.324.614 E srv  llama_server: exiting due to model loading error",
]


def library_bytes(*names: str, canaries: bool = True) -> bytes:
    """A stand-in ``llama.dll``: a header, then NUL-terminated literals with padding."""
    blob = bytearray(b"MZ\x90\x00\x03\x00\x00\x00" + b"\x00" * 24)
    for name in (*(ARCH_TABLE_CANARIES if canaries else ()), *names):
        blob += name.encode("ascii") + b"\x00\x00\x00"
    blob += b"\xff\xfe" + b"\x00" * 8
    return bytes(blob)


def make_engine(root: Path, tag: str, *names: str, canaries: bool = True) -> Path:
    """``<root>/<tag>/`` with a server binary and a ``llama.dll``; returns the binary."""
    directory = root / tag
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / engine_mod.BIN_NAME
    binary.write_bytes(b"stub")
    (directory / "llama.dll").write_bytes(library_bytes(*names, canaries=canaries))
    return binary


@pytest.fixture(autouse=True)
def _fresh_table_cache() -> Any:
    engine_mod._ARCH_TABLE_CACHE.clear()
    yield
    engine_mod._ARCH_TABLE_CACHE.clear()


# ---------------------------------------------------------------------------
# ArchitectureTable
# ---------------------------------------------------------------------------


def test_absent_is_false_and_present_is_true() -> None:
    table = ArchitectureTable.from_bytes(library_bytes("qwen35moe", "deepseek2", "hy_v3"))
    assert table.knows("qwen35moe") is True
    assert table.knows("deepseek2") is True
    assert table.knows("hy_v3") is True
    assert table.knows("k2-horizon") is False
    assert table.knows("longcat-flash-sparse") is False
    assert table.evidence("qwen35moe") == "exact"
    assert table.evidence("k2-horizon") is None


def test_a_name_left_only_inside_a_tail_merged_literal_still_counts() -> None:
    """GNU ld and lld tail-merge string literals: "gemma" may exist only as the
    end of "recurrentgemma", "bert" only as the end of "nomic-bert". Refusing
    on those would refuse models the build loads, so present-as-a-suffix is
    still "assume supported" -- the evidence just says how it was found."""
    blob = b"\x00llama\x00\x00qwen2\x00\x00recurrentgemma\x00nomic-bert\x00Xmamba\x00"
    table = ArchitectureTable.from_bytes(blob)
    assert table.recognised, "gemma is present, merged into recurrentgemma"
    assert table.knows("gemma") is True
    assert table.evidence("gemma") == "suffix"
    assert table.knows("bert") is True
    assert table.evidence("bert") == "suffix"
    # A letter before the name is an identifier character: still a suffix.
    assert table.evidence("mamba") == "suffix"
    assert table.evidence("recurrentgemma") == "exact"
    assert table.evidence("llama") == "exact"


def test_the_live_builds_qwen2_lives_only_inside_rwkv6qwen2() -> None:
    """The bytes around ``qwen2`` in the live, MSVC-built b11037 ``llama.dll``:
    the linker merged the literal into ``rwkv6qwen2``, so ``\\0qwen2\\0`` occurs
    nowhere. A rule demanding a clean preceding byte would refuse every qwen2
    model -- and lose the ``qwen2`` canary, blinding the whole probe."""
    blob = b"\x00llama\x00gemma\x00\x00olmo2\x00mimo2\x00plamo2\x00rwkv6qwen2\x00mell"
    assert b"\x00qwen2\x00" not in blob
    table = ArchitectureTable.from_bytes(blob)
    assert table.recognised
    assert table.knows("qwen2") is True
    assert table.evidence("qwen2") == "suffix"
    assert table.evidence("rwkv6qwen2") == "exact"


def test_a_longer_literal_does_not_make_its_prefix_known() -> None:
    """``name + NUL`` must be in the library: a longer name that STARTS with it
    is not a match -- "k2-horizon-v2" says nothing about "k2-horizon"."""
    table = ArchitectureTable.from_bytes(library_bytes("k2-horizon-v2"))
    assert table.knows("k2-horizon") is False
    assert table.knows("k2-horizon-v2") is True


def test_a_long_run_keeps_its_last_64_characters() -> None:
    run = "x" * 100 + "tailarch"
    table = ArchitectureTable.from_bytes(library_bytes(run))
    assert table.knows("tailarch") is True
    assert table.evidence("tailarch") == "suffix"


@pytest.mark.parametrize("name", [None, "", "unknown", "Llama", "two words", "a" * 65, "qwen/3"])
def test_a_name_that_is_not_architecture_shaped_cannot_be_told(name: str | None) -> None:
    table = ArchitectureTable.from_bytes(library_bytes("qwen3"))
    assert table.knows(name) is None
    assert table.evidence(name) is None


def test_the_canaries_decide_whether_the_table_is_recognised() -> None:
    assert ArchitectureTable.from_bytes(library_bytes()).recognised
    assert not ArchitectureTable.from_bytes(library_bytes("qwen3", canaries=False)).recognised


# ---------------------------------------------------------------------------
# Finding and caching the library
# ---------------------------------------------------------------------------


def test_the_library_beside_the_binary_answers(tmp_path: Path) -> None:
    binary = make_engine(tmp_path, "b11037", "qwen35", "gemma4")
    assert find_llama_library(binary) == binary.parent / "llama.dll"
    assert find_llama_library(binary.parent) == binary.parent / "llama.dll"
    table = architecture_table(binary)
    assert table is not None
    assert table.library == "llama.dll"
    assert knows_architecture(binary, "qwen35") is True
    assert knows_architecture(binary, "k2-horizon") is False


def test_no_library_means_cannot_tell(tmp_path: Path) -> None:
    """A static build, a hand-placed binary: nothing to read is never a refusal."""
    directory = tmp_path / "b1"
    directory.mkdir()
    binary = directory / engine_mod.BIN_NAME
    binary.write_bytes(b"stub")
    assert find_llama_library(binary) is None
    assert architecture_table(binary) is None
    assert knows_architecture(binary, "k2-horizon") is None


def test_a_library_without_the_canaries_is_not_trusted(tmp_path: Path) -> None:
    """A renamed file or a future layout must read as "cannot tell", not "no"."""
    binary = make_engine(tmp_path, "b1", "qwen35", canaries=False)
    assert architecture_table(binary) is None
    assert knows_architecture(binary, "k2-horizon") is None


@pytest.mark.parametrize(
    ("relative", "name"),
    [
        (".", "libllama.so"),
        (".", "libllama.dylib"),
        (".", "libllama.so.0.0.10425"),
        ("lib", "libllama.so"),
    ],
)
def test_the_platform_library_names_are_found(tmp_path: Path, relative: str, name: str) -> None:
    directory = tmp_path / "b1"
    directory.mkdir()
    binary = directory / "llama-server"
    binary.write_bytes(b"stub")
    target = directory / relative
    target.mkdir(parents=True, exist_ok=True)
    (target / name).write_bytes(library_bytes("qwen35"))
    found = find_llama_library(binary)
    assert found is not None and found.name == name
    assert knows_architecture(binary, "qwen35") is True


def test_the_table_is_read_once_and_again_when_the_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = make_engine(tmp_path, "b1", "qwen35")
    library = binary.parent / "llama.dll"
    reads: list[Path] = []
    real_read = engine_mod._read_architecture_table

    def counting(path: Path) -> Any:
        reads.append(path)
        return real_read(path)

    monkeypatch.setattr(engine_mod, "_read_architecture_table", counting)

    assert knows_architecture(binary, "k2-horizon") is False
    assert knows_architecture(binary, "qwen35") is True
    assert len(reads) == 1, "the bytes-derived table is cached, not re-read per question"

    # A reinstall over the same tag: new bytes, a new size, a new mtime.
    library.write_bytes(library_bytes("qwen35", "k2-horizon"))
    stat = library.stat()
    os.utime(library, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    assert knows_architecture(binary, "k2-horizon") is True
    assert len(reads) == 2


def test_an_unreadable_library_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An I/O error (an antivirus lock, a file mid-write) may pass; "cannot tell"
    is returned without remembering it, and the next question reads again."""
    binary = make_engine(tmp_path, "b1", "qwen35")
    library = binary.parent / "llama.dll"
    real_read_bytes = Path.read_bytes
    locked = {"on": True}

    def read_bytes(self: Path) -> bytes:
        if locked["on"] and self.name == "llama.dll":
            raise PermissionError(13, "locked by another process")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    table, cacheable = engine_mod._read_architecture_table(library)
    assert table is None
    assert cacheable is False
    assert knows_architecture(binary, "k2-horizon") is None

    locked["on"] = False
    assert knows_architecture(binary, "k2-horizon") is False, "the failed read was not cached"


def test_an_empty_library_is_a_settled_cannot_tell(tmp_path: Path) -> None:
    binary = make_engine(tmp_path, "b1")
    (binary.parent / "llama.dll").write_bytes(b"")
    assert engine_mod._read_architecture_table(binary.parent / "llama.dll") == (None, True)
    assert knows_architecture(binary, "k2-horizon") is None


# ---------------------------------------------------------------------------
# Engine manager and supervisor
# ---------------------------------------------------------------------------


def _engine_manager(tmp_path: Path) -> EngineManager:
    config = Config(data_dir=tmp_path / "data")
    return EngineManager(config, probe=None)


def test_the_engine_manager_answers_for_the_active_build_and_for_a_pin(tmp_path: Path) -> None:
    manager = _engine_manager(tmp_path)
    make_engine(manager.engines_dir, "b11037", "qwen35")
    make_engine(manager.engines_dir, "b99000", "qwen35", "k2-horizon")
    manager.set_active("b11037")

    assert manager.knows_architecture(None, "k2-horizon") is False
    assert manager.knows_architecture(None, "qwen35") is True
    assert manager.knows_architecture("b99000", "k2-horizon") is True
    # A pin naming a build that is not installed: cannot tell, the spawn says why.
    assert manager.knows_architecture("b10000", "k2-horizon") is None
    table = manager.architecture_table(None)
    assert table is not None and table.library == "llama.dll"


def test_an_install_or_activation_is_announced(tmp_path: Path) -> None:
    manager = _engine_manager(tmp_path)
    binary = make_engine(manager.engines_dir, "b11037", "qwen35")
    heard: list[tuple[str, str]] = []
    manager.on_engine_change = lambda tag, what: heard.append((tag, what))

    manager.set_active("b11037")
    manager._finalize(
        binary.parent,
        binary,
        "cuda",
        engine_mod._SmokeResult(ok=True, detail="ok"),
        activate=False,
    )
    assert heard == [("b11037", "activate"), ("b11037", "install")]


def test_a_failing_listener_never_fails_the_engine_change(tmp_path: Path) -> None:
    manager = _engine_manager(tmp_path)
    make_engine(manager.engines_dir, "b11037", "qwen35")

    def boom(tag: str, what: str) -> None:
        raise RuntimeError("listener broke")

    manager.on_engine_change = boom
    manager.set_active("b11037")  # must not raise
    assert manager.active() is not None


def test_the_supervisor_answers_for_the_build_a_spawn_would_use(tmp_path: Path) -> None:
    binary = make_engine(tmp_path, "b11037", "qwen35")
    config = Config(data_dir=tmp_path / "data")
    supervisor = Supervisor(config, resolve_binary=lambda _tag: binary)
    table = supervisor.architecture_table(None)
    assert table is not None and table.knows("k2-horizon") is False

    def unresolvable(_tag: str | None) -> Path:
        raise FileNotFoundError("no engine installed")

    assert Supervisor(config, resolve_binary=unresolvable).architecture_table(None) is None


# ---------------------------------------------------------------------------
# arch_support: log parsing, the memo, the verdict
# ---------------------------------------------------------------------------


def test_the_two_startup_markers_are_recognised() -> None:
    assert startup_rejection(K2_TAIL) == ("architecture", "k2-horizon")
    assert startup_rejection(
        [
            "llama_model_load: error loading model: error loading model vocabulary: "
            "unknown pre-tokenizer type: 'moonshot-v9'"
        ]
    ) == ("pre_tokenizer", "moonshot-v9")


@pytest.mark.parametrize(
    "line",
    [
        "gguf_init_from_file: failed to open GGUF file 'x.gguf' (No such file or directory)",
        "error: the file does not exist",
        'error while handling argument "--bogus": unknown argument',
        "CUDA error: out of memory",
    ],
)
def test_other_failures_are_not_architecture_rejections(line: str) -> None:
    """A missing file or a bad flag is not an unsupported model, and must never
    be remembered as one."""
    assert startup_rejection([line]) is None


def test_paths_in_a_listed_failure_are_reduced_to_basenames() -> None:
    from studioforge.core.arch_support import redact_paths

    assert redact_paths("failed: 'D:\\data\\engines\\b1\\llama-server.exe' (x)") == (
        "failed: 'llama-server.exe' (x)"
    )
    assert redact_paths("open /home/example/models/k2.gguf failed") == "open k2.gguf failed"
    assert redact_paths("no path here") == "no path here"


def test_a_file_signature_is_mtime_and_size(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF" + b"\x00" * 60)
    signature = file_signature(path)
    assert signature is not None and signature[1] == 64
    assert file_signature(tmp_path / "missing.gguf") is None
    assert file_signature(None) is None


def _rejection(path: Path, *, tag: str | None = "b11037", **overrides: Any) -> StartupRejection:
    signature = file_signature(path)
    assert signature is not None
    fields: dict[str, Any] = {
        "path": str(path),
        "mtime_ns": signature[0],
        "engine_tag": tag,
        "model_id": "infini/K2-Horizon-Q6_K",
        "kind": "architecture",
        "name": "k2-horizon",
        "architecture": "k2-horizon",
        "engine_signature": (1, 100),
        "scan_marker": 10.0,
    }
    fields.update(overrides)
    return StartupRejection(**fields)


def test_the_memo_is_keyed_on_path_mtime_and_build(tmp_path: Path) -> None:
    path = tmp_path / "k2.gguf"
    path.write_bytes(b"GGUF")
    memo = StartupRejectionMemo()
    first = memo.record(_rejection(path))
    again = memo.record(_rejection(path))
    assert len(memo) == 1
    assert again.failures == 2
    assert again.first_failed_at == first.first_failed_at

    kwargs: dict[str, Any] = {"engine_signature": (1, 100), "scan_marker": 10.0}
    assert memo.lookup(path, "b11037", **kwargs) is not None
    assert memo.lookup(path, "b99000", **kwargs) is None, "another build is another key"
    assert memo.lookup(tmp_path / "other.gguf", "b11037", **kwargs) is None


def test_the_memo_lapses_when_the_model_file_changes(tmp_path: Path) -> None:
    path = tmp_path / "k2.gguf"
    path.write_bytes(b"GGUF")
    memo = StartupRejectionMemo()
    memo.record(_rejection(path))
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 7_000_000_000))
    assert memo.lookup(path, "b11037", engine_signature=(1, 100), scan_marker=10.0) is None


def test_the_memo_lapses_on_a_reinstalled_library_or_a_rescan(tmp_path: Path) -> None:
    path = tmp_path / "k2.gguf"
    path.write_bytes(b"GGUF")
    memo = StartupRejectionMemo()
    memo.record(_rejection(path))
    assert memo.lookup(path, "b11037", engine_signature=(2, 100), scan_marker=10.0) is None
    assert len(memo) == 0, "a stale entry is dropped, not kept"

    memo.record(_rejection(path))
    assert memo.lookup(path, "b11037", engine_signature=(1, 100), scan_marker=11.0) is None
    assert len(memo) == 0

    memo.record(_rejection(path))
    assert memo.snapshot()[0]["name"] == "k2-horizon"
    assert memo.clear() == 1
    assert len(memo) == 0


K2_ID = (
    "InfinimindCreations/K2-Horizon-MoVA-36B-A4B-uncensored-GGUF/"
    "K2-Horizon-MoVA-36B-A4B-uncensored-Q6_K"
)


def test_the_refusal_reads_the_way_the_owner_asked() -> None:
    verdict = ArchVerdict(
        model_id=K2_ID,
        architecture="k2-horizon",
        supported=False,
        engine_tag="b11037",
        source="binary",
    )
    assert verdict.message() == (
        f"'{K2_ID}' uses the model architecture 'k2-horizon', which llama.cpp build b11037 "
        "does not include, so it cannot be loaded. No StudioForge setting changes that; it "
        "needs a llama.cpp build that supports 'k2-horizon'."
    )
    assert verdict.note() == "llama.cpp build b11037 does not include the 'k2-horizon' architecture"
    assert "use another model" in verdict.remedy()
    details = verdict.details()
    assert details == {
        "model_id": K2_ID,
        "architecture": "k2-horizon",
        "engine_tag": "b11037",
        "source": "binary",
        "first_failed_at": None,
        "remedy": verdict.remedy(),
        "engine_tag_pinned": False,
    }
    assert verdict.fields() == {"arch_supported": False, "arch_note": verdict.note()}
    assert verdict.fields(with_message=True)["arch_message"] == verdict.message()


def test_a_supported_or_unknown_verdict_says_nothing() -> None:
    for supported in (True, None):
        verdict = ArchVerdict(model_id="m", architecture="qwen35", supported=supported)
        assert verdict.note() is None
        assert verdict.fields() == {"arch_supported": supported}


def test_a_pinned_build_names_the_setting_that_does_fix_it() -> None:
    """The one case where a StudioForge setting is the remedy: a saved
    engine_tag pin on an old build, while the active build includes it."""
    fixable = ArchVerdict(
        model_id="m",
        architecture="gemma4",
        supported=False,
        engine_tag="b10425",
        pinned=True,
        source="binary",
        active_tag="b11037",
        active_supports=True,
    )
    assert "(this model's engine_tag pin)" in fixable.message()
    assert "clear the model's engine_tag to load it there" in fixable.message()
    assert "No StudioForge setting" not in fixable.message()
    assert fixable.remedy().startswith("clear the model's engine_tag")
    assert fixable.details()["active_engine_tag"] == "b11037"
    assert fixable.details()["active_engine_supports"] is True

    unfixable = ArchVerdict(
        model_id="m",
        architecture="k2-horizon",
        supported=False,
        engine_tag="b10425",
        pinned=True,
        source="binary",
        active_tag="b11037",
        active_supports=False,
    )
    assert "No StudioForge setting changes that" in unfixable.message()


def test_a_runtime_verdict_quotes_the_engine_and_the_rejected_name() -> None:
    pre = ArchVerdict(
        model_id="m",
        architecture="k2-horizon",
        supported=False,
        engine_tag="b11037",
        source="runtime",
        rejected_kind="pre_tokenizer",
        rejected_name="moonshot-v9",
        first_failed_at=123.0,
    )
    assert "the pre-tokenizer 'moonshot-v9'" in pre.message()
    assert "rejected at startup ('unknown pre-tokenizer type')" in pre.message()
    assert pre.note() == "llama.cpp build b11037 does not include the pre-tokenizer 'moonshot-v9'"
    assert pre.details()["rejected"] == {"kind": "pre_tokenizer", "name": "moonshot-v9"}
    assert pre.details()["first_failed_at"] == 123.0

    draft = ArchVerdict(
        model_id="m",
        architecture="qwen35",
        supported=False,
        engine_tag="b11037",
        source="runtime",
        rejected_name="k2-horizon",
    )
    assert "The model's own architecture is 'qwen35'" in draft.message()


def test_the_refusal_is_a_400_naming_the_model() -> None:
    error = ArchVerdict(
        model_id="m",
        architecture="k2-horizon",
        supported=False,
        engine_tag="b11037",
        source="binary",
    ).error()
    assert isinstance(error, UnsupportedArchitectureError)
    assert error.status_code == 400
    payload = error.to_payload()["error"]
    assert payload["code"] == "unsupported_architecture"
    assert payload["type"] == "invalid_request_error"
    assert payload["param"] == "model"
    assert payload["studioforge"]["architecture"] == "k2-horizon"
    assert payload["studioforge"]["engine_tag"] == "b11037"
    assert payload["studioforge"]["source"] == "binary"
