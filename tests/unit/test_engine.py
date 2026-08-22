"""Engine manager tests.

Two layers:

* pure-logic tests (asset selection, help parsing, zip flattening, build
  command construction) that run anywhere with mocks;
* live tests against the *real* pinned ``b10425`` engine in the dev data dir,
  each guarded by a skipif so the suite still runs on a machine without it.

The live smoke test is the important one: it is the only thing that proves the
download -> extract -> launch -> ``/health`` path actually works end to end.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest

from studioforge.config import Config
from studioforge.core.engine import (
    BIN_NAME,
    ENGINE_TAG_RE,
    EngineAsset,
    EngineError,
    EngineManager,
    _SmokeResult,  # internal: build tests stub the smoke step
    _verify_archive,  # internal: the pre-extract integrity check
    build_assets,
    build_number,
    extract_engine_archive,
    extract_engine_zip,
    find_server_binary,
    flags_from_help,
    guess_variant,
    parse_asset_name,
    removed_flags_from_help,
)
from studioforge.types import GpuInfo

TAG = "b10425"

# --- the real dev data dir -------------------------------------------------
#
# A handful of tests below exercise the ACTUAL installed engine binary. They
# find it exactly the way the app does -- SF_DATA_DIR if set, else <repo>/data
# -- and skip when it is not there, so a fresh checkout with no engine
# installed still runs the whole file.

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _real_data_dir() -> Path | None:
    env = os.environ.get("SF_DATA_DIR")
    candidates = ([Path(env)] if env else []) + [_REPO_ROOT / "data"]
    for candidate in candidates:
        if (candidate / "engines" / TAG).is_dir():
            return candidate
    return None


REAL_DATA_DIR = _real_data_dir()
REAL_ENGINE = (REAL_DATA_DIR / "engines" / TAG) if REAL_DATA_DIR else None
HAVE_ENGINE = REAL_ENGINE is not None and (REAL_ENGINE / BIN_NAME).is_file()

needs_engine = pytest.mark.skipif(
    not HAVE_ENGINE, reason=f"real {TAG} engine not installed in the dev data dir"
)
windows_only = pytest.mark.skipif(os.name != "nt", reason="engine fixture is a Windows build")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class StubProbe:
    """Structural stand-in for ``core.gpu.GpuProbe``.

    Implemented locally rather than imported so these tests stay independent of
    the GPU module and of whether this box has any GPU at all.
    """

    def __init__(self, gpus: list[GpuInfo], cuda: tuple[int, int] | None) -> None:
        self._gpus = gpus
        self._cuda = cuda

    @property
    def backend(self) -> str:
        return "fake"

    def available(self) -> bool:
        return bool(self._gpus)

    def list_gpus(self) -> list[GpuInfo]:
        return list(self._gpus)

    def get_gpu(self, index: int) -> GpuInfo | None:
        return next((g for g in self._gpus if g.index == index), None)

    def driver_version(self) -> str | None:
        return "610.88"

    def cuda_driver_version(self) -> tuple[int, int] | None:
        return self._cuda

    def shutdown(self) -> None:
        return None


def _gpu(index: int, name: str, cc: tuple[int, int]) -> GpuInfo:
    return GpuInfo(
        index=index,
        name=name,
        total_bytes=32 * 1024**3,
        free_bytes=30 * 1024**3,
        compute_capability=cc,
    )


MIXED_GPUS = [
    _gpu(0, "NVIDIA GeForce RTX 5090", (12, 0)),
    _gpu(1, "NVIDIA GeForce RTX 5090", (12, 0)),
    _gpu(2, "NVIDIA GeForce RTX 3090", (8, 6)),
    _gpu(3, "NVIDIA GeForce RTX 3090", (8, 6)),
]


def _asset(name: str, size: int = 100) -> dict[str, Any]:
    return {
        "name": name,
        "browser_download_url": f"https://example.invalid/{name}",
        "size": size,
    }


#: Mirrors the real GitHub asset list at b10425 (read back from the release API
#: on 2026-08-22). Two facts this fixture must keep honest, because the suite
#: was green for months with names that never existed upstream: every Linux
#: and macOS archive is a ``.tar.gz``, and there is NO ``ubuntu-cuda`` asset --
#: Linux+NVIDIA is the source-build path.
RAW_ASSETS: list[dict[str, Any]] = [
    _asset(f"llama-{TAG}-bin-win-cuda-12.4-x64.zip", 111),
    _asset(f"llama-{TAG}-bin-win-cuda-13.3-x64.zip", 222),
    _asset(f"llama-{TAG}-bin-win-cpu-x64.zip", 333),
    _asset(f"llama-{TAG}-bin-win-vulkan-x64.zip", 444),
    _asset(f"llama-{TAG}-bin-win-rocm-7.14-x64.zip", 555),
    _asset(f"llama-{TAG}-bin-win-cpu-arm64.zip", 666),
    _asset(f"llama-{TAG}-bin-ubuntu-x64.tar.gz", 777),
    _asset(f"llama-{TAG}-bin-ubuntu-arm64.tar.gz", 888),
    _asset(f"llama-{TAG}-bin-ubuntu-vulkan-x64.tar.gz", 999),
    _asset(f"llama-{TAG}-bin-macos-arm64.tar.gz", 1000),
    _asset("cudart-llama-bin-win-cuda-12.4-x64.zip", 11),
    _asset("cudart-llama-bin-win-cuda-13.3-x64.zip", 22),
    _asset(f"llama-{TAG}-bin-win-cuda-13.3-x64.zip.sha256", 1),
]

ASSETS = build_assets(TAG, RAW_ASSETS)


@pytest.fixture
def tmp_config(tmp_path: Path) -> Config:
    config = Config(data_dir=tmp_path / "data")
    config.ensure_dirs()
    return config


@pytest.fixture
def manager(tmp_config: Config) -> EngineManager:
    mgr = EngineManager(tmp_config, probe=StubProbe(MIXED_GPUS, (13, 3)))
    mgr.os_token = "win"
    mgr.arch_token = "x64"
    return mgr


@pytest.fixture
def real_manager() -> EngineManager:
    assert REAL_DATA_DIR is not None
    config = Config(data_dir=REAL_DATA_DIR)
    return EngineManager(config, probe=StubProbe(MIXED_GPUS, (13, 3)))


# ---------------------------------------------------------------------------
# Asset parsing
# ---------------------------------------------------------------------------


def test_parse_asset_name_variants() -> None:
    assert parse_asset_name(f"llama-{TAG}-bin-win-cuda-13.3-x64.zip") == (
        TAG,
        "win",
        "cuda-13.3",
        "x64",
    )
    assert parse_asset_name(f"llama-{TAG}-bin-win-cpu-x64.zip") == (TAG, "win", "cpu", "x64")
    # The Linux/macOS archives are tarballs; a zip-only parser dropped every one.
    assert parse_asset_name(f"llama-{TAG}-bin-ubuntu-x64.tar.gz") == (TAG, "ubuntu", "cpu", "x64")
    assert parse_asset_name(f"llama-{TAG}-bin-ubuntu-vulkan-x64.tar.gz") == (
        TAG,
        "ubuntu",
        "vulkan",
        "x64",
    )
    assert parse_asset_name(f"llama-{TAG}-bin-ubuntu-rocm-7.14-x64.tar.gz") == (
        TAG,
        "ubuntu",
        "rocm-7.14",
        "x64",
    )
    assert parse_asset_name(f"llama-{TAG}-bin-win-rocm-7.14-x64.zip") == (
        TAG,
        "win",
        "rocm-7.14",
        "x64",
    )
    assert parse_asset_name(f"llama-{TAG}-bin-macos-arm64.tar.gz") == (
        TAG,
        "macos",
        "cpu",
        "arm64",
    )
    assert parse_asset_name("cudart-llama-bin-win-cuda-13.3-x64.zip") is None


def test_build_assets_pairs_cudart() -> None:
    by_name = {a.name: a for a in ASSETS}
    cuda133 = by_name[f"llama-{TAG}-bin-win-cuda-13.3-x64.zip"]
    assert cuda133.variant == "cuda-13.3"
    assert cuda133.cuda_version == (13, 3)
    assert cuda133.needs_cudart is True
    assert cuda133.cudart_url is not None
    assert cuda133.cudart_url.endswith("cudart-llama-bin-win-cuda-13.3-x64.zip")
    assert cuda133.size_bytes == 222

    # A cpu build never needs a cuda runtime bundle.
    assert by_name[f"llama-{TAG}-bin-win-cpu-x64.zip"].needs_cudart is False
    # Non-archive assets (checksums) are ignored entirely.
    assert f"llama-{TAG}-bin-win-cuda-13.3-x64.zip.sha256" not in by_name


# ---------------------------------------------------------------------------
# select_asset: the eligibility rule
# ---------------------------------------------------------------------------


def test_select_asset_driver_13_0_picks_12_4(manager: EngineManager) -> None:
    """13.3 > driver 13.0, so only the 12.4 build can actually load."""
    chosen = manager.select_asset(ASSETS, gpus=MIXED_GPUS, cuda_driver=(13, 0))
    assert chosen is not None
    assert chosen.variant == "cuda-12.4"


def test_select_asset_driver_13_4_picks_13_3(manager: EngineManager) -> None:
    chosen = manager.select_asset(ASSETS, gpus=MIXED_GPUS, cuda_driver=(13, 4))
    assert chosen is not None
    assert chosen.variant == "cuda-13.3"


def test_select_asset_driver_13_3_picks_13_3_exactly(manager: EngineManager) -> None:
    chosen = manager.select_asset(ASSETS, gpus=MIXED_GPUS, cuda_driver=(13, 3))
    assert chosen is not None
    assert chosen.variant == "cuda-13.3"


def test_select_asset_old_driver_returns_none_not_cpu(manager: EngineManager) -> None:
    """Driver 12.0 cannot run a 12.4 build; GPU-only forbids falling back to cpu."""
    assert manager.select_asset(ASSETS, gpus=MIXED_GPUS, cuda_driver=(12, 0)) is None


def test_select_asset_never_picks_cpu_or_vulkan(manager: EngineManager) -> None:
    non_cuda = [a for a in ASSETS if not a.is_cuda]
    assert non_cuda, "fixture should contain cpu/vulkan/rocm assets"
    chosen = manager.select_asset(non_cuda, gpus=MIXED_GPUS, cuda_driver=(13, 3))
    assert chosen is None


def test_select_asset_explicit_variant_overrides_driver(tmp_config: Config) -> None:
    tmp_config.engine.cuda_variant = "13.3"
    mgr = EngineManager(tmp_config, probe=StubProbe(MIXED_GPUS, (12, 0)))
    mgr.os_token, mgr.arch_token = "win", "x64"
    chosen = mgr.select_asset(ASSETS, gpus=MIXED_GPUS, cuda_driver=(12, 0))
    assert chosen is not None
    assert chosen.variant == "cuda-13.3"


def test_select_asset_explicit_variant_missing_raises(tmp_config: Config) -> None:
    tmp_config.engine.cuda_variant = "11.8"
    mgr = EngineManager(tmp_config, probe=StubProbe(MIXED_GPUS, (13, 3)))
    mgr.os_token, mgr.arch_token = "win", "x64"
    with pytest.raises(EngineError) as excinfo:
        mgr.select_asset(ASSETS, gpus=MIXED_GPUS, cuda_driver=(13, 3))
    message = str(excinfo.value)
    assert "11.8" in message
    assert "cuda-13.3" in message and "cuda-12.4" in message


def test_select_asset_explicit_cpu_is_refused(tmp_config: Config) -> None:
    tmp_config.engine.cuda_variant = "cpu"
    mgr = EngineManager(tmp_config, probe=StubProbe(MIXED_GPUS, (13, 3)))
    mgr.os_token, mgr.arch_token = "win", "x64"
    with pytest.raises(EngineError, match="GPU-only"):
        mgr.select_asset(ASSETS, gpus=MIXED_GPUS, cuda_driver=(13, 3))


def test_select_asset_linux_nvidia_has_no_prebuilt(manager: EngineManager) -> None:
    """Upstream ships no Linux CUDA archive: None here is what routes a Linux+NVIDIA
    box to the source build, and the cpu/vulkan tarballs must never be picked."""
    manager.os_token, manager.arch_token = "ubuntu", "x64"
    assert manager.select_asset(ASSETS, gpus=MIXED_GPUS, cuda_driver=(13, 3)) is None
    manager.os_token, manager.arch_token = "ubuntu", "arm64"
    assert manager.select_asset(ASSETS, gpus=MIXED_GPUS, cuda_driver=(13, 3)) is None


def test_select_asset_linux_amd_takes_the_rocm_tarball(manager: EngineManager) -> None:
    """The one Linux box a prebuilt serves (ROCm is published as a tarball)."""
    manager.os_token, manager.arch_token = "ubuntu", "x64"
    amd = [_gpu(0, "AMD Radeon RX 7900 XTX", (0, 0))]
    with_rocm = build_assets(
        TAG, [*RAW_ASSETS, _asset(f"llama-{TAG}-bin-ubuntu-rocm-7.14-x64.tar.gz", 1234)]
    )

    chosen = manager.select_asset(with_rocm, gpus=amd, cuda_driver=None)

    assert chosen is not None
    assert chosen.name == f"llama-{TAG}-bin-ubuntu-rocm-7.14-x64.tar.gz"
    assert chosen.needs_cudart is False


def test_select_asset_macos_arm64_has_no_cuda(manager: EngineManager) -> None:
    manager.os_token, manager.arch_token = "macos", "arm64"
    assert manager.select_asset(ASSETS, gpus=MIXED_GPUS, cuda_driver=None) is None


def test_select_asset_amd_box_takes_rocm(manager: EngineManager) -> None:
    amd = [
        GpuInfo(index=0, name="AMD Radeon RX 7900 XTX", total_bytes=1, free_bytes=1),
    ]
    chosen = manager.select_asset(ASSETS, gpus=amd, cuda_driver=None)
    assert chosen is not None
    assert chosen.variant == "rocm-7.14"


# ---------------------------------------------------------------------------
# help parsing (pure)
# ---------------------------------------------------------------------------

SAMPLE_HELP = """----- common params -----

-h,    --help, --usage                  print usage and exit
-c,    --ctx-size N                     size of the prompt context (default: 4096)
                                        (env: LLAMA_ARG_CTX_SIZE)
-fa,   --flash-attn, -fa <on|off|auto>  set Flash Attention use (default: auto)
-sm,   --split-mode {none,layer,row}    how to split the model across GPUs
--spec-draft-n-max N                    number of tokens to draft (default: 3)
--spec-draft-type-k, -ctkd, --cache-type-k-draft TYPE
                                        KV cache data type for K for the draft model
--draft, --draft-n, --draft-max N       the argument has been removed. use --spec-draft-n-max or
                                        --spec-ngram-mod-n-max
                                        (env: LLAMA_ARG_DRAFT_MAX)
--draft-min, --draft-n-min N            the argument has been removed. use --spec-draft-n-min or
                                        --spec-ngram-mod-n-min
"""


def test_flags_from_help_collects_aliases() -> None:
    flags = flags_from_help(SAMPLE_HELP)
    assert {"-h", "--help", "-c", "--ctx-size", "--flash-attn", "-sm", "--split-mode"} <= flags
    assert {"--spec-draft-type-k", "-ctkd", "--cache-type-k-draft"} <= flags
    # Value placeholders must not leak in as flags.
    assert "{none" not in flags
    assert "N" not in flags


def test_removed_flags_from_help_extracts_replacement() -> None:
    removed = removed_flags_from_help(SAMPLE_HELP)
    assert removed["--draft-max"] == "--spec-draft-n-max"
    assert removed["--draft"] == "--spec-draft-n-max"
    assert removed["--draft-n-min"] == "--spec-draft-n-min"
    # A live alias is not "removed", even though it is an old spelling.
    assert "--cache-type-k-draft" not in removed


# ---------------------------------------------------------------------------
# zip flattening
# ---------------------------------------------------------------------------


def _make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def test_extract_flattens_single_top_level_dir(tmp_path: Path) -> None:
    src = _make_zip(
        tmp_path / "flat.zip",
        {
            f"llama-{TAG}-bin/{BIN_NAME}": b"BINARY",
            f"llama-{TAG}-bin/ggml-cuda.dll": b"LIB",
        },
    )
    dest = tmp_path / "engines" / TAG
    binary = extract_engine_zip(src, dest)
    assert binary == dest / BIN_NAME
    assert (dest / BIN_NAME).read_bytes() == b"BINARY"
    assert (dest / "ggml-cuda.dll").is_file()
    assert not (dest / f"llama-{TAG}-bin").exists()


def test_extract_flattens_nested_build_bin(tmp_path: Path) -> None:
    """Linux archives nest ``<name>/build/bin/`` -- the whole prefix is stripped."""
    src = _make_zip(
        tmp_path / "nested.zip",
        {
            f"llama-{TAG}/build/bin/{BIN_NAME}": b"BINARY",
            f"llama-{TAG}/build/bin/libggml.so": b"LIB",
        },
    )
    dest = tmp_path / "engines" / TAG
    extract_engine_zip(src, dest)
    assert (dest / BIN_NAME).is_file()
    assert (dest / "libggml.so").is_file()


def test_extract_keeps_flat_archive_flat(tmp_path: Path) -> None:
    src = _make_zip(tmp_path / "win.zip", {BIN_NAME: b"BINARY", "ggml.dll": b"LIB"})
    dest = tmp_path / "engines" / TAG
    extract_engine_zip(src, dest)
    assert (dest / BIN_NAME).is_file()
    assert (dest / "ggml.dll").is_file()


def test_extract_skips_path_traversal(tmp_path: Path) -> None:
    src = _make_zip(tmp_path / "evil.zip", {"../../pwned.txt": b"NOPE", f"a/{BIN_NAME}": b"BINARY"})
    dest = tmp_path / "engines" / TAG
    extract_engine_zip(src, dest)
    assert not (tmp_path / "pwned.txt").exists()


# ---------------------------------------------------------------------------
# Local inventory
# ---------------------------------------------------------------------------


def _fake_engine(root: Path, tag: str, mtime: float) -> Path:
    directory = root / tag
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / BIN_NAME
    binary.write_bytes(b"fake")
    os.utime(directory, (mtime, mtime))
    return directory


def test_prune_removes_only_the_oldest(tmp_config: Config) -> None:
    root = tmp_config.engines_dir
    old = _fake_engine(root, "b10001", 1_000_000)
    new = _fake_engine(root, "b10002", 2_000_000)
    mgr = EngineManager(tmp_config, probe=StubProbe([], None))

    assert {i.tag for i in mgr.installed()} == {"b10001", "b10002"}
    removed = mgr.prune(keep=1)
    assert removed == ["b10001"]
    assert not old.exists()
    assert new.exists()


def test_prune_never_removes_active_or_pinned(tmp_config: Config) -> None:
    root = tmp_config.engines_dir
    _fake_engine(root, TAG, 1_000)  # pinned_tag default, oldest
    _fake_engine(root, "b10500", 2_000)
    _fake_engine(root, "b10600", 3_000)
    mgr = EngineManager(tmp_config, probe=StubProbe([], None))
    mgr.set_active("b10500")

    removed = mgr.prune(keep=1)
    assert removed == []  # b10600 within keep, b10500 active, TAG pinned
    assert (root / TAG).is_dir()


def test_active_json_round_trip(tmp_config: Config) -> None:
    _fake_engine(tmp_config.engines_dir, "b10500", 2_000)
    mgr = EngineManager(tmp_config, probe=StubProbe([], None))
    mgr.set_active("b10500")

    payload = json.loads((tmp_config.engines_dir / "active.json").read_text(encoding="utf-8"))
    assert payload["tag"] == "b10500"
    info = mgr.active()
    assert info is not None
    assert info.tag == "b10500"
    assert info.active is True


def test_server_binary_missing_tag_raises(manager: EngineManager) -> None:
    with pytest.raises(EngineError, match="not installed"):
        manager.server_binary("b99999")


def test_find_server_binary_handles_nested_bin(tmp_path: Path) -> None:
    nested = tmp_path / "bin"
    nested.mkdir()
    (nested / BIN_NAME).write_bytes(b"x")
    assert find_server_binary(tmp_path) == nested / BIN_NAME


# --- against the real install ---------------------------------------------


@needs_engine
def test_installed_finds_real_engine(real_manager: EngineManager) -> None:
    tags = {info.tag for info in real_manager.installed()}
    assert TAG in tags

    info = real_manager.get(TAG)
    assert info is not None
    assert info.server_binary.is_file()
    assert info.server_binary.name == BIN_NAME
    assert real_manager.server_binary(TAG) == info.server_binary


@needs_engine
def test_get_unknown_tag_is_none(real_manager: EngineManager) -> None:
    assert real_manager.get("b00001") is None


@needs_engine
def test_variant_inferred_from_shipped_libraries(real_manager: EngineManager) -> None:
    """A hand-placed engine dir still reports a useful variant."""
    info = real_manager.get(TAG)
    assert info is not None
    assert info.variant.startswith("cuda")


def test_guess_variant_reads_ggml_libraries(tmp_path: Path) -> None:
    assert guess_variant(tmp_path) == "unknown"
    (tmp_path / ("libggml-cuda.so" if os.name != "nt" else "ggml-cuda.dll")).write_bytes(b"x")
    assert guess_variant(tmp_path) == "cuda"


# ---------------------------------------------------------------------------
# build_from_source: command construction only (never compile)
# ---------------------------------------------------------------------------


def _simulate_build_step(cmd: list[str], phase: str, log_path: Path | None) -> None:
    """Create the on-disk artefacts a real clone/build would leave behind.

    Kept synchronous and outside the async fake runner so the build tests never
    invoke git or cmake -- a real CUDA compile is 30+ minutes, so only command
    construction is under test here.
    """
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(" ".join(cmd) + "\n")
    if phase == "clone":
        src = Path(cmd[-1])
        src.mkdir(parents=True, exist_ok=True)
        (src / "CMakeLists.txt").write_text("project(llama)\n", encoding="utf-8")
    elif phase == "build":
        bin_dir = Path(cmd[2]) / "bin" / "Release"
        bin_dir.mkdir(parents=True, exist_ok=True)
        (bin_dir / BIN_NAME).write_bytes(b"fake")


def test_cuda_arches_dedupes_and_sorts_numerically(manager: EngineManager) -> None:
    assert manager.cuda_arches(MIXED_GPUS) == ["86", "120"]


@pytest.mark.asyncio
async def test_build_from_source_command_construction(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = EngineManager(tmp_config, probe=StubProbe(MIXED_GPUS, (13, 3)))
    calls: list[list[str]] = []

    async def fake_run(cmd: Any, *, cwd: Any, log_path: Path, phase: str, progress: Any) -> int:
        calls.append(list(cmd))
        _simulate_build_step(list(cmd), phase, log_path)
        return 0

    async def fake_smoke(tag: str, tiny_model: Any) -> _SmokeResult:
        return _SmokeResult(True, "stubbed", version_ok=True, version_string="build 10425")

    monkeypatch.setattr(mgr, "_run_logged", fake_run)
    monkeypatch.setattr(mgr, "_smoke", fake_smoke)
    monkeypatch.setenv("SF_VENDOR_LLAMA_CPP", str(tmp_config.data_dir / "no-vendor"))

    arches = mgr.cuda_arches(MIXED_GPUS)
    info = await mgr.build_from_source(TAG, arches=arches)

    clone = next(c for c in calls if c[0] == "git")
    assert clone[:5] == ["git", "clone", "--depth", "1", "--branch"]
    assert clone[5] == TAG

    configure = next(c for c in calls if c[0] == "cmake" and "-S" in c)
    assert "-DGGML_CUDA=ON" in configure
    assert "-DCMAKE_CUDA_ARCHITECTURES=86;120" in configure
    assert "-DCMAKE_BUILD_TYPE=Release" in configure
    assert "-DLLAMA_BUILD_SERVER=ON" in configure
    # The copied llama-server must find its own .so files (and the CUDA
    # toolkit's) after the build tree is gone: $ORIGIN, literal, no shell.
    assert "-DCMAKE_INSTALL_RPATH=$ORIGIN" in configure
    assert "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON" in configure
    assert "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON" in configure

    build = next(c for c in calls if c[0] == "cmake" and "--build" in c)
    assert "--parallel" in build
    assert "llama-server" in build

    assert info.tag == f"{TAG}-local"
    assert info.variant == "source-local"
    assert info.build_log is not None
    assert info.build_log.name == f"engine-build-{TAG}.log"
    assert info.build_log.is_file()
    assert info.server_binary.is_file()
    assert info.server_binary.parent == tmp_config.engines_dir / f"{TAG}-local"


@pytest.mark.asyncio
async def test_build_from_source_dedupes_arches(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = EngineManager(tmp_config, probe=StubProbe(MIXED_GPUS, (13, 3)))
    seen: list[list[str]] = []

    async def fake_run(cmd: Any, **kwargs: Any) -> int:
        seen.append(list(cmd))
        _simulate_build_step(list(cmd), str(kwargs.get("phase")), kwargs.get("log_path"))
        return 0

    async def fake_smoke(tag: str, tiny_model: Any) -> _SmokeResult:
        return _SmokeResult(True, "stubbed")

    monkeypatch.setattr(mgr, "_run_logged", fake_run)
    monkeypatch.setattr(mgr, "_smoke", fake_smoke)
    monkeypatch.setenv("SF_VENDOR_LLAMA_CPP", str(tmp_config.data_dir / "nope"))

    await mgr.build_from_source(TAG, arches=["120", "86", "120", "86", "86"])
    configure = next(c for c in seen if "-S" in c)
    assert "-DCMAKE_CUDA_ARCHITECTURES=86;120" in configure


@pytest.mark.asyncio
async def test_build_from_source_without_arches_raises(tmp_config: Config) -> None:
    mgr = EngineManager(tmp_config, probe=StubProbe([], None))
    with pytest.raises(EngineError, match="CMAKE_CUDA_ARCHITECTURES"):
        await mgr.build_from_source(TAG, arches=[])


# ---------------------------------------------------------------------------
# ensure_engine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_engine_raises_when_nothing_eligible(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_config.engine.allow_source_build = False
    mgr = EngineManager(tmp_config, probe=StubProbe(MIXED_GPUS, (12, 0)))
    mgr.os_token, mgr.arch_token = "win", "x64"

    async def fake_assets(tag: str) -> list[EngineAsset]:
        return list(ASSETS)

    monkeypatch.setattr(mgr, "list_assets", fake_assets)
    with pytest.raises(EngineError) as excinfo:
        await mgr.ensure_engine()
    message = str(excinfo.value)
    assert "sm_120" in message  # names the detected arch
    assert "cuda-13.3" in message  # names what was available
    assert "allow_source_build" in message


@pytest.mark.asyncio
async def test_ensure_engine_falls_back_to_source_build(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = EngineManager(tmp_config, probe=StubProbe(MIXED_GPUS, (12, 0)))
    mgr.os_token, mgr.arch_token = "win", "x64"
    built: dict[str, Any] = {}

    async def fake_assets(tag: str) -> list[EngineAsset]:
        return list(ASSETS)

    async def fake_build(tag: str, *, arches: Any, progress: Any = None) -> Any:
        built["tag"] = tag
        built["arches"] = list(arches)
        directory = _fake_engine(tmp_config.engines_dir, f"{tag}-local", 1_000)
        return mgr._finalize(  # exercises the real finalizer + active.json write
            directory,
            directory / BIN_NAME,
            "source-local",
            _SmokeResult(True, "ok"),
            activate=True,
        )

    monkeypatch.setattr(mgr, "list_assets", fake_assets)
    monkeypatch.setattr(mgr, "build_from_source", fake_build)

    info = await mgr.ensure_engine()
    assert built["tag"] == TAG
    assert built["arches"] == ["86", "120"]
    assert info.variant == "source-local"


@pytest.mark.asyncio
async def test_install_falls_back_to_source_build_on_linux(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`engine --update`, the Setup tab and the MCP install all go through
    install(); on Linux+NVIDIA nothing prebuilt ever matches, and raising there
    made the first-run step in deploy/README.md fail while boot quietly built."""
    mgr = EngineManager(tmp_config, probe=StubProbe(MIXED_GPUS, (13, 3)))
    mgr.os_token, mgr.arch_token = "ubuntu", "x64"
    built: dict[str, Any] = {}

    async def fake_assets(tag: str) -> list[EngineAsset]:
        return list(ASSETS)

    async def fake_build(tag: str, *, arches: Any, progress: Any = None) -> Any:
        built["tag"] = tag
        built["arches"] = list(arches)
        directory = _fake_engine(tmp_config.engines_dir, f"{tag}-local", 1_000)
        return mgr._finalize(
            directory,
            directory / BIN_NAME,
            "source-local",
            _SmokeResult(True, "ok"),
            activate=True,
        )

    monkeypatch.setattr(mgr, "list_assets", fake_assets)
    monkeypatch.setattr(mgr, "build_from_source", fake_build)

    info = await mgr.install(TAG)

    assert built == {"tag": TAG, "arches": ["86", "120"]}
    assert info.tag == f"{TAG}-local" and info.variant == "source-local"


@pytest.mark.asyncio
async def test_install_names_the_source_prerequisites_when_building_is_off(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_config.engine.allow_source_build = False
    mgr = EngineManager(tmp_config, probe=StubProbe(MIXED_GPUS, (13, 3)))
    mgr.os_token, mgr.arch_token = "ubuntu", "x64"

    async def fake_assets(tag: str) -> list[EngineAsset]:
        return list(ASSETS)

    monkeypatch.setattr(mgr, "list_assets", fake_assets)
    with pytest.raises(EngineError) as excinfo:
        await mgr.install(TAG)

    message = str(excinfo.value)
    assert "cmake" in message and "allow_source_build" in message
    # The variants named are this platform's, not the Windows downloads.
    assert "cuda-13.3" not in message
    assert "cpu, vulkan" in message


@pytest.mark.asyncio
async def test_install_reuses_an_existing_local_build(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D27's reuse rule on the explicit path: a second `engine --update` on Linux
    must not recompile for twenty minutes to reach the binary it already has."""
    mgr = EngineManager(tmp_config, probe=StubProbe(MIXED_GPUS, (13, 3)))
    mgr.os_token, mgr.arch_token = "ubuntu", "x64"
    _fake_engine(tmp_config.engines_dir, f"{TAG}-local", 1_000)

    async def fake_smoke(tag: str, tiny_model: Any) -> _SmokeResult:
        return _SmokeResult(True, "stubbed", version_ok=True)

    async def boom(tag: str) -> list[EngineAsset]:  # pragma: no cover - must not run
        raise AssertionError("install hit the network for an engine it already has")

    monkeypatch.setattr(mgr, "_smoke", fake_smoke)
    monkeypatch.setattr(mgr, "list_assets", boom)

    info = await mgr.install(TAG)

    assert info.tag == f"{TAG}-local"
    assert info.active is True


# ---------------------------------------------------------------------------
# Tarball archives (every ubuntu/macos asset upstream publishes)
# ---------------------------------------------------------------------------


def _tarball(path: Path, members: dict[str, bytes], *, link: str | None = None) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755 if name.endswith(BIN_NAME) else 0o644
            archive.addfile(info, io.BytesIO(data))
        if link is not None:
            info = tarfile.TarInfo(link)
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            archive.addfile(info)
    return path


def test_extract_engine_archive_flattens_a_tarball_and_skips_unsafe_members(
    tmp_path: Path,
) -> None:
    archive = _tarball(
        tmp_path / f"llama-{TAG}-bin-ubuntu-x64.tar.gz",
        {
            f"llama-{TAG}/{BIN_NAME}": b"ELF",
            f"llama-{TAG}/libllama.so": b"so",
            f"llama-{TAG}/../escape.txt": b"nope",
        },
        link=f"llama-{TAG}/evil-link",
    )
    dest = tmp_path / "engines" / TAG

    binary = extract_engine_archive(archive, dest)

    assert binary == dest / BIN_NAME, "the shared llama-bNNNN/ prefix is stripped"
    assert (dest / "libllama.so").read_bytes() == b"so"
    assert not (tmp_path / "engines" / "escape.txt").exists()
    assert not (dest / "evil-link").exists()
    if os.name != "nt":
        assert os.access(binary, os.X_OK), "the tarball's execute bit is carried over"


def test_verify_archive_dispatches_on_the_suffix(tmp_path: Path) -> None:
    good = _tarball(tmp_path / "good.tar.gz", {f"llama-{TAG}/{BIN_NAME}": b"ELF"})
    _verify_archive(good)  # does not raise

    truncated = tmp_path / "truncated.tar.gz"
    truncated.write_bytes(good.read_bytes()[:20])
    with pytest.raises(EngineError, match="tar"):
        _verify_archive(truncated)

    not_a_zip = tmp_path / "bogus.zip"
    not_a_zip.write_bytes(b"nope")
    with pytest.raises(EngineError, match="zip"):
        _verify_archive(not_a_zip)


@needs_engine
@pytest.mark.asyncio
async def test_ensure_engine_reuses_installed_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    assert REAL_DATA_DIR is not None
    config = Config(data_dir=REAL_DATA_DIR)
    mgr = EngineManager(config, probe=StubProbe(MIXED_GPUS, (13, 3)))

    async def boom(tag: str) -> list[EngineAsset]:  # pragma: no cover - must not run
        raise AssertionError("ensure_engine hit the network for an installed engine")

    monkeypatch.setattr(mgr, "list_assets", boom)
    info = await mgr.ensure_engine()
    assert info.tag == TAG
    assert info.active is True
    assert mgr.active() is not None


# ---------------------------------------------------------------------------
# ensure_engine at boot: an installed engine is reused, never reinstalled (D27)
#
# The old policy ran the full smoke test (a real GPU micro-load) at every boot
# and treated a failed micro-load as a broken install: it went to GitHub,
# called install() on the same tag, ran the same micro-load again, and
# re-downloaded ~600 MB over a working engine -- with the API port unbound the
# whole time. Every reason that micro-load fails at boot (every GPU full,
# a corrupt tiny model, a driver change) is one a reinstall cannot change.
# ---------------------------------------------------------------------------


def _installed_engine(
    mgr: EngineManager, *, smoke_tested: bool, variant: str = "cuda-13.3"
) -> Path:
    directory = _fake_engine(mgr.engines_dir, TAG, 1_000_000)
    (directory / "engine.json").write_text(
        json.dumps(
            {
                "tag": TAG,
                "variant": variant,
                "smoke_tested": smoke_tested,
                "smoke_tested_at": 1_700_000_000.0 if smoke_tested else None,
                "installed_at": 1_000_000,
            }
        ),
        encoding="utf-8",
    )
    return directory


def _no_network(mgr: EngineManager, monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(tag: str) -> list[EngineAsset]:  # pragma: no cover - must not run
        raise AssertionError("ensure_engine hit the network for an installed engine")

    monkeypatch.setattr(mgr, "list_assets", boom)


@pytest.mark.asyncio
async def test_boot_trusts_a_previously_verified_engine_without_a_micro_load(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    _installed_engine(manager, smoke_tested=True)
    _no_network(manager, monkeypatch)
    micro_loads: list[str] = []

    async def fake_capture(binary: Path, args: Any) -> tuple[int, str]:
        assert list(args) == ["--version"]
        return 0, "version: 1 (build 10425)\n"

    async def fake_smoke(tag: str, tiny_model: Any) -> _SmokeResult:  # pragma: no cover
        micro_loads.append(tag)
        return _SmokeResult(True, "should not run")

    monkeypatch.setattr(manager, "_capture", fake_capture)
    monkeypatch.setattr(manager, "_smoke", fake_smoke)

    info = await manager.ensure_engine()
    assert info.tag == TAG and info.active is True
    assert micro_loads == [], "a build that already passed a micro-load must not re-run it at boot"
    assert manager.active() is not None and manager.active().tag == TAG


@pytest.mark.asyncio
async def test_boot_micro_loads_an_engine_that_was_never_verified(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    _installed_engine(manager, smoke_tested=False)
    _no_network(manager, monkeypatch)
    micro_loads: list[str] = []

    async def fake_capture(binary: Path, args: Any) -> tuple[int, str]:
        return 0, "version: 1 (build 10425)\n"

    async def fake_smoke(tag: str, tiny_model: Any) -> _SmokeResult:
        micro_loads.append(tag)
        return _SmokeResult(True, "ok", version_ok=True, version_string="build 10425")

    monkeypatch.setattr(manager, "_capture", fake_capture)
    monkeypatch.setattr(manager, "_smoke", fake_smoke)

    info = await manager.ensure_engine()
    assert micro_loads == [TAG]
    assert info.smoke_tested is True
    # ...and the verdict is persisted, so the NEXT boot skips it.
    assert manager.get(TAG) is not None and manager.get(TAG).smoke_tested is True


@pytest.mark.asyncio
async def test_a_failed_boot_micro_load_keeps_the_engine_and_never_reinstalls(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full GPU at boot must not turn into a GitHub call and a re-download."""
    _installed_engine(manager, smoke_tested=False)
    _no_network(manager, monkeypatch)
    installs: list[str] = []

    async def fake_capture(binary: Path, args: Any) -> tuple[int, str]:
        return 0, "version: 1 (build 10425)\n"

    async def fake_smoke(tag: str, tiny_model: Any) -> _SmokeResult:
        return _SmokeResult(
            False,
            "micro-load failed: CUDA error: out of memory",
            version_ok=True,
            version_string="build 10425",
        )

    async def fake_install(tag: str, **kwargs: Any) -> Any:  # pragma: no cover
        installs.append(tag)
        raise AssertionError("boot must not reinstall an engine that runs")

    monkeypatch.setattr(manager, "_capture", fake_capture)
    monkeypatch.setattr(manager, "_smoke", fake_smoke)
    monkeypatch.setattr(manager, "install", fake_install)

    info = await manager.ensure_engine()
    assert info.tag == TAG and info.active is True
    assert info.smoke_tested is False, "a failed micro-load is not recorded as a pass"
    assert installs == []


@pytest.mark.asyncio
async def test_a_binary_that_cannot_run_version_is_a_broken_install_and_is_replaced(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    _installed_engine(manager, smoke_tested=True)
    asked: list[str] = []

    async def fake_capture(binary: Path, args: Any) -> tuple[int, str]:
        return 3221225781, "The code execution cannot proceed because ggml-cuda.dll was not found"

    async def fake_assets(tag: str) -> list[EngineAsset]:
        asked.append(tag)
        return []

    monkeypatch.setattr(manager, "_capture", fake_capture)
    monkeypatch.setattr(manager, "list_assets", fake_assets)
    manager.config.engine.allow_source_build = False

    with pytest.raises(EngineError):
        await manager.ensure_engine()
    assert asked == [TAG], "a binary that does not run --version must be reinstalled"


def test_driver_too_old_for_the_installed_build_is_one_warning(tmp_config: Config) -> None:
    mgr = EngineManager(tmp_config, probe=StubProbe(MIXED_GPUS, (12, 4)))
    _installed_engine(mgr, smoke_tested=True, variant="cuda-13.3")
    info = mgr.get(TAG)
    assert info is not None
    text = mgr._warn_driver_too_old(info)
    assert text is not None
    assert "driver only advertises CUDA 12.4" in text
    assert "engine.cuda_variant" in text  # names the knob, not just the symptom

    # A driver that CAN run the build says nothing; so does a non-CUDA variant.
    ok = EngineManager(tmp_config, probe=StubProbe(MIXED_GPUS, (13, 3)))
    assert ok._warn_driver_too_old(info) is None
    info.variant = "source-local"
    assert mgr._warn_driver_too_old(info) is None


@pytest.mark.asyncio
async def test_two_installs_of_one_tag_share_one_download(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Setup tab's Install button clicked during the boot install must wait,
    not stream into the same ``.part`` file."""
    downloads: list[str] = []
    entered = asyncio.Event()
    release = asyncio.Event()

    async def fake_locked(tag: str, *, progress: Any, force: bool) -> Any:
        downloads.append(tag)
        entered.set()
        await release.wait()
        _fake_engine(manager.engines_dir, tag, 1_000_000)
        return manager.get(tag)

    monkeypatch.setattr(manager, "_install_locked", fake_locked)

    first = asyncio.create_task(manager.install(TAG))
    await entered.wait()
    second = asyncio.create_task(manager.install(TAG))
    await asyncio.sleep(0.05)
    assert downloads == [TAG], "the second install must wait for the first"
    release.set()
    await asyncio.gather(first, second)
    assert downloads == [TAG, TAG]  # ran in turn, never together


# ---------------------------------------------------------------------------
# Release discovery + update check
#
# Regression cover for 2026-08-18: llama.cpp published a PRERELEASE tagged
# ``v0.1.2`` carrying no assets at all, above the ordinary ``b10488`` /
# ``b10486`` build releases. ``list_releases`` returned every tag unfiltered and
# three call sites did ``latest = releases[0]; update_available = latest !=
# current``, so the Server tab advertised "Engine b10425 -- v0.1.2 is available"
# behind an Install button that could only fail with "available variants:
# <none>".
# ---------------------------------------------------------------------------


def _win_cuda_assets(tag: str) -> list[str]:
    """The Windows CUDA subset of a real build release's 26 assets."""
    return [
        f"llama-{tag}-bin-win-cuda-12.4-x64.zip",
        f"llama-{tag}-bin-win-cuda-13.3-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-13.3-x64.zip",
    ]


def _release(
    tag: str,
    *,
    prerelease: bool = False,
    draft: bool = False,
    assets: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "tag_name": tag,
        "prerelease": prerelease,
        "draft": draft,
        "assets": [_asset(name) for name in (assets or [])],
    }


#: The shape ``GET /repos/ggml-org/llama.cpp/releases`` really returned on
#: 2026-08-18, verified against the live API.
RELEASES_2026_08_18: list[dict[str, Any]] = [
    _release("v0.1.2", prerelease=True),
    _release("b10488", assets=_win_cuda_assets("b10488")),
    _release("b10486", assets=_win_cuda_assets("b10486")),
    _release("b10485", assets=_win_cuda_assets("b10485")),
]


def _github(releases: list[dict[str, Any]], seen: list[Any] | None = None) -> httpx.AsyncClient:
    """A client answering the two GitHub endpoints the manager calls."""
    by_tag = {str(entry["tag_name"]): entry for entry in releases}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/releases"):
            if seen is not None:
                seen.append(request.url.params.get("per_page"))
            return httpx.Response(200, json=releases)
        entry = by_tag.get(path.rsplit("/", 1)[-1])
        if entry is None:
            return httpx.Response(404, json={"message": "Not Found"})
        return httpx.Response(200, json=entry)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _github_manager(
    config: Config,
    releases: list[dict[str, Any]],
    *,
    cuda: tuple[int, int] | None = (13, 3),
    seen: list[Any] | None = None,
) -> EngineManager:
    mgr = EngineManager(config, probe=StubProbe(MIXED_GPUS, cuda), client=_github(releases, seen))
    mgr.os_token, mgr.arch_token = "win", "x64"
    return mgr


def test_engine_tag_regex_accepts_only_build_tags() -> None:
    assert ENGINE_TAG_RE.match("b10488")
    assert not ENGINE_TAG_RE.match("v0.1.2")
    assert not ENGINE_TAG_RE.match("b10425-local")  # a local build is not upstream


def test_build_number_parses_tags_and_local_builds() -> None:
    assert build_number("b10488") == 10488
    assert build_number("b10425-local") == 10425
    assert build_number("v0.1.2") is None
    assert build_number(None) is None
    # The bug this exists to prevent: string ordering flips at a digit boundary.
    assert build_number("b10000") > build_number("b9999")  # type: ignore[operator]
    assert "b10000" < "b9999"


@pytest.mark.asyncio
async def test_list_releases_drops_the_v0_1_2_prerelease(tmp_config: Config) -> None:
    seen: list[Any] = []
    mgr = _github_manager(tmp_config, RELEASES_2026_08_18, seen=seen)
    assert await mgr.list_releases(limit=5) == ["b10488", "b10486", "b10485"]
    # Over-fetched, so filtering cannot starve the caller of its `limit` tags.
    assert int(seen[0]) >= 20


@pytest.mark.asyncio
async def test_list_releases_include_prerelease_never_widens_the_tag_scheme(
    tmp_config: Config,
) -> None:
    """``include_prerelease`` widens the release *kind*, never the tag scheme.

    A ``vX.Y.Z`` tag is uninstallable under any flag: the asset parser, the
    engine directory layout and the source-build clone all assume ``bNNNN``.
    """
    releases = [
        _release("v0.1.2", prerelease=True),
        _release("b10490", prerelease=True, assets=_win_cuda_assets("b10490")),
        _release("b10488", assets=_win_cuda_assets("b10488")),
    ]
    mgr = _github_manager(tmp_config, releases)
    assert await mgr.list_releases(limit=5) == ["b10488"]
    assert await mgr.list_releases(limit=5, include_prerelease=True) == ["b10490", "b10488"]


@pytest.mark.asyncio
async def test_list_releases_skips_drafts_and_sorts_numerically(tmp_config: Config) -> None:
    releases = [
        _release("b10485", assets=_win_cuda_assets("b10485")),
        _release("b10999", draft=True, assets=_win_cuda_assets("b10999")),
        _release("b9999", assets=_win_cuda_assets("b9999")),
        _release("b10488", assets=_win_cuda_assets("b10488")),
    ]
    mgr = _github_manager(tmp_config, releases)
    # API order is not trusted, and b9999 sorts BELOW b10485 numerically.
    assert await mgr.list_releases(limit=5) == ["b10488", "b10485", "b9999"]


@pytest.mark.asyncio
async def test_list_releases_network_failure_still_raises(tmp_config: Config) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    mgr = EngineManager(
        tmp_config,
        probe=StubProbe(MIXED_GPUS, (13, 3)),
        client=httpx.AsyncClient(transport=httpx.MockTransport(boom)),
    )
    with pytest.raises(EngineError, match="could not list llama.cpp releases"):
        await mgr.list_releases()


@pytest.mark.asyncio
async def test_check_update_picks_the_newest_installable_build(tmp_config: Config) -> None:
    tmp_config.engine.pinned_tag = TAG
    mgr = _github_manager(tmp_config, RELEASES_2026_08_18)
    mgr.set_active(TAG)

    status = await mgr.check_update()
    assert status["checked"] is True
    assert status["current"] == TAG
    assert status["latest"] == "b10488"
    assert status["latest_variant"] == "cuda-13.3"  # the 13.3 driver's best asset
    assert status["update_available"] is True
    assert status["recent"][:2] == ["b10488", "b10486"]
    assert status["skipped"] == []
    assert "v0.1.2" not in status["recent"]


@pytest.mark.asyncio
async def test_check_update_steps_past_a_tag_whose_zips_are_not_uploaded_yet(
    tmp_config: Config,
) -> None:
    """A build is tagged minutes before its Windows zips finish uploading."""
    releases = [
        _release("v0.1.2", prerelease=True),
        _release("b10488"),  # tagged, no assets yet
        _release("b10486", assets=_win_cuda_assets("b10486")),
        _release("b10485", assets=_win_cuda_assets("b10485")),
    ]
    mgr = _github_manager(tmp_config, releases)
    mgr.set_active(TAG)

    status = await mgr.check_update()
    assert status["latest"] == "b10486"
    assert status["latest_variant"] == "cuda-13.3"
    assert status["update_available"] is True
    assert [entry["tag"] for entry in status["skipped"]] == ["b10488"]
    assert "<none>" in status["skipped"][0]["reason"]


@pytest.mark.asyncio
async def test_check_update_reports_nothing_new_when_active_is_the_newest(
    tmp_config: Config,
) -> None:
    tmp_config.engine.pinned_tag = TAG  # a stale pin must not manufacture an update
    mgr = _github_manager(tmp_config, RELEASES_2026_08_18)
    mgr.set_active("b10488")

    status = await mgr.check_update()
    assert status["current"] == "b10488"  # active.json wins over the pin
    assert status["latest"] == "b10488"
    assert status["update_available"] is False


@pytest.mark.asyncio
async def test_check_update_compares_build_numbers_not_strings(tmp_config: Config) -> None:
    """``b10441`` is installed on this box but not active; newer != different.

    The old ``latest != current`` check called a *downgrade* an update, and
    string ordering additionally breaks across a digit boundary.
    """
    _fake_engine(tmp_config.engines_dir, TAG, 1_000)
    _fake_engine(tmp_config.engines_dir, "b10441", 2_000)  # installed, inactive
    tmp_config.engine.pinned_tag = TAG

    releases = [_release("b10441", assets=_win_cuda_assets("b10441"))]
    mgr = _github_manager(tmp_config, releases)

    mgr.set_active(TAG)
    ahead = await mgr.check_update()
    assert ahead["current"] == TAG
    assert (ahead["latest"], ahead["update_available"]) == ("b10441", True)

    # Now run the installed-but-newer build: upstream's b10441 is not an update.
    mgr.set_active("b10441")
    level = await mgr.check_update()
    assert (level["current"], level["update_available"]) == ("b10441", False)

    # And a real downgrade is never offered, however the strings sort.
    downgrade = _github_manager(tmp_config, [_release("b9999", assets=_win_cuda_assets("b9999"))])
    downgrade.set_active("b10441")
    assert (await downgrade.check_update())["update_available"] is False


@pytest.mark.asyncio
async def test_check_update_offers_a_source_build_when_no_asset_fits(tmp_config: Config) -> None:
    tmp_config.engine.allow_source_build = True
    mgr = _github_manager(tmp_config, RELEASES_2026_08_18, cuda=(12, 0))  # too old for 13.3
    mgr.set_active(TAG)

    status = await mgr.check_update()
    assert status["latest"] == "b10488"
    assert status["latest_variant"] == "source"
    assert status["update_available"] is True
    # Every probed tag is accounted for, with the driver named.
    assert [entry["tag"] for entry in status["skipped"]] == ["b10488", "b10486", "b10485"]
    assert "CUDA 12.0" in status["skipped"][0]["reason"]


@pytest.mark.asyncio
async def test_check_update_offers_nothing_when_source_build_is_disabled(
    tmp_config: Config,
) -> None:
    tmp_config.engine.allow_source_build = False
    mgr = _github_manager(tmp_config, RELEASES_2026_08_18, cuda=(12, 0))
    mgr.set_active(TAG)

    status = await mgr.check_update()
    assert status["latest"] is None
    assert status["latest_variant"] is None
    assert status["update_available"] is False
    assert len(status["skipped"]) == 3


@pytest.mark.asyncio
async def test_check_update_probes_at_most_probe_assets_tags(tmp_config: Config) -> None:
    """Each probe is a GitHub call; the default is three, not the whole list."""
    fetched: list[str] = []
    mgr = _github_manager(tmp_config, RELEASES_2026_08_18, cuda=(12, 0))
    original = mgr.list_assets

    async def counting(tag: str) -> list[EngineAsset]:
        fetched.append(tag)
        return await original(tag)

    mgr.list_assets = counting  # type: ignore[method-assign]
    await mgr.check_update(limit=5, probe_assets=2)
    assert fetched == ["b10488", "b10486"]


@pytest.mark.asyncio
async def test_check_update_falls_back_to_the_pin_without_active_json(tmp_config: Config) -> None:
    tmp_config.engine.pinned_tag = TAG
    mgr = _github_manager(tmp_config, RELEASES_2026_08_18)
    assert (await mgr.check_update())["current"] == TAG


# ---------------------------------------------------------------------------
# The three former copies of the update check now share one implementation
# ---------------------------------------------------------------------------


class _StubRegistry:
    def all(self) -> list[Any]:
        return []


class _StubReport:
    def to_dict(self) -> dict[str, Any]:
        return {"engine": {}}


class _FakeRequest:
    def __init__(self, state: Any) -> None:
        self.app = type("App", (), {"state": state})()


@pytest.mark.asyncio
async def test_capabilities_route_uses_check_update(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/api/capabilities?check_update=true`` must not rebuild the comparison."""
    from studioforge.api import admin_routes
    from studioforge.core import capabilities as capabilities_module

    monkeypatch.setattr(capabilities_module, "build_report", lambda *a, **k: _StubReport())

    tmp_config.engine.pinned_tag = TAG
    mgr = _github_manager(tmp_config, RELEASES_2026_08_18)
    mgr.set_active(TAG)
    state = type(
        "State",
        (),
        {
            "config": tmp_config,
            "probe": StubProbe(MIXED_GPUS, (13, 3)),
            "registry": _StubRegistry(),
            "engine_manager": mgr,
        },
    )()

    payload = await admin_routes.capabilities(_FakeRequest(state), check_update=True)
    update = payload["update"]
    assert update["latest"] == "b10488"
    assert update["latest_variant"] == "cuda-13.3"
    assert update["update_available"] is True
    assert "v0.1.2" not in update["recent"]


class _StubCliManager:
    """Records what the CLI asked for; never touches GitHub or the filesystem."""

    calls: list[dict[str, Any]] = []

    def __init__(self, config: Any, **_kwargs: Any) -> None:
        self.config = config

    async def check_update(self, *, limit: int = 5, probe_assets: int = 3) -> dict[str, Any]:
        type(self).calls.append({"limit": limit, "probe_assets": probe_assets})
        return {
            "checked": True,
            "current": TAG,
            "latest": "b10488",
            "update_available": True,
            "recent": ["b10488", "b10486"],
            "latest_variant": "cuda-13.3",
            "skipped": [{"tag": "b10490", "reason": "available variants: <none>"}],
        }

    async def list_releases(self, limit: int = 30, **_kwargs: Any) -> list[str]:
        return ["b10488", "b10486"]


def _cli(monkeypatch: pytest.MonkeyPatch, tmp_config: Config, *args: str) -> Any:
    """Invoke the real CLI with the config load and the manager stubbed out.

    ``_load`` is replaced rather than pointed at a temp file because it also
    reconfigures global logging onto a directory that outlives the test.
    """
    from typer.testing import CliRunner

    from studioforge import __main__ as main_cli
    from studioforge.core import engine as engine_module

    _StubCliManager.calls = []
    monkeypatch.setattr(main_cli, "_load", lambda _path: tmp_config)
    monkeypatch.setattr(engine_module, "EngineManager", _StubCliManager)
    return CliRunner().invoke(main_cli.app, list(args), catch_exceptions=False)


def test_cli_engine_check_reports_the_verified_tag(
    monkeypatch: pytest.MonkeyPatch, tmp_config: Config
) -> None:
    result = _cli(monkeypatch, tmp_config, "engine", "--check")
    assert result.exit_code == 0
    assert _StubCliManager.calls, "the CLI must go through EngineManager.check_update"
    assert "active: b10425" in result.output  # the ACTIVE tag, not the pin
    assert "b10488" in result.output
    assert "cuda-13.3" in result.output
    assert "skipped b10490" in result.output


def test_cli_engine_list_says_prereleases_are_hidden(
    monkeypatch: pytest.MonkeyPatch, tmp_config: Config
) -> None:
    result = _cli(monkeypatch, tmp_config, "engine", "--list")
    assert result.exit_code == 0
    assert "b10488 b10486" in result.output
    # A list that silently differs from GitHub's front page needs to say so.
    assert "prereleases" in result.output


# ---------------------------------------------------------------------------
# Live verification against the real b10425 binary
# ---------------------------------------------------------------------------


@needs_engine
@windows_only
@pytest.mark.asyncio
async def test_version_string_real(real_manager: EngineManager) -> None:
    version = await real_manager.version_string(TAG)
    assert version is not None
    assert "10425" in version


@needs_engine
@windows_only
@pytest.mark.asyncio
async def test_list_devices_real(real_manager: EngineManager) -> None:
    devices = await real_manager.list_devices(TAG)
    cuda = [d for d in devices if d.startswith("CUDA")]
    if not cuda:
        pytest.skip("no CUDA devices visible to the engine on this box")
    assert len(cuda) == 4
    assert any("5090" in d for d in cuda)
    assert any("3090" in d for d in cuda)


@needs_engine
@windows_only
@pytest.mark.asyncio
async def test_supported_flags_real_and_cached(real_manager: EngineManager) -> None:
    flags = await real_manager.supported_flags(TAG)
    assert {"--ctx-size", "--flash-attn", "--spec-draft-n-max", "--mmproj"} <= flags
    assert "--totally-fake" not in flags
    # Removed spellings are excluded: llama-server parses and ignores them.
    assert "--draft-max" not in flags

    cache = real_manager.engine_dir(TAG) / "flags.txt"
    assert cache.is_file()
    cached = {ln.strip() for ln in cache.read_text(encoding="utf-8").splitlines() if ln.strip()}
    assert "--ctx-size" in cached

    # A second manager must hit the on-disk cache, not the binary.
    fresh = EngineManager(real_manager.config, probe=StubProbe(MIXED_GPUS, (13, 3)))
    assert await fresh.supported_flags(TAG) == flags


@needs_engine
@windows_only
@pytest.mark.asyncio
async def test_removed_flags_real(real_manager: EngineManager) -> None:
    removed = await real_manager.removed_flags(TAG)
    assert removed["--draft-max"] == "--spec-draft-n-max"
    assert removed["--draft"] == "--spec-draft-n-max"
    assert removed["--draft-min"] == "--spec-draft-n-min"
    # b10425 keeps these as live aliases, so they must NOT be flagged.
    assert "--cache-type-k-draft" not in removed
    assert "--n-gpu-layers-draft" not in removed


@needs_engine
@windows_only
@pytest.mark.asyncio
async def test_validate_extra_flags_real(real_manager: EngineManager) -> None:
    assert await real_manager.validate_extra_flags(TAG, "") == []
    assert await real_manager.validate_extra_flags(TAG, "   ") == []
    assert await real_manager.validate_extra_flags(TAG, "--top-k 20") == []
    assert await real_manager.validate_extra_flags(TAG, "--top-k 20 --min-p 0.05") == []
    assert await real_manager.validate_extra_flags(TAG, "--spec-draft-n-max 4") == []
    # Negative numeric values must not be mistaken for flags.
    assert await real_manager.validate_extra_flags(TAG, "--top-k -1") == []

    removed = await real_manager.validate_extra_flags(TAG, "--draft-max 4")
    assert len(removed) == 1
    assert "--draft-max" in removed[0]
    assert "removed in this release" in removed[0]
    assert "--spec-draft-n-max" in removed[0]

    unknown = await real_manager.validate_extra_flags(TAG, "--not-a-flag")
    assert len(unknown) == 1
    assert "unknown flag '--not-a-flag'" in unknown[0]
    assert TAG in unknown[0]

    for owned, phrase in (
        ("--port 9999", "--port"),
        ("--model foo.gguf", "--model"),
        ("--host 0.0.0.0", "--host"),
        ("--n-gpu-layers 10", "--n-gpu-layers"),
        ("-ngl 10", "--n-gpu-layers"),
        ("--alias bob", "--alias"),
    ):
        errors = await real_manager.validate_extra_flags(TAG, owned)
        assert errors, f"{owned} should be rejected"
        assert "managed by StudioForge" in errors[0]
        assert phrase in errors[0]

    meta = await real_manager.validate_extra_flags(TAG, '--chat-template-file "a && rm -rf b"')
    assert meta
    assert "illegal shell character" in meta[0]

    injected = await real_manager.validate_extra_flags(TAG, "--top-k $(whoami)")
    assert injected
    assert "illegal shell character" in injected[0]

    # Sampler chains legitimately use ';' and must still validate.
    assert await real_manager.validate_extra_flags(TAG, "--samplers top_k;top_p") == []


@needs_engine
@windows_only
@pytest.mark.asyncio
async def test_smoke_test_real_engine(real_manager: EngineManager) -> None:
    """The end-to-end proof: launch the real engine and reach /health ok."""
    model = real_manager.find_tiny_model()
    if model is None:
        pytest.skip("no tiny GGUF available for the micro-load")
    ok, detail = await real_manager.smoke_test(TAG, tiny_model=model)
    assert ok, detail
    assert "10425" in detail
    assert model.name in detail


@needs_engine
@windows_only
@pytest.mark.asyncio
async def test_smoke_test_failure_carries_stderr_tail(
    real_manager: EngineManager, tmp_path: Path
) -> None:
    """A failing micro-load must report the child's stderr or it is undiagnosable."""
    bogus = tmp_path / "not-a-model.gguf"
    bogus.write_bytes(b"NOTGGUF" * 64)
    ok, detail = await real_manager.smoke_test(TAG, tiny_model=bogus)
    assert not ok
    assert "Last stderr lines" in detail
    assert len(detail.splitlines()) > 1


# ---------------------------------------------------------------------------
# Cancellation must never orphan the smoke-test child
# ---------------------------------------------------------------------------

_HEALTHY_CHILD = """
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def opt(name, default=None):
    args = sys.argv[1:]
    if name in args:
        i = args.index(name)
        if i + 1 < len(args):
            return args[i + 1]
    return default

PORT = int(opt("--port", "0"))

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def do_GET(self):
        body = json.dumps({"status": "ok"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, fmt, *args):
        pass

ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
"""


@pytest.mark.asyncio
async def test_micro_load_kills_the_child_even_when_the_kill_await_is_cancelled(
    tmp_config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cleanup's own first await can receive the CancelledError.

    A client aborting the smoke-test request (or Ctrl-C during startup) can
    deliver cancellation exactly at the ``asyncio.to_thread(kill_process_tree)``
    await in ``_micro_load``'s finally. The child was spawned in its own
    process group, so skipping that kill leaves a llama-server resident with a
    full CUDA context -- permanently leaked VRAM. The kill must complete
    anyway.
    """
    import psutil

    script = tmp_path / "healthy_child.py"
    script.write_text(_HEALTHY_CHILD, encoding="utf-8")
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"GGUF")

    # Route the spawn through the interpreter (the fake "binary" is a .py).
    real_exec = asyncio.create_subprocess_exec

    async def prefixed_exec(_binary: str, *argv: str, **kwargs: Any) -> Any:
        return await real_exec(sys.executable, str(script), *argv, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", prefixed_exec)

    # Simulate the cancellation landing on the cleanup's first await.
    real_to_thread = asyncio.to_thread

    async def cancelled_to_thread(fn: Any, *args: Any, **kwargs: Any) -> Any:
        from studioforge.core.engine import kill_process_tree as target

        if fn is target:
            raise asyncio.CancelledError
        return await real_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", cancelled_to_thread)

    mgr = EngineManager(tmp_config, probe=StubProbe(MIXED_GPUS, (13, 3)))
    pids_before = {p.pid for p in psutil.process_iter()}

    with pytest.raises(asyncio.CancelledError):
        await mgr._micro_load(Path("fake-llama-server.exe"), model)

    # Find the spawned child (a python running our script) and prove it died.
    deadline = asyncio.get_running_loop().time() + 10.0
    survivors: list[int] = []
    while asyncio.get_running_loop().time() < deadline:
        survivors = []
        for proc in psutil.process_iter():
            try:
                if proc.pid in pids_before:
                    continue
                if any("healthy_child.py" in part for part in proc.cmdline()):
                    survivors.append(proc.pid)
            except psutil.Error:
                continue
        if not survivors:
            break
        await asyncio.sleep(0.2)
    assert survivors == [], (
        f"smoke-test child pid(s) {survivors} survived a cancelled cleanup: "
        "an orphaned llama-server would hold its VRAM forever"
    )
