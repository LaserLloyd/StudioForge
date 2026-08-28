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
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from studioforge.config import Config
from studioforge.core.engine import (
    BIN_NAME,
    ENGINE_TAG_RE,
    RELEASE_MAX_PAGES,
    RELEASE_PAGE_SIZE,
    SKIP_DRAFT,
    SKIP_PRERELEASE,
    SKIP_TAG_SCHEME,
    EngineAsset,
    EngineError,
    EngineManager,
    _raise_for_rate_limit,  # internal: the rate-limit diagnosis
    _SmokeResult,  # internal: build tests stub the smoke step
    _swap_engine_dir,  # internal: the extract-to-sibling-then-swap step
    _verify_archive,  # internal: the pre-extract integrity check
    build_assets,
    build_number,
    describe_release_filter,
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

    async def fake_build(tag: str, *, arches: Any, progress: Any = None, activate: bool = False):
        built["tag"] = tag
        built["arches"] = list(arches)
        built["activate"] = activate
        directory = _fake_engine(tmp_config.engines_dir, f"{tag}-local", 1_000)
        return mgr._finalize(  # exercises the real finalizer + active.json write
            directory,
            directory / BIN_NAME,
            "source-local",
            _SmokeResult(True, "ok"),
            activate=activate,
        )

    monkeypatch.setattr(mgr, "list_assets", fake_assets)
    monkeypatch.setattr(mgr, "build_from_source", fake_build)

    info = await mgr.ensure_engine()
    assert built["tag"] == TAG
    assert built["arches"] == ["86", "120"]
    assert info.variant == "source-local"
    # Bootstrap is the one caller that means "and use it": the box has no other
    # engine, so a build that is not activated leaves nothing to load (D49-4).
    assert built["activate"] is True
    assert info.active is True


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

    async def fake_build(tag: str, *, arches: Any, progress: Any = None, activate: bool = False):
        built["tag"] = tag
        built["arches"] = list(arches)
        built["activate"] = activate
        directory = _fake_engine(tmp_config.engines_dir, f"{tag}-local", 1_000)
        return mgr._finalize(
            directory,
            directory / BIN_NAME,
            "source-local",
            _SmokeResult(True, "ok"),
            activate=activate,
        )

    monkeypatch.setattr(mgr, "list_assets", fake_assets)
    monkeypatch.setattr(mgr, "build_from_source", fake_build)

    info = await mgr.install(TAG)

    # The caller's activate flag rides all the way through the source-build
    # fallback: a plain install(TAG) builds without switching to it (D49-4).
    assert built == {"tag": TAG, "arches": ["86", "120"], "activate": False}
    assert info.tag == f"{TAG}-local" and info.variant == "source-local"
    assert info.active is False
    assert not (tmp_config.engines_dir / "active.json").exists()


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
    # Reuse is not activation (D49-4). This assertion used to read ``is True``
    # -- the reuse branch flipped active.json for anyone who merely re-ran an
    # install, which is the drift the split exists to stop.
    assert info.active is False
    assert not (tmp_config.engines_dir / "active.json").exists()

    activated = await mgr.install(TAG, activate=True)
    assert activated.active is True
    assert (
        json.loads((tmp_config.engines_dir / "active.json").read_text(encoding="utf-8"))["tag"]
        == f"{TAG}-local"
    )


# ---------------------------------------------------------------------------
# D49-4/7/12/13: install and activate are separate, and neither may overwrite
# a live engine directory
# ---------------------------------------------------------------------------


def _stub_prebuilt(
    mgr: EngineManager,
    monkeypatch: pytest.MonkeyPatch,
    *,
    assets: list[EngineAsset] | None = None,
    help_text: str = SAMPLE_HELP,
) -> dict[str, list[Any]]:
    """Everything the download branch of ``install`` would otherwise reach for.

    No GitHub, no ~600 MB stream, no GPU micro-load and -- importantly -- no
    exec of the fake binary the extraction leaves behind: ``_capture`` is
    stubbed, so the flag-cache warm (D49-12) is observable without running
    anything.
    """
    seen: dict[str, list[Any]] = {"downloads": [], "captures": []}

    async def fake_assets(tag: str) -> list[EngineAsset]:
        return list(ASSETS if assets is None else assets)

    async def fake_download(
        url: str, target: Path, expected_bytes: int, phase: str, progress: Any
    ) -> Path:
        seen["downloads"].append(phase)
        _emit_progress(progress, phase)
        target.parent.mkdir(parents=True, exist_ok=True)
        _make_zip(target, {BIN_NAME: b"BINARY", "ggml-cuda.dll": b"LIB"})
        return target

    async def fake_smoke(tag: str, tiny_model: Any) -> _SmokeResult:
        return _SmokeResult(True, "stubbed", version_ok=True, version_string="build 10425")

    async def fake_capture(binary: Path, args: Any) -> tuple[int, str]:
        seen["captures"].append(list(args))
        return 0, help_text

    monkeypatch.setattr(mgr, "list_assets", fake_assets)
    monkeypatch.setattr(mgr, "_download", fake_download)
    monkeypatch.setattr(mgr, "_smoke", fake_smoke)
    monkeypatch.setattr(mgr, "_capture", fake_capture)
    return seen


def _emit_progress(progress: Any, phase: str) -> None:
    if progress is not None:
        progress(phase, 1.0)


@pytest.mark.asyncio
async def test_install_unpacks_a_build_without_making_it_the_active_engine(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The D49-4 behaviour break, stated as a guarantee.

    Installing used to call ``set_active`` as its last step unconditionally, so
    ``engine --update`` could install, smoke-test, fail, print "keeping b10425"
    and exit with ``active.json`` already naming the build that had just failed.
    """
    _stub_prebuilt(manager, monkeypatch)

    info = await manager.install(TAG)

    assert info.tag == TAG and info.variant == "cuda-13.3"
    assert info.active is False
    assert not (manager.engines_dir / "active.json").exists()
    assert manager.active() is not None  # the fallback still resolves a build
    assert manager._read_active() is None


@pytest.mark.asyncio
async def test_install_activates_only_when_the_caller_asks(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``activate=True`` is what bootstrap and the first-run checklist pass."""
    _stub_prebuilt(manager, monkeypatch)

    info = await manager.install(TAG, activate=True)

    assert info.active is True
    payload = json.loads((manager.engines_dir / "active.json").read_text(encoding="utf-8"))
    assert payload["tag"] == TAG


@pytest.mark.asyncio
async def test_a_fresh_install_warms_the_flags_cache(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``EngineInfo.flags`` reads ``flags.txt``; only validation used to write it.

    A build reported "no flags" until somebody happened to validate an
    expert-tier string against it, which reads as an engine that supports
    nothing at all (D49-12).
    """
    seen = _stub_prebuilt(manager, monkeypatch)

    await manager.install(TAG)

    assert seen["captures"] == [["--help"]], "exactly one --help of the verified binary"
    assert (manager.engine_dir(TAG) / "flags.txt").is_file()
    info = manager.get(TAG)
    assert info is not None
    assert "--ctx-size" in info.flags


@pytest.mark.asyncio
async def test_activate_refuses_a_tag_that_is_not_installed(
    manager: EngineManager,
) -> None:
    """Writing active.json for an absent build is drift wearing a success message."""
    _fake_engine(manager.engines_dir, "b10500", 1_000)

    with pytest.raises(EngineError) as excinfo:
        await manager.activate("b99999")

    message = str(excinfo.value)
    assert "b99999" in message
    assert "b10500" in message  # names what IS installed
    assert not (manager.engines_dir / "active.json").exists()


@pytest.mark.asyncio
async def test_activate_writes_active_json_warms_the_flags_and_leaves_the_pin_alone(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Activation is one write plus a warm; the pin belongs to the caller (D49-4).

    ``engine.pinned_tag`` is deliberately NOT touched here: the route, the GUI
    and the CLI helper set it beside this call so the two halves cannot be
    written from two places with two opinions.
    """
    _fake_engine(manager.engines_dir, "b10500", 2_000)
    manager.config.engine.pinned_tag = TAG
    captures: list[Any] = []

    async def fake_capture(binary: Path, args: Any) -> tuple[int, str]:
        captures.append(list(args))
        return 0, SAMPLE_HELP

    monkeypatch.setattr(manager, "_capture", fake_capture)

    info = await manager.activate("b10500")

    assert info.tag == "b10500" and info.active is True
    payload = json.loads((manager.engines_dir / "active.json").read_text(encoding="utf-8"))
    assert payload["tag"] == "b10500"
    assert captures == [["--help"]]
    assert (manager.engine_dir("b10500") / "flags.txt").is_file()
    assert manager.config.engine.pinned_tag == TAG


@pytest.mark.asyncio
async def test_an_unreadable_help_does_not_fail_the_activation(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cold flag cache is a cosmetic problem; a refused activation is not."""
    _fake_engine(manager.engines_dir, "b10500", 2_000)

    async def boom(binary: Path, args: Any) -> tuple[int, str]:
        raise OSError("not an executable")

    monkeypatch.setattr(manager, "_capture", boom)

    info = await manager.activate("b10500")
    assert info.active is True


@pytest.mark.asyncio
async def test_reusing_an_installed_build_keeps_its_variant_and_installed_at(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The already-present branch stamped "prebuilt"/now over the truth (D49-7).

    Both stamps are load-bearing: ``_warn_driver_too_old`` only fires for a
    ``cuda-*`` variant, so overwriting a source build's variant disarmed it, and
    ``prune`` orders by ``installed_at``, so restamping it made the oldest
    engine look like the newest one.
    """
    _installed_engine(manager, smoke_tested=True, variant="source-local")

    async def fake_smoke(tag: str, tiny_model: Any) -> _SmokeResult:
        return _SmokeResult(True, "stubbed", version_ok=True)

    async def boom(tag: str) -> list[EngineAsset]:  # pragma: no cover - must not run
        raise AssertionError("a present engine must not be re-downloaded")

    monkeypatch.setattr(manager, "_smoke", fake_smoke)
    monkeypatch.setattr(manager, "list_assets", boom)

    info = await manager.install(TAG)

    assert info.variant == "source-local"
    assert info.installed_at == 1_000_000
    assert info.active is False


@pytest.mark.asyncio
async def test_a_reinstall_is_refused_while_loaded_models_hold_the_tag(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failed-smoke fallthrough used to re-download over a live directory.

    On Windows a running llama-server holds its own ``.exe`` and DLLs open, so
    the extraction fails partway and leaves the LIVE engine half-overwritten --
    breaking models that were working a second earlier (D49-7).
    """
    _installed_engine(manager, smoke_tested=False)
    manager.tag_in_use = lambda tag: ["vendor/Model-A-Q4_K_M", "vendor/Model-B-Q8_0"]

    async def failing_smoke(tag: str, tiny_model: Any) -> _SmokeResult:
        return _SmokeResult(False, "micro-load failed: CUDA error: out of memory")

    async def boom(tag: str) -> list[EngineAsset]:  # pragma: no cover - must not run
        raise AssertionError("the refusal must happen before any download")

    monkeypatch.setattr(manager, "_smoke", failing_smoke)
    monkeypatch.setattr(manager, "list_assets", boom)

    with pytest.raises(EngineError) as excinfo:
        await manager.install(TAG)

    message = str(excinfo.value)
    assert "vendor/Model-A-Q4_K_M" in message and "vendor/Model-B-Q8_0" in message
    assert "unload" in message.lower()


@pytest.mark.asyncio
async def test_force_does_not_get_past_the_in_use_refusal(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``force`` re-downloads a healthy build -- it is not an override for this."""
    _installed_engine(manager, smoke_tested=True)
    manager.tag_in_use = lambda tag: ["vendor/Model-A-Q4_K_M"]
    _stub_prebuilt(manager, monkeypatch)

    with pytest.raises(EngineError, match="vendor/Model-A-Q4_K_M"):
        await manager.install(TAG, force=True)


@pytest.mark.asyncio
async def test_a_hook_that_cannot_answer_refuses_rather_than_assuming_idle(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Cannot tell" must not read as "nothing is running"."""
    _installed_engine(manager, smoke_tested=True)

    def unavailable(tag: str) -> list[str]:
        raise RuntimeError("supervisor is not listable")

    manager.tag_in_use = unavailable
    _stub_prebuilt(manager, monkeypatch)

    with pytest.raises(EngineError, match="could not check"):
        await manager.install(TAG, force=True)


@pytest.mark.asyncio
async def test_a_failed_swap_leaves_the_installed_engine_exactly_as_it_was(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Extraction goes to a sibling and swaps; a failure is a no-op (D49-7)."""
    from studioforge.core import engine as engine_module

    dest = _fake_engine(manager.engines_dir, TAG, 1_000_000)
    (dest / "ggml-cuda.dll").write_bytes(b"ORIGINAL")
    _stub_prebuilt(manager, monkeypatch)

    def refuse(staged: Path, target: Path) -> None:
        raise EngineError(f"could not replace the existing engine at {target}: WinError 32")

    monkeypatch.setattr(engine_module, "_swap_engine_dir", refuse)

    with pytest.raises(EngineError, match="could not replace the existing engine"):
        await manager.install(TAG, force=True)

    assert (dest / BIN_NAME).read_bytes() == b"fake"
    assert (dest / "ggml-cuda.dll").read_bytes() == b"ORIGINAL"
    # ...and the staging tree is cleaned up rather than left to be mistaken for
    # an install (``installed()`` skips underscore-prefixed directories anyway).
    assert not (dest.parent / f"_{dest.name}.new").exists()
    assert {info.tag for info in manager.installed()} == {TAG}


def test_swap_engine_dir_replaces_the_old_directory_and_cleans_up(tmp_path: Path) -> None:
    dest = tmp_path / "engines" / TAG
    dest.mkdir(parents=True)
    (dest / BIN_NAME).write_bytes(b"OLD")
    staged = tmp_path / "engines" / f"_{TAG}.new"
    staged.mkdir()
    (staged / BIN_NAME).write_bytes(b"NEW")

    _swap_engine_dir(staged, dest)

    assert (dest / BIN_NAME).read_bytes() == b"NEW"
    assert not staged.exists()
    assert list((tmp_path / "engines").iterdir()) == [dest]


def test_swap_engine_dir_puts_the_old_directory_back_when_the_move_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The half-second between "moved aside" and "moved in" must not lose it."""
    dest = tmp_path / "engines" / TAG
    dest.mkdir(parents=True)
    (dest / BIN_NAME).write_bytes(b"OLD")
    staged = tmp_path / "engines" / f"_{TAG}.new"
    staged.mkdir()

    real_rename = Path.rename

    def flaky(self: Path, target: Any) -> Path:
        if self == staged:
            raise OSError(32, "the process cannot access the file")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky)

    with pytest.raises(EngineError, match="could not move the new engine"):
        _swap_engine_dir(staged, dest)

    assert (dest / BIN_NAME).read_bytes() == b"OLD"


@pytest.mark.asyncio
async def test_the_disk_precheck_names_what_it_needs_and_what_is_free(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ENOSPC used to arrive as a bare OSError from mid-stream (D49-13)."""
    from studioforge.core import diskspace

    big = build_assets(
        TAG,
        [
            _asset(f"llama-{TAG}-bin-win-cuda-13.3-x64.zip", 800 * 1024**2),
            _asset("cudart-llama-bin-win-cuda-13.3-x64.zip", 20 * 1024**2),
        ],
    )
    _stub_prebuilt(manager, monkeypatch, assets=big)
    monkeypatch.setattr(
        diskspace, "disk_report", lambda path, need: {"drive": "E:\\", "free_bytes": 1024**3}
    )

    with pytest.raises(EngineError) as excinfo:
        await manager.install(TAG)

    message = str(excinfo.value)
    assert "E:\\" in message
    assert "2.0 GiB required" in message
    assert "1.0 GiB free" in message


@pytest.mark.asyncio
async def test_unmeasurable_free_space_does_not_block_the_install(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disk we cannot measure is not a full disk; refusing on a guess is worse."""
    from studioforge.core import diskspace

    _stub_prebuilt(manager, monkeypatch)
    monkeypatch.setattr(
        diskspace, "disk_report", lambda path, need: {"error": "WinError 21", "drive": "?"}
    )

    info = await manager.install(TAG)
    assert info.tag == TAG


def test_a_rate_limited_github_reply_names_the_reset_time_and_the_token_vars() -> None:
    """403 + ``x-ratelimit-remaining: 0`` is a diagnosis, not a generic failure.

    Through ``raise_for_status`` it was indistinguishable from a repository
    that had been made private, and the fix -- wait, or set a token
    ``_api_headers`` has honoured all along -- appeared nowhere (D49-13).
    """
    reset = 1_780_000_000
    resp = httpx.Response(
        403,
        headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(reset)},
        request=httpx.Request("GET", "https://api.github.invalid/releases"),
    )

    with pytest.raises(EngineError) as excinfo:
        _raise_for_rate_limit(resp, "list llama.cpp releases")

    message = str(excinfo.value)
    assert "list llama.cpp releases" in message
    assert time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(reset)) in message
    assert "GITHUB_TOKEN" in message and "GH_TOKEN" in message

    # A 403 that is not a rate limit, and a 429 with quota left, both pass through.
    _raise_for_rate_limit(httpx.Response(403), "list llama.cpp releases")
    _raise_for_rate_limit(
        httpx.Response(429, headers={"x-ratelimit-remaining": "12"}), "list llama.cpp releases"
    )
    _raise_for_rate_limit(httpx.Response(200), "list llama.cpp releases")


@pytest.mark.asyncio
async def test_list_releases_surfaces_the_rate_limit_rather_than_a_403(
    tmp_config: Config,
) -> None:
    def limited(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1780000000"},
            json={"message": "API rate limit exceeded"},
        )

    mgr = EngineManager(
        tmp_config,
        probe=StubProbe(MIXED_GPUS, (13, 3)),
        client=httpx.AsyncClient(transport=httpx.MockTransport(limited)),
    )

    with pytest.raises(EngineError, match="GITHUB_TOKEN"):
        await mgr.list_releases(limit=5)


# ---------------------------------------------------------------------------
# D49-6: the extra-flags revalidation sweep
# ---------------------------------------------------------------------------


def _record(model_id: str, extra_flags: Any) -> Any:
    return SimpleNamespace(id=model_id, settings=SimpleNamespace(extra_flags=extra_flags))


@pytest.mark.asyncio
async def test_the_sweep_names_the_model_and_the_flag_the_new_build_rejects(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The D2 failure, caught at the switch instead of never (D49-6).

    llama-server *ignores* flags it does not recognise, so a build that dropped
    ``--draft-max`` loads the model, looks healthy, and silently runs without
    speculative decoding. Nothing re-ran the save-time validation on an engine
    change, so nothing ever noticed.
    """
    _fake_engine(manager.engines_dir, TAG, 1_000)

    async def fake_capture(binary: Path, args: Any) -> tuple[int, str]:
        return 0, SAMPLE_HELP

    monkeypatch.setattr(manager, "_capture", fake_capture)

    offenders = await manager.revalidate_extra_flags(
        TAG,
        [
            _record("vendor/Clean-Q4_K_M", "--ctx-size 8192"),
            _record("vendor/Stale-Q4_K_M", "--draft-max 4"),
            _record("vendor/Unset-Q4_K_M", ""),
            _record("vendor/None-Q4_K_M", None),
        ],
    )

    assert [entry["model_id"] for entry in offenders] == ["vendor/Stale-Q4_K_M"]
    detail = "; ".join(offenders[0]["errors"])
    assert "--draft-max" in detail
    assert "removed in this release" in detail
    assert "--spec-draft-n-max" in detail


@pytest.mark.asyncio
async def test_the_sweep_is_silent_when_every_saved_flag_still_validates(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_engine(manager.engines_dir, TAG, 1_000)

    async def fake_capture(binary: Path, args: Any) -> tuple[int, str]:
        return 0, SAMPLE_HELP

    monkeypatch.setattr(manager, "_capture", fake_capture)

    assert (
        await manager.revalidate_extra_flags(
            TAG, [_record("vendor/A", "--ctx-size 4096"), _record("vendor/B", "--flash-attn on")]
        )
        == []
    )


@pytest.mark.asyncio
async def test_the_sweep_never_raises_however_broken_the_input(
    manager: EngineManager,
) -> None:
    """It runs mid-activation; an exception here would abort a switch that worked.

    The engine is not installed at all, so validation cannot read a ``--help``:
    that becomes an error string against the model, not a traceback.
    """
    offenders = await manager.revalidate_extra_flags(
        "b99999",
        [
            _record("vendor/Broken", "--ctx-size 4096"),
            SimpleNamespace(settings=None),  # no id, no settings object
            SimpleNamespace(),  # not a record at all
        ],
    )

    assert [entry["model_id"] for entry in offenders] == ["vendor/Broken"]
    assert "cannot validate" in "; ".join(offenders[0]["errors"])


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

    async def fake_locked(tag: str, *, progress: Any, force: bool, activate: bool) -> Any:
        # The keyword list is the contract install() calls this with; a double
        # that lags it raises TypeError *inside* the lock, so the waiter never
        # wakes and the test hangs rather than failing (D49-4 added activate).
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
#: 2026-08-18, verified against the live API. Its build entries are
#: ``prerelease: false`` because that is what upstream published *then*; ten
#: days later they were all ``true`` and this fixture stopped describing the
#: world (D49-1). It stays as the 2026-08-18 regression case -- the asset-less
#: ``v0.1.2`` tag -- and the current metadata lives in the fixtures beside the
#: prerelease tests below.
RELEASES_2026_08_18: list[dict[str, Any]] = [
    _release("v0.1.2", prerelease=True),
    _release("b10488", assets=_win_cuda_assets("b10488")),
    _release("b10486", assets=_win_cuda_assets("b10486")),
    _release("b10485", assets=_win_cuda_assets("b10485")),
]


def _github(
    releases: list[dict[str, Any]],
    seen: list[Any] | None = None,
    *,
    pages: list[list[dict[str, Any]]] | None = None,
) -> httpx.AsyncClient:
    """A client answering the two GitHub endpoints the manager calls.

    ``pages`` serves ``?page=N`` from a list of pages, the way the real API
    does; without it every request gets ``releases`` (a short page, so the
    manager stops after one). ``seen`` collects ``(per_page, page)`` so a test
    can assert how the walk was actually driven (D49-2).
    """
    served = pages if pages is not None else [releases]
    by_tag = {str(entry["tag_name"]): entry for page in served for entry in page}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/releases"):
            number = int(request.url.params.get("page") or 1)
            if seen is not None:
                seen.append((request.url.params.get("per_page"), number))
            body = served[number - 1] if 1 <= number <= len(served) else []
            return httpx.Response(200, json=body)
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
    pages: list[list[dict[str, Any]]] | None = None,
) -> EngineManager:
    mgr = EngineManager(
        config,
        probe=StubProbe(MIXED_GPUS, cuda),
        client=_github(releases, seen, pages=pages),
    )
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
    # A full page every time, numbered from 1 (D49-2). The old
    # ``per_page=min(100, max(limit*2, 20))`` made the ANSWER depend on the
    # limit: with most entries filtered, limit=5 fetched 20 and returned none
    # while limit=50 fetched 100 and returned 26, from one repository.
    assert seen == [("100", 1)]


@pytest.mark.asyncio
async def test_a_prerelease_flagged_build_release_is_still_offered(tmp_config: Config) -> None:
    """The 2026-08-28 blackout, in one assertion (D49-1).

    Upstream began publishing its ordinary ``bNNNN`` builds with GitHub's
    ``prerelease`` flag set: 71 of the newest 100 entries, measured live that
    day. The 2026-08-18 filter rejected every prerelease outright, so all 120
    builds newer than the installed one became invisible to the GUI, to
    ``engine --check`` and to ``GET /api/engine/releases`` at once -- and the
    empty answer was indistinguishable from "upstream published nothing".
    A ``bNNNN`` tag is a build release whatever the flag says.
    """
    releases = [
        _release("b10669", prerelease=True, assets=_win_cuda_assets("b10669")),
        _release("b10550", prerelease=True, assets=_win_cuda_assets("b10550")),
        _release("b10549", assets=_win_cuda_assets("b10549")),
    ]
    mgr = _github_manager(tmp_config, releases)

    assert await mgr.list_releases(limit=5) == ["b10669", "b10550", "b10549"]
    scan = mgr.last_release_scan
    assert scan is not None
    assert (scan["examined"], scan["kept"], scan["skipped"]) == (3, 3, 0)


@pytest.mark.asyncio
async def test_a_draft_build_release_is_never_offered(tmp_config: Config) -> None:
    """A draft is unpublished: its assets 404, so the flag still decides."""
    releases = [
        _release("b10999", draft=True, prerelease=True, assets=_win_cuda_assets("b10999")),
        _release("b10549", prerelease=True, assets=_win_cuda_assets("b10549")),
    ]
    mgr = _github_manager(tmp_config, releases)

    assert await mgr.list_releases(limit=5) == ["b10549"]
    assert mgr.last_release_scan is not None
    assert mgr.last_release_scan["reasons"] == {SKIP_DRAFT: 1}


@pytest.mark.asyncio
async def test_a_prerelease_version_tag_is_still_rejected(tmp_config: Config) -> None:
    """The case the filter was written for on 2026-08-18 stays rejected.

    ``v0.1.2`` carried no prebuilt archive at all, and D49-1 did not soften
    that: the *tag scheme* rejects it, which is the check that was always doing
    the work. The reason recorded is the prerelease one, because that test runs
    first -- either way the tag never reaches an install.
    """
    releases = [
        _release("v0.1.2", prerelease=True),
        _release("b10549", prerelease=True, assets=_win_cuda_assets("b10549")),
    ]
    mgr = _github_manager(tmp_config, releases)

    assert await mgr.list_releases(limit=5) == ["b10549"]
    assert mgr.last_release_scan is not None
    assert mgr.last_release_scan["reasons"] == {SKIP_PRERELEASE: 1}

    # ...and a non-prerelease vX.Y.Z is rejected by the scheme alone.
    plain = _github_manager(tmp_config, [_release("v0.1.3"), _release("b10549")])
    assert await plain.list_releases(limit=5) == ["b10549"]
    assert plain.last_release_scan is not None
    assert plain.last_release_scan["reasons"] == {SKIP_TAG_SCHEME: 1}


@pytest.mark.asyncio
async def test_list_releases_include_prerelease_never_widens_the_tag_scheme(
    tmp_config: Config,
) -> None:
    """``include_prerelease`` widens the release *kind*, never the tag scheme.

    A ``vX.Y.Z`` tag is uninstallable under any flag: the asset parser, the
    engine directory layout and the source-build clone all assume ``bNNNN``.
    Since D49-1 the flag makes no difference to a ``bNNNN`` build either way --
    it only ever governed the tags that are rejected regardless.
    """
    releases = [
        _release("v0.1.2", prerelease=True),
        _release("b10490", prerelease=True, assets=_win_cuda_assets("b10490")),
        _release("b10488", assets=_win_cuda_assets("b10488")),
    ]
    mgr = _github_manager(tmp_config, releases)
    assert await mgr.list_releases(limit=5) == ["b10490", "b10488"]
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


# ---------------------------------------------------------------------------
# D49-2: the walk is paged, and D49-3: what it dropped is reportable
# ---------------------------------------------------------------------------


def _page(tags: list[str], **kwargs: Any) -> list[dict[str, Any]]:
    return [_release(tag, assets=_win_cuda_assets(tag), **kwargs) for tag in tags]


@pytest.mark.asyncio
async def test_list_releases_pages_on_until_it_has_the_count_it_was_asked_for(
    tmp_config: Config,
) -> None:
    """A run of ineligible entries costs another request, not the answer (D49-2).

    The single over-fetched page this used to do meant a page full of tags the
    filter rejects starved the caller: on 2026-08-28 the newest 100 entries
    were 71 prereleases and the answer to ``limit=5`` was an empty list.
    """
    seen: list[Any] = []
    ineligible = [_release(f"v0.{index}.0") for index in range(RELEASE_PAGE_SIZE)]
    builds = _page([f"b{10600 + index}" for index in range(RELEASE_PAGE_SIZE)], prerelease=True)
    mgr = _github_manager(tmp_config, [], seen=seen, pages=[ineligible, builds])

    assert await mgr.list_releases(limit=5) == ["b10699", "b10698", "b10697", "b10696", "b10695"]
    assert seen == [("100", 1), ("100", 2)]
    scan = mgr.last_release_scan
    assert scan is not None
    assert scan["examined"] == 200
    assert scan["pages"] == 2
    assert scan["reasons"] == {SKIP_TAG_SCHEME: RELEASE_PAGE_SIZE}


@pytest.mark.asyncio
async def test_list_releases_stops_at_the_page_budget_and_says_what_it_saw(
    tmp_config: Config,
) -> None:
    """ "Fewer than asked" is the honest contract past the budget (D49-2/D49-3).

    The docstring used to claim the over-fetch meant it "cannot return fewer
    than asked", which was false in exactly the case that mattered. Three pages
    is the cap; what was examined and why it was dropped is recorded so a
    surface can say "checked 300 releases" instead of "none".
    """
    seen: list[Any] = []
    pages = [
        [_release(f"v{page}.{index}.0") for index in range(RELEASE_PAGE_SIZE)] for page in range(5)
    ]
    mgr = _github_manager(tmp_config, [], seen=seen, pages=pages)

    assert await mgr.list_releases(limit=5) == []
    assert len(seen) == RELEASE_MAX_PAGES
    scan = mgr.last_release_scan
    assert scan is not None
    assert scan["pages"] == RELEASE_MAX_PAGES
    assert scan["examined"] == RELEASE_PAGE_SIZE * RELEASE_MAX_PAGES
    assert scan["kept"] == 0
    assert "checked 300 releases" in describe_release_filter(scan)


@pytest.mark.asyncio
async def test_list_releases_stops_on_a_short_page_without_a_wasted_request(
    tmp_config: Config,
) -> None:
    """A repository with three releases must not be asked for page 2."""
    seen: list[Any] = []
    mgr = _github_manager(tmp_config, [], seen=seen, pages=[_page(["b10549", "b10548", "b10547"])])

    assert await mgr.list_releases(limit=20) == ["b10549", "b10548", "b10547"]
    assert seen == [("100", 1)]
    assert mgr.last_release_scan is not None
    assert mgr.last_release_scan["pages"] == 1


def test_describe_release_filter_counts_each_reason_by_name() -> None:
    """The one renderer every surface shares (D49-3).

    Before this, the GUI update line, the List-releases dialog and
    ``engine --check`` each said "none" in their own words and none of them
    could say why -- which is the only part a user can act on.
    """
    text = describe_release_filter(
        {
            "examined": 100,
            "kept": 26,
            "skipped": 74,
            "pages": 1,
            "reasons": {SKIP_PRERELEASE: 71, SKIP_TAG_SCHEME: 3},
        }
    )
    assert text == (
        "checked 100 releases; 74 filtered (71 prerelease non-build tags, 3 non-bNNNN tags)"
    )
    # Nothing to say stays empty, so a caller can append it unconditionally.
    assert describe_release_filter(None) == ""
    assert describe_release_filter({"examined": 0}) == ""
    assert describe_release_filter({"examined": 3, "skipped": 0, "reasons": {}}) == (
        "checked 3 releases"
    )


@pytest.mark.asyncio
async def test_check_update_carries_the_filter_counts_without_disturbing_skipped(
    tmp_config: Config,
) -> None:
    """``filtered``/``filter_summary`` are new; ``skipped`` keeps its shape.

    Existing consumers iterate ``skipped`` for ``entry["tag"]``/``["reason"]``
    -- the per-tag asset-probe failures -- so the release-filter counts land in
    a separate key rather than being mixed in (D49-3).
    """
    tmp_config.engine.pinned_tag = TAG
    mgr = _github_manager(tmp_config, RELEASES_2026_08_18)
    mgr.set_active(TAG)

    status = await mgr.check_update()

    assert status["skipped"] == []  # every probed tag had an asset
    assert status["filtered"]["examined"] == 4
    assert status["filtered"]["reasons"] == {SKIP_PRERELEASE: 1}
    assert status["filter_summary"] == (
        "checked 4 releases; 1 filtered (1 prerelease non-build tag)"
    )


@pytest.mark.asyncio
async def test_check_update_says_every_release_was_filtered_rather_than_nothing(
    tmp_config: Config,
) -> None:
    """The blackout as a user would have met it: nothing offered, and why."""
    tmp_config.engine.allow_source_build = False
    mgr = _github_manager(tmp_config, [_release(f"v0.{index}.0") for index in range(4)])
    mgr.set_active(TAG)

    status = await mgr.check_update()

    assert status["latest"] is None and status["update_available"] is False
    assert status["recent"] == []
    assert status["filter_summary"] == ("checked 4 releases; 4 filtered (4 non-bNNNN tags)")


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


# ---------------------------------------------------------------------------
# D49-4/5/10 at the REST surface
# ---------------------------------------------------------------------------


def _engine_state(config: Config, manager: Any, records: Any = ()) -> Any:
    return type(
        "State",
        (),
        {
            "config": config,
            "engine_manager": manager,
            "registry": SimpleNamespace(all=lambda: list(records)),
        },
    )()


def _stub_capture(manager: EngineManager, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_capture(binary: Path, args: Any) -> tuple[int, str]:
        return 0, SAMPLE_HELP

    monkeypatch.setattr(manager, "_capture", fake_capture)


@pytest.mark.asyncio
async def test_engine_status_reports_drift_only_when_the_pin_and_the_active_build_differ(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``active.json`` wins at load time, so a lone pin edit changes nothing.

    Before D49-5 that disagreement was a boot-time log line and nothing else:
    ``PATCH /api/config`` with a new ``engine.pinned_tag`` returned 200 and the
    box kept running the old build, with no surface saying so.
    """
    from studioforge.api import mgmt_routes

    tmp_config.engine.pinned_tag = TAG
    mgr = EngineManager(tmp_config, probe=StubProbe(MIXED_GPUS, (13, 3)))
    _fake_engine(mgr.engines_dir, TAG, 1_000)
    _fake_engine(mgr.engines_dir, "b10549", 2_000)
    _stub_capture(mgr, monkeypatch)
    request = _FakeRequest(_engine_state(tmp_config, mgr))

    mgr.set_active(TAG)
    assert (await mgmt_routes.engine_status(request))["drift"] is None

    mgr.set_active("b10549")
    payload = await mgmt_routes.engine_status(request)
    assert payload["drift"] == {"pinned": TAG, "active": "b10549"}
    assert payload["pinned_tag"] == TAG
    assert payload["active"]["tag"] == "b10549"


@pytest.mark.asyncio
async def test_engine_status_carries_the_install_progress_snapshot(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ~600 MB download is otherwise a blind wait for a REST caller (D49-10).

    Read with ``getattr`` on purpose -- the Dashboard polls this route on a
    timer, so a manager that has never run an install in this process must
    report nothing rather than fail.
    """
    from studioforge.api import mgmt_routes

    request = _FakeRequest(_engine_state(manager.config, manager))
    assert (await mgmt_routes.engine_status(request))["install_progress"] is None

    _stub_prebuilt(manager, monkeypatch)
    await manager.install(TAG)

    progress = (await mgmt_routes.engine_status(request))["install_progress"]
    assert progress["tag"] == TAG
    assert progress["done"] is True and progress["error"] is None

    # A manager without the attribute at all degrades to null, not a 500.
    bare = SimpleNamespace(active=lambda: None, installed=list)
    blank = await mgmt_routes.engine_status(_FakeRequest(_engine_state(manager.config, bare)))
    assert blank["install_progress"] is None


@pytest.mark.asyncio
async def test_the_install_route_does_not_activate_unless_asked(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``activate`` defaults to False on the wire too (D49-4)."""
    from studioforge.api import mgmt_routes

    _stub_prebuilt(manager, monkeypatch)
    request = _FakeRequest(_engine_state(manager.config, manager))

    # Called with the route's own defaults spelled out: reached through FastAPI
    # these arrive as ``Body(False)``, and the point is the value they carry.
    payload = await mgmt_routes.engine_install(request, tag=TAG, force=False, activate=False)
    assert payload["tag"] == TAG
    assert payload["active"] is False
    assert not (manager.engines_dir / "active.json").exists()

    reinstalled = await mgmt_routes.engine_install(request, tag=TAG, force=False, activate=True)
    assert reinstalled["active"] is True
    assert manager._read_active() == TAG


@pytest.mark.asyncio
async def test_the_activate_route_switches_pins_sweeps_and_names_the_previous_build(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The three writes D49-5 keeps together, plus the note nobody expects.

    A running llama-server child keeps the build it was launched with, which is
    the single most-reported surprise about switching engines -- so the route
    that switches says so in its own response.
    """
    from studioforge.api import mgmt_routes

    manager.config.engine.pinned_tag = TAG
    _fake_engine(manager.engines_dir, TAG, 1_000)
    _fake_engine(manager.engines_dir, "b10549", 2_000)
    manager.set_active(TAG)
    _stub_capture(manager, monkeypatch)
    records = [
        _record("vendor/Stale-Q4_K_M", "--draft-max 4"),
        _record("vendor/Fine-Q4_K_M", "--ctx-size 4096"),
    ]
    request = _FakeRequest(_engine_state(manager.config, manager, records))

    payload = await mgmt_routes.engine_activate(request, tag="b10549")

    assert payload["tag"] == "b10549"
    assert payload["previous"] == TAG
    assert payload["note"] == mgmt_routes.ENGINE_ACTIVATE_NOTE
    assert "POST /api/restart/backend" in payload["note"]
    assert [entry["model_id"] for entry in payload["offenders"]] == ["vendor/Stale-Q4_K_M"]
    assert "--draft-max" in "; ".join(payload["offenders"][0]["errors"])

    # Both halves are written: the one loads read, and the one a restart reads.
    assert manager._read_active() == "b10549"
    assert manager.config.engine.pinned_tag == "b10549"
    assert "pinned_tag: b10549" in manager.config.config_path.read_text(encoding="utf-8")

    # ...and the drift the pair exists to prevent is gone.
    assert (await mgmt_routes.engine_status(request))["drift"] is None


@pytest.mark.asyncio
async def test_the_activate_route_refuses_a_build_that_is_not_installed(
    manager: EngineManager,
) -> None:
    """Activation is not an implicit download; the pin must not move either."""
    from studioforge.api import mgmt_routes

    manager.config.engine.pinned_tag = TAG
    _fake_engine(manager.engines_dir, TAG, 1_000)
    manager.set_active(TAG)
    request = _FakeRequest(_engine_state(manager.config, manager))

    with pytest.raises(EngineError, match="b99999"):
        await mgmt_routes.engine_activate(request, tag="b99999")

    assert manager._read_active() == TAG
    assert manager.config.engine.pinned_tag == TAG


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

    last_release_scan: dict[str, Any] | None = {
        "examined": 100,
        "kept": 26,
        "skipped": 74,
        "pages": 1,
        "reasons": {SKIP_PRERELEASE: 71, SKIP_TAG_SCHEME: 3},
    }


class _EngineCliManager:
    """A CLI double that writes the one file the ordering argument is about.

    ``active.json`` is written by :meth:`activate` and by an ``activate=True``
    install, and by nothing else -- exactly as the real manager behaves since
    D49-4. That is what lets a test assert what ``engine --update`` *left
    behind* rather than what it printed, which is the whole point: the old code
    printed "keeping b10425" over an ``active.json`` that already said b10488.
    """

    calls: list[str] = []
    smoke_ok: bool = True
    offenders: list[dict[str, Any]] = []

    def __init__(self, config: Any, **_kwargs: Any) -> None:
        self.config = config

    # --- helpers ---------------------------------------------------------

    def _write_active(self, tag: str) -> None:
        self.config.engines_dir.mkdir(parents=True, exist_ok=True)
        (self.config.engines_dir / "active.json").write_text(
            json.dumps({"tag": tag}), encoding="utf-8"
        )

    @staticmethod
    def _info(tag: str, *, active: bool) -> Any:
        return SimpleNamespace(
            tag=tag,
            variant="cuda-13.3",
            path=Path("engines") / tag,
            active=active,
            smoke_tested=True,
        )

    # --- the manager surface the CLI uses --------------------------------

    async def check_update(self, *, limit: int = 5, probe_assets: int = 3) -> dict[str, Any]:
        type(self).calls.append(f"check_update(limit={limit})")
        return {
            "checked": True,
            "current": TAG,
            "latest": "b10488",
            "update_available": True,
            "recent": ["b10488"],
            "latest_variant": "cuda-13.3",
            "skipped": [],
            "filtered": {},
            "filter_summary": "checked 10 releases",
        }

    async def install(self, tag: str, *, activate: bool = False, **_kwargs: Any) -> Any:
        type(self).calls.append(f"install({tag}, activate={activate})")
        if activate:
            self._write_active(tag)
        return self._info(tag, active=activate)

    async def smoke_test(self, tag: str, **_kwargs: Any) -> tuple[bool, str]:
        type(self).calls.append(f"smoke_test({tag})")
        return type(self).smoke_ok, f"micro-load detail for {tag}"

    async def activate(self, tag: str) -> Any:
        type(self).calls.append(f"activate({tag})")
        self._write_active(tag)
        return self._info(tag, active=True)

    async def revalidate_extra_flags(self, tag: str, records: Any) -> list[dict[str, Any]]:
        type(self).calls.append(f"revalidate_extra_flags({tag})")
        return list(type(self).offenders)

    async def ensure_engine(self) -> Any:
        return self._info(TAG, active=True)

    def installed(self) -> list[Any]:
        return []


def _active_tag(config: Config) -> str | None:
    path = config.engines_dir / "active.json"
    if not path.is_file():
        return None
    return str(json.loads(path.read_text(encoding="utf-8"))["tag"])


def _cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_config: Config,
    *args: str,
    manager_cls: type[Any] = _StubCliManager,
) -> Any:
    """Invoke the real CLI with the config load and the manager stubbed out.

    ``_load`` is replaced rather than pointed at a temp file because it also
    reconfigures global logging onto a directory that outlives the test.
    """
    from typer.testing import CliRunner

    from studioforge import __main__ as main_cli
    from studioforge.core import engine as engine_module

    _StubCliManager.calls = []
    _EngineCliManager.calls = []
    monkeypatch.setattr(main_cli, "_load", lambda _path: tmp_config)
    monkeypatch.setattr(engine_module, "EngineManager", manager_cls)
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


def test_cli_engine_list_says_what_the_filter_actually_drops(
    monkeypatch: pytest.MonkeyPatch, tmp_config: Config
) -> None:
    """The sentence under the list is now true, and counted (D49-1/D49-3).

    It used to read "prereleases and non-bNNNN tags are hidden", which on
    2026-08-28 was a claim defending a blackout of 120 build releases. The tag
    scheme is the filter that means anything, and the counts say how much the
    list differs from GitHub's front page.
    """
    result = _cli(monkeypatch, tmp_config, "engine", "--list")
    assert result.exit_code == 0
    assert "b10488 b10486" in result.output
    assert "checked 100 releases; 74 filtered" in result.output
    assert "prereleases are NOT" in result.output
    assert "drafts and non-bNNNN tags are hidden" in result.output


# ---------------------------------------------------------------------------
# D49-4/5/6 at the CLI: install, smoke-test, and only then activate
# ---------------------------------------------------------------------------


def test_cli_engine_update_keeps_the_current_engine_when_the_smoke_test_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_config: Config
) -> None:
    """The message and the filesystem finally agree.

    ``install()`` used to activate unconditionally and the smoke test ran
    afterwards, so this branch printed "keeping b10425" while ``active.json``
    already said b10488 -- the failed build was live and the output said the
    opposite. The install is now ``activate=False``, so the lie is not
    expressible.
    """
    _EngineCliManager.smoke_ok = False
    tmp_config.engines_dir.mkdir(parents=True, exist_ok=True)
    (tmp_config.engines_dir / "active.json").write_text(json.dumps({"tag": TAG}), encoding="utf-8")
    try:
        result = _cli(monkeypatch, tmp_config, "engine", "--update", manager_cls=_EngineCliManager)
    finally:
        _EngineCliManager.smoke_ok = True

    assert result.exit_code == 1
    assert _EngineCliManager.calls == [
        "check_update(limit=10)",
        "install(b10488, activate=False)",
        "smoke_test(b10488)",
    ]
    assert "activate(b10488)" not in _EngineCliManager.calls
    assert _active_tag(tmp_config) == TAG, "the failed build must not be live"
    assert tmp_config.engine.pinned_tag != "b10488"
    assert "keeping b10425 active" in result.output
    assert "stays installed but unused" in result.output


def test_cli_engine_update_activates_pins_and_sweeps_once_the_smoke_test_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_config: Config
) -> None:
    """The three writes D49-5 keeps together, in the only safe order."""
    _EngineCliManager.offenders = [
        {"model_id": "vendor/Stale-Q4_K_M", "errors": ["unknown flag '--draft-max'"]}
    ]
    try:
        result = _cli(monkeypatch, tmp_config, "engine", "--update", manager_cls=_EngineCliManager)
    finally:
        _EngineCliManager.offenders = []

    assert result.exit_code == 0
    assert _EngineCliManager.calls[:5] == [
        "check_update(limit=10)",
        "install(b10488, activate=False)",
        "smoke_test(b10488)",
        "activate(b10488)",
        "revalidate_extra_flags(b10488)",
    ]
    assert _active_tag(tmp_config) == "b10488"
    assert tmp_config.engine.pinned_tag == "b10488"
    # ...and the pin is on disk, so it survives the restart it exists for.
    assert "pinned_tag: b10488" in tmp_config.config_path.read_text(encoding="utf-8")
    # The sweep is a warning, never a refusal: llama-server ignores flags it
    # does not know, which is exactly why it has to be said out loud (D49-6).
    assert "WARNING vendor/Stale-Q4_K_M" in result.output
    assert "--draft-max" in result.output
    assert "Restart engines" in result.output


def test_cli_engine_activate_switches_and_pins_an_installed_build(
    monkeypatch: pytest.MonkeyPatch, tmp_config: Config
) -> None:
    """``--activate`` is the half of the split that has no download in it (D49-4)."""
    result = _cli(
        monkeypatch, tmp_config, "engine", "--activate", "b10488", manager_cls=_EngineCliManager
    )

    assert result.exit_code == 0
    assert _EngineCliManager.calls == [
        "activate(b10488)",
        "revalidate_extra_flags(b10488)",
    ]
    assert "check_update(limit=10)" not in _EngineCliManager.calls
    assert _active_tag(tmp_config) == "b10488"
    assert tmp_config.engine.pinned_tag == "b10488"
    assert "now active and pinned" in result.output


def test_cli_engine_install_leaves_the_active_engine_alone_and_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_config: Config
) -> None:
    result = _cli(
        monkeypatch, tmp_config, "engine", "--install", "b10488", manager_cls=_EngineCliManager
    )

    assert result.exit_code == 0
    assert _EngineCliManager.calls == ["install(b10488, activate=False)"]
    assert _active_tag(tmp_config) is None
    assert "the active engine is unchanged" in result.output
    assert "engine --activate b10488" in result.output


def test_cli_engine_install_can_activate_after_the_install_when_asked(
    monkeypatch: pytest.MonkeyPatch, tmp_config: Config
) -> None:
    result = _cli(
        monkeypatch,
        tmp_config,
        "engine",
        "--install",
        "b10488",
        "--activate-after-install",
        manager_cls=_EngineCliManager,
    )

    assert result.exit_code == 0
    assert _EngineCliManager.calls == [
        "install(b10488, activate=True)",
        "activate(b10488)",
        "revalidate_extra_flags(b10488)",
    ]
    assert _active_tag(tmp_config) == "b10488"
    assert tmp_config.engine.pinned_tag == "b10488"


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
