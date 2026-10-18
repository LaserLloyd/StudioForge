"""Backend-aware engine selection, checksum verification and per-backend builds.

Offline throughout: GitHub is a ``httpx.MockTransport``, downloads are stubbed,
and nothing is compiled or launched.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest

from studioforge.config import Config
from studioforge.core.engine import (
    BIN_NAME,
    EngineAsset,
    EngineChecksumError,
    EngineError,
    EngineManager,
    EngineUnverifiedError,
    Verification,
    _SmokeResult,
    build_assets,
    cmake_backend_flags,
    detected_backend,
    engine_record,
    gpu_backend,
    guess_variant,
    parse_digest,
    parse_sha256_text,
    sha256_file,
    variant_backend,
)
from studioforge.types import GpuInfo
from tests.unit.test_engine import (
    ASSETS,
    MIXED_GPUS,
    RAW_ASSETS,
    TAG,
    StubProbe,
    _asset,
    _simulate_build_step,
    _stub_prebuilt,
)

GIB = 1024**3


def gpu(index: int, name: str, **kw: Any) -> GpuInfo:
    return GpuInfo(index=index, name=name, total_bytes=24 * GIB, free_bytes=24 * GIB, **kw)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manager_for(
    tmp_path: Path,
    gpus: list[GpuInfo],
    *,
    os_token: str = "ubuntu",
    arch: str = "x64",
    handler: Any = None,
    **engine_cfg: Any,
) -> EngineManager:
    config = Config(data_dir=tmp_path / "data")
    config.ensure_dirs()
    for key, value in engine_cfg.items():
        setattr(config.engine, key, value)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler)) if handler else None
    mgr = EngineManager(config, probe=StubProbe(gpus, (13, 3)), client=client)
    mgr.os_token, mgr.arch_token = os_token, arch
    return mgr


# ---------------------------------------------------------------------------
# backend inference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("variant", "backend"),
    [
        ("cuda-13.3", "cuda"),
        ("cuda", "cuda"),
        ("rocm-7.14", "rocm"),
        ("hip-radeon", "rocm"),
        ("vulkan", "vulkan"),
        ("metal", "metal"),
        ("cpu", ""),
        ("source-local", ""),
        ("unknown", ""),
    ],
)
def test_variant_backend(variant: str, backend: str) -> None:
    assert variant_backend(variant) == backend


def test_gpu_backend_prefers_the_probes_answer_then_vendor_then_name() -> None:
    assert gpu_backend(gpu(0, "anything", backend="vulkan")) == "vulkan"
    assert gpu_backend(gpu(0, "anything", vendor="amd")) == "rocm"
    assert gpu_backend(gpu(0, "NVIDIA GeForce RTX 5090")) == "cuda"
    assert gpu_backend(gpu(0, "Apple M2 Max")) == "metal"
    assert gpu_backend(gpu(0, "Intel Arc A770")) == "vulkan"
    # legacy reading preserved: an unrecognised non-NVIDIA name is assumed AMD
    assert gpu_backend(gpu(0, "AMD Radeon RX 7900 XTX")) == "rocm"
    assert gpu_backend(gpu(0, "Mystery Accelerator")) == "rocm"


def test_detected_backend() -> None:
    assert detected_backend([]) == "cuda"  # nothing detected: the historical path
    assert detected_backend(MIXED_GPUS) == "cuda"
    assert detected_backend([gpu(0, "x", backend="vulkan")]) == "vulkan"
    assert detected_backend([gpu(0, "x", backend="vulkan"), gpu(1, "y", backend="rocm")]) == "rocm"


def test_guess_variant_recognises_hip_and_metal_libraries(tmp_path: Path) -> None:
    (tmp_path / "libggml-hip.so").write_bytes(b"")
    assert guess_variant(tmp_path) == "rocm"
    other = tmp_path / "mac"
    other.mkdir()
    (other / "libggml-metal.dylib").write_bytes(b"")
    assert guess_variant(other) == "metal"


# ---------------------------------------------------------------------------
# select_asset per backend
# ---------------------------------------------------------------------------

LINUX_RAW = [
    _asset(f"llama-{TAG}-bin-ubuntu-x64.tar.gz", 1),
    _asset(f"llama-{TAG}-bin-ubuntu-vulkan-x64.tar.gz", 2),
    _asset(f"llama-{TAG}-bin-ubuntu-rocm-7.0-x64.tar.gz", 3),
    _asset(f"llama-{TAG}-bin-ubuntu-rocm-7.14-x64.tar.gz", 4),
]
WIN_RAW = [
    *RAW_ASSETS,
    _asset(f"llama-{TAG}-bin-win-hip-radeon-x64.zip", 5),
]


def test_rocm_gpu_picks_the_highest_rocm_archive_on_linux(tmp_path: Path) -> None:
    mgr = manager_for(tmp_path, [])
    amd = [gpu(0, "AMD Radeon RX 7900 XTX", backend="rocm", gfx_target="gfx1100")]
    chosen = mgr.select_asset(build_assets(TAG, LINUX_RAW), gpus=amd, cuda_driver=None)
    assert chosen is not None and chosen.variant == "rocm-7.14"


def test_vulkan_gpu_picks_the_vulkan_archive_never_cpu(tmp_path: Path) -> None:
    mgr = manager_for(tmp_path, [])
    vk = [gpu(0, "Intel Arc A770", backend="vulkan")]
    chosen = mgr.select_asset(build_assets(TAG, LINUX_RAW), gpus=vk, cuda_driver=None)
    assert chosen is not None and chosen.variant == "vulkan"
    # on Windows too, and nothing is chosen when the release has no Vulkan archive
    win = manager_for(tmp_path, [], os_token="win")
    assert (
        win.select_asset(build_assets(TAG, WIN_RAW), gpus=vk, cuda_driver=None).variant == "vulkan"
    )  # type: ignore[union-attr]
    no_vulkan = [a for a in build_assets(TAG, LINUX_RAW) if a.variant != "vulkan"]
    assert mgr.select_asset(no_vulkan, gpus=vk, cuda_driver=None) is None


def test_windows_hip_archive_ranks_below_a_versioned_rocm_one(tmp_path: Path) -> None:
    mgr = manager_for(tmp_path, [], os_token="win")
    amd = [gpu(0, "AMD Radeon RX 7900 XTX", backend="rocm")]
    assets = build_assets(TAG, WIN_RAW)
    assert mgr.select_asset(assets, gpus=amd, cuda_driver=None).variant == "rocm-7.14"  # type: ignore[union-attr]
    only_hip = [a for a in assets if a.variant.startswith("hip")]
    assert mgr.select_asset(only_hip, gpus=amd, cuda_driver=None).variant == "hip-radeon"  # type: ignore[union-attr]


def test_apple_gpu_picks_the_macos_arm64_archive_labelled_metal(tmp_path: Path) -> None:
    mgr = manager_for(tmp_path, [], os_token="macos", arch="arm64")
    assets = build_assets(TAG, RAW_ASSETS)
    mac = next(a for a in assets if a.os_token == "macos")
    assert mac.variant == "metal" and mac.backend == "metal"
    m2 = [gpu(0, "Apple M2 Max", backend="metal", unified_memory=True)]
    assert mgr.select_asset(assets, gpus=m2, cuda_driver=None) is mac
    # an NVIDIA description of the same box still finds nothing (no CUDA on macOS)
    assert mgr.select_asset(assets, gpus=MIXED_GPUS, cuda_driver=(13, 3)) is None


def test_cuda_behaviour_is_unchanged_for_nvidia(tmp_path: Path) -> None:
    mgr = manager_for(tmp_path, MIXED_GPUS, os_token="win")
    chosen = mgr.select_asset(ASSETS, gpus=MIXED_GPUS, cuda_driver=(13, 3))
    assert chosen is not None and chosen.variant == "cuda-13.3"
    # ...even when a Vulkan archive exists: the CUDA probe never selects it.
    assert not chosen.variant.startswith("vulkan")


def test_explicit_backend_argument_overrides_detection(tmp_path: Path) -> None:
    mgr = manager_for(tmp_path, [], os_token="win")
    chosen = mgr.select_asset(ASSETS, gpus=MIXED_GPUS, cuda_driver=(13, 3), backend="vulkan")
    assert chosen is not None and chosen.variant == "vulkan"


# ---------------------------------------------------------------------------
# build_assets: digests, sidecars, sums
# ---------------------------------------------------------------------------


def test_build_assets_reads_github_digest_sidecar_and_sums() -> None:
    zip_name = f"llama-{TAG}-bin-win-cuda-13.3-x64.zip"
    digest = "ab" * 32
    raw = [
        {**_asset(zip_name, 10), "digest": f"sha256:{digest.upper()}"},
        _asset(f"llama-{TAG}-bin-win-vulkan-x64.zip", 11),
        _asset(f"llama-{TAG}-bin-win-vulkan-x64.zip.sha256", 1),
        _asset("SHA256SUMS.txt", 1),
        {**_asset("cudart-llama-bin-win-cuda-13.3-x64.zip", 5), "digest": f"sha256:{'cd' * 32}"},
    ]
    by_variant = {a.variant: a for a in build_assets(TAG, raw)}
    cuda, vulkan = by_variant["cuda-13.3"], by_variant["vulkan"]
    assert cuda.sha256 == digest  # normalised to lower case
    assert cuda.cudart_sha256 == "cd" * 32 and cuda.needs_cudart
    assert vulkan.sha256 is None
    assert vulkan.checksum_url and vulkan.checksum_url.endswith("vulkan-x64.zip.sha256")
    assert vulkan.sums_url and vulkan.sums_url.endswith("SHA256SUMS.txt")
    assert cuda.checksum_url is None


def test_parse_digest_rejects_other_algorithms_and_junk() -> None:
    assert parse_digest(f"sha256:{'0f' * 32}") == "0f" * 32
    assert parse_digest("sha1:abcd") is None
    assert parse_digest("sha256:short") is None
    assert parse_digest(None) is None
    assert parse_digest(12) is None


H1, H2 = "11" * 32, "22" * 32


@pytest.mark.parametrize(
    ("text", "name", "expected"),
    [
        (f"{H1}  a.zip\n{H2}  b.zip\n", "b.zip", H2),
        (f"{H1} *a.zip\n", "a.zip", H1),
        (f"{H1}  ./dist/a.zip\n", "a.zip", H1),
        (f"SHA256 (a.zip) = {H1}\nSHA256 (b.zip) = {H2}\n", "b.zip", H2),
        (f"{H1}\n", "anything.zip", H1),  # a sidecar: one bare hash
        (f"{H1}\n{H2}\n", "a.zip", None),  # bare hashes in a multi-entry file are unattributable
        (f"{H1}  a.zip\n", "other.zip", None),
        ("no hashes here\n", "a.zip", None),
        ("", "a.zip", None),
    ],
)
def test_parse_sha256_text(text: str, name: str, expected: str | None) -> None:
    assert parse_sha256_text(text, name) == expected


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "blob"
    path.write_bytes(b"x" * (3 * 1024 * 1024 + 7))
    assert sha256_file(path) == sha(b"x" * (3 * 1024 * 1024 + 7))


# ---------------------------------------------------------------------------
# _expected_digest / _verify_download
# ---------------------------------------------------------------------------


def make_asset(name: str = "llama-b1-bin-win-cuda-13.3-x64.zip", **kw: Any) -> EngineAsset:
    return EngineAsset(
        tag="b1",
        name=name,
        url=f"https://example.invalid/{name}",
        size_bytes=1,
        variant="cuda-13.3",
        needs_cudart=False,
        cudart_url=None,
        **kw,
    )


@pytest.mark.asyncio
async def test_expected_digest_order_github_then_sidecar_then_sums(tmp_path: Path) -> None:
    served: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        served.append(str(request.url))
        if request.url.path.endswith(".sha256"):
            return httpx.Response(200, text=f"{H1}  llama-b1-bin-win-cuda-13.3-x64.zip\n")
        if request.url.path.endswith("SHA256SUMS"):
            return httpx.Response(200, text=f"{H2}  llama-b1-bin-win-cuda-13.3-x64.zip\n")
        return httpx.Response(404)

    mgr = manager_for(tmp_path, [], handler=handler)
    assert await mgr._expected_digest(make_asset(sha256=H1)) == (H1, "github-digest")
    assert served == []  # nothing fetched when the API already said
    side = make_asset(
        checksum_url="https://example.invalid/x.sha256", sums_url="https://e.invalid/SHA256SUMS"
    )
    assert await mgr._expected_digest(side) == (H1, "sha256-sidecar")
    sums_only = make_asset(sums_url="https://e.invalid/SHA256SUMS")
    assert await mgr._expected_digest(sums_only) == (H2, "sha256sums")
    none, why = await mgr._expected_digest(make_asset())
    assert none is None and "publishes no checksum" in why


@pytest.mark.asyncio
async def test_expected_digest_survives_a_failing_or_useless_checksum_file(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "dead" in request.url.host:
            raise httpx.ConnectError("down")
        return httpx.Response(200, text="not a checksum file\n")

    mgr = manager_for(tmp_path, [], handler=handler)
    digest, why = await mgr._expected_digest(
        make_asset(checksum_url="https://dead.invalid/x.sha256")
    )
    assert digest is None and "could not be fetched" in why
    digest, why = await mgr._expected_digest(make_asset(sums_url="https://ok.invalid/SHA256SUMS"))
    assert digest is None and "has no entry" in why


@pytest.mark.asyncio
async def test_verify_download_accepts_a_matching_archive(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    archive.write_bytes(b"payload")
    mgr = manager_for(tmp_path, [])
    result = await mgr._verify_download(make_asset(sha256=sha(b"payload")), archive)
    assert result == Verification(True, "github-digest", sha(b"payload"))
    assert archive.exists()


@pytest.mark.asyncio
async def test_verify_download_rejects_and_deletes_a_mismatch(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    archive.write_bytes(b"tampered")
    mgr = manager_for(tmp_path, [])
    with pytest.raises(EngineChecksumError) as excinfo:
        await mgr._verify_download(make_asset(sha256=sha(b"payload")), archive)
    assert excinfo.value.code == "engine_checksum_mismatch"
    assert excinfo.value.details["expected"] == sha(b"payload")
    assert excinfo.value.details["actual"] == sha(b"tampered")
    assert not archive.exists()


@pytest.mark.asyncio
async def test_verify_download_without_a_checksum_is_unverified_and_loud(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    archive = tmp_path / "a.zip"
    archive.write_bytes(b"payload")
    mgr = manager_for(tmp_path, [])
    result = await mgr._verify_download(make_asset(), archive)
    assert result.verified is False and result.sha256 is None
    assert result.by.startswith("none:")
    assert result.warnings and "WITHOUT checksum verification" in result.warnings[0]
    assert archive.exists()  # installed, but never silently


@pytest.mark.asyncio
async def test_require_verified_refuses_an_unpublished_checksum(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    archive.write_bytes(b"payload")
    mgr = manager_for(tmp_path, [], require_verified=True)
    with pytest.raises(EngineUnverifiedError) as excinfo:
        await mgr._verify_download(make_asset(), archive)
    assert excinfo.value.code == "engine_unverified_refused"
    assert not archive.exists()


@pytest.mark.asyncio
async def test_cudart_bundle_is_checked_when_a_digest_exists_and_warned_about_otherwise(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "cudart-x.zip"
    bundle.write_bytes(b"runtime")
    mgr = manager_for(tmp_path, [])
    base = Verification(True, "github-digest", "a" * 64)
    ok = await mgr._verify_companion(make_asset(cudart_sha256=sha(b"runtime")), bundle, base)
    assert ok == base
    warned = await mgr._verify_companion(make_asset(), bundle, base)
    assert warned.verified is True and "CUDA runtime bundle" in warned.warnings[0]
    with pytest.raises(EngineChecksumError):
        await mgr._verify_companion(make_asset(cudart_sha256="0" * 64), bundle, base)
    assert not bundle.exists()


# ---------------------------------------------------------------------------
# install end to end: what lands in engine.json / engine list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_install_records_backend_verification_and_smoke_in_engine_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = manager_for(tmp_path, MIXED_GPUS, os_token="win")
    _stub_prebuilt(mgr, monkeypatch)
    payload = b"archive"

    async def digest(asset: EngineAsset) -> tuple[str | None, str]:
        return sha(payload), "github-digest"

    async def download(url: str, target: Path, expected: int, phase: str, progress: Any) -> Path:
        _make_zip_with(target, payload)
        return target

    monkeypatch.setattr(mgr, "_expected_digest", digest)
    monkeypatch.setattr(mgr, "_download", download)
    # the stub archive is a different file than `payload`; hash what is on disk
    monkeypatch.setattr("studioforge.core.engine.sha256_file", lambda p: sha(payload))

    info = await mgr.install(TAG)

    assert (info.backend, info.verified, info.verified_by) == ("cuda", True, "github-digest")
    assert info.smoke_result == "passed" and info.archive_sha256 == sha(payload)
    meta = json.loads((mgr.engine_dir(TAG) / "engine.json").read_text(encoding="utf-8"))
    assert meta["backend"] == "cuda" and meta["verified"] is True
    assert meta["verified_by"] == "github-digest" and meta["smoke_result"] == "passed"
    (record,) = mgr.list_engines()
    assert record["tag"] == TAG and record["backend"] == "cuda"
    assert record["verified"] is True and record["smoke_result"] == "passed"
    assert isinstance(record["installed_at"], float)
    # the archive was verified; the fixture's CUDA runtime bundle has no digest
    assert len(record["warnings"]) == 1 and "CUDA runtime bundle" in record["warnings"][0]


def _make_zip_with(target: Path, marker: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr(BIN_NAME, b"BINARY")
        archive.writestr("marker", marker)


@pytest.mark.asyncio
async def test_unverified_install_is_recorded_and_survives_a_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = manager_for(tmp_path, MIXED_GPUS, os_token="win")
    _stub_prebuilt(mgr, monkeypatch)

    info = await mgr.install(TAG)

    assert info.verified is False and info.verified_by.startswith("none:")
    assert info.warnings and "WITHOUT checksum" in info.warnings[0]
    reread = mgr.get(TAG)
    assert reread is not None and reread.verified is False and reread.warnings == info.warnings
    # a second install() reuses the engine on disk and must not forget the verdict
    again = await mgr.install(TAG)
    assert again.verified is False and again.warnings == info.warnings
    assert again.backend == "cuda"


@pytest.mark.asyncio
async def test_a_checksum_mismatch_installs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = manager_for(tmp_path, MIXED_GPUS, os_token="win")
    _stub_prebuilt(mgr, monkeypatch)

    async def wrong(asset: EngineAsset) -> tuple[str | None, str]:
        return "f" * 64, "github-digest"

    monkeypatch.setattr(mgr, "_expected_digest", wrong)
    with pytest.raises(EngineChecksumError):
        await mgr.install(TAG)
    assert mgr.get(TAG) is None
    assert not (mgr.engines_dir / f"_{TAG}.new").exists()


def test_legacy_engine_json_reads_as_unknown_not_unverified(tmp_path: Path) -> None:
    mgr = manager_for(tmp_path, [])
    directory = mgr.engines_dir / TAG
    directory.mkdir(parents=True)
    (directory / BIN_NAME).write_bytes(b"x")
    (directory / "engine.json").write_text(
        json.dumps({"tag": TAG, "variant": "cuda-12.4", "smoke_tested": True}), encoding="utf-8"
    )
    info = mgr.get(TAG)
    assert info is not None
    assert info.verified is None and info.backend == "cuda" and info.smoke_result == "passed"
    assert engine_record(info)["verified"] is None


@pytest.mark.asyncio
async def test_manual_smoke_test_result_is_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = manager_for(tmp_path, MIXED_GPUS, os_token="win")
    _stub_prebuilt(mgr, monkeypatch)
    await mgr.install(TAG)

    async def failing(tag: str, tiny_model: Any) -> _SmokeResult:
        return _SmokeResult(False, "boom", version_ok=True)

    monkeypatch.setattr(mgr, "_smoke", failing)
    ok, _ = await mgr.smoke_test(TAG)
    assert ok is False
    info = mgr.get(TAG)
    assert info is not None and info.smoke_result == "failed"
    assert info.smoke_tested is True  # an earlier pass is not erased by a later failure


# ---------------------------------------------------------------------------
# source builds per backend
# ---------------------------------------------------------------------------


def test_cmake_backend_flags() -> None:
    assert cmake_backend_flags("cuda", ["120", "86", "120"]) == [
        "-DGGML_CUDA=ON",
        "-DCMAKE_CUDA_ARCHITECTURES=86;120",
    ]
    assert cmake_backend_flags("rocm", ["gfx1151", "gfx1100"]) == [
        "-DGGML_HIP=ON",
        "-DAMDGPU_TARGETS=gfx1100;gfx1151",
    ]
    assert cmake_backend_flags("vulkan", []) == ["-DGGML_VULKAN=ON"]
    assert cmake_backend_flags("metal", []) == ["-DGGML_METAL=ON"]


def test_cmake_flags_never_invent_a_target() -> None:
    with pytest.raises(EngineError, match="CMAKE_CUDA_ARCHITECTURES"):
        cmake_backend_flags("cuda", [])
    with pytest.raises(EngineError, match="AMDGPU_TARGETS"):
        cmake_backend_flags("rocm", [])
    with pytest.raises(EngineError, match="unknown backend"):
        cmake_backend_flags("directml", ["x"])


def test_target_arches_come_from_the_probe(tmp_path: Path) -> None:
    amd = [
        gpu(0, "AMD Radeon RX 7900 XTX", backend="rocm", gfx_target="gfx1100"),
        gpu(1, "AMD Radeon RX 7900 XTX", backend="rocm", gfx_target="gfx1100"),
        gpu(2, "AMD Radeon Graphics", backend="rocm", gfx_target="gfx1151"),
        gpu(3, "unknown gfx", backend="rocm"),
    ]
    mgr = manager_for(tmp_path, amd)
    assert mgr.target_arches("rocm") == ["gfx1100", "gfx1151"]
    assert mgr.target_arches("vulkan") == [] and mgr.target_arches("metal") == []
    assert mgr.target_arches("cuda", MIXED_GPUS) == ["86", "120"]


def _record_build(mgr: EngineManager, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {"calls": [], "envs": []}

    async def fake_run(cmd: Any, *, cwd: Any, log_path: Path, phase: str, progress: Any, env=None):
        seen["calls"].append(list(cmd))
        seen["envs"].append(env)
        _simulate_build_step(list(cmd), phase, log_path)
        return 0

    async def fake_smoke(tag: str, tiny_model: Any) -> _SmokeResult:
        return _SmokeResult(True, "stubbed", version_ok=True, version_string="build 1")

    monkeypatch.setattr(mgr, "_run_logged", fake_run)
    monkeypatch.setattr(mgr, "_smoke", fake_smoke)
    monkeypatch.setenv("SF_VENDOR_LLAMA_CPP", str(mgr.config.data_dir / "no-vendor"))
    return seen


@pytest.mark.asyncio
async def test_rocm_source_build_uses_the_probes_gfx_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    amd = [gpu(0, "AMD Radeon Graphics", backend="rocm", gfx_target="gfx1151", unified_memory=True)]
    mgr = manager_for(tmp_path, amd)
    seen = _record_build(mgr, monkeypatch)
    monkeypatch.setattr("studioforge.core.engine._hip_build_env", lambda: {"HIP_PATH": "/opt/rocm"})

    info = await mgr.build_from_source(TAG, arches=mgr.target_arches("rocm"))

    configure = next(c for c in seen["calls"] if c[0] == "cmake" and "-S" in c)
    assert "-DGGML_HIP=ON" in configure and "-DAMDGPU_TARGETS=gfx1151" in configure
    assert not any("GGML_CUDA" in a or "CUDA_ARCHITECTURES" in a for a in configure)
    # the HIP toolchain environment reaches both cmake invocations
    cmake_envs = [e for c, e in zip(seen["calls"], seen["envs"], strict=True) if c[0] == "cmake"]
    assert cmake_envs and all(e == {"HIP_PATH": "/opt/rocm"} for e in cmake_envs)
    assert info.backend == "rocm" and info.variant == "source-local"
    assert info.verified is False and info.verified_by.startswith("source-build")
    assert info.warnings == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "flag"), [("vulkan", "-DGGML_VULKAN=ON"), ("metal", "-DGGML_METAL=ON")]
)
async def test_vulkan_and_metal_source_builds_need_no_arch(
    backend: str, flag: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = manager_for(tmp_path, [gpu(0, "any", backend=backend)])
    seen = _record_build(mgr, monkeypatch)
    info = await mgr.build_from_source(TAG, arches=[])
    configure = next(c for c in seen["calls"] if c[0] == "cmake" and "-S" in c)
    assert flag in configure
    assert not any("GGML_CUDA" in a for a in configure)
    assert info.backend == backend


@pytest.mark.asyncio
async def test_install_falls_back_to_a_rocm_source_build_when_no_archive_fits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    amd = [gpu(0, "AMD Radeon RX 7900 XTX", backend="rocm", gfx_target="gfx1100")]
    mgr = manager_for(tmp_path, amd)
    built: dict[str, Any] = {}

    async def fake_assets(tag: str) -> list[EngineAsset]:
        return build_assets(TAG, [_asset(f"llama-{TAG}-bin-ubuntu-x64.tar.gz", 1)])  # cpu only

    async def fake_build(tag: str, *, arches: Any, progress: Any = None, activate: bool = False):
        built["arches"] = list(arches)
        raise EngineError("stop here")

    monkeypatch.setattr(mgr, "list_assets", fake_assets)
    monkeypatch.setattr(mgr, "build_from_source", fake_build)
    with pytest.raises(EngineError, match="stop here"):
        await mgr.install(TAG)
    assert built["arches"] == ["gfx1100"]


@pytest.mark.asyncio
async def test_building_off_names_the_backends_toolchain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = manager_for(tmp_path, [gpu(0, "AMD Radeon", backend="rocm")], allow_source_build=False)

    async def fake_assets(tag: str) -> list[EngineAsset]:
        return build_assets(TAG, [_asset(f"llama-{TAG}-bin-ubuntu-x64.tar.gz", 1)])

    monkeypatch.setattr(mgr, "list_assets", fake_assets)
    with pytest.raises(EngineError) as excinfo:
        await mgr.install(TAG)
    message = str(excinfo.value)
    assert "cmake" in message and "ROCm" in message and "nvcc" not in message
