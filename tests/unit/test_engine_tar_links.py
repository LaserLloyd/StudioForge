"""Linux engine tarballs ship soname symlinks the loader needs.

System test 2026-10-18: ``llama-b11669-bin-ubuntu-cuda-13.4-x64.tar.gz`` installed with the
links dropped, so ``llama-server`` died with exit 127 ("libllama-common.so.0: cannot open shared
object file") and the smoke test refused the engine. The links are now recreated, but only when
they stay inside the engine directory and end at a real file.
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import pytest

from studioforge.core.engine import extract_engine_archive

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="Linux/macOS tarballs only; symlinks need privileges on Windows"
)


def _add_file(tar: tarfile.TarFile, name: str, data: bytes = b"x", mode: int = 0o755) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    tar.addfile(info, io.BytesIO(data))


def _add_link(tar: tarfile.TarFile, name: str, target: str, *, hard: bool = False) -> None:
    info = tarfile.TarInfo(name)
    info.type = tarfile.LNKTYPE if hard else tarfile.SYMTYPE
    info.linkname = target
    tar.addfile(info)


def _archive(path: Path, build) -> Path:  # type: ignore[no-untyped-def]
    with tarfile.open(path, "w:gz") as tar:
        build(tar)
    return path


def test_soname_chain_is_recreated_in_any_archive_order(tmp_path: Path) -> None:
    def build(tar: tarfile.TarFile) -> None:
        # Links listed BEFORE their targets and as a chain: libggml.so -> .so.0 -> .so.0.27.0
        _add_link(tar, "llama-b1/libggml.so", "libggml.so.0")
        _add_link(tar, "llama-b1/libggml.so.0", "libggml.so.0.27.0")
        _add_file(tar, "llama-b1/libggml.so.0.27.0", b"lib")
        _add_file(tar, "llama-b1/llama-server", b"bin")

    dest = tmp_path / "engine"
    binary = extract_engine_archive(_archive(tmp_path / "e.tar.gz", build), dest)
    assert binary == dest / "llama-server"
    assert (dest / "libggml.so.0").is_symlink()
    assert (dest / "libggml.so").read_bytes() == b"lib"
    assert (dest / "libggml.so.0").readlink() == Path("libggml.so.0.27.0")


def test_links_that_escape_or_dangle_are_refused(tmp_path: Path) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("keep out")

    def build(tar: tarfile.TarFile) -> None:
        _add_file(tar, "llama-b1/llama-server", b"bin")
        _add_link(tar, "llama-b1/escape.so", "../../secret.txt")
        _add_link(tar, "llama-b1/absolute.so", str(outside))
        _add_link(tar, "llama-b1/dangling.so", "missing.so.0")

    dest = tmp_path / "engines" / "b1"
    extract_engine_archive(_archive(tmp_path / "e.tar.gz", build), dest)
    for name in ("escape.so", "absolute.so", "dangling.so"):
        assert not (dest / name).exists() and not (dest / name).is_symlink(), name
    assert outside.read_text() == "keep out"


def test_hard_links_become_copies_inside_the_engine_dir(tmp_path: Path) -> None:
    def build(tar: tarfile.TarFile) -> None:
        _add_file(tar, "llama-b1/llama-server", b"bin")
        _add_file(tar, "llama-b1/libx.so.1.2", b"hard")
        _add_link(tar, "llama-b1/libx.so.1", "llama-b1/libx.so.1.2", hard=True)

    dest = tmp_path / "engine"
    extract_engine_archive(_archive(tmp_path / "e.tar.gz", build), dest)
    assert (dest / "libx.so.1").read_bytes() == b"hard"
    assert not (dest / "libx.so.1").is_symlink()
