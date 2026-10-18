"""Link members in engine archives (2.0.0 README install test).

Upstream's prebuilt Linux llama.cpp tarballs ship ``libllama.so -> libllama.so.0``
style symlinks. The extractor used to drop every link member, so the downloaded
engine could not find its own libraries, failed its smoke test, and
``studioforge engine --update`` was unusable on Linux. Links are now recreated
inside the extraction root; absolute and root-escaping targets are still refused.
"""

from __future__ import annotations

import io
import os
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

from studioforge.core.engine import BIN_NAME, extract_engine_archive, extract_engine_zip

TAG = "b9999"
PREFIX = f"llama-{TAG}"


def _tar(
    path: Path,
    files: dict[str, bytes],
    symlinks: dict[str, str] | None = None,
    hardlinks: dict[str, str] | None = None,
) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755 if name.endswith(BIN_NAME) else 0o644
            archive.addfile(info, io.BytesIO(data))
        for name, target in (symlinks or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            archive.addfile(info)
        for name, target in (hardlinks or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.LNKTYPE
            info.linkname = target
            archive.addfile(info)
    return path


def _is_link_or_copy(path: Path, expected: bytes) -> bool:
    """A real symlink on POSIX; a copied file where symlinks are not used (Windows)."""
    if os.name != "nt":
        return path.is_symlink() and path.read_bytes() == expected
    return path.is_file() and path.read_bytes() == expected


def test_relative_symlinks_are_recreated(tmp_path: Path) -> None:
    archive = _tar(
        tmp_path / "linux.tar.gz",
        {
            f"{PREFIX}/{BIN_NAME}": b"ELF",
            f"{PREFIX}/libllama.so.0": b"so-real",
            f"{PREFIX}/lib/libggml.so.0.9": b"ggml-real",
        },
        symlinks={
            f"{PREFIX}/libllama.so": "libllama.so.0",
            f"{PREFIX}/lib/libggml.so": "libggml.so.0.9",
            f"{PREFIX}/lib/libllama.so": "../libllama.so.0",
        },
    )
    dest = tmp_path / "engines" / TAG

    binary = extract_engine_archive(archive, dest)

    assert binary == dest / BIN_NAME
    assert _is_link_or_copy(dest / "libllama.so", b"so-real")
    assert _is_link_or_copy(dest / "lib" / "libggml.so", b"ggml-real")
    assert _is_link_or_copy(dest / "lib" / "libllama.so", b"so-real")
    if os.name != "nt":
        # The link text is relative, so the engine dir can be moved (the swap
        # renames staging -> engines/<tag>) without dangling.
        assert (dest / "libllama.so").readlink().as_posix() == "libllama.so.0"


def test_a_symlink_chain_resolves(tmp_path: Path) -> None:
    archive = _tar(
        tmp_path / "chain.tar.gz",
        {f"{PREFIX}/{BIN_NAME}": b"ELF", f"{PREFIX}/libx.so.1.2.3": b"x"},
        symlinks={
            f"{PREFIX}/libx.so": "libx.so.1",  # listed before its own target link
            f"{PREFIX}/libx.so.1": "libx.so.1.2.3",
        },
    )
    dest = tmp_path / "out"
    extract_engine_archive(archive, dest)
    assert (dest / "libx.so").read_bytes() == b"x"
    assert (dest / "libx.so.1").read_bytes() == b"x"


def test_absolute_symlink_targets_are_rejected(tmp_path: Path) -> None:
    archive = _tar(
        tmp_path / "abs.tar.gz",
        {f"{PREFIX}/{BIN_NAME}": b"ELF", f"{PREFIX}/libllama.so.0": b"so"},
        symlinks={
            f"{PREFIX}/evil": "/etc/passwd",
            f"{PREFIX}/evil-win": "C:/Windows/System32/config",
        },
    )
    dest = tmp_path / "out"
    extract_engine_archive(archive, dest)
    assert not os.path.lexists(dest / "evil")
    assert not os.path.lexists(dest / "evil-win")
    assert (dest / BIN_NAME).is_file()


def test_symlinks_that_climb_out_of_the_root_are_rejected(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("outside", encoding="utf-8")
    archive = _tar(
        tmp_path / "escape.tar.gz",
        {f"{PREFIX}/{BIN_NAME}": b"ELF", f"{PREFIX}/sub/real.so": b"r"},
        symlinks={
            f"{PREFIX}/up": "../../secret.txt",
            f"{PREFIX}/sub/up": "../../../secret.txt",
            f"{PREFIX}/dirup": "..",
            # Lexically inside the root, but only because ``a/..`` is collapsed.
            f"{PREFIX}/sub/deep": "real.so",
            f"{PREFIX}/sub/viadot": "deep/../../../../secret.txt",
        },
    )
    dest = tmp_path / "out"
    extract_engine_archive(archive, dest)
    for name in ("up", "dirup", "sub/up", "sub/viadot"):
        assert not os.path.lexists(dest / name), name
    # Nothing ever pointed outside: the outside file is untouched.
    assert secret.read_text(encoding="utf-8") == "outside"
    assert _is_link_or_copy(dest / "sub" / "deep", b"r")


@pytest.mark.skipif(os.name == "nt", reason="needs real symlinks to build the chain")
def test_a_chain_that_only_escapes_on_the_real_filesystem_is_refused(tmp_path: Path) -> None:
    """``p/q/b -> a/../../../z`` is lexically inside the root but really above it."""
    archive = _tar(
        tmp_path / "chain-escape.tar.gz",
        {f"{PREFIX}/{BIN_NAME}": b"ELF", f"{PREFIX}/x/keep": b"k"},
        symlinks={
            f"{PREFIX}/p/q/a": "../../x",
            f"{PREFIX}/p/q/b": "a/../../../z",
        },
    )
    dest = tmp_path / "root" / "out"
    extract_engine_archive(archive, dest)
    assert (dest / "p" / "q" / "a").is_symlink()
    if os.path.lexists(dest / "p" / "q" / "b"):
        assert (dest / "p" / "q" / "b").resolve().is_relative_to(dest.resolve())


def test_hardlink_members_become_links_to_the_extracted_file(tmp_path: Path) -> None:
    archive = _tar(
        tmp_path / "hard.tar.gz",
        {f"{PREFIX}/{BIN_NAME}": b"ELF", f"{PREFIX}/libllama.so.0": b"so"},
        hardlinks={
            f"{PREFIX}/libllama.so": f"{PREFIX}/libllama.so.0",
            f"{PREFIX}/bad": "/etc/passwd",
        },
    )
    dest = tmp_path / "out"
    extract_engine_archive(archive, dest)
    assert (dest / "libllama.so").read_bytes() == b"so"
    assert not os.path.lexists(dest / "bad")


def _zip_with_link(path: Path, link_target: str) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{PREFIX}/{BIN_NAME}", b"ELF")
        archive.writestr(f"{PREFIX}/libllama.so.0", b"so")
        info = zipfile.ZipInfo(f"{PREFIX}/libllama.so")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, link_target)
    return path


def test_zip_symlink_members_are_links_not_text_files(tmp_path: Path) -> None:
    dest = tmp_path / "out"
    extract_engine_zip(_zip_with_link(tmp_path / "ok.zip", "libllama.so.0"), dest)
    assert _is_link_or_copy(dest / "libllama.so", b"so")

    bad = tmp_path / "bad"
    extract_engine_zip(_zip_with_link(tmp_path / "bad.zip", "/etc/passwd"), bad)
    assert not os.path.lexists(bad / "libllama.so")


def test_drive_relative_and_stream_member_names_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``C:x/evil.dll`` is not absolute on Windows but lands outside ``dest``; refuse colons."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)  # where a drive-relative write would land on Windows
    drive = tmp_path.drive or "C:"
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"{PREFIX}/{BIN_NAME}", b"EXE")
        zf.writestr(f"{PREFIX}/ggml.dll", b"DLL")
        zf.writestr(f"{PREFIX}/{drive}escape/evil.dll", b"PWN")
        zf.writestr(f"{PREFIX}/{BIN_NAME}:hidden", b"ADS")
    dest = tmp_path / "engines" / TAG

    binary = extract_engine_zip(archive, dest)

    assert binary == dest / BIN_NAME and binary.read_bytes() == b"EXE"
    assert not list(cwd.rglob("evil.dll"))
    assert not list(tmp_path.rglob("evil.dll"))
    assert sorted(p.name for p in dest.iterdir()) == sorted([BIN_NAME, "ggml.dll"])

    tarball = _tar(
        tmp_path / "evil.tar.gz",
        {f"{PREFIX}/{BIN_NAME}": b"ELF", f"{PREFIX}/{drive}escape/evil.so": b"PWN"},
        symlinks={f"{PREFIX}/libx.so": f"{drive}escape/evil.so"},
    )
    tar_dest = tmp_path / "engines" / "tar"
    extract_engine_archive(tarball, tar_dest)
    assert not list(cwd.rglob("evil.so")) and not (tar_dest / "libx.so").exists()
