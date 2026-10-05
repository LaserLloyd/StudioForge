#!/usr/bin/env python3
"""Install or update a ui-theme/ folder from the theme repository.

    python ui-theme/update.py                    update this folder to the latest release
    python ui-theme/update.py --check            report only; exit 1 if an update exists
    python ui-theme/update.py --ref v1.2.0       pin a release tag (or a branch or commit)
    python update.py --dest static/ui-theme      first install: create the folder there

It is a command-line tool; nothing loads it in a page. Python 3.9 or newer,
standard library only. What it does:

  1. Learns the latest version from the repository's main branch (one small
     file) and downloads that release tag's archive from GitHub (if the tag is
     not published yet it says so and stops). --ref picks something else.
  2. Takes only the archive's ui-theme/ folder and checks every file against
     that folder's files.json (complete and unaltered download).
  3. Writes only the files the bundle owns: new and changed files are
     replaced, files an older bundle had and the new one dropped are removed,
     and anything else in the folder (an app's own file) is left alone.

It refuses to overwrite a bundle file that was edited by hand (one that matches
neither the installed nor the new release) unless you pass --force: the bundle
is replaced whole on every update, so keep app styles in the app. A run that
stops half way (a file held open) is safe to repeat: it resumes.

files.json proves the download is complete and unaltered; trust in where it
came from rests on HTTPS to github.com and on pinning a release tag. Hashes
are taken over LF line endings, so a checkout that turned the files into CRLF
still counts as unedited.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import ssl
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

REPO = "LaserLloyd/ThemeForge"
BUNDLE = "ui-theme"
MANIFEST = "files.json"
TIMEOUT = 60
NAME = "ThemeForge"
#: Names this bundle shipped under before; a folder whose VERSION carries one is
#: still recognised as this bundle (ThemeForge was released as unifyingTheme).
LEGACY_NAMES = ("unifyingTheme",)


class UpdateError(Exception):
    pass


def digest(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


# ---------------------------------------------------------------- network


def fetch(url: str) -> bytes:
    if not url.startswith("https://"):
        raise UpdateError(f"refusing a non-HTTPS URL: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "ui-theme-update"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise UpdateError(f"{url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(reason):
            raise UpdateError(
                f"{url}: this Python cannot verify HTTPS certificates. On macOS with a python.org "
                "build, run 'Install Certificates.command' from its Applications folder. Or download "
                f"https://github.com/{REPO} as a zip and pass --source <the zip's ui-theme folder>."
            ) from exc
        raise UpdateError(f"{url}: {reason}") from exc


def raw_url(repo: str, ref: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"


def remote_version(repo: str, ref: str) -> str:
    """The version string of the bundle at ref, from its VERSION file."""
    text = fetch(raw_url(repo, ref, f"{BUNDLE}/VERSION")).decode("utf-8").split()
    if len(text) < 2 or text[0] != NAME:
        raise UpdateError(f"{repo}@{ref} has no {BUNDLE}/VERSION in the expected format")
    return text[1]


def resolve_ref(repo: str, ref: str) -> str:
    """'latest' -> the release tag named in main's VERSION (v<version>)."""
    if ref != "latest":
        return ref
    return "v" + remote_version(repo, "main")


def download_bundle(repo: str, ref: str, latest: bool) -> tuple[dict, str]:
    try:
        data = fetch(f"https://codeload.github.com/{repo}/tar.gz/{ref}")
    except UpdateError as exc:
        if "HTTP 404" in str(exc) and latest:
            raise UpdateError(f"release {ref} is not tagged on GitHub yet; pass --ref main to install "
                              "the development tree, or --ref <an older tag>") from exc
        raise
    return bundle_from_archive(data), ref


# ---------------------------------------------------------------- reading bundles


def bundle_from_archive(data: bytes) -> dict:
    """Every regular file under <top>/ui-theme/ in a repository tar.gz, keyed by
    its path inside ui-theme/. Nothing is extracted to disk: members are read
    into memory, and links, devices and paths that would escape are refused."""
    out: dict = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            parts = PurePosixPath(member.name).parts
            if len(parts) < 3 or parts[1] != BUNDLE:
                continue
            rel = PurePosixPath(*parts[2:])
            if rel.is_absolute() or ".." in rel.parts or "\\" in member.name:
                raise UpdateError(f"unsafe path in archive: {member.name}")
            if member.isdir():
                continue
            if not member.isfile():
                raise UpdateError(f"unexpected link or device in archive: {member.name}")
            handle = tar.extractfile(member)
            out[rel.as_posix()] = handle.read() if handle else b""
    if not out:
        raise UpdateError(f"the archive has no {BUNDLE}/ folder")
    return out


def bundle_from_folder(folder: Path) -> dict:
    if not folder.is_dir():
        raise UpdateError(f"{folder} is not a folder")
    return {p.relative_to(folder).as_posix(): p.read_bytes()
            for p in sorted(folder.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def verify(files: dict) -> dict:
    if MANIFEST not in files:
        raise UpdateError(f"the new bundle has no {MANIFEST}")
    manifest = json.loads(files[MANIFEST].decode("utf-8"))
    listed = manifest.get("files")
    if not isinstance(listed, dict) or not listed:
        raise UpdateError(f"the new bundle's {MANIFEST} has no files table")
    problems = [f"missing {rel}" for rel in listed if rel not in files]
    problems += [f"corrupt {rel}" for rel, h in listed.items() if rel in files and digest(files[rel]) != h]
    problems += [f"unlisted {rel}" for rel in files if rel not in listed and rel != MANIFEST]
    if problems:
        raise UpdateError("the new bundle failed verification: " + "; ".join(problems))
    for rel in listed:
        if PurePosixPath(rel).is_absolute() or ".." in PurePosixPath(rel).parts:
            raise UpdateError(f"unsafe path in {MANIFEST}: {rel}")
    return manifest


def read_manifest(folder: Path):
    path = folder / MANIFEST
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise UpdateError(f"{path} is not valid JSON: {exc}") from exc
    return data if isinstance(data.get("files"), dict) else None


def describe(manifest) -> str:
    if not manifest:
        return "(nothing installed)"
    return f"{manifest.get('name', NAME)} {manifest.get('version', '?')} {manifest.get('hash', '')}".strip()


# ---------------------------------------------------------------- the target folder


TEMP_NAME = re.compile(r"^\.(.+)\.[A-Za-z0-9_]{8}$")


def leftover_temp(rel: str, owned: set) -> bool:
    """A temporary file write_file() left behind when a run was killed."""
    path = PurePosixPath(rel)
    m = TEMP_NAME.match(path.name)
    return bool(m) and (path.parent / m.group(1)).as_posix() in owned


def local_state(folder: Path, new_files: dict):
    """(installed manifest, edited bundle files, files the bundle does not own).

    A bundle file counts as edited only if it matches neither the installed
    manifest nor the new bundle, so a run that stopped half way (a file held
    open on Windows) resumes cleanly. A missing bundle file is simply written
    again. With no files.json (a copy made before files.json existed) the
    bundle owns exactly the paths the new bundle ships; a non-empty folder
    without a VERSION that names the bundle is refused."""
    if not folder.exists():
        return None, [], []
    have = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts)
    manifest = read_manifest(folder)
    if manifest is None:
        version_file = folder / "VERSION"
        names = tuple(n + " " for n in (NAME, *LEGACY_NAMES))
        is_bundle = version_file.is_file() and \
            version_file.read_text(encoding="utf-8", errors="replace").startswith(names)
        if have and not is_bundle:
            raise UpdateError(f"{folder} is not empty and is not a ui-theme folder (no VERSION/files.json); "
                              "pick an empty or new folder")
        owned = set(new_files)
        return None, [], [rel for rel in have if rel not in owned and rel != MANIFEST
                          and not leftover_temp(rel, owned)]
    owned = set(manifest["files"]) | set(new_files)
    edited = []
    for rel, expected in manifest["files"].items():
        path = folder / rel
        if not path.is_file():
            continue
        actual = digest(path.read_bytes())
        if actual != expected and (rel not in new_files or actual != digest(new_files[rel])):
            edited.append(rel)
    foreign = [rel for rel in have if rel not in owned and rel != MANIFEST and not leftover_temp(rel, owned)]
    return manifest, edited, foreign


def write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data.replace(b"\r\n", b"\n"))
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.2 * (attempt + 1))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def apply(folder: Path, files: dict, old_manifest, new_manifest) -> list:
    """Write every changed bundle file (files.json last), remove files the old
    bundle owned and the new one does not. Returns the changed paths."""
    old = old_manifest["files"] if old_manifest else {}
    owned = set(old) | set(new_manifest["files"])
    for path in folder.rglob(".*") if folder.exists() else []:
        rel = path.relative_to(folder).as_posix()
        if path.is_file() and leftover_temp(rel, owned):
            path.unlink()
    changed = []
    for rel in sorted(new_manifest["files"]):
        target = folder / rel
        if old.get(rel) == new_manifest["files"][rel] and target.is_file() \
                and digest(target.read_bytes()) == new_manifest["files"][rel]:
            continue
        write_file(target, files[rel])
        changed.append(rel)
    for rel in sorted(old):
        if rel not in new_manifest["files"] and (folder / rel).is_file():
            (folder / rel).unlink()
            changed.append(f"{rel} (removed)")
    write_file(folder / MANIFEST, files[MANIFEST])
    return changed


# ---------------------------------------------------------------- main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default="latest",
                        help="release tag, branch or commit (default: latest, the newest release tag)")
    parser.add_argument("--repo", default=REPO, help=f"the GitHub repository as user/name (default: {REPO})")
    parser.add_argument("--dest", help="the ui-theme folder to install into or update "
                                       "(default: the folder this script is in)")
    parser.add_argument("--source", help="a local repository .tar.gz, or a ui-theme folder, instead of GitHub")
    parser.add_argument("--check", action="store_true", help="report only; exit 1 when an update is available")
    parser.add_argument("--force", action="store_true",
                        help="overwrite hand-edited bundle files, and reinstall the same version")
    args = parser.parse_args(argv)

    script = globals().get("__file__")
    if not args.dest and not script:
        print("error: pass --dest <folder> (the bootstrap one-liner has no folder of its own)", file=sys.stderr)
        return 2
    target = Path(args.dest).resolve() if args.dest else Path(script).resolve().parent
    try:
        if args.check and not args.source:
            installed = read_manifest(target)
            if args.ref == "latest":
                version = remote_version(args.repo, "main")
                ref = "v" + version
            else:
                ref = args.ref
                version = remote_version(args.repo, ref)
            print(f"{target}\n  installed: {describe(installed)}\n  available: {NAME} {version} ({args.repo}@{ref})")
            current = installed.get("version") if installed else None
            print("  up to date" if current == version else "  update available")
            return 0 if current == version else 1

        if args.source:
            source = Path(args.source)
            files = bundle_from_folder(source) if source.is_dir() else bundle_from_archive(source.read_bytes())
            origin = str(source)
        else:
            ref = resolve_ref(args.repo, args.ref)
            files, used = download_bundle(args.repo, ref, latest=args.ref == "latest")
            origin = f"{args.repo}@{used}"
        new = verify(files)
        current, edited, foreign = local_state(target, files)
    except (UpdateError, OSError, ValueError, tarfile.TarError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"{target}")
    print(f"  installed: {describe(current)}")
    print(f"  available: {describe(new)}  ({origin})")
    if foreign:
        print("  not part of the bundle (left alone): " + ", ".join(foreign))
    if edited:
        print("  edited by hand: " + ", ".join(edited))
    same = current is not None and current.get("hash") == new.get("hash")
    if args.check:
        if edited:
            print("  the copy was edited by hand: updating needs --force (move the edits into the app first)")
            return 1
        print("  up to date" if same else "  update available")
        return 0 if same else 1
    if same and not edited and not args.force:
        print("  already up to date")
        return 0
    if edited and not args.force:
        print("error: refusing to overwrite the files above. The bundle is replaced whole on every "
              "update: move your changes into your app's own CSS, then rerun (or pass --force).",
              file=sys.stderr)
        return 1
    try:
        changed = apply(target, files, current, new)
    except OSError as exc:
        print(f"error: could not write {target}: {exc}. Close anything holding the files and rerun: "
              "the update resumes where it stopped.", file=sys.stderr)
        return 2
    print(f"  updated -> {describe(new)}")
    if changed:
        print("  changed: " + ", ".join(changed))
    print("  Pages pick it up on the next load if the folder is served with Cache-Control: no-cache "
          "(or ?v= from each file's content). See README.md in the folder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
