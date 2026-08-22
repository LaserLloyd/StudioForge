"""Deep-link handling for HuggingFace's "Use this model" button.

The URL shape asserted here was read off live HuggingFace model pages, not
guessed:
``lmstudio://open_from_hf?model=lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from studioforge.config import Config
from studioforge.core import protocol
from studioforge.core.protocol import (
    LMSTUDIO_SCHEME,
    OWN_SCHEME,
    DeepLink,
    _delete_windows_scheme,
    _read_windows_command,
    _write_windows_scheme,
    gui_url_for,
    handler_command,
    parse_deep_link,
    register,
    status,
    unregister,
)
from studioforge.errors import BadRequestError


@pytest.fixture(autouse=True)
def _preserve_real_scheme_registrations():
    """Put the machine's real URL-scheme registrations back after every test.

    These tests exercise the genuine Windows registry rather than a fake, so on
    a machine where StudioForge is actually registered as the handler they were
    quietly destructive: each one ended in a bare ``unregister``, which deleted
    the developer's own ``studioforge://`` association. Snapshot both schemes
    and restore them verbatim -- including deleting one that was absent.
    """
    if os.name != "nt":
        yield
        return
    before = {scheme: _read_windows_command(scheme) for scheme in (OWN_SCHEME, LMSTUDIO_SCHEME)}
    try:
        yield
    finally:
        for scheme, command in before.items():
            if command:
                _write_windows_scheme(scheme, command)
            else:
                _delete_windows_scheme(scheme)


# The exact links HuggingFace serves for these two repos.
REAL_HF_LINKS = [
    (
        "lmstudio://open_from_hf?model=lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF",
        "lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF",
    ),
    (
        "lmstudio://open_from_hf?model=ggml-org/SmolVLM-256M-Instruct-GGUF",
        "ggml-org/SmolVLM-256M-Instruct-GGUF",
    ),
]

windows_only = pytest.mark.skipif(os.name != "nt", reason="Windows registry only")


def make_config(tmp_path: Path) -> Config:
    config = Config(data_dir=tmp_path / "data")
    config.ensure_dirs()
    return config


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("url", "expected_repo"), REAL_HF_LINKS)
def test_parses_real_huggingface_links(url: str, expected_repo: str) -> None:
    link = parse_deep_link(url)
    assert link.action == "open_from_hf"
    assert link.repo_id == expected_repo
    assert link.is_download


def test_parses_own_scheme_with_quant() -> None:
    link = parse_deep_link("studioforge://download?model=bartowski/Foo-GGUF&quant=Q4_K_M")
    assert link.repo_id == "bartowski/Foo-GGUF"
    assert link.quant == "Q4_K_M"
    assert link.is_download


def test_parses_repo_in_path_form() -> None:
    link = parse_deep_link("studioforge://download/owner/repo-GGUF")
    assert link.repo_id == "owner/repo-GGUF"


def test_url_encoded_repo_is_decoded() -> None:
    link = parse_deep_link("lmstudio://open_from_hf?model=owner%2Frepo-GGUF")
    assert link.repo_id == "owner/repo-GGUF"


def test_model_id_link_is_not_a_download() -> None:
    link = parse_deep_link("studioforge://models?id=publisher/repo/file")
    assert link.is_download is False
    assert link.model_id == "publisher/repo/file"


def test_unknown_scheme_rejected() -> None:
    with pytest.raises(BadRequestError, match="unsupported scheme"):
        parse_deep_link("ollama://open?model=foo/bar")


def test_non_link_rejected() -> None:
    with pytest.raises(BadRequestError, match="not a deep link"):
        parse_deep_link("just-some-text")


def test_malformed_repo_rejected() -> None:
    """'owner/repo' is the only shape HF uses; anything else is a bug or an attack."""
    with pytest.raises(BadRequestError, match="owner/repo"):
        parse_deep_link("lmstudio://open_from_hf?model=too/many/segments")


def test_empty_model_param_is_not_a_download() -> None:
    link = parse_deep_link("lmstudio://open_from_hf?model=")
    assert link.repo_id is None
    assert not link.is_download


# ---------------------------------------------------------------------------
# GUI URL construction
# ---------------------------------------------------------------------------


def test_gui_url_targets_the_download_tab(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    link = parse_deep_link(REAL_HF_LINKS[0][0])
    url = gui_url_for(link, config)
    assert url.startswith(f"http://127.0.0.1:{config.gui.port}/")
    assert "tab=download" in url
    assert "repo=lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF" in url


def test_gui_url_rewrites_wildcard_bind(tmp_path: Path) -> None:
    """0.0.0.0 is a bind address, not something a browser can open."""
    config = make_config(tmp_path)
    config.gui.host = "0.0.0.0"
    url = gui_url_for(parse_deep_link(REAL_HF_LINKS[0][0]), config)
    assert "0.0.0.0" not in url
    assert "127.0.0.1" in url


def test_gui_url_keeps_explicit_host(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.gui.host = "192.168.1.50"
    url = gui_url_for(parse_deep_link(REAL_HF_LINKS[0][0]), config)
    assert "192.168.1.50" in url


def test_gui_url_includes_quant(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    link = parse_deep_link("studioforge://download?model=a/b&quant=Q4_K_M")
    assert "quant=Q4_K_M" in gui_url_for(link, config)


def test_gui_url_plain_when_no_target(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    url = gui_url_for(DeepLink(action="open"), config)
    assert url.endswith("/")
    assert "tab=" not in url


def test_handler_command_is_runnable() -> None:
    command = handler_command()
    assert command
    assert command[-1] == "open"


# ---------------------------------------------------------------------------
# Registration (Windows)
# ---------------------------------------------------------------------------


@windows_only
def test_register_and_unregister_own_scheme(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    try:
        result = register(config)
        assert result["scheme"] == OWN_SCHEME
        current = status(config)
        assert current["studioforge"], "own scheme should be registered"
        assert "open" in str(current["studioforge"])
    finally:
        unregister(config)
    assert status(config)["studioforge"] is None


@windows_only
def test_lmstudio_takeover_backs_up_and_restores(tmp_path: Path) -> None:
    """Taking over LM Studio's scheme must be perfectly reversible.

    The user still has LM Studio installed and sharing the model library, so an
    irreversible hijack would break a working app.

    This test writes to the real HKCU, so it snapshots *both* schemes up front
    and puts them back verbatim -- including deleting one that was absent. It
    previously ended with a bare ``unregister``, which wiped a real
    ``studioforge://`` registration off the developer's machine, and it asserted
    the taken-over command differed from the one found, which is false on a
    machine where the takeover is already applied.
    """
    config = make_config(tmp_path)
    # Plant a known third-party handler so the test does not depend on what
    # this machine happens to have registered (which may already be ours).
    _write_windows_scheme(LMSTUDIO_SCHEME, "C:\Program Files\Some Other App\other.exe%1")
    before_lmstudio = status(config)["lmstudio"]
    try:
        result = register(config, takeover_lmstudio=True)
        assert result["lmstudio_taken_over"] is True
        assert result["lmstudio_previous"] == before_lmstudio

        taken = str(status(config)["lmstudio"])
        assert "studioforge" in taken.lower()

        backup = json.loads((config.data_dir / "protocol-backup.json").read_text("utf-8"))
        assert backup["lmstudio_command"] == before_lmstudio

        # The invariant that actually matters: a round trip returns exactly
        # what was there, whatever that was.
        unregister(config)
        assert status(config)["lmstudio"] == before_lmstudio
    finally:
        unregister(config)

    assert status(config)["lmstudio"] == before_lmstudio, "the handler found must be restored"


@windows_only
def test_repeated_takeover_does_not_clobber_the_backup(tmp_path: Path) -> None:
    """A second register must not record OUR command as the thing to restore."""
    config = make_config(tmp_path)
    _write_windows_scheme(LMSTUDIO_SCHEME, "C:\Program Files\Some Other App\other.exe%1")
    before = status(config)["lmstudio"]
    try:
        register(config, takeover_lmstudio=True)
        register(config, takeover_lmstudio=True)
        backup = json.loads((config.data_dir / "protocol-backup.json").read_text("utf-8"))
        assert backup["lmstudio_command"] == before
    finally:
        unregister(config)
    assert status(config)["lmstudio"] == before


@windows_only
def test_register_without_takeover_leaves_lmstudio_alone(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    before = status(config)["lmstudio"]
    try:
        register(config, takeover_lmstudio=False)
        assert status(config)["lmstudio"] == before
    finally:
        unregister(config)


# ---------------------------------------------------------------------------
# Registration (Linux)
# ---------------------------------------------------------------------------


def _stub_xdg_tools(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[list[str]]:
    """Make the Linux branch hermetic: record the xdg tools, never run them.

    ``xdg-mime default`` writes the developer's real ``~/.config/mimeapps.list``
    and would leave ``lmstudio://`` bound to a desktop file in a pytest tmp dir
    that ``_unregister_linux`` never unbinds. Pretend every tool exists so the
    branch is exercised on every host, and capture argv instead of spawning.
    ``XDG_CONFIG_HOME`` is redirected too, so a future code path that forgets
    the stub still cannot reach the real config.
    """
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    calls: list[list[str]] = []
    monkeypatch.setattr(protocol.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(protocol.subprocess, "run", lambda argv, **_: calls.append(list(argv)))
    return calls


@pytest.mark.skipif(os.name == "nt", reason="POSIX desktop files only")
def test_linux_desktop_file_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The desktop file is the contract; the xdg tools are recorded, not run."""
    calls = _stub_xdg_tools(monkeypatch, tmp_path)
    config = make_config(tmp_path)
    result = register(config, takeover_lmstudio=True)
    desktop = Path(str(result["desktop_file"]))
    assert desktop.is_file()
    text = desktop.read_text(encoding="utf-8")
    assert f"x-scheme-handler/{OWN_SCHEME}" in text
    assert f"x-scheme-handler/{LMSTUDIO_SCHEME}" in text
    assert "%u" in text
    assert any(argv[0].endswith("xdg-mime") for argv in calls), "the takeover must be applied"
    unregister(config)
    assert not desktop.exists()


def test_linux_takeover_binds_both_schemes_through_xdg_mime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin the exact ``xdg-mime`` calls, on every platform.

    ``_register_linux``/``_unregister_linux`` are pure Python plus two
    subprocess calls, so driving them directly lets the Windows rig (where the
    author works) and CI both check the Linux contract instead of leaving it to
    a test that only ever runs on a POSIX box.
    """
    calls = _stub_xdg_tools(monkeypatch, tmp_path)
    config = make_config(tmp_path)
    result = protocol._register_linux(config, takeover_lmstudio=True)
    desktop = Path(str(result["desktop_file"]))
    assert desktop.is_file()
    assert calls[0][0].endswith("update-desktop-database")
    xdg_mime = [argv[1:] for argv in calls if argv[0].endswith("xdg-mime")]
    assert xdg_mime == [
        ["default", desktop.name, f"x-scheme-handler/{OWN_SCHEME}"],
        ["default", desktop.name, f"x-scheme-handler/{LMSTUDIO_SCHEME}"],
    ]

    calls.clear()
    removed = protocol._unregister_linux(config)
    assert removed["removed"] == str(desktop)
    assert not desktop.exists()
    assert [argv[0] for argv in calls] == ["/usr/bin/update-desktop-database"]


def test_linux_register_without_tools_still_writes_the_desktop_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No xdg-utils on the box (minimal containers): the file is still the
    contract, and nothing is spawned."""
    calls = _stub_xdg_tools(monkeypatch, tmp_path)
    monkeypatch.setattr(protocol.shutil, "which", lambda name: None)
    config = make_config(tmp_path)
    result = protocol._register_linux(config, takeover_lmstudio=False)
    text = Path(str(result["desktop_file"])).read_text(encoding="utf-8")
    assert f"x-scheme-handler/{OWN_SCHEME}" in text
    assert f"x-scheme-handler/{LMSTUDIO_SCHEME}" not in text
    assert calls == []


@windows_only
def test_takeover_with_no_previous_handler_is_still_reversible(tmp_path: Path) -> None:
    """LM Studio not installed: nothing to back up, so nothing to restore.

    Leaving our command on `lmstudio://` after unregister would point
    HuggingFace's button at a handler that stops existing the moment
    StudioForge is uninstalled.
    """
    config = make_config(tmp_path)
    _delete_windows_scheme(LMSTUDIO_SCHEME)

    register(config, takeover_lmstudio=True)
    assert "studioforge" in str(status(config)["lmstudio"]).lower()

    unregister(config)
    assert status(config)["lmstudio"] is None


@windows_only
def test_a_lost_backup_does_not_record_our_own_command_as_the_original(
    tmp_path: Path,
) -> None:
    """Re-taking an already-taken scheme must not make US the restore target.

    If the data dir is wiped while the takeover is applied, the next register
    saw our own command as "previous" and stored it -- so a later "give it back
    to LM Studio" handed it back to StudioForge, permanently.
    """
    config = make_config(tmp_path)
    register(config, takeover_lmstudio=True)
    (config.data_dir / "protocol-backup.json").unlink(missing_ok=True)

    register(config, takeover_lmstudio=True)

    backup_file = config.data_dir / "protocol-backup.json"
    recorded = (
        json.loads(backup_file.read_text("utf-8")).get("lmstudio_command")
        if backup_file.is_file()
        else None
    )
    assert recorded != handler_command_string(), "we recorded ourselves as the original"


def handler_command_string() -> str:
    from studioforge.core.protocol import _quoted_command

    return _quoted_command()
