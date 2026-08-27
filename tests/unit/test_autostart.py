"""Login-autostart registration.

The Windows path writes a VBScript shim into the Startup folder; the quoting in
that file is the part most likely to break silently, because a bad quote yields
a shim that runs at every login and does nothing visible.
"""

from __future__ import annotations

import codecs
import os
from pathlib import Path

import pytest

from studioforge.config import Config
from studioforge.core import autostart

windows_only = pytest.mark.skipif(os.name != "nt", reason="Windows Startup folder")
posix_only = pytest.mark.skipif(os.name == "nt", reason="systemd user units")


def read_shim(path: Path) -> str:
    """Decode the shim the way wscript.exe will: UTF-16 LE behind its BOM."""
    raw = path.read_bytes()
    assert raw.startswith(codecs.BOM_UTF16_LE)
    return raw[len(codecs.BOM_UTF16_LE) :].decode("utf-16-le")


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config(data_dir=tmp_path / "data")
    cfg.ensure_dirs()
    cfg.save(cfg.data_dir / "config.yaml")
    cfg.source_path = cfg.data_dir / "config.yaml"
    return cfg


@pytest.fixture
def fake_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the Startup folder so tests never touch the real one."""
    target = tmp_path / "Startup"
    monkeypatch.setattr(autostart, "startup_dir", lambda: target)
    return target


def test_launch_command_includes_the_config_path(config: Config) -> None:
    """A login shell may not carry SF_DATA_DIR, so the path must be explicit."""
    argv = autostart.launch_command(config)
    assert "serve" in argv
    assert str(config.config_path) in argv
    assert "--open" not in argv


def test_launch_command_can_open_the_gui(config: Config) -> None:
    assert "--open" in autostart.launch_command(config, open_gui=True)


@windows_only
def test_enable_writes_a_hidden_launch_shim(config: Config, fake_startup: Path) -> None:
    result = autostart.enable(config)
    assert result.enabled
    shim = fake_startup / "StudioForge.vbs"
    assert shim.is_file()

    text = read_shim(shim)
    # 0 = hidden window, False = do not wait: a login shim must not flash a
    # console or block the logon sequence.
    assert ", 0, False" in text
    assert "WScript.Shell" in text
    assert "serve" in text
    assert str(config.config_path) in text


@windows_only
def test_shim_quotes_paths_containing_spaces(tmp_path: Path, fake_startup: Path) -> None:
    """A path with a space must be quoted or the shim silently fails at logon.

    The spaced path is built HERE, deliberately: an earlier version relied on
    the checkout (or pytest's tmp dir) happening to live under a directory
    with a space in its name -- true on the author's box, false for a stranger
    cloning to ``~/dev/studioforge``, whose very first ``pytest`` went red
    (WP17 F1).
    """
    spaced = tmp_path / "Studio Forge data"
    cfg = Config(data_dir=spaced)
    cfg.ensure_dirs()
    cfg.save(cfg.data_dir / "config.yaml")
    cfg.source_path = cfg.data_dir / "config.yaml"
    assert " " in str(cfg.config_path)

    autostart.enable(cfg)
    text = read_shim(fake_startup / "StudioForge.vbs")
    run_line = next(line for line in text.splitlines() if line.startswith("shell.Run"))
    # Inside a VBScript string literal a quote is escaped by doubling it.
    assert '""' in run_line, run_line
    spaced_parts = [p for p in autostart.launch_command(cfg) if " " in p]
    assert spaced_parts, "the config path itself contains a space"
    for part in spaced_parts:
        assert f'""{part}""' in run_line, f"{part!r} was not quoted"


@windows_only
def test_enable_is_idempotent_and_disable_removes(config: Config, fake_startup: Path) -> None:
    autostart.enable(config)
    autostart.enable(config)
    assert len(list(fake_startup.glob("*.vbs"))) == 1

    assert autostart.status(config).enabled is True
    result = autostart.disable(config)
    assert result.enabled is False
    assert not (fake_startup / "StudioForge.vbs").exists()
    assert autostart.status(config).enabled is False


@windows_only
def test_disable_when_not_enabled_is_not_an_error(config: Config, fake_startup: Path) -> None:
    result = autostart.disable(config)
    assert result.enabled is False
    assert "was not enabled" in result.detail


@windows_only
def test_status_reports_the_real_file_not_our_belief(config: Config, fake_startup: Path) -> None:
    autostart.enable(config)
    (fake_startup / "StudioForge.vbs").unlink()  # user deleted it by hand
    assert autostart.status(config).enabled is False


@windows_only
def test_shim_is_utf16le_with_bom_never_utf8(config: Config, fake_startup: Path) -> None:
    """The VBScript engine parses ANSI or BOM-marked UTF-16 and nothing else.

    A UTF-8 BOM -- written 'on purpose' once, to survive an accented data
    dir -- arrived as three garbage characters and every login died on
    'Invalid character / 800A0408 / Line 1 Char 1'. Lived on this rig from
    2026-08-22 to 2026-08-26.
    """
    autostart.enable(config, tray=True)
    raw = (fake_startup / autostart.WINDOWS_SHIM).read_bytes()
    assert raw.startswith(codecs.BOM_UTF16_LE)
    assert not raw.startswith(codecs.BOM_UTF8)
    body = raw[len(codecs.BOM_UTF16_LE) :].decode("utf-16-le")
    assert "\r\n" in body  # explicit CRLF: write_bytes translates nothing
    assert body.startswith("'")  # the first parsed character is a comment


@windows_only
def test_status_still_reads_an_old_utf8_bom_shim(config: Config, fake_startup: Path) -> None:
    """Installs that pre-date the fix hold a UTF-8-BOM shim; status() must
    still classify it (as broken-at-login as it is) instead of mis-reading."""
    autostart.enable(config, tray=True)
    shim = fake_startup / autostart.WINDOWS_SHIM
    text = read_shim(shim)
    shim.write_text(text, encoding="utf-8-sig")
    assert "tray" in autostart.status(config).detail


@posix_only
def test_linux_unit_is_written(config: Config, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(autostart.shutil, "which", lambda name: None)
    result = autostart.enable(config)
    unit = tmp_path / "config" / "systemd" / "user" / "studioforge.service"
    assert unit.is_file()
    text = unit.read_text(encoding="utf-8")
    assert "ExecStart=" in text
    assert f"SF_DATA_DIR={config.data_dir}" in text
    assert "TimeoutStopSec=120" in text  # long enough to stop every child
    assert "systemctl" in result.detail


def test_status_describe_is_human_readable(config: Config) -> None:
    described = autostart.status(config).describe()
    assert "autostart" in described
    assert "enabled" in described


class TestTrayAutostart:
    """Autostart can launch the tray instead of the bare server.

    Regression cover for "the server starts at login but there is no tray
    icon": ``autostart enable`` could only ever write a ``serve`` shim, so the
    icon was never part of logging in.
    """

    def test_tray_launch_command_uses_the_tray_entry_point(self, config: Config) -> None:
        argv = autostart.launch_command(config, tray=True)

        assert "tray" in argv
        assert "serve" not in argv
        # The config path still travels explicitly: a login shell carries no
        # SF_DATA_DIR.
        assert str(config.config_path) in argv

    def test_tray_launch_never_uses_the_console_script(
        self, config: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """studioforge.exe is a console-subsystem launcher, and which() finds
        whichever venv's copy is on PATH -- possibly a different checkout's.
        A shim baked with it runs the tray against a hidden (or, one Settings
        toggle later, visible) console. The tray shim is always
        pythonw -m studioforge, pinned to this install."""
        monkeypatch.setattr(
            autostart.shutil, "which", lambda name: r"C:\other\venv\Scripts\studioforge.exe"
        )
        argv = autostart.launch_command(config, tray=True)
        assert "studioforge.exe" not in argv[0].lower()
        assert argv[1:3] == ["-m", "studioforge"]

    def test_open_gui_is_ignored_for_the_tray(self, config: Config) -> None:
        # The tray's own menu opens the panel; --open belongs to `serve`.
        argv = autostart.launch_command(config, open_gui=True, tray=True)
        assert "--open" not in argv

    def test_server_mode_is_unchanged(self, config: Config) -> None:
        argv = autostart.launch_command(config, open_gui=True)
        assert "serve" in argv
        assert "--open" in argv
        assert "tray" not in argv

    @pytest.mark.skipif(os.name != "nt", reason="Windows Startup folder")
    def test_enable_writes_a_tray_shim_and_status_says_so(
        self, config: Config, fake_startup: Path
    ) -> None:
        result = autostart.enable(config, tray=True)
        assert result.enabled
        body = read_shim(fake_startup / autostart.WINDOWS_SHIM)
        assert "tray" in body
        assert " serve " not in body

        # Status must distinguish the two modes, or it cannot answer the
        # question people actually ask it.
        assert "tray" in autostart.status(config).detail

        autostart.enable(config, tray=False)
        assert "no tray icon" in autostart.status(config).detail

    @pytest.mark.skipif(os.name == "nt", reason="non-Windows behaviour")
    def test_tray_is_refused_on_linux(self, config: Config) -> None:
        # The Linux mechanism is a headless systemd user unit; a tray started
        # from it would fail at login with no display.
        with pytest.raises(autostart.AutostartError, match="Windows-only"):
            autostart.enable(config, tray=True)
