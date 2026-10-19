"""The CLI finds the INSTALLED data directory instead of silently using ``<checkout>/data`` (D81).

The 2026-10-19 defect: ``studioforge users passwd`` with no ``--config`` and no
``SF_DATA_DIR`` acted on an empty, unrelated install while the service ran from a data
directory named only by its Startup launcher. These tests build fake launchers/units in
scratch directories; the suite-wide ``SF_NO_INSTALL_DISCOVERY`` is switched off per test and
every location the locator reads (Startup folder, XDG config, checkout) is redirected, so a
real install on the developer's machine can never be reached.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from studioforge import install_locator as loc
from studioforge.__main__ import app as cli
from studioforge.config import Config
from studioforge.core import autostart
from studioforge.core.instance_lock import InstanceLock

pytestmark = pytest.mark.usefixtures("restore_logging")

runner = CliRunner()


@pytest.fixture(autouse=True)
def _discovery_on(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("SF_NO_INSTALL_DISCOVERY", "SF_DATA_DIR", "SF_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    # Nothing under the real home / AppData / checkout can be consulted.
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdgdata"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))
    monkeypatch.setattr(loc, "_checkout_data_dir", lambda: None)
    monkeypatch.setattr("studioforge.config._checkout_data_dir", lambda: None)


def make_install(root: Path) -> Path:
    data = root / "live data"  # a space, like the real ones
    config = Config(data_dir=data, models={"dir": root / "models"}, logging={"level": "ERROR"})
    config.ensure_dirs()
    return config.save(data / "config.yaml")


# --- the parsers (strings in, path out) ---------------------------------------


def test_shim_with_a_quoted_config_path() -> None:
    text = (
        "' Created by 'studioforge autostart enable'.\n"
        'Set shell = CreateObject("WScript.Shell")\n'
        'shell.CurrentDirectory = "C:\\Users\\me\\live data"\n'
        'shell.Run "C:\\py\\pythonw.exe -m studioforge tray --config ""C:\\Users\\me\\live data\\config.yaml""", 0, False\n'  # noqa: E501
    )
    assert loc.config_from_shim(text) == Path("C:\\Users\\me\\live data\\config.yaml")


def test_shim_with_a_plain_config_path_and_an_old_shim_without_one() -> None:
    plain = 'shell.Run "C:\\py\\pythonw.exe -m studioforge tray --config C:\\data\\config.yaml", 0, False\n'  # noqa: E501
    assert loc.config_from_shim(plain) == Path("C:\\data\\config.yaml")
    old = 'shell.CurrentDirectory = "C:\\old\\data"\nshell.Run "studioforge serve", 0, False\n'
    assert loc.config_from_shim(old) == Path("C:\\old\\data") / "config.yaml"
    assert loc.config_from_shim("nothing here") is None


def test_unit_with_sf_data_dir_or_config_flag() -> None:
    env_unit = '[Service]\nEnvironment="SF_DATA_DIR=/srv/sf data"\nExecStart=/opt/sf/bin/studioforge serve\n'  # noqa: E501
    assert loc.config_from_unit(env_unit) == Path("/srv/sf data/config.yaml")
    plain = (
        "[Service]\nEnvironment=SF_DATA_DIR=%h/studioforge/data\nExecStart=/x/studioforge serve\n"
    )
    assert loc.config_from_unit(plain, home=Path("/home/me")) == Path(
        "/home/me/studioforge/data/config.yaml"
    )
    flag = (
        "[Service]\nExecStart=/x/python -m studioforge serve --config '/d d/config.yaml' --open\n"
    )
    assert loc.config_from_unit(flag) == Path("/d d/config.yaml")
    assert loc.config_from_unit("[Service]\nExecStart=/x/studioforge serve\n") is None


def test_local_env_bat_expands_variables_and_ignores_the_unresolvable() -> None:
    text = '@echo off\nset "SF_DATA_DIR=C:\\Users\\%USERNAME%\\StudioForge-data"\n'
    assert loc.data_dir_from_local_env(text, {"USERNAME": "me"}) == Path(
        "C:\\Users\\me\\StudioForge-data"
    )
    assert loc.data_dir_from_local_env(text, {}) is None
    assert loc.data_dir_from_local_env('REM set "SF_DATA_DIR=C:\\x"\n', {}) is None
    assert loc.data_dir_from_local_env('set "SF_DATA_DIR=%~dp0data"\n', {}) is None
    two = 'set "SF_DATA_DIR=C:\\a"\nset "SF_DATA_DIR=C:\\b"\n'
    assert loc.data_dir_from_local_env(two, {}) == Path("C:\\b")


# --- order of precedence ------------------------------------------------------


def test_explicit_wins_then_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    other = tmp_path / "other" / "config.yaml"
    assert loc.locate(other) == loc.Located(other, "--config")
    monkeypatch.setenv("SF_DATA_DIR", str(tmp_path / "d"))
    assert loc.locate(None) == loc.Located(None, "SF_DATA_DIR")
    monkeypatch.setenv("SF_CONFIG", str(tmp_path / "c.yaml"))
    assert loc.locate(None) == loc.Located(None, "SF_CONFIG")


def test_discovery_can_be_turned_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    live = make_install(tmp_path)
    monkeypatch.setattr(loc, "_candidates", lambda: [loc.Candidate(live, "a launcher")])
    monkeypatch.setenv("SF_NO_INSTALL_DISCOVERY", "1")
    assert loc.locate(None).config_path is None


def test_the_first_candidate_is_used_when_nothing_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = make_install(tmp_path / "a")
    second = make_install(tmp_path / "b")
    monkeypatch.setattr(
        loc,
        "_candidates",
        lambda: [loc.Candidate(first, "launcher A"), loc.Candidate(second, "unit B")],
    )
    located = loc.locate(None)
    assert located.config_path == first and located.source == "launcher A"


def test_a_running_instance_beats_the_launcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher = make_install(tmp_path / "a")
    running = make_install(tmp_path / "b")
    monkeypatch.setattr(
        loc,
        "_candidates",
        lambda: [loc.Candidate(launcher, "launcher A"), loc.Candidate(running, "unit B")],
    )
    lock = InstanceLock(running.parent)
    assert lock.acquire()
    try:
        located = loc.locate(None)
    finally:
        lock.release()
    assert located.config_path == running
    assert f"the running server (pid {os.getpid()}" in located.source


def test_a_stale_lock_file_is_not_a_running_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher = make_install(tmp_path / "a")
    stale = make_install(tmp_path / "b")
    InstanceLock(stale.parent).acquire()  # writes the payload
    # (released below: the OS lock goes, the file and its pid stay -> holder() is "us"; use a dead pid)  # noqa: E501
    (stale.parent / ".instance.lock").write_text('{"pid": 999999999, "create_time": 1.0}')
    monkeypatch.setattr(
        loc,
        "_candidates",
        lambda: [loc.Candidate(launcher, "launcher A"), loc.Candidate(stale, "unit B")],
    )
    assert loc.locate(None).config_path == launcher


def test_with_no_launcher_the_old_default_applies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(loc, "_candidates", lambda: [])
    assert loc.locate(None) == loc.Located(None, "the default location")


# --- the real launchers, written by the real code -----------------------------


@pytest.mark.skipif(os.name != "nt", reason="the Startup-folder launcher is Windows-only")
def test_the_shim_autostart_writes_is_found_by_the_locator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    live = make_install(tmp_path)
    startup = tmp_path / "Startup"
    monkeypatch.setattr(autostart, "startup_dir", lambda: startup)
    monkeypatch.setattr(loc, "startup_dir", lambda: startup)
    config = Config(data_dir=live.parent)
    config.source_path = live
    autostart.enable(config, tray=True)
    found = loc._candidates()
    assert [c.config_path.resolve() for c in found] == [live.resolve()]
    assert "Startup launcher" in found[0].source and found[0].foreign_reason is None
    assert loc.locate(None).config_path == live


@pytest.mark.skipif(os.name == "nt", reason="systemd units are a POSIX thing")
def test_the_user_unit_autostart_writes_is_found_by_the_locator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    live = make_install(tmp_path)
    unit_dir = tmp_path / "xdg" / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    (unit_dir / autostart.LINUX_UNIT).write_text(
        f'[Service]\nEnvironment="SF_DATA_DIR={live.parent}"\nExecStart=/x/studioforge serve\n'
    )
    assert [c.config_path.resolve() for c in loc._candidates()] == [live.resolve()]


# --- end to end through the CLI ------------------------------------------------


def test_a_plain_users_command_acts_on_the_discovered_install_not_the_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SF_GPU_PROBE", "null")
    live = make_install(tmp_path / "live")
    monkeypatch.setattr(loc, "_candidates", lambda: [loc.Candidate(live, "the installed launcher")])
    add = runner.invoke(
        cli, ["users", "add", "alice", "--password-stdin"], input="a-long-enough-pw\n"
    )
    assert add.exit_code == 0, add.output
    assert f"acting on {live.parent}" in add.stderr
    assert "from the installed launcher" in add.stderr
    assert (live.parent / "registry.sqlite3").is_file()
    # ... and nothing was created in the default location.
    default = Config(data_dir=tmp_path / "nowhere")
    assert not default.data_dir.exists()
    listed = runner.invoke(cli, ["users", "list"])
    assert "alice" in listed.stdout


def test_config_path_follows_the_discovered_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    live = make_install(tmp_path)
    monkeypatch.setattr(loc, "_candidates", lambda: [loc.Candidate(live, "launcher")])
    result = runner.invoke(cli, ["config", "--path"])
    assert result.stdout.strip() == str(live)


# --- a launcher is adopted only when it is ours -------------------------------


def test_interpreter_parsers() -> None:
    plain = 'shell.Run "C:\\v\\Scripts\\pythonw.exe -m studioforge tray --config C:\\d\\c.yaml", 0, False\n'  # noqa: E501
    assert loc.interpreter_from_shim(plain) == Path("C:\\v\\Scripts\\pythonw.exe")
    quoted = 'shell.Run ""C:\\my py\\pythonw.exe" -m studioforge tray", 0, False\n'
    assert loc.interpreter_from_shim(quoted) is None or "pythonw" in str(
        loc.interpreter_from_shim(quoted)
    )
    unit = "[Service]\nExecStart=%h/sf/.venv/bin/python -m studioforge serve\n"
    assert loc.interpreter_from_unit(unit, home=Path("/h")) == Path("/h/sf/.venv/bin/python")
    assert loc.interpreter_from_unit("[Service]\n") is None


def test_only_this_environments_interpreter_belongs() -> None:
    import sys

    assert loc.belongs_to_this_environment(Path(sys.executable))
    assert loc.belongs_to_this_environment(Path(sys.prefix) / "bin" / "studioforge")
    assert not loc.belongs_to_this_environment(Path("/somewhere/else/python"))
    assert not loc.belongs_to_this_environment(None)


@pytest.mark.skipif(os.name != "nt", reason="the Startup-folder launcher is Windows-only")
def test_a_shim_for_another_python_environment_is_foreign(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    live = make_install(tmp_path)
    startup = tmp_path / "Startup"
    startup.mkdir()
    other_python = tmp_path / "other-venv" / "Scripts" / "pythonw.exe"
    shim = f'shell.Run "{other_python} -m studioforge tray --config ""{live}""", 0, False\n'
    (startup / autostart.WINDOWS_SHIM).write_text(shim, encoding="utf-8")
    monkeypatch.setattr(loc, "startup_dir", lambda: startup)
    (candidate,) = loc._candidates()
    assert candidate.foreign_reason and "different Python environment" in candidate.foreign_reason
    located = loc.locate(None)
    assert located.config_path is None
    assert [f.config_path for f in located.foreign] == [live]


def test_foreign_installs_are_reported_not_adopted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    other = make_install(tmp_path / "other")
    mine = make_install(tmp_path / "mine")
    foreign = loc.Candidate(other, "a unit", "owned by the user 'studioforge'")
    monkeypatch.setattr(loc, "_candidates", lambda: [foreign, loc.Candidate(mine, "local-env")])
    located = loc.locate(None)
    assert located.config_path == mine and len(located.foreign) == 1
    monkeypatch.setattr(loc, "_candidates", lambda: [foreign])
    only = loc.locate(None)
    assert only.config_path is None and only.foreign[0].config_path == other


@pytest.mark.skipif(os.name == "nt", reason="uids are a POSIX thing")
def test_a_data_dir_owned_by_another_uid_is_foreign(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(os, "getuid", lambda: tmp_path.stat().st_uid + 4242)
    reason = loc._foreign_owner_reason(tmp_path)
    assert reason and "sudo -u" in reason and f"SF_DATA_DIR={tmp_path}" in reason


def test_users_commands_refuse_a_foreign_only_install_and_say_what_to_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SF_GPU_PROBE", "null")
    other = make_install(tmp_path / "other")
    monkeypatch.setattr(
        loc, "_candidates", lambda: [loc.Candidate(other, "a unit", "different environment")]
    )
    result = runner.invoke(cli, ["users", "list"])
    assert result.exit_code == 2
    assert str(other.parent) in result.stderr and "--config" in result.stderr
    assert not (other.parent / "registry.sqlite3").exists(), "nothing was opened"
    explicit = runner.invoke(cli, ["users", "list", "--config", str(other)])
    assert explicit.exit_code == 0 and f"acting on {other.parent}" in explicit.stderr
    assert runner.invoke(cli, ["config", "--path"]).exit_code == 2


def test_only_the_account_commands_discover_an_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`server`, `setup` and the model commands resolve the directory as they always did."""
    from studioforge import cli_commands

    live = make_install(tmp_path / "live")
    calls: list[int] = []

    def fake_candidates() -> list[loc.Candidate]:
        calls.append(1)
        return [loc.Candidate(live, "x")]

    monkeypatch.setattr(loc, "_candidates", fake_candidates)
    monkeypatch.setattr("studioforge.config._checkout_data_dir", lambda: tmp_path / "plain")
    config = cli_commands._config(None)  # not an account command
    assert calls == [] and config.data_dir != live.parent


def test_local_env_variants_the_launchers_accept() -> None:
    cases = {
        'set "SF_DATA_DIR=C:\\d"  rem my data\n': "C:\\d",
        'set "SF_DATA_DIR=C:\\d" & rem my data\n': "C:\\d",
        '@set "SF_DATA_DIR=C:\\d"\n': "C:\\d",
        'if not defined SF_DATA_DIR set "SF_DATA_DIR=C:\\d"\n': "C:\\d",
        "set SF_DATA_DIR=C:\\d & echo hi\n": "C:\\d",
        "set SF_DATA_DIR=C:\\d\r\nset OTHER=1\r\n": "C:\\d",
    }
    for text, expected in cases.items():
        assert loc.data_dir_from_local_env(text, {}) == Path(expected), text
    for text in (
        "rem set SF_DATA_DIR=C:\\x\n",
        ":: set SF_DATA_DIR=C:\\x\n",
        "echo set SF_DATA_DIR=C:\\x\n",
    ):
        assert loc.data_dir_from_local_env(text, {}) is None, text
