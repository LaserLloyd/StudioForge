"""Log files rotate, and a shared file on Windows can never make rotation lose data (D69 §15).

Before this, ``studioforge.log``, ``watchdog.log`` and ``tray-server.log`` grew
without bound (61 MB of watchdog.log, nearly all of it httpx INFO lines). The
stdlib ``RotatingFileHandler`` is not safe here as it stands: on Windows a
file another process holds cannot be renamed, and the stdlib handler shifts
its backups before it tries the live file, so every failed attempt pushed the
backups out by one and dropped the record that triggered it.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from studioforge import logfiles
from studioforge import logging as sf_logging
from studioforge.config import RESTART_REQUIRED_KEYS, Config
from studioforge.logfiles import (
    AppendFileHandler,
    SafeRotatingFileHandler,
    backup_pattern,
    backups_of,
    prune_backups,
    rotate_if_large,
)


def _record(message: str) -> logging.LogRecord:
    return logging.LogRecord("test", logging.INFO, __file__, 0, message, None, None)


def _handler(path: Path, *, max_bytes: int = 400, backups: int = 3, retry_s: float = 300.0) -> Any:
    handler = SafeRotatingFileHandler(
        path, max_bytes=max_bytes, backup_count=backups, retry_s=retry_s
    )
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    return handler


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


@pytest.fixture()
def root_logging() -> Iterator[None]:
    """configure_logging replaces the root handlers process-wide; put them back."""
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    installed = sf_logging._file_handler
    sf_logging._file_handler = None
    try:
        yield
    finally:
        current = sf_logging._file_handler
        if current is not None and current is not installed:
            current.close()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in saved[0]:
            root.addHandler(handler)
        root.setLevel(saved[1])
        sf_logging._file_handler = installed


# ---------------------------------------------------------------------------
# The handler
# ---------------------------------------------------------------------------


def test_the_owner_rotates_at_the_limit_and_keeps_the_newest_backups(tmp_path: Path) -> None:
    path = tmp_path / "studioforge.log"
    handler = _handler(path, max_bytes=400, backups=2)
    try:
        for i in range(200):
            handler.emit(_record(f"line {i:04d} " + "x" * 30))
    finally:
        handler.close()

    backups = backups_of(path)
    assert len(backups) == 2, "pruned to backup_count"
    assert all(backup_pattern(path).match(b.name) for b in backups)
    assert path.stat().st_size < 400 + 200
    # Nothing was lost between the newest backup and the live file: the last
    # record is live, and the newest backup ends right before the live file's
    # first record.
    live = [line for line in _lines(path) if line.startswith("INFO line")]
    assert live[-1].startswith("INFO line 0199")
    older = [line for line in _lines(backups[-1]) if line.startswith("INFO line")]
    assert int(older[-1].split()[2]) + 1 == int(live[0].split()[2])
    assert handler.rotations > 0 and handler.last_rotation_error is None
    # The new file says where the old one went.
    assert f"log rotated; the previous file is {backups[-1].name}" in path.read_text("utf-8")


def test_a_failed_rename_loses_nothing_and_touches_no_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What a Windows sharing violation does to the owner: nothing, visibly."""
    path = tmp_path / "studioforge.log"
    old_backup = tmp_path / "studioforge.20260101-000000.log"
    old_backup.write_text("old backup\n", encoding="utf-8")

    def refuse(*_a: object) -> None:
        raise PermissionError(32, "The process cannot access the file")

    monkeypatch.setattr(logfiles.os, "rename", refuse)
    handler = _handler(path, max_bytes=200, backups=1)
    try:
        for i in range(60):
            handler.emit(_record(f"record {i}"))
    finally:
        handler.close()

    text = path.read_text(encoding="utf-8")
    assert all(f"record {i}\n" in text for i in range(60)), "a record was dropped"
    assert old_backup.read_text(encoding="utf-8") == "old backup\n", "a backup was touched"
    assert backups_of(path) == [old_backup]
    # Said once, in the file itself, then left alone until retry_s passes.
    assert text.count("log rotation deferred (PermissionError") == 1
    assert "still appending to it, and retrying every 300 s" in text
    assert handler.last_rotation_error is not None


def test_the_deferred_rotation_is_retried_after_retry_s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "watchdog.log"
    real_rename = os.rename
    refusing = {"on": True}

    def rename(src: Any, dst: Any) -> None:
        if refusing["on"]:
            raise PermissionError(32, "held elsewhere")
        real_rename(src, dst)

    clock = {"now": 1000.0}
    monkeypatch.setattr(logfiles.os, "rename", rename)
    monkeypatch.setattr(logfiles.time, "monotonic", lambda: clock["now"])
    handler = _handler(path, max_bytes=100, backups=2, retry_s=60.0)
    try:
        for i in range(10):
            handler.emit(_record(f"early {i}"))
        assert backups_of(path) == []
        refusing["on"] = False  # the other process let go
        handler.emit(_record("still inside the back-off"))
        assert backups_of(path) == []
        clock["now"] += 61.0
        handler.emit(_record("after the back-off"))
        assert len(backups_of(path)) == 1
        assert handler.last_rotation_error is None
    finally:
        handler.close()


def test_a_long_deferral_is_noted_once_per_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A holder that keeps the file for days must not add a line every retry."""
    path = tmp_path / "studioforge.log"
    real_rename = os.rename
    refusing = {"on": True}

    def rename(src: Any, dst: Any) -> None:
        if refusing["on"]:
            raise PermissionError(32, "held elsewhere")
        real_rename(src, dst)

    clock = {"now": 0.0}
    monkeypatch.setattr(logfiles.os, "rename", rename)
    monkeypatch.setattr(logfiles.time, "monotonic", lambda: clock["now"])
    handler = _handler(path, max_bytes=100, backups=5, retry_s=60.0)
    try:
        for _ in range(10):  # ten back-off windows, all refused
            handler.emit(_record("x" * 40))
            clock["now"] += 61.0
        assert path.read_text(encoding="utf-8").count("log rotation deferred") == 1
        refusing["on"] = False
        handler.emit(_record("rotates now"))
        assert len(backups_of(path)) == 1
        refusing["on"] = True  # a new episode is noted again
        for _ in range(4):
            handler.emit(_record("y" * 40))
            clock["now"] += 61.0
        assert path.read_text(encoding="utf-8").count("log rotation deferred") == 1
    finally:
        handler.close()


@pytest.mark.skipif(os.name != "nt", reason="the sharing violation is a Windows behaviour")
def test_a_real_windows_sharing_violation_is_deferred_not_fatal(tmp_path: Path) -> None:
    """Another handle without FILE_SHARE_DELETE -- what the tray held for weeks."""
    path = tmp_path / "studioforge.log"
    handler = _handler(path, max_bytes=150, backups=2, retry_s=300.0)
    other = path.open("a", encoding="utf-8")  # noqa: SIM115 - held open on purpose
    try:
        for i in range(40):
            handler.emit(_record(f"shared {i}"))
    finally:
        other.close()
        handler.close()
    text = path.read_text(encoding="utf-8")
    assert all(f"shared {i}\n" in text for i in range(40))
    assert backups_of(path) == []
    assert "log rotation deferred" in text


def test_max_bytes_zero_never_rotates(tmp_path: Path) -> None:
    path = tmp_path / "studioforge.log"
    handler = _handler(path, max_bytes=0)
    try:
        for i in range(100):
            handler.emit(_record(f"line {i} " + "y" * 40))
    finally:
        handler.close()
    assert backups_of(path) == []
    assert len(_lines(path)) == 100


def test_the_owner_follows_a_rotation_done_by_someone_else(tmp_path: Path) -> None:
    """POSIX lets another process rename the file under our open handle; the
    owner then reopens the new file instead of rotating the fresh one."""
    if os.name == "nt":
        pytest.skip("a file held open cannot be renamed on Windows")
    path = tmp_path / "studioforge.log"
    handler = _handler(path, max_bytes=100, backups=3)
    try:
        handler.emit(_record("before " + "z" * 120))
        moved = tmp_path / "studioforge.20260101-000000.log"
        path.rename(moved)
        path.write_text("started by the other process\n", encoding="utf-8")
        handler.emit(_record("after"))
    finally:
        handler.close()
    assert backups_of(path) == [moved]
    assert _lines(path)[-1] == "INFO after"


def test_prune_never_touches_anything_but_this_logs_backups(tmp_path: Path) -> None:
    path = tmp_path / "studioforge.log"
    keep = [
        tmp_path / "studioforge.20260101-000000.log",
        tmp_path / "studioforge.20260102-000000.log",
        tmp_path / "studioforge.20260102-000000_1.log",
    ]
    bystanders = [
        path,
        tmp_path / "serve-stderr.log",
        tmp_path / "watchdog.20260101-000000.log",
        tmp_path / "studioforge.log.1",
        tmp_path / "studioforge.notes.log",
    ]
    for item in keep + bystanders:
        item.write_text("x\n", encoding="utf-8")

    removed = prune_backups(path, 2)

    assert removed == [keep[0]]
    assert backups_of(path) == keep[1:], "the _N suffix sorts after its plain stamp"
    assert all(item.exists() for item in bystanders)


def test_the_guest_handler_never_holds_the_file(tmp_path: Path) -> None:
    path = tmp_path / "studioforge.log"
    guest = AppendFileHandler(path)
    guest.setFormatter(logging.Formatter("%(message)s"))
    guest.emit(_record("from the tray"))
    # With the file closed after every record, the owner's rename can succeed
    # at any moment -- on Windows this is the whole point.
    moved = tmp_path / "studioforge.20260101-000000.log"
    path.rename(moved)
    guest.emit(_record("after the owner rotated"))
    guest.close()
    assert _lines(moved) == ["from the tray"]
    assert _lines(path) == ["after the owner rotated"]


# ---------------------------------------------------------------------------
# Files nobody may hold: the tray's server console
# ---------------------------------------------------------------------------


def test_rotate_if_large(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "tray-server.log"
    assert rotate_if_large(path, max_bytes=100, backup_count=2) is None  # missing
    path.write_text("small\n", encoding="utf-8")
    assert rotate_if_large(path, max_bytes=100, backup_count=2) is None
    path.write_text("b" * 150, encoding="utf-8")
    assert rotate_if_large(path, max_bytes=0, backup_count=2) is None  # never
    backup = rotate_if_large(path, max_bytes=100, backup_count=2)
    assert backup is not None and backup.read_text(encoding="utf-8") == "b" * 150
    assert not path.exists()

    path.write_text("c" * 150, encoding="utf-8")

    def refuse(*_a: object) -> None:
        raise PermissionError(32, "an old child still holds it")

    monkeypatch.setattr(logfiles.os, "rename", refuse)
    assert rotate_if_large(path, max_bytes=100, backup_count=2) is None
    assert path.read_text(encoding="utf-8") == "c" * 150


def test_the_tray_rotates_the_server_console_before_it_starts_a_server(tmp_path: Path) -> None:
    pytest.importorskip("pystray")
    from studioforge.tray.tray_app import TrayApp

    config = Config(data_dir=tmp_path / "data")
    config.logging.file_max_mb = 1
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    console = config.logs_dir / "tray-server.log"
    console.write_bytes(b"old console\n" * 100_000)  # > 1 MiB

    app = TrayApp(config, client=object(), create_icon=False)  # type: ignore[arg-type]
    handle = app._open_server_log()
    try:
        handle.write(b"new child\n")
    finally:
        handle.close()

    backups = backups_of(console)
    assert len(backups) == 1 and backups[0].stat().st_size > 1 << 20
    assert console.read_bytes() == b"new child\n"


# ---------------------------------------------------------------------------
# Wiring: who owns studioforge.log, the watchdog, the config keys
# ---------------------------------------------------------------------------


def _file_handlers() -> list[logging.Handler]:
    return [
        h
        for h in logging.getLogger().handlers
        if isinstance(h, (SafeRotatingFileHandler, AppendFileHandler, logging.FileHandler))
    ]


def test_configure_logging_owner_rotates_and_a_guest_appends(
    tmp_path: Path, root_logging: None
) -> None:
    sf_logging.configure_logging("INFO", log_dir=tmp_path, owner=True, max_bytes=5 << 20)
    (owner,) = _file_handlers()
    assert isinstance(owner, SafeRotatingFileHandler) and owner.maxBytes == 5 << 20

    sf_logging.configure_logging("INFO", log_dir=tmp_path, owner=False)
    (guest,) = _file_handlers()
    assert isinstance(guest, AppendFileHandler)
    # The replaced owner was closed, not just detached: it no longer pins the file.
    assert owner.stream is None
    # Not sticky: whoever builds the app (create_app, the server) owns the file,
    # whatever ran earlier in the process -- a test suite included.
    sf_logging.configure_logging("INFO", log_dir=tmp_path)
    (again,) = _file_handlers()
    assert isinstance(again, SafeRotatingFileHandler)


def test_only_serve_owns_the_server_log(tmp_path: Path, root_logging: None) -> None:
    import inspect

    from studioforge import __main__ as cli

    assert "_load(config_path, owns_log=True)" in inspect.getsource(cli.serve)
    for command in (cli.tray_cmd, cli.engine_cmd, cli.scan, cli.capabilities_cmd):
        source = inspect.getsource(command)
        assert "_load(config_path)" in source and "owns_log" not in source, command.__name__

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"data_dir: {(tmp_path / 'data').as_posix()}\n", encoding="utf-8")
    cli._load(config_path)
    (guest,) = _file_handlers()
    assert isinstance(guest, AppendFileHandler)
    cli._load(config_path, owns_log=True)
    (owner,) = _file_handlers()
    assert isinstance(owner, SafeRotatingFileHandler)
    assert owner.maxBytes == 20 << 20 and owner.backupCount == 5


def test_the_watchdog_rotates_its_log_and_silences_the_probe_lines(
    tmp_path: Path, root_logging: None
) -> None:
    from studioforge.watchdog.__main__ import configure_logging as watchdog_logging

    config = Config(data_dir=tmp_path / "data")
    config.logging.file_max_mb = 7
    config.logging.file_backups = 2
    path = watchdog_logging(config, "INFO")

    assert path == config.logs_dir / "watchdog.log"
    handler = next(
        h for h in logging.getLogger().handlers if isinstance(h, SafeRotatingFileHandler)
    )
    assert handler.maxBytes == 7 << 20 and handler.backupCount == 2
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING
    handler.close()


def test_the_watchdog_tail_reaches_into_the_newest_backup(tmp_path: Path) -> None:
    from studioforge.watchdog.server import tail_file

    live = tmp_path / "studioforge.log"
    live.write_text("log rotated\n", encoding="utf-8")
    (tmp_path / "studioforge.20260101-000000.log").write_text("ancient\n", encoding="utf-8")
    (tmp_path / "studioforge.20260102-000000.log").write_text("a\nb\nc\nd\n", encoding="utf-8")
    assert tail_file(live, 3) == ["c", "d", "log rotated"]
    assert tail_file(live, 1) == ["log rotated"]
    model = tmp_path / "some-model.log"
    model.write_text("only\n", encoding="utf-8")
    assert tail_file(model, 5) == ["only"]


def test_the_limits_are_config_keys_that_need_a_restart() -> None:
    config = Config(data_dir="/tmp/sf-rotation")
    assert config.logging.file_max_mb == 20 and config.logging.file_backups == 5
    assert {"logging.file_max_mb", "logging.file_backups"} <= RESTART_REQUIRED_KEYS
    with pytest.raises(ValueError):
        Config(data_dir="/tmp/sf-rotation", logging={"file_backups": 0})
    with pytest.raises(ValueError):
        Config(data_dir="/tmp/sf-rotation", logging={"file_max_mb": -1})


def test_backup_names_sort_in_the_order_they_were_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Within one second the counter keeps rising, even past a pruned name,
    and ``_10`` sorts after ``_9``."""
    path = tmp_path / "watchdog.log"
    frozen = time.struct_time((2026, 9, 23, 21, 5, 7, 2, 266, 0))
    monkeypatch.setattr(logfiles.time, "localtime", lambda *_a: frozen)
    made = []
    for _ in range(12):
        name = logfiles._backup_name(path)
        name.write_text("x", encoding="utf-8")
        made.append(name)
    assert made[0].name == "watchdog.20260923-210507.log"
    assert made[1].name == "watchdog.20260923-210507_1.log"
    assert backups_of(path) == made
    prune_backups(path, 3)
    assert backups_of(path) == made[-3:]
    assert logfiles._backup_name(path).name == "watchdog.20260923-210507_12.log"
