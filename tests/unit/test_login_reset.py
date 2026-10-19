"""``<data>/RESET_LOGIN``: lockout recovery for someone with filesystem access (D81).

What must hold: the sentinel resets or creates the named (default: owner) account; it is
read once, at ``serve`` start, and deleted before anything changes so it cannot replay;
a password never reaches a log line or the console; and nothing reachable over the
network can create, read or trigger it.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import studioforge.__main__ as entry
from studioforge import login_reset
from studioforge.api.app import build_state, create_app
from studioforge.config import Config
from studioforge.core.accounts import AccountStore
from studioforge.db import Database
from studioforge.login_reset import PASSWORD_FILE_NAME, SENTINEL_NAME, apply_sentinel, parse

OLD = "the-old-password-1"
NEW = "a-brand-new-password"
SRC = Path(__file__).resolve().parents[2] / "src" / "studioforge"


@pytest.fixture
def data(tmp_path: Path) -> Path:
    path = tmp_path / "data"
    path.mkdir()
    return path


@pytest.fixture
def store(data: Path) -> Iterator[AccountStore]:
    db = Database(data / "registry.sqlite3")
    db.migrate()
    yield AccountStore(db, data)
    db.close()


def sentinel(data: Path, text: str | bytes) -> Path:
    path = data / SENTINEL_NAME
    if isinstance(text, str):
        text = text.encode("utf-8")
    path.write_bytes(text)
    return path


# --- parsing ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b"", ("", "")),
        (b"\r\n", ("", "")),
        (b"alice", ("alice", "")),
        (b"alice\n", ("alice", "")),
        (b"alice:", ("alice", "")),
        (b"alice:pw-1234567890", ("alice", "pw-1234567890")),
        (b"alice:pw-1234567890 \r\n", ("alice", "pw-1234567890")),  # `echo a:b > f` on cmd.exe
        (b"\xef\xbb\xbfalice:pw-1234567890", ("alice", "pw-1234567890")),  # Notepad's BOM
        (b"alice:a:b:c-1234567", ("alice", "a:b:c-1234567")),  # the first colon splits
        (b":only-a-password1", ("", "only-a-password1")),
        (b"alice:first-line-pw\nignored second line", ("alice", "first-line-pw")),
    ],
)
def test_parse(raw: bytes, expected: tuple[str, str]) -> None:
    assert parse(raw) == expected


# --- applying --------------------------------------------------------------


def test_no_sentinel_is_a_no_op(data: Path, store: AccountStore) -> None:
    assert apply_sentinel(data, store) is None


def test_a_password_in_the_file_resets_the_named_account(data: Path, store: AccountStore) -> None:
    store.create_account("alice", OLD)
    path = sentinel(data, f"alice:{NEW}")
    result = apply_sentinel(data, store)
    assert result is not None and result.applied and not result.created
    assert result.username == "alice" and result.password_file is None
    assert not path.exists(), "the sentinel is consumed"
    assert store.verify_password("alice", NEW) is not None
    assert store.verify_password("alice", OLD) is None


def test_an_empty_sentinel_resets_the_owner_with_a_generated_password(
    data: Path, store: AccountStore
) -> None:
    store.create_account("alice", OLD)
    store.create_account("bob", OLD)
    sentinel(data, "")
    result = apply_sentinel(data, store)
    assert result is not None and result.applied and result.username == "alice"
    assert result.password_file == data / PASSWORD_FILE_NAME
    generated = (data / PASSWORD_FILE_NAME).read_text(encoding="utf-8").strip()
    assert len(generated) >= 20
    assert store.verify_password("alice", generated) is not None
    assert store.verify_password("alice", OLD) is None
    assert store.verify_password("bob", OLD) is not None, "other accounts are untouched"


def test_a_bare_name_generates_a_password_for_that_account(data: Path, store: AccountStore) -> None:
    store.create_account("alice", OLD)
    store.create_account("bob", OLD)
    sentinel(data, "bob\n")
    result = apply_sentinel(data, store)
    assert result is not None and result.username == "bob"
    generated = (data / PASSWORD_FILE_NAME).read_text(encoding="utf-8").strip()
    assert store.verify_password("bob", generated) is not None
    assert store.verify_password("alice", OLD) is not None


def test_a_missing_account_is_created_as_the_owner_when_there_is_none(
    data: Path, store: AccountStore
) -> None:
    sentinel(data, f"carol:{NEW}")
    result = apply_sentinel(data, store)
    assert result is not None and result.created
    account = store.get_account("carol")
    assert account is not None and account.role == "owner"
    assert store.verify_password("carol", NEW) is not None


def test_an_empty_sentinel_on_a_fresh_install_creates_owner(
    data: Path, store: AccountStore
) -> None:
    sentinel(data, "")
    result = apply_sentinel(data, store)
    assert result is not None and result.created and result.username == "owner"
    generated = (data / PASSWORD_FILE_NAME).read_text(encoding="utf-8").strip()
    assert store.verify_password("owner", generated) is not None


def test_a_reset_re_enables_a_disabled_account(data: Path, store: AccountStore) -> None:
    store.create_account("alice", OLD)
    store.db.execute_write("UPDATE accounts SET disabled = 1 WHERE username = 'alice'")
    assert store.verify_password("alice", OLD) is None
    sentinel(data, f"alice:{NEW}")
    apply_sentinel(data, store)
    assert store.verify_password("alice", NEW) is not None


def test_the_generated_password_file_is_not_left_world_readable(
    data: Path, store: AccountStore
) -> None:
    import os
    import stat

    sentinel(data, "")
    apply_sentinel(data, store)
    if os.name != "nt":
        mode = stat.S_IMODE((data / PASSWORD_FILE_NAME).stat().st_mode)
        assert mode & 0o077 == 0


# --- consumed, never replayed ----------------------------------------------


def test_the_sentinel_is_consumed_and_a_second_start_does_nothing(
    data: Path, store: AccountStore
) -> None:
    store.create_account("alice", OLD)
    sentinel(data, f"alice:{NEW}")
    assert apply_sentinel(data, store) is not None
    store.set_password("alice", OLD)  # the owner changes it back afterwards
    assert apply_sentinel(data, store) is None
    assert store.verify_password("alice", OLD) is not None


def test_a_sentinel_that_cannot_be_deleted_is_not_applied(
    data: Path, store: AccountStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.create_account("alice", OLD)
    path = sentinel(data, f"alice:{NEW}")
    real_unlink = Path.unlink

    def deny(self: Path, *a: Any, **k: Any) -> None:
        if self.name == SENTINEL_NAME:
            raise PermissionError(13, "Permission denied")
        real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", deny)
    result = apply_sentinel(data, store)
    assert result is not None and not result.applied
    assert "so it was not applied" in (result.error or "")
    assert path.exists()
    assert store.verify_password("alice", OLD) is not None
    assert store.verify_password("alice", NEW) is None


def test_a_rejected_password_still_consumes_the_file(data: Path, store: AccountStore) -> None:
    store.create_account("alice", OLD)
    path = sentinel(data, "alice:short")
    result = apply_sentinel(data, store)
    assert result is not None and not result.applied
    assert "10 characters" in (result.error or "")
    assert not path.exists()
    assert store.verify_password("alice", OLD) is not None


def test_a_sentinel_naming_an_invalid_username_is_rejected(data: Path, store: AccountStore) -> None:
    path = sentinel(data, f"bad/name:{NEW}")
    result = apply_sentinel(data, store)
    assert result is not None and not result.applied and not path.exists()
    assert store.list_accounts() == []


def test_a_directory_or_oversize_file_named_reset_login_is_left_alone_or_refused(
    data: Path, store: AccountStore
) -> None:
    store.create_account("alice", OLD)
    (data / SENTINEL_NAME).mkdir()
    result = apply_sentinel(data, store)
    assert result is not None and not result.applied and (data / SENTINEL_NAME).is_dir()
    (data / SENTINEL_NAME).rmdir()
    big = sentinel(data, b"alice:" + b"x" * 5000)
    result = apply_sentinel(data, store)
    assert result is not None and not result.applied and not big.exists()
    assert store.verify_password("alice", OLD) is not None


# --- no secret leaves through a log line or the console ---------------------


def test_no_password_reaches_the_result_text_or_the_output(
    data: Path, store: AccountStore, capsys: pytest.CaptureFixture[str]
) -> None:
    store.create_account("alice", OLD)
    sentinel(data, f"alice:{NEW}")
    explicit = apply_sentinel(data, store)
    sentinel(data, "")
    generated_run = apply_sentinel(data, store)
    generated = (data / PASSWORD_FILE_NAME).read_text(encoding="utf-8").strip()
    shown = capsys.readouterr()
    assert explicit is not None and generated_run is not None
    for text in (explicit.describe(), generated_run.describe(), shown.out, shown.err):
        assert NEW not in text and generated not in text and OLD not in text


# --- serve: once, at start, before the bind rule ----------------------------


def make_config(tmp_path: Path) -> Config:
    config = Config(
        data_dir=tmp_path / "data",
        models={"dir": tmp_path / "models"},
        server={"port": 11234},
        gui={"port": 18080},
        watchdog={"port": 11235},
        logging={"level": "ERROR"},
    )
    config.ensure_dirs()
    return config


def test_serve_applies_the_sentinel_after_the_port_check_and_before_the_bind_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from typer.testing import CliRunner

    monkeypatch.setenv("SF_DATA_DIR", str(tmp_path / "data"))
    config = make_config(tmp_path)
    config_path = config.save(tmp_path / "data" / "config.yaml")
    sentinel(config.data_dir, f"alice:{NEW}")
    order: list[str] = []
    seen: dict[str, bool] = {}

    monkeypatch.setattr(entry, "_preflight_ports", lambda c: order.append("ports"))

    def enforce(c: Config) -> None:
        order.append("bind-rule")
        db = Database(c.db_path)
        try:
            seen["owner"] = AccountStore(db, c.data_dir).verify_password("alice", NEW) is not None
        finally:
            db.close()
        seen["sentinel_gone"] = not (c.data_dir / SENTINEL_NAME).exists()

    async def fake_serve(c: Config, *, open_gui: bool) -> int:
        order.append("serve")
        return 0

    monkeypatch.setattr(entry, "_enforce_security_start", enforce)
    monkeypatch.setattr(entry, "_serve", fake_serve)
    result = CliRunner().invoke(
        entry.app, ["serve", "--config", str(config_path), "--no-gui", "--no-watchdog"]
    )
    assert result.exit_code == 0, result.output
    assert order == ["ports", "bind-rule", "serve"]
    assert seen == {"owner": True, "sentinel_gone": True}
    assert "RESET_LOGIN consumed: created the account 'alice'" in result.output
    assert NEW not in result.output


def test_a_failed_reset_does_not_stop_the_server_from_starting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    monkeypatch.setenv("SF_DATA_DIR", str(tmp_path / "data"))
    config = make_config(tmp_path)
    config_path = config.save(tmp_path / "data" / "config.yaml")
    sentinel(config.data_dir, "alice:short")
    monkeypatch.setattr(entry, "_preflight_ports", lambda c: None)
    monkeypatch.setattr(entry, "_enforce_security_start", lambda c: None)

    async def fake_serve(c: Config, *, open_gui: bool) -> int:
        return 0

    monkeypatch.setattr(entry, "_serve", fake_serve)
    result = CliRunner().invoke(
        entry.app, ["serve", "--config", str(config_path), "--no-gui", "--no-watchdog"]
    )
    assert result.exit_code == 0
    assert "RESET_LOGIN was not applied" in result.output and "10 characters" in result.output


# --- the network cannot trigger it -----------------------------------------


def test_nothing_that_serves_a_request_knows_the_sentinel_exists() -> None:
    """Only ``__main__`` (the start-up path) imports the module; the request paths never name it."""
    importers: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or "", *(a.name for a in node.names)]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            if any("login_reset" in n for n in names):
                importers.append(rel)
    assert sorted(set(importers)) == ["__main__.py"], importers
    request_paths = ("api", "gui", "mcp", "watchdog", "tray")
    for package in request_paths:
        for path in sorted((SRC / package).rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            assert SENTINEL_NAME not in text and PASSWORD_FILE_NAME not in text, path


def test_the_only_caller_is_the_serve_start_up_path() -> None:
    tree = ast.parse((SRC / "__main__.py").read_text(encoding="utf-8"))
    callers = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(inner, ast.Name) and inner.id == "apply_sentinel" for inner in ast.walk(node)
        )
    }
    assert callers == {"_apply_login_reset"}
    serve = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "serve")
    called = {
        c.func.id
        for c in ast.walk(serve)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
    }
    assert "_apply_login_reset" in called


def test_requests_neither_create_nor_consume_nor_act_on_the_sentinel(tmp_path: Path) -> None:
    config = Config(
        data_dir=tmp_path / "data",
        server={"host": "127.0.0.1", "port": 1234},
        models={"dir": tmp_path / "models"},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
    )
    app = create_app(config, state=build_state(config), start_background=False)
    accounts: AccountStore = app.state.accounts
    accounts.create_account("alice", OLD)
    path = sentinel(config.data_dir, f"alice:{NEW}")
    with TestClient(app) as http:
        http.get("/health")
        http.get("/api/users")
        http.post("/api/users/password", json={"username": "alice", "password": NEW + "x"})
        http.patch("/api/config", json={"models.default_ctx": 4096})
        for name in (SENTINEL_NAME, f"../{SENTINEL_NAME}", PASSWORD_FILE_NAME):
            http.get(f"/{name}")
            http.get(f"/api/logs/models/{name}")
            http.post(f"/api/{name}")
        http.get("/api/status", headers={"X-Reset-Login": "1"})
    assert path.read_text(encoding="utf-8") == f"alice:{NEW}", "no request consumed or edited it"
    assert (
        accounts.verify_password("alice", OLD) is not None
        or accounts.verify_password("alice", NEW + "x") is not None
    )
    assert not (config.data_dir / PASSWORD_FILE_NAME).exists()
    assert accounts.verify_password("alice", NEW) is None, "the sentinel's password was not applied"


def test_docs_name_the_same_file_the_code_reads() -> None:
    readme = (SRC.parents[1] / "README.md").read_text(encoding="utf-8")
    assert re.search(rf"\b{SENTINEL_NAME}\b", readme)
    assert login_reset.SENTINEL_NAME == "RESET_LOGIN"
