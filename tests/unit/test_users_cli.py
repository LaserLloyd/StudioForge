"""``studioforge users list | add | passwd | delete`` and the line each command prints (D81).

The verbs run for real on a scratch data directory (no server, no GPU, no keyring: the
unit conftest pins the file secret backend). Passwords come from ``--password-stdin`` or
``--password-env`` only; no verb takes one as an argument.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from studioforge.__main__ import app as cli
from studioforge.config import Config
from studioforge.core.accounts import AccountStore
from studioforge.db import Database

runner = CliRunner()

PW1 = "first-long-password"
PW2 = "second-long-password"


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SF_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SF_GPU_PROBE", "null")
    config = Config(
        data_dir=tmp_path / "data",
        models={"dir": tmp_path / "models"},
        server={"port": 11234},
        gui={"port": 18080},
        watchdog={"port": 11235},
        logging={"level": "ERROR"},
    )
    config.ensure_dirs()
    return config.save(tmp_path / "data" / "config.yaml")


def run(cfg: Path, *args: str, input: str | None = None):  # noqa: A002 - CliRunner's name
    return runner.invoke(cli, [*args, "--config", str(cfg)], input=input)


def store_for(cfg: Path) -> tuple[Database, AccountStore]:
    config = Config(data_dir=cfg.parent)
    db = Database(config.db_path)
    db.migrate()
    return db, AccountStore(db, config.data_dir)


def add(cfg: Path, name: str, password: str = PW1):
    return run(cfg, "users", "add", name, "--password-stdin", input=password + "\n")


# --- add / create ----------------------------------------------------------


def test_add_is_create_and_the_first_account_is_the_owner(cfg: Path) -> None:
    first = add(cfg, "alice")
    assert first.exit_code == 0, first.output
    assert "Created account 'alice' (owner)" in first.stdout
    second = run(cfg, "users", "create", "bob", "--password-stdin", input=PW2 + "\n")
    assert second.exit_code == 0 and "(admin)" in second.stdout
    listed = run(cfg, "users", "list", "--json")
    assert [(a["username"], a["role"]) for a in json.loads(listed.stdout)] == [
        ("alice", "owner"),
        ("bob", "admin"),
    ]


def test_add_refuses_a_duplicate_and_a_short_password(cfg: Path) -> None:
    assert add(cfg, "alice").exit_code == 0
    dup = add(cfg, "ALICE")
    assert dup.exit_code == 1 and "already exists" in dup.stderr
    short = add(cfg, "carol", "short")
    assert short.exit_code == 1 and "10 characters" in short.stderr


def test_no_verb_takes_a_password_as_an_argument(cfg: Path) -> None:
    for verb in ("add", "create", "passwd"):
        result = run(cfg, "users", verb, "alice", "--password", "hunter2hunter2")
        assert result.exit_code == 2, (verb, result.output)
        assert "hunter2hunter2" not in result.stdout


def test_add_reads_the_password_from_an_env_var(cfg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SF_TEST_ADD_PW", PW1)
    result = run(cfg, "users", "add", "alice", "--password-env", "SF_TEST_ADD_PW")
    assert result.exit_code == 0 and PW1 not in result.output
    db, store = store_for(cfg)
    try:
        assert store.verify_password("alice", PW1) is not None
    finally:
        db.close()


# --- passwd ----------------------------------------------------------------


def test_passwd_changes_the_password_and_the_old_one_stops_working(cfg: Path) -> None:
    add(cfg, "alice")
    changed = run(cfg, "users", "passwd", "alice", "--password-stdin", input=PW2 + "\n")
    assert changed.exit_code == 0 and "Changed the password of 'alice'" in changed.stdout
    db, store = store_for(cfg)
    try:
        assert store.verify_password("alice", PW2) is not None
        assert store.verify_password("alice", PW1) is None
    finally:
        db.close()


def test_passwd_for_a_missing_account_says_how_to_list(cfg: Path) -> None:
    result = run(cfg, "users", "passwd", "ghost", "--password-stdin", input=PW1 + "\n")
    assert result.exit_code == 1
    assert "no account named 'ghost'" in result.stderr and "users list" in result.stderr


def test_passwd_with_an_unset_env_var_is_one_line(cfg: Path) -> None:
    add(cfg, "alice")
    result = run(cfg, "users", "passwd", "alice", "--password-env", "SF_NEVER_SET_ANYWHERE")
    assert result.exit_code == 1 and "not set" in result.stderr


# --- delete ----------------------------------------------------------------


def test_delete_removes_the_account_and_its_password(cfg: Path) -> None:
    add(cfg, "alice")
    add(cfg, "bob", PW2)
    result = run(cfg, "users", "delete", "bob", "--yes")
    assert result.exit_code == 0 and "Deleted the account 'bob'" in result.stdout
    db, store = store_for(cfg)
    try:
        assert [a.username for a in store.list_accounts()] == ["alice"]
        assert store.verify_password("bob", PW2) is None
        assert store.secrets.get("owner.bob.hash") is None
    finally:
        db.close()


def test_delete_asks_first_and_no_means_no(cfg: Path) -> None:
    add(cfg, "alice")
    add(cfg, "bob", PW2)
    declined = run(cfg, "users", "delete", "bob", input="n\n")
    assert declined.exit_code == 0 and "Nothing deleted" in declined.stdout
    assert "bob" in run(cfg, "users", "list").stdout
    accepted = run(cfg, "users", "delete", "bob", input="y\n")
    assert accepted.exit_code == 0 and "Deleted" in accepted.stdout


def test_delete_a_missing_account_is_an_error(cfg: Path) -> None:
    add(cfg, "alice")
    result = run(cfg, "users", "delete", "ghost", "--yes")
    assert result.exit_code == 1 and "no account named 'ghost'" in result.stderr


def test_delete_refuses_the_last_account_unless_forced(cfg: Path) -> None:
    add(cfg, "alice")
    refused = run(cfg, "users", "delete", "alice", "--yes")
    assert refused.exit_code == 1 and "last account that can sign in" in refused.stderr
    assert "--force" in refused.stderr
    assert "alice" in run(cfg, "users", "list").stdout
    forced = run(cfg, "users", "delete", "alice", "--yes", "--force")
    assert forced.exit_code == 0 and "No accounts remain" in forced.stdout
    assert "No accounts" in run(cfg, "users", "list").stdout


def test_deleting_the_owner_promotes_the_oldest_remaining_account(cfg: Path) -> None:
    add(cfg, "alice")
    add(cfg, "bob", PW2)
    add(cfg, "carol", PW2)
    assert run(cfg, "users", "delete", "alice", "--yes").exit_code == 0
    roles = {
        a["username"]: a["role"] for a in json.loads(run(cfg, "users", "list", "--json").stdout)
    }
    assert roles == {"bob": "owner", "carol": "admin"}


def test_a_deleted_accounts_panel_session_no_longer_validates(cfg: Path) -> None:
    add(cfg, "alice")
    add(cfg, "bob", PW2)
    db, store = store_for(cfg)
    try:
        before = store.password_fingerprint("bob")
    finally:
        db.close()
    run(cfg, "users", "delete", "bob", "--yes")
    db, store = store_for(cfg)
    try:
        assert store.get_account("bob") is None
        assert store.password_fingerprint("bob") != before
    finally:
        db.close()


# --- the line every account command prints ---------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ("users", "list"),
        ("keys", "list"),
        ("migrate", "--dry-run"),
        ("doctor",),
    ],
)
def test_account_commands_say_which_data_dir_and_database_they_act_on(
    cfg: Path, args: tuple[str, ...]
) -> None:
    result = run(cfg, *args)
    assert f"acting on {cfg.parent}" in result.stderr, result.output
    assert "registry.sqlite3" in result.stderr
    assert "from --config" in result.stderr


def test_the_line_goes_to_stderr_so_json_stays_clean(cfg: Path) -> None:
    add(cfg, "alice")
    result = run(cfg, "users", "list", "--json")
    assert "acting on" in result.stderr
    assert json.loads(result.stdout)[0]["username"] == "alice"


def test_the_line_names_the_environment_when_that_decided(
    cfg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = runner.invoke(cli, ["users", "list"])
    assert "from SF_DATA_DIR" in result.stderr and str(cfg.parent) in result.stderr


def test_delete_refuses_when_only_disabled_accounts_and_no_admin_key_would_remain(
    cfg: Path,
) -> None:
    add(cfg, "alice")
    add(cfg, "bob", PW2)
    db, store = store_for(cfg)
    try:
        store.db.execute_write("UPDATE accounts SET disabled = 1 WHERE username = 'bob'")
    finally:
        db.close()
    refused = run(cfg, "users", "delete", "alice", "--yes")
    assert refused.exit_code == 1 and "last account that can sign in" in refused.stderr


def test_an_active_admin_key_lets_the_last_account_go_without_force(cfg: Path) -> None:
    add(cfg, "alice")
    assert run(cfg, "keys", "create", "--name", "ops", "--scope", "admin").exit_code == 0
    assert run(cfg, "users", "delete", "alice", "--yes").exit_code == 0


def test_the_owner_is_promoted_from_the_enabled_accounts_only(cfg: Path) -> None:
    add(cfg, "alice")
    add(cfg, "bob", PW2)
    add(cfg, "carol", PW2)
    db, store = store_for(cfg)
    try:
        store.db.execute_write("UPDATE accounts SET disabled = 1 WHERE username = 'bob'")
    finally:
        db.close()
    assert run(cfg, "users", "delete", "alice", "--yes").exit_code == 0
    roles = {
        a["username"]: a["role"] for a in json.loads(run(cfg, "users", "list", "--json").stdout)
    }
    assert roles["carol"] == "owner" and roles["bob"] == "admin"


def test_a_forced_delete_of_the_last_account_keeps_a_network_bound_server_closed(
    cfg: Path,
) -> None:
    """Deleting the last account must not silently reopen a running, network-bound server."""
    from studioforge.api.auth import auth_enforced

    add(cfg, "alice")
    config = Config(data_dir=cfg.parent, server={"host": "0.0.0.0"})  # noqa: S104
    db, store = store_for(cfg)
    try:
        assert auth_enforced(config, store)
        assert run(cfg, "users", "delete", "alice", "--yes", "--force").exit_code == 0
        store._invalidate_posture()
        assert not store.has_owner()
        assert store.had_owner_cached()
        assert auth_enforced(config, store), "an account existed, so credentials stay required"
    finally:
        db.close()


def test_a_fresh_install_with_no_account_ever_is_still_open_on_loopback(cfg: Path) -> None:
    from studioforge.api.auth import auth_enforced

    db, store = store_for(cfg)
    try:
        assert not store.had_owner_cached()
        assert not auth_enforced(Config(data_dir=cfg.parent), store)
    finally:
        db.close()
