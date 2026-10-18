"""The ``studioforge`` CLI: lms-style commands as thin clients of the local API.

Commands that act on the running service are exercised against a fake server (an
``httpx.MockTransport`` swapped in for the client) and, for the auth round trip,
against the real FastAPI app. Commands that work on the data directory (``keys``,
``users``, ``migrate``, ``doctor``, ``support-bundle``) run for real on a scratch
directory.
"""

from __future__ import annotations

import json
import sqlite3
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import studioforge.cli_commands as commands
from studioforge.__main__ import app as cli
from studioforge.api.app import build_state, create_app
from studioforge.cli_client import CliError, LocalClient, base_url_for
from studioforge.config import Config

runner = CliRunner()


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch install: loopback, scratch ports, null GPU probe, own data dir."""
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


def invoke(scratch: Path, *args: str, **kwargs: Any) -> Any:
    return runner.invoke(cli, [*args, "--config", str(scratch)], **kwargs)


class FakeServer:
    """Routes ``(METHOD, path)`` to canned JSON and remembers every request."""

    def __init__(self, routes: dict[tuple[str, str], Any]) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.routes.get((request.method, request.url.path))
        if callable(reply):
            reply = reply(request)
        if reply is None:
            return httpx.Response(404, json={"error": {"message": "no such route", "code": "x"}})
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(200, json=reply)

    def bodies(self, method: str, path: str) -> list[Any]:
        return [
            json.loads(r.content or b"null")
            for r in self.requests
            if r.method == method and r.url.path == path
        ]


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Callable[[dict[tuple[str, str], Any]], FakeServer]:
    def install(routes: dict[tuple[str, str], Any]) -> FakeServer:
        fake = FakeServer(routes)
        real = LocalClient

        def make(config: Config, **kwargs: Any) -> LocalClient:
            return real(config, transport=httpx.MockTransport(fake), **kwargs)

        monkeypatch.setattr(commands, "LocalClient", make)
        return fake

    return install


MODELS = {
    "models": [
        {
            "id": "org/Llama-8B",
            "size_bytes": 5 * 1024**3,
            "quant": "Q4_K_M",
            "architecture": "llama",
            "loaded": True,
            "state": "ready",
        },
        {
            "id": "org/Qwen-4B",
            "size_bytes": 3 * 1024**3,
            "quant": "Q8_0",
            "architecture": "qwen3",
            "loaded": False,
        },
    ]
}


# ---------------------------------------------------------------------------
# ls / ps / load / unload / get
# ---------------------------------------------------------------------------


def test_ls_lists_the_library(scratch: Path, server: Any) -> None:
    server({("GET", "/api/models"): MODELS})
    result = invoke(scratch, "ls")
    assert result.exit_code == 0, result.output
    assert (
        "org/Llama-8B" in result.output and "5.0 GiB" in result.output and "Q8_0" in result.output
    )
    loaded = invoke(scratch, "ls", "--loaded")
    assert "org/Llama-8B" in loaded.output and "Qwen-4B" not in loaded.output
    as_json = json.loads(invoke(scratch, "ls", "--json").output)
    assert [m["id"] for m in as_json] == ["org/Llama-8B", "org/Qwen-4B"]


def test_ls_on_an_empty_library_says_what_to_do(scratch: Path, server: Any) -> None:
    server({("GET", "/api/models"): {"models": []}})
    assert "studioforge get" in invoke(scratch, "ls").output


def test_ps_shows_what_is_loaded_where(scratch: Path, server: Any) -> None:
    server(
        {
            ("GET", "/api/status"): {
                "loaded": [
                    {
                        "model_id": "org/Llama-8B",
                        "state": "ready",
                        "plan": {"devices": [0, 1], "ctx_size": 32768},
                        "ttl_s": 600,
                        "active_requests": 1,
                    }
                ]
            }
        }
    )
    out = invoke(scratch, "ps").output
    assert "org/Llama-8B" in out and "0,1" in out and "32768" in out
    server({("GET", "/api/status"): {"loaded": []}})
    assert "Nothing is loaded" in invoke(scratch, "ps").output


def test_load_defaults_to_the_native_route_with_ttl_and_context(scratch: Path, server: Any) -> None:
    fake = server({("POST", "/api/v1/models/load"): {"context_length": 8192, "status": "loaded"}})
    result = invoke(scratch, "load", "org/Llama-8B", "--ttl", "900", "--context-length", "8192")
    assert result.exit_code == 0, result.output
    assert fake.bodies("POST", "/api/v1/models/load") == [
        {"model": "org/Llama-8B", "context_length": 8192, "ttl": 900}
    ]
    assert "Loaded org/Llama-8B (context 8192)" in result.output


def test_load_with_gpus_uses_the_placement_route_then_records_the_ttl(
    scratch: Path, server: Any
) -> None:
    fake = server(
        {
            ("POST", "/api/models/org/Llama-8B/load"): {"model_id": "org/Llama-8B"},
            ("POST", "/api/v1/models/load"): {"status": "loaded"},
        }
    )
    result = invoke(scratch, "load", "org/Llama-8B", "--gpu", "0,1", "--ctx", "4096", "--ttl", "60")
    assert result.exit_code == 0, result.output
    assert fake.bodies("POST", "/api/models/org/Llama-8B/load") == [
        {"devices": [0, 1], "ctx_size": 4096}
    ]
    assert fake.bodies("POST", "/api/v1/models/load") == [{"model": "org/Llama-8B", "ttl": 60}]


def test_load_with_a_preset_asks_for_the_recommended_load(scratch: Path, server: Any) -> None:
    fake = server({("POST", "/api/models/m/load-recommended"): {"ctx_size": 8192}})
    result = invoke(scratch, "load", "m", "--preset", "dual_5090")
    assert result.exit_code == 0, result.output
    body = fake.bodies("POST", "/api/models/m/load-recommended")[0]
    assert body["prefer_mode"] == "dual_5090" and body["ctx_size"] == 8192  # models.default_ctx


def test_load_rejects_a_bad_gpu_list(scratch: Path, server: Any) -> None:
    server({})
    result = invoke(scratch, "load", "m", "--gpu", "a,b")
    assert result.exit_code != 0 and "--gpu" in result.output


def test_unload_one_or_all(scratch: Path, server: Any) -> None:
    fake = server(
        {
            ("POST", "/api/v1/models/unload"): {"instance_id": "m"},
            ("POST", "/api/models/unload-all"): {"unloaded": ["m"]},
        }
    )
    assert invoke(scratch, "unload", "m").exit_code == 0
    assert fake.bodies("POST", "/api/v1/models/unload") == [{"model": "m"}]
    assert invoke(scratch, "unload", "--all").exit_code == 0
    assert fake.bodies("POST", "/api/models/unload-all") == [{}]
    assert invoke(scratch, "unload").exit_code != 0  # neither
    assert invoke(scratch, "unload", "m", "--all").exit_code != 0  # both


def test_get_queues_a_download_and_follows_it_to_the_end(
    scratch: Path, server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    polls = iter(
        [
            {
                "downloads": [
                    {
                        "group_id": "g1",
                        "status": "running",
                        "downloaded_bytes": 50,
                        "total_bytes": 100,
                    }
                ]
            },
            {
                "downloads": [
                    {
                        "group_id": "g1",
                        "status": "completed",
                        "downloaded_bytes": 100,
                        "total_bytes": 100,
                    }
                ]
            },
        ]
    )
    fake = server(
        {
            ("POST", "/api/downloads"): {"group_id": "g1", "quant": "Q4_K_M", "total_bytes": 100},
            ("GET", "/api/downloads"): lambda _r: next(polls),
        }
    )
    monkeypatch.setattr(commands.time, "sleep", lambda _s: None)
    result = invoke(scratch, "get", "org/Model-GGUF", "--quant", "Q4_K_M")
    assert result.exit_code == 0, result.output
    assert fake.bodies("POST", "/api/downloads") == [
        {"repo_id": "org/Model-GGUF", "quant": "Q4_K_M"}
    ]
    assert (
        "50%" in result.output and "100%" in result.output and "studioforge load" in result.output
    )


def test_get_no_wait_returns_at_once_and_a_failed_download_is_an_error(
    scratch: Path, server: Any
) -> None:
    server({("POST", "/api/downloads"): {"group_id": "g1", "quant": "Q4", "total_bytes": 10}})
    assert invoke(scratch, "get", "a/b", "--no-wait").exit_code == 0
    server(
        {
            ("POST", "/api/downloads"): {"group_id": "g1", "quant": "Q4", "total_bytes": 10},
            ("GET", "/api/downloads"): {"downloads": [{"group_id": "g1", "status": "failed"}]},
        }
    )
    failed = invoke(scratch, "get", "a/b")
    assert failed.exit_code == 1 and "failed" in failed.output


# ---------------------------------------------------------------------------
# errors read like sentences
# ---------------------------------------------------------------------------


def test_a_server_error_is_one_line_with_its_code_and_fix(scratch: Path, server: Any) -> None:
    server(
        {
            ("GET", "/api/models"): httpx.Response(
                503,
                json={
                    "error": {
                        "message": "still starting",
                        "code": "server_busy",
                        "fix": "Retry in a moment.",
                    }
                },
            )
        }
    )
    result = invoke(scratch, "ls")
    assert result.exit_code == 1
    assert (
        "[server_busy] still starting" in result.output
        and "fix: Retry in a moment." in result.output
    )
    assert "Traceback" not in result.output


def test_a_server_that_is_not_running_says_how_to_start_it(scratch: Path) -> None:
    # nothing listens on the scratch port 11234
    result = invoke(scratch, "ls")
    assert result.exit_code == 1
    assert "is not running" in result.output and "studioforge server start" in result.output


# ---------------------------------------------------------------------------
# the client against the real app: it mints its own key when the server wants one
# ---------------------------------------------------------------------------


def test_the_client_mints_a_local_key_when_authentication_is_on(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.config import load_config

    config = load_config(scratch)
    app = create_app(config, state=build_state(config), start_background=False)
    app.state.accounts.create_key("someone-else", "read")  # authentication is now enforced
    cli_key = config.data_dir / "cli.key"
    assert not cli_key.exists()
    with LocalClient(config) as client:
        client._http = TestClient(app, client=("127.0.0.1", 50000))  # the real app, in-process
        status = client.get("/api/status")
        assert "loaded" in status
        assert cli_key.is_file()  # minted on the 401, then reused
        assert client.get("/api/models") is not None
    principal = app.state.accounts.authenticate(cli_key.read_text(encoding="utf-8").strip())
    assert principal is not None and principal.name == "local-cli" and principal.scope == "admin"


def test_the_client_does_not_mint_a_key_for_a_server_that_does_not_ask(scratch: Path) -> None:
    from studioforge.config import load_config

    config = load_config(scratch)
    app = create_app(config, state=build_state(config), start_background=False)
    with LocalClient(config) as client:
        client._http = TestClient(app, client=("127.0.0.1", 50000))
        client.get("/api/status")
    assert not (config.data_dir / "cli.key").exists()


def test_a_non_credential_403_is_not_retried_with_a_key(scratch: Path, server: Any) -> None:
    fake = server(
        {
            ("GET", "/api/status"): httpx.Response(
                403, json={"error": {"message": "no", "code": "permission_denied"}}
            )
        }
    )
    result = invoke(scratch, "ps")
    assert result.exit_code == 1 and len(fake.requests) == 1


def test_base_url_uses_loopback_and_the_scheme(tmp_path: Path) -> None:
    assert base_url_for(Config(data_dir=tmp_path)) == "http://127.0.0.1:1234"
    wide = Config(data_dir=tmp_path, server={"host": "0.0.0.0", "tls": {"mode": "self_signed"}})
    assert base_url_for(wide) == "https://127.0.0.1:1234"
    assert str(CliError("x", "y")) == "x\n  fix: y"


# ---------------------------------------------------------------------------
# update / server / log
# ---------------------------------------------------------------------------


def test_update_check_apply_rollback(scratch: Path, server: Any) -> None:
    fake = server(
        {
            ("GET", "/api/update"): {
                "current_version": "1.0.0",
                "latest_version": "2.0.0",
                "update_available": True,
                "configured": True,
            },
            ("POST", "/api/update/install"): {"installed": "2.0.0"},
            ("POST", "/api/update/rollback"): {"rolled_back": True},
        }
    )
    assert "an update is available" in invoke(scratch, "update", "check").output
    assert invoke(scratch, "update", "apply", "--yes", "--tag", "v2").exit_code == 0
    assert fake.bodies("POST", "/api/update/install") == [{"tag": "v2", "confirm": True}]
    assert invoke(scratch, "update", "rollback", "--yes").exit_code == 0
    assert fake.bodies("POST", "/api/update/rollback") == [{"confirm": True}]
    declined = invoke(scratch, "update", "apply", input="n\n")
    assert declined.exit_code != 0 and len(fake.bodies("POST", "/api/update/install")) == 1


def test_update_check_when_updates_are_not_configured(scratch: Path, server: Any) -> None:
    server(
        {
            ("GET", "/api/update"): {
                "current_version": "1.0.0",
                "configured": False,
                "note": "set update.repo",
            }
        }
    )
    assert "update checks are off" in invoke(scratch, "update", "check").output


def test_server_status_reads_health(scratch: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import studioforge.server_control as control

    monkeypatch.setattr(
        control,
        "server_status",
        lambda config, **kw: control.ServerStatus(
            True, "http://127.0.0.1:11234", 4242, "2.0.0", 12.0
        ),
    )
    out = invoke(scratch, "server", "status").output
    assert "running at http://127.0.0.1:11234" in out and "pid 4242" in out
    monkeypatch.setattr(
        control, "server_status", lambda config, **kw: control.ServerStatus(False, "http://x")
    )
    assert invoke(scratch, "server", "status").exit_code == 1


def test_server_stop_asks_the_running_server_first(
    scratch: Path, server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import studioforge.server_control as control

    fake = server({("POST", "/api/server/stop"): {"stopping": True}})
    states = iter([control.ServerStatus(True, "u", None), control.ServerStatus(False, "u", None)])
    monkeypatch.setattr(control, "server_status", lambda config, **kw: next(states))
    monkeypatch.setattr(control.time, "sleep", lambda _s: None)
    result = invoke(scratch, "server", "stop")
    assert result.exit_code == 0 and "stopped" in result.output
    assert fake.bodies("POST", "/api/server/stop") == [{"confirm": True}]


def test_log_stream_prints_the_tail_once(scratch: Path, server: Any) -> None:
    server({("GET", "/api/logs"): {"lines": [{"message": "alpha"}, "beta", {"msg": "gamma"}]}})
    result = invoke(scratch, "log", "stream", "--no-follow", "-n", "2")
    assert result.exit_code == 0
    assert result.output.split() == ["beta", "gamma"]


def test_new_lines_overlap_detection() -> None:
    assert commands._new_lines([], ["a", "b"]) == ["a", "b"]
    assert commands._new_lines(["a", "b"], ["b", "c"]) == ["c"]
    assert commands._new_lines(["a", "b"], ["x", "y"]) == ["x", "y"]


# ---------------------------------------------------------------------------
# keys / users (direct on the data directory)
# ---------------------------------------------------------------------------


def test_keys_create_list_revoke(scratch: Path) -> None:
    created = invoke(scratch, "keys", "create", "--name", "agent", "--scope", "read", "--json")
    assert created.exit_code == 0, created.output
    payload = json.loads(created.output)
    token = payload["token"]
    assert token.startswith("sf-r-") and payload["key"]["scope"] == "read"
    listed = invoke(scratch, "keys", "list")
    assert "agent" in listed.output and token not in listed.output
    assert "local-cli" in listed.output  # the first key also gives local tools their own
    as_json = json.loads(invoke(scratch, "keys", "list", "--json").output)
    assert token not in json.dumps(as_json)
    revoked = invoke(scratch, "keys", "revoke", payload["key"]["id"])
    assert revoked.exit_code == 0 and "Revoked" in revoked.output
    assert "agent" not in invoke(scratch, "keys", "list").output  # hidden unless --all
    assert "agent" in invoke(scratch, "keys", "list", "--all").output
    missing = invoke(scratch, "keys", "revoke", "nope")
    assert missing.exit_code == 1


def test_keys_create_prints_the_token_once_and_warns_that_auth_is_on(scratch: Path) -> None:
    result = invoke(scratch, "keys", "create", "--name", "first")
    assert result.exit_code == 0
    assert result.output.count("sf-i-") == 1  # inference is the default scope
    assert "Authentication is now ON" in result.output
    second = invoke(scratch, "keys", "create", "--name", "second")
    assert "Authentication is now ON" not in second.output


def test_keys_create_validates(scratch: Path) -> None:
    bad = invoke(scratch, "keys", "create", "--name", "x", "--scope", "root")
    assert bad.exit_code == 1 and "scope" in bad.output
    assert invoke(scratch, "keys", "create", "--name", " ").exit_code == 1


def test_users_create_passwd_list(scratch: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SF_TEST_PW", "a-long-enough-password")
    made = invoke(scratch, "users", "create", "owner", "--password-env", "SF_TEST_PW")
    assert made.exit_code == 0 and "owner" in made.output
    assert "a-long-enough-password" not in made.output
    monkeypatch.setenv("SF_TEST_PW", "another-long-password")
    assert (
        invoke(scratch, "users", "passwd", "owner", "--password-env", "SF_TEST_PW").exit_code == 0
    )
    assert "owner" in invoke(scratch, "users", "list").output
    unset = invoke(scratch, "users", "create", "x", "--password-env", "SF_NOT_SET_ANYWHERE")
    assert unset.exit_code == 1 and "not set" in unset.output
    short = invoke(scratch, "users", "create", "y", input="short\nshort\n")
    assert short.exit_code == 1 and "10 characters" in short.output


def test_users_create_reads_a_password_from_stdin(scratch: Path) -> None:
    result = invoke(
        scratch, "users", "create", "owner", "--password-stdin", input="stdin-long-password\n"
    )
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# migrate / doctor / support-bundle
# ---------------------------------------------------------------------------


def old_database(path: Path, upto: int = 8) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at REAL, name TEXT)"
    )
    connection.executemany(
        "INSERT INTO schema_migrations VALUES (?, 0, 'old')", [(v,) for v in range(1, upto + 1)]
    )
    connection.execute("CREATE TABLE precious (x TEXT)")
    connection.execute("INSERT INTO precious VALUES ('keep me')")
    connection.commit()
    connection.close()


def test_migrate_dry_run_changes_nothing(scratch: Path) -> None:
    db = scratch.parent / "registry.sqlite3"
    old_database(db)
    before = db.read_bytes()
    result = invoke(scratch, "migrate", "--dry-run", "--json")
    summary = json.loads(result.output)
    assert summary["current"] == 8 and summary["pending"] == [
        {"version": 9, "name": "009_accounts_keys.sql"}
    ]
    assert summary["dry_run"] is True and "backup" not in summary
    assert db.read_bytes() == before
    assert not (scratch.parent / "backups").exists()
    human = invoke(scratch, "migrate", "--dry-run").output
    assert "Would apply" in human and "009" in human and "nothing was changed" in human


def test_migrate_applies_after_taking_a_backup(scratch: Path) -> None:
    db = scratch.parent / "registry.sqlite3"
    old_database(db)
    result = invoke(scratch, "migrate", "--json")
    summary = json.loads(result.output)
    assert summary["applied"] == [9] and summary["current"] == 9
    backup = Path(summary["backup"])
    assert backup.is_file() and backup.name == "registry.sqlite3"
    assert backup.parent.parent.name == "backups" and "schema8" in backup.parent.name
    saved = sqlite3.connect(backup)
    try:
        assert saved.execute("SELECT x FROM precious").fetchone() == ("keep me",)
        assert saved.execute("SELECT MAX(version) FROM schema_migrations").fetchone() == (8,)
    finally:
        saved.close()
    now = sqlite3.connect(db)
    try:
        assert now.execute("SELECT name FROM sqlite_master WHERE name='api_keys'").fetchone()
    finally:
        now.close()
    assert "current" in invoke(scratch, "migrate").output  # nothing left


def test_migrate_prunes_only_its_own_backup_folders(scratch: Path) -> None:
    """The container's .premigrate snapshots and the operator's folders share backups/."""
    import os

    backups = scratch.parent / "backups"
    foreign = [backups / f"2.0.0-2026101{i}T000000Z" for i in range(6)] + [backups / "mine"]
    for folder in foreign:
        folder.mkdir(parents=True)
        (folder / ".premigrate").write_text("from 2.0.0", encoding="utf-8")
        os.utime(folder, (1.0, 1.0))  # older than anything migrate writes
    db = scratch.parent / "registry.sqlite3"
    old_database(db)
    summary = json.loads(invoke(scratch, "migrate", "--json").output)
    assert Path(summary["backup"]).is_file()
    assert all(folder.is_dir() for folder in foreign)


def test_migrate_with_no_database_previews_the_whole_chain(scratch: Path) -> None:
    summary = json.loads(invoke(scratch, "migrate", "--dry-run", "--json").output)
    assert summary["exists"] is False and summary["current"] == 0 and len(summary["pending"]) == 9


def test_doctor_text_and_json_and_exit_codes(scratch: Path) -> None:
    ok = invoke(scratch, "doctor", "--json")
    payload = json.loads(ok.output)
    assert payload["findings"] and ok.exit_code in (0, 1)
    text = invoke(scratch, "doctor")
    assert "Authentication" in text.output and "ok," in text.output
    strict = invoke(scratch, "doctor", "--strict")
    assert strict.exit_code == 1  # no GPU under the null probe is a warning, so --strict fails


def test_doctor_fails_on_an_unsafe_bind(scratch: Path, tmp_path: Path) -> None:
    config = Config(
        data_dir=tmp_path / "data",
        server={"host": "0.0.0.0", "port": 11234},
        gui={"port": 18080},
        watchdog={"port": 11235},
    )
    config.save(scratch)
    result = invoke(scratch, "doctor")
    assert result.exit_code == 1
    assert "FAIL" in result.output and "studioforge setup" in result.output


def test_support_bundle_writes_a_zip(scratch: Path, tmp_path: Path) -> None:
    out = tmp_path / "bundle.zip"
    result = invoke(scratch, "support-bundle", "--output", str(out))
    assert result.exit_code == 0 and out.is_file()
    assert "config.redacted.yaml" in zipfile.ZipFile(out).namelist()
    assert "Credentials are removed" in result.output
