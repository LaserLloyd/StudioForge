"""Click-through tests for the new pages, driven by NiceGUI's simulated user.

``TestClient`` proves a page builds; it never presses a button. These tests run the
page builders under ``nicegui.testing.user_simulation`` (no browser, no port) and press
the real controls, so the async handlers behind them are exercised: create and revoke a
key (one-time dialog), accounts, updates, diagnostics, serving and the wizard's steps.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from nicegui import ui
from nicegui.testing.user import User
from nicegui.testing.user_simulation import user_simulation

from studioforge.api.auth import GUARD
from studioforge.auditlog import AuditLog
from studioforge.config import Config
from studioforge.core.accounts import AccountStore
from studioforge.db import Database
from studioforge.gui import tabs
from studioforge.gui.tabs import users_keys
from tests.unit.test_gui import _FakeState, make_record

PASSWORD = "correct horse battery"


@pytest.fixture(autouse=True)
def _fresh_guard() -> None:
    GUARD.reset()


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config(data_dir=tmp_path / "data")
    cfg.models.dir = tmp_path / "models"
    cfg.models.dir.mkdir(parents=True, exist_ok=True)
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def store(tmp_path: Path) -> AccountStore:
    db = Database(tmp_path / "registry.sqlite3")
    db.migrate()
    return AccountStore(db, tmp_path)


@pytest.fixture
def state(config: Config, store: AccountStore) -> _FakeState:
    fake = _FakeState(config)
    fake.accounts = store  # type: ignore[attr-defined]
    fake.audit = AuditLog(config.logs_dir)  # type: ignore[attr-defined]
    return fake


@pytest.fixture
def ctx(config: Config, state: _FakeState) -> tabs.GuiContext:
    return tabs.GuiContext(config=config, api_state=state)


@asynccontextmanager
async def panel(
    ctx: tabs.GuiContext, builder: Callable[[tabs.GuiContext], Any]
) -> AsyncIterator[User]:
    def root() -> None:
        builder(ctx)

    try:
        async with user_simulation(root=root) as user:
            await user.open("/")
            yield user
    finally:
        # The simulation wipes NiceGUI's process-global app (routes, run config, shared
        # head HTML). ``gui.app`` remembers that it already mounted itself, so tell it
        # to do so again for whichever test builds the real panel next.
        from nicegui import core

        from studioforge.gui import app as gui_app

        core.app.middleware_stack = None
        core.app.user_middleware.clear()
        gui_app._NICEGUI_MOUNTED = False
        gui_app._PAGES_REGISTERED = False
        gui_app._THEME_ASSETS_REGISTERED = False


def all_text(user: User) -> str:
    """Every piece of text and every table row the page is currently holding."""
    parts: list[str] = []
    for element in user.client.elements.values():
        parts.append(str(getattr(element, "text", "") or ""))
        parts.append(str(element.props.get("label", "") or ""))
        parts.append(str(element.props.get("rows", "") or ""))
        parts.append(str(element.props.get("model-value", "") or ""))
    return "\n".join(parts)


def labels_starting(user: User, prefix: str) -> list[str]:
    return [
        str(e.text)
        for e in user.client.elements.values()
        if isinstance(e, ui.label) and str(e.text).startswith(prefix)
    ]


def only(user: User, **kwargs: Any) -> Any:
    return user.find(**kwargs).elements.pop()


# ---------------------------------------------------------------------------
# Users & Keys
# ---------------------------------------------------------------------------


async def test_creating_a_key_shows_its_token_once_and_then_not_again(
    ctx: tabs.GuiContext, store: AccountStore, state: _FakeState
) -> None:
    async with panel(ctx, users_keys.render) as user:
        await user.should_see("Who can use this server")
        only(user, kind=ui.expansion, content="Create a key").open()
        user.find("Key name (what will use it)").type("robot")
        user.find("Create key").click()
        await user.should_see("Key 'robot' created")
        (token,) = labels_starting(user, "sf-i-")
        assert store.authenticate(token, count=False) is not None
        await user.should_see("Copy it now")

        user.find("I have copied it").click()
        await user.should_not_see("Copy it now")
        # The row is in the table; the token is nowhere on the page any more.
        assert "robot" in all_text(user)
        assert token not in all_text(user)
    assert [k.name for k in store.list_keys()] == ["robot"]
    assert any(e["action"] == "panel.key.create" for e in state.audit.tail(5))  # type: ignore[attr-defined]


async def test_a_missing_key_name_is_reported_not_created(
    ctx: tabs.GuiContext, store: AccountStore
) -> None:
    async with panel(ctx, users_keys.render) as user:
        only(user, kind=ui.expansion, content="Create a key").open()
        user.find("Create key").click()  # no name typed
        await user.should_see("not allowed")
    assert store.list_keys() == []


async def test_revoking_a_key_asks_first(ctx: tabs.GuiContext, store: AccountStore) -> None:
    record, token = store.create_key("old-bot", "read")
    async with panel(ctx, users_keys.render) as user:
        user.find(kind=ui.table).trigger("revoke", args=record.id)
        await user.should_see("Revoke 'old-bot'?")
        user.find("Cancel").click()
        assert store.find_key(record.id).revoked is False  # type: ignore[union-attr]

        user.find(kind=ui.table).trigger("revoke", args=record.id)
        await user.should_see("Revoke 'old-bot'?")
        user.find("Revoke key").click()
        await user.should_see("Key 'old-bot' revoked.")
        assert token not in all_text(user)
    assert store.find_key(record.id).revoked is True  # type: ignore[union-attr]


async def test_creating_an_account_checks_the_passwords(
    ctx: tabs.GuiContext, store: AccountStore
) -> None:
    async with panel(ctx, users_keys.render) as user:
        only(user, kind=ui.expansion, content="Create an account").open()
        user.find("Username").type("alice")
        user.find("Password").type(PASSWORD)
        user.find("Type the password again").type("something else entirely")
        user.find("Create account").click()
        await user.should_see("do not match")
        assert not store.has_owner()

        only(user, content="Type the password again").set_value(PASSWORD)
        user.find("Create account").click()
        await user.should_see("Account 'alice' created (owner).")
    assert store.verify_password("alice", PASSWORD) is not None


# ---------------------------------------------------------------------------
# Updates
# ---------------------------------------------------------------------------


class _Status:
    def __init__(self, **extra: Any) -> None:
        self.data = {
            "current_version": "9.9.9",
            "configured": True,
            "update_available": False,
            "releases_installed": ["rel-1"],
            "current_release": "rel-1",
            "previous_release": "rel-0",
            **extra,
        }

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


class _Updater:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def status_sync(self) -> _Status:
        return _Status()

    async def check(self) -> _Status:
        self.calls.append("check")
        return _Status(
            update_available=True, latest_version="10.0.0", latest_tag="v10.0.0", checked_at=1.0
        )

    async def install(self, tag: Any, *, drain: Any, restart: bool) -> dict[str, Any]:
        self.calls.append(f"install {tag}")
        return {"message": "Update installed. The server is restarting."}

    async def rollback(self) -> dict[str, Any]:
        self.calls.append("rollback")
        return {"message": "Rolled back."}


async def test_check_install_and_roll_back_from_the_updates_page(
    ctx: tabs.GuiContext, state: _FakeState
) -> None:
    from studioforge.gui.tabs import updates

    updater = _Updater()
    state.updater = updater  # type: ignore[attr-defined]
    async with panel(ctx, updates.render) as user:
        await user.should_see("Running version 9.9.9.")
        user.find("Check for updates").click()
        await user.should_see("Version 10.0.0 is available")

        user.find("Install 10.0.0").click()
        await user.should_see("Install this update?")
        user.find("Cancel").click()
        assert updater.calls == ["check"]
        user.find("Install 10.0.0").click()
        await user.should_see("Install this update?")
        user.find("Install and restart").click()
        await user.should_see("Update installed. The server is restarting.")

        user.find("Roll back to the previous release").click()
        await user.should_see("Go back to the previous release?")
        user.find("Roll back and restart").click()
        await user.should_see("Rolled back.")
    assert updater.calls == ["check", "install v10.0.0", "rollback"]


async def test_updates_page_with_no_source_disables_the_check(
    ctx: tabs.GuiContext, state: _FakeState
) -> None:
    from studioforge.gui.tabs import updates

    class _Unconfigured(_Updater):
        def status_sync(self) -> _Status:
            return _Status(configured=False, note=updates.NOT_CONFIGURED, releases_installed=[])

    state.updater = _Unconfigured()  # type: ignore[attr-defined]
    async with panel(ctx, updates.render) as user:
        await user.should_see("No update source is set")
        assert not only(user, content="Check for updates").enabled


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class _Report:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def to_dict(self) -> dict[str, Any]:
        return self.data


async def test_run_checks_lists_the_findings_with_their_fixes(
    ctx: tabs.GuiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.gui.tabs import diagnostics

    report = _Report(
        {
            "worst": "fail",
            "counts": {"ok": 1, "info": 0, "warn": 1, "fail": 1},
            "findings": [
                {"severity": "ok", "title": "Ports", "message": "all free", "fix": "n/a"},
                {"severity": "fail", "title": "Engine", "message": "none", "fix": "install"},
                {"severity": "warn", "title": "Disk", "message": "low", "fix": "free space"},
            ],
        }
    )
    monkeypatch.setattr(diagnostics, "run_doctor", lambda cfg, **kw: report)
    async with panel(ctx, diagnostics.render) as user:
        user.find("Run checks").click()
        await user.should_see("Fix: install")
        await user.should_see("Fix: free space")
        text = all_text(user)
        assert "1 failed" in text and "Something is broken" in text
        assert "Fix: n/a" not in text  # a passing check gets no "Fix" line


async def test_the_support_bundle_button_downloads_a_zip(
    ctx: tabs.GuiContext, state: _FakeState, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.gui.tabs import diagnostics

    report = _Report({"worst": "ok", "counts": {"ok": 1}, "findings": []})
    monkeypatch.setattr(diagnostics, "run_doctor", lambda cfg, **kw: report)
    async with panel(ctx, diagnostics.render) as user:
        user.find("Download support bundle").click()
        await user.should_see("Support bundle ready. Credentials were removed.")
    assert any(e["action"] == "panel.support_bundle" for e in state.audit.tail(5))  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------


async def test_saving_the_port_goes_through_the_settings_route(
    ctx: tabs.GuiContext, state: _FakeState, config: Config
) -> None:
    from studioforge.gui.tabs import serving

    async with panel(ctx, serving.listening_card) as user:
        only(user, kind=ui.number, content="Server port").set_value(4321)
        user.find("Save").click()
        await user.should_see("saved: server.port")
        await user.should_see("restart required")
    on_disk = yaml.safe_load(config.config_path.read_text(encoding="utf-8"))
    assert on_disk["server"]["port"] == 4321
    line = next(e for e in state.audit.tail(5) if e["action"] == "panel.config.set")  # type: ignore[attr-defined]
    assert line["target"] == "server.port" and "4321" not in str(line)


async def test_the_https_card_saves_the_mode(ctx: tabs.GuiContext, config: Config) -> None:
    from studioforge.gui.tabs import serving

    async with panel(ctx, serving.tls_card) as user:
        only(user, kind=ui.select, content="HTTPS mode").set_value("self_signed")
        user.find("Save").click()
        await user.should_see("saved: server.tls.mode")
    on_disk = yaml.safe_load(config.config_path.read_text(encoding="utf-8"))
    assert on_disk["server"]["tls"]["mode"] == "self_signed"


# ---------------------------------------------------------------------------
# The wizard, one step at a time
# ---------------------------------------------------------------------------


class _Engines:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def active(self) -> Any:
        return None

    async def install(self, tag: str, **kwargs: Any) -> Any:
        self.calls.append(f"install {tag}")
        return SimpleNamespace(tag=tag)

    async def smoke_test(self, tag: str) -> tuple[bool, str]:
        self.calls.append(f"smoke {tag}")
        return True, "ran"


class _Manager:
    def __init__(self) -> None:
        self.loaded: list[str] = []

    async def load_recommended(self, model_id: str, ctx_size: int, **kwargs: Any) -> Any:
        self.loaded.append(model_id)

    async def test_model(self, model_id: str, prompt: Any, *, keep_loaded: bool) -> dict[str, Any]:
        return {"ok": True, "text": "Hello from the model", "tokens_per_second": 40.0}


class _Registry:
    def __init__(self, records: list[Any]) -> None:
        self.records = records

    def all(self) -> list[Any]:
        return list(self.records)

    def scan(self, *, force: bool = False) -> None:
        return None


async def test_the_wizard_steps_run_when_their_buttons_are_pressed(
    ctx: tabs.GuiContext,
    state: _FakeState,
    store: AccountStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studioforge.gui.tabs import setup_wizard_page

    async def fake_activate(request: Any, tag: str) -> dict[str, Any]:
        return {"tag": tag}

    monkeypatch.setattr("studioforge.api.mgmt_routes.engine_activate", fake_activate)
    state.engine_manager = _Engines()
    state.manager = _Manager()
    state.registry = _Registry([make_record("alpha")])  # type: ignore[assignment]
    async with panel(ctx, setup_wizard_page.render) as user:
        await user.should_see("Get StudioForge ready in eight steps")

        # 1. a mismatch is refused, a match creates the owner
        user.find("Password (at least 10 characters)").type(PASSWORD)
        user.find("Type the password again").type("nope")
        user.find("Create owner account").click()
        await user.should_see("The two passwords do not match")
        assert not store.has_owner()
        only(user, content="Type the password again").set_value(PASSWORD)
        user.find("Create owner account").click()
        await user.should_see("Owner account: done")
        assert store.has_owner()

        # 2. hardware
        user.find("Detect hardware").click()
        await user.should_see("GPU 0: RTX 5090")

        # 3. engine
        user.find("Install and test the engine").click()
        await user.should_see("Engine: done")
        assert state.engine_manager.calls[-1].startswith("smoke")

        # 5. load the model that is already in the library
        user.find("Get this model ready").click()
        await user.should_see("Starter model: done")
        assert state.manager.loaded == ["alpha"]

        # 6. test chat
        user.find("Send a test message").click()
        await user.should_see("Reply: Hello from the model")

        # 7. connection details
        user.find("Show the connection details").click()
        await user.should_see("Base URL")
        assert "http://127.0.0.1:1234/v1" in all_text(user)


async def test_the_lan_exposure_shows_the_admin_key_once(
    ctx: tabs.GuiContext, state: _FakeState, store: AccountStore, config: Config
) -> None:
    from studioforge.gui.tabs import setup_wizard_page

    store.create_account("alice", PASSWORD)
    async with panel(ctx, setup_wizard_page.render) as user:
        radios = [e for e in user.client.elements.values() if isinstance(e, ui.radio)]
        exposure = next(r for r in radios if "lan" in r.options)
        exposure.set_value("lan")
        user.find("Apply").click()
        await user.should_see("Your admin key")
        (token,) = labels_starting(user, "sf-a-")
        assert store.authenticate(token, count=False).scope == "admin"  # type: ignore[union-attr]
        await user.should_see("Restart required")
        user.find("I have copied it").click()
        await user.should_not_see("Your admin key")
        assert token not in all_text(user)
    on_disk = yaml.safe_load(config.config_path.read_text(encoding="utf-8"))
    assert on_disk["server"]["host"] == "0.0.0.0"
    assert on_disk["server"]["tls"]["mode"] == "self_signed"


# ---------------------------------------------------------------------------
# Integrations, metrics, the setup banner
# ---------------------------------------------------------------------------


async def test_the_key_picker_renames_the_placeholder_in_every_snippet(
    ctx: tabs.GuiContext, store: AccountStore
) -> None:
    from studioforge.gui.tabs import integrations

    store.create_key("dispatch", "inference")
    async with panel(ctx, integrations.render) as user:
        await user.should_see("<your-api-key>")
        picker = only(user, kind=ui.select, content="Which key will it use?")
        picker.set_value("dispatch")
        await user.should_see("<paste-the-dispatch-key-here>")
        assert "<your-api-key>" not in all_text(user).replace("No keys exist yet", "")


async def test_a_key_made_on_the_integrations_page_appears_in_the_picker(
    ctx: tabs.GuiContext, store: AccountStore
) -> None:
    from studioforge.gui.tabs import integrations

    async with panel(ctx, integrations.render) as user:
        only(user, kind=ui.expansion, content="Create a key").open()
        user.find("Key name (what will use it)").type("claude-agent")
        user.find("Create key").click()
        await user.should_see("Key 'claude-agent' created")
        (token,) = labels_starting(user, "sf-i-")
        user.find("I have copied it").click()
        assert token not in all_text(user)
        picker = only(user, kind=ui.select, content="Which key will it use?")
        assert "claude-agent" in picker.options


async def test_the_metrics_block_reads_the_text_when_it_is_opened(
    ctx: tabs.GuiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.gui.tabs import logs

    reads: list[int] = []

    async def fake_text(_ctx: Any) -> str:
        reads.append(1)
        return "studioforge_up 1\n"

    monkeypatch.setattr(logs, "_metrics_text", fake_text)
    async with panel(ctx, logs.metrics_panel) as user:
        await user.should_see("Metrics (Prometheus text)")
        assert reads == []  # nothing is asked of the models until it is opened
        only(user, kind=ui.expansion, content="Metrics (Prometheus text)").open()
        await user.should_see("studioforge_up 1")
    assert reads == [1]


async def test_the_setup_banner_opens_the_wizard(
    ctx: tabs.GuiContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nicegui.testing.user_navigate import UserNavigate

    from studioforge.gui.tabs import overview

    opened: list[Any] = []
    monkeypatch.setattr(UserNavigate, "to", lambda self, target, *a, **k: opened.append(target))
    async with panel(ctx, overview._setup_banner) as user:
        await user.should_see("Finish setup")
        user.find("Open the setup guide").click()
    assert opened == ["/setup"]
