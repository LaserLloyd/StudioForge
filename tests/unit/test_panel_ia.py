"""The panel's pages as built into the four-tab shell (the IA itself is in test_panel_four_tabs).

Pages are checked two ways: the rendered HTML through ``TestClient`` (a page that
crashes becomes an error card with a 200, so the text is asserted), and the pure
functions and flows each page is built on.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.api.auth import GUARD
from studioforge.auditlog import AuditLog
from studioforge.config import Config
from studioforge.core.accounts import AccountStore
from studioforge.db import Database
from studioforge.errors import StudioForgeError
from studioforge.gui import state as st
from studioforge.gui import tabs
from studioforge.gui.app import create_gui_app
from studioforge.types import EngineInfo
from tests.unit.test_gui import _FakeState, make_record


def adv(slug: str) -> str:
    """The deep link to one Advanced page."""
    return f"/?section=Advanced&page={slug}"


@pytest.fixture(autouse=True)
def _fresh_guard() -> Iterator[None]:
    GUARD.reset()
    yield
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


def ctx_of(config: Config, state: Any) -> tabs.GuiContext:
    return tabs.GuiContext(config=config, api_state=state)


def page(config: Config, state: Any, path: str = "/") -> str:
    with TestClient(create_gui_app(config, api_state=state)) as client:
        response = client.get(path)
    assert response.status_code == 200
    return response.text


def working(state: _FakeState) -> None:
    """Make the install look finished: a model indexed and an engine active."""
    state.registry._records = [make_record("alpha-8b")]
    engine = EngineInfo(
        tag="b1",
        path=Path("engines/b1"),
        server_binary=Path("engines/b1/llama-server.exe"),
        variant="cuda-13",
        version_string="b1",
        smoke_tested=True,
        active=True,
    )
    state.engine_manager = SimpleNamespace(
        active=lambda: engine,
        list=lambda: [engine],
        installed=lambda: [engine],
        get=lambda tag: engine,
    )


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


def test_every_tab_and_every_advanced_page_renders_without_an_error_card(
    config: Config, state: _FakeState
) -> None:
    from studioforge.gui import nav

    paths = ["/", "/?section=Models", "/?section=Test%20Chat"]
    paths += [adv(slug) for slug in nav.ADVANCED_SLUGS]
    for path in paths:
        assert "could not be rendered" not in page(config, state, path), path


def test_the_deep_link_grammar_in_state_is_unchanged() -> None:
    assert st.DEEP_LINK_TABS["download"] == "Download"
    assert st.DEEP_LINK_TABS["server"] == "Server"


def test_a_deep_link_to_a_new_section_renders(config: Config, state: _FakeState) -> None:
    assert "could not be rendered" not in page(config, state, adv("diagnostics"))


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def test_a_fresh_install_lands_on_the_overview_with_the_setup_banner(
    config: Config, state: _FakeState
) -> None:
    text = page(config, state)
    assert "Finish setup" in text and "Open the setup guide" in text


def test_a_working_install_has_no_setup_banner(config: Config, state: _FakeState) -> None:
    working(state)
    text = page(config, state)
    assert "Finish setup" not in text
    assert "could not be rendered" not in text


def test_no_owner_account_is_not_a_reason_to_finish_setup_on_loopback(
    config: Config, state: _FakeState
) -> None:
    from studioforge.gui.tabs.overview import needs_setup

    working(state)
    assert not state.accounts.has_owner()  # type: ignore[attr-defined]
    assert needs_setup(ctx_of(config, state)) is False


def test_a_network_bind_with_nobody_to_sign_in_is_a_reason(
    config: Config, state: _FakeState
) -> None:
    from studioforge.gui.tabs.overview import needs_setup

    working(state)
    config.server.host = "0.0.0.0"
    assert needs_setup(ctx_of(config, state)) is True
    state.accounts.create_account("alice", "correct horse battery")  # type: ignore[attr-defined]
    assert needs_setup(ctx_of(config, state)) is False


def test_the_base_url_card_follows_the_scheme_and_port(config: Config, state: _FakeState) -> None:
    from studioforge.gui.tabs.overview import base_urls, primary_base_url, server_root

    config.server.host = "127.0.0.1"
    config.server.port = 4321
    ctx = ctx_of(config, state)
    assert primary_base_url(ctx) == "http://127.0.0.1:4321/v1"
    assert server_root(ctx) == "http://127.0.0.1:4321"
    assert "http://127.0.0.1:4321/v1" in page(config, state)
    config.server.tls.mode = "self_signed"
    assert base_urls(ctx)[0]["url"] == "https://127.0.0.1:4321/v1"
    assert "https://127.0.0.1:4321/v1" in page(config, state)


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------


def test_serving_warns_when_the_server_would_refuse_to_start(
    config: Config, state: _FakeState
) -> None:
    config.server.host = "0.0.0.0"
    text = page(config, state, adv("serving"))
    assert "The server will refuse to start" in text
    assert "page=accounts" in text  # the link to Accounts &amp; keys
    from studioforge.gui.tabs.widgets import is_remote_without_admin

    ctx = ctx_of(config, state)
    assert is_remote_without_admin(ctx) is True
    state.accounts.create_account("alice", "correct horse battery")  # type: ignore[attr-defined]
    assert is_remote_without_admin(ctx) is False
    # With the owner in place the panel is gated; an admin key gets through, with no warning.
    _, token = state.accounts.create_key("ops", "admin")  # type: ignore[attr-defined]
    with TestClient(create_gui_app(config, api_state=state)) as client:
        signed_in = client.get(adv("serving"), headers={"Authorization": f"Bearer {token}"})
    assert "The server will refuse to start" not in signed_in.text


def test_serving_has_no_warning_on_loopback(config: Config, state: _FakeState) -> None:
    assert "The server will refuse to start" not in page(config, state, adv("serving"))


def test_serving_shows_the_certificate_fingerprint(
    config: Config, state: _FakeState, monkeypatch: pytest.MonkeyPatch
) -> None:
    config.server.tls.mode = "self_signed"
    monkeypatch.setattr(
        "studioforge.tlsutil.describe_tls",
        lambda cfg: {"mode": "self_signed", "fingerprint_sha256": "AA:BB:CC", "not_after": None},
    )
    text = page(config, state, adv("serving"))
    assert "AA:BB:CC" in text and "Certificate fingerprint" in text


def test_serving_offers_the_documented_settings(config: Config, state: _FakeState) -> None:
    text = page(config, state, adv("serving"))
    for needle in (
        "Server port",
        "Server bind address",
        "Control panel bind address",
        "Watchdog bind address",
        "HTTPS mode",
        "Unload after idle",
        "Allowed websites",
    ):
        assert needle in text, needle


# ---------------------------------------------------------------------------
# Placement / Engines / Advanced
# ---------------------------------------------------------------------------


def test_peers_says_they_arrive_with_multi_host_support(config: Config, state: _FakeState) -> None:
    text = page(config, state, adv("peers"))
    assert "Peers and the placement decision log arrive with multi-host support" in text
    assert "milestone M3" in text


def test_the_old_tab_renderers_still_work_on_their_own(config: Config, state: _FakeState) -> None:
    """Each module's ``render(ctx)`` is kept; composition added public functions beside it."""
    from studioforge.gui.tabs import dashboard, server, setup

    for module in (dashboard, server, setup):
        assert callable(module.render)
    for name in ("overview_panels", "placement_panels"):
        assert callable(getattr(dashboard, name))
    for name in (
        "checklist_cards",
        "library_cards",
        "engine_cards",
        "placement_cards",
        "serving_cards",
        "advanced_cards",
    ):
        assert callable(getattr(setup, name))
    for name in ("engine_panels", "integration_panels", "health_panels", "config_panels"):
        assert callable(getattr(server, name))


# ---------------------------------------------------------------------------
# Integrations
# ---------------------------------------------------------------------------


def test_snippets_use_the_servers_address_and_never_carry_a_token(
    config: Config, state: _FakeState
) -> None:
    from studioforge.gui.tabs.integrations import ANY_KEY, key_label, snippet_set

    config.server.port = 4321
    ctx = ctx_of(config, state)
    plain = snippet_set(ctx, ANY_KEY, "alpha-8b")
    assert plain["base_url"] == "http://127.0.0.1:4321/v1"
    assert plain["mcp_url"] == "http://127.0.0.1:4321/mcp"
    assert "<your-api-key>" in plain["openai_sdk"]
    assert "alpha-8b" in plain["curl"]
    named = snippet_set(ctx, "dispatch", None)
    assert key_label("dispatch") == "<paste-the-dispatch-key-here>"
    assert "<paste-the-dispatch-key-here>" in named["curl"]
    assert "Authorization: Bearer" in named["curl"]


def test_integrations_page_renders_every_snippet_block(config: Config, state: _FakeState) -> None:
    text = page(config, state, adv("integrations"))
    for title in (
        "OpenClaw",
        "OpenAI Python SDK",
        "Anthropic Python SDK",
        "DisPatch",
        "MCP address (for agents)",
        "Claude Code: --mcp-config file",
        "Claude Agent SDK: mcp_servers",
        "Create a key",
        "Point OpenClaw at this server",
    ):
        assert title in text, title


def test_the_key_picker_lists_active_keys_only(
    config: Config, state: _FakeState, store: AccountStore
) -> None:
    from studioforge.gui.tabs.integrations import key_choices

    record, _ = store.create_key("alpha", "read")
    store.create_key("beta", "inference")
    store.revoke_key(record.id)
    assert key_choices(ctx_of(config, state)) == ["beta"]


# ---------------------------------------------------------------------------
# Updates
# ---------------------------------------------------------------------------


class _Status:
    def __init__(self, **extra: Any) -> None:
        self.data = {
            "current_version": "9.9.9",
            "configured": True,
            "update_available": False,
            "releases_installed": ["rel-1", "rel-2"],
            "current_release": "rel-2",
            "previous_release": "rel-1",
            "checked_at": 1_800_000_000.0,
            **extra,
        }

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


class _Updater:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def status_sync(self) -> _Status:
        return _Status()

    async def check(self) -> _Status:
        self.calls.append(("check", None))
        return _Status(update_available=True, latest_version="10.0.0", latest_tag="v10.0.0")

    async def install(self, tag: Any, *, drain: Any, restart: bool) -> dict[str, Any]:
        self.calls.append(("install", (tag, restart)))
        return {"installed": tag, "message": "installed"}

    async def rollback(self) -> dict[str, Any]:
        self.calls.append(("rollback", None))
        return {"message": "rolled back"}


def test_updates_page_without_a_source_explains_how_to_turn_it_on(
    config: Config, state: _FakeState
) -> None:
    text = page(config, state, adv("about"))  # the fake state has no updater at all
    assert "Updates are not available on this server build" in text or "update.repo" in text


def test_updates_page_lists_the_installed_releases(config: Config, state: _FakeState) -> None:
    state.updater = _Updater()  # type: ignore[attr-defined]
    text = page(config, state, adv("about"))
    assert "Running version 9.9.9" in text
    assert "rel-2 (running)" in text and "Previous release: rel-1" in text


def test_updates_summary_covers_every_state() -> None:
    from studioforge.gui.tabs.updates import NOT_CONFIGURED, summary_lines

    assert NOT_CONFIGURED in summary_lines({"configured": False, "current_version": "1"})[-1] or (
        "update.repo" in " ".join(summary_lines({"configured": False, "current_version": "1"}))
    )
    assert any("failed" in line for line in summary_lines({"error": "offline"}))
    assert any(
        "10.0.0 is available" in line
        for line in summary_lines(
            {"update_available": True, "latest_version": "10.0.0", "checked_at": 1.0}
        )
    )
    assert any("up to date" in line for line in summary_lines({"checked_at": 1.0}))
    assert any("Not checked" in line for line in summary_lines({}))


async def test_check_install_and_roll_back_go_through_the_update_handlers(
    config: Config, state: _FakeState
) -> None:
    from studioforge.gui.tabs import updates

    updater = _Updater()
    state.updater = updater  # type: ignore[attr-defined]
    ctx = ctx_of(config, state)
    status = await updates.check_for_updates(ctx)
    assert status["update_available"] is True
    result = await updates.install_update(ctx, "v10.0.0")
    assert result["installed"] == "v10.0.0"
    await updates.roll_back(ctx)
    assert updater.calls == [("check", None), ("install", ("v10.0.0", True)), ("rollback", None)]
    actions = [e["action"] for e in state.audit.tail(10)]  # type: ignore[attr-defined]
    assert actions.count("panel.update.install") == 2 and "panel.update.rollback" in actions


async def test_install_and_roll_back_are_refused_for_a_remote_viewer(
    config: Config, state: _FakeState, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.gui.tabs import updates

    updater = _Updater()
    state.updater = updater  # type: ignore[attr-defined]
    monkeypatch.setattr(tabs, "viewer_host", lambda: "192.168.1.50")
    ctx = ctx_of(config, state)
    with pytest.raises(StudioForgeError):
        await updates.install_update(ctx, None)
    with pytest.raises(StudioForgeError):
        await updates.roll_back(ctx)
    assert updater.calls == []


# ---------------------------------------------------------------------------
# Logs & Metrics
# ---------------------------------------------------------------------------


def test_logs_and_metrics_has_the_collapsible_metrics_block(
    config: Config, state: _FakeState
) -> None:
    text = page(config, state, adv("logs"))
    assert "Metrics (Prometheus text)" in text
    assert "/metrics" in text


async def test_the_metrics_panel_reads_the_same_renderer_as_the_endpoint(
    config: Config, state: _FakeState, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.gui.tabs import logs

    async def fake_render(api_state: Any) -> str:
        assert api_state is state
        return "studioforge_up 1\n"

    monkeypatch.setattr("studioforge.core.metrics.render_metrics", fake_render)
    assert await logs._metrics_text(ctx_of(config, state)) == "studioforge_up 1\n"


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


REPORT = {
    "ok": False,
    "worst": "fail",
    "counts": {"ok": 1, "info": 0, "warn": 1, "fail": 1},
    "findings": [
        {"severity": "ok", "title": "Ports", "message": "free", "fix": "", "code": "ports"},
        {"severity": "fail", "title": "Engine", "message": "none", "fix": "install one"},
        {"severity": "warn", "title": "Disk", "message": "low", "fix": "free space"},
    ],
}


def test_diagnostics_orders_the_worst_findings_first_and_summarises() -> None:
    from studioforge.gui.tabs.diagnostics import ordered_findings, summary_text

    titles = [f["title"] for f in ordered_findings(REPORT)]
    assert titles == ["Engine", "Disk", "Ports"]
    summary = summary_text(REPORT)
    assert "1 failed" in summary and "1 warning(s)" in summary and "Fix the items" in summary
    assert "fine" in summary_text({"counts": {"ok": 3}, "worst": "ok"})


def test_diagnostics_page_renders_with_the_buttons(config: Config, state: _FakeState) -> None:
    text = page(config, state, adv("diagnostics"))
    assert "Run checks" in text and "Download support bundle" in text
    assert "removed" in text  # the sentence about credentials


async def test_run_checks_runs_the_doctor_off_the_loop(
    config: Config, state: _FakeState, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.gui.tabs import diagnostics

    seen: dict[str, Any] = {}

    class _Report:
        def to_dict(self) -> dict[str, Any]:
            return REPORT

    def fake_doctor(cfg: Config, **kwargs: Any) -> _Report:
        seen["config"], seen["kwargs"] = cfg, kwargs
        return _Report()

    monkeypatch.setattr(diagnostics, "run_doctor", fake_doctor)
    report = await diagnostics.run_checks(ctx_of(config, state))
    assert report["worst"] == "fail"
    assert seen["config"] is config and seen["kwargs"]["accounts"] is state.accounts  # type: ignore[attr-defined]


async def test_the_support_bundle_has_no_credentials_and_leaves_an_audit_line(
    config: Config, state: _FakeState, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.gui.tabs import diagnostics

    config.server.api_key = "legacy-key-SECRET-123456"
    _, token = state.accounts.create_key("agent", "admin")  # type: ignore[attr-defined]

    class _Report:
        def to_dict(self) -> dict[str, Any]:
            return REPORT

    monkeypatch.setattr(diagnostics, "run_doctor", lambda cfg, **kw: _Report())
    name, payload = await diagnostics.make_support_bundle(ctx_of(config, state))
    assert name.startswith("studioforge-support-") and name.endswith(".zip")
    assert payload[:2] == b"PK"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        blob = b"".join(archive.read(member) for member in archive.namelist())
    assert b"legacy-key-SECRET-123456" not in blob
    assert token.encode() not in blob
    line = next(e for e in state.audit.tail(5) if e["action"] == "panel.support_bundle")  # type: ignore[attr-defined]
    assert line["surface"] == "panel" and line["target"].endswith("bytes")


async def test_diagnostics_actions_are_refused_for_a_remote_viewer(
    config: Config, state: _FakeState, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.gui.tabs import diagnostics

    monkeypatch.setattr(tabs, "viewer_host", lambda: "192.168.1.50")
    ctx = ctx_of(config, state)
    with pytest.raises(StudioForgeError):
        await diagnostics.run_checks(ctx)
    with pytest.raises(StudioForgeError):
        await diagnostics.make_support_bundle(ctx)
