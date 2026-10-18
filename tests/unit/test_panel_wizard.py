"""The first-run wizard page: ``/setup``, the in-process services, and one step at a time.

``PanelServices`` is the wizard's ``WizardServices`` over the running server's own objects.
These tests give it fakes for the engine, downloader, manager and registry (nothing here
touches a GPU, a port or the network), and drive ``SetupWizard`` through it exactly as the
page does: one step at a time on a worker thread, with async work handed back to the loop.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_type_hints

import pytest
from fastapi.testclient import TestClient

from studioforge.api.auth import GUARD
from studioforge.auditlog import AuditLog
from studioforge.config import Config
from studioforge.core.accounts import AccountStore
from studioforge.db import Database
from studioforge.gui import tabs
from studioforge.gui.app import create_gui_app
from studioforge.gui.tabs import setup_wizard_page as page_mod
from studioforge.gui.tabs.setup_wizard_page import (
    PEER_LABEL,
    STEPS,
    PanelServices,
    adopt_model,
    drive_step,
    scrub_secret,
    starter_for,
)
from studioforge.setup_wizard import (
    STARTER_MODELS,
    SetupWizard,
    StepResult,
    WizardOptions,
    WizardServices,
)
from tests.unit.test_gui import _FakeState, make_record

GIB = 1024**3
PASSWORD = "correct horse battery"


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


class Engines:
    """A scripted engine manager."""

    def __init__(self, *, ok: bool = True, active: str | None = None) -> None:
        self.ok = ok
        self._active = SimpleNamespace(tag=active, smoke_tested=True) if active else None
        self.calls: list[tuple[str, Any]] = []

    def active(self) -> Any:
        return self._active

    async def install(self, tag: str, *, force: bool, activate: bool, progress: Any = None) -> Any:
        self.calls.append(("install", (tag, force, activate)))
        if progress:
            progress("download", 0.5)
        return SimpleNamespace(tag=tag)

    async def smoke_test(self, tag: str) -> tuple[bool, str]:
        self.calls.append(("smoke_test", tag))
        return self.ok, "ran" if self.ok else "could not start"

    async def smoke_check(self, tag: str) -> tuple[bool, str, str]:
        ok, detail = await self.smoke_test(tag)
        return ok, "passed" if ok else "failed", detail


class Manager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.threads: list[int] = []

    async def load_recommended(self, model_id: str, ctx_size: int, **kwargs: Any) -> Any:
        self.threads.append(threading.get_ident())
        self.calls.append(("load", (model_id, ctx_size, kwargs)))
        return SimpleNamespace()

    async def test_model(self, model_id: str, prompt: Any, *, keep_loaded: bool) -> dict[str, Any]:
        self.calls.append(("test", (model_id, prompt, keep_loaded)))
        return {"ok": True, "text": "Hello!", "tokens_per_second": 55.0}


class Downloader:
    def __init__(self, statuses: list[str]) -> None:
        self._statuses = list(statuses)
        self.canceled: list[str] = []
        self.done = 0

    def group(self, group_id: str) -> list[Any]:
        return [SimpleNamespace(downloaded_bytes=self.done, total_bytes=100)]

    def group_status(self, group_id: str) -> str:
        self.done += 40
        return self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]

    async def cancel(self, group_id: str, *, delete_partial: bool = True) -> None:
        self.canceled.append(group_id)


class Registry:
    def __init__(self, records: list[Any] | None = None) -> None:
        self.records = list(records or [])
        self.scans = 0

    def all(self) -> list[Any]:
        return list(self.records)

    def scan(self, *, force: bool = False) -> None:
        self.scans += 1


@pytest.fixture
def state(config: Config, store: AccountStore) -> _FakeState:
    fake = _FakeState(config)
    fake.accounts = store  # type: ignore[attr-defined]
    fake.audit = AuditLog(config.logs_dir)  # type: ignore[attr-defined]
    fake.manager = Manager()
    fake.engine_manager = Engines()
    fake.registry = Registry()  # type: ignore[assignment]
    return fake


@pytest.fixture
def ctx(config: Config, state: _FakeState) -> tabs.GuiContext:
    return tabs.GuiContext(config=config, api_state=state)


def services(ctx: tabs.GuiContext, **kwargs: Any) -> PanelServices:
    return PanelServices(ctx, sleep=lambda _s: None, poll_s=0, **kwargs)


def audit_actions(state: _FakeState) -> list[str]:
    return [e["action"] for e in state.audit.tail(50)]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def test_the_setup_page_shows_the_eight_steps(config: Config, state: _FakeState) -> None:
    with TestClient(create_gui_app(config, api_state=state)) as client:
        response = client.get("/setup")
    assert response.status_code == 200
    text = response.text
    assert "could not be rendered" not in text
    for title in (
        "Owner account",
        "Detect hardware",
        "Engine",
        "Model libraries",
        "Starter model",
        "Test chat",
        "Connect your tools",
        "Who can reach it",
    ):
        assert title in text, title
    for control in ("Create owner account", "Detect hardware", "Back", "Next", "Skip", "Finish"):
        assert control in text, control
    assert "Only this computer (recommended)" in text
    assert PEER_LABEL in text and "Coming in a later release" in text


def test_the_setup_page_is_behind_the_sign_in_gate(
    config: Config, state: _FakeState, store: AccountStore
) -> None:
    store.create_key("agent", "read")
    with TestClient(create_gui_app(config, api_state=state)) as client:
        assert client.get("/setup", headers={"accept": "text/html"}, follow_redirects=False)
        refused = client.get("/setup", follow_redirects=False)
        assert refused.status_code == 401
        _, token = store.create_key("ops", "admin")
        allowed = client.get("/setup", headers={"Authorization": f"Bearer {token}"})
    assert allowed.status_code == 200 and "Owner account" in allowed.text


def test_the_step_keys_mirror_the_wizards_own_order() -> None:
    assert [key for key, _ in STEPS] == [
        "account",
        "hardware",
        "engine",
        "library",
        "model",
        "chat",
        "connect",
        "exposure",
    ]
    assert STEPS[0][1].startswith("1. ") and STEPS[-1][1].startswith("8. ")


def test_panel_services_implements_every_method_of_the_protocol() -> None:
    for name in get_type_hints(WizardServices):  # attributes, if any
        assert hasattr(PanelServices, name)
    members = [
        n
        for n in vars(WizardServices)
        if not n.startswith("_") and callable(vars(WizardServices)[n])
    ]
    assert len(members) >= 15
    for name in members:
        assert callable(getattr(PanelServices, name)), name


# ---------------------------------------------------------------------------
# PanelServices
# ---------------------------------------------------------------------------


def test_owner_account_through_the_store(ctx: tabs.GuiContext, state: _FakeState) -> None:
    svc = services(ctx, actor="owner:alice")
    assert svc.has_owner() is False
    svc.create_owner("alice", PASSWORD)
    assert svc.has_owner() is True
    line = next(e for e in state.audit.tail(5) if e["action"] == "panel.account.create")  # type: ignore[attr-defined]
    assert line["actor"] == "owner:alice" and line["target"] == "alice (owner)"
    assert PASSWORD not in str(state.audit.tail(5))  # type: ignore[attr-defined]
    with pytest.raises(Exception, match="at least"):
        svc.create_owner("bob", "short")


def test_hardware_comes_from_the_probe(ctx: tabs.GuiContext) -> None:
    info = services(ctx).detect_hardware()
    assert info["backend"] == "fake"
    assert info["gpus"][0]["name"] == "RTX 5090" and info["gpus"][0]["total_bytes"] == 32 * GIB


def test_starter_follows_the_smallest_gpu(ctx: tabs.GuiContext) -> None:
    chosen = starter_for(services(ctx))  # one 32 GiB card
    assert chosen["repo"] == next(e["repo"] for e in STARTER_MODELS if e["min_vram_gib"] == 20.0)


def test_engine_state_and_install(
    ctx: tabs.GuiContext, state: _FakeState, monkeypatch: pytest.MonkeyPatch
) -> None:
    activated: list[str] = []

    async def fake_activate(request: Any, tag: str) -> dict[str, Any]:
        activated.append(tag)
        return {"tag": tag}

    monkeypatch.setattr("studioforge.api.mgmt_routes.engine_activate", fake_activate)
    svc = services(ctx)
    assert svc.engine_state() == {
        "active": None,
        "smoke_tested": False,
        "pinned_tag": ctx.config.engine.pinned_tag,
    }
    result = svc.ensure_engine(None)
    assert result == {"tag": ctx.config.engine.pinned_tag, "smoke_tested": True, "message": "ran"}
    assert state.engine_manager.calls[0] == (  # type: ignore[attr-defined]
        "install",
        (ctx.config.engine.pinned_tag, False, False),
    )
    assert activated == [ctx.config.engine.pinned_tag]
    assert svc.progress["phase"] == "smoke test"
    assert "panel.engine.install" in audit_actions(state)


def test_a_failed_smoke_test_is_not_activated(
    ctx: tabs.GuiContext, state: _FakeState, monkeypatch: pytest.MonkeyPatch
) -> None:
    state.engine_manager = Engines(ok=False)
    monkeypatch.setattr(
        "studioforge.api.mgmt_routes.engine_activate",
        lambda *a, **k: pytest.fail("activated an engine that did not start"),
    )
    result = services(ctx).ensure_engine("b1")
    assert result["smoke_tested"] is False and result["message"] == "could not start"


def test_libraries_are_listed_and_the_choice_is_saved_through_set_config(
    ctx: tabs.GuiContext, state: _FakeState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (ctx.config.models.dir / "a.gguf").write_bytes(b"x")  # type: ignore[operator]
    lmstudio = tmp_path / "lmstudio"
    lmstudio.mkdir()
    (lmstudio / "b.gguf").write_bytes(b"x")
    monkeypatch.setattr("studioforge.setup_wizard.detect_lmstudio_library", lambda: [lmstudio])
    svc = services(ctx)
    found = svc.find_libraries()
    assert [(f["source"], f["models"]) for f in found] == [("configured", 1), ("lmstudio", 1)]
    svc.set_library(str(lmstudio))
    assert Path(ctx.config.models.dir) == lmstudio  # type: ignore[arg-type]
    assert "panel.config.set" in audit_actions(state)
    line = next(e for e in state.audit.tail(5) if e["action"] == "panel.config.set")  # type: ignore[attr-defined]
    assert line["target"] == "models.dir"


def test_library_models_skip_personas(ctx: tabs.GuiContext, state: _FakeState) -> None:
    state.registry = Registry(  # type: ignore[assignment]
        [make_record("alpha"), make_record("persona", virtual=True)]
    )
    rows = services(ctx).library_models()
    assert [r["id"] for r in rows] == ["alpha"]
    assert rows[0]["size_bytes"] > 0


def test_a_download_is_polled_to_completion_and_matched_in_the_library(
    ctx: tabs.GuiContext, state: _FakeState, monkeypatch: pytest.MonkeyPatch
) -> None:
    state.downloader = Downloader(["running", "running", "completed"])
    state.registry = Registry([make_record("Qwen3-4B-Q4_K_M")])  # type: ignore[assignment]
    asked: dict[str, Any] = {}

    async def fake_start(request: Any, **kwargs: Any) -> dict[str, Any]:
        asked.update(kwargs)
        return {"group_id": "g1", "files": ["Qwen3-4B-Q4_K_M.gguf"], "quant": "Q4_K_M"}

    monkeypatch.setattr("studioforge.api.mgmt_routes.start_download", fake_start)
    svc = services(ctx)
    result = svc.download_model("unsloth/Qwen3-4B-GGUF")
    assert result == {"model_id": "Qwen3-4B-Q4_K_M"}
    assert asked == {
        "repo_id": "unsloth/Qwen3-4B-GGUF",
        "quant": None,
        "include_mmproj": True,
        "force": False,
    }
    assert state.registry.scans == 1  # type: ignore[attr-defined]
    assert svc.progress["phase"] == "indexing"
    assert "panel.download.start" in audit_actions(state)


@pytest.mark.parametrize("status", ["failed", "canceled"])
def test_a_download_that_ends_badly_says_where_to_look(
    ctx: tabs.GuiContext, state: _FakeState, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    state.downloader = Downloader(["running", status])

    async def fake_start(request: Any, **kwargs: Any) -> dict[str, Any]:
        return {"group_id": "g1", "files": []}

    monkeypatch.setattr("studioforge.api.mgmt_routes.start_download", fake_start)
    with pytest.raises(RuntimeError, match=f"download of owner/x {status}"):
        services(ctx).download_model("owner/x")


def test_a_download_can_be_stopped_from_the_page(
    ctx: tabs.GuiContext, state: _FakeState, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloader = Downloader(["running"])
    state.downloader = downloader

    async def fake_start(request: Any, **kwargs: Any) -> dict[str, Any]:
        return {"group_id": "g9", "files": []}

    monkeypatch.setattr("studioforge.api.mgmt_routes.start_download", fake_start)
    svc = services(ctx)
    svc._sleep = lambda _s: svc.cancelled.set()  # the user presses Stop during the first wait
    with pytest.raises(RuntimeError, match="was canceled"):
        svc.download_model("owner/x")
    assert downloader.canceled == ["g9"]


def test_a_missing_downloader_is_explained(ctx: tabs.GuiContext) -> None:
    with pytest.raises(RuntimeError, match="Downloads are not available"):
        services(ctx).download_model("owner/x")


def test_load_and_test_chat_use_the_manager(ctx: tabs.GuiContext, state: _FakeState) -> None:
    svc = services(ctx)
    assert svc.load_model("alpha") == {"ctx_size": ctx.config.models.default_ctx}
    reply = svc.test_chat("alpha")
    assert reply == {"ok": True, "reply": "Hello!", "tokens_per_second": 55.0, "error": None}
    calls = state.manager.calls  # type: ignore[attr-defined]
    assert calls[0] == ("load", ("alpha", ctx.config.models.default_ctx, {"source": "gui"}))
    assert calls[1] == ("test", ("alpha", None, True))


def test_base_url_is_the_servers_own_address(ctx: tabs.GuiContext) -> None:
    ctx.config.server.port = 4321
    svc = services(ctx)
    assert svc.base_url() == "http://127.0.0.1:4321"
    assert svc.ensure_server() == "http://127.0.0.1:4321"


def test_exposure_local_is_a_no_op_on_a_loopback_install(
    ctx: tabs.GuiContext, state: _FakeState
) -> None:
    result = services(ctx).apply_exposure("local")
    assert result["hosts"] == "127.0.0.1"
    assert "panel.config.set" not in audit_actions(state)


def test_exposure_lan_binds_the_network_with_https_and_says_restart(
    ctx: tabs.GuiContext, state: _FakeState
) -> None:
    result = services(ctx).apply_exposure("lan")
    assert result == {"hosts": "0.0.0.0", "tls": "self_signed", "restart_required": True}
    line = next(e for e in state.audit.tail(5) if e["action"] == "panel.config.set")  # type: ignore[attr-defined]
    assert line["target"] == "gui.host, server.host, server.tls.mode, watchdog.host"
    # Back to local-only writes the loopback addresses again.
    ctx.config.server.host = ctx.config.gui.host = ctx.config.watchdog.host = "0.0.0.0"
    assert services(ctx).apply_exposure("local")["restart_required"] is True


def test_the_admin_key_is_created_once_and_audited_without_its_value(
    ctx: tabs.GuiContext, state: _FakeState, store: AccountStore
) -> None:
    token = services(ctx).create_admin_key("setup-admin")
    assert token and token.startswith("sf-a-")
    assert store.authenticate(token, count=False).scope == "admin"  # type: ignore[union-attr]
    assert token not in str(state.audit.tail(5))  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Threads and the loop
# ---------------------------------------------------------------------------


async def test_async_work_runs_on_the_servers_loop_not_the_worker_thread(
    ctx: tabs.GuiContext, state: _FakeState
) -> None:
    loop = asyncio.get_running_loop()
    svc = PanelServices(ctx, loop=loop)
    loop_thread = threading.get_ident()
    await asyncio.to_thread(svc.load_model, "alpha")
    assert state.manager.threads == [loop_thread]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Driving the wizard one step at a time
# ---------------------------------------------------------------------------


def make_wizard(ctx: tabs.GuiContext, **options: Any) -> tuple[SetupWizard, PanelServices]:
    svc = services(ctx)
    return SetupWizard(svc, WizardOptions(interactive=False, **options)), svc


def test_each_step_works_on_its_own_and_in_any_order(
    ctx: tabs.GuiContext, state: _FakeState, store: AccountStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    wizard, _ = make_wizard(ctx)
    # Steps 6 and 7 first: nothing was loaded, and that is a clear "skipped", not a crash.
    assert drive_step(wizard, "chat").status == "skipped"
    assert drive_step(wizard, "connect").status == "done"
    # The owner account is optional.
    skipped = drive_step(wizard, "account")
    assert skipped.status == "skipped" and "local-only" in skipped.detail
    assert not store.has_owner()
    hardware = drive_step(wizard, "hardware")
    assert hardware.status == "done" and "1 GPU(s)" in hardware.detail


def test_creating_the_owner_from_the_form_values(ctx: tabs.GuiContext, store: AccountStore) -> None:
    wizard, _ = make_wizard(ctx, username="alice", password=PASSWORD)
    result = drive_step(wizard, "account")
    assert (result.status, result.data["username"]) == ("done", "alice")
    assert store.has_owner()
    again = drive_step(wizard, "account")
    assert again.status == "skipped" and "already exists" in again.detail


def test_a_step_that_raises_is_a_failed_result_with_the_reason(ctx: tabs.GuiContext) -> None:
    wizard, _ = make_wizard(ctx, username="alice", password="short")
    result = drive_step(wizard, "account")
    assert result.status == "failed"
    assert "at least 10 characters" in result.detail
    assert wizard.results[-1] is result


def test_the_library_step_rejects_a_folder_that_does_not_exist(
    ctx: tabs.GuiContext, tmp_path: Path
) -> None:
    wizard, _ = make_wizard(ctx, library=str(tmp_path / "nope"))
    result = drive_step(wizard, "library")
    assert result.status == "failed" and "does not exist" in result.detail


def test_the_peer_exposure_is_not_available_and_changes_nothing(
    ctx: tabs.GuiContext, state: _FakeState
) -> None:
    wizard, _ = make_wizard(ctx, exposure="peer")
    result = drive_step(wizard, "exposure")
    assert result.status == "manual" and "milestone M3" in result.detail
    assert "panel.config.set" not in audit_actions(state)


def test_network_exposure_needs_the_owner_account(
    ctx: tabs.GuiContext, store: AccountStore
) -> None:
    wizard, _ = make_wizard(ctx, exposure="lan")
    refused = drive_step(wizard, "exposure")
    assert refused.status == "failed" and "needs an owner account" in refused.detail
    store.create_account("alice", PASSWORD)
    done = drive_step(wizard, "exposure")
    assert done.status == "done"
    key = scrub_secret(done)
    assert key and key.startswith("sf-a-")
    assert "admin_key" not in done.data  # removed before anything else can see it
    assert scrub_secret(done) is None


def test_adopting_a_model_lets_the_chat_step_run(ctx: tabs.GuiContext, state: _FakeState) -> None:
    wizard, _ = make_wizard(ctx)
    adopt_model(wizard, "alpha")
    result = drive_step(wizard, "chat")
    assert result.status == "done" and "tok/s" in result.detail
    assert result.data["reply"] == "Hello!"


async def test_the_whole_wizard_runs_through_the_panel_services(
    ctx: tabs.GuiContext, state: _FakeState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same eight steps ``studioforge setup`` runs, here against the live objects."""
    state.engine_manager = Engines(active="b1")  # already installed and checked
    state.registry = Registry([make_record("alpha")])  # type: ignore[assignment]
    library = tmp_path / "lib"
    library.mkdir()
    svc = PanelServices(ctx, loop=asyncio.get_running_loop())
    wizard = SetupWizard(
        svc,
        WizardOptions(
            interactive=False,
            username="alice",
            password=PASSWORD,
            library=str(library),
            model="alpha",
        ),
    )
    results: list[StepResult] = await asyncio.to_thread(wizard.run)
    by_key = {r.key: r.status for r in results}
    assert by_key == {
        "account": "done",
        "hardware": "done",
        "engine": "done",
        "library": "done",
        "model": "done",
        "chat": "done",
        "connect": "done",
        "exposure": "done",
    }
    assert wizard.ok
    assert Path(ctx.config.models.dir) == library  # type: ignore[arg-type]
    connect = next(r for r in results if r.key == "connect")
    assert connect.data["snippets"]["base_url"].endswith("/v1")
    assert PASSWORD not in str(state.audit.tail(50))  # type: ignore[attr-defined]


def test_the_page_module_exports_what_the_app_registers() -> None:
    assert callable(page_mod.render)
