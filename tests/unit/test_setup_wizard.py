"""The first-run wizard: eight steps, runnable with no questions asked.

``SetupWizard`` is orchestration over a ``WizardServices`` object, so most of this
is a fake. ``LiveServices`` (the real thing) is exercised where it is local and
safe: the owner account, the library folders, the config edits for exposure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from studioforge.config import Config, load_config
from studioforge.setup_wizard import (
    STARTER_MODELS,
    LiveServices,
    SetupWizard,
    WizardOptions,
    detect_lmstudio_library,
    pick_starter,
)

GIB = 1024**3


class FakeServices:
    """A scripted machine. Every call is recorded so tests can assert what was and was not done."""

    def __init__(self, **overrides: Any) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.owner = False
        self.gpus = [{"index": 0, "name": "Fake 24G", "total_bytes": 24 * GIB}]
        self.engine = {"active": None, "smoke_tested": False, "pinned_tag": "b9999"}
        self.libraries: list[dict[str, Any]] = []
        self.models: list[dict[str, Any]] = []
        self.chat: dict[str, Any] = {"ok": True, "reply": "hello", "tokens_per_second": 42.0}
        self.engine_result: dict[str, Any] = {"tag": "b9999", "smoke_tested": True, "message": "ok"}
        self.fail: set[str] = set()
        self.__dict__.update(overrides)

    def _call(self, name: str, *args: Any) -> None:
        self.calls.append((name, args))
        if name in self.fail:
            raise RuntimeError(f"{name} blew up")

    def names(self) -> list[str]:
        return [n for n, _ in self.calls]

    def has_owner(self) -> bool:
        return self.owner

    def create_owner(self, username: str, password: str) -> None:
        self._call("create_owner", username)
        self.owner = True

    def detect_hardware(self) -> dict[str, Any]:
        self._call("detect_hardware")
        return {"backend": "fake", "gpus": self.gpus}

    def engine_state(self) -> dict[str, Any]:
        return self.engine

    def ensure_engine(self, tag: str | None) -> dict[str, Any]:
        self._call("ensure_engine", tag)
        return self.engine_result

    def find_libraries(self) -> list[dict[str, Any]]:
        return self.libraries

    def set_library(self, path: str) -> None:
        self._call("set_library", path)

    def ensure_server(self) -> str:
        self._call("ensure_server")
        return "http://127.0.0.1:11234"

    def library_models(self) -> list[dict[str, Any]]:
        return self.models

    def download_model(self, repo: str) -> dict[str, Any]:
        self._call("download_model", repo)
        return {"model_id": repo.split("/")[-1]}

    def load_model(self, model_id: str) -> dict[str, Any]:
        self._call("load_model", model_id)
        return {"ctx_size": 8192}

    def test_chat(self, model_id: str) -> dict[str, Any]:
        self._call("test_chat", model_id)
        return self.chat

    def base_url(self) -> str:
        return "http://127.0.0.1:11234"

    def create_admin_key(self, name: str) -> str | None:
        self._call("create_admin_key", name)
        return "sf-a-" + "k" * 43

    def apply_exposure(self, mode: str) -> dict[str, Any]:
        self._call("apply_exposure", mode)
        return {"hosts": "0.0.0.0" if mode == "lan" else "127.0.0.1"}


class ScriptedPrompter:
    def __init__(self, answers: list[Any]) -> None:
        self.answers = list(answers)
        self.asked: list[str] = []

    def _next(self, question: str) -> Any:
        self.asked.append(question)
        return self.answers.pop(0)

    def ask(self, question: str, default: str | None = None, *, secret: bool = False) -> str:
        return str(self._next(question))

    def confirm(self, question: str, default: bool = True) -> bool:
        return bool(self._next(question))

    def choose(self, question: str, options: list[str], default: int = 0) -> int:
        return int(self._next(question))


def statuses(wizard: SetupWizard) -> dict[str, str]:
    return {r.key: r.status for r in wizard.results}


def run(services: FakeServices, **options: Any) -> SetupWizard:
    wizard = SetupWizard(services, WizardOptions(**options))
    wizard.run()
    return wizard


# ---------------------------------------------------------------------------
# non-interactive
# ---------------------------------------------------------------------------


def test_a_fully_flagged_run_completes_every_step() -> None:
    services = FakeServices(engine={"active": None, "smoke_tested": False})
    services.libraries = [{"path": "C:/lm", "source": "lmstudio", "models": 3}]
    wizard = run(
        services,
        username="owner",
        password="a-long-enough-password",
        library="lmstudio",
        model="unsloth/Qwen3-8B-GGUF",
    )
    assert wizard.ok
    assert statuses(wizard) == {
        "account": "done",
        "hardware": "done",
        "engine": "done",
        "library": "done",
        "model": "done",
        "chat": "done",
        "connect": "done",
        "exposure": "done",
    }
    names = services.names()
    assert (
        names.index("create_owner") < names.index("detect_hardware") < names.index("ensure_engine")
    )
    assert ("set_library", ("C:/lm",)) in services.calls
    assert ("download_model", ("unsloth/Qwen3-8B-GGUF",)) in services.calls
    assert ("load_model", ("Qwen3-8B-GGUF",)) in services.calls
    assert ("test_chat", ("Qwen3-8B-GGUF",)) in services.calls


def test_the_summary_is_json_and_carries_the_snippets() -> None:
    wizard = run(FakeServices(), skip_account=True, skip_model=True)
    summary = wizard.summary()
    json.dumps(summary)
    connect = next(s for s in summary["steps"] if s["step"] == 7)
    assert connect["data"]["snippets"]["base_url"] == "http://127.0.0.1:11234/v1"
    assert "mcp__studioforge__*" in connect["data"]["snippets"]["mcp_agent_sdk"]


def test_local_use_needs_no_account() -> None:
    services = FakeServices()
    wizard = run(services, skip_model=True)
    assert statuses(wizard)["account"] == "skipped"
    assert "create_owner" not in services.names()


def test_an_existing_owner_is_left_alone() -> None:
    services = FakeServices(owner=True)
    wizard = run(services, username="owner", password="a-long-enough-password", skip_model=True)
    assert "create_owner" not in services.names()
    assert statuses(wizard)["account"] == "skipped"


def test_a_passing_engine_is_not_reinstalled() -> None:
    services = FakeServices(engine={"active": "b1", "smoke_tested": True})
    wizard = run(services, skip_model=True)
    assert statuses(wizard)["engine"] == "done" and "ensure_engine" not in services.names()


def test_a_failed_smoke_test_is_reported_and_the_wizard_keeps_going() -> None:
    services = FakeServices(
        engine_result={"tag": "b9999", "smoke_tested": False, "message": "driver too old"}
    )
    wizard = run(services, skip_model=True)
    assert statuses(wizard)["engine"] == "failed" and not wizard.ok
    assert statuses(wizard)["connect"] == "done"  # later independent steps still ran
    assert "driver too old" in wizard.results[2].detail


def test_a_step_that_raises_is_a_failed_step_not_a_crash() -> None:
    services = FakeServices(fail={"detect_hardware"})
    wizard = run(services, skip_model=True)
    assert statuses(wizard)["hardware"] == "failed"
    assert "blew up" in next(r for r in wizard.results if r.key == "hardware").detail
    assert len(wizard.results) == 8


def test_no_gpu_is_a_failed_hardware_step_with_a_fix() -> None:
    wizard = run(FakeServices(gpus=[]), skip_model=True)
    step = next(r for r in wizard.results if r.key == "hardware")
    assert step.status == "failed" and "driver" in step.detail


def test_a_library_flag_that_is_not_a_folder_fails_that_step(tmp_path: Path) -> None:
    wizard = run(FakeServices(), library=str(tmp_path / "missing"), skip_model=True)
    assert statuses(wizard)["library"] == "failed"
    folder = tmp_path / "models"
    folder.mkdir()
    services = FakeServices()
    wizard = run(services, library=str(folder), skip_model=True)
    assert ("set_library", (str(folder),)) in services.calls


def test_the_configured_library_wins_over_lm_studio() -> None:
    services = FakeServices()
    services.libraries = [
        {"path": "C:/mine", "source": "configured", "models": 1},
        {"path": "C:/lm", "source": "lmstudio", "models": 9},
    ]
    run(services, skip_model=True)
    assert not [c for c in services.calls if c[0] == "set_library"]


def test_a_model_already_in_the_library_is_loaded_not_downloaded() -> None:
    services = FakeServices()
    services.models = [{"id": "big", "size_bytes": 9}, {"id": "small", "size_bytes": 1}]
    run(services)
    assert "download_model" not in services.names()
    assert ("load_model", ("small",)) in services.calls  # smallest first: it will surely fit


def test_an_empty_library_downloads_a_starter_that_fits_the_smallest_gpu() -> None:
    services = FakeServices(gpus=[{"index": 0, "name": "g", "total_bytes": 8 * GIB}])
    run(services)
    repo = next(args[0] for name, args in services.calls if name == "download_model")
    assert repo == pick_starter(8.0)["repo"] == STARTER_MODELS[0]["repo"]


@pytest.mark.parametrize(
    ("vram", "repo"),
    [(4, "Qwen3-4B"), (12, "Qwen3-8B"), (24, "Qwen3-14B"), (48, "Qwen3-32B")],
)
def test_starter_choice_scales_with_the_smallest_card(vram: float, repo: str) -> None:
    assert repo in pick_starter(vram)["repo"]


def test_a_failed_chat_is_reported_with_its_error() -> None:
    services = FakeServices(chat={"ok": False, "error": "the model answered with no text"})
    services.models = [{"id": "m", "size_bytes": 1}]
    wizard = run(services)
    step = next(r for r in wizard.results if r.key == "chat")
    assert step.status == "failed" and "no text" in step.detail


def test_skip_flags_skip() -> None:
    services = FakeServices()
    wizard = run(services, skip_engine=True, skip_model=True, skip_chat=True)
    assert statuses(wizard)["engine"] == statuses(wizard)["model"] == "skipped"
    assert statuses(wizard)["chat"] == "skipped"
    assert not {"ensure_engine", "download_model", "load_model", "test_chat"} & set(
        services.names()
    )


# ---------------------------------------------------------------------------
# exposure
# ---------------------------------------------------------------------------


def test_lan_exposure_needs_the_account_and_hands_out_an_admin_key_once() -> None:
    services = FakeServices()
    wizard = run(
        services,
        username="owner",
        password="a-long-enough-password",
        skip_model=True,
        exposure="lan",
    )
    step = next(r for r in wizard.results if r.key == "exposure")
    assert step.status == "done" and step.data["mode"] == "lan"
    assert step.data["admin_key"].startswith("sf-a-")
    assert ("apply_exposure", ("lan",)) in services.calls


def test_lan_exposure_without_an_account_fails_and_changes_nothing() -> None:
    services = FakeServices()
    wizard = run(services, skip_account=True, skip_model=True, exposure="lan")
    step = next(r for r in wizard.results if r.key == "exposure")
    assert step.status == "failed" and "owner account" in step.detail
    assert "apply_exposure" not in services.names()


def test_peer_exposure_is_a_documented_placeholder() -> None:
    services = FakeServices()
    wizard = run(services, skip_model=True, exposure="peer")
    step = next(r for r in wizard.results if r.key == "exposure")
    assert step.status == "manual" and "M3" in step.detail
    assert "apply_exposure" not in services.names()


def test_an_unknown_exposure_is_refused() -> None:
    wizard = run(FakeServices(), skip_model=True, exposure="everywhere")
    assert statuses(wizard)["exposure"] == "failed"


# ---------------------------------------------------------------------------
# interactive
# ---------------------------------------------------------------------------


def test_the_interactive_run_asks_and_uses_the_answers() -> None:
    services = FakeServices(engine={"active": None, "smoke_tested": False, "pinned_tag": "b1"})
    services.libraries = [{"path": "C:/lm", "source": "lmstudio", "models": 2}]
    prompter = ScriptedPrompter(
        ["boss", "a-long-enough-password", True, True, False, 1]
        # username, password, engine?, use LM Studio library?, download starter? (no), exposure
    )
    wizard = SetupWizard(services, WizardOptions(interactive=True), prompter=prompter)
    wizard.run()
    assert ("create_owner", ("boss",)) in services.calls
    assert ("set_library", ("C:/lm",)) in services.calls
    assert statuses(wizard)["model"] == "skipped"  # declined the starter
    assert ("apply_exposure", ("lan",)) in services.calls  # chose option 1 = LAN
    assert any("password" in q.lower() for q in prompter.asked)


def test_declining_the_engine_download_skips_it() -> None:
    services = FakeServices(engine={"active": None, "smoke_tested": False, "pinned_tag": "b1"})
    prompter = ScriptedPrompter(["", "", False, False, 0])
    wizard = SetupWizard(
        services,
        WizardOptions(interactive=True, skip_model=True, skip_account=True),
        prompter=prompter,
    )
    wizard.run()
    assert statuses(wizard)["engine"] == "skipped" and "ensure_engine" not in services.names()


# ---------------------------------------------------------------------------
# LiveServices on a scratch machine
# ---------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fake_home = tmp_path / "home"
    (fake_home / ".lmstudio" / "models" / "pub" / "repo").mkdir(parents=True)
    (fake_home / ".lmstudio" / "models" / "pub" / "repo" / "m.gguf").write_bytes(b"GGUF")
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    return fake_home


def scratch_config(tmp_path: Path) -> Config:
    config = Config(
        data_dir=tmp_path / "data",
        gui={"port": 18080},
        watchdog={"port": 11235},
        server={"port": 11234},
        logging={"level": "ERROR"},
    )
    config.ensure_dirs()
    config.save(tmp_path / "data" / "config.yaml")
    config.source_path = tmp_path / "data" / "config.yaml"
    return config


def test_lm_studio_library_is_detected_in_the_users_home(home: Path) -> None:
    found = detect_lmstudio_library()
    assert [p.resolve() for p in found] == [(home / ".lmstudio" / "models").resolve()]


def test_live_find_libraries_reports_configured_and_lm_studio(tmp_path: Path, home: Path) -> None:
    config = scratch_config(tmp_path)
    live = LiveServices(config)
    libs = live.find_libraries()
    assert {lib["source"] for lib in libs} == {"lmstudio"} and libs[0]["models"] == 1
    mine = tmp_path / "mine"
    mine.mkdir()
    live.set_library(str(mine))
    again = live.find_libraries()
    assert [lib["source"] for lib in again][0] == "configured"
    assert load_config(tmp_path / "data" / "config.yaml").models.dir == mine  # persisted


def test_live_exposure_edits_the_config_and_local_reverts_it(tmp_path: Path) -> None:
    config = scratch_config(tmp_path)
    live = LiveServices(config)
    result = live.apply_exposure("lan")
    assert result["restart_required"] is True
    saved = load_config(tmp_path / "data" / "config.yaml")
    assert saved.server.host == saved.gui.host == saved.watchdog.host == "0.0.0.0"
    assert saved.server.tls.mode == "self_signed"
    live.apply_exposure("local")
    saved = load_config(tmp_path / "data" / "config.yaml")
    assert saved.server.host == saved.gui.host == saved.watchdog.host == "127.0.0.1"


def test_live_owner_and_admin_key(tmp_path: Path) -> None:
    live = LiveServices(scratch_config(tmp_path))
    assert not live.has_owner()
    live.create_owner("owner", "a-long-enough-password")
    assert live.has_owner()
    token = live.create_admin_key("setup-admin")
    assert token and token.startswith("sf-a-")
    assert (tmp_path / "data" / "cli.key").is_file()  # local tools keep working afterwards


def test_the_cli_runs_the_wizard_non_interactively(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.__main__ import app as cli

    monkeypatch.setenv("SF_GPU_PROBE", "null")
    monkeypatch.setenv("SF_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("WIZARD_PW", "a-long-enough-password")
    scratch_config(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "setup",
            "--non-interactive",
            "--username",
            "owner",
            "--password-env",
            "WIZARD_PW",
            "--library",
            "lmstudio",
            "--skip-engine",
            "--skip-model",
            "--skip-chat",
            "--json",
            "--config",
            str(tmp_path / "data" / "config.yaml"),
        ],
    )
    summary = json.loads(result.output[result.output.index("{") :])
    by_key = {s["key"]: s["status"] for s in summary["steps"]}
    assert by_key["account"] == "done" and by_key["library"] == "done"
    assert by_key["engine"] == "skipped" and by_key["exposure"] == "done"
    # no GPU under the null probe: that step fails honestly, and so the run says so
    assert by_key["hardware"] == "failed" and result.exit_code == 1
    assert "a-long-enough-password" not in result.output


# ---------------------------------------------------------------------------
# LiveServices over HTTP (a fake server behind the client)
# ---------------------------------------------------------------------------


def live_with(tmp_path: Path, routes: dict[tuple[str, str], Any]) -> tuple[LiveServices, list[Any]]:
    import httpx

    from studioforge.cli_client import LocalClient

    seen: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, json.loads(request.content or b"null")))
        reply = routes.get((request.method, request.url.path))
        if callable(reply):
            reply = reply()
        if reply is None:
            return httpx.Response(404, json={"error": {"message": "nope", "code": "x"}})
        return httpx.Response(200, json=reply)

    config = scratch_config(tmp_path)
    live = LiveServices(config)
    live._client = LocalClient(config, transport=httpx.MockTransport(handler))
    return live, seen


def test_live_download_waits_for_the_group_then_finds_the_model_by_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import studioforge.setup_wizard as module

    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    polls = iter(
        [
            {
                "downloads": [
                    {"group_id": "g", "status": "running", "downloaded_bytes": 1, "total_bytes": 4}
                ]
            },
            {
                "downloads": [
                    {
                        "group_id": "g",
                        "status": "completed",
                        "downloaded_bytes": 4,
                        "total_bytes": 4,
                    }
                ]
            },
        ]
    )
    live, seen = live_with(
        tmp_path,
        {
            ("POST", "/api/downloads"): {"group_id": "g"},
            ("GET", "/api/downloads"): lambda: next(polls),
            ("POST", "/api/models/scan"): {},
            ("GET", "/api/models"): {
                "models": [
                    {"id": "other/thing", "repo": "other/thing"},
                    {
                        "id": "unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf",
                        "repo": "unsloth/Qwen3-4B-GGUF",
                    },
                ]
            },
        },
    )
    assert live.download_model("unsloth/Qwen3-4B-GGUF") == {
        "model_id": "unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"
    }
    assert [m for m, p, _ in seen if p == "/api/models/scan"]


def test_live_download_failure_is_an_error_with_the_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import studioforge.setup_wizard as module

    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    live, _ = live_with(
        tmp_path,
        {
            ("POST", "/api/downloads"): {"group_id": "g"},
            ("GET", "/api/downloads"): {"downloads": [{"group_id": "g", "status": "failed"}]},
        },
    )
    with pytest.raises(RuntimeError, match="failed"):
        live.download_model("a/b")


def test_live_load_uses_the_servers_own_recommendation_and_chat_reads_the_test(
    tmp_path: Path,
) -> None:
    live, seen = live_with(
        tmp_path,
        {
            ("POST", "/api/models/m/load-recommended"): {"ctx_size": 16384},
            ("POST", "/api/models/m/test"): {"ok": True, "text": "hi", "tokens_per_second": 55.5},
        },
    )
    assert live.load_model("m") == {"ctx_size": 16384}
    assert ("POST", "/api/models/m/load-recommended", {"ctx_size": 8192}) in seen
    assert live.test_chat("m") == {
        "ok": True,
        "reply": "hi",
        "tokens_per_second": 55.5,
        "error": None,
    }
    failing, _ = live_with(tmp_path / "b", {("POST", "/api/models/m/test"): {"ok": False}})
    assert failing.test_chat("m")["ok"] is False


def test_live_library_models_skips_virtual_ones(tmp_path: Path) -> None:
    live, _ = live_with(
        tmp_path,
        {("GET", "/api/models"): {"models": [{"id": "a"}, {"id": "v", "is_virtual": True}]}},
    )
    assert [m["id"] for m in live.library_models()] == ["a"]


def test_ensure_server_starts_one_only_when_none_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import studioforge.server_control as control

    live = LiveServices(scratch_config(tmp_path))
    started: list[bool] = []
    monkeypatch.setattr(
        control, "server_status", lambda config, **kw: control.ServerStatus(True, "http://up")
    )
    monkeypatch.setattr(control, "start_server", lambda config, **kw: started.append(True))
    assert live.ensure_server() == "http://up" and not started
    monkeypatch.setattr(
        control, "server_status", lambda config, **kw: control.ServerStatus(False, "http://down")
    )
    monkeypatch.setattr(
        control, "start_server", lambda config, **kw: control.ServerStatus(True, "http://now")
    )
    assert live.ensure_server() == "http://now"
