"""System-tray app.

pystray needs a real desktop session and a Win32 message pump, so nothing here
runs an icon. What *is* tested is everything that decides what the user sees or
what the server is asked to do: the drawn icon, the status line, menu shape and
enabled-predicates, URL derivation, the single-instance guard, and the exact
method/path/body of every API call a menu item makes.

The API-key assertions matter more than they look: the key is a credential the
tray holds in memory and must put in exactly one place (an Authorization
header) and no other -- not a URL, not a balloon, not the clipboard.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

pystray = pytest.importorskip("pystray", reason="the tray needs pystray")

from studioforge.config import Config  # noqa: E402
from studioforge.core import autostart  # noqa: E402
from studioforge.tray import tray_app  # noqa: E402
from studioforge.tray.tray_app import (  # noqa: E402
    STATE_CRASHED,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_STOPPED,
    ApiResult,
    ServerSnapshot,
    TrayApp,
    acquire_single_instance,
    api_docs_url,
    control_panel_url,
    fallback_mcp_url,
    make_icon_image,
    server_command,
    snapshot_from_status,
    status_text,
)

# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeCall:
    method: str
    path: str
    payload: dict[str, Any] | None = None


@dataclass
class FakeClient:
    """Records every call and answers from a canned table."""

    replies: dict[str, ApiResult] = field(default_factory=dict)
    calls: list[FakeCall] = field(default_factory=list)
    default: ApiResult = field(default_factory=lambda: ApiResult(True, {}))
    #: Absolute-URL replies (the watchdog's /health lives on another port).
    urls: dict[str, ApiResult] = field(default_factory=dict)

    def get(self, path: str, *, timeout: float = 10.0) -> ApiResult:
        self.calls.append(FakeCall("GET", path))
        return self.replies.get(path, self.default)

    def get_url(self, url: str, *, timeout: float = 10.0) -> ApiResult:
        self.calls.append(FakeCall("GET", url))
        return self.urls.get(url, ApiResult(False, error="not answering"))

    def post(
        self, path: str, payload: dict[str, Any] | None = None, *, timeout: float = 60.0
    ) -> ApiResult:
        self.calls.append(FakeCall("POST", path, payload))
        return self.replies.get(path, self.default)


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config(data_dir=tmp_path / "data")
    cfg.ensure_dirs()
    cfg.save(cfg.data_dir / "config.yaml")
    cfg.source_path = cfg.data_dir / "config.yaml"
    return cfg


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def app(config: Config, client: FakeClient) -> TrayApp:
    """A TrayApp with no pystray.Icon: constructing one needs a message pump."""
    return TrayApp(config, client=client, create_icon=False)


def menu_items(app: TrayApp) -> list[Any]:
    """Real menu items, separators dropped."""
    return [i for i in app._build_menu().items if i is not pystray.Menu.SEPARATOR]


def labels(app: TrayApp) -> list[str]:
    return [i.text for i in menu_items(app)]


def item_named(app: TrayApp, prefix: str) -> Any:
    for item in menu_items(app):
        if item.text.startswith(prefix):
            return item
    raise AssertionError(f"no menu item starting with {prefix!r} in {labels(app)}")


# ---------------------------------------------------------------------------
# Icon drawing
# ---------------------------------------------------------------------------


def test_icon_is_rgba_at_the_requested_size() -> None:
    image = make_icon_image(True, size=32)
    assert image.mode == "RGBA"
    assert image.size == (32, 32)


def test_running_and_stopped_icons_differ() -> None:
    """The whole point of the status dot: the tray shows state at a glance."""
    running = make_icon_image(True).tobytes()
    stopped = make_icon_image(False).tobytes()
    assert running != stopped


def test_icon_survives_a_16px_render() -> None:
    """16px is the size Windows actually paints in the notification area."""
    assert make_icon_image(False, size=16).size == (16, 16)
    assert make_icon_image(True, size=16).tobytes() != make_icon_image(False, size=16).tobytes()


# ---------------------------------------------------------------------------
# Status line
# ---------------------------------------------------------------------------


def test_status_line_reports_models_and_free_vram() -> None:
    snapshot = ServerSnapshot(reachable=True, loaded_models=2, free_vram_bytes=44_231_000_000)
    text = status_text(STATE_RUNNING, snapshot)
    assert text.startswith("Running")
    assert "2 models loaded" in text
    assert "41.2 GiB free" in text


def test_status_line_uses_the_singular_for_one_model() -> None:
    snapshot = ServerSnapshot(reachable=True, loaded_models=1, free_vram_bytes=2**30)
    assert "1 model loaded" in status_text(STATE_RUNNING, snapshot)


def test_status_line_when_the_server_is_unreachable() -> None:
    text = status_text(STATE_RUNNING, ServerSnapshot(reachable=False))
    assert "Running" in text
    assert "not answering" in text
    assert "GiB" not in text


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (STATE_STOPPED, "Stopped"),
        (STATE_STARTING, "Starting..."),
        (STATE_CRASHED, "Crashed"),
    ],
)
def test_status_line_for_each_state(state: str, expected: str) -> None:
    assert expected in status_text(state, ServerSnapshot())


def test_snapshot_sums_free_vram_across_gpus() -> None:
    snapshot = snapshot_from_status(
        {
            "loaded": [{"model_id": "a"}, {"model_id": "b"}, {"model_id": "c"}],
            "gpus": [{"free_bytes": 2**30}, {"free_bytes": 3 * 2**30}],
        }
    )
    assert snapshot.reachable
    assert snapshot.loaded_models == 3
    assert snapshot.free_vram_bytes == 4 * 2**30


def test_snapshot_tolerates_a_payload_missing_everything() -> None:
    """A grown/renamed status field must not blank the only visible readout."""
    snapshot = snapshot_from_status({})
    assert snapshot.reachable
    assert snapshot.loaded_models == 0
    assert snapshot.free_vram_bytes == 0


# ---------------------------------------------------------------------------
# Menu construction
# ---------------------------------------------------------------------------


def test_menu_has_the_expected_items(app: TrayApp) -> None:
    found = labels(app)
    for expected in (
        "Open control panel",
        "Open API docs",
        "Open logs folder",
        "Open models folder",
        "Copy MCP URL",
        "Copy MCP PIN",
        "Start at login",
        "Start server",
        "Stop server",
        "Quit",
    ):
        assert expected in found, found
    assert any(t.startswith("Unload all models") for t in found), found
    assert any(t.startswith("Restart engines") for t in found), found
    assert any(t.startswith("Restart server") for t in found), found


def test_the_two_restarts_are_labelled_distinguishably(app: TrayApp) -> None:
    engines = item_named(app, "Restart engines").text
    server = item_named(app, "Restart server").text
    assert engines != server
    assert "API stays up" in engines
    assert "whole process" in server


def test_status_line_is_the_first_item_and_disabled(app: TrayApp) -> None:
    app.state = STATE_RUNNING
    app.snapshot = ServerSnapshot(reachable=True, loaded_models=0, free_vram_bytes=2**30)
    first = menu_items(app)[0]
    assert first.text == status_text(app.state, app.snapshot)
    assert first.enabled is False


def test_exactly_one_default_item_and_it_opens_the_control_panel(app: TrayApp) -> None:
    """Left-clicking the icon fires the default item -- it must be the GUI."""
    defaults = [i.text for i in menu_items(app) if i.default]
    assert defaults == ["Open control panel"]


@pytest.mark.parametrize(
    ("state", "start", "stop", "restart_server", "unload", "restart_engines"),
    [
        (STATE_STOPPED, True, False, False, False, False),
        (STATE_STARTING, False, True, True, False, False),
        (STATE_RUNNING, False, True, True, True, True),
        (STATE_CRASHED, True, False, False, False, False),
    ],
)
def test_enabled_predicates_follow_the_state(
    app: TrayApp,
    state: str,
    start: bool,
    stop: bool,
    restart_server: bool,
    unload: bool,
    restart_engines: bool,
) -> None:
    app.state = state
    assert item_named(app, "Start server").enabled is start
    assert item_named(app, "Stop server").enabled is stop
    assert item_named(app, "Restart server").enabled is restart_server
    assert item_named(app, "Unload all models").enabled is unload
    assert item_named(app, "Restart engines").enabled is restart_engines


def test_start_at_login_reflects_the_autostart_module(
    app: TrayApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        autostart,
        "status",
        lambda cfg: autostart.AutostartStatus(True, "test", None),
    )
    assert item_named(app, "Start at login").checked is True
    monkeypatch.setattr(
        autostart,
        "status",
        lambda cfg: autostart.AutostartStatus(False, "test", None),
    )
    assert item_named(app, "Start at login").checked is False


# ---------------------------------------------------------------------------
# Single-instance guard
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_single_instance_guard_is_exclusive_and_releasable() -> None:
    port = _free_port()
    first = acquire_single_instance(port=port)
    assert first is not None
    try:
        assert acquire_single_instance(port=port) is None
    finally:
        first.close()
    second = acquire_single_instance(port=port)
    assert second is not None
    second.close()


def test_the_guard_port_does_not_collide_with_clawforge() -> None:
    """Both trays live in the same notification area; a shared mutex port would
    make each look 'already running' to the other."""
    assert tray_app.SINGLE_INSTANCE_PORT != 47821


# ---------------------------------------------------------------------------
# URL building
# ---------------------------------------------------------------------------


def test_control_panel_url_follows_the_configured_gui_port(config: Config) -> None:
    config.gui.port = 8080
    before = control_panel_url(config)
    config.gui.port = 9099
    after = control_panel_url(config)
    assert before != after
    assert ":9099" in after
    assert "9099" not in before


def test_api_docs_url_follows_the_configured_server_port(config: Config) -> None:
    config.server.port = 4321
    url = api_docs_url(config)
    assert url.endswith("/docs")
    assert ":4321/" in url


def test_urls_do_not_hardcode_localhost_or_a_port(config: Config) -> None:
    config.gui.host = "10.1.2.3"
    config.gui.port = 7777
    config.server.host = "10.1.2.3"
    config.server.port = 7778
    panel = control_panel_url(config)
    docs = api_docs_url(config)
    assert "localhost" not in panel and "localhost" not in docs
    assert panel.startswith("http://10.1.2.3:7777")
    assert docs.startswith("http://10.1.2.3:7778")


def test_wildcard_bind_becomes_loopback_not_a_dead_address(config: Config) -> None:
    config.gui.host = "0.0.0.0"
    config.server.host = "0.0.0.0"
    assert "0.0.0.0" not in control_panel_url(config)
    assert "0.0.0.0" not in api_docs_url(config)


def test_fallback_mcp_url_uses_the_configured_path_and_port(config: Config) -> None:
    config.server.host = "10.9.8.7"
    config.server.port = 5150
    config.mcp.path = "/mcp"
    assert fallback_mcp_url(config) == "http://10.9.8.7:5150/mcp"


def test_server_command_passes_the_config_path_explicitly(config: Config) -> None:
    """A tray started from the Startup folder may not carry SF_DATA_DIR."""
    argv = server_command(config)
    assert argv[1:4] == ["-m", "studioforge", "serve"]
    assert "--config" in argv
    assert str(config.config_path) in argv


# ---------------------------------------------------------------------------
# Action handlers -> API calls
# ---------------------------------------------------------------------------


def test_unload_all_posts_to_the_unload_endpoint(app: TrayApp, client: FakeClient) -> None:
    app.unload_all_models()
    assert client.calls == [FakeCall("POST", "/api/models/unload-all", None)]


def test_restart_engines_posts_to_the_backend_endpoint(app: TrayApp, client: FakeClient) -> None:
    app.restart_engines()
    assert client.calls == [FakeCall("POST", "/api/restart/backend", None)]


def test_restart_server_always_confirms(app: TrayApp, client: FakeClient) -> None:
    """Without confirm the endpoint correctly refuses, which from a menu item
    is indistinguishable from a broken menu item."""
    app.api_restart_server()
    assert client.calls == [FakeCall("POST", "/api/restart/server", {"confirm": True})]


def test_restart_server_confirms_through_the_menu_path_too(
    app: TrayApp, client: FakeClient
) -> None:
    app.owns_child = False  # attached to someone else's server: the API is the only lever
    app.restart_server()
    posts = [c for c in client.calls if c.method == "POST"]
    assert posts == [FakeCall("POST", "/api/restart/server", {"confirm": True})]
    assert posts[0].payload == {"confirm": True}


def test_restart_server_prefers_a_local_restart_when_the_tray_owns_the_child(
    app: TrayApp, client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The API restart respawns detached and exits, which would orphan the pid
    the tray supervises and leave a second server it cannot stop."""
    seen: list[str] = []
    monkeypatch.setattr(TrayApp, "stop_server", lambda self, *a, **k: seen.append("stop"))
    monkeypatch.setattr(TrayApp, "start_server", lambda self: seen.append("start"))
    app.owns_child = True
    app.restart_server()
    assert seen == ["stop", "start"]
    assert [c for c in client.calls if c.path == "/api/restart/server"] == []


def test_status_poll_reads_the_status_endpoint(app: TrayApp, client: FakeClient) -> None:
    client.replies["/api/status"] = ApiResult(
        True, {"loaded": [{"model_id": "a"}], "gpus": [{"free_bytes": 2**30}]}
    )
    app._poll_status()
    assert FakeCall("GET", "/api/status") in client.calls
    assert app.snapshot.loaded_models == 1
    assert app.snapshot.free_vram_bytes == 2**30


def test_an_unreachable_server_marks_the_snapshot_unreachable(
    app: TrayApp, client: FakeClient
) -> None:
    client.default = ApiResult(False, error="the server is not answering")
    app._poll_status()
    assert app.snapshot.reachable is False
    assert "not answering" in status_text(STATE_RUNNING, app.snapshot)


def test_mcp_info_prefers_the_recommended_url_from_the_server(
    app: TrayApp, client: FakeClient
) -> None:
    client.replies["/api/mcp/info"] = ApiResult(
        True, {"recommended": "http://100.64.1.2:1234/mcp", "pin": "12345678"}
    )
    assert app.mcp_info() == ("http://100.64.1.2:1234/mcp", "12345678")


def test_mcp_info_falls_back_to_the_config_when_the_server_is_down(
    app: TrayApp, client: FakeClient, config: Config
) -> None:
    client.default = ApiResult(False, error="down")
    config.mcp.pin = "87654321"
    url, pin = app.mcp_info()
    assert url == fallback_mcp_url(config)
    assert pin == "87654321"


def test_adopt_running_server_attaches_without_owning(app: TrayApp, client: FakeClient) -> None:
    client.replies["/health"] = ApiResult(True, {"status": "ok"})
    assert app.adopt_running_server() is True
    assert app.state == STATE_RUNNING
    assert app.owns_child is False


def test_adopt_running_server_is_false_when_nothing_answers(
    app: TrayApp, client: FakeClient
) -> None:
    client.default = ApiResult(False, error="down")
    assert app.adopt_running_server() is False
    assert app.state == STATE_STOPPED


# ---------------------------------------------------------------------------
# Autostart delegation
# ---------------------------------------------------------------------------


def test_toggling_autostart_delegates_to_core_autostart(
    app: TrayApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    enabled = False
    calls: list[str] = []

    def fake_status(cfg: Config) -> autostart.AutostartStatus:
        return autostart.AutostartStatus(enabled, "test", None)

    def fake_enable(cfg: Config, **kwargs: Any) -> autostart.AutostartStatus:
        calls.append("enable")
        return autostart.AutostartStatus(True, "test", None)

    def fake_disable(cfg: Config) -> autostart.AutostartStatus:
        calls.append("disable")
        return autostart.AutostartStatus(False, "test", None)

    monkeypatch.setattr(autostart, "status", fake_status)
    monkeypatch.setattr(autostart, "enable", fake_enable)
    monkeypatch.setattr(autostart, "disable", fake_disable)

    app._on_toggle_autostart()
    _join_tray_threads()
    assert calls == ["enable"]

    enabled = True
    app._on_toggle_autostart()
    _join_tray_threads()
    assert calls == ["enable", "disable"]


def _join_tray_threads(timeout: float = 5.0) -> None:
    """Handlers run off the message-pump thread, so wait for them."""
    import threading

    for thread in threading.enumerate():
        if thread.name.startswith("studioforge-tray-") and thread is not threading.current_thread():
            thread.join(timeout=timeout)


# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------


def test_the_api_key_is_sent_as_a_bearer_header(config: Config) -> None:
    config.server.api_key = "sk-tray-secret-value"
    headers = tray_app.HttpApiClient(config)._headers()
    assert headers["Authorization"] == "Bearer sk-tray-secret-value"


def test_no_authorization_header_without_a_key(config: Config) -> None:
    config.server.api_key = None
    assert "Authorization" not in tray_app.HttpApiClient(config)._headers()


def test_the_api_key_never_appears_in_a_user_visible_string(
    config: Config, client: FakeClient
) -> None:
    """It must live in exactly one place: an Authorization header."""
    secret = "sk-tray-secret-value"
    config.server.api_key = secret
    config.mcp.pin = "12345678"
    app = TrayApp(config, client=client, create_icon=False)
    app.state = STATE_RUNNING
    app.snapshot = ServerSnapshot(reachable=True, loaded_models=1, free_vram_bytes=2**30)

    visible: list[str] = [
        status_text(app.state, app.snapshot),
        control_panel_url(config),
        api_docs_url(config),
        fallback_mcp_url(config),
        *labels(app),
    ]
    url, pin = app.mcp_info()
    visible.extend([url, pin or ""])

    notes: list[str] = []
    app._notify = lambda message, title=tray_app.APP_NAME: notes.append(f"{title} {message}")  # type: ignore[method-assign]
    app._on_stop()
    _join_tray_threads()
    visible.extend(notes)

    for text in visible:
        assert secret not in text, text


def test_copying_the_pin_does_not_copy_the_api_key(
    config: Config, client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    config.server.api_key = "sk-tray-secret-value"
    config.mcp.pin = "12345678"
    copied: list[str] = []
    monkeypatch.setattr(tray_app, "copy_to_clipboard", lambda text: copied.append(text) or True)
    app = TrayApp(config, client=client, create_icon=False)
    client.default = ApiResult(False, error="down")
    app._on_copy_mcp_pin()
    _join_tray_threads()
    assert copied == ["12345678"]


def test_the_tray_toggle_enables_TRAY_autostart_not_server_only(
    app: TrayApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The icon the user just switched on must be there at the next login.

    `autostart.enable()` without `tray=True` writes a bare `serve` shim: the
    server comes up headless and the tray is absent, while the checkbox still
    reads as enabled because a shim exists.
    """
    seen: list[dict[str, Any]] = []

    monkeypatch.setattr(
        autostart, "status", lambda cfg: autostart.AutostartStatus(False, "test", None)
    )

    def fake_enable(cfg: Config, **kwargs: Any) -> autostart.AutostartStatus:
        seen.append(kwargs)
        return autostart.AutostartStatus(True, "test", None)

    monkeypatch.setattr(autostart, "enable", fake_enable)

    app._on_toggle_autostart()
    _join_tray_threads()

    assert seen == [{"tray": True}]


def test_stop_works_during_the_crash_restart_window(
    app: TrayApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash-looping server must be stoppable from the menu.

    `owns_child` is cleared the instant a supervised child exits, so during the
    restart backoff Stop was refused with "the tray did not start this" and
    `user_stopped` stayed False -- so the supervisor respawned regardless and
    the loop could not be broken.
    """
    stopped: list[bool] = []
    monkeypatch.setattr(app, "stop_server", lambda: stopped.append(True))

    # Exactly the post-crash state: child gone, restart pending.
    app.proc = None
    app.owns_child = False
    app.user_stopped = False
    app.state = STATE_STARTING

    app._on_stop()
    _join_tray_threads()

    assert stopped == [True], "Stop was refused while the tray was supervising a restart"


# ---------------------------------------------------------------------------
# D28: exactly one respawner, and a deliberate restart is not a crash
#
# Before: a GUI "Restart server" went to the watchdog, which killed the tray's
# child AND spawned a replacement; the tray counted a crash attempt and spawned
# too; the loser exited 3 on the port conflict, which the tray counted again,
# and again, until "Crashed -- see the logs folder" sat next to a healthy
# server. Now the watchdog leaves the respawn to the tray and says so on its
# /health, the tray reads that, and a port-conflict exit is never respawned.
# ---------------------------------------------------------------------------


def _watchdog_says(client: FakeClient, config: Config, payload: dict[str, Any]) -> None:
    client.urls[tray_app.watchdog_health_url(config)] = ApiResult(True, payload)


def test_a_port_conflict_exit_is_never_a_crash(app: TrayApp) -> None:
    assert app.classify_exit(tray_app.EXIT_PORT_CONFLICT) == tray_app.KIND_PORT_TAKEN


def test_an_exit_during_a_watchdog_restart_is_a_requested_restart(
    app: TrayApp, client: FakeClient, config: Config
) -> None:
    _watchdog_says(client, config, {"status": "down", "restart_in_progress": {"since": 1.0}})
    assert app.classify_exit(1) == tray_app.KIND_RESTART_REQUESTED


def test_an_exit_with_no_restart_in_progress_is_a_crash(
    app: TrayApp, client: FakeClient, config: Config
) -> None:
    _watchdog_says(client, config, {"status": "down", "restart_in_progress": None})
    assert app.classify_exit(1) == tray_app.KIND_CRASH
    # No watchdog at all: nobody else is restarting anything.
    client.urls.clear()
    assert app.classify_exit(1) == tray_app.KIND_CRASH


def test_a_requested_restart_respawns_without_spending_a_crash_attempt(app: TrayApp) -> None:
    app.user_stopped = False
    app.restarts = 0
    respawn, notice = app._child_exited_locked(tray_app.KIND_RESTART_REQUESTED, 1)
    assert respawn is True
    assert app.restarts == 0, "a restart somebody asked for is not a crash attempt"
    assert app.state == STATE_STARTING
    assert notice is not None and "unexpectedly" not in notice


def test_a_crash_spends_an_attempt_and_says_so(app: TrayApp) -> None:
    app.user_stopped = False
    app.restarts = 0
    respawn, notice = app._child_exited_locked(tray_app.KIND_CRASH, 1)
    assert respawn is True and app.restarts == 1
    assert notice is not None and "unexpectedly" in notice
    # ...until the budget is gone.
    app.restarts = tray_app.MAX_RESTARTS
    respawn, notice = app._child_exited_locked(tray_app.KIND_CRASH, 1)
    assert respawn is False and app.state == STATE_CRASHED and app.user_stopped is True


def test_a_port_conflict_exit_waits_for_the_holder_and_adopts_a_studioforge(
    app: TrayApp, client: FakeClient
) -> None:
    app.user_stopped = False
    respawn, notice = app._child_exited_locked(tray_app.KIND_PORT_TAKEN, 3)
    assert respawn is False, "respawning cannot free a port someone else holds"
    assert app.state == STATE_STARTING
    assert app._port_holder_deadline is not None
    assert "in use" in app.status_line()

    # Nothing answers yet: keep waiting, no state change.
    client.default = ApiResult(False, error="down")
    assert app._poll_unowned_locked() is False
    assert app.state == STATE_STARTING

    # A StudioForge server comes up on the port (the watchdog's replacement,
    # or another install): attach to it, do not fight it.
    client.default = ApiResult(True, {"status": "ok"})
    assert app._poll_unowned_locked() is True
    assert app.state == STATE_RUNNING
    assert app.adopted is True and app.owns_child is False
    assert app._port_holder_deadline is None


def test_a_port_conflict_that_persists_names_the_fix(
    app: TrayApp, client: FakeClient, config: Config
) -> None:
    app.user_stopped = False
    app._child_exited_locked(tray_app.KIND_PORT_TAKEN, 3)
    client.default = ApiResult(False, error="down")
    app._port_holder_deadline = 0.0  # the grace period has passed
    assert app._poll_unowned_locked() is True
    assert app.state == STATE_CRASHED
    assert app.user_stopped is True, "no more respawn attempts against a taken port"
    line = app.status_line()
    assert str(config.server.port) in line and "Start server" in line
    assert "logs folder" not in line, "the generic crash advice is the wrong next action here"


def test_respawn_attaches_when_a_server_already_answers(
    app: TrayApp, client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawned: list[bool] = []
    monkeypatch.setattr(app, "_spawn", lambda: spawned.append(True) or True)
    app.user_stopped = False
    app.proc = None

    client.default = ApiResult(True, {"status": "ok"})
    app._respawn_or_adopt()
    assert spawned == [] and app.state == STATE_RUNNING and app.adopted is True

    app.state = STATE_STARTING
    app.adopted = False
    client.default = ApiResult(False, error="down")
    app._respawn_or_adopt()
    assert spawned == [True]


def test_stop_refuses_an_adopted_server_instead_of_lying(
    app: TrayApp, client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attached to someone else's server, Stop must not say 'stopped; VRAM
    released' about a process that is still running."""
    stopped: list[bool] = []
    notices: list[str] = []
    monkeypatch.setattr(app, "stop_server", lambda: stopped.append(True))
    monkeypatch.setattr(
        app, "_notify", lambda message, title=tray_app.APP_NAME: notices.append(message)
    )
    client.replies["/health"] = ApiResult(True, {"status": "ok"})
    assert app.adopt_running_server() is True

    app._on_stop()
    _join_tray_threads()

    assert stopped == []
    assert notices and "not started by the tray" in notices[0]


def test_stop_takes_down_a_watchdog_left_over_from_a_restart(
    app: TrayApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a watchdog-driven restart the watchdog is nobody's descendant, so
    a tree kill of the server misses it; Stop from the tray means everything."""
    killed: list[int] = []
    monkeypatch.setattr(tray_app, "find_watchdog_pids", lambda config: [3131])
    monkeypatch.setattr(tray_app, "kill_process_tree", lambda pid, **kw: killed.append(pid))

    class LiveProc:
        pid = 1

        def poll(self) -> int | None:
            return None

        def wait(self, timeout: float = 0) -> int:
            return 0

    app.proc = LiveProc()  # type: ignore[assignment]
    app.owns_child = True
    app.state = STATE_RUNNING
    app.stop_server(timeout=0.1)
    assert killed == [1, 3131]
    assert app.state == STATE_STOPPED


def test_status_line_prefers_a_known_detail_for_a_down_state() -> None:
    snap = ServerSnapshot()
    assert status_text(STATE_CRASHED, snap, "Port 1234 is held by another program") == (
        "Port 1234 is held by another program"
    )
    assert status_text(STATE_STARTING, snap, "Restarting...") == "Restarting..."
    # A running server always shows the live numbers, whatever detail was set.
    live = ServerSnapshot(reachable=True, loaded_models=1, free_vram_bytes=2**30)
    assert status_text(STATE_RUNNING, live, "stale detail").startswith("Running")


def test_exit_code_constant_matches_serve() -> None:
    """The tray reads back the code __main__._preflight_ports exits with."""
    from studioforge.core.ports import EXIT_PORT_CONFLICT

    assert tray_app.EXIT_PORT_CONFLICT == EXIT_PORT_CONFLICT == 3


def test_the_server_is_told_the_tray_launched_it(
    app: TrayApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SF_SUPERVISOR=tray is what makes `POST /api/restart/server` exit for us
    to respawn instead of racing us with a respawn of its own (D28)."""
    captured: dict[str, Any] = {}

    class _Proc:
        pid = 99

        def poll(self) -> None:
            return None

    def fake_popen(argv: list[str], **kwargs: Any) -> Any:
        captured["env"] = kwargs.get("env") or {}
        return _Proc()

    monkeypatch.setattr(tray_app.subprocess, "Popen", fake_popen)
    with app._lock:
        assert app._spawn() is True
    assert captured["env"].get(tray_app.ENV_SUPERVISOR) == "tray"


def test_the_restart_exit_code_is_a_requested_restart_without_asking_anyone(
    app: TrayApp, client: FakeClient
) -> None:
    client.urls.clear()  # no watchdog answering at all
    assert app.classify_exit(tray_app.EXIT_RESTART_REQUESTED) == (tray_app.KIND_RESTART_REQUESTED)


# ---------------------------------------------------------------------------
# A port-conflict exit on a port that is NOT the server port (2026-08-19)
# ---------------------------------------------------------------------------


def test_port_conflict_with_a_free_server_port_respawns_after_a_short_delay(
    app: TrayApp, client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first restart after the V2 switch: serve died on the WATCHDOG port
    (left running for adoption) while 1234 was free. The tray must retry the
    spawn instead of waiting two minutes and blaming LM Studio for 1234."""
    spawned: list[bool] = []
    monkeypatch.setattr(app, "_spawn", lambda: spawned.append(True) or True)
    monkeypatch.setattr(app, "_server_port_is_free", lambda: True)
    app.user_stopped = False
    client.default = ApiResult(False, error="down")

    app._child_exited_locked(tray_app.KIND_PORT_TAKEN, 3)
    assert app._port_conflict_retry_at is not None
    # Before the retry delay: nothing yet.
    assert app._poll_unowned_locked() is False
    assert spawned == []
    # Delay elapsed: respawn, once.
    app._port_conflict_retry_at = 0.0
    assert app._poll_unowned_locked() is True
    assert spawned == [True]
    assert app.state == STATE_STARTING
    assert app._port_conflict_respawns == 1
    assert app._port_holder_deadline is None


def test_port_conflict_respawns_are_bounded_and_the_report_names_the_real_port(
    app: TrayApp, client: FakeClient, monkeypatch: pytest.MonkeyPatch, config: Config
) -> None:
    from studioforge.core.ports import PortConflict, PortHolder

    spawned: list[bool] = []
    monkeypatch.setattr(app, "_spawn", lambda: spawned.append(True) or True)
    monkeypatch.setattr(app, "_server_port_is_free", lambda: True)
    monkeypatch.setattr(
        tray_app,
        "check_startup_ports",
        lambda cfg: [
            PortConflict(
                role="watchdog",
                port=1235,
                host="0.0.0.0",
                holder=PortHolder(pid=4242, name="python.exe"),
            )
        ],
    )
    app.user_stopped = False
    client.default = ApiResult(False, error="down")

    for _ in range(tray_app.MAX_PORT_CONFLICT_RESPAWNS):
        app._child_exited_locked(tray_app.KIND_PORT_TAKEN, 3)
        app._port_conflict_retry_at = 0.0
        assert app._poll_unowned_locked() is True
    assert len(spawned) == tray_app.MAX_PORT_CONFLICT_RESPAWNS

    # One more conflict: no further spawn; at the deadline the REAL port is named.
    app._child_exited_locked(tray_app.KIND_PORT_TAKEN, 3)
    app._port_conflict_retry_at = 0.0
    app._port_holder_deadline = 0.0
    assert app._poll_unowned_locked() is True
    assert len(spawned) == tray_app.MAX_PORT_CONFLICT_RESPAWNS
    assert app.state == STATE_CRASHED
    assert "1235" in app.status_line() and "watchdog" in app.status_line()
    assert str(config.server.port) not in app.status_line() or config.server.port == 1235
