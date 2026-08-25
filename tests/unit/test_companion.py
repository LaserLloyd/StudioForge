"""Tests for the ``studioforge-companion`` package (``sfctl``).

The companion is a **separate distribution** (``packages/studioforge-companion``)
that deliberately does not depend on ``studioforge``, so it is not installed
into this repo's venv and is not on ``sys.path`` by default. Rather than add it
to the root ``pyproject.toml`` -- which would couple the server's dependency set
to the client's and defeat the split -- its ``src`` directory is prepended here.

Most of the suite runs against a **real server**: the actual FastAPI app from
``studioforge.api.app.create_app`` under uvicorn in a background thread, with a
real API key, mirroring ``tests/contract/conftest.py``. A stub would not prove
the thing that matters -- that the client parses the server's real error
envelope, real auth rejection and real payload shapes.

The MCP proxy is tested against two **real MCP servers** started in-process
(``mcp.server.mcpserver.MCPServer`` over streamable HTTP), because the whole
point of the proxy is transport behaviour: what happens to the merged tool list
when one upstream is down.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import typer

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPANION_SRC = REPO_ROOT / "packages" / "studioforge-companion" / "src"
if str(COMPANION_SRC) not in sys.path:
    sys.path.insert(0, str(COMPANION_SRC))

from studioforge_companion import cli as cli_module  # noqa: E402
from studioforge_companion.client import (  # noqa: E402
    ApiError,
    AuthFailed,
    ServerUnreachable,
    StudioForgeClient,
)
from studioforge_companion.config import (  # noqa: E402
    CompanionConfig,
    CompanionConfigError,
    ServerProfile,
    config_path,
    load_companion_config,
    normalize_url,
    redact,
    save_companion_config,
    set_value,
)
from studioforge_companion.mcp_proxy import (  # noqa: E402
    RECOVERY_PREFIX,
    McpProxy,
    Upstream,
    expose_name,
)
from typer.testing import CliRunner  # noqa: E402

API_KEY = "sf-companion-test-key-9f3a"
TINY_MODEL_ID = "tiny-test-model"

on_posix = pytest.mark.skipif(os.name == "nt", reason="POSIX file modes only")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_round_trip_multiple_servers(tmp_path: Path) -> None:
    path = tmp_path / "companion.toml"
    cfg = CompanionConfig(
        default="rig",
        servers={
            "rig": ServerProfile(
                name="rig",
                url="http://100.64.0.3:1234",
                api_key="sf-secret-key-value",
                watchdog_url="http://100.64.0.3:9999",
                timeout_s=42.0,
            ),
            "laptop": ServerProfile(name="laptop", url="http://192.168.1.50:1234"),
        },
    )
    saved = save_companion_config(cfg, path)
    assert saved == path

    loaded = load_companion_config(path)
    assert loaded.default == "rig"
    assert set(loaded.servers) == {"rig", "laptop"}
    rig = loaded.servers["rig"]
    assert rig.name == "rig"
    assert rig.url == "http://100.64.0.3:1234"
    assert rig.api_key == "sf-secret-key-value"
    assert rig.watchdog_url == "http://100.64.0.3:9999"
    assert rig.timeout_s == 42.0
    assert loaded.servers["laptop"].api_key is None


def test_load_missing_file_is_empty(tmp_path: Path) -> None:
    cfg = load_companion_config(tmp_path / "nope.toml")
    assert cfg.servers == {}
    assert cfg.default is None


def test_profile_resolution_default_and_single(tmp_path: Path) -> None:
    cfg = CompanionConfig(
        default="rig",
        servers={
            "rig": ServerProfile(name="rig", url="http://a:1234"),
            "laptop": ServerProfile(name="laptop", url="http://b:1234"),
        },
    )
    assert cfg.profile().name == "rig"
    assert cfg.profile("laptop").name == "laptop"

    single = CompanionConfig(servers={"only": ServerProfile(name="only", url="http://c:1234")})
    assert single.profile().name == "only"


def test_profile_unknown_name_lists_known_names() -> None:
    cfg = CompanionConfig(
        servers={
            "rig": ServerProfile(name="rig", url="http://a:1234"),
            "laptop": ServerProfile(name="laptop", url="http://b:1234"),
        }
    )
    with pytest.raises(CompanionConfigError) as info:
        cfg.profile("desktop")
    message = str(info.value)
    assert "desktop" in message
    assert "rig" in message and "laptop" in message
    assert info.value.exit_code == 2


def test_profile_no_servers_is_actionable() -> None:
    with pytest.raises(CompanionConfigError) as info:
        CompanionConfig().profile()
    assert "sfctl servers add" in str(info.value)


def test_profile_ambiguous_without_default() -> None:
    cfg = CompanionConfig(
        servers={
            "a": ServerProfile(name="a", url="http://a:1234"),
            "b": ServerProfile(name="b", url="http://b:1234"),
        }
    )
    with pytest.raises(CompanionConfigError) as info:
        cfg.profile()
    assert "servers use" in str(info.value)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("host:1234", "http://host:1234"),
        ("http://host:1234/", "http://host:1234"),
        ("https://host/", "https://host"),
        ("  100.64.0.3:1234  ", "http://100.64.0.3:1234"),
        ("http://host:1234", "http://host:1234"),
    ],
)
def test_url_normalization(raw: str, expected: str) -> None:
    assert normalize_url(raw) == expected


@pytest.mark.parametrize("raw", ["http://host:1234/api", "host:1234/v1", "ftp://host", ""])
def test_url_normalization_rejects(raw: str) -> None:
    with pytest.raises(CompanionConfigError):
        normalize_url(raw)


def test_watchdog_url_derivation() -> None:
    profile = ServerProfile(name="rig", url="http://100.64.0.3:1234")
    assert profile.effective_watchdog_url == "http://100.64.0.3:1235"
    assert profile.watchdog_mcp_url == "http://100.64.0.3:1235/mcp"
    assert profile.mcp_url == "http://100.64.0.3:1234/mcp"
    assert profile.api_base == "http://100.64.0.3:1234/api"
    assert profile.openai_base == "http://100.64.0.3:1234/v1"

    explicit = ServerProfile(
        name="rig", url="http://100.64.0.3:1234", watchdog_url="http://other:4321"
    )
    assert explicit.effective_watchdog_url == "http://other:4321"

    no_port = ServerProfile(name="rig", url="https://rig.example.com")
    assert no_port.effective_watchdog_url == "https://rig.example.com:1235"


@on_posix
def test_saved_config_is_0600(tmp_path: Path) -> None:
    path = tmp_path / "companion.toml"
    save_companion_config(
        CompanionConfig(servers={"rig": ServerProfile(name="rig", url="http://a:1234")}), path
    )
    assert (path.stat().st_mode & 0o777) == 0o600


def test_redact_never_returns_the_key() -> None:
    key = "sf-super-secret-key-1234567890"
    masked = redact(key)
    assert masked is not None
    assert masked != key
    assert key not in masked
    assert len(masked) < len(key)
    assert redact("short") == "***"
    assert redact(None) is None
    assert redact("") is None


def test_set_value_dotted(tmp_path: Path) -> None:
    path = tmp_path / "companion.toml"
    set_value("servers.rig.url", "rig.local:1234", path=path)
    set_value("servers.rig.api_key", "sf-abc-123456", path=path)
    set_value("servers.rig.timeout_s", "12.5", path=path)
    set_value("server.watchdog_url", "rig.local:9999", path=path)

    cfg = load_companion_config(path)
    assert cfg.default == "rig"
    rig = cfg.servers["rig"]
    assert rig.url == "http://rig.local:1234"
    assert rig.api_key == "sf-abc-123456"
    assert rig.timeout_s == 12.5
    assert rig.watchdog_url == "http://rig.local:9999"

    set_value("default", "rig", path=path)
    assert load_companion_config(path).default == "rig"

    with pytest.raises(CompanionConfigError):
        set_value("default", "ghost", path=path)
    with pytest.raises(CompanionConfigError):
        set_value("servers.rig.nonsense", "x", path=path)
    with pytest.raises(CompanionConfigError):
        set_value("servers.rig.timeout_s", "not-a-number", path=path)


def test_config_path_is_platform_appropriate() -> None:
    path = config_path()
    assert path.name == "companion.toml"
    assert "studioforge" in str(path).lower()


def test_bad_toml_is_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "companion.toml"
    path.write_text("this is not = = toml", encoding="utf-8")
    with pytest.raises(CompanionConfigError) as info:
        load_companion_config(path)
    assert "not valid TOML" in str(info.value)


# ---------------------------------------------------------------------------
# live server fixture
# ---------------------------------------------------------------------------


def _write_tiny_model(models_dir: Path) -> None:
    """Put one real (if minimal) GGUF in the library.

    Built with the repo's own synthetic GGUF writer so the registry's real
    parser accepts it -- ``models info`` and ``models plan`` need an actual
    registry entry to be meaningful.
    """
    from tests.unit.test_gguf import llm_kv, write_gguf

    target = models_dir / "testpub" / f"{TINY_MODEL_ID}-GGUF"
    target.mkdir(parents=True, exist_ok=True)
    write_gguf(
        target / f"{TINY_MODEL_ID}-Q8_0.gguf",
        llm_kv(),
        [("blk.0.attn_q.weight", (128, 128), 8)],
    )


class ServerHandle:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key

    def profile(self, *, api_key: str | None = None, timeout_s: float = 60.0) -> ServerProfile:
        return ServerProfile(
            name="live",
            url=self.base_url,
            api_key=self.api_key if api_key is None else api_key,
            timeout_s=timeout_s,
        )


@pytest.fixture(scope="module")
def live_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ServerHandle]:
    """The real app under uvicorn in a thread, with an API key set."""
    import uvicorn

    from studioforge.api.app import create_app
    from studioforge.config import Config

    root = tmp_path_factory.mktemp("companion-live")
    models_dir = root / "models"
    models_dir.mkdir()
    _write_tiny_model(models_dir)

    port = free_port()
    config = Config(
        data_dir=root / "data",
        server={
            "host": "127.0.0.1",
            "port": port,
            "api_key": API_KEY,
            "request_timeout_s": 60.0,
            "drain_timeout_s": 2.0,
        },
        gui={"enabled": False, "port": free_port()},
        watchdog={"enabled": False, "port": free_port()},
        models={
            "dir": models_dir,
            "default_ctx": 2048,
            "default_ttl_s": 300,
            "auto_load_pinned": False,
        },
        logging={"level": "WARNING"},
    )
    # start_background=False: no engine install, no supervisor loop. Everything
    # this suite asserts is read-only management surface.
    app = create_app(config, start_background=False)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    )
    thread = threading.Thread(target=server.run, name="companion-test-server", daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            if httpx.get(f"{base_url}/health", timeout=2.0).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if not thread.is_alive():
            raise RuntimeError("test server thread died during startup")
        time.sleep(0.2)
    else:
        raise RuntimeError(f"test server never became healthy at {base_url}")

    try:
        yield ServerHandle(base_url, API_KEY)
    finally:
        server.should_exit = True
        thread.join(timeout=30)


@pytest.fixture
def live_model(live_server: ServerHandle) -> str:
    """Id of the synthetic model, as the registry chose to name it."""

    async def go() -> str:
        async with StudioForgeClient(live_server.profile()) as client:
            payload = await client.models()
        for record in payload["models"]:
            if TINY_MODEL_ID in str(record["id"]):
                return str(record["id"])
        raise AssertionError(f"synthetic model missing: {payload}")

    return asyncio.run(go())


# ---------------------------------------------------------------------------
# client against the live server
# ---------------------------------------------------------------------------


async def test_client_health_and_status(live_server: ServerHandle) -> None:
    async with StudioForgeClient(live_server.profile()) as client:
        health = await client.health()
        assert health["status"] == "ok"

        status = await client.status()
        assert "gpus" in status
        assert "version" in status
        assert isinstance(status.get("loaded"), list)

        version = await client.version()
        assert version["version"]


async def test_client_models_and_gpus(live_server: ServerHandle, live_model: str) -> None:
    async with StudioForgeClient(live_server.profile()) as client:
        models = await client.models()
        assert models["count"] >= 1
        assert any(TINY_MODEL_ID in str(m["id"]) for m in models["models"])

        gpus = await client.gpus()
        assert "gpus" in gpus and "backend" in gpus

        settings = await client.settings(live_model)
        assert isinstance(settings, dict)

        introspected = await client.introspect(live_model)
        assert introspected["loaded"] is False


async def test_client_get_config_redacts_secrets(live_server: ServerHandle) -> None:
    async with StudioForgeClient(live_server.profile()) as client:
        payload = await client.get_config()
    rendered = json.dumps(payload)
    assert API_KEY not in rendered
    assert payload["config"]["server"]["api_key"]
    assert "config_path" in payload


async def test_client_plan(live_server: ServerHandle, live_model: str) -> None:
    async with StudioForgeClient(live_server.profile()) as client:
        plan = await client.plan(live_model, ctx_size=2048)
    assert plan["model_id"] == live_model
    assert "fits" in plan


async def test_wrong_key_raises_auth_failed(live_server: ServerHandle) -> None:
    async with StudioForgeClient(live_server.profile(api_key="sf-wrong-key")) as client:
        with pytest.raises(AuthFailed) as info:
            await client.status()
    assert info.value.exit_code == 5
    assert "key" in str(info.value).lower()


async def test_unreachable_port_raises_server_unreachable() -> None:
    profile = ServerProfile(name="dead", url=f"http://127.0.0.1:{free_port()}", timeout_s=5.0)
    async with StudioForgeClient(profile) as client:
        with pytest.raises(ServerUnreachable) as info:
            await client.status()
    message = str(info.value)
    assert info.value.exit_code == 4
    assert profile.url in message
    assert "sfctl recover" in message
    assert "Traceback" not in message


async def test_unknown_model_error_envelope(live_server: ServerHandle) -> None:
    async with StudioForgeClient(live_server.profile()) as client:
        with pytest.raises(ApiError) as info:
            await client.settings("no-such-model-anywhere")
    error = info.value
    assert error.exit_code == 1
    assert "does not exist" in error.message
    assert error.status_code == 404
    assert error.code == "model_not_found"


async def test_vram_rejection_preserves_suggestions(live_server: ServerHandle) -> None:
    """A rejection's ``suggestions`` must survive parsing.

    A "won't fit in VRAM" refusal is only actionable if the advice attached to
    the server's ``studioforge`` diagnostics block reaches the caller, so this
    asserts on the parsing rather than on any particular hardware verdict: the
    planner's own rejection payload is fed through a stub route that shapes it
    exactly as :class:`studioforge.errors.InsufficientVramError` does.
    """
    from studioforge.errors import InsufficientVramError

    suggestions = [
        "reduce ctx to 8192 (fits with 1.2 GiB spare)",
        "use --kv-type q8_0 to halve the KV cache",
    ]
    payload = InsufficientVramError(
        "model 'huge' needs 96.0 GiB but only 48.0 GiB is free across 2 GPUs",
        details={
            "required_bytes": 103079215104,
            "available_bytes": 51539607552,
            "suggestions": suggestions,
        },
    ).to_payload()

    transport = httpx.MockTransport(lambda request: httpx.Response(507, json=payload))
    profile = ServerProfile(name="stub", url="http://stub:1234")
    async with StudioForgeClient(profile) as client:
        client.http._transport = transport  # noqa: SLF001 - inject the shaped response
        with pytest.raises(ApiError) as info:
            await client.load("huge", ctx_size=999999)

    error = info.value
    assert error.code == "insufficient_vram"
    assert error.status_code == 507
    assert error.suggestions == suggestions
    assert error.details["required_bytes"] == 103079215104


async def test_plan_rejection_from_live_server_carries_advice(
    live_server: ServerHandle, live_model: str
) -> None:
    """An absurd ctx either fits (tiny model) or is refused with advice."""
    async with StudioForgeClient(live_server.profile()) as client:
        try:
            plan = await client.plan(live_model, ctx_size=100_000_000)
        except ApiError as exc:
            assert exc.suggestions or exc.message
            return
    if not plan.get("fits"):
        assert plan.get("suggestions") is not None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

runner = CliRunner()


def _all_output(result: Any) -> str:
    """stdout plus stderr, whichever click version is in play."""
    text = result.output or ""
    try:
        if result.stderr:
            text += result.stderr
    except (ValueError, AttributeError):
        pass
    return text


def _invoke(server: ServerHandle, *args: str, env: dict[str, str] | None = None) -> Any:
    base = ["--url", server.base_url, "--api-key", server.api_key, "--no-color"]
    return runner.invoke(cli_module.app, [*base, *args], env=env, catch_exceptions=False)


@pytest.mark.parametrize(
    ("flags", "expected_tool", "expected_args"),
    [
        (["--restart"], "restart_server", {"confirm": True}),
        (["--nuke"], "nuke_all_models", {"confirm": True}),
        (["--kill", "qwen"], "kill_model", {"model_name": "qwen"}),
        ([], "health", {}),
        # The READ side. `recover` reached 4 of the watchdog's 10 tools -- a
        # health check and three ways to kill things -- so when the main
        # server was wedged (the situation the watchdog exists for) the CLI
        # offered no way to LOOK at the box before choosing which.
        (["--gpus"], "gpu_status", {}),
        (["--logs", "50"], "tail_logs", {"n": 50}),
        (["--logs", "50", "--log-model", "qwen"], "tail_logs", {"n": 50, "model_id": "qwen"}),
        (["--config"], "get_config", {}),
    ],
)
def test_recover_sends_arguments_the_watchdog_actually_accepts(
    monkeypatch: Any, flags: list[str], expected_tool: str, expected_args: dict[str, Any]
) -> None:
    """sfctl recover must send confirm=True and the tool's real parameter names.

    It used to send ``{}`` for restart/nuke -- the watchdog answered with its
    confirmation refusal, which printed as success and exited 0, so the
    recovery command recovered nothing. And ``--kill`` sent ``model_id`` where
    the tool's parameter is ``model_name``, failing schema validation. All
    three destructive paths were non-functional.
    """
    from studioforge_companion import mcp_proxy as proxy_module

    calls: list[tuple[str, dict[str, Any]]] = []

    class _Result:
        is_error = False
        structured_content: dict[str, Any] = {"ok": True}

    async def fake_call(profile: Any, tool: str, arguments: dict[str, Any]) -> Any:
        calls.append((tool, dict(arguments)))
        return _Result()

    monkeypatch.setattr(proxy_module, "call_watchdog_tool", fake_call)
    monkeypatch.setattr(proxy_module, "result_text", lambda result: "ok")

    result = runner.invoke(
        cli_module.app,
        ["--url", "http://127.0.0.1:1", "--no-color", "recover", *flags, "--yes"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, _all_output(result)
    assert calls == [(expected_tool, expected_args)]


@pytest.mark.parametrize(
    "command",
    [
        ["status"],
        ["models", "list"],
        ["config", "get"],
        ["openclaw-setup"],
        ["leases", "list"],
    ],
)
def test_cli_commands_render(live_server: ServerHandle, command: list[str]) -> None:
    result = _invoke(live_server, *command)
    assert result.exit_code == 0, _all_output(result)
    assert result.output.strip()
    assert API_KEY not in _all_output(result)


@pytest.mark.parametrize(
    "command",
    [
        ["status"],
        ["models", "list"],
        ["config", "get"],
        ["openclaw-setup"],
        ["leases", "list"],
    ],
)
def test_cli_commands_json(live_server: ServerHandle, command: list[str]) -> None:
    result = _invoke(live_server, *command, "--json")
    assert result.exit_code == 0, _all_output(result)
    parsed = json.loads(result.output)
    assert parsed is not None
    assert API_KEY not in _all_output(result)


def test_cli_models_info_and_plan(live_server: ServerHandle, live_model: str) -> None:
    for command in (["models", "info", live_model], ["models", "plan", live_model]):
        plain = _invoke(live_server, *command)
        assert plain.exit_code == 0, _all_output(plain)
        assert API_KEY not in _all_output(plain)

        as_json = _invoke(live_server, *command, "--json")
        assert as_json.exit_code == 0, _all_output(as_json)
        parsed = json.loads(as_json.output)
        assert isinstance(parsed, dict)
        assert API_KEY not in _all_output(as_json)


def test_cli_config_get_single_key(live_server: ServerHandle) -> None:
    result = _invoke(live_server, "config", "get", "models.default_ctx")
    assert result.exit_code == 0, _all_output(result)
    assert "2048" in result.output


def test_cli_config_get_secret_is_redacted(live_server: ServerHandle) -> None:
    result = _invoke(live_server, "config", "get", "server.api_key")
    assert result.exit_code == 0, _all_output(result)
    assert API_KEY not in _all_output(result)
    assert result.output.strip()


def test_cli_openclaw_setup_hides_key_by_default(live_server: ServerHandle) -> None:
    redacted = _invoke(live_server, "openclaw-setup", "--json")
    assert API_KEY not in redacted.output
    revealed = _invoke(live_server, "openclaw-setup", "--reveal-key", "--json")
    assert API_KEY in revealed.output  # opt-in only


def test_redactor_masks_an_mcp_pin() -> None:
    """The PIN is a credential, and it is NOT called `api_key`.

    `GET /api/openclaw-setup` returns the MCP pairing PIN as `mcp_pin`, a name
    that matched none of the redactor's hints -- so the default output carried
    it in clear while the human-mode path printed "(API key redacted...)".

    Asserted against the redactor directly, not through the live server: the
    server only fills `mcp_pin` in when `server.api_key` is set, so an
    end-to-end check passes vacuously on a fixture with auth off and proves
    nothing. (Written after exactly that mistake.)
    """
    masked = cli_module._redact_tree({"mcp_pin": "12345678", "nested": {"pin": "87654321"}})
    assert masked["mcp_pin"] != "12345678", "the MCP pin was rendered unredacted"
    assert masked["nested"]["pin"] != "87654321"
    # The sentinel for "auth is disabled" must survive: redacting it once
    # produced the nonsense value `not-...ed` in the printed snippets.
    assert cli_module._redact_tree({"api_key": "not-required"})["api_key"] == "not-required"


def test_cli_help_documents_exit_codes() -> None:
    result = runner.invoke(cli_module.app, ["--help"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "Exit codes" in result.output
    for fragment in ("unreachable", "auth failed", "confirmation required"):
        assert fragment in result.output


# -- exit codes -------------------------------------------------------------


def test_exit_code_4_unreachable() -> None:
    dead = f"http://127.0.0.1:{free_port()}"
    result = runner.invoke(
        cli_module.app, ["--url", dead, "--no-color", "status"], catch_exceptions=False
    )
    assert result.exit_code == 4
    assert "Traceback" not in _all_output(result)


def test_exit_code_5_bad_key(live_server: ServerHandle) -> None:
    result = runner.invoke(
        cli_module.app,
        ["--url", live_server.base_url, "--api-key", "sf-nope", "--no-color", "status"],
        catch_exceptions=False,
    )
    assert result.exit_code == 5


def test_exit_code_1_unknown_model(live_server: ServerHandle) -> None:
    result = _invoke(live_server, "models", "info", "definitely-not-a-model")
    assert result.exit_code == 1


def test_exit_code_2_missing_argument(live_server: ServerHandle) -> None:
    result = runner.invoke(cli_module.app, ["models", "info"], catch_exceptions=False)
    assert result.exit_code == 2


def test_exit_code_2_unknown_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "companion.toml"
    save_companion_config(
        CompanionConfig(servers={"rig": ServerProfile(name="rig", url="http://a:1234")}), config
    )
    monkeypatch.setenv("SF_COMPANION_CONFIG", str(config))
    result = runner.invoke(
        cli_module.app, ["-s", "ghost", "--no-color", "status"], catch_exceptions=False
    )
    assert result.exit_code == 2
    assert "rig" in _all_output(result)


def test_delete_without_yes_in_non_tty_exits_3(live_server: ServerHandle, live_model: str) -> None:
    """No prompt, no hang, nothing deleted -- scripts depend on all three."""
    result = _invoke(live_server, "models", "delete", live_model)
    assert result.exit_code == 3
    assert "--yes" in _all_output(result)

    async def still_there() -> bool:
        async with StudioForgeClient(live_server.profile()) as client:
            payload = await client.models()
        return any(str(m["id"]) == live_model for m in payload["models"])

    assert asyncio.run(still_there())


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------


def test_release_tags_accepts_both_shapes() -> None:
    """``engine/releases`` yields bare tags, ``update/releases`` yields objects."""
    assert cli_module._release_tags({"releases": ["b10427", "b10426"]}) == ["b10427", "b10426"]
    assert cli_module._release_tags({"releases": [{"tag": "v0.2.0"}, {"version": "0.1.0"}]}) == [
        "v0.2.0",
        "0.1.0",
    ]
    assert cli_module._release_tags({"error": "github unreachable"}) == []
    assert cli_module._release_tags(None) == []


def test_new_lines_uses_overlap_not_reprint() -> None:
    """``logs --follow`` polls a tail, so overlap is how it avoids duplicates."""
    previous = ["a", "b", "c"]
    assert cli_module._new_lines(previous, ["b", "c", "d"]) == ["d"]
    assert cli_module._new_lines(previous, ["a", "b", "c"]) == []
    assert cli_module._new_lines([], ["a"]) == ["a"]
    assert cli_module._new_lines(previous, ["x", "y"]) == ["x", "y"]


def test_group_status_prefers_the_unfinished_state() -> None:
    assert cli_module._group_status({"completed", "running"}) == "running"
    assert cli_module._group_status({"completed", "failed"}) == "failed"
    assert cli_module._group_status({"completed"}) == "completed"


def test_coerce_like_uses_the_current_value_as_the_type_hint() -> None:
    assert cli_module._coerce_like(8192, "4096") == 4096
    assert cli_module._coerce_like(True, "no") is False
    assert cli_module._coerce_like(0.5, "0.25") == 0.25
    assert cli_module._coerce_like([], '["a"]') == ["a"]
    assert cli_module._coerce_like(None, "text") == "text"
    assert cli_module._coerce_like("f16", "q8_0") == "q8_0"
    with pytest.raises(CompanionConfigError):
        cli_module._coerce_like(8192, "not-a-number")


def test_redact_tree_masks_secret_looking_keys() -> None:
    payload = {
        "server": {"api_key": "sf-secret-value-1234", "port": 1234},
        "hf": {"token": "hf_secret_value_1234"},
        "models": {"dir": "/models"},
    }
    masked = cli_module._redact_tree(payload)
    rendered = json.dumps(masked)
    assert "sf-secret-value-1234" not in rendered
    assert "hf_secret_value_1234" not in rendered
    assert masked["server"]["port"] == 1234
    assert masked["models"]["dir"] == "/models"


# ---------------------------------------------------------------------------
# MCP proxy
# ---------------------------------------------------------------------------


class McpUpstream:
    """A real MCP server over streamable HTTP, in a background thread."""

    def __init__(self, name: str, tools: dict[str, str]) -> None:
        from mcp.server.mcpserver import MCPServer

        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}/mcp"
        server: Any = MCPServer(name)
        for tool_name, description in tools.items():
            self._register(server, tool_name, description)
        self._server = server
        self._uvicorn: Any = None
        self._thread: threading.Thread | None = None

    @staticmethod
    def _register(server: Any, tool_name: str, description: str) -> None:
        async def handler(argument: str = "") -> str:
            return f"{tool_name} ran with {argument!r}"

        handler.__name__ = tool_name
        server.tool(name=tool_name, description=description)(handler)

    def start(self) -> None:
        import uvicorn

        app = self._server.streamable_http_app()
        self._uvicorn = uvicorn.Server(
            uvicorn.Config(
                app, host="127.0.0.1", port=self.port, log_level="error", access_log=False
            )
        )
        self._thread = threading.Thread(target=self._uvicorn.run, daemon=True)
        self._thread.start()
        deadline = time.time() + 30
        while time.time() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.5)
                if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                    return
            time.sleep(0.1)
        raise RuntimeError(f"fake MCP upstream never came up on {self.port}")

    def stop(self) -> None:
        if self._uvicorn is not None:
            self._uvicorn.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=20)


MANAGEMENT_TOOLS = {
    "list_models": "List models (management).",
    "get_config": "Read server config (management).",
}
WATCHDOG_TOOLS = {
    "get_config": "Read watchdog config.",
    "health": "Watchdog health diagnosis.",
    "restart_server": "Restart the wedged main server.",
}


@pytest.fixture(scope="module")
def fake_management() -> Iterator[McpUpstream]:
    upstream = McpUpstream("fake-management", MANAGEMENT_TOOLS)
    upstream.start()
    try:
        yield upstream
    finally:
        upstream.stop()


@pytest.fixture(scope="module")
def fake_watchdog() -> Iterator[McpUpstream]:
    upstream = McpUpstream("fake-watchdog", WATCHDOG_TOOLS)
    upstream.start()
    try:
        yield upstream
    finally:
        upstream.stop()


def _proxy(management_url: str, watchdog_url: str) -> McpProxy:
    proxy = McpProxy(ServerProfile(name="fake", url="http://127.0.0.1:1"))
    proxy.management = Upstream(label="management", url=management_url, is_watchdog=False)
    proxy.watchdog = Upstream(label="recovery", url=watchdog_url, is_watchdog=True)
    return proxy


def test_expose_name_rule() -> None:
    assert expose_name("list_models", is_watchdog=False) == "list_models"
    assert expose_name("get_config", is_watchdog=False) == "get_config"
    # Colliding watchdog names are prefixed; watchdog-only names are not.
    assert expose_name("get_config", is_watchdog=True) == f"{RECOVERY_PREFIX}get_config"
    assert expose_name("health", is_watchdog=True) == f"{RECOVERY_PREFIX}health"
    assert expose_name("restart_server", is_watchdog=True) == "restart_server"
    assert expose_name("nuke_all_models", is_watchdog=True) == "nuke_all_models"
    assert expose_name("gpu_status", is_watchdog=True) == "gpu_status"


async def test_proxy_merges_both_upstreams(
    fake_management: McpUpstream, fake_watchdog: McpUpstream
) -> None:
    proxy = _proxy(fake_management.url, fake_watchdog.url)
    table = await proxy.build_table()

    # Management keeps bare names.
    assert "list_models" in table
    assert "get_config" in table
    assert table["get_config"].upstream is proxy.management

    # Watchdog collisions are prefixed, watchdog-only names are not.
    assert f"{RECOVERY_PREFIX}get_config" in table
    assert f"{RECOVERY_PREFIX}health" in table
    assert "restart_server" in table
    assert table["restart_server"].upstream is proxy.watchdog
    assert table[f"{RECOVERY_PREFIX}get_config"].remote_name == "get_config"

    # Descriptions survive, and the renamed ones explain themselves.
    assert "management" in (table["list_models"].tool.description or "")
    assert RECOVERY_PREFIX in (table[f"{RECOVERY_PREFIX}get_config"].tool.description or "")


async def test_proxy_forwards_calls(
    fake_management: McpUpstream, fake_watchdog: McpUpstream
) -> None:
    import mcp.types as types

    proxy = _proxy(fake_management.url, fake_watchdog.url)

    managed = await proxy.on_call_tool(
        None, types.CallToolRequestParams(name="list_models", arguments={"argument": "mgmt"})
    )
    assert managed.is_error is not True
    assert "list_models ran with 'mgmt'" in _result_text(managed)

    recovered = await proxy.on_call_tool(
        None,
        types.CallToolRequestParams(
            name=f"{RECOVERY_PREFIX}get_config", arguments={"argument": "wd"}
        ),
    )
    assert recovered.is_error is not True
    # Forwarded under its REMOTE name, not the prefixed one.
    assert "get_config ran with 'wd'" in _result_text(recovered)


async def test_watchdog_tools_survive_management_being_down(fake_watchdog: McpUpstream) -> None:
    """The whole reason the control plane is split in two."""
    import mcp.types as types

    dead = f"http://127.0.0.1:{free_port()}/mcp"
    proxy = _proxy(dead, fake_watchdog.url)

    table = await proxy.build_table()
    assert "restart_server" in table
    assert f"{RECOVERY_PREFIX}health" in table

    called = await proxy.on_call_tool(
        None, types.CallToolRequestParams(name="restart_server", arguments={"argument": "go"})
    )
    assert called.is_error is not True
    assert "restart_server ran with 'go'" in _result_text(called)

    # Management tools are still advertised so the agent knows they exist...
    assert "list_models" in table
    assert "not answering" in (table["list_models"].tool.description or "")

    # ...and invoking one yields an error RESULT (not a raised protocol error)
    # that names the recovery path.
    failed = await proxy.on_call_tool(
        None, types.CallToolRequestParams(name="list_models", arguments={})
    )
    assert failed.is_error is True
    text = _result_text(failed)
    assert "restart_server" in text
    assert "not answering" in text


async def test_proxy_survives_watchdog_being_down(fake_management: McpUpstream) -> None:
    import mcp.types as types

    dead = f"http://127.0.0.1:{free_port()}/mcp"
    proxy = _proxy(fake_management.url, dead)

    table = await proxy.build_table()
    assert "list_models" in table
    assert not any(name.startswith(RECOVERY_PREFIX) for name in table)

    unknown = await proxy.on_call_tool(
        None, types.CallToolRequestParams(name="restart_server", arguments={})
    )
    assert unknown.is_error is True
    assert "unknown tool" in _result_text(unknown)


async def test_proxy_list_tools_handler(
    fake_management: McpUpstream, fake_watchdog: McpUpstream
) -> None:
    proxy = _proxy(fake_management.url, fake_watchdog.url)
    result = await proxy.on_list_tools(None, None)
    names = {tool.name for tool in result.tools}
    assert {"list_models", "get_config", "restart_server"} <= names
    assert f"{RECOVERY_PREFIX}health" in names
    for tool in result.tools:
        assert tool.input_schema is not None


async def test_proxy_served_over_the_real_protocol(
    fake_management: McpUpstream, fake_watchdog: McpUpstream
) -> None:
    """Drive the built server through an actual MCP session, not the handlers.

    ``sfctl mcp`` serves this same ``Server`` over stdio; connecting a real
    client to it in-process exercises the protocol layer (schemas, result
    encoding, instructions) that direct handler calls skip.
    """
    from mcp import Client

    proxy = _proxy(fake_management.url, fake_watchdog.url)
    async with Client(proxy.build_server()) as client:
        listed = await client.list_tools()
        names = {tool.name for tool in listed.tools}
        assert {"list_models", "get_config", "restart_server"} <= names
        assert f"{RECOVERY_PREFIX}get_config" in names

        called = await client.call_tool("restart_server", {"argument": "now"})
        assert called.is_error is not True
        assert "restart_server ran with 'now'" in _result_text(called)

        instructions = client.instructions or ""
        assert RECOVERY_PREFIX in instructions


def test_describe_exception_unwraps_and_trims() -> None:
    """Transport failures arrive wrapped in task groups and can be enormous."""
    from studioforge_companion.mcp_proxy import MAX_ERROR_CHARS, describe_exception

    inner = ConnectionRefusedError("All connection attempts failed")
    grouped = ExceptionGroup("unhandled errors in a TaskGroup", [inner])
    described = describe_exception(grouped)
    assert "TaskGroup" not in described
    assert "All connection attempts failed" in described

    huge = describe_exception(RuntimeError("x" * 5000))
    assert len(huge) <= MAX_ERROR_CHARS
    assert huge.endswith("...")

    assert describe_exception(ValueError()) == "ValueError"


def test_proxy_instructions_document_the_mapping() -> None:
    proxy = McpProxy(ServerProfile(name="rig", url="http://rig:1234"))
    text = proxy.instructions()
    assert "recovery_" in text
    assert "restart_server" in text
    assert "WHEN" in text.upper()
    assert "http://rig:1235" in text


def _result_text(result: Any) -> str:
    from studioforge_companion.mcp_proxy import result_text

    return result_text(result)


# ---------------------------------------------------------------------------
# openclaw-setup without auth: the "not-required" sentinel is not a secret
# ---------------------------------------------------------------------------
#
# The server sends the literal string "not-required" when no API key is
# configured. Redacting it produced "OPENAI_API_KEY=not-...ed" and, worse, a
# copy-pasteable "sfctl config-local set server.api_key=not-...ed" that would
# store garbage as a key.


def test_management_fallback_tools_contains_the_huggingface_pair() -> None:
    """Both halves of the download flow must stay visible while the server is down.

    An agent that cannot see ``repo_details`` does not know the capability
    exists, and would download a quant on a guess instead of asking.
    """
    from studioforge_companion.mcp_proxy import MANAGEMENT_FALLBACK_TOOLS

    tool_names = {name for name, _ in MANAGEMENT_FALLBACK_TOOLS}
    assert "search_models" in tool_names
    assert "repo_details" in tool_names


def test_proxy_instructions_name_the_download_flow_and_the_vram_story() -> None:
    proxy = McpProxy(ServerProfile(name="rig", url="http://rig:1234"))
    text = proxy.instructions()
    assert "repo_details" in text
    assert "download_model" in text
    assert "reclaim_orphan_engines" in text
    assert "vram_orphan_count" in text


def test_watchdog_unprefixed_contains_reclaim_orphan_engines() -> None:
    """The unprefixed watchdog tool set includes reclaim_orphan_engines."""
    from studioforge_companion.mcp_proxy import WATCHDOG_UNPREFIXED

    assert "reclaim_orphan_engines" in WATCHDOG_UNPREFIXED


def test_redact_tree_keeps_the_no_key_sentinel_readable() -> None:
    tree = {
        "inference": {"OPENAI_API_KEY": "not-required"},
        "companion_config": {"server.api_key": "not-required"},
    }
    out = cli_module._redact_tree(tree)
    assert out["inference"]["OPENAI_API_KEY"] == "not-required"
    assert out["companion_config"]["server.api_key"] == "not-required"


def test_redact_tree_still_redacts_a_real_key() -> None:
    out = cli_module._redact_tree({"server.api_key": "sf-really-secret-value"})
    assert "sf-really-secret-value" not in json.dumps(out)


def test_openclaw_setup_without_auth_renders_honestly(monkeypatch: Any) -> None:
    payload = {
        "inference": {
            "OPENAI_BASE_URL": "http://rig:1234/v1",
            "OPENAI_API_KEY": "not-required",
        },
        "mcp": {"mcpServers": {"studioforge": {"command": "sfctl", "args": ["mcp"]}}},
        "companion_config": {
            "server.url": "http://rig:1234",
            "server.api_key": "not-required",
        },
    }
    monkeypatch.setattr(cli_module, "with_client", lambda work: payload)
    result = runner.invoke(
        cli_module.app,
        ["--url", "http://rig:1234", "--no-color", "openclaw-setup"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "not-required" in result.output
    assert "not-...ed" not in result.output, "the sentinel must never be redacted"
    # No suggestion to store the sentinel as a key, and no misleading
    # "redacted" note when there was never a key to redact.
    assert "server.api_key=" not in result.output
    assert "--reveal-key" not in result.output
    assert "server.url=http://rig:1234" in result.output


def test_recover_explains_a_credential_refusal_instead_of_cannot_reach(monkeypatch: Any) -> None:
    """The watchdog guards the recovery tools with the PIN/API key; a 401 must
    say so and give the pairing recipe, not 'cannot reach the watchdog'."""
    from studioforge_companion import mcp_proxy as proxy_module

    async def refuse(profile: Any, tool: str, arguments: dict[str, Any]) -> Any:
        raise RuntimeError("HTTPStatusError: Client error '401 Unauthorized' for url ...")

    monkeypatch.setattr(proxy_module, "call_watchdog_tool", refuse)
    result = runner.invoke(
        cli_module.app,
        ["--url", "http://127.0.0.1:1", "--no-color", "recover"],
        catch_exceptions=False,
    )
    out = _all_output(result)
    assert result.exit_code != 0
    assert "needs a credential" in out
    assert "sfctl servers add" in out and "--api-key <PIN>" in out
    assert "Setup" in out
    assert "cannot reach" not in out


def test_recover_probes_the_watchdog_when_the_mcp_error_hides_the_status(monkeypatch: Any) -> None:
    """The MCP client says only 'Server returned an error response' for a 401;
    recover must probe the watchdog itself and still name the credential fix."""
    from studioforge_companion import mcp_proxy as proxy_module

    async def opaque(profile: Any, tool: str, arguments: dict[str, Any]) -> Any:
        raise RuntimeError("MCPError: Server returned an error response")

    async def probe(profile: Any, **kwargs: Any) -> str:
        return "unauthorized"

    monkeypatch.setattr(proxy_module, "call_watchdog_tool", opaque)
    monkeypatch.setattr(proxy_module, "probe_watchdog_auth", probe)
    result = runner.invoke(
        cli_module.app,
        ["--url", "http://127.0.0.1:1", "--no-color", "recover"],
        catch_exceptions=False,
    )
    out = _all_output(result)
    assert result.exit_code != 0
    assert "needs a credential" in out and "--api-key <PIN>" in out


def test_recover_still_says_unreachable_when_the_watchdog_is_down(monkeypatch: Any) -> None:
    from studioforge_companion import mcp_proxy as proxy_module

    async def opaque(profile: Any, tool: str, arguments: dict[str, Any]) -> Any:
        raise RuntimeError("ConnectError: All connection attempts failed")

    async def probe(profile: Any, **kwargs: Any) -> str:
        return "unreachable"

    monkeypatch.setattr(proxy_module, "call_watchdog_tool", opaque)
    monkeypatch.setattr(proxy_module, "probe_watchdog_auth", probe)
    result = runner.invoke(
        cli_module.app,
        ["--url", "http://127.0.0.1:1", "--no-color", "recover"],
        catch_exceptions=False,
    )
    out = _all_output(result)
    assert "cannot reach the StudioForge watchdog" in out
    assert "needs a credential" not in out


# ---------------------------------------------------------------------------
# GPU leases: the read side is what a co-tenant needs
# ---------------------------------------------------------------------------


def test_status_renders_a_standing_lease(monkeypatch: Any, live_server: ServerHandle) -> None:
    """`/api/status` has carried the lease book since D43 and `sfctl status`
    threw it away, so nothing in the CLI could tell you the rig was LEASED --
    and a harness holding all four cards makes every load fail with a refusal
    that reads like a broken rig.

    Injected at the client, because a real lease needs a real CUDA device and
    CI has none. What is under test is the rendering, not the server.
    """
    real_status = cli_module.StudioForgeClient.status

    async def status_with_a_lease(self: Any) -> Any:
        payload = await real_status(self)
        payload["leases"] = [
            {
                "id": "lease-abc",
                "devices": [0, 1],
                "holder": "crucibleforge",
                "model_ids": [],
                "reason": "benchmark run",
                "idle_s": 12,
                "expires_at": time.time() + 600,
            }
        ]
        return payload

    monkeypatch.setattr(cli_module.StudioForgeClient, "status", status_with_a_lease)
    result = _invoke(live_server, "status")
    assert result.exit_code == 0, _all_output(result)
    output = _all_output(result)
    assert "lease-abc" in output
    assert "crucibleforge" in output
    # An EMPTY model list is the STRONGEST claim -- nobody may plan onto those
    # cards -- so it must never render as "-", which reads like no restriction.
    assert "nothing may load" in output


def test_the_lease_plane_is_reachable_from_the_cli(live_server: ServerHandle) -> None:
    """`/api/leases` has been live with no CLI in front of it: create, list,
    touch and release all had to go through `sfctl mcp`."""
    from studioforge_companion import cli as cli_mod

    for name in ("list", "add", "release", "touch"):
        assert name in {c.name for c in cli_mod.leases_app.registered_commands}, name

    created = _invoke(
        live_server, "leases", "add", "--devices", "0", "--holder", "pytest", "--json"
    )
    if created.exit_code != 0:
        # CI has no NVIDIA driver, so the probe reports zero cards and the
        # server refuses device 0. That refusal still proves the request
        # reached `/api/leases` with the right body -- which is the wiring
        # under test -- so assert on it rather than skipping silently.
        refusal = json.loads(created.output)
        assert "CUDA device" in str(refusal.get("error", "")), refusal
        pytest.skip("no CUDA devices on this machine; lease round-trip needs one")
    lease_id = json.loads(created.output)["id"]
    try:
        listed = json.loads(_invoke(live_server, "leases", "list", "--json").output)
        assert any(entry["id"] == lease_id for entry in listed["leases"])
        # And status sees it, which is the whole point.
        status = json.loads(_invoke(live_server, "status", "--json").output)
        assert any(entry["id"] == lease_id for entry in status["leases"])
        touched = _invoke(live_server, "leases", "touch", lease_id, "--json")
        assert touched.exit_code == 0, _all_output(touched)
    finally:
        released = _invoke(live_server, "leases", "release", lease_id, "--json")
        assert released.exit_code == 0, _all_output(released)
    after = json.loads(_invoke(live_server, "leases", "list", "--json").output)
    assert not any(entry["id"] == lease_id for entry in after["leases"])


def test_the_headline_read_commands_exist(live_server: ServerHandle) -> None:
    """0.2.0's own features were reachable only through `sfctl mcp`."""
    from studioforge_companion import cli as cli_mod

    top = {c.name or c.callback.__name__.replace("_", "-") for c in cli_mod.app.registered_commands}
    assert "search" in top
    model_commands = {
        c.name or c.callback.__name__.replace("_", "-")
        for c in cli_mod.models_app.registered_commands
    }
    assert {"options", "repo", "load-recommended"} <= model_commands


# ---------------------------------------------------------------------------
# Minor contract fixes
# ---------------------------------------------------------------------------


def test_declining_a_prompt_is_a_different_exit_code_from_needing_one(monkeypatch: Any) -> None:
    """Both used to be 3, so a wrapper could not tell "pass --yes" (retryable)
    from "a human said no" (must not be retried)."""
    from studioforge_companion.client import EXIT_CONFIRM, EXIT_DECLINED

    assert EXIT_CONFIRM != EXIT_DECLINED

    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli_module.typer, "confirm", lambda *a, **k: False)
    with pytest.raises(typer.Exit) as declined:
        cli_module._confirm("do the thing?", yes=False)
    assert declined.value.exit_code == EXIT_DECLINED

    monkeypatch.setattr(cli_module.sys.stdin, "isatty", lambda: False, raising=False)
    with pytest.raises(typer.Exit) as needed:
        cli_module._confirm("do the thing?", yes=False)
    assert needed.value.exit_code == EXIT_CONFIRM


async def test_a_download_that_vanishes_is_not_reported_as_finished() -> None:
    """The follower returned its STARTING payload when the group stopped being
    listed, so a never-observed download printed the same success line as a
    verified one -- carrying status="queued"."""

    class _Client:
        async def downloads(self) -> dict[str, Any]:
            return {"downloads": []}

    result = await cli_module._follow_download(
        _Client(),  # type: ignore[arg-type]
        "group-1",
        {"repo_id": "someone/model-GGUF", "quant": "Q4_K_M", "status": "queued"},
    )
    assert result["vanished"] is True
    assert result["status"] != "queued"


def test_an_id_column_is_not_squeezed_to_nothing() -> None:
    """Model ids are `publisher/repo/file-stem`. Rich divides a narrow terminal
    proportionally, so the id folded to a few characters a line while
    single-digit numeric columns kept their width."""
    table = cli_module._table("Model", "State", "Ctx", "Port", "PID")
    by_header = {str(column.header): column for column in table.columns}
    assert by_header["Model"].min_width == cli_module._ID_COLUMN_MIN_WIDTH
    assert by_header["Ctx"].min_width is None
