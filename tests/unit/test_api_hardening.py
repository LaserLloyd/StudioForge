"""API/MCP contract hardening from the WP13 audit.

* on an install with no ``server.api_key``, routes that change the box need a
  local caller or the MCP PIN (D32); reads, inference and load/unload stay open;
* load arguments are validated once, in the manager, for every caller;
* a malformed ``messages`` array is a 400, not a 500;
* ``DELETE /api/models/{id}`` on an unloaded model works (it 500ed on a call to
  a supervisor method that does not exist) and refuses when a persona rides
  the base's files;
* ``mcp.enabled: false`` really leaves the endpoint unmounted;
* ``ModelSettings`` refuses values that would die as an opaque child exit.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import Request

from studioforge.api.auth import (
    PIN_IN_QUERY_NOTE,
    PIN_WITHHELD_NOTE,
    REMOTE_ADMIN_NOTE,
    check_request,
    cross_site_browser_request,
    is_admin_mutation,
    may_reveal_pin,
)
from studioforge.config import Config
from studioforge.core.manager import validate_load_args
from studioforge.errors import AuthError, BadRequestError
from studioforge.types import ModelSettings


def make_request(
    path: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    client: tuple[str, int] | None = ("192.168.1.50", 5000),
) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope: dict[str, Any] = {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": raw,
        "scheme": "http",
        "server": ("0.0.0.0", 1234),
        "client": client,
        "root_path": "",
    }
    return Request(scope)


def open_config(pin: str | None = "12345678") -> Config:
    config = Config(data_dir="/tmp/sf-api-hardening")
    config.server.api_key = None
    config.mcp.pin = pin
    return config


# ---------------------------------------------------------------------------
# D32: remote admin on an open install
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("PATCH", "/api/config"),
        ("POST", "/api/restart/server"),
        ("POST", "/api/restart/backend"),
        ("POST", "/api/engine/install"),
        # Activating rewrites active.json AND engine.pinned_tag: it moves the
        # whole box onto a different binary, so it is as much a box change as
        # the install that fetched it (D49-5).
        ("POST", "/api/engine/activate"),
        ("POST", "/api/engine/smoke-test"),
        ("POST", "/api/update/install"),
        ("POST", "/api/update/rollback"),
        ("POST", "/api/vram/reclaim"),
        ("POST", "/api/downloads"),
        ("DELETE", "/api/downloads/some-group"),
        ("DELETE", "/api/models/vendor/Some-Model-Q4_K_M"),
        ("DELETE", "/api/adapters/some-adapter"),
        ("DELETE", "/api/virtual-models/persona"),
        # Persistent per-model writes (D41): a pin drives the boot autoload and
        # the reconciler, and saved settings shape every future load. Both
        # outlive the instance, so they are box changes, not residency.
        ("POST", "/api/models/vendor/Some-Model-Q4_K_M/pin"),
        ("PUT", "/api/models/vendor/Some-Model/settings"),
        ("PATCH", "/api/models/vendor/Some-Model/settings"),
        # A GPU lease (D43) takes cards away from everyone else on the box.
        ("POST", "/api/leases"),
        ("DELETE", "/api/leases/abc123def456"),
        ("POST", "/api/leases/abc123def456/touch"),
    ],
)
def test_admin_mutations_from_the_lan_need_a_credential_on_an_open_install(
    method: str, path: str
) -> None:
    config = open_config()
    assert is_admin_mutation(method, path)
    with pytest.raises(AuthError) as excinfo:
        check_request(make_request(path, method=method), config)
    assert excinfo.value.status_code == 403
    assert excinfo.value.code == "remote_admin_requires_credential"
    assert "server.api_key" in excinfo.value.message
    assert excinfo.value.message == REMOTE_ADMIN_NOTE


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/config"),
        ("GET", "/api/status"),
        ("GET", "/api/models"),
        ("GET", "/v1/models"),
        ("POST", "/v1/chat/completions"),
        ("POST", "/api/models/vendor/Some-Model/load"),
        ("POST", "/api/models/vendor/Some-Model/unload"),
        ("POST", "/api/models/unload-all"),
        ("GET", "/api/models/vendor/Some-Model/settings"),
        ("GET", "/api/leases"),
        ("POST", "/api/models/scan"),
        ("POST", "/api/virtual-models"),
        # The WP18-21 surfaces. Residency stays open (LM Studio parity, D32):
        # an exact-context load, a per-model restart (a forced reload of one
        # model, not a process restart -- /api/restart/* is the guarded one),
        # the two benchmarks, and the three new reads.
        ("POST", "/api/models/vendor/Some-Model/load-recommended"),
        ("POST", "/api/models/vendor/Some-Model/restart"),
        ("POST", "/api/models/vendor/Some-Model/benchmark"),
        ("POST", "/api/models/vendor/Some-Model/benchmark-parallel"),
        ("GET", "/api/models/vendor/Some-Model/profiles"),
        ("GET", "/api/models/vendor/Some-Model/parallel-observations"),
        ("GET", "/api/vram/holders"),
        # A POST by shape, a read by effect: it execs one binary's --help and
        # compares strings. Gating it only stopped a remote operator checking
        # expert flags BEFORE the save that makes them stick -- and that save
        # (PUT /settings) is gated, which is where the box change happens (D49-11).
        ("POST", "/api/engine/validate-flags"),
    ],
)
def test_reads_inference_and_residency_stay_open_from_the_lan(method: str, path: str) -> None:
    config = open_config()
    assert not is_admin_mutation(method, path)
    check_request(make_request(path, method=method), config)


def test_local_and_in_process_callers_are_trusted() -> None:
    config = open_config()
    check_request(make_request("/api/config", method="PATCH", client=("127.0.0.1", 40000)), config)
    check_request(make_request("/api/config", method="PATCH", client=("::1", 40000)), config)
    check_request(make_request("/api/config", method="PATCH", client=None), config)


def test_the_pin_admits_a_remote_admin_call_either_way_it_is_sent() -> None:
    config = open_config()
    check_request(
        make_request("/api/config", method="PATCH", headers={"X-MCP-Pin": "12345678"}), config
    )
    # sfctl sends the PIN as the bearer token.
    check_request(
        make_request(
            "/api/restart/server", method="POST", headers={"Authorization": "Bearer 12345678"}
        ),
        config,
    )
    with pytest.raises(AuthError):
        check_request(
            make_request("/api/config", method="PATCH", headers={"X-MCP-Pin": "00000000"}), config
        )


def test_without_a_pin_configured_remote_admin_is_refused_outright() -> None:
    config = open_config(pin=None)
    config.mcp.pin_required = False
    with pytest.raises(AuthError) as excinfo:
        check_request(make_request("/api/config", method="PATCH"), config)
    assert excinfo.value.status_code == 403


LOOPBACK = ("127.0.0.1", 40000)


@pytest.mark.parametrize(
    "origin",
    [
        "http://evil.example",
        "https://evil.example:443",
        "null",
        # Another local web app is also a loopback peer -- and is not this
        # server, so the port is part of the comparison.
        "http://127.0.0.1:8188",
        "http://localhost:8080",
    ],
)
def test_a_cross_site_browser_request_on_loopback_is_not_local(origin: str) -> None:
    """The operator's browser is a loopback peer; any page it shows can drive it."""
    config = open_config()
    headers = {"Origin": origin, "Host": "127.0.0.1:1234"}
    assert cross_site_browser_request(make_request("/api/config", headers=headers))
    with pytest.raises(AuthError) as excinfo:
        check_request(
            make_request("/api/config", method="PATCH", headers=headers, client=LOOPBACK), config
        )
    assert excinfo.value.status_code == 403
    assert excinfo.value.code == "remote_admin_requires_credential"
    # ...and the PIN is withheld from such a page: with ACAO `*` the body is
    # readable cross-origin, and the PIN is the only credential on the box.
    assert not may_reveal_pin(
        make_request("/api/mcp/info", headers=headers, client=LOOPBACK), config
    )


@pytest.mark.parametrize(
    ("origin", "host"),
    [
        ("http://127.0.0.1:1234", "127.0.0.1:1234"),
        ("http://localhost:1234", "LOCALHOST:1234"),
        ("http://[::1]:1234", "[::1]:1234"),
        # Implicit default ports on either side.
        ("http://localhost", "localhost:80"),
        ("http://localhost:80", "localhost"),
    ],
)
def test_a_same_origin_browser_request_is_still_local(origin: str, host: str) -> None:
    config = open_config()
    headers = {"Origin": origin, "Host": host}
    assert not cross_site_browser_request(make_request("/api/config", headers=headers))
    check_request(
        make_request("/api/config", method="PATCH", headers=headers, client=LOOPBACK), config
    )
    assert may_reveal_pin(make_request("/api/mcp/info", headers=headers, client=LOOPBACK), config)


def test_a_cross_site_page_still_gets_in_with_a_real_credential() -> None:
    """The Origin rule removes ambient trust; it is not a CSRF token scheme."""
    evil = {"Origin": "http://evil.example", "Host": "127.0.0.1:1234"}
    config = open_config()
    check_request(
        make_request(
            "/api/config",
            method="PATCH",
            headers={**evil, "X-MCP-Pin": "12345678"},
            client=LOOPBACK,
        ),
        config,
    )
    config.server.api_key = "the-real-key"
    check_request(
        make_request(
            "/api/config",
            method="PATCH",
            headers={**evil, "Authorization": "Bearer the-real-key"},
            client=LOOPBACK,
        ),
        config,
    )
    # With a key set there is no ambient credential to ride, so the reveal
    # rule is the old one: a credential was required to get here.
    assert may_reveal_pin(make_request("/api/mcp/info", headers=evil, client=LOOPBACK), config)


def test_origin_is_only_consulted_when_a_browser_sent_one() -> None:
    from types import SimpleNamespace

    config = open_config()
    # No Origin: sfctl, the watchdog, curl, every OpenAI client.
    check_request(
        make_request("/api/config", method="PATCH", headers={"Host": "x"}, client=LOOPBACK), config
    )
    # Origin without Host is not something a browser produces; pass.
    assert not cross_site_browser_request(
        make_request("/api/config", headers={"Origin": "http://evil.example"})
    )
    # The GUI's in-process shim has neither headers nor a peer.
    shim = SimpleNamespace(app=None)
    assert not cross_site_browser_request(shim)
    assert may_reveal_pin(shim, config)


def test_a_configured_api_key_is_the_credential_as_before() -> None:
    config = open_config()
    config.server.api_key = "the-real-key"
    check_request(
        make_request(
            "/api/config", method="PATCH", headers={"Authorization": "Bearer the-real-key"}
        ),
        config,
    )
    with pytest.raises(AuthError) as excinfo:
        check_request(make_request("/api/config", method="PATCH"), config)
    assert excinfo.value.status_code == 401


# ---------------------------------------------------------------------------
# load argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ctx_size": 0},
        {"ctx_size": -100},
        {"ctx_size": 10**9},
        {"parallel": 0},
        {"parallel": -1},
        {"parallel": 1000},
        {"kv_cache_type": "pwned"},
    ],
)
def test_bad_load_arguments_are_a_400_naming_the_parameter(kwargs: dict[str, Any]) -> None:
    args: dict[str, Any] = {"ctx_size": None, "parallel": None, "kv_cache_type": None}
    args.update(kwargs)
    with pytest.raises(BadRequestError) as excinfo:
        validate_load_args(**args)
    (name,) = kwargs
    assert excinfo.value.param == name
    assert name in excinfo.value.message


def test_good_load_arguments_pass() -> None:
    validate_load_args(ctx_size=None, parallel=None, kv_cache_type=None)
    validate_load_args(ctx_size=32768, parallel=4, kv_cache_type="q8_0")
    validate_load_args(ctx_size=1, parallel=1, kv_cache_type="auto")


async def test_the_manager_validates_before_touching_the_registry() -> None:
    from tests.unit.test_load_retry import StubPlanner, StubSupervisor, make_manager

    manager = make_manager(StubSupervisor(), StubPlanner())
    with pytest.raises(BadRequestError):
        await manager.load("test/model", ctx_size=-5)
    with pytest.raises(BadRequestError):
        await manager.load("test/model", parallel=0)


# ---------------------------------------------------------------------------
# ModelSettings bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_value",
    [
        {"parallel": 0},
        {"parallel": -1},
        {"batch_size": 0},
        {"threads": 0},
        {"main_gpu": -1},
        {"cache_reuse": -1},
        {"rope_freq_scale": 0.0},
        {"rope_freq_base": -1.0},
        {"device_override": [0, -1]},
        {"engine_tag": "../../etc"},
        {"engine_tag": "b1/x"},
    ],
)
def test_model_settings_refuse_values_that_would_die_in_the_child(
    field_value: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        ModelSettings(**field_value)


def test_model_settings_keep_the_sane_values() -> None:
    settings = ModelSettings(
        parallel=2,
        batch_size=512,
        main_gpu=0,
        cache_reuse=0,
        rope_freq_scale=0.5,
        device_override=[1, 0],
        engine_tag=" b10425 ",
        reasoning_budget=-1,
    )
    assert settings.engine_tag == "b10425"
    assert settings.reasoning_budget == -1


def test_engine_dir_refuses_a_tag_that_escapes_the_engines_tree(tmp_path: Any) -> None:
    from studioforge.core.engine import EngineError, EngineManager

    manager = EngineManager(Config(data_dir=tmp_path))
    for tag in ("../..", "b1/../../x", "", "..", "C:\\x"):
        with pytest.raises(EngineError):
            manager.engine_dir(tag)
    assert manager.engine_dir("b10425") == tmp_path / "engines" / "b10425"
    assert manager.engine_dir("local-2026.08_cuda13").name == "local-2026.08_cuda13"


# ---------------------------------------------------------------------------
# D32 end to end through the middleware
# ---------------------------------------------------------------------------


def _open_app(tmp_path: Any, pin: str = "12345678", pin_required: bool = True) -> Any:
    from studioforge.api.app import build_state, create_app

    config = Config(
        data_dir=tmp_path / "data",
        server={"host": "0.0.0.0", "port": 1234},
        models={"dir": tmp_path / "models"},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
        mcp={"pin": pin, "pin_required": pin_required},
    )
    return create_app(config, state=build_state(config), start_background=False)


def test_a_cross_site_page_on_the_operators_browser_gets_403_and_no_pin(tmp_path: Any) -> None:
    """Shipped default: CORS `*`, no key. The preflight succeeds for any origin,
    the peer is loopback, and the response is readable cross-origin -- so the
    auth gate, not CORS, has to be what says no."""
    from fastapi.testclient import TestClient

    evil = {"Origin": "http://evil.example"}
    with TestClient(_open_app(tmp_path), client=("127.0.0.1", 50000)) as browser:
        refused = browser.patch("/api/config", json={"models.default_ctx": 4096}, headers=evil)
        assert refused.status_code == 403
        assert refused.json()["error"]["code"] == "remote_admin_requires_credential"
        info = browser.get("/api/mcp/info", headers=evil)
        assert info.status_code == 200
        assert info.json()["pin"] is None
        assert info.json()["pin_note"] == PIN_WITHHELD_NOTE
        # The query carrier is refused outright (D44), and never advertised.
        assert all("?pin=" not in a for a in info.json()["auth"]["alternatives"])
        # Same origin (TestClient's Host is "testserver") is the operator's own tab.
        same = {"Origin": "http://testserver"}
        assert browser.get("/api/mcp/info", headers=same).json()["pin"] == "12345678"
        ok = browser.patch("/api/config", json={"models.default_ctx": 4096}, headers=same)
        assert ok.status_code == 200, ok.text


def test_pin_required_off_does_not_open_the_mcp_plane_to_the_lan(tmp_path: Any) -> None:
    from fastapi.testclient import TestClient

    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "t", "version": "0"},
        },
    }
    accept = {"Accept": "application/json, text/event-stream"}
    with TestClient(_open_app(tmp_path, pin_required=False), client=("192.168.1.50", 50000)) as lan:
        refused = lan.post("/mcp", json=initialize, headers=accept)
        assert refused.status_code == 403
        assert refused.json()["error"]["code"] == "remote_admin_requires_credential"
        # /api/mcp/info tells that caller the truth about what it will be held to.
        info = lan.get("/api/mcp/info").json()
        assert info["pin_required"] is True
        assert info["pin"] is None
        assert info["pin_note"] == PIN_WITHHELD_NOTE
        with_pin = lan.post("/mcp", json=initialize, headers={**accept, "X-MCP-Pin": "12345678"})
        assert with_pin.status_code == 200, with_pin.text
    with TestClient(
        _open_app(tmp_path / "b", pin_required=False), client=("127.0.0.1", 50000)
    ) as local:
        assert local.post("/mcp", json=initialize, headers=accept).status_code == 200
        assert local.get("/api/mcp/info").json()["pin_required"] is False


def test_a_lan_caller_gets_403_on_an_admin_mutation_and_the_pin_admits_it(tmp_path: Any) -> None:
    from fastapi.testclient import TestClient

    app = _open_app(tmp_path)
    with TestClient(app, client=("192.168.1.50", 50000)) as lan:
        refused = lan.patch("/api/config", json={"models.default_ctx": 4096})
        assert refused.status_code == 403
        assert refused.json()["error"]["code"] == "remote_admin_requires_credential"
        # Reads and residency stay open.
        assert lan.get("/api/config").status_code == 200
        assert lan.get("/v1/models").status_code == 200
        allowed = lan.patch(
            "/api/config", json={"models.default_ctx": 4096}, headers={"X-MCP-Pin": "12345678"}
        )
        assert allowed.status_code == 200, allowed.text
    # A second app: the MCP session manager runs once per app instance.
    with TestClient(_open_app(tmp_path / "b"), client=("127.0.0.1", 50000)) as local:
        assert local.patch("/api/config", json={"models.default_ctx": 8192}).status_code == 200


def test_the_open_post_allowlist_admits_validate_flags_and_nothing_beside_it(
    tmp_path: Any,
) -> None:
    """The one exception to the ``/api/engine/`` prefix, end to end (D49-11).

    The prefix rule is deliberately body-blind and path-only. That is the right
    coarseness for ``POST /api/engine/install`` (600 MB onto this disk) and the
    wrong one for ``validate-flags``, which changes nothing. The allowlist is
    checked ahead of the prefix, so this route -- and only this route -- is
    served to a LAN caller with no credential.
    """
    from fastapi.testclient import TestClient

    _reset_guard()
    with TestClient(_open_app(tmp_path), client=("192.168.1.50", 50000)) as lan:
        checked = lan.post("/api/engine/validate-flags", json={"extra_flags": "--top-k 20"})
        assert checked.status_code == 200, checked.text
        assert "errors" in checked.json()

        for path, body in (
            ("/api/engine/install", {"tag": "b10549"}),
            ("/api/engine/activate", {"tag": "b10549"}),
            ("/api/engine/smoke-test", {"tag": "b10549"}),
        ):
            refused = lan.post(path, json=body)
            assert refused.status_code == 403, f"{path}: {refused.text}"
            assert refused.json()["error"]["code"] == "remote_admin_requires_credential"


def test_validate_flags_takes_a_string_or_a_list_and_names_the_shape_otherwise(
    tmp_path: Any,
) -> None:
    """Both call shapes, because both callers exist (D49-11).

    The GUI holds one editable string; ``sfctl`` and most scripts hold an
    argv-shaped list. A list used to 400 with a raw pydantic dump that never
    said which shape was wanted.
    """
    from fastapi.testclient import TestClient

    _reset_guard()
    with TestClient(_open_app(tmp_path), client=("192.168.1.50", 50000)) as lan:
        assert lan.post("/api/engine/validate-flags", json={"extra_flags": ""}).json()["ok"] is True
        as_list = lan.post(
            "/api/engine/validate-flags", json={"extra_flags": ["--top-k", "20", "--min-p"]}
        )
        assert as_list.status_code == 200, as_list.text

        bad = lan.post("/api/engine/validate-flags", json={"extra_flags": {"top_k": 20}})
        assert bad.status_code == 400
        message = bad.json()["error"]["message"]
        assert "list of strings" in message and "string of flags" in message
        assert "got dict" in message

        # A list with a non-string names the offending index rather than
        # answering "got list" to someone who did send a list.
        mixed = lan.post("/api/engine/validate-flags", json={"extra_flags": ["--top-k", 20]})
        assert mixed.status_code == 400
        assert "item 1 is int" in mixed.json()["error"]["message"]


def test_persisting_a_load_profile_is_gated_though_the_route_itself_is_open(
    tmp_path: Any,
) -> None:
    """``POST /load-recommended`` is deliberately ungated -- residency is open,
    LM Studio parity -- but ``persist: true`` writes settings, exactly what
    ``_ADMIN_SETTINGS_SUFFIXES`` protects. ``is_admin_mutation`` decides from
    method and path alone and cannot see a body field, so the handler asks
    (D32/D48). The 404s below are the ordinary unknown-model answer: reaching
    one is how a caller proves it got past the gate.
    """
    from fastapi.testclient import TestClient

    _reset_guard()
    route = "/api/models/vendor/Nope-Q4_K_M/load-recommended"
    persisting = {"ctx_size": 8192, "persist": True}

    with TestClient(_open_app(tmp_path), client=("192.168.1.50", 50000)) as lan:
        refused = lan.post(route, json=persisting)
        assert refused.status_code == 403
        assert refused.json()["error"]["code"] == "remote_admin_requires_credential"
        # The refusal is about the flag, not the route: the same request
        # without it is still served.
        assert lan.post(route, json={"ctx_size": 8192}).status_code == 404
        # ...and the pairing PIN admits the flag, as it does on a gated path.
        with_pin = lan.post(route, json=persisting, headers={"X-MCP-Pin": "12345678"})
        assert with_pin.status_code == 404, with_pin.text
        # The retired query form never satisfies the body gate, not even with
        # the CORRECT pin: it is refused before any comparison, and the refusal
        # says why the URL that used to work does not, rather than reading as a
        # wrong PIN. Pinned here because this gate is checked in the handler,
        # not the middleware, and could regrow the old leniency on its own.
        in_query = lan.post(f"{route}?pin=12345678", json=persisting)
        assert in_query.status_code == 403, in_query.text
        error = in_query.json()["error"]
        assert error["code"] == "remote_admin_requires_credential"
        assert PIN_IN_QUERY_NOTE in error["message"]

    with TestClient(_open_app(tmp_path / "b"), client=("127.0.0.1", 50000)) as local:
        assert local.post(route, json=persisting).status_code == 404


def test_delete_of_an_unloaded_model_is_not_a_500(tmp_path: Any) -> None:
    """``_instance_holding`` called ``supervisor.all()``, which does not exist."""
    from fastapi.testclient import TestClient

    app = _open_app(tmp_path)
    with TestClient(app, client=("127.0.0.1", 50000)) as local:
        response = local.delete("/api/models/vendor/Nope-Q4_K_M")
        # 404 (unknown model), never 500 (AttributeError on the guard).
        assert response.status_code == 404, response.text


def test_mcp_disabled_leaves_the_endpoint_unmounted(tmp_path: Any) -> None:
    from fastapi.testclient import TestClient

    from studioforge.api.app import build_state, create_app

    config = Config(
        data_dir=tmp_path / "data",
        models={"dir": tmp_path / "models"},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
        mcp={"enabled": False, "pin": "12345678"},
    )
    app = create_app(config, state=build_state(config), start_background=False)
    with TestClient(app, client=("127.0.0.1", 50000)) as local:
        assert local.post("/mcp", json={}, headers={"X-MCP-Pin": "12345678"}).status_code == 404
    assert getattr(app.state, "management_mcp", None) is None


# ---------------------------------------------------------------------------
# D44: an eight-digit PIN is only a secret while guessing is slow
# ---------------------------------------------------------------------------


def _reset_guard() -> None:
    from studioforge.api.auth import GUARD

    GUARD.reset()


def test_a_pin_in_the_query_string_is_refused(tmp_path: Any) -> None:
    """It used to be parsed and accepted. A URL is written to reverse-proxy
    access logs, browser history and shell history, none of which expire --
    and neither does the PIN. Header or bearer only."""
    from fastapi.testclient import TestClient

    _reset_guard()
    body = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    accept = {"accept": "application/json, text/event-stream"}
    with TestClient(_open_app(tmp_path), client=("192.168.1.50", 5000)) as lan:
        refused = lan.post("/mcp?pin=12345678", json=body, headers=accept)
        assert refused.status_code == 401, refused.text
        message = refused.json()["error"]["message"]
        # And it explains itself, so a URL that used to work does not merely
        # look like a wrong PIN.
        assert "?pin=" in message and "X-MCP-Pin" in message
    _reset_guard()
    with TestClient(_open_app(tmp_path), client=("192.168.1.50", 5000)) as lan:
        ok = lan.post("/mcp", json=body, headers={**accept, "X-MCP-Pin": "12345678"})
        assert ok.status_code != 401, ok.text


def test_the_watchdog_also_refuses_a_pin_in_the_query_string() -> None:
    """The recovery surface carries the destructive tools; it took ?pin= too."""
    from studioforge.watchdog.server import _pin_from_request

    scope = {"headers": [], "query_string": b"pin=12345678"}
    assert _pin_from_request(scope) is None


def test_wrong_pins_are_locked_out_with_a_doubling_backoff() -> None:
    """Measured on the rig: eight wrong PINs, ~13ms each, no counter, no
    lockout -- 10^8 walked by one machine in hours."""
    _reset_guard()
    from studioforge.api.auth import GUARD, check_request

    config = open_config()
    config.mcp.pin_required = True

    def attempt(pin: str) -> int:
        request = make_request("/mcp", method="POST", headers={"X-MCP-Pin": pin})
        try:
            check_request(request, config)
        except AuthError as exc:
            return exc.status_code
        return 200

    # Three free tries, all plain 401s -- an operator mistyping is not punished.
    assert [attempt("00000000") for _ in range(3)] == [401, 401, 401]
    # The fourth arms the lockout, and the fifth attempt is refused before any
    # comparison happens.
    assert attempt("00000000") == 401
    assert attempt("00000000") == 429
    # Even a CORRECT PIN loses while the lockout stands: refused before the
    # compare, so guessing buys nothing.
    assert attempt("12345678") == 429

    # The 429 tells the caller how long, in the shape sfctl parses.
    request = make_request("/mcp", method="POST", headers={"X-MCP-Pin": "00000000"})
    with pytest.raises(AuthError) as caught:
        check_request(request, config)
    assert caught.value.code == "too_many_credential_attempts"
    assert caught.value.details["retry_after_s"] >= 1

    # A success clears the record.
    GUARD.reset()
    assert attempt("12345678") == 200
    _reset_guard()


def test_the_lockout_only_counts_callers_that_offered_a_credential() -> None:
    """An open install serves plenty of credential-free traffic (that is the
    LM Studio parity the product depends on). Counting those would let a
    chatty poller lock the operator's own address out."""
    _reset_guard()
    from studioforge.api.auth import GUARD, check_request

    config = open_config()
    for _ in range(20):
        check_request(make_request("/v1/models"), config)
    assert GUARD.retry_after("192.168.1.50") == 0.0
    _reset_guard()


def test_a_lockout_is_per_client_address() -> None:
    """One sprayer must not lock the operator out of their own rig."""
    _reset_guard()
    from studioforge.api.auth import GUARD, check_request

    config = open_config()
    config.mcp.pin_required = True
    for _ in range(6):
        with pytest.raises(AuthError):
            check_request(
                make_request(
                    "/mcp",
                    method="POST",
                    headers={"X-MCP-Pin": "00000000"},
                    client=("192.168.1.99", 5000),
                ),
                config,
            )
    assert GUARD.retry_after("192.168.1.99") > 0
    assert GUARD.retry_after("192.168.1.50") == 0.0
    _reset_guard()
