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

from studioforge.api.auth import REMOTE_ADMIN_NOTE, check_request, is_admin_mutation
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


def _open_app(tmp_path: Any, pin: str = "12345678") -> Any:
    from studioforge.api.app import build_state, create_app

    config = Config(
        data_dir=tmp_path / "data",
        server={"host": "0.0.0.0", "port": 1234},
        models={"dir": tmp_path / "models"},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
        mcp={"pin": pin},
    )
    return create_app(config, state=build_state(config), start_background=False)


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
