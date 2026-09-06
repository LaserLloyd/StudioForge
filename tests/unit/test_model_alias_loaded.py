"""The ``loaded`` model alias (plan item 2.7 / D57).

``auto``/``current``/``default``/``local-model`` all resolve to the static
``models.default_model`` (``_require_model``, config only). ``loaded`` is a
second, separate alias resolved against LIVE server state instead: the
largest resident (``state == "ready"``) instance that still has a free slot
(``active_requests < plan.parallel``). It is applied as its own step
(``_resolve_loaded_alias``), right after ``_require_model`` and before
``_resolve_or_404``, at every one of the five call sites in
``openai_routes.py`` -- so both the streaming and non-streaming halves of
``/v1/chat/completions`` resolve it, because they share that same call site
above the branch, and it never touches the registry's alias table, so
``GET /v1/models`` is never asked to carry a synthetic ``loaded`` entry.

First section below unit-tests the resolution helpers directly (no app, no
HTTP -- fast, precise coverage of the ranking and tie-break rules). Second
section drives the real FastAPI app through ``TestClient`` with a stubbed
upstream, exactly like ``test_request_priority.py``, to prove the alias
resolves through the actual route wiring (both chat-completions branches,
the 503 shape, `/v1/models` non-pollution, and that `default` still works).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.api import openai_routes
from studioforge.api.app import build_state, create_app
from studioforge.api.openai_routes import (
    LOADED_MODEL_ALIAS,
    _largest_ready_free_instance,
    _resolve_loaded_alias,
)
from studioforge.config import Config
from studioforge.errors import NoLoadedModelError
from studioforge.types import GgufMeta, InstanceInfo, InstanceState, LoadPlan, ModelRecord
from tests.unit.test_catalog_routes import FakeProbe, FakeRegistry, FakeSupervisor

MESSAGES = [{"role": "user", "content": "hello"}]


def _record(model_id: str, param_count: int | None) -> ModelRecord:
    return ModelRecord(
        id=model_id,
        name=model_id.rsplit("/", 1)[-1],
        path=Path(f"/models/{model_id.replace('/', '_')}.gguf"),
        meta=GgufMeta(param_count=param_count),
    )


def _plan(model_id: str, *, parallel: int = 2, ctx_size: int = 8192) -> LoadPlan:
    return LoadPlan(
        model_id=model_id,
        devices=[0],
        ctx_size=ctx_size,
        parallel=parallel,
        ctx_per_slot=ctx_size,
        max_parallel=parallel,
    )


def _instance(
    model_id: str,
    *,
    state: InstanceState = "ready",
    plan: LoadPlan | None = None,
    active_requests: int = 0,
) -> InstanceInfo:
    return InstanceInfo(
        model_id=model_id,
        state=state,
        plan=plan if plan is not None else _plan(model_id),
        active_requests=active_requests,
    )


class Registry(FakeRegistry):
    """``FakeRegistry`` plus the ``touch`` every JIT load path calls
    (``ensure_loaded`` -> ``registry.touch``), same as test_request_priority.py."""

    def touch(self, model_id: str) -> None:
        return None


class FakeState:
    """The two attributes ``_largest_ready_free_instance`` reads -- nothing
    else, so a test that reaches past them is a test with the wrong fixture."""

    def __init__(self, registry: FakeRegistry, supervisor: FakeSupervisor) -> None:
        self.registry = registry
        self.supervisor = supervisor


class SupervisorWithBaseUrl(FakeSupervisor):
    """FakeSupervisor plus base_url(), needed only by the streaming path."""

    def base_url(self, model_id: str) -> str | None:
        return "http://127.0.0.1:1/fake"


# ---------------------------------------------------------------------------
# _largest_ready_free_instance -- pure unit tests, no app
# ---------------------------------------------------------------------------


class TestLargestReadyFreeInstance:
    def test_two_residents_the_larger_by_param_count_is_chosen(self) -> None:
        registry = FakeRegistry(
            [_record("v/small", 8_000_000_000), _record("v/big", 27_000_000_000)]
        )
        supervisor = FakeSupervisor([_instance("v/small"), _instance("v/big")])
        instance = _largest_ready_free_instance(FakeState(registry, supervisor))
        assert instance is not None
        assert instance.model_id == "v/big"

    def test_the_largest_full_is_skipped_for_the_next_one_down(self) -> None:
        registry = FakeRegistry(
            [_record("v/small", 8_000_000_000), _record("v/big", 27_000_000_000)]
        )
        supervisor = FakeSupervisor(
            [
                _instance("v/small", active_requests=0),
                _instance("v/big", plan=_plan("v/big", parallel=2), active_requests=2),
            ]
        )
        instance = _largest_ready_free_instance(FakeState(registry, supervisor))
        assert instance is not None
        assert instance.model_id == "v/small", (
            "the big one has no free slot (2 active == parallel 2)"
        )

    def test_no_resident_instances_returns_none(self) -> None:
        assert _largest_ready_free_instance(FakeState(FakeRegistry([]), FakeSupervisor([]))) is None

    def test_every_resident_full_returns_none(self) -> None:
        registry = FakeRegistry([_record("v/a", 1)])
        supervisor = FakeSupervisor(
            [_instance("v/a", plan=_plan("v/a", parallel=1), active_requests=1)]
        )
        assert _largest_ready_free_instance(FakeState(registry, supervisor)) is None

    @pytest.mark.parametrize("bad_state", ["loading", "stopped", "failed", "unloading"])
    def test_non_ready_instances_are_never_candidates(self, bad_state: InstanceState) -> None:
        """A "loading" instance has no proven capacity yet; the rest cannot
        serve at all -- none may outrank a smaller but genuinely ready one."""
        registry = FakeRegistry([_record("v/small", 1), _record("v/big", 999_000_000_000)])
        supervisor = FakeSupervisor(
            [_instance("v/small", state="ready"), _instance("v/big", state=bad_state)]
        )
        instance = _largest_ready_free_instance(FakeState(registry, supervisor))
        assert instance is not None
        assert instance.model_id == "v/small"

    def test_an_instance_with_no_plan_is_never_a_candidate(self) -> None:
        registry = FakeRegistry([_record("v/a", 1)])
        supervisor = FakeSupervisor([InstanceInfo(model_id="v/a", state="ready", plan=None)])
        assert _largest_ready_free_instance(FakeState(registry, supervisor)) is None

    def test_unmeasured_param_count_ties_break_on_context_total(self) -> None:
        """Two records with no GGUF metadata both score 0 on param_count, so
        the tie-break is ctx_total = ctx_size * parallel."""
        registry = FakeRegistry([_record("v/a", None), _record("v/b", None)])
        supervisor = FakeSupervisor(
            [
                _instance("v/a", plan=_plan("v/a", parallel=1, ctx_size=8192)),  # ctx_total 8192
                _instance("v/b", plan=_plan("v/b", parallel=2, ctx_size=8192)),  # ctx_total 16384
            ]
        )
        instance = _largest_ready_free_instance(FakeState(registry, supervisor))
        assert instance is not None
        assert instance.model_id == "v/b"

    def test_a_measured_model_always_outranks_an_unmeasured_one(self) -> None:
        """Even a small measured model must beat a huge but stale/unscanned
        record whose param_count could not be read -- 0 never wins."""
        registry = FakeRegistry([_record("v/measured", 1_000_000), _record("v/unmeasured", None)])
        supervisor = FakeSupervisor(
            [
                _instance("v/measured", plan=_plan("v/measured", ctx_size=2048)),
                _instance("v/unmeasured", plan=_plan("v/unmeasured", ctx_size=999_999)),
            ]
        )
        instance = _largest_ready_free_instance(FakeState(registry, supervisor))
        assert instance is not None
        assert instance.model_id == "v/measured"

    def test_an_instance_missing_from_the_registry_does_not_crash(self) -> None:
        """A child mid-shutdown can outlive its registry row; this must
        compete as unmeasured (0), never raise."""
        supervisor = FakeSupervisor([_instance("v/ghost")])
        instance = _largest_ready_free_instance(FakeState(FakeRegistry([]), supervisor))
        assert instance is not None
        assert instance.model_id == "v/ghost"


# ---------------------------------------------------------------------------
# _resolve_loaded_alias -- pure unit tests, no app
# ---------------------------------------------------------------------------


class TestResolveLoadedAlias:
    def test_alias_constant_is_the_word_loaded(self) -> None:
        assert LOADED_MODEL_ALIAS == "loaded"

    def test_names_other_than_loaded_pass_through_unchanged(self) -> None:
        state = FakeState(FakeRegistry([]), FakeSupervisor([]))
        for name in ("vendor/model-Q4_K_M", "auto", "default", "current", "local-model", ""):
            assert _resolve_loaded_alias(state, name) == name

    @pytest.mark.parametrize("spelling", ["loaded", "LOADED", "Loaded", "  loaded  ", "LoAdEd"])
    def test_case_and_whitespace_insensitive(self, spelling: str) -> None:
        registry = FakeRegistry([_record("v/a", 1)])
        supervisor = FakeSupervisor([_instance("v/a")])
        assert _resolve_loaded_alias(FakeState(registry, supervisor), spelling) == "v/a"

    def test_raises_no_loaded_model_with_the_right_shape_when_nothing_qualifies(self) -> None:
        state = FakeState(FakeRegistry([]), FakeSupervisor([]))
        with pytest.raises(NoLoadedModelError) as exc_info:
            _resolve_loaded_alias(state, "loaded")
        assert exc_info.value.status_code == 503
        assert exc_info.value.code == "no_loaded_model"
        assert exc_info.value.error_type == "server_error"


# ---------------------------------------------------------------------------
# Through the real route wiring: TestClient + a stubbed upstream, exactly
# like test_request_priority.py -- no engine, no network, no rig.
# ---------------------------------------------------------------------------

SMALL_ID = "vendor/small-Q4_K_M"
BIG_ID = "vendor/big-Q5_K_M"


@pytest.fixture()
def app(tmp_path: Path) -> Any:
    config = Config(
        data_dir=tmp_path / "data",
        server={"host": "127.0.0.1", "port": 1234},
        models={"dir": tmp_path / "models"},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
    )
    built = create_app(config, state=build_state(config), start_background=False)
    built.state.registry = Registry(
        [_record(SMALL_ID, 8_000_000_000), _record(BIG_ID, 27_000_000_000)]
    )
    built.state.probe = FakeProbe()
    built.state.planner.probe = FakeProbe()
    built.state.manager.registry = built.state.registry
    # Both residents ready with a free slot: these tests are about which one
    # gets PICKED, not about a load.
    supervisor = SupervisorWithBaseUrl([_instance(SMALL_ID), _instance(BIG_ID)])
    built.state.supervisor = supervisor
    built.state.manager.supervisor = supervisor
    return built


@pytest.fixture()
def upstream(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every payload the gateway hands to llama-server, newest last."""
    sent: list[dict[str, Any]] = []

    async def fake_forward(
        state: Any, record: Any, path: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        sent.append(dict(payload))
        return {"id": "cmpl-test", "object": "chat.completion", "choices": []}

    monkeypatch.setattr(openai_routes, "_forward", fake_forward)
    return sent


def test_non_streaming_chat_completions_resolves_loaded_to_the_largest(
    app: Any, upstream: list[dict[str, Any]]
) -> None:
    with TestClient(app) as http:
        response = http.post("/v1/chat/completions", json={"model": "loaded", "messages": MESSAGES})
    assert response.status_code == 200, response.text
    assert len(upstream) == 1
    assert upstream[0]["model"] == BIG_ID


def test_streaming_chat_completions_resolves_loaded_too(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The streaming branch shares the SAME resolution call site, above the
    `if payload.get("stream")` split -- proved here by stubbing the one
    function only the streaming path calls and reading which record it got."""
    seen: dict[str, Any] = {}

    async def fake_stream_upstream(
        state: Any, record: Any, url: str, payload: dict[str, Any], started: float
    ):
        seen["record_id"] = record.id
        seen["payload_model"] = payload.get("model")
        yield b"data: [DONE]\n\n"

    monkeypatch.setattr(openai_routes, "_stream_upstream", fake_stream_upstream)
    with TestClient(app) as http:
        response = http.post(
            "/v1/chat/completions",
            json={"model": "loaded", "messages": MESSAGES, "stream": True},
        )
    assert response.status_code == 200, response.text
    assert seen.get("record_id") == BIG_ID
    assert seen.get("payload_model") == BIG_ID


def test_loaded_alias_is_case_insensitive_over_http(
    app: Any, upstream: list[dict[str, Any]]
) -> None:
    with TestClient(app) as http:
        response = http.post("/v1/chat/completions", json={"model": "LOADED", "messages": MESSAGES})
    assert response.status_code == 200, response.text
    assert upstream[0]["model"] == BIG_ID


def test_default_alias_still_works_unaffected_by_the_new_one(
    app: Any, upstream: list[dict[str, Any]]
) -> None:
    app.state.config.models.default_model = SMALL_ID
    with TestClient(app) as http:
        response = http.post(
            "/v1/chat/completions", json={"model": "default", "messages": MESSAGES}
        )
    assert response.status_code == 200, response.text
    assert upstream[0]["model"] == SMALL_ID


def test_no_loaded_model_when_nothing_is_resident_is_503(app: Any) -> None:
    empty = SupervisorWithBaseUrl([])
    app.state.supervisor = empty
    app.state.manager.supervisor = empty
    with TestClient(app) as http:
        response = http.post("/v1/chat/completions", json={"model": "loaded", "messages": MESSAGES})
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "no_loaded_model"
    assert body["error"]["type"] == "server_error"
    assert response.headers.get("Retry-After")


def test_no_loaded_model_when_every_resident_is_full_is_503(app: Any) -> None:
    full = SupervisorWithBaseUrl(
        [
            _instance(SMALL_ID, plan=_plan(SMALL_ID, parallel=1), active_requests=1),
            _instance(BIG_ID, plan=_plan(BIG_ID, parallel=1), active_requests=1),
        ]
    )
    app.state.supervisor = full
    app.state.manager.supervisor = full
    with TestClient(app) as http:
        response = http.post("/v1/chat/completions", json={"model": "loaded", "messages": MESSAGES})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "no_loaded_model"


def test_v1_models_is_not_polluted_with_a_synthetic_loaded_entry(app: Any) -> None:
    """Aliases are request-side (plan item 2.7): the catalogue must list only
    the two real models, never a 'loaded' row."""
    with TestClient(app) as http:
        data = http.get("/v1/models").json()["data"]
    ids = {m["id"] for m in data}
    assert ids == {SMALL_ID, BIG_ID}
    assert LOADED_MODEL_ALIAS not in ids
