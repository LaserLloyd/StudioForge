"""Client attribution and in-flight request records (D70, items 20 + 21).

Two facts nothing recorded before D70: WHO loaded a model (``loaded_by`` was
the route literal every caller passed) and WHOSE requests are running against
it (``active_requests`` was a bare count). The ``X-SF-Client`` label now rides
on the load's ``source`` -- ``jit:/v1/chat/completions (clawchat)``, the route
literal kept in front -- and the supervisor keeps a bounded ``in_flight``
window of ``{id, started_at, client}`` beside the count, removed on every end.
``sfctl status`` renders both as ``Client`` and ``Started`` columns.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from studioforge_companion.cli import in_flight_cells

from studioforge.api import openai_routes
from studioforge.core import supervisor as supervisor_module
from studioforge.core.attribution import (
    MAX_CLIENT_LABEL,
    attributed_source,
    client_label,
    client_of,
    split_source,
)
from studioforge.core.supervisor import IN_FLIGHT_RECORDS_MAX, Supervisor
from studioforge.errors import ModelBusyError
from studioforge.mcp.management import _compact_instance
from studioforge.types import InFlightRequest, InstanceInfo
from tests.unit.test_catalog_routes import (  # noqa: F401 - fixture and helpers
    MODEL_ID,
    app,
    loaded,
    make_plan,
)
from tests.unit.test_companion import (  # noqa: F401 - fixture and helpers
    ServerHandle,
    _all_output,
    _invoke,
    cli_module,
    live_server,
)
from tests.unit.test_gateway_lifecycle import CountingSupervisor, FakeState
from tests.unit.test_gateway_lifecycle import make_record as lifecycle_record
from tests.unit.test_supervisor import (  # noqa: F401 - fixtures and helpers
    config,
    fake_binary,
    fake_sup,
    make_binary,
    sup,
)
from tests.unit.test_supervisor import make_plan as sup_plan
from tests.unit.test_supervisor import make_record as sup_record

# ---------------------------------------------------------------------------
# The label and the source convention
# ---------------------------------------------------------------------------


def test_client_label_is_cleaned_and_bounded() -> None:
    assert client_label(None) is None
    assert client_label("   ") is None
    assert client_label("  crucible \t forge ") == "crucible forge"
    assert client_label("bot (judge)") == "bot judge", "parentheses are the source delimiter"
    assert client_label("x" * 500) == "x" * MAX_CLIENT_LABEL


def test_attributed_source_keeps_the_route_literal_in_front() -> None:
    assert attributed_source("jit:/v1/chat/completions", "clawchat") == (
        "jit:/v1/chat/completions (clawchat)"
    )
    assert attributed_source("jit:/v1/chat/completions", None) == "jit:/v1/chat/completions"
    assert attributed_source("mcp:load_model", "  ") == "mcp:load_model"


def test_split_source_reads_back_what_attributed_source_wrote() -> None:
    composed = attributed_source("api:/api/models/{id}/load", "crucibleforge-judge")
    assert split_source(composed) == ("api:/api/models/{id}/load", "crucibleforge-judge")
    assert split_source("mcp:load_model") == ("mcp:load_model", None)
    assert split_source("gui") == ("gui", None)
    assert split_source(None) == (None, None)
    assert split_source("") == ("", None)


def test_client_of_prefers_the_label_and_falls_back_to_the_peer() -> None:
    labelled = SimpleNamespace(
        headers={"x-sf-client": " clawchat "}, client=SimpleNamespace(host="10.0.0.5")
    )
    unlabelled = SimpleNamespace(headers={}, client=SimpleNamespace(host="10.0.0.5"))
    in_process = SimpleNamespace(headers={}, client=None)
    assert client_of(labelled) == "clawchat"
    assert client_of(unlabelled) == "10.0.0.5", "the unlabelled caller is the one to catch"
    assert client_of(in_process) is None
    assert client_of(object()) is None


# ---------------------------------------------------------------------------
# The supervisor's window
# ---------------------------------------------------------------------------


def _resident(supervisor: Supervisor, tmp_path: Path, model_id: str = "qwen2.5-7b") -> InstanceInfo:
    """A ready instance in the table without spawning anything."""
    record = sup_record(tmp_path, model_id)
    plan = sup_plan(model_id)
    info = InstanceInfo(model_id=model_id, state="ready", port=18100, plan=plan)
    supervisor._instances[model_id] = supervisor_module._Instance(
        info=info,
        record=record,
        plan=plan,
        port=18100,
        engine_tag=None,
        draft=None,
        adapters=(),
        log_path=tmp_path / "child.log",
    )
    return info


def test_a_start_describes_the_request_and_an_end_removes_exactly_it(
    config: Any,  # noqa: F811 - the imported fixture
    tmp_path: Path,
) -> None:
    supervisor = sup(config, make_binary(tmp_path))
    info = _resident(supervisor, tmp_path)
    before = time.time()
    first = supervisor.mark_request_start(info.model_id, client="clawchat")
    second = supervisor.mark_request_start(info.model_id, client="10.0.0.5")
    assert first and second and first != second
    assert info.active_requests == 2
    assert [entry.client for entry in info.in_flight] == ["clawchat", "10.0.0.5"]
    assert [entry.id for entry in info.in_flight] == [first, second]
    assert all(before <= entry.started_at <= time.time() for entry in info.in_flight)

    supervisor.mark_request_end(info.model_id, request_id=first, tokens_per_second=12.5)
    assert info.active_requests == 1
    assert [entry.id for entry in info.in_flight] == [second], (
        "the ended one left, the other stayed"
    )
    assert info.last_tokens_per_second == 12.5

    supervisor.mark_request_end(info.model_id, request_id=second)
    assert info.active_requests == 0
    assert info.in_flight == []


def test_an_end_without_an_id_trims_the_oldest(
    config: Any,  # noqa: F811 - the imported fixture
    tmp_path: Path,
) -> None:
    """The pre-D70 callers (benchmarks, the smoke test) still work, FIFO."""
    supervisor = sup(config, make_binary(tmp_path))
    info = _resident(supervisor, tmp_path)
    supervisor.mark_request_start(info.model_id, client="old")
    kept = supervisor.mark_request_start(info.model_id, client="new")
    supervisor.mark_request_end(info.model_id)
    assert [entry.id for entry in info.in_flight] == [kept]
    supervisor.mark_request_end(info.model_id)
    assert info.in_flight == [] and info.active_requests == 0


def test_the_window_is_bounded_and_never_longer_than_the_count(
    config: Any,  # noqa: F811 - the imported fixture
    tmp_path: Path,
) -> None:
    supervisor = sup(config, make_binary(tmp_path))
    info = _resident(supervisor, tmp_path)
    ids = [supervisor.mark_request_start(info.model_id) for _ in range(IN_FLIGHT_RECORDS_MAX + 5)]
    assert info.active_requests == IN_FLIGHT_RECORDS_MAX + 5, "every request is counted"
    assert len(info.in_flight) == IN_FLIGHT_RECORDS_MAX, "past the cap: counted, not described"
    # The undescribed ones end first: nothing is removed for them, and the
    # window stays within the count.
    for request_id in ids[IN_FLIGHT_RECORDS_MAX:]:
        supervisor.mark_request_end(info.model_id, request_id=request_id)
    assert len(info.in_flight) == IN_FLIGHT_RECORDS_MAX == info.active_requests
    for request_id in ids[:IN_FLIGHT_RECORDS_MAX]:
        supervisor.mark_request_end(info.model_id, request_id=request_id)
        assert len(info.in_flight) <= info.active_requests
    assert info.in_flight == [] and info.active_requests == 0


def test_unknown_models_are_a_no_op(
    config: Any,  # noqa: F811 - the imported fixture
    tmp_path: Path,
) -> None:
    supervisor = sup(config, make_binary(tmp_path))
    assert supervisor.mark_request_start("nope", client="x") is None
    supervisor.mark_request_end("nope", request_id="whatever")


def test_the_window_serialises_for_api_status() -> None:
    info = InstanceInfo(
        model_id="m",
        state="ready",
        in_flight=[InFlightRequest(id="abc", started_at=1.5, client="clawchat")],
        loaded_by="jit:/v1/chat/completions (clawchat)",
        loaded_by_client="clawchat",
    )
    dumped = info.model_dump(mode="json")
    assert dumped["in_flight"] == [{"id": "abc", "started_at": 1.5, "client": "clawchat"}]
    assert dumped["loaded_by_client"] == "clawchat"
    assert InstanceInfo(model_id="m", state="ready").in_flight == [], "additive: default empty"


async def test_start_stamps_the_loading_client_from_the_source(
    config: Any,  # noqa: F811 - the imported fixture
    tmp_path: Path,
    fake_binary: Path,  # noqa: F811 - the imported fixture
) -> None:
    supervisor = fake_sup(config, fake_binary)
    record = sup_record(tmp_path)
    info = await supervisor.start(
        record, sup_plan(), source=attributed_source("jit:/v1/chat/completions", "clawchat")
    )
    try:
        assert info.loaded_by == "jit:/v1/chat/completions (clawchat)"
        assert info.loaded_by_client == "clawchat"
    finally:
        await supervisor.aclose()


async def test_start_without_a_label_leaves_loaded_by_exactly_as_before(
    config: Any,  # noqa: F811 - the imported fixture
    tmp_path: Path,
    fake_binary: Path,  # noqa: F811 - the imported fixture
) -> None:
    supervisor = fake_sup(config, fake_binary)
    info = await supervisor.start(sup_record(tmp_path), sup_plan(), source="mcp:load_model")
    try:
        assert info.loaded_by == "mcp:load_model"
        assert info.loaded_by_client is None
    finally:
        await supervisor.aclose()


# ---------------------------------------------------------------------------
# The routes thread the label through
# ---------------------------------------------------------------------------


async def test_the_stream_carries_the_client_into_the_record_and_ends_it() -> None:
    supervisor = CountingSupervisor()
    state = FakeState(supervisor, [b'data: {"delta":1}\n\n', b"data: [DONE]\n\n"])
    record = lifecycle_record()
    chunks = [
        chunk
        async for chunk in openai_routes._stream_upstream(
            state, record, "http://x/v1/chat/completions", {}, 0.0, client="clawchat"
        )
    ]
    assert b"[DONE]" in b"".join(chunks)
    assert supervisor.clients == ["clawchat"]
    assert supervisor.ended == ["req-1"], "the end hands back the id the start returned"


async def test_a_disconnect_still_ends_the_record() -> None:
    """The record leaves on the cancel path too: it is in the same finally."""
    supervisor = CountingSupervisor()
    state = FakeState(supervisor, [b'data: {"delta":1}\n\n', b'data: {"delta":2}\n\n'])
    record = lifecycle_record()
    stream = openai_routes._stream_upstream(
        state, record, "http://x/v1/chat/completions", {}, 0.0, client="clawchat"
    )
    await stream.__anext__()
    await stream.aclose()
    assert supervisor.ended == ["req-1"]
    assert supervisor.active == 0


def test_a_jit_load_carries_the_label_on_its_source(
    app: Any,  # noqa: F811 - the imported fixture
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []

    async def refuse(model_id: str, **kwargs: Any) -> Any:
        seen.append({"model_id": model_id, **kwargs})
        raise ModelBusyError("busy", details={"retry_after_s": 1})

    monkeypatch.setattr(app.state.manager, "ensure_loaded", refuse)
    body = {"model": MODEL_ID, "messages": [{"role": "user", "content": "hi"}]}
    with TestClient(app) as http:
        labelled = http.post("/v1/chat/completions", json=body, headers={"X-SF-Client": "clawchat"})
        unlabelled = http.post("/v1/chat/completions", json=body)
    assert labelled.status_code == unlabelled.status_code == 503
    assert seen[0]["source"] == "jit:/v1/chat/completions (clawchat)"
    # No label: the peer address is the attribution -- the caller to catch.
    assert seen[1]["source"] == "jit:/v1/chat/completions (testclient)"


def test_an_explicit_load_route_carries_the_label_on_its_source(
    app: Any,  # noqa: F811 - the imported fixture
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []

    async def record_load(model_id: str, **kwargs: Any) -> InstanceInfo:
        seen.append({"model_id": model_id, **kwargs})
        return InstanceInfo(model_id=model_id, state="ready", plan=make_plan())

    monkeypatch.setattr(app.state.manager, "load", record_load)
    with TestClient(app) as http:
        response = http.post(
            f"/api/models/{MODEL_ID}/load", json={}, headers={"X-SF-Client": "crucibleforge"}
        )
    assert response.status_code == 200, response.text
    assert seen[0]["source"] == "api:/api/models/{id}/load (crucibleforge)"


# ---------------------------------------------------------------------------
# Where it shows: /api/status, the MCP compact row, sfctl
# ---------------------------------------------------------------------------


def _busy_instance(model_id: str = MODEL_ID) -> InstanceInfo:
    now = time.time()
    return InstanceInfo(
        model_id=model_id,
        state="ready",
        port=18100,
        plan=make_plan(model_id),
        active_requests=2,
        loaded_by="jit:/v1/chat/completions (clawchat)",
        loaded_by_client="clawchat",
        in_flight=[
            InFlightRequest(id="aaa", started_at=now - 125.0, client="clawchat"),
            InFlightRequest(id="bbb", started_at=now - 3.0, client="openclaw"),
        ],
    )


def test_api_status_rows_carry_the_window_and_the_loading_client(
    app: Any,  # noqa: F811 - the imported fixture
) -> None:
    loaded(app, _busy_instance())
    with TestClient(app) as http:
        row = http.get("/api/status").json()["loaded"][0]
    assert row["active_requests"] == 2
    assert row["loaded_by_client"] == "clawchat"
    assert [entry["client"] for entry in row["in_flight"]] == ["clawchat", "openclaw"]
    assert set(row["in_flight"][0]) == {"id", "started_at", "client"}


def test_the_mcp_compact_row_carries_both_cheaply() -> None:
    row = _compact_instance(_busy_instance())
    assert row["loaded_by"] == "jit:/v1/chat/completions (clawchat)"
    assert row["loaded_by_client"] == "clawchat"
    assert row["in_flight"] == [
        {
            "id": "aaa",
            "started_at": pytest.approx(time.time() - 125.0, abs=5),
            "client": "clawchat",
        },
        {"id": "bbb", "started_at": pytest.approx(time.time() - 3.0, abs=5), "client": "openclaw"},
    ]


def test_in_flight_cells_busy_idle_and_legacy() -> None:
    now = time.time()
    busy = {
        "in_flight": [
            {"id": "a", "started_at": now - 125.0, "client": "clawchat"},
            {"id": "b", "started_at": now - 3.0, "client": "openclaw"},
            {"id": "c", "started_at": now - 1.0, "client": "clawchat"},
        ]
    }
    assert in_flight_cells(busy) == ("clawchat, openclaw", "2m05s")
    assert in_flight_cells({"in_flight": [], "loaded_by_client": "crucibleforge"}) == (
        "loaded by crucibleforge",
        "-",
    )
    assert in_flight_cells({"active_requests": 0}) == ("-", "-"), "a pre-D70 server"
    assert in_flight_cells({"in_flight": [{"client": None}]}) == ("?", "-"), (
        "malformed, not a crash"
    )


def test_sfctl_status_renders_client_and_started_columns(
    monkeypatch: Any,
    live_server: ServerHandle,  # noqa: F811 - the imported fixture
) -> None:
    real_status = cli_module.StudioForgeClient.status
    now = time.time()

    def _row(model_id: str, **extra: Any) -> dict[str, Any]:
        return {
            "model_id": model_id,
            "state": "ready",
            "plan": {"ctx_size": 4096},
            "port": 9001,
            "pid": 9004,
            "active_requests": 0,
            **extra,
        }

    async def status_with_rows(self: Any) -> Any:
        payload = await real_status(self)
        payload["loaded"] = [
            _row(
                "busy/alpha",
                active_requests=2,
                in_flight=[
                    {"id": "a", "started_at": now - 125.0, "client": "clawchat"},
                    {"id": "b", "started_at": now - 3.0, "client": "openclaw"},
                ],
            ),
            _row("idle/beta", loaded_by_client="crucibleforge"),
            _row("legacy/gamma"),
        ]
        return payload

    monkeypatch.setattr(cli_module.StudioForgeClient, "status", status_with_rows)
    result = _invoke(live_server, "status", env={"COLUMNS": "240"})
    assert result.exit_code == 0, _all_output(result)
    output = _all_output(result)
    assert output.index("Active") < output.index("Client") < output.index("Started")
    assert output.index("Started") < output.index("tok/s")
    busy = next(line for line in output.splitlines() if "busy/alpha" in line)
    idle = next(line for line in output.splitlines() if "idle/beta" in line)
    legacy = next(line for line in output.splitlines() if "legacy/gamma" in line)
    assert "clawchat, openclaw" in busy and "2m05s" in busy
    assert "loaded by crucibleforge" in idle
    assert "loaded by" not in legacy and "clawchat" not in legacy
