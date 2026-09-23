"""Every explicit unload is logged with who asked and what it cut (D70, item 5a).

On 2026-09-20 20:48:17 an explicit unload stopped a model with six streams in
flight; the log showed six ``stream failed`` lines and a ``model_stopped``,
and nothing said who had called. The unload paths -- ``POST
/api/models/{id}/unload``, the MCP ``unload_model`` and ``manager.unload``
itself -- logged nothing. Now the manager logs every deliberate unload with
the entry point, the ``X-SF-Client`` label, the peer address, ``force``,
``active_requests`` and the clients in flight at the moment of the stop;
WARNING when that count is above zero, because this path does not drain.

Behaviour is deliberately unchanged: a busy unload still goes through.
Refusing one is a separate decision that waits for the owner.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.core import manager as manager_module
from studioforge.mcp.management import build_management_mcp
from studioforge.types import InFlightRequest, InstanceInfo
from tests.unit.test_catalog_routes import (  # noqa: F401 - fixture and helpers
    MODEL_ID,
    FakeSupervisor,
    app,
    make_plan,
)
from tests.unit.test_load_retry import (
    StubPlanner,
    StubProbe,
    StubSupervisor,
    make_manager,
    resident,
)
from tests.unit.test_mcp import State, call, state  # noqa: F401 - fixture


class RecordingLog:
    """Stands in for the manager module's structlog logger."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def __getattr__(self, level: str) -> Any:
        if level not in {"debug", "info", "warning", "error", "exception"}:
            raise AttributeError(level)

        def emit(event: str, **fields: Any) -> None:
            self.events.append((level, event, fields))

        return emit

    def named(self, prefix: str) -> list[tuple[str, str, dict[str, Any]]]:
        return [entry for entry in self.events if entry[1].startswith(prefix)]


@pytest.fixture()
def log(monkeypatch: pytest.MonkeyPatch) -> RecordingLog:
    recording = RecordingLog()
    monkeypatch.setattr(manager_module, "log", recording)
    return recording


def _manager_with(instance: InstanceInfo) -> Any:
    supervisor, planner = StubSupervisor(), StubPlanner(probe=StubProbe())
    supervisor.instances[instance.model_id] = instance
    return make_manager(supervisor, planner), supervisor


# ---------------------------------------------------------------------------
# manager.unload
# ---------------------------------------------------------------------------


async def test_an_idle_unload_is_logged_at_info_with_the_caller(log: RecordingLog) -> None:
    manager, supervisor = _manager_with(resident("test/model"))
    assert (
        await manager.unload("test/model", source="rest", client="crucibleforge", peer="10.0.0.9")
        is True
    )
    assert supervisor.stopped == ["test/model"], "behaviour unchanged: the model stopped"
    (level, event, fields), *rest = log.named("explicit unload")
    assert not rest
    assert (level, event) == ("info", "explicit unload")
    assert fields["model_id"] == "test/model"
    assert fields["source"] == "rest"
    assert fields["client"] == "crucibleforge"
    assert fields["peer"] == "10.0.0.9"
    assert fields["force"] is False
    assert fields["active_requests"] == 0
    assert fields["in_flight_clients"] == []


async def test_a_busy_unload_warns_and_names_the_clients_it_cuts(log: RecordingLog) -> None:
    """The 2026-09-20 shape: six streams, and now a line that says whose."""
    busy = resident("test/model")
    busy.active_requests = 6
    busy.loaded_by = "jit:/v1/chat/completions (clawchat)"
    busy.in_flight = [
        InFlightRequest(id=f"r{i}", started_at=1.0, client="clawchat" if i % 2 else "openclaw")
        for i in range(6)
    ]
    manager, supervisor = _manager_with(busy)
    assert await manager.unload("test/model", source="rest", peer="10.0.0.9", force=True) is True
    assert supervisor.stopped == ["test/model"], "still not refused (that is item 5b)"
    (level, event, fields), *rest = log.named("explicit unload")
    assert not rest
    assert level == "warning"
    assert event == "explicit unload cuts live requests"
    assert fields["active_requests"] == 6
    assert fields["in_flight_clients"] == ["clawchat", "openclaw"]
    assert fields["loaded_by"] == "jit:/v1/chat/completions (clawchat)"
    assert fields["force"] is True
    assert fields["client"] is None, "no label sent: the field says so rather than guessing"


async def test_an_unattributed_caller_is_logged_as_in_process(log: RecordingLog) -> None:
    manager, _supervisor = _manager_with(resident("test/model"))
    await manager.unload("test/model")
    (_level, _event, fields), *_ = log.named("explicit unload")
    assert fields["source"] == "in-process"
    assert fields["client"] is None and fields["peer"] is None


async def test_housekeeping_unloads_are_not_explicit(log: RecordingLog) -> None:
    """The benchmarks' leave-as-found unload chose nothing; it stays quiet here."""
    manager, supervisor = _manager_with(resident("test/model"))
    assert await manager.unload("test/model", deliberate=False) is True
    assert supervisor.stopped == ["test/model"]
    assert log.named("explicit unload") == []


async def test_a_model_that_is_not_resident_logs_nothing(log: RecordingLog) -> None:
    supervisor, planner = StubSupervisor(), StubPlanner(probe=StubProbe())
    manager = make_manager(supervisor, planner)
    assert await manager.unload("test/model", source="rest") is False
    assert log.named("explicit unload") == []


async def test_unload_all_logs_one_line_per_resident(log: RecordingLog) -> None:
    idle = resident("test/model")
    busy = resident("busy/model")
    busy.active_requests = 2
    supervisor, planner = StubSupervisor(), StubPlanner(probe=StubProbe())
    supervisor.instances[idle.model_id] = idle
    supervisor.instances[busy.model_id] = busy

    async def stop_all(**_kwargs: Any) -> dict[str, BaseException | None]:
        outcome: dict[str, BaseException | None] = dict.fromkeys(supervisor.instances)
        supervisor.instances.clear()
        return outcome

    supervisor.stop_all = stop_all  # type: ignore[attr-defined]
    manager = make_manager(supervisor, planner)
    unloaded = await manager.unload_all(source="rest", client="ops", peer="10.0.0.9")
    assert sorted(unloaded) == ["busy/model", "test/model"]
    lines = log.named("explicit unload-all")
    assert {(level, event, fields["model_id"]) for level, event, fields in lines} == {
        ("info", "explicit unload-all", "test/model"),
        ("warning", "explicit unload-all cuts live requests", "busy/model"),
    }
    assert all(fields["source"] == "rest" and fields["client"] == "ops" for _, _, fields in lines)


# ---------------------------------------------------------------------------
# The entry points say who they are
# ---------------------------------------------------------------------------


def test_the_rest_route_passes_peer_label_and_source(
    app: Any,  # noqa: F811 - the imported fixture
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []

    async def record_unload(model_id: str, **kwargs: Any) -> bool:
        seen.append({"model_id": model_id, **kwargs})
        return True

    monkeypatch.setattr(app.state.manager, "unload", record_unload)
    with TestClient(app) as http:
        response = http.post(
            f"/api/models/{MODEL_ID}/unload", headers={"X-SF-Client": "crucibleforge-judge"}
        )
    assert response.status_code == 200
    assert response.json() == {"model_id": MODEL_ID, "unloaded": True}
    assert seen == [
        {
            "model_id": MODEL_ID,
            "force": False,
            "source": "rest",
            "client": "crucibleforge-judge",
            "peer": "testclient",
        }
    ]


def test_the_rest_unload_all_route_passes_the_same(
    app: Any,  # noqa: F811 - the imported fixture
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []

    async def record_unload_all(**kwargs: Any) -> list[str]:
        seen.append(dict(kwargs))
        return []

    monkeypatch.setattr(app.state.manager, "unload_all", record_unload_all)
    app.state.supervisor = FakeSupervisor([])
    app.state.manager.supervisor = app.state.supervisor
    with TestClient(app) as http:
        response = http.post("/api/models/unload-all")
    assert response.status_code == 200
    assert seen == [
        {"force": False, "source": "rest", "client": "testclient", "peer": "testclient"}
    ]


async def test_the_mcp_tool_names_itself(
    state: State,  # noqa: F811 - the imported fixture
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []

    async def record_unload(model_id: str, **kwargs: Any) -> bool:
        seen.append({"model_id": model_id, **kwargs})
        return False

    monkeypatch.setattr(state.manager, "unload", record_unload)
    server = build_management_mcp(state)
    payload = await call(server, "unload_model", model_id="anything")
    assert payload == {"ok": True, "model_id": "anything", "unloaded": False}
    assert seen == [{"model_id": "anything", "force": True, "source": "mcp"}]
