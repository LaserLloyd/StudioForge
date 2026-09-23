"""A refused request is logged at the level its code means (D64, CR-3).

Raised by the ClawChat V14 audit (2026-09-12): 1,745 ``request rejected
code=lease_conflict`` lines at INFO between 09-04 and 09-12. ``lease_conflict``
is the end of the vacate protocol -- the client is told to stop polling and
escalate -- and at INFO no WARNING or ERROR filter ever showed it. Meanwhile
every 503 busy signal and the waitable 507 ``gpu_leased`` went out as ERROR
``request failed`` purely because their status is >= 500.

What these pin: ``lease_conflict``, ``insufficient_vram`` and (D66)
``unsupported_architecture`` are WARNING; the documented wait-and-retry family
is INFO whatever its status; anything else keeps the status rule (a 5xx fault
is ERROR, a 4xx is INFO); and the sets are explicit, so the list in
``docs/OPENCLAW-RIG.md`` and the logger agree.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.api import app as app_module
from studioforge.errors import (
    BadRequestError,
    InsufficientVramError,
    LeaseConflictError,
    ModelBusyError,
    ModelLoadError,
    StudioForgeError,
)
from tests.unit.test_catalog_routes import MODEL_ID, app  # noqa: F401 - the fixture


class RecordingLog:
    """Stands in for the app module's structlog logger (see test_load_recommended.RecordingLog)."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str, dict[str, Any]]] = []

    def _record(self, level: str) -> Any:
        return lambda event, **fields: self.lines.append((level, event, fields))

    def __getattr__(self, name: str) -> Any:
        if name in {"debug", "info", "warning", "error", "exception"}:
            return self._record(name)
        raise AttributeError(name)


@pytest.fixture()
def recorded(monkeypatch: pytest.MonkeyPatch) -> RecordingLog:
    recorder = RecordingLog()
    monkeypatch.setattr(app_module, "log", recorder)
    return recorder


def _level_of(recorded: RecordingLog, exc: StudioForgeError) -> tuple[str, str]:
    recorded.lines.clear()
    app_module.log_rejection(exc)
    assert len(recorded.lines) == 1
    level, event, fields = recorded.lines[0]
    assert fields["code"] == exc.code
    return level, event


def test_lease_conflict_is_a_warning(recorded: RecordingLog) -> None:
    assert _level_of(recorded, LeaseConflictError("held")) == ("warning", "request rejected")


def test_insufficient_vram_is_a_warning_not_an_error(recorded: RecordingLog) -> None:
    """A model that does not fit is a refusal to look at, not a server fault."""
    assert _level_of(recorded, InsufficientVramError("too big")) == ("warning", "request rejected")


@pytest.mark.parametrize(
    "exc",
    [
        ModelBusyError("held", code="priority_hold"),
        ModelBusyError("serving"),
        ModelBusyError("benchmarking", code="benchmark_busy"),
        ModelBusyError("benchmarking", code="model_benchmarking"),
        LeaseConflictError("asked", code="lease_vacating"),
        InsufficientVramError("leased", code="gpu_leased"),
    ],
    ids=lambda e: e.code,
)
def test_the_wait_and_retry_family_is_info_whatever_its_status(
    recorded: RecordingLog, exc: StudioForgeError
) -> None:
    assert _level_of(recorded, exc) == ("info", "request rejected")


def test_a_real_fault_is_still_an_error(recorded: RecordingLog) -> None:
    assert _level_of(recorded, ModelLoadError("exited with code 1")) == ("error", "request failed")


def test_an_ordinary_bad_request_is_still_info(recorded: RecordingLog) -> None:
    assert _level_of(recorded, BadRequestError("nope")) == ("info", "request rejected")


def test_the_sets_are_disjoint_and_the_retry_set_is_the_documented_one() -> None:
    assert not app_module.WARNING_REJECTION_CODES & app_module.RETRY_REJECTION_CODES
    # D66 added the model the installed llama.cpp build cannot load at all: a
    # 400 an operator must see, however patiently a client asks for it.
    assert {
        "lease_conflict",
        "insufficient_vram",
        "unsupported_architecture",
    } == app_module.WARNING_REJECTION_CODES
    # docs/OPENCLAW-RIG.md's closed list, less client_quota (a ClawForge2 code).
    assert {
        "priority_hold",
        "model_busy",
        "benchmark_busy",
        "model_benchmarking",
        "lease_vacating",
        "gpu_leased",
    } == app_module.RETRY_REJECTION_CODES


def test_the_error_handler_uses_it(app: Any, recorded: RecordingLog) -> None:  # noqa: F811 - the imported fixture, by design
    async def refuse(*_a: Any, **_k: Any) -> Any:
        raise LeaseConflictError("CUDA [1] is leased to clawforge2")

    app.state.manager.load = refuse
    with TestClient(app, client=("127.0.0.1", 50000)) as http:
        response = http.post(f"/api/models/{MODEL_ID}/load", json={})
    assert response.status_code == 409
    rejected = [line for line in recorded.lines if line[1] == "request rejected"]
    assert rejected == [
        (
            "warning",
            "request rejected",
            {"error": "CUDA [1] is leased to clawforge2", "code": "lease_conflict", "status": 409},
        )
    ]
