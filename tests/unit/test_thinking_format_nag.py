""" "thinking model loads with no reasoning_format" is said once per model (D69 §16).

The warning describes a per-model setting, not a load, and it was logged on
every load of every thinking model: 136 identical lines between 2026-09-13 and
09-22. It is now a WARNING the first time per model per process and DEBUG
after. The alias-collision half of §16 is in ``test_registry.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from studioforge import logging as sf_logging
from studioforge.config import Config
from studioforge.core import manager as manager_module
from studioforge.core.manager import ModelManager
from studioforge.types import ModelCapabilities
from tests.unit.test_load_retry import StubPlanner, StubRegistry, StubSupervisor, make_record

EVENT = "thinking model loads with no reasoning_format"


class _LevelRecorder:
    def __init__(self) -> None:
        self.lines: list[tuple[str, str, dict[str, Any]]] = []

    def __getattr__(self, level: str) -> Any:
        return lambda event, *_a, **kw: self.lines.append((level, event, kw))

    def of(self, level: str) -> list[str]:
        return [kw["model_id"] for lvl, event, kw in self.lines if lvl == level and event == EVENT]


@pytest.fixture()
def recorder(monkeypatch: pytest.MonkeyPatch) -> Iterator[_LevelRecorder]:
    rec = _LevelRecorder()
    monkeypatch.setattr(manager_module, "log", rec)
    sf_logging.reset_first_time()
    yield rec
    sf_logging.reset_first_time()


def _thinking(model_id: str) -> Any:
    record = make_record(model_id)
    return record.model_copy(update={"capabilities": ModelCapabilities(thinking=True)})


async def test_the_nag_is_a_warning_once_per_model_then_debug(recorder: _LevelRecorder) -> None:
    first, second = _thinking("think/one"), _thinking("think/two")
    supervisor = StubSupervisor()
    manager = ModelManager(
        Config(data_dir="/tmp/sf-thinking-nag"),
        registry=StubRegistry({first.id: first, second.id: second}),  # type: ignore[arg-type]
        planner=StubPlanner(),  # type: ignore[arg-type]
        supervisor=supervisor,  # type: ignore[arg-type]
        db=None,  # type: ignore[arg-type]
    )

    for _ in range(3):
        await manager.load(first.id)
        supervisor.instances.clear()
    await manager.load(second.id)

    assert recorder.of("warning") == [first.id, second.id]
    assert recorder.of("debug") == [first.id, first.id]


async def test_a_model_with_a_reasoning_format_is_never_nagged(recorder: _LevelRecorder) -> None:
    record = _thinking("think/formatted")
    record.settings.reasoning_format = "deepseek"
    manager = ModelManager(
        Config(data_dir="/tmp/sf-thinking-nag"),
        registry=StubRegistry({record.id: record}),  # type: ignore[arg-type]
        planner=StubPlanner(),  # type: ignore[arg-type]
        supervisor=StubSupervisor(),  # type: ignore[arg-type]
        db=None,  # type: ignore[arg-type]
    )
    await manager.load(record.id)
    assert recorder.of("warning") == [] and recorder.of("debug") == []
