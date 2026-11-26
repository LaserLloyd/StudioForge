"""D92: a load that ran over its plan into a card's headroom is unloaded and refused.

2026-11-25 on the rig: DeepSeek-R1-0528-Qwen3-8B Q4_K_M at 8 x 79,872 (q8_0 KV) on
the 5090 pair was planned at 54,356 MB and measured 59,831 MB (+10.1%). GPU0 was
left with ~1.4 GiB of 32 GiB; WDDM paged the child and it served at ~5 tok/s per
request instead of ~120 for ten hours, with no error. These tests pin the rule
that would have refused it, and that a correctly sized load is untouched.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from studioforge.config import Config
from studioforge.core.manager import ModelManager
from studioforge.core.planner import PREDICTION_ERROR_WARN_PCT, post_load_headroom_shortfall
from studioforge.errors import InsufficientVramError
from studioforge.types import GB, MB, GpuInfo, LoadPlan, VramEstimate


def card(index: int, *, total_gib: float, free_gib: float) -> GpuInfo:
    total = int(total_gib * GB)
    free = int(free_gib * GB)
    return GpuInfo(
        index=index,
        name=f"GPU{index}",
        total_bytes=total,
        free_bytes=free,
        used_bytes=total - free,
        compute_capability=(12, 0),
    )


def plan_of(planned_mb: int, *, devices: list[int] | None = None, parallel: int = 8) -> LoadPlan:
    return LoadPlan(
        model_id="lmstudio-community/DeepSeek-R1-0528-Qwen3-8B-GGUF/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M",
        devices=devices if devices is not None else [0, 1],
        ctx_size=79872 * parallel,
        parallel=parallel,
        estimate=VramEstimate(weights_bytes=planned_mb * MB),
    )


# -- the rule ---------------------------------------------------------------


def test_last_nights_load_is_caught_on_the_card_it_filled() -> None:
    plan = plan_of(54356)
    gpus = [card(0, total_gib=31.84, free_gib=1.4), card(1, total_gib=31.84, free_gib=3.9)]
    short = post_load_headroom_shortfall(plan, 59831 * MB, gpus, headroom_fraction=0.10)
    assert [row["device"] for row in short] == [0]
    assert short[0]["free_bytes"] == int(1.4 * GB)
    assert short[0]["headroom_bytes"] == int(int(31.84 * GB) * 0.10)


def test_a_load_that_measured_what_it_planned_is_never_refused() -> None:
    # The 7-slot reload of the same model: corrected plan, cards still fairly full.
    plan = plan_of(52000, parallel=7)
    gpus = [card(0, total_gib=31.84, free_gib=2.0), card(1, total_gib=31.84, free_gib=2.0)]
    within_bar = int(52000 * MB * (1 + PREDICTION_ERROR_WARN_PCT / 100.0))
    assert post_load_headroom_shortfall(plan, within_bar, gpus, headroom_fraction=0.10) == []


def test_an_overrun_that_still_leaves_the_headroom_is_served() -> None:
    plan = plan_of(20000)
    gpus = [card(0, total_gib=31.84, free_gib=12.0), card(1, total_gib=31.84, free_gib=12.0)]
    assert post_load_headroom_shortfall(plan, 26000 * MB, gpus, headroom_fraction=0.10) == []


def test_only_the_plans_own_cards_are_judged() -> None:
    # A full card the plan does not use (ComfyUI on CUDA2) is not this load's doing.
    plan = plan_of(20000, devices=[0])
    gpus = [card(0, total_gib=31.84, free_gib=10.0), card(2, total_gib=24.0, free_gib=0.5)]
    assert post_load_headroom_shortfall(plan, 26000 * MB, gpus, headroom_fraction=0.10) == []


def test_no_headroom_configured_means_no_refusal() -> None:
    plan = plan_of(54356)
    gpus = [card(0, total_gib=31.84, free_gib=0.2)]
    assert post_load_headroom_shortfall(plan, 59831 * MB, gpus, headroom_fraction=0.0) == []


def test_nothing_measured_is_not_evidence() -> None:
    plan = plan_of(54356)
    gpus = [card(0, total_gib=31.84, free_gib=0.2)]
    assert post_load_headroom_shortfall(plan, 0, gpus, headroom_fraction=0.10) == []


# -- the manager acts on it ---------------------------------------------------


class _Supervisor:
    def __init__(self) -> None:
        self.stopped: list[str] = []

    async def stop(self, model_id: str) -> None:
        self.stopped.append(model_id)


class _Probe:
    def __init__(self, gpus: list[GpuInfo]) -> None:
        self._gpus = gpus

    def list_gpus(self) -> list[GpuInfo]:
        return list(self._gpus)


def _manager(gpus: list[GpuInfo], *, enabled: bool = True) -> Any:
    config = Config()
    config.planner.headroom_fraction = 0.10
    config.planner.post_load_headroom_check = enabled
    return SimpleNamespace(
        config=config,
        planner=SimpleNamespace(probe=_Probe(gpus)),
        supervisor=_Supervisor(),
    )


def _record() -> Any:
    return SimpleNamespace(
        id="lmstudio-community/DeepSeek-R1-0528-Qwen3-8B-GGUF/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M"
    )


async def test_the_manager_unloads_and_answers_507_with_the_numbers() -> None:
    mgr = _manager([card(0, total_gib=31.84, free_gib=1.4), card(1, total_gib=31.84, free_gib=3.9)])
    with pytest.raises(InsufficientVramError) as caught:
        await ModelManager._enforce_post_load_headroom(mgr, _record(), plan_of(54356), 59831 * MB)
    assert mgr.supervisor.stopped == [_record().id]
    exc = caught.value
    assert exc.status_code == 507
    details = exc.details
    assert details["reason"] == "post_load_overrun"
    assert details["requested_parallel"] == 8
    assert details["over_pct"] == pytest.approx(10.1, abs=0.1)
    assert [row["device"] for row in details["short_devices"]] == [0]
    assert "reduce parallel from 8 to 7" in details["suggestions"]
    assert "unloaded rather than served" in str(exc)


async def test_the_manager_serves_a_load_with_room_left() -> None:
    mgr = _manager([card(0, total_gib=31.84, free_gib=4.5), card(1, total_gib=31.84, free_gib=7.0)])
    await ModelManager._enforce_post_load_headroom(
        mgr, _record(), plan_of(52000, parallel=7), 53000 * MB
    )
    assert mgr.supervisor.stopped == []


async def test_the_switch_turns_it_off() -> None:
    mgr = _manager([card(0, total_gib=31.84, free_gib=1.4)], enabled=False)
    await ModelManager._enforce_post_load_headroom(mgr, _record(), plan_of(54356), 59831 * MB)
    assert mgr.supervisor.stopped == []


async def test_an_unmeasured_load_is_served() -> None:
    mgr = _manager([card(0, total_gib=31.84, free_gib=1.4)])
    await ModelManager._enforce_post_load_headroom(mgr, _record(), plan_of(54356), None)
    assert mgr.supervisor.stopped == []
