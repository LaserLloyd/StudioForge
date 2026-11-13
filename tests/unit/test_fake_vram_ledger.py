"""The fake VRAM ledger (D85): synthetic free memory follows the child's lifecycle.

Before this, nothing in ``src`` ever called ``FakeGpuProbe.consume``: a staging
instance on synthetic cards would load three 30 GB models onto the same 32 GB
card because free memory never moved. Now the supervisor charges a ready stub
child to the probe and credits it back when the child goes, and
``measure_child_vram`` reads the same ledger so the D51 observation row is
written for a stub load.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from studioforge.config import Config
from studioforge.core.gpu import FakeGpuProbe
from studioforge.core.manager import measure_child_vram
from studioforge.core.supervisor import Supervisor
from studioforge.types import GB, GpuInfo, ModelSettings, VramEstimate
from tests.unit.test_supervisor import (
    FAKE_CHILD,
    TEST_PORT_END,
    TEST_PORT_START,
    make_plan,
    make_record,
    resolver,
    wait_for,
)

MIB = 1024 * 1024


def cards(*sizes_gb: int) -> FakeGpuProbe:
    return FakeGpuProbe(
        [
            GpuInfo(
                index=i,
                name="RTX 5090" if gb >= 32 else "RTX 3090",
                total_bytes=gb * GB,
                free_bytes=gb * GB,
                compute_capability=(12, 0) if gb >= 32 else (8, 6),
            )
            for i, gb in enumerate(sizes_gb)
        ]
    )


def free(probe: FakeGpuProbe) -> list[int]:
    return [g.free_bytes for g in probe.list_gpus()]


# ---------------------------------------------------------------------------
# The ledger itself
# ---------------------------------------------------------------------------


def test_attach_takes_memory_and_release_gives_it_back() -> None:
    probe = cards(32, 24)
    taken = probe.attach_child(111, {0: 10 * GB, 1: 6 * GB})
    assert taken == {0: 10 * GB, 1: 6 * GB}
    assert free(probe) == [22 * GB, 18 * GB]
    assert [g.used_bytes for g in probe.list_gpus()] == [10 * GB, 6 * GB]
    assert probe.child_vram(111) == {0: 10 * GB, 1: 6 * GB}
    assert probe.release_child(111) == 16 * GB
    assert free(probe) == [32 * GB, 24 * GB]
    assert probe.child_vram(111) is None


def test_release_of_an_unknown_pid_is_a_no_op() -> None:
    probe = cards(32)
    assert probe.release_child(999) == 0
    assert free(probe) == [32 * GB]


def test_attach_clamps_to_free_memory_and_skips_unknown_cards() -> None:
    probe = cards(8)
    probe.set_free(0, 3 * GB)
    taken = probe.attach_child(1, {0: 5 * GB, 7: GB})
    assert taken == {0: 3 * GB}
    assert free(probe) == [0]
    probe.release_child(1)
    assert free(probe) == [3 * GB]  # gives back what it took, not what it asked for


def test_reattaching_a_pid_does_not_double_charge() -> None:
    probe = cards(32)
    probe.attach_child(5, {0: 4 * GB})
    probe.attach_child(5, {0: 4 * GB})
    assert free(probe) == [28 * GB]


def test_two_children_are_independent() -> None:
    probe = cards(32)
    probe.attach_child(1, {0: 10 * GB})
    probe.attach_child(2, {0: 5 * GB})
    probe.release_child(1)
    assert free(probe) == [27 * GB]


def test_ledger_children_show_up_as_vram_holders() -> None:
    probe = cards(32, 24)
    probe.attach_child(321, {0: 2 * GB, 1: GB})
    rows = {(p.gpu_index, p.pid): p.used_bytes for p in probe.compute_processes()}
    assert rows == {(0, 321): 2 * GB, (1, 321): GB}
    probe.release_child(321)
    assert probe.compute_processes() == []


def test_measure_child_vram_reads_the_ledger() -> None:
    probe = cards(32, 24)
    probe.attach_child(77, {0: 3 * GB, 1: 2 * GB})
    assert measure_child_vram(probe, 77, [0, 1]) == (5 * GB, {0: 3 * GB, 1: 2 * GB})
    # Only the plan's devices count, as for a real measurement.
    assert measure_child_vram(probe, 77, [1]) == (2 * GB, {1: 2 * GB})


# ---------------------------------------------------------------------------
# The supervisor's hooks
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config(data_dir=tmp_path / "data")
    cfg.gateway.child_port_start = TEST_PORT_START
    cfg.gateway.child_port_end = TEST_PORT_END
    cfg.gateway.load_timeout_s = 20.0
    cfg.gateway.health_poll_interval_s = 0.05
    cfg.gateway.max_restarts = 0
    cfg.gateway.restart_backoff_s = 0.05
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def child_script(tmp_path: Path) -> Path:
    path = tmp_path / "engine" / "stub_child.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FAKE_CHILD, encoding="utf-8")
    return path


MakeSup = Callable[..., Supervisor]


@pytest.fixture
def make_sup(config: Config, child_script: Path) -> MakeSup:
    def build(probe: FakeGpuProbe, *, stub_children: bool = True) -> Supervisor:
        return Supervisor(
            config,
            resolve_binary=resolver(child_script),
            launch_prefix=[sys.executable, "-u"],
            probe=probe,
            stub_children=stub_children,
        )

    return build


async def test_a_ready_child_is_charged_to_its_cards_and_credited_on_stop(
    make_sup: MakeSup, tmp_path: Path
) -> None:
    probe = cards(32, 24)
    supervisor = make_sup(probe)
    plan = make_plan(devices=[0, 1], per_gpu_bytes={0: 10 * GB, 1: 6 * GB})
    record = make_record(tmp_path)
    try:
        info = await supervisor.start(record, plan)
        assert free(probe) == [22 * GB, 18 * GB]
        assert probe.child_vram(info.pid or -1) == {0: 10 * GB, 1: 6 * GB}
        assert measure_child_vram(probe, info.pid or -1, plan.devices)[0] == 16 * GB

        await supervisor.stop(record.id)
        assert free(probe) == [32 * GB, 24 * GB]
        report = supervisor.unload_report(record.id)
        assert report is not None
        assert report.vram_before_bytes == 16 * GB
        assert report.vram_reclaimed_bytes == 16 * GB
    finally:
        await supervisor.aclose()


async def test_the_estimate_is_split_over_the_devices_when_the_plan_has_no_per_card_bytes(
    make_sup: MakeSup, tmp_path: Path
) -> None:
    probe = cards(32, 24)
    supervisor = make_sup(probe)
    plan = make_plan(devices=[0, 1], estimate=VramEstimate(weights_bytes=8 * GB))
    try:
        await supervisor.start(make_record(tmp_path), plan)
        assert free(probe) == [28 * GB, 20 * GB]
    finally:
        await supervisor.aclose()


async def test_stub_vram_mib_overrides_the_plan_in_the_plans_proportions(
    make_sup: MakeSup, tmp_path: Path
) -> None:
    probe = cards(32, 24)
    supervisor = make_sup(probe)
    plan = make_plan(devices=[0, 1], per_gpu_bytes={0: 9 * GB, 1: 3 * GB})
    record = make_record(tmp_path, settings=ModelSettings(extra_flags="--stub-vram-mib 4000"))
    try:
        await supervisor.start(record, plan)
        assert free(probe) == [32 * GB - 3000 * MIB, 24 * GB - 1000 * MIB]
    finally:
        await supervisor.aclose()


async def test_a_plan_bigger_than_the_free_memory_is_clamped_not_a_failed_load(
    make_sup: MakeSup, tmp_path: Path
) -> None:
    probe = cards(8)
    probe.set_free(0, 2 * GB)
    supervisor = make_sup(probe)
    try:
        info = await supervisor.start(make_record(tmp_path), make_plan(per_gpu_bytes={0: 6 * GB}))
        assert info.state == "ready"
        assert free(probe) == [0]
    finally:
        await supervisor.aclose()
    assert free(probe) == [2 * GB]


async def test_two_children_share_a_card_and_stopping_one_frees_only_its_share(
    make_sup: MakeSup, tmp_path: Path
) -> None:
    probe = cards(32)
    supervisor = make_sup(probe)
    first = make_record(tmp_path, "model-a")
    second = make_record(tmp_path, "model-b")
    try:
        await supervisor.start(first, make_plan("model-a", per_gpu_bytes={0: 12 * GB}))
        await supervisor.start(second, make_plan("model-b", per_gpu_bytes={0: 8 * GB}))
        assert free(probe) == [12 * GB]
        await supervisor.stop("model-a")
        assert free(probe) == [24 * GB]
    finally:
        await supervisor.aclose()
    assert free(probe) == [32 * GB]


async def test_a_crashed_child_gives_its_memory_back(make_sup: MakeSup, tmp_path: Path) -> None:
    probe = cards(32)
    supervisor = make_sup(probe)  # max_restarts = 0: a crash ends the instance
    record = make_record(tmp_path, settings=ModelSettings(extra_flags="--fake-crash-after 0.5"))
    try:
        await supervisor.start(record, make_plan(per_gpu_bytes={0: 10 * GB}))
        assert free(probe) == [22 * GB]
        assert await wait_for(lambda: free(probe) == [32 * GB], timeout=20.0)
    finally:
        await supervisor.aclose()


async def test_a_relaunched_child_is_charged_once(
    make_sup: MakeSup, config: Config, tmp_path: Path
) -> None:
    config.gateway.max_restarts = 2
    probe = cards(32)
    supervisor = make_sup(probe)
    marker = tmp_path / "crashed-once"
    record = make_record(
        tmp_path,
        settings=ModelSettings(
            extra_flags=f"--fake-crash-after 0.5 --fake-crash-once {marker.as_posix()}"
        ),
    )
    try:
        info = await supervisor.start(record, make_plan(per_gpu_bytes={0: 10 * GB}))
        assert await wait_for(lambda: info.restarts == 1 and info.state == "ready", timeout=30.0)
        assert free(probe) == [22 * GB]
    finally:
        await supervisor.aclose()
    assert free(probe) == [32 * GB]


async def test_without_the_stub_flag_the_probe_is_left_alone(
    make_sup: MakeSup, tmp_path: Path
) -> None:
    probe = cards(32)
    supervisor = make_sup(probe, stub_children=False)
    try:
        await supervisor.start(make_record(tmp_path), make_plan(per_gpu_bytes={0: 10 * GB}))
        assert free(probe) == [32 * GB]
        assert probe.child_vram(1) is None
    finally:
        await supervisor.aclose()


async def test_a_probe_without_a_ledger_is_tolerated(
    config: Config, child_script: Path, tmp_path: Path
) -> None:
    class Bare:
        backend = "fake"

        def list_gpus(self) -> list[GpuInfo]:
            return []

    supervisor = Supervisor(
        config,
        resolve_binary=resolver(child_script),
        launch_prefix=[sys.executable, "-u"],
        probe=Bare(),  # type: ignore[arg-type]
        stub_children=True,
    )
    try:
        info = await supervisor.start(make_record(tmp_path), make_plan())
        assert info.state == "ready"
    finally:
        await supervisor.aclose()
