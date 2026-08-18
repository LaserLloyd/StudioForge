"""Attribution of VRAM holders, and the guards that keep tests off the GPUs.

Written against the 2026-08-18 incident (DECISIONS.md D23), so the fixtures
reproduce its exact shape: three ``llama-server.exe`` processes under our own
engines directory, children of a ``pytest`` process, invisible to NVML's
per-process accounting (``used_bytes: 0`` on Windows/WDDM) and therefore
unattributable by every surface the product had.

The process table is faked rather than real: the classification rules depend on
parent liveness and pid recycling, and neither can be staged with real
processes without racing the OS.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

import psutil
import pytest

from studioforge.core import gpu as gpu_mod
from studioforge.core import vram_holders as vh
from studioforge.core.gpu import FakeGpuProbe
from studioforge.gui import state as st
from studioforge.types import GpuInfo, VramProcess

GIB = 1024 * 1024 * 1024
MIB = 1024 * 1024

ENGINES = Path(r"C:\data\engines") if os.name == "nt" else Path("/data/engines")
ENGINE_EXE = str(ENGINES / "b10425" / ("llama-server.exe" if os.name == "nt" else "llama-server"))


# ---------------------------------------------------------------------------
# A fake process table
# ---------------------------------------------------------------------------


class FakeProcess:
    """Enough of ``psutil.Process`` for this module's queries."""

    def __init__(
        self,
        pid: int,
        name: str,
        *,
        exe: str = "",
        cmdline: list[str] | None = None,
        create_time: float = 1000.0,
        ppid: int = 1,
        status: str = psutil.STATUS_RUNNING,
        denied: bool = False,
    ) -> None:
        self.pid = pid
        self._name = name
        self._exe = exe
        self._cmdline = cmdline or []
        self._create_time = create_time
        self._ppid = ppid
        self._status = status
        self._denied = denied

    @property
    def info(self) -> dict[str, Any]:
        return {"name": self._name}

    @contextlib.contextmanager
    def oneshot(self) -> Any:
        yield

    def _guard(self) -> None:
        if self._denied:
            raise psutil.AccessDenied(self.pid)

    def name(self) -> str:
        self._guard()
        return self._name

    def exe(self) -> str:
        self._guard()
        return self._exe

    def cmdline(self) -> list[str]:
        self._guard()
        return list(self._cmdline)

    def create_time(self) -> float:
        self._guard()
        return self._create_time

    def ppid(self) -> int:
        self._guard()
        return self._ppid

    def status(self) -> str:
        self._guard()
        return self._status


def install_table(monkeypatch: pytest.MonkeyPatch, procs: list[FakeProcess]) -> None:
    """Replace psutil's process table with ``procs`` for one test."""
    table = {proc.pid: proc for proc in procs}

    def process_iter(_attrs: Any = None) -> list[FakeProcess]:
        return list(procs)

    def process(pid: int) -> FakeProcess:
        found = table.get(int(pid))
        if found is None:
            raise psutil.NoSuchProcess(pid)
        return found

    monkeypatch.setattr(psutil, "process_iter", process_iter)
    monkeypatch.setattr(psutil, "Process", process)


def engine_proc(
    pid: int,
    *,
    alias: str = "qwen",
    port: int = 18101,
    ppid: int = 900,
    create_time: float = 1000.0,
    exe: str = ENGINE_EXE,
    port_equals: bool = False,
) -> FakeProcess:
    port_args = [f"--port={port}"] if port_equals else ["--port", str(port)]
    return FakeProcess(
        pid,
        Path(exe).name,
        exe=exe,
        cmdline=[exe, "--alias", alias, *port_args, "--n-gpu-layers", "999"],
        create_time=create_time,
        ppid=ppid,
    )


def pytest_parent(pid: int = 900, create_time: float = 900.0) -> FakeProcess:
    return FakeProcess(
        pid,
        "python.exe",
        exe=sys.executable,
        cmdline=[sys.executable, "-m", "pytest", "tests", "-q"],
        create_time=create_time,
        ppid=800,
    )


# ---------------------------------------------------------------------------
# find_engine_processes
# ---------------------------------------------------------------------------


def test_only_llama_servers_under_our_engines_dir_are_ours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scoping by directory is a safety property, not a nicety.

    A llama-server started by LM Studio or a second install is a VRAM holder we
    report, never a process we may kill -- so it must not appear here at all.
    """
    foreign = str(Path("C:/lmstudio/llama-server.exe" if os.name == "nt" else "/opt/llama-server"))
    install_table(
        monkeypatch,
        [
            engine_proc(101, ppid=900),
            engine_proc(102, exe=foreign, ppid=900),
            FakeProcess(103, "chrome.exe", exe="C:/chrome.exe", cmdline=["chrome"]),
            pytest_parent(),
        ],
    )
    found = vh.find_engine_processes(ENGINES)
    assert [entry.pid for entry in found] == [101]


def test_alias_and_port_are_extracted_in_both_spellings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_table(
        monkeypatch,
        [
            engine_proc(101, alias="qwen3-8b", port=18101),
            engine_proc(102, alias="embed", port=18102, port_equals=True),
            pytest_parent(),
        ],
    )
    found = {entry.pid: entry for entry in vh.find_engine_processes(ENGINES)}
    assert (found[101].alias, found[101].port) == ("qwen3-8b", 18101)
    assert (found[102].alias, found[102].port) == ("embed", 18102)


def test_a_child_of_a_live_process_is_named_not_orphaned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The incident's actual shape: legitimately in use by somebody else."""
    install_table(monkeypatch, [engine_proc(101, ppid=900), pytest_parent()])
    (entry,) = vh.find_engine_processes(ENGINES)
    assert entry.classification == vh.CLASS_CHILD_OF_LIVE
    assert entry.parent_alive is True
    assert entry.parent_name == "python.exe"
    assert entry.parent_cmdline is not None
    assert "pytest" in entry.parent_cmdline


def test_a_dead_parent_makes_an_orphan(monkeypatch: pytest.MonkeyPatch) -> None:
    install_table(monkeypatch, [engine_proc(101, ppid=900)])  # no parent in the table
    (entry,) = vh.find_engine_processes(ENGINES)
    assert entry.classification == vh.CLASS_ORPHAN
    assert entry.parent_alive is False
    assert entry.parent_name is None


def test_a_recycled_parent_pid_is_still_an_orphan(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process that started AFTER its supposed child did not spawn it.

    Without this guard the busiest boxes -- the ones that recycle pids fastest
    -- are exactly the ones where a leak is filed as "somebody's live child"
    and survives every sweep.
    """
    install_table(
        monkeypatch,
        [
            engine_proc(101, ppid=900, create_time=1000.0),
            FakeProcess(900, "explorer.exe", exe="C:/explorer.exe", create_time=5000.0),
        ],
    )
    (entry,) = vh.find_engine_processes(ENGINES)
    assert entry.classification == vh.CLASS_ORPHAN
    assert entry.parent_recycled is True
    assert entry.parent_name == "explorer.exe"


def test_our_own_children_are_never_orphans(monkeypatch: pytest.MonkeyPatch) -> None:
    install_table(monkeypatch, [engine_proc(101, ppid=900)])
    (entry,) = vh.find_engine_processes(ENGINES, own_pids=[101])
    assert entry.classification == vh.CLASS_OURS
    assert entry.is_ours is True


def test_access_denied_processes_are_skipped_quietly(monkeypatch: pytest.MonkeyPatch) -> None:
    denied = engine_proc(101)
    denied._denied = True
    install_table(monkeypatch, [denied, engine_proc(102, ppid=900), pytest_parent()])
    assert [entry.pid for entry in vh.find_engine_processes(ENGINES)] == [102]


def test_a_zombie_parent_does_not_count_as_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    install_table(
        monkeypatch,
        [
            engine_proc(101, ppid=900),
            FakeProcess(900, "python.exe", create_time=900.0, status=psutil.STATUS_ZOMBIE),
        ],
    )
    (entry,) = vh.find_engine_processes(ENGINES)
    assert entry.classification == vh.CLASS_ORPHAN


# ---------------------------------------------------------------------------
# reclaim_orphans
# ---------------------------------------------------------------------------


def test_reclaim_kills_orphans_and_spares_everything_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_table(
        monkeypatch,
        [
            engine_proc(101, alias="leaked", ppid=900),  # parent absent -> orphan
            engine_proc(102, alias="in-use", ppid=901),
            engine_proc(103, alias="ours", ppid=902),
            pytest_parent(901),
            FakeProcess(902, "python.exe", create_time=800.0),
        ],
    )
    killed: list[int] = []
    monkeypatch.setattr(vh, "kill_process_tree", lambda pid, **_kw: killed.append(pid))
    monkeypatch.setattr(vh, "process_is_alive", lambda pid, **_kw: pid not in killed)

    actions = vh.reclaim_orphans(ENGINES, own_pids=[103])
    assert killed == [101]
    assert [action["alias"] for action in actions] == ["leaked"]
    assert actions[0]["killed"] is True


def test_dry_run_kills_nothing_but_reports_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_table(monkeypatch, [engine_proc(101, alias="leaked", ppid=900)])
    monkeypatch.setattr(
        vh, "kill_process_tree", lambda *a, **k: pytest.fail("dry run must not kill")
    )
    actions = vh.reclaim_orphans(ENGINES, dry_run=True)
    assert [(a["pid"], a["dry_run"], a["killed"]) for a in actions] == [(101, True, False)]


def test_reclaim_with_nothing_to_do_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    install_table(monkeypatch, [engine_proc(101, ppid=900), pytest_parent()])
    assert vh.reclaim_orphans(ENGINES) == []


def test_a_survivor_is_reported_as_not_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A kill that returned is not proof; the pid is re-checked afterwards."""
    install_table(monkeypatch, [engine_proc(101, ppid=900)])
    monkeypatch.setattr(vh, "kill_process_tree", lambda *a, **k: None)
    monkeypatch.setattr(vh, "process_is_alive", lambda *a, **k: True)
    (action,) = vh.reclaim_orphans(ENGINES)
    assert action["killed"] is False


# ---------------------------------------------------------------------------
# PDH per-process VRAM
# ---------------------------------------------------------------------------


class FakePdh:
    """The four win32pdh calls this module makes."""

    PDH_FMT_LARGE = 0x00000400

    def __init__(self, data: Any, *, fail: bool = False) -> None:
        self.data = data
        self.fail = fail
        self.collects = 0
        self.closed = False

    def OpenQuery(self) -> str:  # noqa: N802 - win32 spelling
        if self.fail:
            raise OSError("PDH unavailable")
        return "query"

    def MakeCounterPath(self, spec: tuple[Any, ...]) -> str:  # noqa: N802
        return f"\\{spec[1]}({spec[2]})\\{spec[5]}"

    def AddCounter(self, _query: str, _path: str) -> str:  # noqa: N802
        return "counter"

    def CollectQueryData(self, _query: str) -> None:  # noqa: N802
        self.collects += 1

    def GetFormattedCounterArray(self, _counter: str, _fmt: int) -> Any:  # noqa: N802
        return self.data

    def CloseQuery(self, _query: str) -> None:  # noqa: N802
        self.closed = True


@pytest.fixture(autouse=True)
def _clean_pdh_cache() -> Any:
    vh.reset_pdh_cache()
    yield
    vh.reset_pdh_cache()


@pytest.mark.parametrize(
    ("instance", "expected"),
    [
        ("pid_19544_luid_0x00000000_0x0000E5F5_phys_0", 19544),
        ("pid_4_luid_0x00000000_0x00013C35_phys_1", 4),
        ("pid_2468_luid_0x0_0x1_phys_0#1", 2468),
        ("engtype_3D", None),
        ("", None),
        ("pid_notanumber_luid_x", None),
    ],
)
def test_instance_names_yield_pids(instance: str, expected: int | None) -> None:
    assert vh._pid_from_instance(instance) == expected


def test_dedicated_bytes_are_summed_per_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    """One process can hold memory on several adapters; the rows are one pid."""
    monkeypatch.setattr(os, "name", "nt")
    fake = FakePdh(
        {
            "pid_101_luid_0x0_0xA_phys_0": 8 * GIB,
            "pid_101_luid_0x0_0xB_phys_0": 2 * GIB,
            "pid_202_luid_0x0_0xA_phys_0": 300 * MIB,
            "engtype_3D": 5,
        }
    )
    monkeypatch.setattr(vh, "_load_win32pdh", lambda: fake)
    assert vh.pdh_process_dedicated_bytes() == {101: 10 * GIB, 202: 300 * MIB}
    assert fake.closed is True


def test_old_binding_shapes_are_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "name", "nt")
    fake = FakePdh((1, [("pid_101_luid_0x0_0xA_phys_0", 5 * GIB)]))
    monkeypatch.setattr(vh, "_load_win32pdh", lambda: fake)
    assert vh.pdh_process_dedicated_bytes() == {101: 5 * GIB}


def test_an_empty_first_sample_is_retried_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """PDH wildcard counters can need a second collect to have a valid sample."""
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(vh.time, "sleep", lambda _s: None)
    fake = FakePdh({})

    def data(counter: str, fmt: int) -> Any:  # noqa: ARG001
        return {} if fake.collects < 2 else {"pid_7_luid_0x0_0xA_phys_0": 1 * GIB}

    fake.GetFormattedCounterArray = data  # type: ignore[method-assign]
    monkeypatch.setattr(vh, "_load_win32pdh", lambda: fake)
    assert vh.pdh_process_dedicated_bytes() == {7: 1 * GIB}
    assert fake.collects == 2


def test_pdh_failure_is_silent_and_latched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Diagnostics must never raise, and must never warn on every poll."""
    monkeypatch.setattr(os, "name", "nt")
    fake = FakePdh({}, fail=True)
    monkeypatch.setattr(vh, "_load_win32pdh", lambda: fake)
    warnings: list[str] = []
    monkeypatch.setattr(vh.log, "warning", lambda event, **_kw: warnings.append(event))
    assert vh.pdh_process_dedicated_bytes() == {}
    assert vh.pdh_process_dedicated_bytes() == {}
    assert warnings == ["pdh_unavailable"]


def test_samples_are_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "name", "nt")
    fake = FakePdh({"pid_101_luid_0x0_0xA_phys_0": 1 * GIB})
    monkeypatch.setattr(vh, "_load_win32pdh", lambda: fake)
    vh.pdh_process_dedicated_bytes()
    vh.pdh_process_dedicated_bytes()
    assert fake.collects == 1


def test_non_windows_returns_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(
        vh, "_load_win32pdh", lambda: pytest.fail("PDH must not be touched off Windows")
    )
    assert vh.pdh_process_dedicated_bytes() == {}


# ---------------------------------------------------------------------------
# gpu.vram_processes enrichment
# ---------------------------------------------------------------------------


def test_zero_byte_nvml_entries_are_sized_from_pdh(monkeypatch: pytest.MonkeyPatch) -> None:
    """The incident's headline symptom: holders listed, all reporting 0 bytes."""
    probe = FakeGpuProbe(
        [GpuInfo(index=0, name="RTX 5090", total_bytes=32 * GIB, free_bytes=32 * GIB)]
    )
    probe.set_processes(
        [
            VramProcess(gpu_index=0, pid=101, name="llama-server.exe", used_bytes=0),
            VramProcess(gpu_index=0, pid=202, name="dwm.exe", used_bytes=0),
        ]
    )
    monkeypatch.setattr(vh, "pdh_process_dedicated_bytes", lambda **_kw: {101: 15 * GIB})
    found = {entry.pid: entry for entry in gpu_mod.vram_processes(probe)}
    assert found[101].used_bytes == 15 * GIB
    # No PDH figure for a pid means "unknown", which stays 0 rather than a guess.
    assert found[202].used_bytes == 0


def test_real_nvml_numbers_are_never_overwritten(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Linux NVML is authoritative and per-GPU, which is strictly better."""
    probe = FakeGpuProbe(
        [GpuInfo(index=0, name="RTX 3090", total_bytes=24 * GIB, free_bytes=24 * GIB)]
    )
    probe.set_processes(
        [VramProcess(gpu_index=0, pid=101, name="llama-server", used_bytes=7 * GIB)]
    )
    monkeypatch.setattr(
        vh, "pdh_process_dedicated_bytes", lambda **_kw: pytest.fail("PDH not needed")
    )
    (entry,) = gpu_mod.vram_processes(probe)
    assert entry.used_bytes == 7 * GIB


# ---------------------------------------------------------------------------
# The merged view
# ---------------------------------------------------------------------------


def _incident_probe() -> FakeGpuProbe:
    probe = FakeGpuProbe(
        [
            GpuInfo(index=0, name="RTX 5090", total_bytes=32 * GIB, free_bytes=32 * GIB),
            GpuInfo(index=1, name="RTX 5090", total_bytes=32 * GIB, free_bytes=32 * GIB),
        ]
    )
    probe.set_processes(
        [
            VramProcess(gpu_index=0, pid=101, name="llama-server.exe"),
            VramProcess(gpu_index=1, pid=102, name="llama-server.exe"),
            VramProcess(gpu_index=0, pid=301, name="dwm.exe"),
            VramProcess(gpu_index=0, pid=302, name="explorer.exe"),
            VramProcess(gpu_index=0, pid=303, name="msedgewebview2.exe"),
        ]
    )
    return probe


def test_holders_view_names_the_parent_of_a_foreign_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The answer the incident needed and could not get from any endpoint."""
    install_table(
        monkeypatch,
        [
            engine_proc(101, alias="qwen2.5-vl-7b", port=18101, ppid=900),
            engine_proc(102, alias="qwen3-embed", port=18102, ppid=900),
            pytest_parent(),
            FakeProcess(301, "dwm.exe", create_time=10.0),
            FakeProcess(302, "explorer.exe", create_time=10.0),
            FakeProcess(303, "msedgewebview2.exe", create_time=10.0),
        ],
    )
    monkeypatch.setattr(
        vh, "pdh_process_dedicated_bytes", lambda **_kw: {101: 10 * GIB, 102: 15 * GIB}
    )
    view = vh.holders_view(_incident_probe(), ENGINES)

    holders = {row["pid"]: row for row in view["holders"]}
    assert set(holders) == {101, 102}, "desktop noise must be collapsed, not listed"
    assert holders[102]["used_bytes"] == 15 * GIB
    assert holders[102]["used_bytes_source"] == "pdh"
    assert holders[101]["classification"] == vh.CLASS_CHILD_OF_LIVE
    assert holders[101]["parent_name"] == "python.exe"
    assert "pytest" in holders[101]["parent_cmdline"]
    assert holders[101]["alias"] == "qwen2.5-vl-7b"
    assert view["desktop_processes_count"] == 3
    assert view["orphan_count"] == 0


def test_a_big_foreign_holder_is_kept_and_a_small_one_collapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = FakeGpuProbe(
        [GpuInfo(index=0, name="RTX 5090", total_bytes=32 * GIB, free_bytes=32 * GIB)]
    )
    probe.set_processes(
        [
            VramProcess(gpu_index=0, pid=401, name="ComfyUI.exe", used_bytes=12 * GIB),
            VramProcess(gpu_index=0, pid=402, name="dwm.exe", used_bytes=100 * MIB),
        ]
    )
    install_table(
        monkeypatch,
        [
            FakeProcess(401, "ComfyUI.exe", create_time=10.0, ppid=1),
            FakeProcess(402, "dwm.exe", create_time=10.0, ppid=1),
            FakeProcess(1, "services.exe", create_time=1.0),
        ],
    )
    monkeypatch.setattr(vh, "pdh_process_dedicated_bytes", lambda **_kw: {})
    view = vh.holders_view(probe, ENGINES)
    assert [row["pid"] for row in view["holders"]] == [401]
    assert view["holders"][0]["classification"] == vh.CLASS_FOREIGN
    assert view["desktop_processes_count"] == 1


def test_a_split_model_is_one_row_not_two_added_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PDH figure is a per-process total; per-GPU rows would double count.

    NVML lists a two-GPU model once per device. If the view summed those rows
    after each had been back-filled with the same per-process total, a 15 GiB
    model would be reported as 30 GiB -- and the planner's own free-VRAM
    numbers would visibly contradict the holder list.
    """
    probe = FakeGpuProbe(
        [
            GpuInfo(index=0, name="RTX 5090", total_bytes=32 * GIB, free_bytes=32 * GIB),
            GpuInfo(index=1, name="RTX 5090", total_bytes=32 * GIB, free_bytes=32 * GIB),
        ]
    )
    probe.set_processes(
        [
            VramProcess(gpu_index=0, pid=101, name="llama-server.exe"),
            VramProcess(gpu_index=1, pid=101, name="llama-server.exe"),
        ]
    )
    install_table(monkeypatch, [engine_proc(101, ppid=900), pytest_parent()])
    monkeypatch.setattr(vh, "pdh_process_dedicated_bytes", lambda **_kw: {101: 15 * GIB})
    view = vh.holders_view(probe, ENGINES)
    (row,) = view["holders"]
    assert row["used_bytes"] == 15 * GIB
    assert row["gpu_indices"] == [0, 1]


def test_an_engine_process_nvml_cannot_see_is_still_listed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A holder nothing can see is the failure this module exists to prevent."""
    probe = FakeGpuProbe(
        [GpuInfo(index=0, name="RTX 5090", total_bytes=32 * GIB, free_bytes=32 * GIB)]
    )
    install_table(monkeypatch, [engine_proc(101, ppid=900)])
    monkeypatch.setattr(vh, "pdh_process_dedicated_bytes", lambda **_kw: {})
    view = vh.holders_view(probe, ENGINES)
    assert [row["pid"] for row in view["holders"]] == [101]
    assert view["holders"][0]["classification"] == vh.CLASS_ORPHAN
    assert view["orphan_count"] == 1
    assert view["per_process_bytes"] == "unavailable"


# ---------------------------------------------------------------------------
# /api/status enrichment
# ---------------------------------------------------------------------------


def _incident_status_payload() -> dict[str, Any]:
    """The payload exactly as /api/status produced it during the incident."""
    return {
        "vram_processes": [
            {
                "gpu_index": 0,
                "pid": 101,
                "name": "llama-server.exe",
                "used_bytes": 0,
                "is_ours": False,
            },
            {"gpu_index": 0, "pid": 301, "name": "dwm.exe", "used_bytes": 0, "is_ours": False},
            {"gpu_index": 0, "pid": 302, "name": "explorer.exe", "used_bytes": 0, "is_ours": False},
        ]
    }


def test_status_entries_keep_their_keys_and_gain_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_table(
        monkeypatch,
        [
            engine_proc(101, alias="qwen", ppid=900),
            pytest_parent(),
            FakeProcess(301, "dwm.exe", create_time=10.0),
            FakeProcess(302, "explorer.exe", create_time=10.0),
        ],
    )
    monkeypatch.setattr(vh, "pdh_process_dedicated_bytes", lambda **_kw: {101: 10 * GIB})
    payload = _incident_status_payload()
    vh.annotate_status_payload(payload, engines_dir=ENGINES)

    (entry,) = payload["vram_processes"]
    # Existing keys are untouched: the planner's rejection messages read them.
    assert entry["gpu_index"] == 0
    assert entry["pid"] == 101
    assert entry["name"] == "llama-server.exe"
    assert entry["is_ours"] is False
    # And the four that make it answerable.
    assert entry["used_bytes"] == 10 * GIB
    assert entry["classification"] == vh.CLASS_CHILD_OF_LIVE
    assert entry["parent_pid"] == 900
    assert entry["parent_name"] == "python.exe"
    assert "pytest" in entry["parent_cmdline"]
    assert payload["desktop_processes_count"] == 2
    assert payload["vram_orphan_count"] == 0


def test_status_reports_orphans_it_finds(monkeypatch: pytest.MonkeyPatch) -> None:
    install_table(monkeypatch, [engine_proc(101, ppid=900)])
    monkeypatch.setattr(vh, "pdh_process_dedicated_bytes", lambda **_kw: {})
    payload = _incident_status_payload()
    vh.annotate_status_payload(payload, engines_dir=ENGINES)
    assert payload["vram_orphan_count"] == 1
    assert payload["vram_processes"][0]["classification"] == vh.CLASS_ORPHAN


def test_status_annotation_survives_a_missing_key() -> None:
    payload: dict[str, Any] = {"loaded": []}
    assert vh.annotate_status_payload(payload, engines_dir=ENGINES) is payload


# ---------------------------------------------------------------------------
# The HTTP surface
# ---------------------------------------------------------------------------


@pytest.fixture
def app(tmp_path: Path) -> Any:
    from studioforge.api.app import build_state, create_app
    from studioforge.config import Config

    config = Config(
        data_dir=tmp_path / "data",
        server={"host": "127.0.0.1", "port": 1234},
        models={"dir": tmp_path / "models"},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
    )
    return create_app(config, state=build_state(config), start_background=False)


def test_holders_endpoint_answers_who_has_my_vram(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    captured: dict[str, Any] = {}

    def fake_view(probe: Any, engines_dir: Any, *, own_pids: Any = ()) -> dict[str, Any]:
        captured["engines_dir"] = str(engines_dir)
        captured["own_pids"] = set(own_pids)
        return {"holders": [{"pid": 101, "classification": "orphan"}], "orphan_count": 1}

    monkeypatch.setattr(vh, "holders_view", fake_view)
    with TestClient(app) as client:
        payload = client.get("/api/vram/holders").json()
    assert payload["orphan_count"] == 1
    # The route must scope the sweep to OUR engines dir, not the whole box.
    assert captured["engines_dir"].endswith("engines")


def test_reclaim_endpoint_passes_dry_run_through(app: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    seen: dict[str, Any] = {}

    def fake_reclaim(engines_dir: Any, *, own_pids: Any = (), dry_run: bool = False) -> list[Any]:
        seen["dry_run"] = dry_run
        return [{"pid": 101, "alias": "leaked", "killed": not dry_run}]

    monkeypatch.setattr(vh, "reclaim_orphans", fake_reclaim)
    # A local caller: on an open install the reclaim route (it kills processes)
    # is a D32 admin mutation, refused from the LAN without the PIN.
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        preview = client.post("/api/vram/reclaim", json={"dry_run": True}).json()
        assert seen["dry_run"] is True
        assert preview == {
            "dry_run": True,
            "orphans_found": 1,
            "killed": 0,
            "actions": [{"pid": 101, "alias": "leaked", "killed": False}],
        }

        done = client.post("/api/vram/reclaim", json={}).json()
    assert seen["dry_run"] is False
    assert done["killed"] == 1


def test_status_payload_carries_the_attribution(app: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The endpoint an LLM reads must answer the question, not list noise."""
    from fastapi.testclient import TestClient

    def fake_annotate(payload: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        payload["vram_processes"] = [
            {"pid": 101, "name": "llama-server.exe", "classification": "orphan"}
        ]
        payload["desktop_processes_count"] = 17
        payload["vram_orphan_count"] = 1
        return payload

    monkeypatch.setattr(vh, "annotate_status_payload", fake_annotate)
    with TestClient(app) as client:
        payload = client.get("/api/status").json()
    assert payload["desktop_processes_count"] == 17
    assert payload["vram_orphan_count"] == 1
    assert payload["vram_processes"][0]["classification"] == "orphan"


def test_status_still_answers_when_attribution_fails(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A diagnostic that can 500 the status endpoint is worse than none."""
    from fastapi.testclient import TestClient

    def boom(*_a: Any, **_kw: Any) -> None:
        raise RuntimeError("process table exploded")

    monkeypatch.setattr(vh, "annotate_status_payload", boom)
    with TestClient(app) as client:
        response = client.get("/api/status")
    assert response.status_code == 200
    assert "gpus" in response.json()


# ---------------------------------------------------------------------------
# GUI formatting
# ---------------------------------------------------------------------------


def test_holder_origin_names_the_parent() -> None:
    holder = {
        "classification": "child-of-live-process",
        "parent_name": "python.exe",
        "parent_pid": 900,
    }
    assert st.vram_holder_origin(holder) == "child of python.exe (pid 900)"
    assert st.vram_holder_origin({"classification": "orphan"}) == "ORPHAN"
    assert st.vram_holder_origin({"classification": "ours"}) == "ours"
    assert st.vram_holder_origin({"classification": "foreign"}) == "foreign"


def test_holder_line_reads_as_one_row() -> None:
    line = st.vram_holder_line(
        {
            "name": "llama-server.exe",
            "pid": 19544,
            "used_bytes": 15 * GIB,
            "gpu_indices": [0, 1],
            "classification": "orphan",
            "alias": "qwen3-8b",
        }
    )
    assert "llama-server.exe (pid 19544)" in line
    assert "15.00 GiB" in line
    assert "CUDA0,1" in line
    assert "ORPHAN" in line
    assert "alias qwen3-8b" in line


def test_an_unknown_size_says_so_rather_than_zero() -> None:
    line = st.vram_holder_line({"name": "dwm.exe", "pid": 1, "used_bytes": 0})
    assert "size —" in line


def test_only_orphans_offer_reclaim() -> None:
    assert st.vram_holder_is_reclaimable({"classification": "orphan"}) is True
    assert st.vram_holder_is_reclaimable({"classification": "child-of-live-process"}) is False
    assert st.vram_holder_is_reclaimable({"classification": "ours"}) is False


def test_the_note_says_something_in_every_state() -> None:
    quiet = st.vram_holders_note(
        {"holders": [], "desktop_processes_count": 12, "desktop_processes_bytes": 2 * GIB}
    )
    assert "Nothing but desktop applications" in quiet
    assert "12 desktop process(es)" in quiet

    leaking = st.vram_holders_note(
        {"holders": [{"pid": 1}, {"pid": 2}], "orphan_count": 2, "desktop_processes_count": 0}
    )
    assert "2 orphaned" in leaking

    blind = st.vram_holders_note({"holders": [{"pid": 1}], "per_process_bytes": "unavailable"})
    assert "Per-process VRAM is unavailable" in blind
    assert st.vram_holders_note(None) == "VRAM holders unavailable."


# ---------------------------------------------------------------------------
# The guard on the guard
# ---------------------------------------------------------------------------


def test_pytest_config_deselects_the_contract_suite_by_default() -> None:
    """A bare ``pytest`` must not be able to reach the GPUs.

    This asserts the fix for the incident's fourth root cause: ``testpaths =
    ["tests"]`` meant ``pytest`` collected ``tests/contract``, which starts the
    real gateway against the real rig. Deleting ``addopts`` would restore that
    with no other test failing, so the config itself is asserted.
    """
    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    options = config["tool"]["pytest"]["ini_options"]
    assert "not contract" in options["addopts"]
    assert any(marker.startswith("contract:") for marker in options["markers"])


def test_the_contract_conftest_requires_an_explicit_opt_in() -> None:
    """Belt and braces: ``-m contract`` alone must not reach live hardware."""
    root = Path(__file__).resolve().parents[2]
    text = (root / "tests" / "contract" / "conftest.py").read_text(encoding="utf-8")
    assert "SF_RUN_CONTRACT" in text
    assert "pytest_collection_modifyitems" in text
    # And it must not point the gateway at the live data directory any more.
    assert "data_dir=DEV_DATA_DIR" not in text
