"""A port conflict must be a sentence, not a traceback.

Incident (production, the system this replaces): *"two LM Studio instances --
a stale one held :1234, and the 'reset' launched a second server without
killing the first."* StudioForge defaults to the same 1234 (DECISIONS.md D8),
on a box where LM Studio may still be installed, so this is the single most
likely startup failure here.

Untreated it surfaces from inside uvicorn as ``OSError: [WinError 10048]``,
which names no port, no process and no fix.
"""

from __future__ import annotations

import contextlib
import os
import socket
from collections.abc import Iterator
from pathlib import Path

import pytest
import typer

from studioforge.__main__ import _preflight_ports
from studioforge.config import Config
from studioforge.core import ports as ports_module
from studioforge.core.ports import (
    PortConflict,
    PortHolder,
    check_startup_ports,
    describe_conflicts,
    port_is_bindable,
)


@pytest.fixture(autouse=True)
def _no_sf_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("SF_"):
            monkeypatch.delenv(key, raising=False)


@contextlib.contextmanager
def occupied_port() -> Iterator[int]:
    """A real listening socket, so the check is against the OS, not a mock."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    try:
        yield int(sock.getsockname()[1])
    finally:
        sock.close()


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def make_config(tmp_path: Path, **ports: int) -> Config:
    return Config(
        data_dir=tmp_path / "data",
        server={"host": "127.0.0.1", "port": ports.get("server", free_port())},
        gui={"enabled": "gui" in ports, "host": "127.0.0.1", "port": ports.get("gui", 8080)},
        watchdog={
            "enabled": "watchdog" in ports,
            "host": "127.0.0.1",
            "port": ports.get("watchdog", 1235),
        },
        logging={"level": "ERROR"},
    )


def test_bindable_detects_a_held_port() -> None:
    with occupied_port() as port:
        assert port_is_bindable(port, "127.0.0.1") is False


def test_bindable_accepts_a_free_port() -> None:
    assert port_is_bindable(free_port(), "127.0.0.1") is True


def test_a_busy_gateway_port_is_reported_before_uvicorn_starts(tmp_path: Path) -> None:
    """The incident: something already owns 1234 and startup must say so."""
    with occupied_port() as port:
        config = make_config(tmp_path, server=port)
        conflicts = check_startup_ports(config)

    assert [c.role for c in conflicts] == ["server"]
    message = describe_conflicts(conflicts)
    assert str(port) in message
    assert "server.port" in message


def test_every_bound_port_is_checked(tmp_path: Path) -> None:
    with occupied_port() as gui_port, occupied_port() as watchdog_port:
        config = make_config(tmp_path, gui=gui_port, watchdog=watchdog_port)
        conflicts = check_startup_ports(config)

    assert sorted(c.role for c in conflicts) == ["gui", "watchdog"]


def test_disabled_services_are_not_checked(tmp_path: Path) -> None:
    """A disabled GUI cannot clash with anything; reporting it is noise."""
    with occupied_port() as port:
        config = make_config(tmp_path)
        config.gui.enabled = False
        config.gui.port = port
        assert check_startup_ports(config) == []


def test_free_ports_produce_no_conflicts(tmp_path: Path) -> None:
    assert check_startup_ports(make_config(tmp_path)) == []


def test_another_studioforge_is_called_out_by_name() -> None:
    conflict = PortConflict(
        role="server",
        port=1234,
        host="0.0.0.0",
        setting="server.port",
        holder=PortHolder(pid=4242, name="python.exe", is_studioforge=True),
    )

    message = conflict.message()

    assert "another StudioForge instance" in message
    assert "pid 4242" in message
    assert "server.port" in message


def test_lm_studio_is_called_out_by_name() -> None:
    """The actual culprit on this box, with the actual fix."""
    conflict = PortConflict(
        role="server",
        port=1234,
        host="0.0.0.0",
        setting="server.port",
        holder=PortHolder(pid=99, name="LM Studio.exe", is_lmstudio=True),
    )

    message = conflict.message()

    assert "LM Studio" in message
    assert "Quit LM Studio" in message


def test_an_unidentifiable_holder_still_gives_a_usable_message() -> None:
    """psutil often cannot name the owner without elevation: degrade, do not crash."""
    conflict = PortConflict(
        role="server", port=1234, host="0.0.0.0", setting="server.port", holder=PortHolder()
    )

    message = conflict.message()

    assert "could not identify" in message
    assert "server.port" in message


def test_holder_lookup_survives_access_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil

    def denied(kind: str = "inet") -> list[object]:
        raise psutil.AccessDenied()

    monkeypatch.setattr(ports_module.psutil, "net_connections", denied)

    holder = ports_module.find_port_holder(1234)

    assert holder.pid is None


def test_startup_exits_cleanly_instead_of_raising_oserror(tmp_path: Path) -> None:
    """``studioforge serve`` must exit with a message, never a WinError 10048."""
    with occupied_port() as port:
        config = make_config(tmp_path, server=port)
        with pytest.raises(typer.Exit) as excinfo:
            _preflight_ports(config)

    assert excinfo.value.exit_code == 3


def test_startup_proceeds_when_ports_are_free(tmp_path: Path) -> None:
    assert _preflight_ports(make_config(tmp_path)) is None


# ---------------------------------------------------------------------------
# A wildcard listener must be SEEN by the watchdog adoption probe (2026-08-19)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def wildcard_listener() -> Iterator[int]:
    """A listener on 0.0.0.0 -- exactly what the watchdog binds by default."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", 0))
    sock.listen(1)
    try:
        yield int(sock.getsockname()[1])
    finally:
        sock.close()


def test_port_has_listener_sees_a_wildcard_listener() -> None:
    """On Windows, binding 127.0.0.1:port SUCCEEDS beside a 0.0.0.0:port
    listener, so a bind-based 'is anything there' probe says free. The
    connect-based probe must say occupied."""
    with wildcard_listener() as port:
        assert ports_module.port_has_listener(port, "127.0.0.1") is True
    assert ports_module.port_has_listener(port, "127.0.0.1") is False


def test_inspect_running_watchdog_does_not_call_a_wildcard_listener_free(tmp_path: Path) -> None:
    """The first restart after the V2 switch: serve left the watchdog running
    on 0.0.0.0:1235, the replacement 'saw nothing listening', refused to adopt
    and died on the port conflict. The probe must get past the 'nothing is
    listening' gate and actually ask /health (here: a plain socket that is not
    an HTTP server, so the answer is 'did not answer', never 'nothing')."""
    from studioforge.core.ports import inspect_running_watchdog

    with wildcard_listener() as port:
        config = make_config(tmp_path, watchdog=port)
        presence = inspect_running_watchdog(config, timeout_s=1.0)
    assert "nothing is listening" not in presence.reason, presence.reason
    assert presence.adoptable is False  # a bare socket is not our watchdog


@pytest.mark.skipif(os.name == "nt", reason="Windows probes with SO_EXCLUSIVEADDRUSE instead")
def test_the_preflight_sees_past_a_previous_listeners_time_wait() -> None:
    """The preflight must model uvicorn, not a bare socket.

    asyncio's ``create_server`` sets ``SO_REUSEADDR`` on POSIX, so uvicorn
    binds happily over a port whose previous listener left connections in
    TIME_WAIT. A *plain* bind fails for as long as those last -- about a
    minute -- so the preflight refused to start, exited EXIT_PORT_CONFLICT,
    and named a conflict that did not exist. ``supervisor._port_is_bindable``
    already made this split for llama-server children; this is the same fix on
    the path that actually gates startup.
    """
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(1)
    client = socket.create_connection(("127.0.0.1", port))
    served, _ = listener.accept()
    # The server side closes first, so the SERVER's socket is what lingers.
    served.close()
    listener.close()
    client.close()

    plain = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        plain.bind(("127.0.0.1", port))
    except OSError:
        pass  # TIME_WAIT is present: the precondition for this test holds.
    else:
        plain.close()
        pytest.skip("no TIME_WAIT lingered; nothing to prove on this kernel")
    finally:
        plain.close()

    assert ports_module.port_is_bindable(port, "127.0.0.1") is True


@pytest.mark.skipif(os.name == "nt", reason="Windows probes with SO_EXCLUSIVEADDRUSE instead")
def test_a_live_listener_is_still_reported_busy() -> None:
    """SO_REUSEADDR must not turn the probe into a rubber stamp: an actual
    listener still has to read as occupied, or the preflight stops catching
    the double-start it exists for."""
    with wildcard_listener() as port:
        assert ports_module.port_is_bindable(port, "0.0.0.0") is False


def test_the_two_probes_agree_on_this_platform() -> None:
    """Two probes with opposite policies is one wrong probe. They predict
    different processes (uvicorn vs llama-server) but both processes set
    SO_REUSEADDR, so the answers must match."""
    from studioforge.core.supervisor import _port_is_bindable as child_probe

    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = int(probe.getsockname()[1])
    assert ports_module.port_is_bindable(free_port, "127.0.0.1") == child_probe(
        free_port, "127.0.0.1"
    )

    with wildcard_listener() as busy:
        assert ports_module.port_is_bindable(busy, "127.0.0.1") == child_probe(busy, "127.0.0.1")
