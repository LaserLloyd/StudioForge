"""The engine updater's stable channel (D62).

Upstream llama.cpp publishes two kinds of release. A ``bNNNN`` build release
for nearly every merge, every one flagged prerelease (D49-1 already exempts
the tag scheme from that flag), and every few weeks a ``vX.Y.Z`` *version*
release -- not prerelease -- whose only asset is ``nightly-tag.txt``, a
one-line pointer to the build it blesses. Read on 2026-09-09:
``v0.4.0/nightly-tag.txt`` -> ``b10809``, ``v0.3.0/nightly-tag.txt`` ->
``b10621``. So "the stable engine" is the build named by the newest version
release's pointer, and ``engine.update_channel`` decides whether the update
check recommends that build (``stable``, the default) or the newest
installable one (``latest``, the pre-D62 behaviour). These tests pin:

* :meth:`EngineManager.stable_release`: the pointer is followed, a 404 or a
  malformed pointer is ``None`` with a reason and never a raise, and the
  answer is cached on the manager with a shorter retry after a failure;
* :meth:`EngineManager.check_update`: ``stable`` beside ``latest``, the
  channel's ``recommended_tag`` / ``update_recommended``, every pre-D62 key
  untouched, and an unreadable stable channel recommending nothing rather
  than ``latest`` in disguise;
* ``GET /api/engine``: the channel and the stable build beside the inventory,
  through the cache, and ``null`` from a stub manager;
* the config key is validated and ships as ``stable``;
* the one renderer (:func:`describe_stable_channel`) and the panel line, the
  Install target and the CLI follow the channel.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from studioforge import __main__ as main_cli
from studioforge.config import Config
from studioforge.core import engine as engine_module
from studioforge.core.engine import (
    SKIP_TAG_SCHEME,
    STABLE_RELEASE_RETRY_S,
    STABLE_RELEASE_TTL_S,
    EngineAsset,
    EngineManager,
    describe_stable_channel,
)
from studioforge.gui import state as st
from studioforge.gui.tabs import setup as setup_tab
from tests.unit.test_engine import (
    MIXED_GPUS,
    StubProbe,
    _engine_state,
    _fake_engine,
    _FakeRequest,
    _release,
    _stub_capture,
    _win_cuda_assets,
)

ACTIVE = "b10689"
POINTER_PATH = "/ggml-org/llama.cpp/releases/download/v0.4.0/nightly-tag.txt"


def _version_release(
    tag: str,
    *,
    assets: tuple[str, ...] = ("nightly-tag.txt",),
    published_at: str = "2026-09-09T12:00:00Z",
) -> dict[str, Any]:
    entry = _release(tag, assets=list(assets))
    entry["published_at"] = published_at
    return entry


V040 = _version_release("v0.4.0")

#: One page as GitHub serves it: build releases (all prerelease) interleaved
#: with the version releases, newest first.
RELEASES: list[dict[str, Any]] = [
    _release("b10812", prerelease=True, assets=_win_cuda_assets("b10812")),
    _release("b10811", prerelease=True, assets=_win_cuda_assets("b10811")),
    V040,
    _release("b10809", prerelease=True, assets=_win_cuda_assets("b10809")),
    _version_release("v0.3.0", published_at="2026-08-20T00:00:00Z"),
    _release("b10621", prerelease=True, assets=_win_cuda_assets("b10621")),
]


def _github(
    releases: list[dict[str, Any]],
    *,
    latest: dict[str, Any] | None,
    pointer: str = "b10809\n",
    pointer_status: int = 200,
    seen: list[str] | None = None,
) -> httpx.AsyncClient:
    """The four GitHub surfaces the manager reads, served from fixtures.

    ``seen`` collects request paths so a test can count the reads a cache
    saved. The pointer is served from ``github.com`` (the asset download host),
    not the API host, as it is upstream.
    """
    by_tag = {str(entry["tag_name"]): entry for entry in releases}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if seen is not None:
            seen.append(path)
        if request.url.host == "github.com" and "/releases/download/" in path:
            if pointer_status != 200:
                return httpx.Response(pointer_status, text="Not Found")
            return httpx.Response(200, text=pointer)
        if path.endswith("/releases/latest"):
            if latest is None:
                return httpx.Response(404, json={"message": "Not Found"})
            return httpx.Response(200, json=latest)
        if path.endswith("/releases"):
            return httpx.Response(200, json=releases)
        entry = by_tag.get(path.rsplit("/", 1)[-1])
        if entry is None:
            return httpx.Response(404, json={"message": "Not Found"})
        return httpx.Response(200, json=entry)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _manager(
    config: Config,
    *,
    active: str | None = ACTIVE,
    channel: str | None = None,
    cuda: tuple[int, int] | None = (13, 3),
    releases: list[dict[str, Any]] | None = None,
    latest: dict[str, Any] | None = V040,
    client: httpx.AsyncClient | None = None,
    **github: Any,
) -> EngineManager:
    if channel is not None:
        config.engine.update_channel = channel  # type: ignore[assignment]
    mgr = EngineManager(
        config,
        probe=StubProbe(MIXED_GPUS, cuda),
        client=client or _github(releases or RELEASES, latest=latest, **github),
    )
    mgr.os_token, mgr.arch_token = "win", "x64"
    if active:
        mgr.set_active(active)
    return mgr


@pytest.fixture
def tmp_config(tmp_path: Path) -> Config:
    config = Config(data_dir=tmp_path / "data")
    config.ensure_dirs()
    return config


class _Recorder:
    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict[str, Any]]] = []

    def warning(self, event: str, **fields: Any) -> None:
        self.warnings.append((event, fields))

    def __getattr__(self, _name: str) -> Callable[..., None]:
        return lambda *_a, **_k: None


# ---------------------------------------------------------------------------
# stable_release
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stable_release_follows_the_version_releases_pointer(tmp_config: Config) -> None:
    """``releases/latest`` is the newest non-prerelease release -- the version
    release, because every build release is flagged prerelease -- and its
    ``nightly-tag.txt`` names the blessed build."""
    mgr = _manager(tmp_config)
    assert await mgr.stable_release() == {
        "version": "v0.4.0",
        "tag": "b10809",
        "published_at": "2026-09-09T12:00:00Z",
    }
    assert mgr.last_stable_error is None
    # The version releases are still not engines: the build list ignores them
    # by tag scheme (D49), whatever their prerelease flag says.
    assert await mgr.list_releases(limit=10) == ["b10812", "b10811", "b10809", "b10621"]
    assert mgr.last_release_scan is not None
    assert mgr.last_release_scan["reasons"] == {SKIP_TAG_SCHEME: 2}


@pytest.mark.asyncio
async def test_a_missing_pointer_is_none_with_a_reason_not_a_raise(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(engine_module, "log", recorder)
    mgr = _manager(tmp_config, pointer_status=404)
    assert await mgr.stable_release() is None
    assert mgr.last_stable_error is not None
    assert "nightly-tag.txt" in mgr.last_stable_error and "404" in mgr.last_stable_error
    assert [e for e, _ in recorder.warnings] == ["engine.stable.unavailable"]
    # The same reason again after the retry window is one fact, not a second
    # warning: a poller must not turn an outage into sixty warnings an hour.
    mgr._stable_cache = (engine_module.time.monotonic() - 1.0, None)  # noqa: SLF001
    assert await mgr.stable_release() is None
    assert [e for e, _ in recorder.warnings] == ["engine.stable.unavailable"]


@pytest.mark.asyncio
async def test_a_network_failure_never_reaches_a_status_call(tmp_config: Config) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to GitHub")

    mgr = _manager(tmp_config, client=httpx.AsyncClient(transport=httpx.MockTransport(boom)))
    assert await mgr.stable_release() is None
    assert "no route to GitHub" in (mgr.last_stable_error or "")


@pytest.mark.asyncio
async def test_a_build_tag_at_releases_latest_is_not_a_stable_channel(tmp_config: Config) -> None:
    """If upstream ever stops flagging builds prerelease, ``releases/latest``
    names a build. That is a failure with a reason -- the channel must not
    quietly turn into ``latest``, which is the one thing the operator opted out of."""
    build = _release("b10812", assets=_win_cuda_assets("b10812"))
    mgr = _manager(tmp_config, latest=build)
    assert await mgr.stable_release() is None
    assert "b10812" in (mgr.last_stable_error or "") and "vX.Y.Z" in (mgr.last_stable_error or "")


@pytest.mark.asyncio
async def test_a_pointer_that_is_not_a_build_tag_is_refused(tmp_config: Config) -> None:
    mgr = _manager(tmp_config, pointer="v0.4.0\n")
    assert await mgr.stable_release() is None
    assert "not a bNNNN build tag" in (mgr.last_stable_error or "")


@pytest.mark.asyncio
async def test_a_version_release_without_the_pointer_asset_is_refused_before_any_download(
    tmp_config: Config,
) -> None:
    seen: list[str] = []
    mgr = _manager(tmp_config, latest=_version_release("v0.4.0", assets=()), seen=seen)
    assert await mgr.stable_release() is None
    assert "carries no nightly-tag.txt" in (mgr.last_stable_error or "")
    assert not [path for path in seen if "/releases/download/" in path]


@pytest.mark.asyncio
async def test_the_answer_is_cached_and_a_failure_is_retried_sooner(tmp_config: Config) -> None:
    """Two GitHub reads per answer, then none until the TTL runs out; a failure
    stands for a minute, not a quarter of an hour."""
    seen: list[str] = []
    mgr = _manager(tmp_config, seen=seen)

    def reads() -> tuple[int, int]:
        return (
            sum(p.endswith("/releases/latest") for p in seen),
            sum(p == POINTER_PATH for p in seen),
        )

    assert (await mgr.stable_release()) is not None
    assert (await mgr.stable_release()) is not None
    assert reads() == (1, 1)
    assert mgr._stable_cache is not None  # noqa: SLF001 - the TTL under test
    expires_at, _cached = mgr._stable_cache  # noqa: SLF001
    # +1 ms: Windows' monotonic clock ticks every ~15 ms, and (now + TTL) - now
    # can round a hair above TTL when it has not ticked.
    assert 0 < expires_at - engine_module.time.monotonic() <= STABLE_RELEASE_TTL_S + 1e-3
    assert (await mgr.stable_release(refresh=True)) is not None
    assert reads() == (2, 2)
    # Expired: the next read goes back to GitHub.
    mgr._stable_cache = (engine_module.time.monotonic() - 1.0, _cached)  # noqa: SLF001
    assert (await mgr.stable_release()) is not None
    assert reads() == (3, 3)

    failing_seen: list[str] = []
    failing = _manager(tmp_config, pointer_status=404, seen=failing_seen)
    assert await failing.stable_release() is None
    assert await failing.stable_release() is None
    assert sum(p == POINTER_PATH for p in failing_seen) == 1
    assert failing._stable_cache is not None  # noqa: SLF001
    retry_at = failing._stable_cache[0]  # noqa: SLF001
    assert 0 < retry_at - engine_module.time.monotonic() <= STABLE_RELEASE_RETRY_S + 1e-3
    assert STABLE_RELEASE_RETRY_S < STABLE_RELEASE_TTL_S


# ---------------------------------------------------------------------------
# check_update: the channel's verdict beside the old one
# ---------------------------------------------------------------------------

PRE_D62_KEYS = {
    "checked",
    "current",
    "latest",
    "update_available",
    "recent",
    "latest_variant",
    "skipped",
    "filtered",
    "filter_summary",
}


@pytest.mark.asyncio
async def test_check_update_recommends_the_stable_build_by_default(tmp_config: Config) -> None:
    mgr = _manager(tmp_config)
    status = await mgr.check_update(limit=5)

    # Everything a pre-D62 caller reads is still there and still means the same.
    assert set(status) >= PRE_D62_KEYS
    assert status["current"] == ACTIVE
    assert (status["latest"], status["latest_variant"]) == ("b10812", "cuda-13.3")
    assert status["update_available"] is True
    assert status["skipped"] == []
    # And the channel beside it.
    assert status["update_channel"] == "stable"
    assert status["stable"] == {
        "version": "v0.4.0",
        "tag": "b10809",
        "published_at": "2026-09-09T12:00:00Z",
    }
    assert status["stable_variant"] == "cuda-13.3"
    assert status["stable_error"] is None
    assert (status["recommended_tag"], status["recommended_variant"]) == ("b10809", "cuda-13.3")
    assert status["update_recommended"] is True


@pytest.mark.asyncio
async def test_check_update_follows_the_newest_build_on_the_latest_channel(
    tmp_config: Config,
) -> None:
    mgr = _manager(tmp_config, channel="latest")
    status = await mgr.check_update(limit=5)
    assert status["update_channel"] == "latest"
    assert (status["recommended_tag"], status["recommended_variant"]) == ("b10812", "cuda-13.3")
    assert status["update_recommended"] is True
    # The stable build is still named, so the operator can see what they skipped.
    assert status["stable"]["tag"] == "b10809" and status["stable_variant"] == "cuda-13.3"


@pytest.mark.asyncio
async def test_a_stable_build_that_is_not_newer_is_reported_but_not_recommended(
    tmp_config: Config,
) -> None:
    """Never a downgrade, and never a no-op offered as an update (D49)."""
    on_stable = _manager(tmp_config, active="b10809")
    status = await on_stable.check_update(limit=5)
    assert status["recommended_tag"] == "b10809"
    assert status["update_recommended"] is False
    assert status["update_available"] is True  # b10812 exists; the latest channel would take it

    ahead = _manager(tmp_config, active="b10850")
    status = await ahead.check_update(limit=5)
    assert status["recommended_tag"] == "b10809" and status["update_recommended"] is False


@pytest.mark.asyncio
async def test_an_unreadable_stable_channel_recommends_nothing_rather_than_latest(
    tmp_config: Config,
) -> None:
    mgr = _manager(tmp_config, pointer_status=404)
    status = await mgr.check_update(limit=5)
    assert status["stable"] is None and status["stable_variant"] is None
    assert "nightly-tag.txt" in status["stable_error"]
    assert status["recommended_tag"] is None and status["recommended_variant"] is None
    assert status["update_recommended"] is False
    # The newest build is still reported for anyone who wants to switch channel.
    assert status["latest"] == "b10812" and status["update_available"] is True


@pytest.mark.asyncio
async def test_a_stable_build_with_no_asset_for_this_box_falls_back_the_way_latest_does(
    tmp_config: Config,
) -> None:
    """A 12.0 driver fits none of the CUDA archives: with source builds allowed
    the stable build is a source build (D2/D3), and with them off it is
    skipped with the driver named, exactly as ``latest`` is."""
    tmp_config.engine.allow_source_build = True
    mgr = _manager(tmp_config, cuda=(12, 0))
    status = await mgr.check_update(limit=5)
    assert status["latest_variant"] == "source"
    assert status["stable_variant"] == "source"
    assert (status["recommended_tag"], status["recommended_variant"]) == ("b10809", "source")
    assert status["update_recommended"] is True

    tmp_config.engine.allow_source_build = False
    mgr = _manager(tmp_config, cuda=(12, 0))
    status = await mgr.check_update(limit=5)
    assert status["latest"] is None and status["stable_variant"] is None
    assert status["recommended_tag"] is None and status["update_recommended"] is False
    stable_skip = [entry for entry in status["skipped"] if entry["tag"] == "b10809"]
    assert len(stable_skip) == 1 and "CUDA 12.0" in stable_skip[0]["reason"]


@pytest.mark.asyncio
async def test_a_stable_build_whose_release_is_missing_is_not_offered_as_a_source_build(
    tmp_config: Config,
) -> None:
    """A source build clones by tag; a pointer at a tag GitHub does not serve is
    a skip with the 404, not a compile that fails a minute later."""
    tmp_config.engine.allow_source_build = True
    without_b10809 = [entry for entry in RELEASES if entry["tag_name"] != "b10809"]
    mgr = _manager(tmp_config, releases=without_b10809)
    status = await mgr.check_update(limit=5)
    assert status["stable"]["tag"] == "b10809"
    assert status["stable_variant"] is None and status["recommended_tag"] is None
    assert any(e["tag"] == "b10809" and "does not exist" in e["reason"] for e in status["skipped"])


@pytest.mark.asyncio
async def test_the_stable_build_is_probed_once_and_not_at_all_when_it_is_the_newest(
    tmp_config: Config,
) -> None:
    """One GitHub call for the stable tag's assets, and none when the newest
    build already is the stable one."""
    fetched: list[str] = []

    def counting(mgr: EngineManager) -> None:
        original = mgr.list_assets

        async def spy(tag: str) -> list[EngineAsset]:
            fetched.append(tag)
            return await original(tag)

        mgr.list_assets = spy  # type: ignore[method-assign]

    mgr = _manager(tmp_config)
    counting(mgr)
    await mgr.check_update(limit=5)
    assert fetched == ["b10812", "b10809"]

    fetched.clear()
    newest_is_stable = [
        entry for entry in RELEASES if entry["tag_name"] not in ("b10812", "b10811")
    ]
    mgr = _manager(tmp_config, releases=newest_is_stable)
    counting(mgr)
    status = await mgr.check_update(limit=5)
    assert fetched == ["b10809"]
    assert status["latest"] == "b10809" == status["recommended_tag"]


# ---------------------------------------------------------------------------
# GET /api/engine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_status_reports_the_channel_and_the_stable_build(
    tmp_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.api import mgmt_routes

    mgr = _manager(tmp_config, active=None)
    _fake_engine(mgr.engines_dir, ACTIVE, 1_000)
    mgr.set_active(ACTIVE)
    _stub_capture(mgr, monkeypatch)
    request = _FakeRequest(_engine_state(tmp_config, mgr))

    payload = await mgmt_routes.engine_status(request)
    assert {"pinned_tag", "active", "installed", "drift", "install_progress"} <= set(payload)
    assert payload["update_channel"] == "stable"
    assert payload["stable"]["tag"] == "b10809" and payload["stable"]["version"] == "v0.4.0"
    assert payload["stable_error"] is None
    assert payload["stable_installed"] is False

    _fake_engine(mgr.engines_dir, "b10809", 2_000)
    assert (await mgmt_routes.engine_status(request))["stable_installed"] is True

    # A stub manager without the method degrades to null, not a 500 (the
    # Dashboard polls this route on a timer).
    bare = SimpleNamespace(active=lambda: None, installed=list)
    blank = await mgmt_routes.engine_status(_FakeRequest(_engine_state(tmp_config, bare)))
    assert blank["stable"] is None and blank["stable_error"] is None
    assert blank["stable_installed"] is False and blank["update_channel"] == "stable"


# ---------------------------------------------------------------------------
# The config key
# ---------------------------------------------------------------------------


def test_update_channel_ships_as_stable_and_accepts_only_the_two_channels(
    tmp_path: Path,
) -> None:
    assert Config(data_dir=tmp_path).engine.update_channel == "stable"
    assert Config(data_dir=tmp_path, engine={"update_channel": "latest"}).engine.update_channel == (
        "latest"
    )
    with pytest.raises(ValidationError):
        Config(data_dir=tmp_path, engine={"update_channel": "nightly"})


def test_the_setup_tab_promotes_the_channel_into_the_engine_card() -> None:
    """Promoted out of Advanced next to Check for update, with a one-liner."""
    assert "engine.update_channel" in setup_tab.COVERED_KEYS
    assert "vX.Y.Z" in st.CONFIG_FIELD_HELP["engine.update_channel"]


# ---------------------------------------------------------------------------
# One renderer, three surfaces
# ---------------------------------------------------------------------------


def test_describe_stable_channel_covers_every_shape() -> None:
    assert describe_stable_channel(None) == ""
    assert describe_stable_channel({"latest": "b1"}) == ""  # a pre-D62 payload
    assert describe_stable_channel({"stable": None}) == "unavailable"
    assert describe_stable_channel({"stable": None, "stable_error": "boom"}) == "unavailable: boom"
    stable = {"tag": "b10809", "version": "v0.4.0"}
    assert describe_stable_channel({"stable": stable}) == "b10809 (v0.4.0)"
    assert describe_stable_channel({"stable": stable, "stable_variant": "cuda-13.3"}) == (
        "b10809 (v0.4.0, cuda-13.3)"
    )
    assert describe_stable_channel({"stable": {"tag": "b10809"}}) == "b10809"


def _payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "checked": True,
        "current": ACTIVE,
        "latest": "b10812",
        "latest_variant": "cuda-13.3",
        "update_available": True,
        "update_channel": "stable",
        "stable": {"tag": "b10809", "version": "v0.4.0", "published_at": None},
        "stable_variant": "cuda-13.3",
        "stable_error": None,
        "recommended_tag": "b10809",
        "recommended_variant": "cuda-13.3",
        "update_recommended": True,
    }
    base.update(overrides)
    return base


def test_the_panel_line_and_the_install_target_follow_the_channel() -> None:
    recommended = _payload()
    assert st.engine_update_line(recommended) == (
        "Engine b10689 — stable b10809 (v0.4.0, cuda-13.3) is available. "
        "Newest build: b10812 (cuda-13.3)."
    )
    assert st.engine_update_available(recommended) is True
    assert st.engine_install_target(recommended) == "b10809"

    on_stable = _payload(current="b10809", update_recommended=False)
    assert st.engine_update_line(on_stable) == (
        "Engine b10809 is the stable channel's build (v0.4.0). Newest build: b10812 (cuda-13.3)"
        " — set engine.update_channel to 'latest' to be offered it."
    )
    assert st.engine_update_available(on_stable) is False

    ahead = _payload(
        current="b10850", latest="b10850", update_available=False, update_recommended=False
    )
    assert st.engine_update_line(ahead) == (
        "Engine b10850 is ahead of the stable channel, b10809 (v0.4.0, cuda-13.3)."
    )

    unreadable = _payload(
        stable=None,
        stable_variant=None,
        stable_error="boom",
        recommended_tag=None,
        recommended_variant=None,
        update_recommended=False,
    )
    assert st.engine_update_line(unreadable) == (
        "Engine b10689. The stable channel could not be read: boom. Newest build: b10812 "
        "(cuda-13.3) — set engine.update_channel to 'latest' to be offered it."
    )
    assert st.engine_update_available(unreadable) is False
    assert st.engine_install_target(unreadable) == ""

    no_asset = _payload(
        stable_variant=None,
        recommended_tag=None,
        recommended_variant=None,
        update_recommended=False,
    )
    assert st.engine_update_line(no_asset).startswith(
        "Engine b10689. The stable channel's build b10809 (v0.4.0) has no asset this box "
        "can install."
    )

    on_latest = _payload(
        update_channel="latest",
        recommended_tag="b10812",
        recommended_variant="cuda-13.3",
    )
    assert st.engine_update_line(on_latest) == (
        "Engine b10689 — b10812 (cuda-13.3) is available. "
        "Stable channel: b10809 (v0.4.0, cuda-13.3)."
    )
    assert st.engine_install_target(on_latest) == "b10812"


def test_a_payload_from_before_channels_renders_exactly_as_it_did() -> None:
    old = {"checked": True, "current": "b10425", "latest": "b10428", "update_available": True}
    assert st.engine_update_line(old) == "Engine b10425 — b10428 is available."
    assert st.engine_update_available(old) is True
    assert st.engine_install_target(old) == "b10428"


def test_engine_check_prints_the_stable_channel_and_moves_to_its_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Config(data_dir=tmp_path / "data")
    config.ensure_dirs()
    monkeypatch.setattr(main_cli, "_load", lambda _path: config)
    canned: dict[str, Any] = _payload(skipped=[], filter_summary="")

    async def check_update(self: EngineManager, *, limit: int = 5, probe_assets: int = 3) -> Any:
        return dict(canned)

    monkeypatch.setattr(EngineManager, "check_update", check_update)

    result = CliRunner().invoke(main_cli.app, ["engine", "--check"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "active: b10689    newest installable: b10812 (cuda-13.3)" in result.output
    assert "stable channel: b10809 (v0.4.0, cuda-13.3)" in result.output
    assert "engine.update_channel: stable -> recommended: b10809 (cuda-13.3)" in result.output
    assert "run 'studioforge engine --update' to move to b10809" in result.output

    canned.update(
        stable=None,
        stable_variant=None,
        stable_error="boom",
        recommended_tag=None,
        recommended_variant=None,
        update_recommended=False,
    )
    result = CliRunner().invoke(main_cli.app, ["engine", "--check"], catch_exceptions=False)
    assert result.exit_code == 1
    assert "stable channel: unavailable: boom" in result.output
    assert "no llama.cpp release on the stable channel offers a build" in result.output
