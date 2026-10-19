"""The four-tab panel: registry, redirects, the Advanced pages, the Models page and Test Chat.

Pure derivations (``gui/nav.py``, ``gui/models_view.py``, ``gui/status_view.py``) are tested
directly. The pages are tested through ``TestClient`` against the real NiceGUI build, the way
``test_panel_ia.py`` does it: a page that crashes becomes an error card with a 200, so the
assertions look at the text. Nothing here starts a process, binds a port or loads a model.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from nicegui import ui

from studioforge.config import Config
from studioforge.errors import StudioForgeError
from studioforge.gui import models_view as mv
from studioforge.gui import nav, tabs
from studioforge.gui import state as st
from studioforge.gui import status_view as sv
from studioforge.gui.app import TAB_NAMES, create_gui_app
from studioforge.types import GB, GpuInfo, InstanceInfo, LoadPlan
from tests.unit.test_gui import _FakeState, make_record

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config(data_dir=tmp_path / "data")
    cfg.models.dir = tmp_path / "models"
    cfg.models.dir.mkdir(parents=True, exist_ok=True)
    cfg.ensure_dirs()
    return cfg


def fetch(config: Config, state: Any, path: str = "/", *, follow: bool = True) -> Any:
    with TestClient(create_gui_app(config, api_state=state)) as client:
        return client.get(path, follow_redirects=follow)


def tab_labels(text: str) -> list[str]:
    """The labels of every ``q-tab`` in the rendered element tree, in order."""
    return re.findall(r'"tag":"q-tab","props":\{"label":"([^"]+)"', text)


# The names the 2.0.x panel accepted: eleven sections and the seven tabs of 2.0 (plus aliases).
OLD_SECTIONS = (
    "Overview",
    "Models",
    "Serving",
    "Engines",
    "Placement",
    "Integrations",
    "Users & Keys",
    "Updates",
    "Logs & Metrics",
    "Diagnostics",
    "Advanced",
)
OLD_TABS = ("dashboard", "setup", "models", "model", "download", "downloads", "chat")
OLD_TABS += ("server", "settings", "logs", "log")


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_the_registry_has_exactly_four_top_level_tabs_in_order() -> None:
    assert nav.TOP_TABS == ("Status", "Models", "Test Chat", "Advanced")
    assert TAB_NAMES == nav.TOP_TABS
    assert len(set(nav.TOP_TABS)) == 4


def test_the_rendered_top_line_is_those_four_and_nothing_else(config: Config) -> None:
    for path in ("/", "/?section=Models", "/?section=Test%20Chat"):
        assert tab_labels(fetch(config, _FakeState(config), path).text) == list(nav.TOP_TABS), path


def test_the_advanced_navigation_holds_everything_the_brief_names() -> None:
    labels = [page.label for page in nav.ADVANCED_PAGES]
    for wanted in (
        "Serving & network",
        "Accounts & keys",
        "Engines",
        "Watchdog & recovery",
        "Catalog & downloads",
        "Integrations",
        "Peers",
        "Logs & metrics",
        "Config",
        "Diagnostics",
        "About & updates",
    ):
        assert wanted in labels, wanted
    assert len({page.slug for page in nav.ADVANCED_PAGES}) == len(nav.ADVANCED_PAGES)


def test_every_advanced_page_has_a_builder_and_renders(config: Config) -> None:
    from studioforge.gui.tabs import advanced

    assert set(advanced.builder_slugs()) == set(nav.ADVANCED_SLUGS)
    for page in nav.ADVANCED_PAGES:
        text = fetch(config, _FakeState(config), f"/?section=Advanced&page={page.slug}").text
        assert "could not be rendered" not in text, page.slug
        assert page.label.replace("&", "&amp;") in text, page.slug


def test_every_previously_registered_panel_is_reachable_under_advanced_or_a_tab() -> None:
    """Each of the 2.0.x sections lands somewhere that exists."""
    for name in OLD_SECTIONS:
        location = nav.resolve_location({"section": name})
        assert location.tab in nav.TOP_TABS, name
        if location.tab == nav.ADVANCED:
            assert location.page in nav.ADVANCED_SLUGS, name
    reached = {
        nav.resolve_location({"section": name}).page
        for name in OLD_SECTIONS
        if nav.resolve_location({"section": name}).tab == nav.ADVANCED
    }
    # Serving, Engines, Placement, Integrations, Users & Keys, Updates, Logs, Diagnostics.
    assert {
        "serving",
        "engines",
        "placement",
        "integrations",
        "accounts",
        "about",
        "logs",
        "diagnostics",
    } <= reached


def test_every_settings_key_the_old_panel_rendered_is_still_rendered(config: Config) -> None:
    """The catch-all and the dedicated cards moved, they did not shrink."""
    from studioforge.gui.tabs import setup

    config_page = fetch(config, _FakeState(config), "/?section=Advanced&page=config").text
    covered_elsewhere = {
        "serving": ("server.port", "server.host", "gui.host", "watchdog.host"),
        "catalog": ("models.dir", "models.default_ctx", "hf.max_concurrent_downloads"),
        "placement": ("planner.headroom_fraction",),
        "watchdog": (
            "watchdog.enabled",
            "watchdog.port",
            "watchdog.poll_interval_s",
            "watchdog.health_timeout_s",
            "watchdog.wedged_after_failures",
        ),
    }
    for slug, keys in covered_elsewhere.items():
        text = fetch(config, _FakeState(config), f"/?section=Advanced&page={slug}").text
        for key in keys:
            assert key in text, (slug, key)
    assert "gateway.max_restarts" in config_page
    assert "engine.pinned_tag" in setup.COVERED_KEYS


# ---------------------------------------------------------------------------
# Old links redirect
# ---------------------------------------------------------------------------


def test_every_old_name_redirects_to_a_canonical_page_that_renders(config: Config) -> None:
    state = _FakeState(config)
    with TestClient(create_gui_app(config, api_state=state)) as client:
        names = [("section", name) for name in OLD_SECTIONS if name not in ("Models", "Advanced")]
        names += [("tab", name) for name in OLD_TABS if name not in ("models",)]
        names += [("section", name.lower()) for name in OLD_SECTIONS if name == "Overview"]
        for key, value in names:
            first = client.get("/", params={key: value}, follow_redirects=False)
            assert first.status_code == 307, (key, value)
            target = first.headers["location"]
            assert target == "/" or target.startswith("/?section="), (key, value, target)
            # canonical: asking again for the answer does not redirect a second time
            second = client.get(target, follow_redirects=False)
            assert second.status_code == 200, (key, value, target)
            assert "could not be rendered" not in second.text, (key, value)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ({"tab": "dashboard"}, "/"),
        ({"tab": "setup"}, "/"),
        ({"section": "Overview"}, "/"),
        ({"tab": "models"}, "/?section=Models"),
        ({"tab": "download"}, "/?section=Models&filter=downloads"),
        ({"tab": "chat"}, "/?section=Test%20Chat"),
        ({"tab": "server"}, "/?section=Advanced&page=integrations"),
        ({"tab": "settings"}, "/?section=Advanced&page=integrations"),
        ({"tab": "logs"}, "/?section=Advanced&page=logs"),
        ({"section": "Serving"}, "/?section=Advanced&page=serving"),
        ({"section": "Engines"}, "/?section=Advanced&page=engines"),
        ({"section": "Placement"}, "/?section=Advanced&page=placement"),
        ({"section": "Integrations"}, "/?section=Advanced&page=integrations"),
        ({"section": "Users & Keys"}, "/?section=Advanced&page=accounts"),
        ({"section": "users"}, "/?section=Advanced&page=accounts"),
        ({"section": "Updates"}, "/?section=Advanced&page=about"),
        ({"section": "Logs & Metrics"}, "/?section=Advanced&page=logs"),
        ({"section": "Diagnostics"}, "/?section=Advanced&page=diagnostics"),
    ],
)
def test_the_old_names_map_to_their_new_home(query: dict[str, str], expected: str) -> None:
    assert nav.needs_redirect(query) is True
    assert nav.canonical_url(nav.resolve_location(query)) == expected
    # and the answer is itself stable
    again = dict(
        pair.split("=", 1) for pair in expected.removeprefix("/?").split("&") if "=" in pair
    )
    assert nav.needs_redirect({k: v.replace("%20", " ") for k, v in again.items()}) is False


def test_the_protocol_handlers_links_survive_with_their_arguments(config: Config) -> None:
    from studioforge.core import protocol

    link = protocol.gui_url_for(
        protocol.parse_deep_link("lmstudio://open_from_hf?model=owner/repo&quant=Q4_K_M"), config
    )
    assert "tab=download" in link  # the backend still writes the old grammar
    query = dict(pair.split("=", 1) for pair in link.split("?", 1)[1].split("&"))
    response = fetch(
        config,
        _FakeState(config),
        "/?" + "&".join(f"{k}={v}" for k, v in query.items()),
        follow=False,
    )
    assert response.status_code == 307
    assert response.headers["location"].startswith(
        "/?section=Models&filter=downloads&repo=owner/repo"
    )


def test_a_model_link_keeps_the_model(config: Config) -> None:
    response = fetch(config, _FakeState(config), "/?tab=models&model=pub/some-model", follow=False)
    assert response.headers["location"] == "/?section=Models&model=pub/some-model"


def test_a_canonical_link_and_a_bare_slash_are_not_redirected(config: Config) -> None:
    state = _FakeState(config)
    for path in (
        "/",
        "/?section=Models",
        "/?section=Test%20Chat",
        "/?section=Advanced",
        "/?section=Advanced&page=logs",
        "/?section=Models&filter=loaded",
    ):
        assert fetch(config, state, path, follow=False).status_code == 200, path


def test_an_unknown_name_opens_status_not_an_error() -> None:
    assert nav.resolve_location({"tab": "nonsense"}).tab == nav.STATUS
    assert nav.resolve_location({"section": "nonsense", "page": "zzz"}).tab == nav.STATUS
    assert nav.resolve_location({"section": "Advanced", "page": "zzz"}).page == "serving"
    assert nav.resolve_location(None).tab == nav.STATUS


def test_the_old_name_table_covers_the_deep_link_grammar() -> None:
    assert {name.lower() for name in st.DEEP_LINK_TABS} <= set(nav.old_names()) | {"models"}
    for old in OLD_SECTIONS:
        assert old.lower() in nav.old_names() or old.lower() in {"models", "advanced"}


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def _gpu(index: int, used_gib: float = 1.0, total_gib: float = 32.0) -> GpuInfo:
    return GpuInfo(
        index=index,
        name="RTX 5090",
        total_bytes=int(total_gib * GB),
        used_bytes=int(used_gib * GB),
        free_bytes=int((total_gib - used_gib) * GB),
    )


def _instance(model_id: str, *, state: str = "ready", active: int = 0, devices=(0,)) -> Any:
    return InstanceInfo(
        model_id=model_id,
        state=state,  # type: ignore[arg-type]
        port=18100,
        plan=LoadPlan(
            model_id=model_id,
            devices=list(devices),
            ctx_size=8192,
            per_gpu_bytes=dict.fromkeys(devices, 10 * GB),
        ),
        active_requests=active,
        last_tokens_per_second=55.5,
    )


def test_health_chip_orders_boot_drain_failure_and_missing_gpu() -> None:
    assert sv.health_chip({"ready": False, "phase": "scanning models"}).label == "Starting"
    assert "scanning models" in sv.health_chip({"ready": False, "phase": "scanning models"}).detail
    assert sv.health_chip({"ready": True, "error": "no engine"}).label == "Boot failed"
    assert sv.health_chip({"ready": True}, draining=True).label == "Draining"
    failed = sv.health_chip({"ready": True}, instances=[_instance("m", state="failed")])
    assert failed.label == "Attention" and "m" in failed.detail
    assert sv.health_chip({"ready": True}, gpus_seen=False).label == "No GPU"
    assert sv.health_chip({"ready": True}, manager_present=False).label == "Degraded"
    assert sv.health_chip(None).label == "Healthy"


def test_gpu_occupants_name_models_other_programs_and_leases() -> None:
    gpu = _gpu(0, used_gib=14.0)
    lines = sv.gpu_occupant_lines(
        gpu,
        [_instance("alpha", devices=(0,)), _instance("beta", devices=(1,))],
        [SimpleNamespace(devices=[0], holder="comfyui")],
    )
    assert any(line.startswith("alpha") for line in lines)
    assert not any(line.startswith("beta") for line in lines)
    assert any("other programs" in line for line in lines)  # 14 used, 10 ours
    assert any("lease" in line and "comfyui" in line for line in lines)
    assert sv.gpu_occupant_lines(_gpu(3, used_gib=0.0), [], []) == ["nothing loaded"]


def test_busy_label_and_request_totals() -> None:
    assert sv.busy_label(_instance("m", active=2)) == ("Busy", "primary")
    assert sv.busy_label(_instance("m")) == ("Idle", "grey")
    assert sv.busy_label(_instance("m", state="loading")) == ("Loading", "warning")
    assert sv.total_active_requests([_instance("a", active=2), _instance("b", active=1)]) == 3


def test_status_renders_health_gpus_models_and_only_the_allowed_actions(config: Config) -> None:
    state = _FakeState(config)
    state.supervisor = SimpleNamespace(list=lambda: [_instance("alpha-8b", active=1)])
    state.manager = SimpleNamespace(draining=False, leases=SimpleNamespace(all=lambda: []))
    text = fetch(config, state).text
    assert "could not be rendered" not in text
    for needle in ("Healthy", "GPU0 · RTX 5090", "alpha-8b", "Busy", "55.5 tok/s", "Uptime"):
        assert needle in text, needle
    assert "Open in Models" in text and "Unload" in text and "Open logs" in text
    # No settings on Status: none of the controls that moved away.
    for gone in ("Unload all", "Restart engines", "Restart server", "Reclaim", "Pin"):
        assert gone not in text, gone


# ---------------------------------------------------------------------------
# Models: pure helpers
# ---------------------------------------------------------------------------


def test_friendly_name_drops_the_quant_and_the_gguf_noise() -> None:
    plain = make_record("Qwen3-32B-Q4_K_M", quant="Q4_K_M")
    plain = plain.model_copy(update={"name": "Qwen3-32B-Q4_K_M"})
    assert mv.friendly_name(plain) == "Qwen3-32B"
    shard = make_record("vendor/Llama-3.3-70B-Instruct-Q4_K_M-00001-of-00002.gguf")
    assert mv.friendly_name(shard.model_copy(update={"name": shard.id})) == (
        "Llama-3.3-70B-Instruct"
    )
    named = make_record("x").model_copy(update={"name": "Custom Name"})
    assert mv.friendly_name(named) == "Custom Name"
    assert mv.friendly_name(make_record("only_id")) == "only id"


def test_quant_colour_and_context_formatting() -> None:
    assert mv.quant_colour("Q4_K_M") == "primary"
    assert mv.quant_colour("Q8_0") == "accent"
    assert mv.quant_colour("F16") == "deep-orange"
    assert mv.quant_colour("IQ2_XS") == "blue-grey"
    assert mv.quant_colour("") == "grey"
    assert mv.fmt_ctx(131072) == "128k"
    assert mv.fmt_ctx(8192) == "8k"
    assert mv.fmt_ctx(1_048_576) == "1M"
    assert mv.fmt_ctx(None) == st.UNKNOWN


def test_fits_hint_words_the_catalogs_answer_without_recomputing_it() -> None:
    rec = {
        "label": "2x RTX 5090",
        "ctx_per_slot": 131072,
        "fits_now": True,
        "would_evict": [],
    }
    now = mv.fits_hint({"recommended": rec})
    assert (now.text, now.colour) == ("2x RTX 5090 · 128k", "positive")
    evict = mv.fits_hint({"recommended": {**rec, "fits_now": False, "would_evict": ["big"]}})
    assert evict.colour == "warning" and "big" in evict.tooltip
    outside = mv.fits_hint({"recommended": {**rec, "fits_now": False}})
    assert outside.colour == "warning" and "outside StudioForge" in outside.tooltip
    assert mv.fits_hint({"recommended": None}).text == "does not fit"
    assert (
        mv.fits_hint({"arch_supported": False, "arch_note": "no such arch"}).text == "unsupported"
    )
    assert mv.fits_hint(None).text == st.UNKNOWN


def test_row_state_covers_idle_loading_loaded_error_and_the_badges() -> None:
    record = make_record("m")
    idle = mv.row_state(record, None, pinned=False, leased=False)
    assert (idle.primary, idle.badges) == (mv.STATE_IDLE, ())
    both = mv.row_state(record, None, pinned=True, leased=True)
    assert [b[0] for b in both.badges] == ["pinned", "leased"]
    loading = mv.row_state(record, None, pinned=False, leased=False, pending_since=100.0, now=107.0)
    assert loading.primary == mv.STATE_LOADING and loading.progress_s == 7.0
    spawning = mv.row_state(record, _instance("m", state="loading"), pinned=False, leased=False)
    assert spawning.primary == mv.STATE_LOADING
    busy = mv.row_state(record, _instance("m", active=3), pinned=False, leased=False)
    assert busy.primary == mv.STATE_LOADED and "busy · 3 requests" in busy.detail
    assert "55.5 tok/s" in busy.detail
    error = mv.LoadError("insufficient_vram", "needs 40 GiB, 30 usable")
    failed = mv.row_state(record, None, pinned=False, leased=False, error=error)
    assert failed.primary == mv.STATE_ERROR and failed.error is error
    assert failed.detail == "insufficient_vram"
    # a running model has nothing to explain
    up = mv.row_state(record, _instance("m"), pinned=False, leased=False, error=error)
    assert up.primary == mv.STATE_LOADED and up.error is None
    refused = mv.row_state(record, None, pinned=False, leased=False, arch_refusal="old engine")
    assert refused.badges[0][0] == st.UNSUPPORTED_ARCH_BADGE


def test_leased_models_are_named_ones_and_residents_on_leased_cards() -> None:
    lease = SimpleNamespace(model_ids=["named"], devices=[2])
    ids = mv.leased_model_ids(
        [lease], [_instance("on-card", devices=(2,)), _instance("elsewhere", devices=(0,))]
    )
    assert ids == {"named", "on-card"}


def test_filter_rows_applies_the_loaded_chip_and_a_friendly_name_search() -> None:
    records = [
        make_record("Qwen3-32B-Q4_K_M"),
        make_record("Llama-3.1-8B-Instruct-Q8_0", quant="Q8_0", arch="llama"),
    ]
    ids = {r.id for r in mv.filter_rows(records, needle="llama 8b", chip="all", loaded_ids=set())}
    assert ids == {"Llama-3.1-8B-Instruct-Q8_0"}
    loaded = mv.filter_rows(
        records, needle="", chip=nav.FILTER_LOADED, loaded_ids={"Qwen3-32B-Q4_K_M"}
    )
    assert [r.id for r in loaded] == ["Qwen3-32B-Q4_K_M"]
    assert len(mv.filter_rows(records, needle=None, chip="all", loaded_ids=set())) == 2
    # the Downloads chip shows the queue, so it does not narrow the table
    assert len(mv.filter_rows(records, needle="", chip=nav.FILTER_DOWNLOADS, loaded_ids=set())) == 2


def test_fit_bars_show_every_card_and_what_this_load_adds() -> None:
    gpus = [_gpu(0, used_gib=2), _gpu(1, used_gib=4), _gpu(2, used_gib=30, total_gib=24)]
    accepted = {"fits": True, "per_gpu_bytes": {0: 20 * GB, 1: 20 * GB}, "devices": [0, 1]}
    bars = mv.fit_bars(accepted, gpus)
    assert [b.index for b in bars] == [0, 1, 2]
    assert [b.targeted for b in bars] == [True, True, False]
    assert bars[0].add_bytes == 20 * GB and not bars[0].over
    assert bars[2].used_bytes == 30 * GB  # not part of this load, still drawn
    refused = mv.fit_bars({"fits": False, "reason": "too big"}, gpus)
    assert all(b.add_bytes == 0 for b in refused)
    over = mv.fit_bars({"fits": True, "per_gpu_bytes": {1: 40 * GB}, "devices": [1]}, gpus)
    assert over[1].over is True and over[1].add_fraction <= 1.0
    # a reload replaces the model's own footprint rather than stacking on it
    credited = mv.fit_bars(accepted, gpus, credit={0: 2 * GB})
    assert credited[0].used_bytes == 0


@pytest.mark.parametrize(
    ("text", "repo"),
    [
        ("owner/repo", "owner/repo"),
        ("  bartowski/Qwen3-GGUF ", "bartowski/Qwen3-GGUF"),
        ("https://huggingface.co/owner/repo", "owner/repo"),
        ("https://huggingface.co/owner/repo/tree/main", "owner/repo"),
        ("https://huggingface.co/owner/repo/blob/main/x-Q4_K_M.gguf", "owner/repo"),
        ("https://hf.co/owner/repo", "owner/repo"),
        ("https://huggingface.co/models/owner/repo", "owner/repo"),
        ("lmstudio://open_from_hf?model=owner%2Frepo", "owner/repo"),
    ],
)
def test_parse_hf_reference_accepts_ids_pages_and_the_lmstudio_link(text: str, repo: str) -> None:
    assert mv.parse_hf_reference(text) == (repo, None)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "not-a-repo",
        "../../etc/passwd",
        "owner/../repo",
        "https://evil.example/owner/repo",
        "https://huggingface.co/datasets/owner/repo",
        "lmstudio://open_from_hf?model=../x/y",
        "a/b/c",
    ],
)
def test_parse_hf_reference_refuses_everything_else_with_a_sentence(text: str) -> None:
    repo, error = mv.parse_hf_reference(text)
    assert repo is None and error


# ---------------------------------------------------------------------------
# Models: the page and the drawer, rendered
# ---------------------------------------------------------------------------


class _Supervisor:
    def __init__(self, instances: list[Any]) -> None:
        self.instances = instances

    def list(self) -> list[Any]:
        return list(self.instances)

    def get(self, model_id: str) -> Any:
        return next((i for i in self.instances if i.model_id == model_id), None)


class _Manager:
    """Just enough manager for the Models page: a catalog, a dry-run planner, a load recorder."""

    def __init__(self) -> None:
        self.leases = SimpleNamespace(all=lambda: [])
        self.draining = False
        self.loads: list[tuple[str, dict[str, Any]]] = []
        self.fail_with: Exception | None = None

    def catalog(self, *, model: str | None = None, compact: bool = False, refresh: bool = False):
        entry = {
            "id": "alpha-8b",
            "recommended": {
                "label": "2x RTX 5090",
                "ctx_per_slot": 131072,
                "kv_cache_type": "f16",
                "fits_now": True,
                "would_evict": [],
                "load_args": {
                    "model_id": "alpha-8b",
                    "ctx_size": 131072,
                    "parallel": 2,
                    "kv_cache_type": "f16",
                    "devices": [0, 1],
                },
            },
        }
        other = {"id": "beta-70b", "recommended": None, "recommended_basis": "too large"}
        models = [entry, other]
        if model:
            models = [m for m in models if m["id"] == model]
        return {"models": models, "count": len(models)}

    def plan_preview(self, name: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "fits": True,
            "model_id": name,
            "devices": kwargs.get("devices") or [0, 1],
            "split_mode": "layer",
            "ctx_size": kwargs.get("ctx_size") or 8192,
            "parallel": 2,
            "kv_cache_type": "f16",
            "flash_attn": "on",
            "per_gpu_bytes": {0: 9 * GB, 1: 9 * GB},
            "estimate_mb": {"total": 18 * 1024},
            "evict_model_ids": [],
            "notes": [],
        }

    async def load(self, model_id: str, **kwargs: Any) -> Any:
        self.loads.append((model_id, kwargs))
        if self.fail_with is not None:
            raise self.fail_with
        return _instance(model_id)

    def serving_record(self, record: Any) -> Any:
        return record


def _models_state(config: Config, *, loaded: list[Any] | None = None) -> _FakeState:
    state = _FakeState(config)
    state.registry._records = [
        make_record("alpha-8b", quant="Q4_K_M", added_at=2_000.0, mtime=2_000.0),
        make_record("beta-70b", quant="Q8_0", arch="llama", size=70 * GB, added_at=1_000.0),
    ]
    state.supervisor = _Supervisor(loaded or [])
    state.probe = SimpleNamespace(
        backend="fake", list_gpus=lambda: [_gpu(0), _gpu(1), _gpu(2, total_gib=24.0)]
    )
    state.manager = _Manager()
    return state


def test_the_models_page_lists_the_library_with_chips_state_and_the_fits_on_column(
    config: Config,
) -> None:
    state = _models_state(config, loaded=[_instance("alpha-8b")])
    text = fetch(config, state, "/?section=Models").text
    assert "could not be rendered" not in text
    for needle in (
        "alpha-8b",
        "beta-70b",
        "Q4_K_M",
        "Search models",
        "All 2",
        "Loaded 1",
        "Downloads 0",
        "Add model",
        "Rescan",
        "loaded",  # the state chip
        "idle",
        "Fits on",
    ):
        assert needle in text, needle


def test_the_loaded_chip_filters_the_table(config: Config) -> None:
    state = _models_state(config, loaded=[_instance("alpha-8b")])
    text = fetch(config, state, "/?section=Models&filter=loaded").text
    assert "alpha-8b" in text and "beta-70b" not in text


def test_a_search_link_narrows_the_table(config: Config) -> None:
    text = fetch(config, _models_state(config), "/?section=Models&q=70b").text
    assert "beta-70b" in text and "alpha-8b" not in text


def test_the_downloads_chip_has_the_add_box_the_search_and_the_queue(config: Config) -> None:
    text = fetch(config, _models_state(config), "/?section=Models&filter=downloads").text
    assert "could not be rendered" not in text
    for needle in (
        "Add a model",
        "Look up",
        "Search Hugging Face",
        "MTP only",
        "Downloads not available",  # the fake state has no downloader
    ):
        assert needle in text, needle


def test_a_repo_link_opens_the_quant_picker_in_the_downloads_view(config: Config) -> None:
    text = fetch(config, _models_state(config), "/?tab=download&repo=owner/repo&quant=Q4_K_M").text
    assert "owner/repo — Q4_K_M" in text and "Choose which version to download" in text


def test_an_unknown_model_link_says_so_in_the_table(config: Config) -> None:
    text = fetch(config, _models_state(config), "/?section=Models&model=nope/nothing").text
    assert "nope/nothing" in text


_DRAWER_STATE: dict[str, Any] = {}
_DRAWER_REGISTERED = False


def _register_drawer_page() -> None:
    """A page that builds the drawer open on ``?model=``, so its content is in the HTML."""
    global _DRAWER_REGISTERED
    if _DRAWER_REGISTERED:
        return
    _DRAWER_REGISTERED = True

    @ui.page("/_drawer_smoke")
    def _page(model: str = "alpha-8b") -> None:
        from studioforge.gui.tabs import models_page

        drawer = models_page.ModelDrawer(_DRAWER_STATE["ctx"], lambda loc: None)
        drawer.open(model)


def _drawer_html(config: Config, state: Any, model: str = "alpha-8b") -> str:
    _register_drawer_page()
    _DRAWER_STATE["ctx"] = tabs.GuiContext(config=config, api_state=state)
    return fetch(config, state, f"/_drawer_smoke?model={model}").text


def test_the_drawer_prefills_the_planners_recommendation_and_draws_the_fit_before_loading(
    config: Config,
) -> None:
    text = _drawer_html(config, _models_state(config))
    assert "could not be rendered" not in text
    for needle in (
        "alpha-8b",
        "Recommended: 2x RTX 5090 · 128k context · f16 KV",
        "Context (tokens per conversation)",
        "131072",  # the pre-filled context
        "Leave all unchecked to let the planner choose",
        "Advanced options",
        "All settings…",
        "Fits across GPU0, GPU1",  # the verdict, from plan_preview
        "GPU0: ",  # a bar per GPU, with a text alternative (its aria-label)
        "percent used",
        "GPU2",
        "Manage",
        "Sampler defaults",
        "Virtual models &amp; presets",
        "Aliases",
        "Pin",
        "Delete…",
    ):
        assert needle in text, needle
    # nothing was loaded to find that out
    assert "Reload with these settings" not in text


def test_the_drawer_of_a_loaded_model_offers_reload_and_unload(config: Config) -> None:
    state = _models_state(config, loaded=[_instance("alpha-8b", devices=(0, 1))])
    text = _drawer_html(config, state)
    assert "Reload with these settings" in text and "Unload" in text
    assert "GPU0, GPU1" in text  # where it runs now


def test_the_drawer_says_why_a_model_with_no_recommendation_will_not_fit(config: Config) -> None:
    text = _drawer_html(config, _models_state(config), "beta-70b")
    assert "No recommendation" in text


def test_the_drawer_refuses_to_load_what_the_engine_cannot_run(config: Config) -> None:
    state = _models_state(config)
    state.manager.arch_verdicts = lambda records: {
        r.id: SimpleNamespace(
            fields=lambda with_message=False: {
                "arch_supported": False,
                "arch_note": "no k2",
                "arch_message": "this build does not include the k2 architecture",
            }
        )
        for r in records
    }
    text = _drawer_html(config, state)
    assert st.UNSUPPORTED_ARCH_BADGE in text
    assert "does not include the k2 architecture" in text


async def test_every_load_goes_through_manager_load_with_the_catalogs_arguments(
    config: Config,
) -> None:
    from studioforge.gui.tabs import models_page

    state = _models_state(config)
    ctx = tabs.GuiContext(config=config, api_state=state)
    args = {"ctx_size": 131072, "parallel": 2, "kv_cache_type": "f16", "devices": [0, 1]}
    instance = await models_page.run_load(ctx, "alpha-8b", args, force=False)
    assert instance.model_id == "alpha-8b"
    assert state.manager.loads == [("alpha-8b", {**args, "force": False, "source": "gui"})]
    assert "alpha-8b" not in models_page._PENDING and "alpha-8b" not in models_page._ERRORS


async def test_a_failed_load_is_kept_for_the_row_with_its_code_and_sentence(
    config: Config,
) -> None:
    from studioforge.gui.tabs import models_page

    state = _models_state(config)
    state.manager.fail_with = StudioForgeError(
        "alpha-8b needs 40 GiB but 30 GiB are usable.", code="insufficient_vram", status_code=409
    )
    ctx = tabs.GuiContext(config=config, api_state=state)
    try:
        with pytest.raises(StudioForgeError):
            await models_page.run_load(ctx, "alpha-8b", {}, force=True)
        error = models_page._ERRORS["alpha-8b"]
        assert error.code == "insufficient_vram"
        assert error.message == "alpha-8b needs 40 GiB but 30 GiB are usable."
        text = fetch(config, state, "/?section=Models").text
        assert "[insufficient_vram]" in text and "needs 40 GiB but 30 GiB are usable" in text
        assert state.manager.loads[0][1]["force"] is True
    finally:
        models_page._ERRORS.clear()
        models_page._PENDING.clear()


def test_a_loading_row_shows_progress_inline(config: Config) -> None:
    from studioforge.gui.tabs import models_page

    state = _models_state(config)
    models_page._PENDING["alpha-8b"] = 0.0
    try:
        text = fetch(config, state, "/?section=Models").text
    finally:
        models_page._PENDING.clear()
    assert "loading" in text


def test_aliases_are_only_the_names_that_really_resolve_to_the_model(config: Config) -> None:
    from studioforge.gui.tabs import models_page

    state = _models_state(config)
    record = state.registry._records[0]
    state.registry._alias_candidates = lambda r: [r.id, "alpha", "ghost"]
    state.registry.resolve = lambda name: record if name in ("alpha-8b", "alpha") else None
    ctx = tabs.GuiContext(config=config, api_state=state)
    assert models_page.aliases_of(ctx, record) == ["alpha"]
    bare = tabs.GuiContext(config=config, api_state=_FakeState(config))
    assert models_page.aliases_of(bare, record) == []


# ---------------------------------------------------------------------------
# Test Chat
# ---------------------------------------------------------------------------


def test_test_chat_keeps_the_chat_features_and_adds_curl_and_the_load_drawer(
    config: Config,
) -> None:
    text = fetch(config, _models_state(config), "/?section=Test%20Chat").text
    assert "could not be rendered" not in text
    for needle in (
        "Copy as curl",
        "Load options…",
        "Copy all",
        "Clear all",
        "Request settings",
        "Quick tests",
        "Conversation",
    ):
        assert needle in text, needle


def test_chat_curl_repeats_the_request_against_the_gateway_and_never_carries_a_key() -> None:
    text = mv.chat_curl(
        base_url="http://127.0.0.1:1234/v1/",
        model_id="alpha-8b",
        messages=[
            {"role": "system", "content": "Be brief."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what's this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            },
        ],
        sampler={"temperature": 0.2, "stream_options": {"include_usage": True}},
        needs_key=True,
    )
    assert text.startswith("curl -N http://127.0.0.1:1234/v1/chat/completions")
    assert "Authorization: Bearer $STUDIOFORGE_API_KEY" in text
    assert "AAAA" not in text and "<image omitted>" in text
    assert '"temperature": 0.2' in text and "stream_options" not in text
    assert "'\"'\"'" in text  # the apostrophe in "what's this?" is shell-escaped
    keyless = mv.chat_curl(base_url="http://x/v1", model_id="m", messages=[], needs_key=False)
    assert "Authorization" not in keyless


# ---------------------------------------------------------------------------
# Advanced pages that are new or moved
# ---------------------------------------------------------------------------


def test_watchdog_page_has_its_settings_and_the_recovery_controls(config: Config) -> None:
    text = fetch(config, _FakeState(config), "/?section=Advanced&page=watchdog").text
    for needle in (
        "Run the watchdog",
        "Watchdog port",
        "Unload all models",
        "Restart server",
    ):
        assert needle in text, needle


def test_engines_page_has_restart_engines_and_the_old_engine_cards(config: Config) -> None:
    text = fetch(config, _FakeState(config), "/?section=Advanced&page=engines").text
    assert "Restart engines" in text and "llama.cpp engine" in text


def test_placement_page_has_vram_holders_leases_and_tenants(config: Config) -> None:
    text = fetch(config, _FakeState(config), "/?section=Advanced&page=placement").text
    for needle in ("VRAM holders", "GPU leases", "Tenants", "GPUs &amp; memory"):
        assert needle in text, needle


def test_diagnostics_page_carries_the_readiness_checklist(config: Config) -> None:
    text = fetch(config, _FakeState(config), "/?section=Advanced&page=diagnostics").text
    assert "Ready to serve" in text or "Setup" in text
    assert "Run checks" in text


def test_about_page_names_the_version_and_the_build(config: Config) -> None:
    from studioforge import __version__

    text = fetch(config, _FakeState(config), "/?section=Advanced&page=about").text
    assert "About StudioForge" in text and f"v{__version__}" in text and "Build" in text


def test_the_config_page_links_to_the_checks_and_keeps_the_raw_editor(config: Config) -> None:
    text = fetch(config, _FakeState(config), "/?section=Advanced&page=config").text
    assert "Run checks" in text and "Save configuration" in text
    assert "Advanced options, All settings" in text


# ---------------------------------------------------------------------------
# Security and accessibility basics
# ---------------------------------------------------------------------------


def test_the_panel_registers_no_route_of_its_own_for_the_new_tabs(config: Config) -> None:
    """The redirect is the only new behaviour: no new path, no new API surface."""
    app = create_gui_app(config, api_state=_FakeState(config))
    paths = {getattr(route, "path", "") for route in app.routes}
    for forbidden in ("/models", "/status", "/advanced", "/test-chat", "/api/panel"):
        assert forbidden not in paths, forbidden


def test_the_redirect_target_is_always_a_relative_path() -> None:
    nasty = {
        "section": "//evil.example/x",
        "tab": "https://evil.example",
        "repo": "https://evil.example/a/b",
        "model": "//evil.example",
    }
    target = nav.canonical_url(nav.resolve_location(nasty))
    assert target.startswith("/") and not target.startswith("//")
    assert "evil.example" not in target.split("?", 1)[0]


def test_a_remote_viewer_on_an_open_install_cannot_pin_or_delete(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The drawer's box-changing buttons are disabled with the reason, as the old table's were."""
    monkeypatch.setattr(tabs, "viewer_host", lambda: "192.168.1.50")
    text = _drawer_html(config, _models_state(config))
    assert "this changes the server itself" in text  # the D32 note is the buttons' tooltip


def test_interactive_controls_carry_accessible_names(config: Config) -> None:
    text = fetch(config, _models_state(config), "/?section=Models").text
    for needle in (
        '"aria-label":"Search models"',
        '"aria-label":"Sort models"',
        '"role":"button"',
        '"tabindex":"0"',
        '"aria-pressed":"true"',
        '"aria-pressed":"false"',
    ):
        assert needle in text, needle
    drawer = _drawer_html(config, _models_state(config))
    for needle in ('"aria-label":"Close the details"', '"aria-label":"Copy the model id"'):
        assert needle in drawer, needle
