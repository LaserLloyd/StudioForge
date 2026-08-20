"""Tests for the web GUI.

NiceGUI's element tree is awkward to assert on, so these tests target the two
things that actually break in a control panel: the app/auth wiring, and the pure
derivation helpers in :mod:`studioforge.gui.state` where all the real logic
deliberately lives.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from studioforge.config import Config
from studioforge.gui import state as st
from studioforge.gui.app import COOKIE_NAME, LOGIN_PATH, create_gui_app, session_token
from studioforge.types import (
    AdapterAttachment,
    GgufMeta,
    GpuInfo,
    InstanceInfo,
    LoadPlan,
    ModelCapabilities,
    ModelRecord,
    ModelSettings,
    VirtualPreset,
)

GIB = 1024**3


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


class _FakeRegistry:
    def __init__(self, records: list[ModelRecord] | None = None) -> None:
        self._records = records or []

    def all(self) -> list[ModelRecord]:
        return list(self._records)

    def resolve(self, name: str) -> ModelRecord | None:
        return next((r for r in self._records if r.id == name), None)

    def adapters(self) -> list[Any]:
        return []


class _FakeSupervisor:
    def list(self) -> list[InstanceInfo]:
        return []

    def get(self, model_id: str) -> InstanceInfo | None:
        return None


class _FakeProbe:
    backend = "fake"

    def list_gpus(self) -> list[GpuInfo]:
        return [
            GpuInfo(
                index=0,
                name="RTX 5090",
                total_bytes=32 * GIB,
                free_bytes=30 * GIB,
                used_bytes=2 * GIB,
                utilization_pct=12.0,
                temperature_c=41.0,
                compute_capability=(12, 0),
            )
        ]


class _FakeState:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.registry = _FakeRegistry()
        self.supervisor = _FakeSupervisor()
        self.probe = _FakeProbe()
        self.manager = None
        self.planner = None
        self.engine_manager = None
        self.downloader = None
        self.started_at = 0.0


def make_record(
    model_id: str,
    *,
    vocab: int | None = 152064,
    arch: str = "qwen3",
    size: int = 8 * GIB,
    vision: bool = False,
    tools: bool = False,
    thinking: bool = False,
    embedding: bool = False,
    multi_part: bool = False,
    mtime: float = 0.0,
    virtual: bool = False,
    kind: str = "chat",
    quant: str = "Q4_K_M",
    base_model_id: str | None = None,
    preset: VirtualPreset | None = None,
    settings: ModelSettings | None = None,
    last_used_at: float | None = None,
) -> ModelRecord:
    meta = GgufMeta(architecture=arch, n_vocab=vocab or 0, tokenizer_model="gpt2")
    return ModelRecord(
        id=model_id,
        name=model_id,
        kind=kind,  # type: ignore[arg-type]
        path=Path(f"/models/{model_id}.gguf"),
        size_bytes=size,
        quant=quant,
        architecture=arch,
        capabilities=ModelCapabilities(
            vision=vision,
            tools=tools,
            thinking=thinking,
            embedding=embedding,
            multi_part=multi_part,
        ),
        meta=meta,
        settings=settings or ModelSettings(),
        is_virtual=virtual,
        base_model_id=base_model_id,
        preset=preset,
        last_used_at=last_used_at,
        mtime=mtime,
    )


# ---------------------------------------------------------------------------
# App wiring
# ---------------------------------------------------------------------------


def test_create_gui_app_returns_app_and_starts_no_server(config: Config) -> None:
    app = create_gui_app(config, api_state=_FakeState(config))
    assert isinstance(app, FastAPI)


def test_create_gui_app_twice_does_not_raise(config: Config) -> None:
    first = create_gui_app(config, api_state=_FakeState(config))
    second = create_gui_app(config, api_state=_FakeState(config))
    assert isinstance(first, FastAPI)
    assert isinstance(second, FastAPI)
    assert first is not second


def test_index_serves_html(config: Config) -> None:
    app = create_gui_app(config, api_state=_FakeState(config))
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "StudioForge" in response.text


def test_gui_health_is_open(config: Config) -> None:
    app = create_gui_app(config, api_state=_FakeState(config))
    with TestClient(app) as client:
        assert client.get("/gui-health").json()["status"] == "ok"


def test_auth_gate_blocks_and_then_allows(config: Config) -> None:
    config.server.api_key = "sf-secret-key-1234"
    app = create_gui_app(config, api_state=_FakeState(config))
    with TestClient(app) as client:
        anonymous = client.get("/", follow_redirects=False)
        assert anonymous.status_code in (303, 401)

        with_header = client.get("/", headers={"Authorization": f"Bearer {config.server.api_key}"})
        assert with_header.status_code == 200
        assert "StudioForge" in with_header.text

        login = client.post(
            LOGIN_PATH, data={"api_key": config.server.api_key}, follow_redirects=False
        )
        assert login.status_code == 303
        assert COOKIE_NAME in login.cookies
        assert login.cookies[COOKIE_NAME] == session_token(config.server.api_key)

        with_cookie = client.get("/", cookies={COOKIE_NAME: session_token("sf-secret-key-1234")})
        assert with_cookie.status_code == 200


def test_login_rejects_a_wrong_key_without_leaking_it(config: Config) -> None:
    config.server.api_key = "sf-secret-key-1234"
    app = create_gui_app(config, api_state=_FakeState(config))
    with TestClient(app) as client:
        response = client.post(LOGIN_PATH, data={"api_key": "nope"}, follow_redirects=False)
    assert response.status_code == 401
    assert "nope" not in response.text
    assert COOKIE_NAME not in response.cookies


def test_login_cookie_is_not_the_api_key(config: Config) -> None:
    key = "sf-secret-key-1234"
    assert session_token(key) != key
    assert key not in session_token(key)


def test_no_gate_without_an_api_key(config: Config) -> None:
    config.server.api_key = None
    app = create_gui_app(config, api_state=_FakeState(config))
    with TestClient(app) as client:
        assert client.get("/").status_code == 200


# ---------------------------------------------------------------------------
# Proxy safety: no internal absolute URLs
# ---------------------------------------------------------------------------

#: Genuinely external destinations a user clicks through to. Everything else in
#: the GUI must be a relative path so the panel works on a plain-HTTP tailnet,
#: behind ``tailscale serve``'s HTTPS front end, and under any proxy prefix.
URL_ALLOWLIST = frozenset({"https://huggingface.co"})


def _gui_sources() -> list[Path]:
    root = Path(st.__file__).parent
    return sorted(root.rglob("*.py"))


def _string_constants(tree: ast.AST) -> list[str]:
    """Every string literal that is not a docstring."""
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_gui_sources_contain_no_internal_absolute_urls() -> None:
    offenders: list[str] = []
    for path in _gui_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for text in _string_constants(tree):
            for scheme in ("http://", "https://"):
                if scheme not in text:
                    continue
                if any(allowed in text for allowed in URL_ALLOWLIST):
                    continue
                offenders.append(f"{path.name}: {text[:80]!r}")
    assert offenders == [], f"absolute URLs in GUI sources: {offenders}"


def test_url_allowlist_is_actually_used() -> None:
    """The allowlist must describe reality, not be a blanket escape hatch."""
    from studioforge.gui.tabs import download

    assert download.HF_WEB_BASE in URL_ALLOWLIST
    assert len(URL_ALLOWLIST) == 1


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def test_format_bytes() -> None:
    assert st.format_bytes(None) == st.UNKNOWN
    assert st.format_bytes(512) == "512 B"
    assert st.format_bytes(2048) == "2.0 KiB"
    assert st.format_bytes(int(17.99 * GIB), precision=2) == "17.99 GiB"
    assert st.format_bytes(3 * 1024**4).endswith("TiB")


def test_format_duration() -> None:
    assert st.format_duration(None) == st.UNKNOWN
    assert st.format_duration(45) == "45s"
    assert st.format_duration(303) == "5m 03s"
    assert st.format_duration(7500) == "2h 05m"
    assert st.format_duration(-5) == "0s"


def test_ttl_text_prefers_pinned_over_any_countdown() -> None:
    assert st.ttl_text(120, pinned=True) == "pinned"
    assert st.ttl_text(None) == "no TTL"
    assert st.ttl_text(90) == "1m 30s"


def test_instance_ttl_countdown_is_live() -> None:
    import time

    instance = InstanceInfo(
        model_id="m",
        state="ready",
        ttl_s=600,
        started_at=time.time() - 60,
        last_activity_at=time.time() - 60,
    )
    text = st.instance_ttl_text(instance)
    assert text.startswith("9m")
    assert st.instance_ttl_text(instance, pinned=True) == "pinned"
    assert st.instance_ttl_text(None) == st.UNKNOWN


def test_capability_badges() -> None:
    plain = make_record("plain")
    assert st.capability_badges(plain) == []
    rich = make_record(
        "rich", vision=True, tools=True, embedding=True, multi_part=True, virtual=True
    )
    assert st.capability_badges(rich) == [
        "vision",
        "tools",
        "embedding",
        "multi-part",
        "virtual",
    ]


def test_capability_badges_distinguish_virtual_kinds() -> None:
    """persona vs LoRA vs bare alias: three very different VRAM stories."""
    persona = make_record(
        "persona", virtual=True, base_model_id="base", preset=VirtualPreset(temperature=0.7)
    )
    assert st.capability_badges(persona) == ["persona"]

    lora = make_record(
        "lora",
        virtual=True,
        base_model_id="base",
        settings=ModelSettings(adapters=[AdapterAttachment(adapter_id="a", scale=1.0)]),
    )
    assert st.capability_badges(lora) == ["LoRA"]

    both = make_record(
        "both",
        virtual=True,
        base_model_id="base",
        preset=VirtualPreset(system_prompt="You are terse."),
        settings=ModelSettings(adapters=[AdapterAttachment(adapter_id="a", scale=1.0)]),
    )
    assert st.capability_badges(both) == ["persona", "LoRA"]

    alias = make_record("alias", virtual=True, base_model_id="base")
    assert st.capability_badges(alias) == ["virtual"]


def test_status_labels_and_colours() -> None:
    assert st.model_status_label(None) == "unloaded"
    assert st.model_status_label(InstanceInfo(model_id="m", state="ready")) == "loaded"
    assert st.model_status_label(InstanceInfo(model_id="m", state="failed")) == "failed"
    assert st.status_colour("loaded") == "positive"
    assert st.status_colour("nonsense") == "grey"


def test_actual_ctx_uses_what_the_engine_reports() -> None:
    assert st.actual_ctx_text(None) == st.UNKNOWN
    assert st.actual_ctx_text({"loaded": False}) == st.UNKNOWN
    assert st.actual_ctx_text({"loaded": True, "actual": {"n_ctx": 4096}}) == "4096"
    assert st.actual_ctx_text({"loaded": True, "actual": {}}) == st.UNKNOWN


def test_slots_text() -> None:
    assert st.slots_text(None) == st.UNKNOWN
    slots = [{"is_processing": True}, {"is_processing": False}, {"is_processing": False}]
    assert st.slots_text(slots) == "1 busy / 2 idle"


def test_device_text() -> None:
    assert st.device_text(None) == st.UNKNOWN
    plan = LoadPlan(model_id="m", devices=[0, 2], split_mode="row")
    instance = InstanceInfo(model_id="m", state="ready", plan=plan)
    assert st.device_text(instance) == "GPU0, GPU2 (row)"


def test_gpu_helpers_tolerate_missing_telemetry() -> None:
    gpu = GpuInfo(index=1, name="RTX 3090", total_bytes=24 * GIB, free_bytes=24 * GIB)
    assert st.vram_fraction(gpu) == 0.0
    lines = st.gpu_detail_lines(gpu)
    assert any("utilisation unavailable" in line for line in lines)
    assert any("temp n/a" in line for line in lines)
    assert st.ram_text(0, 0) == "system RAM unavailable"


def test_log_line_text() -> None:
    text = st.log_line_text({"ts": 0, "level": "WARNING", "logger": "sf.core", "message": "hi"})
    assert "WARNING" in text
    assert "sf.core" in text
    assert text.endswith("hi")


# ---------------------------------------------------------------------------
# Live activity (LM Studio parity)
# ---------------------------------------------------------------------------


def _introspection(slots: list[dict[str, Any]], **actual: Any) -> dict[str, Any]:
    """Build the shape ``manager.introspect()`` returns, using its own derivation."""
    from studioforge.core.manager import slot_activity

    base = {"n_ctx": 2048, "total_slots": len(slots)}
    base.update(actual)
    return {
        "loaded": True,
        "actual": base,
        "activity": slot_activity(slots),
        "slots": slots,
    }


def test_activity_label_uses_the_engine_derived_string() -> None:
    generating = _introspection(
        [{"id": 0, "n_ctx": 2048, "is_processing": True, "n_prompt_tokens": 10, "n_decoded": 37}]
    )
    assert st.activity_label(generating) == "Generating - 37 tokens"
    assert st.activity_state(generating) == "generating"
    assert st.activity_colour(generating) == "positive"

    ingesting = _introspection(
        [
            {
                "id": 0,
                "is_processing": True,
                "n_prompt_tokens": 100,
                "n_prompt_tokens_processed": 40,
            }
        ]
    )
    assert st.activity_label(ingesting) == "Processing prompt 40/100 (40%)"
    assert st.activity_colour(ingesting) == "warning"

    idle = _introspection([{"id": 0, "is_processing": False}])
    assert st.activity_label(idle) == "Idle"
    assert st.activity_colour(idle) == "grey"
    assert st.activity_label(None) == st.UNKNOWN


def test_activity_slots_and_token_counts() -> None:
    data = _introspection(
        [
            {"id": 0, "is_processing": True, "n_prompt_tokens": 5, "n_decoded": 12},
            {"id": 1, "is_processing": False},
        ]
    )
    assert st.activity_slots_text(data) == "1 busy / 1 idle"
    assert st.tokens_generated(data) == 12
    assert st.tokens_generated(None) == 0


def test_activity_slots_text_falls_back_to_raw_slots() -> None:
    assert st.activity_slots_text({"loaded": True, "slots": [{"is_processing": True}]}) == (
        "1 busy / 0 idle"
    )


def test_speculative_badge_comes_from_slots_not_props() -> None:
    armed = _introspection([{"id": 0, "is_processing": False, "speculative": True}])
    armed["actual"]["speculative"] = True
    assert st.is_speculative(armed) is True
    # A per-slot flag alone is enough, because /props lies about this.
    slot_only = _introspection([{"id": 0, "is_processing": False, "speculative": True}])
    assert st.is_speculative(slot_only) is True
    assert st.is_speculative(_introspection([{"id": 0, "is_processing": False}])) is False
    assert st.is_speculative(None) is False


def test_prompt_cache_hits_are_surfaced() -> None:
    slot = {"prompt_tokens": 109, "prompt_tokens_cached": 100}
    assert st.prompt_cache_text(slot) == "cache hit 100/109 (91%)"
    assert st.prompt_cache_text({"prompt_tokens_cached": 7}) == "cache hit 7"
    assert st.prompt_cache_text({}) == "cache n/a"


def test_slot_line_includes_cache_and_draft() -> None:
    data = _introspection(
        [
            {
                "id": 0,
                "n_ctx": 2048,
                "speculative": True,
                "is_processing": False,
                "n_prompt_tokens": 109,
                "n_prompt_tokens_processed": 0,
                "n_prompt_tokens_cache": 100,
                "id_task": 0,
            }
        ]
    )
    rows = st.activity_slot_rows(data)
    assert len(rows) == 1
    line = st.slot_line(rows[0])
    assert "slot 0" in line
    assert "Idle" in line
    assert "ctx 2048" in line
    assert "cache hit 100/109" in line
    assert "draft" in line
    assert st.activity_slot_rows(None) == []


def test_modalities_and_build_info() -> None:
    data = _introspection([], modalities={"vision": True, "audio": False}, build_info="b10425")
    assert st.modalities_text(data) == "vision"
    assert st.build_info_text(data) == "b10425"
    assert st.modalities_text(None) == st.UNKNOWN
    assert st.modalities_text(_introspection([], modalities={})) == "text only"


# ---------------------------------------------------------------------------
# Speculative decoding results
# ---------------------------------------------------------------------------


def test_acceptance_rate_text() -> None:
    from studioforge.core.manager import draft_stats

    stats = draft_stats({"draft_n": 100, "draft_n_accepted": 80})
    text = st.acceptance_rate_text(stats)
    assert "80.0%" in text
    assert "good" in text

    poor = draft_stats({"draft_n": 100, "draft_n_accepted": 10})
    assert "10.0%" in st.acceptance_rate_text(poor)
    assert "poor" in st.acceptance_rate_text(poor)

    assert st.acceptance_rate_text({"speculative_used": False}) == ("speculative decoding not used")
    assert st.acceptance_rate_text({}) == "speculative decoding not used"


def test_test_result_lines_include_acceptance_and_engine_timings() -> None:
    result = {
        "latency_s": 1.5,
        "completion_tokens": 64,
        "tokens_per_second": 42.7,
        "speculative_used": True,
        "draft_tokens": 50,
        "draft_tokens_accepted": 35,
        "draft_acceptance_rate": 0.7,
        "engine_predicted_per_second": 44.12,
        "engine_prompt_per_second": 5991.6,
    }
    lines = st.test_result_lines(result)
    assert any("42.7 tok/s" in line for line in lines)
    assert any("5991.6" in line for line in lines)
    assert any("70.0%" in line for line in lines)


def test_test_result_lines_for_an_embedding_model() -> None:
    lines = st.test_result_lines({"embedding_dims": 1024, "latency_s": 0.2})
    assert lines[0].startswith("embedding of 1024")


def test_ab_comparison_lines() -> None:
    faster = st.ab_comparison_lines(
        {
            "tokens_per_second": 60.0,
            "speculative_used": True,
            "draft_n": 1,
            "draft_tokens": 100,
            "draft_tokens_accepted": 70,
            "draft_acceptance_rate": 0.7,
        },
        {"tokens_per_second": 40.0},
    )
    assert any("faster" in line for line in faster)
    slower = st.ab_comparison_lines(
        {
            "tokens_per_second": 30.0,
            "speculative_used": True,
            "draft_tokens": 100,
            "draft_tokens_accepted": 10,
            "draft_acceptance_rate": 0.1,
        },
        {"tokens_per_second": 40.0},
    )
    assert any("SLOWER" in line for line in slower)
    neutral = st.ab_comparison_lines({"tokens_per_second": 40.0}, {"tokens_per_second": 40.0})
    assert any("no meaningful difference" in line for line in neutral)
    assert len(st.ab_comparison_lines({}, {})) == 3


# ---------------------------------------------------------------------------
# Fit verdict
# ---------------------------------------------------------------------------


def _fits_preview(**overrides: Any) -> dict[str, Any]:
    preview = {
        "fits": True,
        "model_id": "m",
        "devices": [0],
        "split_mode": "none",
        "ctx_size": 8192,
        "parallel": 1,
        "kv_cache_type": "f16",
        "flash_attn": "auto",
        "per_gpu_bytes": {0: 12 * GIB},
        "evict_model_ids": [],
        "notes": [],
        "estimate_mb": {"total": 12288.0},
        "single_gpu": True,
    }
    preview.update(overrides)
    return preview


def test_fit_verdict_accepted_plan() -> None:
    verdict = st.fit_verdict(_fits_preview())
    assert verdict.fits is True
    assert verdict.headline == "Fits on GPU0"
    assert verdict.colour == "positive"
    assert any("8192" in line for line in verdict.detail_lines)
    assert st.per_gpu_projection_lines(verdict) == ["GPU0: 12.00 GiB projected"]


def test_fit_verdict_multi_gpu_and_eviction() -> None:
    verdict = st.fit_verdict(
        _fits_preview(
            devices=[0, 2],
            split_mode="layer",
            evict_model_ids=["old-model"],
            per_gpu_bytes={"0": GIB, "2": GIB},
        )
    )
    assert "GPU0, GPU2" in verdict.headline
    assert "layer" in verdict.headline
    assert any("old-model" in line for line in verdict.detail_lines)
    assert [index for index, _ in verdict.per_gpu] == [0, 2]


def test_fit_verdict_surfaces_the_fp4_note() -> None:
    note = (
        "placed on a GPU below compute capability 12.0: this quantization runs but "
        "without native acceleration, so expect notably slower prompt processing"
    )
    verdict = st.fit_verdict(_fits_preview(notes=[note]))
    assert verdict.fp4_warning == note
    assert note in verdict.as_text()


def test_fit_verdict_rejected_plan_includes_suggestions() -> None:
    preview = {
        "fits": False,
        "model_id": "m",
        "reason": "no single GPU has room",
        "message": "Cannot load 'm' entirely in VRAM",
        "required_bytes": 40 * GIB,
        "available_bytes": 24 * GIB,
        "per_gpu_free": {0: 20 * GIB, 1: 4 * GIB},
        "max_ctx_that_fits": 2048,
        "suggestions": ["lower ctx_size to 2048", "use a smaller quant"],
        "estimate_mb": {"total": 40960.0},
    }
    verdict = st.fit_verdict(preview)
    assert verdict.fits is False
    assert verdict.colour == "negative"
    assert verdict.headline == "Will not fit in VRAM"
    text = st.fit_verdict_text(preview)
    assert "40.00 GiB" in text
    assert "2048" in text
    assert "lower ctx_size to 2048" in text
    assert st.per_gpu_projection_lines(verdict)[0].endswith("free")


def test_fit_verdict_tolerates_an_empty_dict() -> None:
    verdict = st.fit_verdict({})
    assert verdict.fits is False
    assert verdict.as_text()


# ---------------------------------------------------------------------------
# Deep links (HuggingFace download button)
# ---------------------------------------------------------------------------


def test_deep_link_params_empty_query_changes_nothing() -> None:
    """A plain GET / must still land on the Dashboard, exactly as before."""
    for query in (None, {}):
        params = st.deep_link_params(query)
        assert params == {"tab": None, "repo": None, "quant": None, "model": None, "error": None}
        assert st.initial_tab(params) == "Dashboard"


def test_deep_link_params_download_link() -> None:
    params = st.deep_link_params(
        {"tab": "download", "repo": "lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF"}
    )
    assert params["tab"] == "Download"
    assert params["repo"] == "lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF"
    assert params["quant"] is None
    assert params["error"] is None
    assert st.initial_tab(params) == "Download"


def test_deep_link_params_matches_what_gui_url_for_emits() -> None:
    """Parse the producer's own output, so the two halves cannot drift."""
    from urllib.parse import urlparse

    from studioforge.core.protocol import gui_url_for, parse_deep_link

    config = Config()
    link = parse_deep_link(
        "lmstudio://open_from_hf?model=lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF"
    )
    url = gui_url_for(link, config)
    query = dict(part.split("=", 1) for part in urlparse(url).query.split("&") if "=" in part)
    params = st.deep_link_params(query)
    assert params["tab"] == "Download"
    assert params["repo"] == "lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF"


def test_deep_link_params_models_link() -> None:
    params = st.deep_link_params({"tab": "models", "model": "publisher/repo/some-model"})
    assert params["tab"] == "Models"
    assert params["model"] == "publisher/repo/some-model"
    assert params["repo"] is None
    assert st.initial_tab(params) == "Models"


def test_deep_link_params_infers_the_tab_from_the_target() -> None:
    assert st.deep_link_params({"repo": "owner/repo"})["tab"] == "Download"
    assert st.deep_link_params({"model": "a/b/c"})["tab"] == "Models"


def test_deep_link_params_treats_a_repo_shaped_model_on_download_as_a_repo() -> None:
    params = st.deep_link_params({"tab": "download", "model": "owner/repo"})
    assert params["repo"] == "owner/repo"
    assert params["model"] is None


def test_deep_link_params_rejects_a_traversal_repo() -> None:
    params = st.deep_link_params({"tab": "download", "repo": "../../etc/passwd"})
    assert params["repo"] is None
    assert params["error"] is not None
    # The requested tab is still honoured: that is where the error is shown.
    assert st.initial_tab(params) == "Download"
    # Without a tab there is nothing to open, so it falls back to the dashboard.
    assert st.initial_tab(st.deep_link_params({"repo": "../../etc/passwd"})) == "Dashboard"

    for bad in ("owner", "owner/repo/extra", "/repo", "owner/", "own er/repo", "http://x/y"):
        assert st.deep_link_params({"repo": bad})["repo"] is None, bad


def test_deep_link_params_rejects_a_bad_model_id() -> None:
    assert st.deep_link_params({"model": "../../secrets"})["model"] is None
    assert st.deep_link_params({"model": "a\nb"})["model"] is None


def test_deep_link_params_unknown_tab_falls_back_to_the_dashboard() -> None:
    params = st.deep_link_params({"tab": "nonsense"})
    assert params["tab"] is None
    assert params["error"] is not None
    assert st.initial_tab(params) == "Dashboard"


def test_deep_link_params_accepts_multi_value_and_aliased_keys() -> None:
    params = st.deep_link_params(
        {"tab": ["downloads"], "repo": ["owner/repo"], "quant": ["Q4_K_M"]}
    )
    assert params["tab"] == "Download"
    assert params["repo"] == "owner/repo"
    assert params["quant"] == "Q4_K_M"


def test_deep_link_quant_matching_is_forgiving() -> None:
    assert st.normalise_quant("Q4_K_M") == "q4km"
    assert st.quant_matches("q4-k-m", "Q4_K_M") is True
    assert st.quant_matches("Q4_K_M", "Q4_K_S") is False
    assert st.quant_matches(None, "Q4_K_M") is False
    assert st.quant_matches("", "") is False


def test_deep_link_headline() -> None:
    assert st.deep_link_headline("owner/repo", None) == "owner/repo"
    assert st.deep_link_headline("owner/repo", "Q4_K_M") == "owner/repo — Q4_K_M"


def test_initial_tab_rejects_an_arbitrary_string() -> None:
    assert st.initial_tab({"tab": "Nonsense"}) == "Dashboard"
    assert st.initial_tab({"tab": "Chat"}) == "Chat"


def test_index_honours_a_download_deep_link(config: Config) -> None:
    """The landing page must name the repo without touching the HF API."""
    app = create_gui_app(config, api_state=_FakeState(config))
    with TestClient(app) as client:
        response = client.get("/?tab=download&repo=lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF")
    assert response.status_code == 200
    assert "lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF" in response.text
    assert "From HuggingFace" in response.text
    assert "Choose which version to download" in response.text


def test_index_deep_link_with_a_quant(config: Config) -> None:
    app = create_gui_app(config, api_state=_FakeState(config))
    with TestClient(app) as client:
        response = client.get("/?tab=download&repo=owner/repo&quant=Q4_K_M")
    assert response.status_code == 200
    assert "owner/repo — Q4_K_M" in response.text


def test_index_deep_link_with_a_bad_repo_explains_itself(config: Config) -> None:
    app = create_gui_app(config, api_state=_FakeState(config))
    with TestClient(app) as client:
        response = client.get("/?tab=download&repo=not-a-repo")
    assert response.status_code == 200
    assert "That link could not be used" in response.text


def test_index_without_a_query_still_lands_on_the_dashboard(config: Config) -> None:
    app = create_gui_app(config, api_state=_FakeState(config))
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "From HuggingFace" not in response.text


# ---------------------------------------------------------------------------
# Download tab: search sorting and period
# ---------------------------------------------------------------------------


def test_search_sort_and_period_menus_map_onto_the_hf_search_api() -> None:
    """Every GUI menu entry must be a value HfSearch actually accepts."""
    from studioforge.core.hf_search import SORT_KEYS
    from studioforge.gui.tabs.download import PERIOD_CHOICES, SORT_CHOICES

    assert set(SORT_CHOICES.values()) <= set(SORT_KEYS)
    assert list(SORT_CHOICES) == [
        "Downloads (30d)",
        "Likes",
        "Recently updated",
        "Newly created",
        "Trending",
    ]
    assert list(PERIOD_CHOICES.values()) == [None, 1, 7, 30, 90]


def test_period_tooltip_follows_the_sort() -> None:
    """ "Newly created" filters on birth, everything else on last activity."""
    from studioforge.gui.tabs.download import _period_tooltip

    assert "creation date" in _period_tooltip("created")
    assert "last updated" in _period_tooltip("downloads")


def test_age_label_coarsens_with_distance() -> None:
    from studioforge.gui.tabs.download import _age_label

    assert _age_label(0.0) == "today"
    assert _age_label(0.9) == "today"
    assert _age_label(3.4) == "3d ago"
    assert _age_label(13.9) == "13d ago"
    assert _age_label(14.0) == "2w ago"
    assert _age_label(59.0) == "8w ago"
    assert _age_label(60.0) == "2mo ago"
    assert _age_label(400.0) == "13mo ago"


def test_age_label_is_empty_for_an_unknown_date() -> None:
    """No date means the whole clause is dropped, never "unknown ago"."""
    from studioforge.gui.tabs.download import _age_label

    assert _age_label(None) == ""


def test_download_tab_renders_the_sort_and_period_controls(config: Config) -> None:
    """The selects must survive a real NiceGUI render, tooltip nesting included."""
    app = create_gui_app(config, api_state=_FakeState(config))
    with TestClient(app) as client:
        response = client.get("/?tab=download")
    assert response.status_code == 200
    for label in ("Downloads (30d)", "Recently updated", "Past week", "Any time"):
        assert label in response.text


class _FakeDownloader:
    """The queue panel's whole surface: enough to render an empty queue."""

    def all(self) -> list[Any]:
        return []

    def active(self) -> list[Any]:
        return []

    def group_status(self, group_id: str) -> str:  # pragma: no cover - no rows
        return "queued"

    def queued_remaining_bytes(self) -> int:
        return 38 * GIB


def test_download_tab_shows_the_disk_headroom(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The line has to survive a real render, on the first paint, not a poll later."""
    import shutil

    from studioforge.core import diskspace

    diskspace.clear_cache()
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: (4 * 1024**4, 0, 412 * GIB))
    state = _FakeState(config)
    state.downloader = _FakeDownloader()
    app = create_gui_app(config, api_state=state)
    with TestClient(app) as client:
        response = client.get("/?tab=download")
    diskspace.clear_cache()

    assert response.status_code == 200
    assert "Disk: 412.0 GiB free on" in response.text
    assert "38.0 GiB queued" in response.text


class _FakeOption:
    """One row of the quant picker: what ``LogicalDownload`` gives the tab."""

    def __init__(self, quant: str, total_bytes: int) -> None:
        self.quant = quant
        self.label = f"model-{quant}.gguf"
        self.total_bytes = total_bytes
        self.repo_id = "owner/repo"
        self.mmproj = None


class _FakeRepoFiles:
    def __init__(self, options: list[_FakeOption]) -> None:
        self._options = options

    def logical_models(self) -> list[_FakeOption]:
        return list(self._options)


def test_quant_rows_warn_only_the_quants_that_will_not_fit_on_disk(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """20 GiB left after the queue: the 12 GiB quant is fine, the 30 GiB one is not."""
    import shutil

    from nicegui import ui

    from studioforge.core import diskspace
    from studioforge.gui.tabs import GuiContext
    from studioforge.gui.tabs import download as tab

    diskspace.clear_cache()
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: (4 * 1024**4, 0, 118 * GIB))
    state = _FakeState(config)
    state.downloader = _FakeDownloader()  # 38 GiB still to fetch
    ctx = GuiContext(config=config, api_state=state)
    full = _FakeRepoFiles([_FakeOption("Q4_K_M", 12 * GIB), _FakeOption("Q8_0", 90 * GIB)])

    @ui.page("/_quant_rows_smoke")
    def _page() -> None:
        tab._quant_rows(ctx, full, highlight="Q4_K_M", on_picked=None)

    app = create_gui_app(config, api_state=state)
    with TestClient(app) as client:
        response = client.get("/_quant_rows_smoke")
    diskspace.clear_cache()

    assert response.status_code == 200
    # 118 GiB free - 38 GiB queued = 80 GiB: only the 90 GiB quant overruns it.
    assert response.text.count("not enough disk") == 1
    # The rest of the row survived the change.
    assert "model-Q4_K_M.gguf" in response.text
    assert "from your link" in response.text
    assert "fits one GPU" in response.text


async def test_the_fit_badge_is_replaced_by_the_planners_answer(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WP7's contradiction, closed.

    The weights-only badge said "fits one GPU" beside a context line reading
    ``1x5090: --``. Once the header is read, the badge is the planner's.
    """
    from types import SimpleNamespace

    from studioforge.core import hf_meta
    from studioforge.gui.tabs import GuiContext
    from studioforge.gui.tabs import download as tab

    class _Element:
        def __init__(self) -> None:
            self.text = ""
            self.props_applied: list[str] = []

        def set_text(self, value: str) -> None:
            self.text = value

        def props(self, value: str) -> None:
            self.props_applied.append(value)

    matrix = {
        "source": "remote-gguf-header",
        "placements": [
            {"key": "single_best", "weights_fit": False},
            {"key": "dual_best", "weights_fit": True},
        ],
    }

    async def _arch(*_a: Any, **_k: Any) -> Any:
        return SimpleNamespace(meta=None, source="remote-gguf-header", unavailable=None)

    monkeypatch.setattr(hf_meta, "repo_arch_meta", _arch)
    monkeypatch.setattr(hf_meta, "idle_planner", lambda planner: planner)
    monkeypatch.setattr(hf_meta, "context_matrix", lambda *_a, **_k: matrix)
    monkeypatch.setattr(hf_meta, "context_line", lambda _m: "1x5090: -- · 2x5090: 256k")
    monkeypatch.setattr(hf_meta, "context_tooltip", lambda _m: "tip")
    monkeypatch.setattr(hf_meta, "geometry_line", lambda _m: "attention: full · 65 layers")

    state = _FakeState(config)
    state.planner = object()
    ctx = GuiContext(config=config, api_state=state)
    option = _FakeOption("Q8_0", 30 * GIB)
    label, tip, fit_badge, geometry = _Element(), _Element(), _Element(), _Element()

    await tab._fill_context_lines(
        ctx, _FakeRepoFiles([option]), [(option, label, tip, fit_badge)], geometry
    )

    assert fit_badge.text == "needs multiple GPUs"
    assert fit_badge.props_applied == ["color=warning"]
    assert label.text.startswith("1x5090: --")
    assert tip.text == "tip"
    assert geometry.text.startswith("attention:")


async def test_an_approximate_matrix_leaves_the_first_paint_badge_alone(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No header (a gated repo) means no better answer, so nothing is rewritten."""
    from types import SimpleNamespace

    from studioforge.core import hf_meta
    from studioforge.gui.tabs import GuiContext
    from studioforge.gui.tabs import download as tab

    class _Element:
        def __init__(self) -> None:
            self.text = "fits one GPU"
            self.props_applied: list[str] = []

        def set_text(self, value: str) -> None:
            self.text = value

        def props(self, value: str) -> None:
            self.props_applied.append(value)

    async def _arch(*_a: Any, **_k: Any) -> Any:
        return SimpleNamespace(meta=None, source=None, unavailable="gated repo")

    monkeypatch.setattr(hf_meta, "repo_arch_meta", _arch)
    monkeypatch.setattr(hf_meta, "idle_planner", lambda planner: planner)
    monkeypatch.setattr(
        hf_meta, "context_matrix", lambda *_a, **_k: {"source": None, "placements": []}
    )
    monkeypatch.setattr(hf_meta, "context_line", lambda _m: "1x5090: 128k approx")
    monkeypatch.setattr(hf_meta, "context_tooltip", lambda _m: "tip")
    monkeypatch.setattr(hf_meta, "geometry_line", lambda _m: "geometry")

    state = _FakeState(config)
    state.planner = object()
    ctx = GuiContext(config=config, api_state=state)
    option = _FakeOption("Q8_0", 30 * GIB)
    label, tip, fit_badge, geometry = _Element(), _Element(), _Element(), _Element()

    await tab._fill_context_lines(
        ctx, _FakeRepoFiles([option]), [(option, label, tip, fit_badge)], geometry
    )

    assert fit_badge.text == "fits one GPU"
    assert fit_badge.props_applied == []
    assert label.text.endswith("approx")


def test_download_tab_without_a_downloader_says_so_instead(config: Config) -> None:
    """No disk line where there is no queue to measure against.

    The queue panel is rendered on its own page rather than through
    ``/?tab=download``: every tab paints into the same document, and the Setup
    tab legitimately shows a disk line for the model library, so asserting
    against the whole page would be asserting about the wrong panel.
    """
    from nicegui import ui

    from studioforge.gui.tabs import GuiContext
    from studioforge.gui.tabs import download as tab

    ctx = GuiContext(config=config, api_state=_FakeState(config))

    @ui.page("/_queue_panel_smoke")
    def _page() -> None:
        tab._queue_panel(ctx)

    app = create_gui_app(config, api_state=_FakeState(config))
    with TestClient(app) as client:
        response = client.get("/_queue_panel_smoke")
    assert "Downloads not available" in response.text
    assert "Disk:" not in response.text


def test_index_honours_a_models_deep_link(config: Config) -> None:
    app = create_gui_app(config, api_state=_FakeState(config))
    with TestClient(app) as client:
        response = client.get("/?tab=models&model=publisher/repo/some-model")
    assert response.status_code == 200
    assert "publisher/repo/some-model" in response.text


# ---------------------------------------------------------------------------
# Download queue
# ---------------------------------------------------------------------------


def test_group_download_rows_collapses_a_sharded_model() -> None:
    rows = [
        {
            "id": "g:part1",
            "group_id": "g",
            "repo_id": "bartowski/Foo-GGUF",
            "filename": "part1",
            "status": "running",
            "downloaded_bytes": 1 * GIB,
            "total_bytes": 4 * GIB,
            "speed_bps": 30 * 1024**2,
            "eta_s": 100.0,
            "error": None,
        },
        {
            "id": "g:part2",
            "group_id": "g",
            "repo_id": "bartowski/Foo-GGUF",
            "filename": "part2",
            "status": "queued",
            "downloaded_bytes": 0,
            "total_bytes": 4 * GIB,
            "speed_bps": 0.0,
            "eta_s": None,
            "error": None,
        },
    ]
    groups = st.group_download_rows(rows)
    assert len(groups) == 1
    group = groups[0]
    assert group.group_id == "g"
    assert group.label == "bartowski/Foo-GGUF"
    assert group.file_count == 2
    assert group.done_bytes == 1 * GIB
    assert group.total_bytes == 8 * GIB
    assert group.fraction == 0.125
    assert group.status == "running"  # worst-first: running beats queued
    assert "12%" in group.detail
    assert "2 files" in group.detail


def test_group_download_rows_defers_status_to_the_downloader() -> None:
    rows = [{"group_id": "g", "status": "completed", "downloaded_bytes": 1, "total_bytes": 1}]
    assert st.group_download_rows(rows)[0].status == "completed"
    # The downloader owns the "a group is only done when every file is" rule.
    deferred = st.group_download_rows(rows, status_for=lambda _gid: "paused")
    assert deferred[0].status == "paused"

    # A raising callable must not take the row down with it.
    def boom(_gid: str) -> str:
        raise RuntimeError("nope")

    assert st.group_download_rows(rows, status_for=boom)[0].status == "completed"


def test_group_download_rows_surfaces_the_first_error() -> None:
    rows = [
        {"group_id": "g", "status": "failed", "error": "checksum mismatch"},
        {"group_id": "g", "status": "queued"},
    ]
    group = st.group_download_rows(rows)[0]
    assert group.status == "failed"
    assert group.error == "checksum mismatch"
    assert st.download_status_colour("failed") == "negative"
    assert st.download_status_colour("nonsense") == "grey"


def test_download_group_row_matches_the_real_progress_shape() -> None:
    """Guard against the downloader's payload drifting away from this reader."""
    from studioforge.core.downloader import DownloadProgress

    progress = DownloadProgress(
        id="g:f",
        group_id="g",
        repo_id="pub/repo",
        filename="f.gguf",
        status="running",
        downloaded_bytes=5,
        total_bytes=10,
        speed_bps=1.0,
        eta_s=5.0,
        error=None,
    )
    group = st.group_download_rows([progress.to_dict()])[0]
    assert group.group_id == "g"
    assert group.label == "pub/repo"
    assert group.fraction == 0.5
    assert group.status == "running"


def test_download_fit_verdict() -> None:
    gpus = [
        GpuInfo(index=0, name="a", total_bytes=32 * GIB, free_bytes=30 * GIB),
        GpuInfo(index=1, name="b", total_bytes=24 * GIB, free_bytes=20 * GIB),
    ]
    assert st.download_fit_verdict(None, gpus) == "size unknown"
    assert st.download_fit_verdict(10 * GIB, []) == "no GPU detected"
    assert st.download_fit_verdict(10 * GIB, gpus).startswith("fits one GPU")
    assert st.download_fit_verdict(40 * GIB, gpus).startswith("needs multiple GPUs")
    assert st.download_fit_verdict(400 * GIB, gpus).startswith("will not fit")


# ---------------------------------------------------------------------------
# Fit badge from the planner's own placements
# ---------------------------------------------------------------------------


def context_fit(
    *,
    single: bool | None = True,
    dual: bool = True,
    source: str | None = "remote-gguf-header",
) -> dict[str, Any]:
    """A trimmed ``hf_meta.context_matrix`` payload: just what the badge reads."""
    placements: list[dict[str, Any]] = []
    if single is not None:
        placements.append({"key": "single_best", "label": "1x RTX 5090", "weights_fit": single})
    placements.append({"key": "dual_best", "label": "2x RTX 5090", "weights_fit": dual})
    return {"source": source, "placements": placements}


def test_fit_badge_prefers_a_single_gpu_placement() -> None:
    assert st.fit_badge_from_context(context_fit(single=True)) == ("fits one GPU", "positive")


def test_fit_badge_falls_back_to_multiple_gpus() -> None:
    """The WP7 contradiction: weights alone fit one card, the planner disagrees."""
    assert st.fit_badge_from_context(context_fit(single=False, dual=True)) == (
        "needs multiple GPUs",
        "warning",
    )


def test_fit_badge_says_no_when_nothing_holds_it() -> None:
    assert st.fit_badge_from_context(context_fit(single=False, dual=False)) == (
        "will not fit",
        "negative",
    )


def test_fit_badge_declines_without_a_header() -> None:
    """An approximate matrix is no better than the weights-only badge it would
    be replacing, so the caller keeps that one."""
    assert st.fit_badge_from_context(context_fit(source=None)) is None


def test_fit_badge_declines_without_placements() -> None:
    assert st.fit_badge_from_context({"source": "remote-gguf-header", "placements": []}) is None
    assert st.fit_badge_from_context({}) is None
    assert st.fit_badge_from_context(None) is None


def test_fit_badge_labels_match_the_weights_only_verdict() -> None:
    """A row must not appear to change its mind when only the *source* changed."""
    gpus = [
        GpuInfo(index=0, name="a", total_bytes=32 * GIB, free_bytes=30 * GIB),
        GpuInfo(index=1, name="b", total_bytes=24 * GIB, free_bytes=20 * GIB),
    ]
    for label, _colour in (
        st.fit_badge_from_context(context_fit(single=True)),
        st.fit_badge_from_context(context_fit(single=False, dual=True)),
        st.fit_badge_from_context(context_fit(single=False, dual=False)),
    ):  # type: ignore[misc]
        assert label in {
            st.download_fit_verdict(size, gpus).split(" (")[0]
            for size in (10 * GIB, 40 * GIB, 400 * GIB)
        }


# ---------------------------------------------------------------------------
# Disk space
# ---------------------------------------------------------------------------


def disk(
    *, free: int, queued: int = 0, total: int = 4 * 1024**4, low: bool = False
) -> dict[str, Any]:
    return {
        "path": "D:/models",
        "drive": "D:",
        "total_bytes": total,
        "free_bytes": free,
        "queued_bytes": queued,
        "free_after_queue_bytes": free - queued,
        "low": low,
        "error": None,
    }


def test_disk_line_without_a_queue_says_only_what_is_free() -> None:
    line = st.disk_line(disk(free=412 * GIB))
    assert line == "Disk: 412.0 GiB free on D:"
    assert "queued" not in line


def test_disk_line_projects_the_queue() -> None:
    line = st.disk_line(disk(free=412 * GIB, queued=38 * GIB))
    assert line == "Disk: 412.0 GiB free on D: · 38.0 GiB queued → ~374.0 GiB after downloads"


def test_disk_line_says_short_when_the_queue_overruns() -> None:
    line = st.disk_line(disk(free=30 * GIB, queued=70 * GIB, low=True))
    assert "40.0 GiB SHORT" in line
    assert "-" not in line.split("→")[1]


def test_disk_line_is_empty_without_a_report() -> None:
    """No models.dir configured: the tab shows nothing rather than a zero."""
    assert st.disk_line(None) == ""


def test_disk_line_explains_an_unmeasurable_volume() -> None:
    report = disk(free=0, total=0)
    report["error"] = "OSError: device not ready"
    line = st.disk_line(report)
    assert line.startswith("Disk: unavailable")
    assert "device not ready" in line


def test_disk_is_low_reads_the_reports_own_verdict() -> None:
    assert st.disk_is_low(disk(free=5 * GIB, low=True)) is True
    assert st.disk_is_low(disk(free=500 * GIB)) is False
    assert st.disk_is_low(None) is False


def test_disk_would_overflow_measures_against_what_the_queue_leaves() -> None:
    report = disk(free=100 * GIB, queued=80 * GIB)
    assert st.disk_would_overflow(report, 15 * GIB) is False
    assert st.disk_would_overflow(report, 25 * GIB) is True


def test_disk_would_overflow_stays_quiet_when_it_cannot_tell() -> None:
    """A warning the user can disprove teaches them to ignore the real one."""
    assert st.disk_would_overflow(None, 25 * GIB) is False
    assert st.disk_would_overflow(disk(free=0, total=0), 25 * GIB) is False
    assert st.disk_would_overflow(disk(free=100 * GIB), None) is False


# ---------------------------------------------------------------------------
# Context / slots
# ---------------------------------------------------------------------------


def test_per_slot_ctx_hint_is_silent_for_one_slot() -> None:
    assert st.per_slot_ctx_hint(8192, 1) == ""
    assert st.per_slot_ctx_hint(8192, None) == ""


def test_per_slot_ctx_hint_names_both_numbers() -> None:
    hint = st.per_slot_ctx_hint(8192, 4)
    assert "32768" in hint  # the total llama-server is launched with
    assert "8192" in hint  # what each conversation actually gets
    assert "TOTAL" in hint
    assert st.total_ctx_tokens(8192, 4) == 32768


def test_per_slot_ctx_hint_falls_back_to_the_global_default() -> None:
    hint = st.per_slot_ctx_hint(None, 2, default_ctx=4096)
    assert "8192" in hint


# ---------------------------------------------------------------------------
# Settings form round-trip
# ---------------------------------------------------------------------------


def test_form_round_trip_keeps_none_as_none() -> None:
    settings = ModelSettings()
    form = st.form_from_settings(settings)
    assert form["ctx_size"] is None
    assert form["kv_cache_type"] == ""  # rendered as an empty select == "Auto"
    assert form["device_override"] == ""
    restored = st.settings_from_form(form)
    assert restored == settings
    # The point of all of it: Auto must still mean "ask the planner at load time".
    assert restored.ctx_size is None
    assert restored.kv_cache_type is None
    assert restored.parallel is None
    assert restored.device_override is None


def test_form_round_trip_keeps_concrete_values() -> None:
    settings = ModelSettings(
        ctx_size=16384,
        kv_cache_type="q8_0",
        kv_cache_type_v="q8_0",
        ttl_s=0,
        pinned=True,
        draft_model_id="small",
        device_override=[0, 2],
        parallel=4,
        cont_batching=True,
        flash_attn="on",
        split_mode="row",
        main_gpu=2,
        mlock=True,
        no_mmap=False,
        cache_reuse=512,
        temperature=0.4,
        extra_flags="--verbose",
        adapters=[AdapterAttachment(adapter_id="lora-a", scale=0.75)],
    )
    restored = st.settings_from_form(st.form_from_settings(settings))
    assert restored == settings
    assert restored.device_override == [0, 2]
    assert restored.adapters[0].scale == 0.75
    assert restored.extra_flags == "--verbose"


def test_settings_from_form_treats_blank_text_as_auto() -> None:
    form = st.form_from_settings(ModelSettings(ctx_size=4096, rope_scaling="linear"))
    form["ctx_size"] = ""
    form["rope_scaling"] = "   "
    settings = st.settings_from_form(form)
    assert settings.ctx_size is None
    assert settings.rope_scaling is None


def test_settings_from_form_keeps_empty_extra_flags_as_a_string() -> None:
    form = st.form_from_settings(ModelSettings())
    form["extra_flags"] = "   "
    assert st.settings_from_form(form).extra_flags == ""


def test_settings_from_form_ignores_unknown_keys() -> None:
    form = st.form_from_settings(ModelSettings())
    form["not_a_setting"] = "boom"
    assert st.settings_from_form(form) == ModelSettings()


def test_parse_device_list() -> None:
    assert st.parse_device_list("") is None
    assert st.parse_device_list(None) is None
    assert st.parse_device_list("0, 2") == [0, 2]
    assert st.parse_device_list([1, 3]) == [1, 3]
    with pytest.raises(ValueError):
        st.parse_device_list("cuda0")
    assert st.format_device_list(None) == ""
    assert st.format_device_list([0, 1]) == "0,1"


def test_cache_reuse_hint_says_on_by_default() -> None:
    hint = st.cache_reuse_hint(None, 256)
    assert "ON" in hint
    assert "256" in hint
    assert "inherited default" in hint
    assert "OFF" in st.cache_reuse_hint(0, 256)


# ---------------------------------------------------------------------------
# Draft models
# ---------------------------------------------------------------------------


def test_plausible_draft_models_filters_by_vocab_and_size() -> None:
    target = make_record("big", vocab=152064, arch="qwen3", size=30 * GIB)
    same_family_small = make_record("small", vocab=152064, arch="qwen3", size=1 * GIB)
    other_vocab = make_record("llama", vocab=32000, arch="llama", size=1 * GIB)
    bigger = make_record("bigger", vocab=152064, arch="qwen3", size=60 * GIB)
    embedding = make_record("embed", vocab=152064, arch="qwen3", size=1, kind="embedding")

    candidates = st.plausible_draft_models(
        [target, same_family_small, other_vocab, bigger, embedding], target
    )
    ids = [r.id for r in candidates]
    assert ids == ["small"]
    assert "big" not in ids  # never its own draft
    assert "llama" not in ids  # different vocab size
    assert "bigger" not in ids  # a draft must be smaller than its target


def test_plausible_draft_models_uses_arch_when_vocab_is_unknown() -> None:
    target = make_record("big", vocab=None, arch="qwen3", size=30 * GIB)
    same_arch = make_record("small", vocab=None, arch="qwen3", size=GIB)
    other_arch = make_record("other", vocab=None, arch="gemma3", size=GIB)
    ids = [r.id for r in st.plausible_draft_models([target, same_arch, other_arch], target)]
    assert ids == ["small"]


def test_draft_uncertainty_note() -> None:
    target = make_record("big", vocab=152064)
    good = make_record("small", vocab=152064, size=GIB)
    bad = make_record("mismatch", vocab=32000, size=GIB)
    unknown = make_record("unknown", vocab=None, size=GIB)
    assert st.draft_uncertainty_note(target, None) is None
    assert st.draft_uncertainty_note(target, good) is None
    note = st.draft_uncertainty_note(target, bad)
    assert note is not None and "vocab mismatch" in note
    note = st.draft_uncertainty_note(target, unknown)
    assert note is not None and "could not be verified" in note


# ---------------------------------------------------------------------------
# Masked secrets
# ---------------------------------------------------------------------------


def test_masked_secret_unchanged_is_not_sent() -> None:
    assert st.masked_secret_changed("sf-1...ab", "sf-1...ab") is False


def test_masked_secret_new_value_is_sent() -> None:
    assert st.masked_secret_changed("sf-1...ab", "a-genuinely-new-key") is True


def test_masked_secret_blank_and_placeholder_shapes_are_never_sent() -> None:
    # Blank means "leave it alone", not "clear the key".
    assert st.masked_secret_changed("sf-1...ab", "") is False
    assert st.masked_secret_changed("sf-1...ab", "   ") is False
    assert st.masked_secret_changed("sf-1...ab", None) is False
    assert st.masked_secret_changed(None, "***") is False
    # A *different* redaction (e.g. the field re-rendered from another key) is
    # still a placeholder and must not be written back.
    assert st.masked_secret_changed("sf-1...ab", "abcd...yz") is False


def test_masked_secret_matches_the_api_redaction_format() -> None:
    from studioforge.api.auth import redact

    key = "sf-supersecret-value"
    masked = redact(key)
    assert masked is not None
    assert st.masked_secret(key) == masked
    assert st.masked_secret_changed(masked, masked) is False
    assert st.masked_secret_changed(masked, key) is True


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def test_config_fields_flag_restart_required_keys() -> None:
    by_key = {f.key: f for f in st.CONFIG_FIELDS}
    assert by_key["server.port"].restart_required is True
    assert by_key["gui.port"].restart_required is True
    assert by_key["models.default_ctx"].restart_required is False
    assert set(st.SECRET_CONFIG_KEYS) == {"server.api_key", "hf.token"}


def test_restart_required_keys_filters() -> None:
    assert st.restart_required_keys(["models.default_ctx", "server.port"]) == ["server.port"]


def test_config_value_reads_dotted_paths(config: Config) -> None:
    payload = config.to_yaml_dict()
    assert st.config_value(payload, "server.port") == config.server.port
    assert st.config_value(payload, "nope.nothing") is None


def test_coerce_config_value() -> None:
    assert st.coerce_config_value("int", "8192") == 8192
    assert st.coerce_config_value("int", "") is None
    assert st.coerce_config_value("float", "0.1") == 0.1
    assert st.coerce_config_value("bool", "") is False
    assert st.coerce_config_value("list", "a, b") == ["a", "b"]
    assert st.coerce_config_value("text", "  x ") == "x"
    assert st.coerce_config_value("text", "") is None


def test_quant_affinity_summary() -> None:
    lines = st.quant_affinity_summary(
        {"NVFP4": {"min_compute_capability": "12.0", "mode": "prefer"}}
    )
    assert lines == ["NVFP4: prefers compute capability >= 12.0"]
    assert st.quant_affinity_summary(None) == ["no quant affinity configured"]


def test_quant_affinity_note_never_says_blackwell_only() -> None:
    note = st.QUANT_AFFINITY_NOTE
    assert "never excludes a GPU" in note
    assert "fully usable" in note
    assert "only" not in note.replace("the only mode", "").replace("only on Blackwell", "")


def test_config_panel_shows_real_quant_affinity_defaults(config: Config) -> None:
    families = sorted(config.planner.quant_affinity)
    assert families == ["MXFP4", "NVFP4"]
    for spec in config.planner.quant_affinity.values():
        assert spec.mode == "prefer"  # steering, never exclusion


# ---------------------------------------------------------------------------
# Reasoning format (DECISIONS D12)
# ---------------------------------------------------------------------------


def test_reasoning_format_is_exposed_and_round_trips() -> None:
    settings = ModelSettings(reasoning_format="deepseek", reasoning="on", reasoning_budget=512)
    form = st.form_from_settings(settings)
    assert form["reasoning_format"] == "deepseek"
    assert form["reasoning"] == "on"
    assert st.settings_from_form(form) == settings
    # Unset stays unset, so the global default still applies at launch.
    blank = st.form_from_settings(ModelSettings())
    assert blank["reasoning_format"] == ""
    assert st.settings_from_form(blank).reasoning_format is None


def test_reasoning_format_help_warns_about_empty_replies() -> None:
    assert "empty" in st.REASONING_FORMAT_HELP
    assert "reasoning_content" in st.REASONING_FORMAT_HELP
    safe = st.reasoning_format_hint(None, "none")
    assert "message.content" in safe
    risky = st.reasoning_format_hint("deepseek", "none")
    assert "empty reply" in risky


# ---------------------------------------------------------------------------
# First-run setup
# ---------------------------------------------------------------------------


def test_setup_status_flags_each_missing_piece() -> None:
    items = st.setup_status(
        model_dir=None, model_count=0, gpu_count=0, engine_tag=None, pinned_tag="b10425"
    )
    assert st.setup_is_ready(items) is False
    by_name = {item.name: item for item in items}
    assert by_name["Models indexed"].action == "scan"
    assert by_name["llama.cpp engine"].action == "install-engine"
    assert "b10425" in by_name["llama.cpp engine"].detail
    assert by_name["GPUs detected"].ok is False
    assert all(item.colour == "warning" for item in items)


def test_setup_status_ready_state_has_no_actions() -> None:
    items = st.setup_status(
        model_dir="D:/models",
        model_count=28,
        gpu_count=4,
        engine_tag="b10425",
        pinned_tag="b10425",
    )
    assert st.setup_is_ready(items) is True
    assert all(item.action == "" for item in items)
    assert all(item.icon == "check_circle" for item in items)
    assert any("28 model(s)" in item.detail for item in items)


# ---------------------------------------------------------------------------
# Chat helpers
# ---------------------------------------------------------------------------


def test_vision_attach_reason() -> None:
    assert st.vision_attach_reason(None) is not None
    text = make_record("text-only")
    reason = st.vision_attach_reason(text)
    assert reason is not None and "no vision projector" in reason
    assert st.vision_attach_reason(make_record("vlm", vision=True)) is None


def test_build_chat_content() -> None:
    assert st.build_chat_content("hi", []) == "hi"
    content = st.build_chat_content("look", ["data:image/png;base64,AAA"])
    assert content[0] == {"type": "text", "text": "look"}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_tokens_per_second() -> None:
    assert st.tokens_per_second(0, 1.0) is None
    assert st.tokens_per_second(10, 0.0) is None
    assert st.tokens_per_second(100, 2.0) == 50.0


def test_chat_model_records_exclude_embedding_models() -> None:
    records = [
        make_record("chatty"),
        make_record("embedder", kind="embedding", embedding=True),
    ]
    assert [r.id for r in st.chat_model_records(records)] == ["chatty"]


def test_hidden_chat_models_note_explains_the_absence() -> None:
    assert st.hidden_chat_models_note([make_record("chatty")]) is None
    note = st.hidden_chat_models_note(
        [make_record("chatty"), make_record("embedder", kind="embedding")]
    )
    assert note is not None
    assert "1 model(s) not shown" in note
    assert "embedding" in note
    assert "no chat endpoint" in note


# ---------------------------------------------------------------------------
# Model table sorting / filtering
# ---------------------------------------------------------------------------


def test_sort_models_by_each_key() -> None:
    records = [
        make_record("bravo", size=2 * GIB, last_used_at=100.0, quant="Q8_0"),
        make_record("alpha", size=8 * GIB, last_used_at=None, quant="Q4_K_M"),
        make_record("charlie", size=4 * GIB, last_used_at=200.0, quant="IQ2_XS"),
    ]
    assert [r.id for r in st.sort_models(records, "name")] == ["alpha", "bravo", "charlie"]
    assert [r.id for r in st.sort_models(records, "size")] == ["alpha", "charlie", "bravo"]
    assert [r.id for r in st.sort_models(records, "recent")] == ["charlie", "bravo", "alpha"]
    assert [r.id for r in st.sort_models(records, "quant")] == ["charlie", "alpha", "bravo"]
    loaded = [r.id for r in st.sort_models(records, "loaded", loaded_ids={"charlie"})]
    assert loaded == ["charlie", "alpha", "bravo"]


def test_sort_models_unknown_key_degrades_to_name() -> None:
    """A stale stored preference must never break the table."""
    records = [make_record("b"), make_record("a")]
    assert [r.id for r in st.sort_models(records, "nonsense")] == ["a", "b"]
    assert [r.id for r in st.sort_models(records, None)] == ["a", "b"]


def test_sort_keys_all_have_labels() -> None:
    assert "name" in st.MODEL_COLUMN_BY_KEY
    for key, column in st.MODEL_COLUMN_BY_KEY.items():
        assert column.key == key
        assert column.label


def test_filter_models_matches_more_than_the_id() -> None:
    records = [
        make_record("qwen3-30b", quant="NVFP4", arch="qwen3"),
        make_record("llava-vision", vision=True, arch="llama"),
        make_record("embedder", kind="embedding", embedding=True),
    ]
    assert [r.id for r in st.filter_models(records, "nvfp4")] == ["qwen3-30b"]
    assert [r.id for r in st.filter_models(records, "vision")] == ["llava-vision"]
    assert [r.id for r in st.filter_models(records, "embedding")] == ["embedder"]
    assert [r.id for r in st.filter_models(records, "llama")] == ["llava-vision"]
    assert len(st.filter_models(records, "")) == 3
    assert len(st.filter_models(records, None)) == 3
    assert st.filter_models(records, "no-such-model") == []


def test_filter_models_finds_pinned_models() -> None:
    """Typing what the badge says must find the models wearing it."""
    records = [
        make_record("kept-warm", settings=ModelSettings(pinned=True)),
        make_record("ordinary"),
    ]
    assert [r.id for r in st.filter_models(records, "pinned")] == ["kept-warm"]
    assert [r.id for r in st.filter_models(records, "PINNED")] == ["kept-warm"]


# ---------------------------------------------------------------------------
# Virtual models / presets (D13)
# ---------------------------------------------------------------------------


def test_shares_base_instance_mirrors_the_manager_rule() -> None:
    """Preset-only shares; any ModelSettings delta (adapters included) does not."""
    persona = make_record(
        "persona", virtual=True, base_model_id="base", preset=VirtualPreset(temperature=0.4)
    )
    assert st.shares_base_instance(persona) is True

    lora = make_record(
        "lora",
        virtual=True,
        base_model_id="base",
        settings=ModelSettings(adapters=[AdapterAttachment(adapter_id="a", scale=1.0)]),
    )
    assert st.shares_base_instance(lora) is False

    ctx_override = make_record(
        "ctx", virtual=True, base_model_id="base", settings=ModelSettings(ctx_size=4096)
    )
    assert st.shares_base_instance(ctx_override) is False

    real = make_record("real")
    assert st.shares_base_instance(real) is False


def test_shares_base_instance_agrees_with_manager_serving_record(config: Config) -> None:
    """The GUI's sharing rule and the manager's must be the same rule.

    If ``serving_record`` ever changes its criterion, this test fails and the
    persona dialog's VRAM-cost indicator gets updated with it instead of
    silently lying.
    """
    import inspect

    from studioforge.core.manager import ModelManager

    source = inspect.getsource(ModelManager.serving_record)
    # The manager shares when settings equal the default ModelSettings().
    assert "_DEFAULT_SETTINGS" in source
    persona = make_record(
        "persona", virtual=True, base_model_id="base", preset=VirtualPreset(temperature=0.4)
    )
    assert (persona.settings == ModelSettings()) == st.shares_base_instance(persona)


def test_has_launch_overrides_ignores_adapters() -> None:
    plain = ModelSettings()
    assert st.has_launch_overrides(plain) is False
    with_adapters = ModelSettings(adapters=[AdapterAttachment(adapter_id="a", scale=1.0)])
    assert st.has_launch_overrides(with_adapters) is False
    with_ctx = ModelSettings(ctx_size=8192)
    assert st.has_launch_overrides(with_ctx) is True


def test_virtual_instance_note_names_the_vram_cost() -> None:
    shares, text = st.virtual_instance_note("base-30b", has_adapters=False, has_overrides=False)
    assert shares is True
    assert "no extra VRAM" in text
    assert "base-30b" in text

    shares, text = st.virtual_instance_note("base-30b", has_adapters=True, has_overrides=False)
    assert shares is False
    assert "own llama-server instance" in text
    assert "LoRA adapters" in text
    assert "full VRAM" in text

    shares, text = st.virtual_instance_note("base-30b", has_adapters=False, has_overrides=True)
    assert shares is False
    assert "setting overrides" in text

    shares, text = st.virtual_instance_note("base-30b", has_adapters=True, has_overrides=True)
    assert shares is False
    assert "LoRA adapters" in text and "setting overrides" in text


def test_virtual_base_line() -> None:
    assert st.virtual_base_line(make_record("real")) is None
    persona = make_record(
        "persona", virtual=True, base_model_id="base", preset=VirtualPreset(temperature=0.4)
    )
    line = st.virtual_base_line(persona)
    assert line is not None and "base" in line and "shares its instance" in line
    lora = make_record(
        "lora",
        virtual=True,
        base_model_id="base",
        settings=ModelSettings(adapters=[AdapterAttachment(adapter_id="a", scale=1.0)]),
    )
    line = st.virtual_base_line(lora)
    assert line is not None and "needs its own instance" in line


def test_preset_form_round_trip() -> None:
    preset = VirtualPreset(
        system_prompt="You are terse.",
        temperature=0.4,
        top_p=0.9,
        top_k=40,
        min_p=0.05,
        repeat_penalty=1.1,
        max_tokens=512,
    )
    form = st.form_from_preset(preset)
    assert form["system_prompt"] == "You are terse."
    rebuilt = st.preset_from_form(form)
    assert rebuilt == preset


def test_preset_from_form_blank_collapses_to_none() -> None:
    """An all-blank persona form stores no preset at all, not an empty one."""
    form = st.form_from_preset(None)
    assert form["system_prompt"] == ""
    assert st.preset_from_form(form) is None
    assert st.preset_from_form({"system_prompt": "   "}) is None


def test_preset_from_form_coerces_number_widget_floats() -> None:
    """NiceGUI number inputs hand back floats; int fields must stay ints."""
    preset = st.preset_from_form({"top_k": 40.0, "max_tokens": 256.0})
    assert preset is not None
    assert preset.top_k == 40 and isinstance(preset.top_k, int)
    assert preset.max_tokens == 256 and isinstance(preset.max_tokens, int)


def test_preset_from_form_rejects_invalid_values() -> None:
    with pytest.raises(Exception, match="temperature"):
        st.preset_from_form({"temperature": -1.0})


def test_preset_summary_lines() -> None:
    assert st.preset_summary_lines(None) == []
    lines = st.preset_summary_lines(
        VirtualPreset(system_prompt="You are terse.", temperature=0.4, max_tokens=64)
    )
    assert any("You are terse." in line for line in lines)
    assert any("temperature=0.4" in line and "max_tokens=64" in line for line in lines)
    assert any("client value wins" in line for line in lines)
    long_prompt = "x" * 200
    lines = st.preset_summary_lines(VirtualPreset(system_prompt=long_prompt))
    assert all(len(line) < 120 for line in lines)


# ---------------------------------------------------------------------------
# Dashboard degradation helpers
# ---------------------------------------------------------------------------


def test_fp4_plan_note_surfaces_the_planner_note() -> None:
    note = (
        "placed on a GPU below compute capability 12.0: this quantization runs but "
        "without native acceleration, so expect notably slower prompt processing"
    )
    plan = LoadPlan(model_id="m", devices=[2], notes=[note])
    instance = InstanceInfo(model_id="m", state="ready", plan=plan)
    assert st.fp4_plan_note(instance) == note
    assert st.fp4_plan_note(InstanceInfo(model_id="m", state="ready")) is None
    assert st.fp4_plan_note(None) is None
    boring = LoadPlan(model_id="m", devices=[0], notes=["evicted x"])
    assert st.fp4_plan_note(InstanceInfo(model_id="m", state="ready", plan=boring)) is None


def test_poll_failure_note_reads_as_stale_not_gone() -> None:
    note = st.poll_failure_note(RuntimeError("boom"))
    assert "boom" in note
    assert "retrying" in note
    assert "last good data" in note


# ---------------------------------------------------------------------------
# Rendering the new pieces
# ---------------------------------------------------------------------------


def test_index_renders_a_persona_row_and_chat_note(config: Config) -> None:
    """The Models tab must show a persona distinguishably, and the Chat tab
    must explain why an embedding model is not selectable."""
    state = _FakeState(config)
    state.registry = _FakeRegistry(
        [
            make_record("base-model"),
            make_record(
                "my-persona",
                virtual=True,
                base_model_id="base-model",
                preset=VirtualPreset(system_prompt="You are terse.", temperature=0.4),
            ),
            make_record("embedder", kind="embedding", embedding=True),
        ]
    )
    app = create_gui_app(config, api_state=state)
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "my-persona" in response.text
    assert "persona" in response.text
    assert "shares its instance" in response.text
    assert "You are terse." in response.text
    # The chat tab explains the hidden embedding model instead of failing later.
    assert "no chat endpoint" in response.text


# ---------------------------------------------------------------------------
# Dates: "when did this model arrive?"
# ---------------------------------------------------------------------------


def _epoch(year: int, month: int, day: int, hour: int = 12, minute: int = 0) -> float:
    """Local-time epoch, so the formatter's calendar-day logic is exercised."""
    return time.mktime((year, month, day, hour, minute, 0, 0, 0, -1))


def test_format_when_covers_every_age_band() -> None:
    now = _epoch(2026, 8, 14, 12)
    assert st.format_when(None) == st.UNKNOWN
    assert st.format_when(0) == st.UNKNOWN
    assert st.format_when(now - 5, now=now) == "just now"
    assert st.format_when(now - 60, now=now) == "1 minute ago"
    assert st.format_when(now - 120, now=now) == "2 minutes ago"
    assert st.format_when(now - 3600, now=now) == "1 hour ago"
    assert st.format_when(now - 2 * 3600, now=now) == "2 hours ago"
    assert st.format_when(_epoch(2026, 8, 13, 23, 30), now=now) == "yesterday"
    assert st.format_when(_epoch(2026, 7, 18, 9), now=now) == "18 Jul"
    assert st.format_when(_epoch(2025, 7, 18, 9), now=now) == "18 Jul 2025"


def test_format_when_uses_calendar_days_not_multiples_of_24h() -> None:
    """ "Yesterday" means the previous date to a reader, not "25 hours ago"."""
    now = _epoch(2026, 8, 14, 1)  # 01:00
    # 3 hours earlier is 22:00 the day before: a reader calls that yesterday.
    assert st.format_when(_epoch(2026, 8, 13, 22), now=now) == "yesterday"
    # 30 minutes earlier is still the small hours of today.
    assert st.format_when(now - 1800, now=now) == "30 minutes ago"


def test_format_when_tolerates_a_future_timestamp() -> None:
    """A restored archive or clock skew must not render "-3 hours ago"."""
    now = _epoch(2026, 8, 14, 12)
    assert st.format_when(now + 3600, now=now) == "13:00"


def test_format_datetime_is_absolute() -> None:
    assert st.format_datetime(None) == st.UNKNOWN
    assert st.format_datetime(_epoch(2026, 8, 14, 17, 32)) == "2026-08-14 17:32"


def test_model_added_at_prefers_mtime_over_added_at() -> None:
    """``added_at`` used to churn on every rescan; ``mtime`` is the stable one."""
    record = make_record("m", mtime=_epoch(2026, 8, 14, 17, 32))
    assert st.model_added_at(record) == _epoch(2026, 8, 14, 17, 32)
    # A virtual model has no file of its own, so added_at is the fallback.
    virtual = make_record("v", virtual=True, base_model_id="m")
    assert st.model_added_at(virtual) == virtual.added_at


# ---------------------------------------------------------------------------
# Capability icons
# ---------------------------------------------------------------------------


def test_capability_icons_cover_every_capability_flag() -> None:
    """Every field on ModelCapabilities must have an icon, or it is invisible."""
    fields = set(ModelCapabilities.model_fields)
    assert {item.key for item in st.CAPABILITY_ICONS} == fields


def test_capability_icons_for_a_model_with_none_is_empty() -> None:
    """No placeholder: a "none" chip in every row would be a column of noise."""
    assert st.capability_icons(make_record("plain")) == []


def test_capability_icons_are_ordered_coloured_and_explained() -> None:
    record = make_record(
        "rich", vision=True, tools=True, thinking=True, embedding=True, multi_part=True
    )
    icons = st.capability_icons(record)
    assert [item.key for item in icons] == [
        "vision",
        "tools",
        "thinking",
        "embedding",
        "multi_part",
    ]
    # An icon on its own is a rebus: each one must carry a tooltip and a colour,
    # and the colours must be distinguishable from each other.
    assert all(item.tooltip.strip() for item in icons)
    assert all(item.icon.strip() for item in icons)
    assert len({item.colour for item in icons}) == len(icons)


def test_thinking_tooltip_explains_where_the_thoughts_go() -> None:
    """DECISIONS D12 is the whole reason this capability is worth a badge."""
    thinking = next(item for item in st.CAPABILITY_ICONS if item.key == "thinking")
    lowered = thinking.tooltip.lower()
    assert "inline" in lowered
    assert "reasoning_format" in lowered


def test_capability_signature_groups_identical_feature_sets() -> None:
    a = make_record("a", vision=True, thinking=True)
    b = make_record("b", vision=True, thinking=True)
    c = make_record("c", vision=True)
    assert st.capability_signature(a) == st.capability_signature(b)
    assert st.capability_signature(c) != st.capability_signature(a)
    assert st.capability_signature(make_record("none")) == ""


def test_thinking_is_searchable() -> None:
    """The filter matches the badge text, so 'thinking' must produce one."""
    records = [make_record("reasoner", thinking=True), make_record("plain")]
    assert [r.id for r in st.filter_models(records, "thinking")] == ["reasoner"]


# ---------------------------------------------------------------------------
# Sortable column headers
# ---------------------------------------------------------------------------


def test_default_sort_is_newest_downloaded_first() -> None:
    """A browser with no stored preference lands on the newest downloads."""
    assert st.DEFAULT_SORT_KEY == "date"
    assert st.MODEL_COLUMN_BY_KEY["date"].descending_first is True
    assert st.stored_sort_key(None) == "date"
    assert st.stored_sort_key("nonsense-from-an-old-build") == "date"
    assert st.stored_sort_key("size") == "size"
    assert st.stored_sort_descending(None, "date") is True
    assert st.stored_sort_descending(None, "name") is False
    assert st.stored_sort_descending(False, "date") is False


def test_sort_models_by_download_date() -> None:
    records = [
        make_record("older", mtime=_epoch(2026, 8, 12, 21, 43)),
        make_record("newest", mtime=_epoch(2026, 8, 14, 17, 32)),
        make_record("middle", mtime=_epoch(2026, 8, 14, 0, 13)),
    ]
    assert [r.id for r in st.sort_models(records, "date")] == ["newest", "middle", "older"]
    assert [r.id for r in st.sort_models(records, "date", False)] == [
        "older",
        "middle",
        "newest",
    ]


def test_sort_models_by_architecture_and_type() -> None:
    records = [
        make_record("zeta", arch="qwen3", vision=True),
        make_record("alpha", arch="llama"),
        make_record("beta", arch="qwen3"),
        make_record("embed", arch="bert", kind="embedding", embedding=True),
    ]
    assert [r.id for r in st.sort_models(records, "architecture")] == [
        "embed",
        "alpha",
        "beta",
        "zeta",
    ]
    # Kind first, then the capability set, so like sits with like.
    by_type = [r.id for r in st.sort_models(records, "type")]
    assert by_type[0] == "alpha"  # chat, no capabilities
    assert by_type.index("beta") < by_type.index("zeta")  # plain chat before vision chat
    assert by_type[-1] == "embed"  # embedding kind sorts after chat


def test_sort_models_direction_flag_flips_the_order() -> None:
    records = [
        make_record("bravo", size=2 * GIB),
        make_record("alpha", size=8 * GIB),
        make_record("charlie", size=4 * GIB),
    ]
    assert [r.id for r in st.sort_models(records, "size", True)] == [
        "alpha",
        "charlie",
        "bravo",
    ]
    assert [r.id for r in st.sort_models(records, "size", False)] == [
        "bravo",
        "charlie",
        "alpha",
    ]


def test_sort_models_ties_break_on_id_in_both_directions() -> None:
    """The table repaints on a timer; equal rows must never swap under the cursor."""
    records = [make_record("c", size=GIB), make_record("a", size=GIB), make_record("b", size=GIB)]
    assert [r.id for r in st.sort_models(records, "size", True)] == ["a", "b", "c"]
    assert [r.id for r in st.sort_models(records, "size", False)] == ["a", "b", "c"]
    assert [r.id for r in st.sort_models(records, "date", True)] == ["a", "b", "c"]


def test_next_sort_switches_column_then_reverses() -> None:
    assert st.next_sort("date", True, "name") == ("name", False)
    assert st.next_sort("name", False, "name") == ("name", True)
    assert st.next_sort("name", True, "name") == ("name", False)
    assert st.next_sort("name", False, "size") == ("size", True)
    # A stale element id must not wedge the table.
    assert st.next_sort("name", False, "nonsense") == ("name", False)


def test_sort_direction_text_is_plain_words_not_ascending_descending() -> None:
    """ "Descending" is precise and useless; "newest first" is the real question."""
    assert st.sort_direction_text("date", True) == "newest first"
    assert st.sort_direction_text("date", False) == "oldest first"
    assert st.sort_direction_text("size", True) == "largest first"
    assert st.sort_direction_text("loaded", True) == "loaded first"
    assert st.sort_direction_text("name", False) == "A→Z"
    assert st.sort_direction_text("name", True) == "Z→A"
    # Unknown columns still produce something sayable.
    assert st.sort_direction_text("nonsense", True) == "Z→A"


def test_sort_indicator_marks_only_the_active_column() -> None:
    assert st.sort_indicator("date", "date", True) == "arrow_downward"
    assert st.sort_indicator("date", "date", False) == "arrow_upward"
    assert st.sort_indicator("name", "date", True) == ""


def test_every_sortable_column_has_a_label_and_a_tooltip() -> None:
    for column in st.MODEL_COLUMNS:
        assert column.label.strip()
        assert column.tooltip.strip()
    assert {c.key for c in st.MODEL_COLUMNS} >= {
        "name",
        "size",
        "quant",
        "architecture",
        "date",
        "type",
    }


# ---------------------------------------------------------------------------
# Unload / restart controls
# ---------------------------------------------------------------------------


def test_unload_all_prompt_names_what_is_about_to_go() -> None:
    text = st.unload_all_prompt(["qwen3-30b", "gemma4-12b"])
    assert "qwen3-30b" in text
    assert "gemma4-12b" in text
    assert "2 resident model(s)" in text
    assert "Pinned" in text  # pinned models go too, and that must not surprise
    assert "nothing to unload" in st.unload_all_prompt([])


def test_restart_backend_note_names_each_failure() -> None:
    assert "no models are loaded" in st.restart_backend_note(None)
    assert "no models are loaded" in st.restart_backend_note({"restarted": [], "failed": []})
    text = st.restart_backend_note(
        {"restarted": ["a", "b"], "failed": [{"model_id": "c", "error": "out of VRAM"}]}
    )
    assert "restarted 2 engine(s): a, b" in text
    assert "FAILED c: out of VRAM" in text


def test_restart_server_note_says_how_it_restarts() -> None:
    assert "watchdog" in st.restart_server_note({"via": "watchdog"})
    assert "tray" in st.restart_server_note({"via": "tray"})  # D28
    assert "respawning itself" in st.restart_server_note({"via": "self-respawn"})
    assert st.restart_server_note(None) == "Restart requested."


def test_restart_labels_are_not_interchangeable() -> None:
    """The two restarts are one mis-click apart; each must say what it takes down."""
    assert "stays up" in st.RESTART_ENGINES_HELP
    assert "unavailable" in st.RESTART_SERVER_WARNING
    assert "reconnect" in st.RESTART_SERVER_WARNING


# ---------------------------------------------------------------------------
# Chat: "use the loaded model"
# ---------------------------------------------------------------------------


def _ready(model_id: str, *, last_activity_at: float | None = None) -> InstanceInfo:
    return InstanceInfo(
        model_id=model_id, state="ready", port=9000, last_activity_at=last_activity_at
    )


def test_chat_target_disabled_when_nothing_is_loaded() -> None:
    """Never silently on: an unusable switch must say why it is unusable."""
    records = [make_record("a"), make_record("b")]
    target = st.chat_target(records, [], use_loaded=True, manual_choice="b")
    assert target.switch_disabled is True
    assert target.use_loaded is False
    assert target.disabled_reason == "no model is loaded"
    assert target.picker_disabled is False
    assert target.model_id == "b"
    assert target.label == "no model is loaded"


def test_chat_target_disabled_when_only_a_non_chat_model_is_loaded() -> None:
    records = [make_record("chatty"), make_record("embedder", kind="embedding", embedding=True)]
    target = st.chat_target(records, [_ready("embedder")], use_loaded=True, manual_choice="chatty")
    assert target.switch_disabled is True
    assert target.use_loaded is False
    assert "embedder" in (target.disabled_reason or "")
    assert "no chat endpoint" in (target.disabled_reason or "")
    assert target.model_id == "chatty"


def test_chat_target_with_one_loaded_model() -> None:
    records = [make_record("a"), make_record("b")]
    target = st.chat_target(records, [_ready("a")], use_loaded=True, manual_choice="b")
    assert target.switch_disabled is False
    assert target.use_loaded is True
    assert target.model_id == "a"
    assert target.loaded_id == "a"
    assert target.picker_disabled is True
    assert target.other_loaded == ()
    assert target.label == "a"
    assert target.note == ""


def test_chat_target_picks_and_names_the_most_recently_used() -> None:
    """With several loaded the choice must be visible, not arbitrary."""
    records = [make_record("a"), make_record("b"), make_record("c")]
    instances = [
        _ready("a", last_activity_at=100.0),
        _ready("b", last_activity_at=300.0),
        _ready("c", last_activity_at=200.0),
    ]
    target = st.chat_target(records, instances, use_loaded=True, manual_choice="a")
    assert target.model_id == "b"
    assert target.other_loaded == ("c", "a")
    assert "b" in target.label
    assert "most recently used of 3 loaded" in target.label
    assert "Also loaded: c, a" in target.note


def test_chat_target_off_restores_the_manual_choice() -> None:
    """Turning the switch off must not reset the picker to the first model."""
    records = [make_record("a"), make_record("b")]
    target = st.chat_target(records, [_ready("a")], use_loaded=False, manual_choice="b")
    assert target.use_loaded is False
    assert target.model_id == "b"
    assert target.picker_disabled is False
    # The switch stays operable and still names what it would target.
    assert target.switch_disabled is False
    assert target.loaded_id == "a"


def test_chat_target_ignores_instances_that_are_not_ready() -> None:
    records = [make_record("a")]
    loading = InstanceInfo(model_id="a", state="loading")
    target = st.chat_target(records, [loading], use_loaded=True, manual_choice=None)
    assert target.switch_disabled is True
    assert target.disabled_reason == "no model is loaded"


def test_chat_target_with_no_manual_choice_and_switch_off() -> None:
    target = st.chat_target([], [], use_loaded=False, manual_choice=None)
    assert target.model_id is None
    assert target.switch_disabled is True


# ---------------------------------------------------------------------------
# Benchmarking (the backend is an optional, separately built subsystem)
# ---------------------------------------------------------------------------

_MODES_PAYLOAD = {
    "modes": [
        {"key": "rtx-5090-x1", "label": "1x RTX 5090", "devices": [0], "gpu_name": "RTX 5090"},
        {
            "key": "rtx-5090-x2",
            "label": "2x RTX 5090",
            "devices": [0, 1],
            "gpu_name": "RTX 5090",
        },
        {
            "key": "rtx-3090-x1",
            "label": "1x RTX 3090",
            "devices": [2],
            "gpu_name": "RTX 3090",
            "applicable": False,
            "skipped_reason": "needs 34.0 GiB, 22.5 GiB usable",
        },
        {"key": "all", "label": "All GPUs", "devices": [0, 1, 2, 3], "gpu_name": None},
    ]
}


def test_benchmark_modes_reads_both_endpoint_shapes() -> None:
    rows = st.benchmark_modes(_MODES_PAYLOAD)
    assert [row.key for row in rows] == ["rtx-5090-x1", "rtx-5090-x2", "rtx-3090-x1", "all"]
    assert rows[0].applicable is True  # the global endpoint omits the field
    assert rows[2].applicable is False
    assert "22.5 GiB usable" in rows[2].tooltip
    assert "GPU0, GPU1" in rows[1].detail
    assert st.benchmark_modes(None) == []
    assert st.benchmark_modes({"modes": "nonsense"}) == []


def test_default_selected_modes_pre_ticks_only_the_applicable_ones() -> None:
    rows = st.benchmark_modes(_MODES_PAYLOAD)
    assert st.default_selected_modes(rows) == ["rtx-5090-x1", "rtx-5090-x2", "all"]
    assert st.benchmark_start_disabled_reason([]) is not None
    assert st.benchmark_start_disabled_reason(["all"]) is None


def test_benchmark_progress_names_what_is_running() -> None:
    job = {
        "state": "running",
        "progress": {
            "mode": "2x RTX 5090",
            "phase": "generating",
            "fraction": 0.45,
            "completed": 2,
            "total": 5,
        },
    }
    assert st.benchmark_progress_fraction(job) == 0.45
    text = st.benchmark_progress_text(job)
    assert "2 of 5" in text
    assert "2x RTX 5090" in text
    assert "generating" in text
    assert "45%" in text
    assert st.benchmark_job_state(job) == "running"
    assert st.benchmark_progress_fraction(None) == 0.0
    assert st.benchmark_progress_text({"state": "running"}) == "running"


def test_benchmark_progress_falls_back_to_completed_over_total() -> None:
    job = {"state": "running", "progress": {"completed": 1, "total": 4}}
    assert st.benchmark_progress_fraction(job) == 0.25


_REPORT = {
    "results": [
        {
            "mode": "rtx-5090-x1",
            "label": "1x RTX 5090",
            "devices": [0],
            "load_time_s": 12.5,
            "ttft_s": 0.31,
            "prompt_tokens": 512,
            "prompt_tps": 2100.0,
            "generated_tokens": 128,
            "generation_tps": 95.0,
        },
        {
            "mode": "rtx-5090-x2",
            "label": "2x RTX 5090",
            "devices": [0, 1],
            "load_time_s": 14.0,
            "ttft_s": 0.28,
            "prompt_tokens": 512,
            "prompt_tps": 3400.0,
            "generated_tokens": 128,
            "generation_tps": 78.0,
        },
        {
            "mode": "rtx-3090-x1",
            "label": "1x RTX 3090",
            "devices": [2],
            "applicable": False,
            "skipped_reason": "does not fit",
        },
        {
            "mode": "all",
            "label": "All GPUs",
            "devices": [0, 1, 2, 3],
            "error": "llama-server exited with code 1",
        },
    ]
}


def test_benchmark_result_rows_keep_the_reasons_there_are_no_numbers() -> None:
    rows = st.benchmark_result_rows(_REPORT)
    assert [row.mode for row in rows] == [
        "rtx-5090-x1",
        "rtx-5090-x2",
        "rtx-3090-x1",
        "all",
    ]
    assert rows[0].ran is True
    assert rows[2].ran is False
    assert "does not fit" in rows[2].status_text
    assert "llama-server exited" in rows[3].status_text
    assert st.benchmark_result_rows(None) == []


def test_fastest_modes_reports_generation_and_prompt_separately() -> None:
    """They routinely disagree; a single "fastest" would hide the trade-off."""
    rows = st.benchmark_result_rows(_REPORT)
    generation, prompt = st.fastest_modes(rows)
    assert generation == "rtx-5090-x1"
    assert prompt == "rtx-5090-x2"
    assert st.fastest_modes([]) == (None, None)


def test_report_best_modes_prefers_the_backends_own_winners() -> None:
    rows = st.benchmark_result_rows(_REPORT)
    stored = {**_REPORT, "best_generation_mode": "all", "best_prompt_mode": "rtx-3090-x1"}
    assert st.report_best_modes(stored, rows) == ("all", "rtx-3090-x1")
    # A report without them falls back to the derived winners.
    assert st.report_best_modes(_REPORT, rows) == ("rtx-5090-x1", "rtx-5090-x2")


def test_benchmark_speedup_text() -> None:
    rows = st.benchmark_result_rows(_REPORT)
    text = st.benchmark_speedup_text(rows)
    assert "1x RTX 5090" in text
    assert "faster" in text
    assert st.benchmark_speedup_text(rows[:1]) == ""
    close = st.benchmark_result_rows(
        {
            "results": [
                {"mode": "a", "label": "A", "generation_tps": 100.0},
                {"mode": "b", "label": "B", "generation_tps": 102.0},
            ]
        }
    )
    assert "within 5%" in st.benchmark_speedup_text(close)


def test_benchmark_history_label() -> None:
    now = _epoch(2026, 8, 14, 12)
    label = st.benchmark_history_label({"ts": now - 3600, "report": _REPORT}, now=now)
    assert "1 hour ago" in label
    assert "4 mode(s)" in label
    assert st.benchmark_history_label(None) == st.UNKNOWN


def test_benchmark_unavailable_note_is_reassuring_not_alarming() -> None:
    note = st.BENCHMARK_UNAVAILABLE_NOTE
    assert "not available" in note
    assert "affected" in note


def test_benchmark_tab_reports_unavailable_without_the_subsystem(
    monkeypatch: pytest.MonkeyPatch, config: Config
) -> None:
    """The Models tab must render identically on a build with no benchmarking."""
    from studioforge.api import mgmt_routes
    from studioforge.gui.tabs import GuiContext, benchmark

    ctx = GuiContext(config=config, api_state=_FakeState(config))
    assert benchmark.is_available(ctx) is True
    monkeypatch.delattr(mgmt_routes, "benchmark_job", raising=True)
    assert benchmark.is_available(ctx) is False
    assert benchmark.bridge(ctx) is None


# ---------------------------------------------------------------------------
# Backend capabilities panel
# ---------------------------------------------------------------------------

_CAPABILITIES = {
    "engine": {
        "tag": "b10425",
        "variant": "cuda",
        "version_string": "version: 0.1.0-dev (build 10425, commit 3d9388535)",
        "smoke_tested": True,
        "architecture_count": 142,
        "architectures": ["llama", "qwen3", "gemma4", "bert"],
        "quant_types": ["Q4_K_M", "Q8_0", "NVFP4"],
        "capability_source": "checkout",
        "capability_source_detail": "/engines/b10425/src",
    },
    "hardware": {
        "gpus": [
            {
                "index": 0,
                "name": "RTX 5090",
                "total_bytes": 32 * GIB,
                "free_bytes": 30 * GIB,
                "sm_arch": "120",
            }
        ],
        "total_vram_bytes": 32 * GIB,
        "usable_total_bytes": 28 * GIB,
        "usable_largest_bytes": 28 * GIB,
        "driver_version": "580.00",
        "cuda_driver_version": "13.3",
    },
    "features": {"vision": "mmproj projectors are loaded automatically."},
    "quant_hardware_notes": {"NVFP4": "Runs on every GPU here; accelerated on Blackwell."},
    "library": {
        "model_count": 30,
        "by_architecture": {"gemma4": 9, "llama": 7, "qwen3": 9, "bert": 0},
        "by_quant": {"Q4_K_M": 6, "Q8_0": 2},
        "capabilities": {"vision": 11, "tools": 21, "thinking": 16, "embedding": 3},
        "unsupported_by_engine": [{"model_id": "exotic-moe", "architecture": "brandnew"}],
        "sizing": {
            "fits_one_gpu": 23,
            "needs_multiple_gpus": 7,
            "too_big": 0,
            "too_big_models": [],
            "note": "Weights only; use the per-model fit check for an exact verdict.",
        },
    },
}


def test_sizing_headline_leads_with_the_users_own_models() -> None:
    assert st.sizing_headline(_CAPABILITIES) == (
        "23 of your 30 models fit on one GPU · 7 need a split · 0 too big"
    )
    assert "unavailable" in st.sizing_headline({})


def test_sizing_note_is_shown_verbatim() -> None:
    """It says this is an estimate; paraphrasing would make it sound exact."""
    assert st.sizing_note(_CAPABILITIES) == (
        "Weights only; use the per-model fit check for an exact verdict."
    )
    assert st.sizing_note(None) == ""


def test_engine_summary_names_its_reach() -> None:
    lines = st.engine_summary_lines(_CAPABILITIES)
    joined = " ".join(lines)
    assert "b10425" in joined
    assert "cuda" in joined
    assert "build 10425" in joined
    assert "142 architectures" in joined
    assert "3 quantizations" in joined
    assert "smoke-tested" in joined
    assert "No engine installed" in st.engine_summary_lines({})[0]


def test_capability_source_caveat_is_only_raised_for_a_snapshot() -> None:
    """A bundled list can disagree with the engine actually installed."""
    assert st.capability_source_caveat(_CAPABILITIES) is None
    snapshot = {
        "engine": {
            **_CAPABILITIES["engine"],
            "capability_source": "snapshot",
            "capability_source_detail": "bundled with StudioForge 0.1.0",
        }
    }
    caveat = st.capability_source_caveat(snapshot)
    assert caveat is not None
    assert "out of date" in caveat
    assert "bundled with StudioForge 0.1.0" in caveat


def test_library_chips_are_commonest_first_and_drop_empty_buckets() -> None:
    chips = st.architecture_chips(_CAPABILITIES)
    assert [chip.text for chip in chips] == ["gemma4 ×9", "qwen3 ×9", "llama ×7"]
    assert [chip.text for chip in st.quant_chips(_CAPABILITIES)] == ["Q4_K_M ×6", "Q8_0 ×2"]
    capability_chips = st.library_capability_chips(_CAPABILITIES)
    assert [chip.text for chip in capability_chips] == [
        "vision ×11",
        "tools ×21",
        "thinking ×16",
        "embedding ×3",
    ]
    assert all(chip.tooltip for chip in capability_chips)


def test_unsupported_models_are_surfaced_as_a_warning() -> None:
    """These will not load at all, which outranks everything else on the panel."""
    assert st.unsupported_models(_CAPABILITIES) == [("exotic-moe", "brandnew")]
    warning = st.unsupported_warning(_CAPABILITIES)
    assert warning is not None
    assert "cannot be loaded" in warning
    assert st.unsupported_warning({"library": {"unsupported_by_engine": []}}) is None


def test_feature_and_quant_notes_pass_through() -> None:
    assert st.feature_rows(_CAPABILITIES) == [
        ("vision", "mmproj projectors are loaded automatically.")
    ]
    notes = st.quant_hardware_notes(_CAPABILITIES)
    assert notes[0][0] == "NVFP4"
    # D9: informative, never "unsupported".
    assert "unsupported" not in notes[0][1].lower()


def test_hardware_summary_lines() -> None:
    lines = st.hardware_summary_lines(_CAPABILITIES)
    assert "RTX 5090" in lines[0]
    assert "sm_120" in lines[0]
    assert "usable after headroom" in lines[1]
    assert "580.00" in lines[2]
    assert "GPU-only" in st.hardware_summary_lines({})[0]


def test_engine_update_line_covers_every_state() -> None:
    assert "not run yet" in st.engine_update_line(None)
    assert "not run yet" in st.engine_update_line({"checked": False})
    assert "Could not check" in st.engine_update_line({"checked": True, "error": "429"})
    no_releases = {"checked": True, "current": "b1", "latest": None}
    assert "No installable release" in st.engine_update_line(no_releases)
    available = {
        "checked": True,
        "current": "b10425",
        "latest": "b10428",
        "update_available": True,
    }
    assert st.engine_update_line(available) == "Engine b10425 — b10428 is available."
    assert st.engine_update_available(available) is True
    current = {"checked": True, "current": "b10425", "latest": "b10425"}
    assert "is the latest" in st.engine_update_line(current)
    assert st.engine_update_available(current) is False


def test_engine_update_line_names_the_variant_when_the_check_knows_it() -> None:
    """ "b10488 (cuda-13.3)" vs "(source)" is a download vs a local CUDA compile."""
    prebuilt = {
        "checked": True,
        "current": "b10425",
        "latest": "b10488",
        "update_available": True,
        "latest_variant": "cuda-13.3",
    }
    assert st.engine_update_line(prebuilt) == "Engine b10425 — b10488 (cuda-13.3) is available."
    assert st.engine_update_line({**prebuilt, "latest_variant": "source"}).endswith(
        "b10488 (source) is available."
    )


def test_server_tab_update_check_delegates_to_the_engine_manager() -> None:
    """The GUI must not rebuild ``releases[0] != pinned_tag``.

    That comparison offered llama.cpp's asset-less ``v0.1.2`` prerelease as an
    engine update on 2026-08-18, behind a button that could only ever fail.
    """
    import inspect

    from studioforge.gui.tabs import server as server_tab

    source = inspect.getsource(server_tab._capabilities_panel)
    assert "check_update(limit=" in source
    assert "releases[0]" not in source
    assert "update_available" not in source  # the manager decides, not the GUI


def test_engine_update_note_explains_that_running_models_are_untouched() -> None:
    assert "keeps the build it was launched with" in st.ENGINE_UPDATE_NOTE
    assert "Restart engines" in st.ENGINE_UPDATE_NOTE


def test_filter_architectures() -> None:
    names = st.supported_architectures(_CAPABILITIES)
    assert st.filter_architectures(names, None) == ["bert", "gemma4", "llama", "qwen3"]
    assert st.filter_architectures(names, "qwen") == ["qwen3"]
    assert st.filter_architectures(names, "  ") == ["bert", "gemma4", "llama", "qwen3"]
    assert st.filter_architectures(names, "zzz") == []
    assert st.supported_quant_types(_CAPABILITIES) == ["Q4_K_M", "Q8_0", "NVFP4"]


# ---------------------------------------------------------------------------
# In-process route calls
# ---------------------------------------------------------------------------


def test_api_request_shim_exposes_only_app_state(config: Config) -> None:
    """The restart and benchmark handlers read ``request.app.state`` and nothing
    else; if that ever changes, this test is the tripwire."""
    import inspect

    from studioforge.api import admin_routes, mgmt_routes
    from studioforge.gui.tabs import GuiContext, api_request

    api_state = _FakeState(config)
    request = api_request(GuiContext(config=config, api_state=api_state))
    assert request.app.state is api_state

    for module in (admin_routes, mgmt_routes):
        source = inspect.getsource(module._state)
        assert source.strip().endswith("return request.app.state")


# ---------------------------------------------------------------------------
# Chat sampler values: an explicit 0 is a value, not "use the default"
# ---------------------------------------------------------------------------


def test_number_value_keeps_an_explicit_zero() -> None:
    """Temperature 0 (greedy decoding) must be sent as 0, not become 0.7."""
    assert st.number_value(0, 0.7) == 0.0
    assert st.number_value(0.0, 0.95) == 0.0
    assert st.number_value("0", 0.7) == 0.0


def test_number_value_defaults_only_when_genuinely_unset() -> None:
    assert st.number_value(None, 0.7) == 0.7
    assert st.number_value("", 0.95) == 0.95
    assert st.number_value("   ", 512) == 512.0
    assert st.number_value("not a number", 0.7) == 0.7
    assert st.number_value(0.35, 0.7) == 0.35
    assert st.number_value("1.5", 0.7) == 1.5


def test_chat_tab_does_not_default_samplers_with_boolean_or() -> None:
    """The ``widget.value or default`` idiom silently rewrites 0; ban it here.

    This is a static guard on the chat tab's payload construction: every
    sampler field must go through ``st.number_value`` so an explicit 0 survives.
    """
    import inspect

    from studioforge.gui.tabs import chat

    source = inspect.getsource(chat)
    for needle in ("temperature.value or", "top_p.value or", "max_tokens.value or"):
        assert needle not in source, f"chat.py regressed to boolean-or defaulting: {needle!r}"
    assert "number_value(temperature.value" in source
    assert "number_value(top_p.value" in source
    assert "number_value(max_tokens.value" in source


# ---------------------------------------------------------------------------
# Download row controls per status
# ---------------------------------------------------------------------------


def test_completed_downloads_offer_no_controls() -> None:
    """Pause/Resume/'Cancel (deletes the partial file)' next to a finished
    model read as live controls; a completed row must offer none of them."""
    assert st.download_actions("completed") == []


def test_download_actions_match_what_each_status_can_do() -> None:
    assert st.download_actions("running") == ["pause", "cancel"]
    assert st.download_actions("queued") == ["pause", "cancel"]
    assert st.download_actions("paused") == ["resume", "cancel"]
    assert st.download_actions("failed") == ["resume", "cancel"]
    assert st.download_actions("canceled") == ["resume"]


def test_download_actions_unknown_status_degrades_to_everything() -> None:
    """A future downloader state must not strand the row with no controls."""
    assert st.download_actions("verifying") == ["pause", "resume", "cancel"]


# ---------------------------------------------------------------------------
# Benchmark job terminal state
# ---------------------------------------------------------------------------


def test_benchmark_job_finished_covers_every_terminal_spelling() -> None:
    for state_name in ("succeeded", "completed", "done", "failed", "cancelled", "canceled"):
        assert st.benchmark_job_finished({"state": state_name}) is True
    for state_name in ("queued", "running", "unknown"):
        assert st.benchmark_job_finished({"state": state_name}) is False
    assert st.benchmark_job_finished(None) is False


def test_benchmark_dialog_hides_cancel_once_the_job_is_finished() -> None:
    """The tick handler must drop the Cancel button on a terminal state --
    a Cancel that outlives its job is a button that does nothing."""
    import inspect

    from studioforge.gui.tabs import benchmark

    source = inspect.getsource(benchmark)
    assert "benchmark_job_finished" in source
    assert "cancel_button.set_visibility(False)" in source


def test_format_rate_uses_a_binary_label_for_binary_maths() -> None:
    """The value is divided by 1024**2, so the label must be MiB/s -- calling
    it MB/s overstates the decimal rate by ~5%."""
    assert st.format_rate(1024 * 1024) == "1.0 MiB/s"
    assert st.format_rate(None) == st.UNKNOWN
    assert "MB/s" not in st.format_rate(50_000_000)


def test_log_line_text_does_not_double_prefix_a_rendered_structlog_line() -> None:
    """The ring buffer stores structlog's already-rendered console line; adding
    a second timestamp+level printed every line's metadata twice on the
    Dashboard live log and the Logs tab."""
    rendered = (
        "2026-08-14T13:46:30.922722Z [info     ] db.migration_applied "
        "[studioforge.db] name=001_initial.sql"
    )
    text = st.log_line_text(
        {"ts": 1755178000.0, "level": "INFO", "logger": "studioforge.db", "message": rendered}
    )
    assert text == rendered
    assert text.count("2026-08-14T") == 1
    assert "INFO " not in text  # no second stdlib-style level column


# ---------------------------------------------------------------------------
# Cross-site websocket upgrades are refused (the control channel), even with
# no API key -- a page on any website could otherwise drive the panel.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("origin", "host", "same"),
    [
        ("http://192.168.1.50:8080", "192.168.1.50:8080", True),
        ("http://rig.tailnet.ts.net:8080", "rig.tailnet.ts.net:8080", True),
        ("https://rig.tailnet.ts.net", "rig.tailnet.ts.net:8080", True),  # port ignored
        ("http://[::1]:8080", "[::1]:8080", True),
        ("http://evil.example.com", "192.168.1.50:8080", False),
        ("http://192.168.1.51:8080", "192.168.1.50:8080", False),
        ("null", "192.168.1.50:8080", False),
        (None, "192.168.1.50:8080", True),  # a non-browser client sends no Origin
        ("http://evil.example.com", None, True),  # no Host to compare against
    ],
)
def test_same_origin_websocket_rule(origin: str | None, host: str | None, same: bool) -> None:
    from studioforge.gui.app import _same_origin_websocket

    raw = []
    if origin is not None:
        raw.append((b"origin", origin.encode()))
    if host is not None:
        raw.append((b"host", host.encode()))
    assert _same_origin_websocket({"type": "websocket", "headers": raw}) is same


async def test_the_gate_closes_a_cross_site_websocket_even_without_a_key(config: Config) -> None:
    from studioforge.gui.app import GuiAuthGate

    config.server.api_key = None
    reached = {"inner": False}

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        reached["inner"] = True

    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "websocket.connect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    gate = GuiAuthGate(inner, config)
    scope = {
        "type": "websocket",
        "path": "/_nicegui_ws/socket.io/",
        "headers": [(b"origin", b"http://evil.example.com"), (b"host", b"192.168.1.50:8080")],
    }
    await gate(scope, receive, send)
    assert reached["inner"] is False
    assert sent == [{"type": "websocket.close", "code": 1008}]

    same_site = dict(
        scope, headers=[(b"origin", b"http://192.168.1.50:8080"), (b"host", b"192.168.1.50:8080")]
    )
    await gate(same_site, receive, send)
    assert reached["inner"] is True
