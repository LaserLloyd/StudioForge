"""Tests for the Setup tab.

The tab itself is a renderer; everything it decides lives in
:mod:`studioforge.gui.state` as a pure function, so almost every test here is a
plain function call. The exceptions are deliberate: one render smoke test
(NiceGUI elements have a habit of failing only when actually built), one
"no secret reaches the page" assertion against the real HTML, and one
round-trip that proves a setting saved from this tab lands in ``config.yaml``
through the same route the API uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from studioforge.config import Config
from studioforge.gui import state as st
from studioforge.gui.app import create_gui_app
from studioforge.types import GpuInfo

GIB = 1024**3

API_KEY = "sf-setup-test-key-1234"
HF_TOKEN = "hf_setupfaketoken0987654321"
MCP_PIN = "11223344"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config(data_dir=tmp_path / "data")
    cfg.models.dir = tmp_path / "models"
    cfg.models.dir.mkdir(parents=True, exist_ok=True)
    (cfg.models.dir / "tiny.gguf").write_bytes(b"GGUF")
    cfg.ensure_dirs()
    cfg.save()
    return cfg


class _Registry:
    def __init__(self, records: list[Any] | None = None) -> None:
        self._records = records or []

    def all(self) -> list[Any]:
        return list(self._records)

    def resolve(self, name: str) -> Any:
        return None

    def adapters(self) -> list[Any]:
        return []


class _Supervisor:
    def list(self) -> list[Any]:
        return []

    def get(self, model_id: str) -> Any:
        return None


class _Probe:
    backend = "fake"

    def list_gpus(self) -> list[GpuInfo]:
        return [
            GpuInfo(
                index=0,
                name="RTX 5090",
                total_bytes=32 * GIB,
                free_bytes=30 * GIB,
                used_bytes=2 * GIB,
                compute_capability=(12, 0),
            ),
            GpuInfo(
                index=3,
                name="RTX 3090",
                total_bytes=24 * GIB,
                free_bytes=8 * GIB,
                used_bytes=16 * GIB,
                compute_capability=(8, 6),
            ),
        ]

    def driver_version(self) -> str:
        return "610.88"

    def cuda_driver_version(self) -> tuple[int, int]:
        return (13, 0)

    def compute_processes(self) -> list[Any]:
        return []


class _State:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.registry = _Registry()
        self.supervisor = _Supervisor()
        self.probe = _Probe()
        self.manager = None
        self.planner = None
        self.engine_manager = None
        self.downloader = None
        self.started_at = 0.0


def _ready_kwargs(**overrides: Any) -> dict[str, Any]:
    """Every check passing, so a test can knock exactly one of them down."""
    base: dict[str, Any] = {
        "data_dir": "D:/data",
        "data_dir_writable": True,
        "models_dir": "D:/models",
        "models_dir_exists": True,
        "gguf_count": 28,
        "indexed_count": 28,
        "gpu_count": 4,
        "driver_version": "610.88",
        "cuda_driver": (13, 0),
        "engine_tag": "b10425",
        "engine_smoke_tested": True,
        "pinned_tag": "b10425",
        "api_port": 1234,
        "api_reachable": True,
        "api_port_detail": "listening on 0.0.0.0:1234 (this process)",
        "mcp_pin_set": True,
        "mcp_pin_required": True,
        "hf_token_set": True,
        "autostart_enabled": True,
        "autostart_mechanism": "Windows Startup folder",
        "bind_host": "127.0.0.1",
        "api_key_set": False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Checklist logic
# ---------------------------------------------------------------------------


def test_a_fresh_checkout_fails_every_required_check() -> None:
    checks = st.first_run_checks(
        data_dir="C:/repo/data",
        data_dir_writable=False,
        models_dir=None,
        models_dir_exists=False,
        gguf_count=0,
        indexed_count=0,
        gpu_count=0,
        engine_tag=None,
        pinned_tag="b10425",
        api_port=1234,
        api_reachable=False,
        mcp_pin_set=False,
    )
    assert st.checklist_is_ready(checks) is False
    by_key = {check.key: check for check in checks}
    for key in ("data_dir", "models_dir", "models_indexed", "gpus", "engine", "api_port"):
        assert by_key[key].ok is False, key
        assert by_key[key].required is True, key
    assert by_key["engine"].action == "install-engine"
    assert "b10425" in by_key["engine"].action_label
    assert by_key["models_dir"].action == "detect-library"
    assert by_key["models_indexed"].action == "scan"
    assert by_key["mcp_pin"].action == "generate-pin"


def test_everything_configured_is_ready() -> None:
    checks = st.first_run_checks(**_ready_kwargs())
    assert st.checklist_is_ready(checks) is True
    assert st.checklist_headline(checks).startswith("Ready to serve")
    assert st.checklist_actions(checks) == []
    assert all(check.icon == "check_circle" for check in checks)


def test_optional_items_never_gate_readiness() -> None:
    """A missing HF token and no autostart are states, not problems."""
    checks = st.first_run_checks(
        **_ready_kwargs(hf_token_set=False, autostart_enabled=False, autostart_mechanism="")
    )
    assert st.checklist_is_ready(checks) is True
    by_key = {check.key: check for check in checks}
    assert by_key["hf_token"].required is False
    assert by_key["hf_token"].colour == "grey"
    assert by_key["hf_token"].status_text == "optional"
    assert by_key["autostart"].required is False
    assert "optional item(s) left" in st.checklist_headline(checks)


def test_the_pin_is_only_required_when_pairing_is_enforced() -> None:
    enforced = st.first_run_checks(**_ready_kwargs(mcp_pin_set=False, mcp_pin_required=True))
    assert st.checklist_is_ready(enforced) is False

    relaxed = st.first_run_checks(**_ready_kwargs(mcp_pin_set=False, mcp_pin_required=False))
    assert st.checklist_is_ready(relaxed) is True
    by_key = {check.key: check for check in relaxed}
    assert by_key["mcp_pin"].ok is True
    assert "API key is the credential" in by_key["mcp_pin"].detail


def test_the_headline_names_what_is_outstanding() -> None:
    checks = st.first_run_checks(**_ready_kwargs(engine_tag=None, gpu_count=0))
    headline = st.checklist_headline(checks)
    assert "2 thing(s) to fix" in headline
    assert "llama.cpp engine" in headline
    assert "GPUs" in headline


def test_gpu_detail_carries_the_driver_and_the_exclusions() -> None:
    checks = st.first_run_checks(**_ready_kwargs(excluded_devices=[3, 3, 2]))
    detail = next(check.detail for check in checks if check.key == "gpus")
    assert "driver 610.88" in detail
    assert "driver CUDA 13.0" in detail
    assert "excluded: CUDA2,3" in detail


def test_an_engine_that_was_never_smoke_tested_still_passes_but_says_so() -> None:
    checks = st.first_run_checks(**_ready_kwargs(engine_smoke_tested=False))
    detail = next(check.detail for check in checks if check.key == "engine")
    assert st.checklist_is_ready(checks) is True
    assert "never smoke-tested" in detail


# ---------------------------------------------------------------------------
# Field metadata generated from the pydantic model
# ---------------------------------------------------------------------------


def test_field_specs_cover_every_top_level_config_section() -> None:
    """ "Every setting" has to mean every section, not the ones we remembered."""
    from pydantic import BaseModel

    expected = [
        name
        for name, field in Config.model_fields.items()
        if isinstance(field.annotation, type)
        and issubclass(field.annotation, BaseModel)
        and name not in {"source_path", "data_dir"}
    ]
    specs = st.config_field_specs()
    assert st.config_sections(specs) == expected
    for section in expected:
        assert any(spec.section == section for spec in specs), section


def test_field_specs_cover_every_scalar_key_of_every_section() -> None:
    """The only keys allowed to be missing are the ones with their own widget."""
    from pydantic import BaseModel

    generated = {spec.key for spec in st.config_field_specs()}
    missing: list[str] = []
    for section_name, section_field in Config.model_fields.items():
        annotation = section_field.annotation
        if section_name in {"source_path", "data_dir"}:
            continue
        if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
            continue
        for name, info in annotation.model_fields.items():
            key = f"{section_name}.{info.alias or name}"
            if key not in generated:
                missing.append(key)
    # Both are mappings keyed by CUDA index / quant family: they get row
    # widgets on the GPU card, not a text box, and rendering them as one would
    # silently destroy the other entries.
    assert sorted(missing) == ["planner.quant_affinity", "planner.reserved_mb"]
    assert set(missing) <= st.CUSTOM_WIDGET_KEYS


def test_field_kinds_are_type_aware() -> None:
    specs = st.spec_by_key(st.config_field_specs())
    assert specs["server.port"].kind == "int"
    assert specs["planner.headroom_fraction"].kind == "float"
    assert specs["models.auto_load_pinned"].kind == "bool"
    assert specs["models.dir"].kind == "text"
    assert specs["server.cors_origins"].kind == "list"
    assert specs["planner.on_insufficient"].kind == "select"
    assert specs["planner.on_insufficient"].options == ("evict", "reject")
    assert "q8_0" in specs["models.default_kv_cache_type"].options
    # int | Literal["auto"]: no single widget fits, so free text does, and the
    # config model validates whichever half was typed.
    assert specs["models.default_parallel"].kind == "text"
    assert specs["hf.token"].kind == "secret"


def test_the_logging_json_alias_is_used_not_the_attribute_name() -> None:
    """``logging.json_logs`` is the attribute; ``logging.json`` is the key."""
    keys = {spec.key for spec in st.config_field_specs()}
    assert "logging.json" in keys
    assert "logging.json_logs" not in keys


def test_restart_required_keys_are_flagged_on_the_spec() -> None:
    specs = st.spec_by_key(st.config_field_specs())
    assert specs["server.port"].restart_required is True
    assert specs["gui.port"].restart_required is True
    assert specs["mcp.path"].restart_required is True
    assert specs["models.default_ctx"].restart_required is False
    assert "restart" in specs["server.port"].summary


def test_a_field_without_hand_written_help_still_explains_itself() -> None:
    specs = st.spec_by_key(st.config_field_specs())
    generic = specs["gateway.max_restarts"]
    assert generic.help == ""
    assert "default 3" in generic.summary
    assert "int" in generic.summary


def test_advanced_excludes_covered_keys_secrets_and_mappings() -> None:
    from studioforge.gui.tabs import setup as tab

    specs = st.config_field_specs()
    advanced = st.advanced_field_specs(specs, tab.COVERED_KEYS)
    keys = {spec.key for spec in advanced}
    for covered in tab.COVERED_KEYS:
        assert covered not in keys, covered
    for secret in st.SECRET_KEYS:
        assert secret not in keys, secret
    assert "planner.reserved_mb" not in keys
    assert "planner.quant_affinity" not in keys
    # And it is not empty: the whole point is that the long tail is reachable.
    assert "gateway.max_restarts" in keys
    assert "update.channel" in keys
    assert "watchdog.poll_interval_s" in keys


def test_every_covered_key_is_a_real_config_key() -> None:
    """A typo in COVERED_KEYS would silently hide a key from both places."""
    from studioforge.gui.tabs import setup as tab

    known = {spec.key for spec in st.config_field_specs()}
    assert set(tab.COVERED_KEYS) <= known


def test_every_key_has_exactly_one_control_on_the_tab() -> None:
    from studioforge.gui.tabs import setup as tab

    specs = st.config_field_specs()
    advanced = {spec.key for spec in st.advanced_field_specs(specs, tab.COVERED_KEYS)}
    covered = set(tab.COVERED_KEYS)
    assert covered & advanced == set()
    # Secrets are the only keys that appear in neither list, and each has a
    # hand-built masked control in its own section.
    unplaced = {spec.key for spec in specs} - covered - advanced
    assert unplaced == set(st.SECRET_KEYS) - covered
    assert unplaced == set()


# ---------------------------------------------------------------------------
# Form round-trip and secret handling
# ---------------------------------------------------------------------------


def test_redacted_config_masks_all_three_secrets(config: Config) -> None:
    config.server.api_key = API_KEY
    config.hf.token = HF_TOKEN
    config.mcp.pin = MCP_PIN
    payload = st.redacted_config(config)
    assert payload["server"]["api_key"] != API_KEY
    assert payload["hf"]["token"] != HF_TOKEN
    assert payload["mcp"]["pin"] != MCP_PIN
    assert API_KEY not in yaml.safe_dump(payload)
    assert HF_TOKEN not in yaml.safe_dump(payload)
    assert MCP_PIN not in yaml.safe_dump(payload)


def test_an_untouched_masked_secret_is_never_sent_back(config: Config) -> None:
    """The bug this prevents: overwriting a working key with its placeholder."""
    config.server.api_key = API_KEY
    config.hf.token = HF_TOKEN
    payload = st.redacted_config(config)
    specs = [spec for spec in st.config_field_specs() if spec.key in {"server.api_key", "hf.token"}]
    values = {spec.key: st.spec_display_value(payload, spec) for spec in specs}
    assert st.config_updates_from_form(specs, payload, values) == {}


def test_a_genuinely_new_secret_is_sent(config: Config) -> None:
    config.server.api_key = API_KEY
    payload = st.redacted_config(config)
    specs = [spec for spec in st.config_field_specs() if spec.key == "server.api_key"]
    updates = st.config_updates_from_form(specs, payload, {"server.api_key": "a-brand-new-key"})
    assert updates == {"server.api_key": "a-brand-new-key"}


def test_only_changed_keys_are_sent(config: Config) -> None:
    payload = st.redacted_config(config)
    specs = st.spec_by_key(st.config_field_specs())
    chosen = [specs["models.default_ctx"], specs["models.auto_load_pinned"], specs["server.host"]]
    unchanged = {spec.key: st.spec_display_value(payload, spec) for spec in chosen}
    assert st.config_updates_from_form(chosen, payload, unchanged) == {}

    changed = dict(unchanged)
    changed["models.default_ctx"] = 16384
    assert st.config_updates_from_form(chosen, payload, changed) == {"models.default_ctx": 16384}


def test_a_number_typed_into_the_parallel_field_round_trips_as_an_int() -> None:
    spec = st.spec_by_key(st.config_field_specs())["models.default_parallel"]
    assert st.spec_form_value(spec, "auto") == "auto"
    assert st.spec_form_value(spec, "4") == 4
    assert st.spec_form_value(spec, "") is None


def test_a_list_field_round_trips_through_the_comma_separated_widget(config: Config) -> None:
    payload = st.redacted_config(config)
    spec = st.spec_by_key(st.config_field_specs())["server.cors_origins"]
    assert st.spec_display_value(payload, spec) == "*"
    updates = st.config_updates_from_form(
        [spec], payload, {"server.cors_origins": "https://a.example, https://b.example"}
    )
    assert updates == {"server.cors_origins": ["https://a.example", "https://b.example"]}


def test_mask_secrets_blanks_every_credential_in_a_snippet() -> None:
    snippet = f"OPENAI_API_KEY={API_KEY}\nsfctl servers add rig http://x:1234 --api-key {MCP_PIN}"
    masked = st.mask_secrets(snippet, [API_KEY, MCP_PIN, None, ""])
    assert API_KEY not in masked
    assert MCP_PIN not in masked
    assert masked.count(st.SNIPPET_MASK) == 2
    assert "OPENAI_API_KEY=" in masked


def test_mask_secrets_masks_the_longest_secret_first() -> None:
    """A PIN that happens to be a substring of the key must not half-mask it."""
    key = "1122334455667788"
    masked = st.mask_secrets(f"key={key} pin=1122", ["1122", key])
    assert key not in masked
    assert masked == f"key={st.SNIPPET_MASK} pin={st.SNIPPET_MASK}"


def test_save_result_text_reports_the_restart_it_still_needs() -> None:
    assert st.save_result_text(None) == "nothing changed"
    assert st.save_result_text({"updated": [], "restart_required": []}) == "nothing changed"
    text = st.save_result_text(
        {"updated": ["server.port", "models.default_ctx"], "restart_required": ["server.port"]}
    )
    assert text == "saved: models.default_ctx, server.port — restart required for: server.port"


# ---------------------------------------------------------------------------
# LM Studio detection
# ---------------------------------------------------------------------------


def test_lmstudio_candidate_lines_say_why_each_one_missed() -> None:
    lines = st.lmstudio_candidate_lines(
        [
            {"path": "D:/models", "exists": True, "gguf_count": 28},
            {"path": "E:/gone", "exists": False, "gguf_count": 0},
            {"path": "C:/empty", "exists": True, "gguf_count": 0},
            {"path": "", "exists": True, "gguf_count": 1},
        ]
    )
    assert lines == [
        "D:/models — 28 GGUF file(s)",
        "E:/gone — does not exist",
        "C:/empty — no GGUF files",
    ]


def test_lmstudio_detection_note_covers_all_three_outcomes() -> None:
    assert "No LM Studio library found" in st.lmstudio_detection_note(None, "D:/models")
    assert "already the configured library" in st.lmstudio_detection_note("D:/m", "D:/m")
    assert "Save to point models.dir at it" in st.lmstudio_detection_note("D:/new", "D:/old")


def test_models_dir_status_line_reports_existence_count_and_disk() -> None:
    assert "not set" in st.models_dir_status_line(None, exists=False, gguf_count=0)
    assert "does not exist yet" in st.models_dir_status_line(
        "D:/models", exists=False, gguf_count=0
    )
    line = st.models_dir_status_line(
        "D:/models",
        exists=True,
        gguf_count=28,
        disk={
            "total_bytes": 4 * 1024**4,
            "free_bytes": 412 * GIB,
            "drive": "D:",
            "queued_bytes": 0,
        },
    )
    assert "28 GGUF file(s)" in line
    assert "412.0 GiB free on D:" in line


# ---------------------------------------------------------------------------
# GPU rows and the two multi-tenant knobs
# ---------------------------------------------------------------------------


def test_gpu_rows_join_the_probe_with_the_planner_policy() -> None:
    rows = st.gpu_setup_rows(
        _Probe().list_gpus(),
        excluded_devices=[3],
        reserved_mb={3: 8192},
        holders=[{"pid": 42, "name": "ComfyUI", "used_bytes": 6 * GIB, "gpu_indices": [3]}],
    )
    assert [row.index for row in rows] == [0, 3]
    assert rows[0].excluded is False
    assert rows[0].holders == ""
    assert rows[1].excluded is True
    assert rows[1].reserved_mb == 8192
    summary = rows[1].summary()
    assert "CUDA3" in summary
    assert "EXCLUDED" in summary
    assert "8192 MiB reserved" in summary
    assert "1 process(es) holding 6.00 GiB" in summary
    assert "8.00 GiB free of 24.00 GiB" in summary


def test_a_holder_with_no_measurable_size_still_gets_named() -> None:
    rows = st.gpu_setup_rows(
        _Probe().list_gpus()[:1],
        holders=[{"pid": 7, "name": "llama-server.exe", "used_bytes": 0, "gpu_indices": [0]}],
    )
    assert "size unavailable" in rows[0].summary()


def test_excluded_and_reserved_are_built_from_the_per_gpu_widgets() -> None:
    assert st.excluded_devices_list({0: False, 3: True, 2: True}) == [2, 3]
    assert st.excluded_devices_list({}) == []
    # Zero is "no reservation", not "reserve nothing": it must not be written.
    assert st.reserved_mb_map({0: 0, 1: "", 2: None, 3: 8192}) == {3: 8192}
    assert st.parse_reserved_mb("not a number") == 0
    assert st.parse_reserved_mb(-5) == 0
    assert st.parse_reserved_mb("2048.0") == 2048


def test_the_device_policy_note_describes_both_knobs() -> None:
    empty = st.device_policy_note([], {})
    assert "No device policy set" in empty
    set_note = st.device_policy_note([3], {2: 4096})
    assert "CUDA3" in set_note
    assert "4096 MiB held back on CUDA2" in set_note
    assert "device override still wins" in set_note


def test_the_device_recognition_note_explains_the_ordinals() -> None:
    note = st.DEVICE_RECOGNITION_NOTE
    assert "CUDA ordinal" in note
    assert "CUDA_VISIBLE_DEVICES" in note
    assert "re-probe" in note


# ---------------------------------------------------------------------------
# Engine and network wording
# ---------------------------------------------------------------------------


def test_cuda_variant_note_shows_the_driver_and_why_13_x() -> None:
    auto = st.cuda_variant_note((13, 0), "auto")
    assert "driver CUDA 13.0" in auto
    assert "sm_120" in auto
    pinned = st.cuda_variant_note((12, 4), "13.3")
    assert "pinned to CUDA 13.3" in pinned
    assert "12.4" in pinned
    assert st.UNKNOWN in st.cuda_variant_note(None, "auto")


def test_engine_install_rows_mark_the_active_build() -> None:
    rows = st.engine_install_rows(
        [
            {"tag": "b10425", "variant": "cuda-13.3", "active": True, "smoke_tested": True},
            {"tag": "b10441", "variant": "cuda-13.3", "active": False, "smoke_tested": False},
        ]
    )
    assert rows[0].startswith("★ b10425 (cuda-13.3) · smoke tested")
    assert rows[1].startswith("· b10441")
    assert "not smoke tested" in rows[1]


def test_bind_and_port_notes_say_what_is_actually_exposed() -> None:
    assert "every interface" in st.bind_note("0.0.0.0")
    assert "this machine only" in st.bind_note("127.0.0.1")
    assert "100.64.0.3" in st.bind_note("100.64.0.3")
    assert "LM Studio's default too" in st.port_conflict_note(1234)
    assert "1234" in st.port_conflict_note(1246)


def test_reachable_lines_label_each_address() -> None:
    lines = st.reachable_lines(
        [
            {"kind": "tailscale", "label": "Tailscale", "url": "http://100.64.0.3:1234"},
            {"kind": "loopback", "label": "This machine", "url": "http://127.0.0.1:1234"},
        ]
    )
    assert lines == [
        "Tailscale  http://100.64.0.3:1234",
        "This machine  http://127.0.0.1:1234",
    ]


def test_secret_state_text_never_returns_the_secret() -> None:
    assert st.secret_state_text(None) == "not set"
    assert st.secret_state_text("", unset_note="blank") == "blank"
    shown = st.secret_state_text(API_KEY)
    assert API_KEY not in shown
    assert shown.endswith("34")
    # An 8-character PIN is short enough that redact() shows nothing at all.
    assert st.secret_state_text(MCP_PIN) == "***"


# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------


def test_data_dir_source_names_which_of_the_three_rules_applied() -> None:
    assert st.data_dir_source("D:/data", "C:/repo/data").startswith("SF_DATA_DIR=")
    assert "source checkout" in st.data_dir_source(None, "C:/repo/data")
    assert "platform data directory" in st.data_dir_source(None, None)
    assert "platform data directory" in st.data_dir_source("", None)


def test_where_things_live_lists_every_path_and_no_secret(config: Config) -> None:
    config.server.api_key = API_KEY
    rows = st.where_things_live(config, source="SF_DATA_DIR=D:/data")
    names = [name for name, _ in rows]
    assert names[:6] == [
        "Data directory",
        "Config file",
        "Engines",
        "Logs",
        "Registry database",
        "Downloads (in progress)",
    ]
    joined = " ".join(value for _, value in rows)
    assert str(config.config_path) in joined
    assert API_KEY not in joined
    assert any(name == "Data directory chosen by" for name, _ in rows)


# ---------------------------------------------------------------------------
# Render: the tab has to survive a real NiceGUI build
# ---------------------------------------------------------------------------


def test_the_setup_tab_renders_every_section(config: Config) -> None:
    app = create_gui_app(config, api_state=_State(config))
    with TestClient(app) as client:
        response = client.get("/?tab=setup")
    assert response.status_code == 200
    for heading in (
        "Model library",
        "GPUs &amp; memory",
        "llama.cpp engine",
        "Network &amp; access",
        "Downloads &amp; HuggingFace",
        "Startup &amp; service",
        "Where things live",
    ):
        assert heading in response.text, heading


def test_the_setup_tab_shows_the_checklist_and_the_gpu_policy_controls(config: Config) -> None:
    config.planner.excluded_devices = [3]
    config.planner.reserved_mb = {3: 8192}
    app = create_gui_app(config, api_state=_State(config))
    with TestClient(app) as client:
        response = client.get("/?tab=setup")
    text = response.text
    assert "Data directory" in text
    assert "MCP pairing PIN" in text
    assert "never place models here" in text
    assert "reserve (MiB)" in text
    assert "CUDA3" in text
    assert "8192 MiB reserved" in text


def test_the_advanced_section_renders_the_generated_long_tail(config: Config) -> None:
    app = create_gui_app(config, api_state=_State(config))
    with TestClient(app) as client:
        response = client.get("/?tab=setup")
    assert "Advanced — every other setting" in response.text
    assert "gateway.max_restarts" in response.text
    assert "update.check_interval_h" in response.text


def test_no_secret_reaches_the_rendered_page(config: Config) -> None:
    """The whole point of the masking: assert it against the real HTML."""
    config.server.api_key = API_KEY
    config.hf.token = HF_TOKEN
    config.mcp.pin = MCP_PIN
    app = create_gui_app(config, api_state=_State(config))
    with TestClient(app) as client:
        response = client.get("/?tab=setup", headers={"Authorization": f"Bearer {API_KEY}"})
    assert response.status_code == 200
    assert API_KEY not in response.text
    assert HF_TOKEN not in response.text
    assert MCP_PIN not in response.text


def test_setup_is_a_tab_and_is_deep_linkable() -> None:
    from studioforge.gui.app import TAB_NAMES

    assert TAB_NAMES[1] == "Setup"
    assert st.DEEP_LINK_TABS["setup"] == "Setup"
    assert st.initial_tab(st.deep_link_params({"tab": "setup"})) == "Setup"


def test_a_fresh_install_lands_on_setup_and_a_working_one_on_the_dashboard(
    config: Config,
) -> None:
    from studioforge.gui.app import _default_tab
    from studioforge.gui.tabs import GuiContext

    state = _State(config)
    # No models indexed yet: the Dashboard would be four empty panels.
    assert _default_tab(GuiContext(config=config, api_state=state)) == "Setup"

    state.registry = _Registry([object()])
    state.engine_manager = None
    assert _default_tab(GuiContext(config=config, api_state=state)) == "Dashboard"

    config.models.dir = None
    assert _default_tab(GuiContext(config=config, api_state=state)) == "Setup"


# ---------------------------------------------------------------------------
# The one code path that writes a setting
# ---------------------------------------------------------------------------


async def test_a_setting_saved_from_the_tab_lands_in_config_yaml(config: Config) -> None:
    """Same route PATCH /api/config uses, so the two cannot drift."""
    from studioforge.gui.tabs import GuiContext, apply_config_updates

    ctx = GuiContext(config=config, api_state=_State(config))
    payload = await apply_config_updates(ctx, {"models.default_ctx": 16384})

    assert payload["updated"] == ["models.default_ctx"]
    assert payload["restart_required"] == []
    on_disk = yaml.safe_load(config.config_path.read_text(encoding="utf-8"))
    assert on_disk["models"]["default_ctx"] == 16384
    # And live, because the config object is shared by reference.
    assert config.models.default_ctx == 16384


async def test_a_restart_required_key_is_saved_and_flagged(config: Config) -> None:
    from studioforge.gui.tabs import GuiContext, apply_config_updates

    ctx = GuiContext(config=config, api_state=_State(config))
    payload = await apply_config_updates(ctx, {"server.port": 1246})

    assert payload["restart_required"] == ["server.port"]
    on_disk = yaml.safe_load(config.config_path.read_text(encoding="utf-8"))
    assert on_disk["server"]["port"] == 1246
    assert "restart required for: server.port" in st.save_result_text(payload)


async def test_an_empty_update_never_touches_the_file(config: Config) -> None:
    from studioforge.gui.tabs import GuiContext, apply_config_updates

    ctx = GuiContext(config=config, api_state=_State(config))
    before = config.config_path.read_text(encoding="utf-8")
    payload = await apply_config_updates(ctx, {})
    assert payload == {"updated": [], "restart_required": []}
    assert config.config_path.read_text(encoding="utf-8") == before


async def test_the_per_gpu_maps_round_trip_through_the_same_path(config: Config) -> None:
    """D19's two knobs, which had no GUI at all before this tab."""
    from studioforge.gui.tabs import GuiContext, apply_config_updates

    ctx = GuiContext(config=config, api_state=_State(config))
    await apply_config_updates(
        ctx, {"planner.excluded_devices": [3], "planner.reserved_mb": {3: 8192}}
    )
    on_disk = yaml.safe_load(config.config_path.read_text(encoding="utf-8"))
    assert on_disk["planner"]["excluded_devices"] == [3]
    # ``to_yaml_dict`` dumps in JSON mode, where a mapping key is always a
    # string. Pydantic coerces it back to the CUDA index on load, which is what
    # the reload below asserts -- the in-memory value is never a string.
    assert on_disk["planner"]["reserved_mb"] == {"3": 8192}
    assert config.planner.excluded_devices == [3]
    assert config.planner.reserved_mb == {3: 8192}

    from studioforge.config import load_config

    assert load_config(config.config_path).planner.reserved_mb == {3: 8192}


async def test_an_invalid_value_is_rejected_and_nothing_is_written(config: Config) -> None:
    from studioforge.errors import ConfigError
    from studioforge.gui.tabs import GuiContext, apply_config_updates

    ctx = GuiContext(config=config, api_state=_State(config))
    before = config.config_path.read_text(encoding="utf-8")
    with pytest.raises(ConfigError):
        await apply_config_updates(ctx, {"planner.headroom_fraction": 5.0})
    assert config.config_path.read_text(encoding="utf-8") == before
    assert config.planner.headroom_fraction == 0.10


# ---------------------------------------------------------------------------
# WP17 F4: network exposure is a checklist item
# ---------------------------------------------------------------------------


def _check(checks: list[Any], key: str) -> Any:
    return next(c for c in checks if c.key == key)


def test_loopback_bind_is_private_and_not_required() -> None:
    check = _check(st.first_run_checks(**_ready_kwargs(bind_host="127.0.0.1")), "network")
    assert check.ok is True
    assert check.required is False
    assert "this machine only" in check.detail


def test_lan_bind_without_key_is_a_required_failure_with_a_fix() -> None:
    checks = st.first_run_checks(**_ready_kwargs(bind_host="0.0.0.0", api_key_set=False))
    check = _check(checks, "network")
    assert check.ok is False
    assert check.required is True, "an open LAN port with no key gates readiness"
    assert check.action == "set-api-key"
    assert "NO API key" in check.detail and "/mcp" in check.detail
    assert st.checklist_is_ready(checks) is False


def test_lan_bind_with_key_is_green() -> None:
    check = _check(
        st.first_run_checks(**_ready_kwargs(bind_host="0.0.0.0", api_key_set=True)), "network"
    )
    assert check.ok is True
    assert "protected by server.api_key" in check.detail


@pytest.mark.parametrize("host", ["localhost", "::1", "[::1]", "127.0.0.1"])
def test_loopback_spellings(host: str) -> None:
    assert st._host_is_loopback(host) is True


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "::", "", None])
def test_non_loopback_spellings(host: Any) -> None:
    assert st._host_is_loopback(host) is False
