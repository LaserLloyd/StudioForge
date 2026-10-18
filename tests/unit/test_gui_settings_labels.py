"""The panel's field help comes from the settings schema registry first.

``studioforge.settings_schema.build_registry()`` is the one place a setting is
explained (CLI, docs, API). The generated settings forms read the same words;
``CONFIG_FIELD_HELP`` is only the fallback for keys the registry has no text for.
"""

from __future__ import annotations

import pytest

from studioforge.gui import state as st
from studioforge.settings_schema import build_registry


def test_every_described_registry_key_shows_its_description() -> None:
    specs = st.spec_by_key(st.config_field_specs())
    checked = 0
    for setting in build_registry():
        if not setting.description or setting.key not in specs:
            continue
        assert specs[setting.key].help == setting.description, setting.key
        checked += 1
    assert checked > 50  # the whole generated form, not a sample


def test_the_label_is_still_derived_from_the_key_name() -> None:
    specs = st.spec_by_key(st.config_field_specs())
    assert specs["models.default_ttl_s"].label == "default ttl s"
    assert specs["server.port"].section == "server"


def test_a_key_only_in_the_fallback_table_keeps_its_hand_written_help(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = "gateway.max_restarts"
    registry = {k: v for k, v in st.registry_descriptions().items() if k != key}
    monkeypatch.setattr(st, "registry_descriptions", lambda: registry)
    monkeypatch.setattr(st, "CONFIG_FIELD_HELP", {**st.CONFIG_FIELD_HELP, key: "hand written"})
    specs = st.spec_by_key(st.config_field_specs())
    assert specs[key].help == "hand written"
    # ...and a registry text beats the table when both exist.
    other = "server.port"
    monkeypatch.setattr(st, "CONFIG_FIELD_HELP", {**st.CONFIG_FIELD_HELP, other: "stale"})
    monkeypatch.setattr(st, "registry_descriptions", lambda: {**registry, other: "fresh"})
    assert st.spec_by_key(st.config_field_specs())[other].help == "fresh"


def test_a_broken_registry_falls_back_to_the_table(monkeypatch: pytest.MonkeyPatch) -> None:
    st.registry_descriptions.cache_clear()
    monkeypatch.setattr(
        "studioforge.settings_schema.build_registry",
        lambda: (_ for _ in ()).throw(RuntimeError("no config.example.yaml")),
    )
    try:
        assert st.registry_descriptions() == {}
        specs = st.spec_by_key(st.config_field_specs())
        assert specs["mcp.path"].help == st.CONFIG_FIELD_HELP["mcp.path"]
    finally:
        st.registry_descriptions.cache_clear()


def test_an_explicit_help_table_is_used_as_given() -> None:
    specs = st.spec_by_key(st.config_field_specs(help_text={"server.port": "mine"}))
    assert specs["server.port"].help == "mine"
    assert specs["mcp.path"].help == ""
