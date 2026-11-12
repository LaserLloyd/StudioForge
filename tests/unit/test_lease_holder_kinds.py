"""``leases.holder_kinds``: the operator's own holder families for the lease ``kind`` (D84).

The descriptive ``kind`` on every lease view and ``gpu_leased`` refusal tells a
client whether to wait (``render``/``agent``) or stand down (``benchmark``). The
built-in table only knows generic families; an operator whose programs lease
under their own names adds them in config.yaml, which is merged over it.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from studioforge.config import RESTART_REQUIRED_KEYS, Config, LeasesConfig
from studioforge.core import leases
from studioforge.core.leases import LEASE_KINDS, configure_holder_kinds, lease_kind


@pytest.fixture(autouse=True)
def _reset_configured_kinds() -> Iterator[None]:
    configure_holder_kinds(None)
    yield
    configure_holder_kinds(None)


def test_the_built_in_families_keep_their_kinds() -> None:
    assert lease_kind("benchmark:parallel") == "benchmark"
    assert lease_kind("imagegen:job-7") == "render"
    assert lease_kind("comfyui") == "render"
    assert lease_kind("openclaw-main") == "agent"
    assert lease_kind("somebody-else") == "other"


def test_a_configured_family_is_described_by_its_kind() -> None:
    configure_holder_kinds({"myrenderer": "render", "nightly": "benchmark"})
    assert lease_kind("myrenderer") == "render"
    assert lease_kind("myrenderer-2:job") == "render"
    assert lease_kind("Nightly-Sweep") == "benchmark"
    assert lease_kind("unknown") == "other"


def test_configuration_may_re_describe_a_built_in_family() -> None:
    configure_holder_kinds({"comfyui": "agent"})
    assert lease_kind("comfyui") == "agent"
    assert LEASE_KINDS["comfyui"] == "render"  # the built-in table itself is untouched


def test_reconfiguring_replaces_rather_than_accumulates() -> None:
    configure_holder_kinds({"first": "render"})
    configure_holder_kinds({"second": "agent"})
    assert lease_kind("first") == "other"
    assert lease_kind("second") == "agent"
    configure_holder_kinds({})
    assert leases._CONFIGURED_KINDS == {}


def test_an_unknown_kind_value_is_ignored_by_the_installer() -> None:
    configure_holder_kinds({"odd": "preempt-everything"})
    assert lease_kind("odd") == "other"


def test_the_config_default_is_empty_and_read_at_start() -> None:
    assert Config().leases.holder_kinds == {}
    assert "leases.holder_kinds" in RESTART_REQUIRED_KEYS


def test_the_config_lower_cases_families() -> None:
    cfg = LeasesConfig(holder_kinds={"MyRenderer": "render"})
    assert cfg.holder_kinds == {"myrenderer": "render"}


@pytest.mark.parametrize("family", ["my-renderer", "render:job", "", "-lead"])
def test_a_key_that_cannot_be_a_family_is_refused(family: str) -> None:
    """A holder's family is the part before its first '-' or ':', so a key with
    either could never match and would silently do nothing."""
    with pytest.raises(ValidationError):
        LeasesConfig(holder_kinds={family: "render"})


def test_an_unknown_kind_is_refused_by_the_config() -> None:
    with pytest.raises(ValidationError):
        LeasesConfig(holder_kinds={"myrenderer": "urgent"})  # type: ignore[dict-item]
