"""``EffectiveLaunch.gpu_only`` is a field of the dump, not only a property (D61).

The 2026-09-09 review (finding 4) found ``gpu_only`` was a bare ``@property``:
``compact()`` -- the MCP and ``/v1/models`` view -- named it explicitly, but
``model_dump()`` -- what ``GET /api/status``, ``GET /api/models`` and
``introspect`` return for ``effective`` -- dropped it, while ENGINE-FEATURES.md
listed it among the fields reported per instance. A pydantic
``@computed_field`` puts it in every dump; ``compact()`` is unchanged to the key.
"""

from __future__ import annotations

from studioforge.types import EffectiveLaunch, InstanceInfo

#: The compact view's keys, in order, as they were before D61. The MCP row
#: test in test_mcp.py asserts the values; this pins the shape.
COMPACT_KEYS = [
    "summary",
    "gpu_only",
    "policy_violations",
    "cache_prompt",
    "cache_reuse",
    "cache_ram_mib",
    "cont_batching",
    "kv_unified",
    "slot_prompt_similarity",
    "parallel",
    "ctx_per_slot",
    "spec_type",
    "inert",
]


def test_a_dump_carries_gpu_only_true_for_a_launch_studioforge_composed() -> None:
    dumped = EffectiveLaunch().model_dump()
    assert dumped["gpu_only"] is True and dumped["policy_violations"] == []
    assert EffectiveLaunch().model_dump(mode="json")["gpu_only"] is True


def test_a_dump_carries_gpu_only_false_with_the_violations_named() -> None:
    eff = EffectiveLaunch(policy_violations=["--cpu-moe", "--device none"])
    for dumped in (eff.model_dump(), eff.model_dump(mode="json")):
        assert dumped["gpu_only"] is False
        assert dumped["policy_violations"] == ["--cpu-moe", "--device none"]


def test_gpu_only_is_derived_on_every_dump_not_cached_at_construction() -> None:
    eff = EffectiveLaunch()
    assert eff.model_dump()["gpu_only"] is True
    eff.policy_violations.append("--no-kv-offload")
    assert eff.gpu_only is False and eff.model_dump()["gpu_only"] is False


def test_the_field_is_never_read_as_input() -> None:
    """A dump fed back to the constructor round-trips, and a caller cannot
    assert GPU-only-ness by naming the key: the violations are the only source."""
    eff = EffectiveLaunch(policy_violations=["--device none"], summary="x")
    assert EffectiveLaunch(**eff.model_dump()) == eff
    assert EffectiveLaunch(gpu_only=False).gpu_only is True


def test_compact_is_unchanged_to_the_key() -> None:
    """The MCP and /v1/models subset keeps its shape: the field changed what
    the full dump carries, not what the compact view says."""
    eff = EffectiveLaunch(policy_violations=["--cpu-moe"], summary="s")
    compact = eff.compact()
    assert list(compact) == COMPACT_KEYS
    assert compact["gpu_only"] is False and compact["policy_violations"] == ["--cpu-moe"]
    assert "sources" not in compact and "n_gpu_layers" not in compact


def test_an_instance_dump_carries_it_nested_as_introspect_and_status_do() -> None:
    """``introspect`` returns ``instance.model_dump(mode="json")`` and
    ``GET /api/status`` dumps ``ServerStatus.loaded``; both reach ``effective``
    through the nested model, so the field has to travel there too."""
    instance = InstanceInfo(
        model_id="chat/model",
        state="ready",
        effective=EffectiveLaunch(policy_violations=["--device none"]),
    )
    block = instance.model_dump(mode="json")["effective"]
    assert block is not None and block["gpu_only"] is False
    assert InstanceInfo(model_id="m", state="ready").model_dump()["effective"] is None
