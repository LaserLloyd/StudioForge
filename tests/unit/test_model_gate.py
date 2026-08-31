"""D52: the local-model gate.

The gate is a *routing* primitive -- an agent asks "is what is already loaded
good enough for this job?" and routes to the local ``/v1`` API, a bigger load,
or a cloud model on the answer. Two properties are therefore load-bearing and
get most of the tests here:

* **Unknown never passes.** A model we cannot size cannot promise "at least
  20B"; a tag we cannot verify cannot promise the capability. Both fail closed,
  with a ``why_not`` that distinguishes them ("params 4.0B < 20.0B" is a
  different operational problem from "cannot verify 'uncensored'").
* **A yes carries the model id.** A yes the caller cannot act on is useless, so
  the winner's id/size/modalities/capabilities come back in the same response.

The name parser gets a matrix of its own because model ids on this rig are
dense with numbers that are *not* sizes -- ``Q4_K_M``, ``b10689``, ``BF16``,
``hb16``, ``v2.0`` -- and a parser that reads any of them as a parameter count
would hand an agent a confident wrong answer.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.api.app import build_state, create_app
from studioforge.config import Config
from studioforge.core import model_gate
from studioforge.core.model_gate import (
    SUGAR_TAGS,
    GateRequirement,
    approx_params_b,
    gate_answer,
    gguf_tags,
    modalities_from,
    parse_min_params,
    parse_size_tokens,
    parse_tags,
    tag_verdict,
)
from studioforge.types import (
    GB,
    GgufMeta,
    GpuInfo,
    InstanceInfo,
    ModelCapabilities,
    ModelRecord,
)

BIG = "ReadyArt/Dark-Scarlett-27B-v2.0-GGUF/Dark-Scarlett-27B-v2.0.i1-Q5_K_M_hb16"
SMALL = "HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M"
MOE = "Qwen3.5-122B-A10B-Q4_K_S"


# ---------------------------------------------------------------------------
# Fixtures and stubs
# ---------------------------------------------------------------------------


def make_record(
    model_id: str = BIG,
    *,
    kind: str = "chat",
    quant: str = "Q5_K_M",
    size_bytes: int = 20 * GB,
    param_count: int | None = None,
    vision: bool = False,
    tools: bool = False,
    thinking: bool = False,
    meta: GgufMeta | None = None,
    path: Path | None = None,
) -> ModelRecord:
    if meta is None and param_count is not None:
        meta = GgufMeta(quant_label=quant, param_count=param_count)
    return ModelRecord(
        id=model_id,
        name=model_id.rsplit("/", 1)[-1],
        kind=kind,  # type: ignore[arg-type]
        path=path or Path("/models/thing.gguf"),
        size_bytes=size_bytes,
        quant=quant,
        architecture="llama",
        capabilities=ModelCapabilities(vision=vision, tools=tools, thinking=thinking),
        meta=meta,
    )


def instance(model_id: str = BIG, *, last_activity_at: float | None = None) -> InstanceInfo:
    return InstanceInfo(
        model_id=model_id, state="ready", port=18100, last_activity_at=last_activity_at
    )


class FakeRegistry:
    def __init__(self, records: list[ModelRecord]) -> None:
        self._records = records

    def all(self) -> list[ModelRecord]:
        return list(self._records)

    def get(self, model_id: str) -> ModelRecord | None:
        return next((r for r in self._records if r.id == model_id), None)

    def resolve(self, name: str) -> ModelRecord | None:
        return self.get(name)

    def known_ids(self) -> list[str]:
        return [r.id for r in self._records]

    def openai_list(self) -> list[dict[str, Any]]:
        return [r.openai_dict() for r in self._records]

    def scan(self, *, force: bool = False) -> Any:  # pragma: no cover - lifespan only
        raise RuntimeError("not used")


class FakeSupervisor:
    """Only what the gate touches: ``list()``, plus ``get``/``props``/``slots``
    so the *real* ``Manager.introspect`` can run end to end in the route tests."""

    def __init__(self, instances: list[InstanceInfo], modalities: Any = None) -> None:
        self._instances = instances
        self._modalities = modalities

    def list(self) -> list[InstanceInfo]:
        return list(self._instances)

    def get(self, model_id: str) -> InstanceInfo | None:
        return next((i for i in self._instances if i.model_id == model_id), None)

    async def props(self, model_id: str) -> dict[str, Any] | None:
        return {"modalities": self._modalities} if self._modalities is not None else None

    async def slots(self, model_id: str) -> list[dict[str, Any]]:
        return []


def introspection(modalities: Any) -> dict[str, Any]:
    return {"loaded": True, "actual": {"modalities": modalities}}


def answer(
    requirement: GateRequirement,
    records: list[ModelRecord],
    instances: list[InstanceInfo],
    *,
    live: dict[str, Any] | None = None,
    raises: bool = False,
) -> dict[str, Any]:
    """Run the gate against stubs, synchronously."""

    async def introspect(model_id: str) -> dict[str, Any] | None:
        if raises:
            raise RuntimeError("child is wedged")
        return (live or {}).get(model_id)

    return asyncio.run(
        gate_answer(
            requirement,
            registry=FakeRegistry(records),
            supervisor=FakeSupervisor(instances),
            introspect=introspect,
        )
    )


# ---------------------------------------------------------------------------
# parse_min_params
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        (20, 20.0),
        (1.5, 1.5),
        # Zero is "no bar", not a literal bar: a config default of 0 must not
        # refuse an unsized model with "cannot prove >= 0.0B" (post-review, D52).
        (0, None),
        (0.0, None),
        ("0", None),
        ("0b", None),
        ("0m", None),
        ("20", 20.0),
        ("20b", 20.0),
        ("20B", 20.0),
        ("0.5b", 0.5),
        ("500m", 0.5),
        ("500M", 0.5),
        ("7000m", 7.0),
        ("  20  ", 20.0),
        ("20 B", 20.0),
    ],
)
def test_parse_min_params_reads_the_shapes_a_caller_would_write(
    value: Any, expected: float | None
) -> None:
    assert parse_min_params(value) == expected


@pytest.mark.parametrize("value", ["big", "20g", "abc", "-5", "20b30", "b20", True])
def test_parse_min_params_rejects_junk_and_says_what_it_wanted(value: Any) -> None:
    """The message is the whole point: the route turns it into a 400 body, and a
    400 that does not name the accepted shapes leaves an agent author guessing."""
    with pytest.raises(ValueError, match="min_params"):
        parse_min_params(value)


def test_a_bare_number_means_billions_not_a_raw_count() -> None:
    """The one implicit behaviour in the module, pinned so it cannot drift.

    Nobody asks for a 7,000,000,000-parameter model; they ask for 7B.
    """
    assert parse_min_params(20) == 20.0
    assert parse_min_params("20") == 20.0


# ---------------------------------------------------------------------------
# Name parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        (BIG, (27.0, None)),
        (SMALL, (4.0, None)),  # E4B: an "effective" size is still a total
        (MOE, (122.0, 10.0)),  # MoE: total and routed-active in one id
        ("SmolLM2-1.7B-Instruct", (1.7, None)),  # decimals survive
        ("mistralai/Mixtral-8x7B-Instruct-v0.1", (56.0, None)),  # 8x7B multiplies out
        ("Qwen3-30B-A3B-Instruct-2507-Q8_0", (30.0, 3.0)),
        ("bartowski/gemma-3-27b-it-Q4_K_M", (27.0, None)),  # lowercase b
        ("vendor/nothing-here-Q4_K_M", (None, None)),  # no size token at all
    ],
)
def test_name_parse_matrix(model_id: str, expected: tuple[float | None, float | None]) -> None:
    assert parse_size_tokens(model_id) == expected


@pytest.mark.parametrize(
    "model_id",
    [
        "vendor/model-Q8_0",
        "vendor/model-Q4_K_M",
        "vendor/model-hb16",
        "vendor/model-b10689",
        "vendor/model-BF16",
        "vendor/model-f16",
        "vendor/model-16",
        "vendor/model-v2.0",
        "vendor/model-b10689-hb16-f16-v2.0-16-Q4_K_M",
    ],
)
def test_quant_and_engine_tokens_are_never_read_as_a_size(model_id: str) -> None:
    """Model ids are dense with digits that are not parameter counts.

    Reading ``b10689`` (an engine build) or ``BF16`` (a tensor type) as a size
    would hand an agent a confident wrong answer, which is worse than "unknown"
    -- unknown at least fails closed.
    """
    assert parse_size_tokens(model_id) == (None, None)


# ---------------------------------------------------------------------------
# approx_params_b: the three layers
# ---------------------------------------------------------------------------


def test_exact_metadata_wins_over_the_name() -> None:
    """``general.parameter_count`` is already on GgufMeta, so the exact path is free."""
    record = make_record("vendor/mislabelled-70B-Q4_K_M", param_count=6_800_000_000)
    assert approx_params_b(record) == (6.8, None, "metadata")


def test_a_live_count_from_the_child_outranks_everything() -> None:
    record = make_record(BIG, param_count=27_000_000_000)
    assert approx_params_b(record, live_params=26_500_000_000) == (26.5, None, "metadata")


def test_the_name_answers_when_the_gguf_declares_no_count() -> None:
    assert approx_params_b(make_record(BIG)) == (27.0, None, "name")


def test_metadata_reports_the_moe_active_count() -> None:
    """MoE active params come from throughput's dense-trunk model, which is the
    same number the catalog renders -- two surfaces disagreeing about one
    model's active size would read as a bug."""
    meta = GgufMeta(
        architecture="qwen3moe",
        n_layer=48,
        n_embd=3072,
        n_head=32,
        n_head_kv=4,
        n_embd_head_k=128,
        n_embd_head_v=128,
        n_vocab=151936,
        n_expert=256,
        n_expert_used=8,
        param_count=122_000_000_000,
        quant_label="Q4_K_S",
    )
    total, active, source = approx_params_b(make_record(MOE, meta=meta))
    assert (total, source) == (122.0, "metadata")
    assert active is not None and 0 < active < total


def test_a_dense_model_reports_no_active_count() -> None:
    """``active_params_b`` means "this is an MoE and here is its routed count".

    ``None`` for dense keeps that meaning identical across all three sources --
    the name parser can only ever say it via an ``A10B`` token.
    """
    assert approx_params_b(make_record(BIG, param_count=27_000_000_000))[1] is None


def test_file_size_estimates_a_model_with_no_count_and_no_size_token() -> None:
    record = make_record("vendor/mystery-Q4_K_M", quant="Q4_K_M", size_bytes=8 * GB)
    total, _, source = approx_params_b(record)
    assert source == "estimated"
    assert total is not None and 13.0 < total < 15.0  # 8 GiB / 4.85 bpw


def test_an_unknown_quant_still_estimates_and_still_says_estimated() -> None:
    """A label we do not recognise is not a reason to refuse a number -- the
    whole layer is explicitly approximate -- but the source must stay honest."""
    record = make_record("vendor/mystery-XYZ9", quant="unknown", size_bytes=8 * GB)
    total, _, source = approx_params_b(record)
    assert source == "estimated"
    assert total is not None and total > 0


def test_a_model_with_nothing_to_go_on_is_unknown() -> None:
    record = make_record("vendor/mystery", quant="unknown", size_bytes=0)
    assert approx_params_b(record) == (None, None, "unknown")


def test_an_instance_whose_record_vanished_is_still_sized_from_its_name() -> None:
    """A file deleted under a running child must not make the model unsizeable
    -- unknown fails every bar, so the gate would start refusing a model that is
    loaded and working."""
    assert approx_params_b(BIG) == (27.0, None, "name")
    assert approx_params_b(None) == (None, None, "unknown")


# ---------------------------------------------------------------------------
# Modalities
# ---------------------------------------------------------------------------


def test_modalities_distinguishes_silence_from_text_only() -> None:
    """``None`` and ``[]`` are the same pixel on the Dashboard and opposite
    verdicts to the gate: silence is "unknown" and fails a bar."""
    assert modalities_from(None) is None
    assert modalities_from({"loaded": True, "actual": {}}) is None
    assert modalities_from(introspection({})) == []
    assert modalities_from(introspection({"vision": True, "audio": False})) == ["vision"]
    assert modalities_from(introspection(["vision", "audio"])) == ["vision", "audio"]


def test_the_dashboard_renders_the_gates_own_derivation() -> None:
    """Pins the reuse: if these drift, the screen and the gate disagree about
    whether a running child accepts images."""
    from studioforge.gui import state as st

    assert st.modalities_text(introspection({"vision": True, "audio": False})) == "vision"
    assert st.modalities_text(introspection({})) == "text only"
    assert st.modalities_text(None) == st.UNKNOWN


# ---------------------------------------------------------------------------
# Tag verdicts: the tri-state
# ---------------------------------------------------------------------------


def verdict(tag: str, **kwargs: Any) -> str:
    kwargs.setdefault("record", None)
    kwargs.setdefault("model_id", "vendor/plain-7B")
    kwargs.setdefault("modalities", None)
    kwargs.setdefault("file_tags", frozenset())
    return tag_verdict(tag, **kwargs)


def test_vision_answers_from_the_record_then_from_the_live_child() -> None:
    assert verdict("vision", record=make_record(vision=True)) == "yes"
    # A projector the record predates: the running child is the truthful source.
    assert verdict("vision", record=make_record(vision=False), modalities=["vision"]) == "yes"
    # capabilities IS an answer -- it was derived from the file at scan time.
    assert verdict("vision", record=make_record(vision=False)) == "no"
    assert verdict("vision", record=make_record(vision=False), modalities=[]) == "no"


def test_tools_and_thinking_come_from_the_scanned_capabilities() -> None:
    assert verdict("tools", record=make_record(tools=True)) == "yes"
    assert verdict("tools", record=make_record(tools=False)) == "no"
    assert verdict("thinking", record=make_record(thinking=True)) == "yes"
    assert verdict("thinking", record=make_record(thinking=False)) == "no"


def test_audio_and_video_are_unknown_until_a_child_answers() -> None:
    """No GGUF field and no ModelCapabilities member describes them, so with no
    live answer the honest verdict is ignorance -- which fails the bar."""
    assert verdict("audio", record=make_record()) == "unknown"
    assert verdict("audio", record=make_record(), modalities=["audio"]) == "yes"
    assert verdict("audio", record=make_record(), modalities=["vision"]) == "no"
    assert verdict("video", record=make_record()) == "unknown"


def test_uncensored_is_proven_by_a_name_token_but_never_disproven() -> None:
    """Absence of "uncensored" in a name does not prove a model is censored --
    most names simply do not describe their alignment. So the miss is
    ``unknown`` (which still fails the bar), not a confident ``no``."""
    assert verdict("uncensored", model_id=SMALL) == "yes"
    assert verdict("uncensored", model_id="vendor/Llama-3-8B-abliterated") == "yes"
    assert verdict("uncensored", model_id="vendor/Llama-3-8B-Instruct") == "unknown"


def test_uncensored_is_also_proven_by_the_gguf_general_tags() -> None:
    assert verdict("uncensored", file_tags=frozenset({"abliterated"})) == "yes"
    assert verdict("uncensored", file_tags=frozenset({"chat"})) == "unknown"


def test_curated_tags_read_both_name_tokens_and_file_tags() -> None:
    assert verdict("coding", model_id="Qwen2.5-Coder-32B-Instruct") == "yes"
    # Post-review (D52): a card tag is a claim, not an identity. A real
    # creative-writing merge in this library carries "coding"/"math"/"stem" in
    # its general.tags, so file tags alone can never prove coding -- only a
    # name token ("-Coder-") can. Identity tags (roleplay, uncensored) keep the
    # file path: tagging a card "roleplay" is the author describing the
    # finetune itself.
    assert verdict("coding", file_tags=frozenset({"code"})) == "unknown"
    assert verdict("coding", file_tags=frozenset({"coding"})) == "unknown"
    assert verdict("roleplay", model_id="vendor/Something-RP-12B") == "yes"
    assert verdict("roleplay", file_tags=frozenset({"roleplay"})) == "yes"
    assert verdict("roleplay", file_tags=frozenset({"roleplaying"})) == "yes"


def test_a_live_child_that_answered_outranks_a_stale_vision_capability() -> None:
    """Post-review (D52): the record says what the FILE can do, the child says
    what this PROCESS was launched able to do. A projector deleted after the
    scan -- or a multimodal launched without ``--mmproj`` -- is record-yes /
    live-no, and routing an image there fails. When the child answered its
    modalities, its answer decides vision in both directions."""
    assert verdict("vision", record=make_record(vision=True), modalities=[]) == "no"
    assert verdict("vision", record=make_record(vision=True), modalities=["audio"]) == "no"
    assert verdict("vision", record=make_record(vision=True), modalities=["vision"]) == "yes"
    # A silent child leaves the scan as the best available answer.
    assert verdict("vision", record=make_record(vision=True)) == "yes"


def test_a_supervisor_that_cannot_be_listed_is_a_refusal_not_an_exception() -> None:
    """Post-review (D52): every other failure path degrades; the instance-table
    read must too, or the REST surface 500s where the MCP surface would wrap."""

    class ExplodingSupervisor:
        def list(self) -> Any:
            raise RuntimeError("dictionary changed size during iteration")

    async def introspect(_model_id: str) -> dict[str, Any] | None:
        return None

    answer = asyncio.run(
        gate_answer(
            GateRequirement(min_params_b=1.0),
            registry=FakeRegistry({}),
            supervisor=ExplodingSupervisor(),
            introspect=introspect,
        )
    )
    assert answer["answer"] == "no"
    assert "could not be read" in answer["reason"]
    assert verdict("roleplay", model_id="vendor/Llama-3-8B") == "unknown"


def test_an_arbitrary_tag_falls_through_to_the_generic_matcher() -> None:
    """The extensibility guarantee: a tag nobody anticipated gets a useful
    answer with zero code change, which is what keeps the feature from needing a
    release per tag."""
    assert verdict("qwen", model_id=f"Qwen/{MOE}") == "yes"  # the publisher is a token
    assert verdict("vietnamese", file_tags=frozenset({"vietnamese"})) == "yes"
    assert verdict("vietnamese", model_id="vendor/Llama-3-8B") == "unknown"
    # Whole tokens only, and that cuts both ways: "qwen" does not match the
    # token "qwen3". Unknown rather than yes is the safe direction -- a
    # substring match would fire the generic matcher on almost anything.
    assert verdict("qwen", model_id=MOE) == "unknown"


def test_a_tag_matches_whole_tokens_only() -> None:
    """ "code" must not match "decoder"; a substring match would make the generic
    matcher fire on almost anything."""
    assert verdict("code", model_id="vendor/decoder-only-7B") == "unknown"


# ---------------------------------------------------------------------------
# gguf_tags: the lazy header read
# ---------------------------------------------------------------------------


def gguf_file(tmp_path: Path) -> Path:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF-not-really")
    return path


def test_general_tags_are_read_once_per_path_and_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gate call is a routing primitive an agent may hit before every task, so
    re-parsing a header per call would turn a cheap question into disk I/O."""
    model_gate._FILE_TAGS_CACHE.clear()
    path = gguf_file(tmp_path)
    calls: list[Path] = []

    class FakeGguf:
        kv = {"general.tags": ["Uncensored", "RolePlay", ""]}

    def fake_read(p: Path, **kwargs: Any) -> Any:
        calls.append(p)
        return FakeGguf()

    monkeypatch.setattr("studioforge.core.gguf.read_gguf", fake_read)

    assert gguf_tags(path) == frozenset({"uncensored", "roleplay"})
    assert gguf_tags(path) == frozenset({"uncensored", "roleplay"})
    assert len(calls) == 1, "the cache must not re-read an unchanged file"

    # A replaced file re-reads itself: the key is (path, mtime), so there is no
    # invalidation logic to get wrong.
    import os

    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 100))
    assert gguf_tags(path) == frozenset({"uncensored", "roleplay"})
    assert len(calls) == 2


def test_a_broken_header_yields_no_tags_rather_than_a_failed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gate that 500s because a model file is mid-replacement would be worse
    than useless to the agent depending on it for routing."""
    model_gate._FILE_TAGS_CACHE.clear()
    path = gguf_file(tmp_path)

    def boom(p: Path, **kwargs: Any) -> Any:
        raise OSError("truncated")

    monkeypatch.setattr("studioforge.core.gguf.read_gguf", boom)
    assert gguf_tags(path) == frozenset()
    assert gguf_tags(None) == frozenset()
    assert gguf_tags(tmp_path / "missing.gguf") == frozenset()


# ---------------------------------------------------------------------------
# parse_tags
# ---------------------------------------------------------------------------


def test_parse_tags_normalises_and_drops_empties_silently() -> None:
    assert parse_tags("Vision, CODING ,,roleplay,vision") == frozenset(
        {"vision", "coding", "roleplay"}
    )
    assert parse_tags(None) == frozenset()
    assert parse_tags(",,,") == frozenset(), "a trailing comma is not an error"


def test_parse_tags_folds_the_sugar_flags_into_the_same_set() -> None:
    assert parse_tags("coding", extra=["vision", "tools"]) == frozenset(
        {"coding", "vision", "tools"}
    )


def test_parse_tags_rejects_a_value_that_cannot_be_a_tag() -> None:
    with pytest.raises(ValueError, match="usable tag name"):
        parse_tags("x" * 200)
    with pytest.raises(ValueError, match="usable tag name"):
        parse_tags("bad\x00tag")


# ---------------------------------------------------------------------------
# gate_answer: selection and refusal
# ---------------------------------------------------------------------------


def test_an_empty_box_is_a_no_with_a_reason_and_a_hint() -> None:
    body = answer(GateRequirement(min_params_b=20.0), [], [])
    assert (body["ok"], body["answer"], body["model"]) == (False, "no", None)
    assert body["reason"] == "nothing is loaded"
    assert "load-recommended" in (body["hint"] or "")
    assert body["instances"] == []


def test_no_bar_at_all_is_yes_when_anything_chat_capable_is_loaded() -> None:
    body = answer(GateRequirement(), [make_record()], [instance()])
    assert body["answer"] == "yes"
    assert body["model"] == BIG
    assert body["hint"] is None and body["reason"] is None


def test_a_model_below_the_bar_is_a_no_that_names_the_gap() -> None:
    body = answer(GateRequirement(min_params_b=20.0), [make_record(SMALL)], [instance(SMALL)])
    assert body["answer"] == "no"
    assert body["reason"] == "largest loaded model is 4B, below the 20B bar"
    assert body["instances"][0]["why_not"] == ["params 4.0B < 20.0B"]


def test_an_unsizeable_model_fails_a_bar_it_cannot_be_proven_to_meet() -> None:
    """The gate's promise is "at least this big". A model we cannot size cannot
    make that promise, so unknown fails -- one wasted load beats an agent
    sending 30B work to something that turned out to be 1B."""
    record = make_record("vendor/mystery", quant="unknown", size_bytes=0)
    body = answer(GateRequirement(min_params_b=20.0), [record], [instance("vendor/mystery")])
    assert body["answer"] == "no"
    assert body["instances"][0]["why_not"] == ["unknown size, cannot prove >= 20.0B"]
    assert "no loaded model reports a size" in body["reason"]


def test_the_largest_qualifying_model_wins() -> None:
    records = [make_record(BIG), make_record(SMALL), make_record(MOE)]
    instances = [instance(BIG), instance(SMALL), instance(MOE)]
    body = answer(GateRequirement(min_params_b=1.0), records, instances)
    assert body["model"] == MOE
    assert body["params_b"] == 122.0
    assert body["active_params_b"] == 10.0
    assert body["params_source"] == "name"


def test_a_tie_on_size_goes_to_the_most_recently_active() -> None:
    """Both are equally good answers to "is something big enough loaded", so the
    tie breaks towards the one whose weights and prompt cache are warm."""
    other = BIG.replace("Dark-Scarlett", "Pale-Scarlett")
    records = [make_record(BIG), make_record(other)]
    instances = [
        instance(BIG, last_activity_at=100.0),
        instance(other, last_activity_at=900.0),
    ]
    assert answer(GateRequirement(min_params_b=1.0), records, instances)["model"] == other


def test_an_unsized_model_sorts_last_when_there_is_no_bar() -> None:
    records = [make_record("vendor/mystery", quant="unknown", size_bytes=0), make_record(SMALL)]
    instances = [instance("vendor/mystery"), instance(SMALL)]
    assert answer(GateRequirement(), records, instances)["model"] == SMALL


def test_an_embedding_model_is_never_a_candidate() -> None:
    """It is loaded, healthy, and completely unable to serve a chat request:
    counting it would produce a yes that breaks the caller's very next call."""
    record = make_record("vendor/bge-m3-Q8_0", kind="embedding")
    body = answer(GateRequirement(), [record], [instance("vendor/bge-m3-Q8_0")])
    assert body["answer"] == "no"
    assert body["reason"] == "only an embedding model is loaded, and an embedding model cannot chat"
    assert body["instances"] == []


def test_asking_for_the_embedding_tag_is_refused_with_a_pointer() -> None:
    body = answer(GateRequirement(tags=frozenset({"embedding"})), [make_record()], [instance()])
    assert body["answer"] == "no"
    assert "/v1/embeddings" in body["reason"]


def test_only_ready_instances_count() -> None:
    loading = InstanceInfo(model_id=BIG, state="loading", port=18100)
    assert answer(GateRequirement(), [make_record()], [loading])["answer"] == "no"


# ---------------------------------------------------------------------------
# gate_answer: modalities and tags end to end
# ---------------------------------------------------------------------------


def test_vision_from_live_introspection() -> None:
    record = make_record(BIG, vision=False)
    body = answer(
        GateRequirement(tags=frozenset({"vision"})),
        [record],
        [instance()],
        live={BIG: introspection({"vision": True})},
    )
    assert body["answer"] == "yes"
    assert body["modalities"] == ["vision"]
    assert body["instances"][0]["tags"] == {"vision": "yes"}


def test_vision_falls_back_to_the_scanned_record_when_the_child_is_silent() -> None:
    body = answer(
        GateRequirement(tags=frozenset({"vision"})), [make_record(vision=True)], [instance()]
    )
    assert body["answer"] == "yes"
    assert body["modalities"] == [], "no live answer means nothing to report as modalities"


def test_a_wedged_child_falls_back_to_the_record_rather_than_failing() -> None:
    """``introspect`` talks HTTP to a child that may be mid-restart. Failing the
    *routing* decision over a *reporting* problem would be the wrong trade."""
    body = answer(
        GateRequirement(tags=frozenset({"vision"})),
        [make_record(vision=True)],
        [instance()],
        raises=True,
    )
    assert body["answer"] == "yes"


def test_no_loaded_model_reports_vision() -> None:
    body = answer(GateRequirement(tags=frozenset({"vision"})), [make_record()], [instance()])
    assert body["answer"] == "no"
    assert body["reason"] == "no loaded model reports vision"
    assert body["instances"][0]["why_not"] == ["no vision"]


def test_audio_is_unverifiable_without_a_live_answer_and_says_so() -> None:
    body = answer(GateRequirement(tags=frozenset({"audio"})), [make_record()], [instance()])
    assert body["answer"] == "no"
    assert body["instances"][0]["why_not"] == ["cannot verify 'audio'"]
    assert body["reason"] == "no loaded model can be verified as 'audio'"


def test_audio_passes_when_the_child_reports_it() -> None:
    body = answer(
        GateRequirement(tags=frozenset({"audio"})),
        [make_record()],
        [instance()],
        live={BIG: introspection({"audio": True})},
    )
    assert body["answer"] == "yes"


def test_an_unverifiable_tag_fails_the_bar_the_same_way_an_unknown_size_does() -> None:
    body = answer(
        GateRequirement(tags=frozenset({"uncensored"})), [make_record(BIG)], [instance(BIG)]
    )
    assert body["answer"] == "no"
    assert body["instances"][0]["why_not"] == ["cannot verify 'uncensored'"]


def test_uncensored_passes_off_the_model_id_alone() -> None:
    body = answer(
        GateRequirement(tags=frozenset({"uncensored"})), [make_record(SMALL)], [instance(SMALL)]
    )
    assert body["answer"] == "yes"
    assert body["model"] == SMALL


def test_the_size_gap_is_measured_over_the_models_that_cleared_the_tags() -> None:
    """A 27B that cannot do vision is not "the largest loaded model" for a
    caller who needs vision -- quoting it would send them chasing the wrong gap.
    """
    records = [make_record(BIG, vision=False), make_record(SMALL, vision=True)]
    instances = [instance(BIG), instance(SMALL)]
    body = answer(
        GateRequirement(min_params_b=20.0, tags=frozenset({"vision"})), records, instances
    )
    assert body["reason"] == "largest loaded model is 4B, below the 20B bar"


def test_the_winner_carries_its_capabilities() -> None:
    """Free, and it answers the next question the agent asks."""
    body = answer(GateRequirement(), [make_record(tools=True, thinking=True)], [instance()])
    assert body["capabilities"]["tools"] is True
    assert body["capabilities"]["thinking"] is True
    assert body["capabilities"]["vision"] is False


# ---------------------------------------------------------------------------
# Response shape -- the MCP tool and agent clients build against these keys
# ---------------------------------------------------------------------------

RESPONSE_KEYS = {
    "ok",
    "answer",
    "model",
    "params_b",
    "active_params_b",
    "params_source",
    "modalities",
    "capabilities",
    "checked",
    "instances",
    "reason",
    "hint",
}
ROW_KEYS = {
    "model",
    "params_b",
    "active_params_b",
    "params_source",
    "modalities",
    "tags",
    "meets",
    "why_not",
}


def test_the_response_shape_is_identical_on_yes_and_no() -> None:
    yes = answer(GateRequirement(), [make_record()], [instance()])
    no = answer(GateRequirement(min_params_b=999.0), [make_record()], [instance()])
    assert set(yes) == RESPONSE_KEYS
    assert set(no) == RESPONSE_KEYS
    assert set(yes["instances"][0]) == ROW_KEYS
    assert set(no["instances"][0]) == ROW_KEYS
    assert yes["checked"] == {"min_params_b": None, "tags": []}
    assert no["checked"] == {"min_params_b": 999.0, "tags": []}


def test_a_no_empties_every_winner_field_together() -> None:
    """A populated ``params_b`` beside ``model: null`` would make a client
    wonder whether it described something usable. There is no winner; there is
    nothing to describe."""
    no = answer(GateRequirement(min_params_b=999.0), [make_record()], [instance()])
    assert no["model"] is None
    assert no["params_b"] is None
    assert no["active_params_b"] is None
    assert no["params_source"] is None
    assert no["modalities"] == []
    assert no["capabilities"] is None


def test_the_answer_is_a_plain_yes_or_no_string() -> None:
    """Weak agent clients pattern-match this rather than reading ``ok``."""
    assert answer(GateRequirement(), [make_record()], [instance()])["answer"] == "yes"
    assert answer(GateRequirement(), [], [])["answer"] == "no"


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


class FakeProbe:
    backend = "fake"

    def available(self) -> bool:
        return True

    def list_gpus(self) -> list[GpuInfo]:
        return [
            GpuInfo(
                index=0,
                name="NVIDIA GeForce RTX 5090",
                total_bytes=32 * GB,
                free_bytes=30 * GB,
                used_bytes=2 * GB,
                compute_capability=(12, 0),
            )
        ]

    def get_gpu(self, index: int) -> GpuInfo | None:
        return self.list_gpus()[0] if index == 0 else None

    def compute_processes(self) -> list[Any]:
        return []

    def driver_version(self) -> str | None:
        return None

    def cuda_driver_version(self) -> tuple[int, int] | None:
        return None

    def shutdown(self) -> None:
        return None


@pytest.fixture()
def app(tmp_path: Path) -> Any:
    config = Config(
        data_dir=tmp_path / "data",
        server={"host": "127.0.0.1", "port": 1234},
        models={"dir": tmp_path / "models"},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
    )
    built = create_app(config, state=build_state(config), start_background=False)
    built.state.registry = FakeRegistry([make_record(BIG), make_record(SMALL)])
    built.state.probe = FakeProbe()
    built.state.planner.probe = FakeProbe()
    built.state.manager.registry = built.state.registry
    supervisor = FakeSupervisor([instance(BIG), instance(SMALL)], modalities={"vision": True})
    built.state.supervisor = supervisor
    built.state.manager.supervisor = supervisor
    return built


def test_route_answers_yes_with_the_id_to_use(app: Any) -> None:
    with TestClient(app) as http:
        response = http.get("/api/model-gate", params={"min_params": "20b"})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["answer"] == "yes"
    assert body["model"] == BIG
    assert body["params_b"] == 27.0
    assert set(body) == RESPONSE_KEYS


def test_route_answers_no_with_a_reason_when_the_bar_is_too_high(app: Any) -> None:
    with TestClient(app) as http:
        body = http.get("/api/model-gate", params={"min_params": "200b"}).json()
    assert body["answer"] == "no"
    assert "below the 200B bar" in body["reason"]
    assert body["hint"]


def test_route_reads_live_modalities_through_the_real_manager(app: Any) -> None:
    """End to end through ``Manager.introspect`` and the child's /props, which is
    the same path the Dashboard renders."""
    with TestClient(app) as http:
        body = http.get("/api/model-gate", params={"vision": "true"}).json()
    assert body["answer"] == "yes"
    assert body["modalities"] == ["vision"]


def test_route_maps_the_sugar_flags_onto_the_tags_set(app: Any) -> None:
    with TestClient(app) as http:
        body = http.get(
            "/api/model-gate",
            params={"vision": "true", "thinking": "true", "tags": "coding,vietnamese"},
        ).json()
    assert body["checked"]["tags"] == ["coding", "thinking", "vietnamese", "vision"]


def test_every_sugar_flag_is_actually_exposed_by_the_route(app: Any) -> None:
    """Guards the sugar-vs-tags split the MCP tool mirrors: a flag added to
    SUGAR_TAGS and forgotten in the route would silently do nothing."""
    with TestClient(app) as http:
        body = http.get("/api/model-gate", params=dict.fromkeys(SUGAR_TAGS, "true")).json()
    assert body["checked"]["tags"] == sorted(SUGAR_TAGS)


def test_route_400s_on_junk_min_params(app: Any) -> None:
    with TestClient(app) as http:
        response = http.get("/api/model-gate", params={"min_params": "enormous"})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "min_params"
    assert "20b" in error["message"], "the 400 must name the shapes it accepts"


def test_route_400s_on_a_tags_value_that_cannot_be_tags(app: Any) -> None:
    with TestClient(app) as http:
        response = http.get("/api/model-gate", params={"tags": "x" * 200})
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "tags"


def test_route_is_open_to_a_lan_client(app: Any) -> None:
    """Read-only, so it is not behind the D32 admin gate -- ``is_admin_mutation``
    returns False for any verb outside {POST, PUT, PATCH, DELETE}, so a GET is
    open by construction and needs no entry in any admin list. Routing decisions
    are exactly what LAN clients are supposed to make."""
    with TestClient(app, client=("192.168.1.50", 50000)) as lan:
        response = lan.get("/api/model-gate", params={"min_params": "20b"})
    assert response.status_code == 200
    assert response.json()["model"] == BIG


def test_the_gate_does_not_shadow_a_model_route(app: Any) -> None:
    """The reason the path is /api/model-gate and not /api/models/gate: the
    per-model routes use a greedy ``:path`` converter."""
    with TestClient(app) as http:
        assert http.get("/api/model-gate").status_code == 200
        assert http.get(f"/api/models/{BIG}/introspect").status_code == 200
