"""The ``loaded`` model alias (D58).

``auto``/``current``/``default``/``local-model`` all resolve to the static
``models.default_model`` (``_require_model``, config only). ``loaded`` is a
second, separate alias resolved against LIVE server state instead:

    the largest model currently resident and ready that can serve THIS
    ROUTE'S KIND of request.

Route-kind awareness is the core of D58 (post-review fix-on-top of an earlier
design that only excluded embedders outright): ``want`` -- ``"chat"``,
``"embedding"`` or ``"rerank"`` -- is a required keyword-only argument on both
``_largest_ready_instance`` and ``_resolve_loaded_alias``, threaded through
from each of the five call sites in ``openai_routes.py``, so that
``/v1/embeddings`` naming ``loaded`` can never land on a chat model (the D-1
bug: it used to reach one and then 400 at the ``not_an_embedding_model``
guard) and ``/v1/chat/completions`` can never land on an embedder or a
reranker.

Two other things changed from the earlier design, both load-bearing here:

* the free-slot / active-request criterion is GONE (D58 DEC-2) -- a resident
  already at its own parallel cap is still a valid, still-preferred candidate,
  because nothing downstream actually queues on it being "free"; naming the
  model explicitly would just queue, and ``loaded`` silently downgrading to a
  smaller free model would defeat the whole point of the alias;
* an instance whose registry row has gone missing is NO LONGER a candidate
  at all (D58 DEC-1, filter step 3) -- the previous design sized it from its
  bare id and still picked it, but ``_resolve_or_404`` resolves through the
  very same registry dict, so a selected ghost would 404 downstream anyway;
  skipping it here means that failure never happens, and the caller sees the
  one clean ``no_loaded_model`` 404 instead.

The failure itself changed shape too (D58 DEC-3): nothing about "nothing of
this kind is resident" is transient, so it is a 404
(``invalid_request_error`` / ``no_loaded_model``, ``param="model"``), never a
503, and carries no ``Retry-After``.

A follow-up audit found that premise is FALSE in one window (D58, the
``model_busy`` fix): ``InstanceState`` includes ``"loading"``, and a JIT load
of the wanted kind can be minutes away from ready. ``_resolve_loaded_alias``
now checks ``_loading_instance_for`` before giving up, and raises
``ModelBusyError`` (503, code ``model_busy``, ``details={"loading": <id>,
"retry_after_s": BUSY_RETRY_AFTER_S}``) instead of the 404 when such a load is
in flight -- reusing ``model_busy`` rather than inventing a code because it is
already in OPENCLAW-RIG.md's retry-safe closed list. The 404 survives exactly
when nothing of the wanted kind is either ready OR loading. A ready candidate
still always wins over a loading one (``_largest_ready_instance`` is tried
first, unconditionally), and a loading instance of the WRONG kind never
rescues a request that needed a different one -- ``_loading_instance_for``
applies ``_serves_want`` exactly like the ready-instance filter does.

It is applied as its own step (``_resolve_loaded_alias``), right after
``_require_model`` and before ``_resolve_or_404``, at every one of the five
call sites in ``openai_routes.py`` -- so both the streaming and non-streaming
halves of ``/v1/chat/completions`` resolve it, because they share that same
call site above the branch, and it never touches the registry's alias table,
so ``GET /v1/models`` is never asked to carry a synthetic ``loaded`` entry.

First section below unit-tests the resolution helpers directly (no app, no
HTTP -- fast, precise coverage of the filter, the ranking/tie-break rules,
and the kind predicate). Second section drives the real FastAPI app through
``TestClient`` with a stubbed upstream, exactly like ``test_request_priority.py``,
to prove the alias resolves through the actual route wiring: all five routes
pass the right ``want``, both chat-completions branches, the 404 shape,
`/v1/models` non-pollution, and that `default` still works.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.api import openai_routes
from studioforge.api.app import build_state, create_app
from studioforge.api.openai_routes import (
    LOADED_MODEL_ALIAS,
    _largest_ready_instance,
    _loading_instance_for,
    _resolve_loaded_alias,
    _serves_want,
)
from studioforge.config import Config
from studioforge.core.planner import BUSY_RETRY_AFTER_S
from studioforge.errors import ModelBusyError, NoLoadedModelError
from studioforge.types import (
    GgufMeta,
    InstanceInfo,
    InstanceState,
    LoadPlan,
    ModelCapabilities,
    ModelKind,
    ModelRecord,
)
from tests.unit.test_catalog_routes import FakeProbe, FakeRegistry, FakeSupervisor

MESSAGES = [{"role": "user", "content": "hello"}]


def _record(
    model_id: str,
    param_count: int | None,
    *,
    kind: ModelKind = "chat",
    capabilities: ModelCapabilities | None = None,
) -> ModelRecord:
    return ModelRecord(
        id=model_id,
        name=model_id.rsplit("/", 1)[-1],
        path=Path(f"/models/{model_id.replace('/', '_')}.gguf"),
        kind=kind,
        capabilities=capabilities if capabilities is not None else ModelCapabilities(),
        meta=GgufMeta(param_count=param_count),
    )


def _plan(model_id: str, *, parallel: int = 2, ctx_size: int = 8192) -> LoadPlan:
    return LoadPlan(
        model_id=model_id,
        devices=[0],
        ctx_size=ctx_size,
        parallel=parallel,
        ctx_per_slot=ctx_size,
        max_parallel=parallel,
    )


def _instance(
    model_id: str,
    *,
    state: InstanceState = "ready",
    plan: LoadPlan | None = None,
    active_requests: int = 0,
) -> InstanceInfo:
    return InstanceInfo(
        model_id=model_id,
        state=state,
        plan=plan if plan is not None else _plan(model_id),
        active_requests=active_requests,
    )


class Registry(FakeRegistry):
    """``FakeRegistry`` plus the ``touch`` every JIT load path calls
    (``ensure_loaded`` -> ``registry.touch``), same as test_request_priority.py."""

    def touch(self, model_id: str) -> None:
        return None


class FakeState:
    """The two attributes the helpers read -- nothing else, so a test that
    reaches past them is a test with the wrong fixture."""

    def __init__(self, registry: FakeRegistry, supervisor: FakeSupervisor) -> None:
        self.registry = registry
        self.supervisor = supervisor


class SupervisorWithBaseUrl(FakeSupervisor):
    """FakeSupervisor plus base_url(), needed only by the streaming path."""

    def base_url(self, model_id: str) -> str | None:
        return "http://127.0.0.1:1/fake"


# ---------------------------------------------------------------------------
# _largest_ready_instance -- pure unit tests, no app
# ---------------------------------------------------------------------------


class TestLargestReadyInstance:
    def test_two_residents_the_larger_by_param_count_is_chosen(self) -> None:
        registry = FakeRegistry(
            [_record("v/small", 8_000_000_000), _record("v/big", 27_000_000_000)]
        )
        supervisor = FakeSupervisor([_instance("v/small"), _instance("v/big")])
        instance = _largest_ready_instance(FakeState(registry, supervisor), want="chat")
        assert instance is not None
        assert instance.model_id == "v/big"

    def test_a_saturated_larger_resident_still_outranks_a_free_smaller_one(self) -> None:
        """D58 DEC-2: the free-slot criterion is gone. A 27B pinned at its own
        parallel cap must still beat an 8B sitting idle -- silently handing the
        caller the smaller model on no signal at all is exactly what the
        removed criterion used to do, and it defeats the point of the alias."""
        registry = FakeRegistry(
            [_record("v/small", 8_000_000_000), _record("v/big", 27_000_000_000)]
        )
        supervisor = FakeSupervisor(
            [
                _instance("v/small", active_requests=0),
                _instance("v/big", plan=_plan("v/big", parallel=2), active_requests=2),
            ]
        )
        instance = _largest_ready_instance(FakeState(registry, supervisor), want="chat")
        assert instance is not None
        assert instance.model_id == "v/big", (
            "saturated (2 active == parallel 2) is no longer disqualifying"
        )

    def test_every_resident_at_its_cap_is_still_a_valid_candidate_set(self) -> None:
        """The old design returned None here (both 'full'); D58 has no notion
        of full any more, so the larger one is simply picked."""
        registry = FakeRegistry([_record("v/a", 1_000_000_000), _record("v/b", 2_000_000_000)])
        supervisor = FakeSupervisor(
            [
                _instance("v/a", plan=_plan("v/a", parallel=1), active_requests=1),
                _instance("v/b", plan=_plan("v/b", parallel=1), active_requests=1),
            ]
        )
        instance = _largest_ready_instance(FakeState(registry, supervisor), want="chat")
        assert instance is not None
        assert instance.model_id == "v/b"

    def test_no_resident_instances_returns_none(self) -> None:
        state = FakeState(FakeRegistry([]), FakeSupervisor([]))
        assert _largest_ready_instance(state, want="chat") is None

    @pytest.mark.parametrize("bad_state", ["loading", "stopped", "failed", "unloading"])
    def test_non_ready_instances_are_never_candidates(self, bad_state: InstanceState) -> None:
        """A "loading" instance has no proven capacity yet; the rest cannot
        serve at all -- none may outrank a smaller but genuinely ready one."""
        registry = FakeRegistry([_record("v/small", 1), _record("v/big", 999_000_000_000)])
        supervisor = FakeSupervisor(
            [_instance("v/small", state="ready"), _instance("v/big", state=bad_state)]
        )
        instance = _largest_ready_instance(FakeState(registry, supervisor), want="chat")
        assert instance is not None
        assert instance.model_id == "v/small"

    def test_an_instance_with_no_plan_is_never_a_candidate(self) -> None:
        registry = FakeRegistry([_record("v/a", 1)])
        supervisor = FakeSupervisor([InstanceInfo(model_id="v/a", state="ready", plan=None)])
        state = FakeState(registry, supervisor)
        assert _largest_ready_instance(state, want="chat") is None

    def test_unmeasured_param_count_ties_break_on_context_total(self) -> None:
        """Two records with no GGUF metadata both score 0 on param_count, so
        the tie-break is ctx_total = ctx_size * parallel."""
        registry = FakeRegistry([_record("v/a", None), _record("v/b", None)])
        supervisor = FakeSupervisor(
            [
                _instance("v/a", plan=_plan("v/a", parallel=1, ctx_size=8192)),  # ctx_total 8192
                _instance("v/b", plan=_plan("v/b", parallel=2, ctx_size=8192)),  # ctx_total 16384
            ]
        )
        instance = _largest_ready_instance(FakeState(registry, supervisor), want="chat")
        assert instance is not None
        assert instance.model_id == "v/b"

    def test_a_measured_model_always_outranks_an_unmeasured_one(self) -> None:
        """Even a small measured model (300M -> 0.3B, comfortably above the
        rounding floor at one decimal place) must beat a huge but
        stale/unscanned record whose param_count could not be read -- 0
        never wins."""
        registry = FakeRegistry([_record("v/measured", 300_000_000), _record("v/unmeasured", None)])
        supervisor = FakeSupervisor(
            [
                _instance("v/measured", plan=_plan("v/measured", ctx_size=2048)),
                _instance("v/unmeasured", plan=_plan("v/unmeasured", ctx_size=999_999)),
            ]
        )
        instance = _largest_ready_instance(FakeState(registry, supervisor), want="chat")
        assert instance is not None
        assert instance.model_id == "v/measured"

    def test_ranking_uses_the_records_metadata_not_a_name_guess_from_the_bare_id(self) -> None:
        """D58 DEC-1: filter step 3 guarantees a record is always present, and
        the ranker is handed that record -- not the bare id string -- so a
        genuine metadata param_count wins even when the id itself would parse
        to a bigger (wrong) guess via ``approx_params_b``'s name tier."""
        registry = FakeRegistry(
            [
                _record("vendor/misnamed-27B-instruct", 3_000_000_000),  # id says 27B, meta says 3B
                _record("vendor/small-1B-instruct", 8_000_000_000),  # id says 1B, meta says 8B
            ]
        )
        supervisor = FakeSupervisor(
            [_instance("vendor/misnamed-27B-instruct"), _instance("vendor/small-1B-instruct")]
        )
        instance = _largest_ready_instance(FakeState(registry, supervisor), want="chat")
        assert instance is not None
        assert instance.model_id == "vendor/small-1B-instruct"

    def test_an_instance_missing_from_the_registry_is_skipped_not_sized_by_its_name(self) -> None:
        """D58 DEC-1 reverses the earlier design: no registry row means not a
        candidate at all, because ``_resolve_or_404`` resolves through that
        same dict and would 404 on a selected ghost. So a properly-registered
        1B must beat a 27B-named ghost with no row, not lose to it."""
        registry = FakeRegistry([_record("vendor/small-1B-instruct", 1_000_000_000)])
        supervisor = FakeSupervisor(
            [
                _instance("vendor/small-1B-instruct"),
                _instance("vendor/ghost-27B-instruct"),  # no registry row at all
            ]
        )
        instance = _largest_ready_instance(FakeState(registry, supervisor), want="chat")
        assert instance is not None
        assert instance.model_id == "vendor/small-1B-instruct"

    def test_when_every_resident_is_missing_from_the_registry_there_is_no_candidate(self) -> None:
        supervisor = FakeSupervisor([_instance("v/ghost-a"), _instance("v/ghost-b")])
        state = FakeState(FakeRegistry([]), supervisor)
        assert _largest_ready_instance(state, want="chat") is None

    # -- route-kind awareness (D58 DEC-1, the core of the fix) --------------

    def test_want_chat_never_selects_an_embedding_kind_record(self) -> None:
        registry = FakeRegistry(
            [
                _record("vendor/tiny-embedder", 137_000_000, kind="embedding"),
                _record("vendor/small-chat", 1_000_000_000),
            ]
        )
        supervisor = FakeSupervisor(
            [_instance("vendor/tiny-embedder"), _instance("vendor/small-chat")]
        )
        instance = _largest_ready_instance(FakeState(registry, supervisor), want="chat")
        assert instance is not None
        assert instance.model_id == "vendor/small-chat"

    def test_want_chat_never_selects_a_rerank_kind_record(self) -> None:
        registry = FakeRegistry(
            [
                _record("vendor/tiny-reranker", 137_000_000, kind="rerank"),
                _record("vendor/small-chat", 1_000_000_000),
            ]
        )
        supervisor = FakeSupervisor(
            [_instance("vendor/tiny-reranker"), _instance("vendor/small-chat")]
        )
        instance = _largest_ready_instance(FakeState(registry, supervisor), want="chat")
        assert instance is not None
        assert instance.model_id == "vendor/small-chat"

    def test_want_embedding_selects_an_embedding_kind_record(self) -> None:
        """The D-1 regression itself, at the helper level: a bigger chat
        model resident alongside a smaller embedder must NOT win when the
        caller wants an embedding model."""
        registry = FakeRegistry(
            [
                _record("vendor/big-chat", 27_000_000_000),
                _record("vendor/embedder", 137_000_000, kind="embedding"),
            ]
        )
        supervisor = FakeSupervisor(
            [_instance("vendor/big-chat"), _instance("vendor/embedder")]
        )
        instance = _largest_ready_instance(FakeState(registry, supervisor), want="embedding")
        assert instance is not None
        assert instance.model_id == "vendor/embedder"

    def test_want_embedding_also_honours_the_capability_flag_on_a_different_kind(self) -> None:
        """DEC-1's predicate is ``kind == "embedding" or capabilities.embedding``
        -- exactly the negation of the existing ``not_an_embedding_model``
        guard -- so a record whose ``kind`` is ``"chat"`` but whose
        capabilities flag embedding support must still qualify."""
        registry = FakeRegistry(
            [
                _record(
                    "vendor/multi-purpose",
                    5_000_000_000,
                    capabilities=ModelCapabilities(embedding=True),
                )
            ]
        )
        supervisor = FakeSupervisor([_instance("vendor/multi-purpose")])
        instance = _largest_ready_instance(FakeState(registry, supervisor), want="embedding")
        assert instance is not None
        assert instance.model_id == "vendor/multi-purpose"

    def test_want_rerank_selects_only_a_rerank_kind_record(self) -> None:
        registry = FakeRegistry(
            [
                _record("vendor/big-chat", 27_000_000_000),
                _record("vendor/embedder", 8_000_000_000, kind="embedding"),
                _record("vendor/reranker", 137_000_000, kind="rerank"),
            ]
        )
        supervisor = FakeSupervisor(
            [
                _instance("vendor/big-chat"),
                _instance("vendor/embedder"),
                _instance("vendor/reranker"),
            ]
        )
        instance = _largest_ready_instance(FakeState(registry, supervisor), want="rerank")
        assert instance is not None
        assert instance.model_id == "vendor/reranker"

    @pytest.mark.parametrize(
        ("want", "resident_kinds"),
        [
            ("chat", ["embedding", "rerank"]),
            ("embedding", ["chat", "rerank"]),
            ("rerank", ["chat", "embedding"]),
        ],
    )
    def test_each_want_yields_no_candidate_when_only_other_kinds_are_resident(
        self, want: str, resident_kinds: list[ModelKind]
    ) -> None:
        """The D-1 bug in miniature: models ARE resident and ready, just none
        of the kind this route needs -- must be None, never a wrong-kind pick."""
        registry = FakeRegistry(
            [
                _record(f"vendor/{kind}-model", 27_000_000_000, kind=kind)  # type: ignore[arg-type]
                for kind in resident_kinds
            ]
        )
        supervisor = FakeSupervisor(
            [_instance(f"vendor/{kind}-model") for kind in resident_kinds]
        )
        assert _largest_ready_instance(FakeState(registry, supervisor), want=want) is None

    def test_an_only_resident_embedder_yields_no_chat_candidate(self) -> None:
        """The exact regression the earlier review named: a resident embedder
        must never win a chat request by default just because nothing else is
        loaded."""
        registry = FakeRegistry([_record("vendor/embedder", 137_000_000, kind="embedding")])
        supervisor = FakeSupervisor([_instance("vendor/embedder")])
        state = FakeState(registry, supervisor)
        assert _largest_ready_instance(state, want="chat") is None

    def test_want_is_a_required_keyword_only_argument(self) -> None:
        state = FakeState(FakeRegistry([]), FakeSupervisor([]))
        with pytest.raises(TypeError):
            _largest_ready_instance(state)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# _serves_want -- pure unit tests, no app
# ---------------------------------------------------------------------------


class TestServesWant:
    """`_serves_want` (D58) directly -- the kind predicate both
    `_largest_ready_instance` and `_loading_instance_for` delegate to,
    isolated here from residency/state entirely."""

    def test_none_record_is_never_a_match_for_any_want(self) -> None:
        for want in ("chat", "embedding", "rerank", "video"):
            assert _serves_want(None, want) is False

    def test_want_chat_matches_only_kind_chat(self) -> None:
        assert _serves_want(_record("v/a", 1, kind="chat"), "chat") is True
        assert _serves_want(_record("v/a", 1, kind="embedding"), "chat") is False
        assert _serves_want(_record("v/a", 1, kind="rerank"), "chat") is False

    def test_want_embedding_matches_kind_embedding(self) -> None:
        assert _serves_want(_record("v/a", 1, kind="embedding"), "embedding") is True

    def test_want_embedding_also_honours_the_capability_flag_on_another_kind(self) -> None:
        """Exactly the negation of the `not_an_embedding_model` guard the
        `embeddings` route applies to a client-named model (D52): a
        chat-kind record whose `capabilities.embedding` is set still serves
        an embedding want."""
        record = _record("v/a", 1, kind="chat", capabilities=ModelCapabilities(embedding=True))
        assert _serves_want(record, "embedding") is True

    def test_want_embedding_rejects_a_plain_chat_record_without_the_capability(self) -> None:
        assert _serves_want(_record("v/a", 1, kind="chat"), "embedding") is False

    def test_want_rerank_matches_only_kind_rerank(self) -> None:
        assert _serves_want(_record("v/a", 1, kind="rerank"), "rerank") is True
        assert _serves_want(_record("v/a", 1, kind="chat"), "rerank") is False
        assert _serves_want(_record("v/a", 1, kind="embedding"), "rerank") is False

    def test_an_unrecognised_want_matches_no_record_of_any_kind(self) -> None:
        for kind in ("chat", "embedding", "rerank"):
            assert _serves_want(_record("v/a", 1, kind=kind), "video") is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _loading_instance_for -- pure unit tests, no app
# ---------------------------------------------------------------------------


class TestLoadingInstanceFor:
    """`_loading_instance_for` (D58, the audit's `model_busy` fix) -- the one
    place `state == "loading"` is read as a signal instead of being filtered
    out, exactly as `_largest_ready_instance` filters it (its step 1 treats
    "loading" the same as stopped/failed/unloading: no proven capacity yet)."""

    def test_a_loading_instance_of_the_wanted_kind_is_found(self) -> None:
        registry = FakeRegistry([_record("v/a", 1)])
        supervisor = FakeSupervisor([_instance("v/a", state="loading")])
        found = _loading_instance_for(FakeState(registry, supervisor), want="chat")
        assert found is not None
        assert found.model_id == "v/a"

    def test_a_loading_instance_of_a_different_kind_is_not_found(self) -> None:
        """The core of the audit's second required case: an embedder mid-load
        must not answer a chat want."""
        registry = FakeRegistry([_record("v/embedder", 1, kind="embedding")])
        supervisor = FakeSupervisor([_instance("v/embedder", state="loading")])
        assert _loading_instance_for(FakeState(registry, supervisor), want="chat") is None

    def test_a_ready_instance_is_not_a_loading_candidate(self) -> None:
        registry = FakeRegistry([_record("v/a", 1)])
        supervisor = FakeSupervisor([_instance("v/a", state="ready")])
        assert _loading_instance_for(FakeState(registry, supervisor), want="chat") is None

    @pytest.mark.parametrize("bad_state", ["stopped", "failed", "unloading"])
    def test_every_other_non_loading_state_is_not_a_candidate_either(
        self, bad_state: InstanceState
    ) -> None:
        registry = FakeRegistry([_record("v/a", 1)])
        supervisor = FakeSupervisor([_instance("v/a", state=bad_state)])
        assert _loading_instance_for(FakeState(registry, supervisor), want="chat") is None

    def test_a_loading_instance_with_no_registry_row_is_not_found(self) -> None:
        """Same D58 DEC-1 reasoning as `_largest_ready_instance`'s ghost
        filter: a loading instance whose record has gone missing cannot be
        named to a caller, busy or otherwise."""
        supervisor = FakeSupervisor([_instance("v/ghost", state="loading")])
        state = FakeState(FakeRegistry([]), supervisor)
        assert _loading_instance_for(state, want="chat") is None

    def test_no_instances_at_all_returns_none(self) -> None:
        state = FakeState(FakeRegistry([]), FakeSupervisor([]))
        assert _loading_instance_for(state, want="chat") is None

    def test_any_one_matching_loading_instance_among_several_suffices(self) -> None:
        """Only existence matters here -- the 503 names ONE loading model, not
        a ranked "biggest loading" pick, so any qualifying instance is fine."""
        registry = FakeRegistry([_record("v/a", 1), _record("v/b", 2)])
        supervisor = FakeSupervisor(
            [_instance("v/a", state="loading"), _instance("v/b", state="loading")]
        )
        found = _loading_instance_for(FakeState(registry, supervisor), want="chat")
        assert found is not None
        assert found.model_id in {"v/a", "v/b"}


# ---------------------------------------------------------------------------
# _resolve_loaded_alias -- pure unit tests, no app
# ---------------------------------------------------------------------------


class TestResolveLoadedAlias:
    def test_alias_constant_is_the_word_loaded(self) -> None:
        assert LOADED_MODEL_ALIAS == "loaded"

    def test_names_other_than_loaded_pass_through_unchanged(self) -> None:
        state = FakeState(FakeRegistry([]), FakeSupervisor([]))
        for name in ("vendor/model-Q4_K_M", "auto", "default", "current", "local-model", ""):
            assert _resolve_loaded_alias(state, name, want="chat") == name

    @pytest.mark.parametrize("spelling", ["loaded", "LOADED", "Loaded", "  loaded  ", "LoAdEd"])
    def test_case_and_whitespace_insensitive(self, spelling: str) -> None:
        registry = FakeRegistry([_record("v/a", 1)])
        supervisor = FakeSupervisor([_instance("v/a")])
        state = FakeState(registry, supervisor)
        assert _resolve_loaded_alias(state, spelling, want="chat") == "v/a"

    def test_raises_no_loaded_model_with_the_right_shape_when_nothing_qualifies(self) -> None:
        state = FakeState(FakeRegistry([]), FakeSupervisor([]))
        with pytest.raises(NoLoadedModelError) as exc_info:
            _resolve_loaded_alias(state, "loaded", want="chat")
        assert exc_info.value.status_code == 404
        assert exc_info.value.code == "no_loaded_model"
        assert exc_info.value.error_type == "invalid_request_error"
        assert exc_info.value.param == "model"
        assert exc_info.value.details.get("retry_after_s") is None, (
            "D58 DEC-3: not transient, must carry no Retry-After hint"
        )

    @pytest.mark.parametrize("want", ["chat", "embedding", "rerank"])
    def test_the_404_message_names_the_kind_and_the_remedy(self, want: str) -> None:
        state = FakeState(FakeRegistry([]), FakeSupervisor([]))
        with pytest.raises(NoLoadedModelError) as exc_info:
            _resolve_loaded_alias(state, "loaded", want=want)
        message = str(exc_info.value)
        assert want in message
        assert "loaded" in message
        assert "explicit" in message.lower() or "name a model" in message.lower()

    def test_each_want_gets_its_own_404_even_when_other_kinds_are_resident(self) -> None:
        """The D-1 bug at the alias-resolution level: an embedder is resident
        and ready, but a chat caller must still get the clean 404, never the
        embedder."""
        registry = FakeRegistry([_record("vendor/embedder", 137_000_000, kind="embedding")])
        supervisor = FakeSupervisor([_instance("vendor/embedder")])
        state = FakeState(registry, supervisor)
        with pytest.raises(NoLoadedModelError):
            _resolve_loaded_alias(state, "loaded", want="chat")
        # But the same state resolves fine for the kind that IS resident.
        assert _resolve_loaded_alias(state, "loaded", want="embedding") == "vendor/embedder"

    def test_skipping_a_registry_less_ghost_avoids_a_confusing_downstream_404(self) -> None:
        """If a ghost instance with no registry row were wrongly selected,
        ``_resolve_loaded_alias`` would hand back an id that
        ``_resolve_or_404`` -- which resolves through that very same registry
        dict -- would then 404 on as ``model_not_found``, a confusing error
        for someone who only ever asked for ``loaded``. Filtering the ghost
        out here means the caller sees the ONE clean ``no_loaded_model``
        failure instead."""
        supervisor = FakeSupervisor([_instance("vendor/ghost")])
        state = FakeState(FakeRegistry([]), supervisor)
        with pytest.raises(NoLoadedModelError) as exc_info:
            _resolve_loaded_alias(state, "loaded", want="chat")
        assert exc_info.value.code == "no_loaded_model"

    def test_want_is_a_required_keyword_only_argument(self) -> None:
        state = FakeState(FakeRegistry([]), FakeSupervisor([]))
        with pytest.raises(TypeError):
            _resolve_loaded_alias(state, "loaded")  # type: ignore[call-arg]

    # -- the audit's `model_busy` fix -----------------------------------

    def test_raises_model_busy_when_nothing_ready_but_the_wanted_kind_is_loading(self) -> None:
        """Required case 1, at the helper level: nothing ready but a
        `loading` instance of the wanted kind -- 503 `model_busy`, and
        `details.loading` names the loading model."""
        registry = FakeRegistry([_record("v/a", 1)])
        supervisor = FakeSupervisor([_instance("v/a", state="loading")])
        state = FakeState(registry, supervisor)
        with pytest.raises(ModelBusyError) as exc_info:
            _resolve_loaded_alias(state, "loaded", want="chat")
        assert exc_info.value.status_code == 503
        assert exc_info.value.code == "model_busy"
        assert exc_info.value.details["loading"] == "v/a"
        assert exc_info.value.details["retry_after_s"] == BUSY_RETRY_AFTER_S

    def test_a_loading_instance_of_a_different_kind_does_not_rescue_the_request(self) -> None:
        """Required case 2: an embedding model loading while a chat `loaded`
        request arrives must still be the 404 `no_loaded_model`, not a 503 --
        a load of the wrong kind is no reason to make this caller wait."""
        registry = FakeRegistry([_record("v/embedder", 1, kind="embedding")])
        supervisor = FakeSupervisor([_instance("v/embedder", state="loading")])
        state = FakeState(registry, supervisor)
        with pytest.raises(NoLoadedModelError) as exc_info:
            _resolve_loaded_alias(state, "loaded", want="chat")
        assert exc_info.value.code == "no_loaded_model"

    def test_a_ready_candidate_always_wins_over_a_loading_one(self) -> None:
        """Required case 3: a load in flight for a much bigger model must
        never turn a servable request into a 503 -- `_largest_ready_instance`
        is tried first, unconditionally, so `_loading_instance_for` is never
        even consulted while a ready candidate exists."""
        registry = FakeRegistry([_record("v/ready", 1), _record("v/loading", 999_000_000_000)])
        supervisor = FakeSupervisor(
            [
                _instance("v/ready", state="ready"),
                _instance("v/loading", state="loading"),
            ]
        )
        state = FakeState(registry, supervisor)
        assert _resolve_loaded_alias(state, "loaded", want="chat") == "v/ready"

    def test_model_busy_message_names_the_loading_model_and_the_remedy(self) -> None:
        registry = FakeRegistry([_record("v/a", 1)])
        supervisor = FakeSupervisor([_instance("v/a", state="loading")])
        state = FakeState(registry, supervisor)
        with pytest.raises(ModelBusyError) as exc_info:
            _resolve_loaded_alias(state, "loaded", want="chat")
        message = str(exc_info.value)
        assert "v/a" in message
        assert "loaded" in message


# ---------------------------------------------------------------------------
# Through the real route wiring: TestClient + a stubbed upstream, exactly
# like test_request_priority.py -- no engine, no network, no rig.
# ---------------------------------------------------------------------------

SMALL_ID = "vendor/small-Q4_K_M"
BIG_ID = "vendor/big-Q5_K_M"
EMBED_ID = "vendor/embedder-Q8_0"
RERANK_ID = "vendor/reranker-Q8_0"


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
    built.state.registry = Registry(
        [_record(SMALL_ID, 8_000_000_000), _record(BIG_ID, 27_000_000_000)]
    )
    built.state.probe = FakeProbe()
    built.state.planner.probe = FakeProbe()
    built.state.manager.registry = built.state.registry
    # Both residents ready: these tests are about which one gets PICKED, not
    # about a load. Note D58 DEC-2: no free-slot bias, so active_requests is
    # irrelevant here and left at the default of 0.
    supervisor = SupervisorWithBaseUrl([_instance(SMALL_ID), _instance(BIG_ID)])
    built.state.supervisor = supervisor
    built.state.manager.supervisor = supervisor
    return built


def _set_residents(app: Any, records: list[ModelRecord], instances: list[InstanceInfo]) -> None:
    """Swap the whole resident picture -- registry rows and running instances
    -- keeping the manager's own references in sync, same wiring as the
    ``app`` fixture above."""
    registry = Registry(records)
    supervisor = SupervisorWithBaseUrl(instances)
    app.state.registry = registry
    app.state.manager.registry = registry
    app.state.supervisor = supervisor
    app.state.manager.supervisor = supervisor


@pytest.fixture()
def upstream(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every payload the gateway hands to llama-server, newest last."""
    sent: list[dict[str, Any]] = []

    async def fake_forward(
        state: Any, record: Any, path: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        sent.append(dict(payload))
        return {"id": "cmpl-test", "object": "chat.completion", "choices": []}

    monkeypatch.setattr(openai_routes, "_forward", fake_forward)
    return sent


def test_non_streaming_chat_completions_resolves_loaded_to_the_largest(
    app: Any, upstream: list[dict[str, Any]]
) -> None:
    with TestClient(app) as http:
        response = http.post("/v1/chat/completions", json={"model": "loaded", "messages": MESSAGES})
    assert response.status_code == 200, response.text
    assert len(upstream) == 1
    assert upstream[0]["model"] == BIG_ID


def test_streaming_chat_completions_resolves_loaded_too(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The streaming branch shares the SAME resolution call site, above the
    `if payload.get("stream")` split -- proved here by stubbing the one
    function only the streaming path calls and reading which record it got."""
    seen: dict[str, Any] = {}

    async def fake_stream_upstream(
        state: Any, record: Any, url: str, payload: dict[str, Any], started: float
    ):
        seen["record_id"] = record.id
        seen["payload_model"] = payload.get("model")
        yield b"data: [DONE]\n\n"

    monkeypatch.setattr(openai_routes, "_stream_upstream", fake_stream_upstream)
    with TestClient(app) as http:
        response = http.post(
            "/v1/chat/completions",
            json={"model": "loaded", "messages": MESSAGES, "stream": True},
        )
    assert response.status_code == 200, response.text
    assert seen.get("record_id") == BIG_ID
    assert seen.get("payload_model") == BIG_ID


def test_loaded_alias_is_case_insensitive_over_http(
    app: Any, upstream: list[dict[str, Any]]
) -> None:
    with TestClient(app) as http:
        response = http.post("/v1/chat/completions", json={"model": "LOADED", "messages": MESSAGES})
    assert response.status_code == 200, response.text
    assert upstream[0]["model"] == BIG_ID


def test_default_alias_still_works_unaffected_by_the_new_one(
    app: Any, upstream: list[dict[str, Any]]
) -> None:
    app.state.config.models.default_model = SMALL_ID
    with TestClient(app) as http:
        response = http.post(
            "/v1/chat/completions", json={"model": "default", "messages": MESSAGES}
        )
    assert response.status_code == 200, response.text
    assert upstream[0]["model"] == SMALL_ID


def test_a_saturated_largest_resident_is_still_returned_by_loaded_over_http(
    app: Any, upstream: list[dict[str, Any]]
) -> None:
    """D58 DEC-2, end to end: both residents pinned at their own parallel cap
    must still yield the LARGER one, not a 404 and not a silent downgrade."""
    full = SupervisorWithBaseUrl(
        [
            _instance(SMALL_ID, plan=_plan(SMALL_ID, parallel=1), active_requests=1),
            _instance(BIG_ID, plan=_plan(BIG_ID, parallel=1), active_requests=1),
        ]
    )
    app.state.supervisor = full
    app.state.manager.supervisor = full
    with TestClient(app) as http:
        response = http.post("/v1/chat/completions", json={"model": "loaded", "messages": MESSAGES})
    assert response.status_code == 200, response.text
    assert upstream[0]["model"] == BIG_ID


def test_no_loaded_model_when_nothing_is_resident_is_a_404(app: Any) -> None:
    empty = SupervisorWithBaseUrl([])
    app.state.supervisor = empty
    app.state.manager.supervisor = empty
    with TestClient(app) as http:
        response = http.post("/v1/chat/completions", json={"model": "loaded", "messages": MESSAGES})
    assert response.status_code == 404, response.text
    body = response.json()
    assert body["error"]["code"] == "no_loaded_model"
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["param"] == "model"
    assert "Retry-After" not in response.headers, (
        "D58 DEC-3: this is not transient -- nothing changes until someone loads a model"
    )


def test_an_only_resident_embedder_is_not_handed_to_a_chat_request(app: Any) -> None:
    """The earlier review's exact scenario: an embedding model is the only
    resident on the rig. `loaded` must 404, never hand a chat request an
    embedder it cannot serve."""
    embedder = Registry([_record("vendor/embedder", 137_000_000, kind="embedding")])
    app.state.registry = embedder
    app.state.manager.registry = embedder
    only_embedder = SupervisorWithBaseUrl([_instance("vendor/embedder")])
    app.state.supervisor = only_embedder
    app.state.manager.supervisor = only_embedder
    with TestClient(app) as http:
        response = http.post("/v1/chat/completions", json={"model": "loaded", "messages": MESSAGES})
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "no_loaded_model"


def test_v1_models_is_not_polluted_with_a_synthetic_loaded_entry(app: Any) -> None:
    """Aliases are request-side: the catalogue must list only the two real
    models, never a 'loaded' row."""
    with TestClient(app) as http:
        data = http.get("/v1/models").json()["data"]
    ids = {m["id"] for m in data}
    assert ids == {SMALL_ID, BIG_ID}
    assert LOADED_MODEL_ALIAS not in ids


# ---------------------------------------------------------------------------
# Route wiring: each of the five call sites passes the right `want` (D58
# DEC-4). This is the HTTP-level proof of the route-kind awareness that
# TestLargestReadyInstance and TestResolveLoadedAlias already cover at the
# helper level.
# ---------------------------------------------------------------------------


def test_v1_completions_resolves_loaded_to_the_largest_chat_model(
    app: Any, upstream: list[dict[str, Any]]
) -> None:
    with TestClient(app) as http:
        response = http.post("/v1/completions", json={"model": "loaded", "prompt": "hello"})
    assert response.status_code == 200, response.text
    assert upstream[0]["model"] == BIG_ID


def test_v1_tokenize_resolves_loaded_to_the_same_model_as_chat_completions(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DEC-4: `/v1/tokenize` is `want="chat"` for consistency, not capability
    -- a client tokenizes to budget for the model it is about to chat with, so
    `loaded` there must resolve to the SAME model as `/v1/chat/completions`.
    `/tokenize`'s handler forwards the body verbatim (never rewrites
    `payload["model"]`), so the record actually resolved is read off the
    `_forward` call instead of the payload."""
    seen: dict[str, Any] = {}

    async def fake_forward(
        state: Any, record: Any, path: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        seen["record_id"] = record.id
        seen["path"] = path
        return {"tokens": [1, 2, 3]}

    monkeypatch.setattr(openai_routes, "_forward", fake_forward)
    with TestClient(app) as http:
        response = http.post("/v1/tokenize", json={"model": "loaded", "content": "hello"})
    assert response.status_code == 200, response.text
    assert seen.get("record_id") == BIG_ID


def test_v1_embeddings_with_loaded_reaches_an_embedding_model_not_a_chat_model(
    app: Any, upstream: list[dict[str, Any]]
) -> None:
    """The D-1 bug, reproduced exactly: under the earlier design `loaded` on
    `/v1/embeddings` picked the larger CHAT model and died at the
    `not_an_embedding_model` guard in the embeddings route. A smaller embedder
    resident alongside a much bigger chat model must win here."""
    _set_residents(
        app,
        [_record(BIG_ID, 27_000_000_000), _record(EMBED_ID, 137_000_000, kind="embedding")],
        [_instance(BIG_ID), _instance(EMBED_ID)],
    )
    with TestClient(app) as http:
        response = http.post("/v1/embeddings", json={"model": "loaded", "input": "hello"})
    assert response.status_code == 200, response.text
    assert upstream[0]["model"] == EMBED_ID


def test_v1_rerank_with_loaded_reaches_only_a_rerank_model(
    app: Any, upstream: list[dict[str, Any]]
) -> None:
    _set_residents(
        app,
        [
            _record(BIG_ID, 27_000_000_000),
            _record(EMBED_ID, 8_000_000_000, kind="embedding"),
            _record(RERANK_ID, 137_000_000, kind="rerank"),
        ],
        [_instance(BIG_ID), _instance(EMBED_ID), _instance(RERANK_ID)],
    )
    with TestClient(app) as http:
        response = http.post(
            "/v1/rerank",
            json={"model": "loaded", "query": "q", "documents": ["a", "b"]},
        )
    assert response.status_code == 200, response.text
    assert upstream[0]["model"] == RERANK_ID


def test_v1_embeddings_404s_as_no_loaded_model_when_only_a_chat_model_is_resident(
    app: Any,
) -> None:
    """Must be the clean `no_loaded_model` 404 -- never the old
    `not_an_embedding_model` 400, which would mean it wrongly resolved to the
    chat model first."""
    with TestClient(app) as http:
        response = http.post("/v1/embeddings", json={"model": "loaded", "input": "hello"})
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "no_loaded_model"


def test_v1_rerank_404s_as_no_loaded_model_when_no_rerank_model_is_resident(app: Any) -> None:
    _set_residents(
        app,
        [_record(BIG_ID, 27_000_000_000), _record(EMBED_ID, 8_000_000_000, kind="embedding")],
        [_instance(BIG_ID), _instance(EMBED_ID)],
    )
    with TestClient(app) as http:
        response = http.post(
            "/v1/rerank",
            json={"model": "loaded", "query": "q", "documents": ["a", "b"]},
        )
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "no_loaded_model"


def test_v1_chat_completions_404s_as_no_loaded_model_when_only_non_chat_kinds_are_resident(
    app: Any,
) -> None:
    _set_residents(
        app,
        [
            _record(EMBED_ID, 8_000_000_000, kind="embedding"),
            _record(RERANK_ID, 27_000_000_000, kind="rerank"),
        ],
        [_instance(EMBED_ID), _instance(RERANK_ID)],
    )
    with TestClient(app) as http:
        response = http.post("/v1/chat/completions", json={"model": "loaded", "messages": MESSAGES})
    assert response.status_code == 404, response.text
    assert response.json()["error"]["code"] == "no_loaded_model"


# ---------------------------------------------------------------------------
# The audit's `model_busy` fix, over HTTP: nothing about "nothing of the
# wanted kind is resident" is transient when something of that kind is
# already mid-JIT-load (`InstanceState` includes "loading", and that can take
# minutes) -- `loaded` must 503 `model_busy`, not 404, for that window. These
# are the HTTP-level proof of TestLoadingInstanceFor and the model_busy cases
# in TestResolveLoadedAlias above.
# ---------------------------------------------------------------------------


def test_loaded_alias_returns_503_model_busy_when_nothing_ready_but_the_wanted_kind_is_loading(
    app: Any,
) -> None:
    """Required case 1: nothing ready but a `loading` instance of the wanted
    kind -- 503, code `model_busy`, a `Retry-After` header, and
    `details.loading` naming the loading model."""
    loading_only = SupervisorWithBaseUrl([_instance(BIG_ID, state="loading")])
    app.state.supervisor = loading_only
    app.state.manager.supervisor = loading_only
    with TestClient(app) as http:
        response = http.post("/v1/chat/completions", json={"model": "loaded", "messages": MESSAGES})
    assert response.status_code == 503, response.text
    body = response.json()
    assert body["error"]["code"] == "model_busy"
    assert body["error"]["type"] == "server_error"
    assert body["error"]["studioforge"]["loading"] == BIG_ID
    assert "Retry-After" in response.headers


def test_loaded_alias_loading_instance_of_a_different_kind_still_404s_over_http(
    app: Any,
) -> None:
    """Required case 2, over HTTP: an embedding model loading while a chat
    `loaded` request arrives must still be the 404 `no_loaded_model`, never a
    503 -- the wrong-kind load is no reason to make this caller wait."""
    _set_residents(
        app,
        [_record(EMBED_ID, 8_000_000_000, kind="embedding")],
        [_instance(EMBED_ID, state="loading")],
    )
    with TestClient(app) as http:
        response = http.post("/v1/chat/completions", json={"model": "loaded", "messages": MESSAGES})
    assert response.status_code == 404, response.text
    body = response.json()
    assert body["error"]["code"] == "no_loaded_model"
    assert "Retry-After" not in response.headers


def test_loaded_alias_prefers_a_ready_model_over_one_still_loading_over_http(
    app: Any, upstream: list[dict[str, Any]]
) -> None:
    """Required case 3, over HTTP: a load in flight for the BIGGER model must
    not turn a servable request into a 503 -- the smaller ready resident still
    wins, exactly as it does with no load in flight at all."""
    mixed = SupervisorWithBaseUrl(
        [_instance(SMALL_ID, state="ready"), _instance(BIG_ID, state="loading")]
    )
    app.state.supervisor = mixed
    app.state.manager.supervisor = mixed
    with TestClient(app) as http:
        response = http.post("/v1/chat/completions", json={"model": "loaded", "messages": MESSAGES})
    assert response.status_code == 200, response.text
    assert upstream[0]["model"] == SMALL_ID


def test_loaded_alias_model_busy_over_http_on_the_embeddings_route_too(
    app: Any,
) -> None:
    """Not just `/v1/chat/completions`: `/v1/embeddings` threads the same
    `want` through to the same `_resolve_loaded_alias` call site, so an
    embedding model mid-load must busy an embeddings `loaded` request the
    same way."""
    _set_residents(
        app,
        [_record(EMBED_ID, 8_000_000_000, kind="embedding")],
        [_instance(EMBED_ID, state="loading")],
    )
    with TestClient(app) as http:
        response = http.post("/v1/embeddings", json={"model": "loaded", "input": "hello"})
    assert response.status_code == 503, response.text
    body = response.json()
    assert body["error"]["code"] == "model_busy"
    assert body["error"]["studioforge"]["loading"] == EMBED_ID
    assert "Retry-After" in response.headers
