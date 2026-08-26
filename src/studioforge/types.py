"""Shared domain types.

This module is the vocabulary the whole system speaks: the registry stores
these, the planner returns these, the supervisor launches from these, and the
API/GUI/MCP layers render these. It deliberately has no dependencies beyond
pydantic and config so every layer can import it without cycles.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from studioforge.config import FlashAttn, KvCacheType, SplitMode

ModelKind = Literal["chat", "embedding", "rerank"]
InstanceState = Literal["stopped", "loading", "ready", "failed", "unloading"]

#: What stopped the parallel estimator from offering more slots.
#: ``"explicit"`` means nobody asked it -- a caller pinned the slot count.
#: ``"unknown"`` means the model had no usable metadata to estimate from.
ParallelBound = Literal["vram", "knee", "cap", "explicit", "unknown"]

MB = 1024 * 1024
GB = 1024 * 1024 * 1024


# ---------------------------------------------------------------------------
# GGUF metadata
# ---------------------------------------------------------------------------


class GgufMeta(BaseModel):
    """The subset of GGUF metadata the planner and capability detection need."""

    architecture: str = "unknown"
    n_layer: int = 0
    n_embd: int = 0
    n_head: int = 0
    n_head_kv: int = 0
    n_ctx_train: int = 0
    n_vocab: int = 0
    n_embd_head_k: int = 0
    n_embd_head_v: int = 0
    rope_freq_base: float = 0.0
    n_expert: int = 0
    n_expert_used: int = 0
    file_type: int | None = None
    quant_label: str = "unknown"
    param_count: int | None = None
    tensor_bytes: int = 0
    tokenizer_model: str = ""
    chat_template: str | None = None
    has_vision_tensors: bool = False
    is_mmproj: bool = False
    is_adapter: bool = False
    # mmproj-specific
    vision_n_patch: int | None = None
    vision_image_size: int | None = None
    vision_patch_size: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def head_dim_k(self) -> int:
        """Per-head K dimension, falling back to n_embd/n_head when absent.

        Models with explicit ``attention.key_length`` (e.g. Gemma, DeepSeek MLA)
        must use that value: deriving it from n_embd/n_head is wrong for them and
        silently mis-sizes the KV cache.
        """
        if self.n_embd_head_k > 0:
            return self.n_embd_head_k
        if self.n_head > 0 and self.n_embd > 0:
            return self.n_embd // self.n_head
        return 0

    @property
    def head_dim_v(self) -> int:
        if self.n_embd_head_v > 0:
            return self.n_embd_head_v
        return self.head_dim_k

    @property
    def supports_thinking(self) -> bool:
        """Whether the chat template drives a reasoning/thinking mode.

        Detected from the template rather than the name because the naming is
        chaotic (R1, QwQ, "-thinking", Gemma/Qwen variants that expose an
        ``enable_thinking`` switch). It matters to a user for two practical
        reasons: thoughts land inline in ``content`` under our default
        ``--reasoning-format none`` (DECISIONS.md D12), and a stop sequence can
        trip on text the model narrates while thinking.
        """
        if not self.chat_template:
            return False
        template = self.chat_template
        return any(
            marker in template
            for marker in ("<think>", "</think>", "enable_thinking", "reasoning_content")
        )

    @property
    def supports_tools(self) -> bool:
        """Whether the embedded chat template appears to handle tool calling."""
        if not self.chat_template:
            return False
        template = self.chat_template
        return "tools" in template or "tool_calls" in template


# ---------------------------------------------------------------------------
# Per-model settings (the three-tier flag surface)
# ---------------------------------------------------------------------------


class ModelSettings(BaseModel):
    """Saved per-model launch settings.

    ``None`` means "inherit the global default / let the planner decide", which
    is what keeps "Auto" meaningful: the planner runs against *current* free
    VRAM at load time rather than baking a decision in at save time.
    """

    # --- Tier 1: basic -------------------------------------------------
    ctx_size: int | None = None
    #: Refuse any load whose planned window per slot would land BELOW this,
    #: instead of quietly serving a smaller one. For a model that is a
    #: fallback link in someone's chain: a 61k window that "works" per turn
    #: and then shreds a 51k-prompt session through compaction is worse than
    #: a structured 507 the client can route around. Only applied when the
    #: request itself names no ctx_size -- an explicit ask is honoured (D14).
    min_ctx: int | None = None
    kv_cache_type: KvCacheType | None = None
    kv_cache_type_v: KvCacheType | None = None
    ttl_s: int | None = None
    pinned: bool = False
    draft_model_id: str | None = None
    device_override: list[int] | None = None
    #: The set of CUDA devices the planner MAY choose among for this model --
    #: softer than ``device_override``, which forces an exact placement. A
    #: big dense model with null settings otherwise sprawls across whatever
    #: cards are free, mixed generations included. ``None`` = any card.
    allowed_devices: list[int] | None = None
    engine_tag: str | None = None  # per-model engine pin

    # --- Tier 2: advanced ----------------------------------------------
    batch_size: int | None = None
    ubatch_size: int | None = None
    threads: int | None = None
    threads_batch: int | None = None
    parallel: int | None = None
    cont_batching: bool | None = None
    flash_attn: FlashAttn | None = None
    split_mode: SplitMode | None = None
    main_gpu: int | None = None
    mlock: bool = False
    no_mmap: bool | None = None
    rope_freq_base: float | None = None
    rope_freq_scale: float | None = None
    rope_scaling: str | None = None
    cache_reuse: int | None = None
    # Reasoning/thinking models. Default comes from models.default_reasoning_format
    # ("none"), which keeps thoughts inline in message.content -- llama.cpp's own
    # default ("auto") can leave content EMPTY. See DECISIONS.md D12.
    reasoning_format: Literal["none", "deepseek", "deepseek-legacy"] | None = None
    reasoning: Literal["on", "off", "auto"] | None = None
    reasoning_budget: int | None = None
    no_context_shift: bool | None = None
    #: DEPRECATED and inert. ``--defrag-thold`` is marked deprecated by b10425
    #: and the supervisor no longer emits it, so setting this changes nothing
    #: about the launched child. The field is kept (rather than deleted) purely
    #: so stored settings, the GUI's settings form and older API clients keep
    #: round-tripping; it will be removed once those stop reading it.
    defrag_thold: float | None = None
    #: Upper bound on the slot count the parallel estimator may choose for THIS
    #: model, on top of the global ``MAX_PARALLEL_CAP``. Only meaningful while
    #: ``models.default_parallel`` is ``"auto"`` -- an explicit ``parallel``
    #: is honoured verbatim and is never capped (DECISIONS.md D14/D17).
    max_parallel_cap: int | None = None
    #: Emit ``--kv-unified``: one shared KV pool across slots rather than an
    #: equal slice each. Same total VRAM, but a single long request can then use
    #: the whole pool instead of ``ctx_size``. Default off (``None``) because
    #: llama.cpp only defaults it on when ``--parallel`` is auto, and we always
    #: pass an explicit slot count -- so turning it on is a real behaviour
    #: change that must be verified by a real load before it is recommended.
    kv_unified: bool | None = None
    # Default sampler params, applied by llama-server when a request omits them.
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    # --- speculative decoding (b10425 flag names; gated on engine features) ---
    #: Which drafting strategy to launch with. StudioForge's own sentinel
    #: ``"auto"`` (the default) resolves at launch, in this order:
    #:
    #: 1. ``draft-mtp`` when the GGUF carries ``nextn_predict_layers >= 1`` --
    #:    the model's own multi-token-prediction head, no draft model, no extra
    #:    VRAM. Measured on Qwen3.8-27B Q5_K_S: 37.8 -> 50.7 tok/s (+34%) at
    #:    53% draft acceptance (DECISIONS.md D38).
    #: 2. ``draft-simple`` when a ``draft_model_id`` is set (the pre-WP20
    #:    behaviour).
    #: 3. ``ngram-mod`` for thinking and MoE models -- llama.cpp recommends it
    #:    for output that repeats itself (reasoning, code iteration). Measured
    #:    free on unseen prose (+0.4%, and it emits no drafts at all there).
    #: 4. ``none`` otherwise.
    #:
    #: Every value is distribution-preserving: speculative decoding proposes
    #: tokens and the full model verifies them, so the sampled distribution is
    #: unchanged. This is a speed knob with no quality cost.
    #: Any explicit value (``none``, ``draft-mtp``, ``ngram-mod``, or a comma
    #: list) is honoured verbatim -- and refused with a clear error if the
    #: active engine does not advertise it.
    spec_type: str = "auto"
    spec_draft_n_max: int | None = None
    spec_draft_n_min: int | None = None
    spec_draft_p_min: float | None = None
    draft_device_override: list[int] | None = None
    draft_ctx_size: int | None = None

    #: Per-model chat-template override (a Jinja file passed to
    #: ``--chat-template-file``). ``None`` means "use the template embedded in
    #: the GGUF", which is the default and the right answer almost always.
    #: The override exists because a *baked-in* template can be unusable: a
    #: template using Jinja's ``raise_exception`` that the engine cannot compile
    #: makes certain request shapes fail with a 400 and leaves the user no way
    #: out short of re-quantising the model. Never hardcode a template here --
    #: this is opt-in, per model, and validated at save time by
    #: :func:`validate_chat_template_file`.
    chat_template_file: Path | None = None

    # --- Tier 3: expert escape hatch -----------------------------------
    extra_flags: str = ""

    # --- adapters ------------------------------------------------------
    adapters: list[AdapterAttachment] = Field(default_factory=list)

    @field_validator("ctx_size")
    @classmethod
    def _positive_ctx(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("ctx_size must be positive")
        return v

    @field_validator("spec_type", mode="before")
    @classmethod
    def _spec_type_default(cls, v: Any) -> Any:
        # Rows saved before WP20 carry ``null`` here, and ``null`` meant "the
        # supervisor's own default", which is what "auto" now spells. Coercing
        # rather than widening the annotation keeps exactly one sentinel: a
        # settings object read back from SQLite must not resolve differently
        # from one built in code.
        if v is None or (isinstance(v, str) and not v.strip()):
            return "auto"
        return v

    @field_validator("ttl_s")
    @classmethod
    def _nonneg_ttl(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("ttl_s must be >= 0 (0 means never unload)")
        return v

    @field_validator("max_parallel_cap")
    @classmethod
    def _positive_parallel_cap(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("max_parallel_cap must be >= 1")
        return v

    @field_validator(
        "parallel",
        "batch_size",
        "ubatch_size",
        "threads",
        "threads_batch",
        "draft_ctx_size",
        "reasoning_budget",
        "spec_draft_n_max",
        "spec_draft_n_min",
    )
    @classmethod
    def _positive_counts(cls, v: int | None, info: Any) -> int | None:
        # These all become llama-server flags verbatim; a 0 or a negative
        # value dies as an opaque child exit instead of a message naming the
        # field. (reasoning_budget -1 is llama.cpp's "unlimited"; allowed.)
        if v is None:
            return v
        if info.field_name == "reasoning_budget" and v == -1:
            return v
        if v < 1:
            raise ValueError(f"{info.field_name} must be >= 1")
        return v

    @field_validator("main_gpu", "cache_reuse")
    @classmethod
    def _non_negative(cls, v: int | None, info: Any) -> int | None:
        if v is not None and v < 0:
            raise ValueError(f"{info.field_name} must be >= 0")
        return v

    @field_validator("rope_freq_base", "rope_freq_scale", "spec_draft_p_min")
    @classmethod
    def _positive_floats(cls, v: float | None, info: Any) -> float | None:
        if v is not None and v <= 0:
            raise ValueError(f"{info.field_name} must be > 0")
        return v

    @field_validator("device_override", "draft_device_override")
    @classmethod
    def _device_indices(cls, v: list[int] | None, info: Any) -> list[int] | None:
        if v is None:
            return v
        if any(index < 0 for index in v):
            raise ValueError(f"{info.field_name} entries must be CUDA indices >= 0")
        return v

    @field_validator("engine_tag")
    @classmethod
    def _plain_tag(cls, v: str | None) -> str | None:
        # Becomes a directory name under engines/; one path component only.
        if v is None:
            return None
        cleaned = v.strip()
        if not cleaned:
            return None
        if any(sep in cleaned for sep in ("/", "\\", "..", ":")) or cleaned in {".", ".."}:
            raise ValueError("engine_tag must be a plain build tag such as 'b10425'")
        return cleaned


class AdapterAttachment(BaseModel):
    """A LoRA adapter attached to a model at a given scale."""

    adapter_id: str
    scale: float = 1.0

    @field_validator("scale")
    @classmethod
    def _sane_scale(cls, v: float) -> float:
        if not -10.0 <= v <= 10.0:
            raise ValueError("adapter scale must be within [-10, 10]")
        return v


def validate_chat_template_file(value: Path | str | None) -> Path | None:
    """Check a ``chat_template_file`` override at *save* time.

    Deliberately a plain function rather than a pydantic ``field_validator``:
    the check touches the filesystem, and a validator would run on every
    hydration of stored settings too. A template file that was deleted after it
    was saved would then make the whole ``ModelSettings`` row invalid, and the
    registry would silently fall back to defaults -- turning a fixable "that
    file is gone" into an invisible behaviour change. Saving is the moment the
    user can act on the error, so that is where it is raised.

    Returns the resolved path (or ``None``), and raises ``ValueError`` with a
    message naming the path when the file is missing, is not a file, or cannot
    be read.
    """
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.exists():
        raise ValueError(f"chat_template_file {path} does not exist")
    if not path.is_file():
        raise ValueError(f"chat_template_file {path} is not a file")
    if not os.access(path, os.R_OK):
        raise ValueError(f"chat_template_file {path} is not readable")
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.read(1)
    except OSError as exc:
        raise ValueError(f"chat_template_file {path} could not be read: {exc}") from exc
    return path


#: Preset sampler fields applied to a request payload when (and only when) the
#: request omits them. ``repeat_penalty`` is handled separately because clients
#: may spell it ``repetition_penalty``; ``max_tokens`` because of its
#: ``max_completion_tokens`` alias.
PRESET_SAMPLER_FIELDS: tuple[str, ...] = ("temperature", "top_p", "top_k", "min_p")


class VirtualPreset(BaseModel):
    """Request-time behaviour a virtual model carries: the Ollama ``Modelfile``
    idea grafted onto virtual models.

    Everything here is applied **per request by the gateway**, never at
    ``llama-server`` launch. That distinction is load-bearing: because none of
    these fields changes the child's argv, any number of presets over one base
    can share a single instance -- which is the whole efficiency win over
    "one process per persona". Launch-time overrides (ctx_size, kv cache,
    adapters) live in :class:`ModelSettings` instead and *do* cost a dedicated
    instance.

    Every sampler default yields to an explicit request value; the system
    prompt is prepended, never replacing a client's own system message.
    """

    system_prompt: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    max_tokens: int | None = None

    @field_validator("temperature")
    @classmethod
    def _nonneg_temperature(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("temperature must be >= 0")
        return v

    @field_validator("top_p", "min_p")
    @classmethod
    def _probability(cls, v: float | None) -> float | None:
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError("must be within [0, 1]")
        return v

    @field_validator("top_k")
    @classmethod
    def _nonneg_top_k(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("top_k must be >= 0")
        return v

    @field_validator("repeat_penalty")
    @classmethod
    def _positive_penalty(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("repeat_penalty must be > 0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def _positive_max_tokens(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("max_tokens must be > 0")
        return v

    def is_empty(self) -> bool:
        """True when no field is set, so an all-null preset stores as absent."""
        return all(v is None for v in self.model_dump().values())

    def apply_to_payload(self, payload: dict[str, Any], *, chat: bool) -> None:
        """Fold this preset into an OpenAI request payload, in place.

        The system prompt is prepended as a leading ``system`` message; a
        client's own system message is kept after it, never discarded --
        silently dropping a client's instructions would make the same request
        behave differently against the base and the preset in a way the client
        cannot see. Sampler defaults fill only *absent* fields, including the
        alias spellings (``repetition_penalty``, ``max_completion_tokens``):
        an alias the client sent counts as the client having chosen.
        """
        if chat and self.system_prompt:
            messages = payload.get("messages")
            if isinstance(messages, list):
                payload["messages"] = [
                    {"role": "system", "content": self.system_prompt},
                    *messages,
                ]
        # An explicit JSON null means "unset" in OpenAI semantics, so it is
        # treated the same as an absent key: the preset default applies.
        for name in PRESET_SAMPLER_FIELDS:
            value = getattr(self, name)
            if value is not None and payload.get(name) is None:
                payload[name] = value
        if (
            self.repeat_penalty is not None
            and payload.get("repeat_penalty") is None
            and payload.get("repetition_penalty") is None
        ):
            payload["repeat_penalty"] = self.repeat_penalty
        if (
            self.max_tokens is not None
            and payload.get("max_tokens") is None
            and payload.get("max_completion_tokens") is None
        ):
            payload["max_tokens"] = self.max_tokens


class AdapterRecord(BaseModel):
    """A GGUF LoRA adapter tracked in the registry."""

    id: str
    name: str
    path: Path
    size_bytes: int = 0
    base_architecture: str | None = None
    base_model_hint: str | None = None
    publisher: str | None = None
    repo: str | None = None
    n_layer: int | None = None
    rank: int | None = None


class ModelCapabilities(BaseModel):
    vision: bool = False
    embedding: bool = False
    tools: bool = False
    thinking: bool = False
    multi_part: bool = False


# ---------------------------------------------------------------------------
# Registry records
# ---------------------------------------------------------------------------


class ModelRecord(BaseModel):
    """One logical model: a base GGUF (possibly sharded) plus optional mmproj.

    A *virtual* model (``is_virtual``) shares the base file of another record
    but carries its own adapter set/scales, which is how a LoRA combination is
    selectable by name through the OpenAI API.
    """

    id: str
    name: str
    kind: ModelKind = "chat"
    path: Path
    shards: list[Path] = Field(default_factory=list)
    mmproj_path: Path | None = None
    # Recorded at scan time so the planner needs no filesystem I/O to size
    # the projector (and so tests need no multi-hundred-MB fixture files).
    mmproj_bytes: int = 0
    size_bytes: int = 0
    quant: str = "unknown"
    publisher: str | None = None
    repo: str | None = None
    architecture: str = "unknown"
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    meta: GgufMeta | None = None
    settings: ModelSettings = Field(default_factory=ModelSettings)
    is_virtual: bool = False
    base_model_id: str | None = None
    #: Request-time preset (system prompt / sampler defaults); virtual only.
    preset: VirtualPreset | None = None
    added_at: float = Field(default_factory=time.time)
    last_used_at: float | None = None
    mtime: float = 0.0
    #: True when the newest scan could not re-read this model's file but the
    #: file is still there, so the record was carried over from the previous
    #: scan rather than dropped. A transient read error (a file mid-write, an
    #: antivirus lock, flaky I/O) must not make a configured model vanish from
    #: a client's catalogue.
    stale: bool = False
    #: Why the record is stale (the parse error), for the GUI/logs.
    stale_reason: str | None = None

    @property
    def all_files(self) -> list[Path]:
        files = list(self.shards) if self.shards else [self.path]
        if self.mmproj_path is not None:
            files.append(self.mmproj_path)
        return files

    def openai_dict(self) -> dict[str, Any]:
        """The ``GET /v1/models`` entry.

        Keeps OpenAI's required keys and adds a ``studioforge`` block plus the
        LM Studio-ish top-level hints (``type``/``quantization``) that some
        clients display. Extra keys are ignored by the openai client.
        """
        return {
            "id": self.id,
            "object": "model",
            "created": int(self.added_at),
            "owned_by": self.publisher or "studioforge",
            "type": "embeddings" if self.kind == "embedding" else "llm",
            "quantization": self.quant,
            "arch": self.architecture,
            "capabilities": [k for k, v in self.capabilities.model_dump().items() if v],
            "studioforge": {
                "kind": self.kind,
                "vision": self.capabilities.vision,
                "tools": self.capabilities.tools,
                "size_bytes": self.size_bytes,
                "n_ctx_train": self.meta.n_ctx_train if self.meta else None,
                "is_virtual": self.is_virtual,
                "base_model_id": self.base_model_id,
                "adapters": [a.model_dump() for a in self.settings.adapters],
                "pinned": self.settings.pinned,
                "stale": self.stale,
                "preset": (
                    self.preset.model_dump(exclude_none=True) if self.preset is not None else None
                ),
            },
        }


# ---------------------------------------------------------------------------
# GPU + planner
# ---------------------------------------------------------------------------


class GpuInfo(BaseModel):
    index: int
    name: str
    total_bytes: int
    free_bytes: int
    used_bytes: int = 0
    utilization_pct: float | None = None
    temperature_c: float | None = None
    compute_capability: tuple[int, int] | None = None

    @property
    def cc_str(self) -> str:
        if self.compute_capability is None:
            return "unknown"
        return f"{self.compute_capability[0]}.{self.compute_capability[1]}"

    @property
    def sm_arch(self) -> str:
        """CMake-style arch string, e.g. ``120`` for Blackwell sm_120."""
        if self.compute_capability is None:
            return ""
        return f"{self.compute_capability[0]}{self.compute_capability[1]}"


class VramProcess(BaseModel):
    """A process holding VRAM on one GPU, as reported by NVML.

    Exists so a rejected load can name *who* took the memory. On a box that
    also runs ComfyUI/training jobs, "insufficient VRAM" with no attribution is
    an unactionable message: the numbers are right and the user still cannot
    tell whether to close something or buy something.
    """

    gpu_index: int
    pid: int
    name: str = "unknown"
    used_bytes: int = 0
    #: True when the pid belongs to one of our own llama-server children, so a
    #: rejection can distinguish "your own models" from foreign contention.
    is_ours: bool = False

    def describe(self) -> str:
        return (
            f"{self.used_bytes / GB:.2f} GiB held by {self.name} "
            f"(pid {self.pid}) on CUDA{self.gpu_index}"
        )


class VramEstimate(BaseModel):
    """Breakdown of projected VRAM for one load, in bytes."""

    weights_bytes: int = 0
    kv_bytes: int = 0
    compute_bytes: int = 0
    mmproj_bytes: int = 0
    mmproj_compute_bytes: int = 0
    adapter_bytes: int = 0
    draft_weights_bytes: int = 0
    draft_kv_bytes: int = 0
    cuda_context_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return (
            self.weights_bytes
            + self.kv_bytes
            + self.compute_bytes
            + self.mmproj_bytes
            + self.mmproj_compute_bytes
            + self.adapter_bytes
            + self.draft_weights_bytes
            + self.draft_kv_bytes
            + self.cuda_context_bytes
        )

    def breakdown_mb(self) -> dict[str, float]:
        data = {k: v / MB for k, v in self.model_dump().items()}
        data["total"] = self.total_bytes / MB
        return data


class LoadPlan(BaseModel):
    """An accepted placement decision. GPU-only: n_gpu_layers is always all."""

    model_id: str
    devices: list[int]
    tensor_split: list[float] | None = None
    split_mode: SplitMode = "layer"
    main_gpu: int = 0
    ctx_size: int = 8192
    parallel: int = 1
    kv_cache_type: KvCacheType = "f16"
    kv_cache_type_v: KvCacheType = "f16"
    flash_attn: FlashAttn = "auto"
    estimate: VramEstimate = Field(default_factory=VramEstimate)
    per_gpu_bytes: dict[int, int] = Field(default_factory=dict)
    evict_model_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    # --- concurrency (DECISIONS.md D17) --------------------------------
    #: Slots the estimator judged this placement could sustain. Equal to
    #: :attr:`parallel` on an automatic plan; on an explicit one it simply
    #: repeats the number the caller pinned, because nothing was estimated.
    max_parallel: int = 1
    #: Which bound produced :attr:`max_parallel`.
    parallel_limited_by: ParallelBound = "explicit"
    #: Context each slot gets. Equal to :attr:`ctx_size` -- ``--ctx-size`` is
    #: the TOTAL across slots (D4), so the child is launched with
    #: ``ctx_per_slot * parallel``. Named separately because "context" means
    #: the per-conversation number to every client, and an API payload that
    #: says only ``ctx_size`` next to ``parallel`` is genuinely ambiguous.
    ctx_per_slot: int = 0
    #: Bytes of state one token of context costs, at this model's shape, KV
    #: type and :attr:`ctx_per_slot`. The single number every concurrency and
    #: context estimate divides by.
    #:
    #: This is the *effective* cost -- allocated bytes divided by the context
    #: they cover -- not the uniform ``n_layer * n_head_kv * head_dim``
    #: product. The two are identical for an ordinary model and differ by more
    #: than an order of magnitude for the rest: Gemma-4 31B holds a 1024-token
    #: window on five layers in six, so it costs ~80 KB/token against a uniform
    #: 1.9 MB, and Qwen3.5 caches only every fourth layer. It therefore depends
    #: on the context: a sliding window costs the same at 16k as at 262k, so
    #: the per-token average falls as the window grows. See
    #: ``planner.effective_kv_bytes_per_token``.
    kv_bytes_per_token: int = 0

    # --- launch-time feature resolution (DECISIONS.md D38) --------------
    #: What speculative decoding this launch actually asked the engine for::
    #:
    #:     {"type": "draft-mtp", "draft_n_max": 3, "reason": "...", "draft_model_id": None}
    #:
    #: ``None`` until the supervisor resolves it (the planner does not know the
    #: engine's feature list). ``{"type": "none", ...}`` is a real answer and
    #: means drafting was considered and declined, which is a different thing
    #: from "not yet decided" -- the catalog's `speculative` column needs to be
    #: able to tell those apart.
    speculative: dict[str, Any] | None = None
    #: Why :attr:`split_mode` is not what was asked for, if it is not. Set by
    #: the supervisor when ``auto`` (or an ineligible ``tensor``) is downgraded
    #: to ``layer``; also appended to :attr:`notes`.
    split_mode_reason: str | None = None
    #: How good this placement is, judged structurally at plan time:
    #: ``"optimal"`` (one card, or a same-generation split) or ``"degraded"``
    #: (a split across mixed compute generations -- measured ~half generation
    #: speed on this rig's 5090+3090 splits). ``None`` on plans from before
    #: the field existed. A ranking, not a promise: the benchmark history is
    #: the measured truth.
    placement_tier: Literal["optimal", "degraded"] | None = None

    @property
    def fits_single_gpu(self) -> bool:
        return len(self.devices) == 1

    @property
    def ctx_total(self) -> int:
        """What actually reaches ``--ctx-size``: per-slot context x slots (D4)."""
        return max(1, self.ctx_size) * max(1, self.parallel)


class LoadRejected(BaseModel):
    """A refusal to load, with the numbers and actionable next steps.

    Because there is no CPU fallback, a rejection is terminal for the current
    settings -- so it must always carry something the user can *do*.
    """

    model_id: str
    reason: str
    estimate: VramEstimate = Field(default_factory=VramEstimate)
    required_bytes: int = 0
    available_bytes: int = 0
    per_gpu_free: dict[int, int] = Field(default_factory=dict)
    max_ctx_that_fits: int | None = None
    #: When an explicit ``parallel`` above 1 was refused: the largest slot
    #: count that DOES fit at the requested context, or ``None`` if not even
    #: one does. The window outranks the second slot (D22), so this is the
    #: first lever an agent should reach for.
    max_parallel_that_fits: int | None = None
    suggestions: list[str] = Field(default_factory=list)
    #: Planner notes that also apply to a refusal (e.g. "you asked for more
    #: context than this model was trained for"). Same field name and meaning
    #: as :attr:`LoadPlan.notes` so the GUI renders both the same way.
    notes: list[str] = Field(default_factory=list)
    #: Processes holding VRAM when the refusal was computed. Empty when NVML
    #: cannot enumerate them (containers, WSL, older drivers).
    vram_holders: list[VramProcess] = Field(default_factory=list)
    #: ``[{"model_id": ..., "active_requests": N}]`` -- models the eviction
    #: ladder skipped because they are serving somebody (D36). This is the
    #: difference between "the box is full" and "the box is busy": the first
    #: needs a smaller load, the second needs a wait, and only the second has a
    #: ``retry_after_s``.
    busy_models: list[dict[str, Any]] = Field(default_factory=list)
    #: Seconds worth waiting before retrying, when a *busy* model is what stood
    #: in the way. ``None`` for every other refusal, because "try again later"
    #: is bad advice when nothing is going to change.
    retry_after_s: float | None = None

    def message(self) -> str:
        required_gb = self.required_bytes / GB
        available_gb = self.available_bytes / GB
        text = (
            f"Cannot load '{self.model_id}' entirely in VRAM: needs "
            f"{required_gb:.2f} GiB, {available_gb:.2f} GiB usable. {self.reason}"
        )
        if self.suggestions:
            text += " Suggestions: " + "; ".join(self.suggestions)
        return text


PlanResult = LoadPlan | LoadRejected


# ---------------------------------------------------------------------------
# Running instances
# ---------------------------------------------------------------------------


class InstanceInfo(BaseModel):
    """Public view of a loaded llama-server child."""

    model_id: str
    state: InstanceState
    port: int | None = None
    pid: int | None = None
    engine_tag: str | None = None
    plan: LoadPlan | None = None
    started_at: float | None = None
    last_activity_at: float | None = None
    ttl_s: int | None = None
    active_requests: int = 0
    total_requests: int = 0
    #: WHO asked for this load: ``"mcp:load_model"``, ``"jit:/v1/chat/completions"``,
    #: ``"gui"``, ``"autoload"``, ``"benchmark"``... On a box several clients
    #: share, "a 262144-token model appeared on three GPUs" is not a diagnosable
    #: event without a requester, and the 2026-08-19 log review could not tell
    #: an OpenClaw load from the GUI's (D36).
    loaded_by: str | None = None
    #: Load-priority tier (D46): 1 = the active chat model, 2 = a dispatched
    #: agent, 3 = background. Lower outranks higher when models compete for
    #: VRAM; a load that never said is background, so every pre-D46 instance
    #: behaves exactly as it always did.
    priority: int = 3
    restarts: int = 0
    last_error: str | None = None
    log_path: Path | None = None
    last_tokens_per_second: float | None = None
    #: The resolved speculative-decoding block for this child, mirroring
    #: :attr:`LoadPlan.speculative`. Carried on the instance too because that is
    #: what the Dashboard and ``GET /api/models`` render, and "is this model
    #: drafting?" is not answerable from ``/props`` -- the truthful runtime
    #: signals are ``/slots[].speculative`` (configured) and
    #: ``timings.draft_n`` / ``draft_n_accepted`` (actually working). See
    #: DECISIONS.md D38.
    speculative: dict[str, Any] | None = None

    @property
    def ttl_remaining_s(self) -> float | None:
        """Seconds until idle-unload, or None when pinned/not applicable."""
        if not self.ttl_s or self.state != "ready":
            return None
        if self.active_requests > 0:
            return float(self.ttl_s)
        last = self.last_activity_at or self.started_at
        if last is None:
            return None
        return max(0.0, self.ttl_s - (time.time() - last))


class EngineInfo(BaseModel):
    tag: str
    path: Path
    server_binary: Path
    variant: str = "unknown"  # e.g. "cuda-13.3", "source-local"
    version_string: str | None = None
    smoke_tested: bool = False
    smoke_tested_at: float | None = None
    active: bool = False
    installed_at: float = Field(default_factory=time.time)
    flags: list[str] = Field(default_factory=list)
    build_log: Path | None = None


class GpuLease(BaseModel):
    """A claim on specific CUDA devices that the planner honours (D43).

    Only the models in ``model_ids`` may be planned onto ``devices`` while the
    lease stands; an empty ``model_ids`` means *nobody* may -- the cards are
    held for something outside this server (a ComfyUI run, a training job).
    ``idle_ttl_s`` is the safety net: a lease nobody has touched for that long
    is released by the sweep, so a crashed benchmark or a forgotten reservation
    cannot hold a card forever. ``None`` means held until released.
    """

    id: str
    devices: list[int]
    holder: str
    model_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    created_at: float
    last_activity_at: float
    idle_ttl_s: float | None = 3600.0

    @property
    def idle_s(self) -> float:
        return max(0.0, time.time() - self.last_activity_at)

    @property
    def expires_at(self) -> float | None:
        if self.idle_ttl_s is None:
            return None
        return self.last_activity_at + self.idle_ttl_s


class ServerStatus(BaseModel):
    version: str
    uptime_s: float
    gpus: list[GpuInfo]
    system_ram_total_bytes: int = 0
    system_ram_used_bytes: int = 0
    loaded: list[InstanceInfo] = Field(default_factory=list)
    model_count: int = 0
    #: Every process NVML reports as holding VRAM, ours and foreign alike, so
    #: contention with (say) a ComfyUI run on the same box is visible before a
    #: load is attempted rather than only after it is refused.
    vram_processes: list[VramProcess] = Field(default_factory=list)
    engine: EngineInfo | None = None
    queue_depth: int = 0
    active_downloads: int = 0
    draining: bool = False
    #: Standing GPU leases (D43): which cards are held, by whom, for which
    #: models, and when the sweep will release an idle one.
    leases: list[GpuLease] = Field(default_factory=list)


# Resolve forward references (ModelSettings <-> AdapterAttachment).
ModelSettings.model_rebuild()
ModelRecord.model_rebuild()
