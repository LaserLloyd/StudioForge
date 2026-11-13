"""Several instances of one model, under GPU leases (D91).

One model id used to mean one llama-server child. Two holders of the same model
therefore fought over one instance: on 2026-11-14 at 02:10 a second holder
leased other cards naming the model a first holder was serving three streams
from, its unload cut those streams, and the first holder's next request
reloaded the id on the default context ladder.

This module is the vocabulary of the fix; :class:`~studioforge.core.manager.
ModelManager` does the work.

* **Ids** follow LM Studio's instance identifiers: an instance created without
  a tag is ``<model>:<n>``, ``n`` the smallest free integer from 2 (the base
  resident is the plain id, the first copy ``:2``); a client may ask for a tag
  instead, ``<model>@<tag>`` -- the existing virtual-copy convention. Either
  works in the ``model`` field of every OpenAI-shape and native route.
* **Parity is a server guarantee.** An instance's record is a deep copy of its
  base's settings; ``launch`` may change only the context, the slots and the KV
  types (:data:`LAUNCH_KEYS`). Placement follows the instance's cards.
* **Holder identity** (:class:`Caller`) decides whose instance a request,
  a load or an unload means: the lease id (``X-SF-Lease`` or a body
  ``lease_id``), else the ``X-SF-Client`` label matched exactly to a lease's
  holder, else the key that took the lease. A key shared by two holders never
  makes one of them the other.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from studioforge.errors import BadRequestError, StudioForgeError
from studioforge.types import GpuLease, InstanceInfo, LeaseInstance, ModelRecord

#: The ``launch`` keys a client may set on an instance (brief item 3): its own
#: context per slot, slot count (exact, or a floor and a ceiling) and KV cache
#: types. Everything else is the base's, copied.
LAUNCH_KEYS: tuple[str, ...] = (
    "ctx_size",
    "parallel",
    "min_slots",
    "max_slots",
    "kv_cache_type",
    "kv_cache_type_v",
)
#: Spellings accepted for a launch key: LM Studio's and the llama-server
#: router's ``context_length`` (tokens per slot, as ``ctx_size`` is here).
LAUNCH_KEY_ALIASES: dict[str, str] = {"context_length": "ctx_size"}

#: What a tag may look like: it becomes part of a model id and a URL path.
INSTANCE_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$")
#: ``<model>:<n>`` -- the numbered instance id.
NUMBERED_ID_RE = re.compile(r"^(?P<base>.+):(?P<n>[0-9]{1,6})$")
#: The most copies one ``instances[]`` entry may ask for.
MAX_INSTANCE_COUNT = 16
#: The first number a copy gets: the base resident is the plain id (``:1``).
FIRST_INSTANCE_NUMBER = 2

#: Headers that name the caller's lease (D91, 8a): ``X-SF-Lease`` is the
#: canonical one; ``X-SF-Lease-Id`` is what this server already sends a holder
#: in a vacate request, accepted for symmetry.
LEASE_HEADERS: tuple[str, ...] = ("x-sf-lease", "x-sf-lease-id")

#: Settings an instance does not copy from its base, and why: the placement
#: fields follow the instance's own cards (``device_override`` is them), and
#: a pin is a lifecycle the lease owns.
PLACEMENT_SETTINGS: tuple[str, ...] = (
    "device_override",
    "allowed_devices",
    "main_gpu",
    "draft_device_override",
    "prefer_single_gpu",
)
LIFECYCLE_SETTINGS: tuple[str, ...] = ("pinned",)
#: The launch keys that are also ``ModelSettings`` fields (the slot floor and
#: ceiling are load arguments, not settings).
SETTINGS_LAUNCH_KEYS: tuple[str, ...] = ("ctx_size", "parallel", "kv_cache_type", "kv_cache_type_v")


# ---------------------------------------------------------------------------
# Ids
# ---------------------------------------------------------------------------


def numbered_id(base_id: str, number: int) -> str:
    """``<base>:<n>`` -- LM Studio's instance identifier."""
    return f"{base_id}:{int(number)}"


def tagged_id(base_id: str, tag: str) -> str:
    """``<base>@<tag>`` -- the virtual-copy convention, for a client-chosen name."""
    return f"{base_id}@{tag}"


def parse_numbered_id(model_id: str) -> tuple[str, int] | None:
    """``("X", 2)`` for ``"X:2"``; ``None`` when the id is not a numbered instance id."""
    match = NUMBERED_ID_RE.match(model_id or "")
    if match is None:
        return None
    return match.group("base"), int(match.group("n"))


def next_free_number(base_id: str, taken: Any, *, start: int = FIRST_INSTANCE_NUMBER) -> int:
    """The smallest ``n >= start`` whose ``<base>:<n>`` ``taken(id)`` says is free.

    Gaps are reused: with ``:2`` released and ``:3`` standing, the next copy is
    ``:2`` again.
    """
    number = max(FIRST_INSTANCE_NUMBER, int(start))
    while taken(numbered_id(base_id, number)):
        number += 1
    return number


# ---------------------------------------------------------------------------
# The request shape
# ---------------------------------------------------------------------------


@dataclass
class InstanceRequest:
    """One ``instances[]`` entry of a lease request, shape-checked (D91).

    Triton's ``instance_group`` shape (``count`` copies on the listed ``gpus``),
    LM Studio's id grammar. ``gpus`` defaults to the lease's cards; ``load``
    to ``true`` (the reply then reports each instance as actually loaded).
    """

    model: str
    count: int = 1
    gpus: list[int] | None = None
    tag: str | None = None
    launch: dict[str, Any] = field(default_factory=dict)
    load: bool = True
    origin: str = "requested"
    index: int = 0


def _launch_error(message: str, param: str) -> BadRequestError:
    return BadRequestError(message, param=param, code="launch_key_not_allowed")


def normalise_launch(raw: Any, *, where: str = "launch") -> dict[str, Any]:
    """The ``launch`` object with aliases folded in; any other key is a 400 (item 3).

    Values are type-checked here; the ranges (a context the model was trained
    for, a slot count the cap allows, a KV type the engine knows) are the
    load's own checks, run by the manager before anything is created.
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise BadRequestError(
            f"{where} must be an object of launch settings", param=where, code="invalid_request"
        )
    out: dict[str, Any] = {}
    for key, value in raw.items():
        canonical = LAUNCH_KEY_ALIASES.get(str(key), str(key))
        if canonical not in LAUNCH_KEYS:
            raise _launch_error(
                f"{where}.{key} is not a launch setting an instance may change. An instance may "
                f"set {', '.join(LAUNCH_KEYS)} (context_length is accepted for ctx_size); "
                f"everything else -- the file, chat template, sampling, adapters, engine build, "
                f"reasoning format -- is copied from the base model, because parity with the "
                f"base is the server's guarantee. Change the base model's settings instead.",
                f"{where}.{key}",
            )
        if value is None:
            continue
        if canonical in out and out[canonical] != value:
            raise BadRequestError(
                f"{where} gives the context twice with different values (ctx_size and "
                f"context_length); send one",
                param=f"{where}.{key}",
            )
        if canonical in ("ctx_size", "parallel", "min_slots", "max_slots"):
            if isinstance(value, bool) or not isinstance(value, int) or int(value) < 1:
                raise BadRequestError(
                    f"{where}.{key} must be a positive whole number (got {value!r})",
                    param=f"{where}.{key}",
                )
            value = int(value)
        elif not isinstance(value, str) or not value.strip():
            raise BadRequestError(
                f"{where}.{key} must be a KV cache type such as 'f16' or 'q8_0' (got {value!r})",
                param=f"{where}.{key}",
            )
        else:
            value = value.strip()
        out[canonical] = value
    floor, ceiling = out.get("min_slots"), out.get("max_slots")
    if floor is not None and ceiling is not None and floor > ceiling:
        raise BadRequestError(
            f"{where}.min_slots ({floor}) is above {where}.max_slots ({ceiling})",
            param=f"{where}.min_slots",
        )
    slots = out.get("parallel")
    if slots is not None and (
        (floor is not None and slots < floor) or (ceiling is not None and slots > ceiling)
    ):
        raise BadRequestError(
            f"{where}.parallel ({slots}) lies outside min_slots/max_slots; send parallel alone "
            f"for an exact slot count, or the bounds alone to let the server size it",
            param=f"{where}.parallel",
        )
    return out


def parse_instance_requests(raw: Any) -> list[InstanceRequest]:
    """``instances`` from a lease request, shape-checked; ``[]`` for none.

    ``model_id`` is accepted as a deprecated alias of ``model`` (B.1). An entry
    with an unknown key is a 400 naming it: a typo in ``launch`` must not
    silently become a load at the base's settings.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise BadRequestError(
            'instances must be a list of {"model": "<id>", "count"?, "gpus"?, "tag"?, '
            '"launch"?, "load"?}',
            param="instances",
        )
    known = {"model", "model_id", "count", "gpus", "tag", "launch", "load"}
    out: list[InstanceRequest] = []
    for index, entry in enumerate(raw):
        where = f"instances[{index}]"
        if isinstance(entry, str):
            entry = {"model": entry}
        if not isinstance(entry, Mapping):
            raise BadRequestError(f"{where} must be an object", param=where)
        unknown = sorted(set(entry) - known)
        if unknown:
            raise BadRequestError(
                f"{where} has unknown field(s) {', '.join(unknown)}; an entry takes model, "
                f"count, gpus, tag, launch and load",
                param=f"{where}.{unknown[0]}",
            )
        model = entry.get("model") or entry.get("model_id")
        if not isinstance(model, str) or not model.strip():
            raise BadRequestError(f"{where}.model is required: the model to copy", param=where)
        count = entry.get("count", 1)
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not (1 <= count <= MAX_INSTANCE_COUNT)
        ):
            raise BadRequestError(
                f"{where}.count must be 1..{MAX_INSTANCE_COUNT} (got {count!r})",
                param=f"{where}.count",
            )
        gpus = entry.get("gpus")
        if gpus is not None:
            if (
                not isinstance(gpus, list)
                or not gpus
                or any(isinstance(g, bool) or not isinstance(g, int) for g in gpus)
            ):
                raise BadRequestError(
                    f"{where}.gpus must be a non-empty list of CUDA indices (the lease's cards "
                    f"by default)",
                    param=f"{where}.gpus",
                )
            gpus = sorted({int(g) for g in gpus})
        tag = entry.get("tag")
        if tag is not None:
            if not isinstance(tag, str) or not INSTANCE_TAG_RE.match(tag.strip()):
                raise BadRequestError(
                    f"{where}.tag must be 1-48 letters, digits, '.', '_' or '-', starting with a "
                    f"letter or digit (got {tag!r})",
                    param=f"{where}.tag",
                )
            tag = tag.strip()
        load = entry.get("load", True)
        if not isinstance(load, bool):
            raise BadRequestError(f"{where}.load must be true or false", param=f"{where}.load")
        out.append(
            InstanceRequest(
                model=model.strip(),
                count=int(count),
                gpus=gpus,
                tag=tag,
                launch=normalise_launch(entry.get("launch"), where=f"{where}.launch"),
                load=load,
                index=index,
            )
        )
    return out


def instance_ids_for(
    request: InstanceRequest, base_id: str, taken: Any, reserved: set[str]
) -> list[str]:
    """The ids ``request`` would create: tagged, or numbered with gaps reused.

    ``taken(id)`` answers for everything already standing (ids, aliases);
    ``reserved`` holds the ids earlier entries of the same call claimed, and is
    extended. A tag with ``count > 1`` numbers its copies ``<tag>1``,
    ``<tag>2``, ... (B.1).
    """
    out: list[str] = []
    if request.tag is not None:
        tags = (
            [request.tag]
            if request.count == 1
            else [f"{request.tag}{i}" for i in range(1, request.count + 1)]
        )
        for tag in tags:
            out.append(tagged_id(base_id, tag))
    else:
        for _ in range(request.count):
            number = next_free_number(
                base_id, lambda candidate: bool(taken(candidate)) or candidate in reserved
            )
            candidate = numbered_id(base_id, number)
            reserved.add(candidate)
            out.append(candidate)
            continue
    for candidate in out:
        reserved.add(candidate)
    return out


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def ephemeral_record(
    source: ModelRecord,
    *,
    instance_id: str,
    lease_id: str,
    devices: Sequence[int],
    launch: Mapping[str, Any],
    created_at: float | None = None,
) -> ModelRecord:
    """The registry record of one instance: ``source``, copied, on ``devices``.

    A deep copy of the base's record and settings -- file, shards, projector,
    chat template, sampling, adapters, engine pin, reasoning format, draft, flags
    -- with only :data:`LAUNCH_KEYS` that are settings applied from ``launch``,
    the placement fields following the instance's cards, and no pin (its life
    is its lease's).
    """
    cards = [int(d) for d in devices]
    base_settings = source.settings
    updates: dict[str, Any] = {
        "device_override": cards,
        "allowed_devices": None,
        "prefer_single_gpu": None,
        "pinned": False,
    }
    if base_settings.main_gpu is not None and base_settings.main_gpu not in cards:
        updates["main_gpu"] = None
    draft_cards = base_settings.draft_device_override
    if draft_cards is not None and not set(draft_cards) <= set(cards):
        updates["draft_device_override"] = None
    for key in SETTINGS_LAUNCH_KEYS:
        if launch.get(key) is not None:
            updates[key] = launch[key]
    settings = base_settings.model_copy(deep=True).model_copy(update=updates)
    stamp = {} if created_at is None else {"added_at": float(created_at)}
    return source.model_copy(
        deep=True,
        update={
            "id": instance_id,
            "name": instance_id,
            "settings": settings,
            "is_virtual": True,
            "base_model_id": source.id,
            "ephemeral": True,
            "lease_id": lease_id,
            "last_used_at": None,
            "stale": False,
            "stale_reason": None,
            **stamp,
        },
    )


def parity_view(settings: Any, *, launch_keys: Iterable[str] = LAUNCH_KEYS) -> dict[str, Any]:
    """A settings dump without the fields an instance may differ in (for parity checks)."""
    data = settings.model_dump(mode="json")
    for key in (*launch_keys, *PLACEMENT_SETTINGS, *LIFECYCLE_SETTINGS):
        data.pop(key, None)
    return data


def loaded_shape(instance: InstanceInfo) -> dict[str, Any] | None:
    """What an instance ACTUALLY runs with, from its plan; ``None`` without one."""
    plan = instance.plan
    if plan is None:
        return None
    return {
        "devices": [int(d) for d in plan.devices],
        "ctx_per_slot": int(plan.ctx_per_slot or plan.ctx_size),
        "parallel": int(plan.parallel),
        "kv_cache_type": plan.kv_cache_type,
        "kv_cache_type_v": plan.kv_cache_type_v or plan.kv_cache_type,
        "kv_unified": bool(plan.kv_unified),
        "engine_tag": instance.resolved_engine_tag or instance.engine_tag,
    }


_STATE_WORDS = {"ready": "loaded", "loading": "loading", "unloading": "unloading"}


def instance_row(
    lease: GpuLease,
    spec: LeaseInstance,
    live: InstanceInfo | None = None,
    *,
    live_known: bool = True,
) -> dict[str, Any]:
    """The reply / listing row of one instance (runner requirement 4).

    ``model`` is the exact string a client puts in a chat request. The shape
    fields are the running child's when it is up, else the shape it last ran
    with, else what ``launch`` asked for. ``state`` is the llama-server router
    word (``loaded`` | ``loading`` | ``unloaded``); ``None`` when the caller had
    no supervisor to ask (``live_known=False``).
    """
    shape = (loaded_shape(live) if live is not None else None) or spec.loaded_spec or {}
    launch = spec.launch
    if not live_known:
        state: str | None = None
    elif live is None:
        state = "unloaded"
    else:
        state = _STATE_WORDS.get(live.state, "unloaded")
    ctx = shape.get("ctx_per_slot") or launch.get("ctx_size")
    return {
        "id": spec.id,
        "model": spec.id,
        "base_model_id": spec.base_model_id,
        "lease_id": lease.id,
        "holder": lease.holder,
        "devices": list(shape.get("devices") or spec.devices),
        "ctx_per_slot": ctx,
        "context_length": ctx,
        "parallel": shape.get("parallel") or launch.get("parallel"),
        "kv_cache_type": shape.get("kv_cache_type") or launch.get("kv_cache_type"),
        "kv_cache_type_v": shape.get("kv_cache_type_v") or launch.get("kv_cache_type_v"),
        "state": state,
        "engine_tag": shape.get("engine_tag"),
        "tag": spec.tag,
        "origin": spec.origin,
        "launch": dict(launch),
        "created_at": spec.created_at,
        "ephemeral": True,
    }


# ---------------------------------------------------------------------------
# Holder identity (8a)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Caller:
    """Who a request says it is, for lease ownership decisions (D91, 8a).

    ``lease_ids``: the leases it names (``X-SF-Lease`` / body ``lease_id``).
    ``client``: its ``X-SF-Client`` label, exactly as sent. ``key``: the scoped
    key it authenticated with (``key:<id>``). ``admin``: D32's test. ``peer``:
    its address, which the in-flight records name when it sent no label.
    """

    lease_ids: tuple[str, ...] = ()
    client: str | None = None
    key: str | None = None
    admin: bool = False
    peer: str | None = None

    @property
    def anonymous(self) -> bool:
        return not (self.lease_ids or self.client or self.key)

    @property
    def label(self) -> str | None:
        """What this caller's in-flight records are stamped with: the label, else the peer."""
        return self.client or self.peer


def _same(left: str | None, right: str | None) -> bool:
    return bool(left) and bool(right) and str(left).strip().lower() == str(right).strip().lower()


def names_other_holder(caller: Caller, lease: GpuLease, leases: Iterable[GpuLease]) -> bool:
    """Whether ``caller`` identifies as someone OTHER than ``lease``'s holder (8a).

    True when it names another lease, or sends a label that is not this
    holder's and is either another standing lease's holder exactly or of a
    different family (a label that matches no standing holder but shares the
    family -- ``crucibleforge`` for ``crucibleforge-judge`` -- is not a claim
    to be someone else, and the D55 rules decide as before). Then a key the
    two share proves nothing: every bench run on one client shares one
    inference key, and a key match alone made holder B holder A on 2026-11-14.
    """
    from studioforge.core.leases import holder_family

    if caller.lease_ids and lease.id not in caller.lease_ids:
        return True
    label = (caller.client or "").strip()
    if not label or _same(label, lease.holder):
        return False
    if any(_same(label, other.holder) for other in leases if other.id != lease.id):
        return True
    return holder_family(label) != holder_family(lease.holder)


def claims(caller: Caller | None, lease: GpuLease, leases: Iterable[GpuLease]) -> bool:
    """Whether ``caller`` identifies as ``lease``'s holder, for routing and targeting.

    The lease id it names, else its label equal to the holder (exactly, case
    aside), else -- when its label is no standing holder's -- the key that took
    the lease. Not a proof for a destructive act on its own: the routes add
    D55's proof (``lease_holder_may_act``) before anything is taken down.
    """
    if caller is None or caller.anonymous:
        return False
    standing = list(leases)
    if caller.lease_ids:
        return lease.id in caller.lease_ids
    if caller.client:
        if _same(caller.client, lease.holder):
            return True
        if any(_same(caller.client, other.holder) for other in standing):
            return False
    return bool(caller.key and lease.holder_key and caller.key == lease.holder_key)


def other_clients(instance: InstanceInfo | None, caller: Caller | None) -> list[str]:
    """In-flight clients on ``instance`` that are not ``caller`` (``in-process`` for none)."""
    if instance is None or instance.active_requests <= 0:
        return []
    mine = caller.label if caller is not None else None
    names: list[str] = []
    for entry in instance.in_flight:
        who = entry.client or "in-process"
        if mine and _same(who, mine):
            continue
        if who not in names:
            names.append(who)
    if not names and not instance.in_flight and instance.active_requests > 0 and not mine:
        names.append("unknown")
    return names


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def instance_not_found(
    model_id: str, *, released: Mapping[str, Any] | None = None
) -> StudioForgeError:
    """404 ``instance_not_found``: an instance id with no instance behind it (item 7).

    Never a reload on the default ladder: the id named a copy that belonged to
    a lease, and the lease is the only thing that may bring one back.
    """
    details: dict[str, Any] = {"instance_id": model_id}
    if released:
        details.update(
            {
                "released_at": released.get("at"),
                "lease_id": released.get("lease_id"),
                "reason": released.get("why"),
            }
        )
        message = (
            f"instance '{model_id}' is gone: its lease {released.get('lease_id')} was "
            f"{released.get('why')}. An instance lives and dies with its lease; take a new "
            f"lease with instances[] to get one again."
        )
    else:
        message = (
            f"no instance '{model_id}' is standing. Instance ids (<model>:<n>, <model>@<tag>) "
            f"exist only while the lease that created them stands; GET /api/leases lists them."
        )
    error = StudioForgeError(
        message, code="instance_not_found", status_code=404, param="model", details=details
    )
    error.error_type = "invalid_request_error"
    return error


def instance_released(model_id: str, released: Mapping[str, Any]) -> StudioForgeError:
    """410 ``instance_released``: a request in flight when its instance's lease ended."""
    return StudioForgeError(
        f"instance '{model_id}' was released with its lease {released.get('lease_id')} "
        f"({released.get('why')}) while this request was in flight; the request did not "
        f"complete. Take a new lease to run on a copy again.",
        code="instance_released",
        status_code=410,
        details={
            "instance_id": model_id,
            "lease_id": released.get("lease_id"),
            "reason": released.get("why"),
            "released_at": released.get("at"),
        },
    )


def instance_held(
    model_id: str, lease: GpuLease | None, *, clients: Sequence[str] = (), what: str
) -> StudioForgeError:
    """409 ``instance_held``: the instance belongs to another holder (items 2, 8)."""
    from studioforge.errors import LeaseConflictError

    if lease is not None:
        whose = f"lease {lease.id} ({lease.holder}) holds it on CUDA {sorted(lease.devices)}"
    else:
        whose = f"it is serving {', '.join(clients)}"
    return LeaseConflictError(
        f"Refusing {what}: '{model_id}' belongs to another holder -- {whose}. A second holder "
        f"never replaces the first: take a lease naming the model (you get your own copy, "
        f"<model>:<n>), or send force=true with an admin credential on the exact instance id.",
        code="instance_held",
        param="model",
        details={
            "code": "instance_held",
            "model_id": model_id,
            "lease_id": lease.id if lease is not None else None,
            "holder": lease.holder if lease is not None else None,
            "in_flight_clients": list(clients),
        },
    )
