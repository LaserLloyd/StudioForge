"""The tenant registry and the VRAM Tenant Protocol's server half (D77, D79).

Pinned here, layer by layer: the URL guard that decides which hosts this server
will ever call for a tenant (loopback, the tailnet range, an operator-listed
host; a DNS name is never resolved); the config section and its validation;
registration, merge with a declared entry, deregistration; health polling over
a fake transport with its state words and the bounded summary; the warm relay
and its two refusals; the REST routes behind the same D32 gate as the lease
routes, with URLs withheld from a LAN reader; and ``/api/status`` carrying the
rows. Nothing here starts a tenant, because nothing in the product does.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from studioforge.config import Config, TenantEntry, TenantsConfig
from studioforge.core.leases import LeaseBook
from studioforge.core.tenants import (
    PROTOCOL_VERSION,
    TenantRecord,
    TenantRegistry,
    tenant_view,
)
from studioforge.errors import BadRequestError, TenantNotFoundError, TenantUnreachableError
from studioforge.netguard import (
    TenantUrlError,
    host_allowed,
    host_is_loopback,
    host_is_tailnet,
    validate_tenant_url,
)
from tests.unit.test_catalog_routes import app as app  # noqa: PLC0414 - fixture re-export

LOOPBACK = ("127.0.0.1", 50000)
LAN = ("192.168.1.50", 5000)
SPEECH_HEALTH = "http://127.0.0.1:1236/health"
SPEECH_WARM = "http://127.0.0.1:1236/v1/speech/warm"
SPEECH_VACATE = "http://127.0.0.1:1236/v1/speech/vacate"


# ---------------------------------------------------------------------------
# The URL guard (SSRF)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "localhost", "LOCALHOST", "::1", "127.1", "0x7f000001", "2130706433", "[::1]"],
)
def test_every_spelling_of_loopback_is_loopback(host: str) -> None:
    assert host_is_loopback(host)
    assert host_allowed(host)


@pytest.mark.parametrize("host", ["100.64.0.1", "100.100.7.9", "100.127.255.254"])
def test_the_tailnet_range_is_allowed(host: str) -> None:
    assert host_is_tailnet(host) and host_allowed(host)


@pytest.mark.parametrize(
    "host",
    ["100.63.255.255", "100.128.0.1", "192.168.1.50", "10.0.0.7", "8.8.8.8", "169.254.169.254"],
)
def test_other_addresses_are_refused_unless_listed(host: str) -> None:
    """169.254.169.254 is the cloud metadata address every SSRF writeup names."""
    assert not host_allowed(host)
    assert host_allowed(host, allowed_hosts=[host])


def test_a_dns_name_is_never_resolved_only_matched_verbatim() -> None:
    assert not host_allowed("rig.example.internal")
    assert host_allowed("rig.example.internal", allowed_hosts=["RIG.example.internal."])
    assert not host_allowed("evil.example", allowed_hosts=["rig.example.internal"])


@pytest.mark.parametrize(
    "url",
    [
        "",
        "ftp://127.0.0.1/x",
        "http:///health",
        "http://user:pw@127.0.0.1/health",
        "http://127.0.0.1:99999/health",
        "http://127.0.0.1/he alth",
        "http://192.168.1.50:1236/health",
        "http://169.254.169.254/latest/meta-data",
        "http://" + "a" * 600,
    ],
)
def test_validate_tenant_url_refuses_the_shapes_and_hosts_it_must(url: str) -> None:
    with pytest.raises(TenantUrlError):
        validate_tenant_url(url, field="health_url")


def test_validate_tenant_url_accepts_loopback_tailnet_and_listed_hosts() -> None:
    assert validate_tenant_url(" " + SPEECH_HEALTH + " ", field="health_url") == SPEECH_HEALTH
    assert validate_tenant_url("https://100.64.0.9:8700/health", field="health_url")
    assert validate_tenant_url(
        "http://nas.example.internal:1236/health",
        field="health_url",
        allowed_hosts=["nas.example.internal"],
    )


# ---------------------------------------------------------------------------
# The config section
# ---------------------------------------------------------------------------


def test_the_default_config_has_an_empty_registry_and_the_poller_knobs() -> None:
    config = Config(data_dir="/tmp/sf-tenants")
    assert config.tenants.registry == []
    assert config.tenants.health_interval_s == 30.0
    assert config.tenants.health_timeout_s == 2.0
    assert config.tenants.allowed_hosts == []


def test_a_declared_module_round_trips_with_its_urls_validated() -> None:
    section = TenantsConfig(
        registry=[
            TenantEntry(
                name="speechforge",
                kind="module",
                health_url=SPEECH_HEALTH,
                warm_url=SPEECH_WARM,
                default_priority=2,
                notes="speech-to-text and text-to-speech on one card",
            )
        ]
    )
    entry = section.registry[0]
    assert entry.effective_family == "speechforge"
    assert entry.warm_url == SPEECH_WARM


@pytest.mark.parametrize(
    "entry",
    [
        {"name": "SpeechForge"},
        {"name": "speechforge", "default_priority": 4},
        {"name": "speechforge", "health_url": "http://192.168.1.50:1236/health"},
        {"name": "clawforge2", "kind": "tenant", "warm_url": SPEECH_WARM},
        {"name": "speechforge", "family": "Speech Forge"},
    ],
)
def test_the_config_refuses_a_bad_entry(entry: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        TenantsConfig(registry=[TenantEntry(**entry)])


def test_the_config_refuses_a_name_declared_twice() -> None:
    with pytest.raises(ValueError):
        TenantsConfig(registry=[TenantEntry(name="a"), TenantEntry(name="a")])


def test_an_operator_listed_host_widens_the_guard_for_config_urls() -> None:
    section = TenantsConfig(
        registry=[TenantEntry(name="nas", health_url="http://nas.example.internal:1236/health")],
        allowed_hosts=["nas.example.internal"],
    )
    assert section.registry[0].health_url is not None


# ---------------------------------------------------------------------------
# The registry: declared rows, registration, merge, deregistration
# ---------------------------------------------------------------------------


def _declared() -> TenantsConfig:
    return TenantsConfig(
        registry=[
            TenantEntry(
                name="speechforge",
                kind="module",
                health_url=SPEECH_HEALTH,
                warm_url=SPEECH_WARM,
                default_priority=2,
            ),
            TenantEntry(name="clawforge2", health_url="http://127.0.0.1:8700/health"),
        ]
    )


def test_declared_tenants_appear_unregistered_and_unpolled() -> None:
    registry = TenantRegistry(_declared())
    names = [record.name for record in registry.all()]
    assert names == ["clawforge2", "speechforge"]
    speech = registry.require("speechforge")
    assert speech.source == "config" and speech.kind == "module"
    assert speech.registered is False and speech.health.state == "unpolled"
    with pytest.raises(TenantNotFoundError):
        registry.require("nobody")


def test_a_registration_under_a_declared_name_keeps_the_operators_identity() -> None:
    """The program may fill in where it can be reached; it may not promote itself."""
    registry = TenantRegistry(_declared())
    record = registry.register(
        {
            "name": "speechforge",
            "kind": "tenant",  # ignored: the operator said module
            "version": "0.9.1",
            "protocol": PROTOCOL_VERSION,
            "vacate_url": SPEECH_VACATE,
            "capabilities": ["stt", "tts", "STT"],
            "default_priority": 1,  # ignored for a declared row
        },
        peer="127.0.0.1",
        now=1000.0,
    )
    assert record.kind == "module" and record.source == "config"
    assert record.default_priority == 2
    assert record.version == "0.9.1" and record.capabilities == ["stt", "tts"]
    assert record.vacate_url == SPEECH_VACATE and record.warm_url == SPEECH_WARM
    assert record.registered and record.registered_at == 1000.0 and record.last_seen == 1000.0
    assert record.peer == "127.0.0.1"


def test_a_new_name_registers_with_what_it_says_and_deregisters_away() -> None:
    registry = TenantRegistry(TenantsConfig())
    record = registry.register(
        {
            "name": "dispatch",
            "family": "dispatch",
            "kind": "tenant",
            "health_url": "http://100.64.0.5:7000/health",
            "default_priority": 2,
            "notes": "the voice front end",
        },
        now=5.0,
    )
    assert record.source == "registered" and record.default_priority == 2
    assert len(registry) == 1
    gone = registry.deregister("dispatch", now=9.0)
    assert gone.registered is False and gone.deregistered_at == 9.0
    assert len(registry) == 0


def test_deregistering_a_declared_tenant_keeps_the_row_but_forgets_the_program() -> None:
    registry = TenantRegistry(_declared())
    registry.register({"name": "speechforge", "version": "1.0", "vacate_url": SPEECH_VACATE})
    record = registry.deregister("speechforge", now=42.0)
    assert registry.get("speechforge") is record
    assert record.registered is False and record.version is None
    assert record.vacate_url is None and record.deregistered_at == 42.0


@pytest.mark.parametrize(
    "payload, param",
    [
        ({"name": "Bad Name"}, "name"),
        ({"name": "x", "protocol": "vt.9"}, "protocol"),
        ({"name": "x", "kind": "daemon"}, "kind"),
        ({"name": "x", "version": "has space"}, "version"),
        ({"name": "x", "default_priority": 0}, "default_priority"),
        ({"name": "x", "capabilities": "stt"}, "capabilities"),
        ({"name": "x", "capabilities": ["not ok!"]}, "capabilities"),
        ({"name": "x", "health_url": "http://192.168.1.50/health"}, "health_url"),
        ({"name": "x", "vacate_url": "http://169.254.169.254/"}, "vacate_url"),
        ({"name": "x", "kind": "tenant", "warm_url": SPEECH_WARM}, "warm_url"),
    ],
)
def test_a_registration_is_validated_field_by_field(payload: dict[str, Any], param: str) -> None:
    registry = TenantRegistry(TenantsConfig())
    with pytest.raises(BadRequestError) as excinfo:
        registry.register(payload)
    assert excinfo.value.param == param
    if param in ("health_url", "vacate_url") and "192.168" in str(payload.get(param, "")):
        assert excinfo.value.code == "tenant_url_refused"


def test_the_view_withholds_urls_and_the_peer_from_a_lan_reader() -> None:
    book = LeaseBook()
    lease = book.acquire([3], holder="speechforge-session", priority=2)
    registry = TenantRegistry(_declared(), leases=book)
    registry.register({"name": "speechforge", "vacate_url": SPEECH_VACATE}, peer="127.0.0.1")
    record = registry.require("speechforge")

    public = registry.view(record, reveal_urls=False)
    assert "health_url" not in public and "warm_url" not in public and "vacate_url" not in public
    assert public["health_registered"] and public["warm_registered"]
    assert public["vacate_registered"] is True
    assert public["health_target"] == "127.0.0.1:1236"
    assert "peer" not in json.dumps(public)
    assert public["protocol"] == PROTOCOL_VERSION
    # The tenant's slice of the lease book, by family: the suffixed holder matches.
    assert [row["id"] for row in public["leases"]] == [lease.id]
    assert public["devices"] == [3] and public["lease_count"] == 1

    admin = registry.view(record, reveal_urls=True)
    assert admin["health_url"] == SPEECH_HEALTH and admin["vacate_url"] == SPEECH_VACATE


def test_a_bare_record_views_without_a_book() -> None:
    view = tenant_view(TenantRecord(name="x", family="x"))
    assert view["leases"] == [] and view["devices"] == [] and view["health"]["state"] == "unpolled"


# ---------------------------------------------------------------------------
# Health polling over a fake transport
# ---------------------------------------------------------------------------


class Backend:
    """A fake tenant: answers per path, records every request."""

    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []
        self.responses: dict[str, httpx.Response | Exception] = {}

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            self.calls.append(request)
            answer = self.responses.get(request.url.path)
            if isinstance(answer, Exception):
                raise answer
            if answer is None:
                return httpx.Response(404)
            return answer

        return httpx.MockTransport(handle)


async def test_poll_once_records_healthy_unhealthy_and_unreachable() -> None:
    backend = Backend()
    backend.responses["/health"] = httpx.Response(
        200,
        json={"status": "ready", "version": "0.9.1", "secret": "never-copied", "loaded": True},
    )
    backend.responses["/sick"] = httpx.Response(302, headers={"location": "http://evil.example/"})
    backend.responses["/down"] = httpx.ConnectError("refused")
    config = TenantsConfig(
        registry=[
            TenantEntry(name="up", health_url="http://127.0.0.1:1236/health"),
            TenantEntry(name="sick", health_url="http://127.0.0.1:1237/sick"),
            TenantEntry(name="down", health_url="http://127.0.0.1:1238/down"),
            TenantEntry(name="quiet"),
        ]
    )
    registry = TenantRegistry(config, transport=backend.transport())
    polled = await registry.poll_once(now=100.0)
    assert sorted(record.name for record in polled) == ["down", "sick", "up"]

    up = registry.require("up").health
    assert up.state == "healthy" and up.status_code == 200 and up.checked_at == 100.0
    assert up.summary == {"status": "ready", "version": "0.9.1", "loaded": True}
    assert registry.require("up").last_seen == 100.0

    sick = registry.require("sick").health
    assert sick.state == "unhealthy" and sick.status_code == 302, "a redirect is never followed"
    assert all(call.url.host != "evil.example" for call in backend.calls)

    down = registry.require("down").health
    assert down.state == "unreachable" and down.error == "ConnectError" and down.status_code is None
    assert registry.require("quiet").health.state == "unpolled"
    # The probe names itself and never sends the server's own credentials.
    assert all(call.headers.get("x-sf-client") == "studioforge-tenants" for call in backend.calls)
    await registry.stop()


async def test_the_poller_runs_on_its_interval_and_stops_cleanly() -> None:
    backend = Backend()
    backend.responses["/health"] = httpx.Response(200, json={"ok": True})
    config = TenantsConfig(
        registry=[TenantEntry(name="up", health_url="http://127.0.0.1:1236/health")],
        health_interval_s=0.01,
    )
    registry = TenantRegistry(config, transport=backend.transport())
    registry.start()
    registry.start()  # idempotent
    for _ in range(50):
        await asyncio.sleep(0.01)
        if len(backend.calls) >= 2:
            break
    assert len(backend.calls) >= 2
    await registry.stop()
    seen = len(backend.calls)
    await asyncio.sleep(0.05)
    assert len(backend.calls) == seen, "a stopped poller sends nothing"


async def test_a_declared_tenant_with_no_health_url_is_never_polled() -> None:
    backend = Backend()
    registry = TenantRegistry(
        TenantsConfig(registry=[TenantEntry(name="quiet")]), transport=backend.transport()
    )
    assert await registry.poll_once() == []
    assert backend.calls == []


# ---------------------------------------------------------------------------
# The warm relay
# ---------------------------------------------------------------------------


async def test_warm_relays_one_post_to_a_modules_warm_url_and_reports_its_answer() -> None:
    backend = Backend()
    backend.responses["/v1/speech/warm"] = httpx.Response(
        202, json={"state": "warming", "device": 3, "huge": "x" * 500}
    )
    registry = TenantRegistry(_declared(), transport=backend.transport())
    result = await registry.warm("speechforge", requested_by="gui")
    assert result == {
        "tenant": "speechforge",
        "ok": True,
        "status_code": 202,
        "summary": {"state": "warming", "device": 3},
    }
    (call,) = backend.calls
    assert call.method == "POST" and str(call.url) == SPEECH_WARM
    assert json.loads(call.content) == {"requested_by": "gui", "protocol": PROTOCOL_VERSION}
    await registry.stop()


async def test_warm_refuses_a_plain_tenant_and_reports_a_module_that_will_not_answer() -> None:
    backend = Backend()
    backend.responses["/v1/speech/warm"] = httpx.ConnectError("refused")
    registry = TenantRegistry(_declared(), transport=backend.transport())
    with pytest.raises(BadRequestError) as refused:
        await registry.warm("clawforge2")
    assert refused.value.code == "tenant_not_warmable"
    with pytest.raises(TenantUnreachableError) as unreachable:
        await registry.warm("speechforge")
    assert unreachable.value.details == {"tenant": "speechforge", "error": "ConnectError"}
    backend.responses["/v1/speech/warm"] = httpx.Response(503, json={"status": "busy"})
    with pytest.raises(TenantUnreachableError) as busy:
        await registry.warm("speechforge")
    assert busy.value.details["status_code"] == 503
    with pytest.raises(TenantNotFoundError):
        await registry.warm("nobody")
    await registry.stop()


# ---------------------------------------------------------------------------
# The routes
# ---------------------------------------------------------------------------


def _wire(app: Any) -> TenantRegistry:
    app.state.config.tenants = _declared()
    registry = TenantRegistry(app.state.config.tenants, leases=app.state.manager.leases)
    app.state.tenants = registry
    return registry


def test_register_and_deregister_sit_behind_the_lease_routes_gate(app: Any) -> None:
    """D32 as the lease routes apply it: loopback or a credential; a LAN peer is refused."""
    _wire(app)
    body = {"name": "dispatch", "health_url": "http://100.64.0.5:7000/health", "version": "1.0"}
    stranger = TestClient(app, client=LAN)
    refused = stranger.post("/api/tenants/register", json=body)
    assert refused.status_code == 403
    assert refused.json()["error"]["code"] == "remote_admin_requires_credential"

    local = TestClient(app, client=LOOPBACK)
    created = local.post("/api/tenants/register", json=body)
    assert created.status_code == 200, created.text
    row = created.json()
    assert row["name"] == "dispatch" and row["registered"] is True
    assert row["health_url"] == body["health_url"], "the registrant sees what was stored"

    assert stranger.delete("/api/tenants/dispatch").status_code == 403
    gone = local.delete("/api/tenants/dispatch")
    assert gone.status_code == 200 and gone.json()["deregistered"]["name"] == "dispatch"
    assert local.get("/api/tenants/dispatch").status_code == 404


def test_the_list_is_open_but_withholds_urls_from_a_lan_reader(app: Any) -> None:
    registry = _wire(app)
    registry.register({"name": "speechforge", "version": "0.9.1", "vacate_url": SPEECH_VACATE})
    public = TestClient(app, client=LAN).get("/api/tenants")
    assert public.status_code == 200
    payload = public.json()
    assert payload["count"] == 2 and payload["protocol"] == PROTOCOL_VERSION
    speech = next(row for row in payload["tenants"] if row["name"] == "speechforge")
    assert "health_url" not in speech and "vacate_url" not in speech
    assert speech["version"] == "0.9.1" and speech["health_target"] == "127.0.0.1:1236"
    assert "1236/v1/speech/vacate" not in public.text
    admin = TestClient(app, client=LOOPBACK).get("/api/tenants").json()
    speech_admin = next(row for row in admin["tenants"] if row["name"] == "speechforge")
    assert speech_admin["vacate_url"] == SPEECH_VACATE


def test_a_refused_url_is_a_400_naming_the_field_and_the_rule(app: Any) -> None:
    _wire(app)
    reply = TestClient(app, client=LOOPBACK).post(
        "/api/tenants/register",
        json={"name": "sneaky", "health_url": "http://169.254.169.254/latest/meta-data"},
    )
    assert reply.status_code == 400
    error = reply.json()["error"]
    assert error["code"] == "tenant_url_refused" and error["param"] == "health_url"
    assert "tenants.allowed_hosts" in error["message"]


def test_status_carries_the_tenant_rows_and_a_state_without_a_registry_still_answers(
    app: Any,
) -> None:
    _wire(app)
    payload = TestClient(app, client=LAN).get("/api/status").json()
    assert [row["name"] for row in payload["tenants"]] == ["clawforge2", "speechforge"]
    assert "health_url" not in payload["tenants"][0]
    del app.state.tenants
    again = TestClient(app, client=LAN).get("/api/status").json()
    assert again["tenants"] == [] or isinstance(again["tenants"], list)


def test_warm_over_rest_is_gated_and_relays(app: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _wire(app)
    seen: list[tuple[str, str]] = []

    async def fake_warm(name: str, *, requested_by: str = "x", timeout_s: float = 0.0) -> Any:
        seen.append((name, requested_by))
        return {"tenant": name, "ok": True, "status_code": 202, "summary": None}

    monkeypatch.setattr(registry, "warm", fake_warm)
    assert TestClient(app, client=LAN).post("/api/tenants/speechforge/warm").status_code == 403
    reply = TestClient(app, client=LOOPBACK).post(
        "/api/tenants/speechforge/warm", headers={"X-SF-Client": "pytest"}
    )
    assert reply.status_code == 200 and reply.json()["status_code"] == 202
    assert seen == [("speechforge", "pytest")]


def test_the_admin_prefix_covers_every_tenant_mutation() -> None:
    from studioforge.api.auth import is_admin_mutation

    assert is_admin_mutation("POST", "/api/tenants/register")
    assert is_admin_mutation("DELETE", "/api/tenants/speechforge")
    assert is_admin_mutation("POST", "/api/tenants/speechforge/warm")
    assert not is_admin_mutation("GET", "/api/tenants")
    assert not is_admin_mutation("GET", "/api/tenants/speechforge")


def test_the_example_config_names_the_two_commented_tenants_and_no_real_hosts() -> None:
    """The shipped example shows SpeechForge as a module and ClawForge2 as a tenant,
    commented out, on loopback only."""
    from pathlib import Path

    text = (Path(__file__).resolve().parents[2] / "config.example.yaml").read_text(encoding="utf-8")
    assert "- name: speechforge" in text and "- name: clawforge2" in text
    assert "kind: module" in text and "http://127.0.0.1:1236/v1/speech/warm" in text
    section = text[text.index("tenants:") : text.index("benchmark:")]
    import re

    addresses = set(re.findall(r"\d{1,3}(?:\.\d{1,3}){3}", section))
    assert addresses <= {"127.0.0.1", "100.64.0.0"}, addresses


def test_the_capabilities_name_the_protocol_and_the_registry() -> None:
    from studioforge.core.capabilities import SERVER_FEATURES

    assert SERVER_FEATURES["tenant_registry"] == "D77"
    assert SERVER_FEATURES["tenant_protocol_vt1"] == "D77"
    assert SERVER_FEATURES["lease_auto_placement"] == "D78"


def test_the_tenant_config_has_no_field_a_launcher_would_need() -> None:
    """D79: StudioForge never starts a tenant, so the model has nowhere to put a command.

    A "start it for me" feature needs a command line, a launcher path, argv or a
    working directory; none may appear here without superseding D79 in a
    decision of its own.
    """
    fields = set(TenantEntry.model_fields) | set(TenantsConfig.model_fields)
    forbidden = {"command", "cmd", "argv", "args", "launcher", "exe", "cwd", "working_dir", "start"}
    assert not (fields & forbidden), fields & forbidden
    assert not any(name.startswith(("launch", "start", "spawn", "exec")) for name in fields)
