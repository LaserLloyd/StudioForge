"""The tenant half of the VRAM protocol needs ``inference``, not ``admin`` (2.0.1).

A program that shares this rig's cards (SpeechForge, ClawForge2, ...) holds an
``inference`` key. Taking, renewing and releasing its own lease, and registering
itself as a tenant, must work with that key -- and must not let it touch another
tenant's lease. Everything that is genuinely administrative stays ``admin``.
No GPU, no engine, no port: a TestClient over the real app and the catalog
tests' fake probe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.api.auth import is_tenant_protocol, required_scope
from studioforge.config import Config
from tests.unit.test_catalog_routes import app as app  # noqa: PLC0414 - fixture re-export

LOOPBACK = ("127.0.0.1", 50000)


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _keys(app: Any, *names: str, scope: str = "inference") -> dict[str, dict[str, str]]:
    accounts = app.state.accounts
    return {name: _bearer(accounts.create_key(name, scope)[1]) for name in names}


@pytest.mark.parametrize(
    ("method", "path", "scope"),
    [
        ("POST", "/api/leases", "inference"),
        ("DELETE", "/api/leases/abc123", "inference"),
        ("POST", "/api/leases/abc123/touch", "inference"),
        ("POST", "/api/tenants/register", "inference"),
        ("DELETE", "/api/tenants/speechforge", "inference"),
        # the server makes an outbound call to a registered URL: administrative
        ("POST", "/api/tenants/speechforge/warm", "admin"),
        # lookalikes and unclassified mutations still fail closed
        ("DELETE", "/api/leases", "admin"),
        ("PUT", "/api/leases/abc123", "admin"),
        ("DELETE", "/api/leases/abc123/touch", "admin"),
        ("DELETE", "/api/tenants/register", "admin"),
        ("POST", "/api/tenants", "admin"),
        # reads stay open to every scope
        ("GET", "/api/leases", "read"),
        ("GET", "/api/tenants", "read"),
    ],
)
def test_route_scope_table(method: str, path: str, scope: str) -> None:
    assert required_scope(method, path, Config()) == scope


def test_the_tenant_protocol_predicate_matches_whole_paths_only() -> None:
    assert is_tenant_protocol("post", "/api/leases/")
    assert not is_tenant_protocol("POST", "/api/leases/a/b")
    assert not is_tenant_protocol("POST", "/api/leasesX")
    assert not is_tenant_protocol("GET", "/api/leases")


def test_an_inference_key_takes_renews_and_releases_its_own_lease(app: Any) -> None:
    mine = _keys(app, "speechforge")["speechforge"]
    with TestClient(app, client=LOOPBACK) as http:
        created = http.post(
            "/api/leases", json={"devices": [0], "holder": "speechforge"}, headers=mine
        )
        assert created.status_code == 200, created.text
        body = created.json()
        lease_id = body["id"]
        assert "holder_key" not in body, "the owning key is never shown"
        assert "holder_key" not in http.get("/api/leases", headers=mine).text

        touched = http.post(f"/api/leases/{lease_id}/touch", headers=mine)
        assert touched.status_code == 200, touched.text

        released = http.delete(f"/api/leases/{lease_id}", headers=mine)
        assert released.status_code == 200, released.text
        assert released.json()["released"]["id"] == lease_id
        assert http.get("/api/leases", headers=mine).json()["count"] == 0


def test_another_inference_key_cannot_renew_or_release_it(app: Any) -> None:
    keys = _keys(app, "speechforge", "clawforge2")
    with TestClient(app, client=LOOPBACK) as http:
        lease_id = http.post(
            "/api/leases",
            json={"devices": [0], "holder": "speechforge"},
            headers=keys["speechforge"],
        ).json()["id"]

        for denied in (
            http.post(f"/api/leases/{lease_id}/touch", headers=keys["clawforge2"]),
            http.delete(f"/api/leases/{lease_id}", headers=keys["clawforge2"]),
        ):
            assert denied.status_code == 403, denied.text
            error = denied.json()["error"]
            assert error["code"] == "permission_denied"
            assert "speechforge" not in error["message"], "the owner is not named to the asker"

        # still there, still its owner's
        assert http.get("/api/leases", headers=keys["clawforge2"]).json()["count"] == 1
        owner_touch = http.post(f"/api/leases/{lease_id}/touch", headers=keys["speechforge"])
        assert owner_touch.status_code == 200


def test_an_unknown_lease_is_a_404_for_an_inference_key_not_a_403(app: Any) -> None:
    keys = _keys(app, "speechforge")
    with TestClient(app, client=LOOPBACK) as http:
        gone = http.delete("/api/leases/nosuchlease", headers=keys["speechforge"])
        assert gone.status_code == 404


def test_an_admin_key_releases_any_lease_but_its_own_are_not_an_inference_keys(
    app: Any,
) -> None:
    inference = _keys(app, "speechforge")["speechforge"]
    admin = _keys(app, "root", scope="admin")["root"]
    with TestClient(app, client=LOOPBACK) as http:
        theirs = http.post(
            "/api/leases", json={"devices": [0], "holder": "speechforge"}, headers=inference
        ).json()["id"]
        assert http.delete(f"/api/leases/{theirs}", headers=admin).status_code == 200

        # a lease an admin key took belongs to that key: the tenant key cannot touch it
        ours = http.post(
            "/api/leases", json={"devices": [0], "holder": "ops"}, headers=admin
        ).json()["id"]
        assert http.post(f"/api/leases/{ours}/touch", headers=inference).status_code == 403
        assert http.delete(f"/api/leases/{ours}", headers=inference).status_code == 403
        assert http.delete(f"/api/leases/{ours}", headers=admin).status_code == 200


def test_a_read_key_cannot_take_a_lease_and_admin_only_routes_still_need_admin(
    app: Any,
) -> None:
    read = _keys(app, "viewer", scope="read")["viewer"]
    inference = _keys(app, "speechforge")["speechforge"]
    admin = _keys(app, "root", scope="admin")["root"]
    with TestClient(app, client=LOOPBACK) as http:
        assert http.post("/api/leases", json={"devices": [0]}, headers=read).status_code == 403

        # warm is an administrative relay: inference is refused by scope
        refused = http.post("/api/tenants/speechforge/warm", headers=inference)
        assert refused.status_code == 403
        assert refused.json()["error"]["code"] == "permission_denied"
        passed = http.post("/api/tenants/speechforge/warm", headers=admin)
        assert passed.status_code != 403  # 404 (unknown tenant) or a relay error, never the gate

        # unclassified mutations under the same prefixes stay admin
        assert http.post("/api/tenants", json={}, headers=inference).status_code == 403
        assert http.put("/api/leases/x", json={}, headers=inference).status_code == 403


def test_tenant_registration_is_bound_to_the_registering_key(app: Any) -> None:
    keys = _keys(app, "speechforge", "clawforge2")
    admin = _keys(app, "root", scope="admin")["root"]
    body = {"name": "speechforge", "version": "1.0"}
    with TestClient(app, client=LOOPBACK) as http:
        first = http.post("/api/tenants/register", json=body, headers=keys["speechforge"])
        assert first.status_code == 200, first.text
        # the same key refreshes (the boot path)
        again = http.post("/api/tenants/register", json=body, headers=keys["speechforge"])
        assert again.status_code == 200
        # another inference key can neither take it over nor remove it
        takeover = http.post("/api/tenants/register", json=body, headers=keys["clawforge2"])
        assert takeover.status_code == 403
        removal = http.delete("/api/tenants/speechforge", headers=keys["clawforge2"])
        assert removal.status_code == 403
        # an admin may
        assert http.delete("/api/tenants/speechforge", headers=admin).status_code == 200
        # and once it is gone the name is free for anyone
        free = http.post("/api/tenants/register", json=body, headers=keys["clawforge2"])
        assert free.status_code == 200
        own = http.delete("/api/tenants/speechforge", headers=keys["clawforge2"])
        assert own.status_code == 200


def test_lease_ownership_survives_a_restart() -> None:
    from studioforge.core.leases import LeaseBook, lease_from_row, lease_row

    lease = LeaseBook().acquire([0], holder="speechforge", holder_key="key:k1")
    restored = lease_from_row(lease_row(lease))
    assert restored.holder_key == "key:k1"
    assert "holder_key" not in restored.model_dump()


def test_the_database_round_trips_the_owning_key(tmp_path: Path) -> None:
    from studioforge.core.leases import LeaseBook, lease_row
    from studioforge.db import Database

    db = Database(tmp_path / "registry.sqlite3")
    db.migrate()
    lease = LeaseBook().acquire([0], holder="speechforge", holder_key="key:k1")
    db.save_lease(lease_row(lease))
    assert db.list_leases()[0]["holder_key"] == "key:k1"
    db.close()


# ---------------------------------------------------------------------------
# Review fixes (2.0.1): the MCP release, the unload waiver, the vacate URL,
# and the leases no live key owns
# ---------------------------------------------------------------------------


@pytest.fixture()
def mcp_app(tmp_path: Path) -> Any:
    from studioforge.api.app import build_state, create_app
    from tests.unit.test_mcp_hardening import make_config

    config = make_config(tmp_path)
    return create_app(config, state=build_state(config), start_background=False)


@pytest.mark.parametrize("mode", ["legacy", "2026-07-28"])
async def test_the_mcp_release_honours_the_owning_key(mcp_app: Any, mode: str) -> None:
    """``release_gpus`` is ``inference`` scope: it must not release another key's lease."""
    from tests.unit.test_mcp_hardening import bearer, error_code, session

    accounts = mcp_app.state.accounts
    owner_record, owner = accounts.create_key("speechforge", "inference")
    _r, other = accounts.create_key("clawforge2", "inference")
    _r, admin = accounts.create_key("root", "admin")
    book = mcp_app.state.manager.leases
    lease = book.acquire([0], holder="speechforge", holder_key=f"key:{owner_record.id}")
    async with mcp_app.router.lifespan_context(mcp_app):
        async with session(mcp_app, mode, **bearer(other)) as client:
            denied = await client.call_tool("release_gpus", {"lease_id": lease.id})
            assert error_code(denied) == "permission_denied"
            assert "speechforge" not in str(denied.structured_content)
        assert book.get(lease.id) is not None, "a foreign inference key released the lease"
        async with session(mcp_app, mode, **bearer(owner)) as client:
            done = await client.call_tool("release_gpus", {"lease_id": lease.id})
            assert done.structured_content["ok"] is True
        assert book.get(lease.id) is None

        theirs = book.acquire([1], holder="speechforge", holder_key=f"key:{owner_record.id}")
        async with session(mcp_app, mode, **bearer(admin)) as client:
            done = await client.call_tool("release_gpus", {"lease_id": theirs.id})
            assert done.structured_content["ok"] is True
        assert book.get(theirs.id) is None


def _fake_request(principal: Any, *, label: str = "", token: str = "", peer: str = "127.0.0.1"):
    from types import SimpleNamespace

    from starlette.datastructures import Headers

    headers = {}
    if label:
        headers["X-SF-Client"] = label
    if token:
        headers["X-SF-Vacate-Token"] = token
    return SimpleNamespace(
        state=SimpleNamespace(principal=principal),
        headers=Headers(headers=headers),
        client=SimpleNamespace(host=peer),
    )


def test_the_holder_waiver_is_bound_to_the_owning_key_not_the_shared_loopback_peer() -> None:
    """D55's same-peer proof means nothing when every tenant is a loopback peer."""
    from types import SimpleNamespace

    from studioforge.api.mgmt_routes import lease_holder_may_act
    from studioforge.authkeys import Principal
    from studioforge.core.leases import LeaseBook

    class _Key:
        def __init__(self, ident: str, live: bool = True) -> None:
            self.id, self._live = ident, live

        def active(self, now: float | None = None) -> bool:
            return self._live

    keys = [_Key("a"), _Key("b")]
    state = SimpleNamespace(accounts=SimpleNamespace(list_keys=lambda **_: keys))
    lease = LeaseBook().acquire(
        [0],
        holder="speechforge",
        holder_key="key:a",
        holder_peer="127.0.0.1",
        vacate_url="http://127.0.0.1:9/vacate",
        vacate_token="t" * 32,
    )
    a = Principal("key", "a", "speechforge", "inference")
    b = Principal("key", "b", "clawforge2", "inference")
    assert lease_holder_may_act(_fake_request(a), state, lease)
    # another key on the same machine, spelling the holder's label: refused
    assert not lease_holder_may_act(_fake_request(b, label="speechforge"), state, lease)
    # ... unless it holds the lease's own secret
    assert lease_holder_may_act(_fake_request(b, label="speechforge", token="t" * 32), state, lease)
    # once the owning key is revoked, the D55 peer proof reclaims it
    keys[0] = _Key("a", live=False)
    assert lease_holder_may_act(_fake_request(b, label="speechforge"), state, lease)
    assert not lease_holder_may_act(_fake_request(b, label="clawforge2"), state, lease)
    assert not lease_holder_may_act(
        _fake_request(b, label="speechforge", peer="10.0.0.9"), state, lease
    )


def test_a_non_admin_vacate_url_obeys_the_tenant_url_rule(app: Any) -> None:
    keys = _keys(app, "speechforge")
    admin = _keys(app, "root", scope="admin")["root"]
    token = "v" * 32
    with TestClient(app, client=LOOPBACK) as http:
        refused = http.post(
            "/api/leases",
            json={
                "devices": [0],
                "holder": "speechforge",
                "vacate_url": "http://203.0.113.7:9/vacate",
                "vacate_token": token,
            },
            headers=keys["speechforge"],
        )
        assert refused.status_code == 400, refused.text
        assert refused.json()["error"]["code"] == "tenant_url_refused"
        assert http.get("/api/leases", headers=admin).json()["count"] == 0

        loopback = http.post(
            "/api/leases",
            json={
                "devices": [0],
                "holder": "speechforge",
                "vacate_url": "http://127.0.0.1:9/vacate",
                "vacate_token": token,
            },
            headers=keys["speechforge"],
        )
        assert loopback.status_code == 200, loopback.text
        released = http.delete(f"/api/leases/{loopback.json()['id']}", headers=keys["speechforge"])
        assert released.status_code == 200

        # an admin keeps the pre-2.0.1 rule: any host but this server
        by_admin = http.post(
            "/api/leases",
            json={
                "devices": [0],
                "holder": "ops",
                "vacate_url": "http://203.0.113.7:9/vacate",
                "vacate_token": token,
            },
            headers=admin,
        )
        assert by_admin.status_code == 200, by_admin.text


def test_a_lease_from_before_the_upgrade_is_released_by_its_holders_proof(app: Any) -> None:
    """``release_stale`` at a tenant's boot must not 403 on its own pre-2.0.1 lease."""
    keys = _keys(app, "speechforge", "clawforge2")
    book = app.state.manager.leases
    with TestClient(app, client=LOOPBACK) as http:
        # as migration 010 leaves an older row: no holder_key, the registering peer kept
        legacy = book.acquire([0], holder="speechforge", holder_peer=LOOPBACK[0])
        stranger = http.delete(
            f"/api/leases/{legacy.id}",
            headers={**keys["clawforge2"], "X-SF-Client": "clawforge2"},
        )
        assert stranger.status_code == 403, stranger.text
        unlabelled = http.delete(f"/api/leases/{legacy.id}", headers=keys["speechforge"])
        assert unlabelled.status_code == 403
        own = http.delete(
            f"/api/leases/{legacy.id}",
            headers={**keys["speechforge"], "X-SF-Client": "speechforge"},
        )
        assert own.status_code == 200, own.text


def test_a_key_bound_lease_is_not_released_by_another_key_spelling_its_label(app: Any) -> None:
    keys = _keys(app, "speechforge", "clawforge2")
    with TestClient(app, client=LOOPBACK) as http:
        lease_id = http.post(
            "/api/leases",
            json={"devices": [0], "holder": "speechforge"},
            headers=keys["speechforge"],
        ).json()["id"]
        spoofed = http.delete(
            f"/api/leases/{lease_id}",
            headers={**keys["clawforge2"], "X-SF-Client": "speechforge"},
        )
        assert spoofed.status_code == 403, spoofed.text


def test_a_rotated_key_reclaims_its_lease_and_its_registration(app: Any) -> None:
    accounts = app.state.accounts
    old_record, old = accounts.create_key("speechforge", "inference")
    _r, new = accounts.create_key("speechforge-2", "inference")
    labelled_old = {**_bearer(old), "X-SF-Client": "speechforge"}
    labelled_new = {**_bearer(new), "X-SF-Client": "speechforge"}
    body = {"name": "speechforge", "version": "1.0"}
    with TestClient(app, client=LOOPBACK) as http:
        lease_id = http.post(
            "/api/leases", json={"devices": [0], "holder": "speechforge"}, headers=labelled_old
        ).json()["id"]
        assert (
            http.post("/api/tenants/register", json=body, headers=labelled_old).status_code == 200
        )
        # while the old key lives, the new one is a stranger
        assert (
            http.post("/api/tenants/register", json=body, headers=labelled_new).status_code == 403
        )
        accounts.revoke_key(old_record.id)
        assert http.delete(f"/api/leases/{lease_id}", headers=labelled_new).status_code == 200
        assert (
            http.post("/api/tenants/register", json=body, headers=labelled_new).status_code == 200
        )
