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
