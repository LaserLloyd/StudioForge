#!/usr/bin/env python3
"""Post-restart verification of D53 / D54 against a LIVE StudioForge.

Read-only by design: every check is a GET, plus the MCP streamable-HTTP
handshake (``initialize`` -> ``notifications/initialized`` -> ``tools/list``
-> ``tools/call server_status``), which changes nothing. The one negative
probe -- a ``POST /v1/chat/completions`` expected to be refused with
``507 gpu_leased`` -- is OFF unless ``--probe-507`` is passed, and even then
it runs only when a foreign lease on ``/api/leases`` covers **every** GPU the
server reports, so the request can never trigger a load, touch a card or add
a token to somebody's benchmark: it is refused before the planner runs
(``manager.lease_check``, D53). httpx only, no repo imports.

The second opt-in probe, ``--probe-vacate``, proves the D56 vacate protocol
end to end against a SPARE card: it starts a one-shot HTTP listener on
loopback, takes a class-3 scratch tenant lease on the card with that listener
as its ``vacate_url``, asks for the same card at class 2, and checks the whole
exchange -- ``409 lease_vacating`` + ``Retry-After: 15``, the POST that lands
on the listener (``X-SF-Vacate-Token``, ``X-SF-Lease-Id``, the six body
fields), the lease reading ``vacating`` / ``delivered``, the dedupe on a
re-ask, and the grant after the tenant releases. It refuses to run unless a
card is free of every lease AND every resident model (a lease grant evicts
idle residents, D36), never sends ``force``, names no ``model_ids`` so nothing
is steered, and releases both of its leases in a ``finally``. ``POST
/api/leases`` is D32-gated, so this needs a loopback caller or the PIN.

Usage::

    python scripts/verify_rig_improvements_live.py [--base http://127.0.0.1:1234]
                                                  [--probe-507] [--model <id>]
                                                  [--probe-vacate] [--pin <pin>]

Exit 0 when every check passes, 1 otherwise. Run it from the rig itself:
``/api/mcp/info`` reveals the PIN to a loopback caller only (D32/D44).
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

EXPECTED_TOOL_COUNT = 20
LEASE_KEYS = ("state", "holder_family", "kind", "idle_s", "expires_at", "retry_after_s")
LEASE_STATES = {"active", "idle", "expiring", "vacating"}
LEASE_KINDS = {"benchmark", "render", "agent", "other"}
#: Holders the vacate probe uses; anything it leaves behind is released on exit.
PROBE_TENANT = "verify-vacate-tenant"
PROBE_ASKER = "verify-vacate-bench"
VACATE_BODY_KEYS = {
    "lease_id",
    "devices",
    "requester",
    "requester_priority",
    "deadline_s",
    "deadline_at",
}


def holder_family(holder: str) -> str:
    """The server's rule (D53/D56): everything before the first ``-`` or ``:``."""
    return re.split(r"[-:]", holder or "", maxsplit=1)[0].strip().lower() or (holder or "")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append(Check(name, ok, detail))

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def render(self) -> str:
        width = max(len(c.name) for c in self.checks) if self.checks else 10
        lines = [f"{'CHECK':<{width}}  RESULT  DETAIL", "-" * (width + 40)]
        for check in self.checks:
            verdict = "PASS" if check.ok else "FAIL"
            lines.append(f"{check.name:<{width}}  {verdict:<6}  {check.detail}")
        lines.append("-" * (width + 40))
        lines.append(f"{len(self.checks) - len(self.failed)} passed, {len(self.failed)} failed")
        return "\n".join(lines)


def _get(client: Any, url: str) -> Any:
    response = client.get(url)
    response.raise_for_status()
    return response.json()


def _has_keys(row: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    return [k for k in keys if k not in row]


# ---------------------------------------------------------------------------
# REST checks
# ---------------------------------------------------------------------------


def check_health(client: Any, base: str, report: Report) -> str | None:
    try:
        health = _get(client, f"{base}/health")
    except Exception as exc:  # noqa: BLE001 - the report IS the error path
        report.add("GET /health", False, f"{exc}")
        return None
    version = health.get("version")
    report.add("GET /health", bool(version), f"version={version} draining={health.get('draining')}")
    return version


def check_status(client: Any, base: str, report: Report) -> dict[str, Any]:
    try:
        status = _get(client, f"{base}/api/status")
    except Exception as exc:  # noqa: BLE001
        report.add("GET /api/status", False, f"{exc}")
        return {}
    report.add("GET /api/status", True, f"gpus={len(status.get('gpus', []))}")

    leases = status.get("leases")
    if not isinstance(leases, list):
        report.add("status.leases[] is a list", False, f"got {type(leases).__name__}")
    elif not leases:
        report.add(
            "status.leases[] carry D53 fields", True, "no lease standing -- nothing to inspect"
        )
    else:
        problems: list[str] = []
        for lease in leases:
            missing = _has_keys(lease, LEASE_KEYS)
            if missing:
                problems.append(f"{lease.get('id')}: missing {missing}")
            if lease.get("state") not in LEASE_STATES:
                problems.append(f"{lease.get('id')}: state={lease.get('state')!r}")
            if lease.get("kind") not in LEASE_KINDS:
                problems.append(f"{lease.get('id')}: kind={lease.get('kind')!r}")
            holder = str(lease.get("holder") or "")
            if lease.get("holder_family") != holder_family(holder):
                problems.append(f"{lease.get('id')}: holder_family={lease.get('holder_family')!r}")
            retry = lease.get("retry_after_s")
            if lease.get("expires_at") is not None and not (
                isinstance(retry, int) and 1 <= retry <= 300
            ):
                problems.append(f"{lease.get('id')}: retry_after_s={retry!r}")
        summary = ", ".join(
            f"{lease.get('holder')}[{lease.get('state')}/{lease.get('kind')}] "
            f"idle={lease.get('idle_s')}s retry={lease.get('retry_after_s')}"
            for lease in leases
        )
        report.add(
            "status.leases[] carry D53 fields",
            not problems,
            "; ".join(problems) if problems else summary,
        )

    loaded = status.get("loaded")
    if not isinstance(loaded, list):
        report.add("status.loaded[] carry effective + prompt_cache", False, "loaded[] missing")
    elif not loaded:
        report.add("status.loaded[] carry effective + prompt_cache", True, "nothing loaded")
    else:
        problems = []
        details = []
        for row in loaded:
            missing = _has_keys(row, ("effective", "prompt_cache", "launch_args"))
            if missing:
                problems.append(f"{row.get('model_id')}: missing {missing}")
                continue
            effective = row["effective"]
            cache = row["prompt_cache"]
            if effective is not None and "summary" not in effective:
                problems.append(f"{row.get('model_id')}: effective has no summary")
            if cache is not None and not {"processed_total", "cached_total", "hit_ratio"} <= set(
                cache
            ):
                problems.append(f"{row.get('model_id')}: prompt_cache shape {sorted(cache)}")
            if any(
                "\\" in token or (len(token) > 1 and token[1] == ":") or token.startswith("/")
                for token in row["launch_args"] or []
            ):
                problems.append(f"{row.get('model_id')}: launch_args carries a path")
            hit = None if cache is None else cache.get("hit_ratio")
            details.append(
                f"{row.get('model_id')}: {effective['summary'] if effective else 'effective=null'}"
                f" | prompt_cache={'null' if cache is None else f'hit_ratio={hit}'}"
            )
        report.add(
            "status.loaded[] carry effective + prompt_cache",
            not problems,
            "; ".join(problems) if problems else " || ".join(details),
        )
    return status


def check_models(client: Any, base: str, report: Report) -> None:
    try:
        models = _get(client, f"{base}/api/models")
    except Exception as exc:  # noqa: BLE001
        report.add("GET /api/models rows carry effective", False, f"{exc}")
        return
    rows = models.get("models", [])
    missing = [m.get("id") for m in rows if "effective" not in m]
    warm = [m.get("id") for m in rows if m.get("effective")]
    report.add(
        "GET /api/models rows carry effective",
        not missing and bool(rows),
        f"missing on {missing}" if missing else f"{len(rows)} rows, effective non-null on {warm}",
    )


def check_v1_models(client: Any, base: str, report: Report) -> None:
    try:
        listing = _get(client, f"{base}/v1/models")
    except Exception as exc:  # noqa: BLE001
        report.add("GET /v1/models studioforge block", False, f"{exc}")
        return
    entries = listing.get("data", [])
    problems = []
    for entry in entries:
        block = entry.get("studioforge")
        if not isinstance(block, dict) or "state" not in block:
            problems.append(f"{entry.get('id')}: no studioforge block")
            continue
        if entry.get("state") == "loaded" and "effective" not in block:
            problems.append(f"{entry.get('id')}: loaded but no studioforge.effective")
        if entry.get("state") != "loaded" and "effective" in block:
            problems.append(f"{entry.get('id')}: not loaded yet carries effective")
    loaded_ids = [e.get("id") for e in entries if e.get("state") == "loaded"]
    report.add(
        "GET /v1/models studioforge block",
        not problems and bool(entries),
        "; ".join(problems) if problems else f"{len(entries)} entries, loaded={loaded_ids}",
    )


# ---------------------------------------------------------------------------
# The one negative probe (opt-in)
# ---------------------------------------------------------------------------


def probe_507(
    client: Any, base: str, status: dict[str, Any], model: str | None, report: Report
) -> None:
    name = "POST /v1/chat/completions -> 507 gpu_leased"
    try:
        leases = _get(client, f"{base}/api/leases").get("leases", [])
    except Exception as exc:  # noqa: BLE001
        report.add(name, False, f"/api/leases: {exc}")
        return
    gpu_indices = {g.get("index") for g in status.get("gpus", [])}
    leased = set()
    for lease in leases:
        leased.update(lease.get("devices", []))
    if not leases or not gpu_indices or not gpu_indices <= leased:
        report.add(
            name,
            True,
            f"skipped: leases cover {sorted(leased)} of GPUs {sorted(gpu_indices)} -- the probe "
            f"only runs when a foreign lease holds EVERY card, so it can never trigger a load",
        )
        return
    loaded_ids = {row.get("model_id") for row in status.get("loaded", [])}
    lease_models = {m for lease in leases for m in lease.get("model_ids", [])}
    if model is None:
        try:
            rows = _get(client, f"{base}/api/models").get("models", [])
        except Exception as exc:  # noqa: BLE001
            report.add(name, False, f"/api/models: {exc}")
            return
        candidates = [
            r.get("id")
            for r in rows
            if r.get("id") not in loaded_ids
            and r.get("id") not in lease_models
            and not r.get("is_virtual")
        ]
        model = candidates[0] if candidates else None
    if model is None or model in loaded_ids or model in lease_models:
        report.add(name, True, "skipped: no unloaded, unleased model to name")
        return
    response = client.post(
        f"{base}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "stream": False,
        },
        headers={"X-SF-Client": "verify_rig_improvements_live"},
    )
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        body = {"raw": response.text[:500]}
    error = body.get("error", {}) if isinstance(body, dict) else {}
    ok = (
        response.status_code == 507
        and error.get("code") == "gpu_leased"
        and "leased" in str(error.get("message", ""))
        and "0.00 GiB" not in str(error.get("message", ""))
        and bool(response.headers.get("Retry-After"))
        and isinstance(error.get("studioforge", {}).get("lease"), dict)
    )
    print("\n--- 507 envelope ---")
    print(f"HTTP {response.status_code}  Retry-After: {response.headers.get('Retry-After')}")
    print(json.dumps(body, indent=2)[:4000])
    print("--- end envelope ---\n")
    report.add(
        name,
        ok,
        f"model={model} status={response.status_code} code={error.get('code')} "
        f"retry_after={response.headers.get('Retry-After')}",
    )


# ---------------------------------------------------------------------------
# The vacate probe (opt-in): a scratch tenant on a spare card, asked to leave
# ---------------------------------------------------------------------------


class VacateListener:
    """A one-shot loopback HTTP server that records every POST it receives."""

    def __init__(self) -> None:
        from http.server import BaseHTTPRequestHandler, HTTPServer

        hits: list[dict[str, Any]] = []
        self.hits = hits

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server's spelling
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length).decode("utf-8", "replace")
                try:
                    body: Any = json.loads(raw)
                except json.JSONDecodeError:
                    body = {"raw": raw}
                hits.append(
                    {
                        "path": self.path,
                        "token": self.headers.get("X-SF-Vacate-Token"),
                        "lease_id": self.headers.get("X-SF-Lease-Id"),
                        "content_type": self.headers.get("Content-Type"),
                        "body": body,
                    }
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"vacating": true}')

            def log_message(self, *_args: Any) -> None:  # silence http.server
                return

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def wait_for_hit(self, timeout_s: float) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.hits:
                return self.hits[0]
            time.sleep(0.1)
        return None


def _error_of(response: Any) -> dict[str, Any]:
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        return {}
    return body.get("error", {}) if isinstance(body, dict) else {}


def _spare_card(status: dict[str, Any], leases: list[dict[str, Any]]) -> tuple[int | None, str]:
    """A GPU no lease covers and no model is resident on, else why there is none."""
    gpu_indices = sorted(int(g.get("index")) for g in status.get("gpus", []) if "index" in g)
    leased: set[int] = set()
    for lease in leases:
        leased.update(int(d) for d in lease.get("devices", []))
    resident: set[int] = set()
    for row in status.get("loaded", []):
        plan = row.get("plan") or {}
        resident.update(int(d) for d in (plan.get("devices") or row.get("devices") or []))
    free = [i for i in gpu_indices if i not in leased and i not in resident]
    if not gpu_indices:
        return None, "the server reports no GPUs"
    if not free:
        return None, (
            f"no spare card: GPUs {gpu_indices}, leased {sorted(leased)}, "
            f"resident {sorted(resident)} -- the probe never leases a card something is on"
        )
    # The highest index: on this rig the 3090s, the cards a benchmark wants least.
    return free[-1], f"GPUs {gpu_indices}, leased {sorted(leased)}, resident {sorted(resident)}"


def _release_probe_leases(client: Any, base: str, headers: dict[str, str]) -> list[str]:
    """Release every lease this probe's holders still hold. Never raises."""
    released: list[str] = []
    try:
        leases = _get(client, f"{base}/api/leases").get("leases", [])
    except Exception:  # noqa: BLE001
        return released
    for lease in leases:
        if lease.get("holder") in (PROBE_TENANT, PROBE_ASKER):
            try:
                client.delete(f"{base}/api/leases/{lease['id']}", headers=headers)
                released.append(str(lease["id"]))
            except Exception:  # noqa: BLE001
                continue
    return released


def probe_vacate(
    client: Any, base: str, status: dict[str, Any], pin: str | None, report: Report
) -> None:
    """The D56 exchange, end to end, against a spare card. Cleans up after itself."""
    name = "vacate: tenant asked, 409 lease_vacating, POST delivered, release -> grant"
    headers = {"X-MCP-Pin": pin} if pin else {}
    try:
        leases = _get(client, f"{base}/api/leases").get("leases", [])
    except Exception as exc:  # noqa: BLE001
        report.add(name, False, f"/api/leases: {exc}")
        return
    card, why = _spare_card(status, leases)
    if card is None:
        report.add(name, True, f"skipped: {why}")
        return

    listener = VacateListener()
    listener.start()
    token = "vt-" + secrets.token_hex(12)
    vacate_url = f"http://127.0.0.1:{listener.port}/vacate"
    problems: list[str] = []
    steps: list[str] = []
    tenant_id: str | None = None
    asker_id: str | None = None
    try:
        # 1. the scratch tenant registers where it may be asked
        created = client.post(
            f"{base}/api/leases",
            json={
                "devices": [card],
                "holder": PROBE_TENANT,
                "reason": "verify_rig_improvements_live --probe-vacate (scratch tenant)",
                "idle_ttl_s": 300,
                "priority": 3,
                "vacate_url": vacate_url,
                "vacate_token": token,
            },
            headers=headers,
        )
        if created.status_code != 200:
            error = _error_of(created)
            hint = (
                " -- POST /api/leases is D32-gated: run this on the rig, or pass --pin"
                if created.status_code == 403
                else ""
            )
            report.add(
                name,
                False,
                f"tenant lease refused: HTTP {created.status_code} {error.get('code')}{hint}",
            )
            return
        tenant = created.json()
        tenant_id = str(tenant["id"])
        steps.append(f"tenant {tenant_id} on CUDA {card}")
        if tenant.get("state") != "active":
            problems.append(f"a fresh 300 s lease reads state={tenant.get('state')!r} (W1)")
        if tenant.get("vacate_registered") is not True or tenant.get("vacate_url") != vacate_url:
            problems.append("the registrant was not shown its own vacate_url")
        if tenant.get("priority") != 3:
            problems.append(f"tenant priority={tenant.get('priority')!r}")
        if token in created.text:
            problems.append("the vacate_token was echoed in the create reply")

        # 2. a class-2 asker wants the card
        ask_body = {
            "devices": [card],
            "holder": PROBE_ASKER,
            "reason": "verify_rig_improvements_live --probe-vacate (asker)",
            "idle_ttl_s": 300,
            "priority": 2,
        }
        asked = client.post(f"{base}/api/leases", json=ask_body, headers=headers)
        error = _error_of(asked)
        vacate = (error.get("studioforge") or {}).get("vacate") or {}
        if asked.status_code == 200:
            asker_id = str(asked.json()["id"])
            problems.append("the asker was granted at once: the tenant was never asked")
        elif asked.status_code != 409 or error.get("code") != "lease_vacating":
            problems.append(f"ask: HTTP {asked.status_code} {error.get('code')!r}")
        else:
            steps.append("409 lease_vacating")
            if asked.headers.get("Retry-After") != "15":
                problems.append(f"Retry-After={asked.headers.get('Retry-After')!r}, not 15")
            if vacate.get("state") != "vacating" or vacate.get("requested") != [tenant_id]:
                problems.append(f"vacate block {vacate!r}")
            if (error.get("studioforge") or {}).get("retry_after_s") != 15:
                problems.append("retry_after_s != 15")
            rows = (error.get("studioforge") or {}).get("leases") or []
            if not rows or rows[0].get("state") != "vacating":
                problems.append("the 409's lease row is not state=vacating")
            if token in asked.text:
                problems.append("the vacate_token appeared in the 409")
        print("\n--- 409 envelope ---")
        print(f"HTTP {asked.status_code}  Retry-After: {asked.headers.get('Retry-After')}")
        print(json.dumps(_error_of(asked), indent=2)[:3000])
        print("--- end envelope ---\n")

        # 3. the POST lands on the listener
        hit = listener.wait_for_hit(12.0)
        if hit is None:
            problems.append("no vacate POST reached the listener within 12 s")
        else:
            steps.append("POST delivered")
            if hit.get("token") != token:
                problems.append("X-SF-Vacate-Token did not match the registered token")
            if hit.get("lease_id") != tenant_id:
                problems.append(f"X-SF-Lease-Id={hit.get('lease_id')!r}")
            body = hit.get("body") or {}
            missing = sorted(VACATE_BODY_KEYS - set(body))
            if missing:
                problems.append(f"vacate body lacks {missing}")
            if body.get("lease_id") != tenant_id or body.get("devices") != [card]:
                problems.append(
                    f"vacate body names {body.get('lease_id')!r} {body.get('devices')!r}"
                )
            if body.get("requester") != PROBE_ASKER or body.get("requester_priority") != 2:
                problems.append("vacate body requester/priority wrong")
            print("--- vacate POST as received ---")
            print(
                json.dumps(
                    {
                        **hit,
                        "token": "<matches>" if hit.get("token") == token else hit.get("token"),
                    },
                    indent=2,
                )
            )
            print("--- end ---\n")

        # 4. the book says vacating / delivered, and a re-ask sends nothing new
        delivered = None
        for _ in range(20):
            row = next(
                (
                    r
                    for r in _get(client, f"{base}/api/leases").get("leases", [])
                    if r.get("id") == tenant_id
                ),
                None,
            )
            delivered = (row or {}).get("vacate_delivery")
            if delivered == "delivered":
                break
            time.sleep(0.25)
        if row is None:
            problems.append("the tenant lease vanished")
        else:
            if row.get("state") != "vacating":
                problems.append(f"/api/leases row state={row.get('state')!r}, not vacating")
            if delivered != "delivered":
                problems.append(
                    f"vacate_delivery={delivered!r} (status {row.get('vacate_delivery_status')!r})"
                )
            steps.append(f"row: state={row.get('state')} delivery={delivered}")
        again = client.post(f"{base}/api/leases", json=ask_body, headers=headers)
        again_error = _error_of(again)
        again_vacate = (again_error.get("studioforge") or {}).get("vacate") or {}
        if again.status_code == 200:
            asker_id = str(again.json()["id"])
            problems.append("the re-ask was granted while the tenant still held the card")
        elif again_error.get("code") != "lease_vacating" or again_vacate.get("requested") != []:
            requested = again_vacate.get("requested")
            problems.append(
                f"re-ask: {again.status_code} {again_error.get('code')!r} requested={requested!r}"
            )
        elif len(listener.hits) != 1:
            problems.append(f"{len(listener.hits)} POSTs reached the listener; expected exactly 1")
        else:
            steps.append("re-ask deduped")

        # 5. the tenant releases; the same ask is granted
        released = client.delete(f"{base}/api/leases/{tenant_id}", headers=headers)
        if released.status_code != 200:
            problems.append(f"tenant release: HTTP {released.status_code}")
        else:
            tenant_id = None
            granted = client.post(f"{base}/api/leases", json=ask_body, headers=headers)
            if granted.status_code != 200:
                code = _error_of(granted).get("code")
                problems.append(f"post-release ask: HTTP {granted.status_code} {code!r}")
            else:
                asker_id = str(granted.json()["id"])
                steps.append(f"granted {asker_id} at priority {granted.json().get('priority')}")
                if (
                    granted.json().get("priority") != 2
                    or granted.json().get("holder") != PROBE_ASKER
                ):
                    problems.append("the grant's holder/priority are wrong")
    finally:
        if asker_id:
            client.delete(f"{base}/api/leases/{asker_id}", headers=headers)
        if tenant_id:
            client.delete(f"{base}/api/leases/{tenant_id}", headers=headers)
        leftovers = _release_probe_leases(client, base, headers)
        listener.stop()
        if leftovers:
            problems.append(f"leftover probe leases released on exit: {leftovers}")
    report.add(name, not problems, "; ".join(problems) if problems else " -> ".join(steps))


# ---------------------------------------------------------------------------
# MCP: raw streamable-HTTP handshake
# ---------------------------------------------------------------------------


def _rpc_payload(response: Any) -> dict[str, Any]:
    """The JSON-RPC message in a streamable-HTTP reply (plain JSON or SSE)."""
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        last: dict[str, Any] = {}
        for line in response.text.splitlines():
            if line.startswith("data:"):
                try:
                    candidate = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict) and "jsonrpc" in candidate:
                    last = candidate
        return last
    try:
        data = response.json()
    except Exception:  # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


def _tool_result_json(result: dict[str, Any]) -> Any:
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    for item in result.get("content", []):
        if item.get("type") == "text":
            try:
                return json.loads(item["text"])
            except (json.JSONDecodeError, KeyError):
                continue
    return None


def check_mcp(client: Any, base: str, report: Report) -> None:
    try:
        info = _get(client, f"{base}/api/mcp/info")
    except Exception as exc:  # noqa: BLE001
        report.add("GET /api/mcp/info", False, f"{exc}")
        return
    path = info.get("path") or "/mcp"
    pin = info.get("pin")
    pin_required = bool(info.get("pin_required"))
    report.add(
        "GET /api/mcp/info",
        True,
        f"path={path} pin_required={pin_required} pin={'revealed' if pin else 'withheld'}",
    )
    if pin_required and not pin:
        report.add(
            "MCP handshake",
            False,
            "PIN required but not revealed -- run this on the rig (loopback) so /api/mcp/info "
            "hands it over (D32/D44)",
        )
        return

    headers = {"Accept": "application/json, text/event-stream"}
    if pin:
        headers["X-MCP-Pin"] = pin
    url = f"{base}{path}"

    def rpc(payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        response = client.post(url, json=payload, headers=headers)
        return response, _rpc_payload(response)

    response, init = rpc(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "verify_rig_improvements_live", "version": "0"},
            },
        }
    )
    session = response.headers.get("mcp-session-id")
    ok = response.status_code == 200 and "result" in init
    report.add(
        "MCP initialize",
        ok,
        f"status={response.status_code} session={'yes' if session else 'no'} "
        f"server={init.get('result', {}).get('serverInfo', {})}",
    )
    if not ok:
        return
    if session:
        headers["Mcp-Session-Id"] = session
    client.post(
        url, json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=headers
    )

    response, listing = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    tools = listing.get("result", {}).get("tools", [])
    names = sorted(t.get("name", "") for t in tools)
    report.add(
        f"MCP tools/list == {EXPECTED_TOOL_COUNT}",
        len(tools) == EXPECTED_TOOL_COUNT,
        f"{len(tools)} tools: {', '.join(names)}",
    )

    response, called = rpc(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "server_status", "arguments": {}},
        }
    )
    result = called.get("result", {})
    payload = _tool_result_json(result)
    if not isinstance(payload, dict) or result.get("isError"):
        report.add("MCP server_status", False, f"status={response.status_code} result={result}")
        return
    problems = []
    leases = payload.get("leases")
    if not isinstance(leases, list):
        problems.append("no leases[]")
    else:
        for lease in leases:
            if lease.get("state") not in LEASE_STATES or "holder_family" not in lease:
                problems.append(f"lease {lease.get('id')} lacks state/holder_family")
    loaded = payload.get("loaded")
    if not isinstance(loaded, list):
        problems.append("no loaded[]")
    else:
        for row in loaded:
            missing = _has_keys(row, ("prompt_cache", "effective"))
            if missing:
                problems.append(f"{row.get('model_id')}: missing {missing}")
    report.add(
        "MCP server_status leases.state + loaded.prompt_cache",
        not problems,
        "; ".join(problems)
        if problems
        else f"leases={len(leases or [])} loaded={[r.get('model_id') for r in loaded or []]}",
    )


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="http://127.0.0.1:1234")
    parser.add_argument(
        "--probe-507",
        action="store_true",
        help="send ONE max_tokens=1 chat request naming an unloaded model, only if a foreign "
        "lease covers every GPU; expect 507 gpu_leased + Retry-After (off by default)",
    )
    parser.add_argument("--model", default=None, help="model id for --probe-507 (auto-picked)")
    parser.add_argument(
        "--probe-vacate",
        action="store_true",
        help="take a scratch class-3 lease on a SPARE card (no lease, no resident model) with a "
        "loopback listener as its vacate_url, ask for the card at class 2, and check the D56 "
        "exchange end to end; both leases are released on exit (off by default)",
    )
    parser.add_argument(
        "--pin",
        default=None,
        help="MCP pairing PIN, sent as X-MCP-Pin on the vacate probe's lease calls when not "
        "running from the rig itself",
    )
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args(argv)

    try:
        import httpx
    except ImportError:
        print("httpx is required: .venv/Scripts/python.exe -m pip install httpx", file=sys.stderr)
        return 2

    report = Report()
    base = args.base.rstrip("/")
    with httpx.Client(timeout=args.timeout) as client:
        check_health(client, base, report)
        status = check_status(client, base, report)
        check_models(client, base, report)
        check_v1_models(client, base, report)
        if args.probe_507:
            probe_507(client, base, status, args.model, report)
        else:
            report.add(
                "POST /v1/chat/completions -> 507 gpu_leased",
                True,
                "not run (pass --probe-507 to send the one refused request)",
            )
        if args.probe_vacate:
            probe_vacate(client, base, status, args.pin, report)
        else:
            report.add(
                "vacate: tenant asked, 409 lease_vacating, POST delivered, release -> grant",
                True,
                "not run (pass --probe-vacate to exercise D56 against a spare card)",
            )
        check_mcp(client, base, report)

    print(report.render())
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
