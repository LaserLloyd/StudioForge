"""Every request the auth middleware refuses leaves one log line: which rule, from where.

2026-11-15: a long run from another machine got a 429 and stopped. The server had
two rules that answer 429 (the per-address credential lockout and a key's
requests-per-minute limit) and logged neither; only a metrics counter moved, so
nobody could say which rule fired or which caller had tripped it. These pin the
``auth refused`` line: the code, status, address, client label and message --
never the credential -- WARNING for 401/429, INFO for a scope refusal, and
throttled per (address, code) with a count of what was held back.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.api import app as app_module
from studioforge.errors import AuthError
from tests.unit.test_auth_hardening import GOOD_KEY, make_app

WRONG_KEY = "sf-i-not-the-key-but-shaped-like-one-0123456789"  # scrub-ok: fixture


class RecordingLog:
    """Stands in for the app module's structlog logger."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str, dict[str, Any]]] = []

    def _record(self, level: str) -> Any:
        return lambda event, **fields: self.lines.append((level, event, fields))

    def __getattr__(self, name: str) -> Any:
        if name in {"debug", "info", "warning", "error", "exception"}:
            return self._record(name)
        raise AttributeError(name)

    def refusals(self) -> list[tuple[str, dict[str, Any]]]:
        return [(level, fields) for level, event, fields in self.lines if event == "auth refused"]


@pytest.fixture()
def recorded(monkeypatch: pytest.MonkeyPatch) -> Iterator[RecordingLog]:
    recorder = RecordingLog()
    monkeypatch.setattr(app_module, "log", recorder)
    app_module._auth_refusal_seen.clear()
    yield recorder
    app_module._auth_refusal_seen.clear()


def _request(peer: str = "100.64.0.9", **headers: str) -> Any:
    return SimpleNamespace(
        client=SimpleNamespace(host=peer),
        headers={k.lower().replace("_", "-"): v for k, v in headers.items()},
        method="POST",
        url=SimpleNamespace(path="/v1/chat/completions"),
    )


def _lockout(seconds: int = 4) -> AuthError:
    return AuthError(
        "Too many incorrect credentials from this address.",
        code="too_many_credential_attempts",
        status_code=429,
        details={"retry_after_s": seconds},
    )


def test_a_lockout_is_a_warning_naming_the_rule_the_address_and_the_client(
    recorded: RecordingLog,
) -> None:
    app_module.log_auth_refusal(_request(x_sf_client="bench-runner"), _lockout())
    [(level, fields)] = recorded.refusals()
    assert level == "warning"
    assert fields["code"] == "too_many_credential_attempts"
    assert fields["status"] == 429
    assert fields["peer"] == "100.64.0.9"
    assert fields["client"] == "bench-runner"
    assert fields["path"] == "/v1/chat/completions"
    assert fields["retry_after_s"] == 4
    assert "suppressed" not in fields


def test_a_key_over_its_limit_is_a_warning_naming_the_key(recorded: RecordingLog) -> None:
    exc = AuthError(
        "The API key 'operator' is limited to 20 requests per minute. Retry in 7s.",
        code="key_rate_limited",
        status_code=429,
        details={"retry_after_s": 7, "limit_rpm": 20},
    )
    app_module.log_auth_refusal(_request(), exc)
    [(level, fields)] = recorded.refusals()
    assert level == "warning"
    assert fields["code"] == "key_rate_limited"
    assert "'operator'" in fields["error"]
    assert fields["retry_after_s"] == 7


def test_a_scope_refusal_is_info(recorded: RecordingLog) -> None:
    exc = AuthError("needs admin", code="permission_denied", status_code=403)
    app_module.log_auth_refusal(_request(), exc)
    assert [level for level, _ in recorded.refusals()] == ["info"]


def test_repeats_are_held_back_and_counted_on_the_next_line(
    recorded: RecordingLog, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(app_module.time, "monotonic", lambda: clock[0])
    for _ in range(5):
        app_module.log_auth_refusal(_request(), _lockout())
    assert len(recorded.refusals()) == 1
    # Another address, or another rule from the same address, is its own line.
    app_module.log_auth_refusal(_request(peer="100.64.0.10"), _lockout())
    app_module.log_auth_refusal(_request(), AuthError("wrong key", code="invalid_api_key"))
    assert len(recorded.refusals()) == 3
    clock[0] += app_module.AUTH_REFUSAL_LOG_INTERVAL_S
    app_module.log_auth_refusal(_request(), _lockout())
    level, fields = recorded.refusals()[-1]
    assert fields["peer"] == "100.64.0.9"
    assert fields["suppressed"] == 4


def test_client_headers_are_trimmed_to_printable_text(recorded: RecordingLog) -> None:
    app_module.log_auth_refusal(
        _request(x_sf_client="run\x1b[31m-" + "x" * 200, user_agent="agent/1.0\r\nInjected: yes"),
        _lockout(),
    )
    [(_, fields)] = recorded.refusals()
    assert fields["client"] == ("run[31m-" + "x" * 200)[:64]
    assert "\r" not in fields["user_agent"] and "\n" not in fields["user_agent"]


def test_a_wrong_key_through_the_middleware_logs_the_rule_and_never_the_key(
    tmp_path: Path, recorded: RecordingLog
) -> None:
    from studioforge.api.auth import GUARD

    GUARD.reset()
    try:
        app = make_app(tmp_path, server={"api_key": GOOD_KEY})
        with TestClient(app, client=("192.168.1.50", 5000)) as lan:
            refused = lan.get(
                "/v1/models",
                headers={"Authorization": f"Bearer {WRONG_KEY}", "X-SF-Client": "stale-config"},
            )
        assert refused.status_code == 401
        [(level, fields)] = recorded.refusals()
        assert level == "warning"
        assert fields["status"] == 401
        assert fields["peer"] == "192.168.1.50"
        assert fields["client"] == "stale-config"
        assert fields["path"] == "/v1/models"
        assert all(WRONG_KEY not in str(value) for value in fields.values())
        assert all(GOOD_KEY not in str(value) for value in fields.values())
    finally:
        GUARD.reset()
