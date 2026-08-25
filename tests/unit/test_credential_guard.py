"""The lockout that makes an eight-digit PIN defensible (D44).

Measured on the live rig before this existed: eight wrong PINs in a row, each
answered in ~13 ms, no counter and no lockout. One machine walks 10^8 in
hours, and on the shipped default install that PIN is the only thing in front
of ``restart_server``, ``nuke_all_models`` and ``set_config``.
"""

from __future__ import annotations

from studioforge.credential_guard import CredentialGuard, client_key


def test_the_first_attempts_are_free_then_the_lockout_doubles() -> None:
    guard = CredentialGuard(free_attempts=3, base_lockout_s=1.0, max_lockout_s=300.0)
    assert [guard.record_failure("10.0.0.1") for _ in range(3)] == [0.0, 0.0, 0.0]
    assert guard.record_failure("10.0.0.1") == 1.0
    assert guard.record_failure("10.0.0.1") == 2.0
    assert guard.record_failure("10.0.0.1") == 4.0
    assert guard.record_failure("10.0.0.1") == 8.0


def test_the_lockout_is_capped() -> None:
    """Long enough that brute force is pointless, short enough that an
    operator locked out by a stale script is not stuck for the afternoon."""
    guard = CredentialGuard(free_attempts=0, base_lockout_s=1.0, max_lockout_s=300.0)
    for _ in range(40):
        lockout = guard.record_failure("10.0.0.1")
    assert lockout == 300.0


def test_a_huge_failure_count_does_not_compute_a_huge_number() -> None:
    """``2 ** 100000`` is a real number in Python; computing it per request
    would be the denial of service the guard exists to prevent."""
    guard = CredentialGuard(free_attempts=0, base_lockout_s=1.0, max_lockout_s=300.0)
    for _ in range(5000):
        guard.record_failure("10.0.0.1")
    assert guard.record_failure("10.0.0.1") == 300.0


def test_a_success_clears_the_record() -> None:
    guard = CredentialGuard(free_attempts=1, base_lockout_s=5.0)
    guard.record_failure("10.0.0.1")
    guard.record_failure("10.0.0.1")
    assert guard.retry_after("10.0.0.1") > 0
    guard.record_success("10.0.0.1")
    assert guard.retry_after("10.0.0.1") == 0.0


def test_clients_are_counted_separately() -> None:
    """One sprayer must not lock the operator out of their own rig."""
    guard = CredentialGuard(free_attempts=0, base_lockout_s=5.0)
    guard.record_failure("10.0.0.1")
    assert guard.retry_after("10.0.0.1") > 0
    assert guard.retry_after("10.0.0.2") == 0.0


def test_an_in_process_caller_is_never_throttled() -> None:
    """The GUI renders panels by invoking route handlers directly: no peer,
    no network, nothing to rate-limit."""
    guard = CredentialGuard(free_attempts=0, base_lockout_s=5.0)
    assert guard.record_failure(None) == 0.0
    assert guard.retry_after(None) == 0.0


def test_a_stale_record_is_forgotten() -> None:
    """An operator who got it wrong last Tuesday starts clean."""
    guard = CredentialGuard(free_attempts=0, base_lockout_s=5.0, window_s=0.0)
    guard.record_failure("10.0.0.1")
    assert guard.retry_after("10.0.0.1") == 0.0


def test_tracking_is_bounded() -> None:
    """A spray from many addresses is exactly when unbounded growth would be a
    memory-exhaustion bug."""
    guard = CredentialGuard(free_attempts=0, max_tracked=16)
    for n in range(200):
        guard.record_failure(f"10.0.0.{n}")
    assert len(guard._records) <= 16


def test_client_key_normalises() -> None:
    assert client_key(" 10.0.0.1 ") == "10.0.0.1"
    assert client_key("FE80::1") == "fe80::1"
    assert client_key("") is None
    assert client_key(None) is None
