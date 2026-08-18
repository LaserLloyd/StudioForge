"""MCP reachability and the pairing PIN.

Two separate concerns that arrive together in practice: *where* the MCP endpoint
can be reached from another machine, and *how* a client proves it is allowed to.
"""

from __future__ import annotations

import ipaddress
from typing import Any

import pytest
from fastapi import Request

from studioforge.api.auth import check_request, extract_pin, is_mcp_path
from studioforge.config import Config, generate_pin
from studioforge.core import netinfo
from studioforge.errors import AuthError


def make_request(path: str, headers: dict[str, str] | None = None, query: str = "") -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope: dict[str, Any] = {
        "type": "http",
        "method": "GET",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query.encode(),
        "headers": raw,
        "scheme": "http",
        "server": ("127.0.0.1", 1234),
        "client": ("127.0.0.1", 5000),
        "root_path": "",
    }
    return Request(scope)


def make_config(**kwargs: Any) -> Config:
    config = Config(data_dir="/tmp/sf-mcp")
    config.server.api_key = kwargs.get("api_key")
    config.mcp.pin = kwargs.get("pin")
    config.mcp.pin_required = kwargs.get("pin_required", True)
    return config


# ---------------------------------------------------------------------------
# Address discovery
# ---------------------------------------------------------------------------


def test_tailscale_addresses_rank_first() -> None:
    """A tailnet address survives a network change; a LAN address does not."""
    addresses = netinfo.local_addresses()
    kinds = [a.kind for a in addresses]
    # Loopback is always present, so the list is never empty.
    assert "loopback" in kinds
    if "tailscale" in kinds:
        assert kinds.index("tailscale") == 0
    if "lan" in kinds and "loopback" in kinds:
        assert kinds.index("lan") < kinds.index("loopback")


def test_every_discovered_address_is_a_valid_ipv4() -> None:
    for address in netinfo.local_addresses():
        parsed = ipaddress.ip_address(address.ip)
        assert parsed.version == 4


def test_link_local_and_multicast_are_never_advertised() -> None:
    assert netinfo._classify("169.254.10.1") is None
    assert netinfo._classify("224.0.0.1") is None
    assert netinfo._classify("::1") is None
    assert netinfo._classify("not-an-ip") is None


def test_cgnat_range_is_classified_as_tailscale() -> None:
    assert netinfo._classify("100.100.7.42") == "tailscale"
    assert netinfo._classify("192.168.1.5") == "lan"
    assert netinfo._classify("127.0.0.1") == "loopback"


def test_reachable_urls_include_the_path() -> None:
    urls = netinfo.reachable_urls(1234, "/mcp")
    assert urls
    assert all(u["url"].endswith("/mcp") for u in urls)
    assert all(u["url"].startswith("http://") for u in urls)


def test_a_specific_bind_address_wins_over_discovery() -> None:
    """Bound to one interface means it answers on exactly one address.

    Advertising the others would be a lie.
    """
    urls = netinfo.reachable_urls(1234, "/mcp", host="192.168.1.50")
    assert urls == [
        {
            "ip": "192.168.1.50",
            "kind": "bound",
            "label": "Configured bind address",
            "url": "http://192.168.1.50:1234/mcp",
        }
    ]


def test_wildcard_binds_are_expanded_not_advertised_literally() -> None:
    for wildcard in ("0.0.0.0", "::", ""):
        urls = netinfo.reachable_urls(1234, "/mcp", host=wildcard)
        assert all(wildcard not in u["url"] for u in urls if wildcard)
        assert urls


def test_primary_url_prefers_tailscale() -> None:
    url = netinfo.primary_url(1234, "/mcp")
    addresses = netinfo.local_addresses()
    assert url == f"http://{addresses[0].ip}:1234/mcp"


# ---------------------------------------------------------------------------
# PIN generation
# ---------------------------------------------------------------------------


def test_generated_pins_are_random_and_numeric() -> None:
    pins = {generate_pin() for _ in range(50)}
    assert len(pins) > 45, "PINs should not collide at this rate"
    for pin in pins:
        assert pin.isdigit()
        assert len(pin) == 8


def test_pin_is_minted_on_load_when_required() -> None:
    config = Config(data_dir="/tmp/sf-mcp-pin")
    assert config.mcp.pin_required is True
    # load_config mints one; a bare Config starts empty by design.
    assert config.mcp.pin is None


def test_short_pins_are_rejected() -> None:
    from pydantic import ValidationError

    from studioforge.config import McpConfig

    with pytest.raises(ValidationError):
        McpConfig(pin="123")
    assert McpConfig(pin="   ").pin is None


# ---------------------------------------------------------------------------
# PIN authentication, scoped to the MCP path
# ---------------------------------------------------------------------------


def test_pin_extracted_from_every_accepted_carrier() -> None:
    assert extract_pin(make_request("/mcp", {"X-MCP-Pin": " 1234 "})) == "1234"
    assert extract_pin(make_request("/mcp", {"X-StudioForge-Pin": "5678"})) == "5678"
    assert extract_pin(make_request("/mcp", query="pin=9012")) == "9012"
    assert extract_pin(make_request("/mcp")) is None


def test_is_mcp_path_matches_the_endpoint_and_children() -> None:
    config = make_config(pin="12345678")
    assert is_mcp_path("/mcp", config)
    assert is_mcp_path("/mcp/messages", config)
    assert not is_mcp_path("/mcpx", config)
    assert not is_mcp_path("/api/status", config)


def test_correct_pin_authorizes_the_mcp_path() -> None:
    config = make_config(api_key="the-key", pin="12345678")
    check_request(make_request("/mcp", {"X-MCP-Pin": "12345678"}), config)


def test_pin_also_accepted_as_a_bearer_token() -> None:
    """MCP clients often only offer a bearer field."""
    config = make_config(api_key="the-key", pin="12345678")
    check_request(make_request("/mcp", {"Authorization": "Bearer 12345678"}), config)


def test_wrong_pin_is_refused_with_an_actionable_message() -> None:
    config = make_config(api_key="the-key", pin="12345678")
    with pytest.raises(AuthError) as excinfo:
        check_request(make_request("/mcp", {"X-MCP-Pin": "00000000"}), config)
    assert excinfo.value.code == "invalid_mcp_pin"
    assert "/api/mcp/info" in excinfo.value.message


def test_pin_does_not_work_anywhere_else() -> None:
    """The PIN is a scoped pairing code, not a second API key."""
    config = make_config(api_key="the-key", pin="12345678")
    for path in ("/api/status", "/v1/models", "/v1/chat/completions"):
        with pytest.raises(AuthError):
            check_request(make_request(path, {"X-MCP-Pin": "12345678"}), config)


def test_api_key_still_works_on_the_mcp_path() -> None:
    config = make_config(api_key="the-key", pin="12345678")
    check_request(make_request("/mcp", {"Authorization": "Bearer the-key"}), config)


def test_mcp_pin_enforced_even_when_no_api_key_is_set() -> None:
    """An open server must not leave the control plane its least-guarded part."""
    config = make_config(api_key=None, pin="12345678")
    check_request(make_request("/api/status"), config)  # open, as documented
    with pytest.raises(AuthError):
        check_request(make_request("/mcp"), config)


def test_pin_can_be_disabled() -> None:
    config = make_config(api_key=None, pin="12345678", pin_required=False)
    check_request(make_request("/mcp"), config)


def test_health_needs_no_credential_even_with_a_pin() -> None:
    config = make_config(api_key="the-key", pin="12345678")
    check_request(make_request("/health"), config)


def test_deep_health_requires_the_api_key() -> None:
    """/health?deep=true runs a real completion against every loaded model.

    The shallow liveness form must stay credential-free (watchdogs poll it),
    but the deep form is genuine inference and must never ride the public
    exemption.
    """
    config = make_config(api_key="the-key")
    check_request(make_request("/health"), config)  # shallow: still free
    check_request(make_request("/health", query="deep=false"), config)
    check_request(make_request("/health", query="deep=0"), config)
    with pytest.raises(AuthError):
        check_request(make_request("/health", query="deep=true"), config)
    with pytest.raises(AuthError):
        check_request(make_request("/health", query="deep=1"), config)
    # Junk values fail closed rather than sneaking past the gate.
    with pytest.raises(AuthError):
        check_request(make_request("/health", query="deep=banana"), config)
    # With the key it works exactly like any authenticated request.
    check_request(
        make_request("/health", headers={"Authorization": "Bearer the-key"}, query="deep=true"),
        config,
    )


def test_deep_health_stays_open_when_no_key_is_configured() -> None:
    config = make_config(api_key=None, pin=None)
    check_request(make_request("/health", query="deep=true"), config)


def test_non_ascii_credential_is_a_401_not_a_500() -> None:
    """Starlette decodes headers as latin-1; a byte >= 0x80 must not crash auth.

    ``hmac.compare_digest`` raises TypeError for non-ASCII str operands, which
    would escape the middleware as a 500 with a traceback instead of a 401.
    """
    config = make_config(api_key="the-key", pin="12345678")
    with pytest.raises(AuthError):
        check_request(
            make_request("/api/status", headers={"Authorization": "Bearer kéy"}), config
        )
    with pytest.raises(AuthError):
        check_request(make_request("/mcp", headers={"X-MCP-Pin": "pïn"}), config)


def test_no_credentials_configured_means_open() -> None:
    config = make_config(api_key=None, pin=None)
    check_request(make_request("/mcp"), config)
    check_request(make_request("/api/status"), config)


def test_generated_pin_is_persisted_and_stable(tmp_path: Any) -> None:
    """A pairing code that changes on restart breaks every paired client.

    The PIN is minted when absent; if that mint is not written back, the next
    start mints a different one and anything already paired is locked out.
    """
    from studioforge.config import load_config

    config_path = tmp_path / "config.yaml"
    first = load_config(config_path, create=True)
    assert first.mcp.pin, "a pin should be minted on first run"
    assert config_path.is_file()

    second = load_config(config_path)
    assert second.mcp.pin == first.mcp.pin, "the pin changed across a reload"

    # And it must actually be in the file, not just in memory.
    import yaml

    on_disk = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert on_disk["mcp"]["pin"] == first.mcp.pin


def test_pin_minted_into_a_preexisting_config_is_written_back(tmp_path: Any) -> None:
    """Upgrading an install that predates the PIN must still be stable."""
    import yaml

    from studioforge.config import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"server": {"port": 1234}}), encoding="utf-8"
    )

    first = load_config(config_path)
    assert first.mcp.pin
    second = load_config(config_path)
    assert second.mcp.pin == first.mcp.pin


# ---------------------------------------------------------------------------
# WP17 F2/F3: the PIN is a credential, so no open surface may hand it out
# ---------------------------------------------------------------------------


def _request_from(host: str | None) -> Any:
    scope: dict[str, Any] = {
        "type": "http",
        "method": "GET",
        "path": "/api/mcp/info",
        "headers": [],
        "query_string": b"",
        "scheme": "http",
        "server": ("0.0.0.0", 1234),
        "client": (host, 5000) if host else None,
        "root_path": "",
    }
    return Request(scope)


def test_pin_withheld_from_remote_callers_when_no_api_key() -> None:
    """Shipped default: api_key unset, bound to 0.0.0.0. The PIN is then the only
    credential on the MCP plane, so an open endpoint must not return it to the LAN."""
    from studioforge.api.auth import may_reveal_pin

    config = make_config(api_key=None, pin="12345678")
    assert may_reveal_pin(_request_from("192.168.1.77"), config) is False
    assert may_reveal_pin(_request_from("127.0.0.1"), config) is True
    # An in-process call (the GUI invoking the handler) has no peer and is trusted.
    assert may_reveal_pin(_request_from(None), config) is True


def test_pin_revealed_when_a_credential_was_required() -> None:
    from studioforge.api.auth import may_reveal_pin

    config = make_config(api_key="the-key", pin="12345678")
    assert may_reveal_pin(_request_from("192.168.1.77"), config) is True


def test_redact_config_dict_covers_every_secret_including_the_pin() -> None:
    from studioforge.api.auth import SECRET_CONFIG_PATHS, redact_config_dict

    config = make_config(api_key="key-value-1234567890", pin="12345678")
    config.hf.token = "hf_abcdefghijklmnopqrstuvwxyz"
    data = redact_config_dict(config.to_yaml_dict())
    assert data["server"]["api_key"] != "key-value-1234567890"
    assert data["mcp"]["pin"] == "***"
    assert data["hf"]["token"].startswith("hf_a") and "..." in data["hf"]["token"]
    assert ("mcp", "pin") in SECRET_CONFIG_PATHS
