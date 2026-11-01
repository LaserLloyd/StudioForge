"""STRICT_GPU: quoted and look-alike-dash spellings, and device-filter variables."""

from __future__ import annotations

import pytest

from studioforge.core.engine import (
    child_environment,
    normalize_flag_token,
    policy_family,
    refuse_policy_flags,
)
from studioforge.core.supervisor import launch_policy_violations


@pytest.mark.parametrize(
    "token",
    [
        '"--cpu-moe"',
        "'--cpu-moe'",
        "‐‐cpu-moe",
        "——cpu-moe",
        "−−cpu-moe",
        "--cpu‑moe",
        '"——cpu-moe"',
    ],
)
def test_disguised_offload_flag_is_still_refused(token: str) -> None:
    assert policy_family(token) is not None
    assert refuse_policy_flags([token])
    assert launch_policy_violations(["llama-server", token])


def test_disguised_value_flag_is_read_by_the_launch_report() -> None:
    argv = ["llama-server", '"--n-gpu-layers"', "12"]
    assert launch_policy_violations(argv) == ["--n-gpu-layers 12"]
    assert launch_policy_violations(["llama-server", "——fit", "on"]) == ["--fit on"]


def test_harmless_tokens_are_untouched() -> None:
    assert normalize_flag_token("--ctx-size") == "--ctx-size"
    assert refuse_policy_flags(["--temp", "0.5", "-1"]) == []
    assert launch_policy_violations(["llama-server", "--temp", "0.5"]) == []


def test_visible_device_variables_are_stripped() -> None:
    names = [
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GGML_VK_VISIBLE_DEVICES",
        "hip_visible_devices",
    ]
    env, stripped = child_environment({**dict.fromkeys(names, "1"), "PATH": "/bin"})
    assert env == {"PATH": "/bin"}
    assert sorted(stripped) == sorted(names)
