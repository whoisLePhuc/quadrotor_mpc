#!/usr/bin/env python3
"""Fail fast when v2.0.1 release metadata or runtime contracts drift."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import tomllib

CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mujoco_native import load_native_mujoco_config  # noqa: E402
from native_monte_carlo import load_native_monte_carlo_protocol  # noqa: E402

EXPECTED_VERSION = "2.0.1"
NATIVE_CONFIGS = (
    "mujoco_native.yaml",
    "mujoco_native_dynamic.yaml",
    "mujoco_native_estimation.yaml",
    "mujoco_native_ccmpc.yaml",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    project = tomllib.loads((CODE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    require(project["project"]["version"] == EXPECTED_VERSION, "pyproject version mismatch")
    require(
        f"## {EXPECTED_VERSION} " in (CODE_ROOT / "CHANGELOG.md").read_text(encoding="utf-8"),
        "CHANGELOG is missing the release heading",
    )
    require((REPOSITORY_ROOT / "LICENSE").is_file(), "root LICENSE is missing")
    require((CODE_ROOT / "LICENSE").is_file(), "wheel LICENSE is missing")
    require((REPOSITORY_ROOT / ".github/workflows/ci.yml").is_file(), "CI workflow is missing")
    require((CODE_ROOT / "uv.lock").is_file(), "uv.lock is missing")

    scripts = project["project"].get("scripts", {})
    expected_scripts = {
        "quadrotor-mpc-run",
        "quadrotor-mpc-sim",
        "quadrotor-mpc-sweep",
        "quadrotor-mpc-native",
        "quadrotor-mpc-monte-carlo",
        "quadrotor-mpc-dashboard",
    }
    require(expected_scripts <= set(scripts), "console-script packaging is incomplete")

    config_summaries: dict[str, dict[str, float | bool]] = {}
    for filename in NATIVE_CONFIGS:
        config = load_native_mujoco_config(CODE_ROOT / "config" / filename)
        if config.safety_fallback.enabled and config.safety_fallback.reject_on_deadline_miss:
            require(
                config.safety_fallback.solve_deadline_s <= config.mpc_timestep_s,
                f"{filename}: solve deadline exceeds the MPC period",
            )
        config_summaries[filename] = {
            "period_s": config.mpc_timestep_s,
            "deadline_s": config.safety_fallback.solve_deadline_s,
            "supervisor": config.safety_fallback.enabled,
        }

    ccmpc = load_native_mujoco_config(CODE_ROOT / "config" / "mujoco_native_ccmpc.yaml")
    require(ccmpc.safety_fallback.enabled, "CC-MPC safety supervisor must be enabled")
    require(
        abs(ccmpc.safety_fallback.solve_deadline_s - ccmpc.mpc_timestep_s) <= 1e-12,
        "CC-MPC acceptance deadline must equal its control period",
    )

    protocol = load_native_monte_carlo_protocol(
        CODE_ROOT / "config" / "native_monte_carlo.yaml"
    )
    require(protocol.trials == 50, "release Monte Carlo must use 50 trials per cell")
    require(protocol.minimum_trials_for_claim <= protocol.trials, "claim sample gate is impossible")
    require(protocol.base_config_path.is_file(), "Monte Carlo base config is missing")

    root_readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    forbidden_claims = (
        "Guarantees probabilistic safety",
        "Runs in real time — ~14 ms",
        "config/scenarios/two_static.yaml",
        "config/scenarios/one_moving.yaml",
        "notebooks/interactive_demo.ipynb",
    )
    for claim in forbidden_claims:
        require(claim not in root_readme, f"README retains unsupported claim/path: {claim}")
    require("VALIDATED_WITH_LIMITATIONS" in root_readme, "README omits validation status")

    print(
        json.dumps(
            {
                "status": "PASS",
                "version": EXPECTED_VERSION,
                "native_configs": config_summaries,
                "monte_carlo_trials_per_cell": protocol.trials,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
