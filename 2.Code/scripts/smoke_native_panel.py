#!/usr/bin/env python3
"""Create, update, reset and close the real Qt safety panel."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from quadrotor_mpc.interfaces.desktop.panel import DesktopPanelProcess  # noqa: E402
from quadrotor_mpc.interfaces.desktop.viewer import load_native_mujoco_config  # noqa: E402


def sample() -> dict[str, object]:
    return {
        "time_s": 0.05,
        "position": [0.0, 0.0, 1.0],
        "control": [0.0, 0.0, 0.0, 0.0],
        "goal_distance_m": 3.0,
        "min_clearance_m": 0.4,
        "solver_time_ms": 42.0,
        "collided": False,
        "paused": False,
        "completed": False,
        "solver_status": "SOLVED_SAFE",
        "primary_solver_status": "Solve_Succeeded",
        "primary_solver_success": True,
        "primary_solver_iterations": 10,
        "primary_solver_primal_residual": 1e-7,
        "primary_solver_dual_residual": 1e-7,
        "command_source": "PRIMARY_NMPC",
        "solution_accepted": True,
        "fallback_active": False,
        "fallback_level": 0,
        "fallback_reason": "",
        "deadline_missed": False,
        "safety_assurance_status": "GUARANTEE_ELIGIBLE",
        "risk_semantics": "joint",
        "risk_allocation_method": "uniform",
        "risk_budget_total": 0.10,
        "risk_budget_allocated": 0.10,
        "risk_budget_remaining": 0.0,
        "risk_constraint_count": 63,
        "risk_budget_status": "BUDGET_OK",
        "minimum_chance_residual_m": 0.01,
        "maximum_slack_m": 0.0,
        "horizon_terminal_position_sigma": [0.02, 0.02, 0.03],
        "maximum_projected_uncertainty_m": 0.03,
        "maximum_tightened_safety_radius_m": 0.75,
    }


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    config = load_native_mujoco_config(CODE_ROOT / "config" / "mujoco_native_ccmpc.yaml")
    panel = DesktopPanelProcess(
        config.panel,
        "v2.0.1 release smoke",
        config.panel_runtime_context(),
    )
    panel.start()
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline and not panel.is_alive():
        time.sleep(0.05)
    if not panel.is_alive():
        raise RuntimeError("Qt safety-panel process exited during startup")
    panel.publish(sample())
    time.sleep(0.4)
    panel.reset()
    time.sleep(0.2)
    panel.close()
    if panel.exitcode not in (0, None):
        raise RuntimeError(f"Qt safety-panel process exited with code {panel.exitcode}")
    print("Qt safety-panel smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
