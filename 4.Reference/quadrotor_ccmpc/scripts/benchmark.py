#! /usr/bin/env python3
"""
Benchmark CC-MPC across scenarios, solvers, and configs.

Usage:
    python scripts/benchmark.py                          # all tests
    python scripts/benchmark.py --scenario corridor       # single scenario
    python scripts/benchmark.py --solver OSQP             # single solver
    python scripts/benchmark.py --quick                   # minimal test

Output: benchmark_results.md
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Import CCMPC directly (no display needed)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from ccmpc.ccmpc import CCMPC
from ccmpc.dynamics import QuadrotorDynamics
from ccmpc.obstacle import EllipsoidalObstacle, ObstacleManager
from ccmpc.uncertainty import UncertaintyPropagator as UP


@dataclass
class BenchResult:
    scenario: str = ""
    solver: str = ""
    fov: str = ""
    steps: int = 0
    goal_reached: bool = False
    final_error: float = 0.0
    solve_times_ms: list[float] = field(default_factory=list)
    cte_rmse: float = 0.0
    hdg_rmse_deg: float = 0.0
    failures: int = 0
    status: str = ""


def run_bench(
    scenario: str,
    solver: str,
    fov_enabled: bool,
    max_steps: int = 500,
    goal_threshold: float = 0.4,
) -> BenchResult:
    """Run a single benchmark test."""
    res = BenchResult(scenario=scenario, solver=solver,
                      fov="ON" if fov_enabled else "OFF")

    # Load scenario
    pkg = Path(__file__).parent.parent
    import yaml
    scenario_path = pkg / "config" / f"{scenario}.yaml"
    if not scenario_path.exists():
        scenario_path = pkg / "config" / "simulation.yaml"
    sim_cfg = yaml.safe_load(scenario_path.read_text())

    start = sim_cfg["start"]
    goal = np.array(sim_cfg["goal"], dtype=np.float64)

    # Build config
    cfg = {
        "model": {"quadrotor": {"g":9.81,"kD":0.5,"k_phi":1,"k_theta":1,"k_vz":1,
                                "tau_phi":0.2,"tau_theta":0.2,"tau_vz":0.4}},
        "controller": {
            "prediction": {"horizon_time":1.2,"timestep":0.06,"max_iter":2,
                           "tolerance":0.01,"solver":solver},
            "weights": {"terminal_cost":[10,10,10],"control_cost":[0.1,0.1,0.5,0.05],
                        "yaw_cost":5.0,"logistic_cost":{"Q_o":0.5,"lambda_o":2.0,"r_o":1.5}},
            "fov": {"enabled":fov_enabled,"hfov_deg":170,"vfov_deg":120,
                    "max_range":10,"slack_penalty":10},
            "obstacle": {"delta":0.03,"mav_radius":0.4,"safety_margin":0.1,
                         "slack_penalty":1000,"max_obstacles":2},
            "limits":{"max_roll":0.35,"max_pitch":0.35,"max_vert_vel":1.0,
                      "max_yaw_rate":2.0,"max_speed":3.0},
            "uncertainty":{"process_noise_pos":0.01,"process_noise_vel":0.1,
                          "process_noise_att":0.02,"init_pos_noise":0.05,
                          "init_vel_noise":0.1,"init_att_noise":0.03}
        }
    }

    mpc = CCMPC(cfg)
    dynamics = mpc.dynamics
    obs_mgr = ObstacleManager.from_config(str(scenario_path))
    state = np.array(start, dtype=np.float64)
    init_state = state.copy()

    # Run simulation
    solve_times = []
    cte_hist = []
    hdg_err_hist = []
    failures = 0

    res.steps = max_steps
    for i in range(max_steps):
        # Check goal
        if np.linalg.norm(state[:3] - goal) < goal_threshold:
            res.goal_reached = True
            res.steps = i
            break

        # MPC update every 3 sim steps (dt=0.06s / 0.02s)
        if i % 3 == 0:
            noisy = UP.add_measurement_noise(state)
            obs_mgr.update(mpc.dt)
            t0 = time.perf_counter()
            x_t, u_t = mpc.solve(noisy, goal, obstacle_manager=obs_mgr)
            t_solve = time.perf_counter() - t0
            solve_times.append(t_solve * 1000)
            if mpc._problem.status != 'optimal':
                failures += 1
            last_control = u_t[:, 0].copy() if u_t is not None else np.zeros(4)
        else:
            last_control = last_control if 'last_control' in dir() else np.zeros(4)

        state = dynamics.discrete(state, last_control, 0.02)

        # Metrics
        goal_vec = goal - init_state[:3]
        goal_dir = goal_vec / max(np.linalg.norm(goal_vec), 1e-6)
        mav_vec = state[:3] - init_state[:3]
        cte = float(np.linalg.norm(mav_vec - np.dot(mav_vec, goal_dir) * goal_dir))
        cte_hist.append(cte)
        desired_yaw = math.atan2(goal[1] - state[1], goal[0] - state[0])
        hdg_err = (state[8] - desired_yaw + math.pi) % (2 * math.pi) - math.pi
        hdg_err_hist.append(abs(hdg_err))

    res.final_error = float(np.linalg.norm(state[:3] - goal))
    res.solve_times_ms = solve_times
    res.failures = failures
    res.status = mpc._problem.status if hasattr(mpc._problem, 'status') else 'N/A'

    if cte_hist:
        res.cte_rmse = float(np.sqrt(np.mean(np.square(cte_hist))))
    if hdg_err_hist:
        res.hdg_rmse_deg = float(np.degrees(np.sqrt(np.mean(np.square(hdg_err_hist)))))

    return res


def print_table(results: list[BenchResult]) -> str:
    """Generate markdown table."""
    lines = [
        "| Scenario | Solver | FOV | Goal? | Steps | Avg ms | P95 ms | Failures | CTE RMSE | HDG RMSE |",
        "|----------|--------|-----|-------|-------|--------|--------|----------|----------|----------|",
    ]
    for r in results:
        avg = np.mean(r.solve_times_ms) if r.solve_times_ms else 0
        p95 = sorted(r.solve_times_ms)[int(len(r.solve_times_ms) * 0.95)] if len(r.solve_times_ms) > 1 else avg
        goal = "✅" if r.goal_reached else "❌"
        lines.append(
            f"| {r.scenario:8s} | {r.solver:7s} | {r.fov:3s} "
            f"| {goal} | {r.steps:5d} | {avg:6.0f} | {p95:6.0f} "
            f"| {r.failures:8d} | {r.cte_rmse:.3f} | {r.hdg_rmse_deg:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="CC-MPC Benchmark")
    parser.add_argument("--scenario", default=None, help="Single scenario")
    parser.add_argument("--solver", default=None, help="Single solver")
    parser.add_argument("--quick", action="store_true", help="Minimal test")
    parser.add_argument("--output", default="benchmark_results.md", help="Output file")
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else ["simulation", "simulation_static", "simulation_corridor"]
    solvers = [args.solver] if args.solver else ["CLARABEL", "OSQP"]
    fov_states = [False, True] if not args.quick else [False]

    results: list[BenchResult] = []
    total = len(scenarios) * len(solvers) * len(fov_states)
    count = 0

    print(f"Benchmark: {total} runs\n")
    for sc in scenarios:
        for sv in solvers:
            for fov_on in fov_states:
                count += 1
                fov_label = "FOV ON" if fov_on else "no FOV"
                print(f"  [{count}/{total}] {sc} / {sv} / {fov_label}...", end=" ", flush=True)
                try:
                    r = run_bench(sc, sv, fov_on)
                    avg = np.mean(r.solve_times_ms) if r.solve_times_ms else 0
                    status = "✅" if r.goal_reached else f"❌ ({r.final_error:.2f}m)"
                    print(f"{status}  avg={avg:.0f}ms  steps={r.steps}")
                    results.append(r)
                except Exception as e:
                    print(f"❌ ERROR: {e}")

    md = print_table(results)
    print(f"\n---\n{md}")

    out_path = Path(args.output)
    out_path.write_text(f"# CC-MPC Benchmark Results\n\n{md}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
