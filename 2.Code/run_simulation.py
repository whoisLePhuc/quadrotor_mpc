#!/usr/bin/env python3
"""Command-line entry point for ODE-based MPC/CC-MPC experiments."""

from __future__ import annotations

import argparse
import json

import numpy as np

from resource_paths import resolve_input_path, resolve_output_path
from simulation.config import load_scenario
from simulation.runner import SimulationResult, SimulationRunner
from simulation.visualizer import save_report


def _print_metrics(result: SimulationResult) -> None:
    metrics = result.metrics
    clearance = "n/a" if metrics.min_clearance_m is None else f"{metrics.min_clearance_m:.3f} m"
    print(
        f"[{result.mode:13s}] success={metrics.success!s:5s} collision={metrics.collision!s:5s} "
        f"final={metrics.final_error_m:.3f} m clearance={clearance} "
        f"solve_mean={metrics.mean_solver_ms:.1f} ms p95={metrics.p95_solver_ms:.1f} ms "
        f"deadline_miss={metrics.deadline_miss_rate:.0%}"
    )


def _aggregate(results: list[SimulationResult]) -> dict:
    metrics = [item.metrics for item in results]
    clearances = [item.min_clearance_m for item in metrics if item.min_clearance_m is not None]
    return {
        "mode": results[0].mode,
        "trials": len(results),
        "success_rate": float(np.mean([item.success for item in metrics])),
        "collision_rate": float(np.mean([item.collision for item in metrics])),
        "mean_final_error_m": float(np.mean([item.final_error_m for item in metrics])),
        "mean_tracking_rmse_m": float(np.mean([item.tracking_rmse_m for item in metrics])),
        "worst_min_clearance_m": float(min(clearances)) if clearances else None,
        "mean_solver_ms": float(np.mean([item.mean_solver_ms for item in metrics])),
        "p95_solver_ms": float(np.percentile([item.p95_solver_ms for item in metrics], 95)),
        "mean_deadline_miss_rate": float(np.mean([item.deadline_miss_rate for item in metrics])),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/scenarios/static_obstacle.yaml")
    parser.add_argument("--controller-config", default="config/controller.yaml")
    parser.add_argument("--mode", choices=("ccmpc", "deterministic"), default="ccmpc")
    parser.add_argument("--backend", choices=("scipy", "cvxpy"), default="scipy")
    parser.add_argument("--compare", action="store_true", help="run deterministic MPC and CC-MPC")
    parser.add_argument("--trials", type=int, default=1, help="number of Monte Carlo seeds")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.trials < 1:
        raise SystemExit("--trials must be >= 1")
    scenario_path = resolve_input_path(args.config)
    controller_path = resolve_input_path(args.controller_config)
    output = resolve_output_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scenario = load_scenario(scenario_path)
    modes = ["deterministic", "ccmpc"] if args.compare else [args.mode]
    initial_seed = scenario.seed if args.seed is None else args.seed

    grouped: dict[str, list[SimulationResult]] = {mode: [] for mode in modes}
    for trial in range(args.trials):
        seed = initial_seed + trial
        for mode in modes:
            result = SimulationRunner(
                scenario,
                controller_path,
                mode=mode,
                backend=args.backend,
                seed=seed,
            ).run()
            result.save(output)
            grouped[mode].append(result)
            _print_metrics(result)

    aggregate = {mode: _aggregate(results) for mode, results in grouped.items()}
    aggregate_path = output / f"{scenario.name}-aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"aggregate: {aggregate_path}")

    if not args.no_plot:
        representative = [grouped[mode][0] for mode in modes]
        report_path = save_report(representative, scenario, output / f"{scenario.name}-report.png")
        print(f"report:    {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
