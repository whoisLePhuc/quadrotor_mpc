#!/usr/bin/env python3
"""Run a reproducible MPC experiment and generate a complete report bundle."""

from __future__ import annotations

import argparse

from quadrotor_mpc.application.experiments.manager import save_experiment
from quadrotor_mpc.application.simulation.config import load_scenario
from quadrotor_mpc.application.simulation.runner import SimulationResult, SimulationRunner
from quadrotor_mpc.infrastructure.resources import resolve_input_path, resolve_output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/scenarios/static_obstacle.yaml")
    parser.add_argument("--controller-config", default="config/controller.yaml")
    parser.add_argument("--mode", choices=("ccmpc", "deterministic"), default="ccmpc")
    parser.add_argument("--backend", choices=("scipy", "cvxpy"), default="scipy")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--delta", type=float, default=None, help="override CC-MPC risk delta")
    parser.add_argument("--fov", action="store_true", help="enable FOV soft constraints")
    parser.add_argument("--output-root", default="outputs")
    return parser


def _print_result(result: SimulationResult) -> None:
    metrics = result.metrics
    clearance = "N/A" if metrics.min_clearance_m is None else f"{metrics.min_clearance_m:.3f} m"
    print(
        f"[{result.mode:13s} seed={result.seed:04d}] "
        f"success={metrics.success} collision={metrics.collision} "
        f"final={metrics.final_error_m:.3f} m clearance={clearance} "
        f"p95={metrics.p95_solver_ms:.1f} ms"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.trials < 1:
        raise SystemExit("--trials must be >= 1")
    if args.delta is not None and not 0.0 < args.delta <= 0.5:
        raise SystemExit("--delta must be in (0, 0.5]")
    scenario_path = resolve_input_path(args.config)
    controller_path = resolve_input_path(args.controller_config)
    output_root = resolve_output_path(args.output_root)
    scenario = load_scenario(scenario_path)
    modes = ["deterministic", "ccmpc"] if args.compare else [args.mode]
    first_seed = scenario.seed if args.seed is None else args.seed
    override: dict = {}
    if args.delta is not None:
        override.setdefault("controller", {}).setdefault("obstacle", {})["delta"] = args.delta
    if args.fov:
        override.setdefault("controller", {}).setdefault("fov", {})["enabled"] = True

    results: list[SimulationResult] = []
    for trial in range(args.trials):
        for mode in modes:
            result = SimulationRunner(
                scenario,
                controller_path,
                mode=mode,
                backend=args.backend,
                seed=first_seed + trial,
                controller_override=override,
            ).run()
            results.append(result)
            _print_result(result)

    artifacts = save_experiment(
        results,
        scenario,
        controller_path,
        output_root,
        command=["quadrotor-mpc-run", *argv] if argv is not None else None,
    )
    print(f"run_id:  {artifacts.run_id}")
    print(f"folder:  {artifacts.directory}")
    print(f"report:  {artifacts.interactive_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
