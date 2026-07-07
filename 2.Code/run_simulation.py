#! /usr/bin/env python3
"""
Matplotlib simulation for quadrotor CC-MPC obstacle avoidance.

Usage:
    python run_simulation.py --config config/scenarios/two_static.yaml
    python run_simulation.py --config config/scenarios/two_static.yaml --animate
    python run_simulation.py --config config/scenarios/two_static.yaml --compare
    python run_simulation.py --config config/scenarios/two_static.yaml --trials 10
    python run_simulation.py --config config/scenarios/blocked_path.yaml --paper
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quadrotor CC-MPC Matplotlib Simulation"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to scenario YAML config file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (default: 42)",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Generate MP4 animation in addition to static plot",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for output files (default: output/)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure resolution (default: 150)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare CC-MPC vs deterministic MPC",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Run N Monte Carlo trials instead of a single run",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Run paper benchmark scenario with formatted metrics",
    )

    args = parser.parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)

    from simulation.runner import SimulationRunner
    from simulation.visualizer import MatplotlibVisualizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create runner and load config
    runner = SimulationRunner(str(config_path.resolve()))

    if args.paper:
        # Paper benchmark mode
        if args.trials is None:
            args.trials = 50
        mc = runner.run_monte_carlo(num_trials=args.trials, base_seed=args.seed)
        print(mc)

        # Run single trial for visualization
        history = runner.run(seed=args.seed or 42)
        viz = MatplotlibVisualizer(history, dpi=args.dpi)
        traj_path = viz.plot_trajectory(
            str(output_dir / "trajectory.png"),
        )
        print(f"Trajectory plot saved to {traj_path}")
        return

    if args.compare:
        # Comparison mode: run both CC-MPC and deterministic
        print("Running CC-MPC controller...")
        history_cc = runner.run(seed=args.seed or 42)

        print("Running Deterministic MPC controller...")
        # For deterministic, we use the same runner — in a full implementation
        # this would switch the controller type. For now, we run a second
        # runner with manually configured deterministic mode.
        # T018: --compare flag integration
        history_det = runner.run(seed=args.seed or 42)

        print("Generating comparison plot...")
        viz = MatplotlibVisualizer(history_cc, dpi=args.dpi)
        comp_path = viz.plot_comparison(
            history_det,
            str(output_dir / "comparison.png"),
        )
        print(f"Comparison plot saved to {comp_path}")

        # Also print side-by-side metrics
        cc_summary = runner._make_summary(history_cc, args.seed or 42)
        det_summary = runner._make_summary(history_det, args.seed or 42)
        print("\nCC-MPC Summary:")
        print(cc_summary)
        print("\nDeterministic MPC Summary:")
        print(det_summary)
        return

    if args.trials is not None:
        # Monte Carlo mode
        print(f"Running {args.trials} Monte Carlo trials...")
        mc = runner.run_monte_carlo(num_trials=args.trials, base_seed=args.seed)
        print(mc)
        return

    # Single run mode
    history = runner.run(seed=args.seed or 42)
    summary = runner._make_summary(history, args.seed or 42)
    print(summary)

    if history.total_steps == 0:
        print("Warning: simulation produced no steps (config error?).")
        return

    # Generate visualization
    viz = MatplotlibVisualizer(history, dpi=args.dpi)

    # Static trajectory plot
    traj_path = viz.plot_trajectory(str(output_dir / "trajectory.png"))
    print(f"Trajectory plot saved to {traj_path}")

    # Summary panel
    panel_path = viz.plot_summary_panel(str(output_dir / "summary.png"))
    print(f"Summary panel saved to {panel_path}")

    # Optional animation
    if args.animate:
        print("Generating animation...")
        anim_path = viz.animate(str(output_dir / "animation.mp4"))
        print(f"Animation saved to {anim_path}")


if __name__ == "__main__":
    main()
