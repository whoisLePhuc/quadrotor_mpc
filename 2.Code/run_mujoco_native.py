#!/usr/bin/env python3
"""Run the production NMPC/MuJoCo loop in a native desktop window."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from mujoco_native import NativeMuJoCoViewer, load_native_mujoco_config
from run_coupled import run_coupled_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/mujoco_native.yaml")
    parser.add_argument("--duration", type=float, help="override simulation duration in seconds")
    parser.add_argument(
        "--camera",
        choices=("follow", "fixed"),
        help="override the configured camera mode",
    )
    parser.add_argument(
        "--realtime-factor",
        type=float,
        help="1.0 is real time; 0 disables wall-clock pacing",
    )
    parser.add_argument("--no-trail", action="store_true")
    parser.add_argument("--no-prediction", action="store_true")
    parser.add_argument("--show-contacts", action="store_true")
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="validate YAML and exit without importing MuJoCo",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = base / config_path
    config = load_native_mujoco_config(config_path)

    if args.duration is not None:
        if args.duration <= 0.0:
            raise SystemExit("--duration must be > 0")
        config = replace(config, duration_s=float(args.duration))

    viewer = config.viewer
    if args.realtime_factor is not None:
        if args.realtime_factor < 0.0:
            raise SystemExit("--realtime-factor must be >= 0")
        viewer = replace(viewer, realtime_factor=float(args.realtime_factor))
    if args.camera is not None:
        viewer = replace(viewer, camera_mode=args.camera)
    if args.no_trail:
        viewer = replace(viewer, show_trail=False)
    if args.no_prediction:
        viewer = replace(viewer, show_prediction=False)
    if args.show_contacts:
        viewer = replace(viewer, show_contacts=True)

    print(f"configuration: {config_path}")
    print(f"scenario:      {config.name}")
    print(
        f"timing:        MPC {config.mpc_timestep_s:.3f}s | "
        f"MuJoCo {config.mujoco_timestep_s:.4f}s | duration {config.duration_s:.1f}s"
    )
    if args.validate_config:
        print("configuration is valid")
        return 0

    runtime = NativeMuJoCoViewer(viewer)
    result = run_coupled_simulation(
        x0_vals=config.start,
        goal_pos=config.goal_position,
        goal_euler=config.goal_euler,
        bounds=config.bounds,
        obstacles=[dict(item) for item in config.obstacles],
        margin=config.safety_margin,
        sim_seconds=config.duration_s,
        mpc_dt=config.mpc_timestep_s,
        n_horizon=config.horizon_steps,
        max_iter=config.max_solver_iterations,
        mj_dt=config.mujoco_timestep_s,
        runtime=runtime,
        stop_on_goal=config.stop_on_goal,
        goal_tolerance=config.goal_tolerance_m,
        stop_on_collision=config.stop_on_collision,
    )

    if len(result["pos"]):
        final_position = result["pos"][-1]
        final_error = float(
            np.linalg.norm(
                final_position
                - np.array(
                    [
                        config.goal_position["x"],
                        config.goal_position["y"],
                        config.goal_position["z"],
                    ]
                )
            )
        )
        min_clearance = float(np.min(result["clearance"]))
        print(f"final position: {np.array2string(final_position, precision=3)}")
        print(f"final error:    {final_error:.3f} m")
        print(f"min clearance:  {min_clearance:.3f} m")
    print(f"collision:      {result['collided']}")
    print(f"termination:    {result['termination_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
