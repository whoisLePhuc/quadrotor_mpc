#!/usr/bin/env python3
"""Run the production NMPC/MuJoCo loop in a native desktop window."""

from __future__ import annotations

import argparse
import importlib
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

from mujoco_native import (
    InteractiveMuJoCoRuntime,
    NativeMuJoCoConfig,
    load_native_mujoco_config,
)
from native_telemetry import load_native_recording


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
        "--no-panel",
        action="store_true",
        help="run only the MuJoCo window without the Qt control/plot panel",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="disable telemetry artifacts for this run",
    )
    parser.add_argument(
        "--replay",
        metavar="RUN_DIRECTORY",
        help="replay a native recording without solving NMPC",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="validate YAML and exit without importing MuJoCo",
    )
    return parser


def _resolve_input_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    from_cwd = (Path.cwd() / path).resolve()
    return from_cwd if from_cwd.exists() else (base / path).resolve()


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore", message="The ONNX feature is not available.*")
    warnings.filterwarnings("ignore", message="The opcua feature is not available.*")
    warnings.filterwarnings("ignore", message="The approximateMPC feature requires PyTorch.*")
    args = build_parser().parse_args(argv)
    base = Path(__file__).resolve().parent
    recording = None
    if args.replay:
        replay_path = _resolve_input_path(args.replay, base)
        recording = load_native_recording(replay_path)
        config = NativeMuJoCoConfig.from_mapping(recording["scenario"])
        config_label = replay_path / "scenario.yaml"
    else:
        config_path = _resolve_input_path(args.config, base)
        config = load_native_mujoco_config(config_path)
        config_label = config_path

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
    config = replace(config, viewer=viewer)

    print(f"configuration: {config_label}")
    print(f"scenario:      {config.name}")
    print(
        f"timing:        MPC {config.mpc_timestep_s:.3f}s | "
        f"MuJoCo {config.mujoco_timestep_s:.4f}s | duration {config.duration_s:.1f}s"
    )
    if args.validate_config:
        print("configuration is valid")
        return 0

    panel_enabled = config.panel.enabled and not args.no_panel
    if panel_enabled:
        try:
            importlib.import_module("PySide6.QtWidgets")
            importlib.import_module("pyqtgraph")
        except (ImportError, OSError) as exc:
            raise SystemExit(
                f"desktop panel could not start: {exc}. "
                "Install with `python -m pip install -r requirements-ui.txt` "
                "and ensure the Linux EGL/XCB runtime libraries are present, "
                "or pass --no-panel."
            ) from exc

    runtime = InteractiveMuJoCoRuntime(
        config,
        base_dir=base,
        enable_panel=panel_enabled,
        enable_recording=not args.no_record and recording is None,
    )
    if recording is not None:
        from native_replay import replay_native_recording

        print(f"replay:         {recording['directory']}")
        result = replay_native_recording(config, recording, runtime)
    else:
        from run_coupled import run_coupled_simulation

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
    recording_path = runtime.finalize(result) if recording is None else None

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
    if recording_path is not None:
        print(f"recording:      {recording_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
