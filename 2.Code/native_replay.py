"""Replay a recorded native run without invoking the NMPC solver."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from mujoco_plant import MuJoCoPlant
from obstacle_motion import predict_obstacle_positions
from quad_mpc_core import DRONE_RADIUS
from run_coupled import CoupledRunContext, CoupledStep
from runtime_control import CommandName


def replay_native_recording(config: Any, recording: dict[str, Any], runtime: Any) -> dict[str, Any]:
    rows = recording["rows"]
    states = recording["states"]
    if len(rows) == 0:
        raise ValueError("recording contains no telemetry samples")

    obstacles = [dict(item) for item in config.obstacles]
    plant = MuJoCoPlant(
        config.start,
        config.goal_position,
        obstacles,
        mj_dt=config.mujoco_timestep_s,
    )
    goal = np.asarray(
        [
            config.goal_position["x"],
            config.goal_position["y"],
            config.goal_position["z"],
        ],
        dtype=float,
    )
    context = CoupledRunContext(
        start_position=states[0, :3].copy(),
        goal_position=goal,
        obstacles=tuple(obstacles),
        safety_radii=np.asarray(
            [obstacle["radius"] + config.safety_margin + DRONE_RADIUS for obstacle in obstacles],
            dtype=float,
        ),
        controller_timestep_s=config.mpc_timestep_s,
        horizon_steps=config.horizon_steps,
        total_steps=len(rows),
    )
    paused = False
    step_once = False
    index = 0
    termination = "replay_completed"
    positions: list[np.ndarray] = []
    clearances: list[float] = []
    collided = False

    try:
        runtime.open(plant, context)
        while index < len(rows):
            if not runtime.is_running():
                termination = "viewer_closed"
                break
            for command in runtime.poll_commands():
                name = command.name
                if name == CommandName.STOP:
                    termination = "user_stopped"
                    index = len(rows)
                    break
                if name == CommandName.TOGGLE_PAUSE:
                    paused = not paused
                elif name == CommandName.STEP:
                    paused = True
                    step_once = True
                elif name == CommandName.RESET:
                    index = 0
                    positions.clear()
                    clearances.clear()
                    collided = False
                    runtime.on_reset()
            if index >= len(rows):
                break
            if paused and not step_once:
                runtime.on_idle(True)
                time.sleep(0.01)
                continue

            row = rows[index]
            state = states[index]
            time_s = float(row["time_s"])
            if len(recording["obstacle_predictions"]) == len(rows):
                obstacle_predictions = recording["obstacle_predictions"][index]
            else:
                obstacle_predictions = predict_obstacle_positions(
                    obstacles,
                    time_s,
                    config.horizon_steps + 1,
                    config.mpc_timestep_s,
                )
            obstacle_positions = obstacle_predictions[:, 0, :]
            plant.set_state_13(state, obstacle_positions=obstacle_positions)
            prediction = None
            if len(recording["predicted_positions"]) == len(rows):
                candidate = recording["predicted_positions"][index]
                if candidate.size:
                    prediction = candidate
            control = np.asarray(
                [
                    row["thrust_deviation"],
                    row["tau_x"],
                    row["tau_y"],
                    row["tau_z"],
                ],
                dtype=float,
            )
            collided = collided or bool(int(row["collided"]))
            clearance = float(row["min_clearance_m"])
            keep_running = runtime.on_step(
                CoupledStep(
                    step_index=int(row["step_index"]),
                    time_s=time_s,
                    state_13=state,
                    control=control,
                    obstacle_positions=obstacle_positions,
                    obstacle_predictions=obstacle_predictions,
                    predicted_positions=prediction,
                    min_clearance_m=clearance,
                    goal_distance_m=float(row["goal_distance_m"]),
                    solver_time_ms=float(row["solver_time_ms"]),
                    collided=collided,
                    paused=paused,
                )
            )
            if not keep_running:
                termination = "viewer_closed"
                break
            positions.append(state[:3].copy())
            clearances.append(clearance)
            index += 1
            step_once = False
    finally:
        runtime.close()

    return {
        "t": np.asarray([float(row["time_s"]) for row in rows[: len(positions)]]),
        "pos": np.asarray(positions),
        "clearance": np.asarray(clearances),
        "collided": collided,
        "termination_reason": termination,
        "replay": True,
    }
