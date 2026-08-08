"""Replay a recorded native run without invoking the NMPC solver."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from quadrotor_mpc.application.native.commands import CommandName
from quadrotor_mpc.application.native.runtime import CoupledRunContext, CoupledStep
from quadrotor_mpc.control.nmpc.core import DRONE_RADIUS
from quadrotor_mpc.core.obstacle_motion import predict_obstacle_positions
from quadrotor_mpc.infrastructure.mujoco.plant import MuJoCoPlant


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
                    estimated_state_13=(
                        recording["estimated_states"][index]
                        if len(recording.get("estimated_states", [])) == len(rows)
                        else state
                    ),
                    error_covariance_12=(
                        recording["error_covariances"][index]
                        if len(recording.get("error_covariances", [])) == len(rows)
                        else np.zeros((12, 12), dtype=float)
                    ),
                    estimated_obstacle_states=None,
                    obstacle_covariances=None,
                    estimated_obstacle_predictions=(
                        recording["estimated_obstacle_predictions"][index]
                        if len(recording.get("estimated_obstacle_predictions", [])) == len(rows)
                        else obstacle_predictions
                    ),
                    vehicle_measurement_available=True,
                    obstacle_measurement_available=None,
                    vehicle_measurement_state_13=None,
                    obstacle_measurement_positions=None,
                    predicted_positions=prediction,
                    min_clearance_m=clearance,
                    goal_distance_m=float(row["goal_distance_m"]),
                    solver_time_ms=float(row["solver_time_ms"]),
                    collided=collided,
                    paused=paused,
                    predicted_covariances=(
                        recording["predicted_error_covariance_horizons"][index]
                        if len(
                            recording.get(
                                "predicted_error_covariance_horizons",
                                [],
                            )
                        )
                        == len(rows)
                        else None
                    ),
                    predicted_obstacle_covariances=(
                        recording["predicted_obstacle_covariance_horizons"][index]
                        if len(
                            recording.get(
                                "predicted_obstacle_covariance_horizons",
                                [],
                            )
                        )
                        == len(rows)
                        else None
                    ),
                    chance_margins=(
                        recording["chance_residual_horizons"][index]
                        if len(recording.get("chance_residual_horizons", [])) == len(rows)
                        else None
                    ),
                    risk_allocations=(
                        recording["risk_allocation_horizons"][index]
                        if len(recording.get("risk_allocation_horizons", [])) == len(rows)
                        else None
                    ),
                    slacks=(
                        recording["slack_horizons"][index]
                        if len(recording.get("slack_horizons", [])) == len(rows)
                        else None
                    ),
                    solver_status=str(row.get("solver_status", "")),
                    primary_solver_status=str(
                        row.get("primary_solver_status", "")
                    ),
                    primary_solver_success=bool(
                        int(row.get("primary_solver_success", 1) or 0)
                    ),
                    primary_solver_iterations=int(
                        row.get("primary_solver_iterations", 0) or 0
                    ),
                    primary_solver_primal_residual=float(
                        row.get("primary_solver_primal_residual", 0.0) or 0.0
                    ),
                    primary_solver_dual_residual=float(
                        row.get("primary_solver_dual_residual", 0.0) or 0.0
                    ),
                    command_source=str(
                        row.get("command_source", "PRIMARY_NMPC")
                    ),
                    solution_accepted=bool(
                        int(row.get("solution_accepted", 1) or 0)
                    ),
                    fallback_active=bool(
                        int(row.get("fallback_active", 0) or 0)
                    ),
                    fallback_level=int(
                        row.get("fallback_level", 0) or 0
                    ),
                    fallback_reason=str(row.get("fallback_reason", "")),
                    consecutive_rejections=int(
                        row.get("consecutive_rejections", 0) or 0
                    ),
                    deadline_missed=bool(
                        int(row.get("deadline_missed", 0) or 0)
                    ),
                    safety_assurance_status=str(
                        row.get("safety_assurance_status", "")
                    ),
                    residual_status=str(
                        row.get("residual_status", "UNAVAILABLE")
                    ),
                    horizon_assurance_status=str(
                        row.get("horizon_assurance_status", "")
                    ),
                    horizon_assurance_eligible=bool(
                        int(row.get("horizon_assurance_eligible", 0) or 0)
                    ),
                    horizon_assurance_reason=str(
                        row.get("horizon_assurance_reason", "")
                    ),
                    horizon_assurance_failed_checks=tuple(
                        check
                        for check in str(
                            row.get("horizon_assurance_failed_checks", "")
                        ).split(";")
                        if check
                    ),
                    assurance_schema_version=int(
                        row.get("assurance_schema_version", 2) or 2
                    ),
                    projected_uncertainties=(
                        recording["projected_uncertainty_horizons"][index]
                        if len(recording.get("projected_uncertainty_horizons", []))
                        == len(rows)
                        else None
                    ),
                    tightened_safety_radii=(
                        recording["tightened_safety_radius_horizons"][index]
                        if len(recording.get("tightened_safety_radius_horizons", []))
                        == len(rows)
                        else None
                    ),
                    risk_semantics=str(row.get("risk_semantics", "")),
                    risk_allocation_method=str(
                        row.get("risk_allocation_method", "")
                    ),
                    risk_budget_total=(
                        None
                        if row.get("risk_budget_total", "") in ("", None)
                        else float(row["risk_budget_total"])
                    ),
                    risk_budget_allocated=float(
                        row.get("risk_budget_allocated", 0.0) or 0.0
                    ),
                    risk_budget_remaining=(
                        None
                        if row.get("risk_budget_remaining", "") in ("", None)
                        else float(row["risk_budget_remaining"])
                    ),
                    risk_constraint_count=int(
                        row.get("risk_constraint_count", 0) or 0
                    ),
                    risk_budget_status=str(
                        row.get("risk_budget_status", "")
                    ),
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
