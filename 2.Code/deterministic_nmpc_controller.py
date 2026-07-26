"""do-mpc adapter implementing the native belief-based controller contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from controller_interface import (
    ControlGoal,
    ControlSolution,
    ObstacleBelief,
    VehicleBelief,
)
from quad_mpc_core import (
    DRONE_RADIUS,
    STATE_NAMES,
    build_controller,
    build_model,
    make_mpc_tvp_fun,
)


def _extract_nominal_states(mpc, fallback_state: np.ndarray) -> np.ndarray:
    axes: list[np.ndarray] = []
    try:
        for name in STATE_NAMES:
            values = np.asarray(mpc.data.prediction(("_x", name)), dtype=float)
            axes.append(values.reshape(-1))
    except (AttributeError, KeyError, TypeError, ValueError):
        return fallback_state.reshape(1, 13).copy()
    length = min((len(axis) for axis in axes), default=0)
    if length < 1:
        return fallback_state.reshape(1, 13).copy()
    return np.column_stack([axis[-length:] for axis in axes])


class DeterministicNMPCController:
    """Adapter preserving the existing 13-state deterministic NMPC behavior."""

    def __init__(
        self,
        *,
        bounds: Mapping[str, float],
        obstacle_specs: Sequence[Mapping[str, Any]],
        margin: float,
        horizon_steps: int,
        timestep_s: float,
        max_iter: int = 60,
        cached=None,
    ):
        self._obstacle_specs = tuple(dict(item) for item in obstacle_specs)
        self._margin = float(margin)
        self._horizon_steps = int(horizon_steps)
        self._timestep_s = float(timestep_s)
        if self._horizon_steps < 1:
            raise ValueError("horizon_steps must be >= 1")
        if self._timestep_s <= 0.0:
            raise ValueError("timestep_s must be > 0")

        if cached is not None:
            self.model, self.mpc, self._obstacle_tvp_idx, self._goal_state = cached
            self.mpc.reset_history()
        else:
            self.model, self._obstacle_tvp_idx = build_model(
                self._timestep_s,
                self._obstacle_specs,
            )
            self.mpc = build_controller(
                self.model,
                self._obstacle_specs,
                self._obstacle_tvp_idx,
                bounds,
                self._margin,
                self._horizon_steps,
                self._timestep_s,
                max_iter=max_iter,
            )
            self._goal_state = {
                "pos": {"x": 0.0, "y": 0.0, "z": 1.0},
                "euler": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            }
            self.mpc.set_tvp_fun(
                make_mpc_tvp_fun(
                    self.mpc.get_tvp_template(),
                    self._goal_state,
                    self._obstacle_specs,
                    self._obstacle_tvp_idx,
                    self._horizon_steps,
                    self._timestep_s,
                )
            )
            self.mpc.setup()

    @property
    def horizon_steps(self) -> int:
        return self._horizon_steps

    def reset(self, belief: VehicleBelief) -> None:
        self.mpc.reset_history()
        self._goal_state.pop("obstacle_predictions", None)
        self.mpc.x0 = belief.mean_state_13.reshape(-1, 1)
        self.mpc.set_initial_guess()

    def solve(
        self,
        belief: VehicleBelief,
        obstacles: Sequence[ObstacleBelief],
        goal: ControlGoal,
        time_s: float,
    ) -> ControlSolution:
        if len(obstacles) != len(self._obstacle_specs):
            raise ValueError(
                f"controller expects {len(self._obstacle_specs)} obstacle beliefs, "
                f"got {len(obstacles)}"
            )
        prediction_steps = self._horizon_steps + 1
        if obstacles:
            obstacle_predictions = np.asarray(
                [
                    obstacle.mean_positions(prediction_steps, self._timestep_s)
                    for obstacle in obstacles
                ],
                dtype=float,
            )
        else:
            obstacle_predictions = np.empty((0, prediction_steps, 3), dtype=float)

        self._goal_state["pos"] = {
            axis: float(value) for axis, value in zip(("x", "y", "z"), goal.position)
        }
        # The TVP helper accepts a quaternion directly to avoid an unnecessary
        # quaternion -> Euler -> quaternion conversion at the interface boundary.
        self._goal_state["quaternion_wxyz"] = goal.quaternion_wxyz
        self._goal_state["obstacle_predictions"] = obstacle_predictions
        self._goal_state["prediction_time_s"] = float(time_s)

        command = np.asarray(
            self.mpc.make_step(belief.mean_state_13.reshape(-1, 1)),
            dtype=float,
        ).reshape(4)
        nominal_states = _extract_nominal_states(self.mpc, belief.mean_state_13)
        horizon_length = nominal_states.shape[0]
        predicted_covariances = np.zeros((horizon_length, 12, 12), dtype=float)

        margins = np.empty((horizon_length, len(obstacles)), dtype=float)
        for obstacle_index, obstacle in enumerate(obstacles):
            positions = obstacle_predictions[obstacle_index]
            if positions.shape[0] != horizon_length:
                source = np.linspace(0.0, 1.0, positions.shape[0])
                target = np.linspace(0.0, 1.0, horizon_length)
                positions = np.column_stack(
                    [np.interp(target, source, positions[:, axis]) for axis in range(3)]
                )
            safe_distance = (
                obstacle.shape.bounding_radius_m + self._margin + DRONE_RADIUS
            )
            margins[:, obstacle_index] = (
                np.linalg.norm(nominal_states[:, :3] - positions, axis=1) - safe_distance
            )

        return ControlSolution(
            command=command,
            nominal_states=nominal_states,
            predicted_covariances=predicted_covariances,
            chance_margins=margins,
            risk_allocations=np.zeros_like(margins),
            slacks=np.maximum(0.0, -margins),
            solver_status="SOLVED_DETERMINISTIC",
        )
