"""do-mpc adapter implementing the native belief-based controller contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from quadrotor_mpc.control.nmpc.chance_constraints import (
    ChanceConstraintOptions,
    build_spherical_chance_profile,
    evaluate_spherical_constraints,
)
from quadrotor_mpc.control.nmpc.core import (
    DRONE_RADIUS,
    INPUT_NAMES,
    STATE_NAMES,
    build_controller,
    build_model,
    make_mpc_tvp_fun,
)
from quadrotor_mpc.control.nmpc.covariance import (
    CovariancePropagationOptions,
    HorizonCovariancePropagator,
)
from quadrotor_mpc.core.contracts import (
    ControlGoal,
    ControlSolution,
    ObstacleBelief,
    VehicleBelief,
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


def _extract_nominal_controls(
    mpc,
    fallback_command: np.ndarray,
    horizon_length: int,
) -> np.ndarray:
    control_steps = max(int(horizon_length) - 1, 0)
    if control_steps == 0:
        return np.empty((0, 4), dtype=float)
    axes: list[np.ndarray] = []
    try:
        for name in INPUT_NAMES:
            values = np.asarray(mpc.data.prediction(("_u", name)), dtype=float)
            axes.append(values.reshape(-1))
    except (AttributeError, KeyError, TypeError, ValueError):
        return np.repeat(
            np.asarray(fallback_command, dtype=float).reshape(1, 4),
            control_steps,
            axis=0,
        )
    available = min((len(axis) for axis in axes), default=0)
    if available < 1:
        return np.repeat(
            np.asarray(fallback_command, dtype=float).reshape(1, 4),
            control_steps,
            axis=0,
        )
    extracted = np.column_stack([axis[-available:] for axis in axes])
    if available >= control_steps:
        return extracted[:control_steps].copy()
    padding = np.repeat(extracted[-1:, :], control_steps - available, axis=0)
    return np.vstack([extracted, padding])


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
        covariance_options: CovariancePropagationOptions | None = None,
        chance_options: ChanceConstraintOptions | None = None,
    ):
        self._obstacle_specs = tuple(dict(item) for item in obstacle_specs)
        self._margin = float(margin)
        self._horizon_steps = int(horizon_steps)
        self._timestep_s = float(timestep_s)
        if self._horizon_steps < 1:
            raise ValueError("horizon_steps must be >= 1")
        if self._timestep_s <= 0.0:
            raise ValueError("timestep_s must be > 0")
        self._covariance_options = (
            CovariancePropagationOptions() if covariance_options is None else covariance_options
        )
        self._chance_options = (
            ChanceConstraintOptions() if chance_options is None else chance_options
        )
        if self._chance_options.enabled and not self._covariance_options.enabled:
            raise ValueError(
                "spherical chance constraints require covariance propagation to be enabled"
            )
        self._covariance_propagator = HorizonCovariancePropagator(
            self._covariance_options,
            self._timestep_s,
        )
        self._last_nominal_states: np.ndarray | None = None
        self._last_nominal_controls: np.ndarray | None = None

        if cached is not None:
            if self._chance_options.enabled:
                raise ValueError(
                    "a deterministic cached controller cannot be reused for enabled "
                    "chance constraints"
                )
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
                penalty=(
                    self._chance_options.slack_penalty
                    if self._chance_options.enabled
                    else 1e4
                ),
                soft_obstacle_constraints=self._chance_options.soft_constraint,
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
                    self._margin,
                )
            )
            self.mpc.setup()

    @property
    def horizon_steps(self) -> int:
        return self._horizon_steps

    def reset(self, belief: VehicleBelief) -> None:
        self.mpc.reset_history()
        for key in (
            "obstacle_predictions",
            "obstacle_projected_sigmas",
            "obstacle_betas",
            "obstacle_risk_allocations",
            "obstacle_safe_distances",
        ):
            self._goal_state.pop(key, None)
        self._last_nominal_states = None
        self._last_nominal_controls = None
        self.mpc.x0 = belief.mean_state_13.reshape(-1, 1)
        self.mpc.set_initial_guess()

    def _seed_nominal(
        self,
        belief: VehicleBelief,
        goal: ControlGoal,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shift the prior solution or construct a deterministic first-tick seed."""
        steps = self._horizon_steps + 1
        if (
            self._last_nominal_states is not None
            and self._last_nominal_states.shape == (steps, 13)
            and self._last_nominal_controls is not None
            and self._last_nominal_controls.shape == (self._horizon_steps, 4)
        ):
            states = np.vstack(
                [self._last_nominal_states[1:], self._last_nominal_states[-1:]]
            )
            controls = np.vstack(
                [self._last_nominal_controls[1:], self._last_nominal_controls[-1:]]
            )
            states[0] = belief.mean_state_13
            return states, controls

        states = np.repeat(belief.mean_state_13.reshape(1, 13), steps, axis=0)
        interpolation = np.linspace(0.0, 1.0, steps)[:, None]
        states[:, :3] = (
            belief.mean_state_13[:3]
            + interpolation * (goal.position - belief.mean_state_13[:3])
        )
        controls = np.zeros((self._horizon_steps, 4), dtype=float)
        return states, controls

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

        seed_states, seed_controls = self._seed_nominal(belief, goal)
        if self._covariance_options.enabled:
            (
                tightening_vehicle_covariances,
                tightening_obstacle_covariances,
            ) = self._covariance_propagator.propagate(
                belief,
                obstacles,
                seed_states,
                seed_controls,
            )
        else:
            tightening_vehicle_covariances = np.zeros(
                (prediction_steps, 12, 12),
                dtype=float,
            )
            tightening_obstacle_covariances = np.zeros(
                (prediction_steps, len(obstacles), 6, 6),
                dtype=float,
            )
        base_safety_radii = np.asarray(
            [
                obstacle.shape.bounding_radius_m + self._margin + DRONE_RADIUS
                for obstacle in obstacles
            ],
            dtype=float,
        )
        chance_profile = build_spherical_chance_profile(
            vehicle_positions=seed_states[:, :3],
            obstacle_positions=obstacle_predictions,
            vehicle_covariances=tightening_vehicle_covariances,
            obstacle_covariances=tightening_obstacle_covariances,
            base_safety_radii_m=base_safety_radii,
            options=self._chance_options,
        )

        self._goal_state["pos"] = {
            axis: float(value) for axis, value in zip(("x", "y", "z"), goal.position)
        }
        # The TVP helper accepts a quaternion directly to avoid an unnecessary
        # quaternion -> Euler -> quaternion conversion at the interface boundary.
        self._goal_state["quaternion_wxyz"] = goal.quaternion_wxyz
        self._goal_state["obstacle_predictions"] = obstacle_predictions
        self._goal_state["obstacle_projected_sigmas"] = (
            chance_profile.projected_sigmas_m.T
        )
        self._goal_state["obstacle_betas"] = chance_profile.gaussian_quantiles.T
        self._goal_state["obstacle_risk_allocations"] = (
            chance_profile.risk_allocations.T
        )
        self._goal_state["obstacle_safe_distances"] = (
            chance_profile.safety_radii_m.T
        )
        self._goal_state["prediction_time_s"] = float(time_s)

        command = np.asarray(
            self.mpc.make_step(belief.mean_state_13.reshape(-1, 1)),
            dtype=float,
        ).reshape(4)
        solver_stats = getattr(self.mpc, "solver_stats", {})
        primary_solver_success = bool(solver_stats.get("success", True))
        primary_solver_status = str(
            solver_stats.get(
                "return_status",
                "SUCCESS" if primary_solver_success else "FAILED",
            )
        )
        solver_iterations = max(0, int(solver_stats.get("iter_count", 0)))
        iteration_history = solver_stats.get("iterations", {})

        def final_residual(name: str) -> float:
            values = iteration_history.get(name, ())
            if not values:
                return 0.0
            value = float(values[-1])
            return (
                value
                if np.isfinite(value) and value >= 0.0
                else np.finfo(float).max
            )

        def residual_status_for(name: str) -> str:
            # A missing residual must never masquerade as a perfect 0.0.
            values = iteration_history.get(name, ())
            if not values:
                return "UNAVAILABLE"
            value = float(values[-1])
            if not np.isfinite(value):
                return "INVALID"
            return "AVAILABLE"

        primal_residual = final_residual("inf_pr")
        dual_residual = final_residual("inf_du")
        residual_statuses = {
            residual_status_for("inf_pr"),
            residual_status_for("inf_du"),
        }
        if residual_statuses == {"UNAVAILABLE"}:
            residual_status = "UNAVAILABLE"
        elif "INVALID" in residual_statuses:
            residual_status = "INVALID"
        else:
            residual_status = "AVAILABLE"
        nominal_states = _extract_nominal_states(self.mpc, belief.mean_state_13)
        horizon_length = nominal_states.shape[0]
        nominal_controls = _extract_nominal_controls(
            self.mpc,
            command,
            horizon_length,
        )
        self._last_nominal_states = nominal_states.copy()
        self._last_nominal_controls = nominal_controls.copy()
        if (
            self._covariance_options.enabled
            and horizon_length != tightening_vehicle_covariances.shape[0]
        ):
            (
                predicted_covariances,
                predicted_obstacle_covariances,
            ) = self._covariance_propagator.propagate(
                belief,
                obstacles,
                nominal_states,
                nominal_controls,
            )
        else:
            predicted_covariances = tightening_vehicle_covariances[:horizon_length].copy()
            predicted_obstacle_covariances = (
                tightening_obstacle_covariances[:horizon_length].copy()
            )

        used_obstacle_predictions = obstacle_predictions
        used_sigmas = chance_profile.projected_sigmas_m
        used_risks = chance_profile.risk_allocations
        used_safety_radii = chance_profile.safety_radii_m
        used_risk_metadata = chance_profile
        if horizon_length != prediction_steps:
            target = np.linspace(0.0, 1.0, horizon_length)
            source = np.linspace(0.0, 1.0, prediction_steps)
            used_obstacle_predictions = np.asarray(
                [
                    np.column_stack(
                        [
                            np.interp(target, source, positions[:, axis])
                            for axis in range(3)
                        ]
                    )
                    for positions in obstacle_predictions
                ],
                dtype=float,
            )
            used_sigmas = np.column_stack(
                [
                    np.interp(target, source, chance_profile.projected_sigmas_m[:, index])
                    for index in range(len(obstacles))
                ]
            )
            # Reallocate for the actual output horizon.  Interpolating a joint
            # allocation would change its sum whenever the number of returned
            # nodes differs from the configured grid.
            used_risk_metadata = build_spherical_chance_profile(
                vehicle_positions=nominal_states[:, :3],
                obstacle_positions=used_obstacle_predictions,
                vehicle_covariances=predicted_covariances,
                obstacle_covariances=predicted_obstacle_covariances,
                base_safety_radii_m=base_safety_radii,
                options=self._chance_options,
            )
            used_sigmas = used_risk_metadata.projected_sigmas_m
            used_risks = used_risk_metadata.risk_allocations
            used_safety_radii = used_risk_metadata.safety_radii_m
        margins, slacks = evaluate_spherical_constraints(
            vehicle_positions=nominal_states[:, :3],
            obstacle_positions=used_obstacle_predictions,
            safety_radii_m=used_safety_radii,
            distance_smoothing_m2=self._chance_options.distance_smoothing_m2,
        )
        if self._chance_options.enabled:
            solver_status = (
                "SOLVED_SAFE"
                if not slacks.size
                or float(np.max(slacks)) <= self._chance_options.slack_tolerance_m
                else "SOLVED_WITH_SLACK"
            )
        else:
            solver_status = "SOLVED_DETERMINISTIC"

        return ControlSolution(
            command=command,
            nominal_states=nominal_states,
            predicted_covariances=predicted_covariances,
            chance_margins=margins,
            risk_allocations=used_risks,
            slacks=slacks,
            solver_status=solver_status,
            predicted_obstacle_covariances=predicted_obstacle_covariances,
            projected_uncertainties=used_sigmas,
            tightened_safety_radii=used_safety_radii,
            risk_semantics=used_risk_metadata.risk_semantics,
            risk_allocation_method=used_risk_metadata.risk_allocation_method,
            risk_budget_total=used_risk_metadata.configured_total_epsilon,
            risk_budget_allocated=used_risk_metadata.allocated_epsilon,
            risk_budget_remaining=used_risk_metadata.remaining_epsilon,
            risk_constraint_count=used_risk_metadata.active_constraint_count,
            risk_budget_status=used_risk_metadata.budget_status,
            primary_solver_status=primary_solver_status,
            primary_solver_success=primary_solver_success,
            primary_solver_iterations=solver_iterations,
            primary_solver_primal_residual=primal_residual,
            primary_solver_dual_residual=dual_residual,
            residual_status=residual_status,
        )
