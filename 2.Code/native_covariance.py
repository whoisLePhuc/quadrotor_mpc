"""Prediction-horizon covariance propagation for the native 13-state controller.

Vehicle uncertainty lives in the 12D local error state
``[delta_p, delta_v, delta_theta, delta_omega]``.  The nominal quaternion state
is never expanded into a redundant 13D covariance.  Obstacle uncertainty uses
the 6D constant-velocity tracker state ``[position, velocity]``.

This module only predicts uncertainty.  It does not tighten obstacle
constraints or allocate risk; those operations belong to the chance-constraint
controller stage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from controller_interface import (
    CONTROL_SIZE,
    VEHICLE_ERROR_STATE_SIZE,
    VEHICLE_STATE_SIZE,
    ObstacleBelief,
    VehicleBelief,
)
from native_estimation import (
    GRAVITY_MPS2,
    inject_error,
    propagate_nominal,
    quaternion_to_rotation_matrix,
    state_error,
)
from vehicle import DEFAULT_QUADROTOR

DEFAULT_FEEDBACK_STATE_WEIGHTS = (
    8.0,
    8.0,
    10.0,
    2.0,
    2.0,
    3.0,
    4.0,
    4.0,
    3.0,
    1.0,
    1.0,
    1.0,
)
DEFAULT_FEEDBACK_CONTROL_WEIGHTS = (1.0, 1.0, 1.0, 1.0)


def _positive(value: Any, label: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and > 0")
    return number


def _nonnegative(value: Any, label: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and >= 0")
    return number


def _positive_vector(
    value: Any,
    size: int,
    label: str,
    default: Sequence[float],
) -> tuple[float, ...]:
    array = np.asarray(default if value is None else value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{label} must contain {size} finite values > 0")
    return tuple(float(item) for item in array)


@dataclass(frozen=True, slots=True)
class CovariancePropagationOptions:
    """Validated horizon-propagation configuration."""

    enabled: bool = False
    mode: str = "open_loop"
    linearization: str = "analytic_first_order"
    acceleration_process_std_mps2: float = 0.35
    angular_acceleration_process_std_radps2: float = 0.60
    obstacle_acceleration_process_std_mps2: float = 0.50
    finite_difference_epsilon: float = 1e-6
    minimum_covariance_eigenvalue: float = 1e-12
    feedback_state_weights: tuple[float, ...] = DEFAULT_FEEDBACK_STATE_WEIGHTS
    feedback_control_weights: tuple[float, ...] = DEFAULT_FEEDBACK_CONTROL_WEIGHTS

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CovariancePropagationOptions:
        mode = str(raw.get("mode", "open_loop")).lower()
        if mode not in {"open_loop", "feedback_lqr"}:
            raise ValueError(
                "controller.covariance_propagation.mode must be 'open_loop' or 'feedback_lqr'"
            )
        linearization = str(raw.get("linearization", "analytic_first_order")).lower()
        if linearization not in {
            "analytic_first_order",
            "finite_difference_rk4",
        }:
            raise ValueError(
                "controller.covariance_propagation.linearization must be "
                "'analytic_first_order' or 'finite_difference_rk4'"
            )
        return cls(
            enabled=bool(raw.get("enabled", False)),
            mode=mode,
            linearization=linearization,
            acceleration_process_std_mps2=_nonnegative(
                raw.get("acceleration_process_std_mps2", 0.35),
                "controller.covariance_propagation.acceleration_process_std_mps2",
            ),
            angular_acceleration_process_std_radps2=_nonnegative(
                raw.get("angular_acceleration_process_std_radps2", 0.60),
                "controller.covariance_propagation.angular_acceleration_process_std_radps2",
            ),
            obstacle_acceleration_process_std_mps2=_nonnegative(
                raw.get("obstacle_acceleration_process_std_mps2", 0.50),
                "controller.covariance_propagation.obstacle_acceleration_process_std_mps2",
            ),
            finite_difference_epsilon=_positive(
                raw.get("finite_difference_epsilon", 1e-6),
                "controller.covariance_propagation.finite_difference_epsilon",
            ),
            minimum_covariance_eigenvalue=_positive(
                raw.get("minimum_covariance_eigenvalue", 1e-12),
                "controller.covariance_propagation.minimum_covariance_eigenvalue",
            ),
            feedback_state_weights=_positive_vector(
                raw.get("feedback_state_weights"),
                VEHICLE_ERROR_STATE_SIZE,
                "controller.covariance_propagation.feedback_state_weights",
                DEFAULT_FEEDBACK_STATE_WEIGHTS,
            ),
            feedback_control_weights=_positive_vector(
                raw.get("feedback_control_weights"),
                CONTROL_SIZE,
                "controller.covariance_propagation.feedback_control_weights",
                DEFAULT_FEEDBACK_CONTROL_WEIGHTS,
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "linearization": self.linearization,
            "acceleration_process_std_mps2": self.acceleration_process_std_mps2,
            "angular_acceleration_process_std_radps2": (
                self.angular_acceleration_process_std_radps2
            ),
            "obstacle_acceleration_process_std_mps2": (self.obstacle_acceleration_process_std_mps2),
            "finite_difference_epsilon": self.finite_difference_epsilon,
            "minimum_covariance_eigenvalue": self.minimum_covariance_eigenvalue,
            "feedback_state_weights": list(self.feedback_state_weights),
            "feedback_control_weights": list(self.feedback_control_weights),
        }


def project_covariance(covariance: np.ndarray, floor: float) -> np.ndarray:
    """Return a symmetric PSD covariance with a small numerical eigenvalue floor."""
    symmetric = 0.5 * (np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    projected = (eigenvectors * np.maximum(eigenvalues, float(floor))) @ eigenvectors.T
    return 0.5 * (projected + projected.T)


def vehicle_process_covariance(
    dt: float,
    acceleration_std_mps2: float,
    angular_acceleration_std_radps2: float,
) -> np.ndarray:
    """Discretize independent white linear/angular acceleration disturbances."""
    timestep = _positive(dt, "dt")
    covariance = np.zeros(
        (VEHICLE_ERROR_STATE_SIZE, VEHICLE_ERROR_STATE_SIZE),
        dtype=float,
    )
    for offset, variance in (
        (0, float(acceleration_std_mps2) ** 2),
        (6, float(angular_acceleration_std_radps2) ** 2),
    ):
        position = slice(offset, offset + 3)
        velocity = slice(offset + 3, offset + 6)
        covariance[position, position] = np.eye(3) * variance * timestep**4 / 4.0
        cross = np.eye(3) * variance * timestep**3 / 2.0
        covariance[position, velocity] = cross
        covariance[velocity, position] = cross
        covariance[velocity, velocity] = np.eye(3) * variance * timestep**2
    return covariance


def obstacle_transition(dt: float) -> np.ndarray:
    transition = np.eye(6)
    transition[:3, 3:] = np.eye(3) * _positive(dt, "dt")
    return transition


def obstacle_process_covariance(dt: float, acceleration_std_mps2: float) -> np.ndarray:
    timestep = _positive(dt, "dt")
    variance = float(acceleration_std_mps2) ** 2
    return variance * np.block(
        [
            [np.eye(3) * timestep**4 / 4.0, np.eye(3) * timestep**3 / 2.0],
            [np.eye(3) * timestep**3 / 2.0, np.eye(3) * timestep**2],
        ]
    )


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=float,
    )


def analytic_error_state_jacobians(
    state_13: np.ndarray,
    control_4: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """First-order discrete Jacobians of the 12D right-error dynamics."""
    state = np.asarray(state_13, dtype=float).reshape(VEHICLE_STATE_SIZE)
    control = np.asarray(control_4, dtype=float).reshape(CONTROL_SIZE)
    timestep = _positive(dt, "dt")
    rotation = quaternion_to_rotation_matrix(state[6:10])
    omega = state[10:13]
    inertia_diagonal = np.asarray(
        DEFAULT_QUADROTOR.inertia_kg_m2,
        dtype=float,
    )
    inertia = np.diag(inertia_diagonal)
    inverse_inertia = np.diag(1.0 / inertia_diagonal)
    damping = np.diag(
        [
            DEFAULT_QUADROTOR.angular_damping_nms,
            DEFAULT_QUADROTOR.angular_damping_nms,
            DEFAULT_QUADROTOR.yaw_damping_nms,
        ]
    )
    thrust_acceleration = GRAVITY_MPS2 + control[0] / DEFAULT_QUADROTOR.mass_kg

    continuous_state = np.zeros(
        (VEHICLE_ERROR_STATE_SIZE, VEHICLE_ERROR_STATE_SIZE),
        dtype=float,
    )
    continuous_state[:3, 3:6] = np.eye(3)
    continuous_state[3:6, 3:6] = -DEFAULT_QUADROTOR.linear_damping_per_s * np.eye(3)
    continuous_state[3:6, 6:9] = -thrust_acceleration * rotation @ _skew(np.array([0.0, 0.0, 1.0]))
    continuous_state[6:9, 6:9] = -_skew(omega)
    continuous_state[6:9, 9:12] = np.eye(3)
    continuous_state[9:12, 9:12] = inverse_inertia @ (
        _skew(inertia @ omega) - _skew(omega) @ inertia - damping
    )

    continuous_control = np.zeros(
        (VEHICLE_ERROR_STATE_SIZE, CONTROL_SIZE),
        dtype=float,
    )
    continuous_control[3:6, 0] = (rotation @ np.array([0.0, 0.0, 1.0])) / DEFAULT_QUADROTOR.mass_kg
    continuous_control[9:12, 1:4] = inverse_inertia

    return (
        np.eye(VEHICLE_ERROR_STATE_SIZE) + timestep * continuous_state,
        timestep * continuous_control,
    )


def finite_difference_error_state_jacobians(
    state_13: np.ndarray,
    control_4: np.ndarray,
    dt: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerically linearize one RK4 step in the quaternion local-error chart."""
    state = np.asarray(state_13, dtype=float).reshape(VEHICLE_STATE_SIZE)
    control = np.asarray(control_4, dtype=float).reshape(CONTROL_SIZE)
    perturbation_size = _positive(epsilon, "epsilon")
    nominal_next = propagate_nominal(state, control, dt)

    state_transition = np.empty(
        (VEHICLE_ERROR_STATE_SIZE, VEHICLE_ERROR_STATE_SIZE),
        dtype=float,
    )
    for column in range(VEHICLE_ERROR_STATE_SIZE):
        perturbation = np.zeros(VEHICLE_ERROR_STATE_SIZE)
        perturbation[column] = perturbation_size
        perturbed_next = propagate_nominal(
            inject_error(state, perturbation),
            control,
            dt,
        )
        state_transition[:, column] = state_error(nominal_next, perturbed_next) / perturbation_size

    control_transition = np.empty(
        (VEHICLE_ERROR_STATE_SIZE, CONTROL_SIZE),
        dtype=float,
    )
    for column in range(CONTROL_SIZE):
        perturbed_control = control.copy()
        perturbed_control[column] += perturbation_size
        perturbed_next = propagate_nominal(state, perturbed_control, dt)
        control_transition[:, column] = (
            state_error(nominal_next, perturbed_next) / perturbation_size
        )
    return state_transition, control_transition


class HorizonCovariancePropagator:
    """Propagate vehicle and obstacle beliefs over one MPC horizon."""

    def __init__(self, options: CovariancePropagationOptions, timestep_s: float):
        self.options = options
        self.timestep_s = _positive(timestep_s, "timestep_s")

    def _linearizations(
        self,
        nominal_states: np.ndarray,
        nominal_controls: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        transitions: list[np.ndarray] = []
        input_matrices: list[np.ndarray] = []
        for state, control in zip(nominal_states[:-1], nominal_controls):
            if self.options.linearization == "analytic_first_order":
                transition, input_matrix = analytic_error_state_jacobians(
                    state,
                    control,
                    self.timestep_s,
                )
            else:
                transition, input_matrix = finite_difference_error_state_jacobians(
                    state,
                    control,
                    self.timestep_s,
                    self.options.finite_difference_epsilon,
                )
            transitions.append(transition)
            input_matrices.append(input_matrix)
        return np.asarray(transitions), np.asarray(input_matrices)

    def _feedback_gains(
        self,
        transitions: np.ndarray,
        input_matrices: np.ndarray,
    ) -> np.ndarray:
        """Compute finite-horizon local gains with ``u = u_nominal + K delta_x``."""
        state_cost = np.diag(self.options.feedback_state_weights)
        control_cost = np.diag(self.options.feedback_control_weights)
        value_matrix = state_cost.copy()
        gains = np.zeros(
            (len(transitions), CONTROL_SIZE, VEHICLE_ERROR_STATE_SIZE),
            dtype=float,
        )
        for index in range(len(transitions) - 1, -1, -1):
            transition = transitions[index]
            input_matrix = input_matrices[index]
            hessian = control_cost + input_matrix.T @ value_matrix @ input_matrix
            gain = -np.linalg.solve(
                hessian,
                input_matrix.T @ value_matrix @ transition,
            )
            closed_loop = transition + input_matrix @ gain
            value_matrix = (
                state_cost
                + gain.T @ control_cost @ gain
                + closed_loop.T @ value_matrix @ closed_loop
            )
            value_matrix = 0.5 * (value_matrix + value_matrix.T)
            gains[index] = gain
        return gains

    def propagate_vehicle(
        self,
        belief: VehicleBelief,
        nominal_states: np.ndarray,
        nominal_controls: np.ndarray,
    ) -> np.ndarray:
        states = np.asarray(nominal_states, dtype=float)
        controls = np.asarray(nominal_controls, dtype=float)
        if states.ndim != 2 or states.shape[1] != VEHICLE_STATE_SIZE:
            raise ValueError("nominal_states must have shape (N+1, 13)")
        expected_controls = max(states.shape[0] - 1, 0)
        if controls.shape != (expected_controls, CONTROL_SIZE):
            raise ValueError(f"nominal_controls must have shape ({expected_controls}, 4)")

        horizon = np.empty(
            (states.shape[0], VEHICLE_ERROR_STATE_SIZE, VEHICLE_ERROR_STATE_SIZE),
            dtype=float,
        )
        horizon[0] = project_covariance(
            belief.error_covariance_12,
            self.options.minimum_covariance_eigenvalue,
        )
        if states.shape[0] == 1:
            return horizon

        transitions, input_matrices = self._linearizations(states, controls)
        if self.options.mode == "feedback_lqr":
            gains = self._feedback_gains(transitions, input_matrices)
            transitions = transitions + np.einsum(
                "nij,njk->nik",
                input_matrices,
                gains,
            )

        process_covariance = vehicle_process_covariance(
            self.timestep_s,
            self.options.acceleration_process_std_mps2,
            self.options.angular_acceleration_process_std_radps2,
        )
        for index, transition in enumerate(transitions):
            horizon[index + 1] = project_covariance(
                transition @ horizon[index] @ transition.T + process_covariance,
                self.options.minimum_covariance_eigenvalue,
            )
        return horizon

    def propagate_obstacles(
        self,
        obstacles: Sequence[ObstacleBelief],
        horizon_length: int,
    ) -> np.ndarray:
        steps = int(horizon_length)
        if steps < 1:
            raise ValueError("horizon_length must be >= 1")
        result = np.empty((steps, len(obstacles), 6, 6), dtype=float)
        if not obstacles:
            return result
        transition = obstacle_transition(self.timestep_s)
        process_covariance = obstacle_process_covariance(
            self.timestep_s,
            self.options.obstacle_acceleration_process_std_mps2,
        )
        for obstacle_index, obstacle in enumerate(obstacles):
            result[0, obstacle_index] = project_covariance(
                obstacle.covariance_6,
                self.options.minimum_covariance_eigenvalue,
            )
            for step in range(steps - 1):
                result[step + 1, obstacle_index] = project_covariance(
                    transition @ result[step, obstacle_index] @ transition.T + process_covariance,
                    self.options.minimum_covariance_eigenvalue,
                )
        return result

    def propagate(
        self,
        belief: VehicleBelief,
        obstacles: Sequence[ObstacleBelief],
        nominal_states: np.ndarray,
        nominal_controls: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        vehicle = self.propagate_vehicle(
            belief,
            nominal_states,
            nominal_controls,
        )
        obstacle = self.propagate_obstacles(obstacles, len(nominal_states))
        return vehicle, obstacle
