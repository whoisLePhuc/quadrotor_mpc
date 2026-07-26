"""Native sensor simulation and belief estimation for the 13-state plant.

Ground truth enters this module only through :class:`NativeSensorSimulator`.
Controllers receive immutable ``VehicleBelief`` and ``ObstacleBelief`` objects
produced by the estimators; they never receive a MuJoCo plant or truth state.

Vehicle attitude uses a right-multiplicative local error:

    q_true = q_nominal (*) Exp(delta_theta)

so the nominal state has 13 elements while its covariance has 12.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quadrotor_mpc.core.contracts import ObstacleBelief, SphericalObstacle, VehicleBelief
from quadrotor_mpc.core.vehicle import DEFAULT_QUADROTOR

GRAVITY_MPS2 = 9.81
_ERROR_SIZE = 12


def _probability(value: Any, label: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return number


def _nonnegative(value: Any, label: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and >= 0")
    return number


def _positive(value: Any, label: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and > 0")
    return number


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


@dataclass(frozen=True, slots=True)
class SensorOptions:
    """Noise, dropout and optional unmodelled bias random walks."""

    position_std_m: float = 0.03
    velocity_std_mps: float = 0.06
    attitude_std_rad: float = 0.015
    angular_rate_std_radps: float = 0.02
    obstacle_position_std_m: float = 0.05
    vehicle_dropout_probability: float = 0.0
    obstacle_dropout_probability: float = 0.0
    position_bias_rw_std_m_sqrt_s: float = 0.0
    velocity_bias_rw_std_mps_sqrt_s: float = 0.0
    attitude_bias_rw_std_rad_sqrt_s: float = 0.0
    angular_rate_bias_rw_std_radps_sqrt_s: float = 0.0
    obstacle_bias_rw_std_m_sqrt_s: float = 0.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SensorOptions:
        values = {
            "position_std_m": _nonnegative(
                raw.get("position_std_m", 0.03), "estimation.sensor.position_std_m"
            ),
            "velocity_std_mps": _nonnegative(
                raw.get("velocity_std_mps", 0.06), "estimation.sensor.velocity_std_mps"
            ),
            "attitude_std_rad": _nonnegative(
                raw.get("attitude_std_rad", 0.015), "estimation.sensor.attitude_std_rad"
            ),
            "angular_rate_std_radps": _nonnegative(
                raw.get("angular_rate_std_radps", 0.02),
                "estimation.sensor.angular_rate_std_radps",
            ),
            "obstacle_position_std_m": _nonnegative(
                raw.get("obstacle_position_std_m", 0.05),
                "estimation.sensor.obstacle_position_std_m",
            ),
            "vehicle_dropout_probability": _probability(
                raw.get("vehicle_dropout_probability", 0.0),
                "estimation.sensor.vehicle_dropout_probability",
            ),
            "obstacle_dropout_probability": _probability(
                raw.get("obstacle_dropout_probability", 0.0),
                "estimation.sensor.obstacle_dropout_probability",
            ),
        }
        for name in (
            "position_bias_rw_std_m_sqrt_s",
            "velocity_bias_rw_std_mps_sqrt_s",
            "attitude_bias_rw_std_rad_sqrt_s",
            "angular_rate_bias_rw_std_radps_sqrt_s",
            "obstacle_bias_rw_std_m_sqrt_s",
        ):
            values[name] = _nonnegative(raw.get(name, 0.0), f"estimation.sensor.{name}")
        return cls(**values)


@dataclass(frozen=True, slots=True)
class VehicleFilterOptions:
    acceleration_process_std_mps2: float = 0.35
    angular_acceleration_process_std_radps2: float = 0.60
    initial_position_std_m: float = 0.08
    initial_velocity_std_mps: float = 0.12
    initial_attitude_std_rad: float = 0.04
    initial_angular_rate_std_radps: float = 0.08

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> VehicleFilterOptions:
        return cls(
            **{
                name: _nonnegative(raw.get(name, default), f"estimation.vehicle_filter.{name}")
                for name, default in (
                    ("acceleration_process_std_mps2", 0.35),
                    ("angular_acceleration_process_std_radps2", 0.60),
                    ("initial_position_std_m", 0.08),
                    ("initial_velocity_std_mps", 0.12),
                    ("initial_attitude_std_rad", 0.04),
                    ("initial_angular_rate_std_radps", 0.08),
                )
            }
        )


@dataclass(frozen=True, slots=True)
class ObstacleFilterOptions:
    acceleration_process_std_mps2: float = 0.50
    initial_position_std_m: float = 0.10
    initial_velocity_std_mps: float = 0.75

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ObstacleFilterOptions:
        return cls(
            acceleration_process_std_mps2=_nonnegative(
                raw.get("acceleration_process_std_mps2", 0.50),
                "estimation.obstacle_filter.acceleration_process_std_mps2",
            ),
            initial_position_std_m=_nonnegative(
                raw.get("initial_position_std_m", 0.10),
                "estimation.obstacle_filter.initial_position_std_m",
            ),
            initial_velocity_std_mps=_nonnegative(
                raw.get("initial_velocity_std_mps", 0.75),
                "estimation.obstacle_filter.initial_velocity_std_mps",
            ),
        )


@dataclass(frozen=True, slots=True)
class EstimationOptions:
    enabled: bool = False
    estimator_type: str = "error_state_ekf"
    seed: int = 7
    minimum_covariance_eigenvalue: float = 1e-12
    sensor: SensorOptions = field(default_factory=SensorOptions)
    vehicle_filter: VehicleFilterOptions = field(default_factory=VehicleFilterOptions)
    obstacle_filter: ObstacleFilterOptions = field(default_factory=ObstacleFilterOptions)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> EstimationOptions:
        estimator_type = str(raw.get("type", "error_state_ekf")).lower()
        if estimator_type != "error_state_ekf":
            raise ValueError("estimation.type must be 'error_state_ekf'")
        return cls(
            enabled=bool(raw.get("enabled", False)),
            estimator_type=estimator_type,
            seed=int(raw.get("seed", 7)),
            minimum_covariance_eigenvalue=_positive(
                raw.get("minimum_covariance_eigenvalue", 1e-12),
                "estimation.minimum_covariance_eigenvalue",
            ),
            sensor=SensorOptions.from_mapping(_mapping(raw.get("sensor", {}), "estimation.sensor")),
            vehicle_filter=VehicleFilterOptions.from_mapping(
                _mapping(raw.get("vehicle_filter", {}), "estimation.vehicle_filter")
            ),
            obstacle_filter=ObstacleFilterOptions.from_mapping(
                _mapping(raw.get("obstacle_filter", {}), "estimation.obstacle_filter")
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        def fields(instance: Any) -> dict[str, Any]:
            return {name: getattr(instance, name) for name in instance.__dataclass_fields__}

        return {
            "enabled": self.enabled,
            "type": self.estimator_type,
            "seed": self.seed,
            "minimum_covariance_eigenvalue": self.minimum_covariance_eigenvalue,
            "sensor": fields(self.sensor),
            "vehicle_filter": fields(self.vehicle_filter),
            "obstacle_filter": fields(self.obstacle_filter),
        }


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for ``[w, x, y, z]`` quaternions."""
    w1, x1, y1, z1 = np.asarray(left, dtype=float)
    w2, x2, y2, z2 = np.asarray(right, dtype=float)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=float).copy()
    result[1:] *= -1.0
    return result


def rotation_vector_to_quaternion(rotation_vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotation_vector, dtype=float).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-10:
        quaternion = np.concatenate([[1.0 - angle * angle / 8.0], 0.5 * vector])
    else:
        half_angle = 0.5 * angle
        quaternion = np.concatenate([[np.cos(half_angle)], np.sin(half_angle) * vector / angle])
    return quaternion / np.linalg.norm(quaternion)


def quaternion_to_rotation_vector(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=float).reshape(4)
    quaternion = quaternion / np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    vector_norm = float(np.linalg.norm(quaternion[1:]))
    if vector_norm < 1e-10:
        return 2.0 * quaternion[1:]
    angle = 2.0 * np.arctan2(vector_norm, np.clip(quaternion[0], -1.0, 1.0))
    return angle * quaternion[1:] / vector_norm


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def inject_error(state_13: np.ndarray, error_12: np.ndarray) -> np.ndarray:
    state = np.asarray(state_13, dtype=float).reshape(13).copy()
    error = np.asarray(error_12, dtype=float).reshape(12)
    state[:3] += error[:3]
    state[3:6] += error[3:6]
    state[6:10] = quaternion_multiply(state[6:10], rotation_vector_to_quaternion(error[6:9]))
    state[6:10] /= np.linalg.norm(state[6:10])
    state[10:13] += error[9:12]
    return state


def state_error(reference_13: np.ndarray, value_13: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference_13, dtype=float).reshape(13)
    value = np.asarray(value_13, dtype=float).reshape(13)
    attitude_error = quaternion_multiply(quaternion_conjugate(reference[6:10]), value[6:10])
    return np.concatenate(
        [
            value[:3] - reference[:3],
            value[3:6] - reference[3:6],
            quaternion_to_rotation_vector(attitude_error),
            value[10:13] - reference[10:13],
        ]
    )


def _state_derivative(state_13: np.ndarray, control_4: np.ndarray) -> np.ndarray:
    state = np.asarray(state_13, dtype=float).reshape(13)
    control = np.asarray(control_4, dtype=float).reshape(4)
    position, velocity = state[:3], state[3:6]
    quaternion, omega = state[6:10], state[10:13]
    del position
    thrust_acceleration = GRAVITY_MPS2 + control[0] / DEFAULT_QUADROTOR.mass_kg
    acceleration = (
        quaternion_to_rotation_matrix(quaternion) @ np.array([0.0, 0.0, thrust_acceleration])
        - np.array([0.0, 0.0, GRAVITY_MPS2])
        - DEFAULT_QUADROTOR.linear_damping_per_s * velocity
    )
    quaternion_rate = 0.5 * quaternion_multiply(quaternion, np.concatenate([[0.0], omega]))
    inertia = np.asarray(DEFAULT_QUADROTOR.inertia_kg_m2, dtype=float)
    damping = np.array(
        [
            DEFAULT_QUADROTOR.angular_damping_nms,
            DEFAULT_QUADROTOR.angular_damping_nms,
            DEFAULT_QUADROTOR.yaw_damping_nms,
        ]
    )
    angular_acceleration = (
        control[1:] - np.cross(omega, inertia * omega) - damping * omega
    ) / inertia
    return np.concatenate([velocity, acceleration, quaternion_rate, angular_acceleration])


def propagate_nominal(state_13: np.ndarray, control_4: np.ndarray, dt: float) -> np.ndarray:
    """RK4 propagation matching the controller's nominal rigid-body model."""
    state = np.asarray(state_13, dtype=float).reshape(13)
    control = np.asarray(control_4, dtype=float).reshape(4)
    timestep = _positive(dt, "dt")
    k1 = _state_derivative(state, control)
    k2 = _state_derivative(state + 0.5 * timestep * k1, control)
    k3 = _state_derivative(state + 0.5 * timestep * k2, control)
    k4 = _state_derivative(state + timestep * k3, control)
    result = state + timestep * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    result[6:10] /= np.linalg.norm(result[6:10])
    return result


def _project_covariance(covariance: np.ndarray, floor: float) -> np.ndarray:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


@dataclass(frozen=True, slots=True)
class VehicleMeasurement:
    state_13: np.ndarray
    covariance_12: np.ndarray


@dataclass(frozen=True, slots=True)
class ObstacleMeasurement:
    obstacle_index: int
    position_world: np.ndarray
    covariance_3: np.ndarray


@dataclass(frozen=True, slots=True)
class SensorFrame:
    vehicle: VehicleMeasurement | None
    obstacles: tuple[ObstacleMeasurement, ...]
    obstacle_available: np.ndarray


class NativeSensorSimulator:
    """Seeded Gaussian full-state sensor and identified obstacle detections."""

    def __init__(self, options: SensorOptions, seed: int):
        self.options = options
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._vehicle_bias = np.zeros(_ERROR_SIZE)
        self._obstacle_bias = np.empty((0, 3))

    def reset(self, obstacle_count: int) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._vehicle_bias = np.zeros(_ERROR_SIZE)
        self._obstacle_bias = np.zeros((int(obstacle_count), 3))

    def _advance_biases(self, dt: float) -> None:
        scale = np.sqrt(max(float(dt), 0.0))
        standard_deviations = np.concatenate(
            [
                np.full(3, self.options.position_bias_rw_std_m_sqrt_s),
                np.full(3, self.options.velocity_bias_rw_std_mps_sqrt_s),
                np.full(3, self.options.attitude_bias_rw_std_rad_sqrt_s),
                np.full(3, self.options.angular_rate_bias_rw_std_radps_sqrt_s),
            ]
        )
        self._vehicle_bias += self._rng.normal(0.0, standard_deviations * scale)
        self._obstacle_bias += self._rng.normal(
            0.0,
            self.options.obstacle_bias_rw_std_m_sqrt_s * scale,
            self._obstacle_bias.shape,
        )

    def sample(
        self,
        vehicle_truth_13: np.ndarray,
        obstacle_truth_positions: np.ndarray,
        *,
        dt: float,
        force_measurements: bool = False,
    ) -> SensorFrame:
        truth = np.asarray(vehicle_truth_13, dtype=float).reshape(13)
        obstacle_positions = np.asarray(obstacle_truth_positions, dtype=float).reshape(-1, 3)
        if obstacle_positions.shape[0] != self._obstacle_bias.shape[0]:
            raise ValueError("sensor obstacle count changed without reset")
        self._advance_biases(dt)

        vehicle_available = force_measurements or (
            self._rng.random() >= self.options.vehicle_dropout_probability
        )
        vehicle_measurement: VehicleMeasurement | None = None
        measurement_std = np.concatenate(
            [
                np.full(3, self.options.position_std_m),
                np.full(3, self.options.velocity_std_mps),
                np.full(3, self.options.attitude_std_rad),
                np.full(3, self.options.angular_rate_std_radps),
            ]
        )
        if vehicle_available:
            measurement_error = self._vehicle_bias + self._rng.normal(0.0, measurement_std)
            vehicle_measurement = VehicleMeasurement(
                state_13=inject_error(truth, measurement_error),
                covariance_12=np.diag(measurement_std**2),
            )

        available = np.zeros(obstacle_positions.shape[0], dtype=bool)
        obstacle_measurements: list[ObstacleMeasurement] = []
        obstacle_covariance = np.eye(3, dtype=float) * self.options.obstacle_position_std_m**2
        for index, position in enumerate(obstacle_positions):
            detected = force_measurements or (
                self._rng.random() >= self.options.obstacle_dropout_probability
            )
            if not detected:
                continue
            available[index] = True
            obstacle_measurements.append(
                ObstacleMeasurement(
                    obstacle_index=index,
                    position_world=position
                    + self._obstacle_bias[index]
                    + self._rng.normal(0.0, self.options.obstacle_position_std_m, 3),
                    covariance_3=obstacle_covariance.copy(),
                )
            )
        return SensorFrame(
            vehicle=vehicle_measurement,
            obstacles=tuple(obstacle_measurements),
            obstacle_available=available,
        )


class ErrorStateEkf:
    """12D local-error EKF around a normalized 13D quaternion nominal state."""

    def __init__(
        self,
        options: VehicleFilterOptions,
        *,
        minimum_covariance_eigenvalue: float = 1e-12,
    ):
        self.options = options
        self.minimum_covariance_eigenvalue = float(minimum_covariance_eigenvalue)
        self.state_13: np.ndarray | None = None
        self.covariance_12: np.ndarray | None = None

    def reset(self, measurement: VehicleMeasurement) -> None:
        self.state_13 = np.asarray(measurement.state_13, dtype=float).reshape(13).copy()
        initial_std = np.concatenate(
            [
                np.full(3, self.options.initial_position_std_m),
                np.full(3, self.options.initial_velocity_std_mps),
                np.full(3, self.options.initial_attitude_std_rad),
                np.full(3, self.options.initial_angular_rate_std_radps),
            ]
        )
        self.covariance_12 = np.diag(initial_std**2)

    def predict(self, control_4: np.ndarray, dt: float) -> None:
        if self.state_13 is None or self.covariance_12 is None:
            raise RuntimeError("vehicle estimator must be reset before predict")
        nominal_next = propagate_nominal(self.state_13, control_4, dt)
        epsilon = 1e-6
        transition = np.empty((_ERROR_SIZE, _ERROR_SIZE), dtype=float)
        for column in range(_ERROR_SIZE):
            perturbation = np.zeros(_ERROR_SIZE)
            perturbation[column] = epsilon
            perturbed_next = propagate_nominal(
                inject_error(self.state_13, perturbation), control_4, dt
            )
            transition[:, column] = state_error(nominal_next, perturbed_next) / epsilon

        acceleration_variance = self.options.acceleration_process_std_mps2**2
        angular_variance = self.options.angular_acceleration_process_std_radps2**2
        timestep = float(dt)
        process_covariance = np.zeros((_ERROR_SIZE, _ERROR_SIZE), dtype=float)
        for offset, variance in ((0, acceleration_variance), (6, angular_variance)):
            position_slice = slice(offset, offset + 3)
            velocity_slice = slice(offset + 3, offset + 6)
            process_covariance[position_slice, position_slice] = (
                np.eye(3) * variance * timestep**4 / 4.0
            )
            cross = np.eye(3) * variance * timestep**3 / 2.0
            process_covariance[position_slice, velocity_slice] = cross
            process_covariance[velocity_slice, position_slice] = cross
            process_covariance[velocity_slice, velocity_slice] = np.eye(3) * variance * timestep**2

        self.state_13 = nominal_next
        self.covariance_12 = _project_covariance(
            transition @ self.covariance_12 @ transition.T + process_covariance,
            self.minimum_covariance_eigenvalue,
        )

    def update(self, measurement: VehicleMeasurement | None) -> None:
        if measurement is None:
            return
        if self.state_13 is None or self.covariance_12 is None:
            raise RuntimeError("vehicle estimator must be reset before update")
        innovation = state_error(self.state_13, measurement.state_13)
        innovation_covariance = self.covariance_12 + measurement.covariance_12
        gain = np.linalg.solve(innovation_covariance.T, self.covariance_12.T).T
        correction = gain @ innovation
        self.state_13 = inject_error(self.state_13, correction)
        identity = np.eye(_ERROR_SIZE)
        remainder = identity - gain
        posterior_covariance = (
            remainder @ self.covariance_12 @ remainder.T + gain @ measurement.covariance_12 @ gain.T
        )
        # First-order covariance reset for q+ = q (*) Exp(delta_theta_hat).
        reset_jacobian = np.eye(_ERROR_SIZE)
        reset_jacobian[6:9, 6:9] -= 0.5 * _skew(correction[6:9])
        self.covariance_12 = _project_covariance(
            reset_jacobian @ posterior_covariance @ reset_jacobian.T,
            self.minimum_covariance_eigenvalue,
        )

    @property
    def belief(self) -> VehicleBelief:
        if self.state_13 is None or self.covariance_12 is None:
            raise RuntimeError("vehicle estimator has no belief before reset")
        return VehicleBelief(self.state_13, self.covariance_12)


class ConstantVelocityObstacleFilter:
    """Linear 6D position/velocity Kalman filter with white acceleration noise."""

    def __init__(
        self,
        options: ObstacleFilterOptions,
        *,
        minimum_covariance_eigenvalue: float = 1e-12,
    ):
        self.options = options
        self.minimum_covariance_eigenvalue = float(minimum_covariance_eigenvalue)
        self.state_6: np.ndarray | None = None
        self.covariance_6: np.ndarray | None = None

    def reset(self, measurement: ObstacleMeasurement) -> None:
        self.state_6 = np.concatenate(
            [np.asarray(measurement.position_world, dtype=float).reshape(3), np.zeros(3)]
        )
        self.covariance_6 = np.diag(
            np.concatenate(
                [
                    np.full(3, self.options.initial_position_std_m**2),
                    np.full(3, self.options.initial_velocity_std_mps**2),
                ]
            )
        )

    @staticmethod
    def transition(dt: float) -> np.ndarray:
        transition = np.eye(6)
        transition[:3, 3:] = np.eye(3) * float(dt)
        return transition

    def predict(self, dt: float) -> None:
        if self.state_6 is None or self.covariance_6 is None:
            raise RuntimeError("obstacle filter must be reset before predict")
        transition = self.transition(dt)
        variance = self.options.acceleration_process_std_mps2**2
        timestep = float(dt)
        process_covariance = variance * np.block(
            [
                [np.eye(3) * timestep**4 / 4.0, np.eye(3) * timestep**3 / 2.0],
                [np.eye(3) * timestep**3 / 2.0, np.eye(3) * timestep**2],
            ]
        )
        self.state_6 = transition @ self.state_6
        self.covariance_6 = _project_covariance(
            transition @ self.covariance_6 @ transition.T + process_covariance,
            self.minimum_covariance_eigenvalue,
        )

    def update(self, measurement: ObstacleMeasurement | None) -> None:
        if measurement is None:
            return
        if self.state_6 is None or self.covariance_6 is None:
            raise RuntimeError("obstacle filter must be reset before update")
        observation = np.zeros((3, 6))
        observation[:, :3] = np.eye(3)
        innovation = measurement.position_world - observation @ self.state_6
        innovation_covariance = (
            observation @ self.covariance_6 @ observation.T + measurement.covariance_3
        )
        gain = np.linalg.solve(innovation_covariance.T, (self.covariance_6 @ observation.T).T).T
        self.state_6 = self.state_6 + gain @ innovation
        identity = np.eye(6)
        remainder = identity - gain @ observation
        self.covariance_6 = _project_covariance(
            remainder @ self.covariance_6 @ remainder.T + gain @ measurement.covariance_3 @ gain.T,
            self.minimum_covariance_eigenvalue,
        )


@dataclass(frozen=True, slots=True)
class EstimationSnapshot:
    vehicle_belief: VehicleBelief
    obstacle_beliefs: tuple[ObstacleBelief, ...]
    vehicle_measurement_available: bool
    obstacle_measurement_available: np.ndarray
    vehicle_measurement_state_13: np.ndarray | None
    obstacle_measurement_positions: np.ndarray


class NativeBeliefEstimator:
    """Own sensors, vehicle ESEKF and one obstacle KF per scene obstacle."""

    def __init__(
        self,
        options: EstimationOptions,
        obstacle_specs: Sequence[Mapping[str, Any]],
        *,
        horizon_steps: int,
        timestep_s: float,
    ):
        if not options.enabled:
            raise ValueError("NativeBeliefEstimator requires estimation.enabled=true")
        self.options = options
        self.obstacle_specs = tuple(dict(obstacle) for obstacle in obstacle_specs)
        self.horizon_steps = int(horizon_steps)
        self.timestep_s = _positive(timestep_s, "timestep_s")
        self.sensor = NativeSensorSimulator(options.sensor, options.seed)
        self.vehicle_filter = ErrorStateEkf(
            options.vehicle_filter,
            minimum_covariance_eigenvalue=options.minimum_covariance_eigenvalue,
        )
        self.obstacle_filters = tuple(
            ConstantVelocityObstacleFilter(
                options.obstacle_filter,
                minimum_covariance_eigenvalue=options.minimum_covariance_eigenvalue,
            )
            for _ in self.obstacle_specs
        )
        self._last_snapshot: EstimationSnapshot | None = None

    def _snapshot(self, frame: SensorFrame) -> EstimationSnapshot:
        obstacle_beliefs: list[ObstacleBelief] = []
        for index, (spec, tracker) in enumerate(zip(self.obstacle_specs, self.obstacle_filters)):
            if tracker.state_6 is None or tracker.covariance_6 is None:
                raise RuntimeError("obstacle estimator has no belief before reset")
            positions = np.stack(
                [
                    (
                        ConstantVelocityObstacleFilter.transition(step * self.timestep_s)
                        @ tracker.state_6
                    )[:3]
                    for step in range(self.horizon_steps + 1)
                ]
            )
            obstacle_beliefs.append(
                ObstacleBelief(
                    mean_state_6=tracker.state_6,
                    covariance_6=tracker.covariance_6,
                    shape=SphericalObstacle(float(spec["radius"])),
                    name=str(spec.get("name", f"obstacle_{index}")),
                    predicted_positions=positions,
                )
            )
        obstacle_measurements = np.full((len(self.obstacle_specs), 3), np.nan, dtype=float)
        for measurement in frame.obstacles:
            obstacle_measurements[measurement.obstacle_index] = measurement.position_world
        return EstimationSnapshot(
            vehicle_belief=self.vehicle_filter.belief,
            obstacle_beliefs=tuple(obstacle_beliefs),
            vehicle_measurement_available=frame.vehicle is not None,
            obstacle_measurement_available=frame.obstacle_available.copy(),
            vehicle_measurement_state_13=(
                None if frame.vehicle is None else frame.vehicle.state_13.copy()
            ),
            obstacle_measurement_positions=obstacle_measurements,
        )

    def reset(
        self,
        vehicle_truth_13: np.ndarray,
        obstacle_truth_positions: np.ndarray,
    ) -> EstimationSnapshot:
        self.sensor.reset(len(self.obstacle_specs))
        frame = self.sensor.sample(
            vehicle_truth_13,
            obstacle_truth_positions,
            dt=0.0,
            force_measurements=True,
        )
        if frame.vehicle is None or len(frame.obstacles) != len(self.obstacle_filters):
            raise RuntimeError("forced startup measurements were not produced")
        self.vehicle_filter.reset(frame.vehicle)
        by_index = {item.obstacle_index: item for item in frame.obstacles}
        for index, tracker in enumerate(self.obstacle_filters):
            tracker.reset(by_index[index])
        self._last_snapshot = self._snapshot(frame)
        return self._last_snapshot

    def advance(
        self,
        vehicle_truth_13: np.ndarray,
        obstacle_truth_positions: np.ndarray,
        previous_control_4: np.ndarray,
        *,
        dt: float,
    ) -> EstimationSnapshot:
        self.vehicle_filter.predict(previous_control_4, dt)
        for tracker in self.obstacle_filters:
            tracker.predict(dt)
        frame = self.sensor.sample(
            vehicle_truth_13,
            obstacle_truth_positions,
            dt=dt,
        )
        self.vehicle_filter.update(frame.vehicle)
        measurements = {item.obstacle_index: item for item in frame.obstacles}
        for index, tracker in enumerate(self.obstacle_filters):
            tracker.update(measurements.get(index))
        self._last_snapshot = self._snapshot(frame)
        return self._last_snapshot

    @property
    def snapshot(self) -> EstimationSnapshot:
        if self._last_snapshot is None:
            raise RuntimeError("estimator must be reset before reading a snapshot")
        return self._last_snapshot
