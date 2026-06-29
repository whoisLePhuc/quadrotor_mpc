"""Uncertainty propagation for quadrotor Chance-Constrained MPC.

Canonical State9:
    [x, y, z, vx, vy, vz, roll, pitch, yaw]

This module provides the covariance API needed by the core CC-MPC controller:

    propagator = UncertaintyPropagator.from_config(config)
    Gamma_list = propagator.propagate(Gamma_0, x_traj, u_traj, dynamics)
    Sigma_pos = propagator.position_covariance(Gamma_list[k])

The module is independent from the simulation runtime layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ccmpc.types import (
    CONTROL_DIM,
    STATE_DIM,
    FloatArray,
    as_control_command4,
    as_gamma9x9,
    as_sigma3x3,
    as_state9,
)


POSITION_SLICE = slice(0, 3)
VELOCITY_SLICE = slice(3, 6)
ATTITUDE_SLICE = slice(6, 9)


class UncertaintyError(RuntimeError):
    """Base exception raised by uncertainty code."""


class UncertaintyConfigError(UncertaintyError):
    """Raised when uncertainty configuration is invalid."""


class UncertaintyPropagationError(UncertaintyError):
    """Raised when covariance propagation fails."""


@runtime_checkable
class LinearizableDynamicsProtocol(Protocol):
    """Minimal dynamics interface required for covariance propagation."""

    def linearize(
        self,
        x_bar: FloatArray,
        u_bar: FloatArray,
        dt: float,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return A, B, C linearization matrices."""


@dataclass(frozen=True)
class UncertaintyConfig:
    """Configuration for State9 covariance propagation.

    Noise values are interpreted as standard deviations.
    """

    dt: float
    process_noise_pos: float
    process_noise_vel: float
    process_noise_att: float
    init_pos_noise: float
    init_vel_noise: float
    init_att_noise: float
    vio_drift_pos: float = 0.0
    vio_drift_vel: float = 0.0
    vio_drift_att: float = 0.0
    covariance_floor: float = 1e-12

    def __post_init__(self) -> None:
        """Validate numeric fields."""
        _as_positive_float(self.dt, "dt")
        for name in (
            "process_noise_pos",
            "process_noise_vel",
            "process_noise_att",
            "init_pos_noise",
            "init_vel_noise",
            "init_att_noise",
            "vio_drift_pos",
            "vio_drift_vel",
            "vio_drift_att",
            "covariance_floor",
        ):
            _as_non_negative_float(getattr(self, name), name)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "UncertaintyConfig":
        """Create config from full project config or uncertainty-only mapping.

        Full project config shape:

            config["controller"]["prediction"]["timestep"]
            config["controller"]["uncertainty"]

        Direct uncertainty-only shape:

            {"dt": ..., "process_noise_pos": ..., ...}
        """
        if not isinstance(config, dict):
            raise UncertaintyConfigError("config must be a dictionary.")

        if "controller" in config:
            controller_cfg = config.get("controller")
            if not isinstance(controller_cfg, dict):
                raise UncertaintyConfigError("config['controller'] must be a dictionary.")

            prediction_cfg = controller_cfg.get("prediction", {})
            if prediction_cfg is None:
                prediction_cfg = {}
            if not isinstance(prediction_cfg, dict):
                raise UncertaintyConfigError(
                    "config['controller']['prediction'] must be a dictionary."
                )

            uncertainty_cfg = controller_cfg.get("uncertainty")
            if not isinstance(uncertainty_cfg, dict):
                raise UncertaintyConfigError(
                    "config['controller']['uncertainty'] must be a dictionary."
                )

            dt = uncertainty_cfg.get(
                "dt",
                prediction_cfg.get("timestep", prediction_cfg.get("dt")),
            )
        else:
            uncertainty_cfg = config
            dt = uncertainty_cfg.get("dt", uncertainty_cfg.get("timestep"))

        if dt is None:
            raise UncertaintyConfigError(
                "Missing uncertainty timestep. Provide 'dt' or "
                "controller.prediction.timestep."
            )

        return cls(
            dt=float(dt),
            process_noise_pos=float(_required(uncertainty_cfg, "process_noise_pos")),
            process_noise_vel=float(_required(uncertainty_cfg, "process_noise_vel")),
            process_noise_att=float(_required(uncertainty_cfg, "process_noise_att")),
            init_pos_noise=float(_required(uncertainty_cfg, "init_pos_noise")),
            init_vel_noise=float(_required(uncertainty_cfg, "init_vel_noise")),
            init_att_noise=float(_required(uncertainty_cfg, "init_att_noise")),
            vio_drift_pos=float(uncertainty_cfg.get("vio_drift_pos", 0.0)),
            vio_drift_vel=float(uncertainty_cfg.get("vio_drift_vel", 0.0)),
            vio_drift_att=float(uncertainty_cfg.get("vio_drift_att", 0.0)),
            covariance_floor=float(uncertainty_cfg.get("covariance_floor", 1e-12)),
        )


class UncertaintyPropagator:
    """Propagate State9 covariance over a nominal MPC trajectory.

    Discrete propagation:

        Gamma_{k+1} = A_k Gamma_k A_k.T + Q

    where ``A_k`` comes from ``dynamics.linearize(x_k, u_k, dt)``.
    """

    def __init__(self, config: UncertaintyConfig) -> None:
        if not isinstance(config, UncertaintyConfig):
            raise UncertaintyConfigError(
                "UncertaintyPropagator expects UncertaintyConfig."
            )

        self.config = config
        self.dt = config.dt
        self.Gamma_0: FloatArray = self.initial_covariance()
        self.Q: FloatArray = self.process_covariance()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "UncertaintyPropagator":
        """Create propagator from a config dictionary."""
        return cls(UncertaintyConfig.from_config(config))

    def initial_covariance(self) -> FloatArray:
        """Return initial State9 covariance Gamma_0."""
        variances = grouped_state_variances(
            position_std=self.config.init_pos_noise,
            velocity_std=self.config.init_vel_noise,
            attitude_std=self.config.init_att_noise,
        )
        return as_gamma9x9(np.diag(variances))

    def process_covariance(self) -> FloatArray:
        """Return per-step State9 process covariance Q."""
        process_variances = grouped_state_variances(
            position_std=self.config.process_noise_pos,
            velocity_std=self.config.process_noise_vel,
            attitude_std=self.config.process_noise_att,
        )
        drift_variances = grouped_state_variances(
            position_std=self.config.vio_drift_pos,
            velocity_std=self.config.vio_drift_vel,
            attitude_std=self.config.vio_drift_att,
        )

        variances = process_variances + self.dt * drift_variances

        if self.config.covariance_floor > 0.0:
            variances = variances + self.config.covariance_floor

        return as_gamma9x9(np.diag(variances))

    def propagate(
        self,
        Gamma_0: FloatArray | None,
        x_traj: FloatArray,
        u_traj: FloatArray,
        dynamics: LinearizableDynamicsProtocol,
        *,
        dt: float | None = None,
    ) -> list[FloatArray]:
        """Propagate covariance along a nominal trajectory.

        Accepted trajectory layouts:
            x_traj: (9, N+1) or (N+1, 9)
            u_traj: (4, N)   or (N, 4)

        Returns:
            [Gamma_0, Gamma_1, ..., Gamma_N]
        """
        step_dt = self.dt if dt is None else _as_positive_float(dt, "dt")
        gamma = self.Gamma_0.copy() if Gamma_0 is None else as_gamma9x9(Gamma_0)

        states = as_state_trajectory_state_major(x_traj)
        controls = as_control_trajectory_control_major(u_traj)

        horizon = controls.shape[1]
        if states.shape[1] != horizon + 1:
            raise UncertaintyPropagationError(
                "x_traj must contain exactly one more state than u_traj controls: "
                f"x_traj={states.shape}, u_traj={controls.shape}."
            )

        if not hasattr(dynamics, "linearize"):
            raise UncertaintyPropagationError(
                "dynamics must provide linearize(x_bar, u_bar, dt)."
            )

        gamma_list = [gamma.copy()]

        for k in range(horizon):
            x_bar = as_state9(states[:, k])
            u_bar = as_control_command4(controls[:, k])

            try:
                A_k, _, _ = dynamics.linearize(x_bar, u_bar, step_dt)
            except Exception as exc:
                raise UncertaintyPropagationError(
                    f"dynamics.linearize failed at step {k}: {exc}"
                ) from exc

            A = as_dynamics_jacobian(A_k, name=f"A_{k}")
            gamma = A @ gamma @ A.T + self.Q
            gamma = project_to_psd_gamma(gamma)
            gamma_list.append(gamma.copy())

        return gamma_list

    def position_covariance(self, Gamma: FloatArray) -> FloatArray:
        """Extract position covariance Sigma from State9 covariance."""
        gamma = as_gamma9x9(Gamma)
        return as_sigma3x3(gamma[POSITION_SLICE, POSITION_SLICE])

    def velocity_covariance(self, Gamma: FloatArray) -> FloatArray:
        """Extract velocity covariance from State9 covariance."""
        gamma = as_gamma9x9(Gamma)
        return as_sigma3x3(gamma[VELOCITY_SLICE, VELOCITY_SLICE])

    def attitude_covariance(self, Gamma: FloatArray) -> FloatArray:
        """Extract attitude covariance from State9 covariance."""
        gamma = as_gamma9x9(Gamma)
        return as_sigma3x3(gamma[ATTITUDE_SLICE, ATTITUDE_SLICE])

    def reset_initial_covariance(self, Gamma_0: FloatArray | None = None) -> None:
        """Reset stored initial covariance."""
        self.Gamma_0 = self.initial_covariance() if Gamma_0 is None else as_gamma9x9(Gamma_0)


def grouped_state_variances(
    *,
    position_std: float,
    velocity_std: float,
    attitude_std: float,
) -> FloatArray:
    """Return State9 variance vector from grouped standard deviations."""
    pos = _as_non_negative_float(position_std, "position_std")
    vel = _as_non_negative_float(velocity_std, "velocity_std")
    att = _as_non_negative_float(attitude_std, "attitude_std")

    return np.array(
        [
            pos**2,
            pos**2,
            pos**2,
            vel**2,
            vel**2,
            vel**2,
            att**2,
            att**2,
            att**2,
        ],
        dtype=np.float64,
    )


def as_state_trajectory_state_major(value: FloatArray) -> FloatArray:
    """Validate trajectory and return state-major shape (9, T)."""
    array = _as_finite_array(value, "x_traj")

    if array.ndim != 2:
        raise UncertaintyPropagationError(
            f"x_traj must be 2D, got shape {array.shape}."
        )

    if array.shape[0] == STATE_DIM:
        return array.copy()

    if array.shape[1] == STATE_DIM:
        return array.T.copy()

    raise UncertaintyPropagationError(
        f"x_traj must have shape (9, T) or (T, 9), got {array.shape}."
    )


def as_control_trajectory_control_major(value: FloatArray) -> FloatArray:
    """Validate trajectory and return control-major shape (4, T)."""
    array = _as_finite_array(value, "u_traj")

    if array.ndim != 2:
        raise UncertaintyPropagationError(
            f"u_traj must be 2D, got shape {array.shape}."
        )

    if array.shape[0] == CONTROL_DIM:
        return array.copy()

    if array.shape[1] == CONTROL_DIM:
        return array.T.copy()

    raise UncertaintyPropagationError(
        f"u_traj must have shape (4, T) or (T, 4), got {array.shape}."
    )


def as_dynamics_jacobian(value: FloatArray, *, name: str = "A") -> FloatArray:
    """Validate dynamics state Jacobian with shape (9, 9)."""
    array = _as_finite_array(value, name)

    if array.shape != (STATE_DIM, STATE_DIM):
        raise UncertaintyPropagationError(
            f"{name} must have shape (9, 9), got {array.shape}."
        )

    return array.copy()


def project_to_psd_gamma(Gamma: FloatArray, *, floor: float = 0.0) -> FloatArray:
    """Symmetrize and project State9 covariance to the PSD cone."""
    gamma = as_gamma9x9(symmetrize(Gamma))
    floor_value = _as_non_negative_float(floor, "floor")

    eigvals, eigvecs = np.linalg.eigh(gamma)
    eigvals_clipped = np.maximum(eigvals, floor_value)
    projected = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    return as_gamma9x9(symmetrize(projected))


def symmetrize(matrix: FloatArray) -> FloatArray:
    """Return 0.5 * (matrix + matrix.T) for a square finite matrix."""
    array = _as_finite_array(matrix, "matrix")

    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise UncertaintyPropagationError(
            f"matrix must be square, got shape {array.shape}."
        )

    return 0.5 * (array + array.T)


def _required(mapping: dict[str, Any], key: str) -> Any:
    """Return required mapping value or raise config error."""
    if key not in mapping:
        raise UncertaintyConfigError(f"Missing uncertainty config key: {key}")
    return mapping[key]


def _as_finite_array(value: FloatArray, name: str) -> FloatArray:
    """Convert to finite float64 NumPy array."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise UncertaintyPropagationError(
            f"{name} must be convertible to float64 array."
        ) from exc

    if not np.all(np.isfinite(array)):
        raise UncertaintyPropagationError(f"{name} must contain only finite values.")

    return array


def _as_non_negative_float(value: float, name: str) -> float:
    """Validate finite scalar >= 0."""
    if isinstance(value, bool):
        raise UncertaintyConfigError(f"{name} must be a finite scalar, got bool.")

    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise UncertaintyConfigError(f"{name} must be a finite scalar.") from exc

    if not np.isfinite(scalar) or scalar < 0.0:
        raise UncertaintyConfigError(f"{name} must be finite and >= 0.")

    return scalar


def _as_positive_float(value: float, name: str) -> float:
    """Validate finite scalar > 0."""
    if isinstance(value, bool):
        raise UncertaintyConfigError(f"{name} must be a finite scalar, got bool.")

    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise UncertaintyConfigError(f"{name} must be a finite scalar.") from exc

    if not np.isfinite(scalar) or scalar <= 0.0:
        raise UncertaintyConfigError(f"{name} must be finite and > 0.")

    return scalar


__all__ = [
    "ATTITUDE_SLICE",
    "LinearizableDynamicsProtocol",
    "POSITION_SLICE",
    "UncertaintyConfig",
    "UncertaintyConfigError",
    "UncertaintyError",
    "UncertaintyPropagationError",
    "UncertaintyPropagator",
    "VELOCITY_SLICE",
    "as_control_trajectory_control_major",
    "as_dynamics_jacobian",
    "as_state_trajectory_state_major",
    "grouped_state_variances",
    "project_to_psd_gamma",
    "symmetrize",
]
