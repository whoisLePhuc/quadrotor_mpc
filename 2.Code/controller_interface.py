"""Belief-based controller contract for the native 13-state simulation.

This module intentionally has no dependency on MuJoCo, CasADi or do-mpc.  It is
the boundary between state estimation and control: simulation code supplies
beliefs, while controller implementations return a normalized solution object.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

VEHICLE_STATE_SIZE = 13
VEHICLE_ERROR_STATE_SIZE = 12
OBSTACLE_STATE_SIZE = 6
CONTROL_SIZE = 4


def _readonly_array(value, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    result = array.copy()
    result.setflags(write=False)
    return result


def _covariance(value, size: int, label: str) -> np.ndarray:
    covariance = _readonly_array(value, (size, size), label)
    if not np.allclose(covariance, covariance.T, rtol=1e-8, atol=1e-10):
        raise ValueError(f"{label} must be symmetric")
    if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-9:
        raise ValueError(f"{label} must be positive semidefinite")
    return covariance


@dataclass(frozen=True, slots=True)
class VehicleBelief:
    """Nominal quaternion state and 12D local error-state covariance."""

    mean_state_13: np.ndarray
    error_covariance_12: np.ndarray

    def __post_init__(self) -> None:
        mean = _readonly_array(
            self.mean_state_13,
            (VEHICLE_STATE_SIZE,),
            "VehicleBelief.mean_state_13",
        )
        quaternion_norm = float(np.linalg.norm(mean[6:10]))
        if quaternion_norm <= 1e-12:
            raise ValueError("VehicleBelief quaternion must have non-zero norm")
        normalized = mean.copy()
        normalized[6:10] /= quaternion_norm
        normalized.setflags(write=False)
        object.__setattr__(self, "mean_state_13", normalized)
        object.__setattr__(
            self,
            "error_covariance_12",
            _covariance(
                self.error_covariance_12,
                VEHICLE_ERROR_STATE_SIZE,
                "VehicleBelief.error_covariance_12",
            ),
        )

    @classmethod
    def exact(cls, state_13) -> VehicleBelief:
        """Create a zero-covariance belief for the deterministic baseline."""
        return cls(
            mean_state_13=np.asarray(state_13, dtype=float).reshape(VEHICLE_STATE_SIZE),
            error_covariance_12=np.zeros(
                (VEHICLE_ERROR_STATE_SIZE, VEHICLE_ERROR_STATE_SIZE),
                dtype=float,
            ),
        )


@runtime_checkable
class ObstacleShape(Protocol):
    """Geometry required by a controller without exposing scene implementation."""

    @property
    def bounding_radius_m(self) -> float: ...


@dataclass(frozen=True, slots=True)
class SphericalObstacle:
    radius_m: float

    def __post_init__(self) -> None:
        radius = float(self.radius_m)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("SphericalObstacle.radius_m must be finite and > 0")
        object.__setattr__(self, "radius_m", radius)

    @property
    def bounding_radius_m(self) -> float:
        return self.radius_m


@dataclass(frozen=True, slots=True)
class ObstacleBelief:
    """6D position/velocity belief plus optional horizon mean prediction."""

    mean_state_6: np.ndarray
    covariance_6: np.ndarray
    shape: ObstacleShape
    name: str = ""
    predicted_positions: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mean_state_6",
            _readonly_array(
                self.mean_state_6,
                (OBSTACLE_STATE_SIZE,),
                "ObstacleBelief.mean_state_6",
            ),
        )
        object.__setattr__(
            self,
            "covariance_6",
            _covariance(
                self.covariance_6,
                OBSTACLE_STATE_SIZE,
                "ObstacleBelief.covariance_6",
            ),
        )
        if not isinstance(self.shape, ObstacleShape):
            raise TypeError("ObstacleBelief.shape must implement ObstacleShape")
        object.__setattr__(self, "name", str(self.name))
        if self.predicted_positions is not None:
            prediction = np.asarray(self.predicted_positions, dtype=float)
            if prediction.ndim != 2 or prediction.shape[1] != 3 or prediction.shape[0] < 1:
                raise ValueError(
                    "ObstacleBelief.predicted_positions must have shape (steps, 3)"
                )
            if not np.all(np.isfinite(prediction)):
                raise ValueError(
                    "ObstacleBelief.predicted_positions must contain only finite values"
                )
            prediction = prediction.copy()
            prediction.setflags(write=False)
            object.__setattr__(self, "predicted_positions", prediction)

    def mean_positions(self, steps: int, dt: float) -> np.ndarray:
        """Return the supplied horizon or constant-velocity extrapolation."""
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if self.predicted_positions is not None:
            if self.predicted_positions.shape[0] != steps:
                raise ValueError(
                    "ObstacleBelief prediction length does not match controller horizon"
                )
            return self.predicted_positions
        times = np.arange(steps, dtype=float)[:, None] * float(dt)
        return self.mean_state_6[:3] + times * self.mean_state_6[3:]


@dataclass(frozen=True, slots=True)
class ControlGoal:
    position: np.ndarray
    quaternion_wxyz: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position",
            _readonly_array(self.position, (3,), "ControlGoal.position"),
        )
        quaternion = _readonly_array(
            self.quaternion_wxyz,
            (4,),
            "ControlGoal.quaternion_wxyz",
        )
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-12:
            raise ValueError("ControlGoal quaternion must have non-zero norm")
        normalized = quaternion.copy()
        normalized /= norm
        normalized.setflags(write=False)
        object.__setattr__(self, "quaternion_wxyz", normalized)


@dataclass(frozen=True, slots=True)
class ControlSolution:
    """Backend-independent result returned for every controller tick."""

    command: np.ndarray
    nominal_states: np.ndarray
    predicted_covariances: np.ndarray
    chance_margins: np.ndarray
    risk_allocations: np.ndarray
    slacks: np.ndarray
    solver_status: str

    def __post_init__(self) -> None:
        command = _readonly_array(self.command, (CONTROL_SIZE,), "ControlSolution.command")
        states = np.asarray(self.nominal_states, dtype=float)
        if states.ndim != 2 or states.shape[1] != VEHICLE_STATE_SIZE:
            raise ValueError("ControlSolution.nominal_states must have shape (steps, 13)")
        if states.shape[0] < 1 or not np.all(np.isfinite(states)):
            raise ValueError("ControlSolution.nominal_states must be finite and non-empty")
        covariances = np.asarray(self.predicted_covariances, dtype=float)
        expected_covariance_shape = (
            states.shape[0],
            VEHICLE_ERROR_STATE_SIZE,
            VEHICLE_ERROR_STATE_SIZE,
        )
        if covariances.shape != expected_covariance_shape:
            raise ValueError(
                "ControlSolution.predicted_covariances must have shape "
                f"{expected_covariance_shape}"
            )
        margins = np.asarray(self.chance_margins, dtype=float)
        risks = np.asarray(self.risk_allocations, dtype=float)
        slacks = np.asarray(self.slacks, dtype=float)
        if margins.ndim != 2 or margins.shape[0] != states.shape[0]:
            raise ValueError("ControlSolution.chance_margins must have shape (steps, obstacles)")
        if risks.shape != margins.shape or slacks.shape != margins.shape:
            raise ValueError(
                "ControlSolution risk_allocations and slacks must match chance_margins"
            )
        for label, array in (
            ("nominal_states", states),
            ("predicted_covariances", covariances),
            ("chance_margins", margins),
            ("risk_allocations", risks),
            ("slacks", slacks),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"ControlSolution.{label} must contain only finite values")
            copied = array.copy()
            copied.setflags(write=False)
            object.__setattr__(self, label, copied)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "solver_status", str(self.solver_status))

    @property
    def predicted_positions(self) -> np.ndarray:
        return self.nominal_states[:, :3]


class Controller(Protocol):
    """Common controller lifecycle used by native, headless and future batch runs."""

    @property
    def horizon_steps(self) -> int: ...

    def reset(self, belief: VehicleBelief) -> None: ...

    def solve(
        self,
        belief: VehicleBelief,
        obstacles: Sequence[ObstacleBelief],
        goal: ControlGoal,
        time_s: float,
    ) -> ControlSolution: ...
