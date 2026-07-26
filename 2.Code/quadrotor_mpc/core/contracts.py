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
                raise ValueError("ObstacleBelief.predicted_positions must have shape (steps, 3)")
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
    predicted_obstacle_covariances: np.ndarray | None = None
    projected_uncertainties: np.ndarray | None = None
    tightened_safety_radii: np.ndarray | None = None
    risk_semantics: str = ""
    risk_allocation_method: str = ""
    risk_budget_total: float | None = None
    risk_budget_allocated: float = 0.0
    risk_budget_remaining: float | None = None
    risk_constraint_count: int = 0
    risk_budget_status: str = ""
    primary_solver_status: str = ""
    primary_solver_success: bool = True
    primary_solver_iterations: int = 0
    primary_solver_primal_residual: float = 0.0
    primary_solver_dual_residual: float = 0.0
    command_source: str = "PRIMARY_NMPC"
    solution_accepted: bool = True
    fallback_active: bool = False
    fallback_level: int = 0
    fallback_reason: str = ""
    consecutive_rejections: int = 0
    solve_time_ms: float = 0.0
    deadline_missed: bool = False
    safety_assurance_status: str = ""

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
                f"ControlSolution.predicted_covariances must have shape {expected_covariance_shape}"
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
        projected_uncertainties = (
            np.zeros_like(margins)
            if self.projected_uncertainties is None
            else np.asarray(self.projected_uncertainties, dtype=float)
        )
        tightened_safety_radii = (
            np.zeros_like(margins)
            if self.tightened_safety_radii is None
            else np.asarray(self.tightened_safety_radii, dtype=float)
        )
        if (
            projected_uncertainties.shape != margins.shape
            or tightened_safety_radii.shape != margins.shape
        ):
            raise ValueError(
                "ControlSolution projected_uncertainties and tightened_safety_radii "
                "must match chance_margins"
            )
        if self.predicted_obstacle_covariances is None:
            obstacle_covariances = np.zeros(
                (states.shape[0], margins.shape[1], OBSTACLE_STATE_SIZE, OBSTACLE_STATE_SIZE),
                dtype=float,
            )
        else:
            obstacle_covariances = np.asarray(
                self.predicted_obstacle_covariances,
                dtype=float,
            )
        expected_obstacle_covariance_shape = (
            states.shape[0],
            margins.shape[1],
            OBSTACLE_STATE_SIZE,
            OBSTACLE_STATE_SIZE,
        )
        if obstacle_covariances.shape != expected_obstacle_covariance_shape:
            raise ValueError(
                "ControlSolution.predicted_obstacle_covariances must have shape "
                f"{expected_obstacle_covariance_shape}"
            )
        for label, array in (
            ("nominal_states", states),
            ("predicted_covariances", covariances),
            ("predicted_obstacle_covariances", obstacle_covariances),
            ("chance_margins", margins),
            ("risk_allocations", risks),
            ("slacks", slacks),
            ("projected_uncertainties", projected_uncertainties),
            ("tightened_safety_radii", tightened_safety_radii),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"ControlSolution.{label} must contain only finite values")
            copied = array.copy()
            copied.setflags(write=False)
            object.__setattr__(self, label, copied)
        for label, array in (
            ("predicted_covariances", covariances),
            ("predicted_obstacle_covariances", obstacle_covariances),
        ):
            if not np.allclose(array, np.swapaxes(array, -1, -2), rtol=1e-8, atol=1e-10):
                raise ValueError(f"ControlSolution.{label} must be symmetric")
            if array.size and float(np.min(np.linalg.eigvalsh(array))) < -1e-9:
                raise ValueError(f"ControlSolution.{label} must be positive semidefinite")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "solver_status", str(self.solver_status))
        object.__setattr__(self, "risk_semantics", str(self.risk_semantics))
        object.__setattr__(
            self,
            "risk_allocation_method",
            str(self.risk_allocation_method),
        )
        for label in (
            "risk_budget_total",
            "risk_budget_remaining",
        ):
            value = getattr(self, label)
            if value is not None:
                number = float(value)
                if not np.isfinite(number) or number < -1e-12:
                    raise ValueError(
                        f"ControlSolution.{label} must be finite and nonnegative"
                    )
                object.__setattr__(self, label, max(0.0, number))
        allocated = float(self.risk_budget_allocated)
        if not np.isfinite(allocated) or allocated < -1e-12:
            raise ValueError(
                "ControlSolution.risk_budget_allocated must be finite and nonnegative"
            )
        object.__setattr__(self, "risk_budget_allocated", max(0.0, allocated))
        count = int(self.risk_constraint_count)
        if count < 0:
            raise ValueError("ControlSolution.risk_constraint_count must be >= 0")
        object.__setattr__(self, "risk_constraint_count", count)
        object.__setattr__(
            self,
            "risk_budget_status",
            str(self.risk_budget_status),
        )
        object.__setattr__(
            self,
            "primary_solver_status",
            str(self.primary_solver_status),
        )
        object.__setattr__(
            self,
            "primary_solver_success",
            bool(self.primary_solver_success),
        )
        primary_solver_iterations = int(self.primary_solver_iterations)
        if primary_solver_iterations < 0:
            raise ValueError(
                "ControlSolution.primary_solver_iterations must be >= 0"
            )
        object.__setattr__(
            self,
            "primary_solver_iterations",
            primary_solver_iterations,
        )
        for label in (
            "primary_solver_primal_residual",
            "primary_solver_dual_residual",
        ):
            residual = float(getattr(self, label))
            if not np.isfinite(residual) or residual < 0.0:
                raise ValueError(
                    f"ControlSolution.{label} must be finite and >= 0"
                )
            object.__setattr__(self, label, residual)
        object.__setattr__(self, "command_source", str(self.command_source))
        object.__setattr__(
            self,
            "solution_accepted",
            bool(self.solution_accepted),
        )
        object.__setattr__(
            self,
            "fallback_active",
            bool(self.fallback_active),
        )
        fallback_level = int(self.fallback_level)
        if fallback_level < 0:
            raise ValueError("ControlSolution.fallback_level must be >= 0")
        object.__setattr__(self, "fallback_level", fallback_level)
        object.__setattr__(
            self,
            "fallback_reason",
            str(self.fallback_reason),
        )
        consecutive_rejections = int(self.consecutive_rejections)
        if consecutive_rejections < 0:
            raise ValueError(
                "ControlSolution.consecutive_rejections must be >= 0"
            )
        object.__setattr__(
            self,
            "consecutive_rejections",
            consecutive_rejections,
        )
        solve_time_ms = float(self.solve_time_ms)
        if not np.isfinite(solve_time_ms) or solve_time_ms < 0.0:
            raise ValueError(
                "ControlSolution.solve_time_ms must be finite and >= 0"
            )
        object.__setattr__(self, "solve_time_ms", solve_time_ms)
        object.__setattr__(
            self,
            "deadline_missed",
            bool(self.deadline_missed),
        )
        object.__setattr__(
            self,
            "safety_assurance_status",
            str(self.safety_assurance_status),
        )

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
