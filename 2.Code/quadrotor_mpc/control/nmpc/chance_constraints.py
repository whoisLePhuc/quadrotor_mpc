"""Spherical chance-constraint tightening for the native NMPC controller.

Covariance remains outside the nonlinear program.  This module projects the
relative position covariance onto a collision normal computed from a nominal
trajectory and returns deterministic, time-varying safety radii:

    r_tight = r_drone + r_obstacle + r_margin + beta * sigma

Risk allocation is delegated to :mod:`native_risk_budget`.  This keeps the
geometric projection independent of whether epsilon is interpreted as an
individual constraint risk or a joint receding-horizon budget.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np

from quadrotor_mpc.control.nmpc.risk_budget import (
    RiskAllocation,
    RiskBudgetOptions,
    allocate_risk_budget,
)
from quadrotor_mpc.core.timing import TimingRecorder


def _positive(value: Any, label: str, *, allow_zero: bool = False) -> float:
    number = float(value)
    valid = number >= 0.0 if allow_zero else number > 0.0
    if not np.isfinite(number) or not valid:
        relation = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{label} must be finite and {relation}")
    return number


@dataclass(frozen=True, slots=True)
class ChanceConstraintOptions:
    """Configuration for spherical chance constraints and their risk semantics."""

    enabled: bool = False
    individual_epsilon: float = 0.05
    risk_budget: RiskBudgetOptions = field(default_factory=RiskBudgetOptions)
    soft_constraint: bool = True
    slack_penalty: float = 1_000_000.0
    slack_tolerance_m: float = 1e-6
    minimum_direction_norm_m: float = 1e-8
    distance_smoothing_m2: float = 1e-6

    def __post_init__(self) -> None:
        epsilon = float(self.individual_epsilon)
        if not np.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
            raise ValueError(
                "controller.chance_constraints.individual_epsilon must be in (0, 0.5)"
            )
        object.__setattr__(self, "individual_epsilon", epsilon)
        if not isinstance(self.risk_budget, RiskBudgetOptions):
            raise TypeError(
                "controller.chance_constraints.risk_budget must be RiskBudgetOptions"
            )
        object.__setattr__(
            self,
            "slack_penalty",
            _positive(
                self.slack_penalty,
                "controller.chance_constraints.slack_penalty",
            ),
        )
        object.__setattr__(
            self,
            "slack_tolerance_m",
            _positive(
                self.slack_tolerance_m,
                "controller.chance_constraints.slack_tolerance_m",
                allow_zero=True,
            ),
        )
        object.__setattr__(
            self,
            "minimum_direction_norm_m",
            _positive(
                self.minimum_direction_norm_m,
                "controller.chance_constraints.minimum_direction_norm_m",
            ),
        )
        object.__setattr__(
            self,
            "distance_smoothing_m2",
            _positive(
                self.distance_smoothing_m2,
                "controller.chance_constraints.distance_smoothing_m2",
                allow_zero=True,
            ),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ChanceConstraintOptions:
        constraint_type = str(raw.get("type", "spherical")).lower()
        if constraint_type != "spherical":
            raise ValueError(
                "controller.chance_constraints.type must be 'spherical' in Stage 5"
            )
        risk_raw = raw.get("risk_budget", {})
        if not isinstance(risk_raw, Mapping):
            raise TypeError(
                "controller.chance_constraints.risk_budget must be a mapping"
            )
        return cls(
            enabled=bool(raw.get("enabled", False)),
            individual_epsilon=raw.get("individual_epsilon", 0.05),
            risk_budget=RiskBudgetOptions.from_mapping(risk_raw),
            soft_constraint=bool(raw.get("soft_constraint", True)),
            slack_penalty=raw.get("slack_penalty", 1_000_000.0),
            slack_tolerance_m=raw.get("slack_tolerance_m", 1e-6),
            minimum_direction_norm_m=raw.get("minimum_direction_norm_m", 1e-8),
            distance_smoothing_m2=raw.get("distance_smoothing_m2", 1e-6),
        )

    @property
    def beta(self) -> float:
        """One-sided Gaussian quantile ``Phi^-1(1-epsilon)``."""
        return float(NormalDist().inv_cdf(1.0 - self.individual_epsilon))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "type": "spherical",
            "individual_epsilon": self.individual_epsilon,
            "risk_budget": self.risk_budget.to_mapping(),
            "soft_constraint": self.soft_constraint,
            "slack_penalty": self.slack_penalty,
            "slack_tolerance_m": self.slack_tolerance_m,
            "minimum_direction_norm_m": self.minimum_direction_norm_m,
            "distance_smoothing_m2": self.distance_smoothing_m2,
        }


@dataclass(frozen=True, slots=True)
class SphericalChanceProfile:
    """Precomputed values supplied to one NMPC solve."""

    collision_normals: np.ndarray
    projected_sigmas_m: np.ndarray
    tightenings_m: np.ndarray
    safety_radii_m: np.ndarray
    risk_allocations: np.ndarray
    gaussian_quantiles: np.ndarray
    risk_semantics: str
    risk_allocation_method: str
    configured_total_epsilon: float | None
    allocated_epsilon: float
    remaining_epsilon: float | None
    active_constraint_count: int
    budget_status: str

    def __post_init__(self) -> None:
        sigmas = np.asarray(self.projected_sigmas_m, dtype=float)
        expected = sigmas.shape
        if sigmas.ndim != 2:
            raise ValueError("projected_sigmas_m must have shape (steps, obstacles)")
        normals = np.asarray(self.collision_normals, dtype=float)
        if normals.shape != expected + (3,):
            raise ValueError("collision_normals must have shape (steps, obstacles, 3)")
        arrays = {
            "projected_sigmas_m": sigmas,
            "tightenings_m": np.asarray(self.tightenings_m, dtype=float),
            "safety_radii_m": np.asarray(self.safety_radii_m, dtype=float),
            "risk_allocations": np.asarray(self.risk_allocations, dtype=float),
            "gaussian_quantiles": np.asarray(self.gaussian_quantiles, dtype=float),
        }
        for label, array in arrays.items():
            if array.shape != expected:
                raise ValueError(f"{label} must have shape {expected}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{label} must contain only finite values")
            if np.any(array < 0.0):
                raise ValueError(f"{label} must be nonnegative")
            copied = array.copy()
            copied.setflags(write=False)
            object.__setattr__(self, label, copied)
        if not np.all(np.isfinite(normals)):
            raise ValueError("collision_normals must contain only finite values")
        copied_normals = normals.copy()
        copied_normals.setflags(write=False)
        object.__setattr__(self, "collision_normals", copied_normals)
        object.__setattr__(self, "risk_semantics", str(self.risk_semantics))
        object.__setattr__(
            self,
            "risk_allocation_method",
            str(self.risk_allocation_method),
        )
        object.__setattr__(
            self,
            "active_constraint_count",
            int(self.active_constraint_count),
        )
        object.__setattr__(self, "budget_status", str(self.budget_status))
        for label in (
            "configured_total_epsilon",
            "remaining_epsilon",
        ):
            value = getattr(self, label)
            if value is not None:
                number = float(value)
                if not np.isfinite(number) or number < -1e-12:
                    raise ValueError(f"{label} must be finite and nonnegative")
                object.__setattr__(self, label, max(0.0, number))
        allocated = float(self.allocated_epsilon)
        if not np.isfinite(allocated) or allocated < -1e-12:
            raise ValueError("allocated_epsilon must be finite and nonnegative")
        object.__setattr__(self, "allocated_epsilon", max(0.0, allocated))


def _fallback_direction(relative_covariance: np.ndarray) -> np.ndarray:
    """Use the largest-variance direction when the nominal centers coincide."""
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (relative_covariance + relative_covariance.T)
    )
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return direction / norm


def build_spherical_chance_profile(
    *,
    vehicle_positions: np.ndarray,
    obstacle_positions: np.ndarray,
    vehicle_covariances: np.ndarray,
    obstacle_covariances: np.ndarray,
    base_safety_radii_m: np.ndarray,
    options: ChanceConstraintOptions,
    timing_recorder: TimingRecorder | None = None,
) -> SphericalChanceProfile:
    """Project relative covariance and build deterministic tightened radii.

    ``obstacle_positions`` is obstacle-major ``(obstacles, steps, 3)`` while
    covariance and returned profile arrays are step-major.
    """

    positions = np.asarray(vehicle_positions, dtype=float)
    obstacle_means = np.asarray(obstacle_positions, dtype=float)
    vehicle_covariance = np.asarray(vehicle_covariances, dtype=float)
    obstacle_covariance = np.asarray(obstacle_covariances, dtype=float)
    base_radii = np.asarray(base_safety_radii_m, dtype=float).reshape(-1)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("vehicle_positions must have shape (steps, 3)")
    steps = positions.shape[0]
    obstacle_count = base_radii.shape[0]
    if obstacle_means.shape != (obstacle_count, steps, 3):
        raise ValueError(
            "obstacle_positions must have shape (obstacles, steps, 3)"
        )
    if vehicle_covariance.shape != (steps, 12, 12):
        raise ValueError("vehicle_covariances must have shape (steps, 12, 12)")
    if obstacle_covariance.shape != (steps, obstacle_count, 6, 6):
        raise ValueError(
            "obstacle_covariances must have shape (steps, obstacles, 6, 6)"
        )
    for label, array in (
        ("vehicle_positions", positions),
        ("obstacle_positions", obstacle_means),
        ("vehicle_covariances", vehicle_covariance),
        ("obstacle_covariances", obstacle_covariance),
        ("base_safety_radii_m", base_radii),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{label} must contain only finite values")
    if np.any(base_radii <= 0.0):
        raise ValueError("base_safety_radii_m must be > 0")

    normals = np.empty((steps, obstacle_count, 3), dtype=float)
    sigmas = np.zeros((steps, obstacle_count), dtype=float)
    for step in range(steps):
        vehicle_position_covariance = vehicle_covariance[step, :3, :3]
        for obstacle in range(obstacle_count):
            relative_covariance = (
                vehicle_position_covariance
                + obstacle_covariance[step, obstacle, :3, :3]
            )
            relative = positions[step] - obstacle_means[obstacle, step]
            distance = float(np.linalg.norm(relative))
            if distance > options.minimum_direction_norm_m:
                normal = relative / distance
            else:
                normal = _fallback_direction(relative_covariance)
            variance = float(normal @ relative_covariance @ normal)
            normals[step, obstacle] = normal
            sigmas[step, obstacle] = np.sqrt(max(variance, 0.0))

    risk_context = (
        timing_recorder.measure("risk_allocation_time_ms")
        if timing_recorder is not None
        else nullcontext()
    )
    with risk_context:
        risk: RiskAllocation = allocate_risk_budget(
            steps=steps,
            obstacle_count=obstacle_count,
            enabled=options.enabled,
            individual_epsilon=options.individual_epsilon,
            options=options.risk_budget,
        )
    tightening_context = (
        timing_recorder.measure("tightening_time_ms")
        if timing_recorder is not None
        else nullcontext()
    )
    with tightening_context:
        tightening = risk.gaussian_quantiles * sigmas
        safety_radii = base_radii[None, :] + tightening
    return SphericalChanceProfile(
        collision_normals=normals,
        projected_sigmas_m=sigmas,
        tightenings_m=tightening,
        safety_radii_m=safety_radii,
        risk_allocations=risk.epsilons,
        gaussian_quantiles=risk.gaussian_quantiles,
        risk_semantics=risk.semantics,
        risk_allocation_method=risk.allocation,
        configured_total_epsilon=risk.configured_total_epsilon,
        allocated_epsilon=risk.allocated_epsilon,
        remaining_epsilon=risk.remaining_epsilon,
        active_constraint_count=risk.active_constraint_count,
        budget_status=risk.budget_status,
    )


def evaluate_spherical_constraints(
    *,
    vehicle_positions: np.ndarray,
    obstacle_positions: np.ndarray,
    safety_radii_m: np.ndarray,
    distance_smoothing_m2: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return chance residuals and minimum required nonnegative slacks."""

    vehicle = np.asarray(vehicle_positions, dtype=float)
    obstacles = np.asarray(obstacle_positions, dtype=float)
    safe = np.asarray(safety_radii_m, dtype=float)
    if vehicle.ndim != 2 or vehicle.shape[1] != 3:
        raise ValueError("vehicle_positions must have shape (steps, 3)")
    if obstacles.ndim != 3 or obstacles.shape[2] != 3:
        raise ValueError("obstacle_positions must have shape (obstacles, steps, 3)")
    expected = (vehicle.shape[0], obstacles.shape[0])
    if obstacles.shape[1] != vehicle.shape[0] or safe.shape != expected:
        raise ValueError("safety_radii_m must have shape (steps, obstacles)")
    relative = vehicle[:, None, :] - np.swapaxes(obstacles, 0, 1)
    distance = np.sqrt(
        np.sum(relative * relative, axis=2) + float(distance_smoothing_m2)
    )
    residual = distance - safe
    return residual, np.maximum(0.0, -residual)
