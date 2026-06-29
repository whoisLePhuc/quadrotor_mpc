"""Typed configuration schema for quadrotor CC-MPC simulation.

This module defines immutable dataclasses used after YAML parsing.

Important:
    - This file should not read YAML files.
    - This file should not contain legacy conversion logic.
    - This file should not construct runtime/controller/engine objects.
    - Validation logic should live in validation.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ccmpc.types import FloatArray


class ConfigError(ValueError):
    """Raised when a simulation config is malformed."""


class WorldConvention(str, Enum):
    """Supported world-frame conventions."""

    Z_UP = "Z_UP"


class ObstacleType(str, Enum):
    """Supported obstacle geometry types."""

    BOX_ELLIPSOID = "box_ellipsoid"


class ObstacleMotionModel(str, Enum):
    """Supported obstacle prediction models."""

    CONSTANT_VELOCITY = "constant_velocity"


@dataclass(frozen=True)
class ScenarioMetadata:
    """Human-readable scenario metadata."""

    id: str
    name: str
    description: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorldBounds:
    """Axis-aligned world bounds."""

    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]


@dataclass(frozen=True)
class WorldConfig:
    """World frame and environment configuration."""

    frame: str
    convention: WorldConvention
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    bounds: WorldBounds | None = None


@dataclass(frozen=True)
class GoalConfig:
    """Goal position and goal-reaching threshold."""

    position: FloatArray
    threshold: float


@dataclass(frozen=True)
class ObstacleCovarianceConfig:
    """Obstacle uncertainty represented by standard deviations."""

    position_std: tuple[float, float, float]
    velocity_std: tuple[float, float, float]


@dataclass(frozen=True)
class ObstacleSpec:
    """Typed obstacle specification loaded from scenario config."""

    id: str
    type: ObstacleType
    frame: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw: float
    velocity: tuple[float, float, float]
    covariance: ObstacleCovarianceConfig | None = None
    motion_model: ObstacleMotionModel = ObstacleMotionModel.CONSTANT_VELOCITY
    active: bool = True
    appearance_time: float | None = None
    disappearance_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SuccessConfig:
    """Scenario-level success criteria."""

    goal_threshold: float
    require_collision_free: bool = True
    require_altitude_valid: bool = True
    max_final_speed: float | None = None


@dataclass(frozen=True)
class TerminationConfig:
    """Scenario-level termination conditions."""

    max_time: float
    max_steps: int | None = None
    terminate_on_collision: bool = True
    terminate_on_altitude_violation: bool = True
    terminate_on_goal_reached: bool = True
    terminate_on_solver_failure: bool = False


@dataclass(frozen=True)
class RuntimeOverrides:
    """Scenario-specific runtime overrides."""

    sim_dt: float | None = None
    max_time: float | None = None
    max_steps: int | None = None


@dataclass(frozen=True)
class ScenarioConfig:
    """Canonical typed scenario configuration."""

    schema_version: str
    scenario: ScenarioMetadata
    world: WorldConfig
    initial_state: FloatArray
    goal: GoalConfig
    obstacles: tuple[ObstacleSpec, ...]
    success: SuccessConfig
    termination: TerminationConfig
    runtime_overrides: RuntimeOverrides | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)