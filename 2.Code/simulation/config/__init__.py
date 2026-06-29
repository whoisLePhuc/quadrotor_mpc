"""Configuration layer for the quadrotor CC-MPC simulation."""

from simulation.config.schema import (
    ConfigError,
    GoalConfig,
    ObstacleCovarianceConfig,
    ObstacleMotionModel,
    ObstacleSpec,
    ObstacleType,
    RuntimeOverrides,
    ScenarioConfig,
    ScenarioMetadata,
    SuccessConfig,
    TerminationConfig,
    WorldBounds,
    WorldConfig,
    WorldConvention,
)

__all__ = [
    "ConfigError",
    "GoalConfig",
    "ObstacleCovarianceConfig",
    "ObstacleMotionModel",
    "ObstacleSpec",
    "ObstacleType",
    "RuntimeOverrides",
    "ScenarioConfig",
    "ScenarioMetadata",
    "SuccessConfig",
    "TerminationConfig",
    "WorldBounds",
    "WorldConfig",
    "WorldConvention",
]