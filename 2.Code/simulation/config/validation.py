"""Validation helpers for typed scenario configuration.

This module validates already-parsed ``ScenarioConfig`` objects. It does not
read YAML files, does not convert legacy dictionaries, and does not construct
runtime/controller/engine objects.

Expected usage:
    data = load_yaml(path)
    config = parse_canonical_or_legacy(data)
    validate_scenario_config(config)
    return config
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

import numpy as np

from ccmpc.types import as_goal3, as_position3, as_state9, as_velocity3
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


T = TypeVar("T")

SUPPORTED_WORLD_FRAMES: tuple[str, ...] = ("W",)
SUPPORTED_OBSTACLE_FRAMES: tuple[str, ...] = ("W",)


def validate_scenario_config(config: ScenarioConfig) -> None:
    """Validate a complete canonical scenario config.

    Raises
    ------
    ConfigError
        If any field violates the scenario config contract.
    """
    _require_type(config, ScenarioConfig, "config")

    validate_schema_version(config.schema_version)
    validate_scenario_metadata(config.scenario)
    validate_world_config(config.world)

    as_state9(config.initial_state)
    validate_goal_config(config.goal)

    validate_success_config(config.success)
    validate_termination_config(config.termination)

    if config.runtime_overrides is not None:
        validate_runtime_overrides(config.runtime_overrides)

    validate_obstacle_specs(config.obstacles)

    _validate_state_inside_world_bounds(config.initial_state, config.world)
    _validate_goal_inside_world_bounds(config.goal, config.world)
    _validate_termination_runtime_consistency(
        config.termination,
        config.runtime_overrides,
    )


def validate_schema_version(schema_version: str) -> None:
    """Validate scenario schema version string."""
    _require_type(schema_version, str, "schema_version")
    if not schema_version.strip():
        raise ConfigError("schema_version must be a non-empty string.")

    if schema_version != "legacy" and not schema_version.startswith("1."):
        raise ConfigError(
            "Unsupported scenario schema_version. "
            f"Expected '1.x' or 'legacy', got {schema_version!r}."
        )


def validate_scenario_metadata(metadata: ScenarioMetadata) -> None:
    """Validate scenario metadata."""
    _require_type(metadata, ScenarioMetadata, "scenario")

    _require_non_empty_string(metadata.id, "scenario.id")
    _require_non_empty_string(metadata.name, "scenario.name")

    if metadata.description is not None:
        _require_type(metadata.description, str, "scenario.description")

    _require_type(metadata.tags, tuple, "scenario.tags")
    for index, tag in enumerate(metadata.tags):
        _require_non_empty_string(tag, f"scenario.tags[{index}]")


def validate_world_config(world: WorldConfig) -> None:
    """Validate world frame, convention, gravity, and bounds."""
    _require_type(world, WorldConfig, "world")

    _require_non_empty_string(world.frame, "world.frame")
    if world.frame not in SUPPORTED_WORLD_FRAMES:
        raise ConfigError(
            f"Unsupported world.frame {world.frame!r}. "
            f"Supported frames: {SUPPORTED_WORLD_FRAMES}."
        )

    _require_type(world.convention, WorldConvention, "world.convention")
    if world.convention is not WorldConvention.Z_UP:
        raise ConfigError("Only Z_UP world convention is supported.")

    _validate_float_tuple(world.gravity, 3, "world.gravity")

    if world.bounds is not None:
        validate_world_bounds(world.bounds)


def validate_world_bounds(bounds: WorldBounds) -> None:
    """Validate axis-aligned world bounds."""
    _require_type(bounds, WorldBounds, "world.bounds")

    _validate_bounds_pair(bounds.x, "world.bounds.x")
    _validate_bounds_pair(bounds.y, "world.bounds.y")
    _validate_bounds_pair(bounds.z, "world.bounds.z")


def validate_goal_config(goal: GoalConfig) -> None:
    """Validate goal position and threshold."""
    _require_type(goal, GoalConfig, "goal")

    as_goal3(goal.position)
    _require_positive_finite(goal.threshold, "goal.threshold")


def validate_obstacle_specs(obstacles: Iterable[ObstacleSpec]) -> None:
    """Validate all obstacle specifications and reject duplicate IDs."""
    if not isinstance(obstacles, tuple):
        raise ConfigError("obstacles must be a tuple[ObstacleSpec, ...].")

    seen_ids: set[str] = set()

    for index, obstacle in enumerate(obstacles):
        validate_obstacle_spec(obstacle, index=index)

        if obstacle.id in seen_ids:
            raise ConfigError(f"Duplicate obstacle id: {obstacle.id!r}.")
        seen_ids.add(obstacle.id)


def validate_obstacle_spec(obstacle: ObstacleSpec, *, index: int | None = None) -> None:
    """Validate one obstacle specification."""
    prefix = "obstacle" if index is None else f"obstacles[{index}]"

    _require_type(obstacle, ObstacleSpec, prefix)

    _require_non_empty_string(obstacle.id, f"{prefix}.id")

    _require_type(obstacle.type, ObstacleType, f"{prefix}.type")
    if obstacle.type is not ObstacleType.BOX_ELLIPSOID:
        raise ConfigError(
            f"Unsupported {prefix}.type {obstacle.type!r}. "
            f"Only {ObstacleType.BOX_ELLIPSOID.value!r} is supported initially."
        )

    _require_non_empty_string(obstacle.frame, f"{prefix}.frame")
    if obstacle.frame not in SUPPORTED_OBSTACLE_FRAMES:
        raise ConfigError(
            f"Unsupported {prefix}.frame {obstacle.frame!r}. "
            f"Supported frames: {SUPPORTED_OBSTACLE_FRAMES}."
        )

    as_position3(obstacle.position)
    as_velocity3(obstacle.velocity)

    _validate_float_tuple(obstacle.size, 3, f"{prefix}.size")
    for axis, value in zip(("length", "width", "height"), obstacle.size, strict=True):
        _require_positive_finite(value, f"{prefix}.size.{axis}")

    _require_finite_number(obstacle.yaw, f"{prefix}.yaw")

    if obstacle.covariance is not None:
        validate_obstacle_covariance(obstacle.covariance, name=f"{prefix}.covariance")

    _require_type(obstacle.motion_model, ObstacleMotionModel, f"{prefix}.motion_model")
    if obstacle.motion_model is not ObstacleMotionModel.CONSTANT_VELOCITY:
        raise ConfigError(
            f"Unsupported {prefix}.motion_model {obstacle.motion_model!r}. "
            "Only constant_velocity is supported initially."
        )

    _require_type(obstacle.active, bool, f"{prefix}.active")

    if obstacle.appearance_time is not None:
        _require_non_negative_finite(
            obstacle.appearance_time,
            f"{prefix}.appearance_time",
        )

    if obstacle.disappearance_time is not None:
        _require_non_negative_finite(
            obstacle.disappearance_time,
            f"{prefix}.disappearance_time",
        )

    if (
        obstacle.appearance_time is not None
        and obstacle.disappearance_time is not None
        and obstacle.appearance_time >= obstacle.disappearance_time
    ):
        raise ConfigError(
            f"{prefix}.appearance_time must be smaller than "
            f"{prefix}.disappearance_time."
        )

    _require_type(obstacle.metadata, dict, f"{prefix}.metadata")


def validate_obstacle_covariance(
    covariance: ObstacleCovarianceConfig,
    *,
    name: str = "obstacle.covariance",
) -> None:
    """Validate obstacle covariance standard-deviation representation."""
    _require_type(covariance, ObstacleCovarianceConfig, name)

    _validate_float_tuple(covariance.position_std, 3, f"{name}.position_std")
    _validate_float_tuple(covariance.velocity_std, 3, f"{name}.velocity_std")

    for idx, value in enumerate(covariance.position_std):
        _require_non_negative_finite(value, f"{name}.position_std[{idx}]")

    for idx, value in enumerate(covariance.velocity_std):
        _require_non_negative_finite(value, f"{name}.velocity_std[{idx}]")


def validate_success_config(success: SuccessConfig) -> None:
    """Validate success criteria."""
    _require_type(success, SuccessConfig, "success")

    _require_positive_finite(success.goal_threshold, "success.goal_threshold")
    _require_type(success.require_collision_free, bool, "success.require_collision_free")
    _require_type(success.require_altitude_valid, bool, "success.require_altitude_valid")

    if success.max_final_speed is not None:
        _require_non_negative_finite(success.max_final_speed, "success.max_final_speed")


def validate_termination_config(termination: TerminationConfig) -> None:
    """Validate termination conditions."""
    _require_type(termination, TerminationConfig, "termination")

    _require_positive_finite(termination.max_time, "termination.max_time")

    if termination.max_steps is not None:
        _require_positive_integer(termination.max_steps, "termination.max_steps")

    _require_type(
        termination.terminate_on_collision,
        bool,
        "termination.terminate_on_collision",
    )
    _require_type(
        termination.terminate_on_altitude_violation,
        bool,
        "termination.terminate_on_altitude_violation",
    )
    _require_type(
        termination.terminate_on_goal_reached,
        bool,
        "termination.terminate_on_goal_reached",
    )
    _require_type(
        termination.terminate_on_solver_failure,
        bool,
        "termination.terminate_on_solver_failure",
    )


def validate_runtime_overrides(overrides: RuntimeOverrides) -> None:
    """Validate scenario-specific runtime overrides."""
    _require_type(overrides, RuntimeOverrides, "runtime_overrides")

    if overrides.sim_dt is not None:
        _require_positive_finite(overrides.sim_dt, "runtime_overrides.sim_dt")

    if overrides.max_time is not None:
        _require_positive_finite(overrides.max_time, "runtime_overrides.max_time")

    if overrides.max_steps is not None:
        _require_positive_integer(overrides.max_steps, "runtime_overrides.max_steps")


def _validate_state_inside_world_bounds(state: object, world: WorldConfig) -> None:
    """Reject initial state outside configured world bounds."""
    if world.bounds is None:
        return

    state_array = as_state9(state)
    position = state_array[0:3]

    _validate_point_inside_bounds(
        position,
        world.bounds,
        name="initial_state.position",
    )


def _validate_goal_inside_world_bounds(goal: GoalConfig, world: WorldConfig) -> None:
    """Reject goal outside configured world bounds."""
    if world.bounds is None:
        return

    position = as_goal3(goal.position)
    _validate_point_inside_bounds(position, world.bounds, name="goal.position")


def _validate_point_inside_bounds(
    point: object,
    bounds: WorldBounds,
    *,
    name: str,
) -> None:
    """Validate point [x, y, z] is inside world bounds."""
    position = as_position3(point)

    checks = (
        ("x", position[0], bounds.x),
        ("y", position[1], bounds.y),
        ("z", position[2], bounds.z),
    )

    for axis, value, (lower, upper) in checks:
        if value < lower or value > upper:
            raise ConfigError(
                f"{name}.{axis}={value} is outside world bounds "
                f"[{lower}, {upper}]."
            )


def _validate_termination_runtime_consistency(
    termination: TerminationConfig,
    overrides: RuntimeOverrides | None,
) -> None:
    """Validate simple consistency between termination and runtime overrides."""
    if overrides is None:
        return

    if overrides.max_time is not None and overrides.max_time > termination.max_time:
        raise ConfigError(
            "runtime_overrides.max_time must not be larger than "
            "termination.max_time. Resolve precedence explicitly in AppConfig."
        )

    if (
        overrides.max_steps is not None
        and termination.max_steps is not None
        and overrides.max_steps > termination.max_steps
    ):
        raise ConfigError(
            "runtime_overrides.max_steps must not be larger than "
            "termination.max_steps. Resolve precedence explicitly in AppConfig."
        )


def _validate_float_tuple(value: object, expected_len: int, name: str) -> None:
    """Validate tuple of finite numeric values."""
    if not isinstance(value, tuple):
        raise ConfigError(f"{name} must be a tuple of length {expected_len}.")

    if len(value) != expected_len:
        raise ConfigError(f"{name} must have length {expected_len}, got {len(value)}.")

    for idx, item in enumerate(value):
        _require_finite_number(item, f"{name}[{idx}]")


def _validate_bounds_pair(value: object, name: str) -> None:
    """Validate bounds tuple (lower, upper)."""
    _validate_float_tuple(value, 2, name)
    lower, upper = value  # type: ignore[misc]

    if lower >= upper:
        raise ConfigError(f"{name} lower bound must be smaller than upper bound.")


def _require_type(value: object, expected_type: type[T], name: str) -> None:
    """Require expected runtime type."""
    if not isinstance(value, expected_type):
        raise ConfigError(
            f"{name} must be {expected_type.__name__}, got {type(value).__name__}."
        )


def _require_non_empty_string(value: object, name: str) -> None:
    """Require non-empty string."""
    _require_type(value, str, name)
    if not value.strip():
        raise ConfigError(f"{name} must be a non-empty string.")


def _require_finite_number(value: object, name: str) -> None:
    """Require int or float, excluding bool, NaN, and Inf."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a finite number.")

    if not np.isfinite(float(value)):
        raise ConfigError(f"{name} must be finite.")


def _require_positive_finite(value: object, name: str) -> None:
    """Require finite number strictly greater than zero."""
    _require_finite_number(value, name)
    if float(value) <= 0.0:
        raise ConfigError(f"{name} must be > 0.")


def _require_non_negative_finite(value: object, name: str) -> None:
    """Require finite number greater than or equal to zero."""
    _require_finite_number(value, name)
    if float(value) < 0.0:
        raise ConfigError(f"{name} must be >= 0.")


def _require_positive_integer(value: object, name: str) -> None:
    """Require positive integer, excluding bool."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ConfigError(f"{name} must be > 0.")


__all__ = [
    "SUPPORTED_OBSTACLE_FRAMES",
    "SUPPORTED_WORLD_FRAMES",
    "validate_goal_config",
    "validate_obstacle_covariance",
    "validate_obstacle_spec",
    "validate_obstacle_specs",
    "validate_runtime_overrides",
    "validate_scenario_config",
    "validate_scenario_metadata",
    "validate_schema_version",
    "validate_success_config",
    "validate_termination_config",
    "validate_world_bounds",
    "validate_world_config",
]
