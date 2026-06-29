"""Legacy scenario conversion utilities.

This module converts the old flat demo scenario dictionaries into the canonical
``ScenarioConfig`` dataclass tree.

Important:
    - This file does not read YAML files.
    - This file does not construct runtime/controller/engine objects.
    - This file should not be the long-term scenario format.
    - Loader code should call this only when ``schema_version`` is absent.

Legacy flat format example:
    start: [0, 0, 1, 0, 0, 0, 0, 0, 0]
    goal: [6, 4, 2.5]
    target_altitude: 2.0
    obstacles:
      - position: [2.5, 1.0, 1.5]
        size: [0.6, 0.6, 1.0]
        yaw: 0.3
        velocity: [0.2, 0.0, 0.0]
    goal_threshold: 0.4
    sim_timestep: 0.02
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ccmpc.types import as_goal3, as_state9

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
    WorldConfig,
    WorldConvention,
)


LegacyDict = Mapping[str, Any]


def convert_legacy_scenario(
    data: LegacyDict,
    *,
    scenario_id: str = "legacy_scenario",
    scenario_name: str = "Legacy scenario",
) -> ScenarioConfig:
    """Convert a legacy flat scenario dictionary into ``ScenarioConfig``.

    Parameters
    ----------
    data:
        Parsed YAML dictionary using the old flat demo schema.
    scenario_id:
        Stable ID to assign to the converted scenario if the legacy file does
        not provide one.
    scenario_name:
        Human-readable name to assign to the converted scenario if the legacy
        file does not provide one.

    Returns
    -------
    ScenarioConfig
        Canonical typed scenario config with ``schema_version='legacy'``.

    Raises
    ------
    ConfigError
        If required legacy keys are missing or malformed.
    """
    if not isinstance(data, Mapping):
        raise ConfigError(
            f"Legacy scenario must be a mapping/dict, got {type(data).__name__}."
        )

    initial_state = as_state9(_get_required(data, "start"))
    goal_position = as_goal3(_get_required(data, "goal"))

    goal_threshold = _as_positive_float(
        data.get("goal_threshold", 0.4),
        "goal_threshold",
    )

    obstacles = tuple(
        _convert_legacy_obstacle(raw_obstacle, index=index)
        for index, raw_obstacle in enumerate(data.get("obstacles", []) or [])
    )

    sim_dt = data.get("sim_timestep", None)
    runtime_overrides = RuntimeOverrides(
        sim_dt=_as_positive_float(sim_dt, "sim_timestep") if sim_dt is not None else None,
        max_time=_as_positive_float(data["max_time"], "max_time")
        if "max_time" in data
        else None,
        max_steps=_as_positive_int(data["max_steps"], "max_steps")
        if "max_steps" in data
        else None,
    )

    termination = TerminationConfig(
        max_time=_as_positive_float(data.get("max_time", 30.0), "max_time"),
        max_steps=_as_positive_int(data["max_steps"], "max_steps")
        if "max_steps" in data
        else None,
        terminate_on_collision=_as_bool(
            data.get("terminate_on_collision", True),
            "terminate_on_collision",
        ),
        terminate_on_altitude_violation=_as_bool(
            data.get("terminate_on_altitude_violation", True),
            "terminate_on_altitude_violation",
        ),
        terminate_on_goal_reached=_as_bool(
            data.get("terminate_on_goal_reached", True),
            "terminate_on_goal_reached",
        ),
        terminate_on_solver_failure=_as_bool(
            data.get("terminate_on_solver_failure", False),
            "terminate_on_solver_failure",
        ),
    )

    metadata = ScenarioMetadata(
        id=str(data.get("scenario_id", data.get("id", scenario_id))),
        name=str(data.get("scenario_name", data.get("name", scenario_name))),
        description=(
            str(data["description"])
            if "description" in data and data["description"] is not None
            else "Converted from legacy flat scenario format."
        ),
        tags=("legacy",),
    )

    raw_metadata = {
        "source_format": "legacy_flat",
        "legacy_keys": sorted(str(key) for key in data.keys()),
    }
    if "target_altitude" in data:
        raw_metadata["target_altitude"] = data["target_altitude"]

    return ScenarioConfig(
        schema_version="legacy",
        scenario=metadata,
        world=WorldConfig(
            frame="W",
            convention=WorldConvention.Z_UP,
            gravity=(0.0, 0.0, -9.81),
            bounds=None,
        ),
        initial_state=initial_state,
        goal=GoalConfig(
            position=goal_position,
            threshold=goal_threshold,
        ),
        obstacles=obstacles,
        success=SuccessConfig(
            goal_threshold=goal_threshold,
            require_collision_free=True,
            require_altitude_valid=True,
        ),
        termination=termination,
        runtime_overrides=runtime_overrides,
        raw_metadata=raw_metadata,
    )


def is_legacy_scenario_dict(data: object) -> bool:
    """Return True when a parsed dict looks like the old flat scenario schema."""
    if not isinstance(data, Mapping):
        return False

    return "schema_version" not in data and ("start" in data or "goal" in data)


def _convert_legacy_obstacle(raw_obstacle: object, *, index: int) -> ObstacleSpec:
    """Convert one old-style obstacle dict into ``ObstacleSpec``."""
    name = f"obstacles[{index}]"

    if not isinstance(raw_obstacle, Mapping):
        raise ConfigError(f"{name} must be a mapping/dict.")

    position = _as_float_tuple3(_get_required(raw_obstacle, "position"), f"{name}.position")
    size = _as_float_tuple3(_get_required(raw_obstacle, "size"), f"{name}.size")
    velocity = _as_float_tuple3(raw_obstacle.get("velocity", (0.0, 0.0, 0.0)), f"{name}.velocity")

    yaw = _as_float(raw_obstacle.get("yaw", 0.0), f"{name}.yaw")

    covariance = None
    if "covariance" in raw_obstacle and raw_obstacle["covariance"] is not None:
        covariance = _convert_legacy_obstacle_covariance(
            raw_obstacle["covariance"],
            name=f"{name}.covariance",
        )

    obstacle_id = str(raw_obstacle.get("id", f"obs_{index + 1:03d}"))

    return ObstacleSpec(
        id=obstacle_id,
        type=ObstacleType.BOX_ELLIPSOID,
        frame=str(raw_obstacle.get("frame", "W")),
        position=position,
        size=size,
        yaw=yaw,
        velocity=velocity,
        covariance=covariance,
        motion_model=ObstacleMotionModel.CONSTANT_VELOCITY,
        active=_as_bool(raw_obstacle.get("active", True), f"{name}.active"),
        appearance_time=_as_optional_non_negative_float(
            raw_obstacle.get("appearance_time"),
            f"{name}.appearance_time",
        ),
        disappearance_time=_as_optional_non_negative_float(
            raw_obstacle.get("disappearance_time"),
            f"{name}.disappearance_time",
        ),
        metadata={
            "source_format": "legacy_flat",
        },
    )


def _convert_legacy_obstacle_covariance(
    raw_covariance: object,
    *,
    name: str,
) -> ObstacleCovarianceConfig:
    """Convert optional legacy obstacle covariance fields."""
    if not isinstance(raw_covariance, Mapping):
        raise ConfigError(f"{name} must be a mapping/dict.")

    position_std = _as_float_tuple3(
        raw_covariance.get("position_std", (0.0, 0.0, 0.0)),
        f"{name}.position_std",
    )
    velocity_std = _as_float_tuple3(
        raw_covariance.get("velocity_std", (0.0, 0.0, 0.0)),
        f"{name}.velocity_std",
    )

    return ObstacleCovarianceConfig(
        position_std=position_std,
        velocity_std=velocity_std,
    )


def _get_required(data: Mapping[str, Any], key: str) -> Any:
    """Return required key from mapping or raise ConfigError."""
    if key not in data:
        raise ConfigError(f"Missing required legacy scenario key: {key!r}.")
    return data[key]


def _as_float(value: object, name: str) -> float:
    """Convert value to finite float."""
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a finite number, got bool.")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a finite number.") from exc

    if result != result or result in (float("inf"), float("-inf")):
        raise ConfigError(f"{name} must be finite.")

    return result


def _as_positive_float(value: object, name: str) -> float:
    """Convert value to finite float > 0."""
    result = _as_float(value, name)
    if result <= 0.0:
        raise ConfigError(f"{name} must be > 0.")
    return result


def _as_optional_non_negative_float(value: object, name: str) -> float | None:
    """Convert optional finite float >= 0."""
    if value is None:
        return None

    result = _as_float(value, name)
    if result < 0.0:
        raise ConfigError(f"{name} must be >= 0.")
    return result


def _as_positive_int(value: object, name: str) -> int:
    """Convert value to positive int, rejecting bool."""
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a positive integer, got bool.")

    if not isinstance(value, int):
        raise ConfigError(f"{name} must be a positive integer.")

    if value <= 0:
        raise ConfigError(f"{name} must be > 0.")

    return value


def _as_bool(value: object, name: str) -> bool:
    """Validate bool."""
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be bool.")
    return value


def _as_float_tuple3(value: object, name: str) -> tuple[float, float, float]:
    """Convert a sequence to a finite 3-float tuple."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(f"{name} must be a sequence of 3 finite numbers.")

    if len(value) != 3:
        raise ConfigError(f"{name} must have length 3, got {len(value)}.")

    return (
        _as_float(value[0], f"{name}[0]"),
        _as_float(value[1], f"{name}[1]"),
        _as_float(value[2], f"{name}[2]"),
    )


__all__ = [
    "convert_legacy_scenario",
    "is_legacy_scenario_dict",
]
