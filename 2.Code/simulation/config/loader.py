"""Scenario configuration loader.

This module is the public entry point for loading scenario YAML files into the
canonical ``ScenarioConfig`` dataclass tree.

Responsibilities:
    - read YAML file
    - parse canonical scenario schema
    - dispatch legacy flat scenario schema to legacy adapter
    - validate final ScenarioConfig

Non-responsibilities:
    - construct runtime/controller/engine objects
    - mutate global configuration
    - interpret MPC solver internals
    - implement scenario validation rules directly
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from ccmpc.types import as_goal3, as_state9

from simulation.config.legacy import (
    convert_legacy_scenario,
    is_legacy_scenario_dict,
)
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
from simulation.config.validation import validate_scenario_config


PathLike = str | Path
YamlDict = Mapping[str, Any]


def load_scenario_config(
    path: PathLike,
    *,
    validate: bool = True,
    warn_on_legacy: bool = True,
) -> ScenarioConfig:
    """Load a scenario YAML file and return validated ``ScenarioConfig``.

    Parameters
    ----------
    path:
        Path to a canonical or legacy scenario YAML file.
    validate:
        If True, run ``validate_scenario_config`` before returning.
    warn_on_legacy:
        If True, emit a warning when loading legacy flat scenario format.

    Returns
    -------
    ScenarioConfig
        Parsed and optionally validated scenario config.
    """
    yaml_path = Path(path)
    data = read_yaml_mapping(yaml_path)
    config = parse_scenario_config(
        data,
        source_path=yaml_path,
        warn_on_legacy=warn_on_legacy,
    )

    if validate:
        validate_scenario_config(config)

    return config


def read_yaml_mapping(path: PathLike) -> dict[str, Any]:
    """Read a YAML file and require the top-level object to be a mapping."""
    yaml_module = _import_yaml()
    yaml_path = Path(path)

    if not yaml_path.exists():
        raise ConfigError(f"Scenario config file does not exist: {yaml_path}")

    if not yaml_path.is_file():
        raise ConfigError(f"Scenario config path is not a file: {yaml_path}")

    try:
        text = yaml_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Failed to read scenario config file: {yaml_path}") from exc

    try:
        data = yaml_module.safe_load(text)
    except Exception as exc:  # pragma: no cover - PyYAML raises multiple subclasses.
        raise ConfigError(f"Failed to parse YAML file: {yaml_path}") from exc

    if data is None:
        raise ConfigError(f"Scenario config file is empty: {yaml_path}")

    if not isinstance(data, Mapping):
        raise ConfigError(
            "Scenario config top-level YAML object must be a mapping/dict, "
            f"got {type(data).__name__}."
        )

    return dict(data)


def parse_scenario_config(
    data: YamlDict,
    *,
    source_path: PathLike | None = None,
    warn_on_legacy: bool = True,
) -> ScenarioConfig:
    """Parse a canonical or legacy scenario dictionary into ``ScenarioConfig``."""
    if not isinstance(data, Mapping):
        raise ConfigError(
            f"Scenario config data must be a mapping/dict, got {type(data).__name__}."
        )

    if is_legacy_scenario_dict(data):
        if warn_on_legacy:
            warnings.warn(
                "Loading legacy flat scenario format. "
                "Please migrate to canonical schema_version: '1.0'.",
                UserWarning,
                stacklevel=2,
            )

        scenario_id = _scenario_id_from_path(source_path) if source_path else "legacy_scenario"
        scenario_name = scenario_id.replace("_", " ").replace("-", " ").title()

        return convert_legacy_scenario(
            data,
            scenario_id=scenario_id,
            scenario_name=scenario_name,
        )

    return parse_canonical_scenario(data)


def parse_canonical_scenario(data: YamlDict) -> ScenarioConfig:
    """Parse canonical scenario schema into ``ScenarioConfig``.

    Expected top-level canonical structure:
        schema_version
        scenario
        world
        initial_state
        goal
        obstacles
        success
        termination
        runtime_overrides
    """
    schema_version = _as_non_empty_string(
        _get_required(data, "schema_version"),
        "schema_version",
    )

    scenario = _parse_scenario_metadata(_get_required_mapping(data, "scenario"))
    world = _parse_world_config(_get_required_mapping(data, "world"))
    initial_state = _parse_initial_state(_get_required_mapping(data, "initial_state"))
    goal = _parse_goal_config(_get_required_mapping(data, "goal"))

    obstacles = tuple(
        _parse_obstacle_spec(raw_obstacle, index=index)
        for index, raw_obstacle in enumerate(_get_optional_sequence(data, "obstacles", default=()))
    )

    success = _parse_success_config(_get_required_mapping(data, "success"))
    termination = _parse_termination_config(_get_required_mapping(data, "termination"))

    runtime_overrides = None
    if "runtime_overrides" in data and data["runtime_overrides"] is not None:
        runtime_overrides = _parse_runtime_overrides(
            _get_required_mapping(data, "runtime_overrides")
        )

    raw_metadata = {}
    if "metadata" in data and data["metadata"] is not None:
        raw_metadata = dict(_get_required_mapping(data, "metadata"))

    return ScenarioConfig(
        schema_version=schema_version,
        scenario=scenario,
        world=world,
        initial_state=initial_state,
        goal=goal,
        obstacles=obstacles,
        success=success,
        termination=termination,
        runtime_overrides=runtime_overrides,
        raw_metadata=raw_metadata,
    )


def _parse_scenario_metadata(data: YamlDict) -> ScenarioMetadata:
    """Parse scenario metadata section."""
    tags_raw = data.get("tags", ())
    if tags_raw is None:
        tags = ()
    else:
        tags = tuple(_as_non_empty_string(tag, f"scenario.tags[{idx}]") for idx, tag in enumerate(_require_sequence(tags_raw, "scenario.tags")))

    description = data.get("description", None)
    if description is not None:
        description = str(description)

    return ScenarioMetadata(
        id=_as_non_empty_string(_get_required(data, "id"), "scenario.id"),
        name=_as_non_empty_string(_get_required(data, "name"), "scenario.name"),
        description=description,
        tags=tags,
    )


def _parse_world_config(data: YamlDict) -> WorldConfig:
    """Parse world section."""
    convention_raw = _as_non_empty_string(
        _get_required(data, "convention"),
        "world.convention",
    )

    try:
        convention = WorldConvention(convention_raw)
    except ValueError as exc:
        raise ConfigError(f"Unsupported world.convention: {convention_raw!r}.") from exc

    gravity = _as_float_tuple3(data.get("gravity", (0.0, 0.0, -9.81)), "world.gravity")

    bounds = None
    if "bounds" in data and data["bounds"] is not None:
        bounds_data = _get_required_mapping(data, "bounds")
        bounds = WorldBounds(
            x=_as_float_tuple2(_get_required(bounds_data, "x"), "world.bounds.x"),
            y=_as_float_tuple2(_get_required(bounds_data, "y"), "world.bounds.y"),
            z=_as_float_tuple2(_get_required(bounds_data, "z"), "world.bounds.z"),
        )

    return WorldConfig(
        frame=_as_non_empty_string(_get_required(data, "frame"), "world.frame"),
        convention=convention,
        gravity=gravity,
        bounds=bounds,
    )


def _parse_initial_state(data: YamlDict) -> np.ndarray:
    """Parse initial_state section."""
    return as_state9(_get_required(data, "state9"))


def _parse_goal_config(data: YamlDict) -> GoalConfig:
    """Parse goal section."""
    threshold = _as_positive_float(data.get("threshold", 0.4), "goal.threshold")

    return GoalConfig(
        position=as_goal3(_get_required(data, "position")),
        threshold=threshold,
    )


def _parse_obstacle_spec(raw_obstacle: object, *, index: int) -> ObstacleSpec:
    """Parse one obstacle spec."""
    name = f"obstacles[{index}]"
    data = _require_mapping(raw_obstacle, name)

    type_raw = _as_non_empty_string(data.get("type", ObstacleType.BOX_ELLIPSOID.value), f"{name}.type")
    try:
        obstacle_type = ObstacleType(type_raw)
    except ValueError as exc:
        raise ConfigError(f"Unsupported {name}.type: {type_raw!r}.") from exc

    motion_raw = _as_non_empty_string(
        data.get("motion_model", ObstacleMotionModel.CONSTANT_VELOCITY.value),
        f"{name}.motion_model",
    )
    try:
        motion_model = ObstacleMotionModel(motion_raw)
    except ValueError as exc:
        raise ConfigError(f"Unsupported {name}.motion_model: {motion_raw!r}.") from exc

    covariance = None
    if "covariance" in data and data["covariance"] is not None:
        covariance = _parse_obstacle_covariance(
            _get_required_mapping(data, "covariance"),
            name=f"{name}.covariance",
        )

    metadata = {}
    if "metadata" in data and data["metadata"] is not None:
        metadata = dict(_get_required_mapping(data, "metadata"))

    return ObstacleSpec(
        id=_as_non_empty_string(data.get("id", f"obs_{index + 1:03d}"), f"{name}.id"),
        type=obstacle_type,
        frame=_as_non_empty_string(data.get("frame", "W"), f"{name}.frame"),
        position=_as_float_tuple3(_get_required(data, "position"), f"{name}.position"),
        size=_as_float_tuple3(_get_required(data, "size"), f"{name}.size"),
        yaw=_as_float(data.get("yaw", 0.0), f"{name}.yaw"),
        velocity=_as_float_tuple3(data.get("velocity", (0.0, 0.0, 0.0)), f"{name}.velocity"),
        covariance=covariance,
        motion_model=motion_model,
        active=_as_bool(data.get("active", True), f"{name}.active"),
        appearance_time=_as_optional_non_negative_float(
            data.get("appearance_time"),
            f"{name}.appearance_time",
        ),
        disappearance_time=_as_optional_non_negative_float(
            data.get("disappearance_time"),
            f"{name}.disappearance_time",
        ),
        metadata=metadata,
    )


def _parse_obstacle_covariance(
    data: YamlDict,
    *,
    name: str,
) -> ObstacleCovarianceConfig:
    """Parse obstacle covariance section."""
    return ObstacleCovarianceConfig(
        position_std=_as_float_tuple3(
            data.get("position_std", (0.0, 0.0, 0.0)),
            f"{name}.position_std",
        ),
        velocity_std=_as_float_tuple3(
            data.get("velocity_std", (0.0, 0.0, 0.0)),
            f"{name}.velocity_std",
        ),
    )


def _parse_success_config(data: YamlDict) -> SuccessConfig:
    """Parse success section."""
    return SuccessConfig(
        goal_threshold=_as_positive_float(
            _get_required(data, "goal_threshold"),
            "success.goal_threshold",
        ),
        require_collision_free=_as_bool(
            data.get("require_collision_free", True),
            "success.require_collision_free",
        ),
        require_altitude_valid=_as_bool(
            data.get("require_altitude_valid", True),
            "success.require_altitude_valid",
        ),
        max_final_speed=_as_optional_non_negative_float(
            data.get("max_final_speed"),
            "success.max_final_speed",
        ),
    )


def _parse_termination_config(data: YamlDict) -> TerminationConfig:
    """Parse termination section."""
    return TerminationConfig(
        max_time=_as_positive_float(_get_required(data, "max_time"), "termination.max_time"),
        max_steps=_as_optional_positive_int(data.get("max_steps"), "termination.max_steps"),
        terminate_on_collision=_as_bool(
            data.get("terminate_on_collision", True),
            "termination.terminate_on_collision",
        ),
        terminate_on_altitude_violation=_as_bool(
            data.get("terminate_on_altitude_violation", True),
            "termination.terminate_on_altitude_violation",
        ),
        terminate_on_goal_reached=_as_bool(
            data.get("terminate_on_goal_reached", True),
            "termination.terminate_on_goal_reached",
        ),
        terminate_on_solver_failure=_as_bool(
            data.get("terminate_on_solver_failure", False),
            "termination.terminate_on_solver_failure",
        ),
    )


def _parse_runtime_overrides(data: YamlDict) -> RuntimeOverrides:
    """Parse runtime_overrides section."""
    return RuntimeOverrides(
        sim_dt=_as_optional_positive_float(data.get("sim_dt"), "runtime_overrides.sim_dt"),
        max_time=_as_optional_positive_float(data.get("max_time"), "runtime_overrides.max_time"),
        max_steps=_as_optional_positive_int(data.get("max_steps"), "runtime_overrides.max_steps"),
    )


def _scenario_id_from_path(path: PathLike | None) -> str:
    """Derive a scenario ID from a path stem."""
    if path is None:
        return "legacy_scenario"
    stem = Path(path).stem.strip()
    return stem if stem else "legacy_scenario"


def _import_yaml() -> Any:
    """Import PyYAML lazily and raise ConfigError if unavailable."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise ConfigError(
            "PyYAML is required to load scenario YAML files. "
            "Install it with `pip install pyyaml`."
        ) from exc

    return yaml


def _get_required(data: Mapping[str, Any], key: str) -> Any:
    """Get required mapping key."""
    if key not in data:
        raise ConfigError(f"Missing required config key: {key!r}.")
    return data[key]


def _get_required_mapping(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Get required key and require mapping value."""
    return dict(_require_mapping(_get_required(data, key), key))


def _get_optional_sequence(
    data: Mapping[str, Any],
    key: str,
    *,
    default: Sequence[Any],
) -> Sequence[Any]:
    """Get optional sequence from mapping."""
    if key not in data or data[key] is None:
        return default
    return _require_sequence(data[key], key)


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    """Require mapping/dict value."""
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping/dict, got {type(value).__name__}.")
    return value


def _require_sequence(value: object, name: str) -> Sequence[Any]:
    """Require non-string sequence value."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(f"{name} must be a sequence/list.")
    return value


def _as_non_empty_string(value: object, name: str) -> str:
    """Convert value to non-empty string."""
    if not isinstance(value, str):
        raise ConfigError(f"{name} must be a string.")
    if not value.strip():
        raise ConfigError(f"{name} must be a non-empty string.")
    return value


def _as_float(value: object, name: str) -> float:
    """Convert value to finite float."""
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a finite number, got bool.")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a finite number.") from exc

    if not np.isfinite(result):
        raise ConfigError(f"{name} must be finite.")

    return result


def _as_positive_float(value: object, name: str) -> float:
    """Convert value to finite float > 0."""
    result = _as_float(value, name)
    if result <= 0.0:
        raise ConfigError(f"{name} must be > 0.")
    return result


def _as_optional_positive_float(value: object, name: str) -> float | None:
    """Convert optional value to finite float > 0."""
    if value is None:
        return None
    return _as_positive_float(value, name)


def _as_optional_non_negative_float(value: object, name: str) -> float | None:
    """Convert optional value to finite float >= 0."""
    if value is None:
        return None

    result = _as_float(value, name)
    if result < 0.0:
        raise ConfigError(f"{name} must be >= 0.")
    return result


def _as_optional_positive_int(value: object, name: str) -> int | None:
    """Convert optional value to positive int."""
    if value is None:
        return None
    return _as_positive_int(value, name)


def _as_positive_int(value: object, name: str) -> int:
    """Convert value to positive int, rejecting bool."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ConfigError(f"{name} must be > 0.")
    return value


def _as_bool(value: object, name: str) -> bool:
    """Validate boolean value."""
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be bool.")
    return value


def _as_float_tuple2(value: object, name: str) -> tuple[float, float]:
    """Convert a sequence to a finite 2-float tuple."""
    seq = _require_sequence(value, name)
    if len(seq) != 2:
        raise ConfigError(f"{name} must have length 2, got {len(seq)}.")
    return (_as_float(seq[0], f"{name}[0]"), _as_float(seq[1], f"{name}[1]"))


def _as_float_tuple3(value: object, name: str) -> tuple[float, float, float]:
    """Convert a sequence to a finite 3-float tuple."""
    seq = _require_sequence(value, name)
    if len(seq) != 3:
        raise ConfigError(f"{name} must have length 3, got {len(seq)}.")
    return (
        _as_float(seq[0], f"{name}[0]"),
        _as_float(seq[1], f"{name}[1]"),
        _as_float(seq[2], f"{name}[2]"),
    )


__all__ = [
    "load_scenario_config",
    "parse_canonical_scenario",
    "parse_scenario_config",
    "read_yaml_mapping",
]
