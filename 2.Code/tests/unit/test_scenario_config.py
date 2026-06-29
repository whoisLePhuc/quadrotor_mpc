"""Unit tests for scenario config validation.

Target modules:
    simulation.config.schema
    simulation.config.validation

These tests verify the Phase 3 config-layer contract:
- canonical ScenarioConfig can be validated
- invalid State9/Goal3 shapes are rejected
- duplicate obstacle IDs are rejected
- invalid obstacle dimensions are rejected
- out-of-bounds goal is rejected
- invalid runtime override values are rejected
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

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


def make_valid_obstacle(
    *,
    obstacle_id: str = "obs_001",
    position: tuple[float, float, float] = (2.0, 0.0, 1.5),
    size: tuple[float, float, float] = (0.6, 0.6, 1.0),
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ObstacleSpec:
    """Create a valid obstacle for config validation tests."""
    return ObstacleSpec(
        id=obstacle_id,
        type=ObstacleType.BOX_ELLIPSOID,
        frame="W",
        position=position,
        size=size,
        yaw=0.0,
        velocity=velocity,
        covariance=ObstacleCovarianceConfig(
            position_std=(0.05, 0.05, 0.05),
            velocity_std=(0.02, 0.02, 0.02),
        ),
        motion_model=ObstacleMotionModel.CONSTANT_VELOCITY,
        active=True,
    )


def make_valid_scenario_config() -> ScenarioConfig:
    """Create a minimal valid canonical ScenarioConfig."""
    return ScenarioConfig(
        schema_version="1.0",
        scenario=ScenarioMetadata(
            id="empty_world",
            name="Empty world",
            description="Baseline scenario for config validation tests.",
            tags=("baseline", "unit-test"),
        ),
        world=WorldConfig(
            frame="W",
            convention=WorldConvention.Z_UP,
            gravity=(0.0, 0.0, -9.81),
            bounds=WorldBounds(
                x=(-10.0, 10.0),
                y=(-10.0, 10.0),
                z=(0.0, 6.0),
            ),
        ),
        initial_state=np.array(
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        ),
        goal=GoalConfig(
            position=np.array([5.0, 0.0, 2.0], dtype=np.float64),
            threshold=0.4,
        ),
        obstacles=(make_valid_obstacle(),),
        success=SuccessConfig(
            goal_threshold=0.4,
            require_collision_free=True,
            require_altitude_valid=True,
        ),
        termination=TerminationConfig(
            max_time=30.0,
            max_steps=3000,
            terminate_on_collision=True,
            terminate_on_altitude_violation=True,
            terminate_on_goal_reached=True,
            terminate_on_solver_failure=False,
        ),
        runtime_overrides=RuntimeOverrides(
            sim_dt=0.02,
            max_time=None,
            max_steps=None,
        ),
    )


def test_valid_scenario_config() -> None:
    """A complete canonical ScenarioConfig should pass validation."""
    config = make_valid_scenario_config()

    validate_scenario_config(config)


def test_reject_invalid_initial_state_shape() -> None:
    """ScenarioConfig rejects initial_state that is not State9 shape (9,)."""
    config = make_valid_scenario_config()
    invalid_config = replace(
        config,
        initial_state=np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )

    with pytest.raises((ConfigError, ValueError), match="State9"):
        validate_scenario_config(invalid_config)


def test_reject_invalid_goal_shape() -> None:
    """ScenarioConfig rejects goal.position that is not Goal3 shape (3,)."""
    config = make_valid_scenario_config()
    invalid_goal = replace(
        config.goal,
        position=np.array([5.0, 0.0], dtype=np.float64),
    )
    invalid_config = replace(config, goal=invalid_goal)

    with pytest.raises((ConfigError, ValueError), match="Goal3"):
        validate_scenario_config(invalid_config)


def test_reject_duplicate_obstacle_id() -> None:
    """ScenarioConfig rejects duplicate obstacle IDs."""
    config = make_valid_scenario_config()
    obstacle_a = make_valid_obstacle(obstacle_id="duplicate_obs")
    obstacle_b = make_valid_obstacle(
        obstacle_id="duplicate_obs",
        position=(3.0, 1.0, 1.5),
    )
    invalid_config = replace(config, obstacles=(obstacle_a, obstacle_b))

    with pytest.raises(ConfigError, match="Duplicate obstacle id"):
        validate_scenario_config(invalid_config)


def test_reject_negative_obstacle_size() -> None:
    """ScenarioConfig rejects obstacle dimensions that are <= 0."""
    config = make_valid_scenario_config()
    invalid_obstacle = replace(
        config.obstacles[0],
        size=(0.6, -0.2, 1.0),
    )
    invalid_config = replace(config, obstacles=(invalid_obstacle,))

    with pytest.raises(ConfigError, match="size"):
        validate_scenario_config(invalid_config)


def test_reject_goal_outside_world_bounds() -> None:
    """ScenarioConfig rejects goal position outside world bounds."""
    config = make_valid_scenario_config()
    invalid_goal = replace(
        config.goal,
        position=np.array([50.0, 0.0, 2.0], dtype=np.float64),
    )
    invalid_config = replace(config, goal=invalid_goal)

    with pytest.raises(ConfigError, match="outside world bounds"):
        validate_scenario_config(invalid_config)


def test_reject_invalid_sim_dt() -> None:
    """ScenarioConfig rejects non-positive runtime_overrides.sim_dt."""
    config = make_valid_scenario_config()
    invalid_overrides = replace(config.runtime_overrides, sim_dt=0.0)
    invalid_config = replace(config, runtime_overrides=invalid_overrides)

    with pytest.raises(ConfigError, match="sim_dt"):
        validate_scenario_config(invalid_config)
