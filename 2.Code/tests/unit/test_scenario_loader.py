"""Unit tests for scenario config loader.

Target module:
    simulation.config.loader

These tests verify that loader.py can:
- load canonical scenario YAML
- load legacy flat scenario YAML
- reject missing/empty/malformed YAML files
- parse canonical dictionaries directly
- warn when parsing legacy scenario dictionaries
"""

from __future__ import annotations

import pytest

from simulation.config.loader import (
    load_scenario_config,
    parse_canonical_scenario,
    parse_scenario_config,
)
from simulation.config.schema import (
    ConfigError,
    ObstacleType,
    ScenarioConfig,
)


CANONICAL_SCENARIO_YAML = """
schema_version: "1.0"

scenario:
  id: "loader_test"
  name: "Loader test scenario"
  description: "Canonical scenario used by loader unit tests."
  tags: ["unit-test", "canonical"]

world:
  frame: "W"
  convention: "Z_UP"
  gravity: [0.0, 0.0, -9.81]
  bounds:
    x: [-10.0, 10.0]
    y: [-10.0, 10.0]
    z: [0.0, 6.0]

initial_state:
  state9: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

goal:
  position: [5.0, 0.0, 2.0]
  threshold: 0.4

obstacles:
  - id: "obs_001"
    type: "box_ellipsoid"
    frame: "W"
    position: [2.0, 0.0, 1.5]
    size: [0.6, 0.6, 1.0]
    yaw: 0.0
    velocity: [0.0, 0.0, 0.0]
    motion_model: "constant_velocity"
    active: true
    covariance:
      position_std: [0.05, 0.05, 0.05]
      velocity_std: [0.02, 0.02, 0.02]

success:
  goal_threshold: 0.4
  require_collision_free: true
  require_altitude_valid: true

termination:
  max_time: 30.0
  max_steps: 3000
  terminate_on_collision: true
  terminate_on_altitude_violation: true
  terminate_on_goal_reached: true
  terminate_on_solver_failure: false

runtime_overrides:
  sim_dt: 0.02
"""

LEGACY_SCENARIO_YAML = """
start: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
goal: [5.0, 0.0, 2.0]
target_altitude: 2.0
goal_threshold: 0.4
sim_timestep: 0.02
max_time: 30.0
obstacles:
  - position: [2.0, 0.0, 1.5]
    size: [0.6, 0.6, 1.0]
    yaw: 0.0
    velocity: [0.0, 0.0, 0.0]
"""


def test_load_canonical_scenario_yaml(tmp_path) -> None:
    """load_scenario_config loads and validates canonical YAML."""
    scenario_path = tmp_path / "canonical_scenario.yaml"
    scenario_path.write_text(CANONICAL_SCENARIO_YAML, encoding="utf-8")

    config = load_scenario_config(scenario_path)

    assert isinstance(config, ScenarioConfig)
    assert config.schema_version == "1.0"
    assert config.scenario.id == "loader_test"
    assert config.world.frame == "W"
    assert config.initial_state.shape == (9,)
    assert config.goal.position.shape == (3,)
    assert config.goal.threshold == pytest.approx(0.4)
    assert len(config.obstacles) == 1
    assert config.obstacles[0].id == "obs_001"
    assert config.obstacles[0].type is ObstacleType.BOX_ELLIPSOID
    assert config.runtime_overrides is not None
    assert config.runtime_overrides.sim_dt == pytest.approx(0.02)


def test_load_legacy_scenario_yaml(tmp_path) -> None:
    """load_scenario_config loads legacy flat YAML through the legacy adapter."""
    scenario_path = tmp_path / "legacy_scenario.yaml"
    scenario_path.write_text(LEGACY_SCENARIO_YAML, encoding="utf-8")

    with pytest.warns(UserWarning, match="legacy flat scenario"):
        config = load_scenario_config(scenario_path)

    assert isinstance(config, ScenarioConfig)
    assert config.schema_version == "legacy"
    assert config.scenario.id == "legacy_scenario"
    assert config.initial_state.shape == (9,)
    assert config.goal.position.shape == (3,)
    assert config.goal.threshold == pytest.approx(0.4)
    assert config.success.goal_threshold == pytest.approx(0.4)
    assert config.runtime_overrides is not None
    assert config.runtime_overrides.sim_dt == pytest.approx(0.02)
    assert config.raw_metadata["target_altitude"] == 2.0


def test_loader_rejects_missing_file(tmp_path) -> None:
    """load_scenario_config rejects paths that do not exist."""
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(ConfigError, match="does not exist"):
        load_scenario_config(missing_path)


def test_loader_rejects_empty_yaml(tmp_path) -> None:
    """load_scenario_config rejects empty YAML files."""
    scenario_path = tmp_path / "empty.yaml"
    scenario_path.write_text("", encoding="utf-8")

    with pytest.raises(ConfigError, match="empty"):
        load_scenario_config(scenario_path)


def test_loader_rejects_invalid_top_level_yaml(tmp_path) -> None:
    """load_scenario_config rejects YAML whose top-level object is not a mapping."""
    scenario_path = tmp_path / "invalid_top_level.yaml"
    scenario_path.write_text("- item_1\n- item_2\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="top-level"):
        load_scenario_config(scenario_path)


def test_parse_canonical_scenario() -> None:
    """parse_canonical_scenario parses a canonical dictionary directly."""
    data = {
        "schema_version": "1.0",
        "scenario": {
            "id": "direct_parse",
            "name": "Direct parse scenario",
            "tags": ["unit-test"],
        },
        "world": {
            "frame": "W",
            "convention": "Z_UP",
            "gravity": [0.0, 0.0, -9.81],
            "bounds": {
                "x": [-10.0, 10.0],
                "y": [-10.0, 10.0],
                "z": [0.0, 6.0],
            },
        },
        "initial_state": {
            "state9": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "goal": {
            "position": [5.0, 0.0, 2.0],
            "threshold": 0.4,
        },
        "obstacles": [],
        "success": {
            "goal_threshold": 0.4,
            "require_collision_free": True,
            "require_altitude_valid": True,
        },
        "termination": {
            "max_time": 30.0,
            "max_steps": 3000,
        },
        "runtime_overrides": {
            "sim_dt": 0.02,
        },
    }

    config = parse_canonical_scenario(data)

    assert config.schema_version == "1.0"
    assert config.scenario.id == "direct_parse"
    assert config.scenario.tags == ("unit-test",)
    assert config.initial_state.shape == (9,)
    assert config.goal.position.shape == (3,)
    assert config.obstacles == ()
    assert config.runtime_overrides is not None
    assert config.runtime_overrides.sim_dt == pytest.approx(0.02)


def test_parse_legacy_scenario_warns() -> None:
    """parse_scenario_config warns when parsing legacy flat scenario dictionaries."""
    data = {
        "start": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "goal": [5.0, 0.0, 2.0],
        "goal_threshold": 0.4,
        "sim_timestep": 0.02,
        "obstacles": [],
    }

    with pytest.warns(UserWarning, match="legacy flat scenario"):
        config = parse_scenario_config(data)

    assert config.schema_version == "legacy"
    assert config.initial_state.shape == (9,)
    assert config.goal.position.shape == (3,)
    assert config.runtime_overrides is not None
    assert config.runtime_overrides.sim_dt == pytest.approx(0.02)
