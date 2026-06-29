"""Unit tests for CC-MPC fallback controller.

Target module:
    ccmpc.controllers.fallback_controller

These tests verify the deterministic safe-control behavior used when the
optimization-based CC-MPC controller fails or returns an invalid solution.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ccmpc.controllers.fallback_controller import (
    FallbackConfig,
    FallbackConfigError,
    FallbackController,
    FallbackInputError,
    FallbackMode,
    FallbackResult,
    FallbackStatus,
    parse_fallback_mode,
    world_velocity_to_body_yaw,
    wrap_angle_pi,
)


def make_state(
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 1.0,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> np.ndarray:
    """Create canonical State9."""
    return np.array(
        [x, y, z, vx, vy, vz, roll, pitch, yaw],
        dtype=np.float64,
    )


def make_full_config() -> dict:
    """Create full project-style fallback config."""
    return {
        "controller": {
            "limits": {
                "max_roll": 0.2,
                "max_pitch": 0.3,
                "max_vert_vel": 1.5,
                "max_yaw_rate": 0.7,
                "min_altitude": 0.4,
            },
            "fallback": {
                "mode": "brake",
                "brake_gain": 0.2,
                "brake_deadband": 0.05,
                "land_descent_rate": 0.4,
                "goal_yaw_gain": 0.1,
            },
        }
    }


def make_fallback_only_config() -> dict:
    """Create fallback-only config."""
    return {
        "mode": "land",
        "max_roll": 0.25,
        "max_pitch": 0.35,
        "max_vertical_velocity": 1.2,
        "max_yaw_rate": 0.9,
        "brake_gain": 0.15,
        "brake_deadband": 0.04,
        "land_descent_rate": 0.25,
        "z_min": 0.5,
        "goal_yaw_gain": 0.2,
    }


def test_parse_fallback_mode_accepts_enum_and_strings() -> None:
    """parse_fallback_mode should accept enum, lowercase, uppercase, and dotted names."""
    assert parse_fallback_mode(FallbackMode.HOVER) is FallbackMode.HOVER
    assert parse_fallback_mode("hover") is FallbackMode.HOVER
    assert parse_fallback_mode("BRAKE") is FallbackMode.BRAKE
    assert parse_fallback_mode("FallbackMode.LAND") is FallbackMode.LAND


def test_parse_fallback_mode_rejects_invalid_value() -> None:
    """parse_fallback_mode should reject unsupported values."""
    with pytest.raises(FallbackConfigError, match="Invalid fallback mode"):
        parse_fallback_mode("unknown")


def test_fallback_config_defaults() -> None:
    """Default config should be valid and use hover mode."""
    config = FallbackConfig()

    assert config.mode is FallbackMode.HOVER
    assert config.max_roll > 0.0
    assert config.max_pitch > 0.0
    assert config.max_vertical_velocity > 0.0
    assert config.max_yaw_rate > 0.0
    assert config.land_descent_rate <= config.max_vertical_velocity


def test_fallback_config_from_none() -> None:
    """FallbackConfig.from_config(None) should return defaults."""
    config = FallbackConfig.from_config(None)

    assert config == FallbackConfig()


def test_fallback_config_from_fallback_only_config() -> None:
    """FallbackConfig.from_config should parse fallback-only dict."""
    config = FallbackConfig.from_config(make_fallback_only_config())

    assert config.mode is FallbackMode.LAND
    assert config.max_roll == pytest.approx(0.25)
    assert config.max_pitch == pytest.approx(0.35)
    assert config.max_vertical_velocity == pytest.approx(1.2)
    assert config.max_yaw_rate == pytest.approx(0.9)
    assert config.brake_gain == pytest.approx(0.15)
    assert config.brake_deadband == pytest.approx(0.04)
    assert config.land_descent_rate == pytest.approx(0.25)
    assert config.z_min == pytest.approx(0.5)
    assert config.goal_yaw_gain == pytest.approx(0.2)


def test_fallback_config_from_full_config() -> None:
    """FallbackConfig.from_config should merge controller limits and fallback settings."""
    config = FallbackConfig.from_config(make_full_config())

    assert config.mode is FallbackMode.BRAKE
    assert config.max_roll == pytest.approx(0.2)
    assert config.max_pitch == pytest.approx(0.3)
    assert config.max_vertical_velocity == pytest.approx(1.5)
    assert config.max_yaw_rate == pytest.approx(0.7)
    assert config.z_min == pytest.approx(0.4)
    assert config.brake_gain == pytest.approx(0.2)
    assert config.land_descent_rate == pytest.approx(0.4)
    assert config.goal_yaw_gain == pytest.approx(0.1)


def test_fallback_config_rejects_bad_config_type() -> None:
    """FallbackConfig.from_config should reject non-dict config."""
    with pytest.raises(FallbackConfigError, match="dictionary"):
        FallbackConfig.from_config(["bad"])  # type: ignore[arg-type]


def test_fallback_config_rejects_negative_limit() -> None:
    """FallbackConfig should reject negative limits."""
    with pytest.raises(FallbackConfigError, match="max_roll"):
        FallbackConfig(max_roll=-0.1)


def test_fallback_config_rejects_zero_positive_limit() -> None:
    """FallbackConfig should require positive command limits."""
    with pytest.raises(FallbackConfigError, match="max_pitch"):
        FallbackConfig(max_pitch=0.0)


def test_fallback_config_rejects_land_rate_above_vertical_limit() -> None:
    """land_descent_rate should not exceed max_vertical_velocity."""
    with pytest.raises(FallbackConfigError, match="land_descent_rate"):
        FallbackConfig(max_vertical_velocity=0.2, land_descent_rate=0.3)


def test_fallback_controller_default() -> None:
    """FallbackController() should use default config."""
    controller = FallbackController()

    assert isinstance(controller.config, FallbackConfig)
    assert controller.config.mode is FallbackMode.HOVER


def test_fallback_controller_from_config() -> None:
    """FallbackController.from_config should construct controller from dict."""
    controller = FallbackController.from_config(make_full_config())

    assert controller.config.mode is FallbackMode.BRAKE
    assert controller.config.max_roll == pytest.approx(0.2)


def test_fallback_controller_rejects_invalid_config_object() -> None:
    """FallbackController should reject non-FallbackConfig object."""
    with pytest.raises(FallbackConfigError, match="FallbackConfig"):
        FallbackController(config=object())  # type: ignore[arg-type]


def test_fallback_result_valid() -> None:
    """FallbackResult should validate command and metadata."""
    result = FallbackResult(
        command=np.zeros(4, dtype=np.float64),
        mode="hover",
        status="success",
        reason="unit-test",
        clipped=False,
        horizontal_speed=0.1,
        altitude=1.0,
        metadata={"a": 1},
    )

    assert result.command.shape == (4,)
    assert result.mode is FallbackMode.HOVER
    assert result.status is FallbackStatus.SUCCESS
    assert result.reason == "unit-test"
    assert result.metadata == {"a": 1}


def test_fallback_result_rejects_bad_command() -> None:
    """FallbackResult should reject invalid ControlCommand4."""
    with pytest.raises(Exception):
        FallbackResult(
            command=np.zeros(3, dtype=np.float64),
            mode=FallbackMode.HOVER,
            status=FallbackStatus.SUCCESS,
            reason="bad",
        )


def test_fallback_result_rejects_negative_horizontal_speed() -> None:
    """FallbackResult should reject negative diagnostic speed."""
    with pytest.raises(FallbackConfigError, match="horizontal_speed"):
        FallbackResult(
            command=np.zeros(4, dtype=np.float64),
            mode=FallbackMode.HOVER,
            status=FallbackStatus.SUCCESS,
            reason="bad",
            horizontal_speed=-0.1,
        )


def test_hover_returns_zero_command() -> None:
    """HOVER mode should return zero command by default."""
    controller = FallbackController()
    state = make_state(z=1.0)

    result = controller.hover(state)

    assert result.mode is FallbackMode.HOVER
    assert result.status is FallbackStatus.SUCCESS
    assert result.clipped is False
    assert np.allclose(result.command, [0.0, 0.0, 0.0, 0.0])
    assert result.horizontal_speed == pytest.approx(0.0)
    assert result.altitude == pytest.approx(1.0)


def test_compute_uses_default_mode() -> None:
    """compute should use config.mode when mode override is None."""
    controller = FallbackController(FallbackConfig(mode=FallbackMode.LAND))
    state = make_state(z=2.0)

    result = controller.compute(state, reason="default-mode")

    assert result.mode is FallbackMode.LAND
    assert result.command[2] < 0.0
    assert result.reason == "default-mode"


def test_land_above_z_min_returns_negative_vz() -> None:
    """LAND mode should command descent when altitude is above z_min."""
    controller = FallbackController(
        FallbackConfig(
            mode=FallbackMode.LAND,
            z_min=0.5,
            land_descent_rate=0.3,
            max_vertical_velocity=1.0,
        )
    )
    state = make_state(z=1.2)

    result = controller.land(state)

    assert result.mode is FallbackMode.LAND
    assert result.command[0] == pytest.approx(0.0)
    assert result.command[1] == pytest.approx(0.0)
    assert result.command[2] == pytest.approx(-0.3)
    assert result.command[3] == pytest.approx(0.0)


def test_land_at_z_min_returns_hover_vz() -> None:
    """LAND mode should stop descending at or below z_min."""
    controller = FallbackController(
        FallbackConfig(
            z_min=0.5,
            land_descent_rate=0.3,
            max_vertical_velocity=1.0,
        )
    )

    at_min = controller.land(make_state(z=0.5))
    below_min = controller.land(make_state(z=0.4))

    assert at_min.command[2] == pytest.approx(0.0)
    assert below_min.command[2] == pytest.approx(0.0)


def test_brake_world_x_velocity_commands_negative_pitch() -> None:
    """Positive world x velocity at yaw=0 should command negative pitch."""
    controller = FallbackController(
        FallbackConfig(
            mode=FallbackMode.BRAKE,
            brake_gain=0.2,
            max_pitch=0.5,
        )
    )
    state = make_state(vx=1.0, vy=0.0, yaw=0.0)

    result = controller.brake(state)

    assert result.mode is FallbackMode.BRAKE
    assert result.command[0] == pytest.approx(0.0)
    assert result.command[1] == pytest.approx(-0.2)
    assert result.command[2] == pytest.approx(0.0)
    assert result.command[3] == pytest.approx(0.0)
    assert result.horizontal_speed == pytest.approx(1.0)


def test_brake_world_y_velocity_commands_positive_roll_at_yaw_zero() -> None:
    """Positive world y velocity at yaw=0 should command positive roll."""
    controller = FallbackController(
        FallbackConfig(
            mode=FallbackMode.BRAKE,
            brake_gain=0.2,
            max_roll=0.5,
        )
    )
    state = make_state(vx=0.0, vy=1.0, yaw=0.0)

    result = controller.brake(state)

    assert result.command[0] == pytest.approx(0.2)
    assert result.command[1] == pytest.approx(0.0)


def test_brake_respects_yaw_transform() -> None:
    """BRAKE should rotate world velocity into body-yaw frame."""
    controller = FallbackController(
        FallbackConfig(
            mode=FallbackMode.BRAKE,
            brake_gain=0.2,
            max_roll=0.5,
            max_pitch=0.5,
        )
    )
    state = make_state(vx=0.0, vy=1.0, yaw=math.pi / 2.0)

    result = controller.brake(state)

    # With yaw=pi/2 and world velocity +y, body forward velocity is +1.
    assert result.command[0] == pytest.approx(0.0, abs=1e-12)
    assert result.command[1] == pytest.approx(-0.2)


def test_brake_deadband_returns_hover() -> None:
    """BRAKE should fall back to hover if horizontal speed is too small."""
    controller = FallbackController(
        FallbackConfig(
            mode=FallbackMode.BRAKE,
            brake_gain=0.2,
            brake_deadband=0.1,
        )
    )
    state = make_state(vx=0.05, vy=0.0, yaw=0.0)

    result = controller.brake(state)

    assert np.allclose(result.command, [0.0, 0.0, 0.0, 0.0])
    assert result.mode is FallbackMode.BRAKE


def test_brake_clips_roll_pitch() -> None:
    """BRAKE command should be clipped to roll/pitch limits."""
    controller = FallbackController(
        FallbackConfig(
            mode=FallbackMode.BRAKE,
            brake_gain=10.0,
            max_roll=0.2,
            max_pitch=0.3,
        )
    )
    state = make_state(vx=1.0, vy=1.0, yaw=0.0)

    result = controller.brake(state)

    assert result.clipped is True
    assert result.status is FallbackStatus.CLIPPED
    assert result.command[0] == pytest.approx(0.2)
    assert result.command[1] == pytest.approx(-0.3)


def test_clip_command_limits_all_channels() -> None:
    """clip_command should limit phi, theta, vz, and psi_rate."""
    controller = FallbackController(
        FallbackConfig(
            max_roll=0.2,
            max_pitch=0.3,
            max_vertical_velocity=0.4,
            max_yaw_rate=0.5,
        )
    )

    clipped, was_clipped = controller.clip_command(
        np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float64)
    )

    assert was_clipped is True
    assert np.allclose(clipped, [0.2, -0.3, 0.4, -0.5])


def test_clip_command_no_clipping() -> None:
    """clip_command should report False when command is already within limits."""
    controller = FallbackController()
    command = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float64)

    clipped, was_clipped = controller.clip_command(command)

    assert was_clipped is False
    assert np.allclose(clipped, command)


def test_goal_yaw_rate_zero_when_gain_zero() -> None:
    """Goal should not affect yaw rate if goal_yaw_gain is zero."""
    controller = FallbackController(FallbackConfig(goal_yaw_gain=0.0))
    state = make_state(x=0.0, y=0.0, yaw=0.0)
    goal = np.array([0.0, 1.0, 1.0], dtype=np.float64)

    result = controller.hover(state, goal)

    assert result.command[3] == pytest.approx(0.0)


def test_goal_yaw_rate_points_toward_goal() -> None:
    """Positive goal_yaw_gain should command yaw rate toward goal bearing."""
    controller = FallbackController(
        FallbackConfig(
            goal_yaw_gain=0.5,
            max_yaw_rate=10.0,
        )
    )
    state = make_state(x=0.0, y=0.0, yaw=0.0)
    goal = np.array([0.0, 1.0, 1.0], dtype=np.float64)

    result = controller.hover(state, goal)

    assert result.command[3] == pytest.approx(0.5 * math.pi / 2.0)


def test_goal_yaw_rate_is_clipped() -> None:
    """Yaw-rate command from goal alignment should be clipped."""
    controller = FallbackController(
        FallbackConfig(
            goal_yaw_gain=10.0,
            max_yaw_rate=0.2,
        )
    )
    state = make_state(x=0.0, y=0.0, yaw=0.0)
    goal = np.array([0.0, 1.0, 1.0], dtype=np.float64)

    result = controller.hover(state, goal)

    assert result.clipped is True
    assert result.command[3] == pytest.approx(0.2)


def test_compute_rejects_invalid_state() -> None:
    """compute should reject non-State9 inputs."""
    controller = FallbackController()

    with pytest.raises(Exception):
        controller.compute(np.zeros(8, dtype=np.float64))


def test_compute_rejects_invalid_goal() -> None:
    """compute should reject non-Goal3 inputs."""
    controller = FallbackController()

    with pytest.raises(Exception):
        controller.compute(make_state(), goal=np.zeros(2, dtype=np.float64))


def test_compute_preserves_metadata() -> None:
    """compute should preserve diagnostic metadata."""
    controller = FallbackController()
    state = make_state(z=1.0)

    result = controller.compute(
        state,
        reason="solver_failed",
        metadata={"solver_status": "infeasible"},
    )

    assert result.reason == "solver_failed"
    assert result.metadata == {"solver_status": "infeasible"}


def test_world_velocity_to_body_yaw_identity() -> None:
    """At yaw=0, body velocity should equal world velocity."""
    vx_body, vy_body = world_velocity_to_body_yaw(
        np.array([1.0, 2.0], dtype=np.float64),
        yaw=0.0,
    )

    assert vx_body == pytest.approx(1.0)
    assert vy_body == pytest.approx(2.0)


def test_world_velocity_to_body_yaw_pi_over_two() -> None:
    """At yaw=pi/2, world +y should become body +x."""
    vx_body, vy_body = world_velocity_to_body_yaw(
        np.array([0.0, 1.0], dtype=np.float64),
        yaw=math.pi / 2.0,
    )

    assert vx_body == pytest.approx(1.0)
    assert vy_body == pytest.approx(0.0, abs=1e-12)


def test_world_velocity_to_body_yaw_rejects_bad_shape() -> None:
    """world_velocity_to_body_yaw should require shape (2,)."""
    with pytest.raises(FallbackInputError, match="shape"):
        world_velocity_to_body_yaw(np.zeros(3, dtype=np.float64), yaw=0.0)


def test_world_velocity_to_body_yaw_rejects_nan() -> None:
    """world_velocity_to_body_yaw should reject NaN values."""
    with pytest.raises(FallbackInputError, match="finite"):
        world_velocity_to_body_yaw(np.array([np.nan, 0.0]), yaw=0.0)


def test_wrap_angle_pi() -> None:
    """wrap_angle_pi should map angles to [-pi, pi)."""
    assert wrap_angle_pi(0.0) == pytest.approx(0.0)
    assert wrap_angle_pi(2.0 * math.pi) == pytest.approx(0.0)
    assert wrap_angle_pi(-2.0 * math.pi) == pytest.approx(0.0)
    assert wrap_angle_pi(3.0 * math.pi / 2.0) == pytest.approx(-math.pi / 2.0)
