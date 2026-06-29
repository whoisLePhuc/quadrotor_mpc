"""Fallback controller for quadrotor CC-MPC.

This module provides deterministic safe commands when the optimization-based
controller cannot produce a valid command.

Canonical state
---------------
State9 = [x, y, z, vx, vy, vz, roll, pitch, yaw]

Canonical control command
-------------------------
ControlCommand4 = [phi_cmd, theta_cmd, vz_cmd, psi_rate_cmd]

The fallback controller is intentionally independent from CVXPY, obstacle
managers, physics engines, and simulation runtime code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from ccmpc.types import (
    FloatArray,
    as_control_command4,
    as_goal3,
    as_state9,
)


class FallbackError(ValueError):
    """Base exception raised by fallback controller."""


class FallbackConfigError(FallbackError):
    """Raised when fallback configuration is invalid."""


class FallbackInputError(FallbackError):
    """Raised when fallback inputs are invalid."""


class FallbackMode(str, Enum):
    """Supported fallback control modes."""

    HOVER = "hover"
    BRAKE = "brake"
    LAND = "land"


class FallbackStatus(str, Enum):
    """Fallback result status."""

    SUCCESS = "success"
    CLIPPED = "clipped"
    NOT_NEEDED = "not_needed"


@dataclass(frozen=True)
class FallbackConfig:
    """Configuration for fallback control.

    Parameters
    ----------
    mode:
        Default fallback mode.
    max_roll:
        Absolute roll command limit in rad.
    max_pitch:
        Absolute pitch command limit in rad.
    max_vertical_velocity:
        Absolute vertical velocity command limit.
    max_yaw_rate:
        Absolute yaw-rate command limit.
    brake_gain:
        Horizontal velocity braking gain in rad per m/s before clipping.
    brake_deadband:
        Horizontal speed below this value produces hover instead of brake.
    land_descent_rate:
        Positive magnitude for descent speed.  LAND command uses
        ``vz_cmd = -land_descent_rate`` when above ``z_min``.
    z_min:
        Minimum safe altitude.  LAND mode switches to hover at or below this
        altitude.
    goal_yaw_gain:
        Optional yaw alignment gain used when goal is provided.
    """

    mode: FallbackMode = FallbackMode.HOVER
    max_roll: float = 0.25
    max_pitch: float = 0.25
    max_vertical_velocity: float = 1.0
    max_yaw_rate: float = 0.8
    brake_gain: float = 0.15
    brake_deadband: float = 0.05
    land_descent_rate: float = 0.3
    z_min: float = 0.5
    goal_yaw_gain: float = 0.0

    def __post_init__(self) -> None:
        """Validate numeric config."""
        object.__setattr__(self, "mode", parse_fallback_mode(self.mode))

        _require_positive(self.max_roll, "max_roll")
        _require_positive(self.max_pitch, "max_pitch")
        _require_positive(self.max_vertical_velocity, "max_vertical_velocity")
        _require_positive(self.max_yaw_rate, "max_yaw_rate")
        _require_non_negative(self.brake_gain, "brake_gain")
        _require_non_negative(self.brake_deadband, "brake_deadband")
        _require_non_negative(self.land_descent_rate, "land_descent_rate")
        _require_non_negative(self.z_min, "z_min")
        _require_non_negative(self.goal_yaw_gain, "goal_yaw_gain")

        if self.land_descent_rate > self.max_vertical_velocity:
            raise FallbackConfigError(
                "land_descent_rate must be <= max_vertical_velocity."
            )

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "FallbackConfig":
        """Create fallback config from full project or fallback-only config.

        Supported fallback-only shape:

            {
                "mode": "hover",
                "max_roll": 0.25,
                ...
            }

        Supported full project shape:

            {
                "controller": {
                    "limits": {...},
                    "fallback": {...}
                }
            }

        If no config is provided, defaults are used.
        """
        if config is None:
            return cls()

        if not isinstance(config, dict):
            raise FallbackConfigError("config must be a dictionary or None.")

        if "controller" in config:
            controller_cfg = config.get("controller")
            if not isinstance(controller_cfg, dict):
                raise FallbackConfigError("config['controller'] must be a dictionary.")

            fallback_cfg = controller_cfg.get("fallback", {})
            if fallback_cfg is None:
                fallback_cfg = {}
            if not isinstance(fallback_cfg, dict):
                raise FallbackConfigError(
                    "config['controller']['fallback'] must be a dictionary."
                )

            limits_cfg = controller_cfg.get("limits", {})
            if limits_cfg is None:
                limits_cfg = {}
            if not isinstance(limits_cfg, dict):
                raise FallbackConfigError(
                    "config['controller']['limits'] must be a dictionary."
                )

            merged = {
                "max_roll": limits_cfg.get("max_roll"),
                "max_pitch": limits_cfg.get("max_pitch"),
                "max_vertical_velocity": limits_cfg.get(
                    "max_vert_vel",
                    limits_cfg.get("max_vertical_velocity"),
                ),
                "max_yaw_rate": limits_cfg.get("max_yaw_rate"),
                "z_min": limits_cfg.get("min_altitude", limits_cfg.get("z_min")),
            }
            merged = {key: value for key, value in merged.items() if value is not None}
            merged.update(fallback_cfg)
            data = merged
        else:
            data = dict(config)

        return cls(
            mode=parse_fallback_mode(data.get("mode", FallbackMode.HOVER)),
            max_roll=float(data.get("max_roll", 0.25)),
            max_pitch=float(data.get("max_pitch", 0.25)),
            max_vertical_velocity=float(
                data.get("max_vertical_velocity", data.get("max_vert_vel", 1.0))
            ),
            max_yaw_rate=float(data.get("max_yaw_rate", 0.8)),
            brake_gain=float(data.get("brake_gain", 0.15)),
            brake_deadband=float(data.get("brake_deadband", 0.05)),
            land_descent_rate=float(data.get("land_descent_rate", 0.3)),
            z_min=float(data.get("z_min", data.get("min_altitude", 0.5))),
            goal_yaw_gain=float(data.get("goal_yaw_gain", 0.0)),
        )


@dataclass(frozen=True)
class FallbackResult:
    """Result produced by FallbackController."""

    command: FloatArray
    mode: FallbackMode
    status: FallbackStatus
    reason: str
    clipped: bool = False
    horizontal_speed: float = 0.0
    altitude: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result."""
        object.__setattr__(self, "command", as_control_command4(self.command))
        object.__setattr__(self, "mode", parse_fallback_mode(self.mode))

        if isinstance(self.status, str):
            object.__setattr__(self, "status", FallbackStatus(self.status))

        if not isinstance(self.reason, str):
            raise FallbackInputError("reason must be a string.")

        if not isinstance(self.clipped, bool):
            raise FallbackInputError("clipped must be bool.")

        object.__setattr__(
            self,
            "horizontal_speed",
            _as_non_negative_float(self.horizontal_speed, "horizontal_speed"),
        )
        object.__setattr__(
            self,
            "altitude",
            _as_non_negative_float(self.altitude, "altitude"),
        )

        if not isinstance(self.metadata, dict):
            raise FallbackInputError("metadata must be a dictionary.")
        object.__setattr__(self, "metadata", dict(self.metadata))


class FallbackController:
    """Deterministic safety fallback for CC-MPC.

    Modes
    -----
    HOVER:
        Command zero roll, pitch, vertical velocity, and yaw rate.

    BRAKE:
        Command roll/pitch opposite the horizontal velocity direction.  The
        velocity is rotated into the yaw-only body frame before command
        generation.

    LAND:
        Command slow descent until altitude reaches ``z_min``.  At or below
        ``z_min``, it switches to hover.
    """

    def __init__(self, config: FallbackConfig | None = None) -> None:
        self.config = FallbackConfig() if config is None else config

        if not isinstance(self.config, FallbackConfig):
            raise FallbackConfigError(
                "FallbackController expects FallbackConfig or None."
            )

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "FallbackController":
        """Create controller from full project or fallback-only config."""
        return cls(FallbackConfig.from_config(config))

    def compute(
        self,
        estimated_state: FloatArray,
        goal: FloatArray | None = None,
        *,
        mode: FallbackMode | str | None = None,
        reason: str = "fallback",
        metadata: dict[str, Any] | None = None,
    ) -> FallbackResult:
        """Compute fallback command.

        Parameters
        ----------
        estimated_state:
            Current estimated State9.
        goal:
            Optional Goal3.  When ``goal_yaw_gain > 0``, yaw-rate may align
            toward the goal direction.
        mode:
            Optional per-call mode override.
        reason:
            Human-readable reason for fallback activation.
        metadata:
            Optional extra diagnostics.
        """
        state = as_state9(estimated_state)
        fallback_mode = self.config.mode if mode is None else parse_fallback_mode(mode)

        if goal is None:
            goal_array = None
        else:
            goal_array = as_goal3(goal)

        if fallback_mode is FallbackMode.HOVER:
            raw_command = self._hover_command(state, goal_array)
        elif fallback_mode is FallbackMode.BRAKE:
            raw_command = self._brake_command(state, goal_array)
        elif fallback_mode is FallbackMode.LAND:
            raw_command = self._land_command(state, goal_array)
        else:  # pragma: no cover - parse_fallback_mode prevents this.
            raise FallbackInputError(f"Unsupported fallback mode: {fallback_mode}")

        command, clipped = self.clip_command(raw_command)

        horizontal_speed = float(np.linalg.norm(state[3:5]))
        altitude = float(state[2])
        status = FallbackStatus.CLIPPED if clipped else FallbackStatus.SUCCESS

        return FallbackResult(
            command=command,
            mode=fallback_mode,
            status=status,
            reason=reason,
            clipped=clipped,
            horizontal_speed=horizontal_speed,
            altitude=altitude,
            metadata={} if metadata is None else dict(metadata),
        )

    def hover(
        self,
        estimated_state: FloatArray,
        goal: FloatArray | None = None,
        *,
        reason: str = "hover",
    ) -> FallbackResult:
        """Compute HOVER fallback command."""
        return self.compute(
            estimated_state,
            goal,
            mode=FallbackMode.HOVER,
            reason=reason,
        )

    def brake(
        self,
        estimated_state: FloatArray,
        goal: FloatArray | None = None,
        *,
        reason: str = "brake",
    ) -> FallbackResult:
        """Compute BRAKE fallback command."""
        return self.compute(
            estimated_state,
            goal,
            mode=FallbackMode.BRAKE,
            reason=reason,
        )

    def land(
        self,
        estimated_state: FloatArray,
        goal: FloatArray | None = None,
        *,
        reason: str = "land",
    ) -> FallbackResult:
        """Compute LAND fallback command."""
        return self.compute(
            estimated_state,
            goal,
            mode=FallbackMode.LAND,
            reason=reason,
        )

    def clip_command(self, command: FloatArray) -> tuple[FloatArray, bool]:
        """Clip command to configured safety limits."""
        raw = as_control_command4(command)

        clipped = np.array(
            [
                np.clip(raw[0], -self.config.max_roll, self.config.max_roll),
                np.clip(raw[1], -self.config.max_pitch, self.config.max_pitch),
                np.clip(
                    raw[2],
                    -self.config.max_vertical_velocity,
                    self.config.max_vertical_velocity,
                ),
                np.clip(raw[3], -self.config.max_yaw_rate, self.config.max_yaw_rate),
            ],
            dtype=np.float64,
        )

        was_clipped = bool(not np.allclose(raw, clipped, atol=0.0, rtol=0.0))
        return as_control_command4(clipped), was_clipped

    def _hover_command(
        self,
        state: FloatArray,
        goal: FloatArray | None,
    ) -> FloatArray:
        """Return hover command."""
        return make_fallback_command4(
            0.0,
            0.0,
            0.0,
            self._goal_yaw_rate(state, goal),
        )

    def _brake_command(
        self,
        state: FloatArray,
        goal: FloatArray | None,
    ) -> FloatArray:
        """Return horizontal braking command.

        Body-frame velocity convention:
            x_body forward
            y_body left

        Approximate command convention used here:
            positive theta accelerates +x_body
            positive phi accelerates -y_body

        Therefore:
            theta_cmd = -gain * vx_body
            phi_cmd   =  gain * vy_body
        """
        vx_world = float(state[3])
        vy_world = float(state[4])
        yaw = float(state[8])

        vx_body, vy_body = world_velocity_to_body_yaw(
            np.array([vx_world, vy_world], dtype=np.float64),
            yaw,
        )

        horizontal_speed = float(np.hypot(vx_body, vy_body))
        if horizontal_speed <= self.config.brake_deadband:
            return self._hover_command(state, goal)

        phi_cmd = self.config.brake_gain * vy_body
        theta_cmd = -self.config.brake_gain * vx_body

        return make_fallback_command4(
            phi_cmd,
            theta_cmd,
            0.0,
            self._goal_yaw_rate(state, goal),
        )

    def _land_command(
        self,
        state: FloatArray,
        goal: FloatArray | None,
    ) -> FloatArray:
        """Return slow descent command, or hover if already at z_min."""
        altitude = float(state[2])

        if altitude <= self.config.z_min:
            vz_cmd = 0.0
        else:
            vz_cmd = -self.config.land_descent_rate

        return make_fallback_command4(
            0.0,
            0.0,
            vz_cmd,
            self._goal_yaw_rate(state, goal),
        )

    def _goal_yaw_rate(
        self,
        state: FloatArray,
        goal: FloatArray | None,
    ) -> float:
        """Optional yaw-rate command toward goal direction."""
        if goal is None or self.config.goal_yaw_gain <= 0.0:
            return 0.0

        dx = float(goal[0] - state[0])
        dy = float(goal[1] - state[1])

        if np.hypot(dx, dy) <= 1e-9:
            return 0.0

        desired_yaw = float(np.arctan2(dy, dx))
        yaw_error = wrap_angle_pi(desired_yaw - float(state[8]))
        return self.config.goal_yaw_gain * yaw_error



def make_fallback_command4(
    phi_cmd: float,
    theta_cmd: float,
    vz_cmd: float,
    psi_rate_cmd: float,
) -> FloatArray:
    """Build validated ControlCommand4 without relying on keyword API.

    Canonical order:
        [phi_cmd, theta_cmd, vz_cmd, psi_rate_cmd]
    """
    return as_control_command4(
        np.array(
            [phi_cmd, theta_cmd, vz_cmd, psi_rate_cmd],
            dtype=np.float64,
        )
    )

def parse_fallback_mode(value: FallbackMode | str) -> FallbackMode:
    """Parse fallback mode from enum or string."""
    if isinstance(value, FallbackMode):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if "." in normalized:
            normalized = normalized.split(".")[-1]

        try:
            return FallbackMode(normalized)
        except ValueError as exc:
            valid = ", ".join(item.value for item in FallbackMode)
            raise FallbackConfigError(
                f"Invalid fallback mode {value!r}. Valid values: {valid}."
            ) from exc

    raise FallbackConfigError(
        f"Fallback mode must be FallbackMode or str, got {type(value).__name__}."
    )


def world_velocity_to_body_yaw(velocity_xy_world: FloatArray, yaw: float) -> tuple[float, float]:
    """Rotate world-frame horizontal velocity into yaw-only body frame.

    Body convention:
        x forward
        y left

    Formula:
        vx_body =  cos(yaw) * vx_world + sin(yaw) * vy_world
        vy_body = -sin(yaw) * vx_world + cos(yaw) * vy_world
    """
    velocity = np.asarray(velocity_xy_world, dtype=np.float64)

    if velocity.shape != (2,):
        raise FallbackInputError(
            f"velocity_xy_world must have shape (2,), got {velocity.shape}."
        )

    if not np.all(np.isfinite(velocity)):
        raise FallbackInputError("velocity_xy_world must contain only finite values.")

    yaw_value = _as_finite_float(yaw, "yaw")
    ct = float(np.cos(yaw_value))
    st = float(np.sin(yaw_value))

    vx_body = ct * float(velocity[0]) + st * float(velocity[1])
    vy_body = -st * float(velocity[0]) + ct * float(velocity[1])

    return vx_body, vy_body


def wrap_angle_pi(angle: float) -> float:
    """Wrap angle to interval [-pi, pi)."""
    angle_value = _as_finite_float(angle, "angle")
    return float((angle_value + np.pi) % (2.0 * np.pi) - np.pi)


def _as_finite_float(value: float, name: str) -> float:
    """Validate finite scalar float."""
    if isinstance(value, bool):
        raise FallbackConfigError(f"{name} must be a finite scalar, got bool.")

    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise FallbackConfigError(f"{name} must be a finite scalar.") from exc

    if not np.isfinite(scalar):
        raise FallbackConfigError(f"{name} must be finite.")

    return scalar


def _as_non_negative_float(value: float, name: str) -> float:
    """Validate finite scalar >= 0."""
    scalar = _as_finite_float(value, name)

    if scalar < 0.0:
        raise FallbackConfigError(f"{name} must be >= 0.")

    return scalar


def _require_non_negative(value: float, name: str) -> None:
    """Require finite scalar >= 0."""
    _as_non_negative_float(value, name)


def _require_positive(value: float, name: str) -> None:
    """Require finite scalar > 0."""
    scalar = _as_finite_float(value, name)

    if scalar <= 0.0:
        raise FallbackConfigError(f"{name} must be > 0.")


__all__ = [
    "FallbackConfig",
    "FallbackConfigError",
    "FallbackController",
    "FallbackError",
    "FallbackInputError",
    "FallbackMode",
    "FallbackResult",
    "FallbackStatus",
    "make_fallback_command4",
    "parse_fallback_mode",
    "world_velocity_to_body_yaw",
    "wrap_angle_pi",
]
