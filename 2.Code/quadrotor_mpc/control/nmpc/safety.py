"""Safety acceptance gates and deterministic fallback control.

The primary NMPC controller is allowed to propose a command.  This supervisor
owns the final apply/reject decision so solver failures, late solutions and
soft-constraint violations cannot silently cross the controller boundary.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from quadrotor_mpc.core.contracts import (
    ControlGoal,
    Controller,
    ControlSolution,
    ObstacleBelief,
    VehicleBelief,
)
from quadrotor_mpc.core.vehicle import DEFAULT_QUADROTOR


def _finite_nonnegative(value: Any, label: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and >= 0")
    return number


def _finite_positive(value: Any, label: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and > 0")
    return number


@dataclass(frozen=True, slots=True)
class SafetyFallbackOptions:
    """Validated Stage 6 acceptance and fallback policy."""

    enabled: bool = False
    solve_deadline_s: float = 0.05
    reject_on_deadline_miss: bool = True
    guarantee_slack_tolerance_m: float = 1e-6
    maximum_acceptable_slack_m: float = 0.08
    constraint_tolerance_m: float = 1e-6
    maximum_solver_residual: float = 1e-3
    command_bound_tolerance: float = 1e-6
    hold_last_command_steps: int = 1
    emergency_after_consecutive_rejections: int = 20
    position_kp_xy: float = 2.0
    position_kd_xy: float = 1.8
    vertical_kp: float = 4.0
    vertical_kd: float = 2.4
    attitude_kp: float = 0.004
    attitude_kd: float = 0.0008
    emergency_rate_damping: float = 0.0006
    maximum_tilt_rad: float = 0.30

    def __post_init__(self) -> None:
        for label in (
            "solve_deadline_s",
            "guarantee_slack_tolerance_m",
            "maximum_acceptable_slack_m",
            "constraint_tolerance_m",
            "maximum_solver_residual",
            "command_bound_tolerance",
        ):
            object.__setattr__(
                self,
                label,
                _finite_nonnegative(
                    getattr(self, label),
                    f"controller.safety_fallback.{label}",
                ),
            )
        for label in (
            "position_kp_xy",
            "position_kd_xy",
            "vertical_kp",
            "vertical_kd",
            "attitude_kp",
            "attitude_kd",
            "emergency_rate_damping",
            "maximum_tilt_rad",
        ):
            object.__setattr__(
                self,
                label,
                _finite_positive(
                    getattr(self, label),
                    f"controller.safety_fallback.{label}",
                ),
            )
        hold_steps = int(self.hold_last_command_steps)
        emergency_steps = int(self.emergency_after_consecutive_rejections)
        if hold_steps < 0:
            raise ValueError(
                "controller.safety_fallback.hold_last_command_steps must be >= 0"
            )
        if emergency_steps < 1:
            raise ValueError(
                "controller.safety_fallback.emergency_after_consecutive_rejections "
                "must be >= 1"
            )
        if emergency_steps <= hold_steps:
            raise ValueError(
                "controller.safety_fallback.emergency_after_consecutive_rejections "
                "must exceed hold_last_command_steps"
            )
        if self.guarantee_slack_tolerance_m > self.maximum_acceptable_slack_m:
            raise ValueError(
                "controller.safety_fallback.guarantee_slack_tolerance_m must not "
                "exceed maximum_acceptable_slack_m"
            )
        object.__setattr__(self, "hold_last_command_steps", hold_steps)
        object.__setattr__(
            self,
            "emergency_after_consecutive_rejections",
            emergency_steps,
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SafetyFallbackOptions:
        return cls(
            enabled=bool(raw.get("enabled", False)),
            solve_deadline_s=raw.get("solve_deadline_s", 0.05),
            reject_on_deadline_miss=bool(
                raw.get("reject_on_deadline_miss", True)
            ),
            guarantee_slack_tolerance_m=raw.get(
                "guarantee_slack_tolerance_m",
                1e-6,
            ),
            maximum_acceptable_slack_m=raw.get(
                "maximum_acceptable_slack_m",
                0.08,
            ),
            constraint_tolerance_m=raw.get("constraint_tolerance_m", 1e-6),
            maximum_solver_residual=raw.get(
                "maximum_solver_residual",
                1e-3,
            ),
            command_bound_tolerance=raw.get("command_bound_tolerance", 1e-6),
            hold_last_command_steps=raw.get("hold_last_command_steps", 1),
            emergency_after_consecutive_rejections=raw.get(
                "emergency_after_consecutive_rejections",
                20,
            ),
            position_kp_xy=raw.get("position_kp_xy", 2.0),
            position_kd_xy=raw.get("position_kd_xy", 1.8),
            vertical_kp=raw.get("vertical_kp", 4.0),
            vertical_kd=raw.get("vertical_kd", 2.4),
            attitude_kp=raw.get("attitude_kp", 0.004),
            attitude_kd=raw.get("attitude_kd", 0.0008),
            emergency_rate_damping=raw.get(
                "emergency_rate_damping",
                0.0006,
            ),
            maximum_tilt_rad=raw.get("maximum_tilt_rad", 0.30),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


def _quaternion_to_euler(quaternion_wxyz: np.ndarray) -> tuple[float, float, float]:
    qw, qx, qy, qz = np.asarray(quaternion_wxyz, dtype=float)
    roll = math.atan2(
        2.0 * (qw * qx + qy * qz),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )
    pitch = math.asin(
        float(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    )
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return roll, pitch, yaw


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class SafeFallbackController:
    """Decorate any controller with Stage 6 acceptance gates and fallback."""

    def __init__(
        self,
        primary: Controller,
        *,
        options: SafetyFallbackOptions,
        bounds: Mapping[str, float],
    ):
        self.primary = primary
        self.options = options
        self._limits_lower, self._limits_upper = self._control_limits(bounds)
        self._last_accepted_command: np.ndarray | None = None
        self._fallback_position: np.ndarray | None = None
        self._fallback_yaw: float | None = None
        self._consecutive_rejections = 0

    @staticmethod
    def _control_limits(
        bounds: Mapping[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        thrust = _finite_positive(bounds["thrust"], "controller.bounds.thrust")
        torque_rp = _finite_positive(
            bounds["torque_rp"],
            "controller.bounds.torque_rp",
        )
        torque_yaw = _finite_positive(
            bounds["torque_yaw"],
            "controller.bounds.torque_yaw",
        )
        lower = np.array(
            [
                -min(thrust, DEFAULT_QUADROTOR.mass_kg * 9.81),
                -torque_rp,
                -torque_rp,
                -torque_yaw,
            ],
            dtype=float,
        )
        upper = np.array(
            [
                min(
                    thrust,
                    DEFAULT_QUADROTOR.max_upward_thrust_deviation_n,
                ),
                torque_rp,
                torque_rp,
                torque_yaw,
            ],
            dtype=float,
        )
        return lower, upper

    @property
    def horizon_steps(self) -> int:
        return int(self.primary.horizon_steps)

    def reset(self, belief: VehicleBelief) -> None:
        self.primary.reset(belief)
        self._last_accepted_command = None
        self._fallback_position = None
        self._fallback_yaw = None
        self._consecutive_rejections = 0

    def _rejection_reason(
        self,
        solution: ControlSolution,
        elapsed_s: float,
    ) -> str | None:
        if not solution.primary_solver_success:
            return "PRIMARY_SOLVER_FAILED"
        if max(
            solution.primary_solver_primal_residual,
            solution.primary_solver_dual_residual,
        ) > self.options.maximum_solver_residual:
            return "SOLVER_RESIDUAL_EXCEEDED"
        if (
            solution.risk_semantics == "joint"
            and solution.risk_budget_status != "BUDGET_OK"
        ):
            return "RISK_BUDGET_INVALID"
        tolerance = self.options.command_bound_tolerance
        if np.any(solution.command < self._limits_lower - tolerance) or np.any(
            solution.command > self._limits_upper + tolerance
        ):
            return "COMMAND_OUT_OF_BOUNDS"
        if np.any(solution.slacks < -self.options.constraint_tolerance_m):
            return "NEGATIVE_SLACK_INVALID"
        if solution.chance_margins.size:
            compensated = solution.chance_margins + solution.slacks
            if float(np.min(compensated)) < -self.options.constraint_tolerance_m:
                return "NONLINEAR_RESIDUAL_INVALID"
        maximum_slack = (
            0.0
            if not solution.slacks.size
            else float(np.max(solution.slacks))
        )
        if maximum_slack > self.options.maximum_acceptable_slack_m:
            return "SLACK_LIMIT_EXCEEDED"
        if (
            self.options.reject_on_deadline_miss
            and self.options.solve_deadline_s > 0.0
            and elapsed_s > self.options.solve_deadline_s
        ):
            return "DEADLINE_MISSED"
        return None

    def _latch_fallback_reference(self, belief: VehicleBelief) -> None:
        if self._fallback_position is None:
            self._fallback_position = belief.mean_state_13[:3].copy()
            self._fallback_yaw = _quaternion_to_euler(
                belief.mean_state_13[6:10]
            )[2]

    def _position_hold_command(self, belief: VehicleBelief) -> np.ndarray:
        self._latch_fallback_reference(belief)
        assert self._fallback_position is not None
        assert self._fallback_yaw is not None
        state = belief.mean_state_13
        position_error = self._fallback_position - state[:3]
        velocity = state[3:6]
        desired_acceleration_xy = (
            self.options.position_kp_xy * position_error[:2]
            - self.options.position_kd_xy * velocity[:2]
        )
        _, _, yaw = _quaternion_to_euler(state[6:10])
        desired_roll = (
            desired_acceleration_xy[0] * math.sin(yaw)
            - desired_acceleration_xy[1] * math.cos(yaw)
        ) / 9.81
        desired_pitch = (
            desired_acceleration_xy[0] * math.cos(yaw)
            + desired_acceleration_xy[1] * math.sin(yaw)
        ) / 9.81
        desired_roll = float(
            np.clip(
                desired_roll,
                -self.options.maximum_tilt_rad,
                self.options.maximum_tilt_rad,
            )
        )
        desired_pitch = float(
            np.clip(
                desired_pitch,
                -self.options.maximum_tilt_rad,
                self.options.maximum_tilt_rad,
            )
        )
        roll, pitch, current_yaw = _quaternion_to_euler(state[6:10])
        attitude_error = np.array(
            [
                desired_roll - roll,
                desired_pitch - pitch,
                _wrap_angle(self._fallback_yaw - current_yaw),
            ],
            dtype=float,
        )
        torques = (
            self.options.attitude_kp * attitude_error
            - self.options.attitude_kd * state[10:13]
        )
        thrust_deviation = DEFAULT_QUADROTOR.mass_kg * (
            self.options.vertical_kp * position_error[2]
            - self.options.vertical_kd * velocity[2]
        )
        command = np.concatenate(([thrust_deviation], torques))
        return np.clip(command, self._limits_lower, self._limits_upper)

    def _emergency_hover_command(self, belief: VehicleBelief) -> np.ndarray:
        command = np.zeros(4, dtype=float)
        command[1:] = (
            -self.options.emergency_rate_damping
            * belief.mean_state_13[10:13]
        )
        return np.clip(command, self._limits_lower, self._limits_upper)

    def _fallback_command(
        self,
        belief: VehicleBelief,
    ) -> tuple[np.ndarray, int, str]:
        if (
            self._last_accepted_command is not None
            and self._consecutive_rejections
            <= self.options.hold_last_command_steps
        ):
            return self._last_accepted_command.copy(), 1, "HOLD_LAST_ACCEPTED"
        if (
            self._consecutive_rejections
            < self.options.emergency_after_consecutive_rejections
        ):
            try:
                return self._position_hold_command(belief), 2, "POSITION_HOLD_PD"
            except (ArithmeticError, FloatingPointError, ValueError):
                pass
        return self._emergency_hover_command(belief), 3, "EMERGENCY_HOVER"

    def _empty_solution(
        self,
        belief: VehicleBelief,
        obstacles: Sequence[ObstacleBelief],
        primary_status: str,
    ) -> ControlSolution:
        steps = self.horizon_steps + 1
        obstacle_count = len(obstacles)
        nominal_states = np.repeat(
            belief.mean_state_13.reshape(1, 13),
            steps,
            axis=0,
        )
        vehicle_covariances = np.repeat(
            belief.error_covariance_12.reshape(1, 12, 12),
            steps,
            axis=0,
        )
        obstacle_covariances = np.zeros(
            (steps, obstacle_count, 6, 6),
            dtype=float,
        )
        for index, obstacle in enumerate(obstacles):
            obstacle_covariances[:, index] = obstacle.covariance_6
        matrix = np.zeros((steps, obstacle_count), dtype=float)
        return ControlSolution(
            command=np.zeros(4, dtype=float),
            nominal_states=nominal_states,
            predicted_covariances=vehicle_covariances,
            chance_margins=matrix,
            risk_allocations=matrix,
            slacks=matrix,
            solver_status="PRIMARY_EXCEPTION",
            predicted_obstacle_covariances=obstacle_covariances,
            projected_uncertainties=matrix,
            tightened_safety_radii=matrix,
            primary_solver_status=primary_status,
            primary_solver_success=False,
            risk_budget_status="FALLBACK_NO_PRIMARY_SOLUTION",
        )

    def solve(
        self,
        belief: VehicleBelief,
        obstacles: Sequence[ObstacleBelief],
        goal: ControlGoal,
        time_s: float,
    ) -> ControlSolution:
        started = time.perf_counter()
        try:
            primary_solution = self.primary.solve(
                belief,
                obstacles,
                goal,
                time_s,
            )
        # The supervisor is the containment boundary for arbitrary controller
        # backends, so backend exceptions must become fallback decisions.
        except Exception as exc:  # noqa: BLE001
            elapsed_s = time.perf_counter() - started
            primary_status = f"EXCEPTION:{type(exc).__name__}:{exc}"
            primary_solution = self._empty_solution(
                belief,
                obstacles,
                primary_status,
            )
            rejection_reason = "PRIMARY_SOLVE_EXCEPTION"
        else:
            elapsed_s = time.perf_counter() - started
            rejection_reason = (
                self._rejection_reason(primary_solution, elapsed_s)
                if self.options.enabled
                else None
            )

        deadline_missed = (
            self.options.solve_deadline_s > 0.0
            and elapsed_s > self.options.solve_deadline_s
        )
        if not self.options.enabled:
            return replace(
                primary_solution,
                solve_time_ms=elapsed_s * 1000.0,
                deadline_missed=deadline_missed,
            )

        if rejection_reason is None:
            self._last_accepted_command = primary_solution.command.copy()
            self._fallback_position = None
            self._fallback_yaw = None
            self._consecutive_rejections = 0
            maximum_slack = (
                0.0
                if not primary_solution.slacks.size
                else float(np.max(primary_solution.slacks))
            )
            if deadline_missed:
                assurance = "NOT_GUARANTEED_DEADLINE_MISS"
            elif maximum_slack <= self.options.guarantee_slack_tolerance_m:
                assurance = "GUARANTEE_ELIGIBLE"
            else:
                assurance = "NOT_GUARANTEED_POSITIVE_SLACK"
            return replace(
                primary_solution,
                command=np.clip(
                    primary_solution.command,
                    self._limits_lower,
                    self._limits_upper,
                ),
                command_source="PRIMARY_NMPC",
                solution_accepted=True,
                fallback_active=False,
                fallback_level=0,
                fallback_reason="",
                consecutive_rejections=0,
                solve_time_ms=elapsed_s * 1000.0,
                deadline_missed=deadline_missed,
                safety_assurance_status=assurance,
            )

        self._consecutive_rejections += 1
        self._latch_fallback_reference(belief)
        fallback_command, fallback_level, command_source = self._fallback_command(
            belief
        )
        return replace(
            primary_solution,
            command=fallback_command,
            solver_status=f"FALLBACK_{command_source}",
            command_source=command_source,
            solution_accepted=False,
            fallback_active=True,
            fallback_level=fallback_level,
            fallback_reason=rejection_reason,
            consecutive_rejections=self._consecutive_rejections,
            solve_time_ms=elapsed_s * 1000.0,
            deadline_missed=deadline_missed,
            safety_assurance_status="NOT_GUARANTEED_FALLBACK_ACTIVE",
        )
