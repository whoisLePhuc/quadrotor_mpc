from __future__ import annotations

import importlib.util
import time
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from controller_interface import (
    ControlGoal,
    ControlSolution,
    ObstacleBelief,
    SphericalObstacle,
    VehicleBelief,
)
from mujoco_native import load_native_mujoco_config
from native_safety_fallback import (
    SafeFallbackController,
    SafetyFallbackOptions,
)

CODE_ROOT = Path(__file__).resolve().parents[1]
BOUNDS = {
    "thrust": 0.085,
    "torque_rp": 0.002,
    "torque_yaw": 0.0004,
}


def vehicle_belief(
    *,
    position=(0.0, 0.0, 1.0),
    velocity=(0.0, 0.0, 0.0),
    body_rate=(0.0, 0.0, 0.0),
) -> VehicleBelief:
    return VehicleBelief(
        mean_state_13=np.array(
            [
                *position,
                *velocity,
                1.0,
                0.0,
                0.0,
                0.0,
                *body_rate,
            ],
            dtype=float,
        ),
        error_covariance_12=np.eye(12) * 1e-3,
    )


def obstacle_belief() -> ObstacleBelief:
    return ObstacleBelief(
        mean_state_6=np.array([2.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        covariance_6=np.eye(6) * 1e-3,
        shape=SphericalObstacle(0.2),
    )


def solution(
    *,
    command=(0.01, 0.0, 0.0, 0.0),
    slack=0.0,
    primary_success=True,
    risk_status="BUDGET_OK",
) -> ControlSolution:
    steps = 3
    nominal = np.repeat(
        vehicle_belief().mean_state_13.reshape(1, 13),
        steps,
        axis=0,
    )
    slacks = np.full((steps, 1), slack, dtype=float)
    margins = -slacks
    return ControlSolution(
        command=np.asarray(command, dtype=float),
        nominal_states=nominal,
        predicted_covariances=np.repeat(
            (np.eye(12) * 1e-3)[None, :, :],
            steps,
            axis=0,
        ),
        predicted_obstacle_covariances=np.repeat(
            (np.eye(6) * 1e-3)[None, None, :, :],
            steps,
            axis=0,
        ),
        chance_margins=margins,
        risk_allocations=np.full((steps, 1), 0.1 / steps),
        slacks=slacks,
        projected_uncertainties=np.full((steps, 1), 0.1),
        tightened_safety_radii=np.full((steps, 1), 0.8),
        solver_status="SOLVED_SAFE" if slack == 0.0 else "SOLVED_WITH_SLACK",
        risk_semantics="joint",
        risk_allocation_method="uniform",
        risk_budget_total=0.1,
        risk_budget_allocated=0.1,
        risk_budget_remaining=0.0,
        risk_constraint_count=steps,
        risk_budget_status=risk_status,
        primary_solver_status="Solve_Succeeded",
        primary_solver_success=primary_success,
    )


class ScriptedController:
    horizon_steps = 2

    def __init__(self, script, delay_s: float = 0.0):
        self.script = list(script)
        self.delay_s = float(delay_s)
        self.reset_count = 0

    def reset(self, _belief):
        self.reset_count += 1

    def solve(self, _belief, _obstacles, _goal, _time_s):
        if self.delay_s:
            time.sleep(self.delay_s)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def supervisor(primary, **option_overrides) -> SafeFallbackController:
    options = SafetyFallbackOptions(
        enabled=True,
        solve_deadline_s=option_overrides.pop("solve_deadline_s", 1.0),
        **option_overrides,
    )
    return SafeFallbackController(primary, options=options, bounds=BOUNDS)


GOAL = ControlGoal(
    position=np.array([3.0, 2.0, 2.5]),
    quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
)
OBSTACLES = [obstacle_belief()]


class SafetyFallbackConfigurationTests(unittest.TestCase):
    def test_ccmpc_config_enables_stage6_policy_and_round_trips(self):
        config = load_native_mujoco_config(
            CODE_ROOT / "config" / "mujoco_native_ccmpc.yaml"
        )
        self.assertTrue(config.safety_fallback.enabled)
        self.assertTrue(config.safety_fallback.reject_on_deadline_miss)
        self.assertAlmostEqual(
            config.safety_fallback.maximum_acceptable_slack_m,
            0.08,
        )
        self.assertEqual(type(config).from_mapping(config.to_mapping()), config)

    def test_legacy_config_defaults_to_disabled(self):
        config = load_native_mujoco_config(
            CODE_ROOT / "config" / "mujoco_native.yaml"
        )
        mapping = config.to_mapping()
        mapping["controller"].pop("safety_fallback")
        loaded = type(config).from_mapping(mapping)
        self.assertFalse(loaded.safety_fallback.enabled)

    def test_invalid_slack_threshold_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            SafetyFallbackOptions(
                guarantee_slack_tolerance_m=0.1,
                maximum_acceptable_slack_m=0.05,
            )


class SafetySupervisorTests(unittest.TestCase):
    def test_safe_primary_solution_is_applied_and_guarantee_eligible(self):
        controller = supervisor(ScriptedController([solution()]))
        result = controller.solve(vehicle_belief(), OBSTACLES, GOAL, 0.0)
        self.assertTrue(result.solution_accepted)
        self.assertFalse(result.fallback_active)
        self.assertEqual(result.command_source, "PRIMARY_NMPC")
        self.assertEqual(result.safety_assurance_status, "GUARANTEE_ELIGIBLE")

    def test_positive_slack_can_be_applied_only_as_not_guaranteed(self):
        controller = supervisor(ScriptedController([solution(slack=0.02)]))
        result = controller.solve(vehicle_belief(), OBSTACLES, GOAL, 0.0)
        self.assertTrue(result.solution_accepted)
        self.assertEqual(
            result.safety_assurance_status,
            "NOT_GUARANTEED_POSITIVE_SLACK",
        )

    def test_large_slack_uses_hold_last_then_position_hold(self):
        accepted = solution(command=(0.02, 0.0001, 0.0, 0.0))
        rejected = solution(slack=0.09)
        controller = supervisor(
            ScriptedController([accepted, rejected, rejected]),
            hold_last_command_steps=1,
        )
        first = controller.solve(vehicle_belief(), OBSTACLES, GOAL, 0.0)
        second = controller.solve(vehicle_belief(), OBSTACLES, GOAL, 0.05)
        third = controller.solve(
            vehicle_belief(velocity=(0.2, -0.1, 0.05)),
            OBSTACLES,
            GOAL,
            0.10,
        )
        np.testing.assert_allclose(second.command, first.command)
        self.assertEqual(second.fallback_level, 1)
        self.assertEqual(second.command_source, "HOLD_LAST_ACCEPTED")
        self.assertEqual(second.fallback_reason, "SLACK_LIMIT_EXCEEDED")
        self.assertEqual(third.fallback_level, 2)
        self.assertEqual(third.command_source, "POSITION_HOLD_PD")
        self.assertTrue(np.all(third.command >= controller._limits_lower))
        self.assertTrue(np.all(third.command <= controller._limits_upper))

    def test_unsuccessful_backend_solution_is_rejected(self):
        controller = supervisor(
            ScriptedController([solution(primary_success=False)]),
            hold_last_command_steps=0,
        )
        result = controller.solve(vehicle_belief(), OBSTACLES, GOAL, 0.0)
        self.assertFalse(result.solution_accepted)
        self.assertEqual(result.fallback_reason, "PRIMARY_SOLVER_FAILED")
        self.assertEqual(result.command_source, "POSITION_HOLD_PD")

    def test_joint_budget_failure_is_rejected(self):
        controller = supervisor(
            ScriptedController([solution(risk_status="BUDGET_EXCEEDED")]),
            hold_last_command_steps=0,
        )
        result = controller.solve(vehicle_belief(), OBSTACLES, GOAL, 0.0)
        self.assertEqual(result.fallback_reason, "RISK_BUDGET_INVALID")

    def test_excessive_solver_residual_is_rejected(self):
        candidate = replace(
            solution(),
            primary_solver_primal_residual=0.01,
        )
        controller = supervisor(
            ScriptedController([candidate]),
            hold_last_command_steps=0,
        )
        result = controller.solve(vehicle_belief(), OBSTACLES, GOAL, 0.0)
        self.assertEqual(result.fallback_reason, "SOLVER_RESIDUAL_EXCEEDED")

    def test_out_of_bounds_command_is_rejected(self):
        controller = supervisor(
            ScriptedController([solution(command=(1.0, 0.0, 0.0, 0.0))]),
            hold_last_command_steps=0,
        )
        result = controller.solve(vehicle_belief(), OBSTACLES, GOAL, 0.0)
        self.assertEqual(result.fallback_reason, "COMMAND_OUT_OF_BOUNDS")
        self.assertTrue(np.all(result.command <= controller._limits_upper))

    def test_deadline_miss_rejects_stale_command(self):
        controller = supervisor(
            ScriptedController([solution()], delay_s=0.01),
            solve_deadline_s=0.001,
            hold_last_command_steps=0,
        )
        result = controller.solve(vehicle_belief(), OBSTACLES, GOAL, 0.0)
        self.assertTrue(result.deadline_missed)
        self.assertEqual(result.fallback_reason, "DEADLINE_MISSED")
        self.assertFalse(result.solution_accepted)

    def test_solver_exception_returns_normalized_fallback_solution(self):
        controller = supervisor(
            ScriptedController([RuntimeError("injected failure")]),
            hold_last_command_steps=0,
        )
        result = controller.solve(vehicle_belief(), OBSTACLES, GOAL, 0.0)
        self.assertEqual(result.fallback_reason, "PRIMARY_SOLVE_EXCEPTION")
        self.assertFalse(result.primary_solver_success)
        self.assertEqual(result.nominal_states.shape, (3, 13))
        self.assertEqual(
            result.predicted_obstacle_covariances.shape,
            (3, 1, 6, 6),
        )

    def test_repeated_failures_escalate_to_emergency_hover(self):
        controller = supervisor(
            ScriptedController(
                [solution(primary_success=False), solution(primary_success=False)]
            ),
            hold_last_command_steps=0,
            emergency_after_consecutive_rejections=2,
        )
        first = controller.solve(vehicle_belief(), OBSTACLES, GOAL, 0.0)
        second = controller.solve(
            vehicle_belief(body_rate=(0.5, -0.25, 0.1)),
            OBSTACLES,
            GOAL,
            0.05,
        )
        self.assertEqual(first.command_source, "POSITION_HOLD_PD")
        self.assertEqual(second.command_source, "EMERGENCY_HOVER")
        self.assertEqual(second.fallback_level, 3)
        self.assertAlmostEqual(second.command[0], 0.0)

    def test_reset_clears_rejection_state_and_last_command(self):
        primary = ScriptedController(
            [solution(), solution(primary_success=False), solution(primary_success=False)]
        )
        controller = supervisor(primary, hold_last_command_steps=1)
        belief = vehicle_belief()
        controller.reset(belief)
        controller.solve(belief, OBSTACLES, GOAL, 0.0)
        fallback = controller.solve(belief, OBSTACLES, GOAL, 0.05)
        self.assertEqual(fallback.command_source, "HOLD_LAST_ACCEPTED")
        controller.reset(belief)
        after_reset = controller.solve(belief, OBSTACLES, GOAL, 0.0)
        self.assertEqual(after_reset.command_source, "POSITION_HOLD_PD")
        self.assertEqual(after_reset.consecutive_rejections, 1)
        self.assertEqual(primary.reset_count, 2)


@unittest.skipUnless(
    importlib.util.find_spec("mujoco"),
    "MuJoCo is not installed",
)
class SafetyFallbackClosedLoopTests(unittest.TestCase):
    def test_fault_sequence_reaches_plant_without_crashing_loop(self):
        from run_coupled import run_coupled_simulation

        primary = ScriptedController(
            [
                solution(command=(0.01, 0.0, 0.0, 0.0)),
                RuntimeError("injected solver fault"),
                RuntimeError("injected solver fault"),
            ]
        )
        options = SafetyFallbackOptions(
            enabled=True,
            solve_deadline_s=1.0,
            hold_last_command_steps=1,
            emergency_after_consecutive_rejections=10,
        )
        config = load_native_mujoco_config(
            CODE_ROOT / "config" / "mujoco_native_ccmpc.yaml"
        )
        result = run_coupled_simulation(
            x0_vals=config.start,
            goal_pos=config.goal_position,
            goal_euler=config.goal_euler,
            bounds=config.bounds,
            obstacles=[dict(config.obstacles[0])],
            margin=config.safety_margin,
            sim_seconds=0.16,
            mpc_dt=config.mpc_timestep_s,
            n_horizon=2,
            max_iter=10,
            mj_dt=config.mujoco_timestep_s,
            controller=primary,
            safety_fallback_options=options,
        )
        self.assertEqual(
            result["command_source"].tolist(),
            ["PRIMARY_NMPC", "HOLD_LAST_ACCEPTED", "POSITION_HOLD_PD"],
        )
        self.assertEqual(
            result["fallback_reason"].tolist(),
            ["", "PRIMARY_SOLVE_EXCEPTION", "PRIMARY_SOLVE_EXCEPTION"],
        )
        np.testing.assert_array_equal(
            result["solution_accepted"],
            [True, False, False],
        )
        self.assertFalse(result["collided"])
        self.assertFalse(np.isnan(result["pos"]).any())


if __name__ == "__main__":
    unittest.main()
