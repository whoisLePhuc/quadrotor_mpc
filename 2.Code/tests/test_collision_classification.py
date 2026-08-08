"""Tests for collision classification (spec section 8).

Locks obstacle-vs-ground classification, signed clearance, episode
accumulation and the plant -> runtime -> trial -> artifact -> aggregator chain.
"""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

from quadrotor_mpc.core.collision import (
    CollisionAccumulator,
    CollisionObservation,
    CollisionType,
    classify_collision,
)
from quadrotor_mpc.core.contracts import (
    ControlSolution,
)

CODE_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_AVAILABLE = importlib.util.find_spec("mujoco") is not None
CASADI_AVAILABLE = importlib.util.find_spec("casadi") is not None
DO_MPC_AVAILABLE = importlib.util.find_spec("do_mpc") is not None


class ClassifyCollisionTests(unittest.TestCase):
    def test_classify_collision_table(self):
        cases = [
            (False, False, CollisionType.NONE),
            (True, False, CollisionType.OBSTACLE),
            (False, True, CollisionType.GROUND),
            (True, True, CollisionType.OBSTACLE_AND_GROUND),
        ]
        for obstacle, ground, expected in cases:
            with self.subTest(obstacle=obstacle, ground=ground):
                self.assertEqual(classify_collision(obstacle, ground), expected)

    def test_unknown_legacy_is_never_produced_by_classification(self):
        for obstacle in (False, True):
            for ground in (False, True):
                result = classify_collision(obstacle, ground)
                self.assertNotEqual(result, CollisionType.UNKNOWN_LEGACY)


def observation(
    *,
    obstacle=False,
    ground=False,
    obstacle_clearance=None,
    ground_clearance=None,
) -> CollisionObservation:
    return CollisionObservation(
        obstacle_collision_detected=obstacle,
        ground_collision_detected=ground,
        minimum_obstacle_clearance_m=obstacle_clearance,
        minimum_ground_clearance_m=ground_clearance,
    )


class CollisionAccumulatorTests(unittest.TestCase):
    def test_no_collision_episode(self):
        accumulator = CollisionAccumulator()
        for index in range(3):
            accumulator.observe(observation(obstacle_clearance=0.4), time_s=index * 0.05)
        summary = accumulator.finalize()
        self.assertFalse(summary.collided)
        self.assertEqual(summary.collision_type, CollisionType.NONE)
        self.assertIsNone(summary.first_collision_time_s)
        self.assertEqual(summary.minimum_obstacle_clearance_m, 0.4)

    def test_obstacle_mid_episode_remains_true(self):
        accumulator = CollisionAccumulator()
        accumulator.observe(observation(obstacle_clearance=0.3), time_s=0.0)
        accumulator.observe(
            observation(obstacle=True, obstacle_clearance=-0.01),
            time_s=0.05,
        )
        accumulator.observe(observation(obstacle_clearance=0.2), time_s=0.10)
        summary = accumulator.finalize()
        self.assertTrue(summary.obstacle_collision_detected)
        self.assertFalse(summary.ground_collision_detected)
        self.assertEqual(summary.collision_type, CollisionType.OBSTACLE)
        self.assertEqual(summary.first_obstacle_collision_time_s, 0.05)
        self.assertEqual(summary.first_collision_time_s, 0.05)

    def test_ground_mid_episode_remains_true(self):
        accumulator = CollisionAccumulator()
        accumulator.observe(observation(ground_clearance=1.0), time_s=0.0)
        accumulator.observe(
            observation(ground=True, ground_clearance=-0.02),
            time_s=0.05,
        )
        accumulator.observe(observation(ground_clearance=0.5), time_s=0.10)
        summary = accumulator.finalize()
        self.assertTrue(summary.ground_collision_detected)
        self.assertFalse(summary.obstacle_collision_detected)
        self.assertEqual(summary.collision_type, CollisionType.GROUND)

    def test_obstacle_then_ground_yields_both_and_first_times(self):
        accumulator = CollisionAccumulator()
        accumulator.observe(
            observation(obstacle=True, obstacle_clearance=-0.1, ground_clearance=0.9),
            time_s=1.0,
        )
        accumulator.observe(
            observation(ground=True, ground_clearance=-0.05),
            time_s=2.0,
        )
        summary = accumulator.finalize()
        self.assertEqual(summary.collision_type, CollisionType.OBSTACLE_AND_GROUND)
        self.assertEqual(summary.first_collision_time_s, 1.0)
        self.assertEqual(summary.first_obstacle_collision_time_s, 1.0)
        self.assertEqual(summary.first_ground_collision_time_s, 2.0)
        self.assertEqual(summary.first_collision_type, CollisionType.OBSTACLE)

    def test_ground_then_obstacle_first_times_are_order_aware(self):
        accumulator = CollisionAccumulator()
        accumulator.observe(
            observation(ground=True, ground_clearance=-0.05),
            time_s=1.0,
        )
        accumulator.observe(
            observation(obstacle=True, obstacle_clearance=-0.1),
            time_s=2.0,
        )
        summary = accumulator.finalize()
        self.assertEqual(summary.collision_type, CollisionType.OBSTACLE_AND_GROUND)
        self.assertEqual(summary.first_ground_collision_time_s, 1.0)
        self.assertEqual(summary.first_obstacle_collision_time_s, 2.0)
        self.assertEqual(summary.first_collision_time_s, 1.0)
        self.assertEqual(summary.first_collision_type, CollisionType.GROUND)

    def test_first_time_is_not_overwritten_by_later_collision(self):
        accumulator = CollisionAccumulator()
        accumulator.observe(
            observation(obstacle=True),
            time_s=1.0,
        )
        accumulator.observe(
            observation(obstacle=True),
            time_s=3.0,
        )
        summary = accumulator.finalize()
        self.assertEqual(summary.first_obstacle_collision_time_s, 1.0)
        self.assertEqual(summary.first_collision_time_s, 1.0)

    def test_minimum_clearance_takes_minimum_and_ignores_none(self):
        accumulator = CollisionAccumulator()
        accumulator.observe(
            observation(obstacle_clearance=None, ground_clearance=1.5),
            time_s=0.0,
        )
        accumulator.observe(
            observation(obstacle_clearance=0.3, ground_clearance=None),
            time_s=0.05,
        )
        accumulator.observe(
            observation(obstacle_clearance=0.1, ground_clearance=0.8),
            time_s=0.10,
        )
        summary = accumulator.finalize()
        self.assertEqual(summary.minimum_obstacle_clearance_m, 0.1)
        self.assertEqual(summary.minimum_ground_clearance_m, 0.8)

    def test_non_finite_clearance_raises_validation_error(self):
        accumulator = CollisionAccumulator()
        with self.assertRaisesRegex(ValueError, "must be finite or None"):
            accumulator.observe(
                observation(ground_clearance=float("nan")),
                time_s=0.0,
            )
        with self.assertRaisesRegex(ValueError, "must be finite or None"):
            accumulator.observe(
                observation(obstacle_clearance=float("inf")),
                time_s=0.0,
            )

    def test_collided_invariant_matches_two_booleans(self):
        for obstacle in (False, True):
            for ground in (False, True):
                accumulator = CollisionAccumulator()
                accumulator.observe(observation(obstacle=obstacle, ground=ground), time_s=0.0)
                summary = accumulator.finalize()
                with self.subTest(obstacle=obstacle, ground=ground):
                    self.assertEqual(
                        summary.collided,
                        summary.obstacle_collision_detected or summary.ground_collision_detected,
                    )
                    self.assertEqual(summary.collided, obstacle or ground)


class ClearanceGeometryTests(unittest.TestCase):
    def _clearance(self, z: float) -> float:
        from quadrotor_mpc.control.nmpc.core import DRONE_RADIUS

        return float(z) - DRONE_RADIUS

    def test_ground_clearance_positive_just_above_contact(self):
        self.assertGreater(self._clearance(0.08), 0.0)

    def test_ground_clearance_zero_at_contact_tolerance(self):
        self.assertAlmostEqual(self._clearance(0.07), 0.0, places=12)

    def test_ground_clearance_negative_when_penetrating(self):
        self.assertLess(self._clearance(0.03), 0.0)

    def test_rotated_ellipsoid_clearance_respects_orientation(self):
        from quadrotor_mpc.control.ccmpc.risk import collision_clearance

        # Ellipsoid with radii (2.0, 0.5, 0.5) rotated 90 degrees about z:
        # the long axis now points along the world +y direction.
        theta = np.pi / 2.0
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        radii = np.array([2.0, 0.5, 0.5], dtype=float)
        omega = rotation @ np.diag(1.0 / radii**2) @ rotation.T
        center = np.zeros(3)

        long_axis_point = np.array([0.0, 2.5, 0.0])
        short_axis_point = np.array([0.7, 0.0, 0.0])
        self.assertGreater(
            collision_clearance(long_axis_point, center, omega),
            collision_clearance(short_axis_point, center, omega),
        )
        boundary = np.array([0.0, 2.0, 0.0])
        self.assertAlmostEqual(
            collision_clearance(boundary, center, omega),
            0.0,
            places=10,
        )


def _solution() -> ControlSolution:
    steps = 3
    mean_state = np.array(
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=float,
    )
    nominal = np.repeat(mean_state.reshape(1, 13), steps, axis=0)
    margins = np.zeros((steps, 1), dtype=float)
    return ControlSolution(
        command=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
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
        slacks=margins.copy(),
        projected_uncertainties=np.full((steps, 1), 0.1),
        tightened_safety_radii=np.full((steps, 1), 0.8),
        solver_status="SOLVED_SAFE",
        risk_semantics="joint",
        risk_allocation_method="uniform",
        risk_budget_total=0.1,
        risk_budget_allocated=0.1,
        risk_budget_remaining=0.0,
        risk_constraint_count=steps,
        risk_budget_status="BUDGET_OK",
        primary_solver_status="Solve_Succeeded",
        primary_solver_success=True,
        primary_solver_primal_residual=1e-7,
        primary_solver_dual_residual=1e-7,
        primary_solver_primal_residual_status="AVAILABLE",
        primary_solver_dual_residual_status="AVAILABLE",
        residual_status="AVAILABLE",
    )


class _ScriptedController:
    horizon_steps = 2

    def __init__(self, script):
        self.script = list(script)

    def reset(self, _belief):
        pass

    def solve(self, _belief, _obstacles, _goal, _time_s):
        return self.script.pop(0)


@unittest.skipUnless(
    MUJOCO_AVAILABLE and CASADI_AVAILABLE and DO_MPC_AVAILABLE,
    "optional NMPC/MuJoCo dependencies are not installed",
)
class CollisionPipelineIntegrationTests(unittest.TestCase):
    """Section 8.4: plant -> runtime -> trial -> JSON artifact -> aggregator."""

    def _run(self, start, obstacles):
        from quadrotor_mpc.application.native.runtime import run_coupled_simulation
        from quadrotor_mpc.control.nmpc.safety import SafetyFallbackOptions
        from quadrotor_mpc.interfaces.desktop.viewer import (
            load_native_mujoco_config,
        )

        config = load_native_mujoco_config(CODE_ROOT / "config" / "mujoco_native_ccmpc.yaml")
        return run_coupled_simulation(
            x0_vals=start,
            goal_pos=config.goal_position,
            goal_euler=config.goal_euler,
            bounds=config.bounds,
            obstacles=[dict(item) for item in obstacles],
            margin=config.safety_margin,
            sim_seconds=0.10,
            mpc_dt=config.mpc_timestep_s,
            n_horizon=2,
            max_iter=10,
            mj_dt=config.mujoco_timestep_s,
            controller=_ScriptedController([_solution(), _solution()]),
            safety_fallback_options=SafetyFallbackOptions(enabled=False),
            protocol_type="algorithmic_comparison",
        )

    def _trial(self, result):
        from quadrotor_mpc.application.validation.monte_carlo import (
            summarize_native_trial,
        )
        from quadrotor_mpc.interfaces.desktop.viewer import (
            load_native_mujoco_config,
        )

        base = load_native_mujoco_config(CODE_ROOT / "config" / "mujoco_native_ccmpc.yaml")
        mapping = base.to_mapping()
        mapping["simulation"]["duration_s"] = 0.10
        config = type(base).from_mapping(mapping)
        return summarize_native_trial(
            result,
            config,
            mode="joint_uniform",
            noise_label="nominal",
            covariance_scale=1.0,
            seed=1,
        )

    def test_ground_only_scene_is_classified_ground(self):
        result = self._run(
            {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            [],
        )
        self.assertTrue(result["ground_collided"])
        self.assertFalse(result["obstacle_collided"])
        self.assertTrue(result["collided"])
        self.assertEqual(result["collision_type"], "ground")
        self.assertIsNone(result["minimum_obstacle_clearance_m"])
        self.assertLessEqual(result["minimum_ground_clearance_m"], 0.0)
        self.assertEqual(result["first_collision_time_s"], result["t"][0] + 0.05)

        trial = self._trial(result)
        self.assertEqual(trial.collision_type, "ground")
        self.assertTrue(trial.ground_collision)
        self.assertFalse(trial.obstacle_collision)
        self.assertTrue(trial.collision)

    def test_obstacle_only_scene_is_classified_obstacle(self):
        obstacle = {
            "type": "static",
            "x": 1.0,
            "y": 0.0,
            "z": 1.0,
            "radius": 0.4,
        }
        result = self._run(
            {"x": 1.3, "y": 0.0, "z": 1.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            [obstacle],
        )
        self.assertTrue(result["obstacle_collided"])
        self.assertFalse(result["ground_collided"])
        self.assertTrue(result["collided"])
        self.assertEqual(result["collision_type"], "obstacle")
        self.assertLessEqual(result["minimum_obstacle_clearance_m"], 0.0)
        self.assertIsNotNone(result["minimum_ground_clearance_m"])

        trial = self._trial(result)
        self.assertEqual(trial.collision_type, "obstacle")
        self.assertTrue(trial.obstacle_collision)
        self.assertFalse(trial.ground_collision)
        self.assertTrue(trial.collision)

    def test_safe_scene_is_classified_none_with_null_obstacle_clearance(self):
        result = self._run(
            {"x": 0.0, "y": 0.0, "z": 2.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            [],
        )
        self.assertFalse(result["collided"])
        self.assertEqual(result["collision_type"], "none")
        self.assertIsNone(result["minimum_obstacle_clearance_m"])
        self.assertGreater(result["minimum_ground_clearance_m"], 0.0)

    def test_json_artifact_round_trip_and_aggregation(self):
        from quadrotor_mpc.application.validation.monte_carlo import (
            NativeMonteCarloProtocol,
            NativeTrialResult,
            NoiseLevel,
            aggregate_native_trials,
        )

        ground_result = self._run(
            {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            [],
        )
        obstacle_result = self._run(
            {"x": 1.3, "y": 0.0, "z": 1.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            [{"type": "static", "x": 1.0, "y": 0.0, "z": 1.0, "radius": 0.4}],
        )
        safe_result = self._run(
            {"x": 0.0, "y": 0.0, "z": 2.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            [],
        )
        trials = [self._trial(result) for result in (ground_result, obstacle_result, safe_result)]
        for trial in trials:
            payload = json.dumps(trial.to_mapping())
            reloaded = NativeTrialResult(**json.loads(payload))
            self.assertEqual(reloaded.collision_type, trial.collision_type)
            self.assertEqual(reloaded.obstacle_collision, trial.obstacle_collision)
            self.assertEqual(reloaded.ground_collision, trial.ground_collision)

        protocol = NativeMonteCarloProtocol(
            name="collision-integration",
            base_config_path=CODE_ROOT / "config" / "mujoco_native_ccmpc.yaml",
            output_dir=CODE_ROOT / "outputs" / "test",
            modes=("deterministic", "joint_uniform"),
            noise_levels=(NoiseLevel("nominal", 1.0),),
            trials=30,
            first_seed=10,
            confidence_level=0.95,
            minimum_trials_for_claim=30,
            empirical_collision_rate_limit=0.10,
            require_zero_positive_slack=True,
            require_zero_fallback=True,
            require_zero_budget_failures=True,
            timing_percentile=0.99,
        )
        aggregate = aggregate_native_trials(
            trials,
            protocol,
            controller_period_ms=50.0,
        )
        summary = aggregate["noise_levels"]["nominal"]["controllers"]["joint_uniform"]
        self.assertEqual(summary["collision_rate"]["events"], 2)
        self.assertEqual(summary["obstacle_collision_rate"]["events"], 1)
        self.assertEqual(summary["ground_collision_rate"]["events"], 1)
        self.assertEqual(summary["both_collision_rate"]["events"], 0)
        self.assertEqual(summary["unknown_legacy_collision_rate"]["events"], 0)
        self.assertIsNotNone(summary["minimum_obstacle_clearance_m"])
        self.assertIsNotNone(summary["minimum_ground_clearance_m"])

    def test_both_collision_episode_is_not_double_counted(self):
        from quadrotor_mpc.application.validation.monte_carlo import (
            NativeMonteCarloProtocol,
            NativeTrialResult,
            NoiseLevel,
            aggregate_native_trials,
        )

        both = self._trial(
            self._run(
                {"x": 1.3, "y": 0.0, "z": 1.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                [{"type": "static", "x": 1.0, "y": 0.0, "z": 1.0, "radius": 0.4}],
            )
        )
        both = NativeTrialResult(
            **{
                **both.to_mapping(),
                "obstacle_collision": True,
                "ground_collision": True,
                "collision_type": "obstacle_and_ground",
                "collision": True,
            }
        )
        obstacle_only = NativeTrialResult(
            **{
                **both.to_mapping(),
                "obstacle_collision": True,
                "ground_collision": False,
                "collision_type": "obstacle",
            }
        )
        safe = NativeTrialResult(
            **{
                **both.to_mapping(),
                "obstacle_collision": False,
                "ground_collision": False,
                "collision_type": "none",
                "collision": False,
            }
        )
        protocol = NativeMonteCarloProtocol(
            name="collision-double-count",
            base_config_path=CODE_ROOT / "config" / "mujoco_native_ccmpc.yaml",
            output_dir=CODE_ROOT / "outputs" / "test",
            modes=("deterministic", "joint_uniform"),
            noise_levels=(NoiseLevel("nominal", 1.0),),
            trials=30,
            first_seed=10,
            confidence_level=0.95,
            minimum_trials_for_claim=30,
            empirical_collision_rate_limit=0.10,
            require_zero_positive_slack=True,
            require_zero_fallback=True,
            require_zero_budget_failures=True,
            timing_percentile=0.99,
        )
        aggregate = aggregate_native_trials(
            [both, obstacle_only, safe],
            protocol,
            controller_period_ms=50.0,
        )
        summary = aggregate["noise_levels"]["nominal"]["controllers"]["joint_uniform"]
        self.assertEqual(summary["collision_rate"]["events"], 2)
        self.assertEqual(summary["obstacle_collision_rate"]["events"], 2)
        self.assertEqual(summary["ground_collision_rate"]["events"], 1)
        self.assertEqual(summary["both_collision_rate"]["events"], 1)


class LegacyCollisionMigrationTests(unittest.TestCase):
    def test_legacy_true_maps_to_unknown_legacy_not_obstacle(self):
        from quadrotor_mpc.application.validation.monte_carlo import (
            summarize_native_trial,
        )
        from quadrotor_mpc.interfaces.desktop.viewer import (
            load_native_mujoco_config,
        )

        base = load_native_mujoco_config(CODE_ROOT / "config" / "mujoco_native_ccmpc.yaml")
        mapping = base.to_mapping()
        mapping["simulation"]["duration_s"] = 0.10
        config = type(base).from_mapping(mapping)
        legacy = {
            "pos": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            "u": np.zeros((2, 4)),
            "clearance": np.array([-0.01, -0.02]),
            "solver_time_ms": np.array([30.0, 31.0]),
            "risk_semantics": np.array(["joint", "joint"]),
            "risk_allocation_method": np.array(["uniform", "uniform"]),
            "collided": True,
            "termination_reason": "collision",
        }
        trial = summarize_native_trial(
            legacy,
            config,
            mode="joint_uniform",
            noise_label="nominal",
            covariance_scale=1.0,
            seed=1,
        )
        self.assertEqual(trial.collision_type, "unknown_legacy")
        self.assertFalse(trial.obstacle_collision)
        self.assertFalse(trial.ground_collision)
        self.assertTrue(trial.collision)


if __name__ == "__main__":
    unittest.main()
