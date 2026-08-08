"""Tests for controller timing breakdown (spec section 10).

Covers the monotonic recorder utility, the timing contract, deterministic
scope-to-field mapping, the independent total on failure paths, and
serialization/aggregation semantics.
"""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

from quadrotor_mpc.core.contracts import VehicleBelief
from quadrotor_mpc.core.timing import (
    TIMING_FIELD_NAMES,
    ControllerTiming,
    TimingRecorder,
    merge_controller_timing,
    summarize_timing_series,
)

CODE_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_AVAILABLE = importlib.util.find_spec("mujoco") is not None
CASADI_AVAILABLE = importlib.util.find_spec("casadi") is not None
DO_MPC_AVAILABLE = importlib.util.find_spec("do_mpc") is not None


class _FakeClock:
    def __init__(self, values):
        self._values = list(values)
        self._index = 0

    def __call__(self) -> int:
        value = self._values[self._index]
        self._index = min(self._index + 1, len(self._values) - 1)
        return value


def _require(value: float | None) -> float:
    assert value is not None
    return float(value)


class TimingRecorderTests(unittest.TestCase):
    def test_fake_clock_measures_exact_elapsed(self):
        recorder = TimingRecorder(clock_ns=_FakeClock([0, 10, 20, 40, 70]))
        with recorder.measure("seed_trajectory_time_ms"):
            pass
        with recorder.measure("covariance_propagation_time_ms"):
            pass
        snapshot = recorder.snapshot()
        self.assertAlmostEqual(_require(snapshot.seed_trajectory_time_ms), 10 / 1e6)
        self.assertAlmostEqual(
            _require(snapshot.covariance_propagation_time_ms), 20 / 1e6
        )

    def test_exception_in_context_still_records_elapsed(self):
        recorder = TimingRecorder(clock_ns=_FakeClock([0, 25]))
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with recorder.measure("nlp_solve_time_ms"):
                raise RuntimeError("boom")
        snapshot = recorder.snapshot()
        self.assertAlmostEqual(_require(snapshot.nlp_solve_time_ms), 25 / 1e6)

    def test_distinct_scopes_do_not_overwrite(self):
        recorder = TimingRecorder(clock_ns=_FakeClock([0, 5, 5, 9]))
        with recorder.measure("seed_trajectory_time_ms"):
            pass
        with recorder.measure("nlp_solve_time_ms"):
            pass
        snapshot = recorder.snapshot()
        self.assertAlmostEqual(_require(snapshot.seed_trajectory_time_ms), 5 / 1e6)
        self.assertAlmostEqual(_require(snapshot.nlp_solve_time_ms), 4 / 1e6)

    def test_duplicate_scope_fails_fast(self):
        recorder = TimingRecorder(clock_ns=_FakeClock([0, 1, 1]))
        with recorder.measure("nlp_solve_time_ms"):
            pass
        with self.assertRaisesRegex(ValueError, "measured twice"):
            with recorder.measure("nlp_solve_time_ms"):
                pass

    def test_unknown_scope_name_is_rejected(self):
        recorder = TimingRecorder(clock_ns=_FakeClock([0, 1]))
        with recorder.measure("not_a_real_phase"):
            pass
        with self.assertRaisesRegex(ValueError, "unrecognized timing scope"):
            recorder.snapshot()


class ControllerTimingContractTests(unittest.TestCase):
    def test_non_null_values_must_be_finite_and_nonnegative(self):
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite and >= 0"):
                    ControllerTiming(nlp_solve_time_ms=value)

    def test_zero_is_valid_and_distinct_from_missing(self):
        timing = ControllerTiming(nlp_solve_time_ms=0.0)
        self.assertEqual(timing.nlp_solve_time_ms, 0.0)
        self.assertIsNone(timing.geometry_context_time_ms)
        mapping = timing.to_mapping()
        self.assertEqual(mapping["nlp_solve_time_ms"], 0.0)
        self.assertIsNone(mapping["geometry_context_time_ms"])

    def test_none_serializes_to_json_null(self):
        timing = ControllerTiming()
        payload = json.dumps(timing.to_mapping())
        self.assertIn('"nlp_solve_time_ms": null', payload)
        self.assertIn('"geometry_context_time_ms": null', payload)

    def test_zero_serializes_as_zero_not_null(self):
        payload = json.dumps(ControllerTiming(nlp_solve_time_ms=0.0).to_mapping())
        self.assertIn('"nlp_solve_time_ms": 0.0', payload)

    def test_round_trip_preserves_values_and_missing(self):
        timing = ControllerTiming(
            seed_trajectory_time_ms=1.5,
            nlp_solve_time_ms=12.25,
        )
        reloaded = ControllerTiming.from_mapping(timing.to_mapping())
        self.assertIsNotNone(reloaded)
        assert reloaded is not None
        self.assertEqual(reloaded.seed_trajectory_time_ms, 1.5)
        self.assertEqual(reloaded.nlp_solve_time_ms, 12.25)
        self.assertIsNone(reloaded.geometry_context_time_ms)

    def test_merge_preserves_base_and_applies_updates(self):
        merged = merge_controller_timing(
            ControllerTiming(nlp_solve_time_ms=5.0),
            total_controller_time_ms=9.0,
        )
        self.assertEqual(merged.nlp_solve_time_ms, 5.0)
        self.assertEqual(merged.total_controller_time_ms, 9.0)
        self.assertIsNone(merged.geometry_context_time_ms)
        merged_from_none = merge_controller_timing(
            None,
            total_controller_time_ms=3.0,
        )
        self.assertEqual(merged_from_none.total_controller_time_ms, 3.0)
        for name in TIMING_FIELD_NAMES:
            if name != "total_controller_time_ms":
                self.assertIsNone(getattr(merged_from_none, name))


class TimingSeriesTests(unittest.TestCase):
    def test_missing_samples_excluded_from_percentiles_but_counted(self):
        series: list[dict[str, float | None] | None] = [
            {"nlp_solve_time_ms": 10.0},
            {"nlp_solve_time_ms": None},
            None,
            {"nlp_solve_time_ms": 30.0},
        ]
        stats = summarize_timing_series(series)
        nlp = stats["nlp_solve_time_ms"]
        self.assertEqual(nlp["count_available"], 2)
        self.assertEqual(nlp["count_missing"], 2)
        self.assertEqual(nlp["mean_ms"], 20.0)
        self.assertEqual(nlp["p95_ms"], 29.0)
        self.assertEqual(nlp["max_ms"], 30.0)

    def test_no_valid_samples_gives_null_statistics(self):
        stats = summarize_timing_series(
            [
                {"nlp_solve_time_ms": None},
                None,
            ]
        )
        nlp = stats["nlp_solve_time_ms"]
        self.assertEqual(nlp["count_available"], 0)
        self.assertEqual(nlp["count_missing"], 2)
        self.assertIsNone(nlp["mean_ms"])
        self.assertIsNone(nlp["p99_ms"])

    def test_zero_values_are_kept_not_treated_as_missing(self):
        stats = summarize_timing_series([{"nlp_solve_time_ms": 0.0}])
        self.assertEqual(stats["nlp_solve_time_ms"]["count_available"], 1)
        self.assertEqual(stats["nlp_solve_time_ms"]["mean_ms"], 0.0)

    def test_total_field_is_reported_per_tick(self):
        series: list[dict[str, float | None] | None] = [
            {"total_controller_time_ms": 40.0},
            {"total_controller_time_ms": 60.0},
        ]
        stats = summarize_timing_series(series)
        self.assertEqual(stats["total_controller_time_ms"]["p99_ms"], 59.8)
        self.assertEqual(stats["total_controller_time_ms"]["max_ms"], 60.0)


def _vehicle_belief() -> VehicleBelief:
    return VehicleBelief(
        mean_state_13=np.array(
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=float,
        ),
        error_covariance_12=np.eye(12) * 1e-3,
    )


def _obstacle_beliefs(specs):
    from quadrotor_mpc.core.contracts import ObstacleBelief, SphericalObstacle

    beliefs = []
    for index, spec in enumerate(specs):
        beliefs.append(
            ObstacleBelief(
                mean_state_6=np.array(
                    [
                        spec.get("x", 0.0),
                        spec.get("y", 0.0),
                        spec.get("z", 1.0),
                        0.0,
                        0.0,
                        0.0,
                    ],
                    dtype=float,
                ),
                covariance_6=np.eye(6) * 1e-3,
                shape=SphericalObstacle(float(spec["radius"])),
                name=str(spec.get("name", f"obstacle_{index}")),
            )
        )
    return beliefs


@unittest.skipUnless(
    MUJOCO_AVAILABLE and CASADI_AVAILABLE and DO_MPC_AVAILABLE,
    "optional NMPC/MuJoCo dependencies are not installed",
)
class DeterministicControllerTimingTests(unittest.TestCase):
    """Section 10.3: deterministic scope-to-field mapping through the pipeline."""

    @classmethod
    def setUpClass(cls):
        from quadrotor_mpc.control.nmpc.deterministic import (
            DeterministicNMPCController,
        )
        from quadrotor_mpc.interfaces.desktop.viewer import (
            load_native_mujoco_config,
        )

        config = load_native_mujoco_config(
            CODE_ROOT / "config" / "mujoco_native_ccmpc.yaml"
        )
        mapping = config.to_mapping()
        mapping["controller"]["chance_constraints"]["enabled"] = False
        mapping["controller"]["covariance_propagation"]["enabled"] = False
        config = type(config).from_mapping(mapping)
        cls.config = config
        cls.controller = DeterministicNMPCController(
            bounds=config.bounds,
            obstacle_specs=[dict(item) for item in config.obstacles],
            margin=config.safety_margin,
            horizon_steps=config.horizon_steps,
            timestep_s=config.mpc_timestep_s,
            max_iter=config.max_solver_iterations,
            covariance_options=config.covariance_propagation,
            chance_options=config.chance_constraints,
            clock_ns=_FakeClock(list(range(0, 10_000, 7))),
        )

    def _solve_with_fake_clock(self):
        from quadrotor_mpc.core.contracts import ControlGoal

        goal = ControlGoal(
            position=np.array([3.0, 2.0, 2.5]),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        return self.controller.solve(
            _vehicle_belief(),
            _obstacle_beliefs(self.config.obstacles),
            goal,
            0.0,
        )

    def test_success_path_records_all_phases(self):
        solution = self._solve_with_fake_clock()
        timing = solution.controller_timing
        self.assertIsNotNone(timing)
        for name in (
            "seed_trajectory_time_ms",
            "covariance_propagation_time_ms",
            "chance_profile_time_ms",
            "risk_allocation_time_ms",
            "tightening_time_ms",
            "tvp_update_time_ms",
            "nlp_solve_time_ms",
            "post_solve_diagnostic_time_ms",
        ):
            value = getattr(timing, name)
            self.assertIsNotNone(value, name)
            self.assertGreaterEqual(value, 0.0)

    def test_not_applicable_phases_are_none(self):
        timing = self._solve_with_fake_clock().controller_timing
        self.assertIsNotNone(timing)
        assert timing is not None
        self.assertIsNone(timing.geometry_context_time_ms)
        self.assertIsNone(timing.safety_supervisor_time_ms)

    def test_total_is_not_set_inside_primary_controller(self):
        timing = self._solve_with_fake_clock().controller_timing
        self.assertIsNotNone(timing)
        assert timing is not None
        self.assertIsNone(timing.total_controller_time_ms)

    def test_zero_delta_clock_still_records_valid_scope(self):
        recorder = TimingRecorder(clock_ns=_FakeClock([1, 1]))
        with recorder.measure("nlp_solve_time_ms"):
            pass
        self.assertEqual(recorder.snapshot().nlp_solve_time_ms, 0.0)


class TimingFailurePathTests(unittest.TestCase):
    """Section 10.2: solver failure still reports an independent total."""

    @staticmethod
    def _solution():
        from quadrotor_mpc.core.contracts import ControlSolution

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
                (np.eye(12) * 1e-3)[None, :, :], steps, axis=0
            ),
            predicted_obstacle_covariances=np.repeat(
                (np.eye(6) * 1e-3)[None, None, :, :], steps, axis=0
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

    @unittest.skipUnless(
        MUJOCO_AVAILABLE and CASADI_AVAILABLE and DO_MPC_AVAILABLE,
        "optional NMPC/MuJoCo dependencies are not installed",
    )
    def test_exception_path_still_records_total_without_components(self):
        from quadrotor_mpc.application.native.runtime import run_coupled_simulation
        from quadrotor_mpc.control.nmpc.safety import SafetyFallbackOptions
        from quadrotor_mpc.interfaces.desktop.viewer import (
            load_native_mujoco_config,
        )

        class RaisingController:
            horizon_steps = 2

            def reset(self, belief):
                pass

            def solve(self, belief, obstacles, goal, time_s):
                raise RuntimeError("injected solver fault")

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
            sim_seconds=0.10,
            mpc_dt=config.mpc_timestep_s,
            n_horizon=2,
            max_iter=10,
            mj_dt=config.mujoco_timestep_s,
            controller=RaisingController(),
            safety_fallback_options=SafetyFallbackOptions(
                enabled=True,
                solve_deadline_s=1.0,
                hold_last_command_steps=1,
                emergency_after_consecutive_rejections=10,
            ),
        )
        timing = result["controller_timing"][0]
        self.assertIsNotNone(timing["total_controller_time_ms"])
        self.assertGreater(timing["total_controller_time_ms"], 0.0)
        self.assertIsNotNone(timing["safety_supervisor_time_ms"])
        for name in TIMING_FIELD_NAMES:
            if name not in ("total_controller_time_ms", "safety_supervisor_time_ms"):
                self.assertIsNone(timing[name])


class TimingAggregationTests(unittest.TestCase):
    def test_trial_timing_stats_flow_into_aggregate(self):
        from quadrotor_mpc.application.validation.monte_carlo import (
            NativeMonteCarloProtocol,
            NoiseLevel,
            aggregate_native_trials,
            summarize_native_trial,
        )
        from quadrotor_mpc.interfaces.desktop.viewer import (
            load_native_mujoco_config,
        )

        base = load_native_mujoco_config(
            CODE_ROOT / "config" / "mujoco_native_ccmpc.yaml"
        )
        mapping = base.to_mapping()
        mapping["simulation"]["duration_s"] = 0.10
        config = type(base).from_mapping(mapping)
        result = {
            "pos": np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            "u": np.zeros((2, 4)),
            "clearance": np.array([1.0, 1.0]),
            "solver_time_ms": np.array([30.0, 31.0]),
            "primary_solver_success": np.ones(2, dtype=bool),
            "solution_accepted": np.ones(2, dtype=bool),
            "deadline_missed": np.zeros(2, dtype=bool),
            "fallback_active": np.zeros(2, dtype=bool),
            "safety_assurance_status": np.array(
                ["GUARANTEE_ELIGIBLE", "GUARANTEE_ELIGIBLE"]
            ),
            "horizon_assurance_eligible": np.ones(2, dtype=bool),
            "horizon_assurance_reason": np.array(["eligible", "eligible"]),
            "horizon_assurance_failed_checks": np.array([[], []], dtype=object),
            "risk_budget_status": np.array(["BUDGET_OK", "BUDGET_OK"]),
            "risk_budget_total": np.array([0.1, 0.1]),
            "risk_budget_allocated": np.array([0.1, 0.1]),
            "estimated_state": np.column_stack(
                [
                    np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
                    np.zeros((2, 10)),
                ]
            ),
            "collided": False,
            "obstacle_collided": False,
            "ground_collided": False,
            "collision_type": "none",
            "controller_timing": [
                {
                    "nlp_solve_time_ms": 10.0,
                    "total_controller_time_ms": 40.0,
                },
                {
                    "nlp_solve_time_ms": 20.0,
                    "total_controller_time_ms": 60.0,
                },
            ],
            "termination_reason": "completed",
        }
        trial = summarize_native_trial(
            result,
            config,
            mode="joint_uniform",
            noise_label="nominal",
            covariance_scale=1.0,
            seed=1,
        )
        self.assertIsNotNone(trial.timing_stats)
        assert trial.timing_stats is not None
        self.assertEqual(trial.timing_stats["nlp_solve_time_ms"]["count_available"], 2)
        self.assertEqual(trial.timing_stats["nlp_solve_time_ms"]["mean_ms"], 15.0)
        self.assertEqual(
            trial.timing_stats["total_controller_time_ms"]["max_ms"],
            60.0,
        )
        self.assertIsNone(
            trial.timing_stats["geometry_context_time_ms"]["mean_ms"]
        )

        protocol = NativeMonteCarloProtocol(
            name="timing-aggregation",
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
            [trial],
            protocol,
            controller_period_ms=50.0,
        )
        timing = aggregate["noise_levels"]["nominal"]["controllers"]["joint_uniform"]["timing"]
        self.assertEqual(
            timing["total_controller_time_ms"]["count_available_trials"],
            1,
        )
        self.assertEqual(timing["nlp_solve_time_ms"]["p95_ms"], 19.5)
        self.assertEqual(
            aggregate["timing_schema"]["clock"],
            "time.perf_counter_ns",
        )


if __name__ == "__main__":
    unittest.main()
