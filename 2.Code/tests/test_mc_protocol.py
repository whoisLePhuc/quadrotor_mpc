"""Unit tests for Monte Carlo validation protocol semantics.

Covers the truth table from spec 05: algorithmic comparison applies late-but-
valid primary commands, realtime qualification rejects them to fallback, and
both protocols record identical timing facts with different dispositions.
"""

from __future__ import annotations

import unittest

import numpy as np

from quadrotor_mpc.application.validation.monte_carlo import (
    load_native_monte_carlo_protocol,
)
from quadrotor_mpc.application.validation.protocol import (
    ABORT_HOVER_SOURCE,
    FALLBACK_APPLIED_DEADLINE_MISS,
    FALLBACK_APPLIED_PRIMARY_INVALID,
    FALLBACK_SOURCE,
    NONE_SOURCE,
    PRIMARY_APPLIED_LATE,
    PRIMARY_APPLIED_ON_TIME,
    PRIMARY_SOURCE,
    DeadlinePolicy,
    MonteCarloProtocol,
    config_hash_of,
    deadline_facts,
    deadline_policy_for,
    select_applied_command,
    validate_protocol_policy,
)
from quadrotor_mpc.infrastructure.resources import resolve_input_path

ALG = MonteCarloProtocol.ALGORITHMIC_COMPARISON
RT = MonteCarloProtocol.REALTIME_QUALIFICATION


class MonteCarloProtocolTruthTableTests(unittest.TestCase):
    def test_algorithmic_protocol_applies_late_valid_primary_command(self):
        decision = select_applied_command(
            protocol=ALG,
            primary_valid=True,
            deadline_missed=True,
        )
        self.assertEqual(decision.applied_command_source, PRIMARY_SOURCE)
        self.assertEqual(decision.primary_disposition, PRIMARY_APPLIED_LATE)
        self.assertTrue(decision.primary_valid)
        self.assertTrue(decision.deadline_missed)

    def test_realtime_protocol_rejects_late_valid_primary_command(self):
        decision = select_applied_command(
            protocol=RT,
            primary_valid=True,
            deadline_missed=True,
        )
        self.assertEqual(decision.applied_command_source, FALLBACK_SOURCE)
        self.assertEqual(
            decision.primary_disposition,
            FALLBACK_APPLIED_DEADLINE_MISS,
        )
        self.assertTrue(decision.primary_valid)
        self.assertTrue(decision.deadline_missed)

    def test_both_protocols_apply_on_time_valid_primary_command(self):
        for protocol in (ALG, RT):
            decision = select_applied_command(
                protocol=protocol,
                primary_valid=True,
                deadline_missed=False,
            )
            self.assertEqual(decision.applied_command_source, PRIMARY_SOURCE)
            self.assertEqual(decision.primary_disposition, PRIMARY_APPLIED_ON_TIME)

    def test_both_protocols_fallback_for_invalid_primary_command(self):
        for protocol in (ALG, RT):
            decision = select_applied_command(
                protocol=protocol,
                primary_valid=False,
                deadline_missed=True,
            )
            self.assertEqual(decision.applied_command_source, FALLBACK_SOURCE)
            self.assertEqual(
                decision.primary_disposition,
                FALLBACK_APPLIED_PRIMARY_INVALID,
            )
            self.assertFalse(decision.primary_valid)

    def test_deadline_miss_is_recorded_in_algorithmic_protocol(self):
        decision = select_applied_command(
            protocol=ALG,
            primary_valid=True,
            deadline_missed=True,
        )
        self.assertTrue(decision.deadline_missed)
        self.assertEqual(decision.applied_command_source, PRIMARY_SOURCE)

    def test_rejection_reasons_are_preserved(self):
        decision = select_applied_command(
            protocol=RT,
            primary_valid=False,
            deadline_missed=False,
            rejection_reasons=("SOLVER_FAILURE", "RESIDUAL_INVALID"),
        )
        self.assertEqual(
            decision.rejection_reasons,
            ("SOLVER_FAILURE", "RESIDUAL_INVALID"),
        )

    def test_invalid_primary_takes_precedence_over_deadline(self):
        decision = select_applied_command(
            protocol=RT,
            primary_valid=False,
            deadline_missed=True,
        )
        self.assertEqual(
            decision.primary_disposition,
            FALLBACK_APPLIED_PRIMARY_INVALID,
        )

    def test_unknown_protocol_is_rejected(self):
        with self.assertRaises(ValueError):
            deadline_policy_for(MonteCarloProtocol.UNKNOWN_LEGACY)


class MonteCarloProtocolBoundaryTests(unittest.TestCase):
    def test_exactly_equal_to_period_is_not_a_deadline_miss(self):
        missed, overrun = deadline_facts(50.0, 50.0)
        self.assertFalse(missed)
        self.assertEqual(overrun, 0.0)

    def test_time_just_above_period_is_a_deadline_miss(self):
        missed, overrun = deadline_facts(50.001, 50.0)
        self.assertTrue(missed)
        self.assertAlmostEqual(overrun, 0.001, places=6)

    def test_non_finite_timing_is_invalid_not_on_time(self):
        with self.assertRaises(ValueError):
            deadline_facts(float("nan"), 50.0)
        with self.assertRaises(ValueError):
            deadline_facts(float("inf"), 50.0)

    def test_negative_timing_is_invalid(self):
        with self.assertRaises(ValueError):
            deadline_facts(-1.0, 50.0)

    def test_zero_control_period_is_rejected(self):
        with self.assertRaises(ValueError):
            deadline_facts(10.0, 0.0)


class MonteCarloProtocolPolicyTests(unittest.TestCase):
    def test_deadline_policy_is_derived_from_protocol(self):
        self.assertIs(
            deadline_policy_for(ALG),
            DeadlinePolicy.RECORD_ONLY,
        )
        self.assertIs(
            deadline_policy_for(RT),
            DeadlinePolicy.REJECT_TO_FALLBACK,
        )

    def test_protocol_policy_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_protocol_policy(ALG, DeadlinePolicy.REJECT_TO_FALLBACK)
        with self.assertRaises(ValueError):
            validate_protocol_policy(RT, DeadlinePolicy.RECORD_ONLY)

    def test_matching_policy_is_accepted(self):
        validate_protocol_policy(ALG, DeadlinePolicy.RECORD_ONLY)
        validate_protocol_policy(RT, DeadlinePolicy.REJECT_TO_FALLBACK)

    def test_config_hash_is_deterministic_and_order_independent(self):
        first = config_hash_of({"a": 1, "b": [1, 2]})
        second = config_hash_of({"b": [1, 2], "a": 1})
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        self.assertNotEqual(
            first,
            config_hash_of({"a": 1, "b": [1, 3]}),
        )


class MonteCarloProtocolLoaderTests(unittest.TestCase):
    def test_default_protocol_loads_algorithmic_comparison(self):
        protocol = load_native_monte_carlo_protocol(
            resolve_input_path("config/native_monte_carlo.yaml")
        )
        self.assertEqual(protocol.protocol_type, "algorithmic_comparison")
        self.assertAlmostEqual(protocol.control_period_ms, 50.0)
        self.assertEqual(protocol.deadline_clock, "solver_only")
        self.assertEqual(protocol.deadline_policy, "record_only")

    def test_enum_round_trip(self):
        self.assertEqual(MonteCarloProtocol("algorithmic_comparison"), ALG)
        self.assertEqual(MonteCarloProtocol("realtime_qualification"), RT)
        self.assertEqual(MonteCarloProtocol("UNKNOWN_LEGACY"), MonteCarloProtocol.UNKNOWN_LEGACY)

    def test_canonical_command_sources_exist(self):
        self.assertEqual(PRIMARY_SOURCE, "PRIMARY")
        self.assertEqual(FALLBACK_SOURCE, "FALLBACK")
        self.assertEqual(ABORT_HOVER_SOURCE, "ABORT_HOVER")
        self.assertEqual(NONE_SOURCE, "NONE")


class MonteCarloProtocolIntegrationTests(unittest.TestCase):
    """Same captured result must take different paths under each protocol."""

    def _supervised(self, *, reject_on_deadline_miss: bool):
        import time

        from quadrotor_mpc.control.nmpc.safety import (
            SafeFallbackController,
            SafetyFallbackOptions,
        )
        from quadrotor_mpc.core.contracts import ControlGoal, ControlSolution

        steps = 2
        nominal = np.zeros((steps, 13))
        nominal[:, 2] = 1.0
        nominal[:, 6] = 1.0
        solution = ControlSolution(
            command=np.array([0.01, 0.0, 0.0, 0.0]),
            nominal_states=nominal,
            predicted_covariances=np.zeros((steps, 12, 12)),
            chance_margins=np.zeros((steps, 1)),
            risk_allocations=np.zeros((steps, 1)),
            slacks=np.zeros((steps, 1)),
            solver_status="SOLVED",
            predicted_obstacle_covariances=np.zeros((steps, 1, 6, 6)),
            projected_uncertainties=np.zeros((steps, 1)),
            tightened_safety_radii=np.full((steps, 1), 0.8),
            risk_semantics="joint",
            risk_budget_status="BUDGET_OK",
            primary_solver_status="Solve_Succeeded",
            primary_solver_success=True,
            primary_solver_primal_residual=1e-7,
            primary_solver_dual_residual=1e-7,
            residual_status="AVAILABLE",
            primary_solver_primal_residual_status="AVAILABLE",
            primary_solver_dual_residual_status="AVAILABLE",
        )

        class SlowController:
            horizon_steps = 1

            def reset(self, _belief):
                pass

            def solve(self, _b, _o, _g, _t):
                time.sleep(0.002)
                return solution

        options = SafetyFallbackOptions(
            enabled=True,
            solve_deadline_s=0.001,
            reject_on_deadline_miss=reject_on_deadline_miss,
            hold_last_command_steps=0,
            emergency_after_consecutive_rejections=10,
        )
        controller = SafeFallbackController(
            SlowController(),
            options=options,
            bounds={"thrust": 0.085, "torque_rp": 0.002, "torque_yaw": 0.0004},
        )
        from quadrotor_mpc.core.contracts import VehicleBelief

        belief = VehicleBelief.exact(
            np.array([0.0, 0.0, 1.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        )
        return controller.solve(belief, [], ControlGoal(
            position=np.array([1.0, 0.0, 1.0]),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        ), 0.0)

    def test_algorithmic_protocol_applies_late_primary_in_loop(self):
        result = self._supervised(reject_on_deadline_miss=False)
        self.assertTrue(result.deadline_missed)
        self.assertTrue(result.usable_without_deadline_gate)
        self.assertTrue(result.solution_accepted)
        self.assertEqual(result.command_source, "PRIMARY_NMPC")

    def test_realtime_protocol_rejects_late_primary_in_loop(self):
        result = self._supervised(reject_on_deadline_miss=True)
        self.assertTrue(result.deadline_missed)
        self.assertTrue(result.usable_without_deadline_gate)
        self.assertFalse(result.solution_accepted)
        self.assertEqual(result.fallback_reason, "DEADLINE_MISSED")

    def test_applied_command_provenance_plant_input_matches(self):
        from quadrotor_mpc.application.validation.protocol import (
            MonteCarloProtocol,
            select_applied_command,
        )

        decision = select_applied_command(
            protocol=MonteCarloProtocol.ALGORITHMIC_COMPARISON,
            primary_valid=True,
            deadline_missed=True,
        )
        self.assertEqual(decision.applied_command_source, PRIMARY_SOURCE)
        self.assertEqual(decision.primary_disposition, PRIMARY_APPLIED_LATE)

    def test_aggregation_rejects_mixed_protocols(self):
        from pathlib import Path

        from quadrotor_mpc.application.validation.monte_carlo import (
            NativeMonteCarloProtocol,
            aggregate_native_trials,
        )

        class FakeTrial:
            protocol = "algorithmic_comparison"
            mode = "joint"
            noise_label = "nominal"
            seed = 1
            completed = True
            numerical_failure = False
            collision = False
            success = True
            budget_failure_ticks = 0
            positive_slack_ticks = 0
            fallback_ticks = 0
            deadline_miss_count = 0
            primary_application_rate = 1.0
            final_error_m = 0.1
            min_clearance_m = 0.5
            path_length_m = 4.0
            tracking_rmse_m = 0.2
            control_effort = 0.01
            estimation_position_rmse_m = 0.03
            mean_solver_ms = 20.0
            p95_solver_ms = 30.0
            p99_solver_ms = 40.0
            max_solver_ms = 50.0
            primary_success_rate = 1.0
            accepted_solution_rate = 1.0
            deadline_miss_rate = 0.0
            positive_slack_rate = 0.0
            max_slack_m = 0.0
            chance_violation_rate = 0.0
            fallback_rate = 0.0
            guarantee_eligible_ticks = 10
            guarantee_eligible_rate = 1.0
            guarantee_eligible_episode = True
            horizon_eligible_tick_count = 10
            horizon_eligible_tick_rate = 1.0
            horizon_ineligible_reason_counts = {}
            episode_all_ticks_horizon_eligible = True
            episode_any_fallback = False
            episode_any_positive_slack = False
            episode_any_deadline_miss = False
            maximum_budget_error = 0.0
            fallback_application_rate = 0.0
            residual_available_count = 10
            residual_unavailable_count = 0
            residual_invalid_count = 0
            residual_available_rate = 1.0
            residual_invalid_rate = 0.0
            residual_gate_pass_rate = 1.0
            residual_gate_fail_rate = 0.0
            residual_gate_unknown_rate = 0.0
            primal_residual_p50 = 1e-6
            primal_residual_p95 = 1e-6
            primal_residual_p99 = 1e-6
            dual_residual_p50 = 1e-6
            dual_residual_p95 = 1e-6
            dual_residual_p99 = 1e-6
            enforced_profile_count = 10
            missing_enforced_profile_count = 0
            post_solve_diagnostic_profile_count = 0
            total_ticks = 10
            primary_applied_on_time_count = 10
            primary_applied_late_count = 0
            fallback_deadline_count = 0
            fallback_primary_invalid_count = 0
            max_deadline_overrun_ms = 0.0

        class FakeTrialRealtime(FakeTrial):
            protocol = "realtime_qualification"

        protocol = NativeMonteCarloProtocol(
            name="test",
            base_config_path=Path("config/native_monte_carlo.yaml"),
            output_dir=Path("outputs/test"),
            modes=("joint_uniform",),
            noise_levels=(
                __import__(
                    "quadrotor_mpc.application.validation.monte_carlo",
                    fromlist=["NoiseLevel"],
                ).NoiseLevel("nominal", 1.0),
            ),
            trials=2,
            first_seed=1,
            confidence_level=0.95,
            minimum_trials_for_claim=1,
            empirical_collision_rate_limit=0.10,
            require_zero_positive_slack=True,
            require_zero_fallback=True,
            require_zero_budget_failures=True,
            timing_percentile=0.99,
            protocol_type="algorithmic_comparison",
        )
        mixed = [FakeTrial(), FakeTrialRealtime()]
        with self.assertRaises(ValueError):
            aggregate_native_trials(
                mixed,
                protocol,
                controller_period_ms=50.0,
            )


if __name__ == "__main__":
    unittest.main()
