"""Tests for validation mode canonicalization (spec section 12).

Locks the ``joint`` -> ``joint_uniform`` alias, the single deprecation
warning, the explicit risk-metadata provenance and the aggregation guard
that refuses to group different canonical modes.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path

from quadrotor_mpc.application.validation.modes import (
    CANONICAL_MODES,
    MODE_ALIASES,
    canonicalize_mode,
    canonicalize_modes,
)
from quadrotor_mpc.interfaces.desktop.viewer import load_native_mujoco_config

CODE_ROOT = Path(__file__).resolve().parents[1]


class CanonicalizationTests(unittest.TestCase):
    def test_joint_alias_maps_to_joint_uniform(self):
        result = canonicalize_mode("joint")
        self.assertEqual(result.requested_mode, "joint")
        self.assertEqual(result.canonical_mode, "joint_uniform")
        self.assertTrue(result.legacy_alias_used)

    def test_joint_uniform_is_already_canonical(self):
        result = canonicalize_mode("joint_uniform")
        self.assertEqual(result.canonical_mode, "joint_uniform")
        self.assertFalse(result.legacy_alias_used)

    def test_unknown_mode_fails_fast(self):
        for mode in ("teleporting", "joint_adaptive"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    canonicalize_mode(mode)

    def test_joint_adaptive_is_rejected_before_allocator_exists(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            canonicalize_mode("joint_adaptive")
        self.assertNotIn("joint_adaptive", CANONICAL_MODES)
        self.assertNotIn("joint_adaptive", MODE_ALIASES)

    def test_whitespace_and_case_are_normalized(self):
        result = canonicalize_mode("  JOINT  ")
        self.assertEqual(result.canonical_mode, "joint_uniform")
        self.assertTrue(result.legacy_alias_used)

    def test_empty_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            canonicalize_mode("   ")


class WarningTests(unittest.TestCase):
    def test_alias_emits_single_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            canonicalize_modes(["deterministic", "joint"])
        deprecations = [
            warning for warning in caught if issubclass(warning.category, DeprecationWarning)
        ]
        self.assertEqual(len(deprecations), 1)
        self.assertIn("joint_uniform", str(deprecations[0].message))

    def test_canonical_mode_does_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            canonicalize_modes(["joint_uniform"])
        deprecations = [
            warning for warning in caught if issubclass(warning.category, DeprecationWarning)
        ]
        self.assertEqual(len(deprecations), 0)

    def test_warning_is_emitted_once_per_mode_list(self):
        # The protocol is canonicalized once at load: one alias in a mode list
        # produces exactly one warning, not one per control tick or per mode.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            canonicalize_modes(["deterministic", "joint", "individual"])
        deprecations = [
            warning for warning in caught if issubclass(warning.category, DeprecationWarning)
        ]
        self.assertEqual(len(deprecations), 1)


class EndToEndEquivalenceTests(unittest.TestCase):
    def test_alias_and_canonical_build_the_same_controller_config(self):
        base = load_native_mujoco_config(CODE_ROOT / "config" / "mujoco_native_ccmpc.yaml")
        from quadrotor_mpc.application.validation.monte_carlo import (
            effective_native_config,
        )

        alias = effective_native_config(base, mode="joint", covariance_scale=1.0, seed=3)
        canonical = effective_native_config(
            base, mode="joint_uniform", covariance_scale=1.0, seed=3
        )
        self.assertEqual(alias.to_mapping(), canonical.to_mapping())
        self.assertEqual(
            canonical.chance_constraints.risk_budget.semantics,
            "joint",
        )
        self.assertEqual(
            canonical.chance_constraints.risk_budget.allocation,
            "uniform",
        )

    def test_individual_keeps_individual_semantics(self):
        base = load_native_mujoco_config(CODE_ROOT / "config" / "mujoco_native_ccmpc.yaml")
        from quadrotor_mpc.application.validation.monte_carlo import (
            effective_native_config,
        )

        individual = effective_native_config(base, mode="individual", covariance_scale=1.0, seed=3)
        self.assertEqual(
            individual.chance_constraints.risk_budget.semantics,
            "individual",
        )


class ArtifactProvenanceTests(unittest.TestCase):
    def test_protocol_load_records_canonical_and_requested_modes(self):
        from quadrotor_mpc.application.validation.monte_carlo import (
            load_native_monte_carlo_protocol,
        )

        loaded = load_native_monte_carlo_protocol(CODE_ROOT / "config" / "native_monte_carlo.yaml")
        self.assertIn("joint_uniform", loaded.modes)
        self.assertNotIn("joint", loaded.modes)
        self.assertEqual(loaded.requested_modes, loaded.modes)
        self.assertFalse(loaded.mode_aliases_used)

    def test_legacy_checkpoint_trial_mode_is_upgraded_with_provenance(self):
        from quadrotor_mpc.application.validation.monte_carlo import (
            load_trial_checkpoint,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.jsonl"
            trial = {
                "noise_label": "nominal",
                "covariance_scale": 1.0,
                "mode": "joint",
                "seed": 1,
                "expected_steps": 10,
                "completed_steps": 10,
                "completed": True,
                "termination_reason": "completed",
                "success": True,
                "collision": False,
                "numerical_failure": False,
                "final_error_m": 0.1,
                "min_clearance_m": 0.5,
                "path_length_m": 4.0,
                "tracking_rmse_m": 0.2,
                "control_effort": 0.01,
                "estimation_position_rmse_m": 0.03,
                "mean_solver_ms": 25.0,
                "p95_solver_ms": 35.0,
                "p99_solver_ms": 40.0,
                "max_solver_ms": 45.0,
                "primary_success_rate": 1.0,
                "accepted_solution_rate": 1.0,
                "deadline_miss_rate": 0.0,
                "positive_slack_ticks": 0,
                "positive_slack_rate": 0.0,
                "max_slack_m": 0.0,
                "chance_violation_rate": 0.0,
                "fallback_ticks": 0,
                "fallback_rate": 0.0,
                "guarantee_eligible_ticks": 10,
                "guarantee_eligible_rate": 1.0,
                "guarantee_eligible_episode": True,
                "horizon_eligible_tick_count": 10,
                "horizon_eligible_tick_rate": 1.0,
                "horizon_ineligible_reason_counts": {},
                "episode_all_ticks_horizon_eligible": True,
                "episode_any_fallback": False,
                "episode_any_positive_slack": False,
                "episode_any_deadline_miss": False,
                "budget_failure_ticks": 0,
                "maximum_budget_error": 1e-16,
                "residual_available_count": 10,
                "residual_unavailable_count": 0,
                "residual_invalid_count": 0,
                "residual_available_rate": 1.0,
                "residual_invalid_rate": 0.0,
                "residual_gate_pass_rate": 1.0,
                "residual_gate_fail_rate": 0.0,
                "residual_gate_unknown_rate": 0.0,
                "primal_residual_p50": 1e-6,
                "primal_residual_p95": 2e-6,
                "primal_residual_p99": 3e-6,
                "dual_residual_p50": 1e-6,
                "dual_residual_p95": 2e-6,
                "dual_residual_p99": 3e-6,
                "enforced_profile_count": 10,
                "missing_enforced_profile_count": 0,
                "post_solve_diagnostic_profile_count": 0,
            }
            path.write_text(json.dumps(trial) + "\n", encoding="utf-8")
            upgraded = load_trial_checkpoint(directory)[0]
            self.assertEqual(upgraded.mode, "joint_uniform")
            self.assertEqual(upgraded.requested_mode, "joint")
            self.assertTrue(upgraded.legacy_mode_alias_used)


def _protocol(modes=("deterministic", "joint_uniform")):
    from quadrotor_mpc.application.validation.monte_carlo import (
        NativeMonteCarloProtocol,
        NoiseLevel,
    )

    return NativeMonteCarloProtocol(
        name="mode-guard",
        base_config_path=CODE_ROOT / "config" / "mujoco_native_ccmpc.yaml",
        output_dir=CODE_ROOT / "outputs" / "test",
        modes=modes,
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


def _trial(mode: str):
    from quadrotor_mpc.application.validation.monte_carlo import (
        NativeTrialResult,
    )

    risk_semantics = "disabled" if mode == "deterministic" else "joint"
    risk_allocation_method = "none" if mode == "deterministic" else "uniform"
    return NativeTrialResult(
        noise_label="nominal",
        covariance_scale=1.0,
        mode=mode,
        seed=1,
        expected_steps=10,
        completed_steps=10,
        completed=True,
        termination_reason="completed",
        success=True,
        collision=False,
        numerical_failure=False,
        final_error_m=0.1,
        min_clearance_m=0.5,
        path_length_m=4.0,
        tracking_rmse_m=0.2,
        control_effort=0.01,
        estimation_position_rmse_m=0.03,
        mean_solver_ms=25.0,
        p95_solver_ms=35.0,
        p99_solver_ms=40.0,
        max_solver_ms=45.0,
        primary_success_rate=1.0,
        accepted_solution_rate=1.0,
        deadline_miss_rate=0.0,
        positive_slack_ticks=0,
        positive_slack_rate=0.0,
        max_slack_m=0.0,
        chance_violation_rate=0.0,
        fallback_ticks=0,
        fallback_rate=0.0,
        guarantee_eligible_ticks=10,
        guarantee_eligible_rate=1.0,
        guarantee_eligible_episode=True,
        horizon_eligible_tick_count=10,
        horizon_eligible_tick_rate=1.0,
        horizon_ineligible_reason_counts={},
        episode_all_ticks_horizon_eligible=True,
        episode_any_fallback=False,
        episode_any_positive_slack=False,
        episode_any_deadline_miss=False,
        budget_failure_ticks=0,
        maximum_budget_error=1e-16,
        residual_available_count=10,
        residual_unavailable_count=0,
        residual_invalid_count=0,
        residual_available_rate=1.0,
        residual_invalid_rate=0.0,
        residual_gate_pass_rate=1.0,
        residual_gate_fail_rate=0.0,
        residual_gate_unknown_rate=0.0,
        primal_residual_p50=1e-6,
        primal_residual_p95=2e-6,
        primal_residual_p99=3e-6,
        dual_residual_p50=1e-6,
        dual_residual_p95=2e-6,
        dual_residual_p99=3e-6,
        enforced_profile_count=10,
        missing_enforced_profile_count=0,
        post_solve_diagnostic_profile_count=0,
        requested_mode=mode,
        risk_semantics=risk_semantics,
        risk_allocation_method=risk_allocation_method,
        allocator_config_hash=None if mode == "deterministic" else "a" * 64,
        risk_provenance_status="AVAILABLE",
    )


class AggregationGuardTests(unittest.TestCase):
    def test_canonical_modes_aggregate_together(self):
        from quadrotor_mpc.application.validation.monte_carlo import (
            aggregate_native_trials,
        )

        aggregate = aggregate_native_trials(
            [_trial("deterministic"), _trial("joint_uniform")],
            _protocol(),
            controller_period_ms=50.0,
        )
        self.assertIn(
            "joint_uniform",
            aggregate["noise_levels"]["nominal"]["controllers"],
        )
        self.assertEqual(
            aggregate["controller_modes"],
            ["deterministic", "joint_uniform"],
        )

    def test_synthetic_joint_adaptive_is_refused_without_explicit_comparison(self):
        from quadrotor_mpc.application.validation.monte_carlo import (
            aggregate_native_trials,
        )

        with self.assertRaisesRegex(ValueError, "outside the protocol"):
            aggregate_native_trials(
                [_trial("joint_adaptive")],
                _protocol(),
                controller_period_ms=50.0,
            )


if __name__ == "__main__":
    unittest.main()
