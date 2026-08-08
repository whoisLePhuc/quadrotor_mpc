from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from quadrotor_mpc.application.validation.monte_carlo import (
    NativeMonteCarloProtocol,
    NativeTrialResult,
    NoiseLevel,
    aggregate_native_trials,
    append_trial_checkpoint,
    create_validation_directory,
    effective_native_config,
    finalize_validation_artifacts,
    load_native_monte_carlo_protocol,
    load_trial_checkpoint,
    source_provenance,
    summarize_native_trial,
    wilson_interval,
)
from quadrotor_mpc.interfaces.desktop.viewer import load_native_mujoco_config

ROOT = Path(__file__).resolve().parents[1]


def protocol(*, trials: int = 30) -> NativeMonteCarloProtocol:
    return NativeMonteCarloProtocol(
        name="unit-native-mc",
        base_config_path=ROOT / "config" / "mujoco_native_ccmpc.yaml",
        output_dir=ROOT / "outputs" / "test",
        modes=("deterministic", "joint"),
        noise_levels=(NoiseLevel("nominal", 1.0),),
        trials=trials,
        first_seed=10,
        confidence_level=0.95,
        minimum_trials_for_claim=30,
        empirical_collision_rate_limit=0.10,
        require_zero_positive_slack=True,
        require_zero_fallback=True,
        require_zero_budget_failures=True,
        timing_percentile=0.99,
    )


def trial(
    mode: str,
    seed: int,
    *,
    collision: bool = False,
    slack: bool = False,
    fallback: bool = False,
    p99: float = 40.0,
) -> NativeTrialResult:
    return NativeTrialResult(
        noise_label="nominal",
        covariance_scale=1.0,
        mode=mode,
        seed=seed,
        expected_steps=10,
        completed_steps=10,
        completed=True,
        termination_reason="completed",
        success=not collision,
        collision=collision,
        numerical_failure=False,
        final_error_m=0.1,
        min_clearance_m=0.3 if not collision else -0.01,
        path_length_m=4.0,
        tracking_rmse_m=0.2,
        control_effort=0.01,
        estimation_position_rmse_m=0.03,
        mean_solver_ms=25.0,
        p95_solver_ms=35.0,
        p99_solver_ms=p99,
        max_solver_ms=45.0,
        primary_success_rate=1.0,
        accepted_solution_rate=0.9 if fallback else 1.0,
        deadline_miss_rate=0.1 if fallback else 0.0,
        positive_slack_ticks=1 if slack else 0,
        positive_slack_rate=0.1 if slack else 0.0,
        max_slack_m=0.01 if slack else 0.0,
        chance_violation_rate=0.1 if slack else 0.0,
        fallback_ticks=1 if fallback else 0,
        fallback_rate=0.1 if fallback else 0.0,
        guarantee_eligible_ticks=9 if slack else 10,
        guarantee_eligible_rate=0.9 if slack else 1.0,
        guarantee_eligible_episode=mode != "deterministic" and not slack and not fallback,
        horizon_eligible_tick_count=9 if slack else 10,
        horizon_eligible_tick_rate=0.9 if slack else 1.0,
        horizon_ineligible_reason_counts={},
        episode_all_ticks_horizon_eligible=(
            mode != "deterministic" and not slack and not fallback
        ),
        episode_any_fallback=fallback,
        episode_any_positive_slack=slack,
        episode_any_deadline_miss=fallback,
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
    )


class NativeMonteCarloConfigurationTests(unittest.TestCase):
    def test_repository_protocol_is_valid_and_resolves_base_config(self):
        loaded = load_native_monte_carlo_protocol(
            ROOT / "config" / "native_monte_carlo.yaml"
        )
        self.assertEqual(loaded.trials, 50)
        self.assertEqual(loaded.modes, ("deterministic", "individual", "joint"))
        self.assertEqual(
            [level.covariance_scale for level in loaded.noise_levels],
            [0.25, 1.0, 4.0],
        )
        self.assertTrue(loaded.base_config_path.exists())

    def test_covariance_scale_uses_square_root_for_standard_deviation(self):
        base = load_native_mujoco_config(
            ROOT / "config" / "mujoco_native_ccmpc.yaml"
        )
        scaled = effective_native_config(
            base,
            mode="joint",
            covariance_scale=4.0,
            seed=44,
        )
        self.assertAlmostEqual(
            scaled.estimation.sensor.position_std_m,
            2.0 * base.estimation.sensor.position_std_m,
        )
        self.assertAlmostEqual(
            scaled.covariance_propagation.acceleration_process_std_mps2,
            2.0
            * base.covariance_propagation.acceleration_process_std_mps2,
        )
        self.assertEqual(scaled.estimation.seed, 44)

    def test_controller_modes_preserve_explicit_risk_semantics(self):
        base = load_native_mujoco_config(
            ROOT / "config" / "mujoco_native_ccmpc.yaml"
        )
        deterministic = effective_native_config(
            base, mode="deterministic", covariance_scale=1.0, seed=1
        )
        individual = effective_native_config(
            base, mode="individual", covariance_scale=1.0, seed=1
        )
        joint = effective_native_config(
            base, mode="joint", covariance_scale=1.0, seed=1
        )
        self.assertFalse(deterministic.chance_constraints.enabled)
        self.assertFalse(deterministic.covariance_propagation.enabled)
        self.assertEqual(
            individual.chance_constraints.risk_budget.semantics,
            "individual",
        )
        self.assertEqual(joint.chance_constraints.risk_budget.semantics, "joint")


class NativeMonteCarloMetricTests(unittest.TestCase):
    def test_wilson_interval_is_bounded_and_nonzero_for_zero_events(self):
        low, high = wilson_interval(0, 50)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        self.assertLess(high, 0.10)

    def test_native_result_reduction_keeps_slack_fallback_and_budget_separate(self):
        base = load_native_mujoco_config(
            ROOT / "config" / "mujoco_native_ccmpc.yaml"
        )
        config = effective_native_config(
            base, mode="joint", covariance_scale=1.0, seed=4
        )
        steps = 2
        result = {
            "pos": np.array([[0.0, 0.0, 1.0], [3.0, 2.0, 2.5]]),
            "u": np.zeros((steps, 4)),
            "clearance": np.array([0.3, 0.2]),
            "solver_time_ms": np.array([30.0, 60.0]),
            "primary_solver_success": np.ones(steps, dtype=bool),
            "solution_accepted": np.array([True, False]),
            "deadline_missed": np.array([False, True]),
            "fallback_active": np.array([False, True]),
            "safety_assurance_status": np.array(
                ["GUARANTEE_ELIGIBLE", "NOT_GUARANTEED_FALLBACK_ACTIVE"]
            ),
            "horizon_assurance_eligible": np.array([True, False]),
            "horizon_assurance_reason": np.array(["eligible", "fallback_active"]),
            "horizon_assurance_failed_checks": np.array(
                [[], ["fallback_active"]], dtype=object
            ),
            "slack_horizon": np.array([[[0.0]], [[0.02]]]),
            "chance_residual_horizon": np.array([[[0.1]], [[-0.02]]]),
            "risk_budget_status": np.array(["BUDGET_OK", "BUDGET_OK"]),
            "risk_budget_total": np.array([0.1, 0.1]),
            "risk_budget_allocated": np.array([0.1, 0.1]),
            "estimated_state": np.column_stack(
                [np.array([[0.0, 0.0, 1.0], [3.0, 2.0, 2.5]]), np.zeros((2, 10))]
            ),
            "collided": False,
            "termination_reason": "completed",
        }
        # Match the short synthetic episode rather than the 10-second source.
        mapping = config.to_mapping()
        mapping["simulation"]["duration_s"] = 0.1
        short = type(config).from_mapping(mapping)
        reduced = summarize_native_trial(
            result,
            short,
            mode="joint",
            noise_label="nominal",
            covariance_scale=1.0,
            seed=4,
        )
        self.assertEqual(reduced.positive_slack_ticks, 1)
        self.assertEqual(reduced.fallback_ticks, 1)
        self.assertEqual(reduced.budget_failure_ticks, 0)
        self.assertFalse(reduced.guarantee_eligible_episode)

    def test_positive_slack_blocks_claim_despite_zero_collisions(self):
        items = [
            trial("deterministic", seed)
            for seed in range(10, 60)
        ] + [
            trial("joint", seed, slack=True)
            for seed in range(10, 60)
        ]
        aggregate = aggregate_native_trials(
            items,
            protocol(trials=50),
            controller_period_ms=50.0,
        )
        joint = aggregate["noise_levels"]["nominal"]["controllers"]["joint"]
        self.assertEqual(joint["collision_rate"]["events"], 0)
        self.assertEqual(joint["gates"]["claim_status"], "BLOCKED_POSITIVE_SLACK")
        self.assertEqual(
            aggregate["overall"]["stage_status"],
            "VALIDATED_WITH_LIMITATIONS",
        )

    def test_paired_delta_uses_only_matching_seeds(self):
        items = [
            trial("deterministic", seed)
            for seed in range(10, 40)
        ] + [
            trial("joint", seed)
            for seed in range(10, 39)
        ]
        aggregate = aggregate_native_trials(
            items,
            protocol(),
            controller_period_ms=50.0,
        )
        paired = aggregate["noise_levels"]["nominal"]["paired_comparisons"][
            "joint_minus_deterministic"
        ]
        self.assertEqual(len(paired["paired_seeds"]), 29)
        self.assertAlmostEqual(
            paired["metrics"]["min_clearance_m"]["mean"],
            0.0,
        )


class NativeMonteCarloArtifactTests(unittest.TestCase):
    def test_release_campaign_refuses_dirty_source(self):
        base = load_native_mujoco_config(
            ROOT / "config" / "mujoco_native_ccmpc.yaml"
        )
        configured = protocol(trials=1)
        dirty = {
            "status": "DIRTY_GIT_SNAPSHOT",
            "git_commit": "abc123",
            "git_branch": "feature",
            "git_clean": False,
            "source_snapshot_sha256": "f" * 64,
            "source_file_count": 12,
            "distribution_version": None,
        }
        with tempfile.TemporaryDirectory() as temporary:
            configured = NativeMonteCarloProtocol(
                **{
                    **{
                        field: getattr(configured, field)
                        for field in configured.__dataclass_fields__
                    },
                    "output_dir": Path(temporary),
                }
            )
            with (
                patch("quadrotor_mpc.application.validation.monte_carlo.source_provenance", return_value=dirty),
                self.assertRaisesRegex(RuntimeError, "clean Git source"),
            ):
                create_validation_directory(configured, base, run_id="dirty")

    def test_source_provenance_has_release_fields(self):
        provenance = source_provenance(ROOT)
        self.assertIn(
            provenance["status"],
            {
                "CLEAN_GIT_COMMIT",
                "DIRTY_GIT_SNAPSHOT",
                "INSTALLED_DISTRIBUTION",
                "EDITABLE_DISTRIBUTION",
                "UNVERIFIED_DISTRIBUTION",
                "UNVERIFIED_SOURCE_TREE",
            },
        )
        self.assertIn("source_snapshot_sha256", provenance)
        self.assertIn("git_clean", provenance)

    def test_release_campaign_refuses_editable_distribution(self):
        base = load_native_mujoco_config(
            ROOT / "config" / "mujoco_native_ccmpc.yaml"
        )
        configured = protocol(trials=1)
        editable = {
            "status": "EDITABLE_DISTRIBUTION",
            "git_commit": None,
            "git_branch": None,
            "git_clean": None,
            "source_snapshot_sha256": "a" * 64,
            "source_file_count": 12,
            "distribution_version": "2.0.1",
        }
        with tempfile.TemporaryDirectory() as temporary:
            configured = NativeMonteCarloProtocol(
                **{
                    **{
                        field: getattr(configured, field)
                        for field in configured.__dataclass_fields__
                    },
                    "output_dir": Path(temporary),
                }
            )
            with (
                patch("quadrotor_mpc.application.validation.monte_carlo.source_provenance", return_value=editable),
                self.assertRaisesRegex(RuntimeError, "non-editable distribution"),
            ):
                create_validation_directory(configured, base, run_id="editable")

    def test_checkpoint_round_trip_and_complete_bundle(self):
        base = load_native_mujoco_config(
            ROOT / "config" / "mujoco_native_ccmpc.yaml"
        )
        configured = protocol(trials=1)
        items = [trial("deterministic", 10), trial("joint", 10)]
        with tempfile.TemporaryDirectory() as temporary:
            configured = NativeMonteCarloProtocol(
                **{
                    **{
                        field: getattr(configured, field)
                        for field in configured.__dataclass_fields__
                    },
                    "output_dir": Path(temporary),
                }
            )
            dirty = {
                "status": "DIRTY_GIT_SNAPSHOT",
                "git_commit": "abc123",
                "git_branch": "feature",
                "git_clean": False,
                "source_snapshot_sha256": "e" * 64,
                "source_file_count": 12,
                "distribution_version": None,
            }
            with patch("quadrotor_mpc.application.validation.monte_carlo.source_provenance", return_value=dirty):
                directory = create_validation_directory(
                    configured,
                    base,
                    command=["unit-test"],
                    run_id="unit-native-mc",
                    allow_dirty_source=True,
                )
            append_trial_checkpoint(directory, items[0])
            self.assertEqual(load_trial_checkpoint(directory), items[:1])
            artifacts = finalize_validation_artifacts(
                directory,
                items,
                configured,
                base,
            )
            self.assertEqual(load_trial_checkpoint(directory), items)
            aggregate = json.loads(
                artifacts["aggregate"].read_text(encoding="utf-8")
            )
            self.assertEqual(aggregate["overall"]["sample_size"], "INSUFFICIENT")
            self.assertEqual(
                aggregate["overall"]["release_provenance"],
                "FAIL_DIRTY_SOURCE",
            )
            manifest = __import__("yaml").safe_load(
                (directory / "manifest.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["source"]["status"], "DIRTY_GIT_SNAPSHOT")
            self.assertTrue(artifacts["plot"].exists())
            self.assertTrue(artifacts["report"].exists())


if __name__ == "__main__":
    unittest.main()
