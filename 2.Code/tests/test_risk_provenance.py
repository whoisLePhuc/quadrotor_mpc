"""Regression tests for Monte Carlo mode and risk provenance."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from quadrotor_mpc.application.validation.monte_carlo import (
    NativeMonteCarloProtocol,
    NativeMonteCarloRunner,
    NoiseLevel,
    aggregate_native_trials,
    effective_native_config,
    run_native_trial_batch,
    summarize_native_trial,
)
from quadrotor_mpc.interfaces.desktop.viewer import load_native_mujoco_config

CODE_ROOT = Path(__file__).resolve().parents[1]


def _config(*, mode: str = "joint_uniform"):
    base = load_native_mujoco_config(CODE_ROOT / "config" / "mujoco_native_ccmpc.yaml")
    configured = effective_native_config(
        base,
        mode=mode,
        covariance_scale=1.0,
        seed=7,
    )
    mapping = configured.to_mapping()
    mapping["simulation"]["duration_s"] = 0.10
    return type(configured).from_mapping(mapping)


def _result(
    *,
    risk_semantics: tuple[str, ...] = ("joint", "joint"),
    risk_allocation_method: tuple[str, ...] = ("uniform", "uniform"),
) -> dict[str, object]:
    positions = np.array([[0.0, 0.0, 1.0], [3.0, 2.0, 2.5]])
    return {
        "pos": positions,
        "u": np.zeros((2, 4)),
        "clearance": np.array([0.4, 0.3]),
        "solver_time_ms": np.array([10.0, 11.0]),
        "primary_solver_success": np.ones(2, dtype=bool),
        "solution_accepted": np.ones(2, dtype=bool),
        "deadline_missed": np.zeros(2, dtype=bool),
        "fallback_active": np.zeros(2, dtype=bool),
        "safety_assurance_status": np.array(
            ["HORIZON_GUARANTEE_ELIGIBLE", "HORIZON_GUARANTEE_ELIGIBLE"]
        ),
        "horizon_assurance_eligible": np.ones(2, dtype=bool),
        "horizon_assurance_reason": np.array(["eligible", "eligible"]),
        "horizon_assurance_failed_checks": np.array([[], []], dtype=object),
        "risk_budget_status": np.array(["BUDGET_OK", "BUDGET_OK"]),
        "risk_budget_total": np.array([0.1, 0.1]),
        "risk_budget_allocated": np.array([0.1, 0.1]),
        "risk_semantics": np.asarray(risk_semantics, dtype=str),
        "risk_allocation_method": np.asarray(risk_allocation_method, dtype=str),
        "estimated_state": np.column_stack([positions, np.zeros((2, 10))]),
        "collided": False,
        "obstacle_collided": False,
        "ground_collided": False,
        "termination_reason": "completed",
    }


def _protocol(*, requested_mode: str = "joint_uniform") -> NativeMonteCarloProtocol:
    return NativeMonteCarloProtocol(
        name="risk-provenance",
        base_config_path=CODE_ROOT / "config" / "mujoco_native_ccmpc.yaml",
        output_dir=CODE_ROOT / "outputs" / "test",
        modes=("joint_uniform",),
        requested_modes=(requested_mode,),
        mode_aliases_used=requested_mode == "joint",
        noise_levels=(NoiseLevel("nominal", 1.0),),
        trials=2,
        first_seed=7,
        confidence_level=0.95,
        minimum_trials_for_claim=1,
        empirical_collision_rate_limit=0.10,
        require_zero_positive_slack=True,
        require_zero_fallback=True,
        require_zero_budget_failures=True,
        timing_percentile=0.99,
    )


class TrialRiskProvenanceTests(unittest.TestCase):
    def test_effective_protocol_mapping_keeps_requested_and_canonical_modes(self):
        mapping = _protocol(requested_mode="joint").to_mapping()

        self.assertEqual(mapping["modes"], ["joint"])
        self.assertEqual(mapping["canonical_modes"], ["joint_uniform"])
        self.assertTrue(mapping["mode_aliases_used"])

    def test_summary_preserves_alias_and_controller_reported_risk_metadata(self):
        trial = summarize_native_trial(
            _result(),
            _config(),
            mode="joint_uniform",
            requested_mode="joint",
            noise_label="nominal",
            covariance_scale=1.0,
            seed=7,
        )

        self.assertEqual(trial.mode, "joint_uniform")
        self.assertEqual(trial.requested_mode, "joint")
        self.assertTrue(trial.legacy_mode_alias_used)
        self.assertEqual(trial.risk_semantics, "joint")
        self.assertEqual(trial.risk_allocation_method, "uniform")
        self.assertRegex(trial.allocator_config_hash or "", r"^[0-9a-f]{64}$")
        self.assertEqual(trial.risk_provenance_status, "AVAILABLE")
        self.assertEqual(trial.risk_provenance_unavailable_ticks, 0)

    def test_summary_records_partial_unavailability_without_inventing_tick_metadata(self):
        trial = summarize_native_trial(
            _result(
                risk_semantics=("joint", ""),
                risk_allocation_method=("uniform", ""),
            ),
            _config(),
            mode="joint_uniform",
            noise_label="nominal",
            covariance_scale=1.0,
            seed=7,
        )

        self.assertEqual(trial.risk_semantics, "joint")
        self.assertEqual(trial.risk_allocation_method, "uniform")
        self.assertEqual(trial.risk_provenance_status, "PARTIAL_UNAVAILABLE")
        self.assertEqual(trial.risk_provenance_unavailable_ticks, 1)

    def test_summary_rejects_risk_metadata_that_changes_between_ticks(self):
        with self.assertRaisesRegex(ValueError, "risk_semantics.*changes"):
            summarize_native_trial(
                _result(risk_semantics=("joint", "individual")),
                _config(),
                mode="joint_uniform",
                noise_label="nominal",
                covariance_scale=1.0,
                seed=7,
            )

    def test_summary_rejects_controller_metadata_inconsistent_with_mode(self):
        with self.assertRaisesRegex(ValueError, "joint_uniform.*risk_semantics"):
            summarize_native_trial(
                _result(risk_semantics=("individual", "individual")),
                _config(),
                mode="joint_uniform",
                noise_label="nominal",
                covariance_scale=1.0,
                seed=7,
            )

    def test_runner_threads_requested_mode_into_trial_artifact(self):
        runner = NativeMonteCarloRunner(_config())
        with (
            patch(
                "quadrotor_mpc.application.validation.monte_carlo._build_controller",
                return_value=object(),
            ),
            patch(
                "quadrotor_mpc.application.native.runtime.run_coupled_simulation",
                return_value=_result(),
            ),
        ):
            trial = runner.run_trial(
                mode="joint_uniform",
                requested_mode="joint",
                noise_level=NoiseLevel("nominal", 1.0),
                seed=7,
            )

        self.assertEqual(trial.requested_mode, "joint")
        self.assertTrue(trial.legacy_mode_alias_used)

    def test_process_batch_threads_requested_mode_into_runner(self):
        expected = summarize_native_trial(
            _result(),
            _config(),
            mode="joint_uniform",
            requested_mode="joint",
            noise_label="nominal",
            covariance_scale=1.0,
            seed=7,
        )
        with patch.object(
            NativeMonteCarloRunner,
            "run_trial",
            return_value=expected,
        ) as run_trial:
            rows = run_native_trial_batch(
                _config().to_mapping(),
                mode="joint_uniform",
                requested_mode="joint",
                noise_label="nominal",
                covariance_scale=1.0,
                seeds=(7,),
            )

        self.assertEqual(rows[0]["requested_mode"], "joint")
        run_trial.assert_called_once_with(
            mode="joint_uniform",
            requested_mode="joint",
            noise_level=NoiseLevel("nominal", 1.0),
            seed=7,
        )


class AggregationRiskProvenanceTests(unittest.TestCase):
    def _trial(self, *, requested_mode: str = "joint_uniform", seed: int = 7):
        return summarize_native_trial(
            _result(),
            _config(),
            mode="joint_uniform",
            requested_mode=requested_mode,
            noise_label="nominal",
            covariance_scale=1.0,
            seed=seed,
        )

    def test_aggregate_exposes_validated_risk_provenance(self):
        aggregate = aggregate_native_trials(
            [self._trial(requested_mode="joint")],
            _protocol(requested_mode="joint"),
            controller_period_ms=50.0,
        )

        provenance = aggregate["noise_levels"]["nominal"]["controllers"]["joint_uniform"][
            "risk_provenance"
        ]
        self.assertEqual(provenance["controller_mode"], "joint_uniform")
        self.assertEqual(provenance["risk_semantics"], "joint")
        self.assertEqual(provenance["risk_allocation_method"], "uniform")
        self.assertRegex(provenance["allocator_config_hash"], r"^[0-9a-f]{64}$")
        self.assertEqual(provenance["telemetry_status"], "AVAILABLE")
        self.assertEqual(provenance["unavailable_tick_count"], 0)

    def test_partial_provenance_is_retained_but_blocks_execution_gate(self):
        partial = summarize_native_trial(
            _result(
                risk_semantics=("joint", ""),
                risk_allocation_method=("uniform", ""),
            ),
            _config(),
            mode="joint_uniform",
            noise_label="nominal",
            covariance_scale=1.0,
            seed=7,
        )
        aggregate = aggregate_native_trials(
            [partial],
            _protocol(),
            controller_period_ms=50.0,
        )

        summary = aggregate["noise_levels"]["nominal"]["controllers"]["joint_uniform"]
        self.assertEqual(
            summary["risk_provenance"]["telemetry_status"],
            "PARTIAL_UNAVAILABLE",
        )
        self.assertEqual(summary["gates"]["risk_provenance"], "FAIL")
        self.assertEqual(summary["gates"]["execution_integrity"], "FAIL")
        self.assertEqual(
            summary["gates"]["claim_status"],
            "BLOCKED_RISK_PROVENANCE_UNAVAILABLE",
        )

    def test_aggregate_rejects_mode_allocation_mismatch(self):
        invalid = replace(self._trial(), risk_allocation_method="adaptive")
        with self.assertRaisesRegex(ValueError, "risk_allocation_method"):
            aggregate_native_trials(
                [invalid],
                _protocol(),
                controller_period_ms=50.0,
            )

    def test_aggregate_rejects_mixed_allocator_hashes(self):
        first = self._trial(seed=7)
        second = replace(
            self._trial(seed=8),
            allocator_config_hash="f" * 64,
        )
        with self.assertRaisesRegex(ValueError, "allocator_config_hash"):
            aggregate_native_trials(
                [first, second],
                _protocol(),
                controller_period_ms=50.0,
            )

    def test_aggregate_rejects_trial_alias_provenance_mismatch(self):
        wrong = self._trial(requested_mode="joint_uniform")
        with self.assertRaisesRegex(ValueError, "requested_mode"):
            aggregate_native_trials(
                [wrong],
                _protocol(requested_mode="joint"),
                controller_period_ms=50.0,
            )


if __name__ == "__main__":
    unittest.main()
