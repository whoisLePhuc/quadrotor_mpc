"""Unit tests for the horizon assurance classifier.

Covers the truth table from the assurance semantics specification: every gate
is exercised individually against a valid joint fixture, plus boundary values,
multiple-failure precedence, and the regression proving that slack + deadline
alone are not sufficient for eligibility.
"""

from __future__ import annotations

import math
import unittest
from typing import Any

import numpy as np

from quadrotor_mpc.control.nmpc.assurance import (
    ASSURANCE_SCHEMA_VERSION,
    LEGACY_STATUS_MAP,
    HorizonAssuranceDecision,
    HorizonAssuranceInput,
    HorizonAssuranceStatus,
    classify_horizon_assurance,
    normalize_risk_semantics,
)

ELIGIBLE = HorizonAssuranceStatus.HORIZON_GUARANTEE_ELIGIBLE
DETERMINISTIC = HorizonAssuranceStatus.NOT_APPLICABLE_DETERMINISTIC
INDIVIDUAL = HorizonAssuranceStatus.NOT_JOINT_GUARANTEE_INDIVIDUAL_RISK
BUDGET_INVALID = HorizonAssuranceStatus.NOT_GUARANTEED_RISK_BUDGET_INVALID
SOLVER_FAILURE = HorizonAssuranceStatus.NOT_GUARANTEED_PRIMARY_SOLVER_FAILURE
RESIDUAL_UNAVAILABLE = HorizonAssuranceStatus.NOT_GUARANTEED_RESIDUAL_UNAVAILABLE
RESIDUAL_INVALID = HorizonAssuranceStatus.NOT_GUARANTEED_RESIDUAL_INVALID
POSITIVE_SLACK = HorizonAssuranceStatus.NOT_GUARANTEED_POSITIVE_SLACK
DEADLINE_MISS = HorizonAssuranceStatus.NOT_GUARANTEED_DEADLINE_MISS
FALLBACK_ACTIVE = HorizonAssuranceStatus.NOT_GUARANTEED_FALLBACK_ACTIVE
INVALID_NUMERICS = HorizonAssuranceStatus.NOT_GUARANTEED_INVALID_NUMERICS


def valid_joint_input(**overrides: Any) -> HorizonAssuranceInput:
    return HorizonAssuranceInput(
        risk_semantics=str(overrides.get("risk_semantics", "joint")),
        risk_budget_status=(
            None
            if "risk_budget_status" in overrides and overrides["risk_budget_status"] is None
            else str(overrides.get("risk_budget_status", "BUDGET_OK"))
        ),
        primary_solver_success=bool(
            overrides.get("primary_solver_success", True)
        ),
        residual_status=str(overrides.get("residual_status", "AVAILABLE")),
        primal_residual=(
            None
            if "primal_residual" in overrides and overrides["primal_residual"] is None
            else float(overrides.get("primal_residual", 1e-7))
        ),
        dual_residual=(
            None
            if "dual_residual" in overrides and overrides["dual_residual"] is None
            else float(overrides.get("dual_residual", 2e-7))
        ),
        primal_residual_tolerance=float(
            overrides.get("primal_residual_tolerance", 1e-3)
        ),
        dual_residual_tolerance=float(
            overrides.get("dual_residual_tolerance", 1e-3)
        ),
        maximum_slack=float(overrides.get("maximum_slack", 0.0)),
        slack_tolerance=float(overrides.get("slack_tolerance", 1e-6)),
        deadline_missed=bool(overrides.get("deadline_missed", False)),
        fallback_active=bool(overrides.get("fallback_active", False)),
    )


def decision(data: HorizonAssuranceInput) -> HorizonAssuranceDecision:
    return classify_horizon_assurance(data)


class HorizonAssuranceEligibilityTests(unittest.TestCase):
    def test_joint_solution_is_horizon_eligible_only_when_all_gates_pass(self):
        result = decision(valid_joint_input())
        self.assertTrue(result.eligible)
        self.assertEqual(result.status, ELIGIBLE)
        self.assertEqual(result.reason_code, "eligible")
        self.assertEqual(result.failed_checks, ())

    def test_deterministic_solution_is_not_applicable_to_joint_assurance(self):
        result = decision(valid_joint_input(risk_semantics="deterministic"))
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, DETERMINISTIC)

    def test_disabled_risk_semantics_is_treated_as_deterministic(self):
        result = decision(valid_joint_input(risk_semantics="disabled"))
        self.assertEqual(result.status, DETERMINISTIC)

    def test_individual_risk_is_not_labeled_as_joint_guarantee(self):
        result = decision(valid_joint_input(risk_semantics="individual"))
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, INDIVIDUAL)

    def test_joint_uniform_maps_to_joint_semantics(self):
        result = decision(valid_joint_input(risk_semantics="joint_uniform"))
        self.assertTrue(result.eligible)

    def test_invalid_risk_budget_blocks_horizon_eligibility(self):
        result = decision(valid_joint_input(risk_budget_status="BUDGET_EXCEEDED"))
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, BUDGET_INVALID)

    def test_primary_solver_failure_blocks_horizon_eligibility(self):
        result = decision(valid_joint_input(primary_solver_success=False))
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, SOLVER_FAILURE)

    def test_unavailable_residual_is_not_treated_as_zero(self):
        result = decision(valid_joint_input(residual_status="UNAVAILABLE"))
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, RESIDUAL_UNAVAILABLE)
        self.assertIn("residual_unavailable", result.failed_checks)

    def test_unavailable_residual_blocks_even_with_zero_residual_fields(self):
        result = decision(
            valid_joint_input(
                residual_status="UNAVAILABLE",
                primal_residual=0.0,
                dual_residual=0.0,
            )
        )
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, RESIDUAL_UNAVAILABLE)

    def test_nonfinite_residual_blocks_horizon_eligibility(self):
        result = decision(valid_joint_input(primal_residual=math.nan))
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, RESIDUAL_INVALID)

    def test_none_residual_blocks_horizon_eligibility(self):
        result = decision(valid_joint_input(primal_residual=None))
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, RESIDUAL_INVALID)

    def test_residual_above_tolerance_blocks_horizon_eligibility(self):
        result = decision(valid_joint_input(primal_residual=2e-3))
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, RESIDUAL_INVALID)

    def test_slack_above_tolerance_blocks_horizon_eligibility(self):
        result = decision(valid_joint_input(maximum_slack=2e-6))
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, POSITIVE_SLACK)

    def test_deadline_miss_blocks_horizon_eligibility(self):
        result = decision(valid_joint_input(deadline_missed=True))
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, DEADLINE_MISS)

    def test_fallback_command_blocks_horizon_eligibility(self):
        result = decision(valid_joint_input(fallback_active=True))
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, FALLBACK_ACTIVE)

    def test_nonfinite_slack_blocks_horizon_eligibility(self):
        result = decision(valid_joint_input(maximum_slack=math.nan))
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, INVALID_NUMERICS)

    def test_unknown_semantics_is_never_eligible(self):
        result = decision(valid_joint_input(risk_semantics="teleporting"))
        self.assertFalse(result.eligible)
        self.assertIn("risk_semantics_unknown", result.failed_checks)


class HorizonAssuranceBoundaryTests(unittest.TestCase):
    def test_slack_equal_to_tolerance_is_eligible(self):
        result = decision(valid_joint_input(maximum_slack=1e-6))
        self.assertTrue(result.eligible)

    def test_slack_just_above_tolerance_is_not_eligible(self):
        result = decision(
            valid_joint_input(maximum_slack=float(np.nextafter(1e-6, np.inf)))
        )
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, POSITIVE_SLACK)

    def test_primal_residual_equal_to_tolerance_is_eligible(self):
        result = decision(valid_joint_input(primal_residual=1e-3))
        self.assertTrue(result.eligible)

    def test_primal_residual_just_above_tolerance_is_not_eligible(self):
        result = decision(
            valid_joint_input(
                primal_residual=float(np.nextafter(1e-3, np.inf))
            )
        )
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, RESIDUAL_INVALID)


class HorizonAssurancePrecedenceTests(unittest.TestCase):
    def test_multiple_failures_keep_all_failed_checks(self):
        result = decision(
            valid_joint_input(
                primary_solver_success=False,
                deadline_missed=True,
                fallback_active=True,
                maximum_slack=0.5,
            )
        )
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, SOLVER_FAILURE)
        self.assertIn("primary_solver_failure", result.failed_checks)
        self.assertIn("deadline_miss", result.failed_checks)
        self.assertIn("fallback_active", result.failed_checks)
        self.assertIn("positive_slack", result.failed_checks)

    def test_budget_invalid_precedes_solver_failure(self):
        result = decision(
            valid_joint_input(
                risk_budget_status="BUDGET_EXCEEDED",
                primary_solver_success=False,
            )
        )
        self.assertEqual(result.status, BUDGET_INVALID)

    def test_solver_failure_precedes_residual_unavailable(self):
        result = decision(
            valid_joint_input(
                primary_solver_success=False,
                residual_status="UNAVAILABLE",
            )
        )
        self.assertEqual(result.status, SOLVER_FAILURE)

    def test_residual_unavailable_precedes_deadline_miss(self):
        result = decision(
            valid_joint_input(
                residual_status="UNAVAILABLE",
                deadline_missed=True,
            )
        )
        self.assertEqual(result.status, RESIDUAL_UNAVAILABLE)

    def test_deadline_miss_precedes_fallback_active(self):
        result = decision(
            valid_joint_input(
                deadline_missed=True,
                fallback_active=True,
            )
        )
        self.assertEqual(result.status, DEADLINE_MISS)


class HorizonAssuranceRegressionTests(unittest.TestCase):
    def test_zero_slack_and_on_time_are_not_sufficient_for_guarantee_eligibility(self):
        result = decision(
            valid_joint_input(
                risk_semantics="deterministic",
                maximum_slack=0.0,
                deadline_missed=False,
            )
        )
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, DETERMINISTIC)

    def test_zero_slack_individual_mode_is_not_joint_eligible(self):
        result = decision(
            valid_joint_input(
                risk_semantics="individual",
                maximum_slack=0.0,
                deadline_missed=False,
            )
        )
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, INDIVIDUAL)

    def test_zero_slack_with_fallback_is_not_eligible(self):
        result = decision(
            valid_joint_input(
                maximum_slack=0.0,
                fallback_active=True,
            )
        )
        self.assertFalse(result.eligible)
        self.assertEqual(result.status, FALLBACK_ACTIVE)


class HorizonAssuranceContractTests(unittest.TestCase):
    def test_eligible_flag_matches_status(self):
        eligible = decision(valid_joint_input())
        self.assertTrue(eligible.eligible)
        self.assertTrue(eligible.status == ELIGIBLE)
        self.assertEqual(eligible.to_mapping()["horizon_assurance_eligible"], True)

    def test_decision_mapping_serializes_stable_strings(self):
        mapping = decision(valid_joint_input()).to_mapping()
        self.assertEqual(
            mapping["horizon_assurance_status"],
            "HORIZON_GUARANTEE_ELIGIBLE",
        )
        self.assertEqual(mapping["horizon_assurance_reason"], "eligible")
        self.assertEqual(mapping["horizon_assurance_failed_checks"], [])

    def test_failed_checks_serialize_without_loss(self):
        result = decision(
            valid_joint_input(
                deadline_missed=True,
                fallback_active=True,
            )
        )
        mapping = result.to_mapping()
        self.assertEqual(
            set(mapping["horizon_assurance_failed_checks"]),
            {"deadline_miss", "fallback_active"},
        )

    def test_legacy_status_is_mapped_to_unverified_not_eligible(self):
        self.assertEqual(
            LEGACY_STATUS_MAP["GUARANTEE_ELIGIBLE"],
            "LEGACY_GUARANTEE_ELIGIBLE_UNVERIFIED",
        )

    def test_legacy_map_never_produces_new_eligible_status(self):
        for mapped in LEGACY_STATUS_MAP.values():
            self.assertNotEqual(mapped, "HORIZON_GUARANTEE_ELIGIBLE")

    def test_schema_version_is_two(self):
        self.assertEqual(ASSURANCE_SCHEMA_VERSION, 2)


class RiskSemanticsNormalizationTests(unittest.TestCase):
    def test_normalize_disabled_to_deterministic(self):
        self.assertEqual(normalize_risk_semantics("disabled"), "deterministic")
        self.assertEqual(normalize_risk_semantics(""), "deterministic")

    def test_normalize_joint_variants(self):
        self.assertEqual(normalize_risk_semantics("joint"), "joint")
        self.assertEqual(normalize_risk_semantics("joint_uniform"), "joint")
        self.assertEqual(normalize_risk_semantics("joint_adaptive"), "joint")

    def test_normalize_preserves_unknown(self):
        self.assertEqual(normalize_risk_semantics("teleporting"), "teleporting")


class SupervisorAssuranceIntegrationTests(unittest.TestCase):
    """The supervisor must never emit HORIZON_GUARANTEE_ELIGIBLE for
    deterministic, individual or fallback-driven solutions."""

    def _supervised(self, semantics: str, *, fallback: bool = False):
        from dataclasses import replace

        from quadrotor_mpc.core.contracts import ControlSolution

        base = valid_joint_input(
            risk_semantics=semantics,
            fallback_active=fallback,
        )
        decision = classify_horizon_assurance(base)
        steps = 2
        solution = ControlSolution(
            command=np.zeros(4, dtype=float),
            nominal_states=np.tile(
                np.array(
                    [0.0, 0.0, 1.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                ).reshape(1, 13),
                (steps, 1),
            ),
            predicted_covariances=np.repeat(
                (np.eye(12) * 1e-3)[None, :, :], steps, axis=0
            ),
            chance_margins=np.zeros((steps, 1)),
            risk_allocations=np.zeros((steps, 1)),
            slacks=np.zeros((steps, 1)),
            solver_status="SOLVED",
            predicted_obstacle_covariances=np.repeat(
                (np.eye(6) * 1e-3)[None, None, :, :], steps, axis=0
            ),
            projected_uncertainties=np.zeros((steps, 1)),
            tightened_safety_radii=np.full((steps, 1), 0.8),
            risk_semantics=semantics,
            risk_budget_status="BUDGET_OK",
            primary_solver_status="Solve_Succeeded",
            primary_solver_success=True,
            primary_solver_primal_residual=1e-7,
            primary_solver_dual_residual=1e-7,
            residual_status="AVAILABLE",
            safety_assurance_status=decision.status.value,
            horizon_assurance_status=decision.status.value,
            horizon_assurance_eligible=decision.eligible,
            horizon_assurance_reason=decision.reason_code,
            horizon_assurance_failed_checks=decision.failed_checks,
        )
        if fallback:
            solution = replace(solution, fallback_active=True)
        return solution

    def test_deterministic_solution_never_emits_horizon_eligible(self):
        solution = self._supervised("deterministic")
        self.assertEqual(
            solution.horizon_assurance_status,
            "NOT_APPLICABLE_DETERMINISTIC",
        )
        self.assertFalse(solution.horizon_assurance_eligible)

    def test_individual_solution_never_emits_horizon_eligible(self):
        solution = self._supervised("individual")
        self.assertEqual(
            solution.horizon_assurance_status,
            "NOT_JOINT_GUARANTEE_INDIVIDUAL_RISK",
        )
        self.assertFalse(solution.horizon_assurance_eligible)

    def test_fallback_solution_never_emits_horizon_eligible(self):
        solution = self._supervised("joint", fallback=True)
        self.assertEqual(
            solution.horizon_assurance_status,
            "NOT_GUARANTEED_FALLBACK_ACTIVE",
        )
        self.assertFalse(solution.horizon_assurance_eligible)

    def test_joint_solution_emits_horizon_eligible(self):
        solution = self._supervised("joint")
        self.assertEqual(
            solution.horizon_assurance_status,
            "HORIZON_GUARANTEE_ELIGIBLE",
        )
        self.assertTrue(solution.horizon_assurance_eligible)


if __name__ == "__main__":
    unittest.main()
