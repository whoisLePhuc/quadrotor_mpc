"""Regression tests: missing solver residuals must never be treated as zero.

This locks the semantics of :mod:`quadrotor_mpc.control.nmpc.residuals`:

- a residual of exactly ``0.0`` means the backend reported a perfect optimum;
- a missing residual means the condition cannot be evaluated at all;
- these two facts are distinct and must never be conflated.
"""

from __future__ import annotations

import unittest

import numpy as np

from quadrotor_mpc.control.nmpc.assurance import (
    HorizonAssuranceInput,
    HorizonAssuranceStatus,
    classify_horizon_assurance,
)
from quadrotor_mpc.control.nmpc.residuals import (
    MISSING_RESIDUAL,
    ResidualGateStatus,
    ResidualStatus,
    SolverResidual,
    SolverResidualDiagnostics,
    evaluate_residual_gate,
    extract_residual,
)


def available(value: float) -> SolverResidual:
    return SolverResidual(status=ResidualStatus.AVAILABLE, value=value, source="test")


def unavailable() -> SolverResidual:
    return SolverResidual(
        status=ResidualStatus.UNAVAILABLE, value=None, source="test"
    )


def invalid() -> SolverResidual:
    return SolverResidual(status=ResidualStatus.INVALID, value=None, source="test")


class ResidualExtractionTests(unittest.TestCase):
    def test_missing_field_is_unavailable_not_zero(self):
        residual = extract_residual(MISSING_RESIDUAL, source="test")
        self.assertIs(residual.status, ResidualStatus.UNAVAILABLE)
        self.assertIsNone(residual.value)
        self.assertNotEqual(residual.value, 0.0)

    def test_none_is_unavailable_not_zero(self):
        residual = extract_residual(None, source="test")
        self.assertIs(residual.status, ResidualStatus.UNAVAILABLE)
        self.assertIsNone(residual.value)

    def test_empty_collection_is_unavailable_not_zero(self):
        residual = extract_residual([], source="test")
        self.assertIs(residual.status, ResidualStatus.UNAVAILABLE)
        self.assertIsNone(residual.value)

    def test_exact_zero_is_available_zero(self):
        residual = extract_residual(0.0, source="test")
        self.assertIs(residual.status, ResidualStatus.AVAILABLE)
        self.assertEqual(residual.value, 0.0)

    def test_positive_finite_value_is_available(self):
        residual = extract_residual(1.5e-4, source="test")
        self.assertIs(residual.status, ResidualStatus.AVAILABLE)
        self.assertEqual(residual.value, 1.5e-4)

    def test_nan_is_invalid(self):
        residual = extract_residual(float("nan"), source="test")
        self.assertIs(residual.status, ResidualStatus.INVALID)
        self.assertIsNone(residual.value)

    def test_positive_infinity_is_invalid(self):
        residual = extract_residual(float("inf"), source="test")
        self.assertIs(residual.status, ResidualStatus.INVALID)

    def test_negative_infinity_is_invalid(self):
        residual = extract_residual(float("-inf"), source="test")
        self.assertIs(residual.status, ResidualStatus.INVALID)

    def test_negative_residual_is_invalid(self):
        residual = extract_residual(-1e-4, source="test")
        self.assertIs(residual.status, ResidualStatus.INVALID)

    def test_non_numeric_value_is_invalid(self):
        residual = extract_residual("not-a-number", source="test")
        self.assertIs(residual.status, ResidualStatus.INVALID)

    def test_vector_uses_documented_aggregation(self):
        residual = extract_residual([1e-6, 2e-6, 3e-6], source="test")
        self.assertIs(residual.status, ResidualStatus.AVAILABLE)
        self.assertEqual(residual.value, 3e-6)

    def test_vector_with_one_nan_is_invalid(self):
        residual = extract_residual([1e-6, float("nan"), 3e-6], source="test")
        self.assertIs(residual.status, ResidualStatus.INVALID)
        self.assertIsNone(residual.value)


class SolverResidualValueObjectTests(unittest.TestCase):
    def test_available_requires_value(self):
        with self.assertRaises(ValueError):
            SolverResidual(status=ResidualStatus.AVAILABLE, value=None)

    def test_unavailable_rejects_numeric_value(self):
        with self.assertRaises(ValueError):
            SolverResidual(status=ResidualStatus.UNAVAILABLE, value=0.0)

    def test_invalid_rejects_numeric_value(self):
        with self.assertRaises(ValueError):
            SolverResidual(status=ResidualStatus.INVALID, value=1e-6)

    def test_available_rejects_nan(self):
        with self.assertRaises(ValueError):
            SolverResidual(
                status=ResidualStatus.AVAILABLE,
                value=float("nan"),
            )

    def test_available_rejects_infinity(self):
        with self.assertRaises(ValueError):
            SolverResidual(
                status=ResidualStatus.AVAILABLE,
                value=float("inf"),
            )

    def test_available_rejects_negative_value(self):
        with self.assertRaises(ValueError):
            SolverResidual(
                status=ResidualStatus.AVAILABLE,
                value=-1.0,
            )


class ResidualGateTruthTableTests(unittest.TestCase):
    def _gate(self, primal, dual, required: bool = True) -> ResidualGateStatus:
        return evaluate_residual_gate(
            SolverResidualDiagnostics(primal=primal, dual=dual),
            primal_tolerance=1e-3,
            dual_tolerance=1e-3,
            required=required,
        )

    def test_both_available_within_tolerance_passes(self):
        self.assertIs(
            self._gate(available(1e-6), available(2e-6)),
            ResidualGateStatus.PASS,
        )

    def test_primal_above_tolerance_fails(self):
        self.assertIs(
            self._gate(available(1e-2), available(2e-6)),
            ResidualGateStatus.FAIL_THRESHOLD,
        )

    def test_dual_above_tolerance_fails(self):
        self.assertIs(
            self._gate(available(1e-6), available(1e-2)),
            ResidualGateStatus.FAIL_THRESHOLD,
        )

    def test_unavailable_primal_unknown(self):
        self.assertIs(
            self._gate(unavailable(), available(2e-6)),
            ResidualGateStatus.UNKNOWN_UNAVAILABLE,
        )

    def test_unavailable_dual_unknown(self):
        self.assertIs(
            self._gate(available(1e-6), unavailable()),
            ResidualGateStatus.UNKNOWN_UNAVAILABLE,
        )

    def test_both_unavailable_unknown(self):
        self.assertIs(
            self._gate(unavailable(), unavailable()),
            ResidualGateStatus.UNKNOWN_UNAVAILABLE,
        )

    def test_invalid_primal_fails_invalid(self):
        self.assertIs(
            self._gate(invalid(), available(2e-6)),
            ResidualGateStatus.FAIL_INVALID,
        )

    def test_invalid_dual_fails_invalid(self):
        self.assertIs(
            self._gate(available(1e-6), invalid()),
            ResidualGateStatus.FAIL_INVALID,
        )

    def test_invalid_with_unavailable_fails_invalid(self):
        self.assertIs(
            self._gate(invalid(), unavailable()),
            ResidualGateStatus.FAIL_INVALID,
        )

    def test_threshold_wins_over_unavailable_other(self):
        self.assertIs(
            self._gate(available(1e-2), unavailable()),
            ResidualGateStatus.FAIL_THRESHOLD,
        )

    def test_not_required_gate_is_not_required(self):
        self.assertIs(
            self._gate(unavailable(), unavailable(), required=False),
            ResidualGateStatus.NOT_REQUIRED,
        )

    def test_residual_equal_to_tolerance_passes(self):
        self.assertIs(
            self._gate(available(1e-3), available(1e-3)),
            ResidualGateStatus.PASS,
        )

    def test_residual_just_above_tolerance_fails(self):
        self.assertIs(
            self._gate(available(float(np.nextafter(1e-3, np.inf))), available(1e-6)),
            ResidualGateStatus.FAIL_THRESHOLD,
        )


class ResidualAssuranceIntegrationTests(unittest.TestCase):
    def _assure(self, gate: str) -> HorizonAssuranceStatus:
        decision = classify_horizon_assurance(
            HorizonAssuranceInput(
                risk_semantics="joint",
                risk_budget_status="BUDGET_OK",
                primary_solver_success=True,
                residual_gate_status=gate,
                residual_status="AVAILABLE",
                maximum_slack=0.0,
                slack_tolerance=1e-6,
                deadline_missed=False,
                fallback_active=False,
            )
        )
        return decision.status

    def test_missing_residual_cannot_produce_horizon_guarantee(self):
        status = self._assure("UNKNOWN_UNAVAILABLE")
        self.assertEqual(
            status,
            HorizonAssuranceStatus.NOT_GUARANTEED_RESIDUAL_UNAVAILABLE,
        )

    def test_invalid_residual_cannot_produce_horizon_guarantee(self):
        status = self._assure("FAIL_INVALID")
        self.assertEqual(
            status,
            HorizonAssuranceStatus.NOT_GUARANTEED_RESIDUAL_INVALID,
        )

    def test_residual_above_tolerance_cannot_produce_horizon_guarantee(self):
        status = self._assure("FAIL_THRESHOLD")
        self.assertEqual(
            status,
            HorizonAssuranceStatus.NOT_GUARANTEED_RESIDUAL_THRESHOLD_EXCEEDED,
        )

    def test_available_passing_residuals_allow_assurance(self):
        status = self._assure("PASS")
        self.assertEqual(
            status,
            HorizonAssuranceStatus.HORIZON_GUARANTEE_ELIGIBLE,
        )


if __name__ == "__main__":
    unittest.main()
