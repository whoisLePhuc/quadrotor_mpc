from __future__ import annotations

import unittest
from statistics import NormalDist

import numpy as np

from quadrotor_mpc.control.nmpc.risk_budget import (
    RiskBudgetOptions,
    allocate_risk_budget,
)


class RiskBudgetAllocationTests(unittest.TestCase):
    def test_joint_uniform_allocation_sums_to_total_budget(self):
        allocation = allocate_risk_budget(
            steps=4,
            obstacle_count=3,
            enabled=True,
            individual_epsilon=0.05,
            options=RiskBudgetOptions(
                semantics="joint",
                allocation="uniform",
                total_epsilon=0.12,
            ),
        )

        np.testing.assert_allclose(allocation.epsilons, 0.01)
        self.assertAlmostEqual(float(np.sum(allocation.epsilons)), 0.12)
        self.assertAlmostEqual(allocation.allocated_epsilon, 0.12)
        self.assertAlmostEqual(allocation.remaining_epsilon, 0.0)
        self.assertEqual(allocation.active_constraint_count, 12)
        self.assertEqual(allocation.budget_status, "BUDGET_OK")

    def test_joint_quantile_is_computed_from_allocated_cell_risk(self):
        allocation = allocate_risk_budget(
            steps=2,
            obstacle_count=2,
            enabled=True,
            individual_epsilon=0.05,
            options=RiskBudgetOptions(
                semantics="joint",
                total_epsilon=0.04,
            ),
        )
        expected = NormalDist().inv_cdf(1.0 - 0.01)
        np.testing.assert_allclose(allocation.gaussian_quantiles, expected)

    def test_individual_mode_preserves_stage4_semantics(self):
        allocation = allocate_risk_budget(
            steps=3,
            obstacle_count=2,
            enabled=True,
            individual_epsilon=0.05,
            options=RiskBudgetOptions(semantics="individual"),
        )
        np.testing.assert_allclose(allocation.epsilons, 0.05)
        self.assertIsNone(allocation.configured_total_epsilon)
        self.assertIsNone(allocation.remaining_epsilon)
        self.assertEqual(allocation.budget_status, "INDIVIDUAL_ONLY")

    def test_disabled_constraints_have_no_risk_or_quantile(self):
        allocation = allocate_risk_budget(
            steps=3,
            obstacle_count=2,
            enabled=False,
            individual_epsilon=0.05,
            options=RiskBudgetOptions(semantics="joint"),
        )
        np.testing.assert_allclose(allocation.epsilons, 0.0)
        np.testing.assert_allclose(allocation.gaussian_quantiles, 0.0)
        self.assertEqual(allocation.budget_status, "DISABLED")

    def test_zero_obstacles_retains_unspent_joint_budget(self):
        allocation = allocate_risk_budget(
            steps=21,
            obstacle_count=0,
            enabled=True,
            individual_epsilon=0.05,
            options=RiskBudgetOptions(
                semantics="joint",
                total_epsilon=0.10,
            ),
        )
        self.assertEqual(allocation.active_constraint_count, 0)
        self.assertAlmostEqual(allocation.allocated_epsilon, 0.0)
        self.assertAlmostEqual(allocation.remaining_epsilon, 0.10)
        self.assertEqual(allocation.budget_status, "BUDGET_OK")

    def test_stage5_rejects_unimplemented_allocator(self):
        with self.assertRaisesRegex(ValueError, "uniform"):
            RiskBudgetOptions(
                semantics="joint",
                allocation="geometry_aware",
            )

    def test_invalid_joint_budget_is_rejected(self):
        for value in (0.0, 0.5, 1.0):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "total_epsilon"),
            ):
                RiskBudgetOptions(
                    semantics="joint",
                    total_epsilon=value,
                )


if __name__ == "__main__":
    unittest.main()
