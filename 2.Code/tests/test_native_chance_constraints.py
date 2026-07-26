from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from mujoco_native import load_native_mujoco_config
from native_chance_constraints import (
    ChanceConstraintOptions,
    build_spherical_chance_profile,
    evaluate_spherical_constraints,
)
from native_risk_budget import RiskBudgetOptions

CODE_ROOT = Path(__file__).resolve().parents[1]


def profile_inputs():
    vehicle_positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    obstacle_positions = np.array([[[1.0, 0.0, 0.0], [1.5, 0.0, 0.0]]])
    vehicle_covariances = np.zeros((2, 12, 12))
    obstacle_covariances = np.zeros((2, 1, 6, 6))
    return (
        vehicle_positions,
        obstacle_positions,
        vehicle_covariances,
        obstacle_covariances,
    )


class ChanceConstraintMathTests(unittest.TestCase):
    def test_isotropic_relative_covariance_projects_onto_collision_normal(self):
        vehicle, obstacles, vehicle_covariance, obstacle_covariance = profile_inputs()
        vehicle_covariance[:, :3, :3] = np.eye(3) * 0.04
        obstacle_covariance[:, 0, :3, :3] = np.eye(3) * 0.09
        options = ChanceConstraintOptions(enabled=True, individual_epsilon=0.05)

        profile = build_spherical_chance_profile(
            vehicle_positions=vehicle,
            obstacle_positions=obstacles,
            vehicle_covariances=vehicle_covariance,
            obstacle_covariances=obstacle_covariance,
            base_safety_radii_m=np.array([0.6]),
            options=options,
        )

        expected_sigma = np.sqrt(0.04 + 0.09)
        np.testing.assert_allclose(profile.projected_sigmas_m, expected_sigma)
        np.testing.assert_allclose(profile.tightenings_m, options.beta * expected_sigma)
        np.testing.assert_allclose(
            profile.safety_radii_m,
            0.6 + options.beta * expected_sigma,
        )
        np.testing.assert_allclose(profile.risk_allocations, 0.05)
        np.testing.assert_allclose(
            profile.collision_normals[:, 0],
            [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        )

    def test_anisotropic_projection_uses_collision_direction_not_covariance_trace(self):
        vehicle, obstacles, vehicle_covariance, obstacle_covariance = profile_inputs()
        vehicle_covariance[:, :3, :3] = np.diag([0.01, 4.0, 9.0])
        profile = build_spherical_chance_profile(
            vehicle_positions=vehicle,
            obstacle_positions=obstacles,
            vehicle_covariances=vehicle_covariance,
            obstacle_covariances=obstacle_covariance,
            base_safety_radii_m=np.array([0.5]),
            options=ChanceConstraintOptions(enabled=True),
        )
        np.testing.assert_allclose(profile.projected_sigmas_m, 0.1)

    def test_joint_profile_uses_per_constraint_risk_and_budget(self):
        vehicle, obstacles, vehicle_covariance, obstacle_covariance = profile_inputs()
        vehicle_covariance[:, :3, :3] = np.eye(3) * 0.04
        options = ChanceConstraintOptions(
            enabled=True,
            risk_budget=RiskBudgetOptions(
                semantics="joint",
                allocation="uniform",
                total_epsilon=0.02,
            ),
        )
        profile = build_spherical_chance_profile(
            vehicle_positions=vehicle,
            obstacle_positions=obstacles,
            vehicle_covariances=vehicle_covariance,
            obstacle_covariances=obstacle_covariance,
            base_safety_radii_m=np.array([0.5]),
            options=options,
        )

        np.testing.assert_allclose(profile.risk_allocations, 0.01)
        np.testing.assert_allclose(
            profile.tightenings_m,
            profile.gaussian_quantiles * profile.projected_sigmas_m,
        )
        self.assertAlmostEqual(profile.allocated_epsilon, 0.02)
        self.assertAlmostEqual(profile.remaining_epsilon, 0.0)
        self.assertEqual(profile.active_constraint_count, 2)
        self.assertEqual(profile.budget_status, "BUDGET_OK")

    def test_disabled_profile_preserves_deterministic_radius_and_zero_risk(self):
        vehicle, obstacles, vehicle_covariance, obstacle_covariance = profile_inputs()
        vehicle_covariance[:, :3, :3] = np.eye(3)
        profile = build_spherical_chance_profile(
            vehicle_positions=vehicle,
            obstacle_positions=obstacles,
            vehicle_covariances=vehicle_covariance,
            obstacle_covariances=obstacle_covariance,
            base_safety_radii_m=np.array([0.7]),
            options=ChanceConstraintOptions(enabled=False),
        )
        np.testing.assert_allclose(profile.tightenings_m, 0.0)
        np.testing.assert_allclose(profile.safety_radii_m, 0.7)
        np.testing.assert_allclose(profile.risk_allocations, 0.0)

    def test_residual_and_slack_match_soft_spherical_constraint(self):
        residual, slack = evaluate_spherical_constraints(
            vehicle_positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            obstacle_positions=np.array([[[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
            safety_radii_m=np.array([[0.8], [0.8]]),
            distance_smoothing_m2=0.0,
        )
        np.testing.assert_allclose(residual[:, 0], [-0.3, 0.2])
        np.testing.assert_allclose(slack[:, 0], [0.3, 0.0])

    def test_invalid_individual_risk_is_rejected(self):
        for value in (0.0, 0.5, 1.0):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "individual_epsilon"),
            ):
                ChanceConstraintOptions(individual_epsilon=value)


class ChanceConstraintConfigurationTests(unittest.TestCase):
    def test_ccmpc_scenario_enables_spherical_constraints(self):
        config = load_native_mujoco_config(CODE_ROOT / "config" / "mujoco_native_ccmpc.yaml")
        self.assertTrue(config.chance_constraints.enabled)
        self.assertTrue(config.covariance_propagation.enabled)
        self.assertAlmostEqual(config.chance_constraints.individual_epsilon, 0.05)
        self.assertEqual(config.chance_constraints.risk_budget.semantics, "joint")
        self.assertEqual(config.chance_constraints.risk_budget.allocation, "uniform")
        self.assertAlmostEqual(
            config.chance_constraints.risk_budget.total_epsilon,
            0.10,
        )
        self.assertEqual(type(config).from_mapping(config.to_mapping()), config)

    def test_legacy_mapping_defaults_to_disabled(self):
        config = load_native_mujoco_config(CODE_ROOT / "config" / "mujoco_native.yaml")
        mapping = config.to_mapping()
        mapping["controller"].pop("chance_constraints")
        loaded = type(config).from_mapping(mapping)
        self.assertFalse(loaded.chance_constraints.enabled)

    def test_enabled_constraint_requires_covariance_propagation(self):
        config = load_native_mujoco_config(CODE_ROOT / "config" / "mujoco_native_ccmpc.yaml")
        mapping = config.to_mapping()
        mapping["controller"]["covariance_propagation"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "covariance_propagation"):
            type(config).from_mapping(mapping)


if __name__ == "__main__":
    unittest.main()
