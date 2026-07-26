from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from quadrotor_mpc.control.nmpc.covariance import (
    CovariancePropagationOptions,
    HorizonCovariancePropagator,
    analytic_error_state_jacobians,
    finite_difference_error_state_jacobians,
    obstacle_process_covariance,
    obstacle_transition,
)
from quadrotor_mpc.core.contracts import ObstacleBelief, SphericalObstacle, VehicleBelief
from quadrotor_mpc.estimation.native import propagate_nominal
from quadrotor_mpc.interfaces.desktop.viewer import load_native_mujoco_config

CODE_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_STATE = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def hover_trajectory(steps: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    controls = np.zeros((steps, 4), dtype=float)
    states = [IDENTITY_STATE.copy()]
    for control in controls:
        states.append(propagate_nominal(states[-1], control, dt))
    return np.asarray(states), controls


class HorizonVehicleCovarianceTests(unittest.TestCase):
    def test_analytic_linearization_agrees_with_rk4_finite_difference_at_hover(self):
        analytic_state, analytic_control = analytic_error_state_jacobians(
            IDENTITY_STATE,
            np.zeros(4),
            0.05,
        )
        numeric_state, numeric_control = finite_difference_error_state_jacobians(
            IDENTITY_STATE,
            np.zeros(4),
            0.05,
            1e-6,
        )
        self.assertLess(
            np.linalg.norm(analytic_state - numeric_state) / np.linalg.norm(numeric_state),
            0.01,
        )
        self.assertLess(
            np.linalg.norm(analytic_control - numeric_control) / np.linalg.norm(numeric_control),
            0.06,
        )

    def test_open_loop_horizon_starts_at_belief_and_remains_psd(self):
        dt = 0.05
        states, controls = hover_trajectory(8, dt)
        initial_covariance = np.eye(12) * 1e-4
        belief = VehicleBelief(IDENTITY_STATE, initial_covariance)
        propagator = HorizonCovariancePropagator(
            CovariancePropagationOptions(enabled=True, mode="open_loop"),
            dt,
        )
        horizon = propagator.propagate_vehicle(belief, states, controls)

        self.assertEqual(horizon.shape, (9, 12, 12))
        np.testing.assert_allclose(horizon[0], initial_covariance, atol=1e-12)
        np.testing.assert_allclose(horizon, horizon.swapaxes(-1, -2), atol=1e-12)
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(horizon))), -1e-12)
        self.assertGreater(float(np.trace(horizon[-1])), float(np.trace(horizon[0])))
        self.assertTrue(np.all(np.diag(horizon[-1])[:3] > np.diag(horizon[0])[:3]))

    def test_feedback_aware_mode_changes_closed_loop_uncertainty(self):
        dt = 0.05
        states, controls = hover_trajectory(10, dt)
        belief = VehicleBelief(IDENTITY_STATE, np.eye(12) * 2e-3)
        open_loop = HorizonCovariancePropagator(
            CovariancePropagationOptions(enabled=True, mode="open_loop"),
            dt,
        ).propagate_vehicle(belief, states, controls)
        feedback = HorizonCovariancePropagator(
            CovariancePropagationOptions(enabled=True, mode="feedback_lqr"),
            dt,
        ).propagate_vehicle(belief, states, controls)

        self.assertFalse(np.allclose(feedback[-1], open_loop[-1]))
        self.assertLess(float(np.trace(feedback[-1])), float(np.trace(open_loop[-1])))
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(feedback))), -1e-12)


class HorizonObstacleCovarianceTests(unittest.TestCase):
    def test_constant_velocity_covariance_matches_one_step_equation(self):
        dt = 0.1
        initial = np.diag([0.01, 0.02, 0.03, 0.4, 0.5, 0.6])
        obstacle = ObstacleBelief(
            mean_state_6=np.array([1.0, 2.0, 3.0, 0.2, -0.1, 0.0]),
            covariance_6=initial,
            shape=SphericalObstacle(0.25),
        )
        options = CovariancePropagationOptions(
            enabled=True,
            obstacle_acceleration_process_std_mps2=0.4,
        )
        horizon = HorizonCovariancePropagator(options, dt).propagate_obstacles(
            [obstacle],
            4,
        )
        transition = obstacle_transition(dt)
        expected = transition @ initial @ transition.T + obstacle_process_covariance(dt, 0.4)

        self.assertEqual(horizon.shape, (4, 1, 6, 6))
        np.testing.assert_allclose(horizon[1, 0], expected, atol=1e-12)
        self.assertGreater(
            float(np.trace(horizon[-1, 0])),
            float(np.trace(horizon[0, 0])),
        )

    def test_empty_obstacle_horizon_has_stable_shape(self):
        propagator = HorizonCovariancePropagator(
            CovariancePropagationOptions(enabled=True),
            0.05,
        )
        self.assertEqual(
            propagator.propagate_obstacles([], 5).shape,
            (5, 0, 6, 6),
        )


class CovarianceConfigurationTests(unittest.TestCase):
    def test_estimation_scenario_enables_open_loop_propagation(self):
        config = load_native_mujoco_config(CODE_ROOT / "config" / "mujoco_native_estimation.yaml")
        self.assertTrue(config.covariance_propagation.enabled)
        self.assertEqual(config.covariance_propagation.mode, "open_loop")
        self.assertEqual(type(config).from_mapping(config.to_mapping()), config)

    def test_legacy_mapping_defaults_to_disabled(self):
        config = load_native_mujoco_config(CODE_ROOT / "config" / "mujoco_native.yaml")
        mapping = config.to_mapping()
        mapping["controller"].pop("covariance_propagation")
        loaded = type(config).from_mapping(mapping)
        self.assertFalse(loaded.covariance_propagation.enabled)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "open_loop"):
            CovariancePropagationOptions.from_mapping({"enabled": True, "mode": "perfect_feedback"})


if __name__ == "__main__":
    unittest.main()
