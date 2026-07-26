from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from mujoco_native import load_native_mujoco_config
from native_estimation import (
    ConstantVelocityObstacleFilter,
    ErrorStateEkf,
    EstimationOptions,
    NativeBeliefEstimator,
    NativeSensorSimulator,
    ObstacleFilterOptions,
    ObstacleMeasurement,
    SensorOptions,
    VehicleFilterOptions,
    VehicleMeasurement,
    inject_error,
    state_error,
)
from obstacle_motion import obstacle_position

CODE_ROOT = Path(__file__).resolve().parents[1]
IDENTITY_STATE = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


class QuaternionErrorStateTests(unittest.TestCase):
    def test_injection_and_error_are_local_inverses(self):
        error = np.array([0.1, -0.2, 0.3, 0.01, 0.02, -0.03, 0.04, -0.02, 0.03, 0.2, -0.1, 0.05])
        perturbed = inject_error(IDENTITY_STATE, error)
        np.testing.assert_allclose(state_error(IDENTITY_STATE, perturbed), error, atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(perturbed[6:10]), 1.0)


class NativeSensorTests(unittest.TestCase):
    def test_reset_replays_identical_seeded_measurements(self):
        options = SensorOptions(
            position_std_m=0.1,
            velocity_std_mps=0.1,
            attitude_std_rad=0.1,
            angular_rate_std_radps=0.1,
            obstacle_position_std_m=0.1,
        )
        sensor = NativeSensorSimulator(options, seed=42)
        obstacle_truth = np.array([[1.0, 2.0, 3.0]])
        sensor.reset(1)
        first = sensor.sample(IDENTITY_STATE, obstacle_truth, dt=0.05, force_measurements=True)
        sensor.reset(1)
        repeated = sensor.sample(IDENTITY_STATE, obstacle_truth, dt=0.05, force_measurements=True)
        np.testing.assert_allclose(first.vehicle.state_13, repeated.vehicle.state_13)
        np.testing.assert_allclose(
            first.obstacles[0].position_world,
            repeated.obstacles[0].position_world,
        )

    def test_dropout_is_reported_without_leaking_a_measurement(self):
        sensor = NativeSensorSimulator(
            SensorOptions(
                vehicle_dropout_probability=1.0,
                obstacle_dropout_probability=1.0,
            ),
            seed=1,
        )
        sensor.reset(2)
        frame = sensor.sample(
            IDENTITY_STATE,
            np.zeros((2, 3)),
            dt=0.05,
        )
        self.assertIsNone(frame.vehicle)
        self.assertEqual(frame.obstacles, ())
        self.assertFalse(frame.obstacle_available.any())


class NativeVehicleEstimatorTests(unittest.TestCase):
    def test_measurement_update_reduces_local_state_error_and_preserves_psd(self):
        options = VehicleFilterOptions()
        estimator = ErrorStateEkf(options)
        initial_error = np.array(
            [0.2, -0.1, 0.15, 0.05, 0.02, -0.03, 0.08, -0.04, 0.03, 0.1, 0.0, -0.1]
        )
        estimator.reset(
            VehicleMeasurement(
                inject_error(IDENTITY_STATE, initial_error),
                np.eye(12) * 1e-4,
            )
        )
        before = np.linalg.norm(state_error(estimator.state_13, IDENTITY_STATE))
        estimator.update(
            VehicleMeasurement(
                IDENTITY_STATE.copy(),
                np.eye(12) * 1e-4,
            )
        )
        after = np.linalg.norm(state_error(estimator.state_13, IDENTITY_STATE))
        self.assertLess(after, before)
        self.assertGreaterEqual(np.linalg.eigvalsh(estimator.covariance_12).min(), -1e-12)

    def test_predict_without_measurement_grows_uncertainty(self):
        estimator = ErrorStateEkf(VehicleFilterOptions())
        estimator.reset(VehicleMeasurement(IDENTITY_STATE.copy(), np.eye(12) * 1e-3))
        before = float(np.trace(estimator.covariance_12))
        estimator.predict(np.zeros(4), 0.05)
        estimator.update(None)
        self.assertGreater(float(np.trace(estimator.covariance_12)), before)
        self.assertAlmostEqual(np.linalg.norm(estimator.state_13[6:10]), 1.0)


class ObstacleTrackerTests(unittest.TestCase):
    def test_constant_velocity_is_learned_from_position_only_measurements(self):
        tracker = ConstantVelocityObstacleFilter(
            ObstacleFilterOptions(
                acceleration_process_std_mps2=0.05,
                initial_position_std_m=0.1,
                initial_velocity_std_mps=1.0,
            )
        )
        measurement_covariance = np.eye(3) * 1e-6
        tracker.reset(ObstacleMeasurement(0, np.zeros(3), measurement_covariance))
        velocity = np.array([0.2, -0.1, 0.05])
        dt = 0.1
        for step in range(1, 31):
            tracker.predict(dt)
            tracker.update(
                ObstacleMeasurement(
                    0,
                    velocity * (step * dt),
                    measurement_covariance,
                )
            )
        np.testing.assert_allclose(tracker.state_6[3:], velocity, atol=0.01)
        self.assertGreaterEqual(np.linalg.eigvalsh(tracker.covariance_6).min(), -1e-12)


class NativeBeliefPipelineTests(unittest.TestCase):
    def test_estimated_scenario_round_trips_configuration(self):
        config = load_native_mujoco_config(CODE_ROOT / "config" / "mujoco_native_estimation.yaml")
        self.assertTrue(config.estimation.enabled)
        self.assertEqual(config.estimation.estimator_type, "error_state_ekf")
        self.assertEqual(type(config).from_mapping(config.to_mapping()), config)

    def test_obstacle_horizon_comes_from_tracker_not_truth_motion(self):
        obstacle = {
            "name": "sinusoid",
            "type": "dynamic",
            "radius": 0.2,
            "motion": {
                "type": "sinusoidal",
                "center": [1.0, 0.0, 1.0],
                "amplitude": [0.0, 1.0, 0.0],
                "period_s": 2.0,
                "phase_rad": 0.0,
            },
        }
        options = EstimationOptions(
            enabled=True,
            seed=3,
            sensor=SensorOptions(
                position_std_m=0.0,
                velocity_std_mps=0.0,
                attitude_std_rad=0.0,
                angular_rate_std_radps=0.0,
                obstacle_position_std_m=0.0,
            ),
        )
        estimator = NativeBeliefEstimator(
            options,
            [obstacle],
            horizon_steps=4,
            timestep_s=0.1,
        )
        initial_position = obstacle_position(obstacle, 0.0)[None, :]
        estimator.reset(IDENTITY_STATE, initial_position)
        snapshot = estimator.advance(
            IDENTITY_STATE,
            obstacle_position(obstacle, 0.1)[None, :],
            np.zeros(4),
            dt=0.1,
        )
        predicted = snapshot.obstacle_beliefs[0].predicted_positions
        second_differences = np.diff(predicted, n=2, axis=0)
        np.testing.assert_allclose(second_differences, 0.0, atol=1e-12)
        truth_horizon = np.stack(
            [obstacle_position(obstacle, 0.1 + step * 0.1) for step in range(5)]
        )
        self.assertGreater(np.linalg.norm(predicted - truth_horizon), 1e-3)

    def test_pipeline_reset_reproduces_vehicle_and_obstacle_beliefs(self):
        obstacle = {
            "name": "static",
            "type": "static",
            "position": [1.0, 0.0, 1.0],
            "radius": 0.2,
        }
        options = EstimationOptions(enabled=True, seed=99)
        estimator = NativeBeliefEstimator(
            options,
            [obstacle],
            horizon_steps=3,
            timestep_s=0.05,
        )
        truth_obstacle = np.array([[1.0, 0.0, 1.0]])
        first = estimator.reset(IDENTITY_STATE, truth_obstacle)
        repeated = estimator.reset(IDENTITY_STATE, truth_obstacle)
        np.testing.assert_allclose(
            first.vehicle_belief.mean_state_13,
            repeated.vehicle_belief.mean_state_13,
        )
        np.testing.assert_allclose(
            first.obstacle_beliefs[0].mean_state_6,
            repeated.obstacle_beliefs[0].mean_state_6,
        )

    def test_invalid_estimator_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "error_state_ekf"):
            EstimationOptions.from_mapping({"type": "ground_truth"})


if __name__ == "__main__":
    unittest.main()
