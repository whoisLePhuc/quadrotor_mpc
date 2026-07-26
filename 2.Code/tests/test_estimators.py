from __future__ import annotations

import unittest

import numpy as np

from quadrotor_mpc.application.simulation.estimators import ExtendedKalmanEstimator
from quadrotor_mpc.control.ccmpc.dynamics import QuadrotorDynamics


class EstimatorTests(unittest.TestCase):
    def test_ekf_predict_update_reduces_measurement_error(self) -> None:
        estimator = ExtendedKalmanEstimator(
            QuadrotorDynamics(),
            np.eye(9) * 1e-3,
            np.eye(9) * 0.04,
        )
        estimator.reset(np.zeros(9), np.eye(9) * 0.1)
        estimator.predict(np.zeros(4), 0.05)
        measurement = np.ones(9) * 0.2
        prior_error = np.linalg.norm(estimator.state - measurement)
        estimator.update(measurement)
        posterior_error = np.linalg.norm(estimator.state - measurement)
        self.assertLess(posterior_error, prior_error)
        self.assertTrue(np.all(np.linalg.eigvalsh(estimator.covariance) >= -1e-12))


if __name__ == "__main__":
    unittest.main()
