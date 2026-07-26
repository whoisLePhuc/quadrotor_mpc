from __future__ import annotations

import unittest

import numpy as np

from quadrotor_mpc.control.ccmpc.dynamics import QuadrotorDynamics
from quadrotor_mpc.control.ccmpc.uncertainty import UncertaintyPropagator


class UncertaintyTests(unittest.TestCase):
    def test_propagation_stays_symmetric_psd_and_grows(self) -> None:
        propagator = UncertaintyPropagator()
        dynamics = QuadrotorDynamics(k_vz=3.0)
        states = np.zeros((9, 6))
        states[2] = 1.0
        controls = np.zeros((4, 5))
        covariance = propagator.propagate(
            propagator.Gamma_0,
            states,
            controls,
            dynamics,
            dt=0.1,
        )
        self.assertEqual(len(covariance), 6)
        for matrix in covariance:
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
            self.assertGreaterEqual(float(np.linalg.eigvalsh(matrix).min()), -1e-12)
        self.assertGreater(float(np.trace(covariance[-1])), float(np.trace(covariance[0])))


if __name__ == "__main__":
    unittest.main()
