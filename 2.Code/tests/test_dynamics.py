from __future__ import annotations

import unittest

import numpy as np

from quadrotor_mpc.control.ccmpc.dynamics import QuadrotorDynamics, continuous_dynamics


class DynamicsTests(unittest.TestCase):
    def test_hover_is_equilibrium(self) -> None:
        derivative = continuous_dynamics(np.zeros(9), np.zeros(4))
        np.testing.assert_allclose(derivative, np.zeros(9), atol=1e-12)

    def test_positive_pitch_accelerates_forward(self) -> None:
        state = np.zeros(9)
        state[7] = 0.10
        derivative = continuous_dynamics(state, np.zeros(4))
        self.assertGreater(derivative[3], 0.0)
        self.assertAlmostEqual(derivative[4], 0.0, places=12)

    def test_affine_linearization_matches_expansion_point(self) -> None:
        dynamics = QuadrotorDynamics(k_vz=3.0)
        state = np.array([1.0, -0.2, 1.5, 0.3, -0.1, 0.2, 0.04, -0.08, 0.3])
        control = np.array([0.05, -0.06, 0.4, 0.2])
        A, B, C = dynamics.linearize(state, control, 0.06)
        predicted = A @ state + B @ control + C
        actual = dynamics.discrete(state, control, 0.06)
        np.testing.assert_allclose(predicted, actual, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
