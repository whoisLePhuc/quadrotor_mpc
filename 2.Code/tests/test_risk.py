from __future__ import annotations

import math
import unittest

import numpy as np

from ccmpc.risk import (
    chance_constraint_residual,
    collision_clearance,
    symmetric_matrix_sqrt,
)
from ccmpc.utils import Omega_matrix, yaw_to_rotation


class RiskGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.omega = Omega_matrix(
            np.array([0.7, 0.4, 0.9]),
            0.35,
            yaw_to_rotation(math.radians(37.0)),
        )

    def test_symmetric_root_reconstructs_rotated_omega(self) -> None:
        root = symmetric_matrix_sqrt(self.omega)
        np.testing.assert_allclose(root, root.T, atol=1e-12)
        np.testing.assert_allclose(root @ root, self.omega, atol=1e-12)

    def test_uncertainty_tightens_constraint(self) -> None:
        position = np.array([1.5, 0.9, 0.2])
        zero = np.zeros((3, 3))
        noisy = np.diag([0.12, 0.08, 0.05]) ** 2
        nominal = chance_constraint_residual(position, np.zeros(3), self.omega, zero, zero, 0.5)
        tightened = chance_constraint_residual(position, np.zeros(3), self.omega, noisy, noisy, 0.03)
        self.assertLess(tightened, nominal)

    def test_clearance_sign(self) -> None:
        self.assertGreater(collision_clearance(np.array([3.0, 0.0, 0.0]), np.zeros(3), self.omega), 0.0)
        self.assertLess(collision_clearance(np.array([0.05, 0.0, 0.0]), np.zeros(3), self.omega), 0.0)


if __name__ == "__main__":
    unittest.main()
