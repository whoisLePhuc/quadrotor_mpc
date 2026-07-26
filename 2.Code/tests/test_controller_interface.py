from __future__ import annotations

import unittest

import numpy as np

from quadrotor_mpc.core.contracts import (
    ControlGoal,
    ControlSolution,
    ObstacleBelief,
    SphericalObstacle,
    VehicleBelief,
)
from quadrotor_mpc.estimation.truth import exact_obstacle_beliefs


class BeliefContractTests(unittest.TestCase):
    def test_exact_vehicle_belief_normalizes_quaternion_and_has_12d_covariance(self):
        state = np.array([0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0], dtype=float)
        belief = VehicleBelief.exact(state)
        np.testing.assert_allclose(belief.mean_state_13[6:10], [1, 0, 0, 0])
        self.assertEqual(belief.error_covariance_12.shape, (12, 12))
        self.assertFalse(belief.mean_state_13.flags.writeable)

    def test_invalid_covariance_is_rejected(self):
        covariance = np.eye(12)
        covariance[0, 1] = 1.0
        with self.assertRaisesRegex(ValueError, "symmetric"):
            VehicleBelief(
                np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float),
                covariance,
            )

    def test_obstacle_belief_constant_velocity_fallback(self):
        belief = ObstacleBelief(
            mean_state_6=np.array([1, 2, 3, 0.5, -0.25, 0]),
            covariance_6=np.zeros((6, 6)),
            shape=SphericalObstacle(0.2),
        )
        prediction = belief.mean_positions(3, 0.5)
        np.testing.assert_allclose(
            prediction,
            [[1, 2, 3], [1.25, 1.875, 3], [1.5, 1.75, 3]],
        )

    def test_truth_adapter_preserves_nonlinear_motion_horizon(self):
        obstacle = {
            "name": "sine",
            "type": "dynamic",
            "radius": 0.2,
            "motion": {
                "type": "sinusoidal",
                "center": [1, 2, 3],
                "amplitude": [0, 0.5, 0],
                "period_s": 2.0,
                "phase_rad": 0.0,
            },
        }
        belief = exact_obstacle_beliefs([obstacle], 0.0, 2, 0.5)[0]
        np.testing.assert_allclose(
            belief.predicted_positions,
            [[1, 2, 3], [1, 2.5, 3], [1, 2, 3]],
            atol=1e-12,
        )

    def test_control_solution_exposes_position_horizon(self):
        states = np.zeros((3, 13))
        states[:, 6] = 1.0
        states[:, :3] = [[0, 0, 1], [0.1, 0, 1], [0.2, 0, 1]]
        solution = ControlSolution(
            command=np.zeros(4),
            nominal_states=states,
            predicted_covariances=np.zeros((3, 12, 12)),
            chance_margins=np.ones((3, 1)),
            risk_allocations=np.zeros((3, 1)),
            slacks=np.zeros((3, 1)),
            solver_status="TEST",
        )
        np.testing.assert_allclose(solution.predicted_positions, states[:, :3])
        self.assertEqual(solution.solver_status, "TEST")

    def test_goal_normalizes_quaternion(self):
        goal = ControlGoal(position=[1, 2, 3], quaternion_wxyz=[2, 0, 0, 0])
        np.testing.assert_allclose(goal.quaternion_wxyz, [1, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
