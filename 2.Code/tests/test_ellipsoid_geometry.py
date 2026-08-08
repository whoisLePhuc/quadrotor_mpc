"""Regression tests: rotated ellipsoid collision metric uses R @ D @ R.T.

This locks the fix in the ``Omega_matrix`` implementation (see the docstring in
``control/ccmpc/utils.py``).  The previous implementation used
``R_o.T @ diag(...) @ R_o`` which rotates the collision ellipsoid by ``-yaw``
instead of ``+yaw``, mirroring obstacles used for collision checking or
avoidance.

The tests deliberately use an asymmetric ellipsoid (``a != b``), a non-special
yaw (30 deg) and points on the rotated principal axes in the world frame, so a
wrong rotation direction is observable.  Only the production implementation
(``Omega_matrix`` / ``EllipsoidalObstacle.get_omega``) is exercised; expected
geometry is derived independently.
"""

from __future__ import annotations

import unittest

import numpy as np

from quadrotor_mpc.control.ccmpc.obstacle import EllipsoidalObstacle
from quadrotor_mpc.control.ccmpc.utils import Omega_matrix

AXES = np.array([2.0, 0.5, 1.0])
CENTER = np.array([1.2, -0.7, 0.4])
YAW_RAD = float(np.deg2rad(30.0))
MAV_RADIUS = 0.0


def yaw_rotation(yaw_rad: float) -> np.ndarray:
    """Independent z-axis rotation used only to build expected geometry."""
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    return np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


class RotatedEllipsoidMetricTests(unittest.TestCase):
    def test_metric_equals_rotation_congruence_of_local_diagonal(self):
        rotation = yaw_rotation(YAW_RAD)
        metric_local = np.diag(1.0 / np.square(AXES + MAV_RADIUS))

        metric_world = Omega_matrix(AXES, MAV_RADIUS, rotation)

        np.testing.assert_allclose(
            metric_world,
            rotation @ metric_local @ rotation.T,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_rotated_ellipsoid_long_axis_has_unit_metric(self):
        rotation = yaw_rotation(YAW_RAD)
        metric_world = Omega_matrix(AXES, MAV_RADIUS, rotation)

        local_boundary_point = np.array([AXES[0], 0.0, 0.0])
        world_boundary_point = CENTER + rotation @ local_boundary_point
        delta = world_boundary_point - CENTER
        value = float(delta @ metric_world @ delta)

        self.assertAlmostEqual(value, 1.0, places=10)

    def test_rotated_ellipsoid_short_axis_has_unit_metric(self):
        rotation = yaw_rotation(YAW_RAD)
        metric_world = Omega_matrix(AXES, MAV_RADIUS, rotation)

        local_boundary_point = np.array([0.0, AXES[1], 0.0])
        world_boundary_point = CENTER + rotation @ local_boundary_point
        delta = world_boundary_point - CENTER
        value = float(delta @ metric_world @ delta)

        self.assertAlmostEqual(value, 1.0, places=10)

    def test_point_outside_rotated_ellipsoid_scores_above_one(self):
        rotation = yaw_rotation(YAW_RAD)
        metric_world = Omega_matrix(AXES, MAV_RADIUS, rotation)

        outside_local = np.array([1.1 * AXES[0], 0.0, 0.0])
        outside_world = CENTER + rotation @ outside_local
        outside_delta = outside_world - CENTER
        outside_value = float(outside_delta @ metric_world @ outside_delta)

        self.assertGreater(outside_value, 1.0)

    def test_old_transposed_form_would_mirror_the_ellipsoid(self):
        # Guard against reintroducing R.T @ D @ R: with an asymmetric
        # ellipsoid and a 30 deg yaw the wrong form scores the true +yaw
        # long-axis point far away from 1.0.
        rotation = yaw_rotation(YAW_RAD)
        metric_local = np.diag(1.0 / np.square(AXES + MAV_RADIUS))
        metric_wrong = rotation.T @ metric_local @ rotation

        local_boundary_point = np.array([AXES[0], 0.0, 0.0])
        world_boundary_point = CENTER + rotation @ local_boundary_point
        delta = world_boundary_point - CENTER
        wrong_value = float(delta @ metric_wrong @ delta)

        self.assertNotAlmostEqual(wrong_value, 1.0, places=6)


class EllipsoidalObstacleIntegrationTests(unittest.TestCase):
    """Collision geometry produced by the production obstacle class."""

    def _obstacle(self) -> EllipsoidalObstacle:
        # axes = sqrt(3)/2 * size  =>  size = axes * 2 / sqrt(3)
        box_size = AXES * 2.0 / np.sqrt(3.0)
        return EllipsoidalObstacle(
            position=CENTER.tolist(),
            size=box_size.tolist(),
            yaw=YAW_RAD,
        )

    def test_obstacle_axes_match_requested_geometry(self):
        obstacle = self._obstacle()
        np.testing.assert_allclose(obstacle.axes, AXES, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(obstacle.R_o, yaw_rotation(YAW_RAD), atol=1e-12)

    def test_collision_metric_follows_rotated_long_axis(self):
        obstacle = self._obstacle()
        metric_world = obstacle.get_omega(MAV_RADIUS)

        point_inside = obstacle.p_hat + obstacle.R_o @ np.array([1.9, 0.0, 0.0])
        point_outside = obstacle.p_hat + obstacle.R_o @ np.array([2.1, 0.0, 0.0])
        value_inside = float(
            (point_inside - obstacle.p_hat) @ metric_world @ (point_inside - obstacle.p_hat)
        )
        value_outside = float(
            (point_outside - obstacle.p_hat) @ metric_world @ (point_outside - obstacle.p_hat)
        )

        self.assertLessEqual(value_inside, 1.0)
        self.assertGreater(value_outside, 1.0)

    def test_get_omega_delegates_to_production_matrix(self):
        obstacle = self._obstacle()
        expected = Omega_matrix(obstacle.axes, MAV_RADIUS, obstacle.R_o)
        np.testing.assert_allclose(
            obstacle.get_omega(MAV_RADIUS),
            expected,
            rtol=0.0,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
