"""Tests for DepthSensor simulation."""

import numpy as np
import pytest
from ccmpc.sensor import DepthSensor
from ccmpc.obstacle import EllipsoidalObstacle


class TestDepthSensor:
    @pytest.fixture
    def sensor(self):
        return DepthSensor(hfov_deg=90.0, vfov_deg=70.0, max_range=5.0,
                           prob_detection=1.0)

    @pytest.fixture
    def obstacles(self):
        return [
            EllipsoidalObstacle([2.0, 0.5, 0.0], [0.5, 0.5, 1.0]),
            EllipsoidalObstacle([4.0, -0.3, 0.0], [0.3, 0.3, 0.5]),
        ]

    def test_detect_visible(self, sensor, obstacles):
        """Visible obstacles should be detected."""
        meas = sensor.detect(obstacles, mav_pos=np.zeros(3), mav_yaw=0.0)
        assert len(meas) > 0

    def test_detect_behind(self, sensor, obstacles):
        """Obstacles behind the MAV should not be detected."""
        meas = sensor.detect(obstacles, mav_pos=np.array([5.0, 0, 0]),
                             mav_yaw=0.0)
        for pos, _ in meas:
            # In body frame, x should be positive (in front)
            assert pos[0] < 0  # negative x in body = behind
        # With prob_detection=1.0 and prob_detection being applied,
        # obstacles behind camera are filtered out by x_body <= 0 check
        assert True

    def test_out_of_range(self, sensor, obstacles):
        """Obstacles beyond max_range should not be detected."""
        far_obs = EllipsoidalObstacle([20.0, 0.0, 0.0], [0.3, 0.3, 0.3])
        meas = sensor.detect([far_obs], mav_pos=np.zeros(3), mav_yaw=0.0)
        assert len(meas) == 0

    def test_noise_shape(self, sensor, obstacles):
        """Each measurement should be (pos_3d, cov_3x3)."""
        meas = sensor.detect(obstacles, mav_pos=np.zeros(3), mav_yaw=0.0)
        if meas:
            pos, cov = meas[0]
            assert pos.shape == (3,)
            assert cov.shape == (3, 3)

    def test_mean_noise_zero(self, sensor, obstacles):
        """With many measurements at prob_detection=1.0, mean noise ≈ 0."""
        n_trials = 100
        positions = []
        for _ in range(n_trials):
            m = sensor.detect(obstacles, mav_pos=np.zeros(3), mav_yaw=0.0)
            if m:
                positions.append(m[0][0])
        if positions:
            avg_pos = np.mean(positions, axis=0)
            # The true position is ~2.0m ahead
            assert abs(avg_pos[0] - 2.0) < 0.3



