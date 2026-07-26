"""
Depth sensor simulation for obstacle detection.

Simulates a forward-facing depth camera that detects obstacles
within the camera's field of view. Returns noisy measurements
of obstacle positions in the body frame.

This is a simplified model of the U-depth pipeline (Section 3)
from: Lin, Zhu, Alonso-Mora — ICRA 2020
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


class DepthSensor:
    """Simulated forward-facing depth camera.

    Detects obstacles within the camera FOV by checking their
    true positions and adding realistic measurement noise.
    Returns measurements in the body frame (camera frame).
    """

    def __init__(
        self,
        hfov_deg: float = 90.0,
        vfov_deg: float = 70.0,
        max_range: float = 5.0,
        sigma_range: float = 0.02,
        sigma_angular: float = 0.01,
        prob_detection: float = 0.95,
    ):
        self.hfov = math.radians(hfov_deg / 2.0)
        self.vfov = math.radians(vfov_deg / 2.0)
        self.max_range = max_range
        self.sigma_range = sigma_range
        self.sigma_angular = sigma_angular
        self.prob_detection = prob_detection

    def detect(
        self,
        obstacles: list,
        mav_pos: npt.NDArray[np.float64],
        mav_yaw: float,
    ) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Detect obstacles and return measurements.

        Each measurement: (position_body_frame, measurement_covariance)

        The measurement covariance models the quadratic depth error
        characteristic of stereo depth cameras (sigma_depth ∝ d²).
        """
        measurements: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []

        for obs in obstacles:
            # True position in world frame
            p_true_world = obs.p_hat.copy()

            # Transform to body frame
            dx = p_true_world - mav_pos
            ct, st = math.cos(mav_yaw), math.sin(mav_yaw)
            x_body = ct * dx[0] + st * dx[1]
            y_body = -st * dx[0] + ct * dx[1]
            z_body = dx[2]

            # Check if within FOV
            if x_body <= 0:
                continue
            d = math.sqrt(x_body**2 + y_body**2 + z_body**2)
            if d > self.max_range:
                continue
            yaw_angle = abs(math.atan2(y_body, x_body))
            pitch_angle = abs(math.atan2(z_body, x_body))
            if yaw_angle > self.hfov or pitch_angle > self.vfov:
                continue

            # Probabilistic detection (missed detections)
            if np.random.random() > self.prob_detection:
                continue

            # Measurement noise: quadratic in depth for stereo
            sigma_d = self.sigma_range * (1.0 + d**2)

            # Noisy measurement in body frame
            theta_h = math.atan2(y_body, x_body)
            theta_v = math.atan2(z_body, x_body)
            d_noisy = d + np.random.normal(0, sigma_d)
            th_noisy = theta_h + np.random.normal(0, self.sigma_angular)
            tv_noisy = theta_v + np.random.normal(0, self.sigma_angular)

            # Back to Cartesian body frame
            x_m = d_noisy * math.cos(th_noisy) * math.cos(tv_noisy)
            y_m = d_noisy * math.sin(th_noisy)
            z_m = d_noisy * math.sin(tv_noisy)

            # Measurement covariance in body frame (depth-dependent)
            R = np.diag([sigma_d**2, (d * self.sigma_angular)**2, (d * self.sigma_angular)**2])

            measurements.append((np.array([x_m, y_m, z_m]), R))

        return measurements
