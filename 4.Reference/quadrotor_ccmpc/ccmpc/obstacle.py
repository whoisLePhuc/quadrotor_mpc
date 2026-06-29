"""
Obstacle representation and tracking for Chance-Constrained MPC.

Formulas from:
  "Robust Vision-based Obstacle Avoidance for Micro Aerial Vehicles
   in Dynamic Environments" — Lin, Zhu, Alonso-Mora, ICRA 2020

Equations:
  (6)  Constant velocity prediction
  (7)  Box-to-ellipsoid bounding
  (Collision) Omega matrix definition
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from .utils import (
    Omega_matrix,
    Omega_half,
    yaw_to_rotation,
    box_to_ellipsoid_axes,
)


class EllipsoidalObstacle:
    """An ellipsoidal obstacle with position, velocity, size, and orientation.

    The obstacle is a bounding ellipsoid circumscribing a box detection,
    with Kalman-filter-style uncertainty tracking.
    """

    def __init__(
        self,
        position: npt.NDArray[np.float64] | list[float],
        size: npt.NDArray[np.float64] | list[float],
        yaw: float = 0.0,
        velocity: npt.NDArray[np.float64] | list[float] | None = None,
        pos_uncertainty: float = 0.05,
        vel_uncertainty: float = 0.1,
    ):
        # Position and velocity
        self.p_hat: npt.NDArray[np.float64] = np.array(position, dtype=np.float64)
        self.v_hat: npt.NDArray[np.float64] = (
            np.array(velocity, dtype=np.float64) if velocity is not None
            else np.zeros(3)
        )

        # Size (l, w, h) and ellipsoidal axes (a, b, c) per Eq 7
        self.size: npt.NDArray[np.float64] = np.array(size, dtype=np.float64)
        self.axes: npt.NDArray[np.float64] = box_to_ellipsoid_axes(self.size)

        # Orientation
        self.yaw: float = yaw
        self.R_o: npt.NDArray[np.float64] = yaw_to_rotation(yaw)

        # Uncertainty (Kalman filter style)
        self.Sigma: npt.NDArray[np.float64] = np.diag([pos_uncertainty**2] * 3)
        self.Sigma_v: npt.NDArray[np.float64] = np.diag([vel_uncertainty**2] * 3)

    def predict(self, dt: float) -> None:
        r"""Constant-velocity prediction (Eq 6).

        .. math::

            \hat{p}^{k+1} &= \hat{p}^k + \hat{v}^k \Delta t \\
            \Sigma^{k+1} &= \Sigma^k + \Sigma_v \Delta t^2
        """
        self.p_hat += self.v_hat * dt
        self.Sigma += self.Sigma_v * dt**2

    def get_normal(
        self, p_mav: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Unit normal vector from obstacle to MAV.

        .. math::

            n_o = \frac{\hat{p}_{mav} - \hat{p}_o}{\|\hat{p}_{mav} - \hat{p}_o\|}
        """
        diff = p_mav - self.p_hat
        dist = np.linalg.norm(diff)
        if dist < 1e-10:
            return np.array([1.0, 0.0, 0.0])
        return diff / dist

    def get_omega(self, mav_radius: float) -> npt.NDArray[np.float64]:
        r"""Compute the collision matrix :math:`\Omega`.

        See Eq (Collision) in the paper.
        """
        return Omega_matrix(self.axes, mav_radius, self.R_o)

    def get_omega_half(self, mav_radius: float) -> npt.NDArray[np.float64]:
        r"""Cholesky factor :math:`\Omega^{1/2}`."""
        return Omega_half(self.get_omega(mav_radius))

    def distance_to(
        self, p_mav: npt.NDArray[np.float64]
    ) -> float:
        """Euclidean distance from MAV to obstacle center."""
        return float(np.linalg.norm(p_mav - self.p_hat))

    def gaussian_pdf(
        self, measurement: npt.NDArray[np.float64]
    ) -> float:
        """Gaussian PDF for data association (Eq 5).

        Evaluates p(measurement | predicted_state, predicted_cov).
        Higher value = more likely association.
        """
        diff = measurement - self.p_hat
        cov = self.Sigma
        det = np.linalg.det(2.0 * math.pi * cov)
        if det <= 0:
            return 0.0
        mahal = diff @ np.linalg.solve(cov, diff)
        return float(np.exp(-0.5 * mahal) / np.sqrt(det))

    def kalman_update(
        self,
        measurement: npt.NDArray[np.float64],
        R: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """Kalman filter measurement update for position.

        Args:
            measurement: Observed position (3,).
            R: Measurement noise covariance (3x3). Default: identity scaled.
        """
        if R is None:
            R = np.eye(3) * 0.05**2
        # Kalman gain
        S = self.Sigma + R
        K = np.linalg.solve(S, self.Sigma).T  # K = Sigma @ inv(S)
        # Update
        innovation = measurement - self.p_hat
        self.p_hat += K @ innovation
        self.Sigma = (np.eye(3) - K) @ self.Sigma


def associate_detections(
    tracks: list[EllipsoidalObstacle],
    detections: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    threshold: float = 0.01,
    max_distance: float = 2.0,
) -> list[tuple[int, int]]:
    """Associate detections to existing tracks using Gaussian PDF (Eq 5).

    Args:
        tracks: Existing tracked obstacles.
        detections: List of (position, covariance) tuples from sensor.
        threshold: Minimum PDF value for association.
        max_distance: Maximum Euclidean distance for association.

    Returns:
        List of (track_idx, detection_idx) associations.
    """
    associations: list[tuple[int, int]] = []
    assigned_tracks = set()
    assigned_dets = set()

    for ti, track in enumerate(tracks):
        best_pd = 0.0
        best_di = -1
        for di, (pos, _) in enumerate(detections):
            if di in assigned_dets:
                continue
            dist = np.linalg.norm(pos - track.p_hat)
            if dist > max_distance:
                continue
            pd = track.gaussian_pdf(pos)
            if pd > best_pd and pd >= threshold:
                best_pd = pd
                best_di = di

        if best_di >= 0:
            associations.append((ti, best_di))
            assigned_tracks.add(ti)
            assigned_dets.add(best_di)

    return associations


class ObstacleManager:
    """Manages multiple ellipsoidal obstacles over time.

    Handles:
    - Updating positions of moving obstacles
    - Predicting states over the MPC horizon
    - Returning the closest N obstacles to the MPC
    """

    def __init__(self, obstacles: list[EllipsoidalObstacle] | None = None):
        self.obstacles: list[EllipsoidalObstacle] = (
            obstacles if obstacles is not None else []
        )

    @classmethod
    def from_config(cls, config: str | dict) -> "ObstacleManager":
        """Create obstacles from YAML simulation config."""
        if isinstance(config, str):
            import yaml as _yaml
            with open(config) as f:
                config_data = _yaml.safe_load(f)
        else:
            config_data = config

        obs_list: list[EllipsoidalObstacle] = []
        for obs_cfg in config_data.get("obstacles", []):
            obs_list.append(EllipsoidalObstacle(
                position=obs_cfg["position"],
                size=obs_cfg["size"],
                yaw=obs_cfg.get("yaw", 0.0),
                velocity=obs_cfg.get("velocity", [0.0, 0.0, 0.0]),
            ))
        return cls(obs_list)

    def update(self, dt: float) -> None:
        """Advance all obstacles by dt (moving obstacles)."""
        for obs in self.obstacles:
            obs.predict(dt)

    def get_closest(
        self,
        p_mav: npt.NDArray[np.float64],
        k: int = 2,
    ) -> list[EllipsoidalObstacle]:
        """Return the k closest obstacles to the MAV."""
        if not self.obstacles:
            return []
        distances = [obs.distance_to(p_mav) for obs in self.obstacles]
        sorted_pairs = sorted(zip(distances, self.obstacles))
        return [obs for _, obs in sorted_pairs[:k]]

    def predict_horizon(
        self,
        N: int,
        dt: float,
        p_mav_ref: npt.NDArray[np.float64] | None = None,
        max_obs: int = 2,
    ) -> list[list["HorizonObstacleData"]]:
        """Predict obstacle states over the MPC horizon.

        For each obstacle, returns a list of HorizonObstacleData (one per step).

        Args:
            N: Number of horizon steps.
            dt: Timestep.
            p_mav_ref: Reference MAV trajectory for closest selection (shape (3, N+1)).
            max_obs: Maximum number of obstacles to track.

        Returns:
            List of obstacle data sequences per horizon step.
            Outer list: steps k=0..N-1.
            Inner list: obstacles at that step.
        """
        horizon_data: list[list[HorizonObstacleData]] = []
        # Work on copies
        obs_copy = [
            EllipsoidalObstacle(
                o.p_hat.copy(), o.size.copy(), o.yaw, o.v_hat.copy()
            )
            for o in self.obstacles
        ]

        for k in range(N):
            step_data: list[HorizonObstacleData] = []
            # Select closest at this step
            ref_pos = p_mav_ref[:, k] if p_mav_ref is not None else np.zeros(3)
            closest = sorted(
                obs_copy,
                key=lambda o: o.distance_to(ref_pos),
            )[:max_obs]

            for obs in closest:
                L = obs.get_omega_half(0.4)  # default mav_radius
                step_data.append(HorizonObstacleData(
                    position=obs.p_hat.copy(),
                    velocity=obs.v_hat.copy(),
                    axes=obs.axes.copy(),
                    R_o=obs.R_o.copy(),
                    Sigma=obs.Sigma.copy(),
                    L=L,
                ))

            horizon_data.append(step_data)

            # Advance all copies
            for obs in obs_copy:
                obs.predict(dt)

        return horizon_data


class HorizonObstacleData:
    """Snapshot of obstacle data at a single horizon step."""

    def __init__(
        self,
        position: npt.NDArray[np.float64],
        velocity: npt.NDArray[np.float64],
        axes: npt.NDArray[np.float64],
        R_o: npt.NDArray[np.float64],
        Sigma: npt.NDArray[np.float64],
        L: npt.NDArray[np.float64],
    ):
        self.position = position
        self.velocity = velocity
        self.axes = axes
        self.R_o = R_o
        self.Sigma = Sigma
        self.L = L  # Omega^{1/2} (Cholesky factor)
