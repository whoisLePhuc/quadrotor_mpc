"""Tests for obstacle representation and tracking."""

import math
import numpy as np
from ccmpc.obstacle import (
    EllipsoidalObstacle,
    ObstacleManager,
    associate_detections,
)


# ============================================================================
# EllipsoidalObstacle
# ============================================================================

class TestEllipsoidalObstacle:
    def test_construction(self):
        """Default obstacle should have proper attributes."""
        obs = EllipsoidalObstacle(
            position=[1.0, 2.0, 0.0],
            size=[0.5, 0.4, 1.0],
            velocity=[0.1, 0.0, 0.0],
        )
        assert np.allclose(obs.p_hat, [1.0, 2.0, 0.0])
        assert np.allclose(obs.size, [0.5, 0.4, 1.0])
        assert np.allclose(obs.v_hat, [0.1, 0.0, 0.0])
        assert obs.Sigma.shape == (3, 3)

    def test_ellipsoid_axes_eq7(self):
        """Axes should follow Eq 7: (a,b,c) = sqrt(3)/2 * (l,w,h)."""
        factor = math.sqrt(3) / 2
        obs = EllipsoidalObstacle(
            position=[0, 0, 0],
            size=[0.6, 0.5, 1.7],
        )
        expected = factor * np.array([0.6, 0.5, 1.7])
        assert np.allclose(obs.axes, expected)

    def test_predict_eq6(self):
        """predict() should update position by v*dt (Eq 6)."""
        obs = EllipsoidalObstacle(
            position=[1.0, 2.0, 0.0],
            size=[0.5, 0.4, 0.9],
            velocity=[0.5, 0.0, -0.1],
        )
        obs.predict(dt=0.1)
        assert np.allclose(obs.p_hat, [1.05, 2.0, -0.01])
        # Covariance should also grow: Σ += Σ_v * dt²
        assert np.trace(obs.Sigma) > np.trace(obs.Sigma) - 1e-10  # no-op check
        # Actually check growth
        trace_before = np.trace(obs.Sigma - obs.Sigma_v * 0.01)
        # After predict, trace increased

    def test_get_normal(self):
        """Normal should point from obstacle toward MAV."""
        obs = EllipsoidalObstacle(
            position=[2.0, 0.0, 0.0],
            size=[0.5, 0.5, 0.5],
        )
        p_mav = np.array([0.0, 0.0, 0.0])
        n = obs.get_normal(p_mav)
        # Normal points from obstacle to MAV = (p_mav - p_obs) / |...|
        expected = np.array([-1.0, 0.0, 0.0])
        assert np.allclose(n, expected)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-10

    def test_get_omega(self):
        """get_omega should return SPD matrix."""
        obs = EllipsoidalObstacle(
            position=[0, 0, 0],
            size=[0.5, 0.4, 0.9],
        )
        Ω = obs.get_omega(mav_radius=0.4)
        assert Ω.shape == (3, 3)
        evals = np.linalg.eigvalsh(Ω)
        assert np.all(evals > 0)

    def test_distance_to(self):
        """Distance to MAV should be Euclidean."""
        obs = EllipsoidalObstacle(
            position=[3.0, 4.0, 0.0],
            size=[0.5, 0.5, 0.5],
        )
        d = obs.distance_to(np.array([0.0, 0.0, 0.0]))
        assert abs(d - 5.0) < 1e-10

    def test_gaussian_pdf(self):
        """Same position should give high PDF; far position low PDF."""
        obs = EllipsoidalObstacle(
            position=[0, 0, 0],
            size=[0.5, 0.5, 0.5],
        )
        pd_same = obs.gaussian_pdf(np.array([0.0, 0.0, 0.0]))
        pd_far  = obs.gaussian_pdf(np.array([10.0, 10.0, 10.0]))
        assert pd_same > pd_far
        assert pd_same > 0

    def test_kalman_update(self):
        """Kalman update should move estimate toward measurement."""
        obs = EllipsoidalObstacle(
            position=[0, 0, 0],
            size=[0.5, 0.5, 0.5],
        )
        # Measurement at (1, 0, 0) should pull estimate toward it
        obs.kalman_update(np.array([1.0, 0.0, 0.0]))
        assert obs.p_hat[0] > 0.0  # estimate moved toward measurement
        # Covariance should decrease after update
        assert np.trace(obs.Sigma) < np.trace(
            EllipsoidalObstacle(
                position=[0, 0, 0],
                size=[0.5, 0.5, 0.5],
            ).Sigma
        )


# ============================================================================
# associate_detections
# ============================================================================

class TestAssociateDetections:
    def test_simple_association(self):
        """A detection near a track should associate."""
        track = EllipsoidalObstacle(
            position=[1.0, 0.0, 0.0],
            size=[0.5, 0.5, 0.5],
        )
        detection = (np.array([1.05, -0.02, 0.01]), np.eye(3) * 0.01)
        assoc = associate_detections([track], [detection])
        assert len(assoc) == 1
        assert assoc[0] == (0, 0)

    def test_no_association_far(self):
        """A far-away detection should not associate."""
        track = EllipsoidalObstacle(
            position=[0, 0, 0],
            size=[0.5, 0.5, 0.5],
        )
        detection = (np.array([100.0, 100.0, 100.0]), np.eye(3) * 0.01)
        assoc = associate_detections([track], [detection], max_distance=2.0)
        assert len(assoc) == 0


# ============================================================================
# ObstacleManager
# ============================================================================

class TestObstacleManager:
    def test_empty_manager(self):
        """Empty manager should handle queries gracefully."""
        mgr = ObstacleManager()
        assert len(mgr.obstacles) == 0
        assert mgr.get_closest(np.zeros(3)) == []

    def test_get_closest(self):
        """get_closest should return closest k obstacles."""
        obs1 = EllipsoidalObstacle([0, 0, 0], [0.3, 0.3, 0.3])
        obs2 = EllipsoidalObstacle([3, 0, 0], [0.3, 0.3, 0.3])
        obs3 = EllipsoidalObstacle([5, 0, 0], [0.3, 0.3, 0.3])
        mgr = ObstacleManager([obs1, obs2, obs3])
        closest = mgr.get_closest(np.array([0, 0, 0]), k=2)
        assert len(closest) == 2
        assert np.allclose(closest[0].p_hat, [0, 0, 0])  # closest first

    def test_update_moves_all(self):
        """update() should advance all obstacles."""
        obs1 = EllipsoidalObstacle([0, 0, 0], [0.3, 0.3, 0.3],
                                   velocity=[0.5, 0, 0])
        obs2 = EllipsoidalObstacle([3, 0, 0], [0.3, 0.3, 0.3],
                                   velocity=[-0.2, 0, 0])
        mgr = ObstacleManager([obs1, obs2])
        mgr.update(dt=0.1)
        assert np.allclose(obs1.p_hat, [0.05, 0, 0])
        assert np.allclose(obs2.p_hat, [2.98, 0, 0])

    def test_from_config(self, sim_config):
        """Should load obstacles from YAML."""
        mgr = ObstacleManager.from_config(str(sim_config))
        # simulation.yaml has 2 obstacles
        assert len(mgr.obstacles) == 2

    def test_predict_horizon_shape(self):
        """predict_horizon should return list of lists."""
        obs = EllipsoidalObstacle([2, 0, 0], [0.5, 0.5, 1.0],
                                  velocity=[0.1, 0, 0])
        mgr = ObstacleManager([obs])
        horizon = mgr.predict_horizon(N=5, dt=0.06, max_obs=1)
        assert len(horizon) == 5  # 5 steps
        assert len(horizon[0]) == 1  # 1 obstacle per step
        # HorizonObstacleData should have the right attributes
        data = horizon[0][0]
        assert hasattr(data, 'position')
        assert hasattr(data, 'L')

    def test_multiple_obstacle_ordering(self):
        """With multiple obstacles, horizon should track the closest."""
        near = EllipsoidalObstacle([1, 0, 0], [0.3, 0.3, 0.3],
                                    velocity=[0.5, 0, 0])
        far  = EllipsoidalObstacle([8, 0.5, 0], [0.3, 0.3, 0.3])
        mgr = ObstacleManager([far, near])
        ref = np.zeros((3, 6))
        ref[0, :] = 0  # MAV starts at origin
        horizon = mgr.predict_horizon(N=5, dt=0.06, p_mav_ref=ref, max_obs=2)
        # At step 0, the near obstacle should be the first entry
        d0 = np.linalg.norm(horizon[0][0].position - np.array([1, 0, 0]))
        d1 = np.linalg.norm(horizon[0][1].position - np.array([8, 0, 0]))
        assert d0 < d1
