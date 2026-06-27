"""Tests for utility functions (math, geometry, chance constraints)."""

import math
import numpy as np
import pytest
from numpy.linalg import norm
from ccmpc.utils import (
    erfinv,
    Omega_matrix,
    Omega_half,
    chance_constraint_rhs,
    detect_obstacle_in_fov,
    yaw_from_velocity,
    box_to_ellipsoid_axes,
    yaw_to_rotation,
)
from ccmpc.obstacle import EllipsoidalObstacle


# ============================================================================
# erfinv (Eq 16, Lemma 2)
# ============================================================================

class TestErfinv:
    def test_zero(self):
        """erfinv(0) should be 0."""
        assert abs(erfinv(0.0)) < 1e-12

    def test_known_values(self):
        """Known values from standard erf tables."""
        assert abs(erfinv(0.5) - 0.476936) < 1e-5
        assert abs(erfinv(0.9) - 1.163087) < 1e-4
        assert abs(erfinv(0.99) - 1.821386) < 1e-3

    def test_symmetry(self):
        """erfinv(-y) should equal -erfinv(y)."""
        for y in np.linspace(0.001, 0.999, 10):
            assert abs(erfinv(-y) + erfinv(y)) < 1e-10

    def test_identity_with_erf(self):
        """erfinv(erf(x)) should be approximately x."""
        for x in np.linspace(-2.0, 2.0, 10):
            assert abs(erfinv(math.erf(x)) - x) < 1e-10

    def test_out_of_range(self):
        """Should raise ValueError for |y| > 1."""
        try:
            erfinv(1.5)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_delta_003(self):
        """For delta=0.03, erfinv(1-2*0.03) = erfinv(0.94) ≈ 1.56."""
        val = erfinv(1.0 - 2.0 * 0.03)
        assert 1.3 < val < 1.4


# ============================================================================
# Omega_matrix (Eq Collision)
# ============================================================================

class TestOmegaMatrix:
    def test_shape(self):
        """Omega should be 3x3."""
        axes = np.array([0.5, 0.4, 0.9])
        R = np.eye(3)
        Ω = Omega_matrix(axes, mav_radius=0.4, R_o=R)
        assert Ω.shape == (3, 3)

    def test_positive_definite(self):
        """Omega should be SPD (diagonal with positive entries)."""
        axes = np.array([0.5, 0.4, 0.9])
        R = np.eye(3)
        Ω = Omega_matrix(axes, mav_radius=0.4, R_o=R)
        evals = np.linalg.eigvalsh(Ω)
        assert np.all(evals > 0)

    def test_diagonal_with_identity_rotation(self):
        """With R=I, Omega should be diagonal."""
        axes = np.array([0.5, 0.4, 0.9])
        R = np.eye(3)
        Ω = Omega_matrix(axes, mav_radius=0.4, R_o=R)
        assert np.allclose(Ω, np.diag(np.diag(Ω)))

    def test_values(self):
        """Omega[i,i] = 1/(a_i + r)^2."""
        axes = np.array([0.5, 0.4, 0.9])
        r = 0.4
        R = np.eye(3)
        Ω = Omega_matrix(axes, r, R)
        expected = np.diag(np.array([
            1.0 / (0.5 + 0.4)**2,
            1.0 / (0.4 + 0.4)**2,
            1.0 / (0.9 + 0.4)**2,
        ]))
        assert np.allclose(Ω, expected)


# ============================================================================
# Omega_half (Cholesky factor)
# ============================================================================

class TestOmegaHalf:
    def test_reconstruction(self):
        """L @ L^T should equal Omega."""
        axes = np.array([0.5, 0.4, 0.9])
        R = np.eye(3)
        Ω = Omega_matrix(axes, mav_radius=0.4, R_o=R)
        L = Omega_half(Ω)
        assert np.allclose(L @ L.T, Ω)

    def test_lower_triangular(self):
        """L should be lower triangular."""
        axes = np.array([0.5, 0.4, 0.9])
        R = np.eye(3)
        Ω = Omega_matrix(axes, mav_radius=0.4, R_o=R)
        L = Omega_half(Ω)
        assert np.allclose(L, np.tril(L))


# ============================================================================
# chance_constraint_rhs (Eq 16)
# ============================================================================

class TestChanceConstraintRHS:
    def test_zero_uncertainty(self):
        """With zero uncertainty, RHS should be zero."""
        L = np.eye(3)
        Σ_mav = np.zeros((3, 3))
        Σ_obs = np.zeros((3, 3))
        n = np.array([1.0, 0.0, 0.0])
        rhs = chance_constraint_rhs(L, Σ_mav, Σ_obs, n, delta=0.03)
        assert abs(rhs) < 1e-10

    def test_increasing_with_uncertainty(self):
        """RHS should increase when uncertainty increases."""
        L = np.eye(3)
        n = np.array([1.0, 0.0, 0.0])
        δ = 0.03
        rhs_low = chance_constraint_rhs(
            L, np.diag([0.01, 0.01, 0.01]),
            np.diag([0.01, 0.01, 0.01]), n, δ)
        rhs_high = chance_constraint_rhs(
            L, np.diag([0.1, 0.1, 0.1]),
            np.diag([0.1, 0.1, 0.1]), n, δ)
        assert rhs_high > rhs_low

    def test_decreasing_with_delta(self):
        """Larger δ (looser) should give smaller RHS."""
        L = np.eye(3)
        Σ = np.diag([0.05, 0.05, 0.05]) ** 2
        n = np.array([1.0, 0.0, 0.0])
        rhs_strict = chance_constraint_rhs(L, Σ, Σ.copy(), n, delta=0.01)
        rhs_loose  = chance_constraint_rhs(L, Σ, Σ.copy(), n, delta=0.1)
        assert rhs_loose < rhs_strict


# ============================================================================
# detect_obstacle_in_fov
# ============================================================================

class TestFOVDetection:
    @pytest.fixture
    def obs_list(self):
        return [
            EllipsoidalObstacle([3.0, 0.5, 0.0], [0.5, 0.5, 1.0]),
            EllipsoidalObstacle([1.0, 3.0, 0.0], [0.3, 0.3, 0.5]),
        ]

    def test_detect_visible_obstacle(self, obs_list):
        """An obstacle directly ahead should be detected."""
        pos = np.array([0.0, 0.0, 0.0])
        visible = detect_obstacle_in_fov(obs_list, pos, mav_yaw=0.0)
        # Obstacle at (3, 0.5) is at ~9.5° horizontal → within 90° FOV
        assert len(visible) >= 1

    def test_detect_behind_obstacle(self, obs_list):
        """Obstacle behind the camera should not be detected."""
        pos = np.array([5.0, 0.0, 0.0])
        visible = detect_obstacle_in_fov(obs_list, pos, mav_yaw=0.0)
        # Both obstacles are behind the camera (negative x in body frame)
        for _, obs in visible:
            dx = obs.p_hat - pos
            ct, st = math.cos(0), math.sin(0)
            x_body = ct * dx[0] + st * dx[1]
            if x_body <= 0:
                assert False, "Behind obstacle detected!"

    def test_fov_angular_limit(self, obs_list):
        """Obstacle outside FOV should not be detected."""
        far_obs = EllipsoidalObstacle([1.0, 10.0, 0.0], [0.3, 0.3, 0.3])
        pos = np.array([0.0, 0.0, 0.0])
        visible = detect_obstacle_in_fov([far_obs], pos, mav_yaw=0.0,
                                          hfov_deg=60.0)
        # Obstacle at (1, 10) is at ~84° horizontal → outside 60° FOV
        assert len(visible) == 0

    def test_sorted_by_distance(self, obs_list):
        """Results should be sorted by distance (closest first)."""
        pos = np.array([0.0, 0.0, 0.0])
        visible = detect_obstacle_in_fov(obs_list, pos, mav_yaw=0.0)
        distances = [d for d, _ in visible]
        assert distances == sorted(distances)


# ============================================================================
# yaw_from_velocity
# ============================================================================

class TestYawFromVelocity:
    def test_east(self):
        """Positive x → yaw = 0."""
        v = np.array([1.0, 0.0, 0.0])
        assert abs(yaw_from_velocity(v)) < 1e-10

    def test_north(self):
        """Positive y → yaw = π/2."""
        v = np.array([0.0, 1.0, 0.0])
        assert abs(yaw_from_velocity(v) - math.pi / 2) < 1e-10

    def test_west(self):
        """Negative x → yaw = π."""
        v = np.array([-1.0, 0.0, 0.0])
        assert abs(abs(yaw_from_velocity(v)) - math.pi) < 1e-10

    def test_zero_velocity(self):
        """Zero velocity → yaw = 0."""
        v = np.array([0.0, 0.0, 0.0])
        assert abs(yaw_from_velocity(v)) < 1e-10


# ============================================================================
# box_to_ellipsoid_axes (Eq 7)
# ============================================================================

class TestBoxToEllipsoid:
    def test_factor(self):
        """(a,b,c) = sqrt(3)/2 * (l,w,h)."""
        size = np.array([1.0, 1.0, 1.0])
        axes = box_to_ellipsoid_axes(size)
        expected = np.array([math.sqrt(3) / 2] * 3)
        assert np.allclose(axes, expected)

    def test_circumscribes(self):
        """Ellipsoid should circumscribe the box (box corner lies on surface)."""
        size = np.array([0.6, 0.5, 1.7])
        axes = box_to_ellipsoid_axes(size)
        half = size / 2
        corner = half
        val = (corner[0] / axes[0])**2 + (corner[1] / axes[1])**2 + (corner[2] / axes[2])**2
        assert abs(val - 1.0) < 1e-10

    def test_positive(self):
        """Axes should always be positive."""
        assert np.all(box_to_ellipsoid_axes(np.array([0.1, 0.2, 0.3])) > 0)


# ============================================================================
# yaw_to_rotation
# ============================================================================

class TestYawToRotation:
    def test_shape(self):
        """Output should be 3x3."""
        R = yaw_to_rotation(0.5)
        assert R.shape == (3, 3)

    def test_determinant(self):
        """Rotation matrix should have det = 1."""
        R = yaw_to_rotation(0.5)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_orthogonal(self):
        """R^T R should equal I."""
        R = yaw_to_rotation(0.5)
        assert np.allclose(R.T @ R, np.eye(3))

    def test_yaw_zero(self):
        """Yaw=0 should give identity."""
        R = yaw_to_rotation(0.0)
        assert np.allclose(R, np.eye(3))

    def test_yaw_pi(self):
        """Yaw=π should rotate x to -x."""
        R = yaw_to_rotation(math.pi)
        x = np.array([1.0, 0.0, 0.0])
        Rx = R @ x
        assert np.allclose(Rx, [-1.0, 0.0, 0.0])
