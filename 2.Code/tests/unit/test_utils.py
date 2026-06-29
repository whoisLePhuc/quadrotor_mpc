"""Unit tests for CC-MPC math utilities.

Target module:
    ccmpc.utils

These tests cover deterministic math helpers used by obstacle geometry,
chance constraints, yaw/FOV utilities, and controller migration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pytest

from ccmpc.utils import (
    Omega_half,
    Omega_matrix,
    as_matrix3x3,
    box_to_ellipsoid_axes,
    chance_constraint_rhs,
    detect_obstacle_in_fov,
    erfinv,
    is_symmetric,
    symmetrize3x3,
    wrap_angle_pi,
    yaw_from_velocity,
    yaw_to_rotation,
)


@dataclass(frozen=True)
class FakeObstacle:
    """Minimal obstacle object for FOV tests."""

    p_hat: np.ndarray
    axes: np.ndarray


def test_erfinv_basic_values() -> None:
    """erfinv should invert math.erf for representative values."""
    values = [-0.9, -0.5, 0.0, 0.5, 0.9]

    for value in values:
        x = erfinv(value)
        assert math.erf(x) == pytest.approx(value, abs=1e-10)


def test_erfinv_boundary_values() -> None:
    """erfinv(+/-1) should return signed infinity."""
    assert erfinv(1.0) == math.inf
    assert erfinv(-1.0) == -math.inf


def test_erfinv_rejects_out_of_range() -> None:
    """erfinv should reject inputs outside [-1, 1]."""
    with pytest.raises(ValueError, match="argument must be in"):
        erfinv(1.1)

    with pytest.raises(ValueError, match="argument must be in"):
        erfinv(-1.1)


def test_erfinv_rejects_invalid_tolerance() -> None:
    """erfinv should reject invalid tolerance values."""
    with pytest.raises(ValueError, match="tol"):
        erfinv(0.5, tol=0.0)

    with pytest.raises(ValueError, match="tol"):
        erfinv(0.5, tol=np.nan)


def test_erfinv_rejects_invalid_max_iter() -> None:
    """erfinv should reject invalid Newton iteration count."""
    with pytest.raises(ValueError, match="max_iter"):
        erfinv(0.5, max_iter=0)

    with pytest.raises(ValueError, match="max_iter"):
        erfinv(0.5, max_iter=True)  # type: ignore[arg-type]


def test_as_matrix3x3_valid() -> None:
    """as_matrix3x3 should accept finite 3x3 matrices."""
    matrix = np.eye(3, dtype=np.float64)

    validated = as_matrix3x3(matrix, name="M")

    assert validated.shape == (3, 3)
    assert validated.dtype == np.float64
    assert np.allclose(validated, matrix)


def test_as_matrix3x3_rejects_wrong_shape() -> None:
    """as_matrix3x3 should reject non-3x3 matrices."""
    with pytest.raises(ValueError, match="shape"):
        as_matrix3x3(np.zeros((2, 3), dtype=np.float64), name="M")


def test_as_matrix3x3_rejects_nan() -> None:
    """as_matrix3x3 should reject non-finite values."""
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        as_matrix3x3(matrix, name="M")


def test_is_symmetric() -> None:
    """is_symmetric should detect symmetric matrices."""
    symmetric = np.array(
        [
            [1.0, 0.2, 0.3],
            [0.2, 2.0, 0.4],
            [0.3, 0.4, 3.0],
        ],
        dtype=np.float64,
    )
    non_symmetric = symmetric.copy()
    non_symmetric[0, 1] = 10.0

    assert is_symmetric(symmetric) is True
    assert is_symmetric(non_symmetric) is False


def test_symmetrize3x3() -> None:
    """symmetrize3x3 should return 0.5 * (M + M.T)."""
    matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.0, 4.0, 5.0],
            [0.0, 0.0, 6.0],
        ],
        dtype=np.float64,
    )

    result = symmetrize3x3(matrix)

    assert np.allclose(result, 0.5 * (matrix + matrix.T))
    assert is_symmetric(result)


def test_yaw_to_rotation_identity() -> None:
    """yaw_to_rotation(0) should return identity."""
    rotation = yaw_to_rotation(0.0)

    assert rotation.shape == (3, 3)
    assert np.allclose(rotation, np.eye(3))


def test_yaw_to_rotation_pi_over_two() -> None:
    """yaw_to_rotation(pi/2) should rotate x-axis to y-axis."""
    rotation = yaw_to_rotation(math.pi / 2.0)

    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    rotated = rotation @ x_axis

    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-12)


def test_yaw_to_rotation_is_orthonormal() -> None:
    """yaw_to_rotation should produce an orthonormal matrix."""
    rotation = yaw_to_rotation(0.37)

    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_yaw_from_velocity() -> None:
    """yaw_from_velocity should compute atan2(vy, vx)."""
    assert yaw_from_velocity(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert yaw_from_velocity(np.array([0.0, 1.0, 0.0])) == pytest.approx(math.pi / 2.0)
    assert yaw_from_velocity(np.array([-1.0, 0.0, 0.0])) == pytest.approx(math.pi)
    assert yaw_from_velocity(np.array([0.0, 0.0, 1.0])) == pytest.approx(0.0)


def test_wrap_angle_pi() -> None:
    """wrap_angle_pi should map angles to [-pi, pi)."""
    assert wrap_angle_pi(0.0) == pytest.approx(0.0)
    assert wrap_angle_pi(2.0 * math.pi) == pytest.approx(0.0)
    assert wrap_angle_pi(-2.0 * math.pi) == pytest.approx(0.0)
    assert wrap_angle_pi(3.0 * math.pi / 2.0) == pytest.approx(-math.pi / 2.0)


def test_box_to_ellipsoid_axes() -> None:
    """box_to_ellipsoid_axes should apply sqrt(3)/2 scaling."""
    size = np.array([2.0, 4.0, 6.0], dtype=np.float64)

    axes = box_to_ellipsoid_axes(size)

    expected = 0.5 * math.sqrt(3.0) * size
    assert np.allclose(axes, expected)


def test_box_to_ellipsoid_axes_rejects_non_positive_size() -> None:
    """box_to_ellipsoid_axes should reject non-positive dimensions."""
    with pytest.raises(ValueError, match="positive"):
        box_to_ellipsoid_axes(np.array([1.0, 0.0, 1.0], dtype=np.float64))


def test_omega_matrix_axis_aligned() -> None:
    """Omega_matrix should match diagonal formula for axis-aligned obstacles."""
    axes = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    radius = 0.0
    rotation = np.eye(3, dtype=np.float64)

    omega = Omega_matrix(axes, radius, rotation)

    expected = np.diag([1.0, 1.0 / 4.0, 1.0 / 16.0])
    assert np.allclose(omega, expected)
    assert is_symmetric(omega)


def test_omega_matrix_with_radius() -> None:
    """Omega_matrix should inflate axes by mav_radius."""
    axes = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    radius = 0.5
    rotation = np.eye(3, dtype=np.float64)

    omega = Omega_matrix(axes, radius, rotation)

    expected = np.diag(1.0 / (axes + radius) ** 2)
    assert np.allclose(omega, expected)


def test_omega_matrix_rejects_non_positive_axes() -> None:
    """Omega_matrix should reject zero or negative axes."""
    with pytest.raises(ValueError, match="axes"):
        Omega_matrix(
            axes=np.array([1.0, 0.0, 1.0], dtype=np.float64),
            mav_radius=0.0,
            R_o=np.eye(3),
        )


def test_omega_matrix_rejects_negative_radius() -> None:
    """Omega_matrix should reject negative MAV radius."""
    with pytest.raises(ValueError, match="mav_radius"):
        Omega_matrix(
            axes=np.array([1.0, 1.0, 1.0], dtype=np.float64),
            mav_radius=-0.1,
            R_o=np.eye(3),
        )


def test_omega_half_reconstructs_omega() -> None:
    """Omega_half should return L such that L @ L.T == Omega."""
    omega = np.array(
        [
            [2.0, 0.2, 0.1],
            [0.2, 1.5, 0.3],
            [0.1, 0.3, 1.2],
        ],
        dtype=np.float64,
    )

    L = Omega_half(omega)

    assert L.shape == (3, 3)
    assert np.allclose(L @ L.T, omega)


def test_omega_half_rejects_non_symmetric_matrix() -> None:
    """Omega_half should reject non-symmetric matrices."""
    omega = np.eye(3, dtype=np.float64)
    omega[0, 1] = 0.5

    with pytest.raises(ValueError, match="symmetric"):
        Omega_half(omega)


def test_omega_half_rejects_not_positive_definite() -> None:
    """Omega_half should reject symmetric matrices that are not positive definite."""
    omega = np.diag([1.0, 0.0, 1.0]).astype(np.float64)

    with pytest.raises(ValueError, match="positive definite"):
        Omega_half(omega)


def test_chance_constraint_rhs_positive() -> None:
    """chance_constraint_rhs should return a positive margin for delta < 0.5."""
    L = np.eye(3, dtype=np.float64)
    sigma_mav = np.eye(3, dtype=np.float64) * 0.01
    sigma_obs = np.eye(3, dtype=np.float64) * 0.04
    normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    delta = 0.05

    rhs = chance_constraint_rhs(L, sigma_mav, sigma_obs, normal, delta)

    expected = erfinv(1.0 - 2.0 * delta) * math.sqrt(2.0 * 0.05)
    assert rhs == pytest.approx(expected)
    assert rhs > 0.0


def test_chance_constraint_rhs_normal_is_normalized_internally() -> None:
    """chance_constraint_rhs should normalize non-unit n_o internally."""
    L = np.eye(3, dtype=np.float64)
    sigma_mav = np.eye(3, dtype=np.float64) * 0.01
    sigma_obs = np.eye(3, dtype=np.float64) * 0.04
    delta = 0.05

    rhs_unit = chance_constraint_rhs(
        L,
        sigma_mav,
        sigma_obs,
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        delta,
    )
    rhs_scaled = chance_constraint_rhs(
        L,
        sigma_mav,
        sigma_obs,
        np.array([10.0, 0.0, 0.0], dtype=np.float64),
        delta,
    )

    assert rhs_scaled == pytest.approx(rhs_unit)


def test_chance_constraint_rhs_rejects_invalid_delta() -> None:
    """chance_constraint_rhs should require 0 < delta < 0.5."""
    L = np.eye(3, dtype=np.float64)
    sigma = np.eye(3, dtype=np.float64)
    normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    for delta in [0.0, -0.1, 0.5, 1.0]:
        with pytest.raises(ValueError, match="delta"):
            chance_constraint_rhs(L, sigma, sigma, normal, delta)


def test_chance_constraint_rhs_rejects_zero_normal() -> None:
    """chance_constraint_rhs should reject zero normal vector."""
    with pytest.raises(ValueError, match="n_o"):
        chance_constraint_rhs(
            L=np.eye(3, dtype=np.float64),
            Sigma_mav=np.eye(3, dtype=np.float64),
            Sigma_obs=np.eye(3, dtype=np.float64),
            n_o=np.zeros(3, dtype=np.float64),
            delta=0.05,
        )


def test_detect_obstacle_in_fov_front_visible() -> None:
    """Obstacle in front of MAV should be detected."""
    obstacle = FakeObstacle(
        p_hat=np.array([3.0, 0.0, 1.0], dtype=np.float64),
        axes=np.array([0.5, 0.5, 0.5], dtype=np.float64),
    )

    detected = detect_obstacle_in_fov(
        [obstacle],
        mav_pos=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        mav_yaw=0.0,
        hfov_deg=90.0,
        vfov_deg=70.0,
        max_range=5.0,
    )

    assert len(detected) == 1
    assert detected[0][1] is obstacle
    assert detected[0][0] == pytest.approx(2.5)


def test_detect_obstacle_in_fov_behind_not_visible() -> None:
    """Obstacle behind camera should not be detected."""
    obstacle = FakeObstacle(
        p_hat=np.array([-3.0, 0.0, 1.0], dtype=np.float64),
        axes=np.array([0.5, 0.5, 0.5], dtype=np.float64),
    )

    detected = detect_obstacle_in_fov(
        [obstacle],
        mav_pos=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        mav_yaw=0.0,
    )

    assert detected == []


def test_detect_obstacle_in_fov_out_of_range_not_visible() -> None:
    """Obstacle beyond max range should not be detected."""
    obstacle = FakeObstacle(
        p_hat=np.array([20.0, 0.0, 1.0], dtype=np.float64),
        axes=np.array([0.5, 0.5, 0.5], dtype=np.float64),
    )

    detected = detect_obstacle_in_fov(
        [obstacle],
        mav_pos=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        mav_yaw=0.0,
        max_range=5.0,
    )

    assert detected == []


def test_detect_obstacle_in_fov_sorted_by_distance() -> None:
    """Detected obstacles should be sorted by distance to edge."""
    far = FakeObstacle(
        p_hat=np.array([4.0, 0.0, 1.0], dtype=np.float64),
        axes=np.array([0.5, 0.5, 0.5], dtype=np.float64),
    )
    near = FakeObstacle(
        p_hat=np.array([2.0, 0.0, 1.0], dtype=np.float64),
        axes=np.array([0.5, 0.5, 0.5], dtype=np.float64),
    )

    detected = detect_obstacle_in_fov(
        [far, near],
        mav_pos=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        mav_yaw=0.0,
        max_range=5.0,
    )

    assert [item[1] for item in detected] == [near, far]


def test_detect_obstacle_in_fov_respects_yaw() -> None:
    """Changing MAV yaw should rotate the visible direction."""
    obstacle_world_y = FakeObstacle(
        p_hat=np.array([0.0, 3.0, 1.0], dtype=np.float64),
        axes=np.array([0.5, 0.5, 0.5], dtype=np.float64),
    )

    not_detected_when_yaw_zero = detect_obstacle_in_fov(
        [obstacle_world_y],
        mav_pos=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        mav_yaw=0.0,
        hfov_deg=60.0,
    )
    detected_when_yaw_pi_half = detect_obstacle_in_fov(
        [obstacle_world_y],
        mav_pos=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        mav_yaw=math.pi / 2.0,
        hfov_deg=60.0,
    )

    assert not_detected_when_yaw_zero == []
    assert len(detected_when_yaw_pi_half) == 1
    assert detected_when_yaw_pi_half[0][1] is obstacle_world_y


def test_detect_obstacle_in_fov_rejects_invalid_max_range() -> None:
    """detect_obstacle_in_fov should reject non-positive max_range."""
    with pytest.raises(ValueError, match="max_range"):
        detect_obstacle_in_fov(
            [],
            mav_pos=np.zeros(3, dtype=np.float64),
            mav_yaw=0.0,
            max_range=0.0,
        )
