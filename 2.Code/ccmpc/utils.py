"""Math utilities for Chance-Constrained MPC.

This module contains small, deterministic mathematical helpers used by the
core CC-MPC package.  It intentionally does not depend on the simulation layer.

Migrated from the legacy implementation and adapted to the new data contracts.

Main conventions
----------------
- Position/vector quantities are 3D NumPy arrays with dtype float64.
- Covariance matrices are 3x3 symmetric positive semi-definite matrices.
- Rotation matrices are 3x3 finite float64 arrays.
- Yaw rotation follows the project convention: world frame is Z-up.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np

from ccmpc.types import (
    FloatArray,
    as_position3,
    as_sigma3x3,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SQRT_PI_INV: float = 0.5641895835477563  # 1 / sqrt(pi)


# ---------------------------------------------------------------------------
# Inverse error function
# ---------------------------------------------------------------------------


def erfinv(y: float, tol: float = 1e-12, max_iter: int = 50) -> float:
    """Compute inverse error function using Newton iterations.

    Parameters
    ----------
    y:
        Input value in [-1, 1].
    tol:
        Absolute convergence tolerance on ``erf(x) - y``.
    max_iter:
        Maximum Newton iterations.

    Returns
    -------
    float
        Value ``x`` such that ``erf(x) ~= y``.

    Raises
    ------
    ValueError
        If ``y`` is outside [-1, 1], or if ``tol``/``max_iter`` are invalid.

    Notes
    -----
    The initial guess uses the Winitzki-style approximation used in the legacy
    code, followed by Newton refinement.  This avoids requiring SciPy for the
    core package.
    """
    y = _as_finite_scalar(y, "y")

    if tol <= 0.0 or not math.isfinite(tol):
        raise ValueError("tol must be finite and > 0.")

    if isinstance(max_iter, bool) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")

    if y < -1.0 or y > 1.0:
        raise ValueError(f"erfinv({y}): argument must be in [-1, 1].")

    if abs(y) == 1.0:
        return math.copysign(float("inf"), y)

    if y == 0.0:
        return 0.0

    sign = 1.0 if y > 0.0 else -1.0
    ya = abs(y)

    # Winitzki-style rational approximation as Newton initial guess.
    a_const = 0.147
    log_term = math.log(1.0 - ya * ya)
    t = 2.0 / (math.pi * a_const) + log_term / 2.0
    x = sign * math.sqrt(math.sqrt(t * t - log_term / a_const) - t)

    for _ in range(max_iter):
        fx = math.erf(x) - y

        if abs(fx) < tol:
            break

        derivative = 2.0 * _SQRT_PI_INV * math.exp(-x * x)

        if abs(derivative) < 1e-300:
            x += 0.1 if x >= 0.0 else -0.1
            continue

        x -= fx / derivative

    return float(x)


# ---------------------------------------------------------------------------
# Ellipsoid collision geometry
# ---------------------------------------------------------------------------


def Omega_matrix(
    axes: FloatArray,
    mav_radius: float,
    R_o: FloatArray,
) -> FloatArray:
    r"""Compute ellipsoidal collision matrix ``Omega``.

    The obstacle ellipsoid is inflated by the MAV collision radius.

    Formula
    -------
    ``Omega = R_o.T @ diag(1 / (axes + mav_radius)^2) @ R_o``

    Parameters
    ----------
    axes:
        Obstacle semi-principal axes ``[a, b, c]``.
    mav_radius:
        MAV collision radius.  Must be non-negative.
    R_o:
        Obstacle rotation matrix with shape ``(3, 3)``.

    Returns
    -------
    np.ndarray
        Symmetric positive-definite matrix with shape ``(3, 3)``.
    """
    axes_array = as_position3(axes)

    if np.any(axes_array <= 0.0):
        raise ValueError("axes must contain strictly positive semi-axis lengths.")

    radius = _as_finite_scalar(mav_radius, "mav_radius")
    if radius < 0.0:
        raise ValueError("mav_radius must be >= 0.")

    rotation = as_matrix3x3(R_o, name="R_o")

    inv_sq = 1.0 / (axes_array + radius) ** 2
    omega = rotation.T @ np.diag(inv_sq) @ rotation

    # Numerical symmetrization guards against tiny roundoff asymmetry.
    return symmetrize3x3(omega)


def Omega_half(Omega: FloatArray) -> FloatArray:
    r"""Compute Cholesky factor ``L`` such that ``L @ L.T = Omega``.

    Parameters
    ----------
    Omega:
        Symmetric positive-definite 3x3 matrix.

    Returns
    -------
    np.ndarray
        Lower-triangular Cholesky factor.
    """
    omega = as_matrix3x3(Omega, name="Omega")

    if not is_symmetric(omega):
        raise ValueError("Omega must be symmetric.")

    try:
        return np.linalg.cholesky(omega).astype(np.float64)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Omega must be positive definite for Cholesky factorization.") from exc


def box_to_ellipsoid_axes(size: FloatArray) -> FloatArray:
    r"""Convert box dimensions to circumscribing ellipsoid semi-axes.

    Formula
    -------
    ``axes = sqrt(3) / 2 * [length, width, height]``

    Parameters
    ----------
    size:
        Box dimensions ``[length, width, height]``.

    Returns
    -------
    np.ndarray
        Ellipsoid semi-axes ``[a, b, c]``.
    """
    size_array = as_position3(size)

    if np.any(size_array <= 0.0):
        raise ValueError("size must contain strictly positive box dimensions.")

    return 0.5 * math.sqrt(3.0) * size_array


# ---------------------------------------------------------------------------
# Chance constraint helper
# ---------------------------------------------------------------------------


def chance_constraint_rhs(
    L: FloatArray,
    Sigma_mav: FloatArray,
    Sigma_obs: FloatArray,
    n_o: FloatArray,
    delta: float,
) -> float:
    r"""Compute the RHS of the linearized chance constraint.

    The implementation uses the lower-triangular Cholesky convention
    ``L @ L.T = Omega``.  Therefore the covariance contraction is:

    ``n_o.T @ L @ (Sigma_mav + Sigma_obs) @ L.T @ n_o``

    Parameters
    ----------
    L:
        Cholesky factor of collision matrix ``Omega`` with shape ``(3, 3)``.
    Sigma_mav:
        MAV position covariance with shape ``(3, 3)``.
    Sigma_obs:
        Obstacle position covariance with shape ``(3, 3)``.
    n_o:
        Unit normal vector from obstacle to MAV with shape ``(3,)``.
    delta:
        Collision probability threshold.  Must satisfy ``0 < delta < 0.5``.

    Returns
    -------
    float
        Scalar RHS value.
    """
    L_array = as_matrix3x3(L, name="L")
    Sigma_mav_array = as_sigma3x3(Sigma_mav)
    Sigma_obs_array = as_sigma3x3(Sigma_obs)
    n_array = as_position3(n_o)

    delta_value = _as_finite_scalar(delta, "delta")
    if not (0.0 < delta_value < 0.5):
        raise ValueError("delta must satisfy 0 < delta < 0.5.")

    n_norm = float(np.linalg.norm(n_array))
    if n_norm <= 1e-12:
        raise ValueError("n_o must be non-zero.")

    n_unit = n_array / n_norm
    Sigma_combined = Sigma_mav_array + Sigma_obs_array

    inner_cov = L_array @ Sigma_combined @ L_array.T
    inner_cov = symmetrize3x3(inner_cov)

    variance_scaled = float(n_unit @ inner_cov @ n_unit)
    variance_scaled = max(variance_scaled, 0.0)

    sigma_scaled = math.sqrt(2.0 * variance_scaled)
    return float(erfinv(1.0 - 2.0 * delta_value) * sigma_scaled)


# ---------------------------------------------------------------------------
# Field-of-view helper
# ---------------------------------------------------------------------------


def detect_obstacle_in_fov(
    obstacles: Sequence[Any],
    mav_pos: FloatArray,
    mav_yaw: float,
    hfov_deg: float = 90.0,
    vfov_deg: float = 70.0,
    max_range: float = 5.0,
) -> list[tuple[float, Any]]:
    """Return obstacles inside camera field of view, sorted by distance.

    Each obstacle is expected to expose:
        - ``p_hat``: center position, shape ``(3,)``
        - ``axes``: ellipsoid semi-axes, shape ``(3,)``

    The body-frame convention is:
        - x forward
        - y left
        - z up
    """
    position = as_position3(mav_pos)
    yaw = _as_finite_scalar(mav_yaw, "mav_yaw")
    max_range_value = _as_finite_scalar(max_range, "max_range")

    if max_range_value <= 0.0:
        raise ValueError("max_range must be > 0.")

    hfov_half = math.radians(_as_finite_scalar(hfov_deg, "hfov_deg") / 2.0)
    vfov_half = math.radians(_as_finite_scalar(vfov_deg, "vfov_deg") / 2.0)

    if hfov_half <= 0.0 or vfov_half <= 0.0:
        raise ValueError("hfov_deg and vfov_deg must be > 0.")

    results: list[tuple[float, Any]] = []

    for obstacle in obstacles:
        obs_pos = as_position3(getattr(obstacle, "p_hat"))
        obs_axes = as_position3(getattr(obstacle, "axes"))

        if np.any(obs_axes <= 0.0):
            raise ValueError("obstacle.axes must contain strictly positive values.")

        relative = obs_pos - position
        center_distance = float(np.linalg.norm(relative))
        distance_to_edge = max(0.0, center_distance - float(np.max(obs_axes)))

        if distance_to_edge > max_range_value:
            continue

        # Transform world delta to yaw-only body frame.
        ct = math.cos(yaw)
        st = math.sin(yaw)

        x_body = ct * relative[0] + st * relative[1]
        y_body = -st * relative[0] + ct * relative[1]
        z_body = relative[2]

        if x_body <= 0.0:
            continue

        yaw_angle = abs(math.atan2(y_body, x_body))
        pitch_angle = abs(math.atan2(z_body, x_body))

        if yaw_angle <= hfov_half and pitch_angle <= vfov_half:
            results.append((distance_to_edge, obstacle))

    results.sort(key=lambda item: item[0])
    return results


# ---------------------------------------------------------------------------
# Yaw helpers
# ---------------------------------------------------------------------------


def yaw_from_velocity(v: FloatArray) -> float:
    """Compute desired yaw from velocity vector.

    Returns 0 when horizontal speed is too small.
    """
    velocity = as_position3(v)
    horizontal_speed = float(np.linalg.norm(velocity[:2]))

    if horizontal_speed <= 1e-6:
        return 0.0

    return float(math.atan2(velocity[1], velocity[0]))


def yaw_to_rotation(yaw: float) -> FloatArray:
    """Create 3D rotation matrix for yaw angle around the z-axis."""
    yaw_value = _as_finite_scalar(yaw, "yaw")
    ct = math.cos(yaw_value)
    st = math.sin(yaw_value)

    return np.array(
        [
            [ct, -st, 0.0],
            [st, ct, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def wrap_angle_pi(angle: float) -> float:
    """Wrap angle to interval [-pi, pi)."""
    angle_value = _as_finite_scalar(angle, "angle")
    return float((angle_value + math.pi) % (2.0 * math.pi) - math.pi)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def as_matrix3x3(value: FloatArray, *, name: str = "matrix") -> FloatArray:
    """Validate finite 3x3 float64 matrix."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to float64 array.") from exc

    if array.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {array.shape}.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array.copy()


def is_symmetric(matrix: FloatArray, *, atol: float = 1e-9) -> bool:
    """Return True if matrix is symmetric within absolute tolerance."""
    matrix_array = as_matrix3x3(matrix, name="matrix")
    return bool(np.allclose(matrix_array, matrix_array.T, atol=atol, rtol=0.0))


def symmetrize3x3(matrix: FloatArray) -> FloatArray:
    """Return ``0.5 * (matrix + matrix.T)`` after validating shape."""
    matrix_array = as_matrix3x3(matrix, name="matrix")
    return 0.5 * (matrix_array + matrix_array.T)


def _as_finite_scalar(value: float, name: str) -> float:
    """Convert value to finite scalar float."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite scalar, got bool.")

    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar.") from exc

    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")

    return scalar


__all__ = [
    "Omega_half",
    "Omega_matrix",
    "as_matrix3x3",
    "box_to_ellipsoid_axes",
    "chance_constraint_rhs",
    "detect_obstacle_in_fov",
    "erfinv",
    "is_symmetric",
    "symmetrize3x3",
    "wrap_angle_pi",
    "yaw_from_velocity",
    "yaw_to_rotation",
]
