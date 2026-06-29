"""
Math utilities for Chance-Constrained MPC.

Formulas from:
  "Robust Vision-based Obstacle Avoidance for Micro Aerial Vehicles
   in Dynamic Environments" — Lin, Zhu, Alonso-Mora, ICRA 2020
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SQRT_PI_INV: float = 0.5641895835477563  # 1 / sqrt(pi)


# ---------------------------------------------------------------------------
# erfinv (inverse error function)
# ---------------------------------------------------------------------------
def erfinv(y: float, tol: float = 1e-12, max_iter: int = 50) -> float:
    """Inverse error function via Newton's method.

    Args:
        y: Input in [-1, 1].
        tol: Convergence tolerance.
        max_iter: Maximum Newton iterations.

    Returns:
        x such that erf(x) = y.

    Raises:
        ValueError: If |y| > 1.
    """
    if y < -1.0 or y > 1.0:
        raise ValueError(f"erfinv({y}): argument must be in [-1, 1]")
    if abs(y) == 1.0:
        return math.copysign(float("inf"), y)
    if y == 0.0:
        return 0.0

    sign = 1.0 if y > 0.0 else -1.0
    ya = abs(y)

    # Initial guess: Winitzki 2008 rational approximation
    a_const = 0.147
    t = 2.0 / (math.pi * a_const) + math.log(1.0 - ya * ya) / 2.0
    x = sign * math.sqrt(math.sqrt(t * t - math.log(1.0 - ya * ya) / a_const) - t)

    for _ in range(max_iter):
        fx = math.erf(x) - y
        if abs(fx) < tol:
            break
        d = 2.0 * _SQRT_PI_INV * math.exp(-x * x)
        if abs(d) < 1e-300:
            x += 0.1 if x >= 0.0 else -0.1
            continue
        x -= fx / d

    return x


# ---------------------------------------------------------------------------
# Omega matrix (Eq Collision)
# ---------------------------------------------------------------------------
def Omega_matrix(
    axes: npt.NDArray[np.float64],
    mav_radius: float,
    R_o: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Compute the ellipsoidal collision matrix :math:`\Omega`.

    .. math::

        \Omega = R_o^T \cdot
            \operatorname{diag}\!\left(
                \frac{1}{(a+r)^2},\;
                \frac{1}{(b+r)^2},\;
                \frac{1}{(c+r)^2}
            \right) \cdot R_o

    Args:
        axes: Semi-principal axes (a, b, c) of the obstacle ellipsoid.
        mav_radius: MAV collision radius r.
        R_o: Rotation matrix of the obstacle (3x3).

    Returns:
        Omega matrix (3x3).
    """
    inv_sq = 1.0 / (axes + mav_radius) ** 2
    return R_o.T @ np.diag(inv_sq) @ R_o


def Omega_half(Omega: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r"""Cholesky factor :math:`\Omega^{1/2}` such that :math:`L L^T = \Omega`.

    Args:
        Omega: Symmetric positive-definite matrix (3x3).

    Returns:
        Lower-triangular Cholesky factor L.
    """
    return np.linalg.cholesky(Omega)


# ---------------------------------------------------------------------------
# Chance constraint RHS (Eq 16 right-hand side)
# ---------------------------------------------------------------------------
def chance_constraint_rhs(
    L: npt.NDArray[np.float64],
    Sigma_mav: npt.NDArray[np.float64],
    Sigma_obs: npt.NDArray[np.float64],
    n_o: npt.NDArray[np.float64],
    delta: float,
) -> float:
    r"""Compute the RHS of the chance constraint (Eq 16).

    .. math::

        \text{RHS} = \operatorname{erf}^{-1}(1-2\delta) \cdot
            \sqrt{2 \; \mathbf{n}_o^T \Omega^{1/2} (\Sigma + \Sigma_o) \Omega^{1/2} \mathbf{n}_o}

    Args:
        L: Cholesky factor :math:`L` such that :math:`L L^T = \Omega` (3x3, lower triangular).
        Sigma_mav: MAV position covariance (3x3).
        Sigma_obs: Obstacle position covariance (3x3).
        n_o: Unit normal vector from obstacle to MAV (3,).
        delta: Collision probability threshold (e.g. 0.03).

    Returns:
        Scalar RHS value.

    Note:
        FIX (BUG 2 — MEDIUM): Original code computed ``L.T @ Sigma @ L`` (i.e.
        ``(Omega^{1/2})^T Sigma Omega^{1/2}``), but paper Eq 16 requires
        ``Omega^{1/2} Sigma (Omega^{1/2})^T`` with the **symmetric** matrix
        square root.  For diagonal Omega (axis-aligned obstacles) these are
        identical, so the bug was invisible.  For rotated obstacles the
        Cholesky L satisfies ``L @ L^T = Omega``, so the correct contraction
        is ``L @ Sigma @ L^T``, making the product symmetric and matching the
        paper's intention.
    """
    Sigma_combined = Sigma_mav + Sigma_obs
    # Correct form: n^T L (Sigma_mav + Sigma_obs) L^T n  (L lower triangular)
    inner_cov = L @ Sigma_combined @ L.T
    sigma_scaled = np.sqrt(2.0 * n_o @ inner_cov @ n_o)
    return erfinv(1.0 - 2.0 * delta) * sigma_scaled


# ---------------------------------------------------------------------------
# FOV obstacle detection (mpc_python pattern)
# ---------------------------------------------------------------------------
def detect_obstacle_in_fov(
    obstacles: list,
    mav_pos: npt.NDArray[np.float64],
    mav_yaw: float,
    hfov_deg: float = 90.0,
    vfov_deg: float = 70.0,
    max_range: float = 5.0,
) -> list:
    """Return obstacles within the camera's field of view, sorted by distance.

    Args:
        obstacles: List of EllipsoidalObstacle objects.
        mav_pos: MAV position (3,).
        mav_yaw: MAV yaw angle (rad).
        hfov_deg, vfov_deg: Horizontal/vertical FOV in degrees.
        max_range: Maximum detection range.

    Returns:
        List of (distance, obstacle) tuples for obstacles within FOV, sorted.
    """
    hfov = math.radians(hfov_deg / 2.0)
    vfov = math.radians(vfov_deg / 2.0)
    results = []

    for obs in obstacles:
        dx = obs.p_hat - mav_pos
        d = np.linalg.norm(dx)
        dist_to_edge = max(0.0, d - float(np.max(obs.axes)))
        if dist_to_edge > max_range:
            continue

        # Transform to body frame (x forward, y left, z up)
        ct, st = math.cos(mav_yaw), math.sin(mav_yaw)
        x_body = ct * dx[0] + st * dx[1]
        y_body = -st * dx[0] + ct * dx[1]
        z_body = dx[2]

        if x_body <= 0:
            continue  # behind camera

        # Check angular bounds
        yaw_angle = abs(math.atan2(y_body, x_body))
        pitch_angle = abs(math.atan2(z_body, x_body))

        if yaw_angle <= hfov and pitch_angle <= vfov:
            results.append((dist_to_edge, obs))

    results.sort(key=lambda x: x[0])
    return results


# ---------------------------------------------------------------------------
# Yaw reference from velocity
# ---------------------------------------------------------------------------
def yaw_from_velocity(v: npt.NDArray[np.float64]) -> float:
    """Compute desired yaw from velocity vector."""
    return math.atan2(v[1], v[0]) if np.linalg.norm(v[:2]) > 1e-6 else 0.0


# ---------------------------------------------------------------------------
# Ellipsoid bounding (Eq 7)
# ---------------------------------------------------------------------------
def box_to_ellipsoid_axes(
    size: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Convert box dimensions to circumscribing ellipsoid axes (Eq 7).

    .. math::

        (a, b, c) = \frac{\sqrt{3}}{2} (l, w, h)

    Args:
        size: Box dimensions (length, width, height).

    Returns:
        Ellipsoid semi-axes (a, b, c).
    """
    return 0.5 * math.sqrt(3.0) * np.array(size)


# ---------------------------------------------------------------------------
# Yaw to rotation matrix (z-axis rotation)
# ---------------------------------------------------------------------------
def yaw_to_rotation(yaw: float) -> npt.NDArray[np.float64]:
    """Create 3D rotation matrix for a yaw angle about z-axis."""
    ct, st = math.cos(yaw), math.sin(yaw)
    return np.array([
        [ct, -st, 0.0],
        [st,  ct, 0.0],
        [0.0, 0.0, 1.0],
    ])