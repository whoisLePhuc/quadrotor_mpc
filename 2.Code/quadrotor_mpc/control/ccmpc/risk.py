r"""Collision geometry and Gaussian chance-constraint utilities.

This module uses the *symmetric* positive-definite square root of the collision
matrix.  That choice removes the orientation ambiguity of a triangular Cholesky
factor for rotated ellipsoids and makes the implementation match the paper's
notation :math:`\Omega^{1/2}` directly.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy.special import ndtri

Array = npt.NDArray[np.float64]


def symmetric_matrix_sqrt(matrix: Array) -> Array:
    """Return the symmetric PSD square root ``S`` satisfying ``S @ S = matrix``."""
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    values, vectors = np.linalg.eigh(symmetric)
    if float(values.min()) < -1e-10:
        raise ValueError("matrix must be positive semidefinite")
    values = np.maximum(values, 0.0)
    return (vectors * np.sqrt(values)) @ vectors.T


def chance_constraint_residual(
    position: Array,
    obstacle_position: Array,
    omega: Array,
    covariance_mav: Array,
    covariance_obstacle: Array,
    delta: float,
) -> float:
    r"""Return the signed first-order chance-constraint residual.

    With ``S = Omega^(1/2)`` and ``y = S (p - p_o)``, nominal separation is
    ``||y|| >= 1``.  Projecting combined Gaussian uncertainty onto the outward
    normal ``n = y / ||y||`` gives

    ``g = ||y|| - 1 - Phi^-1(1-delta) * sigma_projected``.

    ``g >= 0`` is safe under the local Gaussian approximation.  ``delta=0.5``
    removes probabilistic tightening and therefore represents deterministic MPC.
    """
    if not 0.0 < delta <= 0.5:
        raise ValueError("delta must be in (0, 0.5]")

    S = symmetric_matrix_sqrt(np.asarray(omega, dtype=float))
    diff = np.asarray(position, dtype=float) - np.asarray(obstacle_position, dtype=float)
    transformed = S @ diff
    metric = float(np.linalg.norm(transformed))
    if metric < 1e-12:
        normal = np.array([1.0, 0.0, 0.0])
    else:
        normal = transformed / metric

    covariance = np.asarray(covariance_mav, dtype=float) + np.asarray(
        covariance_obstacle, dtype=float
    )
    variance = float(normal @ S @ covariance @ S.T @ normal)
    sigma = math.sqrt(max(variance, 0.0))
    beta = float(ndtri(1.0 - delta))
    return metric - 1.0 - beta * sigma


def collision_clearance(position: Array, obstacle_position: Array, omega: Array) -> float:
    """Approximate Euclidean clearance from a point to an ellipsoid surface.

    The value is exact along the ray from the ellipsoid centre through ``position``.
    It is positive outside, zero on the surface and negative inside.
    """
    diff = np.asarray(position, dtype=float) - np.asarray(obstacle_position, dtype=float)
    distance = float(np.linalg.norm(diff))
    if distance < 1e-12:
        return -1.0 / math.sqrt(float(np.linalg.eigvalsh(omega).max()))
    metric = math.sqrt(max(float(diff @ omega @ diff), 0.0))
    if metric < 1e-12:
        return -distance
    boundary_distance = distance / metric
    return distance - boundary_distance
