"""Separate/reference 9-state QP CC-MPC implementation.

This package is the older 9-state CC-MPC baseline with its own dynamics,
ellipsoidal-obstacle geometry and uncertainty model.  It is a
reference/experimental implementation with its own interfaces: it is not the
canonical native validation pipeline, and the native Monte Carlo protocol
builds controllers from ``quadrotor_mpc.control.nmpc`` instead.  The
numerical model and geometry helpers remain importable without CVXPY or
MuJoCo; heavy optional dependencies are loaded only when their classes are
requested, which keeps documentation builds and formula tests lightweight.
"""

from __future__ import annotations

from .dynamics import QuadrotorDynamics
from .obstacle import EllipsoidalObstacle, ObstacleManager
from .risk import (
    chance_constraint_residual,
    collision_clearance,
    symmetric_matrix_sqrt,
)
from .uncertainty import UncertaintyPropagator, VIODriftModel

__all__ = [
    "CCMPC",
    "QuadrotorDynamics",
    "EllipsoidalObstacle",
    "ObstacleManager",
    "UncertaintyPropagator",
    "VIODriftModel",
    "chance_constraint_residual",
    "collision_clearance",
    "symmetric_matrix_sqrt",
]


def __getattr__(name: str):
    """Lazy-load the CVXPY controller so core math has no heavy dependency."""
    if name == "CCMPC":
        from .ccmpc import CCMPC

        return CCMPC
    raise AttributeError(name)
