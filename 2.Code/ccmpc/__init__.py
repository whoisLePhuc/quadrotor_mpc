"""Public API for the quadrotor MPC package.

The numerical model and geometry helpers intentionally remain importable without
CVXPY or MuJoCo.  Heavy optional dependencies are loaded only when their classes
are requested, which keeps documentation builds and formula tests lightweight.
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
