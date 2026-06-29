"""
Chance-Constrained MPC for quadrotor obstacle avoidance.

Core public API. Import from here for the canonical interface:

    from ccmpc import CCMPC, QuadrotorDynamics, ObstacleManager, EllipsoidalObstacle
"""

from .ccmpc import CCMPC
from .dynamics import QuadrotorDynamics
from .mujoco_dynamics import (
    euler_to_quat,
    quat_to_euler,
)
from .obstacle import (
    EllipsoidalObstacle,
    HorizonObstacleData,
    ObstacleManager,
    associate_detections,
)
from .sensor import DepthSensor
from .uncertainty import UncertaintyPropagator, VIODriftModel
from .utils import (
    Omega_half,
    Omega_matrix,
    box_to_ellipsoid_axes,
    chance_constraint_rhs,
    detect_obstacle_in_fov,
    erfinv,
    yaw_from_velocity,
    yaw_to_rotation,
)

__all__ = [
    "CCMPC",
    "QuadrotorDynamics",
    "EllipsoidalObstacle",
    "HorizonObstacleData",
    "ObstacleManager",
    "associate_detections",
    "DepthSensor",
    "UncertaintyPropagator",
    "VIODriftModel",
    "Omega_half",
    "Omega_matrix",
    "box_to_ellipsoid_axes",
    "chance_constraint_rhs",
    "detect_obstacle_in_fov",
    "erfinv",
    "euler_to_quat",
    "quat_to_euler",
    "yaw_from_velocity",
    "yaw_to_rotation",
]
