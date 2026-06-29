"""MuJoCo state adapters for the canonical State9 contract.

This module is intentionally NumPy-only. It can be unit-tested without the
optional ``mujoco`` package.

Assumed free-joint layout:
- qpos: [x, y, z, qw, qx, qy, qz]
- qvel: [vx, vy, vz, wx, wy, wz]

Canonical State9 does not contain angular velocity. Therefore, writing State9
into MuJoCo qvel resets the rotational velocity block to zero.
"""

from __future__ import annotations

import math

import numpy as np

from ccmpc.types import FloatArray, as_state9


def euler_zyx_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> FloatArray:
    """Convert Euler ZYX attitude [roll, pitch, yaw] to quaternion [w, x, y, z]."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    quat = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
    norm = np.linalg.norm(quat)
    if norm == 0.0:  # pragma: no cover - defensive guard.
        raise ValueError("Quaternion norm is zero.")
    return quat / norm


def quat_wxyz_to_euler_zyx(quat: FloatArray) -> tuple[float, float, float]:
    """Convert quaternion [w, x, y, z] to Euler ZYX [roll, pitch, yaw]."""
    q = np.asarray(quat, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"quat must have shape (4,), got {q.shape}.")
    norm = np.linalg.norm(q)
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("quat must have finite non-zero norm.")
    w, x, y, z = q / norm

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def write_state9_to_mujoco_freejoint(
    *,
    state: FloatArray,
    qpos: FloatArray,
    qvel: FloatArray,
    qpos_adr: int = 0,
    qvel_adr: int = 0,
) -> None:
    """Write canonical State9 into MuJoCo free-joint qpos/qvel buffers."""
    s = as_state9(state)
    _validate_freejoint_buffers(qpos=qpos, qvel=qvel, qpos_adr=qpos_adr, qvel_adr=qvel_adr)

    qpos[qpos_adr : qpos_adr + 3] = s[0:3]
    qpos[qpos_adr + 3 : qpos_adr + 7] = euler_zyx_to_quat_wxyz(
        roll=float(s[6]),
        pitch=float(s[7]),
        yaw=float(s[8]),
    )

    qvel[qvel_adr : qvel_adr + 3] = s[3:6]
    qvel[qvel_adr + 3 : qvel_adr + 6] = 0.0


def read_state9_from_mujoco_freejoint(
    *,
    qpos: FloatArray,
    qvel: FloatArray,
    qpos_adr: int = 0,
    qvel_adr: int = 0,
) -> FloatArray:
    """Read canonical State9 from MuJoCo free-joint qpos/qvel buffers."""
    _validate_freejoint_buffers(qpos=qpos, qvel=qvel, qpos_adr=qpos_adr, qvel_adr=qvel_adr)

    position = qpos[qpos_adr : qpos_adr + 3]
    quat = qpos[qpos_adr + 3 : qpos_adr + 7]
    velocity = qvel[qvel_adr : qvel_adr + 3]
    roll, pitch, yaw = quat_wxyz_to_euler_zyx(quat)

    return as_state9(
        [
            position[0],
            position[1],
            position[2],
            velocity[0],
            velocity[1],
            velocity[2],
            roll,
            pitch,
            yaw,
        ]
    )


def _validate_freejoint_buffers(
    *,
    qpos: FloatArray,
    qvel: FloatArray,
    qpos_adr: int,
    qvel_adr: int,
) -> None:
    if qpos_adr < 0 or qvel_adr < 0:
        raise ValueError("qpos_adr and qvel_adr must be non-negative.")
    if qpos.shape[0] < qpos_adr + 7:
        raise ValueError(
            f"qpos has size {qpos.shape[0]}, cannot access freejoint qpos at {qpos_adr}."
        )
    if qvel.shape[0] < qvel_adr + 6:
        raise ValueError(
            f"qvel has size {qvel.shape[0]}, cannot access freejoint qvel at {qvel_adr}."
        )


__all__ = [
    "euler_zyx_to_quat_wxyz",
    "quat_wxyz_to_euler_zyx",
    "read_state9_from_mujoco_freejoint",
    "write_state9_to_mujoco_freejoint",
]
