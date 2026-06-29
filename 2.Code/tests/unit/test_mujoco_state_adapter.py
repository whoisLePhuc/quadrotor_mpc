"""Unit tests for MuJoCo State9/qpos/qvel adapters."""

from __future__ import annotations

import numpy as np
import pytest

from ccmpc.types import DataContractError
from simulation.engines.adapters.mujoco_state_adapter import (
    euler_zyx_to_quat_wxyz,
    quat_wxyz_to_euler_zyx,
    read_state9_from_mujoco_freejoint,
    write_state9_to_mujoco_freejoint,
)


def test_euler_quaternion_roundtrip_small_angles() -> None:
    attitude = (0.1, -0.2, 0.3)
    quat = euler_zyx_to_quat_wxyz(*attitude)
    recovered = quat_wxyz_to_euler_zyx(quat)

    assert quat.shape == (4,)
    assert np.linalg.norm(quat) == pytest.approx(1.0)
    np.testing.assert_allclose(recovered, attitude, atol=1e-12)


def test_state9_to_mujoco_freejoint_roundtrip() -> None:
    state = np.array(
        [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.01, -0.02, 0.03],
        dtype=np.float64,
    )
    qpos = np.zeros(7, dtype=np.float64)
    qvel = np.zeros(6, dtype=np.float64)

    write_state9_to_mujoco_freejoint(state=state, qpos=qpos, qvel=qvel)
    recovered = read_state9_from_mujoco_freejoint(qpos=qpos, qvel=qvel)

    np.testing.assert_allclose(recovered, state, atol=1e-12)


def test_state9_adapter_supports_nonzero_freejoint_addresses() -> None:
    state = np.array(
        [1.0, -2.0, 3.0, 0.4, -0.5, 0.6, 0.02, 0.03, -0.04],
        dtype=np.float64,
    )
    qpos = np.zeros(10, dtype=np.float64)
    qvel = np.zeros(9, dtype=np.float64)

    write_state9_to_mujoco_freejoint(
        state=state,
        qpos=qpos,
        qvel=qvel,
        qpos_adr=3,
        qvel_adr=3,
    )
    recovered = read_state9_from_mujoco_freejoint(
        qpos=qpos,
        qvel=qvel,
        qpos_adr=3,
        qvel_adr=3,
    )

    np.testing.assert_allclose(recovered, state, atol=1e-12)


def test_write_state9_resets_angular_velocity() -> None:
    state = np.zeros(9, dtype=np.float64)
    qpos = np.zeros(7, dtype=np.float64)
    qvel = np.ones(6, dtype=np.float64)

    write_state9_to_mujoco_freejoint(state=state, qpos=qpos, qvel=qvel)

    np.testing.assert_allclose(qvel[3:6], np.zeros(3), atol=1e-12)


def test_state_adapter_rejects_invalid_state_shape() -> None:
    with pytest.raises(DataContractError, match="State9"):
        write_state9_to_mujoco_freejoint(
            state=np.zeros(8, dtype=np.float64),
            qpos=np.zeros(7, dtype=np.float64),
            qvel=np.zeros(6, dtype=np.float64),
        )


def test_state_adapter_rejects_short_buffers() -> None:
    with pytest.raises(ValueError, match="qpos"):
        read_state9_from_mujoco_freejoint(
            qpos=np.zeros(6, dtype=np.float64),
            qvel=np.zeros(6, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="qvel"):
        read_state9_from_mujoco_freejoint(
            qpos=np.zeros(7, dtype=np.float64),
            qvel=np.zeros(5, dtype=np.float64),
        )
