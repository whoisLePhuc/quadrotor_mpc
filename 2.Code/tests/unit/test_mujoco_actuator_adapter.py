"""Unit tests for MuJoCo ActuatorCommand4/data.ctrl adapter."""

from __future__ import annotations

import numpy as np
import pytest

from ccmpc.types import DataContractError
from simulation.engines.adapters.mujoco_actuator_adapter import (
    write_actuator_command4_to_ctrl,
)
from simulation.engines.base import EngineConfigurationError


def test_write_actuator_command4_to_ctrl_default_start() -> None:
    ctrl = np.zeros(4, dtype=np.float64)
    command = np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float64)

    applied = write_actuator_command4_to_ctrl(command=command, ctrl=ctrl)

    np.testing.assert_allclose(ctrl, command)
    np.testing.assert_allclose(applied, command)
    assert applied is not command


def test_write_actuator_command4_to_ctrl_offset() -> None:
    ctrl = np.zeros(6, dtype=np.float64)
    command = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    applied = write_actuator_command4_to_ctrl(command=command, ctrl=ctrl, start_index=2)

    np.testing.assert_allclose(ctrl[0:2], [0.0, 0.0])
    np.testing.assert_allclose(ctrl[2:6], command)
    np.testing.assert_allclose(applied, command)


def test_actuator_adapter_rejects_wrong_shape() -> None:
    with pytest.raises(DataContractError, match="ActuatorCommand4"):
        write_actuator_command4_to_ctrl(
            command=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            ctrl=np.zeros(4, dtype=np.float64),
        )


def test_actuator_adapter_rejects_negative_thrust() -> None:
    with pytest.raises(DataContractError, match="non-negative"):
        write_actuator_command4_to_ctrl(
            command=np.array([1.0, -2.0, 3.0, 4.0], dtype=np.float64),
            ctrl=np.zeros(4, dtype=np.float64),
        )


def test_actuator_adapter_rejects_short_ctrl() -> None:
    with pytest.raises(EngineConfigurationError, match="data.ctrl"):
        write_actuator_command4_to_ctrl(
            command=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            ctrl=np.zeros(3, dtype=np.float64),
        )


def test_actuator_adapter_rejects_negative_start_index() -> None:
    with pytest.raises(EngineConfigurationError, match="actuator_start_index"):
        write_actuator_command4_to_ctrl(
            command=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            ctrl=np.zeros(4, dtype=np.float64),
            start_index=-1,
        )
