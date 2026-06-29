"""Unit tests for canonical simulation data contracts.

Target module:
    ccmpc.types

These tests verify the first refactor contract layer:
- State9 ordering and validation
- ControlCommand4 validation
- ActuatorCommand4 non-negative thrust policy
- Gamma9x9 covariance validation
"""

from __future__ import annotations

import numpy as np
import pytest

from ccmpc.types import (
    ControlIndex,
    DataContractError,
    StateIndex,
    as_actuator_command4,
    as_control_command4,
    as_gamma9x9,
    as_state9,
)


def test_state9_valid() -> None:
    """State9 accepts a finite 9D vector and returns float64 ndarray."""
    state = as_state9([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03])

    assert isinstance(state, np.ndarray)
    assert state.shape == (9,)
    assert state.dtype == np.float64
    assert np.all(np.isfinite(state))


def test_state9_wrong_shape() -> None:
    """State9 rejects vectors whose shape is not exactly (9,)."""
    with pytest.raises(DataContractError, match="State9"):
        as_state9([1.0, 2.0, 3.0])

    with pytest.raises(DataContractError, match="State9"):
        as_state9(np.zeros((9, 1)))


def test_state9_nan() -> None:
    """State9 rejects NaN and Inf values."""
    state_with_nan = np.zeros(9)
    state_with_nan[StateIndex.X] = np.nan

    with pytest.raises(DataContractError, match="finite"):
        as_state9(state_with_nan)

    state_with_inf = np.zeros(9)
    state_with_inf[StateIndex.Y] = np.inf

    with pytest.raises(DataContractError, match="finite"):
        as_state9(state_with_inf)


def test_state9_field_order() -> None:
    """State9 field order is [x, y, z, vx, vy, vz, roll, pitch, yaw]."""
    state = as_state9([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 0.2, 0.3])

    assert state[StateIndex.X] == pytest.approx(1.0)
    assert state[StateIndex.Y] == pytest.approx(2.0)
    assert state[StateIndex.Z] == pytest.approx(3.0)

    assert state[StateIndex.VX] == pytest.approx(4.0)
    assert state[StateIndex.VY] == pytest.approx(5.0)
    assert state[StateIndex.VZ] == pytest.approx(6.0)

    assert state[StateIndex.ROLL] == pytest.approx(0.1)
    assert state[StateIndex.PITCH] == pytest.approx(0.2)
    assert state[StateIndex.YAW] == pytest.approx(0.3)


def test_control_command4_valid() -> None:
    """ControlCommand4 accepts [phi_c, theta_c, vz_c, psi_dot_c]."""
    command = as_control_command4([0.1, -0.2, 1.5, 0.3])

    assert isinstance(command, np.ndarray)
    assert command.shape == (4,)
    assert command.dtype == np.float64
    assert np.all(np.isfinite(command))

    assert command[ControlIndex.PHI_C] == pytest.approx(0.1)
    assert command[ControlIndex.THETA_C] == pytest.approx(-0.2)
    assert command[ControlIndex.VZ_C] == pytest.approx(1.5)
    assert command[ControlIndex.PSI_DOT_C] == pytest.approx(0.3)


def test_control_command4_wrong_shape() -> None:
    """ControlCommand4 rejects vectors whose shape is not exactly (4,)."""
    with pytest.raises(DataContractError, match="ControlCommand4"):
        as_control_command4([0.1, 0.2, 0.3])

    with pytest.raises(DataContractError, match="ControlCommand4"):
        as_control_command4(np.zeros((4, 1)))


def test_actuator_command4_negative_rejected() -> None:
    """ActuatorCommand4 rejects negative thrust by default."""
    with pytest.raises(DataContractError, match="non-negative"):
        as_actuator_command4([1.0, 1.0, -0.1, 1.0])

    actuator = as_actuator_command4([1.0, 1.1, 1.2, 1.3])

    assert actuator.shape == (4,)
    assert np.all(actuator >= 0.0)


def test_gamma9x9_valid_psd() -> None:
    """Gamma9x9 accepts a finite symmetric positive semi-definite 9x9 matrix."""
    gamma = np.eye(9) * 0.01

    validated = as_gamma9x9(gamma)

    assert isinstance(validated, np.ndarray)
    assert validated.shape == (9, 9)
    assert validated.dtype == np.float64
    assert np.allclose(validated, validated.T)
    assert np.all(np.linalg.eigvalsh(validated) >= -1e-9)


def test_gamma9x9_non_symmetric_rejected() -> None:
    """Gamma9x9 rejects non-symmetric covariance matrices."""
    gamma = np.eye(9)
    gamma[0, 1] = 0.5
    gamma[1, 0] = 0.0

    with pytest.raises(DataContractError, match="symmetric"):
        as_gamma9x9(gamma)
