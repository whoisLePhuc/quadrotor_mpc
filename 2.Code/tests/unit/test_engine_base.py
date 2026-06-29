"""Unit tests for physics engine base contracts.

Target module:
    simulation.engines.base

These tests verify the engine abstraction layer before implementing concrete
ODE or MuJoCo engines.
"""

from __future__ import annotations

import numpy as np
import pytest

from ccmpc.types import DataContractError

from simulation.engines.base import (
    DEFAULT_MUJOCO_METADATA,
    DEFAULT_ODE_METADATA,
    EngineCommandType,
    EngineConfigurationError,
    EngineMetadata,
    EngineStepError,
    EngineStepStatus,
    EngineType,
    command_dim_for_type,
    make_step_result,
    validate_engine_command,
    validate_step_dt,
)


def test_engine_metadata_valid_ode() -> None:
    """Default ODE metadata should describe a State9 + ControlCommand4 engine."""
    metadata = DEFAULT_ODE_METADATA

    assert metadata.engine_type is EngineType.ODE
    assert metadata.name == "ODEPhysicsEngine"
    assert metadata.command_type is EngineCommandType.CONTROL_COMMAND4
    assert metadata.state_dim == 9
    assert metadata.command_dim == 4
    assert metadata.supports_control_command is True
    assert metadata.supports_actuator_command is False
    assert metadata.uses_quaternion_internal is False
    assert metadata.deterministic is True


def test_engine_metadata_valid_mujoco() -> None:
    """Default MuJoCo metadata should expose State9 while using quaternion internally."""
    metadata = DEFAULT_MUJOCO_METADATA

    assert metadata.engine_type is EngineType.MUJOCO
    assert metadata.name == "MuJoCoPhysicsEngine"
    assert metadata.command_type is EngineCommandType.CONTROL_COMMAND4
    assert metadata.state_dim == 9
    assert metadata.command_dim == 4
    assert metadata.supports_control_command is True
    assert metadata.supports_actuator_command is True
    assert metadata.uses_quaternion_internal is True


def test_engine_metadata_reject_wrong_state_dim() -> None:
    """Engine metadata rejects any public state dimension other than State9."""
    with pytest.raises(EngineConfigurationError, match="state_dim"):
        EngineMetadata(
            engine_type=EngineType.ODE,
            name="BadEngine",
            command_type=EngineCommandType.CONTROL_COMMAND4,
            state_dim=8,
            command_dim=4,
            supports_control_command=True,
        )


def test_engine_metadata_reject_empty_name() -> None:
    """Engine metadata requires a non-empty human-readable name."""
    with pytest.raises(EngineConfigurationError, match="name"):
        EngineMetadata(
            engine_type=EngineType.ODE,
            name="",
            command_type=EngineCommandType.CONTROL_COMMAND4,
            state_dim=9,
            command_dim=4,
            supports_control_command=True,
        )


def test_engine_metadata_reject_inconsistent_command_dim() -> None:
    """Engine metadata rejects command_dim that conflicts with command_type."""
    with pytest.raises(EngineConfigurationError, match="command_dim"):
        EngineMetadata(
            engine_type=EngineType.ODE,
            name="BadCommandDimEngine",
            command_type=EngineCommandType.CONTROL_COMMAND4,
            state_dim=9,
            command_dim=3,
            supports_control_command=True,
        )


def test_command_dim_for_type() -> None:
    """Command dimension helper maps engine command types to canonical dimensions."""
    assert command_dim_for_type(EngineCommandType.CONTROL_COMMAND4) == 4
    assert command_dim_for_type(EngineCommandType.ACTUATOR_COMMAND4) == 4


def test_validate_control_command4() -> None:
    """validate_engine_command accepts finite ControlCommand4 with shape (4,)."""
    command = validate_engine_command(
        np.array([0.1, -0.2, 1.5, 0.3]),
        EngineCommandType.CONTROL_COMMAND4,
    )

    assert isinstance(command, np.ndarray)
    assert command.shape == (4,)
    assert command.dtype == np.float64
    assert np.allclose(command, [0.1, -0.2, 1.5, 0.3])

    with pytest.raises(DataContractError, match="ControlCommand4"):
        validate_engine_command(
            np.array([0.1, -0.2, 1.5]),
            EngineCommandType.CONTROL_COMMAND4,
        )


def test_validate_actuator_command4() -> None:
    """validate_engine_command accepts non-negative ActuatorCommand4."""
    command = validate_engine_command(
        np.array([1.0, 1.1, 1.2, 1.3]),
        EngineCommandType.ACTUATOR_COMMAND4,
    )

    assert isinstance(command, np.ndarray)
    assert command.shape == (4,)
    assert command.dtype == np.float64
    assert np.all(command >= 0.0)

    with pytest.raises(DataContractError, match="non-negative"):
        validate_engine_command(
            np.array([1.0, -0.1, 1.2, 1.3]),
            EngineCommandType.ACTUATOR_COMMAND4,
        )


def test_validate_step_dt() -> None:
    """validate_step_dt accepts finite positive dt and rejects invalid values."""
    assert validate_step_dt(0.02) == pytest.approx(0.02)
    assert validate_step_dt(np.float64(0.01)) == pytest.approx(0.01)

    invalid_values = [0.0, -0.01, np.nan, np.inf, True]

    for value in invalid_values:
        with pytest.raises(EngineStepError, match="dt"):
            validate_step_dt(value)


def test_make_step_result_valid() -> None:
    """make_step_result creates a validated StepResult with canonical State9."""
    state = np.array([0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03])
    command = np.array([0.1, -0.1, 0.5, 0.2])

    result = make_step_result(
        state=state,
        time=0.02,
        dt=0.02,
        step_index=1,
        command_type=EngineCommandType.CONTROL_COMMAND4,
        applied_command=command,
        status=EngineStepStatus.OK,
        diagnostics={"substeps": 1},
    )

    assert result.state.shape == (9,)
    assert result.time == pytest.approx(0.02)
    assert result.dt == pytest.approx(0.02)
    assert result.step_index == 1
    assert result.status is EngineStepStatus.OK
    assert result.command_type is EngineCommandType.CONTROL_COMMAND4
    assert result.applied_command is not None
    assert result.applied_command.shape == (4,)
    assert result.diagnostics == {"substeps": 1}


def test_make_step_result_reject_invalid_state() -> None:
    """make_step_result rejects state that does not satisfy State9 contract."""
    invalid_state = np.array([0.0, 0.0, 1.0])

    with pytest.raises(DataContractError, match="State9"):
        make_step_result(
            state=invalid_state,
            time=0.02,
            dt=0.02,
            step_index=1,
        )


def test_make_step_result_reject_invalid_applied_command() -> None:
    """make_step_result validates applied_command against command_type."""
    state = np.zeros(9)

    with pytest.raises(DataContractError, match="ControlCommand4"):
        make_step_result(
            state=state,
            time=0.02,
            dt=0.02,
            step_index=1,
            command_type=EngineCommandType.CONTROL_COMMAND4,
            applied_command=np.array([0.1, 0.2, 0.3]),
        )


def test_make_step_result_reject_invalid_dt() -> None:
    """make_step_result rejects invalid dt."""
    state = np.zeros(9)

    with pytest.raises(EngineStepError, match="dt"):
        make_step_result(
            state=state,
            time=0.02,
            dt=0.0,
            step_index=1,
        )
