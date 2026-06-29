"""Phase 4 metadata policy tests for MuJoCo engine boundary."""

from __future__ import annotations

import numpy as np
import pytest

from ccmpc.types import DataContractError
from simulation.engines.base import (
    DEFAULT_MUJOCO_METADATA,
    EngineCommandType,
    EngineConfigurationError,
    EngineMetadata,
    EngineType,
    validate_engine_command,
)


def test_default_mujoco_metadata_consumes_actuator_command4_only() -> None:
    """MuJoCo engine boundary should be ActuatorCommand4 -> State9."""
    metadata = DEFAULT_MUJOCO_METADATA

    assert metadata.engine_type is EngineType.MUJOCO
    assert metadata.name == "MuJoCoPhysicsEngine"
    assert metadata.command_type is EngineCommandType.ACTUATOR_COMMAND4
    assert metadata.state_dim == 9
    assert metadata.command_dim == 4
    assert metadata.supports_control_command is False
    assert metadata.supports_actuator_command is True
    assert metadata.uses_quaternion_internal is True
    assert metadata.deterministic is True


def test_mujoco_metadata_rejects_actuator_command_without_support_flag() -> None:
    """EngineMetadata should keep command_type and support flags consistent."""
    with pytest.raises(EngineConfigurationError, match="ACTUATOR_COMMAND4"):
        EngineMetadata(
            engine_type=EngineType.MUJOCO,
            name="BadMuJoCo",
            command_type=EngineCommandType.ACTUATOR_COMMAND4,
            supports_control_command=False,
            supports_actuator_command=False,
            uses_quaternion_internal=True,
        )


def test_mujoco_command_validation_rejects_control_shape() -> None:
    """A MuJoCo command must be a non-negative ActuatorCommand4."""
    valid = validate_engine_command(
        np.array([1.0, 1.1, 1.2, 1.3]),
        DEFAULT_MUJOCO_METADATA.command_type,
    )
    assert valid.shape == (4,)
    assert np.all(valid >= 0.0)

    with pytest.raises(DataContractError, match="non-negative"):
        validate_engine_command(
            np.array([0.1, -0.1, 0.2, 0.3]),
            DEFAULT_MUJOCO_METADATA.command_type,
        )
