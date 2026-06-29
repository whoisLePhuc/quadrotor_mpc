"""MuJoCo actuator adapter for the canonical ActuatorCommand4 contract."""

from __future__ import annotations

from ccmpc.types import FloatArray, as_actuator_command4
from simulation.engines.base import EngineConfigurationError


def write_actuator_command4_to_ctrl(
    *,
    command: FloatArray,
    ctrl: FloatArray,
    start_index: int = 0,
) -> FloatArray:
    """Write ActuatorCommand4 [T1, T2, T3, T4] into MuJoCo ``data.ctrl``.

    Returns a validated copy of the command actually written at the engine
    boundary. Negative thrust values are rejected by the canonical data contract.
    """
    actuator = as_actuator_command4(command)

    if start_index < 0:
        raise EngineConfigurationError("actuator_start_index must be non-negative.")
    if ctrl.shape[0] < start_index + 4:
        raise EngineConfigurationError(
            f"MuJoCo data.ctrl has size {ctrl.shape[0]}, cannot write 4 actuator "
            f"commands at index {start_index}."
        )

    ctrl[start_index : start_index + 4] = actuator
    return actuator.copy()


__all__ = ["write_actuator_command4_to_ctrl"]
