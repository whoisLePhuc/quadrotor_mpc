"""Base physics-engine abstraction for the quadrotor simulation.

This module defines the public contract that every physics backend must follow.
The runtime should depend on this interface only, not on a concrete ODE or
MuJoCo implementation.

Canonical boundary
------------------
All engines expose the same external state contract:

    State9 = [x, y, z, vx, vy, vz, roll, pitch, yaw]

Engines consume exactly one public command contract:

    ControlCommand4  = [phi_c, theta_c, vz_c, psi_dot_c]
    ActuatorCommand4 = [T1, T2, T3, T4]

Phase 4 policy:
- ODEPhysicsEngine consumes ControlCommand4 directly.
- MuJoCoPhysicsEngine consumes ActuatorCommand4 directly.
- ControlCommand4 -> ActuatorCommand4 mixing stays outside MuJoCoPhysicsEngine.
- MuJoCo qpos/qvel/quaternion/MjModel/MjData must not leak to runtime/controller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np

from ccmpc.types import (
    ACTUATOR_DIM,
    CONTROL_DIM,
    STATE_DIM,
    FloatArray,
    as_actuator_command4,
    as_control_command4,
    as_state9,
)


class EngineError(RuntimeError):
    """Base exception raised by physics-engine implementations."""


class EngineConfigurationError(EngineError):
    """Raised when an engine is constructed with invalid configuration."""


class EngineStateError(EngineError):
    """Raised when an engine state is invalid or unavailable."""


class EngineStepError(EngineError):
    """Raised when an engine fails during a physics step."""


class EngineType(str, Enum):
    """Supported physics engine families."""

    ODE = "ode"
    MUJOCO = "mujoco"
    CUSTOM = "custom"


class EngineCommandType(str, Enum):
    """Command contract consumed by a physics engine."""

    CONTROL_COMMAND4 = "control_command4"
    ACTUATOR_COMMAND4 = "actuator_command4"


class EngineStepStatus(str, Enum):
    """Status returned by one physics step."""

    OK = "ok"
    WARNING = "warning"
    FAILED = "failed"


@dataclass(frozen=True)
class EngineMetadata:
    """Static metadata describing one physics engine instance.

    Attributes
    ----------
    engine_type:
        Engine family such as ODE or MuJoCo.
    name:
        Human-readable engine implementation name.
    command_type:
        Primary command contract accepted by ``step``.
    native_dt:
        Native integration timestep if the engine has one. For pure ODE wrappers
        this can be None because runtime ``dt`` is passed explicitly.
    state_dim:
        External canonical state dimension. Should remain 9.
    command_dim:
        Dimension of the primary command vector. Should be 4.
    supports_control_command:
        True if the engine can accept ControlCommand4 at its public boundary.
    supports_actuator_command:
        True if the engine can accept ActuatorCommand4 at its public boundary.
    uses_quaternion_internal:
        True for engines such as MuJoCo that may use quaternion orientation
        internally and expose Euler ZYX only through an adapter.
    deterministic:
        True when repeated runs with the same initial state, command sequence,
        and timestep should produce the same State9 sequence.
    description:
        Optional human-readable description.
    extra:
        Optional engine-specific static metadata. Keep this out of core runtime
        logic.
    """

    engine_type: EngineType
    name: str
    command_type: EngineCommandType
    native_dt: float | None = None
    state_dim: int = STATE_DIM
    command_dim: int = CONTROL_DIM
    supports_control_command: bool = True
    supports_actuator_command: bool = False
    uses_quaternion_internal: bool = False
    deterministic: bool = True
    description: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate metadata consistency."""
        if not self.name.strip():
            raise EngineConfigurationError("EngineMetadata.name must be non-empty.")
        if self.state_dim != STATE_DIM:
            raise EngineConfigurationError(
                f"EngineMetadata.state_dim must be {STATE_DIM}, got {self.state_dim}."
            )
        expected_dim = command_dim_for_type(self.command_type)
        if self.command_dim != expected_dim:
            raise EngineConfigurationError(
                "EngineMetadata.command_dim is inconsistent with command_type: "
                f"expected {expected_dim}, got {self.command_dim}."
            )
        if self.native_dt is not None and self.native_dt <= 0.0:
            raise EngineConfigurationError("EngineMetadata.native_dt must be > 0.")
        if (
            self.command_type is EngineCommandType.CONTROL_COMMAND4
            and not self.supports_control_command
        ):
            raise EngineConfigurationError(
                "EngineMetadata.command_type is CONTROL_COMMAND4 but "
                "supports_control_command is False."
            )
        if (
            self.command_type is EngineCommandType.ACTUATOR_COMMAND4
            and not self.supports_actuator_command
        ):
            raise EngineConfigurationError(
                "EngineMetadata.command_type is ACTUATOR_COMMAND4 but "
                "supports_actuator_command is False."
            )


@dataclass(frozen=True)
class StepResult:
    """Result returned by ``PhysicsEngine.step``.

    The state is always the external canonical State9 after the step.
    """

    state: FloatArray
    time: float
    dt: float
    step_index: int
    status: EngineStepStatus = EngineStepStatus.OK
    message: str | None = None
    command_type: EngineCommandType | None = None
    applied_command: FloatArray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result contract."""
        as_state9(self.state)
        if self.time < 0.0 or not np.isfinite(self.time):
            raise EngineStateError("StepResult.time must be finite and >= 0.")
        if self.dt <= 0.0 or not np.isfinite(self.dt):
            raise EngineStateError("StepResult.dt must be finite and > 0.")
        if self.step_index < 0:
            raise EngineStateError("StepResult.step_index must be >= 0.")
        if self.applied_command is not None and self.command_type is not None:
            validate_engine_command(self.applied_command, self.command_type)


@runtime_checkable
class PhysicsEngineProtocol(Protocol):
    """Structural protocol for physics engines."""

    def reset(self, initial_state: FloatArray) -> None:
        """Reset engine to canonical State9."""

    def step(self, command: FloatArray, dt: float) -> StepResult:
        """Advance physics by dt seconds using the provided command."""

    def get_state(self) -> FloatArray:
        """Return current canonical State9."""

    def get_time(self) -> float:
        """Return current simulation time in seconds."""

    def get_step_index(self) -> int:
        """Return number of completed physics steps."""

    def get_metadata(self) -> EngineMetadata:
        """Return static engine metadata."""

    def close(self) -> None:
        """Release engine resources."""


class PhysicsEngine(ABC):
    """Abstract base class for canonical physics engines."""

    @abstractmethod
    def reset(self, initial_state: FloatArray) -> None:
        """Reset engine to canonical State9."""

    @abstractmethod
    def step(self, command: FloatArray, dt: float) -> StepResult:
        """Advance physics by ``dt`` seconds."""

    @abstractmethod
    def get_state(self) -> FloatArray:
        """Return current canonical State9."""

    @abstractmethod
    def get_time(self) -> float:
        """Return current simulation time in seconds."""

    @abstractmethod
    def get_step_index(self) -> int:
        """Return number of completed physics steps."""

    @abstractmethod
    def get_metadata(self) -> EngineMetadata:
        """Return static engine metadata."""

    def close(self) -> None:
        """Release engine resources. Stateless engines may keep the no-op default."""


def command_dim_for_type(command_type: EngineCommandType) -> int:
    """Return expected vector dimension for a command type."""
    if command_type is EngineCommandType.CONTROL_COMMAND4:
        return CONTROL_DIM
    if command_type is EngineCommandType.ACTUATOR_COMMAND4:
        return ACTUATOR_DIM
    raise EngineConfigurationError(f"Unsupported command type: {command_type!r}.")


def validate_engine_command(
    command: FloatArray,
    command_type: EngineCommandType,
) -> FloatArray:
    """Validate command according to engine command type."""
    if command_type is EngineCommandType.CONTROL_COMMAND4:
        return as_control_command4(command)
    if command_type is EngineCommandType.ACTUATOR_COMMAND4:
        return as_actuator_command4(command)
    raise EngineConfigurationError(f"Unsupported command type: {command_type!r}.")


def validate_step_dt(dt: float) -> float:
    """Validate and return a physics step duration."""
    if isinstance(dt, bool):
        raise EngineStepError("dt must be a finite positive number, got bool.")
    try:
        dt_value = float(dt)
    except (TypeError, ValueError) as exc:
        raise EngineStepError("dt must be a finite positive number.") from exc
    if not np.isfinite(dt_value) or dt_value <= 0.0:
        raise EngineStepError("dt must be finite and > 0.")
    return dt_value


def make_step_result(
    *,
    state: FloatArray,
    time: float,
    dt: float,
    step_index: int,
    command_type: EngineCommandType | None = None,
    applied_command: FloatArray | None = None,
    status: EngineStepStatus = EngineStepStatus.OK,
    message: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> StepResult:
    """Construct and validate a ``StepResult``."""
    return StepResult(
        state=as_state9(state),
        time=float(time),
        dt=validate_step_dt(dt),
        step_index=step_index,
        status=status,
        message=message,
        command_type=command_type,
        applied_command=(
            validate_engine_command(applied_command, command_type)
            if applied_command is not None and command_type is not None
            else None
        ),
        diagnostics={} if diagnostics is None else dict(diagnostics),
    )


DEFAULT_ODE_METADATA: Final[EngineMetadata] = EngineMetadata(
    engine_type=EngineType.ODE,
    name="ODEPhysicsEngine",
    command_type=EngineCommandType.CONTROL_COMMAND4,
    native_dt=None,
    supports_control_command=True,
    supports_actuator_command=False,
    uses_quaternion_internal=False,
    deterministic=True,
    description="Canonical reduced-order ODE physics engine.",
)

DEFAULT_MUJOCO_METADATA: Final[EngineMetadata] = EngineMetadata(
    engine_type=EngineType.MUJOCO,
    name="MuJoCoPhysicsEngine",
    command_type=EngineCommandType.ACTUATOR_COMMAND4,
    native_dt=None,
    supports_control_command=False,
    supports_actuator_command=True,
    uses_quaternion_internal=True,
    deterministic=True,
    description=(
        "MuJoCo physics engine exposing canonical State9 through adapters "
        "and consuming ActuatorCommand4 at the engine boundary."
    ),
)


__all__ = [
    "DEFAULT_MUJOCO_METADATA",
    "DEFAULT_ODE_METADATA",
    "EngineCommandType",
    "EngineConfigurationError",
    "EngineError",
    "EngineMetadata",
    "EngineStateError",
    "EngineStepError",
    "EngineStepStatus",
    "EngineType",
    "PhysicsEngine",
    "PhysicsEngineProtocol",
    "StepResult",
    "command_dim_for_type",
    "make_step_result",
    "validate_engine_command",
    "validate_step_dt",
]
