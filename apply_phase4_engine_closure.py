#!/usr/bin/env python3
"""Apply Phase 4 engine-interface closure refactor for quadrotor_mpc.

Run from repository root:
    python /path/to/apply_phase4_engine_closure.py

This script writes/replaces the Phase 4 engine files and adds MuJoCo
adapter/interface tests. It does not require MuJoCo at patch time.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

ROOT = Path.cwd()
CODE = ROOT / "2.Code"


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"wrote {target}")


BASE_PY = r'''
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
'''


MUJOCO_STATE_ADAPTER_PY = r'''
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
'''


MUJOCO_ACTUATOR_ADAPTER_PY = r'''
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
'''


ADAPTERS_INIT_PY = r'''
"""Adapters that keep concrete physics backend internals behind canonical contracts."""

from simulation.engines.adapters.mujoco_actuator_adapter import write_actuator_command4_to_ctrl
from simulation.engines.adapters.mujoco_state_adapter import (
    euler_zyx_to_quat_wxyz,
    quat_wxyz_to_euler_zyx,
    read_state9_from_mujoco_freejoint,
    write_state9_to_mujoco_freejoint,
)

__all__ = [
    "euler_zyx_to_quat_wxyz",
    "quat_wxyz_to_euler_zyx",
    "read_state9_from_mujoco_freejoint",
    "write_actuator_command4_to_ctrl",
    "write_state9_to_mujoco_freejoint",
]
'''


MUJOCO_ENGINE_PY = r'''
"""MuJoCo physics-engine implementation behind the canonical PhysicsEngine API.

Phase 4 scope is intentionally narrow:
- reset from State9
- step with ActuatorCommand4
- expose State9 and StepResult
- hide MuJoCo qpos/qvel/quaternion/MjModel/MjData from runtime/controller

Full MuJoCo runtime integration, mixer orchestration, logging, scenario runner,
and viewer support are intentionally left to later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ccmpc.types import FloatArray, as_state9
from simulation.engines.adapters.mujoco_actuator_adapter import (
    write_actuator_command4_to_ctrl,
)
from simulation.engines.adapters.mujoco_state_adapter import (
    read_state9_from_mujoco_freejoint,
    write_state9_to_mujoco_freejoint,
)
from simulation.engines.base import (
    DEFAULT_MUJOCO_METADATA,
    EngineCommandType,
    EngineConfigurationError,
    EngineMetadata,
    EngineStateError,
    EngineStepError,
    EngineType,
    PhysicsEngine,
    StepResult,
    make_step_result,
    validate_engine_command,
    validate_step_dt,
)

try:  # Optional dependency. Importing this module must not require MuJoCo.
    import mujoco
except ImportError:  # pragma: no cover - depends on local optional dependency.
    mujoco = None  # type: ignore[assignment]


PathLike = str | Path


@dataclass(frozen=True)
class MuJoCoEngineConfig:
    """Configuration for ``MuJoCoPhysicsEngine``.

    Attributes
    ----------
    xml_path:
        Path to MuJoCo XML model.
    free_joint_name:
        Name of the root free joint. If omitted, qpos/qvel addresses default to 0.
    actuator_start_index:
        First index in ``data.ctrl`` used for canonical [T1, T2, T3, T4].
    copy_state:
        If True, protect internal buffers by copying arrays at the public boundary.
    """

    xml_path: PathLike
    free_joint_name: str | None = None
    actuator_start_index: int = 0
    copy_state: bool = True


class MuJoCoPhysicsEngine(PhysicsEngine):
    """MuJoCo backend behind the canonical PhysicsEngine interface."""

    def __init__(
        self,
        *,
        config: MuJoCoEngineConfig,
        initial_state: FloatArray | None = None,
        metadata: EngineMetadata = DEFAULT_MUJOCO_METADATA,
    ) -> None:
        if mujoco is None:
            raise EngineConfigurationError(
                "MuJoCo is not installed. Install optional dependency `mujoco` "
                "before constructing MuJoCoPhysicsEngine."
            )

        self._validate_metadata(metadata)
        self._validate_config(config)

        xml_path = Path(config.xml_path)
        if not xml_path.exists():
            raise EngineConfigurationError(f"MuJoCo XML path does not exist: {xml_path}")

        self._config = config
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self._model)

        if self._model.nu < config.actuator_start_index + 4:
            raise EngineConfigurationError(
                f"MuJoCo model has nu={self._model.nu}, but ActuatorCommand4 requires "
                f"4 actuators from index {config.actuator_start_index}."
            )

        native_dt = float(self._model.opt.timestep)
        if native_dt <= 0.0 or not np.isfinite(native_dt):
            raise EngineConfigurationError("MuJoCo model timestep must be finite and > 0.")

        self._metadata = replace(metadata, native_dt=native_dt)
        self._qpos_adr, self._qvel_adr = self._resolve_freejoint_addresses()
        self._step_index = 0
        self._closed = False

        if initial_state is not None:
            self.reset(initial_state)

    def reset(self, initial_state: FloatArray) -> None:
        """Reset MuJoCo state from canonical State9."""
        self._ensure_open()
        state = as_state9(initial_state)

        self._data.qpos[:] = self._model.qpos0
        self._data.qvel[:] = 0.0
        self._data.ctrl[:] = 0.0
        self._data.time = 0.0

        write_state9_to_mujoco_freejoint(
            state=state,
            qpos=self._data.qpos,
            qvel=self._data.qvel,
            qpos_adr=self._qpos_adr,
            qvel_adr=self._qvel_adr,
        )
        self._step_index = 0
        mujoco.mj_forward(self._model, self._data)

    def step(self, command: FloatArray, dt: float) -> StepResult:
        """Advance MuJoCo by ``dt`` using ActuatorCommand4."""
        self._ensure_open()

        dt_value = validate_step_dt(dt)
        command_array = validate_engine_command(command, self._metadata.command_type)
        native_dt = float(self._model.opt.timestep)
        substeps_float = dt_value / native_dt
        substeps = int(round(substeps_float))

        if substeps <= 0 or not np.isclose(substeps_float, substeps, rtol=1e-9, atol=1e-12):
            raise EngineStepError(
                f"Runtime dt={dt_value} must be an integer multiple of MuJoCo "
                f"native timestep={native_dt}."
            )

        applied = write_actuator_command4_to_ctrl(
            command=command_array,
            ctrl=self._data.ctrl,
            start_index=self._config.actuator_start_index,
        )

        try:
            ctrl_slice = slice(
                self._config.actuator_start_index,
                self._config.actuator_start_index + 4,
            )
            for _ in range(substeps):
                self._data.ctrl[ctrl_slice] = applied
                mujoco.mj_step(self._model, self._data)
        except Exception as exc:  # pragma: no cover - MuJoCo-specific failures.
            raise EngineStepError("MuJoCo step failed.") from exc

        self._step_index += 1
        state = self.get_state()

        return make_step_result(
            state=state,
            time=float(self._data.time),
            dt=dt_value,
            step_index=self._step_index,
            command_type=self._metadata.command_type,
            applied_command=applied,
            diagnostics={
                "engine": self._metadata.engine_type.value,
                "native_dt": native_dt,
                "substeps": substeps,
                "mujoco_time": float(self._data.time),
            },
        )

    def get_state(self) -> FloatArray:
        """Return current canonical State9."""
        self._ensure_open()
        state = read_state9_from_mujoco_freejoint(
            qpos=self._data.qpos,
            qvel=self._data.qvel,
            qpos_adr=self._qpos_adr,
            qvel_adr=self._qvel_adr,
        )
        return state.copy() if self._config.copy_state else state

    def get_time(self) -> float:
        """Return current MuJoCo simulation time."""
        self._ensure_open()
        return float(self._data.time)

    def get_step_index(self) -> int:
        """Return number of completed public engine steps."""
        self._ensure_open()
        return int(self._step_index)

    def get_metadata(self) -> EngineMetadata:
        """Return static MuJoCo engine metadata."""
        return self._metadata

    def close(self) -> None:
        """Mark the engine as closed."""
        self._closed = True

    @property
    def is_closed(self) -> bool:
        """Return True if close has been called."""
        return self._closed

    def _resolve_freejoint_addresses(self) -> tuple[int, int]:
        if self._config.free_joint_name is None:
            return 0, 0

        joint_id = mujoco.mj_name2id(
            self._model,
            mujoco.mjtObj.mjOBJ_JOINT,
            self._config.free_joint_name,
        )
        if joint_id < 0:
            raise EngineConfigurationError(
                f"Free joint {self._config.free_joint_name!r} not found in MuJoCo model."
            )

        return (
            int(self._model.jnt_qposadr[joint_id]),
            int(self._model.jnt_dofadr[joint_id]),
        )

    @staticmethod
    def _validate_config(config: MuJoCoEngineConfig) -> None:
        if config.actuator_start_index < 0:
            raise EngineConfigurationError("actuator_start_index must be non-negative.")

    @staticmethod
    def _validate_metadata(metadata: EngineMetadata) -> None:
        if metadata.engine_type is not EngineType.MUJOCO:
            raise EngineConfigurationError(
                "MuJoCoPhysicsEngine metadata.engine_type must be EngineType.MUJOCO."
            )
        if metadata.command_type is not EngineCommandType.ACTUATOR_COMMAND4:
            raise EngineConfigurationError(
                "MuJoCoPhysicsEngine must consume ActuatorCommand4. "
                "Use mixer before calling MuJoCoPhysicsEngine.step()."
            )
        if metadata.supports_control_command:
            raise EngineConfigurationError(
                "MuJoCoPhysicsEngine must not accept ControlCommand4 at its public boundary."
            )
        if not metadata.supports_actuator_command:
            raise EngineConfigurationError(
                "MuJoCoPhysicsEngine metadata must support ActuatorCommand4."
            )
        if not metadata.uses_quaternion_internal:
            raise EngineConfigurationError(
                "MuJoCoPhysicsEngine metadata must declare uses_quaternion_internal=True."
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise EngineStateError("MuJoCoPhysicsEngine is closed.")


__all__ = ["MuJoCoEngineConfig", "MuJoCoPhysicsEngine"]
'''


FACTORY_PY = r'''
"""Physics engine factory.

This module centralizes engine construction so runtime code does not need to
know how each concrete engine is initialized.

Current Phase 4 scope:
- ODE engine: supported and consumes ControlCommand4.
- MuJoCo engine: constructed explicitly with MuJoCoEngineFactoryConfig and
  consumes ActuatorCommand4.

The factory does not run controllers, mixers, loggers, scenario runners, or
visualizers. Those belong to later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ccmpc.dynamics import QuadrotorDynamics
from ccmpc.types import FloatArray, as_state9
from simulation.engines.base import (
    DEFAULT_MUJOCO_METADATA,
    DEFAULT_ODE_METADATA,
    EngineConfigurationError,
    EngineMetadata,
    EngineType,
    PhysicsEngine,
)
from simulation.engines.mujoco_engine import MuJoCoEngineConfig, MuJoCoPhysicsEngine
from simulation.engines.ode_engine import DiscreteDynamicsProtocol, ODEPhysicsEngine

PathLike = str | Path


@dataclass(frozen=True)
class ODEEngineFactoryConfig:
    """Configuration used by the factory to build an ODE engine."""

    dynamics_config: PathLike | dict[str, Any] | None = None
    metadata_name: str | None = None
    native_dt: float | None = None


@dataclass(frozen=True)
class MuJoCoEngineFactoryConfig:
    """Configuration used by the factory to build a MuJoCo engine."""

    xml_path: PathLike
    free_joint_name: str | None = None
    actuator_start_index: int = 0


def create_physics_engine(
    engine_type: EngineType | str,
    *,
    initial_state: FloatArray,
    dynamics: DiscreteDynamicsProtocol | None = None,
    ode_config: ODEEngineFactoryConfig | None = None,
    mujoco_config: MuJoCoEngineFactoryConfig | None = None,
    metadata: EngineMetadata | None = None,
) -> PhysicsEngine:
    """Create a physics engine from canonical factory arguments."""
    parsed_engine_type = parse_engine_type(engine_type)
    state = as_state9(initial_state)

    if parsed_engine_type is EngineType.ODE:
        return create_ode_engine(
            initial_state=state,
            dynamics=dynamics,
            config=ode_config,
            metadata=metadata,
        )

    if parsed_engine_type is EngineType.MUJOCO:
        if mujoco_config is None:
            raise EngineConfigurationError(
                "mujoco_config is required for MuJoCoPhysicsEngine. "
                "Provide xml_path, free_joint_name, and actuator_start_index explicitly."
            )
        return create_mujoco_engine(
            initial_state=state,
            config=mujoco_config,
            metadata=metadata,
        )

    if parsed_engine_type is EngineType.CUSTOM:
        raise EngineConfigurationError(
            "CUSTOM engine factory is not implemented. "
            "Instantiate custom engines directly or extend create_physics_engine()."
        )

    raise EngineConfigurationError(f"Unsupported engine type: {parsed_engine_type!r}.")


def create_ode_engine(
    *,
    initial_state: FloatArray,
    dynamics: DiscreteDynamicsProtocol | None = None,
    config: ODEEngineFactoryConfig | None = None,
    metadata: EngineMetadata | None = None,
) -> ODEPhysicsEngine:
    """Create an ``ODEPhysicsEngine``."""
    state = as_state9(initial_state)
    config = ODEEngineFactoryConfig() if config is None else config

    dynamics_obj = dynamics
    if dynamics_obj is None:
        dynamics_obj = create_quadrotor_dynamics(config.dynamics_config)

    metadata_obj = metadata if metadata is not None else DEFAULT_ODE_METADATA
    if config.metadata_name is not None or config.native_dt is not None:
        metadata_obj = replace(
            metadata_obj,
            name=config.metadata_name if config.metadata_name is not None else metadata_obj.name,
            native_dt=config.native_dt if config.native_dt is not None else metadata_obj.native_dt,
        )

    return ODEPhysicsEngine(
        dynamics=dynamics_obj,
        initial_state=state,
        metadata=metadata_obj,
    )


def create_mujoco_engine(
    *,
    initial_state: FloatArray,
    config: MuJoCoEngineFactoryConfig,
    metadata: EngineMetadata | None = None,
) -> MuJoCoPhysicsEngine:
    """Create a ``MuJoCoPhysicsEngine`` from explicit MuJoCo configuration."""
    state = as_state9(initial_state)
    metadata_obj = metadata if metadata is not None else DEFAULT_MUJOCO_METADATA
    return MuJoCoPhysicsEngine(
        config=MuJoCoEngineConfig(
            xml_path=config.xml_path,
            free_joint_name=config.free_joint_name,
            actuator_start_index=config.actuator_start_index,
        ),
        initial_state=state,
        metadata=metadata_obj,
    )


def create_quadrotor_dynamics(
    dynamics_config: PathLike | dict[str, Any] | None = None,
) -> QuadrotorDynamics:
    """Create ``QuadrotorDynamics`` for the ODE engine."""
    if dynamics_config is None:
        return QuadrotorDynamics()
    if isinstance(dynamics_config, Path):
        return QuadrotorDynamics.from_config(str(dynamics_config))
    if isinstance(dynamics_config, str):
        return QuadrotorDynamics.from_config(dynamics_config)
    if isinstance(dynamics_config, dict):
        return QuadrotorDynamics.from_config(dynamics_config)
    raise EngineConfigurationError(
        "dynamics_config must be None, str, pathlib.Path, or dict."
    )


def parse_engine_type(engine_type: EngineType | str) -> EngineType:
    """Parse engine type from enum or string."""
    if isinstance(engine_type, EngineType):
        return engine_type
    if isinstance(engine_type, str):
        normalized = engine_type.strip().lower()
        try:
            return EngineType(normalized)
        except ValueError as exc:
            raise EngineConfigurationError(
                f"Unsupported engine type string: {engine_type!r}."
            ) from exc
    raise EngineConfigurationError(
        f"engine_type must be EngineType or str, got {type(engine_type).__name__}."
    )


__all__ = [
    "MuJoCoEngineFactoryConfig",
    "ODEEngineFactoryConfig",
    "create_mujoco_engine",
    "create_ode_engine",
    "create_physics_engine",
    "create_quadrotor_dynamics",
    "parse_engine_type",
]
'''


ENGINES_INIT_PY = r'''
"""Physics engine package exports."""

from simulation.engines.base import (
    DEFAULT_MUJOCO_METADATA,
    DEFAULT_ODE_METADATA,
    EngineCommandType,
    EngineConfigurationError,
    EngineError,
    EngineMetadata,
    EngineStateError,
    EngineStepError,
    EngineStepStatus,
    EngineType,
    PhysicsEngine,
    PhysicsEngineProtocol,
    StepResult,
    command_dim_for_type,
    make_step_result,
    validate_engine_command,
    validate_step_dt,
)
from simulation.engines.factory import (
    MuJoCoEngineFactoryConfig,
    ODEEngineFactoryConfig,
    create_mujoco_engine,
    create_ode_engine,
    create_physics_engine,
    create_quadrotor_dynamics,
    parse_engine_type,
)
from simulation.engines.mujoco_engine import MuJoCoEngineConfig, MuJoCoPhysicsEngine
from simulation.engines.ode_engine import DiscreteDynamicsProtocol, ODEPhysicsEngine

__all__ = [
    "DEFAULT_MUJOCO_METADATA",
    "DEFAULT_ODE_METADATA",
    "DiscreteDynamicsProtocol",
    "EngineCommandType",
    "EngineConfigurationError",
    "EngineError",
    "EngineMetadata",
    "EngineStateError",
    "EngineStepError",
    "EngineStepStatus",
    "EngineType",
    "MuJoCoEngineConfig",
    "MuJoCoEngineFactoryConfig",
    "MuJoCoPhysicsEngine",
    "ODEEngineFactoryConfig",
    "ODEPhysicsEngine",
    "PhysicsEngine",
    "PhysicsEngineProtocol",
    "StepResult",
    "command_dim_for_type",
    "create_mujoco_engine",
    "create_ode_engine",
    "create_physics_engine",
    "create_quadrotor_dynamics",
    "make_step_result",
    "parse_engine_type",
    "validate_engine_command",
    "validate_step_dt",
]
'''


TEST_MUJOCO_METADATA = r'''
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
'''


TEST_MUJOCO_STATE_ADAPTER = r'''
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
'''


TEST_MUJOCO_ACTUATOR_ADAPTER = r'''
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
'''


TEST_MUJOCO_ENGINE_INTERFACE = r'''
"""Interface tests for MuJoCoPhysicsEngine.

These tests are skipped when optional dependency ``mujoco`` is not installed.
The adapter unit tests still run without MuJoCo.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from ccmpc.types import DataContractError
from simulation.engines.base import EngineCommandType, EngineConfigurationError, EngineType
from simulation.engines.factory import (
    MuJoCoEngineFactoryConfig,
    create_physics_engine,
)
from simulation.engines.mujoco_engine import MuJoCoEngineConfig, MuJoCoPhysicsEngine


_MINIMAL_QUAD_XML = """
<mujoco model="phase4_quad_test">
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body name="quad" pos="0 0 0">
      <freejoint name="root"/>
      <geom name="body" type="box" size="0.1 0.1 0.02" mass="1"/>
      <site name="r1" pos="0.1 0.1 0"/>
      <site name="r2" pos="-0.1 0.1 0"/>
      <site name="r3" pos="-0.1 -0.1 0"/>
      <site name="r4" pos="0.1 -0.1 0"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="m1" site="r1" gear="0 0 1 0 0 0"/>
    <motor name="m2" site="r2" gear="0 0 1 0 0 0"/>
    <motor name="m3" site="r3" gear="0 0 1 0 0 0"/>
    <motor name="m4" site="r4" gear="0 0 1 0 0 0"/>
  </actuator>
</mujoco>
"""


def write_minimal_xml(tmp_path) -> str:
    xml_path = tmp_path / "phase4_quad_test.xml"
    xml_path.write_text(_MINIMAL_QUAD_XML, encoding="utf-8")
    return str(xml_path)


def make_state() -> np.ndarray:
    return np.array(
        [0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.01, -0.02, 0.03],
        dtype=np.float64,
    )


def make_actuator_command() -> np.ndarray:
    return np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)


def test_mujoco_engine_reset_returns_state9(tmp_path) -> None:
    engine = MuJoCoPhysicsEngine(
        config=MuJoCoEngineConfig(
            xml_path=write_minimal_xml(tmp_path),
            free_joint_name="root",
        ),
        initial_state=make_state(),
    )

    state = engine.get_state()

    assert state.shape == (9,)
    np.testing.assert_allclose(state, make_state(), atol=1e-12)
    assert engine.get_metadata().engine_type is EngineType.MUJOCO
    assert engine.get_metadata().command_type is EngineCommandType.ACTUATOR_COMMAND4
    assert engine.get_metadata().native_dt == pytest.approx(0.01)


def test_mujoco_engine_step_returns_step_result(tmp_path) -> None:
    engine = MuJoCoPhysicsEngine(
        config=MuJoCoEngineConfig(
            xml_path=write_minimal_xml(tmp_path),
            free_joint_name="root",
        ),
        initial_state=make_state(),
    )

    result = engine.step(make_actuator_command(), 0.02)

    assert result.state.shape == (9,)
    assert result.command_type is EngineCommandType.ACTUATOR_COMMAND4
    assert result.applied_command is not None
    np.testing.assert_allclose(result.applied_command, make_actuator_command())
    assert result.dt == pytest.approx(0.02)
    assert result.step_index == 1
    assert result.diagnostics["engine"] == "mujoco"
    assert result.diagnostics["substeps"] == 2


def test_mujoco_engine_rejects_control_command_without_mixer(tmp_path) -> None:
    engine = MuJoCoPhysicsEngine(
        config=MuJoCoEngineConfig(
            xml_path=write_minimal_xml(tmp_path),
            free_joint_name="root",
        ),
        initial_state=make_state(),
    )

    with pytest.raises(DataContractError, match="non-negative"):
        engine.step(np.array([0.1, -0.1, 0.2, 0.3], dtype=np.float64), 0.01)


def test_mujoco_engine_rejects_non_multiple_dt(tmp_path) -> None:
    engine = MuJoCoPhysicsEngine(
        config=MuJoCoEngineConfig(
            xml_path=write_minimal_xml(tmp_path),
            free_joint_name="root",
        ),
        initial_state=make_state(),
    )

    with pytest.raises(Exception, match="integer multiple"):
        engine.step(make_actuator_command(), 0.015)


def test_mujoco_engine_does_not_expose_public_model_data_attrs(tmp_path) -> None:
    engine = MuJoCoPhysicsEngine(
        config=MuJoCoEngineConfig(
            xml_path=write_minimal_xml(tmp_path),
            free_joint_name="root",
        ),
        initial_state=make_state(),
    )

    assert not hasattr(engine, "model")
    assert not hasattr(engine, "data")
    assert not hasattr(engine, "qpos")
    assert not hasattr(engine, "qvel")


def test_mujoco_factory_requires_config() -> None:
    with pytest.raises(EngineConfigurationError, match="mujoco_config"):
        create_physics_engine(EngineType.MUJOCO, initial_state=make_state())


def test_mujoco_factory_creates_engine(tmp_path) -> None:
    engine = create_physics_engine(
        EngineType.MUJOCO,
        initial_state=make_state(),
        mujoco_config=MuJoCoEngineFactoryConfig(
            xml_path=write_minimal_xml(tmp_path),
            free_joint_name="root",
        ),
    )

    assert isinstance(engine, MuJoCoPhysicsEngine)
    assert engine.get_metadata().engine_type is EngineType.MUJOCO
'''


PHASE4_DOC = r'''
# Phase 4 Engine Interface Closure

## Scope

This closure refactor finalizes the public physics-engine boundary before Phase 6 logging.

## Canonical contracts

| Engine | Public input | Public output | Internal details |
|---|---|---|---|
| ODEPhysicsEngine | `ControlCommand4 = [phi_c, theta_c, vz_c, psi_dot_c]` | `State9` | reduced-order ODE dynamics |
| MuJoCoPhysicsEngine | `ActuatorCommand4 = [T1, T2, T3, T4]` | `State9` | MuJoCo `MjModel`, `MjData`, `qpos`, `qvel`, quaternion |

## Boundary decisions

- MuJoCo does **not** accept `ControlCommand4` at the engine boundary.
- `ControlCommand4 -> ActuatorCommand4` mixing stays outside `MuJoCoPhysicsEngine`.
- MuJoCo `qpos/qvel` and quaternion state are private backend details.
- Runtime/controller/logging should consume only `StepResult`, `EngineMetadata`, and canonical arrays.

## Files updated/added

- `2.Code/simulation/engines/base.py`
- `2.Code/simulation/engines/factory.py`
- `2.Code/simulation/engines/__init__.py`
- `2.Code/simulation/engines/mujoco_engine.py`
- `2.Code/simulation/engines/adapters/mujoco_state_adapter.py`
- `2.Code/simulation/engines/adapters/mujoco_actuator_adapter.py`
- `2.Code/tests/unit/test_mujoco_metadata_policy.py`
- `2.Code/tests/unit/test_mujoco_state_adapter.py`
- `2.Code/tests/unit/test_mujoco_actuator_adapter.py`
- `2.Code/tests/interface/test_mujoco_engine_interface.py`

## Validation commands

```bash
cd 2.Code
pytest tests/unit/test_engine_base.py -v
pytest tests/unit/test_engine_factory.py -v
pytest tests/unit/test_ode_engine.py -v
pytest tests/unit/test_mujoco_metadata_policy.py -v
pytest tests/unit/test_mujoco_state_adapter.py -v
pytest tests/unit/test_mujoco_actuator_adapter.py -v
pytest tests/interface/test_mujoco_engine_interface.py -v
pytest tests/unit -v
pytest tests/interface -v
```

If MuJoCo is not installed, `test_mujoco_engine_interface.py` should skip.
'''


def patch_existing_tests() -> None:
    """Patch existing tests whose expectations changed with the Phase 4 policy."""
    engine_base_path = ROOT / "2.Code/tests/unit/test_engine_base.py"
    if engine_base_path.exists():
        text = engine_base_path.read_text(encoding="utf-8")
        pattern = re.compile(
            r"def test_engine_metadata_valid_mujoco\(\) -> None:.*?"
            r"(?=def test_engine_metadata_reject_wrong_state_dim\(\) -> None:)",
            re.DOTALL,
        )
        replacement = '''def test_engine_metadata_valid_mujoco() -> None:\n    """Default MuJoCo metadata should expose State9 and consume ActuatorCommand4."""\n    metadata = DEFAULT_MUJOCO_METADATA\n\n    assert metadata.engine_type is EngineType.MUJOCO\n    assert metadata.name == "MuJoCoPhysicsEngine"\n    assert metadata.command_type is EngineCommandType.ACTUATOR_COMMAND4\n    assert metadata.state_dim == 9\n    assert metadata.command_dim == 4\n    assert metadata.supports_control_command is False\n    assert metadata.supports_actuator_command is True\n    assert metadata.uses_quaternion_internal is True\n\n'''
        new_text, count = pattern.subn(replacement, text, count=1)
        if count == 0:
            print("warning: did not patch test_engine_base.py MuJoCo metadata test")
        else:
            engine_base_path.write_text(new_text, encoding="utf-8")
            print(f"patched {engine_base_path}")

    factory_test_path = ROOT / "2.Code/tests/unit/test_engine_factory.py"
    if factory_test_path.exists():
        text = factory_test_path.read_text(encoding="utf-8")
        if "MuJoCoEngineFactoryConfig" not in text:
            text = text.replace(
                "from simulation.engines.factory import (",
                "from simulation.engines.factory import (\n    MuJoCoEngineFactoryConfig,",
                1,
            )
        pattern = re.compile(
            r"def test_create_physics_engine_rejects_mujoco_for_now\(\) -> None:.*?"
            r"(?=def test_create_physics_engine_rejects_custom_for_now\(\) -> None:)",
            re.DOTALL,
        )
        replacement = '''def test_create_physics_engine_mujoco_requires_config() -> None:\n    """MuJoCo factory path requires explicit XML/config instead of implicit defaults."""\n    with pytest.raises(EngineConfigurationError, match="mujoco_config"):\n        create_physics_engine(\n            EngineType.MUJOCO,\n            initial_state=make_state(),\n        )\n\n'''
        new_text, count = pattern.subn(replacement, text, count=1)
        if count == 0:
            print("warning: did not patch test_engine_factory.py MuJoCo rejection test")
        else:
            factory_test_path.write_text(new_text, encoding="utf-8")
            print(f"patched {factory_test_path}")


def append_tracker() -> None:
    tracker = ROOT / "docs/refactor/IMPLEMENTATION_TRACKER.md"
    tracker.parent.mkdir(parents=True, exist_ok=True)
    entry = f"""

## {date.today().isoformat()} — Phase 4 Engine Interface Closure

- **Status:** Implemented, pending local validation.
- **Purpose:** Close Phase 4 before Phase 6 by finalizing the ODE/MuJoCo physics-engine boundary.
- **Design documents checked:** `1.Docs/Simulation_design/11_refactor_plan.md`.
- **Theory documents checked:** Not math-heavy; no dynamics, CC-MPC, covariance, or solver formulas changed.
- **Files changed:**
  - `2.Code/simulation/engines/base.py`
  - `2.Code/simulation/engines/factory.py`
  - `2.Code/simulation/engines/__init__.py`
  - `2.Code/simulation/engines/mujoco_engine.py`
  - `2.Code/simulation/engines/adapters/__init__.py`
  - `2.Code/simulation/engines/adapters/mujoco_state_adapter.py`
  - `2.Code/simulation/engines/adapters/mujoco_actuator_adapter.py`
  - `2.Code/tests/unit/test_mujoco_metadata_policy.py`
  - `2.Code/tests/unit/test_mujoco_state_adapter.py`
  - `2.Code/tests/unit/test_mujoco_actuator_adapter.py`
  - `2.Code/tests/interface/test_mujoco_engine_interface.py`
  - `1.Docs/Simulation_design/PHASE4_ENGINE_CLOSURE.md`
- **Key interfaces/data contracts:**
  - ODE input: `ControlCommand4`; ODE output: `State9`.
  - MuJoCo input: `ActuatorCommand4`; MuJoCo output: `State9`.
  - MuJoCo `qpos/qvel/quaternion/MjModel/MjData` remain backend-private.
- **Tests added/updated:** MuJoCo metadata policy, state adapter, actuator adapter, optional MuJoCo interface tests.
- **Validation command:**
  ```bash
  cd 2.Code
  pytest tests/unit/test_engine_base.py tests/unit/test_engine_factory.py tests/unit/test_ode_engine.py -v
  pytest tests/unit/test_mujoco_metadata_policy.py tests/unit/test_mujoco_state_adapter.py tests/unit/test_mujoco_actuator_adapter.py -v
  pytest tests/interface/test_mujoco_engine_interface.py -v
  ```
- **Known limitations:** Full MuJoCo runtime integration, mixer orchestration, scenario runner, logging, and viewer support are deferred to later phases.
"""
    if tracker.exists():
        tracker.write_text(tracker.read_text(encoding="utf-8") + entry, encoding="utf-8")
    else:
        tracker.write_text("# Implementation Tracker\n" + entry, encoding="utf-8")
    print(f"updated {tracker}")


def main() -> None:
    if not CODE.exists():
        raise SystemExit(
            "Could not find 2.Code/. Run this script from the quadrotor_mpc repository root."
        )

    write("2.Code/simulation/engines/base.py", BASE_PY)
    write("2.Code/simulation/engines/adapters/__init__.py", ADAPTERS_INIT_PY)
    write("2.Code/simulation/engines/adapters/mujoco_state_adapter.py", MUJOCO_STATE_ADAPTER_PY)
    write("2.Code/simulation/engines/adapters/mujoco_actuator_adapter.py", MUJOCO_ACTUATOR_ADAPTER_PY)
    write("2.Code/simulation/engines/mujoco_engine.py", MUJOCO_ENGINE_PY)
    write("2.Code/simulation/engines/factory.py", FACTORY_PY)
    write("2.Code/simulation/engines/__init__.py", ENGINES_INIT_PY)

    write("2.Code/tests/unit/test_mujoco_metadata_policy.py", TEST_MUJOCO_METADATA)
    write("2.Code/tests/unit/test_mujoco_state_adapter.py", TEST_MUJOCO_STATE_ADAPTER)
    write("2.Code/tests/unit/test_mujoco_actuator_adapter.py", TEST_MUJOCO_ACTUATOR_ADAPTER)
    write("2.Code/tests/interface/test_mujoco_engine_interface.py", TEST_MUJOCO_ENGINE_INTERFACE)

    write("1.Docs/Simulation_design/PHASE4_ENGINE_CLOSURE.md", PHASE4_DOC)
    patch_existing_tests()
    append_tracker()

    print("\nPhase 4 engine closure patch applied.")
    print("Next validation:")
    print("  cd 2.Code")
    print("  pytest tests/unit/test_engine_base.py tests/unit/test_engine_factory.py tests/unit/test_ode_engine.py -v")
    print("  pytest tests/unit/test_mujoco_metadata_policy.py tests/unit/test_mujoco_state_adapter.py tests/unit/test_mujoco_actuator_adapter.py -v")
    print("  pytest tests/interface/test_mujoco_engine_interface.py -v")


if __name__ == "__main__":
    main()
