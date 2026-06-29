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
