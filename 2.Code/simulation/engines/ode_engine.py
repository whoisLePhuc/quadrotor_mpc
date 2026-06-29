"""ODE physics-engine implementation.

This module wraps the reduced-order quadrotor dynamics model behind the common
``PhysicsEngine`` interface.

The engine is intentionally small:
    - it owns the current State9
    - it owns simulation time and step index
    - it validates ControlCommand4 and dt
    - it calls ``dynamics.discrete(state, command, dt)``
    - it returns ``StepResult``

It does not:
    - solve MPC
    - parse YAML
    - log CSV files
    - render visualization
    - convert MuJoCo qpos/qvel
    - mix rotor thrusts
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol, runtime_checkable

import numpy as np

from ccmpc.types import FloatArray, as_state9

from simulation.engines.base import (
    DEFAULT_ODE_METADATA,
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


@runtime_checkable
class DiscreteDynamicsProtocol(Protocol):
    """Structural protocol required by ``ODEPhysicsEngine``.

    Existing dynamics implementations only need to provide this method.
    """

    def discrete(self, state: FloatArray, command: FloatArray, dt: float) -> FloatArray:
        """Return next canonical State9 after applying command for dt seconds."""


class ODEPhysicsEngine(PhysicsEngine):
    """Reduced-order ODE physics engine.

    Parameters
    ----------
    dynamics:
        Object implementing ``discrete(state, command, dt)``.
    initial_state:
        Optional initial canonical State9.  If omitted, caller must call
        ``reset`` before ``step`` or ``get_state``.
    metadata:
        Optional metadata override.  The engine still requires
        ``engine_type=EngineType.ODE`` and ``command_type=CONTROL_COMMAND4``.
    initial_time:
        Initial simulation time in seconds.
    copy_state:
        If True, protect internal state by copying arrays at the interface.
    """

    def __init__(
        self,
        *,
        dynamics: DiscreteDynamicsProtocol,
        initial_state: FloatArray | None = None,
        metadata: EngineMetadata = DEFAULT_ODE_METADATA,
        initial_time: float = 0.0,
        copy_state: bool = True,
    ) -> None:
        self._validate_dynamics(dynamics)
        self._validate_metadata(metadata)
        self._validate_initial_time(initial_time)

        self._dynamics = dynamics
        self._metadata = metadata
        self._copy_state = copy_state

        self._state: FloatArray | None = None
        self._time = float(initial_time)
        self._step_index = 0
        self._closed = False

        if initial_state is not None:
            self.reset(initial_state, time=initial_time)

    @classmethod
    def from_dynamics(
        cls,
        dynamics: DiscreteDynamicsProtocol,
        *,
        initial_state: FloatArray,
        name: str | None = None,
        native_dt: float | None = None,
    ) -> "ODEPhysicsEngine":
        """Convenience constructor for existing ``QuadrotorDynamics`` objects."""
        metadata = DEFAULT_ODE_METADATA

        if name is not None or native_dt is not None:
            metadata = replace(
                metadata,
                name=name if name is not None else metadata.name,
                native_dt=native_dt,
            )

        return cls(
            dynamics=dynamics,
            initial_state=initial_state,
            metadata=metadata,
        )

    def reset(self, initial_state: FloatArray, time: float = 0.0) -> None:
        """Reset engine to canonical State9 and reset step counter."""
        self._ensure_open()
        self._validate_initial_time(time)

        state = as_state9(initial_state)
        self._state = state.copy() if self._copy_state else state
        self._time = float(time)
        self._step_index = 0

    def step(self, command: FloatArray, dt: float) -> StepResult:
        """Advance reduced-order ODE dynamics by one runtime step."""
        self._ensure_open()
        self._ensure_initialized()

        dt_value = validate_step_dt(dt)
        command_array = validate_engine_command(command, self._metadata.command_type)

        # mypy cannot infer _state is not None after _ensure_initialized.
        current_state = self._state
        if current_state is None:  # pragma: no cover - defensive guard.
            raise EngineStateError("ODEPhysicsEngine has not been reset.")

        try:
            next_state_raw = self._dynamics.discrete(
                current_state.copy() if self._copy_state else current_state,
                command_array.copy() if self._copy_state else command_array,
                dt_value,
            )
        except Exception as exc:
            raise EngineStepError("ODE dynamics discrete step failed.") from exc

        next_state = as_state9(next_state_raw)

        self._state = next_state.copy() if self._copy_state else next_state
        self._time += dt_value
        self._step_index += 1

        return make_step_result(
            state=self._state,
            time=self._time,
            dt=dt_value,
            step_index=self._step_index,
            command_type=self._metadata.command_type,
            applied_command=command_array,
            diagnostics={
                "engine": self._metadata.engine_type.value,
                "dynamics_class": type(self._dynamics).__name__,
            },
        )

    def get_state(self) -> FloatArray:
        """Return current canonical State9."""
        self._ensure_open()
        self._ensure_initialized()

        if self._state is None:  # pragma: no cover - defensive guard.
            raise EngineStateError("ODEPhysicsEngine has not been reset.")

        return self._state.copy() if self._copy_state else self._state

    def get_time(self) -> float:
        """Return current simulation time in seconds."""
        self._ensure_open()
        return float(self._time)

    def get_step_index(self) -> int:
        """Return number of completed physics steps."""
        self._ensure_open()
        return int(self._step_index)

    def get_metadata(self) -> EngineMetadata:
        """Return static ODE engine metadata."""
        return self._metadata

    def close(self) -> None:
        """Close engine.

        ODE engine has no external resources, but close still marks the object
        unusable to catch accidental reuse.
        """
        self._closed = True

    @property
    def is_initialized(self) -> bool:
        """Return True if reset has provided an initial state."""
        return self._state is not None

    @property
    def is_closed(self) -> bool:
        """Return True if close has been called."""
        return self._closed

    @staticmethod
    def _validate_dynamics(dynamics: object) -> None:
        """Validate dynamics object has required discrete method."""
        if not isinstance(dynamics, DiscreteDynamicsProtocol):
            raise EngineConfigurationError(
                "ODEPhysicsEngine dynamics must implement "
                "discrete(state, command, dt)."
            )

    @staticmethod
    def _validate_metadata(metadata: EngineMetadata) -> None:
        """Validate ODE-specific metadata constraints."""
        if metadata.engine_type is not EngineType.ODE:
            raise EngineConfigurationError(
                "ODEPhysicsEngine metadata.engine_type must be EngineType.ODE."
            )

        if metadata.command_type is not EngineCommandType.CONTROL_COMMAND4:
            raise EngineConfigurationError(
                "ODEPhysicsEngine expects command_type CONTROL_COMMAND4."
            )

        if not metadata.supports_control_command:
            raise EngineConfigurationError(
                "ODEPhysicsEngine metadata must support ControlCommand4."
            )

        if metadata.uses_quaternion_internal:
            raise EngineConfigurationError(
                "ODEPhysicsEngine should not use quaternion internally."
            )

    @staticmethod
    def _validate_initial_time(time: float) -> None:
        """Validate initial/reset time."""
        if isinstance(time, bool):
            raise EngineConfigurationError("initial_time must be finite and >= 0.")

        try:
            value = float(time)
        except (TypeError, ValueError) as exc:
            raise EngineConfigurationError("initial_time must be finite and >= 0.") from exc

        if not np.isfinite(value) or value < 0.0:
            raise EngineConfigurationError("initial_time must be finite and >= 0.")

    def _ensure_initialized(self) -> None:
        """Raise if engine has not been reset."""
        if self._state is None:
            raise EngineStateError(
                "ODEPhysicsEngine has not been reset. "
                "Call reset(initial_state) before step/get_state."
            )

    def _ensure_open(self) -> None:
        """Raise if engine was closed."""
        if self._closed:
            raise EngineStateError("ODEPhysicsEngine is closed.")


__all__ = [
    "DiscreteDynamicsProtocol",
    "ODEPhysicsEngine",
]
