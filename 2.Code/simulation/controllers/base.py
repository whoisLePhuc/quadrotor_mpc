"""Base controller abstraction for the quadrotor simulation.

This module defines the public contract that every controller must follow.
The runtime should depend on this interface only, not on a concrete CCMPC,
PID, LQR, or emergency-stop implementation.

Canonical boundary
------------------
Controllers consume estimated state and scenario context:

    estimated_state = State9 = [x, y, z, vx, vy, vz, roll, pitch, yaw]
    goal            = Goal3  = [x_goal, y_goal, z_goal]
    covariance      = Gamma9x9 | None

Controllers produce a high-level command:

    ControlCommand4 = [phi_c, theta_c, vz_c, psi_dot_c]

Controllers do not:
    - step the physics engine
    - convert MuJoCo qpos/qvel to State9
    - mix rotor thrusts
    - parse YAML files
    - write logs directly
    - render visualization
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ccmpc.types import (
    CONTROL_DIM,
    GOAL_DIM,
    STATE_DIM,
    FloatArray,
    as_control_command4,
    as_control_trajectory4,
    as_gamma9x9,
    as_goal3,
    as_position3,
    as_state9,
    as_trajectory9,
)


class ControllerError(RuntimeError):
    """Base exception raised by controller implementations."""


class ControllerConfigurationError(ControllerError):
    """Raised when a controller is constructed with invalid configuration."""


class ControllerInputError(ControllerError):
    """Raised when controller input violates the canonical contract."""


class ControllerOutputError(ControllerError):
    """Raised when controller output violates the canonical contract."""


class ControllerSolveError(ControllerError):
    """Raised when a controller fails during command computation."""


class ControllerType(str, Enum):
    """Supported controller families."""

    CCMPC = "ccmpc"
    NOMINAL_MPC = "nominal_mpc"
    PID = "pid"
    LQR = "lqr"
    EMERGENCY_STOP = "emergency_stop"
    CUSTOM = "custom"


class ControllerStatus(str, Enum):
    """High-level controller solve status."""

    SUCCESS = "success"
    WARNING = "warning"
    INFEASIBLE = "infeasible"
    MAX_ITER = "max_iter"
    FAILED = "failed"
    FALLBACK = "fallback"
    NOT_READY = "not_ready"


@dataclass(frozen=True)
class ControllerMetadata:
    """Static metadata describing a controller instance.

    Attributes
    ----------
    controller_type:
        Controller family such as CCMPC or PID.
    name:
        Human-readable controller implementation name.
    state_dim:
        Expected canonical state dimension.  Should remain 9.
    goal_dim:
        Expected canonical goal dimension.  Should remain 3.
    command_dim:
        Expected command dimension.  Should remain 4.
    horizon:
        Prediction horizon length if applicable.
    dt:
        Controller/MPC timestep if applicable.
    supports_obstacles:
        True if the controller can consume obstacle predictions.
    supports_covariance:
        True if the controller can consume Gamma9x9 covariance.
    deterministic:
        True when repeated calls with identical input should produce identical
        output, ignoring solver numerical tolerances.
    description:
        Optional human-readable description.
    extra:
        Optional implementation-specific static metadata for logging only.
    """

    controller_type: ControllerType
    name: str
    state_dim: int = STATE_DIM
    goal_dim: int = GOAL_DIM
    command_dim: int = CONTROL_DIM
    horizon: int | None = None
    dt: float | None = None
    supports_obstacles: bool = False
    supports_covariance: bool = False
    deterministic: bool = True
    description: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate metadata consistency."""
        if not self.name.strip():
            raise ControllerConfigurationError(
                "ControllerMetadata.name must be non-empty."
            )

        if self.state_dim != STATE_DIM:
            raise ControllerConfigurationError(
                f"ControllerMetadata.state_dim must be {STATE_DIM}, "
                f"got {self.state_dim}."
            )

        if self.goal_dim != GOAL_DIM:
            raise ControllerConfigurationError(
                f"ControllerMetadata.goal_dim must be {GOAL_DIM}, "
                f"got {self.goal_dim}."
            )

        if self.command_dim != CONTROL_DIM:
            raise ControllerConfigurationError(
                f"ControllerMetadata.command_dim must be {CONTROL_DIM}, "
                f"got {self.command_dim}."
            )

        if self.horizon is not None and self.horizon <= 0:
            raise ControllerConfigurationError(
                "ControllerMetadata.horizon must be > 0 when provided."
            )

        if self.dt is not None and (
            isinstance(self.dt, bool) or not np.isfinite(float(self.dt)) or self.dt <= 0.0
        ):
            raise ControllerConfigurationError(
                "ControllerMetadata.dt must be finite and > 0 when provided."
            )


@dataclass(frozen=True)
class ObstaclePrediction:
    """Predicted obstacle trajectory consumed by controllers.

    This type is deliberately lightweight.  More detailed geometry may be stored
    in ``metadata`` until obstacle modeling is fully separated.

    Attributes
    ----------
    obstacle_id:
        Stable obstacle identifier.
    positions:
        Predicted center positions with shape (T, 3).
    radii:
        Optional ellipsoid/collision radii with shape (3,).
    covariance:
        Optional obstacle position covariance.  Accepted shapes are (3, 3) for
        a constant covariance or (T, 3, 3) for time-varying covariance.
    active:
        Whether this obstacle should be considered by the controller.
    metadata:
        Optional obstacle-specific data for controller adapters/logging.
    """

    obstacle_id: str
    positions: FloatArray
    radii: FloatArray | None = None
    covariance: FloatArray | None = None
    active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate obstacle prediction contract."""
        if not self.obstacle_id.strip():
            raise ControllerInputError("ObstaclePrediction.obstacle_id must be non-empty.")

        positions = _as_position_trajectory(self.positions, "ObstaclePrediction.positions")
        object.__setattr__(self, "positions", positions)

        if self.radii is not None:
            radii = as_position3(self.radii)
            if np.any(radii <= 0.0):
                raise ControllerInputError("ObstaclePrediction.radii must be > 0.")
            object.__setattr__(self, "radii", radii)

        if self.covariance is not None:
            covariance = _as_obstacle_covariance(
                self.covariance,
                horizon=positions.shape[0],
                name="ObstaclePrediction.covariance",
            )
            object.__setattr__(self, "covariance", covariance)

        if not isinstance(self.active, bool):
            raise ControllerInputError("ObstaclePrediction.active must be bool.")

        if not isinstance(self.metadata, dict):
            raise ControllerInputError("ObstaclePrediction.metadata must be dict.")


@dataclass(frozen=True)
class ControllerInput:
    """Input packet consumed by a controller at one decision time.

    Attributes
    ----------
    time:
        Current simulation time in seconds.
    estimated_state:
        State estimate in canonical State9 format.
    goal:
        Goal position in canonical Goal3 format.
    covariance:
        Optional state covariance Gamma9x9.
    obstacle_predictions:
        Tuple of obstacle predictions.  Empty tuple means no known obstacles.
    previous_solution:
        Optional previous controller solution for warm-start.
    reference_trajectory:
        Optional reference State9 trajectory.  Shape (T, 9).
    config:
        Optional already-parsed controller config dictionary/object.  This base
        class does not interpret it.
    metadata:
        Optional runtime context for logging/adapters.
    """

    time: float
    estimated_state: FloatArray
    goal: FloatArray
    covariance: FloatArray | None = None
    obstacle_predictions: tuple[ObstaclePrediction, ...] = ()
    previous_solution: Any | None = None
    reference_trajectory: FloatArray | None = None
    config: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate input contract."""
        object.__setattr__(self, "time", validate_controller_time(self.time))

        object.__setattr__(
            self,
            "estimated_state",
            as_state9(self.estimated_state),
        )

        object.__setattr__(self, "goal", as_goal3(self.goal))

        if self.covariance is not None:
            object.__setattr__(
                self,
                "covariance",
                as_gamma9x9(self.covariance),
            )

        if not isinstance(self.obstacle_predictions, tuple):
            raise ControllerInputError(
                "ControllerInput.obstacle_predictions must be a tuple."
            )

        for index, obstacle in enumerate(self.obstacle_predictions):
            if not isinstance(obstacle, ObstaclePrediction):
                raise ControllerInputError(
                    "ControllerInput.obstacle_predictions must contain "
                    f"ObstaclePrediction objects; item {index} is "
                    f"{type(obstacle).__name__}."
                )

        if self.reference_trajectory is not None:
            object.__setattr__(
                self,
                "reference_trajectory",
                as_trajectory9(self.reference_trajectory, layout="time_major"),
            )

        if not isinstance(self.metadata, dict):
            raise ControllerInputError("ControllerInput.metadata must be dict.")


@dataclass(frozen=True)
class ControllerDiagnostics:
    """Diagnostics returned by a controller call.

    Diagnostics are intended for logging and debugging.  Runtime logic should
    primarily rely on ``ControllerOutput.command`` and ``ControllerStatus``.
    """

    status: ControllerStatus
    success: bool
    solve_time_ms: float | None = None
    objective_value: float | None = None
    iterations: int | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    max_constraint_violation: float | None = None
    min_obstacle_margin: float | None = None
    notes: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate diagnostics contract."""
        if not isinstance(self.status, ControllerStatus):
            raise ControllerOutputError(
                "ControllerDiagnostics.status must be ControllerStatus."
            )

        if not isinstance(self.success, bool):
            raise ControllerOutputError("ControllerDiagnostics.success must be bool.")

        _validate_optional_non_negative_float(
            self.solve_time_ms,
            "ControllerDiagnostics.solve_time_ms",
        )
        _validate_optional_int_non_negative(
            self.iterations,
            "ControllerDiagnostics.iterations",
        )
        _validate_optional_finite_float(
            self.objective_value,
            "ControllerDiagnostics.objective_value",
        )
        _validate_optional_finite_float(
            self.max_constraint_violation,
            "ControllerDiagnostics.max_constraint_violation",
        )
        _validate_optional_finite_float(
            self.min_obstacle_margin,
            "ControllerDiagnostics.min_obstacle_margin",
        )

        if not isinstance(self.fallback_used, bool):
            raise ControllerOutputError("ControllerDiagnostics.fallback_used must be bool.")

        if self.fallback_reason is not None and not isinstance(self.fallback_reason, str):
            raise ControllerOutputError(
                "ControllerDiagnostics.fallback_reason must be str | None."
            )

        if not isinstance(self.notes, tuple):
            raise ControllerOutputError("ControllerDiagnostics.notes must be tuple.")

        for index, note in enumerate(self.notes):
            if not isinstance(note, str):
                raise ControllerOutputError(
                    f"ControllerDiagnostics.notes[{index}] must be str."
                )

        if not isinstance(self.extra, dict):
            raise ControllerOutputError("ControllerDiagnostics.extra must be dict.")


@dataclass(frozen=True)
class ControllerOutput:
    """Output packet returned by a controller at one decision time.

    Attributes
    ----------
    command:
        Canonical ControlCommand4 to apply at the engine boundary.
    predicted_trajectory:
        Optional predicted State9 trajectory with shape (T, 9).
    control_trajectory:
        Optional planned ControlCommand4 trajectory with shape (T, 4).
    diagnostics:
        Structured diagnostics for logging and fallback decisions.
    raw_solution:
        Optional backend-specific solver result.  Runtime should not depend on
        this field.
    metadata:
        Optional output metadata for logging/adapters.
    """

    command: FloatArray
    predicted_trajectory: FloatArray | None = None
    control_trajectory: FloatArray | None = None
    diagnostics: ControllerDiagnostics = field(
        default_factory=lambda: ControllerDiagnostics(
            status=ControllerStatus.SUCCESS,
            success=True,
        )
    )
    raw_solution: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate output contract."""
        object.__setattr__(
            self,
            "command",
            as_control_command4(self.command),
        )

        if self.predicted_trajectory is not None:
            object.__setattr__(
                self,
                "predicted_trajectory",
                as_trajectory9(self.predicted_trajectory, layout="time_major"),
            )

        if self.control_trajectory is not None:
            object.__setattr__(
                self,
                "control_trajectory",
                as_control_trajectory4(
                    self.control_trajectory,
                    layout="time_major",
                ),
            )

        if not isinstance(self.diagnostics, ControllerDiagnostics):
            raise ControllerOutputError(
                "ControllerOutput.diagnostics must be ControllerDiagnostics."
            )

        if not isinstance(self.metadata, dict):
            raise ControllerOutputError("ControllerOutput.metadata must be dict.")


@runtime_checkable
class ControllerProtocol(Protocol):
    """Structural protocol for controllers."""

    def reset(self) -> None:
        """Reset controller internal state."""

    def compute_command(self, input_data: ControllerInput) -> ControllerOutput:
        """Compute one high-level ControlCommand4."""

    def get_metadata(self) -> ControllerMetadata:
        """Return static controller metadata."""

    def close(self) -> None:
        """Release controller resources."""


class Controller(ABC):
    """Abstract base class for all controllers."""

    @abstractmethod
    def reset(self) -> None:
        """Reset controller internal state."""

    @abstractmethod
    def compute_command(self, input_data: ControllerInput) -> ControllerOutput:
        """Compute one high-level ControlCommand4."""

    @abstractmethod
    def get_metadata(self) -> ControllerMetadata:
        """Return static controller metadata."""

    def close(self) -> None:
        """Release controller resources.

        Stateless controllers may keep the default no-op implementation.
        """


def make_success_output(
    *,
    command: FloatArray,
    predicted_trajectory: FloatArray | None = None,
    control_trajectory: FloatArray | None = None,
    solve_time_ms: float | None = None,
    objective_value: float | None = None,
    iterations: int | None = None,
    raw_solution: Any | None = None,
    metadata: dict[str, Any] | None = None,
    diagnostics_extra: dict[str, Any] | None = None,
) -> ControllerOutput:
    """Convenience helper for successful controller output."""
    diagnostics = ControllerDiagnostics(
        status=ControllerStatus.SUCCESS,
        success=True,
        solve_time_ms=solve_time_ms,
        objective_value=objective_value,
        iterations=iterations,
        fallback_used=False,
        extra={} if diagnostics_extra is None else dict(diagnostics_extra),
    )

    return ControllerOutput(
        command=command,
        predicted_trajectory=predicted_trajectory,
        control_trajectory=control_trajectory,
        diagnostics=diagnostics,
        raw_solution=raw_solution,
        metadata={} if metadata is None else dict(metadata),
    )


def make_fallback_output(
    *,
    command: FloatArray,
    reason: str,
    status: ControllerStatus = ControllerStatus.FALLBACK,
    predicted_trajectory: FloatArray | None = None,
    control_trajectory: FloatArray | None = None,
    raw_solution: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> ControllerOutput:
    """Convenience helper for fallback controller output."""
    if not reason.strip():
        raise ControllerOutputError("Fallback reason must be non-empty.")

    diagnostics = ControllerDiagnostics(
        status=status,
        success=False,
        fallback_used=True,
        fallback_reason=reason,
    )

    return ControllerOutput(
        command=command,
        predicted_trajectory=predicted_trajectory,
        control_trajectory=control_trajectory,
        diagnostics=diagnostics,
        raw_solution=raw_solution,
        metadata={} if metadata is None else dict(metadata),
    )


def validate_controller_time(time: float) -> float:
    """Validate controller time in seconds."""
    if isinstance(time, bool):
        raise ControllerInputError("ControllerInput.time must be finite and >= 0.")

    try:
        value = float(time)
    except (TypeError, ValueError) as exc:
        raise ControllerInputError("ControllerInput.time must be finite and >= 0.") from exc

    if not np.isfinite(value) or value < 0.0:
        raise ControllerInputError("ControllerInput.time must be finite and >= 0.")

    return value


def first_control_from_trajectory(
    control_trajectory: FloatArray,
    *,
    layout: str = "time_major",
) -> FloatArray:
    """Extract the first ControlCommand4 from a planned control trajectory.

    Parameters
    ----------
    control_trajectory:
        Time-major shape (T, 4) by default, or control-major shape (4, T).
    layout:
        Either ``"time_major"`` or ``"control_major"``.
    """
    if layout == "time_major":
        trajectory = as_control_trajectory4(control_trajectory, layout="time_major")
        if trajectory.shape[0] == 0:
            raise ControllerOutputError("control_trajectory must contain at least one step.")
        return as_control_command4(trajectory[0, :])

    if layout == "control_major":
        trajectory = as_control_trajectory4(
            control_trajectory,
            layout="control_major",
        )
        if trajectory.shape[1] == 0:
            raise ControllerOutputError("control_trajectory must contain at least one step.")
        return as_control_command4(trajectory[:, 0])

    raise ControllerOutputError(f"Unsupported control trajectory layout: {layout!r}.")


def _as_position_trajectory(value: FloatArray, name: str) -> FloatArray:
    """Validate position trajectory with shape (T, 3)."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ControllerInputError(f"{name} must be convertible to float64 array.") from exc

    if array.ndim != 2 or array.shape[1] != 3:
        raise ControllerInputError(f"{name} must have shape (T, 3), got {array.shape}.")

    if array.shape[0] == 0:
        raise ControllerInputError(f"{name} must contain at least one prediction step.")

    if not np.all(np.isfinite(array)):
        raise ControllerInputError(f"{name} must contain only finite values.")

    return array.copy()


def _as_obstacle_covariance(
    value: FloatArray,
    *,
    horizon: int,
    name: str,
) -> FloatArray:
    """Validate obstacle covariance with shape (3,3) or (T,3,3)."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ControllerInputError(f"{name} must be convertible to float64 array.") from exc

    if array.shape == (3, 3):
        if not _is_symmetric(array):
            raise ControllerInputError(f"{name} must be symmetric.")
        return array.copy()

    if array.shape == (horizon, 3, 3):
        for index in range(horizon):
            if not _is_symmetric(array[index]):
                raise ControllerInputError(f"{name}[{index}] must be symmetric.")
        return array.copy()

    raise ControllerInputError(
        f"{name} must have shape (3, 3) or ({horizon}, 3, 3), "
        f"got {array.shape}."
    )


def _is_symmetric(matrix: FloatArray, *, atol: float = 1e-9) -> bool:
    """Return True if matrix is symmetric within tolerance."""
    return bool(np.allclose(matrix, matrix.T, atol=atol, rtol=0.0))


def _validate_optional_finite_float(value: float | None, name: str) -> None:
    """Validate optional finite float."""
    if value is None:
        return

    if isinstance(value, bool):
        raise ControllerOutputError(f"{name} must be finite float | None.")

    if not np.isfinite(float(value)):
        raise ControllerOutputError(f"{name} must be finite.")


def _validate_optional_non_negative_float(value: float | None, name: str) -> None:
    """Validate optional finite float >= 0."""
    if value is None:
        return

    _validate_optional_finite_float(value, name)
    if float(value) < 0.0:
        raise ControllerOutputError(f"{name} must be >= 0.")


def _validate_optional_int_non_negative(value: int | None, name: str) -> None:
    """Validate optional integer >= 0."""
    if value is None:
        return

    if isinstance(value, bool) or not isinstance(value, int):
        raise ControllerOutputError(f"{name} must be int | None.")

    if value < 0:
        raise ControllerOutputError(f"{name} must be >= 0.")


DEFAULT_CCMPC_METADATA = ControllerMetadata(
    controller_type=ControllerType.CCMPC,
    name="CCMPCController",
    horizon=None,
    dt=None,
    supports_obstacles=True,
    supports_covariance=True,
    deterministic=True,
    description="Chance-constrained MPC controller adapter.",
)

DEFAULT_EMERGENCY_STOP_METADATA = ControllerMetadata(
    controller_type=ControllerType.EMERGENCY_STOP,
    name="EmergencyStopController",
    horizon=1,
    dt=None,
    supports_obstacles=False,
    supports_covariance=False,
    deterministic=True,
    description="Fallback controller that commands a safe stop/hold behavior.",
)


__all__ = [
    "DEFAULT_CCMPC_METADATA",
    "DEFAULT_EMERGENCY_STOP_METADATA",
    "Controller",
    "ControllerConfigurationError",
    "ControllerDiagnostics",
    "ControllerError",
    "ControllerInput",
    "ControllerInputError",
    "ControllerMetadata",
    "ControllerOutput",
    "ControllerOutputError",
    "ControllerProtocol",
    "ControllerSolveError",
    "ControllerStatus",
    "ControllerType",
    "ObstaclePrediction",
    "first_control_from_trajectory",
    "make_fallback_output",
    "make_success_output",
    "validate_controller_time",
]
