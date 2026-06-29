"""Canonical data contracts for the quadrotor CC-MPC simulation.

This module defines the public array contracts used across the refactored
simulation stack.  It intentionally returns ``numpy.ndarray`` objects instead of
custom runtime containers so the current MPC, dynamics, uncertainty, obstacle,
and MuJoCo adapter code can be migrated incrementally.

Canonical contracts
-------------------
State9:
    [x, y, z, vx, vy, vz, roll, pitch, yaw]

Goal3:
    [x_goal, y_goal, z_goal]

ControlCommand4:
    [phi_c, theta_c, vz_c, psi_dot_c]

ActuatorCommand4:
    [T1, T2, T3, T4]

Gamma9x9:
    State covariance matrix.

Sigma3x3:
    Position covariance matrix, usually Gamma9x9[0:3, 0:3].
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final, Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]

STATE_DIM: Final[int] = 9
GOAL_DIM: Final[int] = 3
CONTROL_DIM: Final[int] = 4
ACTUATOR_DIM: Final[int] = 4
POSITION_DIM: Final[int] = 3
VELOCITY_DIM: Final[int] = 3
ATTITUDE_DIM: Final[int] = 3

TrajectoryLayout = Literal["time_major", "state_major"]
ControlTrajectoryLayout = Literal["time_major", "control_major"]


class DataContractError(ValueError):
    """Raised when a value violates a canonical simulation data contract."""


class StateIndex(IntEnum):
    """Indices of the canonical State9 vector."""

    X = 0
    Y = 1
    Z = 2
    VX = 3
    VY = 4
    VZ = 5
    ROLL = 6
    PITCH = 7
    YAW = 8


class ControlIndex(IntEnum):
    """Indices of the canonical ControlCommand4 vector."""

    PHI_C = 0
    THETA_C = 1
    VZ_C = 2
    PSI_DOT_C = 3


class ActuatorIndex(IntEnum):
    """Indices of the canonical ActuatorCommand4 vector."""

    T1 = 0
    T2 = 1
    T3 = 2
    T4 = 3


STATE9_NAMES: Final[tuple[str, ...]] = (
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "roll",
    "pitch",
    "yaw",
)

CONTROL_COMMAND4_NAMES: Final[tuple[str, ...]] = (
    "phi_c",
    "theta_c",
    "vz_c",
    "psi_dot_c",
)

ACTUATOR_COMMAND4_NAMES: Final[tuple[str, ...]] = (
    "T1",
    "T2",
    "T3",
    "T4",
)

GOAL3_NAMES: Final[tuple[str, ...]] = (
    "x_goal",
    "y_goal",
    "z_goal",
)


class State9Parts(NamedTuple):
    """Convenient named split of State9."""

    position: FloatArray
    velocity: FloatArray
    attitude: FloatArray


@dataclass(frozen=True)
class State9View:
    """Read-only semantic view of a State9 array.

    The array itself is copied during construction to prevent accidental
    mutation from the caller.
    """

    array: FloatArray

    @classmethod
    def from_array(cls, value: ArrayLike) -> "State9View":
        return cls(as_state9(value))

    @property
    def x(self) -> float:
        return float(self.array[StateIndex.X])

    @property
    def y(self) -> float:
        return float(self.array[StateIndex.Y])

    @property
    def z(self) -> float:
        return float(self.array[StateIndex.Z])

    @property
    def vx(self) -> float:
        return float(self.array[StateIndex.VX])

    @property
    def vy(self) -> float:
        return float(self.array[StateIndex.VY])

    @property
    def vz(self) -> float:
        return float(self.array[StateIndex.VZ])

    @property
    def roll(self) -> float:
        return float(self.array[StateIndex.ROLL])

    @property
    def pitch(self) -> float:
        return float(self.array[StateIndex.PITCH])

    @property
    def yaw(self) -> float:
        return float(self.array[StateIndex.YAW])

    @property
    def position(self) -> FloatArray:
        return self.array[0:3].copy()

    @property
    def velocity(self) -> FloatArray:
        return self.array[3:6].copy()

    @property
    def attitude(self) -> FloatArray:
        return self.array[6:9].copy()


def _as_float_array(value: ArrayLike, *, name: str, copy: bool = True) -> FloatArray:
    """Convert input to a float64 NumPy array and validate finite values."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"{name} must be convertible to float64 array.") from exc

    if copy:
        array = array.copy()

    if not np.all(np.isfinite(array)):
        raise DataContractError(f"{name} must contain only finite values.")

    return array


def _require_shape(array: FloatArray, expected_shape: tuple[int, ...], *, name: str) -> None:
    """Validate exact array shape."""
    if array.shape != expected_shape:
        raise DataContractError(
            f"{name} must have shape {expected_shape}, got {array.shape}."
        )


def _require_vector(value: ArrayLike, dim: int, *, name: str, copy: bool = True) -> FloatArray:
    """Validate a one-dimensional finite vector with fixed length."""
    array = _as_float_array(value, name=name, copy=copy)
    _require_shape(array, (dim,), name=name)
    return array


def as_state9(value: ArrayLike, *, copy: bool = True) -> FloatArray:
    """Return value as canonical State9.

    Ordering:
        [x, y, z, vx, vy, vz, roll, pitch, yaw]
    """
    return _require_vector(value, STATE_DIM, name="State9", copy=copy)


def as_goal3(value: ArrayLike, *, copy: bool = True) -> FloatArray:
    """Return value as canonical Goal3: [x_goal, y_goal, z_goal]."""
    return _require_vector(value, GOAL_DIM, name="Goal3", copy=copy)


def as_control_command4(value: ArrayLike, *, copy: bool = True) -> FloatArray:
    """Return value as canonical ControlCommand4.

    Ordering:
        [phi_c, theta_c, vz_c, psi_dot_c]
    """
    return _require_vector(value, CONTROL_DIM, name="ControlCommand4", copy=copy)


def as_actuator_command4(
    value: ArrayLike,
    *,
    allow_negative: bool = False,
    copy: bool = True,
) -> FloatArray:
    """Return value as canonical ActuatorCommand4.

    Ordering:
        [T1, T2, T3, T4]

    By default, negative actuator thrust is rejected because rotor thrust is
    physically non-negative for the intended MuJoCo rotor-force model.
    """
    array = _require_vector(value, ACTUATOR_DIM, name="ActuatorCommand4", copy=copy)
    if not allow_negative and np.any(array < 0.0):
        raise DataContractError("ActuatorCommand4 thrust values must be non-negative.")
    return array


def as_position3(value: ArrayLike, *, copy: bool = True) -> FloatArray:
    """Return value as a finite 3D position vector."""
    return _require_vector(value, POSITION_DIM, name="Position3", copy=copy)


def as_velocity3(value: ArrayLike, *, copy: bool = True) -> FloatArray:
    """Return value as a finite 3D velocity vector."""
    return _require_vector(value, VELOCITY_DIM, name="Velocity3", copy=copy)


def as_attitude3(value: ArrayLike, *, copy: bool = True) -> FloatArray:
    """Return value as finite Euler ZYX attitude [roll, pitch, yaw] in radians."""
    return _require_vector(value, ATTITUDE_DIM, name="Attitude3", copy=copy)


def make_state9(
    *,
    x: float,
    y: float,
    z: float,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> FloatArray:
    """Construct a canonical State9 vector from named fields."""
    return as_state9([x, y, z, vx, vy, vz, roll, pitch, yaw])


def make_control_command4(
    *,
    phi_c: float,
    theta_c: float,
    vz_c: float,
    psi_dot_c: float,
) -> FloatArray:
    """Construct a canonical ControlCommand4 vector from named fields."""
    return as_control_command4([phi_c, theta_c, vz_c, psi_dot_c])


def make_actuator_command4(
    *,
    T1: float,
    T2: float,
    T3: float,
    T4: float,
    allow_negative: bool = False,
) -> FloatArray:
    """Construct a canonical ActuatorCommand4 vector from named fields."""
    return as_actuator_command4([T1, T2, T3, T4], allow_negative=allow_negative)


def split_state9(state: ArrayLike) -> State9Parts:
    """Split State9 into position, velocity, and attitude blocks."""
    array = as_state9(state, copy=False)
    return State9Parts(
        position=array[0:3].copy(),
        velocity=array[3:6].copy(),
        attitude=array[6:9].copy(),
    )


def state9_position(state: ArrayLike) -> FloatArray:
    """Extract position [x, y, z] from State9."""
    return as_state9(state, copy=False)[0:3].copy()


def state9_velocity(state: ArrayLike) -> FloatArray:
    """Extract velocity [vx, vy, vz] from State9."""
    return as_state9(state, copy=False)[3:6].copy()


def state9_attitude(state: ArrayLike) -> FloatArray:
    """Extract attitude [roll, pitch, yaw] from State9."""
    return as_state9(state, copy=False)[6:9].copy()


def as_trajectory9(
    value: ArrayLike,
    *,
    layout: TrajectoryLayout = "time_major",
    copy: bool = True,
) -> FloatArray:
    """Validate a trajectory of State9 vectors.

    Supported layouts:
        time_major:  shape (T, 9)
        state_major: shape (9, T)

    The function preserves the input layout. Use ``trajectory9_to_time_major`` if
    downstream code requires shape (T, 9).
    """
    array = _as_float_array(value, name="Trajectory9", copy=copy)
    if array.ndim != 2:
        raise DataContractError(f"Trajectory9 must be 2D, got ndim={array.ndim}.")

    if layout == "time_major":
        if array.shape[1] != STATE_DIM:
            raise DataContractError(
                f"Trajectory9 time_major must have shape (T, 9), got {array.shape}."
            )
    elif layout == "state_major":
        if array.shape[0] != STATE_DIM:
            raise DataContractError(
                f"Trajectory9 state_major must have shape (9, T), got {array.shape}."
            )
    else:
        raise DataContractError(f"Unsupported trajectory layout: {layout!r}.")

    return array


def trajectory9_to_time_major(
    value: ArrayLike,
    *,
    layout: TrajectoryLayout,
    copy: bool = True,
) -> FloatArray:
    """Return a State9 trajectory with shape (T, 9)."""
    array = as_trajectory9(value, layout=layout, copy=copy)
    if layout == "state_major":
        return array.T.copy()
    return array.copy() if copy else array


def as_control_trajectory4(
    value: ArrayLike,
    *,
    layout: ControlTrajectoryLayout = "time_major",
    copy: bool = True,
) -> FloatArray:
    """Validate a trajectory of ControlCommand4 vectors.

    Supported layouts:
        time_major:    shape (T, 4)
        control_major: shape (4, T)
    """
    array = _as_float_array(value, name="ControlTrajectory4", copy=copy)
    if array.ndim != 2:
        raise DataContractError(f"ControlTrajectory4 must be 2D, got ndim={array.ndim}.")

    if layout == "time_major":
        if array.shape[1] != CONTROL_DIM:
            raise DataContractError(
                "ControlTrajectory4 time_major must have shape (T, 4), "
                f"got {array.shape}."
            )
    elif layout == "control_major":
        if array.shape[0] != CONTROL_DIM:
            raise DataContractError(
                "ControlTrajectory4 control_major must have shape (4, T), "
                f"got {array.shape}."
            )
    else:
        raise DataContractError(f"Unsupported control trajectory layout: {layout!r}.")

    return array


def control_trajectory4_to_time_major(
    value: ArrayLike,
    *,
    layout: ControlTrajectoryLayout,
    copy: bool = True,
) -> FloatArray:
    """Return a ControlCommand4 trajectory with shape (T, 4)."""
    array = as_control_trajectory4(value, layout=layout, copy=copy)
    if layout == "control_major":
        return array.T.copy()
    return array.copy() if copy else array


def is_symmetric(
    matrix: ArrayLike,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> bool:
    """Return True if matrix is symmetric within tolerance."""
    array = _as_float_array(matrix, name="matrix", copy=False)
    return array.ndim == 2 and array.shape[0] == array.shape[1] and bool(
        np.allclose(array, array.T, atol=atol, rtol=rtol)
    )


def is_psd(
    matrix: ArrayLike,
    *,
    atol: float = 1e-9,
    symmetrize: bool = True,
) -> bool:
    """Return True if matrix is positive semi-definite within tolerance."""
    array = _as_float_array(matrix, name="matrix", copy=False)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        return False

    if symmetrize:
        array = 0.5 * (array + array.T)

    try:
        eigvals = np.linalg.eigvalsh(array)
    except np.linalg.LinAlgError:
        return False

    return bool(np.min(eigvals) >= -atol)


def as_gamma9x9(
    value: ArrayLike,
    *,
    require_psd: bool = True,
    atol: float = 1e-9,
    copy: bool = True,
) -> FloatArray:
    """Return value as canonical 9x9 state covariance matrix."""
    array = _as_float_array(value, name="Gamma9x9", copy=copy)
    _require_shape(array, (STATE_DIM, STATE_DIM), name="Gamma9x9")

    if not is_symmetric(array, atol=atol, rtol=0.0):
        raise DataContractError("Gamma9x9 must be symmetric.")

    if require_psd and not is_psd(array, atol=atol, symmetrize=True):
        raise DataContractError("Gamma9x9 must be positive semi-definite.")

    return array


def as_sigma3x3(
    value: ArrayLike,
    *,
    require_psd: bool = True,
    atol: float = 1e-9,
    copy: bool = True,
) -> FloatArray:
    """Return value as canonical 3x3 position covariance matrix."""
    array = _as_float_array(value, name="Sigma3x3", copy=copy)
    _require_shape(array, (POSITION_DIM, POSITION_DIM), name="Sigma3x3")

    if not is_symmetric(array, atol=atol, rtol=0.0):
        raise DataContractError("Sigma3x3 must be symmetric.")

    if require_psd and not is_psd(array, atol=atol, symmetrize=True):
        raise DataContractError("Sigma3x3 must be positive semi-definite.")

    return array


def position_covariance_from_gamma9x9(
    gamma: ArrayLike,
    *,
    require_psd: bool = True,
    atol: float = 1e-9,
) -> FloatArray:
    """Extract Sigma3x3 = Gamma9x9[0:3, 0:3]."""
    gamma_array = as_gamma9x9(gamma, require_psd=require_psd, atol=atol, copy=False)
    return as_sigma3x3(
        gamma_array[0:3, 0:3],
        require_psd=require_psd,
        atol=atol,
        copy=True,
    )


def covariance_from_std(std: ArrayLike, *, name: str = "std") -> FloatArray:
    """Create a diagonal covariance matrix from standard deviations.

    Parameters
    ----------
    std:
        One-dimensional vector of standard deviations.  All entries must be
        finite and non-negative.
    """
    std_array = _as_float_array(std, name=name, copy=True)
    if std_array.ndim != 1:
        raise DataContractError(f"{name} must be a 1D vector.")
    if np.any(std_array < 0.0):
        raise DataContractError(f"{name} must contain non-negative values.")
    return np.diag(std_array**2).astype(np.float64)


def assert_state9(value: ArrayLike) -> None:
    """Raise DataContractError if value is not valid State9."""
    as_state9(value, copy=False)


def assert_goal3(value: ArrayLike) -> None:
    """Raise DataContractError if value is not valid Goal3."""
    as_goal3(value, copy=False)


def assert_control_command4(value: ArrayLike) -> None:
    """Raise DataContractError if value is not valid ControlCommand4."""
    as_control_command4(value, copy=False)


def assert_actuator_command4(value: ArrayLike) -> None:
    """Raise DataContractError if value is not valid ActuatorCommand4."""
    as_actuator_command4(value, copy=False)


__all__ = [
    "ACTUATOR_COMMAND4_NAMES",
    "ACTUATOR_DIM",
    "ATTITUDE_DIM",
    "CONTROL_COMMAND4_NAMES",
    "CONTROL_DIM",
    "GOAL3_NAMES",
    "GOAL_DIM",
    "POSITION_DIM",
    "STATE9_NAMES",
    "STATE_DIM",
    "VELOCITY_DIM",
    "ActuatorIndex",
    "ControlIndex",
    "DataContractError",
    "FloatArray",
    "State9Parts",
    "State9View",
    "StateIndex",
    "as_actuator_command4",
    "as_attitude3",
    "as_control_command4",
    "as_control_trajectory4",
    "as_gamma9x9",
    "as_goal3",
    "as_position3",
    "as_sigma3x3",
    "as_state9",
    "as_trajectory9",
    "as_velocity3",
    "assert_actuator_command4",
    "assert_control_command4",
    "assert_goal3",
    "assert_state9",
    "control_trajectory4_to_time_major",
    "covariance_from_std",
    "is_psd",
    "is_symmetric",
    "make_actuator_command4",
    "make_control_command4",
    "make_state9",
    "position_covariance_from_gamma9x9",
    "split_state9",
    "state9_attitude",
    "state9_position",
    "state9_velocity",
    "trajectory9_to_time_major",
]
