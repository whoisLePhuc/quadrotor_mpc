"""Reduced-order quadrotor dynamics for CC-MPC.

The canonical external contracts are:

    State9:
        [x, y, z, vx, vy, vz, roll, pitch, yaw]

    ControlCommand4:
        [phi_c, theta_c, vz_c, psi_dot_c]

The model follows the simplified first-order low-pass Euler approximation used
for real-time CC-MPC prediction.  It is intentionally a reduced-order model, not
a full Newton-Euler rigid-body model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from .types import (
    CONTROL_DIM,
    STATE_DIM,
    FloatArray,
    as_control_command4,
    as_state9,
)


JacobianFunction = Callable[[FloatArray], FloatArray]


class DynamicsError(ValueError):
    """Raised when dynamics inputs or parameters are invalid."""


@dataclass(frozen=True)
class QuadrotorDynamicsParams:
    """Parameters of the reduced-order quadrotor dynamics model.

    Parameters
    ----------
    g:
        Gravity magnitude in m/s².
    kD:
        Linear drag coefficient in 1/s.
    k_phi, k_theta:
        Roll and pitch command gains.
    k_vz:
        Vertical velocity command gain.
    tau_phi, tau_theta, tau_vz:
        First-order response time constants in seconds.
    """

    g: float = 9.81
    kD: float = 0.5
    k_phi: float = 1.0
    k_theta: float = 1.0
    k_vz: float = 3.0
    tau_phi: float = 0.2
    tau_theta: float = 0.2
    tau_vz: float = 0.4

    def __post_init__(self) -> None:
        """Validate model parameters."""
        _require_positive_finite(self.g, "g")
        _require_non_negative_finite(self.kD, "kD")
        _require_positive_finite(self.k_phi, "k_phi")
        _require_positive_finite(self.k_theta, "k_theta")
        _require_positive_finite(self.k_vz, "k_vz")
        _require_positive_finite(self.tau_phi, "tau_phi")
        _require_positive_finite(self.tau_theta, "tau_theta")
        _require_positive_finite(self.tau_vz, "tau_vz")

    def to_kwargs(self) -> dict[str, float]:
        """Return parameters as kwargs for functional dynamics helpers."""
        return asdict(self)


def _body_tilt_factor(phi: float, theta: float) -> tuple[float, float]:
    r"""Compute normalized tilt factors ``F_theta`` and ``F_phi``.

    .. math::

        A &= \sqrt{1 + \tan^2\theta + \tan^2\phi} \\
        F_\theta &= \frac{\tan\theta}{\cos\theta \cdot A} \\
        F_\phi &= \frac{\tan\phi}{\cos\phi \cdot A}

    The small epsilon guards avoid numerical blow-up near Euler singularities.
    """
    t_phi = math.tan(phi)
    t_theta = math.tan(theta)
    tilt_norm = math.sqrt(1.0 + t_theta * t_theta + t_phi * t_phi)

    if tilt_norm < 1e-12:
        return 0.0, 0.0

    c_theta = math.cos(theta)
    c_phi = math.cos(phi)

    f_theta = t_theta / (c_theta * tilt_norm) if abs(c_theta) > 1e-12 else 0.0
    f_phi = t_phi / (c_phi * tilt_norm) if abs(c_phi) > 1e-12 else 0.0

    return f_theta, f_phi


def continuous_dynamics(
    x: npt.ArrayLike,
    u: npt.ArrayLike,
    *,
    g: float = 9.81,
    kD: float = 0.5,
    k_phi: float = 1.0,
    k_theta: float = 1.0,
    k_vz: float = 3.0,
    tau_phi: float = 0.2,
    tau_theta: float = 0.2,
    tau_vz: float = 0.4,
) -> FloatArray:
    r"""Compute continuous-time dynamics ``x_dot = f(x, u)``.

    Parameters
    ----------
    x:
        Canonical State9.
    u:
        Canonical ControlCommand4.
    g, kD, k_phi, k_theta, k_vz, tau_phi, tau_theta, tau_vz:
        Reduced-order model parameters.

    Returns
    -------
    numpy.ndarray
        State derivative with shape ``(9,)``.
    """
    state = as_state9(x)
    command = as_control_command4(u)
    params = QuadrotorDynamicsParams(
        g=g,
        kD=kD,
        k_phi=k_phi,
        k_theta=k_theta,
        k_vz=k_vz,
        tau_phi=tau_phi,
        tau_theta=tau_theta,
        tau_vz=tau_vz,
    )

    vx = state[3]
    vy = state[4]
    vz = state[5]
    phi = state[6]
    theta = state[7]
    psi = state[8]

    phi_c = command[0]
    theta_c = command[1]
    vz_c = command[2]
    psi_dot_c = command[3]

    f_theta, f_phi = _body_tilt_factor(phi, theta)
    c_psi = math.cos(psi)
    s_psi = math.sin(psi)

    # Position kinematics.
    dx = vx
    dy = vy
    dz = vz

    # Reduced-order velocity dynamics.
    dvx = params.g * f_theta * c_psi - params.g * f_phi * s_psi - params.kD * vx
    dvy = params.g * f_theta * s_psi + params.g * f_phi * c_psi - params.kD * vy
    dvz = (params.k_vz * vz_c - vz) / params.tau_vz

    # First-order attitude command response.
    dphi = (params.k_phi * phi_c - phi) / params.tau_phi
    dtheta = (params.k_theta * theta_c - theta) / params.tau_theta
    dpsi = psi_dot_c

    return np.array(
        [dx, dy, dz, dvx, dvy, dvz, dphi, dtheta, dpsi],
        dtype=np.float64,
    )


def discrete_step(
    x: npt.ArrayLike,
    u: npt.ArrayLike,
    dt: float,
    **params: float,
) -> FloatArray:
    """Integrate one timestep using fourth-order Runge-Kutta.

    Parameters
    ----------
    x:
        Current canonical State9.
    u:
        Current canonical ControlCommand4.
    dt:
        Integration timestep in seconds.
    **params:
        Model parameters passed to ``continuous_dynamics``.

    Returns
    -------
    numpy.ndarray
        Next State9 after one integration step.
    """
    state = as_state9(x)
    command = as_control_command4(u)
    dt_value = _require_positive_finite(dt, "dt")

    def f(xk: FloatArray) -> FloatArray:
        return continuous_dynamics(xk, command, **params)

    k1 = f(state)
    k2 = f(state + 0.5 * dt_value * k1)
    k3 = f(state + 0.5 * dt_value * k2)
    k4 = f(state + dt_value * k3)

    return as_state9(state + (dt_value / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def _finite_diff_jacobian(
    function: JacobianFunction,
    x0: npt.ArrayLike,
    *,
    eps: float = 1e-6,
) -> FloatArray:
    """Compute a central finite-difference Jacobian.

    Parameters
    ----------
    function:
        Function from ``R^m`` to ``R^n``.
    x0:
        Point at which to evaluate the Jacobian.
    eps:
        Perturbation size.

    Returns
    -------
    numpy.ndarray
        Jacobian matrix with shape ``(n, m)``.
    """
    if eps <= 0.0 or not np.isfinite(eps):
        raise DynamicsError("eps must be finite and > 0.")

    x0_array = _as_finite_vector(x0, name="x0")
    y0 = _as_finite_vector(function(x0_array), name="function(x0)")

    output_dim = y0.size
    input_dim = x0_array.size
    jacobian = np.zeros((output_dim, input_dim), dtype=np.float64)

    for index in range(input_dim):
        perturbation = np.zeros(input_dim, dtype=np.float64)
        perturbation[index] = eps

        y_plus = _as_finite_vector(function(x0_array + perturbation), name="f(x + dx)")
        y_minus = _as_finite_vector(function(x0_array - perturbation), name="f(x - dx)")

        if y_plus.shape != y0.shape or y_minus.shape != y0.shape:
            raise DynamicsError("Function output shape changed during finite difference.")

        jacobian[:, index] = (y_plus - y_minus) / (2.0 * eps)

    return jacobian


class QuadrotorDynamics:
    """Reduced-order quadrotor dynamics model with linearization support."""

    def __init__(
        self,
        g: float = 9.81,
        kD: float = 0.5,
        k_phi: float = 1.0,
        k_theta: float = 1.0,
        k_vz: float = 3.0,
        tau_phi: float = 0.2,
        tau_theta: float = 0.2,
        tau_vz: float = 0.4,
    ) -> None:
        self.params = QuadrotorDynamicsParams(
            g=g,
            kD=kD,
            k_phi=k_phi,
            k_theta=k_theta,
            k_vz=k_vz,
            tau_phi=tau_phi,
            tau_theta=tau_theta,
            tau_vz=tau_vz,
        )

    @property
    def g(self) -> float:
        """Gravity magnitude in m/s²."""
        return self.params.g

    @property
    def kD(self) -> float:
        """Linear drag coefficient."""
        return self.params.kD

    @property
    def k_phi(self) -> float:
        """Roll command gain."""
        return self.params.k_phi

    @property
    def k_theta(self) -> float:
        """Pitch command gain."""
        return self.params.k_theta

    @property
    def k_vz(self) -> float:
        """Vertical velocity command gain."""
        return self.params.k_vz

    @property
    def tau_phi(self) -> float:
        """Roll response time constant in seconds."""
        return self.params.tau_phi

    @property
    def tau_theta(self) -> float:
        """Pitch response time constant in seconds."""
        return self.params.tau_theta

    @property
    def tau_vz(self) -> float:
        """Vertical velocity response time constant in seconds."""
        return self.params.tau_vz

    @property
    def _params(self) -> dict[str, float]:
        """Backward-compatible parameter dictionary."""
        return self.params.to_kwargs()

    @classmethod
    def from_params(cls, params: QuadrotorDynamicsParams) -> "QuadrotorDynamics":
        """Create dynamics model from QuadrotorDynamicsParams."""
        if not isinstance(params, QuadrotorDynamicsParams):
            raise TypeError(
                "params must be an instance of QuadrotorDynamicsParams."
            )

        return cls(
            g=params.g,
            kD=params.kD,
            k_phi=params.k_phi,
            k_theta=params.k_theta,
            k_vz=params.k_vz,
            tau_phi=params.tau_phi,
            tau_theta=params.tau_theta,
            tau_vz=params.tau_vz,
        )
    
    @classmethod
    def from_config(cls, config: str | Path | Mapping[str, Any]) -> "QuadrotorDynamics":
        """Create dynamics from a YAML config path or parsed config mapping."""
        config_data = _load_config_mapping(config)
        model_config = _get_nested_mapping(config_data, "model", "quadrotor")

        return cls(
            g=_get_required_float(model_config, "g"),
            kD=_get_required_float(model_config, "kD"),
            k_phi=_get_required_float(model_config, "k_phi"),
            k_theta=_get_required_float(model_config, "k_theta"),
            k_vz=_get_required_float(model_config, "k_vz"),
            tau_phi=_get_required_float(model_config, "tau_phi"),
            tau_theta=_get_required_float(model_config, "tau_theta"),
            tau_vz=_get_required_float(model_config, "tau_vz"),
        )

    def continuous(self, x: npt.ArrayLike, u: npt.ArrayLike) -> FloatArray:
        """Return continuous dynamics ``f(x, u)``."""
        return continuous_dynamics(x, u, **self._params)

    def discrete(self, x: npt.ArrayLike, u: npt.ArrayLike, dt: float) -> FloatArray:
        """Return one-step RK4 discrete dynamics."""
        return discrete_step(x, u, dt, **self._params)

    def jacobian_state(self, x: npt.ArrayLike, u: npt.ArrayLike) -> FloatArray:
        r"""Return continuous-time state Jacobian ``∂f/∂x`` with shape ``(9, 9)``."""
        state = as_state9(x)
        command = as_control_command4(u)

        def function(xk: FloatArray) -> FloatArray:
            return self.continuous(xk, command)

        jacobian = _finite_diff_jacobian(function, state)
        _require_shape(jacobian, (STATE_DIM, STATE_DIM), "jacobian_state")
        return jacobian

    def jacobian_control(self, x: npt.ArrayLike, u: npt.ArrayLike) -> FloatArray:
        r"""Return continuous-time control Jacobian ``∂f/∂u`` with shape ``(9, 4)``.

        The reduced-order model is affine in the control input, so this Jacobian
        is sparse and analytical.
        """
        # Validate inputs even though the analytical control Jacobian only
        # depends on parameters. This catches contract violations early.
        as_state9(x)
        as_control_command4(u)

        jacobian = np.zeros((STATE_DIM, CONTROL_DIM), dtype=np.float64)
        jacobian[6, 0] = self.k_phi / self.tau_phi
        jacobian[7, 1] = self.k_theta / self.tau_theta
        jacobian[5, 2] = self.k_vz / self.tau_vz
        jacobian[8, 3] = 1.0

        return jacobian

    def linearize(
        self,
        x_bar: npt.ArrayLike,
        u_bar: npt.ArrayLike,
        dt: float,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        r"""Linearize the discrete dynamics around ``(x_bar, u_bar)``.

        The resulting affine LTV model is:

        .. math::

            x_{k+1} \approx A_k x_k + B_k u_k + C_k

        with:

        .. math::

            A_k = I + \Delta t A_\mathrm{cont}

            B_k = \Delta t B_\mathrm{cont}
                + \frac{\Delta t^2}{2} A_\mathrm{cont} B_\mathrm{cont}

            C_k = x_{k+1}^{true} - A_k \bar{x}_k - B_k \bar{u}_k

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            ``(A_k, B_k, C_k)`` with shapes ``(9, 9)``, ``(9, 4)``, and ``(9,)``.
        """
        state = as_state9(x_bar)
        command = as_control_command4(u_bar)
        dt_value = _require_positive_finite(dt, "dt")

        a_cont = self.jacobian_state(state, command)
        b_cont = self.jacobian_control(state, command)

        a_k = np.eye(STATE_DIM, dtype=np.float64) + dt_value * a_cont
        b_k = dt_value * b_cont + 0.5 * dt_value * dt_value * (a_cont @ b_cont)

        # Affine offset: make the linearized model exactly match the RK4 rollout
        # at the expansion point.
        x_next = self.discrete(state, command, dt_value)
        c_k = x_next - a_k @ state - b_k @ command

        _require_shape(a_k, (STATE_DIM, STATE_DIM), "A_k")
        _require_shape(b_k, (STATE_DIM, CONTROL_DIM), "B_k")
        _require_shape(c_k, (STATE_DIM,), "C_k")

        return a_k, b_k, c_k


def _load_config_mapping(config: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    """Load a config mapping from a path or return the provided mapping."""
    if isinstance(config, Mapping):
        return config

    path = Path(config)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DynamicsError(f"Failed to read dynamics config: {path}") from exc
    except yaml.YAMLError as exc:
        raise DynamicsError(f"Failed to parse YAML config: {path}") from exc

    if not isinstance(data, Mapping):
        raise DynamicsError("Dynamics config top-level object must be a mapping.")

    return data


def _get_nested_mapping(data: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    """Return nested mapping at the provided key path."""
    current: Any = data

    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            joined = ".".join(keys)
            raise DynamicsError(f"Missing config section: {joined}.")
        current = current[key]

    if not isinstance(current, Mapping):
        joined = ".".join(keys)
        raise DynamicsError(f"Config section must be a mapping: {joined}.")

    return current


def _get_required_float(data: Mapping[str, Any], key: str) -> float:
    """Read a required finite float from a mapping."""
    if key not in data:
        raise DynamicsError(f"Missing required dynamics parameter: {key}.")

    return _as_finite_float(data[key], key)


def _as_finite_float(value: object, name: str) -> float:
    """Convert value to a finite float, rejecting bool."""
    if isinstance(value, bool):
        raise DynamicsError(f"{name} must be a finite number, got bool.")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DynamicsError(f"{name} must be a finite number.") from exc

    if not np.isfinite(result):
        raise DynamicsError(f"{name} must be finite.")

    return result


def _require_positive_finite(value: object, name: str) -> float:
    """Require a finite float strictly greater than zero."""
    result = _as_finite_float(value, name)
    if result <= 0.0:
        raise DynamicsError(f"{name} must be > 0.")
    return result


def _require_non_negative_finite(value: object, name: str) -> float:
    """Require a finite float greater than or equal to zero."""
    result = _as_finite_float(value, name)
    if result < 0.0:
        raise DynamicsError(f"{name} must be >= 0.")
    return result


def _as_finite_vector(value: npt.ArrayLike, *, name: str) -> FloatArray:
    """Convert value to a finite one-dimensional float64 vector."""
    array = np.asarray(value, dtype=np.float64)

    if array.ndim != 1:
        raise DynamicsError(f"{name} must be a 1D vector, got ndim={array.ndim}.")

    if not np.all(np.isfinite(array)):
        raise DynamicsError(f"{name} must contain only finite values.")

    return array.copy()


def _require_shape(array: npt.NDArray[np.float64], shape: tuple[int, ...], name: str) -> None:
    """Require an exact NumPy array shape."""
    if array.shape != shape:
        raise DynamicsError(f"{name} must have shape {shape}, got {array.shape}.")


__all__ = [
    "DynamicsError",
    "QuadrotorDynamics",
    "QuadrotorDynamicsParams",
    "continuous_dynamics",
    "discrete_step",
]
