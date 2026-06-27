"""
Quadrotor dynamics model.

Formulas from:
  "Robust Vision-based Obstacle Avoidance for Micro Aerial Vehicles
   in Dynamic Environments" — Lin, Zhu, Alonso-Mora, ICRA 2020

State:  9D  [x, y, z, vx, vy, vz, phi, theta, psi]
Control: 4D [phi_c, theta_c, vz_c, psi_dot_c]
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import yaml


# ---------------------------------------------------------------------------
# Continuous dynamics (Eq 8 + Attitude)
# ---------------------------------------------------------------------------
def _body_tilt_factor(
    phi: float, theta: float
) -> tuple[float, float]:
    r"""Compute the tilt factors :math:`F_\theta, F_\varphi`.

    .. math::

        A &= \sqrt{1 + \tan^2\theta + \tan^2\phi} \\
        F_\theta &= \frac{\tan\theta}{\cos\theta \cdot A} \\
        F_\varphi &= \frac{\tan\phi}{\cos\phi \cdot A}
    """
    t_phi = math.tan(phi)
    t_theta = math.tan(theta)
    A_tilt = math.sqrt(1.0 + t_theta * t_theta + t_phi * t_phi)
    if A_tilt < 1e-12:
        return 0.0, 0.0
    c_theta = math.cos(theta)
    c_phi = math.cos(phi)
    F_theta = t_theta / (c_theta * A_tilt) if abs(c_theta) > 1e-12 else 0.0
    F_phi = t_phi / (c_phi * A_tilt) if abs(c_phi) > 1e-12 else 0.0
    return F_theta, F_phi


def continuous_dynamics(
    x: npt.NDArray[np.float64],
    u: npt.NDArray[np.float64],
    g: float = 9.81,
    kD: float = 0.5,
    k_phi: float = 1.0,
    k_theta: float = 1.0,
    k_vz: float = 1.0,
    tau_phi: float = 0.2,
    tau_theta: float = 0.2,
    tau_vz: float = 0.4,
) -> npt.NDArray[np.float64]:
    r"""Compute :math:`\dot{x} = f(x, u)`.

    Args:
        x: State vector (9,).
        u: Control vector (4,).
        g, kD, k_phi, k_theta, k_vz: Model parameters.
        tau_phi, tau_theta, tau_vz: Time constants.

    Returns:
        State derivative (9,).
    """
    phi = x[6]
    theta = x[7]
    psi = x[8]

    phi_c = u[0]
    theta_c = u[1]
    vz_c = u[2]
    psi_dot_c = u[3]

    # Position derivatives
    dx = x[3:6]  # [vx, vy, vz]

    # Tilt factor
    F_theta, F_phi = _body_tilt_factor(phi, theta)
    c_psi = math.cos(psi)
    s_psi = math.sin(psi)

    # Velocity derivatives (Eq 8 — first-order low-pass Euler)
    dvx = g * F_theta * c_psi - g * F_phi * s_psi - kD * x[3]
    dvy = g * F_theta * s_psi + g * F_phi * c_psi - kD * x[4]
    dvz = (k_vz * vz_c - x[5]) / tau_vz

    # Attitude derivatives (first-order lags)
    dphi = (k_phi * phi_c - phi) / tau_phi
    dtheta = (k_theta * theta_c - theta) / tau_theta
    dpsi = psi_dot_c

    return np.array(
        [dx[0], dx[1], dx[2], dvx, dvy, dvz, dphi, dtheta, dpsi]
    )


# ---------------------------------------------------------------------------
# Discrete integration (RK4)
# ---------------------------------------------------------------------------
def discrete_step(
    x: npt.NDArray[np.float64],
    u: npt.NDArray[np.float64],
    dt: float,
    **params,
) -> npt.NDArray[np.float64]:
    """Integrate dynamics one timestep using RK4.

    Args:
        x: Current state (9,).
        u: Control input (4,).
        dt: Timestep (s).
        **params: Model parameters passed to continuous_dynamics.

    Returns:
        Next state (9,).
    """
    def f(xk):
        return continuous_dynamics(xk, u, **params)

    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------------------------------------------------
# Numerical Jacobian (finite differences)
# ---------------------------------------------------------------------------
def _finite_diff_jacobian(
    f, x0: npt.NDArray[np.float64],
    eps: float = 1e-6,
) -> npt.NDArray[np.float64]:
    """Compute Jacobian of f at x0 via central finite differences.

    Args:
        f: Function R^m -> R^n.
        x0: Point at which to evaluate (m,).
        eps: Perturbation size.

    Returns:
        Jacobian matrix (n, m).
    """
    n = len(f(x0))
    m = len(x0)
    J = np.zeros((n, m))
    for i in range(m):
        dx = np.zeros(m)
        dx[i] = eps
        J[:, i] = (f(x0 + dx) - f(x0 - dx)) / (2.0 * eps)
    return J


# ---------------------------------------------------------------------------
# QuadrotorDynamics class
# ---------------------------------------------------------------------------
class QuadrotorDynamics:
    """Quadrotor dynamics model with linearization support."""

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
    ):
        self.g = g
        self.kD = kD
        self.k_phi = k_phi
        self.k_theta = k_theta
        self.k_vz = k_vz
        self.tau_phi = tau_phi
        self.tau_theta = tau_theta
        self.tau_vz = tau_vz

    @property
    def _params(self) -> dict:
        return {
            "g": self.g,
            "kD": self.kD,
            "k_phi": self.k_phi,
            "k_theta": self.k_theta,
            "k_vz": self.k_vz,
            "tau_phi": self.tau_phi,
            "tau_theta": self.tau_theta,
            "tau_vz": self.tau_vz,
        }

    @classmethod
    def from_config(cls, config: str | dict) -> "QuadrotorDynamics":
        """Create from YAML config path or dict."""
        if isinstance(config, str):
            with open(config) as f:
                config_data = yaml.safe_load(f)
        else:
            config_data = config
        v = config_data["model"]["quadrotor"]
        return cls(
            g=v["g"],
            kD=v["kD"],
            k_phi=v["k_phi"],
            k_theta=v["k_theta"],
            k_vz=v["k_vz"],
            tau_phi=v["tau_phi"],
            tau_theta=v["tau_theta"],
            tau_vz=v["tau_vz"],
        )

    def continuous(self, x: npt.NDArray[np.float64], u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Continuous dynamics f(x, u)."""
        return continuous_dynamics(x, u, **self._params)

    def discrete(self, x: npt.NDArray[np.float64], u: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
        """Discrete-time dynamics (RK4)."""
        return discrete_step(x, u, dt, **self._params)

    def jacobian_state(
        self, x: npt.NDArray[np.float64], u: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Jacobian of continuous dynamics WRT state: :math:`\partial f / \partial x`.

        Returns (9x9) matrix.
        """
        def f(xk):
            return self.continuous(xk, u)
        return _finite_diff_jacobian(f, x)

    def jacobian_control(
        self, x: npt.NDArray[np.float64], u: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        r"""Jacobian of continuous dynamics WRT control: :math:`\partial f / \partial u`.

        Returns (9x4) matrix.
        """
        # Simple analytical Jacobian for control since dynamics are linear in u
        J = np.zeros((9, 4))
        J[6, 0] = self.k_phi / self.tau_phi    # d(dphi)/d(phi_c)
        J[7, 1] = self.k_theta / self.tau_theta # d(dtheta)/d(theta_c)
        J[5, 2] = self.k_vz / self.tau_vz       # d(dvz)/d(vz_c)
        J[8, 3] = 1.0                            # d(dpsi)/d(psi_dot_c)
        return J

    def linearize(
        self,
        x_bar: npt.NDArray[np.float64],
        u_bar: npt.NDArray[np.float64],
        dt: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""Linearize dynamics around (x_bar, u_bar) for discrete-time LTV model.

        Uses second-order Taylor expansion for the B matrix to capture
        cascade coupling (e.g. theta_c → theta → vx within one step).
        Without this, the linearized model at hover predicts zero velocity
        change from any attitude command, causing the MPC to be overly
        aggressive and then oscillate.

        Returns (A_k, B_k, C_k) such that:

        .. math::

            x_{k+1} \approx A_k \, x_k + B_k \, u_k + C_k

        where:
            A_k = I + dt * \partial f / \partial x  (first-order)
            B_k = dt * \partial f / \partial u + (dt^2/2) * A_cont * B_cont
            C_k compensates for the expansion point
        """
        A_cont = self.jacobian_state(x_bar, u_bar)
        B_cont = self.jacobian_control(x_bar, u_bar)
        f_xu = self.continuous(x_bar, u_bar)

        # First-order A (adequate for state dynamics over dt=0.06)
        A_k = np.eye(9) + dt * A_cont

        # Second-order B: captures cascade coupling like theta_c → vx
        # Without this term, at hover (theta=0, vx=0) the model says:
        #   vx_{k+1} = vx_k + dt * (g*theta_k - kD*vx_k)
        #   = 0 + 0.06 * (0 - 0) = 0   ← WRONG, ignores theta rise during step
        # The A_cont*B_cont*dt²/2 term adds this coupling:
        #   vx_{k+1} += dt²/2 * d(dvx)/dtheta * d(dtheta)/dtheta_c * theta_c
        B_k = dt * B_cont + 0.5 * dt * dt * (A_cont @ B_cont)

        # C_k: the affine offset that makes the linearized model exactly match
        # the actual discrete rollout at the expansion point (x_bar, u_bar).
        #
        # FIX (BUG 1 — CRITICAL): The original code computed C_k algebraically
        # from Taylor residuals, which does NOT equal x_next - A_k@x_bar - B_k@u_bar
        # (numerical test shows up to 7 mm/step error that accumulates over horizon).
        # The correct definition is simply:
        #   C_k = x_{k+1}^{true} - A_k @ x_bar - B_k @ u_bar
        # so that the linear model perfectly predicts x_bar -> x_next.
        x_next = discrete_step(x_bar, u_bar, dt, **self._params)
        C_k = x_next - A_k @ x_bar - B_k @ u_bar

        return A_k, B_k, C_k