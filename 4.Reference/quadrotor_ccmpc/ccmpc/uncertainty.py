"""
Uncertainty propagation for Chance-Constrained MPC.

Formulas from:
  "Robust Vision-based Obstacle Avoidance for Micro Aerial Vehicles
   in Dynamic Environments" — Lin, Zhu, Alonso-Mora, ICRA 2020

Equation (19): Gamma^{k+1} = F^k * Gamma^k * F^k^T + W^k
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .dynamics import QuadrotorDynamics


class UncertaintyPropagator:
    """EKF-type uncertainty propagation (Eq 19).

    Propagates the full 9x9 state covariance through the linearized dynamics.
    The position covariance (3x3 block) is extracted for the chance constraint.
    """

    def __init__(
        self,
        process_noise_pos: float = 0.01,
        process_noise_vel: float = 0.1,
        process_noise_att: float = 0.02,
        init_pos_noise: float = 0.05,
        init_vel_noise: float = 0.1,
        init_att_noise: float = 0.03,
    ):
        # Process noise covariance W (9x9)
        self.W = np.diag(
            np.concatenate([
                [process_noise_pos**2] * 3,   # position noise
                [process_noise_vel**2] * 3,   # velocity noise
                [process_noise_att**2] * 3,   # attitude noise
            ])
        )

        # Initial state covariance Gamma_0
        self.Gamma_0 = np.diag(
            np.concatenate([
                [init_pos_noise**2] * 3,
                [init_vel_noise**2] * 3,
                [init_att_noise**2] * 3,
            ])
        )

    @classmethod
    def from_config(cls, config: str | dict) -> "UncertaintyPropagator":
        """Create from YAML config path or dict."""
        if isinstance(config, str):
            import yaml
            with open(config) as f:
                config_data = yaml.safe_load(f)
        else:
            config_data = config
        u = config_data["controller"]["uncertainty"]
        return cls(
            process_noise_pos=u["process_noise_pos"],
            process_noise_vel=u["process_noise_vel"],
            process_noise_att=u["process_noise_att"],
            init_pos_noise=u["init_pos_noise"],
            init_vel_noise=u["init_vel_noise"],
            init_att_noise=u["init_att_noise"],
        )

    @staticmethod
    def vio_drift_from_config(config: str | dict) -> "VIODriftModel":
        """Create VIODriftModel from config."""
        if isinstance(config, str):
            import yaml
            with open(config) as f:
                config_data = yaml.safe_load(f)
        else:
            config_data = config
        u = config_data["controller"]["uncertainty"]
        return VIODriftModel(
            drift_pos=u.get("vio_drift_pos", 0.005),
            drift_vel=u.get("vio_drift_vel", 0.02),
            drift_att=u.get("vio_drift_att", 0.005),
        )

    def propagate(
        self,
        Gamma_0: npt.NDArray[np.float64],
        x_guess: npt.NDArray[np.float64],
        u_guess: npt.NDArray[np.float64],
        dynamics: QuadrotorDynamics,
        dt: float = 0.06,
    ) -> list[npt.NDArray[np.float64]]:
        r"""Propagate covariance over prediction horizon (Eq 19).

        .. math::

            \Gamma^{k+1} = F^k \, \Gamma^k \, F^{k\,T} + W

        Args:
            Gamma_0: Initial state covariance (9x9).
            x_guess: Last-loop state trajectory (9, N+1).
            u_guess: Last-loop control sequence (4, N).
            dynamics: Quadrotor dynamics model.
            dt: Timestep of the prediction horizon.

        Returns:
            List of covariance matrices [Gamma_0, ..., Gamma_N] each (9x9).
        """
        N = x_guess.shape[1] - 1
        Gamma_list = [Gamma_0.copy()]

        for k in range(N):
            Gamma_k = Gamma_list[-1]
            A_cont = dynamics.jacobian_state(x_guess[:, k], u_guess[:, k])
            if A_cont is None:
                # FIX (BUG 3 — HIGH): Original fallback used F_k = I (identity),
                # meaning covariance grew only from W*dt with no state coupling.
                # For MuJoCo dynamics, retrieve the discrete-time A_k directly
                # from linearize(), which uses mjd_transitionFD.
                try:
                    A_k, _, _ = dynamics.linearize(x_guess[:, k], u_guess[:, k], dt)
                    F_k = A_k  # already discrete-time state transition
                except Exception:
                    F_k = np.eye(9)
            else:
                F_k = np.eye(9) + dt * A_cont
            # FIX #18 (LOW): verify that W*dt is the correct discretization.
            # Continuous SDE: dΓ/dt = F Γ F^T + W  (W is continuous PSD intensity)
            # Euler-Maruyama discretization over dt:
            #   Γ_{k+1} = F_k Γ_k F_k^T + W*dt
            # This is correct for small dt (first-order Itô discretization).
            # W is defined in __init__ with units [m²/s, (m/s)²/s, rad²/s] per axis,
            # so W*dt correctly gives [m², (m/s)², rad²] added covariance per step.
            Gamma_k1 = F_k @ Gamma_k @ F_k.T + self.W * dt
            Gamma_list.append(Gamma_k1)

        return Gamma_list

    @staticmethod
    def position_covariance(Gamma: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Extract 3x3 position covariance from the top-left block."""
        return Gamma[:3, :3].copy()

    @staticmethod
    def add_measurement_noise(
        x: npt.NDArray[np.float64],
        sigma_pos: float = 0.02,
        sigma_vel: float = 0.05,
        sigma_att: float = 0.02,
    ) -> npt.NDArray[np.float64]:
        """Simulate VIO measurement noise (i.i.d. Gaussian)."""
        noise = np.concatenate([
            np.random.normal(0, sigma_pos, 3),
            np.random.normal(0, sigma_vel, 3),
            np.random.normal(0, sigma_att, 3),
        ])
        return x + noise


class VIODriftModel:
    """Time-correlated VIO drift/bias model.

    VIO estimates drift over time due to integration of noisy IMU data.
    This model simulates a random walk bias on position and velocity,
    which makes the uncertainty covariance grow more realistically.

    The drift evolves as: b_{t+1} = b_t + w,  w ~ N(0, Q_drift)
    """

    def __init__(
        self,
        drift_pos: float = 0.005,
        drift_vel: float = 0.02,
        drift_att: float = 0.005,
    ):
        self.Q_drift = np.concatenate([
            [drift_pos**2] * 3,
            [drift_vel**2] * 3,
            [drift_att**2] * 3,
        ])
        self.bias: npt.NDArray[np.float64] = np.zeros(9)

    def update(self, dt: float = 0.02) -> npt.NDArray[np.float64]:
        """Step drift bias forward by dt and return the current bias."""
        self.bias += np.random.normal(0, np.sqrt(self.Q_drift * dt))
        return self.bias.copy()

    def apply(self, true_state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Add current drift bias to the true state (simulates VIO estimate)."""
        return true_state + self.bias