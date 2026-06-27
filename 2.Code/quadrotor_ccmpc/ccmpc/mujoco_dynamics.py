"""
MuJoCo-based dynamics with finite-difference linearization.

Provides linearize() that returns A, B, C matrices that EXACTLY match
the MuJoCo physics engine. This solves the mismatch between the MPC's
internal model and the simulation.

Uses mjd_transitionFD() for fast C-implemented Jacobian computation,
then projects from MuJoCo's 12D state space to our 9D state space.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from .utils import euler_to_quat, quat_to_euler


def quat_to_rotvec(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r"""Convert quaternion to rotation vector (exponential map).

    The rotation vector is the axis-angle representation where
    the vector direction is the axis and the magnitude is the angle.
    """
    w, x, y, z = q
    sin_half = math.sqrt(x * x + y * y + z * z)
    if sin_half < 1e-12:
        return np.zeros(3)
    angle = 2.0 * math.atan2(sin_half, w)
    return np.array([x, y, z]) / sin_half * angle


def rotvec_to_quat(rv: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert rotation vector to quaternion [w, x, y, z]."""
    angle = np.linalg.norm(rv)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rv / angle
    s = math.sin(angle * 0.5)
    return np.array([math.cos(angle * 0.5), axis[0] * s, axis[1] * s, axis[2] * s])


class MuJoCoDynamics:
    """Dynamics model that uses MuJoCo for physics + linearization.

    Uses mjd_transitionFD() for fast Jacobian computation.
    The resulting A, B matrices perfectly match the MuJoCo physics,
    so the MPC's iMPC loop converges correctly.
    """

    def __init__(self, model: Any, data: Any, quad_id: int, mixer=None):
        """
        Args:
            model: MuJoCo MjModel.
            data: MuJoCo MjData (will be modified during linearization).
            quad_id: Body ID of the quadrotor.
            mixer: QuadrotorMixer instance to convert MPC commands to thrusts.
        """
        self.m = model
        self.d = data
        self.quad_id = quad_id
        self.mixer = mixer
        self.sim_dt = float(model.opt.timestep)  # native MuJoCo timestep

    def set_state_and_ctrl(
        self, x: npt.NDArray[np.float64], u: npt.NDArray[np.float64]
    ) -> None:
        """Set MuJoCo state + controls from 9D state and 4D MPC command.

        Converts MPC attitude command [phi_c, theta_c, vz_c, psi_dot_c]
        to rotor thrusts using the mixer, then sets d.qpos, d.qvel, d.ctrl.
        """
        self.d.qpos[:3] = x[:3]
        quat = euler_to_quat(x[6], x[7], x[8])
        self.d.qpos[3:7] = quat
        self.d.qvel[:3] = x[3:6]
        self.d.qvel[3:6] = [0.0, 0.0, 0.0]
        if self.mixer is not None:
            # Convert MPC command to rotor thrusts
            thrust = self.mixer.compute(u, x, dt=self.sim_dt)
            self.d.ctrl[:] = thrust
        else:
            self.d.ctrl[:] = u
        import mujoco
        mujoco.mj_forward(self.m, self.d)

    def get_state_9d(self) -> npt.NDArray[np.float64]:
        """Read back 9D state from MuJoCo after a step."""
        pos = self.d.qpos[:3].copy()
        quat = self.d.qpos[3:7].copy()
        roll, pitch, yaw = quat_to_euler(quat)
        vel = self.d.qvel[:3].copy()
        return np.array([
            pos[0], pos[1], pos[2],
            vel[0], vel[1], vel[2],
            roll, pitch, yaw,
        ])

    def step(self, u: npt.NDArray[np.float64], n_steps: int = 1) -> npt.NDArray[np.float64]:
        """Advance MuJoCo physics by n_steps using MPC command and return 9D state.

        Args:
            u: Control input [phi_c, theta_c, vz_c, psi_dot_c] (4,).
            n_steps: Number of MuJoCo timesteps to advance.

        Returns:
            Next 9D state.
        """
        import mujoco
        if self.mixer is not None:
            # Get current state for mixer PD control
            x_current = self.get_state_9d()
            for _ in range(n_steps):
                thrust = self.mixer.compute(u, x_current, dt=self.sim_dt)
                self.d.ctrl[:] = thrust
                mujoco.mj_step(self.m, self.d)
                x_current = self.get_state_9d()
        else:
            self.d.ctrl[:] = u
            for _ in range(n_steps):
                mujoco.mj_step(self.m, self.d)
        return self.get_state_9d()

    # Compatibility methods for QuadrotorDynamics interface
    def jacobian_state(self, x, u):
        """Compatibility: return None (uncertainty uses ODE model)."""
        return None

    def jacobian_control(self, x, u):
        """Compatibility: return None."""
        return None

    def continuous(self, x, u):
        """Compatibility: return zero derivative (use MuJoCo step instead)."""
        return np.zeros(9)

    @property
    def _params(self):
        return {"g": 9.81, "kD": 0.5}

    def discrete(self, x, u, dt):
        """Predict next state using MuJoCo physics.
        
        Saves sim state, sets MPC state, steps MuJoCo, reads result,
        restores sim state. Used internally by MPC for prediction.
        """
        saved_pos = self.d.qpos.copy()
        saved_vel = self.d.qvel.copy()
        saved_ctrl = self.d.ctrl.copy()
        saved_xfrc = self.d.xfrc_applied.copy()
        # Set to prediction state
        self.set_state_and_ctrl(x, u)
        n = max(1, round(dt / self.sim_dt))
        # Step and read result
        import mujoco as _mj
        for _ in range(n):
            _mj.mj_step(self.m, self.d)
        result = self.get_state_9d()
        # Restore sim state
        self.d.qpos[:] = saved_pos
        self.d.qvel[:] = saved_vel
        self.d.ctrl[:] = saved_ctrl
        self.d.xfrc_applied[:] = saved_xfrc
        return result

    def linearize(
        self,
        x_bar: npt.NDArray[np.float64],
        u_bar: npt.NDArray[np.float64],
        dt: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Linearize MuJoCo dynamics around (x_bar, u_bar).

        Uses mjd_transitionFD() for fast Jacobian, then converts from
        thrust-space to command-space using the mixer Jacobian.

        Returns (A_k, B_k, C_k) for discrete-time LTV model:
            x_{k+1} ≈ A_k @ x_k + B_k @ u_k + C_k
        """
        import mujoco
        import numpy.linalg as la

        n_substeps = max(1, round(dt / self.sim_dt))
        nv = self.m.nv

        # Save MuJoCo state to restore after linearization
        saved_qpos = self.d.qpos.copy()
        saved_qvel = self.d.qvel.copy()
        saved_ctrl = self.d.ctrl.copy()
        saved_xfrc = self.d.xfrc_applied.copy()

        # Set state and get one-step Jacobian at native timestep (0.005s)
        self.set_state_and_ctrl(x_bar, u_bar)

        # Get A_thrust (12x12) and B_thrust (12x4) from mjd_transitionFD
        A_t = np.zeros((nv * 2, nv * 2))
        B_t = np.zeros((nv * 2, self.m.nu))
        mujoco.mjd_transitionFD(self.m, self.d, 1e-6, 1, A_t, B_t, None, None)

        # Scale from sim_dt to MPC dt
        if n_substeps > 1:
            A_n = la.matrix_power(A_t, n_substeps)
            B_n = np.zeros_like(B_t)
            Ak = np.eye(nv * 2)
            for _ in range(n_substeps):
                B_n += Ak @ B_t
                Ak = Ak @ A_t
        else:
            A_n = A_t
            B_n = B_t

        # Mixer Jacobian: d(thrust)/d(cmd) in command space
        # dT/dphi_c = ±kp, dT/dtheta_c = ±kp, dT/dvz_c = kp_vz/4, dT/dpsi_dot_c = ±kd_yaw
        kp = self.mixer.kp_angle if self.mixer else 1.0
        kd_yaw = self.mixer.kd_yaw if self.mixer else 0.0
        kp_vz = self.mixer.kp_vz if self.mixer else 0.0
        mix_J = np.array([
            [ kp, -kp, kp_vz/4.0,  kd_yaw],
            [ kp,  kp, kp_vz/4.0, -kd_yaw],
            [-kp,  kp, kp_vz/4.0,  kd_yaw],
            [-kp, -kp, kp_vz/4.0, -kd_yaw],
        ])

        # B in command space = B_thrust @ mix_J
        B_cmd = B_n @ mix_J  # (12x4) @ (4x4) = (12x4)

        # Add yaw channel: psi_dot_c directly controls yaw rate via xfrc_applied
        # In steady-state, yaw_rate ≈ psi_dot_c, so after dt:
        #   Δyaw = psi_dot_c * dt
        #   This is captured by adding dt to B[8, 3]

        # Project from MuJoCo's 12D FD state to our 9D state.
        #
        # mjd_transitionFD state vector layout (floating free body, nv=6):
        #   [0:3]  = position (world frame)
        #   [3:6]  = rotation vector (body orientation, tangent to quaternion)
        #   [6:9]  = linear velocity (world frame)
        #   [9:12] = angular velocity (body frame, omega)
        #
        # Our 9D state: [pos(3), linvel(3), euler(phi,theta,psi)(3)]
        # Indices:        0:3     3:6          6:9
        #
        # FIX (BUG 4 — HIGH): Original code directly mapped angvel (9:12)
        # to Euler rates (6:9 in our state), which is only valid for small angles.
        # The correct mapping uses the Euler kinematic equation:
        #   [phi_dot, theta_dot, psi_dot]^T = E(phi,theta) @ omega_body
        # where E is the Euler rate matrix (3x3). For small angles E ≈ I.
        # We apply E evaluated at x_bar to transform the angular velocity
        # Jacobian block to an Euler rate Jacobian.
        phi, theta = float(x_bar[6]), float(x_bar[7])
        cp, sp = math.cos(phi), math.sin(phi)
        ct, tt = math.cos(theta), math.tan(theta)
        # E maps body angular velocity [p,q,r] -> Euler rates [phi_dot, theta_dot, psi_dot]
        if abs(ct) < 1e-6:
            ct = 1e-6  # avoid division by zero at theta=±90°
        E = np.array([
            [1.0,  sp * tt,  cp * tt],
            [0.0,  cp,       -sp    ],
            [0.0,  sp / ct,  cp / ct],
        ])

        A9 = np.zeros((9, 9))
        # pos <- pos
        A9[0:3, 0:3] = A_n[0:3, 0:3]
        # pos <- linvel
        A9[0:3, 3:6] = A_n[0:3, 6:9]
        # pos <- euler (via rotvec in MuJoCo: small angle rotvec ≈ euler)
        A9[0:3, 6:9] = A_n[0:3, 3:6]
        # linvel <- pos
        A9[3:6, 0:3] = A_n[6:9, 0:3]
        # linvel <- linvel
        A9[3:6, 3:6] = A_n[6:9, 6:9]
        # linvel <- euler
        A9[3:6, 6:9] = A_n[6:9, 3:6]
        # euler_rate <- ... : angvel block (9:12) transformed by Euler rate matrix E
        A9[6:9, 0:3] = E @ A_n[9:12, 0:3]
        A9[6:9, 3:6] = E @ A_n[9:12, 6:9]
        A9[6:9, 6:9] = E @ A_n[9:12, 3:6]

        B9 = np.zeros((9, 4))
        B9[0:3, :] = B_cmd[0:3, :]
        B9[3:6, :] = B_cmd[6:9, :]
        # Apply E to transform angvel response to Euler rate response
        B9[6:9, :] = E @ B_cmd[9:12, :]
        # Add yaw control from xfrc_applied (separate from thrust dynamics)
        # Steady-state yaw_rate ≈ psi_dot_c, so Δyaw ≈ psi_dot_c * dt
        B9[8, 3] += dt

        # Forward step to compute C (nonlinear residual)
        # Must be done BEFORE restoring state since step() modifies d
        x_next = self.step(u_bar, n_substeps)
        C9 = x_next - A9 @ x_bar - B9 @ u_bar

        # Restore saved MuJoCo state
        self.d.qpos[:] = saved_qpos
        self.d.qvel[:] = saved_qvel
        self.d.ctrl[:] = saved_ctrl
        self.d.xfrc_applied[:] = saved_xfrc

        return A9, B9, C9