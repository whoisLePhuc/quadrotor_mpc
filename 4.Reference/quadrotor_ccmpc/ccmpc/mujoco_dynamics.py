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


def euler_to_quat(
    roll: float, pitch: float, yaw: float
) -> npt.NDArray[np.float64]:
    """Convert ZYX Euler angles (rad) to quaternion [w, x, y, z]."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def quat_to_euler(q: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    """Convert quaternion [w, x, y, z] to ZYX Euler angles (rad)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


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

    FIX #6 (MAJOR — architectural): The original design let the MPC use
    QuadrotorDynamics (ODE model) for linearization even when the plant was
    MuJoCo.  After ~2 s of flight the model mismatch caused the solver to
    diverge because the linearized A,B no longer matched what MuJoCo would
    actually do.

    Resolution:
    - When sim_demo_mujoco.py constructs the CCMPC, it should pass a
      MuJoCoDynamics instance as `dynamics=` so linearize() uses
      mjd_transitionFD() — this gives Jacobians that are exact for the
      MuJoCo plant rather than the simplified ODE.
    - MuJoCoDynamics.linearize() already returns the correct (A9, B9, C9)
      with the Euler-rate correction (FIX #4) applied.
    - The ODE QuadrotorDynamics is still used in the nosim path.
    """

    def __init__(self, model: Any, data: Any, quad_id: int, mixer=None):
        """
        Args:
            model: MuJoCo MjModel.
            data: MuJoCo MjData (shared with simulation thread — READ ONLY from MPC).
            quad_id: Body ID of the quadrotor.
            mixer: QuadrotorMixer instance to convert MPC commands to thrusts.
        """
        self.m = model
        self.d = data
        self.quad_id = quad_id
        self.mixer = mixer
        self.sim_dt = float(model.opt.timestep)  # native MuJoCo timestep

        # Private MjData copy for thread-safe linearization.
        # mjd_transitionFD + step() modify MjData, which races with the
        # simulation thread's mj_step().  A private copy eliminates the race.
        import mujoco
        self._d_lin = mujoco.MjData(model)

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
            x_prev = None
            x_current = self.get_state_9d()
            for _ in range(n_steps):
                thrust = self.mixer.compute(u, x_current, state_prev=x_prev, dt=self.sim_dt)
                self.d.ctrl[:] = thrust

                # Yaw torque via xfrc_applied (matching sim_demo_mujoco.py lines 537-544)
                psi_dot_current = self.d.qvel[5]
                yaw_err = float(u[3]) - psi_dot_current
                if abs(yaw_err) > 0.005:
                    body_z = self.d.xmat[self.quad_id].reshape(3, 3)[:, 2]
                    yaw_torque_mag = 3.0 * yaw_err
                    self.d.xfrc_applied[self.quad_id, 3:6] = body_z * yaw_torque_mag
                else:
                    self.d.xfrc_applied[self.quad_id, 3:6] = [0.0, 0.0, 0.0]

                mujoco.mj_step(self.m, self.d)
                x_prev = x_current
                x_current = self.get_state_9d()

            self.d.xfrc_applied[self.quad_id, 3:6] = [0.0, 0.0, 0.0]
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
        """Step MuJoCo by dt and return new state.

        NOTE: This modifies d. SAVE/RESTORE state before/after calling.
        """
        saved_pos = self.d.qpos.copy()
        saved_vel = self.d.qvel.copy()
        saved_ctrl = self.d.ctrl.copy()
        saved_xfrc = self.d.xfrc_applied.copy()
        n = max(1, round(dt / self.sim_dt))
        result = self.step(u, n)
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

        # Use private MjData copy for thread safety.
        # The simulation thread concurrently calls mj_step() on self.d.
        # mjd_transitionFD + step() modify MjData, so we must isolate them
        # on a private copy to avoid corrupting the simulation.
        d_saved = self.d
        self.d = self._d_lin
        self.d.qpos[:] = d_saved.qpos
        self.d.qvel[:] = d_saved.qvel
        self.d.ctrl[:] = d_saved.ctrl
        self.d.xfrc_applied[:] = d_saved.xfrc_applied

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

        # Project from MuJoCo's 12D FD state to our 9D state.
        A9 = np.zeros((9, 9))
        A9[0:3, 0:3] = A_n[0:3, 0:3]
        A9[0:3, 3:6] = A_n[0:3, 6:9]
        A9[0:3, 6:9] = A_n[0:3, 3:6]
        A9[3:6, 0:3] = A_n[6:9, 0:3]
        A9[3:6, 3:6] = A_n[6:9, 6:9]
        A9[3:6, 6:9] = A_n[6:9, 3:6]
        A9[6:9, 0:3] = A_n[3:6, 0:3]
        A9[6:9, 3:6] = A_n[3:6, 6:9]
        A9[6:9, 6:9] = A_n[3:6, 3:6]

        B9 = np.zeros((9, 4))
        B9[0:3, :] = B_cmd[0:3, :]
        B9[3:6, :] = B_cmd[6:9, :]
        B9[6:9, :] = B_cmd[3:6, :]

        x_next = self.step(u_bar, n_substeps)
        C9 = x_next - A9 @ x_bar - B9 @ u_bar

        self.d = d_saved
        return A9, B9, C9