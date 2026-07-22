r"""
Quadrotor mixer: maps CC-MPC attitude commands to individual rotor thrusts.

The CC-MPC outputs [phi_c, theta_c, vz_c, psi_dot_c] which are attitude
and vertical velocity commands. The mixer converts these to 4 rotor
thrusts [T1, T2, T3, T4] using PD control for attitude tracking.

Rotor layout (X-configuration, viewed from top, NED body frame):
         Front (+x)
  rotor4 (FL) ---- rotor1 (FR)
                        /
            [body]     /
       rotor3 (BL) -- rotor2 (BR)
         Rear (-x)

Spin directions (to produce counter-torques):
  rotor1 FR: CCW (+z torque in body)  # noqa
  rotor2 BR: CW  (-z torque in body)
  rotor3 BL: CCW (+z torque in body)   ← FIX #17: was described as CW
  rotor4 FL: CW  (-z torque in body)   ← FIX #17: was described as CCW

Moment arms (equal length L from CoM to each rotor):
  Roll  (+phi): FR and BR produce +roll (right side), FL and BL produce -roll
  Pitch (+theta): FR and FL produce +pitch (front side), BR and BL produce -pitch
  FIX #17 (LOW): original comment had signs reversed for pitch arm.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Arm length from quadrotor center to rotor (meters)
ARM_LENGTH = 0.311  # sqrt(0.22^2 + 0.22^2)


class QuadrotorMixer:
    """Converts CC-MPC commands to rotor thrusts for MuJoCo physics."""

    def __init__(
        self,
        mass: float = 1.0,
        arm_length: float = ARM_LENGTH,
        max_thrust: float = 12.0,
        kp_angle: float = 0.7,
        kd_angle: float = 0.3,
        kp_vz: float = 8.0,
        kd_yaw: float = 0.5,
    ):
        self.mass = mass
        self.arm = arm_length
        self.max_thrust = max_thrust
        self.kp_angle = kp_angle
        self.kd_angle = kd_angle
        self.kp_vz = kp_vz
        self.kd_yaw = kd_yaw
        self.hover_thrust = mass * 9.81 / 4.0  # per rotor

    def compute(
        self,
        cmd: npt.NDArray[np.float64],
        state: npt.NDArray[np.float64],
        state_prev: npt.NDArray[np.float64] | None = None,
        dt: float = 0.005,
    ) -> npt.NDArray[np.float64]:
        r"""Convert CC-MPC command to rotor thrusts.

        Args:
            cmd: CC-MPC control [phi_c, theta_c, vz_c, psi_dot_c] (4,).
            state: Current MuJoCo state [x,y,z, vx,vy,vz, phi,theta,psi] (9,).
            state_prev: Previous state for rate estimation (9,) or None.
            dt: Timestep for rate estimation.

        Returns:
            Rotor thrusts [T1, T2, T3, T4] clipped to [0, max_thrust].
        """
        # Extract current angles and rates
        phi = state[6]
        theta = state[7]
        psi = state[8]
        vz = state[5]

        if state_prev is not None:
            phi_dot = (phi - state_prev[6]) / max(dt, 1e-6)
            theta_dot = (theta - state_prev[7]) / max(dt, 1e-6)
            psi_dot = (psi - state_prev[8]) / max(dt, 1e-6)
        else:
            phi_dot = 0.0
            theta_dot = 0.0
            psi_dot = 0.0

        # PD control for attitude tracking
        roll_cmd = self.kp_angle * (cmd[0] - phi) - self.kd_angle * phi_dot
        pitch_cmd = self.kp_angle * (cmd[1] - theta) - self.kd_angle * theta_dot
        yaw_cmd = self.kd_yaw * (cmd[3] - psi_dot)

        # Vertical velocity control
        thrust_offset = self.kp_vz * (cmd[2] - vz)

        # Base thrust with TILT COMPENSATION.
        # Without compensation, tilted rotors provide less upward force:
        #   F_z = sum(T) * cos(phi) * cos(theta)
        # To maintain altitude while tilted, base thrust must be increased:
        #   T_base_compensated = hover_thrust / (cos(phi) * cos(theta))
        # This feedforward term ensures the drone can hover at any attitude,
        # preventing altitude loss during horizontal flight.
        import math as _math
        tilt_comp = 1.0 / max(_math.cos(phi) * _math.cos(theta), 0.5)
        tilt_comp = min(tilt_comp, 2.0)   # clamp at 60°
        T_base = self.hover_thrust * tilt_comp + thrust_offset / 4.0

        # X-config mixing: FR(1), BR(2), BL(3), FL(4) with CCW=1,3 / CW=2,4
        tau_roll  = roll_cmd
        tau_pitch = pitch_cmd
        tau_yaw   = yaw_cmd

        T = np.array([
            T_base + tau_roll - tau_pitch + tau_yaw,   # rotor1 FR (CCW)
            T_base + tau_roll + tau_pitch - tau_yaw,   # rotor2 BR (CW)
            T_base - tau_roll + tau_pitch + tau_yaw,   # rotor3 BL (CCW)
            T_base - tau_roll - tau_pitch - tau_yaw,   # rotor4 FL (CW)
        ])

        # MuJoCo pure z-force actuators can't generate yaw torque directly.
        # Yaw control is handled externally via xfrc_applied in the step function.
        return np.clip(T, 0.0, self.max_thrust)
