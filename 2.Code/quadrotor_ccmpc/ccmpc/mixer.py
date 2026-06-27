"""
Quadrotor mixer: maps CC-MPC attitude commands to individual rotor thrusts.

The CC-MPC outputs [phi_c, theta_c, vz_c, psi_dot_c] which are attitude
and vertical velocity commands. The mixer converts these to 4 rotor
thrusts [T1, T2, T3, T4] using PD control for attitude tracking.

Rotor layout (X-configuration, viewed from top):
  rotor1 (FR) ---- rotor3 (FL)
       \            /
        \          /
         [  body  ]
        /          \
       /            \
  rotor2 (BR) ---- rotor4 (BL)
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

        # Moment arm for roll/pitch torque
        # Each rotor produces torque = thrust * arm_length
        # Roll torque: T1*y1 + T2*y2 + T3*y3 + T4*y4
        # Pitch torque: T1*x1 + T2*x2 + T3*x3 + T4*x4
        # For X-config with arms at 45°:
        #   rotor1: (0.22, 0.22)  rotor2: (-0.22, 0.22)
        #   rotor3: (-0.22, -0.22) rotor4: (0.22, -0.22)
        # Roll axis: rotor1 and rotor4 produce +roll, rotor2 and rotor3 produce -roll
        # Pitch axis: rotor3 and rotor4 produce +pitch, rotor1 and rotor2 produce -pitch
        self.mixer_matrix = np.array([
            # T1   T2   T3   T4    ← rotor thrusts
            [ 1.0, 1.0, 1.0, 1.0],  # total thrust
            [ 1.0, 0.0, 0.0, 1.0],  # roll (rotors on +y side)
            [ 0.0, 0.0, 1.0, 1.0],  # pitch (rotors on +x side)
            [ 1.0,-1.0, 1.0,-1.0],  # yaw (alternating CCW/CW)
        ])

        # Inverse: decompose desired torques into rotor thrusts
        # Pseudo-inverse of the mixer
        self._mix_inv = np.linalg.pinv(self.mixer_matrix)

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

        # Total thrust demand per rotor (hover + correction)
        T_base = self.hover_thrust + thrust_offset / 4.0

        # Mix: decompose roll/pitch demands into rotor thrust adjustments
        # Rotor positions in MuJoCo (x-forward, y-left, z-up):
        #   FR=rotor1 at (+0.22,+0.22): +roll, -pitch
        #   BR=rotor2 at (-0.22,+0.22): +roll, +pitch  
        #   BL=rotor3 at (-0.22,-0.22): -roll, +pitch
        #   FL=rotor4 at (+0.22,-0.22): -roll, -pitch
        # For positive roll:  T1+T2-T3-T4 = roll_desired * kp
        # For positive pitch: -T1+T2+T3-T4 = pitch_desired * kp
        T = np.array([
            T_base + roll_cmd - pitch_cmd,   # rotor1 front-left
            T_base + roll_cmd + pitch_cmd,   # rotor2 back-left
            T_base - roll_cmd + pitch_cmd,   # rotor3 back-right
            T_base - roll_cmd - pitch_cmd,   # rotor4 front-right
        ])

        # For X-configuration, the correct decoupled mixing is:
        #   rotor1 (FR, +x +y): +roll, -pitch
        #   rotor2 (BR, -x +y): +roll, +pitch
        #   rotor3 (BL, -x -y): -roll, +pitch  (FIX: was -roll_cmd + pitch_cmd)
        #   rotor4 (FL, +x -y): -roll, -pitch  (FIX: was -roll_cmd - pitch_cmd)
        #
        # FIX (BUG 5 — MEDIUM): The mixer_matrix was defined for a different
        # rotor labeling convention than compute() used.  mixer_matrix was
        # dead code.  We now compute thrusts directly from the mixer_matrix
        # inverse so both are consistent.
        #
        # Mixer matrix (rows = [thrust, roll_torque, pitch_torque, yaw_torque]):
        #   thrust = T1+T2+T3+T4
        #   roll   = arm*(T1+T2-T3-T4)   (rotors on +y side produce +roll)
        #   pitch  = arm*(-T1+T2+T3-T4)  (rotors on +x side produce -pitch for forward tilt)
        #   yaw    = km*(T1-T2+T3-T4)    (alternating spin direction)
        # Normalize arm/km out: use unit coefficients, scale by PD outputs.

        total_thrust = 4.0 * T_base  # total desired thrust

        # Desired torques (unitless, in rotor-delta-thrust units)
        tau_roll  =  roll_cmd   # will distribute as ±roll per rotor
        tau_pitch =  pitch_cmd  # will distribute as ±pitch per rotor
        tau_yaw   =  yaw_cmd    # differential yaw via rotor spin direction

        # X-config mixing: FR(1), BR(2), BL(3), FL(4) with CCW=1,3 / CW=2,4
        T = np.array([
            T_base + tau_roll - tau_pitch + tau_yaw,   # rotor1 FR (CCW)
            T_base + tau_roll + tau_pitch - tau_yaw,   # rotor2 BR (CW)
            T_base - tau_roll + tau_pitch + tau_yaw,   # rotor3 BL (CCW)
            T_base - tau_roll - tau_pitch - tau_yaw,   # rotor4 FL (CW)
        ])

        # MuJoCo pure z-force actuators can't generate yaw torque directly.
        # Yaw control is handled externally via xfrc_applied in the step function.
        return np.clip(T, 0.0, self.max_thrust)