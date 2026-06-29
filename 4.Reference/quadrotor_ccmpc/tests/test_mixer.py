"""Tests for QuadrotorMixer: checks thrust distribution correctness."""

import numpy as np
from ccmpc.mixer import QuadrotorMixer


# ============================================================================
# QuadrotorMixer — CRITICAL: previously buggy (pitch torque was zero)
# ============================================================================

class TestQuadrotorMixer:
    def test_hover_thrust(self):
        """At hover with zero command, all rotors should have equal thrust."""
        mixer = QuadrotorMixer(mass=1.0)
        state = np.zeros(9)
        state[2] = 1.0
        T = mixer.compute(np.zeros(4), state)
        assert len(T) == 4
        # All four rotors should be close to hover thrust
        assert np.allclose(T, mixer.hover_thrust, atol=1e-6)

    def test_pitch_forward_creates_pitch_torque(self):
        """CRITICAL: The mixer must create NON-ZERO pitch torque.
        
        With theta_c > 0 (pitch forward command), the rotor thrust
        distribution must produce a net pitch torque.
        Before the fix, T1 and T4 had no pitch term, and T2/T3
        had opposite signs that cancelled → pitch torque = 0.
        
        MuJoCo torque calculation (x-forward, y-left, z-up):
          pitch_torque ∝ -T1 + T2 + T3 - T4
        """
        mixer = QuadrotorMixer(mass=1.0)
        state = np.zeros(9)
        state[2] = 1.0
        cmd = np.array([0.0, 0.35, 0.0, 0.0])
        T = mixer.compute(cmd, state)
        # pitch_torque ∝ -T1 + T2 + T3 - T4
        pitch_torque = -T[0] + T[1] + T[2] - T[3]
        assert pitch_torque > 0.01, (
            f"Pitch torque is {pitch_torque:.4f} — should be > 0. "
            "The mixer produces no net pitch torque!"
        )

    def test_roll_creates_roll_torque(self):
        """With phi_c > 0, rotor distribution must produce roll torque.
        
        MuJoCo roll torque (x-forward, y-left, z-up):
          roll_torque ∝ T1 + T2 - T3 - T4
        """
        mixer = QuadrotorMixer(mass=1.0)
        state = np.zeros(9)
        state[2] = 1.0
        cmd = np.array([0.35, 0.0, 0.0, 0.0])
        T = mixer.compute(cmd, state)
        roll_torque = T[0] + T[1] - T[2] - T[3]
        assert abs(roll_torque) > 0.01, (
            f"Roll torque is {roll_torque:.4f} — should be non-zero."
        )

    def test_pitch_and_roll_decoupled(self):
        """Simultaneous pitch and roll should produce both torques."""
        mixer = QuadrotorMixer(mass=1.0)
        state = np.zeros(9)
        state[2] = 1.0
        cmd = np.array([0.25, 0.25, 0.0, 0.0])
        T = mixer.compute(cmd, state)
        pitch_torque = -T[0] + T[1] + T[2] - T[3]
        roll_torque  =  T[0] + T[1] - T[2] - T[3]
        assert abs(pitch_torque) > 0.01
        assert abs(roll_torque) > 0.01

    def test_thrust_symmetry_pure_pitch(self):
        """With pure pitch, front and back thrust sums should differ.
        
        Front rotors: T1, T4 (positive x in MuJoCo body frame)
        Rear rotors:  T2, T3 (negative x)
        For forward pitch: front thrust sum < rear thrust sum
        """
        mixer = QuadrotorMixer(mass=1.0)
        state = np.zeros(9)
        state[2] = 1.0
        cmd = np.array([0.0, 0.35, 0.0, 0.0])
        T = mixer.compute(cmd, state)
        front_sum = T[0] + T[3]  # front rotors (positive x)
        rear_sum  = T[1] + T[2]  # rear rotors (negative x)
        assert rear_sum > front_sum, (
            f"Rear thrust {rear_sum:.4f} should exceed front thrust "
            f"{front_sum:.4f} for forward pitch."
        )

    def test_thrust_symmetry_pure_roll(self):
        """With pure roll, left and right thrust sums should differ.
        
        Left rotors:  T1, T2 (positive y in MuJoCo)
        Right rotors: T3, T4 (negative y)
        For positive roll: left thrust sum < right thrust sum
        (Or vice versa — just verify they're different)
        """
        mixer = QuadrotorMixer(mass=1.0)
        state = np.zeros(9)
        state[2] = 1.0
        cmd = np.array([0.35, 0.0, 0.0, 0.0])
        T = mixer.compute(cmd, state)
        left_sum  = T[0] + T[1]  # left rotors (positive y)
        right_sum = T[2] + T[3]  # right rotors (negative y)
        assert not np.allclose(left_sum, right_sum, atol=0.01)

    def test_clip_to_max_thrust(self):
        """Thrust should be clipped to max_thrust."""
        mixer = QuadrotorMixer(mass=1.0, max_thrust=10.0)
        state = np.zeros(9)
        state[2] = 1.0
        # Very aggressive command
        cmd = np.array([0.5, 0.5, 5.0, 5.0])
        T = mixer.compute(cmd, state)
        assert np.all(T <= 10.0 + 1e-6)
        assert np.all(T >= 0.0 - 1e-6)

    def test_rate_damping(self):
        """When state_prev is provided, rate damping should reduce commands."""
        mixer = QuadrotorMixer(mass=1.0)
        state = np.array([0, 0, 1, 0, 0, 0, 0.3, 0, 0])  # already has roll
        cmd = np.array([0.0, 0.0, 0.0, 0.0])  # no command → reduce angle
        T_no_rate = mixer.compute(cmd, state, state_prev=None)
        T_with_rate = mixer.compute(cmd, state, state - np.array([0,0,0,0,0,0,0.1,0,0]), dt=0.005)
        # Rate damping should affect the output
        assert not np.allclose(T_no_rate, T_with_rate)

    def test_hover_thrust_by_mass(self):
        """Hover thrust should equal mass * g / 4."""
        mixer = QuadrotorMixer(mass=1.5)
        assert abs(mixer.hover_thrust - 1.5 * 9.81 / 4) < 1e-6
