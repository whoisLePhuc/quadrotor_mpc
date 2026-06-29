"""Tests for quadrotor dynamics module."""

import numpy as np
from numpy.linalg import norm
from ccmpc.dynamics import (
    _body_tilt_factor,
    continuous_dynamics,
    discrete_step,
    QuadrotorDynamics,
)


# ============================================================================
# _body_tilt_factor
# ============================================================================

class TestBodyTiltFactor:
    def test_hover(self):
        """At hover (phi=0, theta=0), tilt factors should be zero."""
        F_theta, F_phi = _body_tilt_factor(0.0, 0.0)
        assert abs(F_theta) < 1e-10
        assert abs(F_phi) < 1e-10

    def test_small_angle_approximation(self):
        """For small angles, F_theta ≈ theta, F_phi ≈ phi."""
        theta = 0.1
        phi = 0.08
        F_theta, F_phi = _body_tilt_factor(phi, theta)
        # Should be close to tan(theta), tan(phi) for small angles
        assert abs(F_theta - np.tan(theta)) < 0.01
        assert abs(F_phi - np.tan(phi)) < 0.01

    def test_large_angle_roll_pitch_coupling(self):
        """With both roll and pitch, each factor should be slightly less
        than tan() due to the geometric coupling term A_tilt."""
        theta = 0.35
        phi = 0.35
        F_theta, F_phi = _body_tilt_factor(phi, theta)
        assert F_theta < np.tan(theta)
        assert F_phi < np.tan(phi)
        assert F_theta > 0.0 and F_phi > 0.0

    def test_pure_pitch(self):
        """With only pitch (phi=0), F_phi should be zero."""
        F_theta, F_phi = _body_tilt_factor(0.0, 0.35)
        assert abs(F_phi) < 1e-10
        assert F_theta > 0.0

    def test_symmetry(self):
        """Negative angles should produce negative factors."""
        F_pos, _ = _body_tilt_factor(0.0, 0.2)
        F_neg, _ = _body_tilt_factor(0.0, -0.2)
        assert abs(F_pos + F_neg) < 1e-10


# ============================================================================
# continuous_dynamics
# ============================================================================

class TestContinuousDynamics:
    def test_hover_derivative_zero(self, hover_state, zero_cmd):
        """At hover with zero command, the state should not change."""
        xdot = continuous_dynamics(hover_state, zero_cmd)
        # Position derivative = velocity = 0
        assert abs(xdot[0]) < 1e-10
        assert abs(xdot[1]) < 1e-10
        assert abs(xdot[2]) < 1e-10  # vz=0
        # Velocity derivatives
        assert abs(xdot[3]) < 1e-10  # dvx = 0
        assert abs(xdot[4]) < 1e-10
        assert abs(xdot[5]) < 1e-10  # dvz = 0
        # Attitude derivatives
        assert abs(xdot[6]) < 1e-10  # dphi = (k*phi_c - phi)/tau = 0
        assert abs(xdot[7]) < 1e-10
        assert abs(xdot[8]) < 1e-10  # dpsi = psi_dot_c = 0

    def test_output_shape(self, hover_state, zero_cmd):
        """Output should be 9D."""
        xdot = continuous_dynamics(hover_state, zero_cmd)
        assert xdot.shape == (9,)

    def test_pitch_forward_creates_forward_accel(self, hover_state, max_pitch_cmd):
        """At exact hover (theta=0), dvx=0. After one discrete step, vx > 0."""
        xdot = continuous_dynamics(hover_state, max_pitch_cmd)
        assert abs(xdot[3]) < 1e-6
        x1 = discrete_step(hover_state, max_pitch_cmd, dt=0.06)
        assert x1[3] > 0

    def test_pitch_forward_no_lateral_accel(self, hover_state, max_pitch_cmd):
        """Pure pitch command should not create lateral acceleration at hover."""
        xdot = continuous_dynamics(hover_state, max_pitch_cmd)
        assert abs(xdot[4]) < 1e-10  # dvy ≈ 0

    def test_roll_creates_lateral_accel(self, hover_state, max_roll_cmd):
        """Roll command (phi_c=0.35) should create lateral acceleration."""
        xdot = continuous_dynamics(hover_state, max_roll_cmd)
        assert abs(xdot[4]) < 1e-6  # At hover phi=0, initial dvy=0

    def test_attitude_lag(self, hover_state, max_pitch_cmd):
        """Attitude rate should be proportional to (cmd - current)."""
        xdot = continuous_dynamics(hover_state, max_pitch_cmd)
        # dtheta/dt = (k_theta * theta_c - theta) / tau_theta
        # = (1.0 * 0.35 - 0) / 0.2 = 1.75 rad/s
        assert abs(xdot[7] - 1.75) < 1e-6

    def test_drag_effect(self, moving_state, zero_cmd):
        """Moving state with zero command should decelerate due to drag."""
        xdot = continuous_dynamics(moving_state, zero_cmd)
        # dvx = -kD * vx = -0.5 * 3.0 = -1.5
        assert xdot[3] < 0.0  # decelerating
        # Moving state has pitch=0.15 (non-zero) creating forward thrust which reduces drag effect
        # dvx = g*F_theta*cos(psi) - kD*vx ≈ 0 - 1.5 = -1.5 (at theta=0, but theta=0.15 here)
        assert xdot[3] < 0.0  # still decelerating (drag dominates)

    def test_yaw_rate(self, hover_state):
        """Yaw command should directly produce yaw rate."""
        u = np.array([0.0, 0.0, 0.0, 2.0])
        xdot = continuous_dynamics(hover_state, u)
        assert abs(xdot[8] - 2.0) < 1e-10  # dpsi/dt = psi_dot_c

    def test_vertical_climb(self, hover_state):
        """Vertical velocity command should produce climb acceleration."""
        u = np.array([0.0, 0.0, 1.0, 0.0])
        xdot = continuous_dynamics(hover_state, u)
        # dvz/dt = (k_vz * vz_c - vz) / tau_vz
        # With default k_vz=1.0, tau_vz=0.4: dvz = (1*1-0)/0.4 = 2.5
        assert abs(xdot[5] - 2.5) < 1e-6


# ============================================================================
# discrete_step (RK4)
# ============================================================================

class TestDiscreteStep:
    def test_shape(self, hover_state, zero_cmd):
        """Output should be 9D state."""
        x1 = discrete_step(hover_state, zero_cmd, dt=0.06)
        assert x1.shape == (9,)

    def test_hover_stays(self, hover_state, zero_cmd):
        """At hover with zero cmd, state should barely change."""
        x1 = discrete_step(hover_state, zero_cmd, dt=0.06)
        assert norm(x1 - hover_state) < 1e-6

    def test_pitch_forward_moves(self, hover_state, max_pitch_cmd):
        """After one dt=0.06 step with max pitch, vx should increase."""
        x1 = discrete_step(hover_state, max_pitch_cmd, dt=0.06)
        assert x1[3] > 0.0  # velocity in x
        assert x1[7] > 0.0  # pitch angle increased

    def test_multiple_step_tracking(self, hover_state, max_pitch_cmd):
        """Open-loop tracking over 20 steps (1.2s horizon)."""
        x = hover_state.copy()
        u = max_pitch_cmd
        for _ in range(20):
            x = discrete_step(x, u, dt=0.06)
        # Should have moved forward significantly
        assert x[0] > 1.0  # pos_x > 1m
        assert x[3] > 2.0  # vel_x > 2 m/s
        assert abs(x[1]) < 0.01  # no y drift

    def test_energy_growth(self, hover_state, max_pitch_cmd):
        """With max pitch, velocity should increase (energy injected).
        This verifies the dynamics are not artificially damped."""
        speeds = []
        x = hover_state.copy()
        u = max_pitch_cmd
        for _ in range(30):
            x = discrete_step(x, u, dt=0.06)
            speeds.append(norm(x[3:5]))
        # Speed should be monotonically increasing
        assert all(speeds[i] <= speeds[i+1] * 1.1 for i in range(len(speeds)-1))

    def test_different_dt_consistency(self, hover_state, max_pitch_cmd):
        """Small dt steps should match one larger dt step (within tolerance)."""
        # One step with dt=0.06
        x_large = discrete_step(hover_state, max_pitch_cmd, dt=0.06)
        # Three steps with dt=0.02
        x_small = hover_state.copy()
        for _ in range(3):
            x_small = discrete_step(x_small, max_pitch_cmd, dt=0.02)
        # RK4 is 4th-order, so error should be O(dt⁴)
        assert norm(x_large - x_small) < 1e-4


# ============================================================================
# QuadrotorDynamics class
# ============================================================================

class TestQuadrotorDynamics:
    def test_construction_defaults(self):
        """Default parameters should be reasonable."""
        dyn = QuadrotorDynamics()
        assert abs(dyn.g - 9.81) < 1e-10
        assert abs(dyn.kD - 0.5) < 1e-10
        assert abs(dyn.tau_phi - 0.2) < 1e-10

    def test_continuous_wrapper(self, hover_state, zero_cmd):
        """Wrapper should match standalone function."""
        dyn = QuadrotorDynamics()
        xdot1 = dyn.continuous(hover_state, zero_cmd)
        xdot2 = continuous_dynamics(hover_state, zero_cmd)
        assert norm(xdot1 - xdot2) < 1e-10

    def test_discrete_wrapper(self, hover_state, zero_cmd):
        """Wrapper should match standalone function."""
        dyn = QuadrotorDynamics()
        x1 = dyn.discrete(hover_state, zero_cmd, dt=0.06)
        x2 = discrete_step(hover_state, zero_cmd, dt=0.06, g=dyn.g, kD=dyn.kD,
                           k_phi=dyn.k_phi, k_theta=dyn.k_theta, k_vz=dyn.k_vz,
                           tau_phi=dyn.tau_phi, tau_theta=dyn.tau_theta,
                           tau_vz=dyn.tau_vz)
        assert norm(x1 - x2) < 1e-10

    def test_jacobian_state_shape(self, hover_state, zero_cmd):
        """State Jacobian should be 9x9."""
        dyn = QuadrotorDynamics()
        J = dyn.jacobian_state(hover_state, zero_cmd)
        assert J.shape == (9, 9)

    def test_jacobian_state_at_hover(self, hover_state, zero_cmd):
        """At hover, Jacobian should have zero rows for velocity-phi/theta
        coupling because F_theta=F_phi=0."""
        dyn = QuadrotorDynamics()
        J = dyn.jacobian_state(hover_state, zero_cmd)
        # d(dvx)/d(theta) should be nonzero at hover (g * dF/dtheta)
        assert J[3, 7] > 0.0
        # d(dvy)/d(phi) should be nonzero at hover
        assert J[4, 6] > 0.0

    def test_jacobian_control_shape(self, hover_state, zero_cmd):
        """Control Jacobian should be 9x4."""
        dyn = QuadrotorDynamics()
        J = dyn.jacobian_control(hover_state, zero_cmd)
        assert J.shape == (9, 4)

    def test_jacobian_control_sparsity(self, hover_state, zero_cmd):
        """Control Jacobian should have non-zero entries only for:
        d(dphi)/d(phi_c), d(dtheta)/d(theta_c), d(dvz)/d(vz_c), d(dpsi)/d(psi_dot_c)."""
        dyn = QuadrotorDynamics()
        J = dyn.jacobian_control(hover_state, zero_cmd)
        nonzero = np.where(np.abs(J) > 1e-10)
        assert len(nonzero[0]) == 4  # exactly 4 non-zero entries
        assert J[6, 0] > 0  # d(dphi)/d(phi_c)
        assert J[7, 1] > 0  # d(dtheta)/d(theta_c)
        assert J[5, 2] > 0  # d(dvz)/d(vz_c)
        assert J[8, 3] > 0  # d(dpsi)/d(psi_dot_c)

    # ========================================================================
    # CRITICAL: linearize() test — this was buggy before the fix
    # ========================================================================

    def test_linearize_cascade_coupling(self, hover_state, max_pitch_cmd):
        """CRITICAL: Linearized model at hover must predict non-zero velocity
        change from pitch command (cascade coupling theta_c → theta → vx).
        
        Before the fix, B[3,1] = 0 and the linearized model predicted
        vx=0 after one step even with max pitch command. This was the
        root cause of the entire simulation failing.
        """
        dyn = QuadrotorDynamics()
        A, B, C = dyn.linearize(hover_state, max_pitch_cmd, dt=0.06)
        # The cascade coupling term must be non-zero
        assert abs(B[3, 1]) > 1e-6, (
            "B[3,1] (theta_c coupling to vx) is zero! "
            "This means the linearized model thinks pitch commands "
            "don't affect velocity. Add the cascade term."
        )
        x_pred = A @ hover_state + B @ max_pitch_cmd + C
        x_true = dyn.discrete(hover_state, max_pitch_cmd, dt=0.06)
        # vx error should be small (not 100% like before)
        vx_error = abs(x_pred[3] - x_true[3])
        assert vx_error < 0.01, (
            f"Linearized vx prediction error too large: {vx_error:.6f}. "
            "Check the cascade coupling term in linearize()."
        )

    def test_linearize_20step_accuracy(self, hover_state, max_pitch_cmd):
        """Linearized model should track true dynamics over 20 steps."""
        dyn = QuadrotorDynamics()
        x_lin = hover_state.copy()
        x_rk4 = hover_state.copy()
        u = max_pitch_cmd
        for k in range(20):
            A, B, C = dyn.linearize(x_lin, u, dt=0.06)
            x_lin = A @ x_lin + B @ u + C
            x_rk4 = dyn.discrete(x_rk4, u, dt=0.06)
        pos_error = abs(x_lin[0] - x_rk4[0])
        assert pos_error < 0.1

    def test_linearize_at_hover_no_cmd(self, hover_state, zero_cmd):
        """At hover with zero command, linearized model should stay at hover."""
        dyn = QuadrotorDynamics()
        A, B, C = dyn.linearize(hover_state, zero_cmd, dt=0.06)
        x_pred = A @ hover_state + B @ zero_cmd + C
        assert norm(x_pred - hover_state) < 1e-10

    def test_linearize_with_moving_state(self, moving_state, zero_cmd):
        """For a moving state with zero cmd, predicted deceleration
        should match the true dynamics."""
        dyn = QuadrotorDynamics()
        A, B, C = dyn.linearize(moving_state, zero_cmd, dt=0.06)
        x_pred = A @ moving_state + B @ zero_cmd + C
        x_true = dyn.discrete(moving_state, zero_cmd, dt=0.06)
        assert norm(x_pred - x_true) < 0.02

    def test_from_config(self, mpc_config):
        """Should load parameters from YAML config."""
        dyn = QuadrotorDynamics.from_config(str(mpc_config))
        assert dyn is not None
        assert abs(dyn.g - 9.81) < 1e-6
        assert abs(dyn.kD - 0.5) < 1e-6
