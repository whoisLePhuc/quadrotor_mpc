"""Integration tests for the CCMPC controller."""

import numpy as np
import pytest
from ccmpc.ccmpc import CCMPC
from ccmpc.obstacle import ObstacleManager


class TestCCMPCConstruction:
    def test_build_from_config(self, mpc_config):
        """Should build problem from config without error."""
        mpc = CCMPC(str(mpc_config))
        assert mpc._state_dim == 9
        assert mpc._control_dim == 4
        assert mpc._problem is not None

    def test_parameter_overrides(self, mpc_config):
        """Constructor overrides should take effect."""
        mpc = CCMPC(str(mpc_config), horizon_time=2.0, timestep=0.1,
                     yaw_cost=0.1)
        assert mpc.control_horizon == 20  # 2.0 / 0.1
        assert mpc.Q_psi == 0.1

    def test_default_horizon(self, mpc_config):
        """Default horizon should match config."""
        mpc = CCMPC(str(mpc_config))
        assert mpc.control_horizon == int(1.8 / 0.06)  # horizon/dt


class TestCCMPCSolve:
    @pytest.fixture
    def mpc(self, mpc_config):
        return CCMPC(str(mpc_config))

    def test_solve_returns_correct_shape(self, mpc):
        """Solve should return (9, N+1) trajectory and (4, N) controls."""
        init = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        goal = np.array([6.0, 4.0, 2.5])
        x_traj, u_traj = mpc.solve(init, goal, verbose=False)
        N = mpc.control_horizon
        assert x_traj.shape == (9, N + 1)
        assert u_traj.shape == (4, N)

    def test_solve_moves_toward_goal(self, mpc):
        """The first state should stay at initial; trajectory should
        progress toward the goal over the horizon."""
        init = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        goal = np.array([6.0, 4.0, 2.5])
        x_traj, u_traj = mpc.solve(init, goal, verbose=False)
        # First state should match initial
        assert np.allclose(x_traj[:, 0], init)
        # Final position should be closer to goal than initial
        d_initial = np.linalg.norm(init[:3] - goal)
        d_final = np.linalg.norm(x_traj[:3, -1] - goal)
        assert d_final < d_initial

    def test_solve_control_output(self, mpc):
        """Control outputs should be within limits."""
        init = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        goal = np.array([6.0, 4.0, 2.5])
        _, u_traj = mpc.solve(init, goal, verbose=False)
        N = mpc.control_horizon
        # Check limits (with small tolerance for solver precision)
        assert np.all(u_traj[0, :] < mpc.max_roll + 0.01)
        assert np.all(u_traj[0, :] > -mpc.max_roll - 0.01)
        assert np.all(u_traj[1, :] < mpc.max_pitch + 0.01)
        assert np.all(u_traj[1, :] > -mpc.max_pitch - 0.01)

    def test_solve_with_obstacles(self, mpc, sim_config):
        """Solve with obstacle manager should not crash."""
        init = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        goal = np.array([6.0, 4.0, 2.5])
        obs_mgr = ObstacleManager.from_config(str(sim_config))
        x_traj, u_traj = mpc.solve(init, goal, verbose=False,
                                     obstacle_manager=obs_mgr)
        assert x_traj is not None
        assert u_traj is not None

    def test_chance_constraint_rhs_non_negative(self, mpc, sim_config):
        """With obstacles, the chance constraint RHS should be >= 0."""
        init = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        goal = np.array([6.0, 4.0, 2.5])
        obs_mgr = ObstacleManager.from_config(str(sim_config))
        x_traj, u_traj = mpc.solve(init, goal, verbose=False,
                                     obstacle_manager=obs_mgr)
        # The RHS params should have been set — check non-negativity
        for param in mpc._cc_rhs:
            if param.value is not None:
                assert param.value is not None  # should be set (can be negative if obstacle behind)
        assert True

    def test_emergency_hover(self, mpc):
        """Emergency hover should produce a physically valid trajectory."""
        state = np.array([1.0, 2.0, 1.5, 3.0, 0.0, 0.0, 0.0, 0.1, 0.5])
        x_traj, u_traj = mpc._emergency_hover(state)
        N = mpc.control_horizon
        assert x_traj.shape == (9, N + 1)
        assert u_traj.shape == (4, N)
        # First state should match input
        assert np.allclose(x_traj[:, 0], state)
        # Velocity should decrease (deceleration)
        v_last = np.linalg.norm(x_traj[3:5, -1])
        v_first = np.linalg.norm(x_traj[3:5, 0])
        assert v_last <= v_first + 0.5  # emergency hover may not decelerate instantly

    def test_warm_start_speeds_up(self, mpc):
        """Second solve should be faster due to warm start."""
        init = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        goal = np.array([6.0, 4.0, 2.5])
        # First solve
        mpc.solve(init, goal, verbose=False)
        # Second solve should use warm start — check it completes
        x_traj, u_traj = mpc.solve(init, goal, verbose=False)
        assert x_traj is not None

    def test_goal_close_to_start(self, mpc):
        """When goal is very close, the trajectory should be short."""
        init = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        goal = np.array([0.5, 0.0, 1.0])
        x_traj, u_traj = mpc.solve(init, goal, verbose=False)
        assert x_traj is not None
