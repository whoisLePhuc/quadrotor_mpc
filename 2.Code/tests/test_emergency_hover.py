"""Regression tests: emergency hover must not wipe the whole control trajectory.

This locks the fix in ``CCMPC._emergency_hover`` (``control/ccmpc/ccmpc.py``).
The previous implementation used ``u_traj[:] = 0.0`` inside the ``on_ground``
branch, destroying every already-computed control for ``k' < k`` the instant
the rollout dipped below ``z = 0.2`` at any later step.  The fix overwrites
only the current control column::

    u_traj[:, k] = 0.0

Invariant under test: when the rollout touches the ground threshold at step
``k > 0``, every column before ``k`` must be preserved and only column ``k``
carries the emergency-hover command.
"""

from __future__ import annotations

import unittest

import numpy as np

from quadrotor_mpc.control.ccmpc.ccmpc import CCMPC
from quadrotor_mpc.control.ccmpc.dynamics import QuadrotorDynamics

# Trajectory convention: u_traj has shape (n_controls=4, horizon=N).
N_CONTROLS = 4
GROUND_THRESHOLD_Z = 0.2

# Dynamics parameters matching config/controller.yaml.
_DYNAMICS = QuadrotorDynamics(
    g=9.81,
    kD=0.5,
    k_phi=1.0,
    k_theta=1.0,
    k_vz=3.0,
    tau_phi=0.33,
    tau_theta=0.33,
    tau_vz=0.4,
)
_MAX_ROLL = 0.25
_MAX_PITCH = 0.25
_MAX_VERT_VEL = 3.0
_MAX_YAW_RATE = 0.8


def _emergency_hover_controller(horizon: int, timestep: float = 0.06) -> CCMPC:
    """Minimal CCMPC instance exposing only what ``_emergency_hover`` touches.

    Building a full ``CCMPC`` would compile a CVXPY problem just to exercise a
    pure fallback rollout; this fixture sets the attributes the method reads so
    the real production method runs against the real production dynamics.
    """
    controller = object.__new__(CCMPC)
    controller.control_horizon = horizon
    controller.dt = timestep
    controller.dynamics = _DYNAMICS
    controller.max_roll = _MAX_ROLL
    controller.max_pitch = _MAX_PITCH
    controller.max_vert_vel = _MAX_VERT_VEL
    controller.max_yaw_rate = _MAX_YAW_RATE
    return controller


def _first_ground_step(x_traj: np.ndarray) -> int | None:
    for step in range(x_traj.shape[1]):
        if x_traj[2, step] < GROUND_THRESHOLD_Z:
            return step
    return None


class EmergencyHoverPreservationTests(unittest.TestCase):
    """Columns before the first on-ground step must survive the hover call."""

    def test_only_current_control_column_is_overwritten(self):
        horizon = 8
        controller = _emergency_hover_controller(horizon)
        # Airborne start with a downward velocity toward a low goal: the
        # rollout crosses z=0.2 mid-horizon, so columns 0..k-1 already hold
        # non-zero controls computed by the airborne branch.
        state = np.array(
            [0.0, 0.0, 1.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0],
            dtype=float,
        )
        goal = np.array([2.0, 2.0, 0.3], dtype=float)

        x_traj, u_traj = controller._emergency_hover(state, goal)
        k_ground = _first_ground_step(x_traj)

        self.assertIsNotNone(k_ground)
        assert k_ground is not None  # type narrowing for the static checker
        self.assertGreater(k_ground, 0)
        self.assertEqual(u_traj.shape, (N_CONTROLS, horizon))
        self.assertEqual(u_traj.dtype, np.float64)
        # Every column before the ground step holds its airborne-branch
        # command; `u_traj[:] = 0.0` would zero all of them.
        np.testing.assert_array_equal(
            u_traj[:, :k_ground] != 0.0,
            np.ones((N_CONTROLS, k_ground), dtype=bool),
        )
        # The ground column is the emergency hover command: pure climb,
        # no tilt, yaw-rate only.
        self.assertEqual(u_traj[0, k_ground], 0.0)
        self.assertEqual(u_traj[1, k_ground], 0.0)
        self.assertAlmostEqual(
            float(u_traj[2, k_ground]),
            _MAX_VERT_VEL,
        )
        self.assertLessEqual(abs(float(u_traj[3, k_ground])), _MAX_YAW_RATE)

    def test_airborne_only_rollout_never_wipes_any_column(self):
        horizon = 8
        controller = _emergency_hover_controller(horizon)
        # Goal above the start: altitude rises, the ground branch never runs.
        state = np.array(
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=float,
        )
        goal = np.array([2.0, 2.0, 2.5], dtype=float)

        x_traj, u_traj = controller._emergency_hover(state, goal)

        self.assertIsNone(_first_ground_step(x_traj))
        # Every horizon column keeps its non-zero airborne command.
        np.testing.assert_array_equal(
            u_traj != 0.0,
            np.ones((N_CONTROLS, horizon), dtype=bool),
        )

    def test_shape_and_dtype_are_preserved_for_non_square_trajectory(self):
        # (4, 5): non-square, so a row/column mix-up is observable.
        horizon = 5
        controller = _emergency_hover_controller(horizon)
        state = np.array(
            [0.0, 0.0, 1.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0],
            dtype=float,
        )
        goal = np.array([2.0, 2.0, 0.3], dtype=float)

        x_traj, u_traj = controller._emergency_hover(state, goal)

        self.assertEqual(u_traj.shape, (N_CONTROLS, horizon))
        self.assertEqual(x_traj.shape, (9, horizon + 1))
        self.assertEqual(u_traj.dtype, np.float64)

    def test_ground_at_first_step_has_no_prior_columns_to_preserve(self):
        # Starting already below z=0.2 exercises the ground branch from k=0;
        # the invariant is that column 0 is the emergency hover command.
        controller = _emergency_hover_controller(horizon=4)
        state = np.array(
            [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=float,
        )
        goal = np.array([2.0, 2.0, 0.3], dtype=float)

        x_traj, u_traj = controller._emergency_hover(state, goal)

        self.assertEqual(_first_ground_step(x_traj), 0)
        self.assertEqual(u_traj[0, 0], 0.0)
        self.assertEqual(u_traj[1, 0], 0.0)
        self.assertAlmostEqual(float(u_traj[2, 0]), _MAX_VERT_VEL)


if __name__ == "__main__":
    unittest.main()
