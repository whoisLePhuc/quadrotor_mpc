"""Regression tests: telemetry must preserve the enforced chance profile.

The enforced profile is the exact set of risk/tightening values loaded into
the NMPC solver for one solve attempt.  A profile recomputed after the solve on
the optimized trajectory is a post-solve diagnostic and must never be reported
as enforced.
"""

from __future__ import annotations

import unittest
from typing import cast

import numpy as np

from quadrotor_mpc.core.chance_profile import (
    ENFORCED_PHASE,
    POST_SOLVE_DIAGNOSTIC_PHASE,
    ChanceProfile,
    ChanceProfileShapeError,
    build_chance_profile_snapshot,
    new_profile_id,
    new_solve_attempt_id,
)


def make_profile(
    *,
    epsilon=None,
    sigma=None,
    margin=None,
    radius=None,
    phase=ENFORCED_PHASE,
    source="seed_trajectory",
    attempt="attempt-1",
    steps=2,
    obstacles=1,
    joint_budget=None,
    allocated=None,
):
    shape = (steps, obstacles)
    epsilon = np.full(shape, 0.01) if epsilon is None else np.asarray(epsilon, dtype=float)
    sigma = np.full(shape, 0.1) if sigma is None else np.asarray(sigma, dtype=float)
    margin = np.full(shape, 0.05) if margin is None else np.asarray(margin, dtype=float)
    radius = np.full(shape, 0.7) if radius is None else np.asarray(radius, dtype=float)
    return build_chance_profile_snapshot(
        control_tick=0,
        solve_attempt_id=attempt,
        phase=phase,
        trajectory_source=source,
        allocation_method="uniform",
        risk_semantics="joint",
        node_indices=tuple(range(steps)),
        obstacle_ids=tuple(f"obstacle_{i}" for i in range(obstacles)),
        epsilon=epsilon,
        projected_sigma=sigma,
        tightening_margin=margin,
        tightened_radius=radius,
        active_mask=np.ones(shape, dtype=bool),
        joint_budget=joint_budget,
        allocated_budget=allocated,
        remaining_budget=(0.0 if joint_budget is not None and allocated is not None else None),
    )


class ChanceProfileSnapshotTests(unittest.TestCase):
    def test_enforced_profile_is_immutable_snapshot(self):
        working = np.array([[0.01, 0.02]])
        profile = build_chance_profile_snapshot(
            control_tick=0,
            solve_attempt_id="a",
            phase=ENFORCED_PHASE,
            trajectory_source="seed",
            allocation_method="uniform",
            risk_semantics="joint",
            node_indices=(0,),
            obstacle_ids=("o1", "o2"),
            epsilon=working,
            projected_sigma=np.array([[0.1, 0.2]]),
            tightening_margin=np.array([[0.05, 0.06]]),
            tightened_radius=np.array([[0.7, 0.8]]),
            active_mask=np.array([[True, True]]),
            joint_budget=0.03,
            allocated_budget=0.03,
            remaining_budget=0.0,
        )
        working[0, 0] = 0.99
        self.assertEqual(profile.epsilon[0, 0], 0.01)
        with self.assertRaises(ValueError):
            profile.epsilon[0, 0] = 0.5

    def test_profile_id_is_deterministic_across_builds(self):
        first = make_profile()
        second = make_profile()
        self.assertEqual(
            first.provenance.profile_id,
            second.provenance.profile_id,
        )

    def test_solve_attempt_id_is_unique(self):
        self.assertNotEqual(new_solve_attempt_id(), new_solve_attempt_id())

    def test_diagnostic_and_enforced_phase_are_distinct(self):
        enforced = make_profile(phase=ENFORCED_PHASE)
        diagnostic = make_profile(phase=POST_SOLVE_DIAGNOSTIC_PHASE)
        self.assertEqual(enforced.provenance.phase, "ENFORCED")
        self.assertEqual(diagnostic.provenance.phase, "POST_SOLVE_DIAGNOSTIC")
        self.assertNotEqual(
            enforced.provenance.profile_id,
            diagnostic.provenance.profile_id,
        )

    def test_active_epsilon_must_be_strictly_between_zero_and_one(self):
        with self.assertRaises(ValueError):
            make_profile(epsilon=np.array([[0.0]]))
        with self.assertRaises(ValueError):
            make_profile(epsilon=np.array([[1.0]]))

    def test_shape_mismatch_raises_structural_error(self):
        with self.assertRaises(ChanceProfileShapeError):
            make_profile(epsilon=np.array([[0.01, 0.02, 0.03]]))

    def test_nonfinite_values_are_rejected(self):
        with self.assertRaises(ValueError):
            make_profile(sigma=np.array([[float("nan")]]))


class ChanceProfileIdentityTests(unittest.TestCase):
    def test_profile_id_is_digest_not_python_hash(self):
        payload = {"epsilon": [[0.01]], "sigma": [[0.1]]}
        digest = new_profile_id(payload)
        self.assertEqual(len(digest), 16)
        self.assertEqual(new_profile_id(payload), digest)


class ChanceProfileSerializationTests(unittest.TestCase):
    def test_round_trip_preserves_payload(self):
        profile = make_profile(
            epsilon=np.array([[0.01, 0.02]]),
            sigma=np.array([[0.1, 0.2]]),
            margin=np.array([[0.05, 0.06]]),
            radius=np.array([[0.7, 0.8]]),
            steps=1,
            obstacles=2,
            joint_budget=0.03,
            allocated=0.03,
        )
        mapping = profile.to_mapping()
        self.assertEqual(mapping["provenance"]["phase"], "ENFORCED")
        self.assertEqual(mapping["epsilon"], [[0.01, 0.02]])
        self.assertEqual(mapping["obstacle_ids"], ["obstacle_0", "obstacle_1"])
        self.assertEqual(mapping["node_indices"], [0])
        self.assertEqual(mapping["joint_budget"], 0.03)
        self.assertNotIn("NaN", str(mapping))

    def test_none_profile_is_not_serialized_as_zero(self):
        self.assertIsNone(None)


class ChanceProfileShapeValidationTests(unittest.TestCase):
    def setUp(self):
        from quadrotor_mpc.control.nmpc.deterministic import (
            DeterministicNMPCController,
        )

        self.controller = object.__new__(DeterministicNMPCController)
        self.controller._horizon_steps = 3
        self.controller._obstacle_specs = (
            {"type": "static", "x": 1.0, "y": 0.0, "z": 1.0, "radius": 0.2},
        )

    def _fake_profile(self):
        class FakeProfile:
            risk_allocations = np.zeros((4, 1))
            safety_radii_m = np.zeros((4, 1))

        return FakeProfile()

    def test_state_shape_mismatch_is_rejected(self):
        with self.assertRaises(ChanceProfileShapeError):
            self.controller._validate_profile_shapes(
                np.zeros((3, 13)),
                np.zeros((3, 4)),
                np.zeros((1, 4, 3)),
                self._fake_profile(),
            )

    def test_control_shape_mismatch_is_rejected(self):
        with self.assertRaises(ChanceProfileShapeError):
            self.controller._validate_profile_shapes(
                np.zeros((4, 13)),
                np.zeros((4, 4)),
                np.zeros((1, 4, 3)),
                self._fake_profile(),
            )

    def test_obstacle_prediction_shape_mismatch_is_rejected(self):
        with self.assertRaises(ChanceProfileShapeError):
            self.controller._validate_profile_shapes(
                np.zeros((4, 13)),
                np.zeros((3, 4)),
                np.zeros((1, 5, 3)),
                self._fake_profile(),
            )

    def test_valid_shapes_are_accepted(self):
        self.controller._validate_profile_shapes(
            np.zeros((4, 13)),
            np.zeros((3, 4)),
            np.zeros((1, 4, 3)),
            self._fake_profile(),
        )


class ChanceProfileEnforcedSemanticsTests(unittest.TestCase):
    """The solver boundary and telemetry must reflect the enforced profile."""

    def test_solver_boundary_receives_enforced_profile_values(self):
        enforced = make_profile(
            epsilon=np.array([[0.005, 0.015, 0.030]]),
            steps=1,
            obstacles=3,
            joint_budget=0.05,
            allocated=0.05,
        )
        goal_state_risk = enforced.epsilon.T
        goal_state_safe = enforced.tightened_radius.T
        np.testing.assert_array_equal(
            goal_state_risk,
            np.array([[0.005], [0.015], [0.030]]),
        )
        np.testing.assert_array_equal(
            goal_state_safe,
            np.array([[0.7], [0.7], [0.7]]),
        )

    def test_telemetry_serializes_enforced_not_post_solve_profile(self):
        enforced = make_profile(
            epsilon=np.array([[0.01]]),
            steps=1,
            obstacles=1,
            phase=ENFORCED_PHASE,
            source="seed_trajectory",
            attempt="attempt-1",
        )
        diagnostic = make_profile(
            epsilon=np.array([[0.04]]),
            steps=1,
            obstacles=1,
            phase=POST_SOLVE_DIAGNOSTIC_PHASE,
            source="optimized_trajectory",
            attempt="attempt-1",
        )
        enforced_payload = enforced.to_mapping()
        diagnostic_payload = diagnostic.to_mapping()
        self.assertEqual(enforced_payload["epsilon"], [[0.01]])
        self.assertEqual(diagnostic_payload["epsilon"], [[0.04]])
        self.assertNotEqual(
            enforced_payload["provenance"]["profile_id"],
            diagnostic_payload["provenance"]["profile_id"],
        )

    def test_budget_metric_uses_enforced_profile_only(self):
        enforced = make_profile(
            epsilon=np.array([[0.01, 0.02]]),
            steps=1,
            obstacles=2,
            joint_budget=0.03,
            allocated=0.03,
        )
        diagnostic = make_profile(
            epsilon=np.array([[0.20, 0.30]]),
            steps=1,
            obstacles=2,
            phase=POST_SOLVE_DIAGNOSTIC_PHASE,
            source="optimized_trajectory",
        )
        self.assertAlmostEqual(
            float(np.sum(enforced.epsilon)),
            0.03,
        )
        self.assertNotAlmostEqual(
            float(np.sum(diagnostic.epsilon)),
            0.03,
        )
        self.assertEqual(enforced.provenance.phase, "ENFORCED")

    def test_solve_result_carries_enforced_profile_with_own_identity(self):
        """The enforced snapshot survives the solve and owns its profile id."""
        solution = self._solve_with_mock_controller()
        self.assertEqual(
            solution.chance_profile_application_status,
            "APPLIED",
        )
        self.assertTrue(solution.chance_profile_enforced_profile_id)
        self.assertIsNotNone(solution.enforced_chance_profile)
        enforced = cast("ChanceProfile", solution.enforced_chance_profile)
        self.assertEqual(enforced.provenance.phase, "ENFORCED")
        self.assertEqual(
            enforced.provenance.profile_id,
            solution.chance_profile_enforced_profile_id,
        )
        self.assertEqual(
            enforced.provenance.solve_attempt_id,
            solution.chance_profile_solve_attempt_id,
        )
        self.assertIsNone(solution.post_solve_diagnostic_profile)

    def test_deterministic_mode_serializes_null_profile(self):
        solution = self._solve_with_mock_controller(deterministic=True)
        self.assertIsNone(solution.enforced_chance_profile)
        self.assertIsNone(solution.post_solve_diagnostic_profile)
        self.assertEqual(
            solution.chance_profile_application_status,
            "NOT_APPLICABLE_DETERMINISTIC",
        )
        self.assertEqual(solution.chance_profile_enforced_profile_id, "")

    def test_diagnostic_profile_never_replaces_enforced_identity(self):
        """When the solver returns a shorter horizon, the enforced profile id
        must still come from the enforced profile, never the diagnostic."""
        solution = self._solve_with_mock_controller(
            returned_horizon=2,
        )
        self.assertIsNotNone(solution.enforced_chance_profile)
        self.assertIsNotNone(solution.post_solve_diagnostic_profile)
        diagnostic = cast(
            "ChanceProfile",
            solution.post_solve_diagnostic_profile,
        )
        self.assertEqual(
            diagnostic.provenance.phase,
            "POST_SOLVE_DIAGNOSTIC",
        )
        enforced = cast("ChanceProfile", solution.enforced_chance_profile)
        self.assertEqual(
            solution.chance_profile_enforced_profile_id,
            enforced.provenance.profile_id,
        )
        self.assertNotEqual(
            solution.chance_profile_enforced_profile_id,
            diagnostic.provenance.profile_id,
        )
        # With uniform allocation enforced and diagnostic epsilon agree, so the
        # provenance identity above is the discriminating assertion.
        self.assertEqual(
            solution.risk_allocations.shape,
            diagnostic.epsilon.shape,
        )
        # Budget metadata must come from the enforced profile: it counts the
        # full (N+1) horizon, not the shortened diagnostic grid.
        self.assertEqual(
            solution.risk_constraint_count,
            3,
        )

    def _solve_with_mock_controller(
        self,
        *,
        deterministic: bool = False,
        returned_horizon: int | None = None,
    ):
        from unittest import mock

        from quadrotor_mpc.control.nmpc.chance_constraints import (
            ChanceConstraintOptions,
        )
        from quadrotor_mpc.control.nmpc.covariance import (
            CovariancePropagationOptions,
        )
        from quadrotor_mpc.control.nmpc.deterministic import (
            DeterministicNMPCController,
        )
        from quadrotor_mpc.control.nmpc.risk_budget import RiskBudgetOptions
        from quadrotor_mpc.core.contracts import (
            ControlGoal,
            ObstacleBelief,
            SphericalObstacle,
            VehicleBelief,
        )

        horizon = 2
        controller = object.__new__(DeterministicNMPCController)
        controller._obstacle_specs = (
            {"type": "static", "x": 1.5, "y": 0.0, "z": 1.0, "radius": 0.2},
        )
        controller._margin = 0.3
        controller._horizon_steps = horizon
        controller._timestep_s = 0.05
        controller._covariance_options = CovariancePropagationOptions(
            enabled=not deterministic,
            mode="open_loop",
        )
        controller._chance_options = ChanceConstraintOptions(
            enabled=not deterministic,
            risk_budget=RiskBudgetOptions(),
        )
        controller._covariance_propagator = mock.MagicMock()

        def _fake_propagate(belief, obstacles, states, controls):
            steps = states.shape[0]
            return (
                np.zeros((steps, 12, 12)),
                np.zeros((steps, len(obstacles), 6, 6)),
            )

        controller._covariance_propagator.propagate.side_effect = _fake_propagate
        controller._last_nominal_states = None
        controller._last_nominal_controls = None
        controller._solve_attempt_counter = 0

        class FakeMpc:
            def __init__(self, horizon_rows: int):
                self.solver_stats = {"success": True, "iterations": {}}
                self.data = mock.MagicMock()
                self.data.prediction.return_value = np.zeros((horizon_rows, 1))

            def make_step(self, _state):
                return np.zeros((4, 1))

        returned = horizon + 1 if returned_horizon is None else max(1, int(returned_horizon))
        mpc = FakeMpc(returned)
        controller.mpc = mpc
        controller.model = mock.MagicMock()
        controller._goal_state = {}
        controller._obstacle_tvp_idx = (0,)

        belief = VehicleBelief(
            mean_state_13=np.array(
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                dtype=float,
            ),
            error_covariance_12=np.eye(12) * 1e-3,
        )
        obstacle = ObstacleBelief(
            mean_state_6=np.array([1.5, 0.0, 1.0, 0, 0, 0]),
            covariance_6=np.eye(6) * 1e-3,
            shape=SphericalObstacle(0.2),
        )
        goal = ControlGoal(
            position=np.array([3.0, 2.0, 2.5]),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        return controller.solve(belief, [obstacle], goal, 0.0)


if __name__ == "__main__":
    unittest.main()
