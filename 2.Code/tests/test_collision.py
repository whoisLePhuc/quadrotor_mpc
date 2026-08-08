"""Regression tests: MuJoCo plant collision detection includes ground contact.

This locks the fix in ``8aa474a`` ("fix: count ground contact as collision in
native MuJoCo plant").  Previously ``check_collision()`` only detected
quadrotor-to-obstacle contacts, so an episode that fell or touched the floor
still reported ``collided=False``.

Invariant under test:

    check_collision() == check_obstacle_collision() OR check_ground_collision()

The tests use real MuJoCo contacts (no viewer, no timing dependency) so a
regression in how ground contacts are recognised is caught, not just the OR
aggregation.
"""

from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

from quadrotor_mpc.infrastructure.mujoco.plant import MuJoCoPlant

GROUND_POSE = {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
AIRBORNE_POSE = {"x": 0.0, "y": 0.0, "z": 2.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
GOAL = {"x": 1.0, "y": 0.0, "z": 1.0}


@unittest.skipUnless(
    importlib.util.find_spec("mujoco"),
    "MuJoCo is not installed",
)
class GroundCollisionTests(unittest.TestCase):
    """Ground contact must count as a collision (the fixed behaviour)."""

    def _plant(self, pose, obstacles=None):
        return MuJoCoPlant(
            dict(pose),
            dict(GOAL),
            [] if obstacles is None else obstacles,
            mj_dt=0.002,
        )

    def test_ground_contact_counts_as_collision(self):
        plant = self._plant(GROUND_POSE)

        self.assertFalse(plant.check_obstacle_collision())
        self.assertTrue(plant.check_ground_collision())
        self.assertTrue(plant.check_collision())

    def test_ground_contact_is_detected_after_a_step(self):
        # Contacts must survive a physics step, not only the initial forward
        # kinematics, because the closed-loop runtime reads them after stepping.
        plant = self._plant(GROUND_POSE)
        plant.apply_control_and_step([0.0, 0.0, 0.0, 0.0], 5, 0.0)

        self.assertFalse(plant.check_obstacle_collision())
        self.assertTrue(plant.check_ground_collision())
        self.assertTrue(plant.check_collision())


@unittest.skipUnless(
    importlib.util.find_spec("mujoco"),
    "MuJoCo is not installed",
)
class AirborneCollisionTests(unittest.TestCase):
    """No contacts at altitude must not be reported as a collision."""

    def test_airborne_vehicle_without_contacts_is_not_collision(self):
        plant = MuJoCoPlant(
            dict(AIRBORNE_POSE),
            dict(GOAL),
            [],
            mj_dt=0.002,
        )

        self.assertFalse(plant.check_obstacle_collision())
        self.assertFalse(plant.check_ground_collision())
        self.assertFalse(plant.check_collision())

    def test_airborne_vehicle_after_steps_has_no_contacts(self):
        plant = MuJoCoPlant(
            dict(AIRBORNE_POSE),
            dict(GOAL),
            [],
            mj_dt=0.002,
        )
        # Hover-equivalent command keeps the drone at altitude.
        plant.apply_control_and_step([0.0, 0.0, 0.0, 0.0], 50, 0.0)

        self.assertFalse(plant.check_collision())


@unittest.skipUnless(
    importlib.util.find_spec("mujoco"),
    "MuJoCo is not installed",
)
class ObstacleCollisionTests(unittest.TestCase):
    """Obstacle-only contacts must still count as a collision."""

    def test_obstacle_contact_counts_as_collision(self):
        obstacle = {
            "type": "static",
            "x": 0.0,
            "y": 0.0,
            "z": 1.0,
            "radius": 0.4,
        }
        # Quadrotor centered on the obstacle: guaranteed contact.
        plant = MuJoCoPlant(
            {"x": 0.0, "y": 0.0, "z": 1.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            dict(GOAL),
            [obstacle],
            mj_dt=0.002,
        )

        self.assertTrue(plant.check_obstacle_collision())
        self.assertFalse(plant.check_ground_collision())
        self.assertTrue(plant.check_collision())


class CollisionAggregationTests(unittest.TestCase):
    """The OR contract of the production collision loop without physics."""

    @staticmethod
    def _plant_with_contacts(contact_pairs):
        """Plant whose ``data.contact`` exposes the given geom-id pairs."""
        plant = object.__new__(MuJoCoPlant)
        plant.quad_geom_ids = {1, 2, 3}
        plant.obstacle_geom_ids = {50}
        plant.ground_geom_id = 100
        plant.hazard_geom_ids = {50, 100}

        def contact(geom1, geom2):
            return mock.Mock(geom1=geom1, geom2=geom2)

        plant.data = mock.Mock(
            ncon=len(contact_pairs),
            contact=[contact(a, b) for a, b in contact_pairs],
        )
        return plant

    def test_collision_aggregates_ground_contact(self):
        plant = self._plant_with_contacts([(1, 100)])  # quad geom + ground

        self.assertTrue(plant.check_collision())
        self.assertTrue(plant.check_ground_collision())
        self.assertFalse(plant.check_obstacle_collision())

    def test_collision_aggregates_obstacle_contact(self):
        plant = self._plant_with_contacts([(2, 50)])  # quad geom + obstacle

        self.assertTrue(plant.check_collision())
        self.assertTrue(plant.check_obstacle_collision())
        self.assertFalse(plant.check_ground_collision())

    def test_collision_is_false_without_hazard_contacts(self):
        plant = self._plant_with_contacts([(1, 200)])  # unrelated geom

        self.assertFalse(plant.check_collision())
        self.assertFalse(plant.check_obstacle_collision())
        self.assertFalse(plant.check_ground_collision())

    def test_collision_is_false_with_no_contacts(self):
        plant = self._plant_with_contacts([])

        self.assertFalse(plant.check_collision())
        self.assertFalse(plant.check_obstacle_collision())
        self.assertFalse(plant.check_ground_collision())


if __name__ == "__main__":
    unittest.main()
