from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mujoco_native import NativeMuJoCoConfig, load_native_mujoco_config
from vehicle import CRAZYFLIE_2


CODE_ROOT = Path(__file__).resolve().parents[1]


class NativeMuJoCoConfigurationTests(unittest.TestCase):
    def test_default_configuration_is_valid(self):
        config = load_native_mujoco_config(CODE_ROOT / "config" / "mujoco_native.yaml")
        self.assertEqual(config.name, "nmpc-mujoco-native-static")
        self.assertEqual(config.viewer.camera_mode, "follow")
        self.assertAlmostEqual(config.bounds["thrust"], 0.085)
        self.assertEqual(len(config.obstacles), 1)

    def test_invalid_obstacle_type_is_rejected(self):
        mapping = {
            "start": {"x": 0, "y": 0, "z": 1},
            "goal": {
                "position": {"x": 1, "y": 0, "z": 1},
                "euler": {},
            },
            "controller": {
                "bounds": {"thrust": 0.08, "torque_rp": 0.001, "torque_yaw": 0.0002}
            },
            "simulation": {},
            "viewer": {},
            "obstacles": [
                {"type": "teleporting", "x": 0, "y": 0, "z": 1, "radius": 0.2}
            ],
        }
        with self.assertRaisesRegex(ValueError, "static.*dynamic"):
            NativeMuJoCoConfig.from_mapping(mapping)

    def test_missing_file_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.yaml"
            with self.assertRaisesRegex(ValueError, "not found"):
                load_native_mujoco_config(path)


class CrazyflieParameterTests(unittest.TestCase):
    def test_menagerie_rigid_body_parameters(self):
        self.assertAlmostEqual(CRAZYFLIE_2.mass_kg, 0.027)
        np.testing.assert_allclose(
            CRAZYFLIE_2.inertia_kg_m2,
            [2.3951e-5, 2.3951e-5, 3.2347e-5],
            rtol=0.0,
            atol=1e-12,
        )
        self.assertAlmostEqual(CRAZYFLIE_2.hover_thrust_n, 0.26487)
        self.assertGreater(CRAZYFLIE_2.max_upward_thrust_deviation_n, 0.0)

    def test_vendored_model_and_license_are_present(self):
        model_root = CODE_ROOT / "models" / "bitcraze_crazyflie_2"
        self.assertTrue((model_root / "cf2.xml").is_file())
        self.assertTrue((model_root / "LICENSE").is_file())
        self.assertGreater(len(list((model_root / "assets").glob("*.obj"))), 30)


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("mujoco", "casadi", "do_mpc")),
    "optional NMPC/MuJoCo dependencies are not installed",
)
class CrazyfliePlantTests(unittest.TestCase):
    def _plant(self, obstacles=None):
        from mujoco_plant import MuJoCoPlant

        return MuJoCoPlant(
            {"x": 0.0, "y": 0.0, "z": 1.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            {"x": 1.0, "y": 0.0, "z": 1.0},
            [] if obstacles is None else obstacles,
            mj_dt=0.002,
        )

    def test_model_compiles_with_upstream_mass_and_inertia(self):
        plant = self._plant()
        self.assertAlmostEqual(float(plant.model.body_mass[plant.quad_id]), 0.027)
        np.testing.assert_allclose(
            plant.model.body_inertia[plant.quad_id], CRAZYFLIE_2.inertia_kg_m2
        )
        self.assertEqual(len(plant.quad_geom_ids), 32)

    def test_hover_equilibrium(self):
        plant = self._plant()
        initial = plant.get_state_13()[:, 0].copy()
        plant.apply_control_and_step([0.0, 0.0, 0.0, 0.0], 500, 0.0)
        final = plant.get_state_13()[:, 0]
        np.testing.assert_allclose(final[:6], initial[:6], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
