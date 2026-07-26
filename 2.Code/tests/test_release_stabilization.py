from __future__ import annotations

import unittest
from pathlib import Path

import tomllib

from quadrotor_mpc.application.validation.monte_carlo import load_native_monte_carlo_protocol
from quadrotor_mpc.infrastructure.resources import resolve_input_path, resource_root
from quadrotor_mpc.interfaces.desktop.viewer import load_native_mujoco_config

CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent


class ReleaseStabilizationTests(unittest.TestCase):
    def test_version_console_scripts_ci_and_license_are_present(self):
        project = tomllib.loads(
            (CODE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(project["project"]["version"], "2.0.1")
        self.assertGreaterEqual(len(project["project"]["scripts"]), 6)
        self.assertTrue((REPOSITORY_ROOT / ".github/workflows/ci.yml").is_file())
        self.assertTrue((REPOSITORY_ROOT / "LICENSE").is_file())
        self.assertTrue((CODE_ROOT / "LICENSE").is_file())

    def test_default_runtime_inputs_resolve_from_source_resources(self):
        self.assertEqual(resource_root(), CODE_ROOT)
        for relative in (
            "config/controller.yaml",
            "config/mujoco_native_ccmpc.yaml",
            "config/native_monte_carlo.yaml",
            "models/bitcraze_crazyflie_2/cf2.xml",
        ):
            self.assertTrue(resolve_input_path(relative).is_file(), relative)

    def test_release_deadline_matches_control_period(self):
        config = load_native_mujoco_config(
            CODE_ROOT / "config" / "mujoco_native_ccmpc.yaml"
        )
        self.assertAlmostEqual(
            config.safety_fallback.solve_deadline_s,
            config.mpc_timestep_s,
        )

    def test_release_protocol_uses_fifty_paired_seeds(self):
        protocol = load_native_monte_carlo_protocol(
            CODE_ROOT / "config" / "native_monte_carlo.yaml"
        )
        self.assertEqual(protocol.trials, 50)
        self.assertEqual(len(protocol.seeds), 50)
        self.assertEqual(protocol.output_dir, (Path.cwd() / "outputs/native_monte_carlo").resolve())

    def test_root_readme_has_no_legacy_broken_paths_or_claims(self):
        readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
        for forbidden in (
            "config/scenarios/two_static.yaml",
            "config/scenarios/one_moving.yaml",
            "notebooks/interactive_demo.ipynb",
            "Guarantees probabilistic safety",
            "Runs in real time — ~14 ms",
        ):
            self.assertNotIn(forbidden, readme)
        self.assertIn("VALIDATED_WITH_LIMITATIONS", readme)


if __name__ == "__main__":
    unittest.main()
