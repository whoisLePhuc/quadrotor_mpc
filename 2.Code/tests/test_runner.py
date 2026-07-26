from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from quadrotor_mpc.application.simulation.config import load_scenario
from quadrotor_mpc.application.simulation.runner import SimulationRunner

ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def test_hover_smoke_run_is_finite_and_reproducible(self) -> None:
        scenario = load_scenario(ROOT / "config/scenarios/hover.yaml")
        scenario.max_time = 0.20
        first = SimulationRunner(
            scenario,
            ROOT / "config/controller.yaml",
            mode="ccmpc",
            backend="scipy",
            seed=99,
        ).run()
        second = SimulationRunner(
            scenario,
            ROOT / "config/controller.yaml",
            mode="ccmpc",
            backend="scipy",
            seed=99,
        ).run()
        self.assertEqual(first.states.shape[1], 9)
        self.assertTrue(np.isfinite(first.states).all())
        np.testing.assert_allclose(first.states, second.states, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
