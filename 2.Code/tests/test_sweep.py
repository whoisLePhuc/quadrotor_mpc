from __future__ import annotations

import unittest
from pathlib import Path

from experiments.sweep import run_parameter_sweep
from simulation.config import load_scenario


ROOT = Path(__file__).resolve().parents[1]


class SweepTests(unittest.TestCase):
    def test_delta_sweep_preserves_seed_and_labels_value(self) -> None:
        scenario = load_scenario(ROOT / "config/scenarios/hover.yaml")
        scenario.max_time = 0.20
        outcomes = run_parameter_sweep(
            scenario,
            ROOT / "config/controller.yaml",
            "delta",
            [0.03, 0.10],
            ["ccmpc"],
            trials=1,
            seed=51,
        )
        self.assertEqual([item.value for item in outcomes], [0.03, 0.10])
        self.assertTrue(all(item.result.seed == 51 for item in outcomes))

    def test_invalid_delta_is_rejected(self) -> None:
        scenario = load_scenario(ROOT / "config/scenarios/hover.yaml")
        with self.assertRaises(ValueError):
            run_parameter_sweep(
                scenario,
                ROOT / "config/controller.yaml",
                "delta",
                [0.0],
                ["ccmpc"],
                trials=1,
            )


if __name__ == "__main__":
    unittest.main()
