from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from experiments.manager import aggregate_results, save_experiment
from simulation.config import load_scenario
from simulation.runner import SimulationRunner


ROOT = Path(__file__).resolve().parents[1]


class ExperimentTests(unittest.TestCase):
    def _short_hover(self):
        scenario = load_scenario(ROOT / "config/scenarios/hover.yaml")
        scenario.max_time = 0.20
        return scenario

    def test_run_serializes_prediction_diagnostics(self) -> None:
        scenario = self._short_hover()
        result = SimulationRunner(
            scenario, ROOT / "config/controller.yaml", seed=77
        ).run()
        self.assertEqual(result.covariances.shape[1:], (9, 9))
        self.assertEqual(result.predicted_trajectories.shape[2], 9)
        self.assertEqual(result.predicted_controls.shape[2], 4)
        self.assertEqual(len(result.chance_residuals), len(result.times))
        self.assertTrue(result.cost_terms)

    def test_complete_experiment_bundle_is_valid(self) -> None:
        scenario = self._short_hover()
        results = [
            SimulationRunner(
                scenario,
                ROOT / "config/controller.yaml",
                mode=mode,
                seed=88,
            ).run()
            for mode in ("deterministic", "ccmpc")
        ]
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = save_experiment(
                results,
                scenario,
                ROOT / "config/controller.yaml",
                temporary,
                run_id="unit-test",
            )
            self.assertTrue(artifacts.static_report.exists())
            self.assertTrue(artifacts.interactive_report.exists())
            manifest = yaml.safe_load(artifacts.manifest.read_text(encoding="utf-8"))
            metrics = json.loads(artifacts.metrics.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 1)
            self.assertIn("paired_comparison", metrics)
            prediction = next(artifacts.directory.glob("*-predictions.npz"))
            with np.load(prediction) as content:
                self.assertIn("predicted_trajectories", content.files)

    def test_aggregate_confidence_intervals_are_bounded(self) -> None:
        scenario = self._short_hover()
        result = SimulationRunner(
            scenario, ROOT / "config/controller.yaml", seed=12
        ).run()
        summary = aggregate_results([result])["ccmpc"]
        low, high = summary["success_rate_ci95"]
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)


if __name__ == "__main__":
    unittest.main()
