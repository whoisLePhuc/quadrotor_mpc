"""Unit tests for matplotlib visualizer.

Target module:
    simulation.visualizer

These tests verify:
- plot_trajectory() produces a valid PNG file
- plot_summary_panel() produces a valid PNG file
- MatplotlibVisualizer handles empty/malformed history
- Obstacle ellipse extraction from config
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simulation.runner import SimulationHistory
from simulation.visualizer import MatplotlibVisualizer


def make_minimal_history() -> SimulationHistory:
    """Create a minimal valid history for visualizer testing."""
    history = SimulationHistory(config=None)
    for i in range(5):
        state = np.array([float(i) * 0.5, float(i) * 0.3, 1.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0],
                         dtype=np.float64)
        command = np.array([0.05, 0.03, 0.1, 0.0], dtype=np.float64)
        history.states.append(state)
        history.commands.append(command)
        history.solve_times.append(10.0)
        history.solver_statuses.append("optimal")
        history.collisions.append(False)
        history.feasibility.append(True)
        history.covariances.append(np.eye(9, dtype=np.float64) * 0.01)
    history.goal_reached = True
    return history


class TestMatplotlibVisualizer:

    def test_plot_trajectory_creates_png(self, tmp_path: Path) -> None:
        history = make_minimal_history()
        viz = MatplotlibVisualizer(history, dpi=50)
        output = tmp_path / "test_traj.png"
        result = viz.plot_trajectory(str(output))
        assert result.exists()
        assert result.suffix == ".png"
        assert result.stat().st_size > 100

    def test_plot_trajectory_default_path(self) -> None:
        history = make_minimal_history()
        viz = MatplotlibVisualizer(history, dpi=50)
        result = viz.plot_trajectory()
        assert result.exists()

    def test_plot_summary_panel_creates_png(self, tmp_path: Path) -> None:
        history = make_minimal_history()
        viz = MatplotlibVisualizer(history, dpi=50)
        output = tmp_path / "test_summary.png"
        result = viz.plot_summary_panel(str(output))
        assert result.exists()
        assert result.stat().st_size > 100

    def test_empty_history_raises(self) -> None:
        history = SimulationHistory(config=None)
        viz = MatplotlibVisualizer(history, dpi=50)
        # Should not crash; might produce empty plot
        result = viz.plot_trajectory()
        assert result.exists()

    def test_obstacle_ellipse_extraction_no_config(self) -> None:
        history = make_minimal_history()
        viz = MatplotlibVisualizer(history, dpi=50)
        ellipses = viz._get_obstacle_ellipses()
        assert ellipses == []

    def test_failure_point_none_when_all_feasible(self) -> None:
        history = make_minimal_history()
        viz = MatplotlibVisualizer(history, dpi=50)
        assert viz._find_failure_point() is None

    def test_failure_point_detected(self) -> None:
        history = make_minimal_history()
        history.feasibility[3] = False
        viz = MatplotlibVisualizer(history, dpi=50)
        assert viz._find_failure_point() == 3
