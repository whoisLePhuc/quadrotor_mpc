"""Unit tests for simulation runner and data containers.

Target module:
    simulation.runner

These tests verify:
- SimulationHistory computed properties (success, trajectory_length, etc.)
- SimulationSummary string formatting
- SimulationRunner config loading and obstacle creation
- MonteCarloSummary aggregation
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from simulation.runner import (
    SimulationHistory,
    SimulationSummary,
    MonteCarloSummary,
    SimulationRunner,
)


def make_dummy_history(config: Any = None) -> SimulationHistory:
    """Create a minimal SimulationHistory for testing computed properties."""
    history = SimulationHistory(config=config)
    # Add 10 steps moving in a straight line
    for i in range(10):
        state = np.array([i * 0.1, i * 0.05, 1.0, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0],
                         dtype=np.float64)
        command = np.array([0.05, 0.03, 0.1, 0.0], dtype=np.float64)
        history.states.append(state)
        history.commands.append(command)
        history.solve_times.append(5.0 + i * 0.5)
        history.solver_statuses.append("optimal")
        history.collisions.append(False)
        history.feasibility.append(True)
    history.goal_reached = True
    return history


class TestSimulationHistory:

    def test_success_when_goal_reached_no_collision(self) -> None:
        history = make_dummy_history()
        history.goal_reached = True
        history.collisions = [False] * 10
        assert history.success is True

    def test_fails_when_collision_detected(self) -> None:
        history = make_dummy_history()
        history.collisions[5] = True
        assert history.success is False

    def test_fails_when_goal_not_reached(self) -> None:
        history = make_dummy_history()
        history.goal_reached = False
        assert history.success is False

    def test_trajectory_length_computed_correctly(self) -> None:
        history = make_dummy_history()
        # Each step moves 0.1 in x, 0.05 in y: per-step distance = sqrt(0.01+0.0025)
        expected_per_step = np.sqrt(0.1**2 + 0.05**2)
        expected_total = expected_per_step * 9  # 9 diffs for 10 points
        assert history.trajectory_length == pytest.approx(expected_total, rel=1e-6)

    def test_avg_solve_time(self) -> None:
        history = make_dummy_history()
        # Solve times: 5.0, 5.5, 6.0, ..., 9.5
        expected_avg = np.mean([5.0 + i * 0.5 for i in range(10)])
        assert history.avg_solve_time == pytest.approx(expected_avg)

    def test_total_steps(self) -> None:
        history = make_dummy_history()
        assert history.total_steps == 10

    def test_empty_history(self) -> None:
        history = SimulationHistory(config=None)  # type: ignore[arg-type]
        assert history.success is False
        assert history.trajectory_length == 0.0
        assert history.avg_solve_time == 0.0
        assert history.total_steps == 0


class TestSimulationSummary:

    def test_summary_includes_key_metrics(self) -> None:
        summary = SimulationSummary(
            success=True,
            goal_reached=True,
            collision_detected=False,
            total_timesteps=100,
            trajectory_length=12.5,
            min_separation=0.3,
            avg_solve_time_ms=45.2,
            max_solve_time_ms=120.0,
            solver_failures=0,
            seed=42,
            controller_type="ccmpc",
        )
        text = str(summary)
        assert "SUCCESS" in text
        assert "12.50 m" in text
        assert "0.300 m" in text
        assert "45.2 ms" in text
        assert "120.0 ms" in text

    def test_failure_summary(self) -> None:
        summary = SimulationSummary(
            success=False,
            goal_reached=False,
            collision_detected=True,
            total_timesteps=50,
            trajectory_length=3.2,
            min_separation=0.0,
            avg_solve_time_ms=30.0,
            max_solve_time_ms=80.0,
            solver_failures=5,
            seed=1,
            controller_type="ccmpc",
        )
        text = str(summary)
        assert "FAIL" in text
        assert "yes" in text  # collision detected


class TestMonteCarloSummary:

    def test_summary_aggregation(self) -> None:
        results = [
            SimulationSummary(True, True, False, 100, 10.0, 0.5, 40.0, 80.0, 0, 1, "ccmpc"),
            SimulationSummary(True, True, False, 100, 11.0, 0.4, 42.0, 90.0, 0, 2, "ccmpc"),
            SimulationSummary(False, False, True, 50, 5.0, 0.0, 45.0, 100.0, 3, 3, "ccmpc"),
        ]
        mc = MonteCarloSummary(
            trials=3,
            success_rate=float(2.0 / 3.0),
            mean_min_separation=float(np.mean([0.5, 0.4, 0.0])),
            std_min_separation=float(np.std([0.5, 0.4, 0.0])),
            mean_trajectory_length=float(np.mean([10.0, 11.0, 5.0])),
            mean_avg_solve_time=float(np.mean([40.0, 42.0, 45.0])),
            results=results,
        )
        text = str(mc)
        assert "67%" in text or "66%" in text  # 2/3 success rate
        assert "Trials" in text or "trials" in text


class TestSimulationRunner:

    def test_load_config_two_static(self) -> None:
        runner = SimulationRunner()
        runner.load_config("2.Code/config/scenarios/two_static.yaml")
        assert runner.config is not None
        assert runner.mpc is not None
        assert runner.dynamics is not None
        assert runner.obstacle_manager is not None
        assert len(runner.obstacle_manager.obstacles) == 2

    def test_load_config_one_moving(self) -> None:
        runner = SimulationRunner()
        runner.load_config("2.Code/config/scenarios/one_moving.yaml")
        assert runner.config is not None
        assert runner.obstacle_manager is not None
        assert len(runner.obstacle_manager.obstacles) == 1

    def test_load_config_blocked_path(self) -> None:
        runner = SimulationRunner()
        runner.load_config("2.Code/config/scenarios/blocked_path.yaml")
        assert runner.config is not None
        assert runner.obstacle_manager is not None
        assert len(runner.obstacle_manager.obstacles) == 3

    def test_runner_raises_without_config(self) -> None:
        runner = SimulationRunner()
        with pytest.raises(RuntimeError, match="No config loaded"):
            runner.run()

    def test_run_returns_history(self) -> None:
        runner = SimulationRunner("2.Code/config/scenarios/two_static.yaml")
        history = runner.run(seed=42)
        assert history is not None
        assert history.total_steps > 0
        assert len(history.states) > 0
        assert len(history.commands) > 0
        assert len(history.solve_times) > 0

    def test_seeded_reproducibility(self) -> None:
        runner = SimulationRunner("2.Code/config/scenarios/two_static.yaml")
        h1 = runner.run(seed=42)
        h2 = runner.run(seed=42)
        # Same seed should produce same trajectory length
        assert h1.trajectory_length == pytest.approx(h2.trajectory_length, rel=1e-3)

    def test_different_seeds_different_results(self) -> None:
        runner = SimulationRunner("2.Code/config/scenarios/two_static.yaml")
        h1 = runner.run(seed=42)
        h2 = runner.run(seed=99)
        # Could produce the same by luck but extremely unlikely
        assert h1.trajectory_length != pytest.approx(h2.trajectory_length, rel=1e-3)

    def test_monte_carlo_mode(self) -> None:
        runner = SimulationRunner("2.Code/config/scenarios/two_static.yaml")
        mc = runner.run_monte_carlo(num_trials=3, base_seed=42)
        assert mc.trials == 3
        assert mc.success_rate >= 0.0
        assert mc.success_rate <= 1.0
        assert mc.mean_trajectory_length > 0
        assert len(mc.results) == 3

    @pytest.mark.slow
    def test_performance_100_steps_under_10s(self) -> None:
        runner = SimulationRunner("2.Code/config/scenarios/two_static.yaml")
        import time
        t0 = time.perf_counter()
        runner.run(seed=42)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"Simulation took {elapsed:.1f}s, expected <10s"
