from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mujoco_native import load_native_mujoco_config
from native_telemetry import (
    NativeRunRecorder,
    RecordingOptions,
    load_native_recording,
    step_to_sample,
)
from obstacle_motion import (
    normalize_obstacle,
    obstacle_position,
    predict_obstacle_positions,
)
from runtime_control import CommandName, LocalCommandQueue, RuntimeCommand

CODE_ROOT = Path(__file__).resolve().parents[1]


class ObstacleMotionTests(unittest.TestCase):
    def test_constant_velocity_position_and_prediction(self):
        obstacle = normalize_obstacle(
            {
                "type": "dynamic",
                "radius": 0.2,
                "motion": {
                    "type": "constant_velocity",
                    "initial_position": [1.0, -0.5, 2.0],
                    "velocity": [0.2, 0.4, -0.1],
                },
            }
        )
        np.testing.assert_allclose(obstacle_position(obstacle, 2.5), [1.5, 0.5, 1.75])
        prediction = predict_obstacle_positions([obstacle], 1.0, 4, 0.25)
        self.assertEqual(prediction.shape, (1, 4, 3))
        np.testing.assert_allclose(prediction[0, 0], [1.2, -0.1, 1.9])
        np.testing.assert_allclose(obstacle_position(obstacle, np.array([[2.5]])), [1.5, 0.5, 1.75])

    def test_three_axis_sinusoid(self):
        obstacle = normalize_obstacle(
            {
                "type": "dynamic",
                "radius": 0.2,
                "motion": {
                    "type": "sinusoidal",
                    "center": [1.0, 2.0, 3.0],
                    "amplitude": [0.5, 1.0, 1.5],
                    "period_s": 4.0,
                },
            }
        )
        np.testing.assert_allclose(obstacle_position(obstacle, 1.0), [1.5, 3.0, 4.5])

    def test_waypoints_interpolate_and_repeat(self):
        obstacle = normalize_obstacle(
            {
                "type": "dynamic",
                "radius": 0.2,
                "motion": {
                    "type": "waypoints",
                    "repeat": True,
                    "points": [
                        {"time_s": 0.0, "position": [0, 0, 1]},
                        {"time_s": 2.0, "position": [2, 0, 1]},
                        {"time_s": 4.0, "position": [2, 2, 1]},
                    ],
                },
            }
        )
        np.testing.assert_allclose(obstacle_position(obstacle, 1.0), [1, 0, 1])
        np.testing.assert_allclose(obstacle_position(obstacle, 5.0), [1, 0, 1])

    def test_legacy_dynamic_schema_remains_supported(self):
        obstacle = normalize_obstacle(
            {"type": "dynamic", "x": 2.0, "z": 1.5, "radius": 0.3, "amp": 1.0, "period": 4.0}
        )
        np.testing.assert_allclose(obstacle_position(obstacle, 1.0), [2.0, 1.0, 1.5])

    def test_invalid_waypoint_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            normalize_obstacle(
                {
                    "type": "dynamic",
                    "radius": 0.2,
                    "motion": {
                        "type": "waypoints",
                        "points": [
                            {"time_s": 1.0, "position": [0, 0, 0]},
                            {"time_s": 1.0, "position": [1, 0, 0]},
                        ],
                    },
                }
            )


class InteractiveConfigurationTests(unittest.TestCase):
    def test_dynamic_scenario_and_round_trip(self):
        config = load_native_mujoco_config(CODE_ROOT / "config" / "mujoco_native_dynamic.yaml")
        self.assertEqual(len(config.obstacles), 3)
        self.assertEqual(config.obstacles[1]["motion"]["type"], "constant_velocity")
        self.assertTrue(config.panel.enabled)
        self.assertTrue(config.recording.enabled)
        round_trip = type(config).from_mapping(config.to_mapping())
        self.assertEqual(round_trip, config)

    def test_command_queue_preserves_order(self):
        queue = LocalCommandQueue()
        queue.put(RuntimeCommand(CommandName.TOGGLE_PAUSE, source="test"))
        queue.put(RuntimeCommand(CommandName.STEP, source="test"))
        self.assertEqual(
            [command.name for command in queue.drain()],
            [CommandName.TOGGLE_PAUSE, CommandName.STEP],
        )
        self.assertEqual(queue.drain(), [])

    def test_run_again_command_round_trip(self):
        message = RuntimeCommand(CommandName.RUN_AGAIN, source="test").as_message()
        self.assertEqual(RuntimeCommand.from_message(message).name, CommandName.RUN_AGAIN)


class NativeRecordingTests(unittest.TestCase):
    def test_recording_round_trip_uses_numeric_arrays(self):
        step = SimpleNamespace(
            step_index=1,
            time_s=0.05,
            state_13=np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float),
            control=np.array([0.01, 0.0, 0.0, 0.0]),
            goal_distance_m=2.0,
            min_clearance_m=0.4,
            solver_time_ms=12.5,
            collided=False,
            paused=False,
            predicted_positions=np.zeros((3, 3), dtype=float),
            obstacle_predictions=np.zeros((1, 3, 3), dtype=float),
            predicted_covariances=np.repeat(
                (np.eye(12) * 0.01)[None, :, :],
                3,
                axis=0,
            ),
            predicted_obstacle_covariances=np.repeat(
                (np.eye(6) * 0.02)[None, None, :, :],
                3,
                axis=0,
            ),
            chance_margins=np.array([[0.2], [0.1], [-0.01]]),
            risk_allocations=np.full((3, 1), 0.05),
            slacks=np.array([[0.0], [0.0], [0.01]]),
            projected_uncertainties=np.full((3, 1), 0.12),
            tightened_safety_radii=np.full((3, 1), 0.85),
            solver_status="SOLVED_WITH_SLACK",
            risk_semantics="joint",
            risk_allocation_method="uniform",
            risk_budget_total=0.10,
            risk_budget_allocated=0.10,
            risk_budget_remaining=0.0,
            risk_constraint_count=3,
            risk_budget_status="BUDGET_OK",
        )
        sample = step_to_sample(step)
        with tempfile.TemporaryDirectory() as directory:
            recorder = NativeRunRecorder(
                RecordingOptions(enabled=True, output_dir="records"),
                "test scenario",
                {"name": "test scenario"},
                base_dir=directory,
            )
            recorder.record_step(step, sample)
            recorder.record_event("pause", 0.05, source="test")
            recorder.write_snapshot(sample)
            run_dir = recorder.finalize(
                {
                    "termination_reason": "completed",
                    "collided": False,
                    "clearance": np.array([0.4]),
                }
            )
            self.assertIsNotNone(run_dir)
            loaded = load_native_recording(run_dir)
            np.testing.assert_allclose(loaded["states"][0], step.state_13)
            self.assertEqual(loaded["predicted_positions"].dtype, np.float64)
            self.assertEqual(loaded["obstacle_predictions"].shape, (1, 1, 3, 3))
            self.assertEqual(
                loaded["predicted_error_covariance_horizons"].shape,
                (1, 3, 12, 12),
            )
            self.assertEqual(
                loaded["predicted_obstacle_covariance_horizons"].shape,
                (1, 3, 1, 6, 6),
            )
            self.assertEqual(loaded["chance_residual_horizons"].shape, (1, 3, 1))
            self.assertEqual(loaded["risk_allocation_horizons"].shape, (1, 3, 1))
            self.assertEqual(loaded["slack_horizons"].shape, (1, 3, 1))
            self.assertEqual(
                loaded["projected_uncertainty_horizons"].shape,
                (1, 3, 1),
            )
            self.assertEqual(
                loaded["tightened_safety_radius_horizons"].shape,
                (1, 3, 1),
            )
            self.assertEqual(sample["solver_status"], "SOLVED_WITH_SLACK")
            self.assertAlmostEqual(sample["minimum_chance_residual_m"], -0.01)
            self.assertAlmostEqual(sample["maximum_slack_m"], 0.01)
            self.assertEqual(sample["risk_semantics"], "joint")
            self.assertEqual(sample["risk_allocation_method"], "uniform")
            self.assertAlmostEqual(sample["risk_budget_total"], 0.10)
            self.assertAlmostEqual(sample["risk_budget_allocated"], 0.10)
            self.assertEqual(sample["risk_constraint_count"], 3)
            self.assertEqual(sample["risk_budget_status"], "BUDGET_OK")
            self.assertEqual(loaded["rows"][0]["risk_semantics"], "joint")
            self.assertEqual(loaded["rows"][0]["risk_budget_status"], "BUDGET_OK")
            self.assertIsNotNone(sample["horizon_terminal_position_sigma"])
            self.assertTrue((run_dir / "snapshot-001.json").is_file())


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("mujoco", "casadi", "do_mpc")),
    "optional NMPC/MuJoCo dependencies are not installed",
)
class InteractiveLoopTests(unittest.TestCase):
    def test_spherical_chance_constraint_reaches_native_solver(self):
        from run_coupled import run_coupled_simulation

        config = load_native_mujoco_config(CODE_ROOT / "config" / "mujoco_native_ccmpc.yaml")
        result = run_coupled_simulation(
            x0_vals=config.start,
            goal_pos=config.goal_position,
            goal_euler=config.goal_euler,
            bounds=config.bounds,
            obstacles=[dict(item) for item in config.obstacles],
            margin=config.safety_margin,
            sim_seconds=0.10,
            mpc_dt=config.mpc_timestep_s,
            n_horizon=4,
            max_iter=30,
            mj_dt=config.mujoco_timestep_s,
            estimation_options=config.estimation,
            covariance_options=config.covariance_propagation,
            chance_options=config.chance_constraints,
        )
        self.assertEqual(len(result["t"]), 2)
        expected_cell_risk = 0.10 / ((4 + 1) * len(config.obstacles))
        np.testing.assert_allclose(
            result["risk_allocation_horizon"],
            expected_cell_risk,
        )
        np.testing.assert_allclose(result["risk_budget_total"], 0.10)
        np.testing.assert_allclose(result["risk_budget_allocated"], 0.10)
        np.testing.assert_allclose(result["risk_budget_remaining"], 0.0, atol=1e-12)
        np.testing.assert_array_equal(result["risk_semantics"], "joint")
        np.testing.assert_array_equal(result["risk_budget_status"], "BUDGET_OK")
        nominal_radii = np.asarray(
            [
                obstacle["radius"] + config.safety_margin + 0.03
                for obstacle in config.obstacles
            ]
        )
        self.assertTrue(
            np.all(result["tightened_safety_radius_horizon"] >= nominal_radii)
        )
        self.assertGreater(float(np.max(result["projected_uncertainty_horizon"])), 0.0)
        self.assertTrue(
            set(result["solver_status"]).issubset(
                {"SOLVED_SAFE", "SOLVED_WITH_SLACK"}
            )
        )

    def test_reset_pause_step_and_stop_commands(self):
        from run_coupled import run_coupled_simulation

        class ScriptedRuntime:
            def __init__(self):
                self.poll_count = 0
                self.steps = []
                self.reset_count = 0
                self.closed = False

            def open(self, _plant, _context):
                return None

            def is_running(self):
                return True

            def poll_commands(self):
                self.poll_count += 1
                script = {
                    1: [RuntimeCommand(CommandName.RESET, source="test")],
                    2: [RuntimeCommand(CommandName.TOGGLE_PAUSE, source="test")],
                    3: [RuntimeCommand(CommandName.STEP, source="test")],
                    4: [RuntimeCommand(CommandName.STOP, source="test")],
                }
                return script.get(self.poll_count, [])

            def on_step(self, step):
                self.steps.append(step)
                return True

            def on_idle(self, _paused):
                return None

            def on_reset(self):
                self.reset_count += 1

            def close(self):
                self.closed = True

        config = load_native_mujoco_config(CODE_ROOT / "config" / "mujoco_native.yaml")
        runtime = ScriptedRuntime()
        result = run_coupled_simulation(
            x0_vals=config.start,
            goal_pos=config.goal_position,
            goal_euler=config.goal_euler,
            bounds=config.bounds,
            obstacles=[dict(item) for item in config.obstacles],
            margin=config.safety_margin,
            sim_seconds=1.0,
            mpc_dt=config.mpc_timestep_s,
            n_horizon=4,
            max_iter=20,
            mj_dt=config.mujoco_timestep_s,
            runtime=runtime,
        )
        self.assertEqual(result["termination_reason"], "user_stopped")
        self.assertEqual(runtime.reset_count, 1)
        self.assertEqual(len(runtime.steps), 1)
        self.assertTrue(runtime.steps[0].paused)
        self.assertEqual(runtime.steps[0].solver_status, "SOLVED_DETERMINISTIC")
        self.assertEqual(runtime.steps[0].predicted_covariances.shape[1:], (12, 12))
        self.assertEqual(
            runtime.steps[0].predicted_obstacle_covariances.shape[2:],
            (6, 6),
        )
        self.assertEqual(runtime.steps[0].chance_margins.shape[1], len(config.obstacles))
        self.assertEqual(runtime.steps[0].risk_allocations.shape, runtime.steps[0].slacks.shape)
        self.assertTrue(runtime.closed)

    def test_goal_completion_holds_until_run_again_then_stop(self):
        from run_coupled import run_coupled_simulation

        class ScriptedRuntime:
            def __init__(self):
                self.poll_count = 0
                self.steps = []
                self.completions = []
                self.reset_count = 0
                self.closed = False

            def open(self, _plant, _context):
                return None

            def is_running(self):
                return True

            def poll_commands(self):
                self.poll_count += 1
                return {
                    2: [RuntimeCommand(CommandName.RUN_AGAIN, source="test")],
                    4: [RuntimeCommand(CommandName.STOP, source="test")],
                }.get(self.poll_count, [])

            def on_step(self, step):
                self.steps.append(step)
                return True

            def on_idle(self, _paused):
                return None

            def on_reset(self):
                self.reset_count += 1

            def on_completed(self, reason):
                self.completions.append(reason)

            def close(self):
                self.closed = True

        config = load_native_mujoco_config(CODE_ROOT / "config" / "mujoco_native.yaml")
        start_goal = {axis: config.start[axis] for axis in ("x", "y", "z")}
        runtime = ScriptedRuntime()
        result = run_coupled_simulation(
            x0_vals=config.start,
            goal_pos=start_goal,
            goal_euler={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            bounds=config.bounds,
            obstacles=[],
            margin=config.safety_margin,
            sim_seconds=1.0,
            mpc_dt=config.mpc_timestep_s,
            n_horizon=4,
            max_iter=20,
            mj_dt=config.mujoco_timestep_s,
            runtime=runtime,
            stop_on_goal=True,
            goal_tolerance=1.0,
        )
        self.assertEqual(result["termination_reason"], "user_stopped")
        self.assertEqual(runtime.completions, ["goal_reached", "goal_reached"])
        self.assertEqual(runtime.reset_count, 1)
        self.assertEqual(len(runtime.steps), 2)
        self.assertTrue(runtime.closed)

    def test_replay_does_not_invoke_controller(self):
        from native_replay import replay_native_recording

        class ReplayRuntime:
            def __init__(self):
                self.steps = []
                self.closed = False

            def open(self, _plant, _context):
                return None

            def is_running(self):
                return True

            def poll_commands(self):
                return []

            def on_step(self, step):
                self.steps.append(step)
                return True

            def on_idle(self, _paused):
                return None

            def on_reset(self):
                return None

            def close(self):
                self.closed = True

        config = load_native_mujoco_config(CODE_ROOT / "config" / "mujoco_native.yaml")
        state = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
        row = {
            "step_index": "1",
            "time_s": "0.05",
            "thrust_deviation": "0.0",
            "tau_x": "0.0",
            "tau_y": "0.0",
            "tau_z": "0.0",
            "goal_distance_m": "3.5",
            "min_clearance_m": "1.0",
            "solver_time_ms": "10.0",
            "collided": "0",
        }
        obstacle_predictions = predict_obstacle_positions(
            config.obstacles, 0.05, config.horizon_steps + 1, config.mpc_timestep_s
        )[None, ...]
        recording = {
            "rows": [row],
            "states": state[None, :],
            "predicted_positions": np.empty((1, 0, 3)),
            "obstacle_predictions": obstacle_predictions,
        }
        runtime = ReplayRuntime()
        result = replay_native_recording(config, recording, runtime)
        self.assertEqual(result["termination_reason"], "replay_completed")
        self.assertEqual(len(runtime.steps), 1)
        self.assertTrue(runtime.closed)


if __name__ == "__main__":
    unittest.main()
