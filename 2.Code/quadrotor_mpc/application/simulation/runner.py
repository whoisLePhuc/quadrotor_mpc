"""Closed-loop experiment runner with uncertainty and solver diagnostics."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from quadrotor_mpc.control.ccmpc.dynamics import QuadrotorDynamics
from quadrotor_mpc.control.ccmpc.obstacle import EllipsoidalObstacle, ObstacleManager
from quadrotor_mpc.control.ccmpc.risk import collision_clearance

from .config import ScenarioConfig
from .controllers import Controller, ControlResult, _deep_update, load_controller
from .estimators import ExtendedKalmanEstimator
from .metrics import SimulationMetrics, compute_metrics

Array = npt.NDArray[np.float64]


@dataclass(slots=True)
class SimulationResult:
    """All data needed to reproduce, inspect and report one closed-loop run."""

    scenario_name: str
    mode: str
    backend: str
    seed: int
    controller_dt: float
    times: Array
    states: Array
    estimated_states: Array
    reference_positions: Array
    controls: Array
    covariances: Array
    clearances: Array
    chance_residuals: Array
    chance_slacks: Array
    solver_times_ms: Array
    solver_iterations: Array
    solver_statuses: list[str]
    obstacle_positions: Array
    control_update_times: Array
    predicted_trajectories: Array
    predicted_controls: Array
    cost_terms: dict[str, Array]
    events: list[dict[str, Any]]
    metrics: SimulationMetrics

    @property
    def stem(self) -> str:
        return f"{self.scenario_name}-{self.mode}-seed-{self.seed}"

    def save(self, output_directory: str | Path) -> tuple[Path, Path]:
        """Save summary, time series, predictions and event log."""
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        summary_path = output / f"{self.stem}-summary.json"
        csv_path = output / f"{self.stem}-timeseries.csv"
        predictions_path = output / f"{self.stem}-predictions.npz"
        events_path = output / f"{self.stem}-events.jsonl"

        summary_path.write_text(
            json.dumps(
                {
                    "scenario": self.scenario_name,
                    "mode": self.mode,
                    "backend": self.backend,
                    "seed": self.seed,
                    "controller_timestep_s": self.controller_dt,
                    "metrics": self.metrics.to_dict(),
                    "artifacts": {
                        "timeseries": csv_path.name,
                        "predictions": predictions_path.name,
                        "events": events_path.name,
                    },
                },
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        cost_names = sorted(self.cost_terms)
        header = [
            "time_s",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "roll",
            "pitch",
            "yaw",
            "x_est",
            "y_est",
            "z_est",
            "vx_est",
            "vy_est",
            "vz_est",
            "roll_est",
            "pitch_est",
            "yaw_est",
            "x_ref",
            "y_ref",
            "z_ref",
            "phi_cmd",
            "theta_cmd",
            "vz_cmd",
            "yaw_rate_cmd",
            "sigma_x",
            "sigma_y",
            "sigma_z",
            "clearance_m",
            "chance_residual",
            "chance_slack",
            "solver_ms",
            "solver_iterations",
        ] + [f"cost_{name}" for name in cost_names]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for index in range(len(self.times)):
                writer.writerow(
                    [
                        self.times[index],
                        *self.states[index],
                        *self.estimated_states[index],
                        *self.reference_positions[index],
                        *self.controls[index],
                        *np.sqrt(np.maximum(np.diag(self.covariances[index])[:3], 0.0)),
                        self.clearances[index],
                        self.chance_residuals[index],
                        self.chance_slacks[index],
                        self.solver_times_ms[index],
                        int(self.solver_iterations[index]),
                        *[self.cost_terms[name][index] for name in cost_names],
                    ]
                )

        np.savez_compressed(
            predictions_path,
            control_update_times=self.control_update_times,
            predicted_trajectories=self.predicted_trajectories,
            predicted_controls=self.predicted_controls,
        )
        with events_path.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event, allow_nan=False) + "\n")
        return summary_path, csv_path


class SimulationRunner:
    """Run one reproducible receding-horizon simulation."""

    def __init__(
        self,
        scenario: ScenarioConfig,
        controller_config: str | Path,
        mode: str = "ccmpc",
        backend: str = "scipy",
        seed: int | None = None,
        controller_override: dict | None = None,
    ) -> None:
        self.scenario = scenario
        self.controller_config = Path(controller_config)
        self.mode = mode
        self.backend = backend
        self.seed = scenario.seed if seed is None else int(seed)
        self.controller_override = controller_override or {}
        self.rng = np.random.default_rng(self.seed)

        with self.controller_config.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        _deep_update(config, self.controller_override)
        model = dict(config["model"]["quadrotor"])
        mismatch = scenario.model_mismatch
        model["kD"] *= mismatch.drag_scale
        model["tau_phi"] *= mismatch.attitude_time_scale
        model["tau_theta"] *= mismatch.attitude_time_scale
        model["tau_vz"] *= mismatch.vertical_time_scale
        self.plant = QuadrotorDynamics(**model)
        self.controller: Controller = load_controller(
            self.controller_config,
            mode,
            backend,
            self.controller_override,
        )
        self.controller.reset()

        self.obstacles = ObstacleManager(
            [
                EllipsoidalObstacle(
                    position=item.position,
                    size=item.size,
                    yaw=item.yaw,
                    velocity=item.velocity,
                    pos_uncertainty=item.pos_uncertainty,
                    vel_uncertainty=item.vel_uncertainty,
                )
                for item in scenario.obstacles
            ]
        )

        uncertainty_cfg = config["controller"]["uncertainty"]
        self.covariance = np.diag(
            np.concatenate(
                [
                    [uncertainty_cfg["init_pos_noise"] ** 2] * 3,
                    [uncertainty_cfg["init_vel_noise"] ** 2] * 3,
                    [uncertainty_cfg["init_att_noise"] ** 2] * 3,
                ]
            )
        )
        self.process_covariance = np.diag(
            np.concatenate(
                [
                    [uncertainty_cfg["process_noise_pos"] ** 2] * 3,
                    [uncertainty_cfg["process_noise_vel"] ** 2] * 3,
                    [uncertainty_cfg["process_noise_att"] ** 2] * 3,
                ]
            )
        )
        noise = scenario.noise
        self.measurement_covariance = np.diag(
            np.array(
                [max(noise.measurement_pos, 1e-6) ** 2] * 3
                + [max(noise.measurement_vel, 1e-6) ** 2] * 3
                + [max(noise.measurement_att, 1e-6) ** 2] * 3
            )
        )
        self.estimator = ExtendedKalmanEstimator(
            self.controller.dynamics,
            self.process_covariance,
            self.measurement_covariance,
        )
        limits = config["controller"]["limits"]
        self.control_limits = np.array(
            [
                limits["max_roll"],
                limits["max_pitch"],
                limits["max_vert_vel"],
                limits["max_yaw_rate"],
            ],
            dtype=float,
        )

    def _measure(self, state: Array) -> Array:
        noise = self.scenario.noise
        sigma = np.array(
            [noise.measurement_pos] * 3 + [noise.measurement_vel] * 3 + [noise.measurement_att] * 3
        )
        return state + self.rng.normal(0.0, sigma)

    def _apply_process_noise(self, state: Array, dt: float) -> Array:
        noise = self.scenario.noise
        disturbed = state.copy()
        disturbed[3:6] += self.rng.normal(0.0, noise.process_vel * math.sqrt(dt), 3)
        disturbed[6:9] += self.rng.normal(0.0, noise.process_att * math.sqrt(dt), 3)
        disturbed[8] = (disturbed[8] + math.pi) % (2.0 * math.pi) - math.pi
        return disturbed

    def _minimum_clearance(self, position: Array) -> tuple[float, bool]:
        if not self.obstacles.obstacles:
            return float("inf"), False
        clearances = [
            collision_clearance(
                position,
                obstacle.p_hat,
                obstacle.get_omega(getattr(self.controller, "mav_radius", 0.4)),
            )
            for obstacle in self.obstacles.obstacles
        ]
        minimum = min(clearances)
        return minimum, minimum <= 0.0

    def _reference_at(self, time_s: float) -> Array:
        if np.linalg.norm(self.scenario.goal - self.scenario.start[:3]) < 1e-12:
            return self.scenario.goal.copy()
        reference_duration = max(0.65 * self.scenario.max_time, self.controller.dt)
        fraction = min(1.0, time_s / reference_duration)
        return self.scenario.start[:3] + fraction * (self.scenario.goal - self.scenario.start[:3])

    def run(self) -> SimulationResult:
        scenario = self.scenario
        dt = scenario.sim_timestep
        maximum_steps = int(math.ceil(scenario.max_time / dt))
        state = scenario.start.copy()
        initial_measurement = self._measure(state)
        self.estimator.reset(initial_measurement, self.covariance)
        estimated = self.estimator.state.copy()
        command = np.zeros(4)
        next_control_time = 0.0
        last_control_result: ControlResult | None = None
        collided = False
        chance_violation_active = False

        times = [0.0]
        states = [state.copy()]
        estimates = [estimated.copy()]
        references = [self._reference_at(0.0)]
        controls = [command.copy()]
        covariances = [self.covariance.copy()]
        clearances: list[float] = []
        residuals = [float("nan")]
        slacks = [0.0]
        solve_times = [0.0]
        iterations = [0]
        statuses: list[str] = []
        obstacle_history = [[obs.p_hat.copy() for obs in self.obstacles.obstacles]]
        update_times: list[float] = []
        predictions: list[Array] = []
        predicted_controls: list[Array] = []
        cost_records: list[dict[str, float]] = [{}]
        events: list[dict[str, Any]] = [{"time_s": 0.0, "type": "simulation_started"}]

        clearance, initial_collision = self._minimum_clearance(state[:3])
        clearances.append(clearance)
        collided |= initial_collision
        if initial_collision:
            events.append({"time_s": 0.0, "type": "collision"})

        for step in range(maximum_steps):
            current_time = step * dt
            solved_this_step = False
            if current_time + 1e-12 >= next_control_time:
                last_control_result = self.controller.solve(
                    estimated,
                    scenario.goal,
                    self.obstacles,
                    self.estimator.covariance,
                )
                command = last_control_result.command
                statuses.append(last_control_result.status)
                update_times.append(current_time)
                predictions.append(last_control_result.trajectory.copy())
                predicted_controls.append(last_control_result.controls.copy())
                next_control_time = current_time + self.controller.dt
                solved_this_step = True

                if not last_control_result.status.startswith("optimal"):
                    events.append(
                        {
                            "time_s": current_time,
                            "type": "solver_warning",
                            "status": last_control_result.status,
                        }
                    )
                if last_control_result.solve_time_ms > self.controller.dt * 1000.0:
                    events.append(
                        {
                            "time_s": current_time,
                            "type": "deadline_miss",
                            "solve_time_ms": last_control_result.solve_time_ms,
                        }
                    )

            state = self.plant.discrete(state, command, dt)
            state = self._apply_process_noise(state, dt)
            self.estimator.predict(command, dt)
            self.estimator.update(self._measure(state))
            estimated = self.estimator.state.copy()
            self.covariance = self.estimator.covariance.copy()
            self.obstacles.update(dt)

            clearance, is_collision = self._minimum_clearance(state[:3])
            if is_collision and not collided:
                events.append({"time_s": (step + 1) * dt, "type": "collision"})
            collided |= is_collision
            elapsed = (step + 1) * dt
            current_residual = (
                last_control_result.min_chance_residual if last_control_result is not None else None
            )
            violates_chance = current_residual is not None and current_residual < 0.0
            if violates_chance and not chance_violation_active:
                events.append(
                    {
                        "time_s": elapsed,
                        "type": "chance_constraint_violation",
                        "residual": current_residual,
                    }
                )
            chance_violation_active = violates_chance

            times.append(elapsed)
            states.append(state.copy())
            estimates.append(estimated.copy())
            references.append(self._reference_at(elapsed))
            controls.append(command.copy())
            covariances.append(self.covariance.copy())
            clearances.append(clearance)
            residuals.append(
                float(current_residual) if current_residual is not None else float("nan")
            )
            slacks.append(last_control_result.max_slack if last_control_result else 0.0)
            solve_times.append(
                last_control_result.solve_time_ms
                if solved_this_step and last_control_result is not None
                else 0.0
            )
            iterations.append(
                last_control_result.iterations
                if solved_this_step and last_control_result is not None
                else 0
            )
            cost_records.append(dict(last_control_result.cost_terms) if last_control_result else {})
            obstacle_history.append([obs.p_hat.copy() for obs in self.obstacles.obstacles])

            goal_error = float(np.linalg.norm(state[:3] - scenario.goal))
            speed = float(np.linalg.norm(state[3:6]))
            if goal_error <= scenario.goal_threshold and speed <= scenario.stop_speed:
                events.append({"time_s": elapsed, "type": "goal_reached"})
                break

        events.append({"time_s": float(times[-1]), "type": "simulation_finished"})
        times_array = np.asarray(times)
        states_array = np.asarray(states)
        estimates_array = np.asarray(estimates)
        controls_array = np.asarray(controls)
        clearances_array = np.asarray(clearances)
        residuals_array = np.asarray(residuals)
        slacks_array = np.asarray(slacks)
        solve_times_array = np.asarray(solve_times)
        iterations_array = np.asarray(iterations, dtype=int)
        metrics = compute_metrics(
            states_array,
            times_array,
            scenario.goal,
            scenario.start[:3],
            clearances_array,
            slacks_array,
            solve_times_array[solve_times_array > 0.0],
            statuses,
            collided,
            scenario.goal_threshold,
            self.controller.dt,
            controls=controls_array,
            estimated_states=estimates_array,
            chance_residuals=residuals_array,
            solver_iterations=iterations_array,
            control_limits=self.control_limits,
        )

        obstacle_array = np.asarray(obstacle_history, dtype=float)
        if not self.obstacles.obstacles:
            obstacle_array = np.empty((len(times_array), 0, 3))
        prediction_array = np.asarray(predictions, dtype=float)
        predicted_control_array = np.asarray(predicted_controls, dtype=float)
        if not predictions:
            prediction_array = np.empty((0, 0, 9))
            predicted_control_array = np.empty((0, 0, 4))

        cost_names = sorted({key for record in cost_records for key in record})
        cost_terms = {
            name: np.asarray([record.get(name, float("nan")) for record in cost_records])
            for name in cost_names
        }
        return SimulationResult(
            scenario_name=scenario.name,
            mode=self.mode,
            backend=self.backend,
            seed=self.seed,
            controller_dt=float(self.controller.dt),
            times=times_array,
            states=states_array,
            estimated_states=estimates_array,
            reference_positions=np.asarray(references),
            controls=controls_array,
            covariances=np.asarray(covariances),
            clearances=clearances_array,
            chance_residuals=residuals_array,
            chance_slacks=slacks_array,
            solver_times_ms=solve_times_array,
            solver_iterations=iterations_array,
            solver_statuses=statuses,
            obstacle_positions=obstacle_array,
            control_update_times=np.asarray(update_times),
            predicted_trajectories=prediction_array,
            predicted_controls=predicted_control_array,
            cost_terms=cost_terms,
            events=events,
            metrics=metrics,
        )
