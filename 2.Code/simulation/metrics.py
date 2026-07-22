"""Research-grade metrics for controller comparison and Monte Carlo studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def point_to_segment_distance(point: Array, start: Array, goal: Array) -> float:
    """Euclidean distance from a 3-D point to the finite start-goal segment."""
    direction = goal - start
    denominator = float(direction @ direction)
    if denominator < 1e-12:
        return float(np.linalg.norm(point - start))
    alpha = float(np.clip((point - start) @ direction / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + alpha * direction)))


@dataclass(slots=True)
class SimulationMetrics:
    success: bool
    collision: bool
    duration_s: float
    time_to_goal_s: float | None
    steps: int
    final_error_m: float
    path_length_m: float
    tracking_rmse_m: float
    mean_speed_mps: float
    max_speed_mps: float
    min_clearance_m: float | None
    mean_clearance_m: float | None
    chance_violation_rate: float
    max_chance_slack: float
    control_effort: float
    control_smoothness: float
    saturation_rate: float
    max_estimation_error_m: float
    mean_solver_ms: float
    median_solver_ms: float
    p95_solver_ms: float
    max_solver_ms: float
    mean_solver_iterations: float
    p95_solver_iterations: float
    deadline_miss_rate: float
    real_time_feasible: bool
    solver_warnings: int

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_percentile(values: Array, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0


def compute_metrics(
    states: Array,
    times: Array,
    goal: Array,
    start: Array,
    clearances: Array,
    slacks: Array,
    solver_times: Array,
    statuses: list[str],
    collision: bool,
    goal_threshold: float,
    controller_timestep: float,
    *,
    controls: Array | None = None,
    estimated_states: Array | None = None,
    chance_residuals: Array | None = None,
    solver_iterations: Array | None = None,
    control_limits: Array | None = None,
) -> SimulationMetrics:
    """Compute metrics with explicit safety, efficiency and timing groups."""
    positions = states[:, :3]
    velocities = states[:, 3:6]
    final_error = float(np.linalg.norm(positions[-1] - goal))
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    tracking_errors = np.array([
        point_to_segment_distance(point, start, goal) for point in positions
    ])
    success = final_error <= goal_threshold and not collision

    finite_clearances = clearances[np.isfinite(clearances)]
    min_clearance = float(finite_clearances.min()) if finite_clearances.size else None
    mean_clearance = float(finite_clearances.mean()) if finite_clearances.size else None

    residuals = (
        np.asarray(chance_residuals, dtype=float)
        if chance_residuals is not None
        else np.empty(0)
    )
    finite_residuals = residuals[np.isfinite(residuals)]
    chance_violation_rate = (
        float(np.mean(finite_residuals < 0.0)) if finite_residuals.size else 0.0
    )

    command_array = (
        np.asarray(controls, dtype=float) if controls is not None else np.empty((0, 4))
    )
    if len(command_array) > 1:
        dt = np.diff(times)
        effort = float(np.sum(np.sum(command_array[1:] ** 2, axis=1) * dt))
        deltas = np.diff(command_array, axis=0)
        smoothness = float(np.sum(deltas**2))
    else:
        effort = 0.0
        smoothness = 0.0

    saturation_rate = 0.0
    if command_array.size and control_limits is not None:
        limits = np.asarray(control_limits, dtype=float)
        saturation_rate = float(np.mean(np.any(
            np.abs(command_array) >= 0.99 * limits,
            axis=1,
        )))

    estimation_error = 0.0
    if estimated_states is not None and len(estimated_states) == len(states):
        estimation_error = float(np.max(np.linalg.norm(
            np.asarray(estimated_states)[:, :3] - positions,
            axis=1,
        )))

    solver_times = np.asarray(solver_times, dtype=float)
    solver_iterations = (
        np.asarray(solver_iterations, dtype=float)
        if solver_iterations is not None
        else np.empty(0)
    )
    solver_iterations = solver_iterations[solver_iterations > 0.0]
    deadline_ms = controller_timestep * 1000.0
    deadline_miss_rate = (
        float(np.mean(solver_times > deadline_ms)) if solver_times.size else 0.0
    )
    p95_solver = _safe_percentile(solver_times, 95)

    speeds = np.linalg.norm(velocities, axis=1)
    return SimulationMetrics(
        success=success,
        collision=collision,
        duration_s=float(times[-1]) if times.size else 0.0,
        time_to_goal_s=float(times[-1]) if success and times.size else None,
        steps=max(0, len(times) - 1),
        final_error_m=final_error,
        path_length_m=path_length,
        tracking_rmse_m=float(np.sqrt(np.mean(tracking_errors**2))),
        mean_speed_mps=float(np.mean(speeds)),
        max_speed_mps=float(np.max(speeds)),
        min_clearance_m=min_clearance,
        mean_clearance_m=mean_clearance,
        chance_violation_rate=chance_violation_rate,
        max_chance_slack=float(slacks.max()) if slacks.size else 0.0,
        control_effort=effort,
        control_smoothness=smoothness,
        saturation_rate=saturation_rate,
        max_estimation_error_m=estimation_error,
        mean_solver_ms=float(solver_times.mean()) if solver_times.size else 0.0,
        median_solver_ms=_safe_percentile(solver_times, 50),
        p95_solver_ms=p95_solver,
        max_solver_ms=float(solver_times.max()) if solver_times.size else 0.0,
        mean_solver_iterations=(
            float(solver_iterations.mean()) if solver_iterations.size else 0.0
        ),
        p95_solver_iterations=_safe_percentile(solver_iterations, 95),
        deadline_miss_rate=deadline_miss_rate,
        real_time_feasible=p95_solver <= deadline_ms,
        solver_warnings=sum(not status.startswith("optimal") for status in statuses),
    )
