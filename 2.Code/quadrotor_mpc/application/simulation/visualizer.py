"""Adaptive publication-style Matplotlib reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from .config import ScenarioConfig
from .runner import SimulationResult

COLORS = {"ccmpc": "#2166ac", "deterministic": "#ef8a62", "nmpc": "#1b9e77"}
COMMAND_LABELS = (r"$\phi_c$", r"$\theta_c$", r"$v_{z,c}$", r"$\dot\psi_c$")


def _label(result: SimulationResult) -> str:
    return f"{result.mode} / seed {result.seed}"


def _finite(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(values)
    return mask, values[mask]


def save_report(
    results: list[SimulationResult],
    scenario: ScenarioConfig,
    output_path: str | Path,
) -> Path:
    """Save a scenario-aware quick report with tracking, safety and timing."""
    if not results:
        raise ValueError("at least one simulation result is required")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_obstacles = bool(scenario.obstacles)

    figure = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = figure.add_gridspec(4, 3, height_ratios=(0.16, 1.25, 1.0, 1.0))
    kpi_axis = figure.add_subplot(grid[0, :])
    trajectory_axis = figure.add_subplot(grid[1:3, 0:2])
    error_axis = figure.add_subplot(grid[1, 2])
    safety_axis = figure.add_subplot(grid[2, 2])
    control_axis = figure.add_subplot(grid[3, 0])
    uncertainty_axis = figure.add_subplot(grid[3, 1])
    solver_axis = figure.add_subplot(grid[3, 2])
    kpi_axis.axis("off")

    for result in results:
        color = COLORS.get(result.mode)
        label = _label(result)
        trajectory_axis.plot(
            result.states[:, 0], result.states[:, 1], color=color, lw=2.2, label=label
        )
        trajectory_axis.plot(
            result.reference_positions[:, 0], result.reference_positions[:, 1],
            color=color, lw=1.0, ls=":", alpha=0.6,
        )
        position_error = np.linalg.norm(result.states[:, :3] - scenario.goal, axis=1)
        error_axis.plot(result.times, position_error, color=color, label=label)

        if has_obstacles:
            mask, clearance = _finite(result.clearances)
            safety_axis.plot(result.times[mask], clearance, color=color, label=label)
        else:
            safety_axis.text(
                0.5, 0.5, "N/A — scenario has no obstacles",
                ha="center", va="center", transform=safety_axis.transAxes,
                color="#5d6d7e",
            )

        for index, command_label in enumerate(COMMAND_LABELS):
            control_axis.plot(
                result.times,
                result.controls[:, index],
                color=color,
                alpha=0.45 + 0.12 * index,
                lw=1.0,
                label=f"{result.mode}: {command_label}",
            )
        sigma = np.sqrt(np.maximum(np.diagonal(result.covariances, axis1=1, axis2=2)[:, :3], 0.0))
        uncertainty_axis.plot(result.times, np.linalg.norm(sigma, axis=1), color=color, label=label)
        mask = result.solver_times_ms > 0.0
        solver_axis.plot(
            result.times[mask], result.solver_times_ms[mask],
            color=color, marker=".", ms=3, lw=1.0, label=label,
        )
        solver_axis.axhline(
            result.controller_dt * 1000.0,
            color=color,
            ls=":",
            lw=1.0,
            alpha=0.7,
        )

    for obstacle in scenario.obstacles:
        axes = np.sqrt(3.0) * obstacle.size / 2.0 + 0.4
        patch = Ellipse(
            xy=obstacle.position[:2],
            width=2.0 * axes[0],
            height=2.0 * axes[1],
            angle=np.degrees(obstacle.yaw),
            facecolor="#d73027",
            edgecolor="#7f0000",
            alpha=0.18,
        )
        trajectory_axis.add_patch(patch)
        if np.linalg.norm(obstacle.velocity[:2]) > 0.0:
            trajectory_axis.arrow(
                obstacle.position[0], obstacle.position[1],
                obstacle.velocity[0], obstacle.velocity[1],
                width=0.015, color="#7f0000", length_includes_head=True,
            )

    trajectory_axis.scatter(
        scenario.start[0], scenario.start[1], marker="o", s=65,
        color="#636e72", edgecolor="white", zorder=5, label="start",
    )
    trajectory_axis.scatter(
        *scenario.goal[:2], marker="*", s=260,
        color="#f1c40f", edgecolor="black", zorder=5, label="goal",
    )
    same_point = np.linalg.norm(scenario.start[:2] - scenario.goal[:2]) < 1e-9
    trajectory_axis.set_title(f"Trajectory — {scenario.name}")
    trajectory_axis.set_xlabel("x [m]")
    trajectory_axis.set_ylabel("y [m]")
    trajectory_axis.set_aspect("equal", adjustable="box")
    if same_point:
        radius = max(0.5, 0.1 * max(abs(scenario.start[2]), 1.0))
        trajectory_axis.set_xlim(scenario.start[0] - radius, scenario.start[0] + radius)
        trajectory_axis.set_ylim(scenario.start[1] - radius, scenario.start[1] + radius)
    trajectory_axis.grid(True, alpha=0.25)
    trajectory_axis.legend(loc="best", fontsize=8)

    error_axis.axhline(scenario.goal_threshold, color="#636e72", ls="--", lw=1)
    error_axis.set(title="Goal error", xlabel="time [s]", ylabel="error [m]")
    safety_axis.axhline(0.0, color="#d73027", ls="--", lw=1)
    safety_axis.set(
        title="Ellipsoid clearance" if has_obstacles else "Safety clearance",
        xlabel="time [s]", ylabel="clearance [m]",
    )
    control_axis.set(title="Control commands", xlabel="time [s]", ylabel="command")
    uncertainty_axis.set(title="Position uncertainty", xlabel="time [s]", ylabel=r"$||\sigma_p||$ [m]")
    solver_axis.set(title="Solver timing", xlabel="time [s]", ylabel="solve [ms]")
    for axis in (error_axis, safety_axis, control_axis, uncertainty_axis, solver_axis):
        axis.grid(True, alpha=0.25)
    error_axis.legend(fontsize=7)
    uncertainty_axis.legend(fontsize=7)
    solver_axis.legend(fontsize=7)

    kpis = []
    for result in results:
        clearance = (
            "N/A" if result.metrics.min_clearance_m is None
            else f"{result.metrics.min_clearance_m:.3f} m"
        )
        kpis.append(
            f"{result.mode}: success={result.metrics.success} · collision={result.metrics.collision} · "
            f"final={result.metrics.final_error_m:.3f} m · clearance={clearance} · "
            f"p95={result.metrics.p95_solver_ms:.1f} ms"
        )
    kpi_axis.text(
        0.5, 0.5, "\n".join(kpis),
        ha="center", va="center", fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f4f6f7", "edgecolor": "#d5d8dc"},
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path
