"""Matplotlib visualizer for simulation results.

Produces static plots, animations, and comparison figures from a
SimulationHistory object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from simulation.runner import SimulationHistory


# Try importing matplotlib — fail gracefully with a clear message
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib.animation import FuncAnimation
    matplotlib.use("Agg")
except ImportError:
    raise ImportError(
        "matplotlib required. Install: pip install matplotlib"
    )

# Ensure ffmpeg is available for animation export
import os as _os
try:
    import imageio_ffmpeg as _iif
    _ffmpeg_path = _iif.get_ffmpeg_exe()
    _os.environ["PATH"] = _os.path.dirname(_ffmpeg_path) + _os.pathsep + _os.environ.get("PATH", "")
    matplotlib.rcParams["animation.ffmpeg_path"] = _ffmpeg_path
except ImportError:
    pass


# Color/style constants matching theory documentation
COLOR_TRAJECTORY = "tab:blue"
COLOR_COMPARISON_DET = "tab:orange"
COLOR_COLLISION = "tab:red"
COLOR_START = "tab:green"
COLOR_GOAL = "tab:red"
COLOR_OBSTACLE = "tab:gray"
COLOR_COVARIANCE = "tab:blue"
LINE_WIDTH = 2
FONT_SIZE = 10
DPI_DEFAULT = 150
DPI_PUBLICATION = 300


class MatplotlibVisualizer:
    """Renders simulation results as publication-quality matplotlib figures."""

    def __init__(
        self,
        history: SimulationHistory,
        *,
        dpi: int = DPI_DEFAULT,
        figsize: tuple[float, float] = (10, 8),
    ) -> None:
        self.history = history
        self.dpi = dpi
        self.figsize = figsize

    def _get_obstacle_ellipses(self) -> list[dict[str, Any]]:
        """Extract obstacle ellipse parameters from config."""
        obstacles = []
        if self.history.config is None:
            return obstacles

        for obs_spec in self.history.config.obstacles:
            cx, cy = obs_spec.position[0], obs_spec.position[1]
            # Convert box size [l, w, h] to ellipsoid axes
            l, w, _ = obs_spec.size
            a = np.sqrt(3) / 2 * l
            b = np.sqrt(3) / 2 * w
            yaw = obs_spec.yaw
            obstacles.append({
                "center": (cx, cy),
                "width": 2 * a,
                "height": 2 * b,
                "angle": np.degrees(yaw),
            })
        return obstacles

    def _get_state_array(self) -> npt.NDArray:
        """Get state history as (T, 9) array."""
        return np.array(self.history.states)

    def _find_failure_point(self) -> int | None:
        """Find first infeasible timestep index, or None."""
        for i, feasible in enumerate(self.history.feasibility):
            if not feasible:
                return i
        for i, coll in enumerate(self.history.collisions):
            if coll:
                return i
        return None

    def plot_trajectory(
        self,
        save_path: str | Path = "output/trajectory.png",
        *,
        show_covariance: bool = True,
        covariance_interval: int = 5,
    ) -> Path:
        """Generate static 2D top-down (X-Y) trajectory plot.

        Parameters
        ----------
        save_path:
            Path for the output PNG file.
        show_covariance:
            If True, overlay covariance contours.
        covariance_interval:
            Draw contours every N timesteps.

        Returns
        -------
        Path
            Path to saved PNG file.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        states = self._get_state_array()
        positions = states[:, :3]  # (T, 3)

        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)

        # Determine plot bounds from trajectory + obstacles
        all_x = list(positions[:, 0])
        all_y = list(positions[:, 1])
        for obs in self._get_obstacle_ellipses():
            all_x.extend([obs["center"][0] - obs["width"] / 2,
                          obs["center"][0] + obs["width"] / 2])
            all_y.extend([obs["center"][1] - obs["height"] / 2,
                          obs["center"][1] + obs["height"] / 2])
        margin = 1.0
        x_min, x_max = min(all_x) - margin, max(all_x) + margin
        y_min, y_max = min(all_y) - margin, max(all_y) + margin

        # Obstacles
        for obs in self._get_obstacle_ellipses():
            ellipse = Ellipse(
                xy=obs["center"],
                width=obs["width"],
                height=obs["height"],
                angle=obs["angle"],
                facecolor=COLOR_OBSTACLE,
                edgecolor="black",
                alpha=0.3,
                linewidth=1,
            )
            ax.add_patch(ellipse)

        # Trajectory
        failure_idx = self._find_failure_point()
        if failure_idx is not None and failure_idx > 0:
            # Plot safe portion
            ax.plot(positions[:failure_idx, 0], positions[:failure_idx, 1],
                    color=COLOR_TRAJECTORY, linewidth=LINE_WIDTH,
                    label="flight path")
            # Plot failure portion in red
            ax.plot(positions[failure_idx:, 0], positions[failure_idx:, 1],
                    color=COLOR_COLLISION, linewidth=LINE_WIDTH,
                    linestyle="--", label="failure")
            # Mark failure point
            ax.scatter(positions[failure_idx, 0], positions[failure_idx, 1],
                       color=COLOR_COLLISION, marker="X", s=200, zorder=5,
                       label="failure")
        else:
            ax.plot(positions[:, 0], positions[:, 1],
                    color=COLOR_TRAJECTORY, linewidth=LINE_WIDTH,
                    label="flight path")

        # Covariance contours
        if show_covariance and len(self.history.covariances) > 0:
            for i, gamma in enumerate(self.history.covariances):
                if i % covariance_interval != 0:
                    continue
                if gamma is None:
                    continue
                # Extract position covariance (3x3 top-left)
                pos_cov = gamma[:3, :3]
                try:
                    w, v = np.linalg.eigh(pos_cov)
                    w = np.maximum(w, 1e-10)
                    angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
                    width = 2 * np.sqrt(w[0]) * 3  # 3-sigma
                    height = 2 * np.sqrt(w[1]) * 3
                    ellipse = Ellipse(
                        xy=(positions[i, 0], positions[i, 1]),
                        width=width, height=height, angle=angle,
                        facecolor="none", edgecolor=COLOR_COVARIANCE,
                        alpha=0.4, linewidth=0.8, linestyle="--",
                    )
                    ax.add_patch(ellipse)
                except np.linalg.LinAlgError:
                    pass

        # Start and goal markers
        ax.scatter(positions[0, 0], positions[0, 1],
                   color=COLOR_START, marker="o", s=120, zorder=5,
                   label="start")
        goal_pos = self.history.config.goal.position if self.history.config else None
        if goal_pos is not None:
            ax.scatter(goal_pos[0], goal_pos[1],
                       color=COLOR_GOAL, marker="D", s=120, zorder=5,
                       label="goal")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("X [m]", fontsize=FONT_SIZE)
        ax.set_ylabel("Y [m]", fontsize=FONT_SIZE)
        ax.set_title("Quadrotor Trajectory — CC-MPC", fontsize=FONT_SIZE + 2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=FONT_SIZE - 1, loc="best")
        ax.set_aspect("equal")

        plt.tight_layout()
        fig.savefig(str(save_path), dpi=DPI_PUBLICATION, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_summary_panel(
        self,
        save_path: str | Path = "output/summary.png",
    ) -> Path:
        """Multi-panel summary figure.

        Layout (2x2):
        - Top-left: X-Y trajectory
        - Top-right: Altitude over time
        - Bottom-left: Control commands over time
        - Bottom-right: Solve time per step
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        states = self._get_state_array()
        positions = states[:, :3]
        t_arr = np.arange(len(states))

        fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=self.dpi)

        # Top-left: X-Y trajectory
        ax = axes[0, 0]
        ax.plot(positions[:, 0], positions[:, 1], color=COLOR_TRAJECTORY,
                linewidth=LINE_WIDTH)
        ax.scatter(positions[0, 0], positions[0, 1],
                   color=COLOR_START, marker="o", s=80)
        if self.history.config:
            g = self.history.config.goal.position
            ax.scatter(g[0], g[1], color=COLOR_GOAL, marker="D", s=80)
        for obs in self._get_obstacle_ellipses():
            ellipse = Ellipse(
                xy=obs["center"], width=obs["width"], height=obs["height"],
                angle=obs["angle"], facecolor=COLOR_OBSTACLE,
                edgecolor="black", alpha=0.3, linewidth=1,
            )
            ax.add_patch(ellipse)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("Trajectory")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # Top-right: Altitude over time
        ax = axes[0, 1]
        ax.plot(t_arr * 0.02, positions[:, 2], color=COLOR_TRAJECTORY,
                linewidth=LINE_WIDTH)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Z [m]")
        ax.set_title("Altitude")
        ax.grid(True, alpha=0.3)

        # Bottom-left: Control commands over time
        ax = axes[1, 0]
        commands = np.array(self.history.commands) if self.history.commands else np.zeros((1, 4))
        t_cmd = np.arange(len(commands)) * 0.02
        ax.plot(t_cmd, commands[:, 0], label=r"$\phi_c$", linewidth=1.5)
        ax.plot(t_cmd, commands[:, 1], label=r"$\theta_c$", linewidth=1.5)
        ax.plot(t_cmd, commands[:, 2], label=r"$v_{z,c}$", linewidth=1.5)
        ax.plot(t_cmd, commands[:, 3], label=r"$\dot{\psi}_c$", linewidth=1.5)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Command")
        ax.set_title("Control Inputs")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom-right: Solve time per step
        ax = axes[1, 1]
        solve_times = np.array(self.history.solve_times) if self.history.solve_times else np.zeros(1)
        t_solve = np.arange(len(solve_times)) * 0.02
        ax.plot(t_solve, solve_times, color="tab:purple", linewidth=1.5)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Solve time [ms]")
        ax.set_title("MPC Solve Time")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(str(save_path), dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def animate(
        self,
        save_path: str | Path = "output/animation.mp4",
        *,
        fps: int = 10,
    ) -> Path:
        """Generate MP4 animation showing simulation evolution.

        Parameters
        ----------
        save_path:
            Output MP4 file path.
        fps:
            Frames per second.

        Returns
        -------
        Path
            Path to saved MP4 file.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        states = self._get_state_array()

        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)

        # Static obstacles
        for obs in self._get_obstacle_ellipses():
            ellipse = Ellipse(
                xy=obs["center"], width=obs["width"], height=obs["height"],
                angle=obs["angle"], facecolor=COLOR_OBSTACLE,
                edgecolor="black", alpha=0.3, linewidth=1,
            )
            ax.add_patch(ellipse)

        # Static elements
        ax.scatter(states[0, 0], states[0, 1], color=COLOR_START,
                   marker="o", s=120, zorder=5, label="start")
        if self.history.config:
            g = self.history.config.goal.position
            ax.scatter(g[0], g[1], color=COLOR_GOAL, marker="D",
                       s=120, zorder=5, label="goal")

        all_x = states[:, 0]
        all_y = states[:, 1]
        margin = 1.0
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("Quadrotor CC-MPC Simulation")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        ax.legend(fontsize=9)

        (trail_line,) = ax.plot([], [], color=COLOR_TRAJECTORY,
                                linewidth=LINE_WIDTH, alpha=0.7)
        (pos_dot,) = ax.plot([], [], "o", color=COLOR_TRAJECTORY,
                             markersize=8, zorder=6)
        time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                            va="top", fontfamily="monospace", fontsize=9)

        def _init() -> tuple:
            trail_line.set_data([], [])
            pos_dot.set_data([], [])
            time_text.set_text("")
            return trail_line, pos_dot, time_text

        def _update(frame: int) -> tuple:
            trail_line.set_data(states[:frame + 1, 0],
                                states[:frame + 1, 1])
            pos_dot.set_data([states[frame, 0]], [states[frame, 1]])
            time_text.set_text(f"t = {frame * 0.02:.1f}s  "
                               f"step {frame}/{len(states)}")
            return trail_line, pos_dot, time_text

        n_frames = max(2, len(states))
        anim = FuncAnimation(fig, _update, frames=n_frames,
                             init_func=_init, blit=True, interval=1000 // fps)

        try:
            anim.save(str(save_path), writer="ffmpeg", fps=fps, dpi=self.dpi)
        except Exception as e:
            print(f"Warning: animation failed to save ({e}). "
                  f"Install ffmpeg for MP4 output.")
            return save_path

        plt.close(fig)
        return save_path

    def plot_comparison(
        self,
        det_history: SimulationHistory,
        save_path: str | Path = "output/comparison.png",
    ) -> Path:
        """Overlaid comparison of CC-MPC vs deterministic MPC trajectories."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        cc_states = self._get_state_array()
        det_states = np.array(det_history.states)

        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)

        # Obstacles
        for obs in self._get_obstacle_ellipses():
            ellipse = Ellipse(
                xy=obs["center"], width=obs["width"], height=obs["height"],
                angle=obs["angle"], facecolor=COLOR_OBSTACLE,
                edgecolor="black", alpha=0.3, linewidth=1,
            )
            ax.add_patch(ellipse)

        # CC-MPC trajectory
        ax.plot(cc_states[:, 0], cc_states[:, 1],
                color=COLOR_TRAJECTORY, linewidth=LINE_WIDTH,
                label="CC-MPC", alpha=0.8)
        # Deterministic MPC trajectory
        ax.plot(det_states[:, 0], det_states[:, 1],
                color=COLOR_COMPARISON_DET, linewidth=LINE_WIDTH,
                label="Deterministic MPC", alpha=0.8, linestyle="--")

        # Start and goal
        ax.scatter(cc_states[0, 0], cc_states[0, 1],
                   color=COLOR_START, marker="o", s=120, zorder=5)
        if self.history.config:
            g = self.history.config.goal.position
            ax.scatter(g[0], g[1], color=COLOR_GOAL, marker="D", s=120, zorder=5)

        all_x = np.concatenate([cc_states[:, 0], det_states[:, 0]])
        all_y = np.concatenate([cc_states[:, 1], det_states[:, 1]])
        margin = 1.0
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

        ax.set_xlabel("X [m]", fontsize=FONT_SIZE)
        ax.set_ylabel("Y [m]", fontsize=FONT_SIZE)
        ax.set_title("CC-MPC vs Deterministic MPC", fontsize=FONT_SIZE + 2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=FONT_SIZE - 1)
        ax.set_aspect("equal")

        plt.tight_layout()
        fig.savefig(str(save_path), dpi=DPI_PUBLICATION, bbox_inches="tight")
        plt.close(fig)
        return save_path
