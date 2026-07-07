"""Simulation runner — orchestrates the CC-MPC simulation loop.

This module contains the data structures and runner class that drive the
matplotlib-based simulation of the quadrotor CC-MPC algorithm.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from ccmpc.obstacle import ObstacleManager, ObstacleState
from ccmpc.types import FloatArray


from simulation.config.loader import load_scenario_config
from simulation.config.schema import ScenarioConfig


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SimulationState:
    """Holds the full system state at a single timestep."""

    k: int                         # Timestep index
    t: float                       # Simulation time (s)
    state: FloatArray              # Quadrotor State9, shape (9,)
    command: FloatArray            # ControlCommand4, shape (4,)
    Gamma: FloatArray | None       # Full covariance, shape (9,9)
    sigma_pos: FloatArray | None   # Position covariance, shape (3,3)
    solve_time_ms: float           # MPC solve wall-clock time (ms)
    solver_status: str             # CVXPY solver return status
    collision: bool                # Whether collision detected
    feasible: bool                 # Whether solver found feasible solution
    goal_dist: float               # Distance to goal (m)


@dataclass
class SimulationHistory:
    """Holds the complete simulation history for one run."""

    config: ScenarioConfig | None
    states: list[FloatArray] = field(default_factory=list)           # T × (9,)
    commands: list[FloatArray] = field(default_factory=list)          # T × (4,)
    covariances: list[FloatArray] = field(default_factory=list)      # T × (9,9)
    solve_times: list[float] = field(default_factory=list)           # (T,)
    solver_statuses: list[str] = field(default_factory=list)         # (T,)
    collisions: list[bool] = field(default_factory=list)             # (T,)
    feasibility: list[bool] = field(default_factory=list)            # (T,)
    goal_reached: bool = False

    @property
    def success(self) -> bool:
        return self.goal_reached and (not any(self.collisions))

    @property
    def min_separation(self) -> float:
        """Minimum distance from quadrotor to nearest obstacle edge across all steps."""
        # This is computed externally during the simulation loop.
        # Returns a placeholder if not yet computed.
        return getattr(self, "_min_separation", float("nan"))

    @min_separation.setter
    def min_separation(self, value: float) -> None:
        self._min_separation = value

    @property
    def trajectory_length(self) -> float:
        """Total path length traveled."""
        if len(self.states) < 2:
            return 0.0
        pos = np.array([s[:3] for s in self.states])
        diffs = np.diff(pos, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    @property
    def avg_solve_time(self) -> float:
        return float(np.mean(self.solve_times)) if self.solve_times else 0.0

    @property
    def total_steps(self) -> int:
        return len(self.states)

    def append(self, s: SimulationState) -> None:
        self.states.append(s.state)
        self.commands.append(s.command)
        if s.Gamma is not None:
            self.covariances.append(s.Gamma)
        self.solve_times.append(s.solve_time_ms)
        self.solver_statuses.append(s.solver_status)
        self.collisions.append(s.collision)
        self.feasibility.append(s.feasible)


@dataclass
class SimulationSummary:
    """Quantitative report printed at end of a simulation run."""

    success: bool
    goal_reached: bool
    collision_detected: bool
    total_timesteps: int
    trajectory_length: float
    min_separation: float
    avg_solve_time_ms: float
    max_solve_time_ms: float
    solver_failures: int
    seed: int
    controller_type: str

    def __str__(self) -> str:
        lines = [
            "=" * 48,
            "Simulation Summary",
            "=" * 48,
            f"  Status:          {'✓ SUCCESS' if self.success else '✗ FAIL'}",
            f"  Goal reached:    {'yes' if self.goal_reached else 'no'}",
            f"  Collision:       {'yes' if self.collision_detected else 'no'}",
            f"  Timesteps:       {self.total_timesteps}",
            f"  Trajectory len:  {self.trajectory_length:.2f} m",
            f"  Min separation:  {self.min_separation:.3f} m",
            f"  Avg solve time:  {self.avg_solve_time_ms:.1f} ms",
            f"  Max solve time:  {self.max_solve_time_ms:.1f} ms",
            f"  Solver failures: {self.solver_failures}",
            f"  Seed:            {self.seed}",
            f"  Controller:      {self.controller_type}",
            "-" * 48,
        ]
        return "\n".join(lines)


@dataclass
class MonteCarloSummary:
    """Aggregated statistics over multiple Monte Carlo trials."""

    trials: int
    success_rate: float
    mean_min_separation: float
    std_min_separation: float
    mean_trajectory_length: float
    mean_avg_solve_time: float
    results: list[SimulationSummary] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            "=" * 48,
            "Monte Carlo Summary",
            "=" * 48,
            f"  Trials:              {self.trials}",
            f"  Success rate:        {self.success_rate * 100:.0f}%",
            f"  Min separation:      {self.mean_min_separation:.3f} ± {self.std_min_separation:.3f} m",
            f"  Trajectory length:   {self.mean_trajectory_length:.2f} m",
            f"  Avg solve time:      {self.mean_avg_solve_time:.1f} ms",
            "-" * 48,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# EKF divergence threshold — covariance trace above this triggers divergence halt
EKF_DIVERGENCE_THRESHOLD: float = 100.0

# Solver timeout threshold (ms) — treat as failure if exceeded
SOLVER_TIMEOUT_MS: float = 500.0


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


def _clip_command(command: FloatArray, limits: dict | None = None) -> FloatArray:
    """Clip control commands to dynamics limits.

    Default limits (from mpc.yaml or CCMPCConfig):
      max_roll: 0.25 rad, max_pitch: 0.25 rad
      max_vert_vel: 3.0 m/s, max_yaw_rate: 0.8 rad/s
    """
    cmd = command.copy()
    if limits:
        cmd[0] = np.clip(cmd[0], -limits.get("max_roll", 0.25), limits.get("max_roll", 0.25))
        cmd[1] = np.clip(cmd[1], -limits.get("max_pitch", 0.25), limits.get("max_pitch", 0.25))
        cmd[2] = np.clip(cmd[2], -limits.get("max_vert_vel", 3.0), limits.get("max_vert_vel", 3.0))
        cmd[3] = np.clip(cmd[3], -limits.get("max_yaw_rate", 0.8), limits.get("max_yaw_rate", 0.8))
    return cmd


class SimulationRunner:
    """Top-level simulation orchestrator.

    Loads a scenario config, creates the necessary components, and runs the
    simulation loop.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config: ScenarioConfig | None = None
        self.mpc: Any = None
        self.dynamics: Any = None
        self.obstacle_manager: ObstacleManager | None = None
        self._last_history: SimulationHistory | None = None

        if config_path is not None:
            self.load_config(config_path)

    def load_config(self, config_path: str | Path) -> None:
        self.config = load_scenario_config(config_path, validate=True)

        from ccmpc.legacy_solver import CCMPC as _CCMPC
        _ref_dir = Path(__file__).parent.parent.parent / "4.Reference" / "quadrotor_ccmpc"
        _mpc_config = _ref_dir / "config" / "mpc.yaml"
        self.mpc = _CCMPC(str(_mpc_config.resolve()))
        self.dynamics = self.mpc.dynamics
        self._mpc_dt = self.mpc.dt

        if self.config is not None and self.config.obstacles:
            from ccmpc.obstacle import (
                ObstacleShape, ObstacleMotionModel,
                default_position_covariance, default_velocity_covariance,
            )
            from ccmpc.utils import box_to_ellipsoid_axes, yaw_to_rotation
            import numpy as np
            obs_states = []
            for o in self.config.obstacles:
                axes = box_to_ellipsoid_axes(np.array(o.size, dtype=np.float64))
                rotation = yaw_to_rotation(o.yaw)
                pos = np.array(o.position, dtype=np.float64)
                vel = np.array(o.velocity, dtype=np.float64)
                cov = default_position_covariance()
                cov_v = default_velocity_covariance()
                if o.covariance is not None:
                    cov = np.diag([s**2 for s in o.covariance.position_std])
                    cov_v = np.diag([s**2 for s in o.covariance.velocity_std])
                obs_state = ObstacleState(
                    obstacle_id=o.id,
                    p_hat=pos,
                    v_hat=vel,
                    axes=axes,
                    R_o=rotation,
                    Sigma=cov,
                    Sigma_v=cov_v,
                    active=o.active,
                    shape=ObstacleShape.ELLIPSOID,
                    motion_model=ObstacleMotionModel(o.motion_model.value),
                )
                obs_states.append(obs_state)
            self.obstacle_manager = ObstacleManager(tuple(obs_states))
        else:
            self.obstacle_manager = ObstacleManager()

    def _get_dt(self) -> float:
        if self.config and self.config.runtime_overrides and self.config.runtime_overrides.sim_dt:
            return self.config.runtime_overrides.sim_dt
        return 0.02

    @property
    def last_history(self) -> SimulationHistory | None:
        return self._last_history

    def run(self, *, seed: int | None = None) -> SimulationHistory:
        if self.config is None:
            raise RuntimeError("No config loaded. Call load_config() first.")

        actual_seed = seed if seed is not None else 42
        np.random.seed(actual_seed)

        start = self.config.initial_state
        goal_pos = self.config.goal.position
        goal_threshold = self.config.goal.threshold
        sim_dt = self._get_dt()

        max_steps = 200
        if self.config.termination.max_steps is not None:
            max_steps = self.config.termination.max_steps
        elif self.config.termination.max_time > 0:
            max_steps = int(self.config.termination.max_time / sim_dt)

        state = start.copy()
        gamma_0 = np.eye(9, dtype=np.float64) * 0.01
        history = SimulationHistory(config=self.config)
        min_separation = float("inf")
        mav_radius = 0.4

        mpc_skip = max(1, round(self._mpc_dt / sim_dt))
        mpc_counter = 0
        last_command = np.zeros(4, dtype=np.float64)

        for k in tqdm(range(max_steps), desc="Simulation", unit="step"):
            t = k * sim_dt
            solver_status = "optimal"
            solve_time_ms = 0.0
            command = last_command.copy()
            feasible = True

            if mpc_counter % mpc_skip == 0:
                t0 = time.perf_counter()
                try:
                    x_traj, u_traj = self.mpc.solve(
                        state, goal_pos,
                        obstacle_manager=self.obstacle_manager,
                        Gamma_0=gamma_0,
                    )
                    solve_time_ms = (time.perf_counter() - t0) * 1000.0
                    if u_traj is not None and u_traj.shape[1] > 0:
                        command = u_traj[:, 0].copy()
                        last_command = command.copy()
                except Exception as exc:
                    solve_time_ms = (time.perf_counter() - t0) * 1000.0
                    solver_status = str(exc)[:50]
                    feasible = False
            mpc_counter += 1

            if np.any(np.isnan(command)) or np.any(np.isinf(command)):
                solver_status = "numerical_error"
                feasible = False

            command = _clip_command(command, {
                "max_roll": 0.25, "max_pitch": 0.25,
                "max_vert_vel": 3.0, "max_yaw_rate": 0.8,
            })

            if self.dynamics is not None:
                state = self.dynamics.discrete(state, command, sim_dt)

            collision = False
            if self.obstacle_manager:
                for obs in self.obstacle_manager.active_obstacles:
                    dist = obs.distance_to_edge(state[:3])
                    if dist <= 0.0:
                        collision = True
                    min_separation = min(min_separation, dist)

            goal_dist = float(np.linalg.norm(state[:3] - goal_pos))
            goal_reached = goal_dist < goal_threshold

            sim_state = SimulationState(
                k=k, t=t, state=state.copy(), command=command.copy(),
                Gamma=None, sigma_pos=None,
                solve_time_ms=solve_time_ms,
                solver_status=solver_status,
                collision=collision,
                feasible=feasible,
                goal_dist=goal_dist,
            )
            history.append(sim_state)

            if goal_reached:
                history.goal_reached = True
                break

            last_command = command

        history.min_separation = min_separation if min_separation != float("inf") else 0.0
        self._last_history = history
        return history

    def run_monte_carlo(
        self,
        num_trials: int,
        *,
        base_seed: int | None = None,
    ) -> MonteCarloSummary:
        """Run N independent trials with seeds base_seed+0..base_seed+N-1.

        All N trials always run regardless of individual failures.
        """
        base = base_seed if base_seed is not None else 42
        summaries: list[SimulationSummary] = []

        for i in range(num_trials):
            trial_seed = base + i
            history = self.run(seed=trial_seed)
            summary = self._make_summary(history, trial_seed)
            summaries.append(summary)

        success_rate = np.mean([s.success for s in summaries])
        mean_sep = np.mean([s.min_separation for s in summaries])
        std_sep = np.std([s.min_separation for s in summaries])
        mean_len = np.mean([s.trajectory_length for s in summaries])
        mean_solve = np.mean([s.avg_solve_time_ms for s in summaries])

        return MonteCarloSummary(
            trials=num_trials,
            success_rate=float(success_rate),
            mean_min_separation=float(mean_sep),
            std_min_separation=float(std_sep),
            mean_trajectory_length=float(mean_len),
            mean_avg_solve_time=float(mean_solve),
            results=summaries,
        )

    def _make_summary(
        self,
        history: SimulationHistory,
        seed: int,
    ) -> SimulationSummary:
        """Build a SimulationSummary from history."""
        return SimulationSummary(
            success=history.success,
            goal_reached=history.goal_reached,
            collision_detected=any(history.collisions),
            total_timesteps=history.total_steps,
            trajectory_length=history.trajectory_length,
            min_separation=history.min_separation,
            avg_solve_time_ms=history.avg_solve_time,
            max_solve_time_ms=max(history.solve_times) if history.solve_times else 0.0,
            solver_failures=sum(1 for f in history.feasibility if not f),
            seed=seed,
            controller_type="ccmpc",
        )
