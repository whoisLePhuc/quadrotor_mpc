"""Controller adapters with a common deterministic/CC-MPC interface."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import yaml
from scipy.optimize import minimize

from quadrotor_mpc.control.ccmpc.dynamics import QuadrotorDynamics
from quadrotor_mpc.control.ccmpc.obstacle import ObstacleManager
from quadrotor_mpc.control.ccmpc.risk import chance_constraint_residual
from quadrotor_mpc.control.ccmpc.uncertainty import UncertaintyPropagator

Array = npt.NDArray[np.float64]


@dataclass(slots=True)
class ControlResult:
    command: Array
    trajectory: Array
    controls: Array
    solve_time_ms: float
    status: str
    max_slack: float = 0.0
    min_chance_residual: float | None = None
    iterations: int = 0
    cost_terms: dict[str, float] = field(default_factory=dict)


class Controller(Protocol):
    dt: float
    dynamics: QuadrotorDynamics

    def reset(self) -> None: ...

    def solve(
        self,
        state: Array,
        goal: Array,
        obstacles: ObstacleManager,
        covariance: Array,
    ) -> ControlResult: ...


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


class ScipyMPCController:
    """Nonlinear soft-constrained MPC implemented with SciPy.

    This backend is dependency-light and intended for executable demonstrations,
    CI smoke tests and parameter studies.  It solves the nonlinear rollout
    directly and uses a squared slack penalty for collision chance constraints.
    The CVXPY backend below remains the paper-faithful sequential convex QP path.
    """

    def __init__(self, config: dict, mode: str = "ccmpc") -> None:
        if mode not in {"ccmpc", "deterministic"}:
            raise ValueError("mode must be 'ccmpc' or 'deterministic'")
        self.mode = mode
        model = config["model"]["quadrotor"]
        self.dynamics = QuadrotorDynamics(**model)
        runtime = config.get("runtime", {})
        self.dt = float(runtime.get("timestep", 0.12))
        self.horizon = int(runtime.get("horizon_steps", 10))
        self.control_blocks = min(self.horizon, int(runtime.get("control_blocks", self.horizon)))
        self.max_iter = int(runtime.get("scipy_max_iter", 24))
        self.ftol = float(runtime.get("scipy_ftol", 1e-4))

        controller = config["controller"]
        limits = controller["limits"]
        self.max_roll = float(limits["max_roll"])
        self.max_pitch = float(limits["max_pitch"])
        self.max_vert_vel = float(limits["max_vert_vel"])
        self.max_yaw_rate = float(limits["max_yaw_rate"])
        self.max_speed = float(limits["max_speed"])
        self.min_altitude = float(limits.get("min_altitude", 0.15))

        obstacle = controller["obstacle"]
        self.delta = float(obstacle["delta"]) if mode == "ccmpc" else 0.5
        self.mav_radius = float(obstacle["mav_radius"])
        self.slack_penalty = float(obstacle.get("slack_penalty", 1000.0))
        self.max_obstacles = int(obstacle.get("max_obstacles", 2))

        weights = controller["weights"]
        self.terminal_weight = np.asarray(weights["terminal_cost"], dtype=float)
        self.control_weight = np.asarray(weights["control_cost"], dtype=float)
        self.yaw_weight = float(weights["yaw_cost"])
        logistic = weights.get("logistic_cost", {})
        self.logistic_weight = float(logistic.get("Q_o", 0.0))
        self.logistic_lambda = float(logistic.get("lambda_o", 2.0))
        self.logistic_radius = float(logistic.get("r_o", 1.5))

        fov = controller.get("fov", {})
        self.fov_enabled = bool(fov.get("enabled", False))
        self.hfov = math.radians(float(fov.get("hfov_deg", 87.0)) / 2.0)
        self.vfov = math.radians(float(fov.get("vfov_deg", 58.0)) / 2.0)
        self.max_range = float(fov.get("max_range", 5.0))
        self.fov_penalty = float(fov.get("slack_penalty", 10.0))

        self.uncertainty = UncertaintyPropagator.from_config(config)
        self._previous_controls: Array | None = None
        self._previous_blocks: Array | None = None
        self._previous_trajectory: Array | None = None
        self._last_command = np.zeros(4)

    def reset(self) -> None:
        self._previous_controls = None
        self._previous_blocks = None
        self._previous_trajectory = None
        self._last_command = np.zeros(4)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        one_step = [
            (-self.max_roll, self.max_roll),
            (-self.max_pitch, self.max_pitch),
            (-self.max_vert_vel, self.max_vert_vel),
            (-self.max_yaw_rate, self.max_yaw_rate),
        ]
        return one_step * self.control_blocks

    def _expand_controls(self, blocks: Array) -> Array:
        """Expand piecewise-constant control blocks over the full horizon."""
        indices = np.minimum(
            np.arange(self.horizon) * self.control_blocks // self.horizon,
            self.control_blocks - 1,
        )
        return blocks[indices]

    def _initial_controls(self, state: Array, goal: Array) -> Array:
        if self._previous_blocks is not None:
            return np.vstack([self._previous_blocks[1:], self._previous_blocks[-1]])

        error = goal - state[:3]
        c, s = math.cos(state[8]), math.sin(state[8])
        error_body_x = c * error[0] + s * error[1]
        error_body_y = -s * error[0] + c * error[1]
        velocity_body_x = c * state[3] + s * state[4]
        velocity_body_y = -s * state[3] + c * state[4]
        command = np.array(
            [
                np.clip(
                    0.10 * error_body_y - 0.04 * velocity_body_y, -self.max_roll, self.max_roll
                ),
                np.clip(
                    0.10 * error_body_x - 0.04 * velocity_body_x, -self.max_pitch, self.max_pitch
                ),
                np.clip(0.7 * error[2] - 0.2 * state[5], -self.max_vert_vel, self.max_vert_vel),
                np.clip(
                    1.2 * _wrap_angle(math.atan2(error[1], error[0]) - state[8]),
                    -self.max_yaw_rate,
                    self.max_yaw_rate,
                ),
            ]
        )
        return np.tile(command, (self.control_blocks, 1))

    def _rollout(self, state: Array, controls: Array) -> Array:
        trajectory = np.empty((self.horizon + 1, 9), dtype=float)
        trajectory[0] = state
        for k in range(self.horizon):
            trajectory[k + 1] = self.dynamics.discrete(trajectory[k], controls[k], self.dt)
        return trajectory

    def _covariance_horizon(self, covariance: Array, x_guess: Array, u_guess: Array) -> list[Array]:
        return self.uncertainty.propagate(
            covariance,
            x_guess.T,
            u_guess.T,
            self.dynamics,
            self.dt,
        )

    def _risk_residuals(
        self,
        trajectory: Array,
        obstacles: ObstacleManager,
        covariance_horizon: list[Array],
    ) -> list[float]:
        residuals: list[float] = []
        closest = obstacles.get_closest(trajectory[0, :3], self.max_obstacles)
        for k in range(1, self.horizon + 1):
            elapsed = k * self.dt
            mav_cov = covariance_horizon[k][:3, :3]
            for obstacle in closest:
                obs_position = obstacle.p_hat + obstacle.v_hat * elapsed
                obs_covariance = obstacle.Sigma + obstacle.Sigma_v * elapsed**2
                residual = chance_constraint_residual(
                    trajectory[k, :3],
                    obs_position,
                    obstacle.get_omega(self.mav_radius),
                    mav_cov,
                    obs_covariance,
                    self.delta,
                )
                residuals.append(float(residual))
        return residuals

    def _risk_slacks(
        self,
        trajectory: Array,
        obstacles: ObstacleManager,
        covariance_horizon: list[Array],
    ) -> list[float]:
        return [
            max(0.0, -value)
            for value in self._risk_residuals(trajectory, obstacles, covariance_horizon)
        ]

    def _cost_diagnostics(
        self,
        state: Array,
        goal: Array,
        trajectory: Array,
        controls: Array,
        previous_command: Array,
        obstacles: ObstacleManager,
        covariance_horizon: list[Array],
    ) -> dict[str, float]:
        """Evaluate the same major terms used by the optimizer for reporting."""
        tracking = 0.0
        yaw = 0.0
        state_limits = 0.0
        logistic = 0.0
        for k in range(1, self.horizon + 1):
            fraction = k / self.horizon
            reference = state[:3] + fraction * (goal - state[:3])
            error = trajectory[k, :3] - reference
            tracking += 0.7 * float(error @ np.diag([1.0, 1.0, 3.0]) @ error)
            yaw_reference = math.atan2(goal[1] - trajectory[k, 1], goal[0] - trajectory[k, 0])
            yaw += 0.1 * self.yaw_weight * _wrap_angle(trajectory[k, 8] - yaw_reference) ** 2
            speed_xy = float(np.linalg.norm(trajectory[k, 3:5]))
            state_limits += 200.0 * max(0.0, speed_xy - self.max_speed) ** 2
            state_limits += 5000.0 * max(0.0, self.min_altitude - trajectory[k, 2]) ** 2

            if self.logistic_weight > 0.0:
                elapsed = k * self.dt
                for obstacle in obstacles.get_closest(trajectory[k, :3], self.max_obstacles):
                    distance = float(
                        np.linalg.norm(
                            trajectory[k, :3] - (obstacle.p_hat + obstacle.v_hat * elapsed)
                        )
                    )
                    exponent = np.clip(
                        self.logistic_lambda * (distance - self.logistic_radius),
                        -50,
                        50,
                    )
                    logistic += self.logistic_weight / (1.0 + math.exp(float(exponent)))

        terminal_error = trajectory[-1, :3] - goal
        terminal = float(np.sum(self.terminal_weight * terminal_error**2))
        control = float(np.sum(self.control_weight * controls**2))
        deltas = np.vstack(
            [
                controls[0] - previous_command,
                np.diff(controls, axis=0),
            ]
        )
        smoothness = 0.4 * float(np.sum(deltas**2))
        slacks = self._risk_slacks(trajectory, obstacles, covariance_horizon)
        risk = self.slack_penalty * float(np.dot(slacks, slacks)) if slacks else 0.0
        fov = self._fov_cost(trajectory)
        terms = {
            "tracking": tracking,
            "yaw": yaw,
            "terminal": terminal,
            "control": control,
            "smoothness": smoothness,
            "state_limits": state_limits,
            "risk": risk,
            "logistic_obstacle": logistic,
            "fov": fov,
        }
        terms["total"] = float(sum(terms.values()))
        return terms

    def _fov_cost(self, trajectory: Array) -> float:
        if not self.fov_enabled:
            return 0.0
        cost = 0.0
        tan_h, tan_v = math.tan(self.hfov), math.tan(self.vfov)
        for k in range(self.horizon):
            delta = trajectory[k + 1, :3] - trajectory[k, :3]
            yaw = trajectory[k, 8]
            c, s = math.cos(yaw), math.sin(yaw)
            x_body = c * delta[0] + s * delta[1]
            y_body = -s * delta[0] + c * delta[1]
            z_body = delta[2]
            violations = (
                max(0.0, -x_body),
                max(0.0, abs(y_body) - tan_h * max(x_body, 0.0)),
                max(0.0, abs(z_body) - tan_v * max(x_body, 0.0)),
                max(0.0, x_body - self.max_range),
            )
            cost += self.fov_penalty * sum(value * value for value in violations)
        return cost

    def solve(
        self,
        state: Array,
        goal: Array,
        obstacles: ObstacleManager,
        covariance: Array,
    ) -> ControlResult:
        state = np.asarray(state, dtype=float)
        goal = np.asarray(goal, dtype=float)
        initial_blocks = self._initial_controls(state, goal)
        initial_controls = self._expand_controls(initial_blocks)
        x_guess = self._rollout(state, initial_controls)
        covariance_horizon = self._covariance_horizon(covariance, x_guess, initial_controls)
        previous_command = self._last_command.copy()

        def objective(flat_controls: Array) -> float:
            blocks = flat_controls.reshape(self.control_blocks, 4)
            controls = self._expand_controls(blocks)
            trajectory = self._rollout(state, controls)
            cost = 0.0
            for k in range(1, self.horizon + 1):
                fraction = k / self.horizon
                reference = state[:3] + fraction * (goal - state[:3])
                error = trajectory[k, :3] - reference
                cost += 0.7 * float(error @ np.diag([1.0, 1.0, 3.0]) @ error)
                yaw_reference = math.atan2(goal[1] - trajectory[k, 1], goal[0] - trajectory[k, 0])
                cost += 0.1 * self.yaw_weight * _wrap_angle(trajectory[k, 8] - yaw_reference) ** 2

                speed_xy = float(np.linalg.norm(trajectory[k, 3:5]))
                cost += 200.0 * max(0.0, speed_xy - self.max_speed) ** 2
                cost += 5000.0 * max(0.0, self.min_altitude - trajectory[k, 2]) ** 2

            terminal_error = trajectory[-1, :3] - goal
            cost += float(np.sum(self.terminal_weight * terminal_error**2))
            cost += float(np.sum(self.control_weight * controls**2))

            deltas = np.vstack([controls[0] - previous_command, np.diff(controls, axis=0)])
            cost += 0.4 * float(np.sum(deltas**2))

            slacks = self._risk_slacks(trajectory, obstacles, covariance_horizon)
            if slacks:
                cost += self.slack_penalty * float(np.dot(slacks, slacks))

            if self.logistic_weight > 0.0:
                for k in range(1, self.horizon + 1):
                    elapsed = k * self.dt
                    for obstacle in obstacles.get_closest(trajectory[k, :3], self.max_obstacles):
                        distance = float(
                            np.linalg.norm(
                                trajectory[k, :3] - (obstacle.p_hat + obstacle.v_hat * elapsed)
                            )
                        )
                        exponent = np.clip(
                            self.logistic_lambda * (distance - self.logistic_radius), -50, 50
                        )
                        cost += self.logistic_weight / (1.0 + math.exp(float(exponent)))
            return cost + self._fov_cost(trajectory)

        started = time.perf_counter()
        result = minimize(
            objective,
            initial_blocks.reshape(-1),
            method="L-BFGS-B",
            bounds=self.bounds,
            options={"maxiter": self.max_iter, "ftol": self.ftol, "maxls": 15},
        )
        solve_time_ms = (time.perf_counter() - started) * 1000.0
        blocks = np.asarray(result.x, dtype=float).reshape(self.control_blocks, 4)
        controls = self._expand_controls(blocks)
        trajectory = self._rollout(state, controls)
        residuals = self._risk_residuals(trajectory, obstacles, covariance_horizon)
        slacks = [max(0.0, -value) for value in residuals]
        cost_terms = self._cost_diagnostics(
            state,
            goal,
            trajectory,
            controls,
            previous_command,
            obstacles,
            covariance_horizon,
        )

        self._previous_controls = controls.copy()
        self._previous_blocks = blocks.copy()
        self._previous_trajectory = trajectory.copy()
        self._last_command = controls[0].copy()
        status = "optimal" if result.success else f"usable:{result.message}"
        return ControlResult(
            command=controls[0].copy(),
            trajectory=trajectory,
            controls=controls,
            solve_time_ms=solve_time_ms,
            status=status,
            max_slack=max(slacks, default=0.0),
            min_chance_residual=min(residuals) if residuals else None,
            iterations=int(getattr(result, "nit", 0) or 0),
            cost_terms=cost_terms,
        )


class CVXPYController:
    """Adapter for the sequential-convex CVXPY implementation from the reference code."""

    def __init__(self, config: str | Path | dict, mode: str = "ccmpc") -> None:
        try:
            from quadrotor_mpc.control.ccmpc.ccmpc import CCMPC
        except ModuleNotFoundError as exc:
            if exc.name == "cvxpy":
                raise RuntimeError(
                    "CVXPY backend requires optional dependencies; run "
                    "`pip install -r requirements.txt` or `pip install -e .[qp]`"
                ) from exc
            raise
        self._controller = CCMPC(config)
        if mode == "deterministic":
            self._controller.delta = 0.5
        self.dt = self._controller.dt
        self.dynamics = self._controller.dynamics
        self.mav_radius = self._controller.mav_radius

    def reset(self) -> None:
        self._controller._previous_trajectory = None
        self._controller._previous_controls = None
        self._controller._previous_covariance = None

    def solve(
        self, state: Array, goal: Array, obstacles: ObstacleManager, covariance: Array
    ) -> ControlResult:
        started = time.perf_counter()
        trajectory, controls = self._controller.solve(
            state,
            goal,
            obstacle_manager=obstacles,
            Gamma_0=covariance,
        )
        solve_time_ms = (time.perf_counter() - started) * 1000.0
        slack_values = [
            float(variable.value)
            for variable in self._controller._cc_slack
            if variable.value is not None
        ]
        status = str(self._controller._problem.status)
        stats = self._controller._problem.solver_stats
        objective_value = self._controller._problem.value
        return ControlResult(
            command=controls[:, 0].copy(),
            trajectory=trajectory.T.copy(),
            controls=controls.T.copy(),
            solve_time_ms=solve_time_ms,
            status=status,
            max_slack=max(slack_values, default=0.0),
            min_chance_residual=(-max(slack_values, default=0.0) if obstacles.obstacles else None),
            iterations=int(getattr(stats, "num_iters", 0) or 0),
            cost_terms={
                "total": float(objective_value)
                if objective_value is not None and np.isfinite(objective_value)
                else float("nan")
            },
        )


def load_controller(
    config_path: str | Path,
    mode: str = "ccmpc",
    backend: str = "scipy",
    config_override: dict | None = None,
) -> Controller:
    """Create a controller using one shared interface."""
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config_override:
        _deep_update(config, config_override)
    if backend == "cvxpy":
        return CVXPYController(config, mode)
    if backend != "scipy":
        raise ValueError("backend must be 'scipy' or 'cvxpy'")
    return ScipyMPCController(config, mode)


def _deep_update(target: dict, updates: dict) -> dict:
    """Recursively update a configuration mapping in-place."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target
