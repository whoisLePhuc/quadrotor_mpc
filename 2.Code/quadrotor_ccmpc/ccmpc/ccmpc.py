"""
Chance-Constrained MPC for quadrotor obstacle avoidance.

Implements the formulations from:
  "Robust Vision-based Obstacle Avoidance for Micro Aerial Vehicles
   in Dynamic Environments" — Lin, Zhu, Alonso-Mora, ICRA 2020
"""

from __future__ import annotations

import math
import pathlib

import cvxpy as opt
import numpy as np
import numpy.typing as npt
import yaml

from .dynamics import QuadrotorDynamics
from .utils import (
    Omega_half,
    yaw_from_velocity,
)
from .obstacle import ObstacleManager
from .uncertainty import UncertaintyPropagator
from . import _problem


class CCMPC:
    """Chance-Constrained Model Predictive Controller.

    Solves a receding-horizon QP with:
    - Linearized quadrotor dynamics (iMPC)
    - Chance-constrained obstacle avoidance (Eq 16)
    - Quadratic terminal, control, and yaw costs
    - State and control limits
    """

    def __init__(
        self,
        config: str | pathlib.Path | dict,
        horizon_time: float | None = None,
        timestep: float | None = None,
        terminal_cost: list[float] | None = None,
        control_cost: list[float] | None = None,
        yaw_cost: float | None = None,
    ) -> None:
        if isinstance(config, (str, pathlib.Path)):
            path = pathlib.Path(config)
            if not path.is_absolute():
                path = pathlib.Path(__file__).parent.parent / config
            with open(path) as f:
                config_data = yaml.safe_load(f)
        else:
            config_data = config

        # Dimensions
        self._state_dim: int = 9
        self._control_dim: int = 4

        # Model parameters
        model_cfg = config_data["model"]["quadrotor"]
        self.dynamics = QuadrotorDynamics(
            g=model_cfg["g"],
            kD=model_cfg["kD"],
            k_phi=model_cfg["k_phi"],
            k_theta=model_cfg["k_theta"],
            k_vz=model_cfg["k_vz"],
            tau_phi=model_cfg["tau_phi"],
            tau_theta=model_cfg["tau_theta"],
            tau_vz=model_cfg["tau_vz"],
        )

        # Prediction horizon
        pred_cfg = config_data["controller"]["prediction"]
        self.dt: float = timestep if timestep is not None else pred_cfg["timestep"]
        horizon = horizon_time if horizon_time is not None else pred_cfg["horizon_time"]
        self.control_horizon: int = int(horizon / self.dt)
        self._max_iter: int = pred_cfg.get("max_iter", 3)
        self._tol: float = pred_cfg.get("tolerance", 0.01)

        # Cost weights
        weights_cfg = config_data["controller"]["weights"]
        tc = terminal_cost if terminal_cost is not None else weights_cfg["terminal_cost"]
        cc = control_cost if control_cost is not None else weights_cfg["control_cost"]
        yc = yaw_cost if yaw_cost is not None else weights_cfg["yaw_cost"]

        # Terminal cost matrix (on position only)
        assert len(tc) == 3, "terminal_cost must be length 3 [x, y, z]"
        self.Qg: npt.NDArray[np.float64] = np.diag(tc)

        # Control cost matrix (on all 4 control inputs)
        assert len(cc) == 4, "control_cost must be length 4"
        self.R: npt.NDArray[np.float64] = np.sqrt(np.diag(cc))
        self.Q_psi: float = float(yc)

        # Obstacle parameters
        obs_cfg = config_data["controller"]["obstacle"]
        self.delta: float = obs_cfg["delta"]
        self.mav_radius: float = obs_cfg["mav_radius"]
        self._slack_penalty: float = obs_cfg["slack_penalty"]
        self._max_obs: int = obs_cfg.get("max_obstacles", 2)

        # Actuator limits
        limits_cfg = config_data["controller"]["limits"]
        self.max_roll: float = limits_cfg["max_roll"]
        self.max_pitch: float = limits_cfg["max_pitch"]
        self.max_vert_vel: float = limits_cfg["max_vert_vel"]
        self.max_yaw_rate: float = limits_cfg["max_yaw_rate"]
        self.max_speed: float = limits_cfg["max_speed"]

        # Solver preference
        self._solver: str = pred_cfg.get("solver", "CLARABEL")

        # Logistic obstacle cost parameters (Eq 12)
        lg_cfg = weights_cfg.get("logistic_cost", {})
        self._lg_Qo: float = lg_cfg.get("Q_o", 0.0)
        self._lg_lam: float = lg_cfg.get("lambda_o", 3.0)
        self._lg_ro: float = lg_cfg.get("r_o", 2.0)

        # Field of View constraints (Eq 17-18)
        fov_cfg = config_data["controller"].get("fov", {})
        self._fov_enabled: bool = fov_cfg.get("enabled", False)
        if self._fov_enabled:
            hfov = math.radians(fov_cfg["hfov_deg"])
            vfov = math.radians(fov_cfg["vfov_deg"])
            d_max = fov_cfg["max_range"]
            th = min(math.tan(hfov / 2.0), 100.0)
            tv = min(math.tan(vfov / 2.0), 100.0)
            # 5 half-spaces in body frame: left, right, bottom, top, depth
            # Constraint: n_j^T @ p_body ≤ m_j  (p_body relative to camera)
            # Left:   y ≥ -x·th  ⇒  -th·x - y ≤ 0   →  n=[-th, -1, 0], m=0
            # Right:  y ≤  x·th  ⇒  -th·x + y ≤ 0   →  n=[-th,  1, 0], m=0
            # Bottom: z ≥ -x·tv  ⇒  -tv·x - z ≤ 0   →  n=[-tv,  0,-1], m=0
            # Top:    z ≤  x·tv  ⇒  -tv·x + z ≤ 0   →  n=[-tv,  0, 1], m=0
            # Depth:  x ≤ d_max  ⇒   x ≤ d_max       →  n=[  1,  0, 0], m=d_max
            self._fov_n_body: list[np.ndarray] = [
                np.array([-th, -1.0, 0.0]),
                np.array([-th,  1.0, 0.0]),
                np.array([-tv,  0.0,-1.0]),
                np.array([-tv,  0.0, 1.0]),
                np.array([ 1.0, 0.0, 0.0]),
            ]
            self._fov_m: list[float] = [0.0, 0.0, 0.0, 0.0, d_max]
            self._fov_slack_penalty: float = fov_cfg.get("slack_penalty", 10000.0)
        else:
            self._fov_n_body = []
            self._fov_m = []
            self._fov_slack_penalty = 0.0

        # Uncertainty
        self.uncertainty = UncertaintyPropagator.from_config(config_data)

        # CVXPY variables
        self._states: opt.Variable = opt.Variable(
            (self._state_dim, self.control_horizon + 1), name="states"
        )
        self._controls: opt.Variable = opt.Variable(
            (self._control_dim, self.control_horizon), name="controls"
        )

        # CVXPY parameters
        self._initial_state: opt.Parameter = opt.Parameter(
            self._state_dim, name="x0"
        )
        self._goal: opt.Parameter = opt.Parameter(3, name="goal")
        self._last_command: opt.Parameter = opt.Parameter(
            self._control_dim, name="last_cmd"
        )

        # Linearized dynamics params
        self._A_params: list[opt.Parameter] = [
            opt.Parameter((self._state_dim, self._state_dim), name=f"A_{k}")
            for k in range(self.control_horizon)
        ]
        self._B_params: list[opt.Parameter] = [
            opt.Parameter((self._state_dim, self._control_dim), name=f"B_{k}")
            for k in range(self.control_horizon)
        ]
        self._C_params: list[opt.Parameter] = [
            opt.Parameter(self._state_dim, name=f"C_{k}")
            for k in range(self.control_horizon)
        ]

        # Chance constraint params — built in _build_problem()
        self._cc_a: list[opt.Parameter] = []
        self._cc_rhs: list[opt.Parameter] = []
        self._cc_slack: list[opt.Variable] = []

        # Reference trajectory params (straight line from current to goal)
        self._ref_x: list[opt.Parameter] = []
        self._ref_y: list[opt.Parameter] = []
        self._ref_z: list[opt.Parameter] = []

        self._yaw_ref: opt.Parameter | None = None

        # Build problem ONCE (DPP-compliant, fixed structure)
        self._build_problem()

        # Warm-start storage
        self._previous_trajectory: npt.NDArray[np.float64] | None = None
        self._previous_controls: npt.NDArray[np.float64] | None = None
        self._previous_covariance: npt.NDArray[np.float64] | None = None

    def _build_problem(self) -> None:
        """Build the QP problem once during init (DPP-compliant structure).
        
        Delegates to _problem module for CVXPY construction.
        """
        _problem.build_qp_problem(self)

    def _idx(self, k: int, i: int) -> int:
        """Flat index for chance constraint params."""
        return k * self._max_obs + i

    def _set_chance_constraint_params(
        self,
        k: int,
        idx: int,
        p_mav_guess: npt.NDArray[np.float64],
        p_obs: npt.NDArray[np.float64],
        L: npt.NDArray[np.float64],
        Sigma_mav: npt.NDArray[np.float64],
        Sigma_obs: npt.NDArray[np.float64],
    ) -> None:
        """Delegate to _problem module."""
        _problem.set_chance_constraint_params(
            self, k, idx, p_mav_guess, p_obs, L, Sigma_mav, Sigma_obs
        )

    def _compute_logistic_cost(
        self,
        x_guess: npt.NDArray[np.float64],
        obstacle_manager,
    ) -> None:
        """Delegate to _problem module."""
        _problem.compute_logistic_cost(self, x_guess, obstacle_manager)

    def _compute_fov_params(
        self,
        x_guess: npt.NDArray[np.float64],
        obstacle_manager=None,
    ) -> None:
        """Delegate to _problem module."""
        _problem.compute_fov_params(self, x_guess, obstacle_manager)

    def _dummy_chance_constraint(self, k: int, idx: int) -> None:
        """Delegate to _problem module."""
        _problem.dummy_chance_constraint(self, k, idx)

    def solve(
        self,
        initial_state: npt.NDArray[np.float64] | list[float],
        goal: npt.NDArray[np.float64] | list[float],
        solver: str | None = None,
        verbose: bool = False,
        obstacle_manager: ObstacleManager | None = None,
        Gamma_0: npt.NDArray[np.float64] | None = None,
        **solver_opts,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Solve the CC-MPC problem.

        Args:
            initial_state: Current state estimate (9,).
            goal: Goal position (3,).
            solver: CVXPY solver name (default: from config).
            verbose: Enable solver verbosity.
            obstacle_manager: Obstacle manager for obstacle data.
            Gamma_0: Initial state covariance (9x9). If None, uses default.
            **solver_opts: Additional CVXPY solver options.

        Returns:
            (state_trajectory (9, N+1), control_sequence (4, N)).
            On failure, returns emergency hover trajectory.
        """
        N = self.control_horizon
        initial_state = np.asarray(initial_state, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)

        # ---- Build initial guess (warm-start) ----
        if self._previous_trajectory is not None and self._previous_controls is not None:
            # Shift previous trajectory left by 1
            x_guess = np.roll(self._previous_trajectory, -1, axis=1)
            x_guess[:, -1] = self._previous_trajectory[:, -1]
            u_guess = np.roll(self._previous_controls, -1, axis=1)
            u_guess[:, -1] = self._previous_controls[:, -1]
            # Sync first state with current initial_state (warm-start alignment)
            x_guess[:, 0] = initial_state
        else:
            # First iteration: initial velocity toward goal for better warm-start
            x_guess = np.tile(initial_state, (N + 1, 1)).T
            goal_dir = goal - initial_state[:3]
            gd_norm = float(np.linalg.norm(goal_dir))
            if gd_norm > 0.1:
                target_speed = min(1.0, gd_norm / self.horizon_time)
                init_vel = goal_dir / gd_norm * target_speed
                for k in range(N):
                    x_guess[:3, k + 1] = x_guess[:3, k] + init_vel * self.dt
                    x_guess[3:6, k + 1] = init_vel
            else:
                for k in range(N):
                    x_guess[:3, k + 1] = x_guess[:3, k] + x_guess[3:6, k] * self.dt
            u_guess = np.zeros((4, N))

        # ---- Pre-compute uncertainty propagation (Eq 19) ----
        if Gamma_0 is None:
            Gamma_0 = self.uncertainty.Gamma_0.copy()
        Gamma_list = self.uncertainty.propagate(
            Gamma_0, x_guess, u_guess, self.dynamics
        )

        # ---- iMPC loop ----
        converged = False
        retry_fresh = True
        for iteration in range(self._max_iter):
            # 1. Set initial state, goal, and last command
            self._initial_state.value = initial_state
            self._goal.value = goal
            if self._previous_controls is not None:
                self._last_command.value = self._previous_controls[:, 0]
            else:
                self._last_command.value = np.zeros(self._control_dim)

            # 2. Linearize dynamics at each step
            for k in range(N):
                x_bar = x_guess[:, k]
                u_bar = u_guess[:, k]
                A_k, B_k, C_k = self.dynamics.linearize(x_bar, u_bar, self.dt)
                self._A_params[k].value = A_k
                self._B_params[k].value = B_k
                self._C_params[k].value = C_k

            # 3. Yaw reference: point toward goal (better for FOV keeping)
            goal_vec = goal - x_guess[:3, 0]
            yaw_desired = math.atan2(goal_vec[1], goal_vec[0])
            self._yaw_ref.value = np.full(N + 1, yaw_desired)

            # 3b. Reference trajectory: straight line from current to goal
            p_start = initial_state[:3]
            for k in range(N):
                frac = (k + 1) / N
                ref_k = p_start + (goal - p_start) * frac
                self._ref_x[k].value = float(ref_k[0])
                self._ref_y[k].value = float(ref_k[1])
                self._ref_z[k].value = float(ref_k[2])

            # 4. Set chance constraint params
            # Predict obstacle positions for each horizon step using
            # the last-loop trajectory as the linearization point.
            obs_predictions: list[list[dict]] = [[] for _ in range(N)]
            if obstacle_manager is not None:
                for i_obs, obs in enumerate(obstacle_manager.obstacles):
                    if i_obs >= self._max_obs:
                        break
                    # Pre-compute Omega^{1/2} for this obstacle (constant size/orientation)
                    inv_sq = 1.0 / (obs.axes + self.mav_radius) ** 2
                    Omega = obs.R_o.T @ np.diag(inv_sq) @ obs.R_o
                    L_const = Omega_half(Omega)

                    pos = obs.p_hat.copy()
                    vel = obs.v_hat.copy()
                    Sigma = obs.Sigma.copy()
                    for k in range(N):
                        p_mav_guess_k = x_guess[:3, k + 1]
                        obs_predictions[k].append(dict(
                            pos=pos.copy(),
                            Sigma=Sigma.copy(),
                            L=L_const,
                        ))
                        pos = pos + vel * self.dt
                        Sigma = Sigma + obs.Sigma_v * self.dt**2

            for k in range(N):
                p_mav_guess_k = x_guess[:3, k + 1]
                Sigma_mav = self.uncertainty.position_covariance(Gamma_list[k + 1])

                for i_obs in range(self._max_obs):
                    if i_obs < len(obs_predictions[k]):
                        pred = obs_predictions[k][i_obs]
                        self._set_chance_constraint_params(
                            k, i_obs,
                            p_mav_guess_k, pred["pos"], pred["L"],
                            Sigma_mav, pred["Sigma"],
                        )
                    else:
                        self._dummy_chance_constraint(k, i_obs)

            # 4a. Set logistic cost gradient (Eq 12)
            self._compute_logistic_cost(x_guess, obstacle_manager)

            # 4b. Set FOV constraint params (Eq 17-18)
            self._compute_fov_params(x_guess, obstacle_manager)

            # 5. Solve QP
            try:
                solver_name = solver or self._solver
                solver_instance = getattr(opt, solver_name)
                # Solver-specific options
                sopts = dict(solver_opts)
                if solver_name == 'CLARABEL':
                    sopts.setdefault('max_iter', 100)
                    sopts.setdefault('tol_gap_abs', 1e-5)
                    sopts.setdefault('tol_gap_rel', 1e-5)
                    sopts.setdefault('tol_feas', 1e-5)
                self._problem.solve(
                    solver=solver_instance,
                    verbose=verbose,
                    warm_start=False,
                    **sopts,
                )
            except Exception as e:
                print(f"  [CCMPC] Iter {iteration}: Solver exception: {e}")
                break

            # 6. Check solver status
            if self._states.value is None:
                if iteration == 0 and retry_fresh:
                    retry_fresh = False
                    x_guess = np.tile(initial_state, (N + 1, 1)).T
                    gd = np.linalg.norm(goal - initial_state[:3])
                    if gd > 0.1:
                        tv = min(2.0, gd / self.horizon_time)
                        iv = (goal - initial_state[:3]) / gd * tv
                        for k in range(N):
                            x_guess[:3, k + 1] = x_guess[:3, k] + iv * self.dt
                            x_guess[3:6, k + 1] = iv
                    u_guess = np.zeros((4, N))
                    self._previous_trajectory = None
                    self._previous_controls = None
                    continue
                if iteration >= self._max_iter - 1:
                    print(f"  [CCMPC] All retries failed -> PID fallback")
                    return self._emergency_hover(initial_state, goal)

            new_x = np.array(self._states.value)
            new_u = np.array(self._controls.value)

            # 7. Check convergence
            max_dev = np.max(np.abs(new_x - x_guess))
            if max_dev < self._tol:
                converged = True
                x_guess = new_x
                u_guess = new_u
                break

            # 8. Update guess for next iMPC iteration
            x_guess = new_x
            u_guess = new_u

        # ---- Store for next control cycle (warm-start) ----
        self._previous_trajectory = np.copy(x_guess)
        self._previous_controls = np.copy(u_guess)
        self._previous_covariance = Gamma_list[-1].copy()

        if verbose:
            status = "converged" if converged else f"max_iter ({self._max_iter})"
            print(f"  [CCMPC] Solved: {status}, "
                  f"cost={self._problem.value:.2f}")

        return x_guess, u_guess

    def _emergency_hover(
        self, state: npt.NDArray[np.float64], goal: npt.NDArray[np.float64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """PID go-to-goal fallback when MPC solver fails.

        Uses simple P controllers for yaw, pitch, and altitude
        to fly toward the goal. Decelerates when close.
        """
        N = self.control_horizon
        dt = self.dt
        x_traj = np.zeros((9, N + 1))
        x_traj[:, 0] = state
        u_traj = np.zeros((4, N))
        # If no goal, just yaw in place
        g = goal if goal is not None else state[:3] + np.array([1, 0, 0])
        for k in range(N):
            p_k = x_traj[:3, k]
            v_k = x_traj[3:6, k]
            speed = float(np.linalg.norm(v_k))
            dist_to_goal = float(np.linalg.norm(p_k - g))
            on_ground = p_k[2] < 0.2
            # Yaw toward goal
            des_yaw_k = math.atan2(g[1] - p_k[1], g[0] - p_k[0])
            yaw_err_k = (des_yaw_k - x_traj[8, k] + math.pi) % (2 * math.pi) - math.pi
            u_traj[3, k] = float(np.clip(2.0 * yaw_err_k, -self.max_yaw_rate, self.max_yaw_rate))
            if on_ground:
                u_traj[:] = 0.0
                u_traj[2, k] = self.max_vert_vel  # max climb, no tilt
                u_traj[3, k] = float(np.clip(2.0 * yaw_err_k, -self.max_yaw_rate, self.max_yaw_rate))
            else:
                # Speed regulation: 2 m/s max, decelerate when close
                speed_target = np.clip(dist_to_goal * 0.5, 0.0, 2.0)
                pitch_cmd = np.clip((speed_target - speed) * 0.3, -self.max_pitch * 0.3, self.max_pitch * 0.3)
                u_traj[1, k] = float(pitch_cmd)
                u_traj[0, k] = float(np.clip(-yaw_err_k * 0.1, -self.max_roll * 0.3, self.max_roll * 0.3))
                z_err = g[2] - p_k[2]
                u_traj[2, k] = float(np.clip(2.0 * z_err, -self.max_vert_vel, self.max_vert_vel))
            x_traj[:, k + 1] = self.dynamics.discrete(x_traj[:, k], u_traj[:, k], dt)
        return x_traj, u_traj

    @property
    def horizon_time(self) -> float:
        """Total prediction horizon in seconds."""
        return self.control_horizon * self.dt
