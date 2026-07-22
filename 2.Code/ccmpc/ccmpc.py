"""
Chance-Constrained MPC for quadrotor obstacle avoidance.

Implements the formulations from:
  "Robust Vision-based Obstacle Avoidance for Micro Aerial Vehicles
   in Dynamic Environments" — Lin, Zhu, Alonso-Mora, ICRA 2020
"""

from __future__ import annotations

import math
import pathlib
import warnings

import cvxpy as opt
import numpy as np
import numpy.typing as npt
import yaml

from .dynamics import QuadrotorDynamics
from .utils import (
    chance_constraint_rhs,
    erfinv,
    yaw_from_velocity,
    Omega_half,
)
from .obstacle import ObstacleManager
from .uncertainty import UncertaintyPropagator


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
        solver: str | None = None,   # FIX #22: allow CLI to override solver
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
        # FIX #1 (CRITICAL): Altitude floor — was missing, drone could plan below ground.
        self._min_altitude: float = limits_cfg.get("min_altitude", 0.15)

        # Solver preference (kwarg overrides config)
        # FIX #22: __init__ now accepts solver= so the CLI arg propagates correctly.
        self._solver: str = solver if solver is not None else pred_cfg.get("solver", "CLARABEL")

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
        # DPP FIX: Instead of storing goal as a 3D Parameter and building
        # quad_form(x - goal_param, Qg), which is NOT DPP when multiplied by
        # another Parameter (dist_scale), we pre-expand the terminal cost:
        #
        #   scale * (x-g)^T Qg (x-g)
        #   = scale * x^T Qg x  -  2*(scale*Qg*g)^T x  + const(ignored)
        #
        # Both `scale * quad_form(x, Qg)` and `-(scale_Qg_g)^T x` are DPP
        # because each Parameter appears in only one atom.
        # We store: _goal_scale (scalar param) and _goal_Qg_vec (3D param = scale*Qg@goal)
        self._goal_scale: opt.Parameter = opt.Parameter(nonneg=True, name="goal_scale")
        self._goal_Qg_vec: opt.Parameter = opt.Parameter(3, name="goal_Qg_vec")
        # Keep _goal for backward compat (used in _emergency_hover distance calc)
        self._goal_np: npt.NDArray[np.float64] = np.zeros(3)

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
        """Build the QP problem once during init (DPP-compliant structure)."""
        cost = 0.0
        constraints = []

        # Yaw reference parameter
        self._yaw_ref = opt.Parameter(self.control_horizon + 1, name="yaw_ref")

        # Chance constraint params (flat: index = k * max_obs + i_obs)
        # Constraint: a_k^T @ p_{k+1} - 1 + slack >= rhs_combined
        # rhs_combined pre-computed as rhs + a_k^T @ p_obs (single scalar param)
        # to avoid Parameter @ Parameter expressions (DPP requirement).
        self._cc_a: list[opt.Parameter] = []
        self._cc_rhs: list[opt.Parameter] = []
        self._cc_slack: list[opt.Variable] = []

        # Logistic cost params (linearized gradient w.r.t. position)
        self._lg_grad: list[opt.Parameter] = []

        # FOV constraint params (5 half-spaces per step)
        self._fov_a: list[opt.Parameter] = []   # normals in world frame (3,)
        self._fov_b: list[opt.Parameter] = []   # RHS scalars
        self._fov_slack: list[opt.Variable] = []  # slack variables

        for k in range(self.control_horizon):
            constraints += [
                self._states[:, k + 1]
                == self._A_params[k] @ self._states[:, k]
                + self._B_params[k] @ self._controls[:, k]
                + self._C_params[k]
            ]

            for i_obs in range(self._max_obs):
                a_k_i = opt.Parameter(3, name=f"a_{k}_{i_obs}")
                rhs_k_i = opt.Parameter(name=f"rhs_{k}_{i_obs}")
                slack_k_i = opt.Variable(nonneg=True, name=f"slack_{k}_{i_obs}")

                self._cc_a.append(a_k_i)
                self._cc_rhs.append(rhs_k_i)
                self._cc_slack.append(slack_k_i)

                p_k1 = self._states[:3, k + 1]
                constraints += [
                    a_k_i @ p_k1 - 1.0 + slack_k_i >= rhs_k_i
                ]
                cost += self._slack_penalty * slack_k_i

            # Logistic obstacle cost (Eq 12) — linearized: grad^T · p_k
            if self._lg_Qo > 0.0:
                g_k = opt.Parameter(3, name=f"lg_{k}")
                self._lg_grad.append(g_k)
                cost += g_k @ self._states[:3, k + 1]

            # FOV obstacle-visibility constraints (Eq 17-18)
            # FIX #16 (MEDIUM): original code only enforced the single worst-violated
            # half-space per step, leaving the other 4 unchecked. We now build one
            # parameter pair per half-space (5 per step) so ALL are enforced.
            # Total extra params: 5 * N (a 3-vector + scalar per half-space per step).
            if self._fov_enabled:
                step_fov_a = []
                step_fov_b = []
                for j in range(5):
                    a_j = opt.Parameter(3, name=f"fov_a_{k}_{j}")
                    b_j = opt.Parameter(name=f"fov_b_{k}_{j}")
                    step_fov_a.append(a_j)
                    step_fov_b.append(b_j)
                s_obs = opt.Variable(nonneg=True, name=f"fov_s_{k}")
                self._fov_a.append(step_fov_a)   # list of 5 Parameter(3)
                self._fov_b.append(step_fov_b)   # list of 5 Parameter
                self._fov_slack.append(s_obs)
                for j in range(5):
                    constraints += [
                        step_fov_b[j] - step_fov_a[j] @ self._states[:3, k + 1] + s_obs >= 0.0
                    ]
                cost += self._fov_slack_penalty * s_obs

            # Per-step reference tracking (mpc_python pattern): 
            # follow a straight line from current to goal.
            ref_x_k = opt.Parameter(name=f"ref_x_{k}")
            ref_y_k = opt.Parameter(name=f"ref_y_{k}")
            ref_z_k = opt.Parameter(name=f"ref_z_{k}")
            self._ref_x.append(ref_x_k)
            self._ref_y.append(ref_y_k)
            self._ref_z.append(ref_z_k)
            # Per-step reference tracking with SEPARATE z weight:
            # z needs ~5x more weight than x,y because:
            # 1. z has separate dynamics (vz_c channel) not coupled to xy speed
            # 2. At large tilt angles, gravity reduces z-lift — need aggressive correction
            # 3. The altitude floor penalty (soft) needs support from the tracking cost
            _w_xy = 0.5   # weight for x, y tracking
            _w_z  = 5.0   # weight for z tracking (much higher than x,y)
            pos_err_xy = opt.vstack([
                self._states[0, k + 1] - ref_x_k,
                self._states[1, k + 1] - ref_y_k,
            ])
            cost += _w_xy * opt.sum_squares(pos_err_xy)
            cost += _w_z  * (self._states[2, k + 1] - ref_z_k) ** 2

            cost += opt.sum_squares(self.R @ self._controls[:, k])
            cost += self.Q_psi * (self._states[8, k] - self._yaw_ref[k]) ** 2

        # Terminal cost — DPP-compliant expanded form:
        #   scale * x_T^T Qg x_T  -  2 * (scale*Qg*goal)^T x_T
        # where x_T = states[:3, -1].
        # This is DPP because each Parameter (_goal_scale, _goal_Qg_vec) appears
        # in exactly one atom and each is affine in Variables.
        x_terminal_pos = self._states[:3, -1]
        cost += (self._goal_scale * opt.quad_form(x_terminal_pos, self.Qg)
                 - 2.0 * self._goal_Qg_vec @ x_terminal_pos)
        # Strong terminal yaw cost
        cost += self.Q_psi * 10.0 * (self._states[8, -1] - self._yaw_ref[-1]) ** 2

        constraints += [self._states[:, 0] == self._initial_state]

        # ---- State bounds — SOFT constraints with slack variables ----
        # Hard state constraints cause INFEASIBILITY when the current state
        # (or linearized model prediction) violates the bound — e.g. high
        # horizontal speed + low altitude means the linearized model predicts
        # the drone will drop below min_altitude even under max braking.
        # Solution: ALL state bounds are soft (slack >= 0, penalized in cost).
        # The slack_penalty is large enough to make violations expensive but
        # the problem always remains feasible.
        N_ctrl = self.control_horizon
        _SPalt  = 5000.0   # altitude floor — very expensive to violate
        _SPvz   = 2000.0   # vertical velocity
        _SPatt  = 500.0    # roll/pitch angles (safety critical)
        _SPspd  = 200.0    # horizontal speed

        # Altitude floor (FIX #1)
        slack_z = opt.Variable(N_ctrl, nonneg=True, name="slack_z")
        constraints += [self._states[2, 1:] + slack_z >= self._min_altitude]
        cost += _SPalt * opt.sum(slack_z)

        # vz state bound (FIX #2)
        slack_vz_hi = opt.Variable(N_ctrl, nonneg=True, name="slack_vz_hi")
        slack_vz_lo = opt.Variable(N_ctrl, nonneg=True, name="slack_vz_lo")
        constraints += [self._states[5, 1:] <= self.max_vert_vel  + slack_vz_hi]
        constraints += [self._states[5, 1:] >= -self.max_vert_vel - slack_vz_lo]
        cost += _SPvz * opt.sum(slack_vz_hi + slack_vz_lo)

        # Roll/pitch state bounds (FIX #3)
        slack_phi_hi = opt.Variable(N_ctrl, nonneg=True, name="slack_phi_hi")
        slack_phi_lo = opt.Variable(N_ctrl, nonneg=True, name="slack_phi_lo")
        constraints += [self._states[6, 1:] <= self.max_roll  + slack_phi_hi]
        constraints += [self._states[6, 1:] >= -self.max_roll - slack_phi_lo]
        cost += _SPatt * opt.sum(slack_phi_hi + slack_phi_lo)

        slack_tht_hi = opt.Variable(N_ctrl, nonneg=True, name="slack_tht_hi")
        slack_tht_lo = opt.Variable(N_ctrl, nonneg=True, name="slack_tht_lo")
        constraints += [self._states[7, 1:] <= self.max_pitch  + slack_tht_hi]
        constraints += [self._states[7, 1:] >= -self.max_pitch - slack_tht_lo]
        cost += _SPatt * opt.sum(slack_tht_hi + slack_tht_lo)

        # Horizontal speed bounds (FIX: soft constraint to prevent infeasibility at high speed)
        slack_vx_hi = opt.Variable(N_ctrl, nonneg=True, name="slack_vx_hi")
        slack_vx_lo = opt.Variable(N_ctrl, nonneg=True, name="slack_vx_lo")
        constraints += [self._states[3, 1:] <= self.max_speed  + slack_vx_hi]
        constraints += [self._states[3, 1:] >= -self.max_speed - slack_vx_lo]
        cost += _SPspd * opt.sum(slack_vx_hi + slack_vx_lo)

        slack_vy_hi = opt.Variable(N_ctrl, nonneg=True, name="slack_vy_hi")
        slack_vy_lo = opt.Variable(N_ctrl, nonneg=True, name="slack_vy_lo")
        constraints += [self._states[4, 1:] <= self.max_speed  + slack_vy_hi]
        constraints += [self._states[4, 1:] >= -self.max_speed - slack_vy_lo]
        cost += _SPspd * opt.sum(slack_vy_hi + slack_vy_lo)

        # Control bounds (HARD — actuator limits never change)
        constraints += [self._controls[0, :] <= self.max_roll,
                         self._controls[0, :] >= -self.max_roll]
        constraints += [self._controls[1, :] <= self.max_pitch,
                         self._controls[1, :] >= -self.max_pitch]
        constraints += [self._controls[2, :] <= self.max_vert_vel,
                         self._controls[2, :] >= -self.max_vert_vel]
        constraints += [self._controls[3, :] <= self.max_yaw_rate,
                         self._controls[3, :] >= -self.max_yaw_rate]

        self._problem = opt.Problem(opt.Minimize(cost), constraints)

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
        """Set CVXPY parameters for one chance constraint (DPP-compliant).

        Computes a = n^T @ L (row vector) and folds a @ p_obs into
        the RHS to avoid Parameter @ Parameter expressions.
        """
        diff = p_mav_guess - p_obs
        dist = np.linalg.norm(diff)
        n_o = diff / dist if dist >= 1e-6 else np.array([1.0, 0.0, 0.0])

        rhs = chance_constraint_rhs(L, Sigma_mav, Sigma_obs, n_o, self.delta)
        a_vec = n_o @ L
        rhs_combined = float(rhs) + float(a_vec @ p_obs)

        flat = self._idx(k, idx)
        self._cc_a[flat].value = a_vec
        self._cc_rhs[flat].value = rhs_combined

    def _compute_logistic_cost(
        self,
        x_guess: npt.NDArray[np.float64],
        obstacle_manager,
    ) -> None:
        """Linearize logistic cost (Eq 12) around guess trajectory.

        J(d) = Q_o / (1 + exp(λ_o(d - r_o)))
        Linearized as: J'(d_guess) · n_o^T · p̂_k  (gradient · position)

        Only active when logistic cost weight Q_o > 0.
        """
        if self._lg_Qo <= 0.0 or not self._lg_grad:
            return

        if obstacle_manager is None or not obstacle_manager.obstacles:
            for k in range(self.control_horizon):
                self._lg_grad[k].value = np.zeros(3)
            return

        obs = obstacle_manager.get_closest(x_guess[:3, 0], k=1)[0]

        for k in range(self.control_horizon):
            p_mav = x_guess[:3, k + 1]
            p_obs = obs.p_hat + obs.v_hat * self.dt * k
            diff = p_mav - p_obs
            d = float(np.linalg.norm(diff))

            if d < 1e-6:
                self._lg_grad[k].value = np.zeros(3)
                continue

            # Logistic value and derivative
            exp_arg = math.exp(self._lg_lam * (d - self._lg_ro))
            J_deriv = -self._lg_lam * self._lg_Qo * exp_arg / ((1.0 + exp_arg) ** 2)

            # Gradient w.r.t. p: J'(d) * n_o^T, capped to avoid solver instability
            n_o = diff / d
            grad = J_deriv * n_o
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm > 5.0:
                grad = grad * (5.0 / grad_norm)
            self._lg_grad[k].value = grad

    def _compute_fov_params(
        self,
        x_guess: npt.NDArray[np.float64],
        obstacle_manager=None,
    ) -> None:
        """Compute FOV constraints: p^{k+1} must be within FOV^k (Eq 17-18).

        The next waypoint p_{k+1} must be visible from the current camera
        position p_k with yaw ψ_k. In body frame:
            n_body_j · R_yaw^T · (p_{k+1} - p_k) ≤ m_j   for each half-space j (j=0..4)

        FIX #16 (MEDIUM): original code only activated the single worst-violated
        half-space. Now all 5 half-space params are filled for every step so the
        solver sees and enforces the complete FOV pyramid.
        """
        if not self._fov_enabled:
            return
        for k in range(self.control_horizon):
            yaw_k = x_guess[8, k]
            ct, st = math.cos(yaw_k), math.sin(yaw_k)
            R_yaw = np.array([
                [ct, -st, 0.0],
                [st,  ct, 0.0],
                [0.0, 0.0, 1.0],
            ])
            p_k = x_guess[:3, k]

            # Fill all 5 half-space constraints for this step
            for j in range(5):
                n_j = self._fov_n_body[j]
                m_j = self._fov_m[j]
                # World-frame normal
                a_world = R_yaw @ n_j
                b_val = float(m_j) + float(a_world @ p_k)
                self._fov_a[k][j].value = a_world
                self._fov_b[k][j].value = b_val

    def _dummy_chance_constraint(self, k: int, idx: int) -> None:
        """Set params to a trivially satisfied constraint (no obstacle)."""
        flat = self._idx(k, idx)
        a_vec = np.array([0.0, 0.0, 1e-6])
        rhs_combined = -1000.0
        self._cc_a[flat].value = a_vec
        self._cc_rhs[flat].value = rhs_combined

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
            # FIX #5 (CRITICAL): Re-integrate from actual current state so the
            # warm-start trajectory is physically consistent.  The old code only
            # set x_guess[:,0] = initial_state but left x_guess[:,1:] from the
            # previous cycle, which can be metres away from any feasible path
            # when the drone has moved significantly between MPC calls.
            x_guess[:, 0] = initial_state
            # Reintegrate first step to anchor the trajectory
            x_guess[:, 1] = self.dynamics.discrete(initial_state, u_guess[:, 0], self.dt)
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
        fresh_attempts = 0      # FIX #4: track how many fresh restarts we've done
        _max_fresh = 2          # allow up to 2 independent fresh restarts
        for iteration in range(self._max_iter):
            # 1. Set initial state, goal, and last command
            self._initial_state.value = initial_state
            # DPP-compliant goal params: set _goal_scale and _goal_Qg_vec
            # instead of _goal (which caused param*param non-DPP violation).
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
            goal_dist = float(np.linalg.norm(goal_vec))
            yaw_desired = math.atan2(goal_vec[1], goal_vec[0])
            self._yaw_ref.value = np.full(N + 1, yaw_desired)

            # FIX #13 + DPP: compute scale and set the two DPP-compliant params.
            # Expanded terminal cost: scale*x^T Qg x - 2*(scale*Qg*goal)^T x
            horizon_reach = self.max_speed * self.horizon_time
            dist_scale = float(np.clip(goal_dist / max(horizon_reach, 1.0), 0.5, 3.0))
            self._goal_scale.value = dist_scale
            self._goal_Qg_vec.value = dist_scale * (self.Qg @ goal)

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
                # FIX #4 (CRITICAL): old code used a single fresh_attempt flag that
                # could only trigger ONE retry, and after that immediately fell through
                # to emergency hover without re-entering the iMPC loop.
                # New logic: up to _max_fresh restarts, each re-entering the full loop.
                if fresh_attempts < _max_fresh:
                    fresh_attempts += 1
                    print(f"  [CCMPC] Solver fail (iter {iteration}), "
                          f"fresh restart {fresh_attempts}/{_max_fresh}...")
                    N = self.control_horizon
                    # Build a new straight-line guess toward goal
                    x_guess = np.tile(initial_state, (N + 1, 1)).T
                    goal_dir = goal - initial_state[:3]
                    gd_norm = float(np.linalg.norm(goal_dir))
                    if gd_norm > 0.1:
                        # Vary target speed per attempt to escape local infeasibility
                        speed_frac = [1.0, 0.5][fresh_attempts - 1]
                        target_speed = min(speed_frac * 2.0, gd_norm / self.horizon_time)
                        init_vel = goal_dir / gd_norm * target_speed
                        for k in range(N):
                            x_guess[:3, k + 1] = x_guess[:3, k] + init_vel * self.dt
                            x_guess[3:6, k + 1] = init_vel
                    u_guess = np.zeros((4, N))
                    # Invalidate warm-start so next call also uses a fresh guess
                    self._previous_trajectory = None
                    self._previous_controls = None
                    # Re-enter iMPC from iteration 0
                    continue
                print(f"  [CCMPC] All {_max_fresh} fresh restarts exhausted -> emergency hover")
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