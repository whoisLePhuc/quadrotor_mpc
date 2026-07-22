"""
CVXPY problem construction for CC-MPC.

Extracted from ccmpc.py to keep the controller class focused on the solve loop.
All CVXPY variable/parameter/constraint creation lives here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import cvxpy as opt
import numpy as np
import numpy.typing as npt

from .utils import chance_constraint_rhs

if TYPE_CHECKING:
    from .ccmpc import CCMPC
    from .obstacle import ObstacleManager


def build_qp_problem(mpc: CCMPC) -> None:
    """Build the CVXPY QP problem and attach all vars/params to the CCMPC instance.

    Sets up:
      - State/control variables, initial state and goal params
      - Linearized dynamics params (A_k, B_k, C_k)
      - Chance constraint params (a, rhs, slack per obstacle per step)
      - Logistic cost gradient params
      - FOV constraint params
      - Reference trajectory params
      - Terminal, control, yaw, and slack costs
      - State and control bound constraints
    """
    N = mpc.control_horizon
    cost: Any = 0.0  # CVXPY expression, not float
    constraints: list = []

    # --- Yaw reference parameter ---
    mpc._yaw_ref = opt.Parameter(N + 1, name="yaw_ref")

    # --- Chance constraint params ---
    _setup_chance_constraint_params(mpc, N)
    for k in range(N):
        for i_obs in range(mpc._max_obs):
            a_k_i = opt.Parameter(3, name=f"a_{k}_{i_obs}")
            rhs_k_i = opt.Parameter(name=f"rhs_{k}_{i_obs}")
            slack_k_i = opt.Variable(nonneg=True, name=f"slack_{k}_{i_obs}")
            mpc._cc_a.append(a_k_i)
            mpc._cc_rhs.append(rhs_k_i)
            mpc._cc_slack.append(slack_k_i)

            p_k1 = mpc._states[:3, k + 1]
            constraints.append(a_k_i @ p_k1 - 1.0 + slack_k_i >= rhs_k_i)
            cost += mpc._slack_penalty * slack_k_i

    # --- Logistic obstacle cost gradient params ---
    _setup_logistic_cost_params(mpc, N)

    # --- FOV constraint params ---
    _setup_fov_params(mpc, N)
    if mpc._fov_enabled:
        for k in range(N):
            for j in range(5):
                a_fov = opt.Parameter(3, name=f"fov_{j}_{k}")
                b_fov = opt.Parameter(name=f"fov_b_{j}_{k}")
                s_fov = opt.Variable(nonneg=True, name=f"fov_s_{j}_{k}")
                mpc._fov_a.append(a_fov)
                mpc._fov_b.append(b_fov)
                mpc._fov_slack.append(s_fov)
                constraints.append(
                    b_fov - a_fov @ mpc._states[:3, k + 1] + s_fov >= 0.0
                )
                cost += mpc._fov_slack_penalty * s_fov

    # --- Reference trajectory params ---
    _setup_reference_params(mpc, N)

    # --- Dynamics constraints ---
    for k in range(N):
        constraints.append(
            mpc._states[:, k + 1]
            == mpc._A_params[k] @ mpc._states[:, k]
            + mpc._B_params[k] @ mpc._controls[:, k]
            + mpc._C_params[k]
        )

    # --- Per-step costs ---
    for k in range(N):
        cost += _build_step_costs(mpc, k)

    # --- Terminal cost ---
    terminal_position = opt.vstack([
        mpc._states[0, -1] - mpc._goal[0],
        mpc._states[1, -1] - mpc._goal[1],
        mpc._states[2, -1] - mpc._goal[2],
    ])
    cost += opt.quad_form(terminal_position, mpc.Qg)
    cost += mpc.Q_psi * 10.0 * (mpc._states[8, -1] - mpc._yaw_ref[-1]) ** 2

    # --- Initial state + bounds ---
    constraints.append(mpc._states[:, 0] == mpc._initial_state)
    _add_bounds(mpc, N, constraints)

    mpc._problem = opt.Problem(opt.Minimize(cost), constraints)


def _setup_chance_constraint_params(mpc: CCMPC, _N: int) -> None:
    """Initialize empty lists for chance constraint parameters."""
    mpc._cc_a = []
    mpc._cc_rhs = []
    mpc._cc_slack = []


def _setup_logistic_cost_params(mpc: CCMPC, N: int) -> None:
    """Initialize logistic cost gradient parameter list."""
    mpc._lg_grad = []
    if mpc._lg_Qo > 0.0:
        for k in range(N):
            g_k = opt.Parameter(3, name=f"lg_{k}")
            mpc._lg_grad.append(g_k)


def _setup_fov_params(mpc: CCMPC, N: int) -> None:
    """Initialize FOV constraint parameter lists."""
    mpc._fov_a = []
    mpc._fov_b = []
    mpc._fov_slack = []


def _setup_reference_params(mpc: CCMPC, N: int) -> None:
    """Create reference trajectory parameters (straight line waypoints)."""
    mpc._ref_x = []
    mpc._ref_y = []
    mpc._ref_z = []
    for k in range(N):
        mpc._ref_x.append(opt.Parameter(name=f"ref_x_{k}"))
        mpc._ref_y.append(opt.Parameter(name=f"ref_y_{k}"))
        mpc._ref_z.append(opt.Parameter(name=f"ref_z_{k}"))


def _build_step_costs(mpc: CCMPC, k: int) -> Any:
    """Build per-step cost contributions: reference tracking, control, yaw, logistic."""
    cost: Any = 0.0

    # Reference tracking
    pos_err = opt.vstack([
        mpc._states[0, k + 1] - mpc._ref_x[k],
        mpc._states[1, k + 1] - mpc._ref_y[k],
        mpc._states[2, k + 1] - mpc._ref_z[k],
    ])
    cost += 0.5 * opt.sum_squares(pos_err)

    # Control effort
    cost += opt.sum_squares(mpc.R @ mpc._controls[:, k])

    # Yaw alignment
    cost += mpc.Q_psi * (mpc._states[8, k] - mpc._yaw_ref[k]) ** 2

    # Logistic obstacle cost
    if mpc._lg_Qo > 0.0 and k < len(mpc._lg_grad) and mpc._lg_grad[k] is not None:
        cost += mpc._lg_grad[k] @ mpc._states[:3, k + 1]

    return cost


def _add_bounds(mpc: CCMPC, N: int, constraints: list) -> None:
    """Add state and control bound constraints."""
    constraints.append(mpc._states[2, 1:] >= 0.1)  # altitude floor

    # Velocity bounds
    constraints.extend([
        mpc._states[3, 1:] <= mpc.max_speed,
        mpc._states[3, 1:] >= -mpc.max_speed,
        mpc._states[4, 1:] <= mpc.max_speed,
        mpc._states[4, 1:] >= -mpc.max_speed,
        mpc._states[5, 1:] <= mpc.max_vert_vel,
        mpc._states[5, 1:] >= -mpc.max_vert_vel,
    ])

    # Attitude bounds
    constraints.extend([
        mpc._states[6, 1:] <= 0.5,
        mpc._states[6, 1:] >= -0.5,
        mpc._states[7, 1:] <= 0.5,
        mpc._states[7, 1:] >= -0.5,
    ])

    # Control bounds
    constraints.extend([
        mpc._controls[0, :] <= mpc.max_roll,
        mpc._controls[0, :] >= -mpc.max_roll,
        mpc._controls[1, :] <= mpc.max_pitch,
        mpc._controls[1, :] >= -mpc.max_pitch,
        mpc._controls[2, :] <= mpc.max_vert_vel,
        mpc._controls[2, :] >= -mpc.max_vert_vel,
        mpc._controls[3, :] <= mpc.max_yaw_rate,
        mpc._controls[3, :] >= -mpc.max_yaw_rate,
    ])


# ---------------------------------------------------------------------------
# Chance constraint param setters (called from CCMPC.solve)
# ---------------------------------------------------------------------------
def set_chance_constraint_params(
    mpc: CCMPC,
    k: int,
    idx: int,
    p_mav_guess: npt.NDArray[np.float64],
    p_obs: npt.NDArray[np.float64],
    L: npt.NDArray[np.float64],
    Sigma_mav: npt.NDArray[np.float64],
    Sigma_obs: npt.NDArray[np.float64],
) -> None:
    """Set CVXPY parameters for one chance constraint (DPP-compliant).

    Computes a = n^T @ L and folds a @ p_obs into the RHS to avoid
    Parameter @ Parameter expressions.
    """
    diff = p_mav_guess - p_obs
    dist = np.linalg.norm(diff)
    n_o = diff / dist if dist >= 1e-6 else np.array([1.0, 0.0, 0.0])

    rhs = chance_constraint_rhs(L, Sigma_mav, Sigma_obs, n_o, mpc.delta)
    a_vec = n_o @ L
    rhs_combined = float(rhs) + float(a_vec @ p_obs)

    flat = mpc._idx(k, idx)
    mpc._cc_a[flat].value = a_vec
    mpc._cc_rhs[flat].value = rhs_combined


def dummy_chance_constraint(mpc: CCMPC, k: int, idx: int) -> None:
    """Set params to a trivially satisfied constraint (no obstacle in slot)."""
    flat = mpc._idx(k, idx)
    a_vec = np.array([0.0, 0.0, 1e-6])
    rhs_combined = -1000.0
    mpc._cc_a[flat].value = a_vec
    mpc._cc_rhs[flat].value = rhs_combined


# ---------------------------------------------------------------------------
# Logistic cost linearization (Eq 12)
# ---------------------------------------------------------------------------
def compute_logistic_cost(
    mpc: CCMPC,
    x_guess: npt.NDArray[np.float64],
    obstacle_manager: ObstacleManager | None,
) -> None:
    """Linearize logistic cost (Eq 12) around guess trajectory.

    J(d) = Q_o / (1 + exp(λ_o(d - r_o)))
    Linearized as: J'(d_guess) · n_o^T · p̂_k  (gradient · position)

    Only active when logistic cost weight Q_o > 0.
    """
    N = mpc.control_horizon
    if mpc._lg_Qo <= 0.0 or not mpc._lg_grad:
        return

    if obstacle_manager is None or not obstacle_manager.obstacles:
        for k in range(N):
            mpc._lg_grad[k].value = np.zeros(3)
        return

    obs = obstacle_manager.get_closest(x_guess[:3, 0], k=1)[0]

    for k in range(N):
        p_mav = x_guess[:3, k + 1]
        p_obs = obs.p_hat + obs.v_hat * mpc.dt * k
        diff = p_mav - p_obs
        d = float(np.linalg.norm(diff))

        if d < 1e-6:
            mpc._lg_grad[k].value = np.zeros(3)
            continue

        exp_arg = math.exp(mpc._lg_lam * (d - mpc._lg_ro))
        J_deriv = -mpc._lg_lam * mpc._lg_Qo * exp_arg / ((1.0 + exp_arg) ** 2)

        n_o = diff / d
        grad = J_deriv * n_o
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > 5.0:
            grad = grad * (5.0 / grad_norm)
        mpc._lg_grad[k].value = grad


# ---------------------------------------------------------------------------
# FOV constraint computation (Eq 17-18)
# ---------------------------------------------------------------------------
def compute_fov_params(
    mpc: CCMPC,
    x_guess: npt.NDArray[np.float64],
    obstacle_manager: ObstacleManager | None = None,
) -> None:
    """Compute FOV constraints for the current guess trajectory.

    For each horizon step k, transforms the camera FOV half-spaces
    into world-frame linear constraints on p_{k+1}.
    """
    if not mpc._fov_enabled or not mpc._fov_a:
        return

    N = mpc.control_horizon
    for k in range(N):
        yaw_k = x_guess[8, k]
        ct, st = math.cos(yaw_k), math.sin(yaw_k)
        R_yaw = np.array([
            [ct, -st, 0.0],
            [st, ct, 0.0],
            [0.0, 0.0, 1.0],
        ])
        p_k = x_guess[:3, k]
        p_k1 = x_guess[:3, k + 1]
        p_body = R_yaw.T @ (p_k1 - p_k)

        # Find worst-violated half-space
        worst_v = -1e9
        worst_j = 0
        for j in range(5):
            v_j = float(mpc._fov_n_body[j] @ p_body - mpc._fov_m[j])
            if v_j > worst_v:
                worst_v = v_j
                worst_j = j

        n_worst = mpc._fov_n_body[worst_j]
        m_worst = mpc._fov_m[worst_j]
        a_world = R_yaw @ n_worst
        b_val = float(m_worst) + float(a_world @ p_k)

        mpc._fov_a[k].value = a_world
        mpc._fov_b[k].value = b_val
