"""
quad_mpc_core.py
=================
Core quadrotor dynamics + nonlinear MPC (CasADi / do-mpc) for the Streamlit app.

Model: full nonlinear rigid-body quadrotor with quaternion attitude (no small-angle
assumption, no Euler gimbal singularity in the dynamics themselves).

    state x = [x,y,z, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz]   (13-dim)
    input u = [thrust_dev, taux, tauy, tauz]              (4-dim)

Controller: real nonlinear MPC via do-mpc/CasADi/IPOPT (interior-point), solved as a
receding-horizon closed loop together with a do-mpc Simulator representing the "true"
plant. This replaces the hand-rolled gradient/FISTA solver from the JS/browser version
with a properly convergent, well-tested optimization solver - no manual tuning of
step-size preconditioning, horizon-length workarounds, etc. needed.

Obstacle avoidance is expressed as a genuine nonlinear (soft) inequality constraint
-- "distance to obstacle center >= radius + margin + drone_radius" -- rather than an
ad-hoc potential-field penalty, so it is enforced by the solver directly.
"""

import numpy as np
from casadi import vertcat, sqrt as ca_sqrt
import do_mpc

from obstacle_motion import obstacle_position
from vehicle import DEFAULT_QUADROTOR

# ---------------------------------------------------------------------------
# physical constants (same values used throughout this project's iterations)
# ---------------------------------------------------------------------------
G = 9.81
M = DEFAULT_QUADROTOR.mass_kg
IXX, IYY, IZZ = DEFAULT_QUADROTOR.inertia_kg_m2
D_LIN = DEFAULT_QUADROTOR.linear_damping_per_s
D_ANG = DEFAULT_QUADROTOR.angular_damping_nms
D_YAW = DEFAULT_QUADROTOR.yaw_damping_nms
DRONE_RADIUS = DEFAULT_QUADROTOR.collision_radius_m

STATE_NAMES = ["x", "y", "z", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "wx", "wy", "wz"]
INPUT_NAMES = ["Tdev", "taux", "tauy", "tauz"]


def quat_from_euler(roll, pitch, yaw):
    """ZYX/aerospace convention Euler -> quaternion [qw,qx,qy,qz]."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quat_to_euler(q):
    """Inverse of quat_from_euler. q = [qw,qx,qy,qz]."""
    qw, qx, qy, qz = q
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return roll, pitch, yaw


def quat_rotate(q, v):
    """Rotate 3-vector v from body frame to world frame by quaternion q (numpy)."""
    qw, qx, qy, qz = q
    qv = np.array([qx, qy, qz])
    t = 2 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def obstacle_pos_at(obs, t):
    """World position of an obstacle at absolute simulation time t."""
    return tuple(obstacle_position(obs, t))


# ---------------------------------------------------------------------------
# symbolic dynamics (shared by the discrete-time RK4 model builder)
# ---------------------------------------------------------------------------
def _dynamics_ca(xvec, uvec):
    x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz = [xvec[i] for i in range(13)]
    Tdev, taux, tauy, tauz = [uvec[i] for i in range(4)]

    az_body = G + Tdev / M
    c1x, c1y, c1z = qy * az_body, -qx * az_body, 0
    c2x = qy * c1z - qz * c1y
    c2y = qz * c1x - qx * c1z
    c2z = qx * c1y - qy * c1x
    ax = 2 * qw * c1x + 2 * c2x
    ay = 2 * qw * c1y + 2 * c2y
    az = az_body + 2 * qw * c1z + 2 * c2z

    vxdot = ax - D_LIN * vx
    vydot = ay - D_LIN * vy
    vzdot = az - G - D_LIN * vz

    qdw = -0.5 * (qx * wx + qy * wy + qz * wz)
    qdx = 0.5 * (qw * wx + qy * wz - qz * wy)
    qdy = 0.5 * (qw * wy + qz * wx - qx * wz)
    qdz = 0.5 * (qw * wz + qx * wy - qy * wx)

    wxdot = (taux - (IZZ - IYY) * wy * wz - D_ANG * wx) / IXX
    wydot = (tauy - (IXX - IZZ) * wz * wx - D_ANG * wy) / IYY
    wzdot = (tauz - (IYY - IXX) * wx * wy - D_YAW * wz) / IZZ

    return vertcat(vx, vy, vz, vxdot, vydot, vzdot, qdw, qdx, qdy, qdz, wxdot, wydot, wzdot)


def build_model(dt, obstacles):
    """
    Discrete-time (explicit RK4) do-mpc model. Using an explicit discretization
    instead of do-mpc's default implicit-collocation continuous-time transcription
    is ~3x faster to solve empirically, at negligible accuracy cost for this dt.

    `obstacles`: list of obstacle dicts (see obstacle_pos_at). Dynamic obstacles get
    their own time-varying-parameter (tvp) entries so the MPC can predict their
    future position analytically instead of assuming they stay put.
    """
    model = do_mpc.model.Model("discrete")

    xvars = [model.set_variable("_x", n) for n in STATE_NAMES]
    uvars = [model.set_variable("_u", n) for n in INPUT_NAMES]

    model.set_variable("_tvp", "xg")
    model.set_variable("_tvp", "yg")
    model.set_variable("_tvp", "zg")
    model.set_variable("_tvp", "qwg")
    model.set_variable("_tvp", "qxg")
    model.set_variable("_tvp", "qyg")
    model.set_variable("_tvp", "qzg")

    dyn_idx = [i for i, o in enumerate(obstacles) if o["type"] == "dynamic"]
    for i in dyn_idx:
        model.set_variable("_tvp", f"obsx_{i}")
        model.set_variable("_tvp", f"obsy_{i}")
        model.set_variable("_tvp", f"obsz_{i}")

    xvec = vertcat(*xvars)
    uvec = vertcat(*uvars)
    k1 = _dynamics_ca(xvec, uvec)
    k2 = _dynamics_ca(xvec + dt / 2 * k1, uvec)
    k3 = _dynamics_ca(xvec + dt / 2 * k2, uvec)
    k4 = _dynamics_ca(xvec + dt * k3, uvec)
    x_next = xvec + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    qnorm = ca_sqrt(x_next[6] ** 2 + x_next[7] ** 2 + x_next[8] ** 2 + x_next[9] ** 2)
    x_next[6] = x_next[6] / qnorm
    x_next[7] = x_next[7] / qnorm
    x_next[8] = x_next[8] / qnorm
    x_next[9] = x_next[9] / qnorm

    for i, n in enumerate(STATE_NAMES):
        model.set_rhs(n, x_next[i])

    model.setup()
    return model, dyn_idx


def build_controller(
    model,
    obstacles,
    dyn_idx,
    bounds,
    margin,
    n_horizon,
    dt,
    max_iter=60,
    use_jit=False,
    w_pos=(2, 2, 14),
    w_vel=(1, 1, 1.5),
    w_att=80.0,
    w_rate=(2, 2, 0.5),
    r_ctrl=(0.05, 0.08, 0.08, 0.05),
    penalty=1e4,
):
    mpc = do_mpc.controller.MPC(model)
    nlpsol_opts = {
        "ipopt.max_iter": max_iter,
        "ipopt.tol": 1e-4,
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.sb": "yes",
    }
    if use_jit:
        # ~30-40% faster per-tick solve, at the cost of a one-time C-compile step
        # (~10-40s). Only worth it when the resulting controller is CACHED and
        # reused across multiple runs (see build_cached_mpc) - a single throwaway
        # run would lose more time to compilation than it saves.
        nlpsol_opts.update(
            {"jit": True, "compiler": "shell", "jit_options": {"flags": ["-O1"], "verbose": False}}
        )
    mpc.set_param(
        n_horizon=n_horizon,
        t_step=dt,
        n_robust=0,
        store_full_solution=True,
        nlpsol_opts=nlpsol_opts,
    )
    mpc.settings.supress_ipopt_output()

    x, y, z = model.x["x"], model.x["y"], model.x["z"]
    vx, vy, vz = model.x["vx"], model.x["vy"], model.x["vz"]
    qw, qx, qy, qz = model.x["qw"], model.x["qx"], model.x["qy"], model.x["qz"]
    wx, wy, wz = model.x["wx"], model.x["wy"], model.x["wz"]
    xg, yg, zg = model.tvp["xg"], model.tvp["yg"], model.tvp["zg"]
    qwg, qxg, qyg, qzg = model.tvp["qwg"], model.tvp["qxg"], model.tvp["qyg"], model.tvp["qzg"]

    pos_cost = w_pos[0] * (x - xg) ** 2 + w_pos[1] * (y - yg) ** 2 + w_pos[2] * (z - zg) ** 2
    vel_cost = w_vel[0] * vx**2 + w_vel[1] * vy**2 + w_vel[2] * vz**2
    # Quaternion tracking cost invariant to the q <-> -q double-cover ambiguity -
    # no manual "hemisphere alignment" bookkeeping needed (unlike the browser version).
    dot = qw * qwg + qx * qxg + qy * qyg + qz * qzg
    att_cost = w_att * (1 - dot**2)
    rate_cost = w_rate[0] * wx**2 + w_rate[1] * wy**2 + w_rate[2] * wz**2

    lterm = pos_cost + vel_cost + att_cost + rate_cost
    mterm = 6 * lterm
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(Tdev=r_ctrl[0], taux=r_ctrl[1], tauy=r_ctrl[2], tauz=r_ctrl[3])

    thrust_lim_down = min(bounds["thrust"], M * G)
    thrust_lim_up = min(bounds["thrust"], DEFAULT_QUADROTOR.max_upward_thrust_deviation_n)
    mpc.bounds["lower", "_u", "Tdev"] = -thrust_lim_down
    mpc.bounds["upper", "_u", "Tdev"] = thrust_lim_up
    torque_rp_lim = min(bounds["torque_rp"], DEFAULT_QUADROTOR.max_roll_pitch_torque_nm)
    torque_yaw_lim = min(bounds["torque_yaw"], DEFAULT_QUADROTOR.max_yaw_torque_nm)
    mpc.bounds["lower", "_u", "taux"] = -torque_rp_lim
    mpc.bounds["upper", "_u", "taux"] = torque_rp_lim
    mpc.bounds["lower", "_u", "tauy"] = -torque_rp_lim
    mpc.bounds["upper", "_u", "tauy"] = torque_rp_lim
    mpc.bounds["lower", "_u", "tauz"] = -torque_yaw_lim
    mpc.bounds["upper", "_u", "tauz"] = torque_yaw_lim

    for i, obs in enumerate(obstacles):
        if obs["type"] == "static":
            cx, cy, cz = obs["x"], obs["y"], obs["z"]
        else:
            cx, cy, cz = model.tvp[f"obsx_{i}"], model.tvp[f"obsy_{i}"], model.tvp[f"obsz_{i}"]
        safe_dist = obs["radius"] + margin + DRONE_RADIUS
        dist_expr = ca_sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 + 1e-6)
        mpc.set_nl_cons(
            f"obstacle_{i}",
            safe_dist - dist_expr,
            ub=0,
            soft_constraint=True,
            penalty_term_cons=penalty,
        )

    return mpc


def _fill_tvp(template, k_or_none, goal_pos, qg, obstacles, dyn_idx, t_abs):
    def setv(key, val):
        if k_or_none is None:
            template[key] = val
        else:
            template["_tvp", k_or_none, key] = val

    setv("xg", goal_pos["x"])
    setv("yg", goal_pos["y"])
    setv("zg", goal_pos["z"])
    setv("qwg", qg[0])
    setv("qxg", qg[1])
    setv("qyg", qg[2])
    setv("qzg", qg[3])
    for i in dyn_idx:
        px, py, pz = obstacle_pos_at(obstacles[i], t_abs)
        setv(f"obsx_{i}", px)
        setv(f"obsy_{i}", py)
        setv(f"obsz_{i}", pz)


def make_mpc_tvp_fun(template, goal_state, obstacles, dyn_idx, n_horizon, dt):
    """`goal_state` is a MUTABLE dict {'pos':..., 'euler':...} read fresh on every
    call (rather than a value baked into the closure at build time) - this is what
    lets the same compiled MPC controller be reused for a new start/goal without
    rebuilding/recompiling the NLP (see build_cached_mpc / CONTROLLER CACHING note
    below)."""

    def tvp_fun(t_now):
        qg = quat_from_euler(
            goal_state["euler"]["roll"], goal_state["euler"]["pitch"], goal_state["euler"]["yaw"]
        )
        for k in range(n_horizon + 1):
            _fill_tvp(template, k, goal_state["pos"], qg, obstacles, dyn_idx, t_now + k * dt)
        return template

    return tvp_fun


def make_sim_tvp_fun(template, goal_state, obstacles, dyn_idx):
    def tvp_fun(t_now):
        qg = quat_from_euler(
            goal_state["euler"]["roll"], goal_state["euler"]["pitch"], goal_state["euler"]["yaw"]
        )
        _fill_tvp(template, None, goal_state["pos"], qg, obstacles, dyn_idx, t_now)
        return template

    return tvp_fun


def build_cached_mpc(bounds, obstacles, margin, n_horizon=20, dt=0.05, max_iter=60, use_jit=False):
    """
    Builds+compiles the MPC controller ONCE for a given (bounds, obstacles, margin)
    configuration. Returns (model, mpc, dyn_idx, goal_state) where `goal_state` is a
    mutable dict the caller updates before each run (see run_simulation's
    `cached=...` argument) to reuse the SAME compiled controller across many
    start/goal combinations without paying the NLP (re)build/compile cost again.

    CONTROLLER CACHING - WHY THIS MATTERS: profiling shows MuJoCo integration itself
    is negligible (<1ms/tick) - essentially ALL per-tick cost is the IPOPT solve,
    and rebuilding+setup()-ing a fresh do-mpc controller on every single "Run" click
    (as earlier versions of this app did) re-pays CasADi's problem-construction cost
    every time even though most of a user's session only changes start/goal (which
    do-mpc already supports changing via tvp, with NO rebuild needed). Cache this
    object (e.g. with `st.cache_resource` in Streamlit, keyed on bounds/obstacles/
    margin) and reuse it across runs. Enabling `use_jit=True` compiles the NLP to
    C (one-time cost of ~10-40s) for a further ~30-40% per-tick speedup - only
    worthwhile when combined with this caching, since a single un-cached run would
    lose more time to compilation than it gains back.
    """
    model, dyn_idx = build_model(dt, obstacles)
    mpc = build_controller(
        model, obstacles, dyn_idx, bounds, margin, n_horizon, dt, max_iter=max_iter, use_jit=use_jit
    )
    goal_state = {"pos": {"x": 0, "y": 0, "z": 1}, "euler": {"roll": 0, "pitch": 0, "yaw": 0}}
    mpc.set_tvp_fun(
        make_mpc_tvp_fun(mpc.get_tvp_template(), goal_state, obstacles, dyn_idx, n_horizon, dt)
    )
    mpc.setup()
    return model, mpc, dyn_idx, goal_state


def run_simulation(
    x0_vals,
    goal_pos,
    goal_euler,
    bounds,
    obstacles,
    margin,
    sim_seconds=12.0,
    dt=0.05,
    n_horizon=20,
    max_iter=60,
    progress_cb=None,
    cached=None,
):
    """
    Runs the full closed-loop MPC simulation and returns a dict of time-series
    arrays ready for plotting: t, pos (N,3), euler (N,3) [roll,pitch,yaw radians],
    u (N,4), and per-step min obstacle clearance.

    `cached`: optional (model, mpc, dyn_idx, goal_state) tuple from
    build_cached_mpc(...) to reuse an already-built/compiled controller instead of
    constructing a fresh one (see build_cached_mpc's docstring for why this matters
    for performance).
    """
    if cached is not None:
        model, mpc, dyn_idx, goal_state = cached
        goal_state["pos"] = goal_pos
        goal_state["euler"] = goal_euler
        mpc.reset_history()
    else:
        model, dyn_idx = build_model(dt, obstacles)
        mpc = build_controller(
            model, obstacles, dyn_idx, bounds, margin, n_horizon, dt, max_iter=max_iter
        )
        goal_state = {"pos": goal_pos, "euler": goal_euler}
        mpc.set_tvp_fun(
            make_mpc_tvp_fun(mpc.get_tvp_template(), goal_state, obstacles, dyn_idx, n_horizon, dt)
        )
        mpc.setup()

    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=dt)
    simulator.set_tvp_fun(
        make_sim_tvp_fun(simulator.get_tvp_template(), goal_state, obstacles, dyn_idx)
    )
    simulator.setup()

    q0 = quat_from_euler(x0_vals.get("roll", 0), x0_vals.get("pitch", 0), x0_vals.get("yaw", 0))
    x0 = np.array([x0_vals["x"], x0_vals["y"], x0_vals["z"], 0, 0, 0, *q0, 0, 0, 0]).reshape(-1, 1)
    mpc.x0 = x0
    simulator.x0 = x0
    mpc.set_initial_guess()

    n_steps = int(sim_seconds / dt)
    ts, poss, eulers, us, clearances = [], [], [], [], []
    x_curr = x0
    t = 0.0
    for k in range(n_steps):
        u0 = mpc.make_step(x_curr)
        x_curr = simulator.make_step(u0)

        ts.append(t)
        poss.append(x_curr[0:3, 0].copy())
        eulers.append(quat_to_euler(x_curr[6:10, 0]))
        us.append(u0.flatten().copy())

        min_clear = np.inf
        for obs in obstacles:
            cx, cy, cz = obstacle_pos_at(obs, t)
            d = (
                np.sqrt(
                    (x_curr[0, 0] - cx) ** 2 + (x_curr[1, 0] - cy) ** 2 + (x_curr[2, 0] - cz) ** 2
                )
                - obs["radius"]
                - DRONE_RADIUS
            )
            min_clear = min(min_clear, d)
        clearances.append(min_clear)

        t += dt
        if progress_cb is not None:
            progress_cb((k + 1) / n_steps)

    return {
        "t": np.array(ts),
        "pos": np.array(poss),
        "euler": np.array(eulers),
        "u": np.array(us),
        "clearance": np.array(clearances),
        "dt": dt,
    }
