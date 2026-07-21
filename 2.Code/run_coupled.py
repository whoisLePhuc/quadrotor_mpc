"""
run_coupled.py
==============
Closed-loop simulation: do-mpc/CasADi MPC (unchanged "brain", quad_mpc_core.py)
computes u0 every MPC tick; MuJoCo (mujoco_plant.py) is the "true" plant that
actually integrates that control -- a genuinely different rigid-body integrator
than the one the MPC uses internally to predict, so this is a real model-mismatch
robustness test, plus we get real contact-based collision detection.
"""
import numpy as np
from quad_mpc_core import (
    build_model, build_controller, make_mpc_tvp_fun, quat_from_euler, quat_to_euler,
    DRONE_RADIUS, obstacle_pos_at,
)
from mujoco_plant import MuJoCoPlant


def run_coupled_simulation(x0_vals, goal_pos, goal_euler, bounds, obstacles, margin,
                            sim_seconds=12.0, mpc_dt=0.05, n_horizon=20, max_iter=60,
                            mj_dt=0.002, progress_cb=None, capture_frames=False, render_every=1,
                            cached=None):
    """
    `cached`: optional (model, mpc, dyn_idx, goal_state) tuple from
    quad_mpc_core.build_cached_mpc(...) to reuse an already-built/compiled MPC
    controller across multiple runs (different start/goal) instead of rebuilding
    it every time - see build_cached_mpc's docstring. Profiling showed MuJoCo
    stepping itself costs <1ms/tick, so essentially all per-tick cost is the MPC
    solve; caching (and optionally JIT) is where the real speedup comes from, not
    anything MuJoCo-specific.
    """
    if cached is not None:
        model, mpc, dyn_idx, goal_state = cached
        goal_state['pos'] = goal_pos
        goal_state['euler'] = goal_euler
        mpc.reset_history()
    else:
        model, dyn_idx = build_model(mpc_dt, obstacles)
        mpc = build_controller(model, obstacles, dyn_idx, bounds, margin, n_horizon, mpc_dt, max_iter=max_iter)
        goal_state = {'pos': goal_pos, 'euler': goal_euler}
        mpc.set_tvp_fun(make_mpc_tvp_fun(mpc.get_tvp_template(), goal_state, obstacles, dyn_idx, n_horizon, mpc_dt))
        mpc.setup()

    plant = MuJoCoPlant(x0_vals, goal_pos, obstacles, mj_dt=mj_dt)

    renderer, render_err = (None, "not requested")
    if capture_frames:
        renderer, render_err = plant.try_create_renderer()

    q0 = quat_from_euler(x0_vals.get('roll', 0), x0_vals.get('pitch', 0), x0_vals.get('yaw', 0))
    x0 = np.array([x0_vals['x'], x0_vals['y'], x0_vals['z'], 0, 0, 0, *q0, 0, 0, 0]).reshape(-1, 1)
    mpc.x0 = x0
    mpc.set_initial_guess()

    n_substeps = max(1, round(mpc_dt/mj_dt))
    n_steps = int(sim_seconds/mpc_dt)

    ts, poss, eulers, us, clearances, frames = [], [], [], [], [], []
    collided = False
    x_curr = x0
    t = 0.0
    for k in range(n_steps):
        u0 = mpc.make_step(x_curr)                      # MPC "brain" (its own model)
        plant.apply_control_and_step(u0.flatten(), n_substeps, t)  # MuJoCo "true" plant
        x_curr = plant.get_state_13()

        ts.append(t)
        poss.append(x_curr[0:3, 0].copy())
        eulers.append(quat_to_euler(x_curr[6:10, 0]))
        us.append(u0.flatten().copy())

        min_clear = np.inf
        for obs in obstacles:
            cx, cy, cz = obstacle_pos_at(obs, t)
            d = np.sqrt((x_curr[0,0]-cx)**2 + (x_curr[1,0]-cy)**2 + (x_curr[2,0]-cz)**2) - obs['radius'] - DRONE_RADIUS
            min_clear = min(min_clear, d)
        clearances.append(min_clear)
        if plant.check_collision():
            collided = True

        if renderer is not None and k % render_every == 0:
            frames.append(plant.render_frame(renderer, camera='fixed_view'))

        t += mpc_dt
        if progress_cb is not None:
            progress_cb((k+1)/n_steps)

    return {
        't': np.array(ts), 'pos': np.array(poss), 'euler': np.array(eulers),
        'u': np.array(us), 'clearance': np.array(clearances),
        'dt': mpc_dt, 'collided': collided,
        'frames': frames, 'render_error': render_err,
    }


if __name__ == '__main__':
    import time
    t0 = time.time()
    result = run_coupled_simulation(
        x0_vals={'x': 0, 'y': 0, 'z': 1, 'roll': 0, 'pitch': 0, 'yaw': 0},
        goal_pos={'x': 3, 'y': 2, 'z': 2.5},
        goal_euler={'roll': 0, 'pitch': 0, 'yaw': np.deg2rad(45)},
        bounds={'thrust': 6.0, 'torque_rp': 0.03, 'torque_yaw': 0.02},
        obstacles=[{'type': 'static', 'x': 1.5, 'y': 1.0, 'z': 2.0, 'radius': 0.5}],
        margin=0.3, sim_seconds=8.0,
    )
    print(f"perf: {(time.time()-t0)/len(result['t'])*1000:.2f} ms/tick (MPC+MuJoCo combined)")
    print("final pos:", result['pos'][-1], " goal: [3,2,2.5]")
    print("min clearance:", result['clearance'].min())
    print("collided (real contact):", result['collided'])
    print("any NaN:", np.isnan(result['pos']).any())
