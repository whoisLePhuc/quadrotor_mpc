"""
MuJoCo physics simulation for CC-MPC quadrotor obstacle avoidance.

Uses MuJoCo physics engine (not ODE integration). CC-MPC outputs attitude
commands [phi_c, theta_c, vz_c, psi_dot_c], which are converted to individual
rotor thrusts via QuadrotorMixer and applied to MuJoCo actuators.

Architecture (threaded, matching mpc_python pattern):
  - Main thread: MuJoCo physics step + render
  - MPC thread: CC-MPC solve loop
  - SharedData for thread-safe state/control exchange

Paper: Lin, Zhu, Alonso-Mora — ICRA 2020
"""

from __future__ import annotations

import argparse
import math
import pathlib
import signal
import threading
import time

import numpy as np
import numpy.typing as npt

import mujoco
import mujoco.viewer
import yaml

from ccmpc.ccmpc import CCMPC
from ccmpc.obstacle import ObstacleManager
from ccmpc.uncertainty import UncertaintyPropagator
from ccmpc.utils import detect_obstacle_in_fov, quat_to_euler
from ccmpc.mixer import QuadrotorMixer


# ---------------------------------------------------------------------------
# Thread-safe data exchange
# ---------------------------------------------------------------------------
class SharedData:
    """Data shared between the physics thread and MPC thread."""

    def __init__(self) -> None:
        self.lock: threading.Lock = threading.Lock()
        self.state: npt.NDArray[np.float64] = np.zeros(9)
        self.goal_reached: bool = False
        self.is_active: bool = True
        self.control: npt.NDArray[np.float64] = np.zeros(4)
        self.last_control: npt.NDArray[np.float64] = np.zeros(4)
        self.x_mpc_world: npt.NDArray[np.float64] | None = None
        self.mpc_elapsed: float = 0.0
        self.position_cov: npt.NDArray[np.float64] = np.eye(3) * 0.01


# ---------------------------------------------------------------------------
# MPC controller thread
# ---------------------------------------------------------------------------
def controller_loop(
    mpc: CCMPC,
    shared: SharedData,
    goal: npt.NDArray[np.float64],
    goal_threshold: float,
    obstacle_manager: ObstacleManager,
    no_fov_filter: bool = False,
    solver: str | None = None,
) -> None:
    """Read state from shared, solve CC-MPC, write control back.

    Includes delay compensation: predicts the state forward by the
    last solve time before computing the next command. This ensures
    the control is computed for the time it will actually be applied.
    """
    while True:
        with shared.lock:
            if not shared.is_active or shared.goal_reached:
                break
            current_state = shared.state.copy()
            last_u = shared.last_control.copy()

        dist = np.linalg.norm(current_state[:3] - goal)
        if dist < goal_threshold:
            with shared.lock:
                shared.goal_reached = True
            break

        # Delay compensation: predict forward by last solve time
        # Using forward Euler on the continuous dynamics to match MuJoCo better
        elapsed = shared.mpc_elapsed
        if elapsed > 0.0:
            xdot = mpc.dynamics.continuous(current_state, last_u)
            pred_state = current_state + xdot * min(elapsed, 0.1)
        else:
            pred_state = current_state.copy()

        # FOV filter: only pass visible obstacles to MPC
        if no_fov_filter:
            obs_for_mpc = obstacle_manager
        else:
            visible = detect_obstacle_in_fov(
                obstacle_manager.obstacles,
                pred_state[:3], pred_state[8],
                hfov_deg=90.0, vfov_deg=70.0, max_range=5.0,
            )
            obs_for_mpc = ObstacleManager([o for _, o in visible]) if visible else None

        noisy_state = UncertaintyPropagator.add_measurement_noise(pred_state)

        obstacle_manager.update(mpc.dt)
        t_start = time.perf_counter()
        x_traj, u_traj = mpc.solve(noisy_state, goal, solver=solver, obstacle_manager=obs_for_mpc)
        solve_time = time.perf_counter() - t_start

        with shared.lock:
            if u_traj is not None and u_traj.shape[1] > 0:
                shared.control[:] = u_traj[:, 0]
            shared.mpc_elapsed = solve_time
            shared.x_mpc_world = x_traj.copy() if x_traj is not None else None
            if hasattr(mpc.uncertainty, 'position_covariance'):
                shared.position_cov = mpc.uncertainty.position_covariance(
                    mpc._previous_covariance
                ) if mpc._previous_covariance is not None else np.eye(3) * 0.01

        time.sleep(0.001)


# ---------------------------------------------------------------------------
# MuJoCo state helpers
# ---------------------------------------------------------------------------
def body_id(model: mujoco.MjModel, name: str) -> int:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if i == -1:
        raise ValueError(f"Body '{name}' not found")
    return i


def read_mujoco_state(
    model: mujoco.MjModel, data: mujoco.MjData, bid: int
) -> npt.NDArray[np.float64]:
    """Read 9D state [x,y,z, vx,vy,vz, roll,pitch,yaw] from MuJoCo."""
    pos = data.xpos[bid].copy()
    vel = data.qvel[:3].copy()
    quat = data.xquat[bid].copy()  # [w,x,y,z]
    roll, pitch, yaw = quat_to_euler(quat)
    return np.array([pos[0], pos[1], pos[2],
                     vel[0], vel[1], vel[2],
                     roll, pitch, yaw])


# ---------------------------------------------------------------------------
# Draw helpers
# ---------------------------------------------------------------------------
def draw_obstacle(viewer: mujoco.viewer.Handle, obstacles: list) -> None:
    for obs in obstacles:
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        r = float(np.max(obs.axes))
        mujoco.mjv_initGeom(
            g, type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([r, 0.0, 0.0], dtype=np.float64),
            pos=np.array(obs.p_hat, dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=np.array([1.0, 0.2, 0.2, 0.45], dtype=np.float32),
        )
        viewer.user_scn.ngeom += 1


def draw_goal(viewer: mujoco.viewer.Handle, goal: npt.NDArray[np.float64]) -> None:
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        g, type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([0.08, 0.0, 0.0], dtype=np.float64),
        pos=np.array(goal, dtype=np.float64),
        mat=np.eye(3).ravel(),
        rgba=np.array([1.0, 0.84, 0.0, 0.7], dtype=np.float32),
    )
    viewer.user_scn.ngeom += 1


def draw_trail(
    viewer: mujoco.viewer.Handle,
    x_hist, y_hist, z_hist, downsample=15,
) -> None:
    if len(x_hist) < 2:
        return
    for i in range(0, len(x_hist) - 1, downsample):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        alpha = (i + 1) / len(x_hist) * 0.6
        p1 = np.array([x_hist[i], y_hist[i], z_hist[i]], dtype=np.float64)
        p2 = np.array([x_hist[i+1], y_hist[i+1], z_hist[i+1]], dtype=np.float64)
        mujoco.mjv_initGeom(
            g, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.array([0.02, 0.0, 0.0], dtype=np.float64),
            pos=np.zeros(3, dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=np.array([0.2, 0.4, 1.0, alpha], dtype=np.float32),
        )
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.02, p1, p2)
        viewer.user_scn.ngeom += 1


def draw_fov(
    viewer: mujoco.viewer.Handle,
    x: float, y: float, z: float,
    yaw: float,
    hfov_deg: float = 87.0,
    vfov_deg: float = 58.0,
    max_range: float = 5.0,
) -> None:
    """Draw camera FOV frustum as wireframe lines."""
    hfov = math.radians(hfov_deg / 2.0)
    vfov = math.radians(vfov_deg / 2.0)
    ct, st = math.cos(yaw), math.sin(yaw)

    # Frustum corners in body frame (x forward, y left, z up)
    d = max_range
    corners_body = np.array([
        [d, -d * math.tan(hfov), -d * math.tan(vfov)],  # front-left-bottom
        [d, -d * math.tan(hfov),  d * math.tan(vfov)],  # front-left-top
        [d,  d * math.tan(hfov),  d * math.tan(vfov)],  # front-right-top
        [d,  d * math.tan(hfov), -d * math.tan(vfov)],  # front-right-bottom
    ])
    # Rotate to world frame (yaw only)
    R = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
    corners = corners_body @ R.T + np.array([x, y, z])

    origin = np.array([x, y, z], dtype=np.float64)
    rgba = np.array([0.8, 0.8, 0.0, 0.2], dtype=np.float32)

    # 4 lines from origin to front corners
    for i in range(4):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            g, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.array([0.008, 0.0, 0.0], dtype=np.float64),
            pos=np.zeros(3, dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=rgba,
        )
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                             0.008, origin, corners[i].astype(np.float64))
        viewer.user_scn.ngeom += 1

    # 4 edges of the front face
    for i in range(4):
        j = (i + 1) % 4
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            g, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.array([0.008, 0.0, 0.0], dtype=np.float64),
            pos=np.zeros(3, dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=rgba,
        )
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                             0.008, corners[i].astype(np.float64),
                             corners[j].astype(np.float64))
        viewer.user_scn.ngeom += 1


def draw_uncertainty(
    viewer: mujoco.viewer.Handle,
    pos: npt.NDArray[np.float64],
    cov: npt.NDArray[np.float64],
    scale: float = 3.0,
) -> None:
    """Draw position uncertainty as a 3σ ellipsoid."""
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    try:
        w, v = np.linalg.eigh(cov)
        w = np.maximum(w, 1e-10)
        axes = np.sqrt(w) * scale
        # Semi-axes as size
        size = np.array([axes[0], axes[1], axes[2]], dtype=np.float64)
        # Rotation matrix from eigenvectors
        mat = v.ravel().astype(np.float64)

        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            g, type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            size=size,
            pos=np.array(pos, dtype=np.float64),
            mat=mat,
            rgba=np.array([1.0, 0.6, 0.0, 0.15], dtype=np.float32),
        )
        viewer.user_scn.ngeom += 1
    except np.linalg.LinAlgError:
        pass


def draw_mpc_preview(
    viewer: mujoco.viewer.Handle, traj: npt.NDArray[np.float64]
) -> None:
    for i in range(traj.shape[1]):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            g, type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.04, 0.0, 0.0], dtype=np.float64),
            pos=np.array([traj[0, i], traj[1, i], traj[2, i]], dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=np.array([0.0, 1.0, 0.2, 0.5], dtype=np.float32),
        )
        viewer.user_scn.ngeom += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quadrotor CC-MPC MuJoCo Simulation"
    )
    parser.add_argument("--scenario", default="simulation",
                        help="Scenario name (config/simulation_<name>.yaml)")
    parser.add_argument("--no-fov", action="store_true",
                        help="Disable FOV constraints")
    parser.add_argument("--no-fov-filter", action="store_true",
                        help="Disable FOV obstacle filter")
    parser.add_argument("--solver", default=None,
                        help="CVXPY solver (CLARABEL, OSQP, SCS)")
    parser.add_argument("--horizon", type=float, default=None,
                        help="Horizon time in seconds (default: from config)")
    args = parser.parse_args()

    pkg_root = pathlib.Path(__file__).parent

    # Load scenario config
    scenario_path = pkg_root / "config" / f"{args.scenario}.yaml"
    if not scenario_path.exists():
        scenario_path = pkg_root / "config" / f"simulation_{args.scenario}.yaml"
    if not scenario_path.exists():
        scenario_path = pkg_root / "config" / "simulation.yaml"
    sim_config = yaml.safe_load(scenario_path.read_text())
    print(f"Scenario: {scenario_path.stem}")
    start = sim_config["start"]
    goal = np.array(sim_config["goal"], dtype=np.float64)
    goal_threshold = sim_config.get("goal_threshold", 0.4)
    sim_dt = sim_config.get("sim_timestep", 0.02)

    # Load MuJoCo model (with actuators)
    model_path = pkg_root / "models" / "quadrotor.xml"
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    quad_id = body_id(m, "quadrotor")

    # Initial state
    init_state = np.array(start, dtype=np.float64)

    # CC-MPC
    mpc_kwargs = {}
    if args.horizon:
        mpc_kwargs["horizon_time"] = args.horizon
    mpc = CCMPC(pkg_root / "config" / "mpc.yaml", **mpc_kwargs)

    # Override FOV if specified via CLI
    if args.no_fov:
        mpc._fov_enabled = False

    # Mixer that maps CC-MPC commands to MuJoCo rotor thrusts
    actual_mass = sum(m.body_mass)
    mixer = QuadrotorMixer(mass=actual_mass)
    print(f"  Mass: {actual_mass:.2f}kg, hover/rotor: {mixer.hover_thrust:.1f}N")

    # Obstacles
    obstacle_manager = ObstacleManager.from_config(str(scenario_path))

    # MPC thread
    shared = SharedData()
    with shared.lock:
        shared.state[:] = init_state

    mpc_thread = threading.Thread(
        target=controller_loop,
        args=(mpc, shared, goal, goal_threshold, obstacle_manager),
        kwargs={"no_fov_filter": args.no_fov_filter, "solver": args.solver},
        daemon=True,
    )

    shutdown_flag = threading.Event()

    def handle_shutdown(signum, frame):
        shutdown_flag.set()

    signal.signal(signal.SIGINT, handle_shutdown)

    # Set initial MuJoCo pose
    d.qpos[:3] = init_state[:3]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # identity quaternion
    mujoco.mj_forward(m, d)

    # Launch viewer
    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        viewer.cam.distance = 7.0
        viewer.cam.azimuth = 60
        viewer.cam.elevation = -30

        fps = 60.0
        render_dt = 1.0 / fps
        sub_steps = max(1, round(sim_dt / m.opt.timestep))

        x_hist = [init_state[0]]
        y_hist = [init_state[1]]
        z_hist = [init_state[2]]
        iter_count = 0
        cte_hist: list[float] = []
        hdg_err_hist: list[float] = []

        sim_start_time = time.perf_counter()
        mpc_thread.start()

        try:
            while viewer.is_running() and not shutdown_flag.is_set():

                dist_to_goal = np.linalg.norm(
                    read_mujoco_state(m, d, quad_id)[:3] - goal
                )
                if shared.goal_reached or dist_to_goal < goal_threshold:
                    cte_rmse = float(np.sqrt(np.mean(np.square(cte_hist)))) if cte_hist else 0.0
                    hdg_rmse = float(np.degrees(np.sqrt(np.mean(np.square(hdg_err_hist))))) if hdg_err_hist else 0.0
                    print(
                        f"\n{'='*50}\n"
                        f"GOAL REACHED!  Distance: {dist_to_goal:.3f}m\n"
                        f"Steps: {iter_count}\n"
                        f"RMSE — CTE: {cte_rmse:.3f}m  |  Heading: {hdg_rmse:.1f}°\n"
                        f"{'='*50}\n"
                    )
                    # Goal animation: flash a celebration marker
                    for _ in range(10):
                        viewer.user_scn.ngeom = 0
                        draw_goal(viewer, goal)
                        draw_obstacle(viewer, obstacle_manager.obstacles)
                        # Pulsing green ring
                        t_frac = (_ / 10.0)
                        radius = 0.3 + t_frac * 0.5
                        rgba = np.array([0.0, 1.0, 0.0, 0.3 + 0.4 * (1 - t_frac)], dtype=np.float32)
                        if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                            g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                            mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                size=np.array([radius, 0.0, 0.0], dtype=np.float64),
                                pos=np.array(goal, dtype=np.float64),
                                mat=np.eye(3).ravel(), rgba=rgba)
                            viewer.user_scn.ngeom += 1
                        viewer.sync()
                        time.sleep(0.1)
                    break

                elapsed_real_time = time.perf_counter() - sim_start_time

                # ---- Sync with MPC thread ----
                with shared.lock:
                    cmd = shared.control.copy()
                    shared.last_control[:] = cmd
                    mpc_traj = shared.x_mpc_world

                # ---- MuJoCo physics step(s) ----
                for _ in range(sub_steps):
                    s = read_mujoco_state(m, d, quad_id)
                    rotor_thrusts = mixer.compute(cmd, s, None, m.opt.timestep)
                    d.ctrl[:] = rotor_thrusts
                    # Yaw torque via xfrc_applied (body-z in world frame)
                    if abs(cmd[3]) > 0.01:
                        body_z = d.xmat[quad_id].reshape(3, 3)[:, 2]
                        yaw_torque_mag = 0.5 * (float(cmd[3]) - d.qvel[5])
                        d.xfrc_applied[quad_id, 3:6] = body_z * yaw_torque_mag
                    else:
                        d.xfrc_applied[quad_id, 3:6] = [0.0, 0.0, 0.0]
                    mujoco.mj_step(m, d)
                d.xfrc_applied[quad_id, 3:6] = [0.0, 0.0, 0.0]

                mujoco_state = read_mujoco_state(m, d, quad_id)

                # ---- Share state with MPC thread ----
                with shared.lock:
                    shared.state[:] = mujoco_state

                # ---- Tracking metrics (CTE, heading error) ----
                # Cross-track error: perpendicular distance from direct line to goal
                goal_vec = goal - init_state[:3]
                goal_dir = goal_vec / max(np.linalg.norm(goal_vec), 1e-6)
                mav_vec = mujoco_state[:3] - init_state[:3]
                cte = float(np.linalg.norm(
                    mav_vec - np.dot(mav_vec, goal_dir) * goal_dir
                ))
                cte_hist.append(cte)
                # Heading error: difference between yaw and direction to goal
                desired_yaw = math.atan2(goal[1] - mujoco_state[1], goal[0] - mujoco_state[0])
                hdg_err = (mujoco_state[8] - desired_yaw + math.pi) % (2 * math.pi) - math.pi
                hdg_err_hist.append(abs(hdg_err))

                # ---- Log ----
                x_hist.append(mujoco_state[0])
                y_hist.append(mujoco_state[1])
                z_hist.append(mujoco_state[2])
                iter_count += 1

                # ---- Draw ----
                viewer.user_scn.ngeom = 0
                draw_goal(viewer, goal)
                draw_obstacle(viewer, obstacle_manager.obstacles)
                draw_uncertainty(viewer, mujoco_state[:3], shared.position_cov)
                draw_fov(viewer, mujoco_state[0], mujoco_state[1], mujoco_state[2],
                         mujoco_state[8], hfov_deg=87.0, vfov_deg=58.0, max_range=5.0)
                draw_trail(viewer, x_hist, y_hist, z_hist)
                if mpc_traj is not None:
                    draw_mpc_preview(viewer, mpc_traj)

                viewer.cam.lookat[:] = mujoco_state[:3]

                if iter_count % 10 == 0:
                    cte_rmse_live = float(np.sqrt(np.mean(np.square(cte_hist)))) if cte_hist else 0.0
                    hdg_rmse_live = float(np.degrees(np.sqrt(np.mean(np.square(hdg_err_hist))))) if hdg_err_hist else 0.0
                    print(
                        f"pos:({mujoco_state[0]:.1f},{mujoco_state[1]:.1f},{mujoco_state[2]:.1f}) "
                        f"| v:{np.linalg.norm(mujoco_state[3:6]):.2f} "
                        f"| CTE:{cte:.2f}m "
                        f"| hdg:{math.degrees(hdg_err):.1f}° "
                        f"| CTE_RMSE:{cte_rmse_live:.3f}m "
                        f"| HDG_RMSE:{hdg_rmse_live:.1f}° "
                        f"| goal:{dist_to_goal:.2f}m"
                    )

                viewer.sync()

                # Frame limiting
                frame_end = time.perf_counter()
                sleep_time = max(0.0, render_dt - (frame_end - sim_start_time - elapsed_real_time))
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            shared.is_active = False
            mpc_thread.join(timeout=2.0)
            if mpc_thread.is_alive():
                print("MPC thread did not shut down cleanly.")
            print("Simulation ended.")


if __name__ == "__main__":
    main()
