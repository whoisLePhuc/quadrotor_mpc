#! /usr/bin/env python3
"""
3D nosim demo for Chance-Constrained MPC quadrotor obstacle avoidance.

Usage:
    python sim_demo_nosim.py
    python sim_demo_nosim.py --scenario corridor
    python sim_demo_nosim.py --no-fov --solver OSQP

Based on: Lin, Zhu, Alonso-Mora — ICRA 2020
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import signal
import time

import numpy as np
import numpy.typing as npt
import yaml

from ccmpc.ccmpc import CCMPC
from ccmpc.dynamics import QuadrotorDynamics
from ccmpc.obstacle import ObstacleManager
from ccmpc.uncertainty import UncertaintyPropagator
from ccmpc.utils import detect_obstacle_in_fov


def _import_mujoco():
    """Import MuJoCo with a working GL backend. Returns (mujoco, mixer) or None."""
    for backend in ['egl', 'glfw', 'osmesa']:
        try:
            os.environ['MUJOCO_GL'] = backend
            import mujoco as mj
            from ccmpc.mixer import QuadrotorMixer
            print(f"  [INFO] MuJoCo backend: {backend}")
            return mj, QuadrotorMixer
        except Exception:
            continue
    # Clear MUJOCO_GL so user sees the original error, not osmesa noise
    os.environ.pop('MUJOCO_GL', None)
    print("  [WARN] MuJoCo not available — no working GL backend found.")
    return None, None


class SimLogger:
    """Records simulation data for post-hoc analysis.

    Logs state, controls, MPC trajectory, and solver metrics
    at every simulation step. Exports to CSV for analysis.
    """

    def __init__(self) -> None:
        self.step: list[int] = []
        self.t: list[float] = []
        # State: [x, y, z, vx, vy, vz, roll, pitch, yaw]
        self.pos_x: list[float] = []
        self.pos_y: list[float] = []
        self.pos_z: list[float] = []
        self.vel_x: list[float] = []
        self.vel_y: list[float] = []
        self.vel_z: list[float] = []
        self.roll: list[float] = []
        self.pitch: list[float] = []
        self.yaw: list[float] = []
        # Control: [phi_c, theta_c, vz_c, psi_dot_c]
        self.u0: list[float] = []
        self.u1: list[float] = []
        self.u2: list[float] = []
        self.u3: list[float] = []
        # Rotor thrusts (MuJoCo only)
        self.T1: list[float] = []
        self.T2: list[float] = []
        self.T3: list[float] = []
        self.T4: list[float] = []
        # MPC info
        self.mpc_traj_x: list[str] = []
        self.mpc_traj_y: list[str] = []
        self.mpc_traj_z: list[str] = []
        self.mpc_solve_ms: list[float] = []
        self.mpc_cost: list[float] = []
        # Metrics
        self.cte: list[float] = []
        self.hdg_err_deg: list[float] = []
        self.goal_dist: list[float] = []

    def record(
        self,
        step: int,
        sim_t: float,
        state: npt.NDArray[np.float64],
        control: npt.NDArray[np.float64],
        thrusts: npt.NDArray[np.float64] | None = None,
        mpc_traj: npt.NDArray[np.float64] | None = None,
        mpc_solve_ms: float = 0.0,
        mpc_cost: float = float("nan"),  # FIX BUG 7: was missing from signature
        cte: float = 0.0,
        hdg_err_deg: float = 0.0,
        goal_dist: float = 0.0,
    ) -> None:
        self.step.append(step)
        self.t.append(sim_t)
        self.pos_x.append(float(state[0]))
        self.pos_y.append(float(state[1]))
        self.pos_z.append(float(state[2]))
        self.vel_x.append(float(state[3]))
        self.vel_y.append(float(state[4]))
        self.vel_z.append(float(state[5]))
        self.roll.append(float(state[6]))
        self.pitch.append(float(state[7]))
        self.yaw.append(float(state[8]))
        self.u0.append(float(control[0]))
        self.u1.append(float(control[1]))
        self.u2.append(float(control[2]))
        self.u3.append(float(control[3]))
        if thrusts is not None:
            self.T1.append(float(thrusts[0]))
            self.T2.append(float(thrusts[1]))
            self.T3.append(float(thrusts[2]))
            self.T4.append(float(thrusts[3]))
        if mpc_traj is not None:
            self.mpc_traj_x.append(str(mpc_traj[0].round(3).tolist()))
            self.mpc_traj_y.append(str(mpc_traj[1].round(3).tolist()))
            self.mpc_traj_z.append(str(mpc_traj[2].round(3).tolist()))
        self.mpc_solve_ms.append(mpc_solve_ms)
        self.mpc_cost.append(mpc_cost)  # FIX BUG 7: now populated
        self.cte.append(cte)
        self.hdg_err_deg.append(hdg_err_deg)
        self.goal_dist.append(goal_dist)

    def save_csv(self, path: str = "sim_data.csv") -> None:
        """Save all logged data to a single CSV file."""
        header = (
            "step,t,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,"
            "roll,pitch,yaw,"
            "u_phi_c,u_theta_c,u_vz_c,u_psi_dot_c,"
            "T1,T2,T3,T4,"
            "mpc_solve_ms,cte,hdg_err_deg,goal_dist,"
            "mpc_traj_x,mpc_traj_y,mpc_traj_z"
        )
        rows: list[str] = [header]
        for i in range(len(self.step)):
            row = (
                f"{self.step[i]},{self.t[i]:.3f},"
                f"{self.pos_x[i]:.4f},{self.pos_y[i]:.4f},{self.pos_z[i]:.4f},"
                f"{self.vel_x[i]:.4f},{self.vel_y[i]:.4f},{self.vel_z[i]:.4f},"
                f"{self.roll[i]:.6f},{self.pitch[i]:.6f},{self.yaw[i]:.6f},"
                f"{self.u0[i]:.6f},{self.u1[i]:.6f},{self.u2[i]:.6f},{self.u3[i]:.6f},"
            )
            if i < len(self.T1):
                row += f"{self.T1[i]:.4f},{self.T2[i]:.4f},{self.T3[i]:.4f},{self.T4[i]:.4f},"
            else:
                row += ",,,,"
            row += (
                f"{self.mpc_solve_ms[i]:.1f},{self.cte[i]:.4f},{self.hdg_err_deg[i]:.2f},{self.goal_dist[i]:.4f},"
            )
            if i < len(self.mpc_traj_x):
                row += f'"{self.mpc_traj_x[i]}","{self.mpc_traj_y[i]}","{self.mpc_traj_z[i]}"'
            else:
                row += ",,"
            rows.append(row)
        import csv
        with open(path, "w") as f:
            f.write("\n".join(rows) + "\n")
        print(f"  [DATA] Saved {len(self.step)} rows to {path}")


class QuadrotorSim:
    """3D quadrotor simulation with CC-MPC control."""

    def __init__(self, args) -> None:
        pkg_root = pathlib.Path(__file__).parent
        scenario_path = pkg_root / "config" / f"{args.scenario}.yaml"
        if not scenario_path.exists():
            scenario_path = pkg_root / "config" / f"simulation_{args.scenario}.yaml"
        if not scenario_path.exists():
            scenario_path = pkg_root / "config" / "simulation.yaml"
        sim_config = yaml.safe_load(scenario_path.read_text())
        print(f"Scenario: {scenario_path.stem}")

        self.args = args
        start = sim_config["start"]
        self.state: npt.NDArray[np.float64] = np.array(start, dtype=np.float64)
        self.goal: npt.NDArray[np.float64] = np.array(sim_config["goal"], dtype=np.float64)
        self.goal_threshold: float = sim_config.get("goal_threshold", 0.4)
        self.sim_dt: float = sim_config.get("sim_timestep", 0.02)

        self.mpc = CCMPC(pkg_root / "config" / "mpc.yaml")
        if args.no_fov:
            self.mpc._fov_enabled = False
        self.mpc_dt: float = self.mpc.dt
        self.mpc_skip: int = max(1, round(self.mpc_dt / self.sim_dt))
        self.dynamics: QuadrotorDynamics = self.mpc.dynamics
        self.last_control: npt.NDArray[np.float64] = np.zeros(4)

        self.obstacle_manager: ObstacleManager = ObstacleManager.from_config(str(scenario_path))
        self.mpc_counter: int = 0
        self.mpc_solve_time: float = 0.0
        self.mpc_traj: npt.NDArray[np.float64] | None = None
        self.iter_count: int = 0

        # Data logger
        self.logger: SimLogger = SimLogger()

        # Engine selection
        self.engine = args.engine or "ode"
        self._mu = None
        if self.engine == "mujoco":
            self._init_mujoco()  # must init BEFORE first MPC solve

        # Tracking metrics
        self.x_hist: list[float] = [self.state[0]]
        self.y_hist: list[float] = [self.state[1]]
        self.z_hist: list[float] = [self.state[2]]
        self.cte_hist: list[float] = []
        self.hdg_err_hist: list[float] = []

        # Setup display
        self._has_display = not args.headless
        if self._has_display:
            global plt
            import matplotlib.pyplot as plt
            self._setup_plot()
        else:
            self.fig = None
            self.ax3d = None

    def _setup_plot(self) -> None:
        plt.style.use("ggplot")
        self.fig: plt.Figure = plt.figure(figsize=(14, 8))
        gs = plt.GridSpec(3, 4, figure=self.fig)

        self.ax3d = self.fig.add_subplot(gs[:, :3], projection="3d")
        self.ax3d.set_xlabel("X [m]")
        self.ax3d.set_ylabel("Y [m]")
        self.ax3d.set_zlabel("Z [m]")
        self.ax3d.set_title("Quadrotor CC-MPC Simulation")
        self.ax3d.scatter(*self.goal, c="gold", s=200, marker="*", label="goal")
        self._obs_patches: list = []
        for obs in self.obstacle_manager.obstacles:
            self._obs_patches.append(
                self._plot_ellipsoid(self.ax3d, obs.p_hat, obs.axes, obs.R_o, "red", 0.3)
            )

        (self.traj_line,) = self.ax3d.plot([], [], [], c="tab:blue", alpha=0.6, label="flight path")
        (self.mpc_line,) = self.ax3d.plot([], [], [], c="tab:green", marker=".", alpha=0.5, label="MPC preview")
        (self.pos_marker,) = self.ax3d.plot([], [], [], "o", c="tab:blue", markersize=8)

        # Subplots
        self.ax_vel = plt.subplot(gs[0, 3])
        self.ax_vel.set_ylabel("v [m/s]")
        self.ax_vel.set_xlabel("t [s]")
        (self.vel_line,) = self.ax_vel.plot([], [], c="tab:blue")

        self.ax_cte = plt.subplot(gs[1, 3])
        self.ax_cte.set_ylabel("CTE [m]")
        self.ax_cte.set_xlabel("t [s]")
        self.ax_cte.axhline(y=0, c="gray", ls="--", lw=0.5)
        (self.cte_line,) = self.ax_cte.plot([], [], c="tab:orange")

        self.ax_hdg = plt.subplot(gs[2, 3])
        self.ax_hdg.set_ylabel("heading err [deg]")
        self.ax_hdg.set_xlabel("t [s]")
        self.ax_hdg.axhline(y=0, c="gray", ls="--", lw=0.5)
        (self.hdg_line,) = self.ax_hdg.plot([], [], c="tab:red")

        # HUD
        self.hud = self.ax3d.text2D(
            0.02, 0.98, "", transform=self.ax3d.transAxes,
            va="top", fontfamily="monospace", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        self.ax3d.set_xlim(-1, 7)
        self.ax3d.set_ylim(-2, 5)
        self.ax3d.set_zlim(0, 4)
        self.ax3d.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.ion()
        plt.show()

    # ----- MuJoCo engine methods (lazy-init, used when --engine mujoco) -----

    def _init_mujoco(self):
        """Lazy initialize MuJoCo model + data + dynamics for engine=mujoco."""
        if self._mu is not None:
            return self._mu
        mj, MixerCls = _import_mujoco()
        if mj is None:
            raise RuntimeError("MuJoCo not available. Use --engine ode")
        pkg_root = pathlib.Path(__file__).parent
        model = mj.MjModel.from_xml_path(str(pkg_root / "models" / "quadrotor.xml"))
        data = mj.MjData(model)
        data.qpos[:3] = self.state[:3]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mj.mj_forward(model, data)
        mixer = MixerCls(mass=sum(model.body_mass))
        self._mu = (mj, model, data, mixer)
        # Replace MPC dynamics with MuJoCo-based version
        from ccmpc.mujoco_dynamics import MuJoCoDynamics
        quad_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "quadrotor")
        self.mpc.dynamics = MuJoCoDynamics(model, data, quad_id, mixer)
        return self._mu

    def _quat_to_euler(self, q):
        w, x, y, z = q
        roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = math.asin(max(-1, min(1, 2*(w*y - z*x))))
        yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return roll, pitch, yaw

    def _step_mujoco(self, control):
        """Advance MuJoCo physics by sim_dt steps. Returns (state, thrusts)."""
        if self._mu is None:
            self._init_mujoco()
        mj, model, data, mixer = self._mu
        sub_steps = max(1, round(self.sim_dt / model.opt.timestep))
        last_thrust = np.zeros(4)
        for _ in range(sub_steps):
            pos = data.xpos[1].copy()
            vel = data.qvel[:3].copy()
            quat = data.xquat[1].copy()
            roll, pitch, yaw = self._quat_to_euler(quat)
            mstate = np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], roll, pitch, yaw])
            last_thrust = mixer.compute(control, mstate, dt=model.opt.timestep)
            data.ctrl[:] = last_thrust
            # Yaw torque via xfrc_applied (body-z torque transformed to world)
            psi_dot_c = float(control[3])
            if abs(psi_dot_c) > 0.01:
                body_z = data.xmat[1].reshape(3, 3)[:, 2]  # body z in world coords
                psi_dot = data.qvel[5]  # z-angular velocity in world
                yaw_torque_mag = 3.0 * (psi_dot_c - psi_dot)
                data.xfrc_applied[1, 3:6] = body_z * yaw_torque_mag
            else:
                data.xfrc_applied[1, 3:6] = [0.0, 0.0, 0.0]
            mj.mj_step(model, data)
        data.xfrc_applied[1, 3:6] = [0.0, 0.0, 0.0]
        pos = data.xpos[1].copy()
        vel = data.qvel[:3].copy()
        quat = data.xquat[1].copy()
        roll, pitch, yaw = self._quat_to_euler(quat)
        state = np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], roll, pitch, yaw])
        return state, last_thrust

    # ----- Plot helpers -----

    def _plot_uncertainty(self, ax, center, cov, scale=3.0):
        """Draw 3σ uncertainty ellipsoid from covariance matrix."""
        try:
            w, v = np.linalg.eigh(cov)
            w = np.maximum(w, 1e-10)
            axes = np.sqrt(w) * scale
            u = np.linspace(0, 2 * np.pi, 12)
            v_ang = np.linspace(0, np.pi, 8)
            x = axes[0] * np.outer(np.cos(u), np.sin(v_ang))
            y = axes[1] * np.outer(np.sin(u), np.sin(v_ang))
            z = axes[2] * np.outer(np.ones_like(u), np.cos(v_ang))
            pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
            pts = pts @ v.T + center
            return ax.plot_surface(
                pts[:, 0].reshape(x.shape), pts[:, 1].reshape(y.shape),
                pts[:, 2].reshape(z.shape), color="orange", alpha=0.12,
                edgecolor="none"
            )
        except np.linalg.LinAlgError:
            return None

    def _plot_ellipsoid(self, ax, center, axes, R, color, alpha):
        u = np.linspace(0, 2 * np.pi, 15)
        v = np.linspace(0, np.pi, 10)
        x = axes[0] * np.outer(np.cos(u), np.sin(v))
        y = axes[1] * np.outer(np.sin(u), np.sin(v))
        z = axes[2] * np.outer(np.ones_like(u), np.cos(v))
        pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        pts = pts @ R.T + center
        return ax.plot_surface(
            pts[:, 0].reshape(x.shape), pts[:, 1].reshape(y.shape),
            pts[:, 2].reshape(z.shape), color=color, alpha=alpha, edgecolor="none"
        )

    def run(self) -> None:
        print(f"  Horizon: {self.mpc.control_horizon} steps ({self.mpc.horizon_time:.1f}s)")
        print(f"  MPC dt: {self.mpc_dt}s  |  Sim dt: {self.sim_dt}s")
        print(f"  Obstacles: {len(self.obstacle_manager.obstacles)}")
        print(f"  delta: {self.mpc.delta}")
        print("Close plot window or press CTRL-C to stop.\n")

        if self._has_display:
            self._render()
        init_state = self.state.copy()

        try:
            while True:
                dist_to_goal = np.linalg.norm(self.state[:3] - self.goal)
                if dist_to_goal < self.goal_threshold:
                    cte_rmse = float(np.sqrt(np.mean(np.square(self.cte_hist)))) if self.cte_hist else 0.0
                    hdg_rmse = float(np.degrees(np.sqrt(np.mean(np.square(self.hdg_err_hist))))) if self.hdg_err_hist else 0.0
                    print(f"\nGOAL REACHED!  Distance: {dist_to_goal:.3f}m")
                    print(f"Steps: {self.iter_count}")
                    print(f"RMSE — CTE: {cte_rmse:.3f}m  |  Heading: {hdg_rmse:.1f}°\n")
                    if self.args.log:
                        self.logger.save_csv(self.args.log)
                    if self._has_display:
                        plt.pause(1.5)
                        plt.close(self.fig)
                    return

                # MPC update
                if self.mpc_counter % self.mpc_skip == 0:
                    noisy_state = UncertaintyPropagator.add_measurement_noise(self.state)

                    # Delay compensation
                    if self.mpc_solve_time > 0:
                        pred_state = self.dynamics.discrete(noisy_state, self.last_control, self.mpc_solve_time)
                    else:
                        pred_state = noisy_state.copy()

                    # FOV filter
                    visible = detect_obstacle_in_fov(
                        self.obstacle_manager.obstacles, pred_state[:3], pred_state[8],
                        hfov_deg=90.0, vfov_deg=70.0, max_range=5.0,
                    )
                    obs_for_mpc = ObstacleManager([o for _, o in visible]) if visible else None
                    self.obstacle_manager.update(self.mpc_dt)

                    t0 = time.perf_counter()
                    x_traj, u_traj = self.mpc.solve(pred_state, self.goal, obstacle_manager=obs_for_mpc)
                    self.mpc_solve_time = time.perf_counter() - t0

                    if x_traj is not None:
                        self.mpc_traj = x_traj.copy()
                    if u_traj is not None and u_traj.shape[1] > 0:
                        self.last_control = u_traj[:, 0].copy()

                self.mpc_counter += 1

                u = self.last_control.copy()

                # Physics step (ODE or MuJoCo)
                thrusts: npt.NDArray[np.float64] | None = None
                if self.engine == "mujoco":
                    self.state, thrusts = self._step_mujoco(u)
                else:
                    self.state = self.dynamics.discrete(self.state, u, self.sim_dt)
                self.iter_count += 1

                # Tracking metrics
                goal_vec = self.goal - init_state[:3]
                goal_dir = goal_vec / max(np.linalg.norm(goal_vec), 1e-6)
                mav_vec = self.state[:3] - init_state[:3]
                cte = float(np.linalg.norm(mav_vec - np.dot(mav_vec, goal_dir) * goal_dir))
                self.cte_hist.append(cte)
                desired_yaw = math.atan2(self.goal[1] - self.state[1], self.goal[0] - self.state[0])
                hdg_err = (self.state[8] - desired_yaw + math.pi) % (2 * math.pi) - math.pi
                hdg_deg = float(math.degrees(abs(hdg_err)))
                self.hdg_err_hist.append(hdg_deg)

                # Log
                self.x_hist.append(self.state[0])
                self.y_hist.append(self.state[1])
                self.z_hist.append(self.state[2])

                # Data logger
                self.logger.record(
                    step=self.iter_count,
                    sim_t=self.iter_count * self.sim_dt,
                    state=self.state,
                    control=u,
                    thrusts=thrusts,
                    mpc_traj=self.mpc_traj,
                    mpc_solve_ms=self.mpc_solve_time * 1000.0,
                    cte=cte,
                    hdg_err_deg=hdg_deg,
                    goal_dist=float(dist_to_goal),
                )

                if self._has_display and self.iter_count % 5 == 0:
                    self._render()

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            if self._has_display:
                plt.ioff()
                plt.show()

    def _render(self) -> None:
        self.ax3d.set_title(f"Quadrotor CC-MPC  |  t={self.iter_count * self.sim_dt:.1f}s")
        ds = max(1, len(self.x_hist) // 500)
        self.traj_line.set_data(self.x_hist[::ds], self.y_hist[::ds])
        self.traj_line.set_3d_properties(self.z_hist[::ds])
        self.pos_marker.set_data([self.state[0]], [self.state[1]])
        self.pos_marker.set_3d_properties([self.state[2]])

        if self.mpc_traj is not None:
            self.mpc_line.set_data(self.mpc_traj[0], self.mpc_traj[1])
            self.mpc_line.set_3d_properties(self.mpc_traj[2])
        else:
            self.mpc_line.set_data([], [])
            self.mpc_line.set_3d_properties([])

        for i, obs in enumerate(self.obstacle_manager.obstacles):
            if i < len(self._obs_patches):
                self._obs_patches[i].remove()
                self._obs_patches[i] = self._plot_ellipsoid(
                    self.ax3d, obs.p_hat, obs.axes, obs.R_o, "red", 0.3
                )

        # Uncertainty ellipsoid
        if hasattr(self.mpc, 'uncertainty') and self.mpc._previous_covariance is not None:
            pos_cov = self.mpc.uncertainty.position_covariance(self.mpc._previous_covariance)
            self._plot_uncertainty(self.ax3d, self.state[:3], pos_cov)

        goal_dist = np.linalg.norm(self.state[:3] - self.goal)
        cte_str = f"CTE: {self.cte_hist[-1]:.3f}m" if self.cte_hist else "CTE: --"
        hdg_str = f"hdg: {math.degrees(self.hdg_err_hist[-1]):.1f}°" if self.hdg_err_hist else "hdg: --"
        self.hud.set_text(
            f"pos: ({self.state[0]:.2f}, {self.state[1]:.2f}, {self.state[2]:.2f})\n"
            f"v: {np.linalg.norm(self.state[3:6]):.2f} m/s\n"
            f"goal: {goal_dist:.2f} m\n"
            f"{cte_str}  |  {hdg_str}\n"
            f"MPC: {self.mpc_solve_time*1000:.0f} ms"
        )

        t_arr = np.arange(len(self.cte_hist)) * self.sim_dt * self.mpc_skip
        if len(t_arr) > 1:
            n = min(len(t_arr), len(self.cte_hist))
            self.cte_line.set_data(t_arr[:n], self.cte_hist[:n])
            self.ax_cte.relim()
            self.ax_cte.autoscale_view(scalex=True, scaley=True)

            self.hdg_line.set_data(t_arr[:n], np.degrees(self.hdg_err_hist[:n]))
            self.ax_hdg.relim()
            self.ax_hdg.autoscale_view(scalex=True, scaley=True)

        plt.draw()
        plt.pause(0.001)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quadrotor CC-MPC Simulation")
    parser.add_argument("--scenario", default="simulation", help="Scenario name")
    parser.add_argument("--engine", default="ode", choices=["ode", "mujoco"],
                        help="Physics engine (ode=kinematic, mujoco=full physics)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display (no matplotlib)")
    parser.add_argument("--no-fov", action="store_true", help="Disable FOV constraints")
    parser.add_argument("--solver", default=None, help="CVXPY solver")
    parser.add_argument("--horizon", type=float, default=None, help="Horizon time (s)")
    parser.add_argument("--log", type=str, default=None,
                        help="Save simulation data to CSV (e.g. --log flight_data.csv)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal.default_int_handler)
    sim = QuadrotorSim(args)
    sim.run()


if __name__ == "__main__":
    main()