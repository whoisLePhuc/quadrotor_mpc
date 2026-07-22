"""Independent MuJoCo rigid-body plant for the legacy 13-state NMPC demo.

MuJoCo is imported lazily so the ODE/CC-MPC command-line simulation does not
require binary or OpenGL dependencies.
"""

from __future__ import annotations

import math

import numpy as np

from quad_mpc_core import DRONE_RADIUS, G, IXX, IYY, IZZ, M, obstacle_pos_at, quat_from_euler


class MuJoCoPlant:
    """Rigid-body plant driven by total thrust and body torque commands."""

    def __init__(self, x0_vals, goal_pos, obstacles, mj_dt: float = 0.002):
        try:
            import mujoco
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MuJoCo is optional; install with `pip install -r requirements-ui.txt`"
            ) from exc

        self.mujoco = mujoco
        self.obstacles = [dict(item) for item in obstacles]
        obstacle_xml: list[str] = []
        for index, obstacle in enumerate(self.obstacles):
            x, y, z = obstacle_pos_at(obstacle, 0.0)
            radius = float(obstacle["radius"])
            if obstacle["type"] == "dynamic":
                obstacle_xml.append(
                    f'<body name="obstacle_{index}" mocap="true" pos="{x} {y} {z}">'
                    f'<geom name="obstacle_geom_{index}" type="sphere" size="{radius}" '
                    'rgba="0.85 0.18 0.12 0.65" contype="1" conaffinity="1"/>'
                    '</body>'
                )
            else:
                obstacle_xml.append(
                    f'<body name="obstacle_{index}" pos="{x} {y} {z}">'
                    f'<geom name="obstacle_geom_{index}" type="sphere" size="{radius}" '
                    'rgba="0.85 0.18 0.12 0.65" contype="1" conaffinity="1"/>'
                    '</body>'
                )

        xml = f"""
        <mujoco model="quadrotor_nmpc_plant">
          <option timestep="{float(mj_dt)}" gravity="0 0 -{G}" integrator="RK4"/>
          <visual><global offwidth="960" offheight="720"/></visual>
          <worldbody>
            <light pos="0 -3 7" dir="0 0 -1"/>
            <geom name="ground" type="plane" size="20 20 0.1" rgba="0.20 0.24 0.30 1"/>
            <camera name="fixed_view" pos="8 -10 7" xyaxes="0.78 0.62 0 -0.30 0.38 0.87"/>
            <body name="quadrotor" pos="0 0 1">
              <freejoint name="quad_free"/>
              <inertial pos="0 0 0" mass="{M}" diaginertia="{IXX} {IYY} {IZZ}"/>
              <geom name="quad_collision" type="sphere" size="{DRONE_RADIUS}" mass="0"
                    rgba="0.08 0.45 0.95 0.75" contype="1" conaffinity="1"/>
              <geom type="box" size="0.32 0.035 0.025" mass="0" rgba="0.1 0.2 0.3 1"/>
              <geom type="box" size="0.035 0.32 0.025" mass="0" rgba="0.1 0.2 0.3 1"/>
            </body>
            {''.join(obstacle_xml)}
          </worldbody>
        </mujoco>
        """
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = float(mj_dt)
        self.quad_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
        self.quad_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "quad_collision"
        )
        self.obstacle_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_geom_{index}")
            for index in range(len(self.obstacles))
        }
        self.dynamic_mocap_ids: dict[int, int] = {}
        for index, obstacle in enumerate(self.obstacles):
            if obstacle["type"] == "dynamic":
                body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{index}"
                )
                self.dynamic_mocap_ids[index] = int(self.model.body_mocapid[body_id])

        quaternion = quat_from_euler(
            x0_vals.get("roll", 0.0), x0_vals.get("pitch", 0.0), x0_vals.get("yaw", 0.0)
        )
        self.data.qpos[:3] = [x0_vals["x"], x0_vals["y"], x0_vals["z"]]
        self.data.qpos[3:7] = quaternion
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _update_dynamic_obstacles(self, time_s: float) -> None:
        for index, mocap_id in self.dynamic_mocap_ids.items():
            self.data.mocap_pos[mocap_id] = obstacle_pos_at(self.obstacles[index], time_s)

    def apply_control_and_step(self, control, n_substeps: int, time_s: float) -> None:
        """Apply ``[thrust_deviation, tau_x, tau_y, tau_z]`` and integrate."""
        thrust_deviation, tau_x, tau_y, tau_z = np.asarray(control, dtype=float)
        for substep in range(int(n_substeps)):
            t = time_s + substep * float(self.model.opt.timestep)
            self._update_dynamic_obstacles(t)
            rotation = self.data.xmat[self.quad_id].reshape(3, 3)
            body_thrust = np.array([0.0, 0.0, M * G + thrust_deviation])
            body_torque = np.array([tau_x, tau_y, tau_z])
            self.data.xfrc_applied[self.quad_id, :3] = rotation @ body_thrust
            self.data.xfrc_applied[self.quad_id, 3:6] = rotation @ body_torque
            self.mujoco.mj_step(self.model, self.data)
        self.data.xfrc_applied[self.quad_id] = 0.0

    def get_state_13(self):
        """Return ``[p_world, v_world, q_wxyz, omega_body]`` as a column vector."""
        position = self.data.qpos[:3].copy()
        quaternion = self.data.qpos[3:7].copy()
        velocity_world = self.data.qvel[:3].copy()
        omega_body = self.data.qvel[3:6].copy()
        return np.concatenate([position, velocity_world, quaternion, omega_body]).reshape(-1, 1)

    def check_collision(self) -> bool:
        """Return true only for quadrotor-to-obstacle contacts (ground excluded)."""
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.quad_geom_id in pair and pair.intersection(self.obstacle_geom_ids):
                return True
        return False

    def try_create_renderer(self):
        try:
            return self.mujoco.Renderer(self.model, height=720, width=960), None
        except Exception as exc:  # OpenGL backend varies by host
            return None, str(exc)

    def render_frame(self, renderer, camera: str = "fixed_view"):
        renderer.update_scene(self.data, camera=camera)
        return renderer.render().copy()
