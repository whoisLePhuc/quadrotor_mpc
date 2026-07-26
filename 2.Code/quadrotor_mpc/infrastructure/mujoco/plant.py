"""Independent MuJoCo rigid-body plant for the legacy 13-state NMPC demo.

MuJoCo is imported lazily so the ODE/CC-MPC command-line simulation does not
require binary or OpenGL dependencies.
"""

from __future__ import annotations

import numpy as np

from quadrotor_mpc.control.nmpc.core import G, M, obstacle_pos_at, quat_from_euler
from quadrotor_mpc.core.vehicle import DEFAULT_QUADROTOR
from quadrotor_mpc.infrastructure.resources import resource_root


def _menagerie_assets() -> dict[str, bytes]:
    """Return the vendored Crazyflie MJCF and OBJ files as a MuJoCo VFS."""
    model_root = resource_root() / "models" / "bitcraze_crazyflie_2"
    paths = [model_root / "cf2.xml", *sorted((model_root / "assets").glob("*.obj"))]
    return {path.relative_to(model_root).as_posix(): path.read_bytes() for path in paths}


class MuJoCoPlant:
    """Crazyflie 2 rigid body driven by total thrust and body torques.

    The airframe, collision meshes, mass and inertia come from Google DeepMind's
    MuJoCo Menagerie model. Scenario objects are added by a separate scene layer.
    """

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
                    "</body>"
                )
            else:
                obstacle_xml.append(
                    f'<body name="obstacle_{index}" pos="{x} {y} {z}">'
                    f'<geom name="obstacle_geom_{index}" type="sphere" size="{radius}" '
                    'rgba="0.85 0.18 0.12 0.65" contype="1" conaffinity="1"/>'
                    "</body>"
                )

        goal_x, goal_y, goal_z = (float(goal_pos[axis]) for axis in ("x", "y", "z"))
        xml = f"""
        <mujoco model="crazyflie_2_nmpc_plant">
          <include file="cf2.xml"/>
          <asset>
            <texture name="workbench_sky" type="skybox" builtin="gradient"
                     rgb1="0.10 0.14 0.22" rgb2="0.02 0.03 0.06" width="512" height="3072"/>
            <texture name="workbench_floor_grid" type="2d" builtin="checker"
                     rgb1="0.16 0.19 0.24" rgb2="0.23 0.27 0.34"
                     width="512" height="512"/>
            <material name="workbench_floor_grid" texture="workbench_floor_grid" texrepeat="20 20"
                      reflectance="0.12"/>
          </asset>
          <visual>
            <global offwidth="960" offheight="720"/>
            <headlight ambient="0.45 0.45 0.45" diffuse="0.75 0.75 0.75"
                       specular="0.20 0.20 0.20"/>
            <map znear="0.01"/>
            <rgba haze="0.15 0.20 0.30 1"/>
          </visual>
          <worldbody>
            <light pos="3 -4 8" dir="-0.2 0.3 -1" directional="true"/>
            <geom name="ground" type="plane" size="20 20 0.1" material="workbench_floor_grid"/>
            <camera name="fixed_view" pos="8 -10 7" xyaxes="0.78 0.62 0 -0.30 0.38 0.87"/>
            <site name="goal_marker" type="sphere" pos="{goal_x} {goal_y} {goal_z}"
                  size="0.12" rgba="1.0 0.78 0.05 0.95"/>
            {"".join(obstacle_xml)}
          </worldbody>
        </mujoco>
        """
        self.model = mujoco.MjModel.from_xml_string(xml, assets=_menagerie_assets())
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = float(mj_dt)
        self.model.opt.gravity[:] = [0.0, 0.0, -G]
        self.quad_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cf2")
        self.quad_geom_ids = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) == self.quad_id
            and int(self.model.geom_contype[geom_id]) != 0
        }
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

        self.reset(x0_vals)

    def _update_dynamic_obstacles(self, time_s: float) -> None:
        for index, mocap_id in self.dynamic_mocap_ids.items():
            self.data.mocap_pos[mocap_id] = obstacle_pos_at(self.obstacles[index], time_s)

    def reset(self, x0_vals) -> None:
        """Reset the same model/data pair without reopening the native viewer."""
        self.mujoco.mj_resetData(self.model, self.data)
        quaternion = quat_from_euler(
            x0_vals.get("roll", 0.0),
            x0_vals.get("pitch", 0.0),
            x0_vals.get("yaw", 0.0),
        )
        self.data.qpos[:3] = [x0_vals["x"], x0_vals["y"], x0_vals["z"]]
        self.data.qpos[3:7] = quaternion
        self.data.qvel[:] = 0.0
        self._update_dynamic_obstacles(0.0)
        self.mujoco.mj_forward(self.model, self.data)

    def set_state_13(
        self,
        state_13: np.ndarray,
        *,
        obstacle_positions: np.ndarray | None = None,
    ) -> None:
        """Set a recorded state for deterministic viewer replay."""
        state = np.asarray(state_13, dtype=float).reshape(13)
        self.data.qpos[:3] = state[:3]
        self.data.qvel[:3] = state[3:6]
        self.data.qpos[3:7] = state[6:10]
        self.data.qvel[3:6] = state[10:13]
        if obstacle_positions is not None:
            positions = np.asarray(obstacle_positions, dtype=float).reshape(-1, 3)
            for index, mocap_id in self.dynamic_mocap_ids.items():
                self.data.mocap_pos[mocap_id] = positions[index]
        self.mujoco.mj_forward(self.model, self.data)

    def apply_control_and_step(self, control, n_substeps: int, time_s: float) -> None:
        """Apply ``[thrust_deviation, tau_x, tau_y, tau_z]`` and integrate."""
        thrust_deviation, tau_x, tau_y, tau_z = np.asarray(control, dtype=float)
        total_thrust = float(
            np.clip(M * G + thrust_deviation, 0.0, DEFAULT_QUADROTOR.max_total_thrust_n)
        )
        tau_x = float(
            np.clip(
                tau_x,
                -DEFAULT_QUADROTOR.max_roll_pitch_torque_nm,
                DEFAULT_QUADROTOR.max_roll_pitch_torque_nm,
            )
        )
        tau_y = float(
            np.clip(
                tau_y,
                -DEFAULT_QUADROTOR.max_roll_pitch_torque_nm,
                DEFAULT_QUADROTOR.max_roll_pitch_torque_nm,
            )
        )
        tau_z = float(
            np.clip(
                tau_z,
                -DEFAULT_QUADROTOR.max_yaw_torque_nm,
                DEFAULT_QUADROTOR.max_yaw_torque_nm,
            )
        )
        for substep in range(int(n_substeps)):
            t = time_s + substep * float(self.model.opt.timestep)
            self._update_dynamic_obstacles(t)
            rotation = self.data.xmat[self.quad_id].reshape(3, 3)
            body_thrust = np.array([0.0, 0.0, total_thrust])
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
            if pair.intersection(self.quad_geom_ids) and pair.intersection(self.obstacle_geom_ids):
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
