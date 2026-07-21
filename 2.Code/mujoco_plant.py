"""
mujoco_plant.py
================
Uses MuJoCo as the "true" physical plant (replacing do-mpc's built-in Simulator),
while the MPC "brain" (quad_mpc_core.py, CasADi/do-mpc/IPOPT) is UNCHANGED and still
predicts using its own simplified quaternion model. This lets us check whether the
MPC is robust to a mismatch between its internal prediction model and a more
trustworthy, independently-implemented rigid-body physics engine - plus we get real
contact-based collision detection (not just a soft-constraint distance check) and
MuJoCo's own renderer for nicer visuals.

State convention check (done empirically, see project notes): MuJoCo's free-joint
qpos = [x,y,z, qw,qx,qy,qz] and qvel = [vx,vy,vz, wx,wy,wz] use EXACTLY the same
conventions as our MPC state vector (quaternion scalar-first, linear velocity in
world frame, angular velocity in body frame) - so no basis conversion is needed,
only reordering into our 13-vector layout.
"""
import numpy as np
import mujoco
from quad_mpc_core import M, G, DRONE_RADIUS, obstacle_pos_at

MJCF_TEMPLATE = """
<mujoco model="quad">
  <option timestep="{mj_dt}" gravity="0 0 -9.81" integrator="RK4"/>
  <default>
    <geom contype="1" conaffinity="1" friction="0.8 0.02 0.02"/>
  </default>
  <worldbody>
    <light directional="true" diffuse="0.8 0.8 0.8" pos="2 2 6" dir="-0.3 -0.3 -1"/>
    <geom name="floor" type="plane" size="15 15 0.1" rgba="0.06 0.07 0.09 1"/>
    <camera name="fixed_view" pos="{cam_x} {cam_y} {cam_z}" mode="targetbody" target="drone"/>
    <body name="drone" pos="{x0} {y0} {z0}">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="{mass}" diaginertia="{ixx} {iyy} {izz}"/>
      <geom name="body" type="box" size="0.10 0.10 0.03" rgba="0.15 0.18 0.24 1"/>
      <geom name="arm1" type="capsule" fromto="-0.45 -0.45 0  0.45 0.45 0" size="0.015" rgba="0.2 0.24 0.32 1"/>
      <geom name="arm2" type="capsule" fromto="-0.45 0.45 0  0.45 -0.45 0" size="0.015" rgba="0.2 0.24 0.32 1"/>
      <geom name="r1" type="cylinder" pos="0.45 0.45 0.02" size="0.09 0.01" rgba="0.22 0.84 0.75 0.9"/>
      <geom name="r2" type="cylinder" pos="0.45 -0.45 0.02" size="0.09 0.01" rgba="0.22 0.84 0.75 0.9"/>
      <geom name="r3" type="cylinder" pos="-0.45 0.45 0.02" size="0.09 0.01" rgba="0.22 0.84 0.75 0.9"/>
      <geom name="r4" type="cylinder" pos="-0.45 -0.45 0.02" size="0.09 0.01" rgba="0.22 0.84 0.75 0.9"/>
      <site name="cog" pos="0 0 0" size="0.01"/>
    </body>
    {obstacle_xml}
  </worldbody>
  <actuator>
    <general name="thrust" site="cog" gear="0 0 1 0 0 0" ctrlrange="0 40" gaintype="fixed" gainprm="1"/>
    <general name="roll_torque"  site="cog" gear="0 0 0 1 0 0" ctrlrange="-3 3" gaintype="fixed" gainprm="1"/>
    <general name="pitch_torque" site="cog" gear="0 0 0 0 1 0" ctrlrange="-3 3" gaintype="fixed" gainprm="1"/>
    <general name="yaw_torque"   site="cog" gear="0 0 0 0 0 1" ctrlrange="-2 2" gaintype="fixed" gainprm="1"/>
  </actuator>
</mujoco>
"""


def build_mjcf(x0_vals, goal_pos, obstacles, mj_dt=0.002):
    mid_x = (x0_vals['x'] + goal_pos['x'])/2
    mid_y = (x0_vals['y'] + goal_pos['y'])/2
    mid_z = (x0_vals['z'] + goal_pos['z'])/2
    span = max(3.0, abs(goal_pos['x']-x0_vals['x']), abs(goal_pos['y']-x0_vals['y'])) 
    cam_x, cam_y, cam_z = mid_x - span*0.9, mid_y - span*1.3, mid_z + span*0.7

    obstacle_xml = ""
    static_i = 0
    dynamic_i = 0
    for obs in obstacles:
        if obs['type'] == 'static':
            obstacle_xml += (
                f'<geom name="obs_static_{static_i}" type="sphere" '
                f'pos="{obs["x"]} {obs["y"]} {obs["z"]}" size="{obs["radius"]}" '
                f'rgba="1 0.54 0.24 0.55" contype="1" conaffinity="1"/>\n'
            )
            static_i += 1
        else:
            obstacle_xml += (
                f'<body name="obs_dynamic_{dynamic_i}" mocap="true" '
                f'pos="{obs["x"]} 0 {obs["z"]}">\n'
                f'  <geom name="obs_dyn_geom_{dynamic_i}" type="sphere" size="{obs["radius"]}" '
                f'rgba="1 0.3 0.85 0.55" contype="1" conaffinity="1"/>\n'
                f'</body>\n'
            )
            dynamic_i += 1
    return MJCF_TEMPLATE.format(
        mj_dt=mj_dt, x0=x0_vals['x'], y0=x0_vals['y'], z0=x0_vals['z'],
        mass=M, ixx=0.012, iyy=0.012, izz=0.020, obstacle_xml=obstacle_xml,
        cam_x=cam_x, cam_y=cam_y, cam_z=cam_z,
    )


class MuJoCoPlant:
    """Wraps a MuJoCo model/data pair as the 'true' plant, with helpers to read the
    state in our 13-vector convention, apply control, step, drive mocap obstacles,
    and check for real (contact-based) collisions."""

    def __init__(self, x0_vals, goal_pos, obstacles, mj_dt=0.002):
        xml = build_mjcf(x0_vals, goal_pos, obstacles, mj_dt=mj_dt)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.obstacles = obstacles
        self.mj_dt = mj_dt

        q0 = _quat_from_euler(x0_vals.get('roll', 0), x0_vals.get('pitch', 0), x0_vals.get('yaw', 0))
        self.data.qpos[:] = [x0_vals['x'], x0_vals['y'], x0_vals['z'], *q0]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        self.dyn_mocap_ids = []
        di = 0
        for obs in obstacles:
            if obs['type'] == 'dynamic':
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'obs_dynamic_{di}')
                self.dyn_mocap_ids.append((self.model.body_mocapid[body_id], obs))
                di += 1

        self.drone_geom_ids = set()
        for gname in ['body', 'arm1', 'arm2', 'r1', 'r2', 'r3', 'r4']:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid >= 0:
                self.drone_geom_ids.add(gid)
        self.obstacle_geom_ids = set()
        for gname in self._all_geom_names():
            if gname.startswith('obs_'):
                self.obstacle_geom_ids.add(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, gname))

    def _all_geom_names(self):
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(self.model.ngeom)]

    def get_state_13(self):
        qpos, qvel = self.data.qpos, self.data.qvel
        return np.array([
            qpos[0], qpos[1], qpos[2],
            qvel[0], qvel[1], qvel[2],
            qpos[3], qpos[4], qpos[5], qpos[6],
            qvel[3], qvel[4], qvel[5],
        ]).reshape(-1, 1)

    def set_mocap_obstacles(self, t):
        for mocap_id, obs in self.dyn_mocap_ids:
            px, py, pz = obstacle_pos_at(obs, t)
            self.data.mocap_pos[mocap_id] = [px, py, pz]

    def apply_control_and_step(self, u_deltaT_tau, n_substeps, t0):
        """u_deltaT_tau = [dT, taux, tauy, tauz] (MPC convention). Converts thrust
        deviation to absolute thrust for MuJoCo's actuator, steps n_substeps times,
        keeping mocap obstacles kinematically updated each substep."""
        dT, taux, tauy, tauz = u_deltaT_tau
        self.data.ctrl[:] = [M*G + dT, taux, tauy, tauz]
        for i in range(n_substeps):
            self.set_mocap_obstacles(t0 + i*self.mj_dt)
            mujoco.mj_step(self.model, self.data)

    def check_collision(self):
        """Returns True if any active contact involves a drone geom and an obstacle geom."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 in self.drone_geom_ids and g2 in self.obstacle_geom_ids) or \
               (g2 in self.drone_geom_ids and g1 in self.obstacle_geom_ids):
                return True
        return False

    def try_create_renderer(self, width=640, height=480):
        """Attempts to create an offscreen renderer. Returns (renderer, None) on
        success, or (None, reason) if no GL backend is available on this machine -
        callers should fall back to the always-available Plotly visualization in
        that case. NOTE: the MUJOCO_GL environment variable (egl/osmesa/glfw) must
        be set BEFORE the mujoco module is first imported by the process; it is
        read once at import time, so switching backends at runtime is not safe."""
        try:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            renderer.update_scene(self.data)
            renderer.render()  # smoke-test an actual render call
            return renderer, None
        except Exception as e:
            return None, str(e)

    def render_frame(self, renderer, camera=None):
        renderer.update_scene(self.data, camera=camera)
        return renderer.render().copy()


def _quat_from_euler(roll, pitch, yaw):
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    return np.array([
        cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy,
    ])
