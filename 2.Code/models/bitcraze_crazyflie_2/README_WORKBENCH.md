# Crazyflie 2 model provenance

This directory vendors the `bitcraze_crazyflie_2` model from
[Google DeepMind MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/bitcraze_crazyflie_2)
at commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`.

The vendored MJCF and mesh assets are unchanged. The workbench creates a separate
scene at runtime to add the ground, goal and scenario obstacles. The upstream MIT
license is preserved in `LICENSE`.

Upstream documents the rigid-body values as:

- mass: `0.027 kg`;
- diagonal inertia: `2.3951e-5 2.3951e-5 3.2347e-5 kg m^2`;
- combined body thrust and body-moment inputs at the inertial frame;
- gyro, accelerometer and quaternion sensors at the IMU site.

Important limitation: upstream states that its actuator `ctrlrange` values are
arbitrary. This workbench therefore uses the upstream mass/inertia and mesh model,
but keeps conservative torque limits in `vehicle.py`. The plant is a body-wrench
model; it does not yet simulate motor lag, individual propeller aerodynamics,
battery voltage sag or rotor downwash.
