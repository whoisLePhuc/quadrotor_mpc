"""Physical parameters for the optional high-fidelity quadrotor track."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QuadrotorParameters:
    """Rigid-body and conservative actuator parameters in SI units."""

    name: str
    mass_kg: float
    inertia_kg_m2: tuple[float, float, float]
    collision_radius_m: float
    max_total_thrust_n: float
    max_roll_pitch_torque_nm: float
    max_yaw_torque_nm: float
    linear_damping_per_s: float
    angular_damping_nms: float
    yaw_damping_nms: float
    source_url: str

    @property
    def hover_thrust_n(self) -> float:
        return self.mass_kg * 9.81

    @property
    def max_upward_thrust_deviation_n(self) -> float:
        return max(0.0, self.max_total_thrust_n - self.hover_thrust_n)


# Mass and inertia are the values used by Google DeepMind MuJoCo Menagerie's
# Bitcraze Crazyflie 2 model. The Menagerie README traces them to the Bitcraze
# datasheet and MIT system identification. Menagerie explicitly states that its
# moment ctrlranges are arbitrary, so the torque limits below are conservative
# workbench assumptions and are intentionally documented as such.
CRAZYFLIE_2 = QuadrotorParameters(
    name="Bitcraze Crazyflie 2",
    mass_kg=0.027,
    inertia_kg_m2=(2.3951e-5, 2.3951e-5, 3.2347e-5),
    collision_radius_m=0.07,
    max_total_thrust_n=0.35,
    max_roll_pitch_torque_nm=0.002,
    max_yaw_torque_nm=0.0004,
    linear_damping_per_s=0.30,
    angular_damping_nms=4.0e-5,
    yaw_damping_nms=1.0e-5,
    source_url=(
        "https://github.com/google-deepmind/mujoco_menagerie/tree/main/bitcraze_crazyflie_2"
    ),
)


DEFAULT_QUADROTOR = CRAZYFLIE_2
