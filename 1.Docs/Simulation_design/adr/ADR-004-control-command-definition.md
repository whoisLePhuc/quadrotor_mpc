# ADR-004: Canonical Control Command Definition

> Status: Proposed
> Date: 2026-06-28
> Project: Quadrotor CC-MPC Simulation
> Related documents:
>
> * `docs/design/04_DATA_MODEL.md`
> * `docs/design/ADR/ADR-003-state-vector-definition.md`

---

## 1. Context

The refactored quadrotor CC-MPC simulation requires a clear definition of the control command exchanged between:

```text
Controller
MPC solver
ODE dynamics
Low-level mixer
MuJoCo adapter
Physics engine
Logger
Runtime loop
```

The project currently involves at least two different meanings of “control”:

```text
High-level controller command
Low-level actuator command
```

The CC-MPC controller produces a 4-dimensional high-level command:

```text
[phi_c, theta_c, vz_c, psi_dot_c]
```

However, a MuJoCo rotor-force simulation expects actuator-level commands such as rotor thrusts:

```text
[T1, T2, T3, T4]
```

If the project does not formally distinguish these two command types, the following bugs can occur:

```text
MPC command is accidentally passed directly to MuJoCo actuators
roll command is interpreted as rotor thrust
pitch command is interpreted as torque
vertical velocity command is interpreted as force
yaw-rate command is interpreted as yaw torque
logger mixes high-level command and actuator command
ODE and MuJoCo engines receive different semantic inputs
```

Therefore, the project needs a formal architectural decision for the canonical control command.

---

## 2. Decision

The simulation shall use `ControlCommand4` as the canonical high-level controller command.

```text
ControlCommand4 = [phi_c, theta_c, vz_c, psi_dot_c]
```

Mathematical notation:

$$
\mathbf{u}
=

\begin{bmatrix}
\phi_c \
\theta_c \
v_{z,c} \
\dot{\psi}_c
\end{bmatrix}
\in
\mathbb{R}^4
$$

Where:

| Index | Field       | Symbol         | Unit  | Meaning                     |
| ----: | ----------- | -------------- | ----- | --------------------------- |
|     0 | `phi_c`     | $\phi_c$       | rad   | Commanded roll angle        |
|     1 | `theta_c`   | $\theta_c$     | rad   | Commanded pitch angle       |
|     2 | `vz_c`      | $v_{z,c}$      | m/s   | Commanded vertical velocity |
|     3 | `psi_dot_c` | $\dot{\psi}_c$ | rad/s | Commanded yaw rate          |

`ControlCommand4` shall not be interpreted as rotor thrust, torque, PWM, motor speed, or MuJoCo actuator control.

---

## 3. Relationship With State9

This ADR depends on `ADR-003-state-vector-definition.md`.

The canonical state is:

```text
State9 = [x, y, z, vx, vy, vz, roll, pitch, yaw]
```

The canonical control is:

```text
ControlCommand4 = [phi_c, theta_c, vz_c, psi_dot_c]
```

Together they define the canonical dynamics interface:

$$
\dot{\mathbf{x}}
=

\mathbf{f}
(
\mathbf{x},
\mathbf{u}
)
$$

or in discrete time:

$$
\mathbf{x}_{k+1}
=

\mathbf{f}_d
(
\mathbf{x}_k,
\mathbf{u}_k
)
$$

where:

| Symbol         | Meaning                       |
| -------------- | ----------------------------- |
| $\mathbf{x}_k$ | `State9` at step $k$          |
| $\mathbf{u}_k$ | `ControlCommand4` at step $k$ |
| $\mathbf{f}_d$ | Discrete-time dynamics model  |

---

## 4. Command Field Semantics

### 4.1 `phi_c`

`phi_c` is the commanded roll angle.

```text
unit: rad
semantic type: commanded attitude
```

It is not:

```text
roll torque
rotor thrust difference
body angular velocity
PWM command
```

In the reduced quadrotor model, roll follows a first-order response:

$$
\dot{\phi}
=

\frac{1}{\tau_\phi}
(
k_\phi \phi_c - \phi
)
$$

---

### 4.2 `theta_c`

`theta_c` is the commanded pitch angle.

```text
unit: rad
semantic type: commanded attitude
```

It is not:

```text
pitch torque
rotor thrust difference
body angular velocity
PWM command
```

In the reduced quadrotor model, pitch follows a first-order response:

$$
\dot{\theta}
=

\frac{1}{\tau_\theta}
(
k_\theta \theta_c - \theta
)
$$

---

### 4.3 `vz_c`

`vz_c` is the commanded vertical velocity.

```text
unit: m/s
semantic type: vertical velocity command
```

It is not:

```text
vertical force
total thrust
altitude command
rotor speed
```

In the reduced quadrotor model:

$$
\dot{v}_z
=

\frac{1}{\tau_{vz}}
(
k_{vz} v_{z,c} - v_z
)
$$

---

### 4.4 `psi_dot_c`

`psi_dot_c` is the commanded yaw rate.

```text
unit: rad/s
semantic type: yaw-rate command
```

It is not:

```text
yaw angle command
yaw torque
rotor drag torque
motor speed difference
```

In the reduced quadrotor model:

$$
\dot{\psi}
=

\dot{\psi}_c
$$

---

## 5. ControlCommand4 Is Not ActuatorCommand4

The project shall distinguish between:

```text
ControlCommand4
ActuatorCommand4
```

---

### 5.1 `ControlCommand4`

`ControlCommand4` is produced by the high-level controller.

```text
ControlCommand4 = [phi_c, theta_c, vz_c, psi_dot_c]
```

Owner:

```text
Controller / CC-MPC
```

Consumed by:

```text
ODE dynamics
LowLevelMixer
Logger
```

May be consumed directly by:

```text
ODEPhysicsEngine
```

Should not be consumed directly by:

```text
MuJoCo rotor-force actuator
```

---

### 5.2 `ActuatorCommand4`

`ActuatorCommand4` is produced by the low-level mixer.

For rotor-thrust simulation:

```text
ActuatorCommand4 = [T1, T2, T3, T4]
```

Mathematically:

$$
\mathbf{T}
=

\begin{bmatrix}
T_1 \
T_2 \
T_3 \
T_4
\end{bmatrix}
\in
\mathbb{R}^4
$$

Where:

| Index | Field | Unit | Meaning        |
| ----: | ----- | ---- | -------------- |
|     0 | `T1`  | N    | Rotor 1 thrust |
|     1 | `T2`  | N    | Rotor 2 thrust |
|     2 | `T3`  | N    | Rotor 3 thrust |
|     3 | `T4`  | N    | Rotor 4 thrust |

Owner:

```text
LowLevelMixer
```

Consumed by:

```text
MuJoCoPhysicsEngine
Actuator-level physics backend
Logger
```

---

## 6. Required Command Flow

The canonical command flow shall be:

```text
Controller
    -> ControlCommand4
    -> LowLevelMixer
    -> ActuatorCommand4
    -> PhysicsEngine
```

For ODE simulation, the simplified dynamics may directly consume `ControlCommand4`:

```text
Controller
    -> ControlCommand4
    -> ODEPhysicsEngine
```

For MuJoCo rotor-force simulation, the engine shall not directly consume `ControlCommand4`.

Correct MuJoCo flow:

```text
Controller
    -> ControlCommand4
    -> QuadrotorMixer
    -> ActuatorCommand4
    -> MuJoCo ctrl
```

Incorrect MuJoCo flow:

```text
Controller
    -> ControlCommand4
    -> MuJoCo ctrl
```

---

## 7. Why ODE Can Consume ControlCommand4 Directly

The reduced ODE model is defined directly using the high-level command:

$$
\mathbf{u}
=

\begin{bmatrix}
\phi_c \
\theta_c \
v_{z,c} \
\dot{\psi}_c
\end{bmatrix}
$$

Therefore, `ODEPhysicsEngine` may implement:

```python
next_state = dynamics.discrete(
    state=current_state,
    control=control_command,
    dt=dt,
)
```

This is valid because the ODE dynamics model is not modeling individual rotors.

---

## 8. Why MuJoCo Should Not Consume ControlCommand4 Directly

MuJoCo rotor-force simulation models actuator forces applied at rotor sites.

Therefore, the MuJoCo engine expects actuator-level input such as:

```text
rotor thrust
general actuator force
external torque
```

`ControlCommand4` does not represent rotor thrust.

Therefore, MuJoCo requires a mapping:

$$
\mathbf{T}
=
\text{Mixer}
(
\mathbf{x},
\mathbf{u}
)
$$

where:

| Symbol       | Data type          |
| ------------ | ------------------ |
| $\mathbf{x}$ | `State9`           |
| $\mathbf{u}$ | `ControlCommand4`  |
| $\mathbf{T}$ | `ActuatorCommand4` |

The mixer may use current roll, pitch, yaw, vertical velocity, and previous state to compute rotor thrust.

---

## 9. Mixer Responsibility

The low-level mixer shall be the only module responsible for converting:

```text
ControlCommand4 -> ActuatorCommand4
```

The mixer may implement:

```text
attitude PD control
vertical velocity control
tilt compensation
rotor thrust allocation
saturation
```

The mixer shall not solve MPC.

The mixer shall not update the simulation state.

The mixer shall not own the physics engine.

The mixer shall be a pure or mostly pure transformation module:

```python
actuator_command = mixer.compute(
    command=control_command,
    state=current_state,
    previous_state=previous_state,
    dt=dt,
)
```

---

## 10. Command Limits

The canonical controller command shall respect configured bounds.

Recommended limits:

| Field       |                    Lower |                   Upper | Unit  |
| ----------- | -----------------------: | ----------------------: | ----- |
| `phi_c`     |          `-max_roll_cmd` |          `max_roll_cmd` | rad   |
| `theta_c`   |         `-max_pitch_cmd` |         `max_pitch_cmd` | rad   |
| `vz_c`      | `-max_vertical_velocity` | `max_vertical_velocity` | m/s   |
| `psi_dot_c` |          `-max_yaw_rate` |          `max_yaw_rate` | rad/s |

These limits shall be defined in configuration and validated at module boundaries.

Example:

```yaml
controller:
  limits:
    max_roll: 0.35
    max_pitch: 0.35
    max_vert_vel: 3.0
    max_yaw_rate: 0.8
```

---

## 11. Suggested Python Type

The refactored codebase should introduce a shared type for `ControlCommand4`.

Recommended location:

```text
simulation/state.py
```

or:

```text
ccmpc/types.py
```

Suggested implementation:

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ControlCommand4:
    data: np.ndarray

    def __post_init__(self):
        if self.data.shape != (4,):
            raise ValueError("ControlCommand4 must have shape (4,)")
        if not np.all(np.isfinite(self.data)):
            raise ValueError("ControlCommand4 contains NaN or Inf")

    @property
    def phi_c(self) -> float:
        return float(self.data[0])

    @property
    def theta_c(self) -> float:
        return float(self.data[1])

    @property
    def vz_c(self) -> float:
        return float(self.data[2])

    @property
    def psi_dot_c(self) -> float:
        return float(self.data[3])
```

The implementation may use raw NumPy arrays internally for performance, but public APIs shall document and validate the ordering.

---

## 12. Suggested Actuator Type

The refactored codebase should also introduce `ActuatorCommand4`.

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ActuatorCommand4:
    data: np.ndarray

    def __post_init__(self):
        if self.data.shape != (4,):
            raise ValueError("ActuatorCommand4 must have shape (4,)")
        if not np.all(np.isfinite(self.data)):
            raise ValueError("ActuatorCommand4 contains NaN or Inf")
        if np.any(self.data < 0.0):
            raise ValueError("Rotor thrust must be non-negative")

    @property
    def T1(self) -> float:
        return float(self.data[0])

    @property
    def T2(self) -> float:
        return float(self.data[1])

    @property
    def T3(self) -> float:
        return float(self.data[2])

    @property
    def T4(self) -> float:
        return float(self.data[3])
```

---

## 13. Public Interface Rules

### Rule 1: Controller output

A controller shall return `ControlCommand4`.

```python
command = controller.compute_command(observation)
```

Return type:

```text
ControlCommand4
```

The controller shall not return rotor thrust unless it is explicitly a low-level controller.

---

### Rule 2: ODE engine input

The ODE physics engine may accept `ControlCommand4`.

```python
next_state = ode_engine.step(
    command=control_command,
    dt=sim_dt,
)
```

This is valid because the ODE dynamics model is defined using high-level command inputs.

---

### Rule 3: MuJoCo engine input

The MuJoCo physics engine shall accept `ActuatorCommand4`.

```python
next_state = mujoco_engine.step(
    actuator_command=actuator_command,
    dt=sim_dt,
)
```

If MuJoCo engine accepts `ControlCommand4`, that engine must explicitly contain or own a mixer adapter. This design is allowed only if documented in `05_ENGINE_INTERFACE.md`.

Preferred architecture:

```text
SimulationRuntime owns the mixer.
MuJoCoPhysicsEngine receives ActuatorCommand4.
```

---

### Rule 4: Logger shall record both command levels

The logger shall distinguish:

```text
control_phi_c
control_theta_c
control_vz_c
control_psi_dot_c

actuator_T1
actuator_T2
actuator_T3
actuator_T4
```

The logger shall not use ambiguous names such as:

```text
u0
u1
u2
u3
ctrl0
ctrl1
ctrl2
ctrl3
```

unless accompanied by metadata mapping.

---

### Rule 5: No silent reinterpretation

No module shall reinterpret a `ControlCommand4` as:

```text
force
thrust
torque
PWM
rotor speed
MuJoCo ctrl
```

without an explicit adapter.

---

## 14. Alternatives Considered

### 14.1 Alternative A: Use rotor thrust as canonical control

```text
u = [T1, T2, T3, T4]
```

Decision: Rejected as canonical controller command.

Reason:

```text
The CC-MPC theory baseline and reduced dynamics model use commanded roll, pitch, vertical velocity, and yaw rate, not individual rotor thrust.
```

Rotor thrust is still valid as actuator-level data.

---

### 14.2 Alternative B: Use torque and total thrust

```text
u = [T, tau_x, tau_y, tau_z]
```

Decision: Rejected for current refactor.

Reason:

```text
This corresponds more closely to a rigid-body Newton-Euler model, but the current CC-MPC model assumes a higher-level command interface.
```

This may be reconsidered if the project moves to a full rigid-body NMPC model.

---

### 14.3 Alternative C: Use body rates

```text
u = [p_cmd, q_cmd, r_cmd, thrust_cmd]
```

Decision: Rejected for current refactor.

Reason:

```text
The current reduced model is expressed in Euler angle command and vertical velocity command, not body-rate command.
```

---

### 14.4 Alternative D: Use acceleration command

```text
u = [ax_cmd, ay_cmd, az_cmd, yaw_rate_cmd]
```

Decision: Rejected.

Reason:

```text
Acceleration commands would require another mapping layer from acceleration to attitude command and are not the current CC-MPC model input.
```

---

## 15. Consequences

### 15.1 Positive consequences

This decision provides:

1. A single high-level control command definition.
2. Clear distinction between controller command and actuator command.
3. Compatibility with the reduced CC-MPC dynamics.
4. Cleaner engine abstraction.
5. Easier logging and debugging.
6. Easier replacement of MuJoCo or ODE engine.
7. Clear responsibility for the low-level mixer.
8. Less risk of passing the wrong vector into the wrong module.

---

### 15.2 Negative consequences

This decision also has limitations:

1. It assumes a high-level attitude/velocity command interface.
2. It does not directly model motor dynamics.
3. It requires a mixer for rotor-force physics engines.
4. It may mismatch a real drone API that expects body-rate or thrust commands.
5. MuJoCo requires additional conversion from high-level command to rotor thrust.
6. Aggressive maneuvers may not be accurately represented by this reduced command model.

These limitations are acceptable for the current refactor target.

---

## 16. Risk Analysis

### Risk 1: Directly passing ControlCommand4 to MuJoCo

Cause:

```text
Developer assumes u means actuator input.
```

Consequence:

```text
MuJoCo receives values in radians or m/s where it expects force-like actuator commands.
Simulation becomes unstable or physically meaningless.
```

Mitigation:

```text
MuJoCoPhysicsEngine shall require ActuatorCommand4 or explicitly own a mixer adapter.
```

---

### Risk 2: Logger ambiguity

Cause:

```text
Logger stores columns as u0, u1, u2, u3.
```

Consequence:

```text
Later analysis cannot determine whether u means high-level command or rotor thrust.
```

Mitigation:

```text
Logger shall use explicit names: control_phi_c, actuator_T1, etc.
```

---

### Risk 3: ODE and MuJoCo mismatch

Cause:

```text
ODE consumes ControlCommand4, MuJoCo consumes ActuatorCommand4.
```

Consequence:

```text
Simulation behavior differs between engines.
```

Mitigation:

```text
Engine interface shall document accepted command type.
Runtime shall explicitly include or skip the mixer depending on engine type.
```

---

### Risk 4: Units mismatch

Cause:

```text
Angles accidentally passed in degrees.
```

Consequence:

```text
Commands become too large by factor of approximately 57.3.
```

Mitigation:

```text
All public control commands use radians.
Config loader converts degrees to radians if degrees are allowed in config.
```

---

## 17. Required Validation

### 17.1 ControlCommand4 validation

A valid `ControlCommand4` shall satisfy:

```text
shape == (4,)
all values finite
phi_c in configured bounds
theta_c in configured bounds
vz_c in configured bounds
psi_dot_c in configured bounds
```

Suggested validation:

```python
def validate_control_command4(
    u: np.ndarray,
    max_roll: float,
    max_pitch: float,
    max_vert_vel: float,
    max_yaw_rate: float,
) -> None:
    if u.shape != (4,):
        raise ValueError("ControlCommand4 must have shape (4,)")
    if not np.all(np.isfinite(u)):
        raise ValueError("ControlCommand4 contains NaN or Inf")

    phi_c, theta_c, vz_c, psi_dot_c = u

    if abs(phi_c) > max_roll:
        raise ValueError("phi_c exceeds max_roll")
    if abs(theta_c) > max_pitch:
        raise ValueError("theta_c exceeds max_pitch")
    if abs(vz_c) > max_vert_vel:
        raise ValueError("vz_c exceeds max_vert_vel")
    if abs(psi_dot_c) > max_yaw_rate:
        raise ValueError("psi_dot_c exceeds max_yaw_rate")
```

---

### 17.2 ActuatorCommand4 validation

A valid rotor-thrust `ActuatorCommand4` shall satisfy:

```text
shape == (4,)
all values finite
all thrust values >= 0
all thrust values <= max_rotor_thrust
```

Suggested validation:

```python
def validate_actuator_command4(
    T: np.ndarray,
    max_rotor_thrust: float,
) -> None:
    if T.shape != (4,):
        raise ValueError("ActuatorCommand4 must have shape (4,)")
    if not np.all(np.isfinite(T)):
        raise ValueError("ActuatorCommand4 contains NaN or Inf")
    if np.any(T < 0.0):
        raise ValueError("Rotor thrust must be non-negative")
    if np.any(T > max_rotor_thrust):
        raise ValueError("Rotor thrust exceeds max_rotor_thrust")
```

---

## 18. Migration Plan

### Phase 1: Introduce command types

Create:

```text
ControlCommand4
ActuatorCommand4
validate_control_command4()
validate_actuator_command4()
```

Possible location:

```text
ccmpc/types.py
```

or:

```text
simulation/types.py
```

---

### Phase 2: Update controller interface

Change controller output from raw `np.ndarray` to documented `ControlCommand4`.

Before:

```python
x_traj, u_seq = mpc.solve(...)
cmd = u_seq[:, 0]
```

After:

```python
solution = controller.compute_command(...)
cmd: ControlCommand4 = solution.command
```

---

### Phase 3: Update mixer interface

The mixer shall explicitly accept `ControlCommand4` and return `ActuatorCommand4`.

```python
actuator_command = mixer.compute(
    command=control_command,
    state=true_state,
    previous_state=previous_state,
    dt=dt,
)
```

---

### Phase 4: Update engine interface

ODE engine:

```python
ode_engine.step(control_command, dt)
```

MuJoCo engine:

```python
mujoco_engine.step(actuator_command, dt)
```

or, if MuJoCo owns the mixer:

```python
mujoco_engine.step_control_command(control_command, state, dt)
```

The chosen design shall be documented in `05_ENGINE_INTERFACE.md`.

---

### Phase 5: Update logger

Logger shall distinguish:

```text
control_phi_c
control_theta_c
control_vz_c
control_psi_dot_c

actuator_T1
actuator_T2
actuator_T3
actuator_T4
```

---

### Phase 6: Add tests

Required tests:

```text
test_control_command4_shape
test_control_command4_units_and_bounds
test_control_command4_nan_rejection
test_actuator_command4_shape
test_actuator_command4_nonnegative
test_mixer_returns_actuator_command4
test_mujoco_engine_rejects_control_command4_without_mixer
test_logger_command_column_names
```

---

## 19. Acceptance Criteria

This ADR is accepted when:

1. `DATA_MODEL.md` defines `ControlCommand4` and `ActuatorCommand4`.
2. Controller output is documented as `ControlCommand4`.
3. Mixer input is documented as `ControlCommand4`.
4. Mixer output is documented as `ActuatorCommand4`.
5. ODE engine command input is documented.
6. MuJoCo engine command input is documented.
7. Logger records high-level command and actuator command separately.
8. Tests validate shape, units, bounds, and non-ambiguity.
9. No public API uses ambiguous command names without metadata.

---

## 20. Decision Summary

The canonical high-level command for the refactored quadrotor CC-MPC simulation is:

```text
ControlCommand4 = [phi_c, theta_c, vz_c, psi_dot_c]
```

This command is:

```text
4-dimensional
high-level
controller-generated
compatible with the reduced CC-MPC dynamics
not rotor thrust
not torque
not PWM
not motor speed
```

The canonical actuator-level command for rotor-force simulation is:

```text
ActuatorCommand4 = [T1, T2, T3, T4]
```

The required conversion path is:

```text
ControlCommand4 -> LowLevelMixer -> ActuatorCommand4
```

Any future change to the canonical control representation must be documented in a new ADR that supersedes this one.

---

## 21. Related Documents

```text
docs/design/04_DATA_MODEL.md
docs/design/05_ENGINE_INTERFACE.md
docs/design/06_CONTROLLER_INTERFACE.md
docs/design/08_LOGGING_AND_METRICS.md
docs/design/ADR/ADR-003-state-vector-definition.md
docs/theory/02_Quadrotor_Dynamics.md
docs/theory/07_Newton_Euler.md
docs/theory/10_State_Space_Model.md
docs/theory/18_Implementation_Notes.md
```
