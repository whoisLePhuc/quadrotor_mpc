# ADR-003: Canonical State Vector Definition

> Status: Proposed
> Date: 2026-06-28
> Project: Quadrotor CC-MPC Simulation
> Related document: `docs/design/04_DATA_MODEL.md`

---

## 1. Context

The refactored quadrotor CC-MPC simulation requires a single canonical state representation shared by:

```text
Simulation runtime
Physics engines
ODE dynamics
MuJoCo adapter
Controller interface
CC-MPC solver
Sensor / estimator modules
Obstacle avoidance modules
Logger
Visualization
```

The current codebase and theory notes use a reduced quadrotor state model with position, velocity, and Euler attitude.

However, there are multiple possible state representations in a quadrotor simulation:

```text
Reduced 9D state with Euler angles
Full Newton-Euler state with quaternion and angular velocity
MuJoCo qpos/qvel state
Position-only state
Pose-only state
State with body-frame velocity
State with world-frame velocity
```

If the project does not define a canonical state vector, different modules may silently assume different state ordering, coordinate frames, or attitude representations. This can cause severe bugs such as:

```text
wrong state indexing
wrong velocity frame
wrong attitude conversion
wrong MPC linearization
wrong logging interpretation
wrong MuJoCo adapter behavior
wrong controller input
```

Therefore, the project needs a formal architectural decision for the canonical state vector.

---

## 2. Decision

The simulation shall use `State9` as the canonical state representation.

```text
State9 = [x, y, z, vx, vy, vz, roll, pitch, yaw]
```

Mathematical notation:

$$
\mathbf{x}
=

\begin{bmatrix}
x \
y \
z \
v_x \
v_y \
v_z \
\phi \
\theta \
\psi
\end{bmatrix}
\in
\mathbb{R}^9
$$

Equivalent block form:

$$
\mathbf{x}
=

\begin{bmatrix}
\mathbf{p} \
\mathbf{v} \
\boldsymbol{\eta}
\end{bmatrix}
$$

where:

$$
\mathbf{p}
=

\begin{bmatrix}
x \
y \
z
\end{bmatrix}
$$

$$
\mathbf{v}
=

\begin{bmatrix}
v_x \
v_y \
v_z
\end{bmatrix}
$$

$$
\boldsymbol{\eta}
=

\begin{bmatrix}
\phi \
\theta \
\psi
\end{bmatrix}
$$

Alias convention:

```text
roll  = phi   = φ
pitch = theta = θ
yaw   = psi   = ψ
```

---

## 3. State Field Definition

| Index | Field   | Symbol   | Unit | Frame    | Meaning                           |
| ----: | ------- | -------- | ---- | -------- | --------------------------------- |
|     0 | `x`     | $x$      | m    | World    | Position along world X            |
|     1 | `y`     | $y$      | m    | World    | Position along world Y            |
|     2 | `z`     | $z$      | m    | World    | Position along world Z / altitude |
|     3 | `vx`    | $v_x$    | m/s  | World    | Velocity along world X            |
|     4 | `vy`    | $v_y$    | m/s  | World    | Velocity along world Y            |
|     5 | `vz`    | $v_z$    | m/s  | World    | Velocity along world Z            |
|     6 | `roll`  | $\phi$   | rad  | Attitude | Roll Euler angle                  |
|     7 | `pitch` | $\theta$ | rad  | Attitude | Pitch Euler angle                 |
|     8 | `yaw`   | $\psi$   | rad  | Attitude | Yaw Euler angle                   |

---

## 4. Coordinate Frame Decision

All position and velocity components in `State9` shall be expressed in the world frame.

```text
x, y, z      -> world-frame position
vx, vy, vz   -> world-frame velocity
roll, pitch, yaw -> body attitude relative to world frame
```

The world frame shall be Z-up:

```text
X_W: horizontal reference axis
Y_W: horizontal lateral axis
Z_W: upward
```

The body frame shall use:

```text
X_B: forward
Y_B: left
Z_B: upward
```

The canonical attitude convention shall be ZYX Euler angles:

$$
\mathbf{R}_B^W
=

\mathbf{R}_Z(\psi)
\mathbf{R}_Y(\theta)
\mathbf{R}_X(\phi)
$$

---

## 5. Control Compatibility

This ADR only defines the state vector, but the chosen `State9` is designed to be compatible with the controller-level command:

```text
ControlCommand4 = [phi_c, theta_c, vz_c, psi_dot_c]
```

Mathematically:

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

This pairing gives the canonical dynamics interface:

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

The control vector is not part of this ADR, but the state vector is intentionally chosen to match the simplified Bebop-style CC-MPC model.

---

## 6. Why Not Use Full Newton-Euler State?

A full Newton-Euler quadrotor state may include:

```text
position
linear velocity
quaternion
body angular velocity
```

Example:

$$
\mathbf{x}_{NE}
=

\begin{bmatrix}
\mathbf{p} \
\dot{\mathbf{p}} \
\mathbf{q} \
\boldsymbol{\omega}
\end{bmatrix}
$$

This representation is more physically complete, but it is not chosen as the canonical controller state for this project.

Reasons:

1. It increases the state dimension.
2. It requires angular velocity states.
3. It requires quaternion handling inside the controller.
4. It makes linearization and MPC problem construction more complex.
5. It does not match the simplified first-order attitude model used by the CC-MPC theory baseline.
6. It is less convenient for the current high-level command interface.

The full Newton-Euler representation may still be used inside high-fidelity physics engines or future research extensions, but it shall not replace `State9` unless a new ADR supersedes this one.

---

## 7. Why Not Use MuJoCo qpos/qvel as Canonical State?

MuJoCo internally represents a free body using generalized coordinates and velocities.

Typical representation:

```text
qpos = [x, y, z, qw, qx, qy, qz]
qvel = [vx, vy, vz, wx, wy, wz]
```

This representation is engine-specific.

It is not chosen as canonical because:

1. It uses quaternion attitude, while the MPC model uses Euler angles.
2. It includes angular velocity, which is not part of the reduced CC-MPC state.
3. It is tied to MuJoCo implementation details.
4. It would force non-MuJoCo modules to depend on MuJoCo-specific data layout.
5. It would make ODE and MuJoCo engines harder to swap.

MuJoCo state shall be treated as engine-internal data.

The MuJoCo adapter shall provide explicit conversion:

```text
State9 -> MuJoCo qpos/qvel
MuJoCo qpos/qvel -> State9
```

---

## 8. Why Use Euler Angles Instead of Quaternion in State9?

Euler angles are selected for the canonical controller state because:

1. The simplified CC-MPC dynamics are written directly in terms of roll, pitch, and yaw.
2. The control commands are commanded roll angle, commanded pitch angle, vertical velocity command, and yaw-rate command.
3. Linearization is simpler in Euler coordinates for the reduced model.
4. The operating region is near-hover or moderate-attitude flight, where Euler singularities are not expected to dominate.
5. Euler angles are easier to interpret in logs and debugging.

Quaternion shall still be allowed in:

```text
MuJoCo engine internals
VIO internal representation
orientation conversion utilities
interpolation utilities
```

Quaternion shall not be passed directly to the CC-MPC controller.

---

## 9. Consequences

### 9.1 Positive Consequences

Using `State9` as canonical state gives:

1. A single state ordering across the codebase.
2. Clear compatibility between dynamics, controller, logger, and scenario config.
3. Smaller MPC decision vector than a full Newton-Euler state.
4. Simpler linearization.
5. Easier debugging and plotting.
6. Easier CSV logging.
7. Easier interface design for ODE and MuJoCo engines.
8. Clear adapter boundary for MuJoCo quaternion state.

---

### 9.2 Negative Consequences

This decision also has limitations:

1. Angular velocity is not represented in the canonical state.
2. High-agility maneuvers may not be accurately modeled.
3. Quaternion singularity avoidance is not available in the controller state.
4. MuJoCo must convert between quaternion and Euler at every state exchange.
5. The reduced model may mismatch full rigid-body dynamics at aggressive attitudes.
6. Euler angles have a gimbal-lock singularity near pitch = ±90 degrees.

These limitations are acceptable for the current CC-MPC simulation target, which focuses on reduced-order planning and obstacle avoidance rather than full aerobatic control.

---

## 10. Alternatives Considered

### 10.1 Alternative A: Full Newton-Euler state

```text
[p, v, quaternion, angular_velocity]
```

Decision: Rejected as canonical state.

Reason:

```text
Too complex for the current CC-MPC planner and not aligned with the simplified first-order model.
```

May be used internally by physics engines.

---

### 10.2 Alternative B: MuJoCo qpos/qvel state

```text
qpos + qvel
```

Decision: Rejected as canonical state.

Reason:

```text
Engine-specific representation. Not suitable as a project-wide data contract.
```

MuJoCo shall be wrapped by an adapter.

---

### 10.3 Alternative C: 12D Euler + angular velocity state

```text
[x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]
```

Decision: Rejected for initial refactor.

Reason:

```text
More physically expressive but requires changing the controller model, dynamics, covariance propagation, and QP dimensions.
```

This may be reconsidered in a future high-fidelity dynamics milestone.

---

### 10.4 Alternative D: Position-only state

```text
[x, y, z]
```

Decision: Rejected.

Reason:

```text
Insufficient for MPC prediction because the dynamics require velocity and attitude.
```

---

### 10.5 Alternative E: Body-frame velocity state

```text
[x, y, z, vx_body, vy_body, vz_body, roll, pitch, yaw]
```

Decision: Rejected.

Reason:

```text
The current simplified dynamics and state-space formulation use world-frame velocity.
```

Body-frame velocity may be computed as derived data when needed.

---

## 11. Required Implementation Rules

### Rule 1: Canonical state ordering

Every module that accepts `State9` shall use the following order:

```text
[x, y, z, vx, vy, vz, roll, pitch, yaw]
```

No module may silently use another ordering.

---

### Rule 2: Explicit adapters for alternative representations

Any module using another representation shall define an explicit adapter.

Examples:

```text
MuJoCoStateAdapter
QuaternionAdapter
BodyFrameVelocityAdapter
```

---

### Rule 3: No direct MuJoCo state in controller

The controller shall not receive:

```text
qpos
qvel
quaternion
MuJoCo data object
```

The controller shall receive only canonical controller input based on `State9`.

---

### Rule 4: Logging shall use canonical field names

Logger columns shall use the following names:

```text
pos_x
pos_y
pos_z
vel_x
vel_y
vel_z
roll
pitch
yaw
```

These names shall map directly to `State9`.

---

### Rule 5: Scenario config shall use State9 ordering

Scenario YAML shall define initial state using the canonical ordering:

```yaml
start: [x, y, z, vx, vy, vz, roll, pitch, yaw]
```

---

### Rule 6: Validation at module boundaries

Every public module boundary shall validate:

```text
shape == (9,)
all values are finite
angles are in radians
position and velocity are world-frame values
```

Recommended validation function:

```python
def validate_state9(x: np.ndarray) -> None:
    if x.shape != (9,):
        raise ValueError("State9 must have shape (9,)")
    if not np.all(np.isfinite(x)):
        raise ValueError("State9 contains NaN or Inf")
```

---

## 12. Required Data Type

The refactored codebase should introduce a shared type for `State9`.

Recommended location:

```text
simulation/state.py
```

or:

```text
ccmpc/types.py
```

Recommended dataclass:

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class State9:
    data: np.ndarray

    def __post_init__(self):
        if self.data.shape != (9,):
            raise ValueError("State9 must have shape (9,)")
        if not np.all(np.isfinite(self.data)):
            raise ValueError("State9 contains NaN or Inf")

    @property
    def position(self) -> np.ndarray:
        return self.data[0:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.data[3:6]

    @property
    def attitude(self) -> np.ndarray:
        return self.data[6:9]

    @property
    def roll(self) -> float:
        return float(self.data[6])

    @property
    def pitch(self) -> float:
        return float(self.data[7])

    @property
    def yaw(self) -> float:
        return float(self.data[8])
```

The implementation may use raw NumPy arrays internally for performance, but public APIs should document and validate that the array follows `State9`.

---

## 13. Required Adapter Interfaces

### 13.1 MuJoCo adapter

The MuJoCo adapter shall implement:

```python
def state9_to_mujoco(state: State9) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert State9 to MuJoCo qpos/qvel.
    """

def mujoco_to_state9(qpos: np.ndarray, qvel: np.ndarray) -> State9:
    """
    Convert MuJoCo qpos/qvel to canonical State9.
    """
```

---

### 13.2 Quaternion adapter

Quaternion conversion shall use the order:

```text
[w, x, y, z]
```

Required functions:

```python
def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert ZYX Euler angles to quaternion [w, x, y, z].
    """

def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """
    Convert quaternion [w, x, y, z] to ZYX Euler angles.
    """
```

---

### 13.3 Body-frame velocity adapter

If body-frame velocity is needed, it shall be derived explicitly.

```python
def world_velocity_to_body_velocity(
    velocity_world: np.ndarray,
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    """
    Convert world-frame velocity to body-frame velocity.
    """
```

Body-frame velocity shall not be stored inside canonical `State9`.

---

## 14. Migration Plan

### Phase 1: Add state type and validation

1. Create `State9` type.
2. Add `validate_state9()`.
3. Add tests for valid and invalid state arrays.

---

### Phase 2: Update config loading

1. Parse `start` from scenario YAML as `State9`.
2. Reject scenarios with incorrect state length.
3. Validate finite values.
4. Document that `start` uses canonical ordering.

---

### Phase 3: Update physics engines

1. ODE engine shall consume and return `State9`.
2. MuJoCo engine shall convert between `State9` and `qpos/qvel`.
3. MuJoCo internals shall not leak into controller modules.

---

### Phase 4: Update controller interface

1. Controller shall receive `estimated_state: State9`.
2. Controller shall return predicted trajectory based on `State9`.
3. Controller shall not access engine-internal state.

---

### Phase 5: Update logging

1. Logger shall write state columns using canonical field names.
2. Logger shall optionally store `true_state` and `estimated_state` separately.
3. Logger shall not store ambiguous `state` without a prefix.

---

### Phase 6: Add regression tests

Required tests:

```text
test_state9_shape
test_state9_field_order
test_state9_nan_rejection
test_mujoco_state_roundtrip
test_scenario_start_state_order
test_logger_state_column_order
test_controller_accepts_state9
```

---

## 15. Compatibility With Current Source

The current implementation already uses a 9D state in several places.

Examples:

```text
config/simulation.yaml:
start: [x, y, z, vx, vy, vz, phi, theta, psi]

sim_demo_nosim.py:
self.state = np.array(start, dtype=np.float64)

logger:
pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw

dynamics.py:
State = [x, y, z, vx, vy, vz, phi, theta, psi]
```

This ADR makes that implicit convention explicit and mandatory.

---

## 16. Risk Analysis

### Risk 1: Silent state-order mismatch

Cause:

```text
A module assumes a different index ordering.
```

Mitigation:

```text
Use State9 dataclass, validation, tests, and clear field names.
```

---

### Risk 2: Quaternion/Euler mismatch

Cause:

```text
MuJoCo uses quaternion, while MPC uses Euler angles.
```

Mitigation:

```text
Keep quaternion inside MuJoCo adapter and provide tested conversion functions.
```

---

### Risk 3: Velocity frame mismatch

Cause:

```text
One module assumes world-frame velocity, another assumes body-frame velocity.
```

Mitigation:

```text
Canonical State9 velocity is always world-frame velocity.
Body-frame velocity must be explicitly named and derived.
```

---

### Risk 4: Future model upgrade breaks API

Cause:

```text
Future high-fidelity model may require angular velocity or rotor states.
```

Mitigation:

```text
Introduce a new ADR before changing canonical state.
Use engine-internal state extensions instead of changing State9 silently.
```

---

## 17. Acceptance Criteria

This ADR is accepted when:

1. `DATA_MODEL.md` defines `State9` consistently.
2. Scenario config uses `start: [x, y, z, vx, vy, vz, roll, pitch, yaw]`.
3. ODE engine accepts and returns `State9`.
4. MuJoCo engine exposes `State9` at public boundary.
5. Controller interface accepts `estimated_state: State9`.
6. Logger records canonical state columns.
7. Unit tests verify state ordering and adapter round-trip.
8. No public API passes MuJoCo `qpos/qvel` directly to the controller.

---

## 18. Decision Summary

The canonical state vector for the refactored quadrotor CC-MPC simulation is:

```text
State9 = [x, y, z, vx, vy, vz, roll, pitch, yaw]
```

This state is:

```text
9-dimensional
world-frame for position and velocity
Euler ZYX for attitude
compatible with the simplified CC-MPC model
compatible with ODE dynamics
compatible with MuJoCo through an adapter
smaller and simpler than full Newton-Euler state
```

Any future change to the canonical state representation must be documented in a new ADR that supersedes this one.

---

## 19. Related Documents

```text
docs/design/04_DATA_MODEL.md
docs/design/05_ENGINE_INTERFACE.md
docs/design/06_CONTROLLER_INTERFACE.md
docs/design/ADR/ADR-004-control-command-definition.md
docs/theory/02_Quadrotor_Dynamics.md
docs/theory/03_Coordinate_Frames.md
docs/theory/05_Euler_Angles.md
docs/theory/06_Quaternion.md
docs/theory/07_Newton_Euler.md
docs/theory/10_State_Space_Model.md
```
