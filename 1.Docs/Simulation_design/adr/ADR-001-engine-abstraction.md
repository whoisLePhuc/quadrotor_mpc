# ADR-001: Physics Engine Abstraction

> Status: Proposed
> Date: 2026-06-28
> Project: Quadrotor CC-MPC Simulation
> Related documents:
>
> * `docs/design/04_DATA_MODEL.md`
> * `docs/design/05_ENGINE_INTERFACE.md`
> * `docs/design/ADR/ADR-003-state-vector-definition.md`
> * `docs/design/ADR/ADR-004-control-command-definition.md`

---

## 1. Context

The quadrotor CC-MPC simulation needs to support multiple physics backends.

The initial refactor target includes at least:

```text
ODEPhysicsEngine
MuJoCoPhysicsEngine
```

These engines have different internal assumptions:

```text
ODEPhysicsEngine:
- Uses reduced-order quadrotor dynamics
- Uses State9 directly
- Uses ControlCommand4 directly
- Does not model individual rotors

MuJoCoPhysicsEngine:
- Uses MuJoCo qpos/qvel internally
- Uses quaternion internally
- May model rotor forces
- Should receive ActuatorCommand4 after mixing
```

Without a common abstraction, the simulation runtime becomes tightly coupled to engine-specific details.

Common failure modes include:

```text
controller code directly reads MuJoCo qpos/qvel
runtime contains many if engine == "mujoco" branches
MPC command is passed directly to MuJoCo actuator ctrl
logger uses different state naming for different engines
ODE and MuJoCo produce incompatible state formats
MuJoCo quaternion leaks into controller API
physics stepping, rendering, logging, and control logic are mixed in one script
```

Therefore, the project needs a formal architecture decision: physics engines must be abstracted behind a common interface.

---

## 2. Decision

The refactored simulation shall define a common `PhysicsEngine` abstraction.

All physics engines shall expose the same public state interface:

```text
State9
```

All physics engines shall implement a common lifecycle:

```text
reset()
step()
get_state()
get_time()
get_metadata()
close()
```

The controller shall not depend on a concrete physics engine.

The runtime shall depend on the abstract engine interface.

The physics engine shall not own:

```text
controller logic
MPC solve logic
logging format
visualization logic
scenario parsing
experiment orchestration
```

The physics engine shall own only:

```text
physics state
engine time
physics stepping
engine-specific state adapters
engine-specific command adapters
```

---

## 3. Intent

The goal of this decision is not to make ODE and MuJoCo physically identical.

The goal is to make them software-compatible at the simulation boundary.

The shared boundary is:

```text
State9 out
engine-specific command in
StepResult out
```

The runtime should be able to switch engines through configuration:

```yaml
simulation:
  engine: ode
```

or:

```yaml
simulation:
  engine: mujoco
```

without changing controller code.

---

## 4. Canonical Data Contracts

This ADR depends on the following canonical data contracts.

### 4.1 State

```text
State9 = [x, y, z, vx, vy, vz, roll, pitch, yaw]
```

Mathematical form:

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

All public engine state output shall be `State9`.

---

### 4.2 High-level control command

```text
ControlCommand4 = [phi_c, theta_c, vz_c, psi_dot_c]
```

Mathematical form:

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

This command is compatible with the reduced ODE dynamics.

---

### 4.3 Actuator command

For rotor-force simulation:

```text
ActuatorCommand4 = [T1, T2, T3, T4]
```

Mathematical form:

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

This command is compatible with actuator-level physics engines such as MuJoCo rotor-force simulation.

---

## 5. Engine Abstraction

The base engine interface shall be:

```python
class PhysicsEngine:
    def reset(self, initial_state: State9) -> None:
        ...

    def step(self, command, dt: float) -> StepResult:
        ...

    def get_state(self) -> State9:
        ...

    def get_time(self) -> float:
        ...

    def get_metadata(self) -> EngineMetadata:
        ...

    def close(self) -> None:
        ...
```

The abstract engine interface shall hide engine-specific implementation details.

Public runtime code shall not directly access:

```text
MuJoCo model
MuJoCo data
MuJoCo qpos
MuJoCo qvel
MuJoCo ctrl
quaternion state
ODE internal arrays
```

unless it is inside an adapter or engine implementation.

---

## 6. Engine Metadata

Each engine shall declare its capabilities through `EngineMetadata`.

```python
@dataclass(frozen=True)
class EngineMetadata:
    engine_type: EngineType
    command_type: EngineCommandType
    state_type: str
    supports_reset: bool
    supports_rendering: bool
    supports_linearization: bool
    supports_internal_substeps: bool
    notes: str = ""
```

Engine type:

```python
class EngineType(str, Enum):
    ODE = "ode"
    MUJOCO = "mujoco"
```

Command type:

```python
class EngineCommandType(str, Enum):
    CONTROL_COMMAND_4 = "ControlCommand4"
    ACTUATOR_COMMAND_4 = "ActuatorCommand4"
```

Example metadata for ODE:

```python
EngineMetadata(
    engine_type=EngineType.ODE,
    command_type=EngineCommandType.CONTROL_COMMAND_4,
    state_type="State9",
    supports_reset=True,
    supports_rendering=False,
    supports_linearization=True,
    supports_internal_substeps=False,
)
```

Example metadata for MuJoCo:

```python
EngineMetadata(
    engine_type=EngineType.MUJOCO,
    command_type=EngineCommandType.ACTUATOR_COMMAND_4,
    state_type="State9",
    supports_reset=True,
    supports_rendering=True,
    supports_linearization=True,
    supports_internal_substeps=True,
)
```

---

## 7. Runtime Dispatch

The runtime shall use engine metadata to determine how to apply commands.

Pseudocode:

```python
true_state = engine.get_state()

observation = estimator.estimate(true_state)
control_command = controller.compute_command(observation)

metadata = engine.get_metadata()

if metadata.command_type == EngineCommandType.CONTROL_COMMAND_4:
    applied_command = control_command

elif metadata.command_type == EngineCommandType.ACTUATOR_COMMAND_4:
    applied_command = mixer.compute(
        command=control_command,
        state=true_state,
        previous_state=previous_true_state,
        dt=sim_dt,
    )

else:
    raise UnsupportedCommandType(metadata.command_type)

step_result = engine.step(applied_command, sim_dt)
```

This keeps engine-specific command handling in the runtime orchestration layer, not inside the controller.

---

## 8. ODE Engine Policy

`ODEPhysicsEngine` shall consume `ControlCommand4` directly.

Reason:

```text
The reduced-order ODE model is defined using commanded roll, commanded pitch, commanded vertical velocity, and commanded yaw rate.
```

The ODE stepping equation is:

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

```text
x_k -> State9
u_k -> ControlCommand4
```

The ODE engine shall not require a mixer.

Allowed flow:

```text
Controller
    -> ControlCommand4
    -> ODEPhysicsEngine
    -> State9
```

---

## 9. MuJoCo Engine Policy

`MuJoCoPhysicsEngine` shall expose `State9` publicly but may use `qpos/qvel` internally.

MuJoCo state conversion:

```text
State9 -> qpos/qvel
qpos/qvel -> State9
```

MuJoCo orientation conversion:

```text
Euler ZYX -> quaternion [w, x, y, z]
quaternion [w, x, y, z] -> Euler ZYX
```

For rotor-force simulation, MuJoCo shall consume `ActuatorCommand4`.

Allowed flow:

```text
Controller
    -> ControlCommand4
    -> LowLevelMixer
    -> ActuatorCommand4
    -> MuJoCoPhysicsEngine
    -> State9
```

Disallowed flow:

```text
Controller
    -> ControlCommand4
    -> MuJoCo ctrl
```

unless MuJoCo engine explicitly owns and documents an internal mixer adapter.

Preferred design:

```text
SimulationRuntime owns the mixer.
MuJoCoPhysicsEngine receives ActuatorCommand4.
```

---

## 10. Engine Ownership Boundaries

### 10.1 Engine owns physics state

The engine shall own:

```text
current true_state
current engine time
internal physics buffers
internal engine-specific state
```

For ODE:

```text
internal State9 array
```

For MuJoCo:

```text
MjModel
MjData
qpos
qvel
ctrl
```

---

### 10.2 Engine does not own controller

Invalid:

```text
MuJoCoPhysicsEngine calls CCMPC.solve()
```

Valid:

```text
SimulationRuntime calls Controller.compute_command()
SimulationRuntime calls PhysicsEngine.step()
```

Reason:

```text
Physics engine simulates dynamics.
Controller computes commands.
Runtime orchestrates interaction.
```

---

### 10.3 Engine does not own logger

Invalid:

```text
PhysicsEngine.step() writes CSV directly
```

Valid:

```text
StepResult -> Logger.record()
```

Reason:

```text
Logging is an experiment-output concern, not a physics concern.
```

---

### 10.4 Engine does not own renderer

Invalid:

```text
PhysicsEngine requires viewer to run
```

Valid:

```text
Renderer reads State9 and draws it
Engine can run headless
```

Reason:

```text
Physics simulation should be possible in batch/headless mode.
```

---

## 11. StepResult Contract

Each engine step shall return `StepResult`.

```python
@dataclass(frozen=True)
class StepResult:
    time: float
    dt: float
    true_state: State9
    applied_command: object
    success: bool
    status: str
    engine_info: dict
```

Required fields:

| Field             | Meaning                            |
| ----------------- | ---------------------------------- |
| `time`            | Engine time after step             |
| `dt`              | Requested or applied timestep      |
| `true_state`      | Canonical state after step         |
| `applied_command` | Command actually applied to engine |
| `success`         | Whether step succeeded             |
| `status`          | Human-readable status              |
| `engine_info`     | Engine-specific diagnostics        |

The `engine_info` field may contain:

```text
MuJoCo substep count
MuJoCo timestep
numerical warning
actuator saturation
external wrench
contact information
adapter diagnostics
```

---

## 12. Adapter Policy

Any engine that uses non-canonical data internally shall provide adapters.

Required MuJoCo adapters:

```python
def state9_to_mujoco(state: State9) -> tuple[np.ndarray, np.ndarray]:
    ...

def mujoco_to_state9(qpos: np.ndarray, qvel: np.ndarray) -> State9:
    ...

def actuator_command4_to_mujoco_ctrl(command: ActuatorCommand4) -> np.ndarray:
    ...
```

Required quaternion utilities:

```python
def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    ...

def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    ...
```

Quaternion storage order shall be:

```text
[w, x, y, z]
```

No adapter may silently change:

```text
state ordering
axis convention
unit convention
rotor ordering
command semantics
```

Any such change must be documented.

---

## 13. Configuration Policy

Engine selection shall be configuration-driven.

Example:

```yaml
simulation:
  engine: ode
```

or:

```yaml
simulation:
  engine: mujoco
```

Engine-specific configuration shall be nested under engine-specific keys.

Example:

```yaml
simulation:
  engine: mujoco

mujoco:
  model_path: models/quadrotor.xml
  timestep_policy: reject
  quad_body_name: quadrotor
```

The runtime shall create engines through a factory.

```python
def create_physics_engine(config) -> PhysicsEngine:
    if config.simulation.engine == "ode":
        return ODEPhysicsEngine(...)

    if config.simulation.engine == "mujoco":
        return MuJoCoPhysicsEngine(...)

    raise ValueError("Unknown physics engine")
```

No script shall manually construct a different engine path outside the factory unless it is a test.

---

## 14. Alternatives Considered

### 14.1 Alternative A: Keep separate scripts for each engine

Example:

```text
sim_demo_nosim.py for ODE
sim_demo_mujoco.py for MuJoCo
```

Decision: Rejected as final architecture.

Reason:

```text
It duplicates runtime logic and makes controller, logger, scenario loading, and metrics inconsistent across engines.
```

This approach may remain temporarily for demos, but it shall not be the refactored architecture.

---

### 14.2 Alternative B: Let controller directly handle engine differences

Example:

```text
Controller checks whether engine is ODE or MuJoCo
Controller outputs different command types depending on engine
```

Decision: Rejected.

Reason:

```text
Controller should compute high-level commands, not manage physics backend details.
```

Engine-specific handling belongs in runtime dispatch and adapters.

---

### 14.3 Alternative C: Use MuJoCo state as the universal state

Example:

```text
qpos/qvel becomes canonical state
```

Decision: Rejected.

Reason:

```text
MuJoCo qpos/qvel is engine-specific and uses quaternion internally.
The controller model uses State9 with Euler attitude.
```

MuJoCo data shall remain internal to `MuJoCoPhysicsEngine`.

---

### 14.4 Alternative D: Force all engines to consume ControlCommand4

Decision: Rejected.

Reason:

```text
This fits ODE but not actuator-level MuJoCo rotor-force simulation.
```

MuJoCo should consume `ActuatorCommand4` unless it explicitly owns a mixer adapter.

---

### 14.5 Alternative E: Force all engines to consume ActuatorCommand4

Decision: Rejected.

Reason:

```text
This would require adding a mixer even for reduced-order ODE simulation, although the ODE model is already defined in terms of ControlCommand4.
```

ODE may consume `ControlCommand4` directly.

---

## 15. Consequences

### 15.1 Positive consequences

This decision provides:

1. Replaceable physics engines.
2. Cleaner runtime orchestration.
3. Clear state boundary through `State9`.
4. Clear command boundary through engine metadata.
5. Reduced controller-engine coupling.
6. Easier testing of ODE and MuJoCo using the same scenarios.
7. Easier logging across engines.
8. Easier future addition of new engines.
9. Less risk of MuJoCo internals leaking into MPC.
10. Better separation of physics, control, logging, and rendering.

---

### 15.2 Negative consequences

This decision introduces:

1. More upfront interface code.
2. More adapter code.
3. Runtime dispatch complexity.
4. Need to validate command type per engine.
5. Need to test state conversion carefully.
6. Possible small conversion error between quaternion and Euler.

These costs are acceptable because they prevent larger architecture-level bugs.

---

## 16. Risks and Mitigations

### Risk 1: Engine abstraction hides physical differences

Cause:

```text
ODE and MuJoCo expose the same interface but use different dynamics.
```

Mitigation:

```text
Document that interface equivalence is not physical equivalence.
Use validation experiments to compare engine behavior.
```

---

### Risk 2: Wrong command type sent to engine

Cause:

```text
Runtime sends ControlCommand4 to MuJoCo rotor actuator.
```

Mitigation:

```text
EngineMetadata.command_type must be checked.
Engine.step() must validate command type.
Tests must cover wrong-command rejection.
```

---

### Risk 3: Adapter conversion bug

Cause:

```text
State9 <-> qpos/qvel conversion is implemented incorrectly.
```

Mitigation:

```text
Add round-trip tests.
Validate quaternion norm.
Validate Euler angle convention.
Compare reset state with get_state() after reset.
```

---

### Risk 4: Logger records inconsistent data across engines

Cause:

```text
ODE logs ControlCommand4.
MuJoCo logs ActuatorCommand4 only.
```

Mitigation:

```text
Logger shall record both control_command and applied_command when available.
Column names shall be explicit.
```

---

### Risk 5: Runtime becomes too complex

Cause:

```text
Runtime dispatch handles engine, mixer, estimator, logger, renderer.
```

Mitigation:

```text
Keep runtime orchestration explicit.
Move engine-specific code into factory and metadata.
Do not spread engine-specific if-statements across the codebase.
```

---

## 17. Implementation Rules

### Rule 1: Controller is engine-agnostic

The controller shall not know whether the physics engine is ODE or MuJoCo.

---

### Rule 2: Engine exposes State9

Every engine shall expose `State9` at public boundaries.

---

### Rule 3: Engine declares command type

Every engine shall declare whether it expects:

```text
ControlCommand4
```

or:

```text
ActuatorCommand4
```

---

### Rule 4: Runtime performs command dispatch

The runtime shall decide whether a mixer is needed based on engine metadata.

---

### Rule 5: Engine internals stay internal

The following shall not appear in controller interfaces:

```text
qpos
qvel
quaternion
MjData
MjModel
MuJoCo ctrl
rotor site id
```

---

### Rule 6: Engine step is side-effect limited

`step()` shall update physics state and engine time.

It shall not:

```text
solve MPC
write CSV
update plots
parse scenario config
create controller
```

---

## 18. Migration Plan

### Phase 1: Create engine interface

Add:

```text
simulation/engines/base.py
simulation/engines/metadata.py
```

Define:

```text
PhysicsEngine
EngineType
EngineCommandType
EngineMetadata
StepResult
```

---

### Phase 2: Wrap ODE engine

Create:

```text
simulation/engines/ode_engine.py
```

Move reduced ODE stepping behind:

```python
ODEPhysicsEngine.step(ControlCommand4, dt)
```

---

### Phase 3: Wrap MuJoCo engine

Create:

```text
simulation/engines/mujoco_engine.py
```

Move MuJoCo stepping behind:

```python
MuJoCoPhysicsEngine.step(ActuatorCommand4, dt)
```

Add adapters:

```text
simulation/engines/adapters/mujoco_state_adapter.py
simulation/engines/adapters/mujoco_actuator_adapter.py
```

---

### Phase 4: Create engine factory

Add:

```text
simulation/engines/factory.py
```

The simulation app shall call:

```python
engine = create_physics_engine(config)
```

---

### Phase 5: Update runtime loop

Runtime shall use:

```python
metadata = engine.get_metadata()
```

to determine whether to call the mixer.

---

### Phase 6: Update tests

Add tests for:

```text
engine metadata
ODE reset/step
MuJoCo reset/step
adapter round-trip
wrong command type rejection
runtime dispatch
```

---

## 19. Required Tests

```text
test_engine_factory_creates_ode_engine
test_engine_factory_creates_mujoco_engine

test_ode_engine_metadata
test_ode_engine_reset_returns_state9
test_ode_engine_step_accepts_control_command4
test_ode_engine_rejects_actuator_command4

test_mujoco_engine_metadata
test_mujoco_engine_reset_returns_state9
test_mujoco_engine_step_accepts_actuator_command4
test_mujoco_engine_rejects_control_command4_without_mixer

test_state9_to_mujoco_roundtrip
test_mujoco_to_state9_roundtrip
test_euler_quaternion_roundtrip

test_runtime_dispatch_skips_mixer_for_ode
test_runtime_dispatch_uses_mixer_for_mujoco
```

---

## 20. Acceptance Criteria

This ADR is accepted when:

1. A `PhysicsEngine` interface exists.
2. ODE and MuJoCo engines implement the interface.
3. Every engine exposes `State9`.
4. Every engine declares its expected command type.
5. Runtime dispatch uses engine metadata.
6. Controller code does not depend on engine type.
7. Logger receives `StepResult`, not engine internals.
8. MuJoCo `qpos/qvel` are hidden behind adapters.
9. Unit tests verify ODE/MuJoCo interface behavior.
10. Demo scripts are refactored to use the engine factory.

---

## 21. Decision Summary

The simulation shall abstract physics engines behind a common `PhysicsEngine` interface.

Public engine state shall be:

```text
State9
```

ODE engine shall consume:

```text
ControlCommand4
```

MuJoCo rotor-force engine shall consume:

```text
ActuatorCommand4
```

Engine-specific data shall remain internal.

The runtime shall use engine metadata to decide whether a mixer is required.

The controller shall be engine-agnostic.

This decision enables a clean simulation architecture where ODE and MuJoCo are interchangeable backends at the software boundary, even though their physical fidelity differs.

---

## 22. Related Documents

```text
docs/design/04_DATA_MODEL.md
docs/design/05_ENGINE_INTERFACE.md
docs/design/06_CONTROLLER_INTERFACE.md
docs/design/07_SCENARIO_CONFIG.md
docs/design/08_LOGGING_AND_METRICS.md
docs/design/ADR/ADR-003-state-vector-definition.md
docs/design/ADR/ADR-004-control-command-definition.md
docs/theory/02_Quadrotor_Dynamics.md
docs/theory/06_Quaternion.md
docs/theory/09_Discretization.md
docs/theory/10_State_Space_Model.md
docs/theory/18_Implementation_Notes.md
```
