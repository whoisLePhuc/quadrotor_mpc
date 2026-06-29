# 01_REQUIREMENTS.md

> Status: Draft
> Scope: Requirements for refactored simulation
> Project: Quadrotor CC-MPC Simulation
> Related documents:
>
> * `00_SIMULATION_VISION.md`
> * `02_ARCHITECTURE.md`
> * `03_RUNTIME_FLOW.md`
> * `04_DATA_MODEL.md`
> * `05_ENGINE_INTERFACE.md`
> * `06_CONTROLLER_INTERFACE.md`
> * `07_SCENARIO_CONFIG.md`
> * `08_LOGGING_AND_METRICS.md`
> * `09_VALIDATION_PLAN.md`
> * `10_KNOWN_LIMITATIONS.md`
> * `11_REFACTOR_PLAN.md`
> * `ADR/ADR-001-engine-abstraction.md`
> * `ADR/ADR-002-single-thread-vs-mpc-thread.md`
> * `ADR/ADR-003-state-vector-definition.md`
> * `ADR/ADR-004-control-command-definition.md`

---

## 1. Purpose

This document defines the functional and non-functional requirements for the refactored quadrotor CC-MPC simulation.

The goal is to convert the high-level vision in `00_SIMULATION_VISION.md` into concrete requirements that can be implemented, tested, and validated.

This document answers:

```text id="2pj3u7"
What shall the system do?
What constraints shall the system obey?
What behavior is required?
What behavior is optional?
What is out of scope?
How will requirements be validated?
```

---

## 2. Requirement Language

This document uses the following requirement language:

| Term         | Meaning                 |
| ------------ | ----------------------- |
| `shall`      | mandatory requirement   |
| `should`     | recommended requirement |
| `may`        | optional feature        |
| `shall not`  | prohibited behavior     |
| `should not` | discouraged behavior    |

Requirement IDs use prefixes:

| Prefix | Category                   |
| ------ | -------------------------- |
| `FR`   | Functional requirement     |
| `NFR`  | Non-functional requirement |
| `CON`  | Constraint                 |
| `VAL`  | Validation requirement     |
| `DOC`  | Documentation requirement  |
| `OOS`  | Out-of-scope item          |

---

## 3. System Scope

The system shall provide a research-grade simulation framework for quadrotor CC-MPC obstacle avoidance.

The system shall include:

```text id="p6u0aa"
canonical data model
scenario configuration
deterministic runtime
physics engine abstraction
controller abstraction
ODE simulation backend
MuJoCo simulation backend
uncertainty propagation
obstacle prediction
low-level mixer for actuator-level engines
structured logging
metrics computation
validation tests
```

The system shall not be treated as a real flight stack.

---

## 4. Stakeholders

Primary stakeholders:

| Stakeholder        | Need                                                   |
| ------------------ | ------------------------------------------------------ |
| Researcher         | run reproducible CC-MPC experiments                    |
| Developer          | refactor code without breaking math/data contracts     |
| Reviewer           | inspect logs, configs, and validation results          |
| Future contributor | add controllers, engines, scenarios, or loggers safely |
| Graduate student   | understand architecture and reproduce experiments      |

---

## 5. Operating Assumptions

The initial refactor assumes:

```text id="e692be"
Python-based simulation
NumPy-based numerical arrays
CVXPY-based QP formulation for CC-MPC
ODE backend for fast deterministic simulation
MuJoCo backend for actuator-level rigid-body simulation
structured YAML config files
CSV/JSON-based experiment outputs
deterministic single-thread runtime as reference mode
```

The system may later support other solvers, engines, and output formats.

---

## 6. Priority Levels

Requirement priority:

| Priority | Meaning                                            |
| -------- | -------------------------------------------------- |
| `P0`     | required for first stable refactor                 |
| `P1`     | required for research-grade usability              |
| `P2`     | useful extension after core architecture is stable |
| `P3`     | future work                                        |

---

# 7. Functional Requirements

---

## FR-001: Canonical State Model

**Priority:** P0
**Source:** `04_DATA_MODEL.md`, `ADR-003-state-vector-definition.md`

The system shall use the canonical state vector:

```text id="wdqyyz"
State9 = [x, y, z, vx, vy, vz, roll, pitch, yaw]
```

The system shall interpret fields as:

| Field              | Unit | Frame     |
| ------------------ | ---- | --------- |
| `x, y, z`          | m    | world     |
| `vx, vy, vz`       | m/s  | world     |
| `roll, pitch, yaw` | rad  | Euler ZYX |

The system shall not use another public state ordering without an explicit adapter.

**Acceptance criteria:**

```text id="t7e7x9"
State9 type exists
State9 validation tests pass
all public engine states are State9-compatible
all controller inputs use State9-compatible state
```

---

## FR-002: Canonical High-Level Control Command

**Priority:** P0
**Source:** `04_DATA_MODEL.md`, `ADR-004-control-command-definition.md`

The system shall use the canonical high-level control command:

```text id="dzqfhf"
ControlCommand4 = [phi_c, theta_c, vz_c, psi_dot_c]
```

where:

| Field       | Unit  | Meaning                     |
| ----------- | ----- | --------------------------- |
| `phi_c`     | rad   | commanded roll              |
| `theta_c`   | rad   | commanded pitch             |
| `vz_c`      | m/s   | commanded vertical velocity |
| `psi_dot_c` | rad/s | commanded yaw rate          |

The controller shall return `ControlCommand4`.

The controller shall not return rotor thrust, torque, PWM, motor speed, or MuJoCo actuator control.

**Acceptance criteria:**

```text id="94kxp4"
ControlCommand4 type exists
controller output command is ControlCommand4
tests reject invalid command shape
tests reject NaN or Inf command values
```

---

## FR-003: Actuator Command Separation

**Priority:** P0
**Source:** `ADR-004-control-command-definition.md`

The system shall define actuator-level command separately:

```text id="6qu5ep"
ActuatorCommand4 = [T1, T2, T3, T4]
```

The system shall not confuse `ControlCommand4` with `ActuatorCommand4`.

For actuator-level physics engines, the runtime shall call a mixer or adapter before stepping the engine.

**Acceptance criteria:**

```text id="96b6mu"
ActuatorCommand4 type exists
MuJoCo engine receives ActuatorCommand4
ODE engine may receive ControlCommand4
logger records both command levels separately
```

---

## FR-004: Scenario Configuration

**Priority:** P0
**Source:** `07_SCENARIO_CONFIG.md`

The system shall load scenario configuration from structured YAML files.

A canonical scenario shall define:

```text id="hrqas3"
schema_version
scenario metadata
world frame
initial_state
goal
obstacles
success criteria
termination criteria
runtime overrides
```

The scenario config shall map to typed config objects.

**Acceptance criteria:**

```text id="5lbu1l"
canonical scenario YAML loads successfully
invalid scenario YAML fails before runtime
initial_state maps to State9
goal maps to Goal3
obstacles map to ObstacleSpec
```

---

## FR-005: Legacy Scenario Adapter

**Priority:** P1
**Source:** `07_SCENARIO_CONFIG.md`

The system should support current flat scenario YAML files during migration.

Legacy mapping:

```text id="1kbwvp"
start -> initial_state.state9
goal -> goal.position
goal_threshold -> success.goal_threshold
sim_timestep -> runtime_overrides.sim_dt
obstacles -> obstacles
```

Legacy support shall emit a warning.

**Acceptance criteria:**

```text id="f7dxzh"
legacy scenario loads through adapter
warning is emitted
legacy fields map to canonical fields
new scenarios use canonical schema
```

---

## FR-006: Physics Engine Interface

**Priority:** P0
**Source:** `05_ENGINE_INTERFACE.md`, `ADR-001-engine-abstraction.md`

The system shall expose physics backends through a common `PhysicsEngine` interface.

Required methods:

```text id="cwdsod"
reset()
step()
get_state()
get_time()
get_metadata()
close()
```

Every physics engine shall expose `State9` at its public boundary.

**Acceptance criteria:**

```text id="hq3rlz"
PhysicsEngine interface exists
ODEPhysicsEngine implements interface
MuJoCoPhysicsEngine implements interface or is marked experimental
engine.get_state() returns State9-compatible state
engine metadata declares expected command type
```

---

## FR-007: ODE Physics Engine

**Priority:** P0
**Source:** `05_ENGINE_INTERFACE.md`

The system shall provide an ODE-based physics engine for deterministic and fast simulation.

The ODE engine shall:

```text id="zc1p0z"
consume ControlCommand4
step reduced quadrotor dynamics
return State9
return StepResult
track simulation time
```

**Acceptance criteria:**

```text id="ni5b2p"
ODE engine reset works
ODE engine step works
ODE engine returns finite State9
empty_world ODE scenario runs through runtime
```

---

## FR-008: MuJoCo Physics Engine

**Priority:** P1
**Source:** `05_ENGINE_INTERFACE.md`

The system shall provide a MuJoCo physics engine or clearly mark it as experimental until validation passes.

The MuJoCo engine shall:

```text id="hx4pmu"
hide qpos/qvel from controller
expose State9 publicly
consume ActuatorCommand4
use explicit State9 <-> MuJoCo adapter
return StepResult
```

**Acceptance criteria:**

```text id="cl3bzm"
MuJoCo reset/get_state round-trip test passes
MuJoCo engine does not expose qpos/qvel to controller
MuJoCo step receives ActuatorCommand4
actuator commands are logged
```

---

## FR-009: Controller Interface

**Priority:** P0
**Source:** `06_CONTROLLER_INTERFACE.md`

The system shall expose controllers through a common `Controller` interface.

Required methods:

```text id="29w0uf"
reset()
compute_command()
get_metadata()
close()
```

The controller shall receive `ControllerInput`.

The controller shall return `ControllerOutput`.

**Acceptance criteria:**

```text id="tgrg9h"
Controller interface exists
CCMPCController implements interface
controller accepts ControllerInput
controller returns ControllerOutput
controller diagnostics are populated
```

---

## FR-010: CC-MPC Controller

**Priority:** P0
**Source:** `06_CONTROLLER_INTERFACE.md`

The system shall wrap the existing CC-MPC solver behind `CCMPCController`.

`CCMPCController` shall:

```text id="qktl2v"
receive estimated State9
receive Goal3
receive covariance
receive obstacle predictions
solve or call the CC-MPC optimization
return first ControlCommand4
return predicted trajectory if available
return diagnostics
use fallback when needed
```

**Acceptance criteria:**

```text id="st26w0"
CCMPCController returns valid ControlCommand4
predicted trajectory has expected shape
diagnostics include solver status and solve time
fallback returns valid ControlCommand4
```

---

## FR-011: Fallback Controller

**Priority:** P1
**Source:** `06_CONTROLLER_INTERFACE.md`

The system shall provide fallback behavior when the controller cannot produce a nominal solution.

Fallback may be triggered by:

```text id="t5sx3v"
QP infeasible
solver error
NaN solution
timeout policy
invalid predicted trajectory
```

Fallback shall still return `ControlCommand4`.

**Acceptance criteria:**

```text id="m84vxq"
forced solver failure triggers fallback
fallback command is valid ControlCommand4
fallback reason is logged
runtime policy handles fallback consistently
```

---

## FR-012: Estimator Interface

**Priority:** P1
**Source:** `03_RUNTIME_FLOW.md`, `04_DATA_MODEL.md`

The system shall separate `true_state` from `estimated_state`.

The initial system shall support:

```text id="a5zwkr"
IdealEstimator
NoisyStateEstimator
```

Estimator output shall include:

```text id="xfwtfq"
estimated_state: State9
covariance: Gamma9x9
```

**Acceptance criteria:**

```text id="21zzx0"
ideal estimator returns true_state as estimated_state
noisy estimator returns valid State9
covariance shape is 9x9
controller receives estimated_state, not raw engine internals
```

---

## FR-013: Obstacle Manager

**Priority:** P0
**Source:** `07_SCENARIO_CONFIG.md`, `09_VALIDATION_PLAN.md`

The system shall manage static and moving obstacles.

Obstacle manager shall:

```text id="rnhecc"
load obstacles from ScenarioConfig
represent obstacle geometry
predict obstacle positions over horizon
provide obstacle predictions to controller
provide collision and margin metrics
```

Initial motion model:

```text id="3wvunm"
constant_velocity
```

**Acceptance criteria:**

```text id="9z6rks"
static obstacle prediction is stable
moving obstacle prediction follows constant velocity
collision/margin metrics are finite
obstacle manager handles empty obstacle list
```

---

## FR-014: Deterministic Runtime

**Priority:** P0
**Source:** `03_RUNTIME_FLOW.md`, `ADR-002-single-thread-vs-mpc-thread.md`

The system shall provide deterministic single-thread runtime as the reference mode.

Reference step order:

```text id="pdkbk9"
engine.get_state()
estimator.estimate()
obstacle_manager.predict_horizon()
controller.compute_command() if due
dispatch_command()
engine.step()
build LogRecord
logger.record()
renderer.render() if due
termination_checker.check()
```

**Acceptance criteria:**

```text id="q27q6v"
runtime executes steps in documented order
controller due logic works
zero-order hold works between controller updates
deterministic ODE run is reproducible within tolerance
```

---

## FR-015: Command Dispatch

**Priority:** P0
**Source:** `03_RUNTIME_FLOW.md`, `05_ENGINE_INTERFACE.md`

The runtime shall dispatch commands according to engine metadata.

Rules:

```text id="dglwjx"
ODE engine: ControlCommand4 -> engine.step()
MuJoCo engine: ControlCommand4 -> LowLevelMixer -> ActuatorCommand4 -> engine.step()
```

The runtime shall reject unsupported command types.

**Acceptance criteria:**

```text id="x3xw1u"
dispatch sends ControlCommand4 to ODE
dispatch uses mixer for MuJoCo
dispatch rejects missing mixer for actuator-level engine
wrong command type is rejected
```

---

## FR-016: Logging

**Priority:** P0
**Source:** `08_LOGGING_AND_METRICS.md`

The system shall produce structured logs for each complete run.

Required outputs:

```text id="c2eq5o"
metadata.json
steps.csv
summary.json
```

Optional outputs:

```text id="21m86d"
events.jsonl
controller_debug.jsonl
obstacle_debug.jsonl
trajectories.npz
```

**Acceptance criteria:**

```text id="zjgzx7"
metadata.json is written
steps.csv is written
summary.json is written
CSV schema is stable
summary metrics match steps.csv
```

---

## FR-017: Metrics

**Priority:** P0
**Source:** `08_LOGGING_AND_METRICS.md`

The system shall compute and log core metrics.

Required metrics:

```text id="2u9uax"
goal_distance
goal_reached
collision_flag
min_obstacle_distance
min_obstacle_margin
controller_solve_time_ms
controller_status
fallback_used
nan_detected
termination_reason
```

**Acceptance criteria:**

```text id="wl6xa8"
metrics are finite when defined
summary includes final_goal_distance
summary includes collision status
summary includes solver/fallback statistics
```

---

## FR-018: Termination Handling

**Priority:** P0
**Source:** `03_RUNTIME_FLOW.md`, `07_SCENARIO_CONFIG.md`

The system shall terminate runs based on defined conditions.

Required termination reasons:

```text id="3q39wc"
goal_reached
collision
max_time
max_steps
altitude_violation
numerical_failure
solver_failure
user_interrupt
```

**Acceptance criteria:**

```text id="rpka1h"
goal reached termination works
max time termination works
collision termination works when enabled
numerical failure produces summary
termination reason is logged
```

---

## FR-019: Scenario-Based CLI Entry Point

**Priority:** P1
**Source:** `11_REFACTOR_PLAN.md`

The system should provide a CLI entry point.

Recommended command:

```bash id="sa0c33"
python scripts/run_simulation.py --scenario empty_world --engine ode
```

The CLI shall load app/config files and call the runtime.

The CLI shall not implement controller, physics, logging, or plotting internals.

**Acceptance criteria:**

```text id="jluqgx"
run_simulation.py exists
CLI can run ODE empty_world scenario
CLI produces metadata.json, steps.csv, summary.json
CLI does not duplicate runtime loop
```

---

## FR-020: Rendering

**Priority:** P2
**Source:** `02_ARCHITECTURE.md`

The system may provide optional rendering.

Initial renderers:

```text id="mmwyuq"
NullRenderer
MatplotlibRenderer
```

Future renderers:

```text id="0e80z6"
MuJoCoViewerRenderer
Open3DRenderer
WebRenderer
```

Rendering shall be passive and shall not mutate runtime or engine state.

**Acceptance criteria:**

```text id="a2kv7b"
headless mode works without renderer
renderer receives snapshots
renderer does not modify engine/controller state
```

---

## FR-021: Validation Suite

**Priority:** P0
**Source:** `09_VALIDATION_PLAN.md`

The system shall include validation tests for core contracts.

Required test categories:

```text id="n8i4rp"
unit tests
interface tests
integration tests
regression tests
validation experiments
```

Minimum merge gate:

```text id="e68myf"
type/data tests
scenario loader tests
ODE engine interface tests
controller interface tests
logging schema tests
ODE empty_world regression
```

**Acceptance criteria:**

```text id="26i42g"
pytest test suite exists
core tests pass
regression tests protect log schema
validation report can be generated
```

---

## FR-022: Threaded Runtime

**Priority:** P3
**Source:** `ADR-002-single-thread-vs-mpc-thread.md`

The system may provide threaded MPC runtime after deterministic runtime is stable.

Threaded runtime shall include:

```text id="03s60e"
state snapshots
command snapshots
timestamps
sequence IDs
command age logging
stale command policy
safe shutdown
```

Threaded runtime shall not be the reference mode.

**Acceptance criteria:**

```text id="r9zvja"
threaded mode is explicitly configured
snapshot atomicity tests pass
stale command is logged
threaded mode uses same Controller and PhysicsEngine interfaces
```

---

# 8. Non-Functional Requirements

---

## NFR-001: Reproducibility

**Priority:** P0

The system shall support reproducible deterministic runs.

Given the same:

```text id="dgj2ts"
scenario
config
initial state
random seed
engine
controller
runtime mode
software versions
```

the deterministic ODE run should produce the same output within numerical tolerance.

**Acceptance criteria:**

```text id="2wwczb"
same seed reproducibility test passes
metadata records config files
metadata records runtime mode
metadata records engine/controller type
```

---

## NFR-002: Traceability

**Priority:** P0

The system shall make simulation results traceable from config to output logs.

A run shall record:

```text id="lud7wk"
scenario id
engine type
controller type
runtime mode
config files
random seed
termination reason
core metrics
```

**Acceptance criteria:**

```text id="wzjyu0"
metadata.json contains required fields
summary.json contains final metrics
steps.csv contains state/control/diagnostic fields
```

---

## NFR-003: Modularity

**Priority:** P0

The system shall separate:

```text id="ke96rk"
runtime
controller
physics engine
mixer
estimator
obstacle manager
logger
renderer
config loader
```

No module shall own unrelated responsibilities.

**Acceptance criteria:**

```text id="4yxqf3"
controller does not step engine
engine does not call controller
logger does not mutate runtime
runtime orchestrates modules through interfaces
```

---

## NFR-004: Testability

**Priority:** P0

Every architectural boundary shall be testable.

The following shall have tests:

```text id="tveqvy"
canonical types
scenario loader
engine interface
controller interface
runtime dispatch
logging schema
metrics
```

**Acceptance criteria:**

```text id="8lv4g7"
tests exist for each boundary
tests can run without GUI
ODE tests can run in CI
MuJoCo tests can be marked optional if environment lacks MuJoCo
```

---

## NFR-005: Numerical Robustness

**Priority:** P0

The system shall detect and handle numerical errors.

The system shall detect:

```text id="ul2utj"
NaN
Inf
invalid covariance
invalid state shape
invalid command shape
solver failure
engine failure
```

**Acceptance criteria:**

```text id="x7bl81"
NaN in true_state terminates or fails validation
NaN in control command is rejected
invalid covariance is rejected
numerical failure is logged
```

---

## NFR-006: Performance Observability

**Priority:** P1

The system shall measure performance even if it does not guarantee hard real-time execution.

The system shall log:

```text id="fqhhuv"
controller_solve_time_ms
engine_step_time_ms if available
step_wall_time_ms
fallback_count
late_solve_count
```

**Acceptance criteria:**

```text id="zd5fh5"
solve time appears in logs
summary includes mean/max solve time
late solve count can be computed
```

---

## NFR-007: Maintainability

**Priority:** P1

The system should be easy to extend and maintain.

Requirements:

```text id="ni40bu"
small interfaces
clear package layout
no hidden global state
no duplicated runtime loops
no duplicated logger schemas
```

**Acceptance criteria:**

```text id="o8zduz"
README points to new entry point
design docs match code structure
demo scripts are thin wrappers or legacy
```

---

## NFR-008: Headless Operation

**Priority:** P0

The system shall run without GUI/display.

Headless operation is required for:

```text id="tx8abg"
CI
batch experiments
regression tests
remote servers
```

**Acceptance criteria:**

```text id="w110bs"
ODE runtime runs headless
logger works headless
renderer can be disabled
tests do not require viewer
```

---

## NFR-009: Config Validation Before Runtime

**Priority:** P0

The system shall validate configs before simulation starts.

Invalid configs shall fail early.

**Acceptance criteria:**

```text id="lk56dg"
invalid State9 length rejected before runtime
invalid goal length rejected before runtime
negative obstacle size rejected before runtime
unsupported engine type rejected before runtime
```

---

## NFR-010: Honest Scope and Limitations

**Priority:** P0

The system shall not hide limitations.

The system shall document:

```text id="3nd5bf"
reduced dynamics
Euler singularity
ODE/MuJoCo mismatch
approximate mixer
Gaussian uncertainty assumption
linearized chance constraints
no real perception pipeline
no real VIO pipeline
no formal stability proof
not flight-certified
```

**Acceptance criteria:**

```text id="njnw4y"
10_KNOWN_LIMITATIONS.md exists
README or docs link to limitations
experimental claims reference limitations
```

---

# 9. Constraints

---

## CON-001: Canonical State Constraint

All public state boundaries shall use `State9`.

No public module shall expose a different state order without an adapter.

---

## CON-002: Controller Command Constraint

All high-level controllers shall output `ControlCommand4`.

---

## CON-003: Actuator Command Constraint

Actuator-level engines shall receive `ActuatorCommand4`, not `ControlCommand4`.

---

## CON-004: Engine Independence Constraint

The controller shall not know whether the active engine is ODE or MuJoCo.

---

## CON-005: Runtime Ownership Constraint

Only runtime shall orchestrate cross-module calls.

Controller, engine, logger, and renderer shall not orchestrate each other.

---

## CON-006: Logger Passive Constraint

Logger shall not affect simulation behavior.

Logger shall only record data passed by runtime.

---

## CON-007: Scenario Declarative Constraint

Scenario files shall describe experiment environment.

Scenario files shall not encode controller internals or solver variables.

---

## CON-008: Deterministic Reference Constraint

Deterministic single-thread runtime shall be the reference mode.

Threaded runtime shall be optional.

---

## CON-009: MuJoCo Adapter Constraint

MuJoCo-specific data such as `qpos`, `qvel`, quaternion layout, body IDs, and actuator IDs shall remain inside MuJoCo engine/adapters.

---

## CON-010: No Hardware-Safety Claim

The simulation shall not be claimed as flight-certified, hardware-safe, or deployment-ready.

---

# 10. Validation Requirements

---

## VAL-001: Data Contract Validation

The system shall include tests for:

```text id="413qc4"
State9
ControlCommand4
ActuatorCommand4
Trajectory9
Gamma9x9
```

---

## VAL-002: Engine Validation

The system shall include interface tests for:

```text id="1fuqlg"
ODEPhysicsEngine
MuJoCoPhysicsEngine if enabled
```

---

## VAL-003: Controller Validation

The system shall include tests verifying that:

```text id="ra6fgu"
CCMPCController returns ControlCommand4
diagnostics are populated
fallback works
predicted trajectory shape is correct
```

---

## VAL-004: Runtime Validation

The system shall include integration tests verifying:

```text id="hp4b9j"
deterministic step order
controller due logic
zero-order hold
command dispatch
termination behavior
```

---

## VAL-005: Logging Validation

The system shall include tests verifying:

```text id="ho56fk"
metadata.json exists
steps.csv header is stable
summary.json exists
summary metrics match step log
```

---

## VAL-006: Regression Validation

The system shall include regression scenarios:

```text id="x1rt6f"
empty_world_ode
static_obstacle_ode
moving_obstacle_ode
```

MuJoCo regression scenarios may be added after MuJoCo adapter and mixer validation pass.

---

# 11. Documentation Requirements

---

## DOC-001: Design Documentation

The following docs shall exist:

```text id="7posft"
00_SIMULATION_VISION.md
01_REQUIREMENTS.md
02_ARCHITECTURE.md
03_RUNTIME_FLOW.md
04_DATA_MODEL.md
05_ENGINE_INTERFACE.md
06_CONTROLLER_INTERFACE.md
07_SCENARIO_CONFIG.md
08_LOGGING_AND_METRICS.md
09_VALIDATION_PLAN.md
10_KNOWN_LIMITATIONS.md
11_REFACTOR_PLAN.md
```

---

## DOC-002: ADR Documentation

The following ADRs shall exist:

```text id="zaxgj3"
ADR-001-engine-abstraction.md
ADR-002-single-thread-vs-mpc-thread.md
ADR-003-state-vector-definition.md
ADR-004-control-command-definition.md
```

---

## DOC-003: README Update

The project README shall eventually document:

```text id="6g5s5e"
project purpose
new architecture
how to run simulation
how to run tests
where logs are written
known limitations
```

---

# 12. Out of Scope

---

## OOS-001: Real Flight Deployment

The refactor shall not provide real flight deployment.

---

## OOS-002: Flight Certification

The refactor shall not claim flight certification or hardware safety.

---

## OOS-003: Full Perception Pipeline

The initial refactor shall not implement full real depth-camera perception.

---

## OOS-004: Full VIO Pipeline

The initial refactor shall not implement full real visual-inertial odometry.

---

## OOS-005: Formal Stability Proof

The initial refactor shall not provide formal closed-loop stability proof.

---

## OOS-006: Formal Recursive Feasibility Proof

The initial refactor shall not prove recursive feasibility.

---

## OOS-007: Full Newton-Euler NMPC

The initial refactor shall not replace the current reduced-order CC-MPC with full Newton-Euler NMPC.

---

# 13. Requirement Traceability Matrix

| Requirement | Design Document                     | Validation                                       |
| ----------- | ----------------------------------- | ------------------------------------------------ |
| FR-001      | `04_DATA_MODEL.md`, `ADR-003`       | `test_state9_field_order`                        |
| FR-002      | `04_DATA_MODEL.md`, `ADR-004`       | `test_control_command4_field_order`              |
| FR-003      | `ADR-004`, `05_ENGINE_INTERFACE.md` | `test_mujoco_uses_actuator_command4`             |
| FR-004      | `07_SCENARIO_CONFIG.md`             | `test_load_canonical_scenario`                   |
| FR-006      | `05_ENGINE_INTERFACE.md`, `ADR-001` | `test_engine_get_state_returns_state9`           |
| FR-007      | `05_ENGINE_INTERFACE.md`            | `test_ode_engine_step_returns_state9`            |
| FR-008      | `05_ENGINE_INTERFACE.md`            | `test_mujoco_state9_roundtrip`                   |
| FR-009      | `06_CONTROLLER_INTERFACE.md`        | `test_controller_returns_controller_output`      |
| FR-010      | `06_CONTROLLER_INTERFACE.md`        | `test_ccmpc_controller_returns_control_command4` |
| FR-014      | `03_RUNTIME_FLOW.md`, `ADR-002`     | `test_runtime_step_order`                        |
| FR-015      | `03_RUNTIME_FLOW.md`                | `test_dispatch_uses_mixer_for_mujoco`            |
| FR-016      | `08_LOGGING_AND_METRICS.md`         | `test_csv_logger_writes_steps`                   |
| FR-017      | `08_LOGGING_AND_METRICS.md`         | `test_goal_distance_metric`                      |
| FR-021      | `09_VALIDATION_PLAN.md`             | `pytest tests`                                   |
| NFR-010     | `10_KNOWN_LIMITATIONS.md`           | documentation review                             |

---

# 14. Minimum Viable Refactor Requirements

The minimum viable refactor shall include:

```text id="fym2w6"
FR-001 State9
FR-002 ControlCommand4
FR-003 ActuatorCommand4 separation
FR-004 ScenarioConfig
FR-006 PhysicsEngine interface
FR-007 ODEPhysicsEngine
FR-009 Controller interface
FR-010 CCMPCController wrapper
FR-014 deterministic runtime
FR-015 command dispatch
FR-016 logging
FR-017 metrics
FR-021 validation suite
```

A refactor that does not satisfy these requirements shall not be considered complete.

---

# 15. Final Acceptance Criteria

The refactored system is accepted when:

1. Canonical data types exist and are tested.
2. Scenario config loader exists and is tested.
3. ODE engine runs through `PhysicsEngine`.
4. CC-MPC runs through `Controller`.
5. Runtime runs deterministic single-thread loop.
6. ODE empty-world scenario reaches the goal.
7. Static obstacle scenario avoids collision.
8. Logs include metadata, steps, and summary.
9. Summary metrics match step logs.
10. No public module changes `State9` or `ControlCommand4` ordering.
11. MuJoCo support is either validated or clearly marked experimental.
12. Known limitations are documented.
13. Tests pass for the minimum merge gate.

---

# 16. Summary

This requirements document defines what the refactored quadrotor CC-MPC simulation must provide.

The most important requirements are:

```text id="4tipvy"
explicit State9 and ControlCommand4 contracts
separation between ControlCommand4 and ActuatorCommand4
engine-agnostic controller interface
controller-agnostic physics engine interface
deterministic reference runtime
canonical scenario configuration
passive structured logging
validation-driven refactor
honest known limitations
```

The requirements convert the project vision into testable implementation targets.

A requirement is not complete until it is implemented, tested, and traceable to the relevant design document.

---

# 17. Related Documents

```text id="58x02l"
docs/design/00_SIMULATION_VISION.md
docs/design/02_ARCHITECTURE.md
docs/design/03_RUNTIME_FLOW.md
docs/design/04_DATA_MODEL.md
docs/design/05_ENGINE_INTERFACE.md
docs/design/06_CONTROLLER_INTERFACE.md
docs/design/07_SCENARIO_CONFIG.md
docs/design/08_LOGGING_AND_METRICS.md
docs/design/09_VALIDATION_PLAN.md
docs/design/10_KNOWN_LIMITATIONS.md
docs/design/11_REFACTOR_PLAN.md

docs/design/ADR/ADR-001-engine-abstraction.md
docs/design/ADR/ADR-002-single-thread-vs-mpc-thread.md
docs/design/ADR/ADR-003-state-vector-definition.md
docs/design/ADR/ADR-004-control-command-definition.md
```
