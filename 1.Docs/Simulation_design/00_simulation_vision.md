# 00_SIMULATION_VISION.md

> Status: Draft
> Scope: Vision document for refactored simulation
> Project: Quadrotor CC-MPC Simulation
> Related documents:
>
> * `01_REQUIREMENTS.md`
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

---

## 1. Purpose

This document defines the vision for the refactored quadrotor CC-MPC simulation.

The purpose is to answer:

```text id="0oqe6d"
Why does this simulation exist?
What research problem does it support?
What should the refactored system become?
What should the system explicitly not become?
What does success look like?
```

This is a high-level vision document.

Detailed requirements, architecture, interfaces, runtime flow, logging, validation, and refactor phases are defined in later documents.

---

## 2. Executive Vision

The project shall become a research-grade simulation platform for studying chance-constrained model predictive control for quadrotor obstacle avoidance under uncertainty.

The refactored simulation shall support:

```text id="3bop59"
goal-directed quadrotor navigation
static and moving obstacle avoidance
uncertainty-aware planning
ODE-based fast simulation
MuJoCo-based higher-fidelity simulation
deterministic experiment replay
structured scenario configuration
stable logging and metrics
validation-driven refactoring
```

The simulation shall not be a loose collection of demo scripts.

It shall become a modular system where:

```text id="b6x1o2"
state definitions are explicit
control command meanings are explicit
physics engines are replaceable
controllers are replaceable
runtime flow is deterministic by default
logs are analysis-ready
limitations are documented
```

---

## 3. Problem Statement

Autonomous quadrotor navigation in cluttered environments is difficult because the vehicle must simultaneously:

```text id="0h7f58"
move toward a goal
respect dynamic constraints
avoid static obstacles
avoid moving obstacles
handle state-estimation uncertainty
handle obstacle-sensing uncertainty
handle process noise and disturbances
compute commands online
```

A deterministic planner that ignores uncertainty may produce trajectories that are nominally collision-free but unsafe when the real state deviates from the planned state.

Therefore, this project focuses on:

```text id="u7iok7"
Chance-Constrained Model Predictive Control
```

The core idea is to plan trajectories such that the probability of collision remains below a specified risk threshold.

---

## 4. Research Motivation

The main research motivation is to study how CC-MPC can be used for quadrotor obstacle avoidance in uncertain environments.

The simulation should help answer questions such as:

```text id="kdz0dr"
Can the quadrotor reach the goal while avoiding obstacles?
How does uncertainty affect obstacle avoidance?
How conservative are the chance constraints?
How often does the solver fail?
How sensitive is the controller to model mismatch?
How different are ODE and MuJoCo results?
How does obstacle motion affect feasibility?
How should fallback behavior be designed?
```

The system shall make these questions measurable through logs, metrics, and validation experiments.

---

## 5. Core Research Question

The central research question is:

```text id="rfgf8g"
Can a quadrotor using CC-MPC safely and reliably navigate to a goal in an obstacle-filled environment while accounting for uncertainty in state estimation, motion, and obstacle prediction?
```

Sub-questions:

```text id="x6z61x"
How should uncertainty be propagated over the prediction horizon?
How should obstacle avoidance be formulated as chance constraints?
How should the controller behave when the QP becomes infeasible?
How should model mismatch between planner and physics engine be measured?
How should simulation results be logged for reproducible analysis?
```

---

## 6. Target Simulation System

The target system is a modular simulation platform composed of:

```text id="q4tr7l"
SimulationRuntime
Controller
PhysicsEngine
Estimator
ObstacleManager
LowLevelMixer
Logger
Renderer
ScenarioConfig
ValidationSuite
```

The system shall support two main physics backends:

```text id="o83i0l"
ODEPhysicsEngine
MuJoCoPhysicsEngine
```

The system shall support one primary controller:

```text id="fgo4ix"
CCMPCController
```

and allow future controllers such as:

```text id="407cbc"
PIDController
LQRController
NominalMPCController
NMPCController
EmergencyStopController
```

---

## 7. What Is Being Simulated

The simulation models the closed-loop behavior of a quadrotor navigation system.

At a high level, the simulation includes:

```text id="jyy88n"
quadrotor state evolution
controller planning
uncertainty propagation
obstacle prediction
collision checking
command application
runtime timing
logging and metrics
```

The canonical high-level state is:

```text id="y6lm52"
State9 = [x, y, z, vx, vy, vz, roll, pitch, yaw]
```

The canonical high-level control command is:

```text id="8kknn6"
ControlCommand4 = [phi_c, theta_c, vz_c, psi_dot_c]
```

The actuator-level command for rotor-force simulation is:

```text id="v6b0kg"
ActuatorCommand4 = [T1, T2, T3, T4]
```

---

## 8. What Is Not Being Simulated Initially

The initial refactor does not attempt to fully simulate:

```text id="r77gbc"
real camera perception pipeline
real visual-inertial odometry pipeline
real motor dynamics
battery voltage sag
wind gusts
ground effect
aerodynamic blade effects
hardware communication delays
real flight-controller firmware
multi-agent communication
formal safety monitor
```

Some of these may be added later.

For the initial refactor, the priority is to build a correct, modular, validated simulation architecture.

---

## 9. Intended Users

The primary users are:

```text id="zcl5nd"
researcher developing CC-MPC algorithms
graduate student studying quadrotor control
developer refactoring the simulation code
reviewer checking reproducibility
future contributor adding engines/controllers/scenarios
```

The system should be understandable by someone who knows:

```text id="r8g5ic"
Python
basic robotics
state-space modeling
MPC concepts
linear algebra
probability and covariance
```

The system should not require reading all demo scripts to understand the architecture.

---

## 10. Primary Use Cases

### 10.1 Run a deterministic ODE simulation

A user should be able to run:

```bash id="h4zg1l"
python scripts/run_simulation.py --scenario empty_world --engine ode
```

Expected output:

```text id="uoj743"
metadata.json
steps.csv
summary.json
```

---

### 10.2 Run an obstacle-avoidance scenario

A user should be able to run a scenario with static or moving obstacles and inspect:

```text id="tkt12y"
goal distance
minimum obstacle distance
minimum obstacle margin
chance-constraint margin
solver status
fallback count
trajectory
```

---

### 10.3 Compare ODE and MuJoCo behavior

A user should be able to run the same scenario with:

```text id="s4jrvr"
ODEPhysicsEngine
MuJoCoPhysicsEngine
```

and compare:

```text id="8kddg9"
trajectory
goal distance
control commands
actuator commands
solver behavior
runtime warnings
success/failure reason
```

The system shall not imply that ODE and MuJoCo are physically equivalent by default.

---

### 10.4 Validate a refactor

A developer should be able to run:

```bash id="wzq0wz"
pytest tests/unit
pytest tests/interface
pytest tests/integration
pytest tests/regression
```

and determine whether the refactor broke:

```text id="tbv8kg"
state ordering
control ordering
engine interface
controller interface
scenario loader
runtime dispatch
logging schema
```

---

### 10.5 Debug a failed run

If a simulation fails, the user should be able to inspect logs and answer:

```text id="0qxhkg"
Which state did the controller receive?
Which command did the controller return?
Was fallback used?
Was the command sent directly to ODE or through the mixer?
Did the physics engine fail?
Was there a collision?
Was there a NaN?
Which obstacle was closest?
What was the termination reason?
```

---

## 11. Vision for Refactored Architecture

The refactored architecture shall follow this conceptual flow:

```text id="i0e07m"
ScenarioConfig
    -> SimulationRuntime
        -> PhysicsEngine
        -> Estimator
        -> ObstacleManager
        -> Controller
        -> LowLevelMixer
        -> Logger
        -> Renderer
```

Reference runtime flow:

```text id="f74y4l"
true_state
-> estimated_state
-> obstacle_predictions
-> controller output
-> command dispatch
-> physics step
-> logging
-> rendering
-> termination check
```

The runtime shall orchestrate modules.

It shall not implement module internals.

---

## 12. System Boundary

The simulation boundary includes:

```text id="2rpocd"
scenario loading
initial state
goal
obstacle definitions
physics stepping
controller execution
uncertainty propagation
obstacle prediction
command dispatch
logging
metrics
validation
```

The boundary excludes:

```text id="w0p1fl"
real flight hardware
real onboard autopilot
real sensor drivers
real ROS deployment
real camera calibration
real human-safety certification
```

If hardware experiments are later added, they shall be defined in a separate architecture document.

---

## 13. Reference Data Contracts

The project shall use explicit canonical data contracts.

### 13.1 Canonical state

```text id="ru9is4"
State9 = [x, y, z, vx, vy, vz, roll, pitch, yaw]
```

Position and velocity are expressed in world frame.

Attitude uses Euler ZYX angles in radians.

---

### 13.2 Canonical controller command

```text id="aql9mh"
ControlCommand4 = [phi_c, theta_c, vz_c, psi_dot_c]
```

This is a high-level controller command.

It is not rotor thrust, motor speed, torque, PWM, or MuJoCo actuator control.

---

### 13.3 Canonical actuator command

```text id="oijzvp"
ActuatorCommand4 = [T1, T2, T3, T4]
```

This is used when a physics engine requires rotor-level actuator input.

---

### 13.4 Canonical covariance

The full state covariance is:

```text id="0coia1"
Gamma9x9
```

The position covariance used by obstacle chance constraints is:

```text id="f8dhc1"
Sigma3x3 = Gamma9x9[0:3, 0:3]
```

---

## 14. Design Principles

### 14.1 Explicit over implicit

State order, command order, units, frames, and ownership shall be explicit.

No module shall rely on hidden assumptions.

---

### 14.2 Interfaces over script coupling

The system shall use interfaces such as:

```text id="fluk8k"
PhysicsEngine
Controller
Logger
Estimator
Renderer
```

instead of coupling core logic to demo scripts.

---

### 14.3 Deterministic first

The reference runtime shall be deterministic single-thread mode.

Threaded runtime may be added later, but it shall not be the initial source of truth.

---

### 14.4 Logs are research artifacts

Logs are not optional debug leftovers.

Logs are part of the research output.

They shall be stable, structured, versioned, and sufficient for analysis.

---

### 14.5 Validation before claims

The project shall not claim that a feature works until it is validated.

Examples:

```text id="8e35i2"
ODE support requires ODE runtime tests.
MuJoCo support requires MuJoCo adapter and mixer tests.
Logging support requires schema tests.
Obstacle avoidance support requires collision/margin tests.
```

---

### 14.6 Limitations are part of the design

The project shall document limitations openly.

Known limitations shall not be hidden behind optimistic wording.

---

## 15. Simulation Modes

The refactored system shall support the following modes.

### 15.1 Deterministic single-thread mode

Primary mode for:

```text id="27jgrq"
debugging
testing
regression
research experiments
paper-style plots
```

This is the reference mode.

---

### 15.2 Headless batch mode

Used for:

```text id="72fay6"
running many scenarios
collecting metrics
CI tests
regression experiments
```

No viewer is required.

---

### 15.3 Real-time viewer mode

Used for:

```text id="fq82m6"
visual inspection
demo videos
qualitative debugging
```

This mode may skip rendering frames, but it shall not silently skip controller logic.

---

### 15.4 Threaded MPC mode

Optional future mode for:

```text id="teqv37"
asynchronous control experiments
real-time visualization
deployment-like timing studies
```

This mode shall use timestamped snapshots and stale-command diagnostics.

---

## 16. Expected Outputs

Each complete run shall produce:

```text id="r2suqx"
metadata.json
steps.csv
summary.json
```

Optional outputs:

```text id="a6jmvp"
events.jsonl
controller_debug.jsonl
obstacle_debug.jsonl
trajectories.npz
plots
validation_report.md
```

A simulation run without analyzable outputs is not considered complete.

---

## 17. Success Criteria

The refactored simulation vision is achieved when:

```text id="cu7l9u"
State9 is consistently used at public boundaries
ControlCommand4 is consistently used as controller output
ActuatorCommand4 is separated from ControlCommand4
ODE engine runs through PhysicsEngine interface
MuJoCo engine is isolated behind adapters
CCMPC runs through Controller interface
runtime flow is deterministic by default
scenario config is structured and validated
logs are stable and analysis-ready
validation tests exist for core contracts
demo scripts are no longer the architecture source
```

---

## 18. Research Success Criteria

A research experiment is successful only if it reports:

```text id="mmv4aq"
scenario id
engine type
controller type
runtime mode
initial state
goal
obstacles
solver status
goal distance
collision status
minimum obstacle margin
chance-constraint margin if available
fallback count
termination reason
known limitations
```

The system shall make these quantities easy to obtain.

---

## 19. Engineering Success Criteria

The engineering refactor is successful when:

```text id="d760ah"
new developers can understand the architecture from docs
tests catch state/control ordering mistakes
ODE and MuJoCo code paths share runtime infrastructure
scenario loading is not duplicated in scripts
logging is not duplicated in scripts
controller code does not know engine internals
engine code does not know controller internals
```

---

## 20. Non-Goals

The initial refactor shall not attempt to deliver:

```text id="2e8fwl"
flight-certified software
real hardware deployment
full nonlinear NMPC replacement
real perception stack
real VIO stack
multi-robot coordination stack
formal stability proof
formal recursive feasibility proof
real-time operating system integration
```

These are valid future directions, but they are not required for the first research-grade simulation refactor.

---

## 21. Long-Term Vision

The long-term vision is a simulation platform where a researcher can:

```text id="hj5kcd"
define a scenario
choose a controller
choose a physics engine
run a deterministic experiment
inspect structured logs
compare against baselines
validate assumptions
extend modules safely
```

Possible future extensions:

```text id="qj4l5m"
full Newton-Euler dynamics
quaternion-based planning state
NMPC controller
hardware-in-the-loop adapter
real perception replay
real VIO replay
multi-agent CC-MPC
Monte Carlo validation
automatic report generation
```

---

## 22. Relationship to Other Documents

This document defines the vision.

Detailed documents define implementation contracts:

| Document                     | Role                                       |
| ---------------------------- | ------------------------------------------ |
| `01_REQUIREMENTS.md`         | Functional and non-functional requirements |
| `02_ARCHITECTURE.md`         | System architecture and module boundaries  |
| `03_RUNTIME_FLOW.md`         | Runtime execution order                    |
| `04_DATA_MODEL.md`           | Canonical state/control/data contracts     |
| `05_ENGINE_INTERFACE.md`     | Physics engine interface                   |
| `06_CONTROLLER_INTERFACE.md` | Controller interface                       |
| `07_SCENARIO_CONFIG.md`      | Scenario YAML schema                       |
| `08_LOGGING_AND_METRICS.md`  | Log schema and metrics                     |
| `09_VALIDATION_PLAN.md`      | Test and validation strategy               |
| `10_KNOWN_LIMITATIONS.md`    | Known model/software/research limitations  |
| `11_REFACTOR_PLAN.md`        | Implementation roadmap                     |

ADR documents explain why key architecture decisions were made.

---

## 23. Acceptance Criteria

This vision document is accepted when:

```text id="it575p"
the purpose of the simulation is clear
the research motivation is clear
the target system boundary is clear
the non-goals are explicit
the core data contracts are named
the expected outputs are defined
the relationship to other design docs is clear
the document does not contradict existing ADRs
```

---

## 24. Summary

The refactored quadrotor CC-MPC simulation shall become a research-grade platform for uncertainty-aware obstacle avoidance experiments.

The central idea is:

```text id="q58sva"
use CC-MPC to compute high-level quadrotor commands
simulate the closed-loop behavior under controlled scenarios
log enough information to analyze safety, performance, and failure modes
validate every architectural contract through tests
```

The project shall prioritize:

```text id="iw5duh"
correctness
traceability
reproducibility
modularity
validation
honest limitations
```

over quick demo behavior.

The final result should be a simulation system that can support serious academic work rather than a set of scripts that happen to produce plots.

---

## 25. Related Documents

```text id="fof79l"
docs/design/01_REQUIREMENTS.md
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
