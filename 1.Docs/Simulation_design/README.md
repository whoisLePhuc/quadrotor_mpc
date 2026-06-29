# Design Documentation

> Project: Quadrotor CC-MPC Simulation
> Scope: Design documentation index and reading guide
> Status: Draft
> Location: `docs/design/`

---

## 1. Purpose

This directory contains the design documentation for the refactored quadrotor CC-MPC simulation.

The goal of these documents is to define the system before implementation, so the source code does not depend on ambiguous demo-script behavior.

The design documentation defines:

```text
system vision
requirements
architecture
runtime flow
canonical data model
engine interface
controller interface
scenario configuration
logging and metrics
validation plan
known limitations
refactor plan
architecture decision records
```

The refactored source code shall conform to these documents.

---

## 2. Recommended Reading Order

| Order | Document                                     | Purpose                                                                     |
| ----: | -------------------------------------------- | --------------------------------------------------------------------------- |
|     1 | `00_SIMULATION_VISION.md`                    | Explains why the simulation exists and what it should become                |
|     2 | `01_REQUIREMENTS.md`                         | Converts the vision into testable requirements                              |
|     3 | `04_DATA_MODEL.md`                           | Defines canonical state, control, command, covariance, and trajectory types |
|     4 | `ADR/ADR-003-state-vector-definition.md`     | Explains why `State9` is the canonical state                                |
|     5 | `ADR/ADR-004-control-command-definition.md`  | Explains why `ControlCommand4` is the canonical controller output           |
|     6 | `05_ENGINE_INTERFACE.md`                     | Defines the physics engine interface                                        |
|     7 | `ADR/ADR-001-engine-abstraction.md`          | Explains why physics engines are abstracted                                 |
|     8 | `06_CONTROLLER_INTERFACE.md`                 | Defines the controller interface                                            |
|     9 | `03_RUNTIME_FLOW.md`                         | Defines the execution order of one simulation loop                          |
|    10 | `ADR/ADR-002-single-thread-vs-mpc-thread.md` | Explains why deterministic single-thread runtime is the reference mode      |
|    11 | `02_ARCHITECTURE.md`                         | Shows the full modular architecture                                         |
|    12 | `07_SCENARIO_CONFIG.md`                      | Defines scenario YAML schema                                                |
|    13 | `08_LOGGING_AND_METRICS.md`                  | Defines log schema and metrics                                              |
|    14 | `09_VALIDATION_PLAN.md`                      | Defines tests and validation experiments                                    |
|    15 | `10_KNOWN_LIMITATIONS.md`                    | Records known model, solver, runtime, and research limitations              |
|    16 | `11_REFACTOR_PLAN.md`                        | Defines implementation phases for the refactor                              |

---

## 3. Document Map

```text
docs/design/
├── README.md
├── 00_SIMULATION_VISION.md
├── 01_REQUIREMENTS.md
├── 02_ARCHITECTURE.md
├── 03_RUNTIME_FLOW.md
├── 04_DATA_MODEL.md
├── 05_ENGINE_INTERFACE.md
├── 06_CONTROLLER_INTERFACE.md
├── 07_SCENARIO_CONFIG.md
├── 08_LOGGING_AND_METRICS.md
├── 09_VALIDATION_PLAN.md
├── 10_KNOWN_LIMITATIONS.md
├── 11_REFACTOR_PLAN.md
└── ADR/
    ├── README.md
    ├── ADR-001-engine-abstraction.md
    ├── ADR-002-single-thread-vs-mpc-thread.md
    ├── ADR-003-state-vector-definition.md
    └── ADR-004-control-command-definition.md
```

---

## 4. Core Design Decisions

The design folder is built around a few non-negotiable decisions.

### 4.1 Canonical state

```text
State9 = [x, y, z, vx, vy, vz, roll, pitch, yaw]
```

Defined in:

```text
04_DATA_MODEL.md
ADR/ADR-003-state-vector-definition.md
```

### 4.2 Canonical controller command

```text
ControlCommand4 = [phi_c, theta_c, vz_c, psi_dot_c]
```

Defined in:

```text
04_DATA_MODEL.md
ADR/ADR-004-control-command-definition.md
06_CONTROLLER_INTERFACE.md
```

### 4.3 Actuator command is separate

```text
ActuatorCommand4 = [T1, T2, T3, T4]
```

The controller shall not output rotor thrust directly.

### 4.4 Physics engines are replaceable

```text
PhysicsEngine
├── ODEPhysicsEngine
└── MuJoCoPhysicsEngine
```

Defined in:

```text
05_ENGINE_INTERFACE.md
ADR/ADR-001-engine-abstraction.md
```

### 4.5 Deterministic runtime is the reference mode

The reference runtime shall be deterministic and single-threaded.

Threaded MPC runtime may be added later, but it is not the default reference architecture.

Defined in:

```text
03_RUNTIME_FLOW.md
ADR/ADR-002-single-thread-vs-mpc-thread.md
```

---

## 5. How to Use These Docs During Refactor

| Task                              | Read First                               |
| --------------------------------- | ---------------------------------------- |
| Add canonical state/control types | `04_DATA_MODEL.md`, `ADR-003`, `ADR-004` |
| Implement ODE or MuJoCo engine    | `05_ENGINE_INTERFACE.md`, `ADR-001`      |
| Wrap CC-MPC solver                | `06_CONTROLLER_INTERFACE.md`             |
| Implement runtime loop            | `03_RUNTIME_FLOW.md`, `ADR-002`          |
| Add scenario YAML loader          | `07_SCENARIO_CONFIG.md`                  |
| Add logger or metrics             | `08_LOGGING_AND_METRICS.md`              |
| Add tests                         | `09_VALIDATION_PLAN.md`                  |
| Make claims about results         | `10_KNOWN_LIMITATIONS.md`                |
| Plan implementation phases        | `11_REFACTOR_PLAN.md`                    |

---

## 6. Rules for Updating Design Docs

When changing an architectural contract, update the design docs first.

Examples of architectural contract changes:

```text
changing State9 ordering
changing ControlCommand4 ordering
changing engine command type policy
changing runtime threading model
changing scenario YAML schema
changing required log columns
changing validation acceptance criteria
```

Required update process:

```text
1. Update the relevant design document.
2. Update or add an ADR if the decision is architectural.
3. Update traceability in requirements or validation docs if needed.
4. Update tests.
5. Only then update implementation.
```

---

## 7. ADR Policy

Architecture Decision Records are stored in:

```text
docs/design/ADR/
```

Use an ADR when a decision:

```text
affects module boundaries
affects public data contracts
affects runtime architecture
affects engine/controller separation
affects long-term maintainability
has meaningful alternatives
```

Current ADRs:

| ADR                                      | Decision                                             |
| ---------------------------------------- | ---------------------------------------------------- |
| `ADR-001-engine-abstraction.md`          | Use `PhysicsEngine` abstraction                      |
| `ADR-002-single-thread-vs-mpc-thread.md` | Use deterministic single-thread runtime as reference |
| `ADR-003-state-vector-definition.md`     | Use `State9` as canonical public state               |
| `ADR-004-control-command-definition.md`  | Use `ControlCommand4` as canonical controller output |

---

## 8. Definition of Design Complete

The design documentation is considered complete enough to start implementation when:

```text
vision is documented
requirements are documented
architecture is documented
data contracts are documented
engine interface is documented
controller interface is documented
runtime flow is documented
scenario config is documented
logging schema is documented
validation plan is documented
known limitations are documented
refactor plan is documented
core ADRs are documented
```

At this point, implementation can begin with:

```text
Phase 1: Baseline Capture
Phase 2: Canonical Types
```

as defined in `11_REFACTOR_PLAN.md`.

---

## 9. Summary

This folder defines the design source of truth for the refactored quadrotor CC-MPC simulation.

The most important principles are:

```text
explicit data contracts
modular architecture
engine/controller separation
deterministic runtime first
structured scenario config
stable logging
validation-driven development
honest limitations
incremental refactor
```

The implementation shall follow these documents rather than treating existing demo scripts as the final architecture.
