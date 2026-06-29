# 11_REFACTOR_PLAN.md

> Status: Draft
> Scope: Ideal implementation plan after design freeze
> Project: Quadrotor CC-MPC Simulation
> Related documents:
>
> * `00_SIMULATION_VISION.md`
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
> * `ADR/ADR-001-engine-abstraction.md`
> * `ADR/ADR-002-single-thread-vs-mpc-thread.md`
> * `ADR/ADR-003-state-vector-definition.md`
> * `ADR/ADR-004-control-command-definition.md`

---

## 1. Purpose

This document defines the refactor plan for the quadrotor CC-MPC simulation.

The purpose is to convert the current demo-oriented implementation into a modular, testable, research-grade simulation architecture.

The refactor shall preserve existing mathematical behavior where possible, while replacing implicit script-level coupling with explicit interfaces.

The final architecture shall conform to:

```text
State9
ControlCommand4
ActuatorCommand4
PhysicsEngine
Controller
SimulationRuntime
ScenarioConfig
LogRecord
RunSummary
```

---

## 2. Refactor Goals

The refactor has the following goals.

### 2.1 Make data contracts explicit

The code shall use canonical data types:

```text
State9 = [x, y, z, vx, vy, vz, roll, pitch, yaw]

ControlCommand4 = [phi_c, theta_c, vz_c, psi_dot_c]

ActuatorCommand4 = [T1, T2, T3, T4]
```

No public module shall use raw ambiguous arrays without documented ordering.

---

### 2.2 Separate runtime orchestration from module internals

The current demo scripts mix several responsibilities:

```text
scenario loading
controller creation
physics stepping
MuJoCo setup
logging
plotting
termination logic
CLI parsing
```

After refactor, these responsibilities shall be separated.

The runtime shall orchestrate modules, not implement their internals.

---

### 2.3 Make ODE and MuJoCo replaceable

The refactored system shall support:

```text
ODEPhysicsEngine
MuJoCoPhysicsEngine
```

through the same `PhysicsEngine` interface.

The controller shall not depend on which engine is used.

---

### 2.4 Make logging stable and analysis-ready

Each run shall produce:

```text
metadata.json
steps.csv
summary.json
optional debug files
```

The log schema shall be stable, versioned, and shared across ODE and MuJoCo runs.

---

### 2.5 Make validation part of the refactor

Every phase shall have tests and acceptance criteria.

No phase is considered complete if it only “runs once”.

---

## 3. Refactor Non-Goals

This refactor shall not attempt to:

```text
rewrite the entire CC-MPC mathematics
replace CVXPY/CLARABEL immediately
implement real camera perception
implement real VIO
prove formal stability
prove recursive feasibility
make the system flight-certified
```

Those are future research or engineering milestones.

---

## 4. Refactor Strategy

The refactor shall be incremental.

The project shall not perform one large rewrite.

Recommended strategy:

```text
1. Freeze design contracts.
2. Add canonical types.
3. Add tests around current behavior.
4. Introduce interfaces.
5. Wrap existing modules behind interfaces.
6. Introduce deterministic runtime.
7. Migrate scenario config.
8. Migrate logging.
9. Port ODE demo.
10. Port MuJoCo demo.
11. Add validation and regression tests.
12. Retire demo scripts as architecture sources.
```

Each phase shall keep the repository runnable.

---

## 5. Target Architecture Snapshot

Target package layout:

```text
quadrotor_ccmpc/
├── ccmpc/
│   ├── __init__.py
│   ├── types.py
│   ├── dynamics.py
│   ├── linearization.py
│   ├── uncertainty.py
│   ├── obstacle.py
│   ├── mixer.py
│   ├── utils.py
│   └── controllers/
│       ├── __init__.py
│       ├── ccmpc_controller.py
│       ├── fallback_controller.py
│       └── solver_adapter.py
│
├── simulation/
│   ├── __init__.py
│   ├── app.py
│   ├── config/
│   ├── runtime/
│   ├── engines/
│   ├── controllers/
│   ├── estimation/
│   ├── obstacles/
│   ├── logging/
│   └── rendering/
│
├── config/
│   ├── controller/
│   ├── runtime/
│   ├── engines/
│   ├── scenarios/
│   └── logging/
│
├── models/
├── scripts/
├── tests/
└── docs/
```

---

## 6. Migration Principles

### 6.1 Preserve behavior before improving behavior

Do not change the controller mathematics and architecture at the same time.

Bad:

```text
rewrite solver
change dynamics
change runtime
change logging
change config
```

in one phase.

Good:

```text
wrap current solver behind Controller interface first
then improve solver internals later
```

---

### 6.2 Tests before deletion

Before deleting or replacing old script logic, add tests or golden logs that capture expected behavior.

---

### 6.3 Adapters before direct rewrites

If current code uses a legacy format, create an adapter first.

Examples:

```text
legacy scenario YAML -> ScenarioConfig
legacy logger fields -> canonical LogRecord
MuJoCo qpos/qvel -> State9
```

---

### 6.4 Demo scripts become thin entry points

After refactor, scripts shall only:

```text
parse CLI
load AppConfig
call SimulationApp.run()
```

They shall not own architecture.

---

## 7. Phase Overview

| Phase | Name                      | Main Outcome                                                |
| ----: | ------------------------- | ----------------------------------------------------------- |
|     0 | Design Freeze             | Docs and ADRs accepted                                      |
|     1 | Baseline Capture          | Current behavior captured before refactor                   |
|     2 | Canonical Types           | `State9`, `ControlCommand4`, `ActuatorCommand4` implemented |
|     3 | Config Layer              | Canonical scenario/config loader implemented                |
|     4 | Engine Interface          | ODE and MuJoCo wrapped behind `PhysicsEngine`               |
|     5 | Controller Interface      | CC-MPC wrapped behind `Controller`                          |
|     6 | Logging Layer             | `LogRecord`, `CSVLogger`, `RunSummary` implemented          |
|     7 | Deterministic Runtime     | Single-thread runtime implemented                           |
|     8 | ODE Migration             | ODE demo runs through new runtime                           |
|     9 | MuJoCo Migration          | MuJoCo headless runtime works through engine/mixer          |
|    10 | Validation Suite          | Unit/integration/regression tests completed                 |
|    11 | Cleanup                   | Legacy scripts simplified or removed                        |
|    12 | Optional Threaded Runtime | Threaded MPC runtime added later                            |

---

# Phase 0: Design Freeze

## 0.1 Objective

Freeze the design documents before large code changes.

## 0.2 Required documents

The following documents shall exist:

```text
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

Required ADRs:

```text
ADR-001-engine-abstraction.md
ADR-002-single-thread-vs-mpc-thread.md
ADR-003-state-vector-definition.md
ADR-004-control-command-definition.md
```

## 0.3 Tasks

```text
review all design docs
check consistency of State9 ordering
check consistency of ControlCommand4 ordering
check ODE/MuJoCo command policy
check runtime mode decision
check logging schema names
check validation acceptance criteria
```

## 0.4 Acceptance criteria

```text
all core docs exist
all ADRs exist
no conflicting state ordering
no conflicting control ordering
ODE/MuJoCo boundary is clear
next phase can start without architectural ambiguity
```

---

# Phase 1: Baseline Capture

## 1.1 Objective

Capture current behavior before refactoring.

This phase prevents accidental behavior loss.

## 1.2 Tasks

Run current demos and collect outputs.

Recommended commands:

```bash
python sim_demo_nosim.py --engine ode --headless --log baseline_ode.csv
```

Optional:

```bash
python sim_demo_nosim.py --engine mujoco --headless --log baseline_mujoco.csv
```

If MuJoCo GUI is available:

```bash
python sim_demo_mujoco.py
```

## 1.3 Capture artifacts

Store:

```text
baseline_ode.csv
baseline_mujoco.csv
stdout logs
config files used
MuJoCo warnings if any
solver status
final goal distance
collision status
```

Recommended directory:

```text
validation/baselines/pre_refactor/
```

## 1.4 Add baseline summary

Create:

```text
validation/baselines/pre_refactor/README.md
```

with:

```text
date
git commit
commands used
environment
whether ODE ran successfully
whether MuJoCo ran successfully
known warnings
final metrics
```

## 1.5 Acceptance criteria

```text
current ODE behavior is captured
current MuJoCo behavior is captured if environment supports it
baseline files are committed or archived
known current failures are documented
```

---

# Phase 2: Canonical Types

## 2.1 Objective

Introduce canonical data types without changing solver behavior.

## 2.2 New files

```text
ccmpc/types.py
```

Optional later:

```text
ccmpc/validation.py
```

## 2.3 Types to implement

```text
State9
Goal3
ControlCommand4
ActuatorCommand4
Trajectory9
ControlTrajectory4
Gamma9x9
Sigma3x3
```

## 2.4 Required behavior

Each type shall validate:

```text
shape
finite values
field ordering
basic physical constraints where applicable
```

Example constraints:

```text
State9.shape == (9,)
ControlCommand4.shape == (4,)
ActuatorCommand4.shape == (4,)
ActuatorCommand4 values >= 0
Gamma9x9.shape == (9, 9)
Gamma9x9 symmetric PSD
```

## 2.5 Migration rule

At this phase, existing functions may still accept raw NumPy arrays internally.

But public boundaries shall begin documenting canonical types.

## 2.6 Tests

Create:

```text
tests/unit/test_types.py
```

Required tests:

```text
test_state9_valid
test_state9_wrong_shape
test_state9_nan
test_state9_field_order

test_control_command4_valid
test_control_command4_wrong_shape
test_control_command4_nan
test_control_command4_field_order

test_actuator_command4_valid
test_actuator_command4_negative_rejected

test_gamma9x9_valid_psd
test_gamma9x9_non_symmetric_rejected
```

## 2.7 Acceptance criteria

```text
canonical types implemented
unit tests pass
existing demos still run
no module changes state/control ordering
```

---

# Phase 3: Config Layer

## 3.1 Objective

Introduce canonical config loading and scenario validation.

## 3.2 New files

```text
simulation/config/schema.py
simulation/config/loader.py
simulation/config/legacy.py
simulation/config/validation.py
```

## 3.3 Config objects

Implement:

```text
ScenarioConfig
WorldConfig
GoalConfig
ObstacleSpec
SuccessConfig
TerminationConfig
RuntimeOverrides
```

## 3.4 Legacy adapter

Support current flat scenario format temporarily.

Mapping:

```text
start -> initial_state.state9
goal -> goal.position
goal_threshold -> success.goal_threshold
sim_timestep -> runtime_overrides.sim_dt
obstacles -> obstacles
```

## 3.5 New scenario directory

Create:

```text
config/scenarios/
```

Migrate current scenarios to:

```text
config/scenarios/default.yaml
config/scenarios/static_obstacles.yaml
config/scenarios/corridor_static.yaml
```

## 3.6 Tests

Create:

```text
tests/unit/test_scenario_config.py
tests/interface/test_scenario_loader_interface.py
```

Required tests:

```text
test_load_canonical_scenario
test_load_legacy_scenario
test_reject_invalid_state9_length
test_reject_invalid_goal_length
test_reject_duplicate_obstacle_id
test_reject_negative_obstacle_size
test_runtime_override_positive
```

## 3.7 Acceptance criteria

```text
canonical scenario loads
legacy scenario loads with warning
invalid scenario fails before runtime
runtime no longer parses raw YAML directly
```

---

# Phase 4: Engine Interface

## 4.1 Objective

Wrap physics backends behind a common `PhysicsEngine` interface.

## 4.2 New files

```text
simulation/engines/base.py
simulation/engines/metadata.py
simulation/engines/factory.py
simulation/engines/ode_engine.py
simulation/engines/mujoco_engine.py
simulation/engines/adapters/mujoco_state_adapter.py
simulation/engines/adapters/mujoco_actuator_adapter.py
```

## 4.3 Core types

Implement:

```text
PhysicsEngine
EngineType
EngineCommandType
EngineMetadata
StepResult
```

## 4.4 ODE engine

`ODEPhysicsEngine` shall:

```text
consume ControlCommand4
own current State9
own engine time
step reduced dynamics
return StepResult
```

## 4.5 MuJoCo engine

`MuJoCoPhysicsEngine` shall:

```text
consume ActuatorCommand4
expose State9
hide qpos/qvel
own MuJoCo model/data
return StepResult
```

## 4.6 Adapter tests

Required tests:

```text
test_state9_to_mujoco_roundtrip
test_mujoco_to_state9_roundtrip
test_quaternion_euler_roundtrip
test_mujoco_engine_rejects_control_command4_without_mixer
```

## 4.7 Interface tests

Create:

```text
tests/interface/test_engine_interface.py
```

Required tests:

```text
test_ode_engine_metadata
test_ode_engine_reset
test_ode_engine_step_returns_state9
test_ode_engine_accepts_control_command4

test_mujoco_engine_metadata
test_mujoco_engine_reset_returns_state9
test_mujoco_engine_accepts_actuator_command4
```

## 4.8 Acceptance criteria

```text
ODE engine works through PhysicsEngine
MuJoCo engine exposes State9
MuJoCo internals do not leak to runtime/controller
wrong command type is rejected
```

---

# Phase 5: Controller Interface

## 5.1 Objective

Wrap CC-MPC solver behind a stable `Controller` interface.

## 5.2 New files

```text
simulation/controllers/base.py
simulation/controllers/metadata.py
simulation/controllers/factory.py
ccmpc/controllers/ccmpc_controller.py
ccmpc/controllers/fallback_controller.py
ccmpc/controllers/solver_adapter.py
```

## 5.3 Core types

Implement:

```text
Controller
ControllerType
ControllerMetadata
ControllerInput
ControllerOutput
ControllerDiagnostics
ControllerStatus
```

## 5.4 CC-MPC wrapper

`CCMPCController` shall:

```text
receive ControllerInput
call existing CCMPC.solve()
return ControllerOutput
return first ControlCommand4
include predicted trajectory
include control trajectory
include diagnostics
handle fallback
```

## 5.5 Do not rewrite solver yet

At this phase, keep current CC-MPC mathematical implementation.

Only wrap it.

## 5.6 Tests

Create:

```text
tests/interface/test_controller_interface.py
tests/unit/test_fallback_controller.py
```

Required tests:

```text
test_controller_accepts_controller_input
test_controller_returns_controller_output
test_controller_output_command_is_control_command4
test_controller_diagnostics_present
test_controller_does_not_return_actuator_command4
test_fallback_returns_control_command4
```

## 5.7 Acceptance criteria

```text
CCMPCController implements Controller
controller output command is ControlCommand4
controller does not know engine type
fallback is available
existing CCMPC tests still pass
```

---

# Phase 6: Logging Layer

## 6.1 Objective

Replace script-local logging with canonical logging.

## 6.2 New files

```text
simulation/logging/base.py
simulation/logging/records.py
simulation/logging/schema.py
simulation/logging/csv_logger.py
simulation/logging/jsonl_logger.py
simulation/logging/memory_logger.py
simulation/logging/null_logger.py
simulation/logging/summary.py
```

## 6.3 Core types

Implement:

```text
LogRecord
RunMetadata
RunSummary
LogEvent
StepMetrics
```

## 6.4 Required outputs

Each run shall produce:

```text
metadata.json
steps.csv
summary.json
```

Optional:

```text
events.jsonl
controller_debug.jsonl
trajectories.npz
obstacle_debug.jsonl
```

## 6.5 CSV schema

Create stable CSV schema with canonical names:

```text
true_pos_x
true_pos_y
true_pos_z
true_vel_x
true_vel_y
true_vel_z
true_roll
true_pitch
true_yaw
control_phi_c
control_theta_c
control_vz_c
control_psi_dot_c
actuator_T1
actuator_T2
actuator_T3
actuator_T4
controller_solve_time_ms
goal_distance
collision_flag
```

## 6.6 Tests

Create:

```text
tests/unit/test_log_record.py
tests/unit/test_metrics.py
tests/interface/test_logger_interface.py
```

Required tests:

```text
test_log_record_valid
test_log_record_rejects_nan_true_state
test_csv_header_matches_schema
test_csv_rows_have_constant_column_count
test_logger_writes_metadata
test_logger_writes_steps
test_logger_writes_summary
test_summary_matches_steps
```

## 6.7 Acceptance criteria

```text
logger is passive
runtime builds LogRecord
logger writes required files
CSV header is stable
summary metrics match step logs
```

---

# Phase 7: Deterministic Runtime

## 7.1 Objective

Implement the reference deterministic single-thread runtime.

## 7.2 New files

```text
simulation/runtime/app.py
simulation/runtime/loop.py
simulation/runtime/timing.py
simulation/runtime/dispatch.py
simulation/runtime/termination.py
simulation/runtime/metrics.py
simulation/runtime/errors.py
```

## 7.3 Runtime responsibilities

The runtime shall:

```text
initialize modules
reset engine
run deterministic loop
call estimator
predict obstacles
call controller when due
dispatch command to engine
call mixer if needed
step physics
build log record
record logs
render if enabled
check termination
return RunSummary
```

## 7.4 Reference step order

```text
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

## 7.5 Command dispatch

ODE path:

```text
ControlCommand4 -> ODEPhysicsEngine.step()
```

MuJoCo path:

```text
ControlCommand4 -> LowLevelMixer -> ActuatorCommand4 -> MuJoCoPhysicsEngine.step()
```

## 7.6 Tests

Create:

```text
tests/integration/test_runtime_ode.py
tests/integration/test_runtime_dispatch.py
```

Required tests:

```text
test_runtime_single_step_ode
test_runtime_controller_due_logic
test_runtime_zero_order_hold
test_runtime_dispatch_control_to_ode
test_runtime_dispatch_actuator_to_mujoco
test_runtime_logs_each_step
test_runtime_terminates_on_goal
```

## 7.7 Acceptance criteria

```text
deterministic ODE runtime runs one step
deterministic ODE runtime runs full empty_world scenario
controller due logic works
zero-order hold works
logging works
no old demo runtime logic required
```

---

# Phase 8: ODE Migration

## 8.1 Objective

Run existing ODE simulation through the new architecture.

## 8.2 Tasks

```text
create ODE engine config
create default runtime config
create default scenario config
create controller config
create logging config
create run_simulation.py entry point
```

## 8.3 New CLI

Recommended command:

```bash
python scripts/run_simulation.py \
  --scenario config/scenarios/default.yaml \
  --engine config/engines/ode.yaml \
  --controller config/controller/ccmpc.yaml \
  --runtime config/runtime/default.yaml \
  --logging config/logging/csv.yaml
```

Optional shorthand:

```bash
python scripts/run_simulation.py --scenario default --engine ode
```

## 8.4 Compare with baseline

Compare:

```text
final goal distance
trajectory shape
solver status
solve time
goal reached flag
collision flag
```

Exact equality is not required if logging/runtime changed, but behavior should remain qualitatively consistent.

## 8.5 Tests

Create:

```text
tests/regression/test_empty_world_ode.py
tests/regression/test_static_obstacle_ode.py
```

Required tests:

```text
test_empty_world_ode_goal_reached
test_empty_world_ode_no_nan
test_static_obstacle_ode_no_collision
test_ode_log_schema_regression
```

## 8.6 Acceptance criteria

```text
ODE default scenario runs through new runtime
logs are generated
summary is generated
empty_world reaches goal
no NaN or Inf
old sim_demo_nosim.py can be reduced to thin wrapper or marked legacy
```

---

# Phase 9: MuJoCo Migration

## 9.1 Objective

Run MuJoCo through the new architecture without leaking MuJoCo internals.

## 9.2 Tasks

```text
create MuJoCoPhysicsEngine
create MuJoCo state adapter
create MuJoCo actuator adapter
create mixer integration
create headless MuJoCo runtime
log actuator commands
validate reset/get_state round-trip
```

## 9.3 Initial scope

Initial MuJoCo migration shall focus on:

```text
headless stepping
State9 output
ActuatorCommand4 input
logging
basic smoke tests
```

Viewer and threaded mode are secondary.

## 9.4 Do not start with threaded mode

The first MuJoCo refactor target shall be deterministic single-thread headless mode.

Threaded viewer mode shall be postponed until Phase 12.

## 9.5 Tests

Create:

```text
tests/interface/test_mujoco_engine_interface.py
tests/integration/test_runtime_mujoco.py
```

Required tests:

```text
test_mujoco_reset_state9_roundtrip
test_mujoco_step_returns_state9
test_mujoco_requires_actuator_command4
test_runtime_mujoco_uses_mixer
test_runtime_mujoco_logs_actuator_command
test_controller_does_not_receive_qpos_qvel
```

## 9.6 Acceptance criteria

```text
MuJoCo engine can reset from State9
MuJoCo engine can step with ActuatorCommand4
MuJoCo output is State9
runtime uses mixer before MuJoCo step
actuator_T1..T4 are logged
MuJoCo tests pass in MuJoCo-enabled environment
```

---

# Phase 10: Validation Suite

## 10.1 Objective

Implement the validation plan defined in `09_VALIDATION_PLAN.md`.

## 10.2 Required validation layers

```text
static checks
unit tests
interface tests
integration tests
regression tests
scenario validation
performance validation
reproducibility validation
```

## 10.3 Minimum merge gate

A refactor PR shall not be merged unless the following pass:

```text
ruff check
unit tests
interface tests
ODE runtime integration test
scenario loader tests
logging schema tests
empty_world_ode regression
```

## 10.4 Validation commands

Recommended:

```bash
ruff check .
pytest tests/unit -v
pytest tests/interface -v
pytest tests/integration -v
pytest tests/regression -v
```

Full validation:

```bash
pytest tests -v --tb=short
```

Optional coverage:

```bash
pytest tests --cov=ccmpc --cov=simulation --cov-report=html
```

## 10.5 Acceptance criteria

```text
all required tests pass
validation report can be generated
summary metrics match logs
reproducibility test passes for deterministic ODE runtime
no NaN in reference scenarios
```

---

# Phase 11: Cleanup and Legacy Retirement

## 11.1 Objective

Remove architectural dependence on demo scripts.

## 11.2 Tasks

```text
mark old demo scripts as legacy
replace duplicated runtime code with SimulationRuntime
replace script-local SimLogger with CSVLogger
replace raw YAML parsing with config loader
replace direct engine branching with engine factory
replace plotting logic with renderer abstraction
```

## 11.3 Script policy

Allowed final scripts:

```text
scripts/run_simulation.py
scripts/benchmark.py
scripts/replay_log.py
scripts/convert_legacy_log.py
```

Legacy scripts may remain temporarily:

```text
sim_demo_nosim.py
sim_demo_mujoco.py
```

but they shall become wrappers around the new runtime or be moved to:

```text
scripts/legacy/
```

## 11.4 Deletion criteria

A legacy code path may be deleted only when:

```text
new runtime covers its functionality
tests cover the replacement
baseline behavior is documented
no docs refer to it as canonical architecture
```

## 11.5 Acceptance criteria

```text
no core architecture depends on demo scripts
no duplicated runtime loop remains
no duplicated logger remains
no raw MuJoCo qpos/qvel reaches controller
README points to new run_simulation.py
```

---

# Phase 12: Optional Threaded Runtime

## 12.1 Objective

Add optional threaded MPC runtime after deterministic runtime is stable.

## 12.2 Scope

Threaded runtime shall support:

```text
physics thread
controller thread
shared immutable snapshots
state sequence id
command sequence id
command age logging
stale command policy
safe shutdown
```

## 12.3 New files

```text
simulation/runtime/threaded.py
simulation/runtime/shared_state.py
```

## 12.4 Required data types

```text
StateSnapshot
CommandSnapshot
SharedRuntimeData
RuntimeThreadError
```

## 12.5 Required log fields

```text
state_time
command_time
command_age_ms
state_sequence_id
command_sequence_id
controller_input_state_sequence_id
stale_command_flag
```

## 12.6 Tests

Create:

```text
tests/integration/test_threaded_runtime.py
```

Required tests:

```text
test_threaded_snapshot_atomicity
test_threaded_command_sequence_id
test_threaded_state_sequence_id
test_threaded_stale_command_detection
test_threaded_safe_shutdown
```

## 12.7 Acceptance criteria

```text
threaded runtime uses same Controller and PhysicsEngine interfaces
no partial state reads
no partial command reads
stale command is logged
thread failure triggers safe shutdown
threaded mode is not default
```

---

## 13. Suggested Work Breakdown

Recommended implementation order for a single developer:

```text
Week 1:
  Phase 0
  Phase 1
  Phase 2

Week 2:
  Phase 3
  Phase 4 ODE only

Week 3:
  Phase 5
  Phase 6

Week 4:
  Phase 7
  Phase 8

Week 5:
  Phase 9 MuJoCo headless
  Phase 10 initial validation

Week 6:
  Phase 11 cleanup
  documentation update
  regression stabilization
```

This schedule is approximate.
Validation quality is more important than speed.

---

## 14. Branching Strategy

Recommended branches:

```text
main
refactor/design-docs
refactor/types
refactor/config
refactor/engine-interface
refactor/controller-interface
refactor/logging
refactor/runtime
refactor/ode-migration
refactor/mujoco-migration
refactor/validation
```

Each branch should be small enough to review.

---

## 15. Pull Request Policy

Each PR shall include:

```text
summary of changes
related design document
related ADR if applicable
tests added
tests run
known limitations
migration notes
```

Recommended PR checklist:

```text
[ ] Follows DATA_MODEL.md
[ ] Does not change State9 ordering
[ ] Does not change ControlCommand4 ordering
[ ] Adds or updates tests
[ ] Updates docs if contract changed
[ ] Keeps demo or runtime runnable
[ ] No new hidden dependency on MuJoCo internals
[ ] No logger side effects on simulation
```

---

## 16. Refactor Risk Register

| Risk                                         | Impact                      | Mitigation                                   |
| -------------------------------------------- | --------------------------- | -------------------------------------------- |
| State ordering accidentally changes          | Invalid controller behavior | Type tests and field-order tests             |
| Command ordering accidentally changes        | Wrong control applied       | `ControlCommand4` tests                      |
| MuJoCo qpos leaks to controller              | Engine coupling             | Adapter/interface tests                      |
| Logging schema changes silently              | Analysis breaks             | Schema regression tests                      |
| Solver behavior changes during wrapper phase | Hard-to-debug regression    | Preserve solver internals first              |
| Runtime becomes too large                    | Maintenance issue           | Split timing, dispatch, metrics, termination |
| MuJoCo migration unstable                    | Delays refactor             | Make MuJoCo phase optional until ODE stable  |
| Threading introduces nondeterminism          | Poor reproducibility        | Keep deterministic runtime as default        |

---

## 17. Definition of Done

The refactor is done when:

```text
canonical types exist and are tested
scenario config loader exists and is tested
PhysicsEngine interface exists
ODEPhysicsEngine works through interface
MuJoCoPhysicsEngine works through interface or is clearly marked experimental
Controller interface exists
CCMPCController works through interface
SimulationRuntime deterministic mode runs ODE scenario
CSVLogger writes metadata, steps, summary
Validation tests pass
README points to new architecture
demo scripts are legacy or thin wrappers
```

---

## 18. Final Acceptance Criteria

The refactored system shall satisfy:

```text
python scripts/run_simulation.py --scenario empty_world --engine ode
```

and produce:

```text
metadata.json
steps.csv
summary.json
```

with:

```text
valid State9 columns
valid ControlCommand4 columns
controller diagnostics
goal distance
termination reason
no NaN
```

The ODE empty-world regression shall pass:

```text
success == true
collision == false
nan_detected == false
final_goal_distance <= goal_threshold
```

The static-obstacle ODE regression shall pass:

```text
collision == false
nan_detected == false
min_obstacle_margin >= configured tolerance
```

MuJoCo support shall not be claimed complete until:

```text
MuJoCo reset/get_state round-trip passes
MuJoCo receives ActuatorCommand4
runtime uses mixer
actuator commands are logged
no qpos/qvel reaches controller
```

---

## 19. Deliverables

Final refactor deliverables:

```text
ccmpc/types.py
simulation/config/
simulation/engines/
simulation/controllers/
simulation/runtime/
simulation/logging/
simulation/estimation/
simulation/obstacles/
simulation/rendering/
config/scenarios/
config/runtime/
config/engines/
config/controller/
config/logging/
scripts/run_simulation.py
tests/unit/
tests/interface/
tests/integration/
tests/regression/
validation/reports/
```

Documentation deliverables:

```text
docs/design/
docs/design/ADR/
README.md updated with new usage
```

---

## 20. Suggested First Commit

The first implementation commit after docs should be:

```text
feat(types): add canonical State9 and command types
```

Files:

```text
ccmpc/types.py
tests/unit/test_types.py
```

It should not modify:

```text
ccmpc/ccmpc.py
ccmpc/dynamics.py
sim_demo_nosim.py
sim_demo_mujoco.py
```

Reason:

```text
Start with low-risk canonical contracts before touching runtime behavior.
```

---

## 21. Suggested Second Commit

The second implementation commit should be:

```text
feat(config): add canonical scenario loader and legacy adapter
```

Files:

```text
simulation/config/schema.py
simulation/config/loader.py
simulation/config/legacy.py
simulation/config/validation.py
tests/unit/test_scenario_config.py
```

Reason:

```text
Runtime needs typed scenarios before orchestration can be cleaned up.
```

---

## 22. Suggested Third Commit

The third implementation commit should be:

```text
feat(engine): add PhysicsEngine interface and ODE engine wrapper
```

Files:

```text
simulation/engines/base.py
simulation/engines/metadata.py
simulation/engines/ode_engine.py
simulation/engines/factory.py
tests/interface/test_engine_interface.py
```

Reason:

```text
ODE is simpler than MuJoCo and should become the first stable backend.
```

---

## 23. Suggested Fourth Commit

The fourth implementation commit should be:

```text
feat(controller): wrap CCMPC behind Controller interface
```

Files:

```text
simulation/controllers/base.py
simulation/controllers/metadata.py
simulation/controllers/factory.py
ccmpc/controllers/ccmpc_controller.py
ccmpc/controllers/fallback_controller.py
tests/interface/test_controller_interface.py
```

Reason:

```text
Runtime can then treat the controller as a stable black box.
```

---

## 24. Suggested Fifth Commit

The fifth implementation commit should be:

```text
feat(runtime): add deterministic simulation runtime
```

Files:

```text
simulation/runtime/loop.py
simulation/runtime/dispatch.py
simulation/runtime/timing.py
simulation/runtime/termination.py
simulation/runtime/metrics.py
tests/integration/test_runtime_ode.py
```

Reason:

```text
This is the first point where the refactored system becomes runnable end-to-end.
```

---

## 25. Summary

The refactor shall proceed incrementally.

The recommended order is:

```text
design freeze
baseline capture
canonical types
scenario config
engine interface
controller interface
logging
deterministic runtime
ODE migration
MuJoCo migration
validation
cleanup
optional threaded runtime
```

The reference target is not a prettier demo script.

The reference target is a research-grade simulation system with:

```text
clear data contracts
engine abstraction
controller abstraction
deterministic runtime
stable logging
validated behavior
documented limitations
```

No phase is complete without tests and acceptance criteria.

---

## 26. Related Documents

```text
docs/design/00_SIMULATION_VISION.md
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

docs/design/ADR/ADR-001-engine-abstraction.md
docs/design/ADR/ADR-002-single-thread-vs-mpc-thread.md
docs/design/ADR/ADR-003-state-vector-definition.md
docs/design/ADR/ADR-004-control-command-definition.md
```
