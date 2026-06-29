# 10_KNOWN_LIMITATIONS.md

> Status: Draft
> Scope: Ideal design after refactor
> Project: Quadrotor CC-MPC Simulation
> Related documents:
>
> * `02_ARCHITECTURE.md`
> * `03_RUNTIME_FLOW.md`
> * `04_DATA_MODEL.md`
> * `05_ENGINE_INTERFACE.md`
> * `06_CONTROLLER_INTERFACE.md`
> * `07_SCENARIO_CONFIG.md`
> * `08_LOGGING_AND_METRICS.md`
> * `09_VALIDATION_PLAN.md`
> * `11_REFACTOR_PLAN.md`
> * `ADR/ADR-001-engine-abstraction.md`
> * `ADR/ADR-002-single-thread-vs-mpc-thread.md`
> * `ADR/ADR-003-state-vector-definition.md`
> * `ADR/ADR-004-control-command-definition.md`

---

## 1. Purpose

This document records the known limitations of the refactored quadrotor CC-MPC simulation.

The purpose is not to weaken the project.

The purpose is to make the research scope explicit.

A limitation is acceptable if it is:

```text
known
documented
validated where possible
visible in logs
not silently presented as solved
```

A limitation is dangerous if it is:

```text
hidden
undocumented
not tested
silently mixed into another module
reported as a research result without qualification
```

This document shall be updated whenever a new limitation is discovered.

---

## 2. Limitation Categories

Known limitations are grouped into the following categories:

```text
model limitations
state representation limitations
control command limitations
physics engine limitations
MuJoCo limitations
controller and solver limitations
chance constraint limitations
uncertainty limitations
obstacle model limitations
estimation and sensing limitations
runtime limitations
logging limitations
validation limitations
software engineering limitations
research-scope limitations
```

Each limitation should include:

```text
description
impact
current status
mitigation
future work
```

---

## 3. Summary Table

| ID      | Limitation                                           | Severity | Status               |
| ------- | ---------------------------------------------------- | -------: | -------------------- |
| LIM-001 | Reduced 9-state dynamics, not full Newton-Euler      |     High | Accepted             |
| LIM-002 | Euler attitude representation has singularity        |   Medium | Accepted             |
| LIM-003 | Small-angle / near-hover assumptions                 |     High | Accepted             |
| LIM-004 | `ControlCommand4` is high-level, not actuator-level  |     High | Managed by mixer     |
| LIM-005 | ODE and MuJoCo are not physically equivalent         |     High | Requires validation  |
| LIM-006 | MuJoCo adapter may introduce conversion error        |   Medium | Requires tests       |
| LIM-007 | Mixer is approximate                                 |     High | Requires calibration |
| LIM-008 | Chance constraints assume Gaussian uncertainty       |     High | Accepted             |
| LIM-009 | Chance constraints are linearized                    |     High | Accepted             |
| LIM-010 | Covariance propagation uses EKF linearization        |   Medium | Accepted             |
| LIM-011 | Obstacle motion model is constant velocity           |   Medium | Accepted             |
| LIM-012 | No real perception pipeline in initial refactor      |     High | Out of scope         |
| LIM-013 | No real VIO pipeline in initial refactor             |     High | Out of scope         |
| LIM-014 | Solver may be infeasible or late                     |     High | Managed by fallback  |
| LIM-015 | Deterministic runtime is not real deployment timing  |   Medium | Accepted             |
| LIM-016 | Threaded runtime is optional and harder to reproduce |   Medium | Future work          |
| LIM-017 | No formal stability proof                            |     High | Out of scope         |
| LIM-018 | No formal recursive feasibility proof                |     High | Out of scope         |
| LIM-019 | Not flight-certified or hardware-safe                | Critical | Explicit non-goal    |
| LIM-020 | Logs cannot prove physical correctness alone         |   Medium | Requires validation  |

---

## 4. LIM-001: Reduced 9-State Dynamics

### Description

The project uses a reduced state vector:

```text
State9 = [x, y, z, vx, vy, vz, roll, pitch, yaw]
```

This is not the full Newton-Euler rigid-body state.

The full Newton-Euler model would include:

```text
position
linear velocity
quaternion or rotation matrix
body angular velocity
motor or actuator dynamics
```

The reduced model does not explicitly model:

```text
body angular rates p, q, r
motor speeds
rotor thrust dynamics
rotor drag torque
full inertia matrix effects
gyroscopic coupling
aerodynamic ground effect
battery voltage effects
wind disturbance
```

### Impact

The reduced model is suitable for high-level CC-MPC planning, but it is not a full physical simulation of a quadrotor.

It may be inaccurate for:

```text
aggressive maneuvers
large roll/pitch angles
fast yaw motion
high-speed flight
near-saturation actuator behavior
strong external disturbances
```

### Current status

Accepted for the initial refactor.

### Mitigation

The project shall:

```text
document State9 as the canonical planning state
validate ODE behavior in simple scenarios
separate ODE engine from MuJoCo engine
avoid claiming full rigid-body fidelity
log engine type and dynamics model type
```

### Future work

Possible future extension:

```text
add FullNewtonEulerState
add body-rate command model
add thrust/torque command interface
add NMPC controller for full rigid-body dynamics
```

---

## 5. LIM-002: Euler Angle Singularity

### Description

The canonical attitude representation uses Euler ZYX angles:

```text
roll
pitch
yaw
```

Euler angles are minimal and intuitive, but they have a singularity at:

$$
\theta = \pm \frac{\pi}{2}
$$

At this pitch angle, roll and yaw become coupled.

### Impact

This is acceptable for normal quadrotor navigation because the drone is expected to fly near hover with moderate roll and pitch.

It is not suitable for:

```text
flips
aerobatic flight
near-vertical pitch
large-angle attitude recovery
global attitude planning
```

### Current status

Accepted.

### Mitigation

The system shall:

```text
keep roll and pitch bounds small
validate angle bounds
use quaternion internally for MuJoCo adapter
test Euler/quaternion round-trip in safe attitude range
avoid validating near singularity as a normal case
```

### Future work

Possible future extension:

```text
use quaternion state for full rigid-body controller
add attitude representation adapter
add separate ADR for quaternion-based planning state
```

---

## 6. LIM-003: Small-Angle and Near-Hover Assumptions

### Description

The reduced dynamics are most accurate near hover and moderate attitude.

The horizontal acceleration model relies on roll and pitch producing horizontal motion in a simplified way.

The model assumes that high-level attitude commands are tracked by a lower-level onboard controller.

### Impact

The simulation may overestimate performance when:

```text
roll or pitch is large
thrust deviates significantly from hover thrust
vertical acceleration is aggressive
drag model is inaccurate
controller commands saturate
```

### Current status

Accepted.

### Mitigation

The project shall:

```text
enforce command bounds
enforce attitude bounds
log command saturation
log speed and altitude violations
validate only within intended operating envelope
```

Recommended bounds should remain conservative.

### Future work

Use full Newton-Euler or identified dynamics if the research scope shifts toward aggressive flight.

---

## 7. LIM-004: High-Level Control Command Is Not Rotor Thrust

### Description

The controller output is:

```text
ControlCommand4 = [phi_c, theta_c, vz_c, psi_dot_c]
```

This is a high-level command.

It is not:

```text
rotor thrust
body torque
PWM
motor speed
MuJoCo actuator ctrl
```

### Impact

If `ControlCommand4` is passed directly to an actuator-level physics engine, the simulation becomes physically meaningless.

For example:

```text
phi_c in radians could be misread as thrust
theta_c in radians could be misread as torque
vz_c in m/s could be misread as force
psi_dot_c in rad/s could be misread as yaw torque
```

### Current status

Managed by `ADR-004-control-command-definition.md`.

### Mitigation

The runtime shall enforce:

```text
ODE engine may consume ControlCommand4
MuJoCo rotor-force engine shall consume ActuatorCommand4
LowLevelMixer converts ControlCommand4 to ActuatorCommand4
logger records both command levels separately
```

### Future work

Add type-level checks so the wrong command type cannot be passed silently.

---

## 8. LIM-005: ODE and MuJoCo Are Not Physically Equivalent

### Description

The ODE engine and MuJoCo engine may expose the same public interface, but they do not necessarily simulate the same physics.

ODE engine:

```text
reduced-order
high-level command model
State9 directly
```

MuJoCo engine:

```text
rigid-body physics
quaternion state internally
actuator-level force input
contact and inertial effects
```

### Impact

A controller that performs well in ODE may not perform equally well in MuJoCo.

Differences may appear in:

```text
altitude response
yaw response
roll/pitch transient response
actuator saturation
numerical stability
goal tracking
obstacle avoidance margin
```

### Current status

Known high-risk limitation.

### Mitigation

The project shall:

```text
log engine_type
log controller dynamics type
validate ODE and MuJoCo separately
avoid claiming ODE/MuJoCo equivalence
define MuJoCo validation scenarios
compare trajectories qualitatively and quantitatively
```

### Future work

Possible future work:

```text
identify ODE model parameters from MuJoCo
calibrate mixer
derive equivalent high-level closed-loop model
add engine-pair validation report
```

---

## 9. LIM-006: MuJoCo State Adapter May Introduce Error

### Description

MuJoCo may use:

```text
qpos
qvel
quaternion
body angular velocity
```

while the public data model uses:

```text
State9 = [x, y, z, vx, vy, vz, roll, pitch, yaw]
```

Therefore, conversion is required:

```text
State9 -> qpos/qvel
qpos/qvel -> State9
```

### Impact

Adapter errors can cause:

```text
incorrect attitude
wrong yaw direction
wrong velocity frame
reset/get_state mismatch
controller receiving wrong state
unstable MuJoCo rollout
```

### Current status

Requires explicit tests.

### Mitigation

The project shall add:

```text
State9 to MuJoCo round-trip tests
Euler/quaternion round-trip tests
reset then get_state consistency test
quaternion normalization check
frame convention tests
```

### Future work

Add a strict adapter validation suite before using MuJoCo results in research claims.

---

## 10. LIM-007: Mixer Is Approximate

### Description

The low-level mixer converts:

```text
ControlCommand4 -> ActuatorCommand4
```

This conversion approximates the behavior of a real lower-level flight controller.

The real drone may use:

```text
inner-loop attitude controller
motor dynamics
rate feedback
thrust compensation
battery compensation
motor saturation logic
hardware-specific allocation
```

### Impact

MuJoCo behavior may be sensitive to mixer tuning.

Poor mixer tuning can cause:

```text
altitude drift
oscillation
yaw instability
roll/pitch overshoot
rotor saturation
NaN or extreme acceleration in physics
```

### Current status

Known high-risk limitation for MuJoCo.

### Mitigation

The project shall:

```text
log ActuatorCommand4
log saturation flags
validate hover behavior
validate roll/pitch/yaw response
separate mixer tests from engine tests
avoid hiding mixer inside controller
```

### Future work

Calibrate mixer against MuJoCo or real flight logs.

---

## 11. LIM-008: Gaussian Uncertainty Assumption

### Description

The chance-constrained formulation assumes Gaussian uncertainty.

This applies to:

```text
state estimation uncertainty
obstacle position uncertainty
motion disturbance uncertainty
```

### Impact

Real uncertainty may be:

```text
non-Gaussian
multi-modal
biased
heavy-tailed
correlated over time
affected by perception failures
```

If the Gaussian assumption is wrong, the stated probability bound may not match real collision probability.

### Current status

Accepted theoretical assumption.

### Mitigation

The project shall:

```text
clearly label chance constraints as Gaussian-based
log covariance values
validate formulas numerically
avoid claiming safety under arbitrary noise distributions
```

### Future work

Possible extensions:

```text
distributionally robust MPC
particle-based uncertainty propagation
scenario MPC
non-Gaussian obstacle tracking
```

---

## 12. LIM-009: Linearized Chance Constraints

### Description

Obstacle collision constraints are nonlinear.

The project uses a linearized half-space approximation to make the chance constraint tractable inside a QP.

### Impact

The approximation may be inaccurate when:

```text
robot is very close to obstacle
linearization point is poor
obstacle geometry is highly rotated or elongated
trajectory changes significantly between iterations
uncertainty is large
```

### Current status

Accepted.

### Mitigation

The project shall:

```text
log min_chance_constraint_margin
log min_obstacle_margin
validate known inside/on-boundary/outside points
use iterative MPC updates
use slack variables
reject or fallback for severe infeasibility
```

### Future work

Possible alternatives:

```text
nonlinear MPC
sequential convex programming with trust regions
sampling-based chance constraint validation
Monte Carlo post-validation
```

---

## 13. LIM-010: EKF-Style Covariance Propagation

### Description

Covariance propagation uses linearized dynamics:

$$
\boldsymbol{\Gamma}^{k+1}
=

\mathbf{F}^k
\boldsymbol{\Gamma}^k
\mathbf{F}^{kT}
+
\mathbf{W}^k
$$

This is efficient but approximate.

### Impact

The covariance prediction may be inaccurate when:

```text
dynamics are strongly nonlinear
horizon is long
attitude is far from hover
process noise is not Gaussian
linearization point is poor
```

### Current status

Accepted.

### Mitigation

The project shall:

```text
validate covariance symmetry
validate covariance PSD
extract Sigma3x3 consistently
log covariance trace if debug enabled
use short prediction horizon
```

### Future work

Possible alternatives:

```text
UKF propagation
Monte Carlo propagation
polynomial chaos expansion
learned uncertainty model
```

---

## 14. LIM-011: Constant-Velocity Obstacle Model

### Description

Obstacle prediction initially assumes:

```text
p_o(k+1) = p_o(k) + v_o(k) * dt
v_o(k+1) = v_o(k)
```

### Impact

This model may be inaccurate for:

```text
accelerating obstacles
humans changing direction
occluded obstacles
dynamic agents with intent
obstacles entering/exiting field of view
```

### Current status

Accepted for initial refactor.

### Mitigation

The system shall:

```text
log obstacle predictions
log closest obstacle id
log obstacle margin
limit claims to constant-velocity obstacle scenarios
```

### Future work

Add:

```text
constant-acceleration model
Kalman-filtered obstacle tracker
multi-hypothesis prediction
learned pedestrian prediction
```

---

## 15. LIM-012: No Real Perception Pipeline in Initial Refactor

### Description

The initial refactor does not implement the full depth-camera pipeline:

```text
depth image
U-depth map
box detection
object tracking
ellipsoid fitting
uncertainty estimation
```

Instead, obstacle states are loaded from scenario config or simulated obstacle manager.

### Impact

The simulation does not validate:

```text
perception latency
depth noise model
false positives
false negatives
occlusion
field-of-view loss
object detection failure
```

### Current status

Out of scope for initial refactor.

### Mitigation

The project shall:

```text
name obstacle source explicitly
log perception mode
avoid claiming real perception robustness
treat obstacle manager as simulated truth or simplified perception
```

### Future work

Add perception module and validate it separately.

---

## 16. LIM-013: No Real VIO Pipeline in Initial Refactor

### Description

The initial refactor may use:

```text
IdealEstimator
NoisyStateEstimator
VIO drift model
```

but does not implement a full real VIO pipeline.

### Impact

The simulation does not validate:

```text
feature tracking
visual-inertial calibration
camera-IMU synchronization
scale drift
tracking loss
estimator reset
real covariance quality
```

### Current status

Out of scope for initial refactor.

### Mitigation

The project shall:

```text
separate true_state and estimated_state
log estimator_type
log covariance
avoid claiming real VIO robustness
```

### Future work

Add external estimator adapter or recorded VIO dataset replay.

---

## 17. LIM-014: Solver May Be Infeasible or Late

### Description

The CC-MPC optimization can become infeasible or slow.

Common causes:

```text
too many close obstacles
large uncertainty
goal behind dense obstacle cluster
poor warm-start
overly strict chance constraints
bad slack penalty
solver numerical issue
```

### Impact

Solver failure can cause:

```text
no valid command
fallback command
goal timeout
unsafe trajectory if slack is overused
missed real-time deadline
```

### Current status

Managed by fallback and diagnostics.

### Mitigation

The controller shall:

```text
report solver status
report solve_time_ms
report fallback_used
retry with fresh initialization if configured
return fallback ControlCommand4 if needed
log max_constraint_violation
log min_chance_constraint_margin
```

### Future work

Evaluate alternative solvers and generated-code solvers for real-time deployment.

---

## 18. LIM-015: Deterministic Runtime Is Not Real Deployment Timing

### Description

The reference runtime is deterministic single-thread.

This is ideal for debugging and reproducibility, but real systems often run:

```text
estimator thread
perception thread
controller thread
physics or plant process
communication thread
```

### Impact

Single-thread validation may hide:

```text
state latency
command latency
thread scheduling jitter
stale commands
asynchronous perception updates
deadline misses
```

### Current status

Accepted.

### Mitigation

The project shall:

```text
log solve_time_ms
log runtime mode
treat deterministic mode as reference, not deployment proof
define threaded runtime separately
```

### Future work

Add threaded runtime with timestamped snapshots and command-age logging.

---

## 19. LIM-016: Threaded Runtime Is Harder to Reproduce

### Description

Threaded runtime can introduce non-determinism from scheduling.

### Impact

Two runs with the same config may differ due to:

```text
different command timing
different state snapshot timing
different solver completion time
OS scheduling
viewer/render overhead
```

### Current status

Optional future mode.

### Mitigation

Threaded runtime shall include:

```text
state_sequence_id
command_sequence_id
state_time
command_time
command_age_ms
stale_command_flag
controller_solve_time_ms
```

### Future work

Only introduce threaded runtime after deterministic runtime passes validation.

---

## 20. LIM-017: No Formal Stability Proof

### Description

The current project does not provide a formal closed-loop stability proof.

### Impact

Even if simulations work, this does not prove:

```text
asymptotic stability
input-to-state stability
robust stability
stability under all obstacle layouts
stability under all solver failures
```

### Current status

Out of scope.

### Mitigation

The project shall avoid claiming formal stability unless a proof is added.

### Future work

Possible future additions:

```text
terminal invariant set
terminal Lyapunov cost
recursive feasibility analysis
robust MPC proof
```

---

## 21. LIM-018: No Formal Recursive Feasibility Proof

### Description

The current implementation does not prove that if the MPC problem is feasible at time step `k`, it remains feasible at `k+1`.

### Impact

The controller may become infeasible due to:

```text
new obstacles
uncertainty growth
bad previous command
state deviation
solver numerical issue
```

### Current status

Out of scope.

### Mitigation

The project uses practical mechanisms:

```text
soft constraints
fallback controller
fresh initialization retry
termination policies
diagnostic logging
```

These mechanisms improve robustness but are not a mathematical proof.

### Future work

Develop recursive feasibility conditions or robust terminal constraints.

---

## 22. LIM-019: Not Flight-Certified or Hardware-Safe

### Description

This simulation is a research software artifact.

It is not:

```text
flight-certified
hardware-safe by itself
real-time certified
validated for human safety
validated for deployment near people
```

### Impact

The software shall not be used directly to fly real hardware without a separate safety layer.

### Current status

Explicit non-goal.

### Mitigation

Any hardware experiment must include:

```text
manual kill switch
geofence
low-level flight controller failsafe
maximum velocity/altitude limits
independent safety monitor
bench testing
controlled environment
```

### Future work

Create a separate hardware safety document if real flight experiments are planned.

---

## 23. LIM-020: Logs Do Not Prove Correctness Alone

### Description

Logs provide evidence of what happened during a run.

They do not prove that:

```text
the math is correct
the model is physically accurate
the controller is safe in all cases
the solver is always feasible
the engine is bug-free
```

### Impact

A run log can support debugging and regression testing, but it cannot replace unit tests, validation experiments, or mathematical analysis.

### Current status

Accepted.

### Mitigation

The project shall use logs together with:

```text
unit tests
interface tests
integration tests
regression tests
validation experiments
manual review
```

### Future work

Add automated validation reports that combine logs, plots, metrics, and test results.

---

## 24. Additional Limitations

### 24.1 No wind model by default

The initial refactor does not include wind gusts or aerodynamic disturbance fields.

Impact:

```text
simulation may be easier than real outdoor flight
```

Mitigation:

```text
document disturbance model
add wind model only as explicit scenario feature
```

---

### 24.2 No ground effect model

The simulation does not model ground effect near low altitude.

Impact:

```text
low-altitude behavior may be inaccurate
```

Mitigation:

```text
avoid low-altitude validation claims
use altitude lower bound
```

---

### 24.3 No battery or motor thermal model

The simulation does not model:

```text
battery voltage sag
motor heating
thrust degradation
ESC limits
```

Impact:

```text
long-run actuator performance may be overestimated
```

---

### 24.4 Limited obstacle geometry

Initial obstacle support focuses on ellipsoidal approximations.

Impact:

```text
complex geometry is approximated
sharp corners and concave shapes are not represented exactly
```

Mitigation:

```text
document obstacle type
use conservative inflation where needed
```

---

### 24.5 Limited multi-agent support

The current refactor focuses on one quadrotor.

Impact:

```text
multi-robot collision avoidance is not fully implemented
```

Future work:

```text
add RobotPredictionHorizon
add inter-robot chance constraints
add decentralized planning tests
```

---

## 25. Limitation Severity Levels

Severity shall be classified as:

| Severity | Meaning                                           |
| -------- | ------------------------------------------------- |
| Low      | Minor inconvenience or documentation issue        |
| Medium   | Can affect analysis but is manageable             |
| High     | Can affect research conclusions                   |
| Critical | Can affect safety or invalidate deployment claims |

---

## 26. Limitation Status Values

Status values:

| Status              | Meaning                                    |
| ------------------- | ------------------------------------------ |
| Accepted            | Known and within current scope             |
| Managed             | Mitigated by design, tests, or logs        |
| Requires validation | Needs test/experiment before relying on it |
| Out of scope        | Not addressed in current refactor          |
| Future work         | Planned extension                          |
| Deprecated          | Limitation no longer applies               |

---

## 27. How to Add a New Limitation

When a new limitation is found, add a new section:

```text
## LIM-XXX: Title

### Description
...

### Impact
...

### Current status
...

### Mitigation
...

### Future work
...
```

Then update:

```text
Summary Table
Validation Plan if needed
Known failure cases if applicable
```

---

## 28. Acceptance Criteria

This document is accepted when:

1. Major model limitations are documented.
2. Euler/quaternion limitations are documented.
3. ODE/MuJoCo mismatch is documented.
4. Mixer limitations are documented.
5. Chance constraint assumptions are documented.
6. Covariance propagation limitations are documented.
7. Solver limitations are documented.
8. Perception and VIO gaps are documented.
9. Runtime limitations are documented.
10. Hardware safety non-goal is explicit.
11. Each limitation includes mitigation or future work.

---

## 29. Summary

The current refactor targets a research-grade simulation architecture, not a fully certified flight stack.

The most important limitations are:

```text
the dynamics model is reduced
Euler attitude has known singularities
ODE and MuJoCo are not physically equivalent by default
MuJoCo requires careful adapter and mixer validation
chance constraints rely on Gaussian and linearization assumptions
covariance propagation is approximate
perception and VIO are simplified or simulated
solver infeasibility is possible
single-thread runtime is not deployment timing
there is no formal stability or recursive feasibility proof
the system is not hardware-safe by itself
```

These limitations are acceptable only because they are explicitly documented and tied to validation, logging, and future work.

---

## 30. Related Documents

```text
docs/design/02_ARCHITECTURE.md
docs/design/03_RUNTIME_FLOW.md
docs/design/04_DATA_MODEL.md
docs/design/05_ENGINE_INTERFACE.md
docs/design/06_CONTROLLER_INTERFACE.md
docs/design/07_SCENARIO_CONFIG.md
docs/design/08_LOGGING_AND_METRICS.md
docs/design/09_VALIDATION_PLAN.md
docs/design/11_REFACTOR_PLAN.md

docs/design/ADR/ADR-001-engine-abstraction.md
docs/design/ADR/ADR-002-single-thread-vs-mpc-thread.md
docs/design/ADR/ADR-003-state-vector-definition.md
docs/design/ADR/ADR-004-control-command-definition.md

docs/theory/05_Euler_Angles.md
docs/theory/06_Quaternion.md
docs/theory/07_Newton_Euler.md
docs/theory/12_CCMPC.md
docs/theory/13_Chance_Constraints.md
docs/theory/14_Covariance_Propagation.md
docs/theory/16_Optimization.md
docs/theory/17_Solver.md
docs/theory/18_Implementation_Notes.md
```
