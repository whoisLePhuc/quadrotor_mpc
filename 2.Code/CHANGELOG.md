# Changelog

## Unreleased — Layered package architecture

- Moved every executable module from the `2.Code` root into the installable
  `quadrotor_mpc` package.
- Grouped domain contracts, controllers, estimation, infrastructure,
  application use cases, reporting and interfaces into explicit layers.
- Repointed all console scripts to packaged CLI adapters while preserving their
  public command names and arguments.
- Added architecture dependency tests and release checks that reject root-level
  runtime modules or non-packaged entry points.
- Updated tests and documentation to use the canonical package imports.

## 2.0.1 — Release stabilization

- Corrected README commands, scope and claims to match measured repository
  behavior and the `VALIDATED_WITH_LIMITATIONS` result.
- Aligned the safety-supervisor acceptance deadline with the 50 ms controller
  period and added cross-field configuration validation.
- Added locked dependency resolution and GitHub Actions gates for lint, tests,
  native configs, Qt lifecycle, build and installed-wheel execution.
- Completed wheel packaging for native modules, console commands, configs and
  the sourced Crazyflie assets.
- Added root/package MIT licenses and a release acceptance checklist.
- Added clean-source enforcement, deterministic source fingerprints and
  source-matching resume checks to native Monte Carlo evidence.

## 2.0.0 — Native Monte Carlo validation

- Added a paired native MuJoCo benchmark for deterministic-estimated,
  individual-risk and joint-uniform controllers.
- Added covariance sweeps at `0.25 Sigma`, `Sigma` and `4 Sigma`, with standard
  deviations scaled by the square root of the covariance multiplier.
- Added Wilson confidence intervals, paired-seed deltas and separate gates for
  execution, empirical collisions, risk accounting, slack, fallback and
  real-time timing.
- Added append-only checkpoints, protocol fingerprints and safe resume for long
  validation campaigns.
- Added raw CSV, aggregate JSON, reproducibility manifest, Markdown report and
  six-panel PNG artifacts.
- Kept finite-sample evidence distinct from an episode-wide probability proof.

## 1.9.0 — Native safety-console integration

- Added a pure, headless-testable telemetry-to-view projection so Qt never
  reinterprets chance, guarantee, risk or fallback semantics.
- Replaced the single dense status line with six independent operational cards
  for episode, applied control, assurance, risk, slack and solver timing.
- Added bounded plots for chance residual/slack, propagated uncertainty,
  tightened radius, joint-risk use, solution acceptance and fallback level.
- Added a visible solve-deadline reference and deduplicated operational
  transition log.
- Added assurance-aware MuJoCo prediction colors and degraded/fallback vehicle
  halos.
- Preserved reset/replay behavior and deterministic profiles without
  fabricating probabilistic status.

## 1.8.0 — Safe slack acceptance and fallback

- Added a backend-independent safety supervisor that accepts or rejects primary
  NMPC solutions before they reach the plant.
- Added gates for backend success, joint-risk status, actuator bounds,
  nonlinear residual/slack consistency, degraded-slack limit and solve
  deadline.
- Added deterministic escalation through last accepted command, bounded
  position-hold PD and rate-damped emergency hover.
- Converted solver exceptions into normalized fallback solutions instead of
  terminating the closed-loop runtime.
- Added guarantee-eligibility versus degraded-positive-slack semantics.
- Extended controller results, telemetry, event recording, replay and desktop
  status with accept/reject and fallback evidence.
- Preserved Reset and Run again reproducibility by clearing supervisor state.

## 1.7.0 — Native joint risk-budget management

- Added explicit `individual` and `joint` chance-constraint risk semantics.
- Added uniform allocation of `total_epsilon` over every active
  horizon-step/obstacle pair with Boole union-bound accounting.
- Added per-constraint Gaussian quantiles from the allocated epsilon instead of
  one global beta.
- Added runtime allocation audits with total, allocated and remaining risk,
  active-constraint count and `BUDGET_OK` / `INDIVIDUAL_ONLY` / `DISABLED`.
- Extended controller results, native telemetry, recording, replay and desktop
  status with risk-budget metadata.
- Preserved Stage 4 individual mode and documented that a receding-horizon
  budget is not an episode-wide probability guarantee.

## 1.6.0 — Native spherical chance constraints

- Added individual Gaussian spherical chance constraints with
  collision-normal covariance projection.
- Added one-iteration shifted-nominal tightening outside the do-mpc NLP.
- Added TVPs for projected sigma, Gaussian quantile, individual risk and
  time-varying tightened safety radius.
- Added explicit soft-constraint slack telemetry and `SOLVED_SAFE` /
  `SOLVED_WITH_SLACK` classification.
- Added a complete native CC-MPC scenario and preserved deterministic
  regression configurations.
- Extended native telemetry, recording, replay and visualization with chance
  residual, risk, slack, projected uncertainty and tightened safety radii.

## 1.5.0 — Native horizon covariance propagation

- Added 12D quaternion error-state covariance propagation along the optimized
  native NMPC horizon with analytic first-order local Jacobians and an optional
  RK4 finite-difference verification mode.
- Added 6D constant-velocity obstacle covariance horizons with white
  acceleration process noise.
- Added `open_loop` and optional feedback-aware `feedback_lqr` propagation
  modes without changing the deterministic NMPC constraint.
- Extended `ControlSolution`, native results, telemetry, recording and replay
  with vehicle and obstacle horizon covariance.
- Added numerical PSD projection, configuration validation and Stage 3
  unit/integration tests.

## 1.4.0 — Native sensor and estimator pipeline

- Added seeded Gaussian vehicle and obstacle sensor simulation with optional
  dropout and unmodelled bias random walks.
- Added a 13D nominal/12D error-state quaternion EKF for the native Crazyflie
  state and a position-only 6D constant-velocity Kalman tracker per obstacle.
- Routed estimated beliefs into the Stage 1 controller interface without
  exposing MuJoCo truth or configured obstacle motion to the controller.
- Reset the sensor RNG, filters and controller belief together for reproducible
  `Reset` and `Run again` episodes.
- Added estimator state, covariance, measurement availability and tracker
  horizons to native results, telemetry recording and replay.
- Preserved exact deterministic configurations and added
  `config/mujoco_native_estimation.yaml` for the Stage 2 pipeline.

## 1.3.0 — Belief-based controller foundation

- Added a backend-independent `Controller` protocol and validated `VehicleBelief`,
  `ObstacleBelief`, `ControlGoal` and `ControlSolution` contracts.
- Wrapped the existing do-mpc controller in `DeterministicNMPCController`.
- Changed every obstacle center to a time-varying parameter so future estimators can
  update obstacle means without recompiling the NLP.
- Routed native MuJoCo execution through the shared controller interface while
  preserving a clearly isolated perfect-information adapter for deterministic regression.
- Exposed covariance, chance-margin, risk-allocation, slack and solver-status fields
  at the runtime boundary for later CC-MPC implementations.

## 1.2.1 — Persistent interactive session

- Kept the native viewer and telemetry panel open after goal, collision stop or
  configured duration; completion now enters a stable held state.
- Added **Run again** (Enter) to reset plant, controller, clock, overlays,
  recording episode and plots, then immediately resume the same scenario.
- Preserved automatic termination for headless and batch workflows.

## 1.2.0 — Interactive native runtime

- Added constant-velocity, three-axis sinusoidal and waypoint obstacle motion with
  one shared predictor used by NMPC, MuJoCo, viewer and replay.
- Added native keyboard controls for pause, single-step, reset, stop and overlays.
- Added a separate PySide6/pyqtgraph control panel with bounded real-time telemetry.
- Added NMPC solve-time measurement, obstacle horizon overlays and safety status.
- Added deterministic recording bundles, snapshots and solver-free native replay.
- Added a dynamic-crossing scenario and regression tests for motion and recording.

## 1.1.0 — Native MuJoCo and sourced Crazyflie model

- Added a native passive MuJoCo desktop entry point with follow/fixed camera,
  real-time pacing, trail, prediction, safety envelope and contact overlays.
- Added a runtime observer contract so the viewer does not duplicate the NMPC or
  physics loop.
- Replaced the unsourced box/sphere quadrotor with the MIT-licensed Bitcraze
  Crazyflie 2 model from Google DeepMind MuJoCo Menagerie.
- Aligned the optional 13-state controller mass, inertia, collision radius and
  actuator bounds with the Crazyflie-scale plant.
- Added model provenance, configuration validation and hover/contact regression tests.

## 1.0.0 — Research workbench release

- Added reproducible run IDs, manifests and configuration snapshots.
- Added EKF state estimation, covariance logging and actual/estimate/reference telemetry.
- Added predicted horizon, predicted controls, chance residual, slack, cost and solver diagnostics.
- Added adaptive PNG and self-contained interactive HTML reports.
- Added paired controller aggregation, confidence intervals, Monte Carlo and parameter sweeps.
- Added multi-page Streamlit dashboard and optional NMPC/MuJoCo page.
- Added rotated-ellipsoid and high-noise scenarios.
- Expanded architecture, protocol, metric, validation and theory-to-code documentation.
