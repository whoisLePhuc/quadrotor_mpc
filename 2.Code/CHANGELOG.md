# Changelog

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
