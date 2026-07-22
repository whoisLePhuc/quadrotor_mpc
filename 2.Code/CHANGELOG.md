# Changelog

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
