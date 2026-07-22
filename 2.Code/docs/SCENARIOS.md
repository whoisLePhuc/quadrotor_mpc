# Scenarios

| Scenario | Purpose |
|---|---|
| `hover` | equilibrium, serialization and regression smoke test |
| `point_to_point` | tracking with noise and model mismatch |
| `static_obstacle` | paired deterministic versus CC-MPC example |
| `moving_obstacle` | constant-velocity prediction and growing obstacle covariance |
| `rotated_ellipsoid` | rotated geometry and non-axis-aligned clearance |
| `high_noise` | stress covariance tightening and estimator robustness |

Create new scenarios in the dashboard or copy a YAML file. Every scenario must specify a 9-element
start state, 3-element goal and positive plant timestep. Use explicit seeds in published results.
