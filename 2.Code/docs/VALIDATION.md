# Validation

Validation is layered:

1. Unit tests: dynamics equilibrium, linearization, rotated matrix square root and covariance PSD.
2. Integration tests: deterministic repeatability, artifact schema and adaptive reports.
3. Scenario regression: hover, point-to-point, static and moving obstacles.
4. Statistical validation: paired Monte Carlo and parameter sweeps.
5. Cross-plant validation: 9-state ODE controller model versus 13-state MuJoCo plant.
6. MuJoCo model validation: sourced mass/inertia, MJCF compilation, hover equilibrium,
   collision contacts and native-viewer lifecycle.
7. Native interaction validation: legacy/new obstacle schemas, horizon agreement,
   command ordering, reset/pause/single-step behavior and recording round-trip.
8. Controller-contract validation: state/covariance dimensions, quaternion
   normalization, positive-semidefinite covariance, horizon shape agreement and
   normalized solver diagnostics.
9. Native estimation validation: seeded reset repeatability, dropout behavior,
   quaternion local-error consistency, ESEKF update contraction, covariance PSD,
   position-only obstacle velocity recovery and truth-free tracker horizons.

For a native release, run both `config/mujoco_native.yaml` and
`config/mujoco_native_dynamic.yaml`. Check that the reported current obstacle
position equals the first point of its displayed prediction, replay loads with
`allow_pickle=False`, and closing either interactive window terminates cleanly.

Before claiming a chance-constraint probability, verify Gaussian assumptions, covariance
calibration, obstacle prediction error, finite-sample collision rate and nonzero slack. FOV and soft
constraints require their own residual plots and must not be inferred from trajectory appearance.

## Stage 1 controller-interface regression

The native deterministic baseline must remain numerically equivalent after the
belief-interface refactor. The verified dynamic scenario result is:

| Check | Result |
|---|---:|
| Completed steps | 200 / 200 |
| Final goal error | 0.00013550 m |
| Minimum clearance | 0.29991421 m |
| Collision | false |
| NaN state | false |

The runtime integration tests also assert that `ControlSolution` diagnostics
reach `CoupledStep`, and that goal completion, `Run again`, reset, pause,
single-step and explicit stop preserve their v1.2.1 behavior.

## Stage 2 estimator checks

Run `config/mujoco_native_estimation.yaml` and report separately:

- vehicle position, velocity, attitude and body-rate RMSE;
- normalized estimation error squared when calibration is evaluated;
- vehicle and obstacle dropout counts;
- covariance minimum eigenvalue;
- obstacle position/velocity RMSE;
- deterministic versus estimated closed-loop safety metrics.

The controller must receive tracker-extrapolated obstacle means. Ground-truth
motion is allowed only in the sensor simulator, collision/clearance metrics and
validation logs. Horizon covariance consistency is a Stage 3 criterion.
