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
10. Horizon covariance validation: initial-belief agreement, local-error
    linearization, vehicle/obstacle horizon dimensions, symmetry, PSD, open-loop
    growth, feedback-aware behavior and numeric recording round-trip.
11. Native spherical chance constraints: Gaussian quantile, relative covariance
    projection, TVP tightening, risk/slack shape agreement, status semantics and
    deterministic-versus-CC-MPC paired regression.
12. Native risk-budget management: legacy individual semantics, uniform joint
    allocation, exact sum audit, per-cell quantiles, configuration round-trip,
    telemetry/recording persistence and joint-versus-individual paired regression.

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
validation logs.

## Stage 3 horizon covariance checks

Run `config/mujoco_native_estimation.yaml` and verify:

- `predicted_error_covariance_horizon.shape == (ticks, N+1, 12, 12)`;
- `predicted_obstacle_covariance_horizon.shape == (ticks, N+1, n_obs, 6, 6)`;
- horizon index zero matches the belief supplied to that controller tick;
- every covariance is finite, symmetric and positive semidefinite;
- terminal position sigma responds to process-noise configuration;
- deterministic control, clearance and collision regression are unchanged
  because Stage 3 does not yet tighten a constraint.

The `feedback_lqr` mode is checked as a propagation approximation, not as a
replacement controller. No chance-constraint guarantee may be claimed until
Stage 4 makes the propagated uncertainty change the optimization constraint.

## Stage 4 spherical chance-constraint checks

Run `config/mujoco_native_estimation.yaml` and
`config/mujoco_native_ccmpc.yaml` with the same seed. Verify:

- the deterministic run preserves the Stage 3 final error and clearance;
- `tightened_safety_radius >= nominal_safety_radius`;
- `tightening == Phi^-1(1-epsilon) * projected_sigma`;
- `chance_residual + slack >= 0` within solver tolerance;
- positive slack is classified as `SOLVED_WITH_SLACK`;
- a run containing positive slack is not reported as satisfying the chance
  guarantee;
- CC-MPC produces a measurably different constraint/clearance from the paired
  deterministic run;
- no covariance, residual, control or state contains NaN.

## Stage 5 risk-budget checks

Run `config/mujoco_native_ccmpc.yaml` and verify:

- `risk_semantics == joint` and `risk_allocation_method == uniform`;
- each solve allocates over `(N+1) * n_obstacles` scalar constraints;
- `sum(risk_allocation_horizon[t]) <= risk_budget_total[t] + tolerance`;
- `risk_budget_allocated == risk_budget_total == 0.10` for the supplied
  three-obstacle scenario;
- each TVP quantile equals `Phi^-1(1-risk_allocation[k,o])`;
- changing back to `individual` reproduces Stage 4 allocations and trajectory;
- joint allocation produces stronger tightening than individual allocation for
  the same belief/covariance;
- `BUDGET_OK` is not interpreted as a safety guarantee when slack is positive.

The joint budget is for one receding prediction horizon. Monte Carlo validation
with a confidence interval is required before making an episode-level collision
rate claim.

Verified 10-second paired run, estimator seed 7:

| Controller | Final error | Minimum clearance | Maximum slack | Status counts |
|---|---:|---:|---:|---:|
| Deterministic dynamic baseline | 0.000135502 m | 0.299914211 m | 0 | `DISABLED` |
| Stage 4 individual, epsilon 0.05 | 0.076558310 m | 0.317015107 m | 0.064665434 m | 161 safe / 39 slack |
| Stage 5 joint uniform, total 0.10 | 0.104444105 m | 0.410519817 m | 0.075120639 m | 114 safe / 86 slack |

For the joint run, every one of 200 solves allocated 63 constraints, the
maximum absolute budget-sum error was \(5.55\times10^{-17}\), and no collision,
NaN state or NaN control occurred. The positive slack prevents a probabilistic
safety guarantee despite correct risk accounting.
