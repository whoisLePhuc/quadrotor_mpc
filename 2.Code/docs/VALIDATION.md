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
13. Native safety-console integration: deterministic/CC-MPC policy projection,
    guarantee/slack/fallback/deadline labeling, risk-budget failure visibility,
    transition deduplication, reset semantics and configuration round-trip.
14. Native Monte Carlo validation: paired seeds, covariance-level scaling,
    Wilson intervals, paired deltas, resumable artifacts and explicit
    probability-claim/timing gates.

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

## Stage 6 safe slack and fallback checks

Verify the supervisor independently of the optimizer using fault injection:

- backend `success=false` rejects the primary solution;
- solver exceptions do not terminate the plant loop;
- command-bound, risk-budget, nonlinear residual, slack-limit and deadline
  gates each produce a specific rejection reason;
- the command source escalates through
  `HOLD_LAST_ACCEPTED -> POSITION_HOLD_PD -> EMERGENCY_HOVER`;
- every fallback command remains inside configured actuator bounds;
- Reset and Run again clear the last command, fallback reference and rejection
  count;
- telemetry/recording/replay preserve primary and fallback status;
- positive slack above the numerical guarantee tolerance is never labeled
  guarantee-eligible.

Verified 10-second Stage 6 run, estimator seed 7, joint uniform budget 0.10 and
100 ms configured solve deadline:

| Metric | Result |
|---|---:|
| Completed ticks | 200 / 200 |
| Primary solver success | 200 / 200 |
| Primary commands applied | 200 |
| Nominal deadline rejects / fallback | 0 / 0 |
| Minimum clearance | 0.410519826 m |
| Maximum slack | 0.075120640 m |
| Guarantee-eligible ticks | 114 |
| Positive-slack, not-guaranteed ticks | 86 |
| Maximum primal / dual solver residual | 9.12e-7 / 6.92e-7 |
| Mean / p95 / p99 solve time | 60.63 / 73.62 / 81.60 ms |
| Maximum solve time | 88.42 ms |
| Collision / NaN | no / no |

The supervisor-disabled Stage 5 regression remains exactly:
final error 0.104444105 m, minimum clearance 0.410519817 m, 114
`SOLVED_SAFE` and 86 `SOLVED_WITH_SLACK`.

The current native NMPC does **not** pass a strict 50 ms real-time gate on the
validation host because p99 solve time is 81.60 ms. Stage 6 validates the
deadline-miss response; it does not claim that the solver is fast enough for a
20 Hz deployment.

## Stage 7 desktop safety-console checks

Verify the presentation model independently of Qt:

- deterministic mode reports `DETERMINISTIC`, `DISABLED` risk/slack and
  `MONITOR ONLY`, never a chance guarantee;
- an accepted zero-slack joint-risk solution reports `PRIMARY NMPC`,
  `GUARANTEE ELIGIBLE`, `BUDGET_OK`, `HARD-SAFE` and `ON TIME`;
- positive slack reports `DEGRADED` and `NOT GUARANTEED`;
- fallback, deadline miss and risk-budget failure use high-visibility warning
  or danger states;
- repeated telemetry in the same state does not duplicate transition alerts;
- Reset and Run again clear history, alerts and prior transition state;
- replay projects the recorded sample fields through the same view model;
- all four native YAML configurations round-trip with bounded alert history.

When optional UI dependencies are available, additionally launch
`config/mujoco_native_ccmpc.yaml` and verify that the Qt panel remains
responsive while MuJoCo runs, closing either window stops cleanly, prediction
color follows assurance state, and the deadline trace matches the configured
50 ms acceptance limit. The Stage 7 paired-regression numbers below are
historical measurements from the former 100 ms policy; v2.0.1 intentionally
changes the command-acceptance deadline to match the control period.

Paired control-path regression against Stage 6 commit `5256c82` in the same
Python/dependency environment:

| Configuration | Stage 6 | Stage 7 | Difference |
|---|---:|---:|---:|
| Deterministic final error | 0.000101204527 m | 0.000101204527 m | 0 |
| Deterministic clearance | 0.299927809264 m | 0.299927809264 m | 0 |
| CC-MPC final error | 0.104444105413 m | 0.104444105413 m | 0 |
| CC-MPC clearance | 0.410519825565 m | 0.410519825565 m | 0 |
| CC-MPC maximum slack | 0.075120639653 m | 0.075120639653 m | 0 |

Both CC-MPC runs produced 114 guarantee-eligible ticks, 86 positive-slack
ticks, zero fallback activations, no collision and no NaN. This paired check
isolates Stage 7 from solver-library version drift and confirms the presentation
integration does not change an applied command.

## Stage 8 native Monte Carlo checks

Run:

```bash
quadrotor-mpc-monte-carlo \
  --config config/native_monte_carlo.yaml \
  --workers 3
```

Verify:

- the raw table contains exactly
  `trials * modes * noise_levels` unique `(noise, mode, seed)` rows;
- deterministic, individual and joint controllers use identical seed sets at
  every uncertainty level;
- covariance scale `s` multiplies configured standard deviations by `sqrt(s)`;
- deterministic mode disables chance tightening while retaining the same
  estimator, plant, obstacles and safety-supervisor policy;
- every Bernoulli episode metric reports its event count, rate and Wilson
  confidence interval;
- risk-budget failures, slack and fallback remain separate metrics;
- `GUARANTEE_ELIGIBLE` requires every completed tick to be eligible;
- any positive slack or fallback blocks the finite-sample chance claim;
- no per-horizon joint budget is reinterpreted as episode collision
  probability;
- p99 timing is checked against the 50 ms controller period;
- interrupted runs resume only after a protocol-fingerprint match;
- manifest, protocol/config snapshots, checkpoint, CSV, aggregate JSON,
  Markdown and PNG are all present.

The default 50-seed, three-level protocol is a descriptive finite-sample
validation. Even a result labeled `EMPIRICALLY_SUPPORTED_NOT_PROVEN` is not a
formal proof of Gaussian calibration or an episode-wide collision guarantee.

Verified Stage 8 campaign, seeds 1000–1029:

| Covariance | Controller | Success | Collision | Mean min clearance | Mean max slack | Fallback episodes |
|---|---|---:|---:|---:|---:|---:|
| `0.25 Sigma` | deterministic | 30/30 | 0/30 | 0.305261 m | 0 | 0/30 |
| `0.25 Sigma` | individual | 30/30 | 0/30 | 0.326352 m | 0.062461 m | 16/30 |
| `0.25 Sigma` | joint | 30/30 | 0/30 | 0.343861 m | 0.056917 m | 14/30 |
| `Sigma` | deterministic | 30/30 | 0/30 | 0.313791 m | 0 | 2/30 |
| `Sigma` | individual | 30/30 | 0/30 | 0.356432 m | 0.061792 m | 19/30 |
| `Sigma` | joint | 30/30 | 0/30 | 0.414314 m | 0.070586 m | 17/30 |
| `4 Sigma` | deterministic | 30/30 | 0/30 | 0.323614 m | 0 | 4/30 |
| `4 Sigma` | individual | 30/30 | 0/30 | 0.467242 m | 0.079857 m | 22/30 |
| `4 Sigma` | joint | 0/30 | 0/30 | 0.660826 m | 0.139326 m | 30/30 |

All 270 episodes completed without collision, NaN or risk-budget failure. The
joint-minus-deterministic mean clearance deltas were `+0.038600 m`,
`+0.100523 m` and `+0.337212 m` as covariance increased. At `4 Sigma`, joint
control traded that clearance for a mean final-error increase of `0.338990 m`;
every episode missed the configured goal threshold.

The campaign status is `VALIDATED_WITH_LIMITATIONS`. For zero collisions in 30
trials, the Wilson 95% upper bound is `0.113513`, above the empirical gate
`0.10`. Every chance-controller cell also contained positive slack and
fallback. Median per-trial solver p99 ranged from `87.523 ms` to `95.470 ms`
for the chance controllers, so the 50 ms real-time gate failed.

## Phase 9 release-stabilization checks

Version 2.0.1 adds release gates around the completed research workbench:

- the configured rejection deadline must not exceed the controller period;
- README commands and claims are checked against the repository;
- all runtime configs, tests and lint run in GitHub Actions;
- the Qt panel lifecycle and an installed wheel are smoke-tested;
- dependency resolution is frozen by `uv.lock`;
- native Monte Carlo release evidence refuses an uncommitted Git tree and
  records a deterministic validation-source SHA-256;
- an explicitly allowed dirty-source campaign is reproducible but marked
  non-release-eligible.

The 30-seed Stage 8 artifact remains unchanged for auditability. Its legacy
commit provenance is documented in `validation/README.md`; it must not be
presented as the clean 50-seed v2.0.1 release campaign.
