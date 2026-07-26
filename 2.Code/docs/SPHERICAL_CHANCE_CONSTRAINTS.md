# Native Spherical Chance Constraints

## Scope

Stage 4 adds Gaussian spherical chance constraints to the native quaternion
NMPC. It supports spherical obstacles only. Covariance propagation remains
outside the do-mpc nonlinear program, and the deterministic controller remains
available with the same plant, estimator and seed.

Stage 5 adds individual-versus-joint semantics and uniform horizon allocation.
See `RISK_BUDGET_MANAGEMENT.md`. Stage 6 applies solver, slack and timing gates
plus a bounded fallback hierarchy after this optimization; see
`SAFE_SLACK_FALLBACK.md`.

## Constraint

For horizon step \(i\) and obstacle \(j\), define the relative mean position:

\[
d_{i,j} = \mu_{p,i} - \mu_{o,i,j},
\qquad
n_{i,j} = \frac{d_{i,j}}{\|d_{i,j}\|}.
\]

The relative position covariance is:

\[
P_{\mathrm{rel},i,j}
=
P_{p,i}^{(3\times3)}
+
P_{o,i,j}^{(3\times3)}.
\]

Independence between vehicle and obstacle estimation errors is assumed. The
standard deviation projected onto the collision normal is:

\[
\sigma_{i,j}
=
\sqrt{
n_{i,j}^{T}P_{\mathrm{rel},i,j}n_{i,j}
}.
\]

For allocated violation probability \(\epsilon_{i,j}\):

\[
\beta_{i,j} = \Phi^{-1}(1-\epsilon_{i,j}),
\qquad
r_{\mathrm{tight},i,j}
=
r_{\mathrm{drone}}+r_{\mathrm{obs},j}+r_{\mathrm{margin}}
+\beta_{i,j}\sigma_{i,j}.
\]

The deterministic nonlinear constraint supplied to do-mpc is:

\[
\|p_i-\mu_{o,i,j}\|
\ge
r_{\mathrm{tight},i,j}-s_{i,j},
\qquad s_{i,j}\ge0.
\]

do-mpc uses the same \(10^{-6}\,\mathrm{m}^2\) smoothing term as the residual
telemetry. The recorded slack is the minimum nonnegative slack required by the
optimized nominal solution.

## Per-tick order

1. Shift the previous optimized state/control trajectory.
2. On the first tick, initialize a straight-line position seed.
3. Linearize the 12D quaternion error-state dynamics around the seed.
4. Propagate vehicle and 6D obstacle covariance over \(N+1\) points.
5. Compute collision normals, projected sigma and tightened radii.
6. Allocate risk and fill do-mpc TVPs for mean, sigma, beta, risk and radius.
7. Solve the native NMPC once.
8. Record chance residual, slack, risk, covariance and solver classification.

This is one outer covariance iteration. Re-solving repeatedly after updating
the nominal trajectory is intentionally deferred.

## Configuration

```yaml
controller:
  covariance_propagation:
    enabled: true
    mode: open_loop

  chance_constraints:
    enabled: true
    type: spherical
    individual_epsilon: 0.05
    risk_budget:
      semantics: joint
      allocation: uniform
      total_epsilon: 0.10
    soft_constraint: true
    slack_penalty: 1000000.0
    slack_tolerance_m: 1.0e-6
```

Run the complete estimator/covariance/CC-NMPC scenario:

```bash
MUJOCO_GL=glfw python run_mujoco_native.py \
  --config config/mujoco_native_ccmpc.yaml
```

## Status semantics

- `SOLVED_SAFE`: optimized chance residuals are nonnegative within tolerance.
- `SOLVED_WITH_SLACK`: a solution exists, but one or more chance constraints
  require positive slack. The chance guarantee is not satisfied.
- `SOLVED_DETERMINISTIC`: chance tightening is disabled.

Fallback and emergency statuses are not implemented through Stage 5.

## Assumptions and limitations

- Gaussian local position errors.
- Vehicle and obstacle errors are independent; cross-covariance is zero.
- Collision normal is frozen from the shifted nominal trajectory.
- In `individual` mode, every horizon-step/obstacle pair uses the same
  `individual_epsilon`; this is not a joint statement.
- In `joint` mode, the uniform allocator accounts for every \(N+1\) NLP state
  node and obstacle. The budget applies to one prediction horizon, not the
  complete receding-horizon episode.
- Spherical obstacle geometry only.
- Positive slack invalidates the chance guarantee and is never hidden.
