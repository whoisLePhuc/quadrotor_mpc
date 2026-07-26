# Native risk-budget management

## Scope

Stage 5 adds explicit risk semantics to the native spherical CC-NMPC. It keeps
the Stage 4 individual mode for regression and adds a uniform joint allocation
over every horizon-step/obstacle chance constraint used by one NMPC solve.

Stage 5 does not implement adaptive/geometry-aware allocation, iterative risk
allocation or an episode-wide probability guarantee. Stage 6 adds a separate
post-solve safety supervisor without changing this allocator.

## Semantics

### Individual mode

Each scalar constraint receives the configured value:

\[
\epsilon_{k,o}=\epsilon_{\mathrm{individual}}.
\]

This is a per-constraint statement. The sum may be larger than one, and the
mode must not be described as a joint-horizon guarantee.

### Joint uniform mode

For \(N_c\) active scalar chance constraints and configured total budget
\(\epsilon_{\mathrm{total}}\):

\[
\epsilon_{k,o}
=
\frac{\epsilon_{\mathrm{total}}}{N_c},
\qquad
\sum_{k,o}\epsilon_{k,o}
=
\epsilon_{\mathrm{total}}.
\]

The native NLP has \(N+1\) state nodes, so:

\[
N_c=(N+1)n_{\mathrm{obs}}.
\]

Every node present in the constraint grid, including the initial node, is
counted. This is conservative but keeps the numerical model and the reported
union-bound accounting identical.

By Boole's inequality:

\[
\mathbb{P}\left(\bigcup_{k,o}\mathcal C_{k,o}\right)
\le
\sum_{k,o}\epsilon_{k,o}
\le
\epsilon_{\mathrm{total}}.
\]

This bound applies to the prediction horizon of one solve. Receding-horizon
replanning introduces overlapping, state-dependent events; therefore the same
number is not automatically an upper bound for collision over the complete
closed-loop episode.

## Tightening

Each allocated cell has its own one-sided Gaussian quantile:

\[
\beta_{k,o}=\Phi^{-1}(1-\epsilon_{k,o}),
\]

and its tightened radius is:

\[
r_{\mathrm{tight},k,o}
=
r_{\mathrm{base},o}
+
\beta_{k,o}\sigma_{k,o}.
\]

The risk allocator and covariance projection remain outside the nonlinear
program. The resulting epsilon, beta and radius arrays are supplied as TVPs.

## Configuration

```yaml
controller:
  chance_constraints:
    enabled: true
    type: spherical
    individual_epsilon: 0.05
    risk_budget:
      semantics: joint
      allocation: uniform
      total_epsilon: 0.10
      tolerance: 1.0e-12
```

Set `semantics: individual` to reproduce Stage 4. In Stage 5,
`allocation: uniform` is the only joint allocator accepted; unsupported names
fail configuration validation instead of silently falling back.

## Runtime audit

Every `ControlSolution`, native result, telemetry sample and recording reports:

- risk semantics and allocation method;
- configured total epsilon;
- allocated and remaining epsilon;
- number of active scalar constraints;
- `BUDGET_OK`, `INDIVIDUAL_ONLY` or `DISABLED`.

Uniform joint allocation raises an error if the allocated sum exceeds the
configured total by more than `tolerance`.

`BUDGET_OK` only validates the allocation arithmetic. A probabilistic safety
claim additionally requires:

- nonpositive solver relaxation within tolerance;
- calibrated Gaussian covariance;
- correct obstacle prediction and independence assumptions;
- Monte Carlo validation with confidence intervals.

Any positive safety slack still invalidates the chance guarantee.
