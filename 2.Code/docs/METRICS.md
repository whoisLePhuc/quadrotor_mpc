# Metrics

| Metric | Meaning |
|---|---|
| Final error | Euclidean distance to goal at termination |
| Tracking RMSE | RMS distance from the finite start-goal segment |
| Path length | Sum of 3-D distances between plant samples |
| Min clearance | Minimum signed ellipsoid clearance; negative is collision |
| Chance violation rate | Fraction of finite logged residuals below zero |
| Control effort | Time integral of squared command norm |
| Control smoothness | Sum of squared command increments |
| Saturation rate | Fraction of samples at 99% or more of any command limit |
| Solver p95 | 95th percentile of nonzero control solve times |
| Deadline miss | Fraction of solves longer than the controller period |
| Positive-slack episode | Episode containing at least one chance slack above tolerance |
| Fallback episode | Episode containing at least one non-primary applied command |
| Guarantee-eligible episode | Chance-enabled episode whose every completed tick is guarantee-eligible |
| Budget-failure episode | Episode containing `BUDGET_EXCEEDED` |
| Collision-rate CI | Wilson interval over geometric collision events across seeds |

`success` requires goal error within the configured threshold and no geometric collision. A run can
reach the goal but still be unsuccessful after colliding. Soft slack is always reported separately.

For native Monte Carlo, solver timing is summarized inside each trial and again
across trials. The real-time gate uses the 95th percentile of per-trial p99
solve times against the native controller period. It is intentionally stricter
than comparing only the global mean.
