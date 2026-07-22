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

`success` requires goal error within the configured threshold and no geometric collision. A run can
reach the goal but still be unsuccessful after colliding. Soft slack is always reported separately.
