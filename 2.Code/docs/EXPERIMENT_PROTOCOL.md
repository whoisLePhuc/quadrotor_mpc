# Experiment protocol

Controller comparisons must use paired conditions:

1. Same scenario, plant model, start and goal.
2. Same seed and therefore the same disturbance distribution.
3. Same stop condition, simulation duration and control frequency.
4. Same obstacle trajectories and measurement assumptions.
5. Record solver failures, slack and deadline misses without filtering.

Use at least 30 seeds for descriptive Monte Carlo work and more for estimating rare collision
probabilities. Report the number of trials together with confidence intervals. A single trajectory
is a diagnostic example, not evidence of robustness.

Primary comparison groups are tracking, safety, efficiency, solver timing and reliability. The
paired delta in `metrics.json` is CC-MPC minus deterministic MPC; its sign must be interpreted per
metric.

## Native Stage 8 protocol

`config/native_monte_carlo.yaml` validates the 13-state MuJoCo path separately
from the earlier 9-state ODE study. Its default matrix is:

- controllers: deterministic-estimated, individual-risk and joint-uniform;
- covariance levels: `0.25 Sigma`, `Sigma` and `4 Sigma`;
- 50 paired seeds per controller and covariance level;
- the same plant, obstacles, duration, estimator structure, actuator limits and
  safety supervisor for every paired comparison.

The level value multiplies covariance. Therefore every affected standard
deviation is multiplied by the square root of that value. Dropout probabilities
and deterministic scenario geometry are not scaled.

Every completed episode is appended to `trials.jsonl`. Resume is permitted only
when the stored protocol fingerprint matches the effective protocol and native
base configuration.

Wilson intervals are reported for episode success, collision, incomplete run,
numerical failure, positive slack, fallback, guarantee eligibility and
risk-budget failure. The report also keeps paired seed deltas for tracking,
clearance, effort, timing, slack and fallback.

No episode-level probability guarantee may be claimed from the configured
per-horizon risk budget. Under the default gate, a chance-controller claim is
blocked by insufficient trials, a collision confidence bound above the
configured empirical limit, any positive slack, fallback, risk-budget failure,
NaN or incomplete episode. Passing this gate means
`EMPIRICALLY_SUPPORTED_NOT_PROVEN`.
