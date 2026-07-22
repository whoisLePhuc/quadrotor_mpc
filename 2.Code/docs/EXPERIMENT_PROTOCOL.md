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
