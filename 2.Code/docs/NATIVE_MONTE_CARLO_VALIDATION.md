# Native Monte Carlo Validation

## Purpose

Stage 8 evaluates the complete native chain:

```text
seeded sensor simulation
  -> vehicle ESEKF and obstacle Kalman trackers
  -> horizon covariance propagation
  -> deterministic / individual-risk / joint-risk NMPC
  -> safety acceptance gates and fallback
  -> 13-state MuJoCo plant and geometric collision truth
```

This is separate from the lightweight 9-state Monte Carlo implementation. The
controller still receives beliefs only; MuJoCo truth is used by the sensor
simulator and validation metrics, never as hidden controller input.

## Default matrix

| Axis | Values |
|---|---|
| Controller | `deterministic`, `individual`, `joint` |
| Covariance | `0.25 Sigma`, `Sigma`, `4 Sigma` |
| Trials | 50 paired seeds per cell |
| Confidence interval | Wilson 95% |
| Plant duration | inherited from `mujoco_native_ccmpc.yaml` |
| Empirical collision gate | upper CI at most 0.10 |
| Timing gate | trial-p99 distribution versus the 50 ms control period |

If covariance is multiplied by \(s\), every corresponding standard deviation
is multiplied by \(\sqrt{s}\). This applies to measurement noise, filter
initial/process uncertainty and prediction-horizon process uncertainty.
Dropout probabilities are unchanged.

## Controller comparisons

- `deterministic`: estimator and safety supervisor are active, but chance
  constraints and horizon covariance tightening are disabled.
- `individual`: Stage 4 individual epsilon semantics.
- `joint`: Stage 5 uniform joint budget over all horizon-step/obstacle pairs.

All three use the same seed, plant, obstacle motion, run duration, goal,
actuator limits and supervisor thresholds within a paired cell.

## Artifacts

Each run directory contains:

```text
manifest.yaml
protocol.yaml
base_config.yaml
trials.jsonl
trials.csv
aggregate.json
report.md
report.png
```

`trials.jsonl` is written after every completed episode in serial mode and
after every completed worker batch in parallel mode. The manifest begins in
`RUNNING` state and changes to `COMPLETED` only after all aggregate artifacts
are written.

Manifest schema 2 also records the Git commit/branch, clean state and a
deterministic SHA-256 over validation-relevant source and configuration files.
A release campaign accepts either a clean Git worktree or a fingerprinted,
non-editable installed distribution. Dirty worktrees, editable installs and
unverified loose source trees/distributions are refused. Git and archive
fingerprints include the native MJCF/OBJ collision model as well as source and
configuration. `--allow-dirty-source` is reserved for smoke campaigns and
produces `FAIL_DIRTY_SOURCE` release provenance.

## Running and resuming

```bash
quadrotor-mpc-monte-carlo \
  --config config/native_monte_carlo.yaml \
  --workers 3 \
  --batch-size 5
```

Serial execution (`--workers 1`) maximizes reproducibility and reuses one
compiled controller per controller/noise cell. Parallel execution uses spawned
processes; each batch still reuses one compiled controller for all seeds in the
batch.

Resume an interrupted campaign with the same effective overrides:

```bash
quadrotor-mpc-monte-carlo \
  --config config/native_monte_carlo.yaml \
  --workers 3 \
  --resume outputs/native_monte_carlo/<run-id>
```

Resume is rejected when the protocol fingerprint or validation-source snapshot
differs.

## Claim semantics

The report distinguishes five ideas that must not be collapsed:

1. A geometric collision is observed using MuJoCo truth.
2. Positive slack means a hard chance constraint was relaxed.
3. Fallback means the primary NMPC command was rejected.
4. `BUDGET_OK` means only that the current prediction-horizon allocation sums
   correctly.
5. A finite Monte Carlo confidence interval is empirical evidence, not proof.

For a chance-controller cell, `EMPIRICALLY_SUPPORTED_NOT_PROVEN` requires
sufficient trials, an acceptable collision upper confidence bound, complete
finite execution, correct risk accounting, and no disqualifying slack or
fallback. The joint budget remains a per-solve union bound; it is never reported
as an episode-wide probability guarantee.

## Verified 30-seed campaign

The implementation campaign used paired seeds 1000–1029 across all nine
controller/noise cells: 270 complete episodes.

- collision, NaN and risk-budget failures: zero;
- joint mean minimum clearance:
  `0.343861 m`, `0.414314 m`, `0.660826 m`;
- deterministic mean minimum clearance:
  `0.305261 m`, `0.313791 m`, `0.323614 m`;
- every chance-controller cell contained positive slack;
- fallback episode rate for joint:
  `46.7%`, `56.7%`, `100%`;
- joint success at `4 Sigma`: `0/30` because the final goal error exceeded the
  configured threshold, despite zero geometric collisions;
- overall status: `VALIDATED_WITH_LIMITATIONS`.

Thirty trials satisfy the descriptive sample-size rule, but zero observed
collisions still gives a Wilson 95% upper bound of `0.113513`. The default
50-trial protocol is retained because zero collisions would then lower the
bound below the `0.10` empirical gate. Positive slack, fallback and the failed
50 ms timing gate independently block a probability claim in the verified
campaign.

The committed 30-seed artifact is retained as historical Stage 8 evidence, but
its manifest names the pre-Phase-8 commit `9f4a1b3`. It is therefore not
release-provenance-complete. Version 2.0.1 prevents recurrence by enforcing a
clean source and snapshot-matched resume; the release-scale 50-seed campaign
must be run after the stabilization source is committed.
