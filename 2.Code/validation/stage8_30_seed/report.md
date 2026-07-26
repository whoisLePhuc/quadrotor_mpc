# Native Monte Carlo Validation — Stage 8

- Overall status: **VALIDATED_WITH_LIMITATIONS**
- Confidence level: 95.0%
- Trials per controller/noise level: 30
- Paired seeds: 1000–1029

## Aggregate results

| Noise | Controller | Success (CI) | Collision (CI) | Min clearance mean | Max slack mean | Solver p99 median | Claim blockers |
|---|---|---:|---:|---:|---:|---:|---|
| quarter_sigma (0.25Σ) | deterministic | 1.000 [0.886, 1.000] | 0.000 [0.000, 0.114] | 0.305261 m | 0.000000 m | 67.460 ms | NOT_APPLICABLE_DETERMINISTIC |
| quarter_sigma (0.25Σ) | individual | 1.000 [0.886, 1.000] | 0.000 [0.000, 0.114] | 0.326352 m | 0.062461 m | 95.470 ms | COLLISION_CONFIDENCE_BOUND, POSITIVE_SLACK, FALLBACK |
| quarter_sigma (0.25Σ) | joint | 1.000 [0.886, 1.000] | 0.000 [0.000, 0.114] | 0.343861 m | 0.056917 m | 92.175 ms | COLLISION_CONFIDENCE_BOUND, POSITIVE_SLACK, FALLBACK |
| nominal_sigma (1Σ) | deterministic | 1.000 [0.886, 1.000] | 0.000 [0.000, 0.114] | 0.313791 m | 0.000000 m | 67.097 ms | NOT_APPLICABLE_DETERMINISTIC |
| nominal_sigma (1Σ) | individual | 1.000 [0.886, 1.000] | 0.000 [0.000, 0.114] | 0.356432 m | 0.061792 m | 94.862 ms | COLLISION_CONFIDENCE_BOUND, POSITIVE_SLACK, FALLBACK |
| nominal_sigma (1Σ) | joint | 1.000 [0.886, 1.000] | 0.000 [0.000, 0.114] | 0.414314 m | 0.070586 m | 87.523 ms | COLLISION_CONFIDENCE_BOUND, POSITIVE_SLACK, FALLBACK |
| four_sigma (4Σ) | deterministic | 1.000 [0.886, 1.000] | 0.000 [0.000, 0.114] | 0.323614 m | 0.000000 m | 61.542 ms | NOT_APPLICABLE_DETERMINISTIC |
| four_sigma (4Σ) | individual | 1.000 [0.886, 1.000] | 0.000 [0.000, 0.114] | 0.467242 m | 0.079857 m | 90.988 ms | COLLISION_CONFIDENCE_BOUND, POSITIVE_SLACK, FALLBACK |
| four_sigma (4Σ) | joint | 0.000 [0.000, 0.114] | 0.000 [0.000, 0.114] | 0.660826 m | 0.139326 m | 92.240 ms | COLLISION_CONFIDENCE_BOUND, POSITIVE_SLACK, FALLBACK |

## Gate interpretation

- Execution integrity: **PASS**
- Required sample size: **PASS**
- Real-time gate: **FAIL**
- Probabilistic claim: **BLOCKED**

A finite Monte Carlo result is empirical evidence only. The configured joint risk is per receding prediction horizon and is not an episode-wide probability guarantee.

Positive slack means the hard chance constraint was relaxed. Fallback means the primary NMPC command was not applied. Either event blocks a probabilistic safety claim under this protocol, even when no geometric collision was observed.

## Reproduction

Use `run_native_monte_carlo.py --config config/native_monte_carlo.yaml`. The manifest records the source revision, environment, protocol fingerprint and expected trial count. `trials.jsonl` is append-only and supports resume.
