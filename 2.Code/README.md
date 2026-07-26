# Quadrotor MPC Research Workbench

A reproducible simulation, comparison and reporting environment for deterministic MPC,
chance-constrained MPC and an optional quaternion NMPC/MuJoCo track.

Version `2.0.1` is a release-stabilized research workbench. The verified native
campaign is `VALIDATED_WITH_LIMITATIONS`: positive slack, fallback and solver
timing currently block a probabilistic-safety or real-time claim.

The workbench is organized around three workflows:

- **Learn**: inspect equations, predicted horizons, covariance and constraint residuals.
- **Run**: execute a closed-loop scenario and generate a complete artifact bundle.
- **Compare**: use paired seeds, Monte Carlo statistics and parameter sweeps.

## Installation

Python 3.10 or newer is required. The locked environment is the preferred path:

```bash
uv sync --all-extras --locked
```

For a standard editable installation:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[full,dev]"
```

The legacy split requirement files remain available for source-checkout use:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-ui.txt
```

## Quick start

Run one tracked experiment:

```bash
quadrotor-mpc-run \
  --config config/scenarios/static_obstacle.yaml \
  --compare
```

Run a 10-seed paired Monte Carlo experiment:

```bash
quadrotor-mpc-run \
  --config config/scenarios/moving_obstacle.yaml \
  --compare --trials 10 --seed 100
```

Open the workbench:

```bash
quadrotor-mpc-dashboard
```

Open the optional NMPC/MuJoCo track in a native Linux desktop window:

```bash
MUJOCO_GL=glfw quadrotor-mpc-native
```

Run the native dynamic-obstacle scenario with the control/telemetry panel:

```bash
MUJOCO_GL=glfw quadrotor-mpc-native \
  --config config/mujoco_native_dynamic.yaml
```

Run the same native track through noisy sensors, a 12D quaternion error-state
EKF and 6D obstacle trackers:

```bash
MUJOCO_GL=glfw quadrotor-mpc-native \
  --config config/mujoco_native_estimation.yaml
```

Run the native spherical CC-MPC with uniform joint-risk allocation, safe
fallback and the Stage 7 desktop safety console:

```bash
MUJOCO_GL=glfw quadrotor-mpc-native \
  --config config/mujoco_native_ccmpc.yaml
```

This configuration distributes `epsilon_total = 0.10` uniformly across all
`(N+1) * n_obstacles` scalar constraints in each NMPC solve. The budget is a
receding-horizon union bound, not an episode-wide guarantee.

Run the Stage 8 paired native Monte Carlo protocol:

```bash
quadrotor-mpc-monte-carlo \
  --config config/native_monte_carlo.yaml \
  --workers 3
```

The default protocol evaluates deterministic-estimated, individual-risk and
joint-uniform controllers with the same 50 seeds at `0.25 Sigma`, `Sigma` and
`4 Sigma`. It writes an append-only checkpoint plus raw CSV, confidence
intervals, paired deltas, claim gates, a Markdown report and a PNG summary.
Interrupted runs can be continued with `--resume <run-directory>`.
Release evidence is accepted from a clean Git source or a fingerprinted,
non-editable installed distribution. A dirty-source smoke campaign requires the
explicit `--allow-dirty-source` flag and is recorded as non-release-eligible
with an exact source snapshot hash.

The interactive session remains open after goal, collision stop, or configured
duration. Use **Run again** in the panel (or `Enter` in the MuJoCo window) to
restart the same scenario, and **Stop**/`Esc` or window close to exit.

The safety console separates episode, applied controller, guarantee
eligibility, risk budget, slack and deadline status. Positive slack and
fallback are always labeled `NOT GUARANTEED`. See
[`docs/DESKTOP_SAFETY_INTERFACE.md`](docs/DESKTOP_SAFETY_INTERFACE.md).

Replay a recorded run without solving the NMPC problem again:

```bash
MUJOCO_GL=glfw quadrotor-mpc-native \
  --replay outputs/native/<timestamp>_<scenario>
```

This native track uses the sourced Bitcraze Crazyflie 2 model from Google DeepMind
MuJoCo Menagerie, not the model under `4.Reference`. See
[`docs/NATIVE_MUJOCO_VIEWER.md`](docs/NATIVE_MUJOCO_VIEWER.md).

The older lightweight command remains available:

```bash
quadrotor-mpc-sim --config config/scenarios/point_to_point.yaml
```

## Dashboard

| Page | Purpose |
|---|---|
| Home | architecture and workflow |
| Scenario Builder | edit, validate and download scenario YAML |
| Live Simulation | 3-D path, predicted horizon, telemetry, uncertainty and solver |
| Compare Controllers | paired deterministic MPC versus CC-MPC |
| Monte Carlo | multi-seed distributions and aggregate metrics |
| Experiment Explorer | reopen manifests, metrics and interactive reports |
| Theory Mode | map equations to code, tests and plots |
| NMPC + MuJoCo | optional 13-state high-fidelity plant track |

## Experiment artifacts

Every `quadrotor-mpc-run` execution creates:

```text
outputs/runs/<run-id>/
├── manifest.yaml
├── scenario.yaml
├── controller.yaml
├── metrics.json
├── comparison.csv
├── report.png
├── report.html
├── *-summary.json
├── *-timeseries.csv
├── *-predictions.npz
└── *-events.jsonl
```

The time series includes actual/estimated/reference states, commands, covariance standard
deviations, clearance, chance residual, slack, solve time, solver iterations and cost terms.
Predicted horizons are stored separately in compressed NumPy format.

## Parameter sweeps

Study the risk-performance tradeoff:

```bash
quadrotor-mpc-sweep \
  --config config/scenarios/static_obstacle.yaml \
  --parameter delta \
  --values 0.01 0.03 0.05 0.10 0.20 \
  --mode both --trials 10
```

Supported sweep parameters are `delta`, `measurement_pos`, `process_vel`, `drag_scale`,
`obstacle_speed_scale` and `horizon_steps`. The sweep writes raw CSV, aggregate JSON and a
four-panel PNG.

## Models and backends

The executable research baseline uses:

```text
x = [px, py, pz, vx, vy, vz, roll, pitch, yaw]
u = [roll_cmd, pitch_cmd, vertical_velocity_cmd, yaw_rate_cmd]
```

- `scipy`: nonlinear rollout with soft chance constraints; suitable for demos, CI and offline sweeps.
- `cvxpy`: sequential-convex QP path that is closer to the reference formulation.
- optional `do-mpc/IPOPT + MuJoCo`: 13-state quaternion model-mismatch experiment using
  the vendored MuJoCo Menagerie Crazyflie 2 rigid body.

Keeping the 9-state controller model and 13-state plant separate is intentional. It prevents a
controller from being validated only against the exact same equations it predicts.

## Chance constraint

With `S = Omega^(1/2)`, `y = S(p-p_obs)` and `n = y/||y||`, the logged residual is

```text
g = ||y|| - 1 - Phi^-1(1-delta) * sqrt(n' S (Sigma + Sigma_obs) S' n)
```

`g >= 0` is locally safe under the Gaussian approximation. A nonzero soft slack must always be
reported; it means the optimization remained numerically usable but the hard probability statement
was relaxed.

## Metrics

Reports cover tracking, safety, efficiency, estimator performance and solver reliability. For a
scientific comparison, use the same plant, start, obstacle path, seed and stop conditions for all
controllers. See [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md).

## Scenarios and tests

The repository includes hover, point-to-point, static obstacle, moving obstacle, rotated ellipsoid
and high-noise stress scenarios. New scenarios can be built in the dashboard.

```bash
python -m unittest discover -s tests -v
```

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/LAYERED_PACKAGE_MIGRATION.md`](docs/LAYERED_PACKAGE_MIGRATION.md)
- [`docs/MODEL_CONVENTIONS.md`](docs/MODEL_CONVENTIONS.md)
- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md)
- [`docs/THEORY_CODE_MAPPING.md`](docs/THEORY_CODE_MAPPING.md)
- [`docs/VALIDATION.md`](docs/VALIDATION.md)
- [`docs/NATIVE_MUJOCO_VIEWER.md`](docs/NATIVE_MUJOCO_VIEWER.md)
- [`docs/CONTROLLER_INTERFACE.md`](docs/CONTROLLER_INTERFACE.md)
- [`docs/NATIVE_SENSOR_ESTIMATION.md`](docs/NATIVE_SENSOR_ESTIMATION.md)
- [`docs/HORIZON_COVARIANCE_PROPAGATION.md`](docs/HORIZON_COVARIANCE_PROPAGATION.md)
- [`docs/SPHERICAL_CHANCE_CONSTRAINTS.md`](docs/SPHERICAL_CHANCE_CONSTRAINTS.md)
- [`docs/RISK_BUDGET_MANAGEMENT.md`](docs/RISK_BUDGET_MANAGEMENT.md)
- [`docs/SAFE_SLACK_FALLBACK.md`](docs/SAFE_SLACK_FALLBACK.md)
- [`docs/DESKTOP_SAFETY_INTERFACE.md`](docs/DESKTOP_SAFETY_INTERFACE.md)
- [`docs/NATIVE_MONTE_CARLO_VALIDATION.md`](docs/NATIVE_MONTE_CARLO_VALIDATION.md)
- [`docs/RELEASE_2_0_1.md`](docs/RELEASE_2_0_1.md)

## Directory map

```text
2.Code/
├── quadrotor_mpc/     installable Python package
│   ├── core/          stable contracts and shared domain values
│   ├── control/       9-state CC-MPC and native quaternion NMPC
│   ├── estimation/    sensor simulation, ESEKF and obstacle tracking
│   ├── infrastructure/ MuJoCo plant and resource adapters
│   ├── application/   simulation, native runtime, experiments and validation
│   ├── reporting/     Plotly, Matplotlib and HTML reports
│   └── interfaces/    CLI, Streamlit and Qt/MuJoCo desktop adapters
├── config/            controller and scenario YAML
├── docs/              architecture and research protocol
├── models/            sourced MuJoCo MJCF and mesh assets
├── tests/             unit, integration and architecture tests
├── validation/        committed evidence
└── pyproject.toml     build metadata and six console entry points
```
