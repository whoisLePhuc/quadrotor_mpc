# Quadrotor MPC Research Workbench

A reproducible simulation, comparison and reporting environment for deterministic MPC,
chance-constrained MPC and an optional quaternion NMPC/MuJoCo track.

The workbench is organized around three workflows:

- **Learn**: inspect equations, predicted horizons, covariance and constraint residuals.
- **Run**: execute a closed-loop scenario and generate a complete artifact bundle.
- **Compare**: use paired seeds, Monte Carlo statistics and parameter sweeps.

## Installation

Python 3.10 or newer is required; Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the Streamlit dashboard and optional NMPC/MuJoCo track:

```bash
python -m pip install -r requirements-ui.txt
```

## Quick start

Run one tracked experiment:

```bash
python run_experiment.py \
  --config config/scenarios/static_obstacle.yaml \
  --compare
```

Run a 10-seed paired Monte Carlo experiment:

```bash
python run_experiment.py \
  --config config/scenarios/moving_obstacle.yaml \
  --compare --trials 10 --seed 100
```

Open the workbench:

```bash
streamlit run dashboard/Home.py
# or: python start_dashboard.py
```

The older lightweight command remains available:

```bash
python run_simulation.py --config config/scenarios/point_to_point.yaml
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

Every `run_experiment.py` execution creates:

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
python run_sweep.py \
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
- optional `do-mpc/IPOPT + MuJoCo`: 13-state quaternion model-mismatch experiment.

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
- [`docs/MODEL_CONVENTIONS.md`](docs/MODEL_CONVENTIONS.md)
- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md)
- [`docs/THEORY_CODE_MAPPING.md`](docs/THEORY_CODE_MAPPING.md)
- [`docs/VALIDATION.md`](docs/VALIDATION.md)

## Directory map

```text
2.Code/
├── ccmpc/             mathematical controller core
├── simulation/        closed-loop runtime and metrics
├── experiments/       tracking, aggregation and sweeps
├── reporting/         Plotly and HTML reports
├── dashboard/         Streamlit multi-page workbench
├── config/            controller and scenario YAML
├── docs/              architecture and research protocol
├── models/            MuJoCo XML
├── tests/             unit and integration tests
├── run_experiment.py  tracked experiment CLI
├── run_sweep.py       parameter-sweep CLI
├── start_dashboard.py dashboard launcher for the active Python environment
└── run_simulation.py  compact backward-compatible CLI
```
