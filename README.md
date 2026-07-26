# Quadrotor MPC Research Workbench

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Validation](https://img.shields.io/badge/validation-with%20limitations-orange)](2.Code/docs/NATIVE_MONTE_CARLO_VALIDATION.md)

A reproducible research workbench for deterministic MPC and chance-constrained
MPC (CC-MPC) obstacle avoidance with a native 13-state MuJoCo/ESEKF validation
track.

> **Current status:** feature-complete research prototype, empirically validated
> with documented limitations. It is not a production flight controller and
> does not currently establish an episode-wide probabilistic safety guarantee
> or 20 Hz real-time feasibility.

## What is implemented

- A 9-state executable MPC/CC-MPC research baseline with paired experiments,
  parameter sweeps, reports and a Streamlit dashboard.
- A separate 13-state quaternion NMPC/MuJoCo plant using the sourced Bitcraze
  Crazyflie 2 model.
- Seeded sensor simulation, a 12D error-state EKF and probabilistic obstacle
  trackers.
- Horizon covariance propagation, spherical chance-constraint tightening and
  individual or joint-uniform risk allocation.
- Explicit soft-slack reporting, solution acceptance gates and bounded fallback
  control.
- A native desktop safety console, recording/replay and paired Monte Carlo
  validation with Wilson confidence intervals.

The controller receives estimated beliefs rather than hidden MuJoCo truth.
The 9-state and 13-state models intentionally remain separate to expose model
mismatch.

## Validation status

The committed Stage 8 campaign contains 270 episodes (30 paired seeds per
controller/noise cell):

- `0/270` geometric collisions, NaNs or risk-budget accounting failures.
- Every chance-controller cell contained positive slack.
- Joint-risk fallback increased from `46.7%` to `100%` across the uncertainty
  sweep.
- Joint risk at \(4\Sigma\) reached the goal in `0/30` episodes.
- Solver p99 did not satisfy the `50 ms` controller period.
- With 30 trials and zero collisions, the Wilson 95% upper bound is `0.113513`,
  above the empirical gate of `0.10`.

The result is therefore `VALIDATED_WITH_LIMITATIONS`; zero observed collisions
must not be presented as a formal probabilistic guarantee. See the
[native validation protocol](2.Code/docs/NATIVE_MONTE_CARLO_VALIDATION.md) and
[raw evidence](2.Code/validation/stage8_30_seed/).

## Installation

Python 3.10 or newer is required. Python 3.12 is used by the current validation
artifact.

### Reproducible environment with uv

```bash
git clone https://github.com/whoisLePhuc/quadrotor_mpc.git
cd quadrotor_mpc/2.Code
uv sync --all-extras --locked
```

### Standard pip installation

```bash
cd quadrotor_mpc/2.Code
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[full,dev]"
```

MuJoCo and the Qt desktop panel need a working desktop OpenGL/EGL runtime. The
ODE baseline and headless Monte Carlo do not need a visible window.

## Quick start

Run a tracked paired ODE experiment:

```bash
cd 2.Code
uv run quadrotor-mpc-run \
  --config config/scenarios/static_obstacle.yaml \
  --compare
```

Run the compact ODE simulation:

```bash
uv run quadrotor-mpc-sim \
  --config config/scenarios/moving_obstacle.yaml \
  --compare
```

Launch the dashboard:

```bash
uv run quadrotor-mpc-dashboard
```

Launch the native CC-MPC scenario on a Linux desktop:

```bash
MUJOCO_GL=glfw uv run quadrotor-mpc-native \
  --config config/mujoco_native_ccmpc.yaml
```

Run the default 50-seed/cell native validation from a clean commit:

```bash
uv run quadrotor-mpc-monte-carlo \
  --config config/native_monte_carlo.yaml \
  --workers 3 \
  --batch-size 5
```

The Monte Carlo command accepts release evidence only from a clean Git tree or
a fingerprinted, non-editable installed distribution. `--allow-dirty-source`
exists only for explicitly non-release smoke campaigns and records the exact
source snapshot as non-release-eligible.

## Repository map

```text
quadrotor_mpc/
├── 1.Docs/
│   ├── Theory/          theory chapters and formula reference
│   └── Paper/           source papers
├── 2.Code/
│   ├── ccmpc/           9-state controller mathematics
│   ├── simulation/      closed-loop ODE runtime
│   ├── config/          native and ODE configurations
│   ├── dashboard/       Streamlit research interface
│   ├── models/          sourced MuJoCo model and license
│   ├── tests/           unit, integration and release tests
│   ├── validation/      committed validation evidence
│   └── docs/            architecture, protocol and limitations
├── 3.Notebooks/         learning and formulation notebooks
├── LICENSE
└── README.md
```

## Scope boundaries

This repository does not currently implement:

- a depth-camera or vision detection pipeline;
- ROS 2 integration;
- hardware-in-the-loop or real flight deployment;
- a solver proven to meet the `50 ms` controller period;
- a formal episode-wide collision-probability proof.

Those are separate future research/deployment tracks, not hidden capabilities
of the current release.

## Documentation

- [Code workbench guide](2.Code/README.md)
- [Architecture](2.Code/docs/ARCHITECTURE.md)
- [Model conventions](2.Code/docs/MODEL_CONVENTIONS.md)
- [Safety fallback](2.Code/docs/SAFE_SLACK_FALLBACK.md)
- [Desktop safety interface](2.Code/docs/DESKTOP_SAFETY_INTERFACE.md)
- [Native Monte Carlo validation](2.Code/docs/NATIVE_MONTE_CARLO_VALIDATION.md)
- [Release stabilization](2.Code/docs/RELEASE_2_0_1.md)
- [Theory map](1.Docs/Theory/README.md)

## References

- H. Zhu and J. Alonso-Mora, “Chance-Constrained Collision Avoidance
  for MAVs in Dynamic Environments,” IEEE RA-L, 2019.
- J. Lin, H. Zhu and J. Alonso-Mora, “Robust Vision-based Obstacle
  Avoidance for MAVs in Dynamic Environments,” IEEE ICRA, 2020.

The papers motivate the formulation; their published performance figures are
not presented as measurements of this implementation.

## License

The workbench is released under the [MIT License](LICENSE). The vendored
Crazyflie model retains its own upstream MIT license in
`2.Code/models/bitcraze_crazyflie_2/LICENSE`.
