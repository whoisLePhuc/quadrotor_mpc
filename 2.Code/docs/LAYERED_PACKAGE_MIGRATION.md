# Layered Package Migration

The refactor after commit `e75d0f3` changes Python import paths without changing
the six public console commands, YAML schemas, runtime behavior or validation
semantics.

## Import mapping

| Previous module | Canonical module |
|---|---|
| `controller_interface` | `quadrotor_mpc.core.contracts` |
| `vehicle` | `quadrotor_mpc.core.vehicle` |
| `obstacle_motion` | `quadrotor_mpc.core.obstacle_motion` |
| `ccmpc.*` | `quadrotor_mpc.control.ccmpc.*` |
| `quad_mpc_core` | `quadrotor_mpc.control.nmpc.core` |
| `deterministic_nmpc_controller` | `quadrotor_mpc.control.nmpc.deterministic` |
| `chance_constrained_nmpc_controller` | `quadrotor_mpc.control.nmpc.chance_constrained` |
| `native_covariance` | `quadrotor_mpc.control.nmpc.covariance` |
| `native_chance_constraints` | `quadrotor_mpc.control.nmpc.chance_constraints` |
| `native_risk_budget` | `quadrotor_mpc.control.nmpc.risk_budget` |
| `native_safety_fallback` | `quadrotor_mpc.control.nmpc.safety` |
| `native_estimation` | `quadrotor_mpc.estimation.native` |
| `belief_from_truth` | `quadrotor_mpc.estimation.truth` |
| `mujoco_plant` | `quadrotor_mpc.infrastructure.mujoco.plant` |
| `run_coupled` | `quadrotor_mpc.application.native.runtime` |
| `native_telemetry` | `quadrotor_mpc.application.native.telemetry` |
| `native_replay` | `quadrotor_mpc.application.native.replay` |
| `native_monte_carlo` | `quadrotor_mpc.application.validation.monte_carlo` |
| `mujoco_native` | `quadrotor_mpc.interfaces.desktop.viewer` |
| `native_ui_model` | `quadrotor_mpc.interfaces.desktop.model` |
| `native_desktop_panel` | `quadrotor_mpc.interfaces.desktop.panel` |

Example:

```python
from quadrotor_mpc.core.contracts import ControlGoal, VehicleBelief
from quadrotor_mpc.control.nmpc.deterministic import DeterministicNMPCController
```

## Stable console interface

The following commands are unchanged:

```text
quadrotor-mpc-run
quadrotor-mpc-sim
quadrotor-mpc-sweep
quadrotor-mpc-native
quadrotor-mpc-monte-carlo
quadrotor-mpc-dashboard
```

Direct execution of removed root scripts such as `python run_simulation.py` is
replaced by the corresponding console command. This keeps source checkout and
installed-wheel behavior identical.

## Dependency policy

- `core` contains no controller, simulator or UI dependency.
- `control` and `estimation` cannot depend on application or presentation code.
- `infrastructure` cannot depend on CLI, dashboard or desktop presentation.
- `application` orchestrates use cases; `interfaces` translate user input and
  render output.
- Package initializers remain lightweight; import concrete implementations from
  their modules.

`tests/test_architecture.py` and `scripts/release_check.py` enforce these rules.
