# Native controller interface

Stage 1 separates the native MuJoCo loop from do-mpc through a belief-based
contract. The contract is defined in `controller_interface.py` and has no
MuJoCo, CasADi or do-mpc dependency.

## Lifecycle

```python
controller.reset(initial_vehicle_belief)

solution = controller.solve(
    vehicle_belief,
    obstacle_beliefs,
    goal,
    time_s,
)
```

`reset()` clears controller history and warm-start state. It is called at
startup, Reset and Run again. `solve()` is the only control action used by
`run_coupled.py`.

## Input contracts

| Type | Meaning | Shape |
|---|---|---:|
| `VehicleBelief.mean_state_13` | `[p, v, q_wxyz, omega]` nominal state | `(13,)` |
| `VehicleBelief.error_covariance_12` | covariance of `[δp, δv, δθ, δω]` | `(12, 12)` |
| `ObstacleBelief.mean_state_6` | obstacle `[position, velocity]` mean | `(6,)` |
| `ObstacleBelief.covariance_6` | obstacle position/velocity covariance | `(6, 6)` |
| `ObstacleBelief.predicted_positions` | optional nonlinear mean horizon | `(N+1, 3)` |
| `ControlGoal.position` | world-frame goal position | `(3,)` |
| `ControlGoal.quaternion_wxyz` | normalized goal attitude | `(4,)` |

Covariances must be finite, symmetric and positive semidefinite. Quaternion
inputs are normalized at construction.

## Output contract

| Field | Meaning | Shape |
|---|---|---:|
| `command` | `[thrust_deviation, τx, τy, τz]` | `(4,)` |
| `nominal_states` | optimized mean state horizon | `(N+1, 13)` |
| `predicted_covariances` | error-state covariance horizon | `(N+1, 12, 12)` |
| `chance_margins` | signed obstacle constraint margins | `(N+1, n_obs)` |
| `risk_allocations` | allocated risk per horizon constraint | `(N+1, n_obs)` |
| `slacks` | nonnegative constraint relaxation | `(N+1, n_obs)` |
| `solver_status` | backend-independent status label | string |

The deterministic adapter returns zero predicted covariance and zero risk
allocation. Its geometric margin and implied violation slack are populated so
the same telemetry schema is already exercised before CC-MPC is introduced.

## Current deterministic adapter

`DeterministicNMPCController` owns all do-mpc-specific operations:

- model/controller construction;
- TVP updates;
- warm-start reset;
- `make_step`;
- optimized-horizon extraction;
- conversion to `ControlSolution`.

All obstacle centers are TVPs. The compiled NLP depends on obstacle count and
shape, but obstacle mean trajectories may change on every tick.

## Belief sources

`belief_from_truth.py` converts native truth into zero-covariance beliefs only
when `estimation.enabled: false`. It remains the deterministic regression
baseline.

When `estimation.enabled: true`, `native_estimation.py` provides:

```text
MuJoCo truth → sensor simulation → vehicle/obstacle estimators → beliefs
```

The vehicle estimator is a 12D error-state EKF around the 13D quaternion
nominal state. Each obstacle has a position-only constant-velocity Kalman
filter. Tracker means—not configured truth motion—form the obstacle horizon.

No production code imports `4.Reference`.
