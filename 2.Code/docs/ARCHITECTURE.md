# Architecture

The workbench separates the controller model, true plant, estimator and experiment layer.

```mermaid
flowchart LR
    A[Scenario and seed] --> B[Estimator and covariance]
    B --> C[Controller interface]
    C --> D[ODE 9D or MuJoCo 13D]
    D --> B
    B --> E[Experiment logger]
    C --> E
    D --> E
    E --> F[Dashboard and reports]
```

The 9-state plant is the executable research baseline. The optional 13-state quaternion/MuJoCo
track is deliberately separate and is used to study model mismatch. Both must not be described as
the same physical model.

Core ownership:

| Layer | Modules | Responsibility |
|---|---|---|
| Mathematics | `ccmpc/` | dynamics, uncertainty, ellipsoid risk and QP |
| Runtime | `simulation/` | controller adapter, plant loop, metrics and static report |
| Experiments | `experiments/` | run identity, manifest, paired statistics and sweeps |
| Reporting | `reporting/` | interactive Plotly figures and HTML |
| UI | `dashboard/` | Learn, Run, Compare and Explore workflows |
| Native MuJoCo | `mujoco_plant.py`, `mujoco_native.py` | sourced Crazyflie plant and passive desktop viewer |
| Native interaction | `runtime_control.py`, `native_desktop_panel.py` | command queue and separate Qt telemetry process |
| Native evidence | `native_telemetry.py`, `native_replay.py` | bounded data, recording bundle and solver-free replay |
| Obstacle motion | `obstacle_motion.py` | one predictor shared by controller, plant, metrics and viewer |
| Controller contract | `controller_interface.py` | belief, goal and normalized solution types |
| Native deterministic adapter | `deterministic_nmpc_controller.py` | do-mpc backend behind the shared contract |
| Native estimation | `native_estimation.py` | seeded sensors, 12D ESEKF and 6D obstacle trackers |
| Exact baseline source | `belief_from_truth.py` | zero-covariance truth adapter used only for regression |

Dependencies point inward: UI and reporting consume experiment/runtime data; the mathematics layer
does not depend on the dashboard.

The native viewer is connected through the `CoupledRuntime` lifecycle protocol.
It never owns the controller or advances physics, so closing/rendering the window
cannot create a second simulation path. `4.Reference` is not a production dependency.

The Qt panel runs in a child process because Qt and GLFW both have main-thread
event-loop requirements. Only plain command and telemetry dictionaries cross the
process boundary. The simulation process remains the sole owner of do-mpc and
MuJoCo state.

## Native controller boundary

`run_coupled.py` no longer calls `mpc.make_step` or reads do-mpc prediction data.
It calls the backend-independent contract:

```python
controller.reset(vehicle_belief)
solution = controller.solve(vehicle_belief, obstacle_beliefs, goal, time_s)
```

The deterministic baseline creates zero-covariance beliefs from MuJoCo truth in
`belief_from_truth.py`. The estimated configuration instead routes truth through
the seeded sensor simulator, 12D vehicle error-state EKF and one 6D
constant-velocity tracker per obstacle in `native_estimation.py`. Both paths
produce the same controller contract without changing runtime signatures.

All obstacle centers are TVPs in the native NMPC model. Geometry and obstacle
count define the compiled NLP; estimated mean trajectories can change at every
control tick without rebuilding it.
