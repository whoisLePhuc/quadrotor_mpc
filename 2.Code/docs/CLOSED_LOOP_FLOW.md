# Closed-loop flow

At every plant step the runner propagates the true state and covariance. At controller ticks it
also solves a new finite-horizon problem and applies only the first command.

```mermaid
sequenceDiagram
    participant R as Runner
    participant E as Estimator
    participant C as MPC controller
    participant P as Plant
    participant L as Logger
    R->>E: noisy measurement
    E->>C: state estimate and covariance
    C-->>R: u0, horizon, residual, slack
    R->>P: apply u0 for dt
    P-->>E: next measurement
    R->>L: states, prediction, cost and events
```

The update interval of the controller may differ from the plant integration step. Solver deadline
misses are recorded against the controller interval, not the plant step.
