# Controller interface

Every controller used by the ODE experiment runner implements:

```python
controller.reset()
result = controller.solve(state_estimate, goal, obstacles, covariance)
```

`ControlResult` contains:

- first receding-horizon command;
- complete predicted state and control horizon;
- solver time, status and iteration count;
- maximum chance slack and minimum residual;
- cost decomposition when the backend exposes it.

The SciPy backend is a nonlinear soft-constrained demonstrator and offline sweep backend. The
CVXPY backend is the sequential-convex QP path. A successful offline solve is not proof of real-time
feasibility; inspect solver p95 and deadline-miss rate.
