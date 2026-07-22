# Theory-to-code mapping

| Theory | Implementation | Test | Observable evidence |
|---|---|---|---|
| Nonlinear 9-state dynamics and RK4 | `ccmpc/dynamics.py` | `test_dynamics.py` | state telemetry |
| Linearization | `ccmpc/dynamics.py` | `test_dynamics.py` | predicted horizon |
| Covariance propagation | `ccmpc/uncertainty.py` | `test_uncertainty.py` | sigma plot |
| EKF predict/update | `simulation/estimators.py` | `test_estimators.py` | actual versus estimate |
| Rotated ellipsoid matrix root | `ccmpc/risk.py` | `test_risk.py` | clearance plot |
| Chance residual | `ccmpc/risk.py` | `test_risk.py` | residual and slack |
| Receding-horizon controller | `simulation/controllers.py` | `test_runner.py` | prediction replay |
| Paired controller comparison | `experiments/manager.py` | `test_experiments.py` | comparison CSV/report |
| Parameter sweep | `experiments/sweep.py` | `test_sweep.py` | sweep CSV/PNG |

For every new theoretical equation, add a code location, a focused test and a plot or metric that
can falsify an incorrect implementation.
