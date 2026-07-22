# Validation

Validation is layered:

1. Unit tests: dynamics equilibrium, linearization, rotated matrix square root and covariance PSD.
2. Integration tests: deterministic repeatability, artifact schema and adaptive reports.
3. Scenario regression: hover, point-to-point, static and moving obstacles.
4. Statistical validation: paired Monte Carlo and parameter sweeps.
5. Cross-plant validation: 9-state ODE controller model versus 13-state MuJoCo plant.
6. MuJoCo model validation: sourced mass/inertia, MJCF compilation, hover equilibrium,
   collision contacts and native-viewer lifecycle.

Before claiming a chance-constraint probability, verify Gaussian assumptions, covariance
calibration, obstacle prediction error, finite-sample collision rate and nonzero slack. FOV and soft
constraints require their own residual plots and must not be inferred from trajectory appearance.
