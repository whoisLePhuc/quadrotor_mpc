"""State-estimator interfaces and a full-state extended Kalman filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from quadrotor_mpc.control.ccmpc.dynamics import QuadrotorDynamics

Array = npt.NDArray[np.float64]


@dataclass(slots=True)
class ExtendedKalmanEstimator:
    """EKF using the controller dynamics and a full-state VIO-like measurement.

    The measurement matrix is identity because the simulation exposes noisy position,
    velocity and attitude. The nonlinear state is propagated with RK4 and covariance
    is propagated with the local continuous-time Jacobian.
    """

    dynamics: QuadrotorDynamics
    process_covariance: Array
    measurement_covariance: Array
    state: Array | None = None
    covariance: Array | None = None

    def reset(self, measurement: Array, covariance: Array) -> None:
        self.state = np.asarray(measurement, dtype=float).copy()
        self.covariance = np.asarray(covariance, dtype=float).copy()

    def predict(self, command: Array, dt: float) -> None:
        if self.state is None or self.covariance is None:
            raise RuntimeError("estimator must be reset before predict")
        jacobian = self.dynamics.jacobian_state(self.state, command)
        transition = np.eye(9) + dt * jacobian
        self.state = self.dynamics.discrete(self.state, command, dt)
        self.covariance = (
            transition @ self.covariance @ transition.T
            + self.process_covariance * dt
        )
        self._symmetrize()

    def update(self, measurement: Array) -> None:
        if self.state is None or self.covariance is None:
            raise RuntimeError("estimator must be reset before update")
        measurement = np.asarray(measurement, dtype=float)
        innovation = measurement - self.state
        innovation[8] = (innovation[8] + np.pi) % (2.0 * np.pi) - np.pi
        innovation_covariance = self.covariance + self.measurement_covariance
        gain = np.linalg.solve(innovation_covariance, self.covariance).T
        self.state = self.state + gain @ innovation
        self.state[8] = (self.state[8] + np.pi) % (2.0 * np.pi) - np.pi

        identity = np.eye(9)
        remainder = identity - gain
        self.covariance = (
            remainder @ self.covariance @ remainder.T
            + gain @ self.measurement_covariance @ gain.T
        )
        self._symmetrize()

    def _symmetrize(self) -> None:
        assert self.covariance is not None
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
        if eigenvalues.min() < 0.0:
            self.covariance = (
                eigenvectors * np.maximum(eigenvalues, 0.0)
            ) @ eigenvectors.T
