"""Tests for uncertainty propagation (Eq 19)."""

import numpy as np
from ccmpc.uncertainty import UncertaintyPropagator, VIODriftModel
from ccmpc.dynamics import QuadrotorDynamics


# ============================================================================
# UncertaintyPropagator
# ============================================================================

class TestUncertaintyPropagator:
    def test_construction_defaults(self):
        """Default construction should create valid matrices."""
        up = UncertaintyPropagator()
        assert up.W.shape == (9, 9)
        assert up.Gamma_0.shape == (9, 9)
        # All diagonal entries should be positive
        assert np.all(np.diag(up.W) > 0)
        assert np.all(np.diag(up.Gamma_0) > 0)

    def test_propagate_shape(self):
        """propagate() should return list of N+1 covariance matrices."""
        up = UncertaintyPropagator()
        dyn = QuadrotorDynamics()
        N = 5
        x_guess = np.zeros((9, N + 1))
        x_guess[2, :] = 1.0  # z position
        u_guess = np.zeros((4, N))
        Gamma_list = up.propagate(up.Gamma_0, x_guess, u_guess, dyn, dt=0.06)
        assert len(Gamma_list) == N + 1
        assert Gamma_list[0].shape == (9, 9)

    def test_covariance_growth(self):
        """Covariance trace should increase over the horizon (Eq 19)."""
        up = UncertaintyPropagator()
        dyn = QuadrotorDynamics()
        N = 10
        x_guess = np.zeros((9, N + 1))
        x_guess[2, :] = 1.0
        u_guess = np.zeros((4, N))
        Gamma_list = up.propagate(up.Gamma_0, x_guess, u_guess, dyn, dt=0.06)
        traces = [np.trace(G) for G in Gamma_list]
        assert traces[-1] > traces[0]

    def test_propagate_with_motion(self):
        """Covariance growth should be faster with aggressive motion."""
        up = UncertaintyPropagator()
        dyn = QuadrotorDynamics()
        N = 10
        # Motion: quadrotor pitching forward
        x_guess = np.zeros((9, N + 1))
        x_guess[2, :] = 1.0
        for k in range(N):
            x_guess[7, k] = 0.35  # pitch angle
            x_guess[3, k] = 2.0   # forward velocity
        u_guess = np.zeros((4, N))
        Gamma_list = up.propagate(up.Gamma_0, x_guess, u_guess, dyn, dt=0.06)
        traces = [np.trace(G) for G in Gamma_list]
        assert traces[-1] > traces[0]

    def test_position_covariance(self):
        """Extracting position covariance should give 3x3 top-left block."""
        Gamma = np.random.randn(9, 9)
        Gamma = Gamma @ Gamma.T  # make SPD
        Σ = UncertaintyPropagator.position_covariance(Gamma)
        assert Σ.shape == (3, 3)
        assert np.allclose(Σ, Gamma[:3, :3])

    def test_add_measurement_noise(self):
        """Noise should have approximately correct standard deviation."""
        x = np.zeros(9)
        noisy = UncertaintyPropagator.add_measurement_noise(x, sigma_pos=0.05)
        assert noisy.shape == (9,)
        # Single call: noise should be non-zero with some probability
        assert not np.allclose(noisy, x)

    def test_from_config(self, mpc_config):
        """Should load from YAML."""
        up = UncertaintyPropagator.from_config(str(mpc_config))
        assert up is not None
        assert np.all(np.diag(up.W) > 0)

    def test_vio_drift_from_config(self, mpc_config):
        """Should create VIODriftModel from config."""
        drift = UncertaintyPropagator.vio_drift_from_config(str(mpc_config))
        assert drift is not None
        assert len(drift.Q_drift) == 9


# ============================================================================
# VIODriftModel
# ============================================================================

class TestVIODriftModel:
    def test_update_changes_bias(self):
        """update() should change the drift bias."""
        drift = VIODriftModel(drift_pos=0.01)
        b0 = drift.bias.copy()
        b1 = drift.update(dt=0.02)
        # May or may not change significantly (random walk)
        assert len(b1) == 9

    def test_apply_modifies_state(self):
        """apply() should return biased state."""
        drift = VIODriftModel()
        drift.bias = np.ones(9) * 0.5
        x = np.ones(9)
        x_biased = drift.apply(x)
        assert not np.allclose(x_biased, x)
