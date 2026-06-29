"""Unit tests for uncertainty propagation.

Target module:
    ccmpc.uncertainty

These tests verify the covariance propagation contract used by the core
CC-MPC controller without depending on CVXPY, obstacle code, or simulation
runtime.
"""

from __future__ import annotations

import numpy as np
import pytest

from ccmpc.uncertainty import (
    UncertaintyConfig,
    UncertaintyConfigError,
    UncertaintyPropagationError,
    UncertaintyPropagator,
    as_control_trajectory_control_major,
    as_dynamics_jacobian,
    as_state_trajectory_state_major,
    grouped_state_variances,
    project_to_psd_gamma,
    symmetrize,
)


class IdentityDynamics:
    """Fake dynamics whose linearized A matrix is identity."""

    def __init__(self) -> None:
        self.calls = 0

    def linearize(
        self,
        x_bar: np.ndarray,
        u_bar: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.calls += 1
        A = np.eye(9, dtype=np.float64)
        B = np.zeros((9, 4), dtype=np.float64)
        C = np.zeros(9, dtype=np.float64)
        return A, B, C


class PositionVelocityDynamics:
    """Fake dynamics with x/y/z position depending on vx/vy/vz."""

    def __init__(self) -> None:
        self.calls = 0

    def linearize(
        self,
        x_bar: np.ndarray,
        u_bar: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.calls += 1
        A = np.eye(9, dtype=np.float64)
        A[0, 3] = dt
        A[1, 4] = dt
        A[2, 5] = dt
        B = np.zeros((9, 4), dtype=np.float64)
        C = np.zeros(9, dtype=np.float64)
        return A, B, C


class BadDynamicsNoLinearize:
    """Fake object without linearize method."""


class BadDynamicsWrongShape:
    """Fake dynamics returning invalid A shape."""

    def linearize(
        self,
        x_bar: np.ndarray,
        u_bar: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.eye(8, dtype=np.float64),
            np.zeros((9, 4), dtype=np.float64),
            np.zeros(9, dtype=np.float64),
        )


class BadDynamicsRaises:
    """Fake dynamics that raises from linearize."""

    def linearize(
        self,
        x_bar: np.ndarray,
        u_bar: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise RuntimeError("linearization failed")


def make_direct_config() -> dict:
    """Create uncertainty-only config."""
    return {
        "dt": 0.1,
        "process_noise_pos": 0.01,
        "process_noise_vel": 0.1,
        "process_noise_att": 0.02,
        "init_pos_noise": 0.05,
        "init_vel_noise": 0.1,
        "init_att_noise": 0.03,
        "vio_drift_pos": 0.005,
        "vio_drift_vel": 0.02,
        "vio_drift_att": 0.005,
        "covariance_floor": 1e-12,
    }


def make_full_config() -> dict:
    """Create full project-style config."""
    return {
        "controller": {
            "prediction": {
                "timestep": 0.1,
            },
            "uncertainty": {
                "process_noise_pos": 0.01,
                "process_noise_vel": 0.1,
                "process_noise_att": 0.02,
                "init_pos_noise": 0.05,
                "init_vel_noise": 0.1,
                "init_att_noise": 0.03,
                "vio_drift_pos": 0.005,
                "vio_drift_vel": 0.02,
                "vio_drift_att": 0.005,
                "covariance_floor": 1e-12,
            },
        }
    }


def make_state_trajectory_state_major(horizon: int = 3) -> np.ndarray:
    """Create valid x_traj with shape (9, N+1)."""
    x_traj = np.zeros((9, horizon + 1), dtype=np.float64)
    x_traj[2, :] = 1.0
    x_traj[0, :] = np.linspace(0.0, 0.3, horizon + 1)
    return x_traj


def make_control_trajectory_control_major(horizon: int = 3) -> np.ndarray:
    """Create valid u_traj with shape (4, N)."""
    return np.zeros((4, horizon), dtype=np.float64)


def test_uncertainty_config_from_direct_config() -> None:
    """UncertaintyConfig.from_config should parse uncertainty-only dict."""
    config = UncertaintyConfig.from_config(make_direct_config())

    assert config.dt == pytest.approx(0.1)
    assert config.process_noise_pos == pytest.approx(0.01)
    assert config.process_noise_vel == pytest.approx(0.1)
    assert config.process_noise_att == pytest.approx(0.02)
    assert config.init_pos_noise == pytest.approx(0.05)
    assert config.init_vel_noise == pytest.approx(0.1)
    assert config.init_att_noise == pytest.approx(0.03)
    assert config.vio_drift_pos == pytest.approx(0.005)
    assert config.vio_drift_vel == pytest.approx(0.02)
    assert config.vio_drift_att == pytest.approx(0.005)


def test_uncertainty_config_from_full_config() -> None:
    """UncertaintyConfig.from_config should parse full project-style config."""
    config = UncertaintyConfig.from_config(make_full_config())

    assert config.dt == pytest.approx(0.1)
    assert config.process_noise_pos == pytest.approx(0.01)
    assert config.init_pos_noise == pytest.approx(0.05)


def test_uncertainty_config_rejects_missing_dt() -> None:
    """UncertaintyConfig should reject configs without dt/timestep."""
    bad_config = make_direct_config()
    bad_config.pop("dt")

    with pytest.raises(UncertaintyConfigError, match="timestep"):
        UncertaintyConfig.from_config(bad_config)


def test_uncertainty_config_rejects_missing_required_key() -> None:
    """UncertaintyConfig should reject missing required noise fields."""
    bad_config = make_direct_config()
    bad_config.pop("process_noise_pos")

    with pytest.raises(UncertaintyConfigError, match="process_noise_pos"):
        UncertaintyConfig.from_config(bad_config)


def test_uncertainty_config_rejects_negative_noise() -> None:
    """UncertaintyConfig should reject negative standard deviations."""
    bad_config = make_direct_config()
    bad_config["process_noise_pos"] = -0.01

    with pytest.raises(UncertaintyConfigError, match="process_noise_pos"):
        UncertaintyConfig.from_config(bad_config)


def test_uncertainty_config_rejects_non_positive_dt() -> None:
    """UncertaintyConfig should reject non-positive dt."""
    bad_config = make_direct_config()
    bad_config["dt"] = 0.0

    with pytest.raises(UncertaintyConfigError, match="dt"):
        UncertaintyConfig.from_config(bad_config)


def test_grouped_state_variances() -> None:
    """grouped_state_variances should expand grouped std values to State9 variances."""
    variances = grouped_state_variances(
        position_std=0.1,
        velocity_std=0.2,
        attitude_std=0.3,
    )

    expected = np.array(
        [
            0.01,
            0.01,
            0.01,
            0.04,
            0.04,
            0.04,
            0.09,
            0.09,
            0.09,
        ],
        dtype=np.float64,
    )

    assert variances.shape == (9,)
    assert np.allclose(variances, expected)


def test_grouped_state_variances_rejects_negative_std() -> None:
    """grouped_state_variances should reject negative standard deviation."""
    with pytest.raises(UncertaintyConfigError, match="position_std"):
        grouped_state_variances(
            position_std=-0.1,
            velocity_std=0.2,
            attitude_std=0.3,
        )


def test_propagator_from_config() -> None:
    """UncertaintyPropagator.from_config should create Gamma_0 and Q."""
    propagator = UncertaintyPropagator.from_config(make_full_config())

    assert propagator.dt == pytest.approx(0.1)
    assert propagator.Gamma_0.shape == (9, 9)
    assert propagator.Q.shape == (9, 9)
    assert np.allclose(propagator.Gamma_0, propagator.initial_covariance())
    assert np.allclose(propagator.Q, propagator.process_covariance())


def test_initial_covariance() -> None:
    """initial_covariance should use init_* noise variances."""
    propagator = UncertaintyPropagator.from_config(make_direct_config())

    Gamma_0 = propagator.initial_covariance()

    expected_diag = grouped_state_variances(
        position_std=0.05,
        velocity_std=0.1,
        attitude_std=0.03,
    )
    assert Gamma_0.shape == (9, 9)
    assert np.allclose(np.diag(Gamma_0), expected_diag)
    assert np.allclose(Gamma_0, Gamma_0.T)


def test_process_covariance() -> None:
    """process_covariance should use process noise and dt-scaled drift variance."""
    propagator = UncertaintyPropagator.from_config(make_direct_config())

    Q = propagator.process_covariance()

    process_variances = grouped_state_variances(
        position_std=0.01,
        velocity_std=0.1,
        attitude_std=0.02,
    )
    drift_variances = grouped_state_variances(
        position_std=0.005,
        velocity_std=0.02,
        attitude_std=0.005,
    )
    expected_diag = process_variances + 0.1 * drift_variances + 1e-12

    assert Q.shape == (9, 9)
    assert np.allclose(np.diag(Q), expected_diag)
    assert np.allclose(Q, Q.T)


def test_position_velocity_attitude_covariance_extractors() -> None:
    """Covariance extractors should return 3x3 covariance blocks."""
    propagator = UncertaintyPropagator.from_config(make_direct_config())
    Gamma = np.diag(np.arange(1, 10, dtype=np.float64))

    Sigma_pos = propagator.position_covariance(Gamma)
    Sigma_vel = propagator.velocity_covariance(Gamma)
    Sigma_att = propagator.attitude_covariance(Gamma)

    assert Sigma_pos.shape == (3, 3)
    assert Sigma_vel.shape == (3, 3)
    assert Sigma_att.shape == (3, 3)
    assert np.allclose(np.diag(Sigma_pos), [1.0, 2.0, 3.0])
    assert np.allclose(np.diag(Sigma_vel), [4.0, 5.0, 6.0])
    assert np.allclose(np.diag(Sigma_att), [7.0, 8.0, 9.0])


def test_as_state_trajectory_state_major_accepts_state_major() -> None:
    """State trajectory helper should accept shape (9, T)."""
    x_traj = make_state_trajectory_state_major(horizon=2)

    result = as_state_trajectory_state_major(x_traj)

    assert result.shape == (9, 3)
    assert np.allclose(result, x_traj)


def test_as_state_trajectory_state_major_accepts_time_major() -> None:
    """State trajectory helper should convert shape (T, 9) to (9, T)."""
    x_traj = make_state_trajectory_state_major(horizon=2)

    result = as_state_trajectory_state_major(x_traj.T)

    assert result.shape == (9, 3)
    assert np.allclose(result, x_traj)


def test_as_state_trajectory_state_major_rejects_bad_shape() -> None:
    """State trajectory helper should reject invalid shapes."""
    with pytest.raises(UncertaintyPropagationError, match="x_traj"):
        as_state_trajectory_state_major(np.zeros((8, 3), dtype=np.float64))


def test_as_control_trajectory_control_major_accepts_control_major() -> None:
    """Control trajectory helper should accept shape (4, T)."""
    u_traj = make_control_trajectory_control_major(horizon=2)

    result = as_control_trajectory_control_major(u_traj)

    assert result.shape == (4, 2)
    assert np.allclose(result, u_traj)


def test_as_control_trajectory_control_major_accepts_time_major() -> None:
    """Control trajectory helper should convert shape (T, 4) to (4, T)."""
    u_traj = make_control_trajectory_control_major(horizon=2)

    result = as_control_trajectory_control_major(u_traj.T)

    assert result.shape == (4, 2)
    assert np.allclose(result, u_traj)


def test_as_control_trajectory_control_major_rejects_bad_shape() -> None:
    """Control trajectory helper should reject invalid shapes."""
    with pytest.raises(UncertaintyPropagationError, match="u_traj"):
        as_control_trajectory_control_major(np.zeros((3, 2), dtype=np.float64))


def test_as_dynamics_jacobian_valid() -> None:
    """as_dynamics_jacobian should accept shape (9, 9)."""
    A = np.eye(9, dtype=np.float64)

    result = as_dynamics_jacobian(A)

    assert result.shape == (9, 9)
    assert np.allclose(result, A)


def test_as_dynamics_jacobian_rejects_bad_shape() -> None:
    """as_dynamics_jacobian should reject non-(9,9) matrix."""
    with pytest.raises(UncertaintyPropagationError, match="shape"):
        as_dynamics_jacobian(np.eye(8, dtype=np.float64))


def test_symmetrize() -> None:
    """symmetrize should compute 0.5 * (M + M.T)."""
    matrix = np.array(
        [
            [1.0, 2.0],
            [0.0, 3.0],
        ],
        dtype=np.float64,
    )

    result = symmetrize(matrix)

    assert np.allclose(result, [[1.0, 1.0], [1.0, 3.0]])


def test_symmetrize_rejects_non_square() -> None:
    """symmetrize should reject non-square matrix."""
    with pytest.raises(UncertaintyPropagationError, match="square"):
        symmetrize(np.zeros((2, 3), dtype=np.float64))


def test_project_to_psd_gamma() -> None:
    """project_to_psd_gamma should preserve a valid PSD Gamma9x9."""
    Gamma = np.eye(9, dtype=np.float64) * 0.1
    Gamma[0, 1] = 0.01
    Gamma[1, 0] = 0.01

    projected = project_to_psd_gamma(Gamma)

    eigvals = np.linalg.eigvalsh(projected)
    assert projected.shape == (9, 9)
    assert np.allclose(projected, projected.T)
    assert np.all(eigvals >= -1e-12)


def test_propagate_identity_dynamics() -> None:
    """For A=I, Gamma should grow by Q every step."""
    propagator = UncertaintyPropagator.from_config(make_direct_config())
    dynamics = IdentityDynamics()
    horizon = 3

    Gamma_0 = propagator.Gamma_0.copy()
    x_traj = make_state_trajectory_state_major(horizon=horizon)
    u_traj = make_control_trajectory_control_major(horizon=horizon)

    Gamma_list = propagator.propagate(
        Gamma_0,
        x_traj,
        u_traj,
        dynamics,
    )

    assert dynamics.calls == horizon
    assert len(Gamma_list) == horizon + 1

    for k, Gamma_k in enumerate(Gamma_list):
        expected = Gamma_0 + k * propagator.Q
        assert Gamma_k.shape == (9, 9)
        assert np.allclose(Gamma_k, expected)
        assert np.allclose(Gamma_k, Gamma_k.T)


def test_propagate_accepts_time_major_layouts() -> None:
    """propagate should accept time-major x/u trajectories."""
    propagator = UncertaintyPropagator.from_config(make_direct_config())
    dynamics = IdentityDynamics()
    horizon = 2

    x_traj = make_state_trajectory_state_major(horizon=horizon).T
    u_traj = make_control_trajectory_control_major(horizon=horizon).T

    Gamma_list = propagator.propagate(
        propagator.Gamma_0,
        x_traj,
        u_traj,
        dynamics,
    )

    assert dynamics.calls == horizon
    assert len(Gamma_list) == horizon + 1


def test_propagate_position_velocity_dynamics_increases_position_covariance() -> None:
    """A with position-velocity coupling should increase position covariance."""
    propagator = UncertaintyPropagator.from_config(make_direct_config())
    dynamics = PositionVelocityDynamics()
    horizon = 1

    x_traj = make_state_trajectory_state_major(horizon=horizon)
    u_traj = make_control_trajectory_control_major(horizon=horizon)

    Gamma_list = propagator.propagate(
        propagator.Gamma_0,
        x_traj,
        u_traj,
        dynamics,
    )

    Sigma_0 = propagator.position_covariance(Gamma_list[0])
    Sigma_1 = propagator.position_covariance(Gamma_list[1])

    assert np.all(np.diag(Sigma_1) > np.diag(Sigma_0))


def test_propagate_uses_default_gamma_when_none() -> None:
    """propagate should use stored Gamma_0 when Gamma_0 argument is None."""
    propagator = UncertaintyPropagator.from_config(make_direct_config())
    dynamics = IdentityDynamics()
    horizon = 2

    Gamma_list = propagator.propagate(
        None,
        make_state_trajectory_state_major(horizon=horizon),
        make_control_trajectory_control_major(horizon=horizon),
        dynamics,
    )

    assert np.allclose(Gamma_list[0], propagator.Gamma_0)


def test_propagate_with_dt_override() -> None:
    """propagate should pass dt override to dynamics.linearize."""
    seen_dt = []

    class DtRecordingDynamics:
        def linearize(
            self,
            x_bar: np.ndarray,
            u_bar: np.ndarray,
            dt: float,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            seen_dt.append(dt)
            return (
                np.eye(9, dtype=np.float64),
                np.zeros((9, 4), dtype=np.float64),
                np.zeros(9, dtype=np.float64),
            )

    propagator = UncertaintyPropagator.from_config(make_direct_config())

    propagator.propagate(
        propagator.Gamma_0,
        make_state_trajectory_state_major(horizon=2),
        make_control_trajectory_control_major(horizon=2),
        DtRecordingDynamics(),
        dt=0.2,
    )

    assert seen_dt == [0.2, 0.2]


def test_propagate_rejects_mismatched_horizon() -> None:
    """propagate should require len(x_traj) = len(u_traj) + 1."""
    propagator = UncertaintyPropagator.from_config(make_direct_config())

    with pytest.raises(UncertaintyPropagationError, match="one more state"):
        propagator.propagate(
            propagator.Gamma_0,
            make_state_trajectory_state_major(horizon=2),
            make_control_trajectory_control_major(horizon=3),
            IdentityDynamics(),
        )


def test_propagate_rejects_dynamics_without_linearize() -> None:
    """propagate should reject dynamics object without linearize method."""
    propagator = UncertaintyPropagator.from_config(make_direct_config())

    with pytest.raises(UncertaintyPropagationError, match="linearize"):
        propagator.propagate(
            propagator.Gamma_0,
            make_state_trajectory_state_major(horizon=1),
            make_control_trajectory_control_major(horizon=1),
            BadDynamicsNoLinearize(),  # type: ignore[arg-type]
        )


def test_propagate_rejects_bad_dynamics_jacobian_shape() -> None:
    """propagate should reject invalid A matrix returned by dynamics."""
    propagator = UncertaintyPropagator.from_config(make_direct_config())

    with pytest.raises(UncertaintyPropagationError, match="shape"):
        propagator.propagate(
            propagator.Gamma_0,
            make_state_trajectory_state_major(horizon=1),
            make_control_trajectory_control_major(horizon=1),
            BadDynamicsWrongShape(),
        )


def test_propagate_wraps_dynamics_exception() -> None:
    """propagate should wrap exceptions from dynamics.linearize."""
    propagator = UncertaintyPropagator.from_config(make_direct_config())

    with pytest.raises(UncertaintyPropagationError, match="linearize failed"):
        propagator.propagate(
            propagator.Gamma_0,
            make_state_trajectory_state_major(horizon=1),
            make_control_trajectory_control_major(horizon=1),
            BadDynamicsRaises(),
        )


def test_reset_initial_covariance_default() -> None:
    """reset_initial_covariance(None) should restore default Gamma_0."""
    propagator = UncertaintyPropagator.from_config(make_direct_config())
    custom_gamma = np.eye(9, dtype=np.float64) * 0.5

    propagator.reset_initial_covariance(custom_gamma)
    assert np.allclose(propagator.Gamma_0, custom_gamma)

    propagator.reset_initial_covariance()
    assert np.allclose(propagator.Gamma_0, propagator.initial_covariance())


def test_reset_initial_covariance_custom() -> None:
    """reset_initial_covariance should accept custom valid Gamma9x9."""
    propagator = UncertaintyPropagator.from_config(make_direct_config())
    custom_gamma = np.eye(9, dtype=np.float64) * 0.2

    propagator.reset_initial_covariance(custom_gamma)

    assert np.allclose(propagator.Gamma_0, custom_gamma)
