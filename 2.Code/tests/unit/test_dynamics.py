"""Unit tests for reduced-order quadrotor dynamics.

Target module:
    ccmpc.dynamics

These tests verify the dynamics contract after refactor:
- State9 and ControlCommand4 validation
- continuous dynamics output shape
- RK4 discrete step output shape
- Jacobian shapes
- control Jacobian analytical entries
- LTV linearization shape
- affine linearization matches nonlinear rollout at expansion point
"""

from __future__ import annotations

import numpy as np
import pytest

from ccmpc.dynamics import (
    QuadrotorDynamics,
    QuadrotorDynamicsParams,
    continuous_dynamics,
    discrete_step,
)
from ccmpc.types import DataContractError


def make_state() -> np.ndarray:
    """Create a representative valid State9."""
    return np.array(
        [
            0.5,   # x
            -0.2,  # y
            1.4,   # z
            0.1,   # vx
            -0.1,  # vy
            0.05,  # vz
            0.03,  # roll / phi
            -0.04, # pitch / theta
            0.2,   # yaw / psi
        ],
        dtype=np.float64,
    )


def make_hover_state() -> np.ndarray:
    """Create near-hover State9."""
    return np.array(
        [
            0.0,  # x
            0.0,  # y
            1.0,  # z
            0.0,  # vx
            0.0,  # vy
            0.0,  # vz
            0.0,  # roll / phi
            0.0,  # pitch / theta
            0.0,  # yaw / psi
        ],
        dtype=np.float64,
    )


def make_command() -> np.ndarray:
    """Create a representative valid ControlCommand4."""
    return np.array(
        [
            0.05,  # phi_c
            -0.04, # theta_c
            0.2,   # vz_c
            0.1,   # psi_dot_c
        ],
        dtype=np.float64,
    )


def test_continuous_dynamics_shape() -> None:
    """continuous_dynamics returns a finite State9 derivative."""
    x = make_state()
    u = make_command()

    xdot = continuous_dynamics(x, u)

    assert isinstance(xdot, np.ndarray)
    assert xdot.shape == (9,)
    assert xdot.dtype == np.float64
    assert np.all(np.isfinite(xdot))


def test_continuous_dynamics_position_derivative_matches_velocity() -> None:
    """The first three derivatives should equal [vx, vy, vz]."""
    x = make_state()
    u = make_command()

    xdot = continuous_dynamics(x, u)

    assert np.allclose(xdot[0:3], x[3:6])


def test_discrete_step_shape() -> None:
    """discrete_step returns a finite next State9."""
    x = make_state()
    u = make_command()
    dt = 0.02

    x_next = discrete_step(x, u, dt)

    assert isinstance(x_next, np.ndarray)
    assert x_next.shape == (9,)
    assert x_next.dtype == np.float64
    assert np.all(np.isfinite(x_next))


def test_discrete_step_reject_invalid_state() -> None:
    """discrete_step rejects state that is not State9."""
    invalid_x = np.zeros(8)
    u = make_command()

    with pytest.raises(DataContractError, match="State9"):
        discrete_step(invalid_x, u, 0.02)


def test_discrete_step_reject_invalid_command() -> None:
    """discrete_step rejects command that is not ControlCommand4."""
    x = make_state()
    invalid_u = np.zeros(3)

    with pytest.raises(DataContractError, match="ControlCommand4"):
        discrete_step(x, invalid_u, 0.02)


def test_discrete_step_reject_invalid_dt() -> None:
    """discrete_step rejects non-positive or non-finite dt."""
    x = make_state()
    u = make_command()

    invalid_dt_values = [0.0, -0.01, np.nan, np.inf, True]

    for invalid_dt in invalid_dt_values:
        with pytest.raises(ValueError, match="dt"):
            discrete_step(x, u, invalid_dt)


def test_quadrotor_dynamics_default_params() -> None:
    """QuadrotorDynamics default params should be internally consistent."""
    dynamics = QuadrotorDynamics()

    assert dynamics.g == pytest.approx(9.81)
    assert dynamics.kD == pytest.approx(0.5)
    assert dynamics.k_phi == pytest.approx(1.0)
    assert dynamics.k_theta == pytest.approx(1.0)
    assert dynamics.k_vz == pytest.approx(3.0)
    assert dynamics.tau_phi == pytest.approx(0.2)
    assert dynamics.tau_theta == pytest.approx(0.2)
    assert dynamics.tau_vz == pytest.approx(0.4)


def test_quadrotor_dynamics_from_params() -> None:
    """QuadrotorDynamics can be constructed from QuadrotorDynamicsParams."""
    params = QuadrotorDynamicsParams(
        g=9.81,
        kD=0.3,
        k_phi=1.2,
        k_theta=1.1,
        k_vz=2.5,
        tau_phi=0.25,
        tau_theta=0.3,
        tau_vz=0.45,
    )

    dynamics = QuadrotorDynamics.from_params(params)

    assert dynamics.kD == pytest.approx(0.3)
    assert dynamics.k_phi == pytest.approx(1.2)
    assert dynamics.k_theta == pytest.approx(1.1)
    assert dynamics.k_vz == pytest.approx(2.5)
    assert dynamics.tau_phi == pytest.approx(0.25)
    assert dynamics.tau_theta == pytest.approx(0.3)
    assert dynamics.tau_vz == pytest.approx(0.45)


def test_quadrotor_dynamics_reject_invalid_params() -> None:
    """Dynamics params reject physically invalid time constants."""
    with pytest.raises(ValueError, match="tau_phi"):
        QuadrotorDynamicsParams(tau_phi=0.0)

    with pytest.raises(ValueError, match="tau_theta"):
        QuadrotorDynamicsParams(tau_theta=-0.1)

    with pytest.raises(ValueError, match="tau_vz"):
        QuadrotorDynamicsParams(tau_vz=np.inf)


def test_continuous_method_matches_function() -> None:
    """QuadrotorDynamics.continuous should call continuous_dynamics with same params."""
    dynamics = QuadrotorDynamics()
    x = make_state()
    u = make_command()

    expected = continuous_dynamics(x, u, **dynamics._params)
    actual = dynamics.continuous(x, u)

    assert np.allclose(actual, expected)


def test_discrete_method_matches_function() -> None:
    """QuadrotorDynamics.discrete should call discrete_step with same params."""
    dynamics = QuadrotorDynamics()
    x = make_state()
    u = make_command()
    dt = 0.02

    expected = discrete_step(x, u, dt, **dynamics._params)
    actual = dynamics.discrete(x, u, dt)

    assert np.allclose(actual, expected)


def test_jacobian_state_shape() -> None:
    """jacobian_state returns A_cont with shape (9, 9)."""
    dynamics = QuadrotorDynamics()
    x = make_state()
    u = make_command()

    a_cont = dynamics.jacobian_state(x, u)

    assert isinstance(a_cont, np.ndarray)
    assert a_cont.shape == (9, 9)
    assert np.all(np.isfinite(a_cont))


def test_jacobian_control_shape() -> None:
    """jacobian_control returns B_cont with shape (9, 4)."""
    dynamics = QuadrotorDynamics()
    x = make_state()
    u = make_command()

    b_cont = dynamics.jacobian_control(x, u)

    assert isinstance(b_cont, np.ndarray)
    assert b_cont.shape == (9, 4)
    assert np.all(np.isfinite(b_cont))


def test_jacobian_control_expected_entries() -> None:
    """Analytical control Jacobian should match first-order command dynamics."""
    dynamics = QuadrotorDynamics(
        k_phi=1.2,
        k_theta=1.4,
        k_vz=3.0,
        tau_phi=0.2,
        tau_theta=0.25,
        tau_vz=0.5,
    )
    x = make_state()
    u = make_command()

    b_cont = dynamics.jacobian_control(x, u)

    assert b_cont[6, 0] == pytest.approx(dynamics.k_phi / dynamics.tau_phi)
    assert b_cont[7, 1] == pytest.approx(dynamics.k_theta / dynamics.tau_theta)
    assert b_cont[5, 2] == pytest.approx(dynamics.k_vz / dynamics.tau_vz)
    assert b_cont[8, 3] == pytest.approx(1.0)

    # Other control entries should remain zero in this reduced-order model.
    expected_nonzero = np.zeros((9, 4), dtype=bool)
    expected_nonzero[6, 0] = True
    expected_nonzero[7, 1] = True
    expected_nonzero[5, 2] = True
    expected_nonzero[8, 3] = True

    assert np.allclose(b_cont[~expected_nonzero], 0.0)


def test_linearize_shapes() -> None:
    """linearize returns A_k, B_k, C_k with canonical LTV dimensions."""
    dynamics = QuadrotorDynamics()
    x_bar = make_state()
    u_bar = make_command()
    dt = 0.02

    a_k, b_k, c_k = dynamics.linearize(x_bar, u_bar, dt)

    assert a_k.shape == (9, 9)
    assert b_k.shape == (9, 4)
    assert c_k.shape == (9,)
    assert np.all(np.isfinite(a_k))
    assert np.all(np.isfinite(b_k))
    assert np.all(np.isfinite(c_k))


def test_linearize_matches_rollout_at_expansion_point() -> None:
    """Affine linear model should exactly match nonlinear rollout at (x_bar, u_bar)."""
    dynamics = QuadrotorDynamics()
    x_bar = make_state()
    u_bar = make_command()
    dt = 0.02

    a_k, b_k, c_k = dynamics.linearize(x_bar, u_bar, dt)

    nonlinear_next = dynamics.discrete(x_bar, u_bar, dt)
    linearized_next = a_k @ x_bar + b_k @ u_bar + c_k

    assert np.allclose(linearized_next, nonlinear_next, atol=1e-10, rtol=1e-10)


def test_linearize_reject_invalid_dt() -> None:
    """linearize rejects non-positive or non-finite dt."""
    dynamics = QuadrotorDynamics()
    x_bar = make_state()
    u_bar = make_command()

    with pytest.raises(ValueError, match="dt"):
        dynamics.linearize(x_bar, u_bar, 0.0)


def test_hover_zero_command_has_no_horizontal_acceleration() -> None:
    """At hover with zero attitude and yaw, horizontal acceleration is zero."""
    x = make_hover_state()
    u = np.zeros(4)

    xdot = continuous_dynamics(x, u)

    assert xdot[3] == pytest.approx(0.0)
    assert xdot[4] == pytest.approx(0.0)


def test_positive_vertical_command_increases_vz_derivative() -> None:
    """Positive vz_c should produce positive vertical acceleration at hover."""
    x = make_hover_state()
    u = np.array([0.0, 0.0, 0.5, 0.0])

    xdot = continuous_dynamics(x, u)

    assert xdot[5] > 0.0
