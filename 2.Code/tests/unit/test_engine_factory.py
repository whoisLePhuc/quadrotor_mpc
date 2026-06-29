"""Unit tests for physics engine factory.

Target module:
    simulation.engines.factory

These tests verify that the factory:
- parses engine type consistently
- creates ODEPhysicsEngine from default dynamics
- creates ODEPhysicsEngine from fake dynamics
- applies ODE factory metadata overrides
- creates QuadrotorDynamics from default/dict config
- rejects MuJoCo/CUSTOM engines until implemented
"""

from __future__ import annotations

import numpy as np
import pytest

from ccmpc.dynamics import QuadrotorDynamics

from simulation.engines.base import (
    EngineConfigurationError,
    EngineType,
)
from simulation.engines.factory import (
    ODEEngineFactoryConfig,
    create_ode_engine,
    create_physics_engine,
    create_quadrotor_dynamics,
    parse_engine_type,
)
from simulation.engines.ode_engine import ODEPhysicsEngine


def make_state() -> np.ndarray:
    """Create a valid canonical State9."""
    return np.array(
        [
            0.0,  # x
            0.0,  # y
            1.0,  # z
            0.0,  # vx
            0.0,  # vy
            0.0,  # vz
            0.0,  # roll
            0.0,  # pitch
            0.0,  # yaw
        ],
        dtype=np.float64,
    )


def make_command() -> np.ndarray:
    """Create a valid canonical ControlCommand4."""
    return np.array(
        [
            0.05,  # phi_c
            -0.04, # theta_c
            0.2,   # vz_c
            0.1,   # psi_dot_c
        ],
        dtype=np.float64,
    )


class FakeDynamics:
    """Small fake dynamics object implementing discrete(state, command, dt)."""

    def __init__(self) -> None:
        self.calls = 0

    def discrete(self, state: np.ndarray, command: np.ndarray, dt: float) -> np.ndarray:
        self.calls += 1
        next_state = state.copy()
        next_state[0] += dt
        next_state[5] += command[2] * dt
        return next_state


def test_parse_engine_type_from_enum() -> None:
    """parse_engine_type accepts EngineType enum values."""
    assert parse_engine_type(EngineType.ODE) is EngineType.ODE
    assert parse_engine_type(EngineType.MUJOCO) is EngineType.MUJOCO
    assert parse_engine_type(EngineType.CUSTOM) is EngineType.CUSTOM


def test_parse_engine_type_from_string() -> None:
    """parse_engine_type accepts case-insensitive strings with whitespace."""
    assert parse_engine_type("ode") is EngineType.ODE
    assert parse_engine_type(" ODE ") is EngineType.ODE
    assert parse_engine_type("mujoco") is EngineType.MUJOCO
    assert parse_engine_type("CUSTOM") is EngineType.CUSTOM


def test_parse_engine_type_rejects_invalid_string() -> None:
    """parse_engine_type rejects unknown engine strings."""
    with pytest.raises(EngineConfigurationError, match="Unsupported engine type"):
        parse_engine_type("pybullet")


def test_parse_engine_type_rejects_invalid_type() -> None:
    """parse_engine_type rejects non-string non-EngineType values."""
    with pytest.raises(EngineConfigurationError, match="engine_type"):
        parse_engine_type(123)  # type: ignore[arg-type]


def test_create_quadrotor_dynamics_default() -> None:
    """create_quadrotor_dynamics(None) returns default QuadrotorDynamics."""
    dynamics = create_quadrotor_dynamics()

    assert isinstance(dynamics, QuadrotorDynamics)
    assert dynamics.g == pytest.approx(9.81)
    assert dynamics.k_vz == pytest.approx(3.0)


def test_create_quadrotor_dynamics_from_dict() -> None:
    """create_quadrotor_dynamics parses a dict accepted by QuadrotorDynamics.from_config."""
    config = {
        "model": {
            "quadrotor": {
                "g": 9.81,
                "kD": 0.25,
                "k_phi": 1.2,
                "k_theta": 1.3,
                "k_vz": 2.5,
                "tau_phi": 0.21,
                "tau_theta": 0.22,
                "tau_vz": 0.45,
            }
        }
    }

    dynamics = create_quadrotor_dynamics(config)

    assert isinstance(dynamics, QuadrotorDynamics)
    assert dynamics.kD == pytest.approx(0.25)
    assert dynamics.k_phi == pytest.approx(1.2)
    assert dynamics.k_theta == pytest.approx(1.3)
    assert dynamics.k_vz == pytest.approx(2.5)
    assert dynamics.tau_phi == pytest.approx(0.21)
    assert dynamics.tau_theta == pytest.approx(0.22)
    assert dynamics.tau_vz == pytest.approx(0.45)


def test_create_quadrotor_dynamics_rejects_invalid_config_type() -> None:
    """create_quadrotor_dynamics rejects unsupported config object types."""
    with pytest.raises(EngineConfigurationError, match="dynamics_config"):
        create_quadrotor_dynamics(3.14)  # type: ignore[arg-type]


def test_create_ode_engine_default() -> None:
    """create_ode_engine creates an initialized ODEPhysicsEngine."""
    initial_state = make_state()

    engine = create_ode_engine(initial_state=initial_state)

    assert isinstance(engine, ODEPhysicsEngine)
    assert engine.get_metadata().engine_type is EngineType.ODE
    assert np.allclose(engine.get_state(), initial_state)
    assert engine.get_time() == pytest.approx(0.0)
    assert engine.get_step_index() == 0


def test_create_ode_engine_with_fake_dynamics() -> None:
    """create_ode_engine can use an injected dynamics object."""
    fake_dynamics = FakeDynamics()
    initial_state = make_state()
    command = make_command()

    engine = create_ode_engine(
        initial_state=initial_state,
        dynamics=fake_dynamics,
    )

    result = engine.step(command, 0.02)

    assert fake_dynamics.calls == 1
    assert result.state[0] == pytest.approx(0.02)
    assert result.state[5] == pytest.approx(command[2] * 0.02)


def test_create_ode_engine_with_metadata_overrides() -> None:
    """ODEEngineFactoryConfig can override metadata name and native_dt."""
    initial_state = make_state()

    engine = create_ode_engine(
        initial_state=initial_state,
        dynamics=FakeDynamics(),
        config=ODEEngineFactoryConfig(
            metadata_name="FactoryODE",
            native_dt=0.02,
        ),
    )

    assert engine.get_metadata().name == "FactoryODE"
    assert engine.get_metadata().native_dt == pytest.approx(0.02)
    assert engine.get_metadata().engine_type is EngineType.ODE


def test_create_physics_engine_ode_from_enum() -> None:
    """create_physics_engine creates ODE engine from EngineType.ODE."""
    initial_state = make_state()

    engine = create_physics_engine(
        EngineType.ODE,
        initial_state=initial_state,
        dynamics=FakeDynamics(),
    )

    assert isinstance(engine, ODEPhysicsEngine)
    assert engine.get_metadata().engine_type is EngineType.ODE
    assert np.allclose(engine.get_state(), initial_state)


def test_create_physics_engine_ode_from_string() -> None:
    """create_physics_engine creates ODE engine from string 'ode'."""
    initial_state = make_state()

    engine = create_physics_engine(
        "ode",
        initial_state=initial_state,
        dynamics=FakeDynamics(),
    )

    assert isinstance(engine, ODEPhysicsEngine)
    assert engine.get_metadata().engine_type is EngineType.ODE


def test_create_physics_engine_rejects_invalid_initial_state() -> None:
    """create_physics_engine validates initial_state before constructing engine."""
    with pytest.raises(ValueError, match="State9"):
        create_physics_engine(
            "ode",
            initial_state=np.zeros(8, dtype=np.float64),
            dynamics=FakeDynamics(),
        )


def test_create_physics_engine_rejects_mujoco_for_now() -> None:
    """MuJoCo factory path should fail explicitly until MuJoCo engine is implemented."""
    with pytest.raises(EngineConfigurationError, match="MuJoCo engine factory"):
        create_physics_engine(
            EngineType.MUJOCO,
            initial_state=make_state(),
        )


def test_create_physics_engine_rejects_custom_for_now() -> None:
    """CUSTOM factory path should fail explicitly until custom engine support is added."""
    with pytest.raises(EngineConfigurationError, match="CUSTOM engine factory"):
        create_physics_engine(
            EngineType.CUSTOM,
            initial_state=make_state(),
        )


def test_create_physics_engine_rejects_unknown_string() -> None:
    """Unknown engine string should be rejected."""
    with pytest.raises(EngineConfigurationError, match="Unsupported engine type"):
        create_physics_engine(
            "unknown_engine",
            initial_state=make_state(),
        )
