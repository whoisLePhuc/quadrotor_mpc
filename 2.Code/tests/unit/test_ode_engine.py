"""Unit tests for ODEPhysicsEngine.

Target module:
    simulation.engines.ode_engine

These tests verify that ODEPhysicsEngine is a thin, safe wrapper around an
object implementing discrete(state, command, dt).
"""

from __future__ import annotations

import numpy as np
import pytest

from ccmpc.types import DataContractError

from simulation.engines.base import (
    DEFAULT_ODE_METADATA,
    EngineCommandType,
    EngineConfigurationError,
    EngineStateError,
    EngineStepError,
    EngineType,
)
from simulation.engines.ode_engine import ODEPhysicsEngine


def make_state() -> np.ndarray:
    """Create a valid canonical State9."""
    return np.array(
        [
            0.0,  # x
            0.0,  # y
            1.0,  # z
            0.1,  # vx
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
    """Deterministic fake dynamics used to isolate ODEPhysicsEngine behavior."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_state: np.ndarray | None = None
        self.last_command: np.ndarray | None = None
        self.last_dt: float | None = None

    def discrete(self, state: np.ndarray, command: np.ndarray, dt: float) -> np.ndarray:
        """Return a predictable next State9."""
        self.calls += 1
        self.last_state = state.copy()
        self.last_command = command.copy()
        self.last_dt = dt

        next_state = state.copy()
        next_state[0:3] = state[0:3] + dt * state[3:6]
        next_state[5] = state[5] + dt * command[2]
        next_state[6] = command[0]
        next_state[7] = command[1]
        next_state[8] = state[8] + dt * command[3]
        return next_state


class FailingDynamics:
    """Fake dynamics that always fails inside discrete()."""

    def discrete(self, state: np.ndarray, command: np.ndarray, dt: float) -> np.ndarray:
        raise RuntimeError("fake dynamics failure")


class InvalidStateDynamics:
    """Fake dynamics that returns an invalid state shape."""

    def discrete(self, state: np.ndarray, command: np.ndarray, dt: float) -> np.ndarray:
        return np.zeros(8, dtype=np.float64)


class NoDiscreteMethod:
    """Invalid dynamics object for constructor validation."""


def test_ode_engine_reset() -> None:
    """reset() initializes state, time, and step index."""
    dynamics = FakeDynamics()
    engine = ODEPhysicsEngine(dynamics=dynamics)

    initial_state = make_state()
    engine.reset(initial_state, time=1.5)

    assert engine.is_initialized is True
    assert engine.get_time() == pytest.approx(1.5)
    assert engine.get_step_index() == 0
    assert np.allclose(engine.get_state(), initial_state)
    assert engine.get_metadata().engine_type is EngineType.ODE
    assert engine.get_metadata().command_type is EngineCommandType.CONTROL_COMMAND4


def test_ode_engine_step_matches_dynamics_discrete() -> None:
    """step() should call dynamics.discrete() and expose its next State9."""
    dynamics = FakeDynamics()
    initial_state = make_state()
    command = make_command()
    dt = 0.02

    expected_next_state = dynamics.discrete(initial_state, command, dt)

    # Use a fresh fake so call counters remain easy to assert.
    dynamics = FakeDynamics()
    engine = ODEPhysicsEngine(dynamics=dynamics, initial_state=initial_state)

    result = engine.step(command, dt)

    assert dynamics.calls == 1
    assert np.allclose(result.state, expected_next_state)
    assert np.allclose(engine.get_state(), expected_next_state)
    assert result.command_type is EngineCommandType.CONTROL_COMMAND4
    assert result.applied_command is not None
    assert np.allclose(result.applied_command, command)
    assert result.diagnostics["engine"] == "ode"
    assert result.diagnostics["dynamics_class"] == "FakeDynamics"


def test_ode_engine_step_updates_time_and_step_index() -> None:
    """step() increments simulation time and completed step count."""
    dynamics = FakeDynamics()
    engine = ODEPhysicsEngine(dynamics=dynamics, initial_state=make_state())

    engine.step(make_command(), 0.02)
    engine.step(make_command(), 0.03)

    assert engine.get_time() == pytest.approx(0.05)
    assert engine.get_step_index() == 2


def test_ode_engine_get_state_returns_copy() -> None:
    """get_state() should not expose mutable internal state by default."""
    dynamics = FakeDynamics()
    initial_state = make_state()
    engine = ODEPhysicsEngine(dynamics=dynamics, initial_state=initial_state)

    state_copy = engine.get_state()
    state_copy[0] = 999.0

    assert engine.get_state()[0] == pytest.approx(initial_state[0])


def test_ode_engine_requires_reset_before_step() -> None:
    """step() rejects use before reset() when no initial_state was provided."""
    dynamics = FakeDynamics()
    engine = ODEPhysicsEngine(dynamics=dynamics)

    with pytest.raises(EngineStateError, match="not been reset"):
        engine.step(make_command(), 0.02)


def test_ode_engine_requires_reset_before_get_state() -> None:
    """get_state() rejects use before reset()."""
    dynamics = FakeDynamics()
    engine = ODEPhysicsEngine(dynamics=dynamics)

    with pytest.raises(EngineStateError, match="not been reset"):
        engine.get_state()


def test_ode_engine_rejects_invalid_initial_state() -> None:
    """reset() rejects invalid State9."""
    dynamics = FakeDynamics()
    engine = ODEPhysicsEngine(dynamics=dynamics)

    with pytest.raises(DataContractError, match="State9"):
        engine.reset(np.zeros(8, dtype=np.float64))


def test_ode_engine_rejects_invalid_command() -> None:
    """step() rejects invalid ControlCommand4 before calling dynamics."""
    dynamics = FakeDynamics()
    engine = ODEPhysicsEngine(dynamics=dynamics, initial_state=make_state())

    with pytest.raises(DataContractError, match="ControlCommand4"):
        engine.step(np.zeros(3, dtype=np.float64), 0.02)

    assert dynamics.calls == 0


def test_ode_engine_rejects_invalid_dt() -> None:
    """step() rejects invalid dt before calling dynamics."""
    dynamics = FakeDynamics()
    engine = ODEPhysicsEngine(dynamics=dynamics, initial_state=make_state())

    invalid_dt_values = [0.0, -0.01, np.nan, np.inf, True]

    for invalid_dt in invalid_dt_values:
        with pytest.raises(EngineStepError, match="dt"):
            engine.step(make_command(), invalid_dt)

    assert dynamics.calls == 0


def test_ode_engine_close_rejects_reuse() -> None:
    """close() marks the engine unusable for future runtime calls."""
    dynamics = FakeDynamics()
    engine = ODEPhysicsEngine(dynamics=dynamics, initial_state=make_state())

    engine.close()

    assert engine.is_closed is True

    with pytest.raises(EngineStateError, match="closed"):
        engine.step(make_command(), 0.02)

    with pytest.raises(EngineStateError, match="closed"):
        engine.get_state()

    with pytest.raises(EngineStateError, match="closed"):
        engine.get_time()

    with pytest.raises(EngineStateError, match="closed"):
        engine.get_step_index()


def test_ode_engine_wraps_dynamics_error() -> None:
    """Exceptions raised by dynamics.discrete() should be wrapped as EngineStepError."""
    engine = ODEPhysicsEngine(
        dynamics=FailingDynamics(),
        initial_state=make_state(),
    )

    with pytest.raises(EngineStepError, match="discrete step failed"):
        engine.step(make_command(), 0.02)


def test_ode_engine_rejects_invalid_next_state_from_dynamics() -> None:
    """Engine rejects dynamics output that is not canonical State9."""
    engine = ODEPhysicsEngine(
        dynamics=InvalidStateDynamics(),
        initial_state=make_state(),
    )

    with pytest.raises(DataContractError, match="State9"):
        engine.step(make_command(), 0.02)


def test_ode_engine_rejects_invalid_dynamics_object() -> None:
    """Constructor rejects objects without discrete(state, command, dt)."""
    with pytest.raises(EngineConfigurationError, match="discrete"):
        ODEPhysicsEngine(dynamics=NoDiscreteMethod())  # type: ignore[arg-type]


def test_ode_engine_from_dynamics_constructor() -> None:
    """from_dynamics() convenience constructor should initialize a valid ODE engine."""
    dynamics = FakeDynamics()
    initial_state = make_state()

    engine = ODEPhysicsEngine.from_dynamics(
        dynamics,
        initial_state=initial_state,
        name="FakeODE",
        native_dt=0.02,
    )

    assert engine.get_metadata().name == "FakeODE"
    assert engine.get_metadata().native_dt == pytest.approx(0.02)
    assert engine.get_metadata().engine_type is EngineType.ODE
    assert np.allclose(engine.get_state(), initial_state)


def test_ode_engine_metadata_is_default_ode_metadata() -> None:
    """Default ODEPhysicsEngine metadata should be DEFAULT_ODE_METADATA."""
    engine = ODEPhysicsEngine(
        dynamics=FakeDynamics(),
        initial_state=make_state(),
    )

    assert engine.get_metadata() == DEFAULT_ODE_METADATA
