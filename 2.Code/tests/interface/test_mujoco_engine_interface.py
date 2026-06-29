"""Interface tests for MuJoCoPhysicsEngine.

These tests are skipped when optional dependency ``mujoco`` is not installed.
The adapter unit tests still run without MuJoCo.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from ccmpc.types import DataContractError
from simulation.engines.base import EngineCommandType, EngineConfigurationError, EngineType
from simulation.engines.factory import (
    MuJoCoEngineFactoryConfig,
    create_physics_engine,
)
from simulation.engines.mujoco_engine import MuJoCoEngineConfig, MuJoCoPhysicsEngine


_MINIMAL_QUAD_XML = """
<mujoco model="phase4_quad_test">
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body name="quad" pos="0 0 0">
      <freejoint name="root"/>
      <geom name="body" type="box" size="0.1 0.1 0.02" mass="1"/>
      <site name="r1" pos="0.1 0.1 0"/>
      <site name="r2" pos="-0.1 0.1 0"/>
      <site name="r3" pos="-0.1 -0.1 0"/>
      <site name="r4" pos="0.1 -0.1 0"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="m1" site="r1" gear="0 0 1 0 0 0"/>
    <motor name="m2" site="r2" gear="0 0 1 0 0 0"/>
    <motor name="m3" site="r3" gear="0 0 1 0 0 0"/>
    <motor name="m4" site="r4" gear="0 0 1 0 0 0"/>
  </actuator>
</mujoco>
"""


def write_minimal_xml(tmp_path) -> str:
    xml_path = tmp_path / "phase4_quad_test.xml"
    xml_path.write_text(_MINIMAL_QUAD_XML, encoding="utf-8")
    return str(xml_path)


def make_state() -> np.ndarray:
    return np.array(
        [0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.01, -0.02, 0.03],
        dtype=np.float64,
    )


def make_actuator_command() -> np.ndarray:
    return np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)


def test_mujoco_engine_reset_returns_state9(tmp_path) -> None:
    engine = MuJoCoPhysicsEngine(
        config=MuJoCoEngineConfig(
            xml_path=write_minimal_xml(tmp_path),
            free_joint_name="root",
        ),
        initial_state=make_state(),
    )

    state = engine.get_state()

    assert state.shape == (9,)
    np.testing.assert_allclose(state, make_state(), atol=1e-12)
    assert engine.get_metadata().engine_type is EngineType.MUJOCO
    assert engine.get_metadata().command_type is EngineCommandType.ACTUATOR_COMMAND4
    assert engine.get_metadata().native_dt == pytest.approx(0.01)


def test_mujoco_engine_step_returns_step_result(tmp_path) -> None:
    engine = MuJoCoPhysicsEngine(
        config=MuJoCoEngineConfig(
            xml_path=write_minimal_xml(tmp_path),
            free_joint_name="root",
        ),
        initial_state=make_state(),
    )

    result = engine.step(make_actuator_command(), 0.02)

    assert result.state.shape == (9,)
    assert result.command_type is EngineCommandType.ACTUATOR_COMMAND4
    assert result.applied_command is not None
    np.testing.assert_allclose(result.applied_command, make_actuator_command())
    assert result.dt == pytest.approx(0.02)
    assert result.step_index == 1
    assert result.diagnostics["engine"] == "mujoco"
    assert result.diagnostics["substeps"] == 2


def test_mujoco_engine_rejects_control_command_without_mixer(tmp_path) -> None:
    engine = MuJoCoPhysicsEngine(
        config=MuJoCoEngineConfig(
            xml_path=write_minimal_xml(tmp_path),
            free_joint_name="root",
        ),
        initial_state=make_state(),
    )

    with pytest.raises(DataContractError, match="non-negative"):
        engine.step(np.array([0.1, -0.1, 0.2, 0.3], dtype=np.float64), 0.01)


def test_mujoco_engine_rejects_non_multiple_dt(tmp_path) -> None:
    engine = MuJoCoPhysicsEngine(
        config=MuJoCoEngineConfig(
            xml_path=write_minimal_xml(tmp_path),
            free_joint_name="root",
        ),
        initial_state=make_state(),
    )

    with pytest.raises(Exception, match="integer multiple"):
        engine.step(make_actuator_command(), 0.015)


def test_mujoco_engine_does_not_expose_public_model_data_attrs(tmp_path) -> None:
    engine = MuJoCoPhysicsEngine(
        config=MuJoCoEngineConfig(
            xml_path=write_minimal_xml(tmp_path),
            free_joint_name="root",
        ),
        initial_state=make_state(),
    )

    assert not hasattr(engine, "model")
    assert not hasattr(engine, "data")
    assert not hasattr(engine, "qpos")
    assert not hasattr(engine, "qvel")


def test_mujoco_factory_requires_config() -> None:
    with pytest.raises(EngineConfigurationError, match="mujoco_config"):
        create_physics_engine(EngineType.MUJOCO, initial_state=make_state())


def test_mujoco_factory_creates_engine(tmp_path) -> None:
    engine = create_physics_engine(
        EngineType.MUJOCO,
        initial_state=make_state(),
        mujoco_config=MuJoCoEngineFactoryConfig(
            xml_path=write_minimal_xml(tmp_path),
            free_joint_name="root",
        ),
    )

    assert isinstance(engine, MuJoCoPhysicsEngine)
    assert engine.get_metadata().engine_type is EngineType.MUJOCO
