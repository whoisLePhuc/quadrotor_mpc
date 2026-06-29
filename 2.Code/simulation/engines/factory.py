"""Physics engine factory.

This module centralizes engine construction so runtime code does not need to
know how each concrete engine is initialized.

Current Phase 4 scope:
- ODE engine: supported and consumes ControlCommand4.
- MuJoCo engine: constructed explicitly with MuJoCoEngineFactoryConfig and
  consumes ActuatorCommand4.

The factory does not run controllers, mixers, loggers, scenario runners, or
visualizers. Those belong to later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ccmpc.dynamics import QuadrotorDynamics
from ccmpc.types import FloatArray, as_state9
from simulation.engines.base import (
    DEFAULT_MUJOCO_METADATA,
    DEFAULT_ODE_METADATA,
    EngineConfigurationError,
    EngineMetadata,
    EngineType,
    PhysicsEngine,
)
from simulation.engines.mujoco_engine import MuJoCoEngineConfig, MuJoCoPhysicsEngine
from simulation.engines.ode_engine import DiscreteDynamicsProtocol, ODEPhysicsEngine

PathLike = str | Path


@dataclass(frozen=True)
class ODEEngineFactoryConfig:
    """Configuration used by the factory to build an ODE engine."""

    dynamics_config: PathLike | dict[str, Any] | None = None
    metadata_name: str | None = None
    native_dt: float | None = None


@dataclass(frozen=True)
class MuJoCoEngineFactoryConfig:
    """Configuration used by the factory to build a MuJoCo engine."""

    xml_path: PathLike
    free_joint_name: str | None = None
    actuator_start_index: int = 0


def create_physics_engine(
    engine_type: EngineType | str,
    *,
    initial_state: FloatArray,
    dynamics: DiscreteDynamicsProtocol | None = None,
    ode_config: ODEEngineFactoryConfig | None = None,
    mujoco_config: MuJoCoEngineFactoryConfig | None = None,
    metadata: EngineMetadata | None = None,
) -> PhysicsEngine:
    """Create a physics engine from canonical factory arguments."""
    parsed_engine_type = parse_engine_type(engine_type)
    state = as_state9(initial_state)

    if parsed_engine_type is EngineType.ODE:
        return create_ode_engine(
            initial_state=state,
            dynamics=dynamics,
            config=ode_config,
            metadata=metadata,
        )

    if parsed_engine_type is EngineType.MUJOCO:
        if mujoco_config is None:
            raise EngineConfigurationError(
                "mujoco_config is required for MuJoCoPhysicsEngine. "
                "Provide xml_path, free_joint_name, and actuator_start_index explicitly."
            )
        return create_mujoco_engine(
            initial_state=state,
            config=mujoco_config,
            metadata=metadata,
        )

    if parsed_engine_type is EngineType.CUSTOM:
        raise EngineConfigurationError(
            "CUSTOM engine factory is not implemented. "
            "Instantiate custom engines directly or extend create_physics_engine()."
        )

    raise EngineConfigurationError(f"Unsupported engine type: {parsed_engine_type!r}.")


def create_ode_engine(
    *,
    initial_state: FloatArray,
    dynamics: DiscreteDynamicsProtocol | None = None,
    config: ODEEngineFactoryConfig | None = None,
    metadata: EngineMetadata | None = None,
) -> ODEPhysicsEngine:
    """Create an ``ODEPhysicsEngine``."""
    state = as_state9(initial_state)
    config = ODEEngineFactoryConfig() if config is None else config

    dynamics_obj = dynamics
    if dynamics_obj is None:
        dynamics_obj = create_quadrotor_dynamics(config.dynamics_config)

    metadata_obj = metadata if metadata is not None else DEFAULT_ODE_METADATA
    if config.metadata_name is not None or config.native_dt is not None:
        metadata_obj = replace(
            metadata_obj,
            name=config.metadata_name if config.metadata_name is not None else metadata_obj.name,
            native_dt=config.native_dt if config.native_dt is not None else metadata_obj.native_dt,
        )

    return ODEPhysicsEngine(
        dynamics=dynamics_obj,
        initial_state=state,
        metadata=metadata_obj,
    )


def create_mujoco_engine(
    *,
    initial_state: FloatArray,
    config: MuJoCoEngineFactoryConfig,
    metadata: EngineMetadata | None = None,
) -> MuJoCoPhysicsEngine:
    """Create a ``MuJoCoPhysicsEngine`` from explicit MuJoCo configuration."""
    state = as_state9(initial_state)
    metadata_obj = metadata if metadata is not None else DEFAULT_MUJOCO_METADATA
    return MuJoCoPhysicsEngine(
        config=MuJoCoEngineConfig(
            xml_path=config.xml_path,
            free_joint_name=config.free_joint_name,
            actuator_start_index=config.actuator_start_index,
        ),
        initial_state=state,
        metadata=metadata_obj,
    )


def create_quadrotor_dynamics(
    dynamics_config: PathLike | dict[str, Any] | None = None,
) -> QuadrotorDynamics:
    """Create ``QuadrotorDynamics`` for the ODE engine."""
    if dynamics_config is None:
        return QuadrotorDynamics()
    if isinstance(dynamics_config, Path):
        return QuadrotorDynamics.from_config(str(dynamics_config))
    if isinstance(dynamics_config, str):
        return QuadrotorDynamics.from_config(dynamics_config)
    if isinstance(dynamics_config, dict):
        return QuadrotorDynamics.from_config(dynamics_config)
    raise EngineConfigurationError(
        "dynamics_config must be None, str, pathlib.Path, or dict."
    )


def parse_engine_type(engine_type: EngineType | str) -> EngineType:
    """Parse engine type from enum or string."""
    if isinstance(engine_type, EngineType):
        return engine_type
    if isinstance(engine_type, str):
        normalized = engine_type.strip().lower()
        try:
            return EngineType(normalized)
        except ValueError as exc:
            raise EngineConfigurationError(
                f"Unsupported engine type string: {engine_type!r}."
            ) from exc
    raise EngineConfigurationError(
        f"engine_type must be EngineType or str, got {type(engine_type).__name__}."
    )


__all__ = [
    "MuJoCoEngineFactoryConfig",
    "ODEEngineFactoryConfig",
    "create_mujoco_engine",
    "create_ode_engine",
    "create_physics_engine",
    "create_quadrotor_dynamics",
    "parse_engine_type",
]
