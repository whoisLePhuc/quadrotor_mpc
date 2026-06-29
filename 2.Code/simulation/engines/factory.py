"""Physics engine factory.

This module centralizes engine construction so runtime code does not need to
know how each concrete engine is initialized.

Current scope:
    - ODE engine: supported
    - MuJoCo engine: intentionally not implemented yet

The MuJoCo path is kept explicit because it needs additional pieces that should
be designed separately:
    - State9 <-> MuJoCo qpos/qvel adapter
    - quaternion <-> Euler conversion policy
    - ControlCommand4 -> ActuatorCommand4 mixer
    - model XML path and MuJoCo timestep handling
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ccmpc.dynamics import QuadrotorDynamics
from ccmpc.types import FloatArray, as_state9

from simulation.engines.base import (
    DEFAULT_ODE_METADATA,
    EngineConfigurationError,
    EngineMetadata,
    EngineType,
    PhysicsEngine,
)
from simulation.engines.ode_engine import (
    DiscreteDynamicsProtocol,
    ODEPhysicsEngine,
)


PathLike = str | Path


@dataclass(frozen=True)
class ODEEngineFactoryConfig:
    """Configuration used by the factory to build an ODE engine.

    Attributes
    ----------
    dynamics_config:
        Optional path or dictionary accepted by ``QuadrotorDynamics.from_config``.
        If omitted, ``QuadrotorDynamics()`` is used.
    metadata_name:
        Optional name override for engine metadata.
    native_dt:
        Optional native timestep metadata.  ODE engine still accepts runtime dt
        at every step.
    """

    dynamics_config: PathLike | dict[str, Any] | None = None
    metadata_name: str | None = None
    native_dt: float | None = None


def create_physics_engine(
    engine_type: EngineType | str,
    *,
    initial_state: FloatArray,
    dynamics: DiscreteDynamicsProtocol | None = None,
    ode_config: ODEEngineFactoryConfig | None = None,
    metadata: EngineMetadata | None = None,
) -> PhysicsEngine:
    """Create a physics engine from canonical factory arguments.

    Parameters
    ----------
    engine_type:
        Engine type enum or string.  Currently supports ``"ode"``.
    initial_state:
        Canonical State9 used to reset the created engine.
    dynamics:
        Optional dynamics object implementing ``discrete(state, command, dt)``.
        For ODE engine, if this is omitted the factory creates
        ``QuadrotorDynamics``.
    ode_config:
        Optional ODE-specific factory config.
    metadata:
        Optional metadata override.  If omitted, default metadata is used.

    Returns
    -------
    PhysicsEngine
        Concrete engine instance behind the common interface.

    Raises
    ------
    EngineConfigurationError
        If the engine type is unsupported or configuration is invalid.
    """
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
        raise EngineConfigurationError(
            "MuJoCo engine factory is not implemented yet. "
            "Implement simulation.engines.mujoco_engine after defining the "
            "State9/qpos-qvel adapter and ControlCommand4/ActuatorCommand4 policy."
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
    """Create an ``ODEPhysicsEngine``.

    If ``dynamics`` is provided, it is used directly.  Otherwise the factory
    constructs ``QuadrotorDynamics`` using ``config.dynamics_config`` when
    available, or default constructor values when not.
    """
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


def create_quadrotor_dynamics(
    dynamics_config: PathLike | dict[str, Any] | None = None,
) -> QuadrotorDynamics:
    """Create ``QuadrotorDynamics`` for the ODE engine.

    Parameters
    ----------
    dynamics_config:
        None, YAML path, or parsed dictionary.
        - None: use ``QuadrotorDynamics()`` defaults.
        - path/dict: use ``QuadrotorDynamics.from_config(...)``.
    """
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
    "ODEEngineFactoryConfig",
    "create_ode_engine",
    "create_physics_engine",
    "create_quadrotor_dynamics",
    "parse_engine_type",
]
