"""Physics engine abstraction layer for quadrotor simulation.

Public imports are collected here so higher-level modules can depend on the
engine package boundary instead of concrete file paths.

Example:
    from simulation.engines import ODEPhysicsEngine, PhysicsEngine
"""

from __future__ import annotations

from simulation.engines.base import (
    DEFAULT_MUJOCO_METADATA,
    DEFAULT_ODE_METADATA,
    EngineCommandType,
    EngineConfigurationError,
    EngineError,
    EngineMetadata,
    EngineStateError,
    EngineStepError,
    EngineStepStatus,
    EngineType,
    PhysicsEngine,
    PhysicsEngineProtocol,
    StepResult,
    command_dim_for_type,
    make_step_result,
    validate_engine_command,
    validate_step_dt,
)
from simulation.engines.ode_engine import (
    DiscreteDynamicsProtocol,
    ODEPhysicsEngine,
)

from simulation.engines.factory import (
    ODEEngineFactoryConfig,
    create_ode_engine,
    create_physics_engine,
    create_quadrotor_dynamics,
    parse_engine_type,
)

__all__ = [
    "DEFAULT_MUJOCO_METADATA",
    "DEFAULT_ODE_METADATA",
    "DiscreteDynamicsProtocol",
    "EngineCommandType",
    "EngineConfigurationError",
    "EngineError",
    "EngineMetadata",
    "EngineStateError",
    "EngineStepError",
    "EngineStepStatus",
    "EngineType",
    "ODEPhysicsEngine",
    "PhysicsEngine",
    "PhysicsEngineProtocol",
    "StepResult",
    "command_dim_for_type",
    "make_step_result",
    "validate_engine_command",
    "validate_step_dt",
    "ODEEngineFactoryConfig",
    "create_ode_engine",
    "create_physics_engine",
    "create_quadrotor_dynamics",
    "parse_engine_type",
]
