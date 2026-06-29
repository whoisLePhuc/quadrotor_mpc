"""Controller factory.

This module is the only public construction point for simulation controllers.
It belongs to the simulation interface layer and must not contain the CC-MPC
mathematical formulation.

Responsibilities
----------------
- Read an already-loaded controller config object/dict.
- Resolve the requested controller type.
- Construct a concrete controller from ``ccmpc.controllers``.
- Return an object that satisfies the simulation Controller interface.

The actual CC-MPC implementation remains in ``ccmpc.controllers``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

# Prefer the refactored metadata module.  Fall back to the pre-split base module
# so this factory can be introduced before all imports are migrated.
try:  # pragma: no cover - fallback branch is for transitional compatibility.
    from simulation.controllers.metadata import (
        ControllerConfigurationError,
        ControllerDiagnostics,
        ControllerInput,
        ControllerMetadata,
        ControllerOutput,
        ControllerOutputError,
        ControllerStatus,
        ControllerType,
    )
except ModuleNotFoundError:  # pragma: no cover
    from simulation.controllers.base import (
        ControllerConfigurationError,
        ControllerDiagnostics,
        ControllerInput,
        ControllerMetadata,
        ControllerOutput,
        ControllerOutputError,
        ControllerStatus,
        ControllerType,
    )

from simulation.controllers.base import Controller, ControllerProtocol


class ControllerFactoryError(RuntimeError):
    """Base exception raised by controller factory code."""


class ControllerFactoryConfigError(ControllerFactoryError):
    """Raised when controller factory configuration is invalid."""


class ControllerFactoryRegistryError(ControllerFactoryError):
    """Raised when a custom controller registry entry is invalid."""


ControllerBuilder = Callable[[Any], ControllerProtocol]


def create_controller(
    config: str | Path | Mapping[str, Any] | Any | None = None,
    *,
    controller_type: ControllerType | str | None = None,
    formulation: Any | None = None,
    solver_adapter: Any | None = None,
    fallback_controller: Any | None = None,
    registry: Mapping[ControllerType | str, ControllerBuilder] | None = None,
) -> ControllerProtocol:
    """Create a controller from config.

    Parameters
    ----------
    config:
        Controller-only config, full project config, config object, config path,
        or None.  Full project config may contain a ``controller`` section.
    controller_type:
        Optional explicit controller type override.  If omitted, the type is
        read from config and defaults to ``ControllerType.CCMPC``.
    formulation:
        Optional CC-MPC formulation object injected into ``CCMPCController``.
        This keeps the factory independent from the mathematical formulation.
    solver_adapter:
        Optional solver adapter injected into ``CCMPCController``.
    fallback_controller:
        Optional fallback controller injected into ``CCMPCController`` or used
        directly for ``EMERGENCY_STOP``.
    registry:
        Optional extension registry for project-specific controllers.

    Returns
    -------
    ControllerProtocol
        An object exposing reset(), compute_command(), get_metadata(), close().

    Notes
    -----
    At the current refactor stage, ``ccmpc.controllers.CCMPCController`` may
    still expose a legacy convenience API.  The factory therefore wraps it in a
    small private bridge that converts ``ControllerInput`` to ``solve(...)`` and
    converts the structured solve result back to ``ControllerOutput``.
    """

    resolved_type = resolve_controller_type(
        config,
        override=controller_type,
    )

    if registry is not None:
        custom = _get_registered_builder(registry, resolved_type)
        if custom is not None:
            return _validate_controller_instance(custom(config))

    if resolved_type is ControllerType.CCMPC:
        return _create_ccmpc_controller(
            config,
            formulation=formulation,
            solver_adapter=solver_adapter,
            fallback_controller=fallback_controller,
        )

    if resolved_type is ControllerType.EMERGENCY_STOP:
        return _create_emergency_stop_controller(
            config,
            fallback_controller=fallback_controller,
        )

    raise ControllerFactoryConfigError(
        f"Controller type {resolved_type.value!r} is not implemented in the "
        "default factory. Provide a registry entry to support it."
    )


def create_controller_from_config(
    config: str | Path | Mapping[str, Any] | Any | None = None,
    **kwargs: Any,
) -> ControllerProtocol:
    """Alias for create_controller(config, **kwargs)."""

    return create_controller(config, **kwargs)


def build_controller(
    config: str | Path | Mapping[str, Any] | Any | None = None,
    **kwargs: Any,
) -> ControllerProtocol:
    """Backward-compatible alias for create_controller(config, **kwargs)."""

    return create_controller(config, **kwargs)


def resolve_controller_type(
    config: str | Path | Mapping[str, Any] | Any | None = None,
    *,
    override: ControllerType | str | None = None,
) -> ControllerType:
    """Resolve controller type from explicit override or config.

    Accepted config keys in either the root config or ``controller`` section:

    - ``type``
    - ``controller_type``
    - ``kind``
    - ``name``

    If no type is provided, CC-MPC is selected because it is the primary
    controller in this project.
    """

    if override is not None:
        return parse_controller_type(override)

    config_mapping = _to_mapping_or_none(config)
    if config_mapping is None:
        return ControllerType.CCMPC

    controller_cfg = _controller_section(config_mapping)

    for key in ("type", "controller_type", "kind", "name"):
        value = controller_cfg.get(key)
        if value is not None:
            return parse_controller_type(value)

    # Some configs use nested prediction/fallback blocks without naming the
    # controller.  The design target for this project defaults to CC-MPC.
    return ControllerType.CCMPC


def parse_controller_type(value: ControllerType | str) -> ControllerType:
    """Parse ControllerType from enum or string."""

    if isinstance(value, ControllerType):
        return value

    if not isinstance(value, str):
        raise ControllerFactoryConfigError(
            f"Controller type must be ControllerType or str, got {type(value).__name__}."
        )

    raw = value.strip()
    if not raw:
        raise ControllerFactoryConfigError("Controller type string must be non-empty.")

    if raw.startswith("ControllerType."):
        raw = raw.split(".", 1)[1]

    normalized = raw.lower().replace("-", "_").replace(" ", "_")

    aliases = {
        "ccmpc": ControllerType.CCMPC,
        "cc_mpc": ControllerType.CCMPC,
        "chance_constrained_mpc": ControllerType.CCMPC,
        "quadrotor_ccmpc": ControllerType.CCMPC,
        "quadrotor_cc_mpc": ControllerType.CCMPC,
        "quadrotor_ccmpc_controller": ControllerType.CCMPC,
        "quadrotor_cc_mpc_controller": ControllerType.CCMPC,
        "ccmpc_controller": ControllerType.CCMPC,
        "nominal_mpc": ControllerType.NOMINAL_MPC,
        "mpc": ControllerType.NOMINAL_MPC,
        "pid": ControllerType.PID,
        "lqr": ControllerType.LQR,
        "emergency_stop": ControllerType.EMERGENCY_STOP,
        "emergency": ControllerType.EMERGENCY_STOP,
        "fallback": ControllerType.EMERGENCY_STOP,
        "safe_stop": ControllerType.EMERGENCY_STOP,
        "stop": ControllerType.EMERGENCY_STOP,
    }

    custom_type = getattr(ControllerType, "CUSTOM", None)
    if custom_type is not None:
        aliases["custom"] = custom_type

    if normalized in aliases:
        return aliases[normalized]

    for item in ControllerType:
        if normalized == item.value.lower() or raw.upper() == item.name:
            return item

    raise ControllerFactoryConfigError(f"Unsupported controller type: {value!r}.")


class _CCMPCControllerBridge(Controller):
    """Private bridge from CCMPCController.solve(...) to ControllerOutput.

    This class is intentionally private to avoid adding a public
    simulation/controllers/ccmpc_adapter.py module that is not listed in the
    architecture document.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def reset(self) -> None:
        self.inner.reset()

    def compute_command(self, input_data: ControllerInput) -> ControllerOutput:
        result = self.inner.solve(
            estimated_state=input_data.estimated_state,
            goal=input_data.goal,
            covariance=input_data.covariance,
            obstacles=input_data.obstacle_predictions,
            reference_trajectory=input_data.reference_trajectory,
            time_s=input_data.time,
            metadata={
                "source": "simulation.controllers.factory",
                "input_metadata": dict(getattr(input_data, "metadata", {}) or {}),
                "previous_solution_provided": input_data.previous_solution is not None,
            },
        )

        diagnostics = _diagnostics_from_ccmpc_result(result)

        return ControllerOutput(
            command=result.command,
            predicted_trajectory=result.predicted_states,
            control_trajectory=result.predicted_controls,
            diagnostics=diagnostics,
            raw_solution=result,
            metadata=dict(result.metadata),
        )

    def get_metadata(self) -> ControllerMetadata:
        config = getattr(self.inner, "config", None)
        name = str(getattr(config, "name", "CCMPCController"))
        horizon = getattr(config, "horizon", None)
        dt = getattr(config, "dt", None)

        return ControllerMetadata(
            controller_type=ControllerType.CCMPC,
            name=name,
            horizon=horizon,
            dt=dt,
            supports_obstacles=True,
            supports_covariance=True,
            deterministic=True,
            description="Factory bridge for ccmpc.controllers.CCMPCController.",
            extra={
                "inner_class": type(self.inner).__name__,
                "bridge": type(self).__name__,
            },
        )

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()


class _EmergencyStopControllerBridge(Controller):
    """Private bridge from FallbackController.compute(...) to ControllerOutput."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def reset(self) -> None:
        # Current fallback controller is stateless.
        reset = getattr(self.inner, "reset", None)
        if callable(reset):
            reset()

    def compute_command(self, input_data: ControllerInput) -> ControllerOutput:
        result = self.inner.compute(
            input_data.estimated_state,
            input_data.goal,
            reason="emergency_stop_controller",
            metadata={
                "source": "simulation.controllers.factory",
                "input_metadata": dict(getattr(input_data, "metadata", {}) or {}),
            },
        )

        diagnostics = ControllerDiagnostics(
            status=ControllerStatus.FALLBACK,
            success=False,
            solve_time_ms=None,
            objective_value=None,
            iterations=None,
            fallback_used=True,
            fallback_reason=result.reason,
            max_constraint_violation=None,
            min_obstacle_margin=None,
            notes=("direct emergency/fallback controller output",),
            extra={
                "mode": str(result.mode.value if hasattr(result.mode, "value") else result.mode),
                "fallback_status": str(
                    result.status.value if hasattr(result.status, "value") else result.status
                ),
                "clipped": result.clipped,
                "horizontal_speed": result.horizontal_speed,
                "altitude": result.altitude,
                **dict(result.metadata),
            },
        )

        return ControllerOutput(
            command=result.command,
            predicted_trajectory=None,
            control_trajectory=None,
            diagnostics=diagnostics,
            raw_solution=result,
            metadata={"source": "EmergencyStopControllerBridge"},
        )

    def get_metadata(self) -> ControllerMetadata:
        return ControllerMetadata(
            controller_type=ControllerType.EMERGENCY_STOP,
            name="EmergencyStopController",
            horizon=1,
            dt=None,
            supports_obstacles=False,
            supports_covariance=False,
            deterministic=True,
            description="Factory bridge for ccmpc.controllers.FallbackController.",
            extra={
                "inner_class": type(self.inner).__name__,
                "bridge": type(self).__name__,
            },
        )

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()


def _create_ccmpc_controller(
    config: str | Path | Mapping[str, Any] | Any | None,
    *,
    formulation: Any | None,
    solver_adapter: Any | None,
    fallback_controller: Any | None,
) -> ControllerProtocol:
    """Create CC-MPC controller bridge."""

    from ccmpc.controllers.ccmpc_controller import CCMPCController

    if solver_adapter is None and fallback_controller is None:
        inner = CCMPCController.from_config(
            config,
            formulation=formulation,
        )
    else:
        inner = CCMPCController(
            config,
            formulation=formulation,
            solver_adapter=solver_adapter,
            fallback_controller=fallback_controller,
        )

    return _validate_controller_instance(_CCMPCControllerBridge(inner))


def _create_emergency_stop_controller(
    config: str | Path | Mapping[str, Any] | Any | None,
    *,
    fallback_controller: Any | None,
) -> ControllerProtocol:
    """Create emergency-stop/fallback controller bridge."""

    if fallback_controller is None:
        if isinstance(config, (str, Path)):
            raise ControllerFactoryConfigError(
                "Emergency-stop controller does not load config files directly. "
                "Pass an already-loaded mapping config or inject fallback_controller."
            )

        from ccmpc.controllers.fallback_controller import FallbackController

        config_mapping = _to_mapping_or_none(config)
        inner = FallbackController.from_config(config_mapping)
    else:
        inner = fallback_controller

    return _validate_controller_instance(_EmergencyStopControllerBridge(inner))


def _diagnostics_from_ccmpc_result(result: Any) -> ControllerDiagnostics:
    """Convert CCMPCSolveResult-like object to ControllerDiagnostics."""

    fallback_reason = None
    fallback_result = getattr(result, "fallback_result", None)
    if fallback_result is not None:
        fallback_reason = getattr(fallback_result, "reason", None)

    status = _status_from_ccmpc_result(result)

    return ControllerDiagnostics(
        status=status,
        success=bool(getattr(result, "success", False)),
        solve_time_ms=getattr(result, "solve_time_ms", None),
        objective_value=getattr(result, "objective_value", None),
        iterations=getattr(result, "iterations", None),
        fallback_used=bool(getattr(result, "used_fallback", False)),
        fallback_reason=fallback_reason,
        max_constraint_violation=None,
        min_obstacle_margin=None,
        notes=(),
        extra={
            "ccmpc_status": getattr(result, "status", None),
            "metadata": dict(getattr(result, "metadata", {}) or {}),
        },
    )


def _status_from_ccmpc_result(result: Any) -> ControllerStatus:
    """Map CCMPC solve status to controller-level status."""

    if bool(getattr(result, "used_fallback", False)):
        return ControllerStatus.FALLBACK

    if bool(getattr(result, "success", False)):
        return ControllerStatus.SUCCESS

    raw_status = str(getattr(result, "status", "")).lower()

    if "infeasible" in raw_status and hasattr(ControllerStatus, "INFEASIBLE"):
        return ControllerStatus.INFEASIBLE

    if "max_iter" in raw_status and hasattr(ControllerStatus, "MAX_ITER"):
        return ControllerStatus.MAX_ITER

    if hasattr(ControllerStatus, "FAILED"):
        return ControllerStatus.FAILED

    # Compatibility with alternative metadata enum variants.
    if hasattr(ControllerStatus, "SOLVER_ERROR"):
        return ControllerStatus.SOLVER_ERROR

    return ControllerStatus.FALLBACK


def _validate_controller_instance(controller: Any) -> ControllerProtocol:
    """Validate that an object satisfies the controller interface."""

    required_methods = ("reset", "compute_command", "get_metadata", "close")

    for method_name in required_methods:
        method = getattr(controller, method_name, None)
        if not callable(method):
            raise ControllerFactoryRegistryError(
                f"Controller object is missing callable method {method_name!r}."
            )

    metadata = controller.get_metadata()
    if not isinstance(metadata, ControllerMetadata):
        raise ControllerFactoryRegistryError(
            "Controller.get_metadata() must return ControllerMetadata."
        )

    return controller


def _get_registered_builder(
    registry: Mapping[ControllerType | str, ControllerBuilder],
    controller_type: ControllerType,
) -> ControllerBuilder | None:
    """Return custom registry builder for a controller type if available."""

    candidates: tuple[ControllerType | str, ...] = (
        controller_type,
        controller_type.value,
        controller_type.name,
    )

    for key in candidates:
        if key in registry:
            builder = registry[key]
            if not callable(builder):
                raise ControllerFactoryRegistryError(
                    f"Registry entry for {key!r} is not callable."
                )
            return builder

    return None


def _to_mapping_or_none(
    config: str | Path | Mapping[str, Any] | Any | None,
) -> dict[str, Any] | None:
    """Convert loaded config-like objects to a plain mapping when possible."""

    if config is None:
        return None

    if isinstance(config, (str, Path)):
        # Paths are intentionally not loaded here.  CCMPCController already
        # owns its legacy config loading path.
        return None

    if isinstance(config, Mapping):
        return dict(config)

    if is_dataclass(config):
        data = asdict(config)
        if isinstance(data, Mapping):
            return dict(data)

    model_dump = getattr(config, "model_dump", None)
    if callable(model_dump):
        data = model_dump()
        if isinstance(data, Mapping):
            return dict(data)

    dict_method = getattr(config, "dict", None)
    if callable(dict_method):
        data = dict_method()
        if isinstance(data, Mapping):
            return dict(data)

    if hasattr(config, "__dict__"):
        data = {
            key: value
            for key, value in vars(config).items()
            if not key.startswith("_")
        }
        if data:
            return data

    raise ControllerFactoryConfigError(
        f"Unsupported controller config type: {type(config).__name__}."
    )


def _controller_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return controller section from full config or the config itself."""

    maybe_controller = config.get("controller")
    if maybe_controller is None:
        return config

    if not isinstance(maybe_controller, Mapping):
        raise ControllerFactoryConfigError(
            "config['controller'] must be a mapping when provided."
        )

    return maybe_controller


__all__ = [
    "ControllerBuilder",
    "ControllerFactoryConfigError",
    "ControllerFactoryError",
    "ControllerFactoryRegistryError",
    "build_controller",
    "create_controller",
    "create_controller_from_config",
    "parse_controller_type",
    "resolve_controller_type",
]
