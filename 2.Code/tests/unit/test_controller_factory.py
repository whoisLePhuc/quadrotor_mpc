"""Unit tests for simulation.controllers.factory.

These tests verify that the controller factory is the public construction point
for simulation controllers.  They deliberately avoid building a real CC-MPC
optimization problem and instead use fake controller modules/classes.

Target module:
    simulation.controllers.factory
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from simulation.controllers.base import (
    ControllerDiagnostics,
    ControllerInput,
    ControllerMetadata,
    ControllerOutput,
    ControllerStatus,
    ObstaclePrediction,
    ControllerType,
)
from simulation.controllers.factory import (
    ControllerFactoryConfigError,
    ControllerFactoryRegistryError,
    build_controller,
    create_controller,
    create_controller_from_config,
    parse_controller_type,
    resolve_controller_type,
)


def make_state9() -> np.ndarray:
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


def make_goal3() -> np.ndarray:
    """Create a valid canonical Goal3."""
    return np.array([1.0, 2.0, 1.5], dtype=np.float64)


def make_gamma9x9() -> np.ndarray:
    """Create a valid Gamma9x9 covariance."""
    return np.eye(9, dtype=np.float64) * 0.01


def make_obstacle_prediction(obstacle_id: str) -> ObstaclePrediction:
    """Create a valid ObstaclePrediction for factory bridge tests."""
    return ObstaclePrediction(
        obstacle_id=obstacle_id,
        positions=np.array(
            [
                [1.0, 0.0, 1.0],
                [1.1, 0.0, 1.0],
                [1.2, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        radii=np.array([0.3, 0.3, 0.5], dtype=np.float64),
        covariance=np.eye(3, dtype=np.float64) * 0.01,
        active=True,
        metadata={"source": "unit-test"},
    )


def make_controller_input(
    *,
    obstacle_predictions: tuple[Any, ...] = (),
    covariance: np.ndarray | None = None,
) -> ControllerInput:
    """Create a valid ControllerInput."""
    return ControllerInput(
        time=0.12,
        estimated_state=make_state9(),
        goal=make_goal3(),
        covariance=covariance,
        obstacle_predictions=obstacle_predictions,
        previous_solution=None,
        reference_trajectory=None,
        config={"horizon": 3, "dt": 0.06},
        metadata={"request_id": "factory-test"},
    )


@dataclass
class FakeFallbackReason:
    """Small fallback result-like object."""

    reason: str = "solver_failed"


class FakeCCMPCResult:
    """Minimal CCMPCSolveResult-like object returned by fake inner controller."""

    def __init__(
        self,
        *,
        success: bool = True,
        used_fallback: bool = False,
        status: str = "optimal",
    ) -> None:
        self.command = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float64)
        self.predicted_states = np.zeros((4, 9), dtype=np.float64)
        self.predicted_states[:, 2] = 1.0
        self.predicted_controls = np.zeros((3, 4), dtype=np.float64)
        self.predicted_controls[0, :] = self.command
        self.success = success
        self.used_fallback = used_fallback
        self.status = status
        self.solve_time_ms = 2.5
        self.objective_value = 12.0
        self.iterations = 3
        self.metadata = {"inner": "fake-ccmpc"}
        self.fallback_result = (
            FakeFallbackReason("formulation_failed") if used_fallback else None
        )


class FakeCCMPCInner:
    """Fake concrete CC-MPC controller implementation."""

    instances: list["FakeCCMPCInner"] = []
    from_config_calls: list[dict[str, Any]] = []
    init_calls: list[dict[str, Any]] = []

    def __init__(
        self,
        config: Any = None,
        *,
        formulation: Any | None = None,
        solver_adapter: Any | None = None,
        fallback_controller: Any | None = None,
    ) -> None:
        self.config_input = config
        self.formulation = formulation
        self.solver_adapter = solver_adapter
        self.fallback_controller = fallback_controller
        self.config = types.SimpleNamespace(name="fake-ccmpc", horizon=3, dt=0.06)
        self.solve_calls: list[dict[str, Any]] = []
        self.reset_calls = 0
        self.close_calls = 0
        type(self).instances.append(self)
        type(self).init_calls.append(
            {
                "config": config,
                "formulation": formulation,
                "solver_adapter": solver_adapter,
                "fallback_controller": fallback_controller,
            }
        )

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        formulation: Any | None = None,
    ) -> "FakeCCMPCInner":
        cls.from_config_calls.append(
            {
                "config": config,
                "formulation": formulation,
            }
        )
        return cls(config, formulation=formulation)

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self.close_calls += 1

    def solve(self, **kwargs: Any) -> FakeCCMPCResult:
        self.solve_calls.append(kwargs)
        return FakeCCMPCResult()


class FakeFallbackMode:
    """Simple enum-like object with .value used by factory diagnostics."""

    value = "hover"


class FakeFallbackStatus:
    """Simple enum-like object with .value used by factory diagnostics."""

    value = "active"


@dataclass
class FakeFallbackResult:
    """Minimal FallbackResult-like object."""

    command: np.ndarray
    reason: str
    mode: Any = FakeFallbackMode()
    status: Any = FakeFallbackStatus()
    clipped: bool = False
    horizontal_speed: float = 0.0
    altitude: float = 1.0
    metadata: dict[str, Any] | None = None


class FakeFallbackInner:
    """Fake concrete fallback controller implementation."""

    instances: list["FakeFallbackInner"] = []
    from_config_calls: list[Any] = []

    def __init__(self) -> None:
        self.compute_calls: list[dict[str, Any]] = []
        self.reset_calls = 0
        self.close_calls = 0
        type(self).instances.append(self)

    @classmethod
    def from_config(cls, config: Any) -> "FakeFallbackInner":
        cls.from_config_calls.append(config)
        return cls()

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self.close_calls += 1

    def compute(
        self,
        estimated_state: np.ndarray,
        goal: np.ndarray,
        *,
        reason: str,
        metadata: dict[str, Any],
    ) -> FakeFallbackResult:
        self.compute_calls.append(
            {
                "estimated_state": estimated_state,
                "goal": goal,
                "reason": reason,
                "metadata": metadata,
            }
        )
        return FakeFallbackResult(
            command=np.zeros(4, dtype=np.float64),
            reason=reason,
            metadata={"inner": "fake-fallback"},
        )


class FakeRegisteredController:
    """Fake custom controller returned by registry."""

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.reset_calls = 0
        self.close_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def compute_command(self, input_data: ControllerInput) -> ControllerOutput:
        return ControllerOutput(
            command=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            diagnostics=ControllerDiagnostics(
                status=ControllerStatus.SUCCESS,
                success=True,
            ),
        )

    def get_metadata(self) -> ControllerMetadata:
        return ControllerMetadata(
            controller_type=ControllerType.PID,
            name="FakeRegisteredController",
            supports_obstacles=False,
            supports_covariance=False,
        )

    def close(self) -> None:
        self.close_calls += 1


class BadRegisteredController:
    """Object that is intentionally missing the controller interface."""

    def get_metadata(self) -> ControllerMetadata:
        return ControllerMetadata(
            controller_type=ControllerType.PID,
            name="BadRegisteredController",
        )


@pytest.fixture(autouse=True)
def reset_fake_classes() -> None:
    """Reset fake class-level call logs before each test."""
    FakeCCMPCInner.instances.clear()
    FakeCCMPCInner.from_config_calls.clear()
    FakeCCMPCInner.init_calls.clear()
    FakeFallbackInner.instances.clear()
    FakeFallbackInner.from_config_calls.clear()


@pytest.fixture
def fake_ccmpc_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake ccmpc.controllers.ccmpc_controller module."""
    module = types.ModuleType("ccmpc.controllers.ccmpc_controller")
    module.CCMPCController = FakeCCMPCInner
    monkeypatch.setitem(sys.modules, "ccmpc.controllers.ccmpc_controller", module)
    return module


@pytest.fixture
def fake_fallback_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake ccmpc.controllers.fallback_controller module."""
    module = types.ModuleType("ccmpc.controllers.fallback_controller")
    module.FallbackController = FakeFallbackInner
    monkeypatch.setitem(sys.modules, "ccmpc.controllers.fallback_controller", module)
    return module


def test_parse_controller_type_accepts_ccmpc_aliases() -> None:
    """Factory should accept common CC-MPC naming variants."""
    assert parse_controller_type(ControllerType.CCMPC) is ControllerType.CCMPC
    assert parse_controller_type("ccmpc") is ControllerType.CCMPC
    assert parse_controller_type("cc-mpc") is ControllerType.CCMPC
    assert parse_controller_type("quadrotor-ccmpc") is ControllerType.CCMPC
    assert parse_controller_type("ControllerType.CCMPC") is ControllerType.CCMPC


def test_parse_controller_type_accepts_emergency_aliases() -> None:
    """Factory should accept common emergency/fallback naming variants."""
    assert parse_controller_type("emergency_stop") is ControllerType.EMERGENCY_STOP
    assert parse_controller_type("fallback") is ControllerType.EMERGENCY_STOP
    assert parse_controller_type("safe-stop") is ControllerType.EMERGENCY_STOP


def test_parse_controller_type_rejects_unknown_type() -> None:
    """Unknown controller type should fail before construction."""
    with pytest.raises(ControllerFactoryConfigError, match="Unsupported controller type"):
        parse_controller_type("unknown-controller")


def test_resolve_controller_type_defaults_to_ccmpc() -> None:
    """Missing type should default to the project's primary CC-MPC controller."""
    assert resolve_controller_type(None) is ControllerType.CCMPC
    assert resolve_controller_type({}) is ControllerType.CCMPC


def test_resolve_controller_type_reads_root_config() -> None:
    """Factory should resolve controller type from root config."""
    assert resolve_controller_type({"type": "ccmpc"}) is ControllerType.CCMPC


def test_resolve_controller_type_reads_nested_controller_config() -> None:
    """Factory should resolve controller type from config['controller']."""
    config = {"controller": {"type": "emergency_stop"}}

    assert resolve_controller_type(config) is ControllerType.EMERGENCY_STOP


def test_resolve_controller_type_accepts_name_alias() -> None:
    """Legacy/full configs may identify the controller using name."""
    config = {"controller": {"name": "quadrotor-ccmpc"}}

    assert resolve_controller_type(config) is ControllerType.CCMPC


def test_resolve_controller_type_override_wins() -> None:
    """Explicit override should take priority over config fields."""
    config = {"controller": {"type": "emergency_stop"}}

    assert (
        resolve_controller_type(config, override=ControllerType.CCMPC)
        is ControllerType.CCMPC
    )


def test_resolve_controller_type_rejects_bad_controller_section() -> None:
    """Nested controller config must be a mapping."""
    with pytest.raises(ControllerFactoryConfigError, match="controller"):
        resolve_controller_type({"controller": "ccmpc"})


def test_create_controller_default_builds_ccmpc_bridge(
    fake_ccmpc_module: types.ModuleType,
) -> None:
    """Default factory construction should build a CC-MPC bridge."""
    controller = create_controller({"controller": {"type": "ccmpc"}})

    metadata = controller.get_metadata()

    assert metadata.controller_type is ControllerType.CCMPC
    assert metadata.name == "fake-ccmpc"
    assert metadata.supports_obstacles is True
    assert metadata.supports_covariance is True
    assert FakeCCMPCInner.from_config_calls
    assert FakeCCMPCInner.instances


def test_factory_aliases_call_create_controller(
    fake_ccmpc_module: types.ModuleType,
) -> None:
    """Backward-compatible aliases should construct controllers too."""
    controller_a = create_controller_from_config({"type": "ccmpc"})
    controller_b = build_controller({"type": "ccmpc"})

    assert controller_a.get_metadata().controller_type is ControllerType.CCMPC
    assert controller_b.get_metadata().controller_type is ControllerType.CCMPC


def test_ccmpc_bridge_compute_command_returns_controller_output(
    fake_ccmpc_module: types.ModuleType,
) -> None:
    """CC-MPC bridge should convert ControllerInput -> ControllerOutput."""
    controller = create_controller({"controller": {"type": "ccmpc"}})
    input_data = make_controller_input(covariance=make_gamma9x9())

    output = controller.compute_command(input_data)

    assert isinstance(output, ControllerOutput)
    assert output.command.shape == (4,)
    assert output.predicted_trajectory is not None
    assert output.predicted_trajectory.shape == (4, 9)
    assert output.control_trajectory is not None
    assert output.control_trajectory.shape == (3, 4)
    assert output.diagnostics.status is ControllerStatus.SUCCESS
    assert output.diagnostics.success is True
    assert output.diagnostics.solve_time_ms == pytest.approx(2.5)
    assert output.diagnostics.objective_value == pytest.approx(12.0)
    assert output.diagnostics.iterations == 3
    assert output.raw_solution is not None


def test_ccmpc_bridge_passes_canonical_input_to_inner_solve(
    fake_ccmpc_module: types.ModuleType,
) -> None:
    """Bridge should pass canonical fields into the inner solve() call."""
    obstacles = (
        make_obstacle_prediction("obs_1"),
        make_obstacle_prediction("obs_2"),
    )
    controller = create_controller({"controller": {"type": "ccmpc"}})
    input_data = make_controller_input(
        obstacle_predictions=obstacles,
        covariance=make_gamma9x9(),
    )

    controller.compute_command(input_data)

    inner = FakeCCMPCInner.instances[-1]
    assert len(inner.solve_calls) == 1
    call = inner.solve_calls[0]

    assert np.allclose(call["estimated_state"], input_data.estimated_state)
    assert np.allclose(call["goal"], input_data.goal)
    assert np.allclose(call["covariance"], input_data.covariance)
    assert call["obstacles"] == obstacles
    assert call["reference_trajectory"] is None
    assert call["time_s"] == pytest.approx(input_data.time)
    assert call["metadata"]["source"] == "simulation.controllers.factory"
    assert call["metadata"]["input_metadata"] == {"request_id": "factory-test"}


def test_ccmpc_bridge_reset_and_close_delegate_to_inner(
    fake_ccmpc_module: types.ModuleType,
) -> None:
    """Bridge lifecycle calls should delegate when the inner controller supports them."""
    controller = create_controller({"type": "ccmpc"})
    inner = FakeCCMPCInner.instances[-1]

    controller.reset()
    controller.close()

    assert inner.reset_calls == 1
    assert inner.close_calls == 1


def test_create_controller_injects_formulation_into_ccmpc_from_config(
    fake_ccmpc_module: types.ModuleType,
) -> None:
    """Factory should inject formulation into CCMPCController.from_config()."""
    formulation = object()

    create_controller({"type": "ccmpc"}, formulation=formulation)

    assert FakeCCMPCInner.from_config_calls[-1]["formulation"] is formulation


def test_create_controller_uses_direct_constructor_when_extra_deps_injected(
    fake_ccmpc_module: types.ModuleType,
) -> None:
    """Injected solver/fallback dependencies should use the direct constructor path."""
    solver_adapter = object()
    fallback_controller = object()

    create_controller(
        {"type": "ccmpc"},
        solver_adapter=solver_adapter,
        fallback_controller=fallback_controller,
    )

    assert FakeCCMPCInner.init_calls[-1]["solver_adapter"] is solver_adapter
    assert FakeCCMPCInner.init_calls[-1]["fallback_controller"] is fallback_controller


def test_create_controller_emergency_stop_builds_fallback_bridge(
    fake_fallback_module: types.ModuleType,
) -> None:
    """Emergency-stop factory target should wrap a fallback controller."""
    controller = create_controller({"type": "emergency_stop"})

    metadata = controller.get_metadata()

    assert metadata.controller_type is ControllerType.EMERGENCY_STOP
    assert metadata.name == "EmergencyStopController"
    assert FakeFallbackInner.from_config_calls
    assert FakeFallbackInner.instances


def test_emergency_stop_bridge_compute_command_returns_controller_output(
    fake_fallback_module: types.ModuleType,
) -> None:
    """Emergency-stop bridge should return canonical ControllerOutput."""
    controller = create_controller({"type": "emergency_stop"})
    input_data = make_controller_input()

    output = controller.compute_command(input_data)

    assert isinstance(output, ControllerOutput)
    assert np.allclose(output.command, np.zeros(4))
    assert output.diagnostics.status is ControllerStatus.FALLBACK
    assert output.diagnostics.success is False
    assert output.diagnostics.fallback_used is True
    assert output.diagnostics.fallback_reason == "emergency_stop_controller"
    assert output.raw_solution is not None


def test_emergency_stop_bridge_passes_canonical_input_to_fallback(
    fake_fallback_module: types.ModuleType,
) -> None:
    """Emergency bridge should pass State9 and Goal3 into fallback compute()."""
    controller = create_controller({"type": "emergency_stop"})
    input_data = make_controller_input()

    controller.compute_command(input_data)

    inner = FakeFallbackInner.instances[-1]
    assert len(inner.compute_calls) == 1
    call = inner.compute_calls[0]

    assert np.allclose(call["estimated_state"], input_data.estimated_state)
    assert np.allclose(call["goal"], input_data.goal)
    assert call["reason"] == "emergency_stop_controller"
    assert call["metadata"]["source"] == "simulation.controllers.factory"


def test_emergency_stop_rejects_path_config_without_injected_fallback(
    fake_fallback_module: types.ModuleType,
) -> None:
    """Fallback bridge should not own config-file loading."""
    with pytest.raises(ControllerFactoryConfigError, match="does not load config files"):
        create_controller(
            Path("controller.yaml"),
            controller_type=ControllerType.EMERGENCY_STOP,
        )


def test_create_controller_rejects_unimplemented_default_type() -> None:
    """PID/LQR/NOMINAL_MPC require a registry entry for now."""
    with pytest.raises(ControllerFactoryConfigError, match="not implemented"):
        create_controller({"type": "pid"})


def test_create_controller_uses_registry_for_custom_controller() -> None:
    """Factory extension point should allow externally registered controllers."""
    config = {"type": "pid", "gain": 1.0}

    controller = create_controller(
        config,
        registry={
            ControllerType.PID: lambda cfg: FakeRegisteredController(cfg),
        },
    )

    assert controller.get_metadata().controller_type is ControllerType.PID
    assert isinstance(controller.compute_command(make_controller_input()), ControllerOutput)


def test_create_controller_registry_can_use_string_keys() -> None:
    """Registry should accept enum value/name string keys for convenience."""
    controller = create_controller(
        {"type": "pid"},
        registry={
            "pid": lambda cfg: FakeRegisteredController(cfg),
        },
    )

    assert controller.get_metadata().controller_type is ControllerType.PID


def test_create_controller_rejects_noncallable_registry_entry() -> None:
    """Registry entries must be callable builders."""
    with pytest.raises(ControllerFactoryRegistryError, match="not callable"):
        create_controller(
            {"type": "pid"},
            registry={
                ControllerType.PID: object(),  # type: ignore[dict-item]
            },
        )


def test_create_controller_rejects_bad_registered_controller() -> None:
    """Registry builder must return an object satisfying the Controller interface."""
    with pytest.raises(ControllerFactoryRegistryError, match="missing callable method"):
        create_controller(
            {"type": "pid"},
            registry={
                ControllerType.PID: lambda cfg: BadRegisteredController(),
            },
        )


def test_create_controller_rejects_registry_controller_bad_metadata() -> None:
    """get_metadata() must return ControllerMetadata."""
    class BadMetadataController(FakeRegisteredController):
        def get_metadata(self) -> dict[str, str]:  # type: ignore[override]
            return {"bad": "metadata"}

    with pytest.raises(ControllerFactoryRegistryError, match="ControllerMetadata"):
        create_controller(
            {"type": "pid"},
            registry={
                ControllerType.PID: lambda cfg: BadMetadataController(cfg),
            },
        )


def test_factory_source_does_not_depend_on_engines_or_mujoco() -> None:
    """Controller factory must not import engine-specific modules."""
    import inspect
    import simulation.controllers.factory as factory_module

    source = inspect.getsource(factory_module).lower()

    assert "simulation.engines" not in source
    assert "mujoco" not in source
    assert "physicsengine" not in source
