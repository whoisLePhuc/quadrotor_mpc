"""Unit tests for orchestrator-style CC-MPC controller.

Target module:
    ccmpc.controllers.ccmpc_controller

These tests intentionally do not build a real CVXPY problem.  Instead, they use
small fake problem/builder objects so the controller orchestration can be tested
independently from optimization model construction.

The real CVXPY variables, constraints, and objective should be covered later in
tests/unit/test_formulation.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ccmpc.controllers.ccmpc_controller import (
    CCMPC,
    CCMPCConfig,
    CCMPCConfigError,
    CCMPCController,
    CCMPCInputError,
    CCMPCOutputError,
    CCMPCSolveResult,
    MPCProblemBundle,
    load_controller_config,
)
from ccmpc.controllers.fallback_controller import FallbackMode
from ccmpc.controllers.solver_adapter import SolverResult, SolverStatus


def make_state(
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 1.0,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> np.ndarray:
    """Create canonical State9."""
    return np.array(
        [x, y, z, vx, vy, vz, roll, pitch, yaw],
        dtype=np.float64,
    )


def make_goal(
    *,
    x: float = 1.0,
    y: float = 2.0,
    z: float = 1.5,
) -> np.ndarray:
    """Create canonical Goal3."""
    return np.array([x, y, z], dtype=np.float64)


def make_gamma(scale: float = 0.01) -> np.ndarray:
    """Create canonical Gamma9x9 covariance."""
    return np.eye(9, dtype=np.float64) * scale


def make_controls_time_major(horizon: int, first: np.ndarray | None = None) -> np.ndarray:
    """Create time-major ControlCommand4 trajectory."""
    controls = np.zeros((horizon, 4), dtype=np.float64)
    if first is None:
        first = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float64)
    controls[0, :] = first
    return controls


def make_states_time_major(horizon: int, state: np.ndarray | None = None) -> np.ndarray:
    """Create time-major State9 trajectory."""
    x = make_state() if state is None else np.asarray(state, dtype=np.float64)
    return np.tile(x.reshape(1, 9), (horizon + 1, 1))


@dataclass
class FakeSolverStats:
    """Minimal CVXPY-like solver stats."""

    num_iters: int | None = 7


class FakeProblem:
    """Minimal CVXPY-like problem for SolverAdapter."""

    def __init__(
        self,
        *,
        status: str = "optimal",
        value: float | None = 1.0,
        raw_return: float | None = None,
        solver_stats: FakeSolverStats | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.status = status
        self.value = value
        self.raw_return = value if raw_return is None else raw_return
        self.solver_stats = FakeSolverStats() if solver_stats is None else solver_stats
        self.raises = raises
        self.solve_kwargs: dict[str, Any] | None = None
        self.solve_calls = 0

    def solve(self, **kwargs: Any) -> float | None:
        """Record kwargs and return configured objective."""
        self.solve_calls += 1
        self.solve_kwargs = kwargs
        if self.raises is not None:
            raise self.raises
        return self.raw_return


class ValueLike:
    """Small object exposing .value like CVXPY Variable/Expression."""

    def __init__(self, value: Any) -> None:
        self.value = value


class FakeFormulation:
    """Formulation returning a preconfigured MPCProblemBundle."""

    def __init__(
        self,
        *,
        problem: FakeProblem | None = None,
        command: Any | None = None,
        predicted_states: Any | None = None,
        predicted_controls: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.problem = FakeProblem() if problem is None else problem
        self.command = command
        self.predicted_states = predicted_states
        self.predicted_controls = predicted_controls
        self.metadata = {} if metadata is None else dict(metadata)
        self.calls: list[dict[str, Any]] = []

    def build(self, **kwargs: Any) -> MPCProblemBundle:
        """Record build kwargs and return bundle."""
        self.calls.append(kwargs)
        return MPCProblemBundle(
            problem=self.problem,
            command=self.command,
            predicted_states=self.predicted_states,
            predicted_controls=self.predicted_controls,
            metadata=self.metadata,
        )


class RawFormulation:
    """Builder that returns only a problem object."""

    def __init__(self, problem: FakeProblem | None = None) -> None:
        self.problem = FakeProblem() if problem is None else problem
        self.calls: list[dict[str, Any]] = []

    def build(self, **kwargs: Any) -> FakeProblem:
        self.calls.append(kwargs)
        return self.problem


class RaisingFormulation:
    """Builder whose build call raises."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = RuntimeError("builder exploded") if exc is None else exc
        self.calls: list[dict[str, Any]] = []

    def build(self, **kwargs: Any) -> MPCProblemBundle:
        self.calls.append(kwargs)
        raise self.exc


class ObjectWithObstacles:
    """Obstacle manager exposing .obstacles."""

    def __init__(self, obstacles: tuple[Any, ...]) -> None:
        self.obstacles = obstacles


class ObjectWithActiveObstacles:
    """Obstacle manager exposing .active_obstacles callable."""

    def __init__(self, obstacles: tuple[Any, ...]) -> None:
        self._obstacles = obstacles

    def active_obstacles(self) -> tuple[Any, ...]:
        return self._obstacles


def make_config(
    *,
    horizon: int = 5,
    dt: float = 0.2,
    fallback_mode: str = "hover",
    require_formulation: bool = False,
    use_fallback_on_failure: bool = True,
) -> dict[str, Any]:
    """Create direct controller config."""
    return {
        "dt": dt,
        "horizon": horizon,
        "fallback_mode": fallback_mode,
        "require_formulation": require_formulation,
        "use_fallback_on_failure": use_fallback_on_failure,
        "solver": "CLARABEL",
    }


def make_full_config() -> dict[str, Any]:
    """Create full project-style controller config."""
    return {
        "controller": {
            "name": "quadrotor-ccmpc",
            "prediction": {
                "timestep": 0.1,
                "horizon": 6,
                "solver": "CLARABEL",
            },
            "fallback": {
                "mode": "land",
                "land_descent_rate": 0.2,
            },
            "solver": {
                "verbose": False,
                "warm_start": True,
            },
            "use_fallback_on_failure": True,
        }
    }


def test_ccmpc_alias_points_to_controller() -> None:
    """CCMPC should remain a compatibility alias for CCMPCController."""
    assert CCMPC is CCMPCController


def test_ccmpc_config_defaults() -> None:
    """Default CCMPCConfig should be valid."""
    config = CCMPCConfig()

    assert config.dt == pytest.approx(0.1)
    assert config.horizon == 15
    assert config.name == "ccmpc"
    assert config.fallback_mode is FallbackMode.HOVER
    assert config.use_fallback_on_failure is True
    assert config.require_formulation is False


def test_ccmpc_config_from_direct_config() -> None:
    """CCMPCConfig.from_config should parse direct config."""
    config = CCMPCConfig.from_config(make_config(horizon=4, dt=0.05, fallback_mode="brake"))

    assert config.dt == pytest.approx(0.05)
    assert config.horizon == 4
    assert config.fallback_mode is FallbackMode.BRAKE


def test_ccmpc_config_from_full_config() -> None:
    """CCMPCConfig.from_config should parse full project config."""
    config = CCMPCConfig.from_config(make_full_config())

    assert config.dt == pytest.approx(0.1)
    assert config.horizon == 6
    assert config.name == "quadrotor-ccmpc"
    assert config.fallback_mode is FallbackMode.LAND


def test_ccmpc_config_from_horizon_time() -> None:
    """CCMPCConfig should derive horizon from horizon_time / dt."""
    config = CCMPCConfig.from_config({"dt": 0.1, "horizon_time": 1.2})

    assert config.horizon == 12


def test_ccmpc_config_rejects_bad_dt() -> None:
    """CCMPCConfig should reject non-positive dt."""
    with pytest.raises(CCMPCConfigError, match="dt"):
        CCMPCConfig(dt=0.0)


def test_ccmpc_config_rejects_bad_horizon() -> None:
    """CCMPCConfig should reject non-positive horizon."""
    with pytest.raises(CCMPCConfigError, match="horizon"):
        CCMPCConfig(horizon=0)


def test_ccmpc_config_rejects_bad_config_type() -> None:
    """CCMPCConfig.from_config should require mapping or None."""
    with pytest.raises(CCMPCConfigError, match="mapping"):
        CCMPCConfig.from_config(["bad"])  # type: ignore[arg-type]


def test_load_controller_config_none() -> None:
    """load_controller_config(None) should return empty dict."""
    assert load_controller_config(None) == {}


def test_load_controller_config_mapping() -> None:
    """load_controller_config should copy mapping config."""
    config = {"dt": 0.1}

    loaded = load_controller_config(config)

    assert loaded == config
    assert loaded is not config


def test_load_controller_config_missing_file() -> None:
    """load_controller_config should reject missing file path."""
    with pytest.raises(CCMPCConfigError, match="not found"):
        load_controller_config(Path("does_not_exist.yaml"))


def test_mpc_problem_bundle_valid() -> None:
    """MPCProblemBundle should accept a fake problem."""
    problem = FakeProblem()

    bundle = MPCProblemBundle(problem=problem, metadata={"source": "test"})

    assert bundle.problem is problem
    assert bundle.metadata == {"source": "test"}


def test_mpc_problem_bundle_rejects_none_problem() -> None:
    """MPCProblemBundle requires a problem."""
    with pytest.raises(CCMPCOutputError, match="problem"):
        MPCProblemBundle(problem=None)


def test_mpc_problem_bundle_rejects_bad_metadata() -> None:
    """MPCProblemBundle metadata must be a dict."""
    with pytest.raises(CCMPCOutputError, match="metadata"):
        MPCProblemBundle(problem=FakeProblem(), metadata=["bad"])  # type: ignore[arg-type]


def test_solve_without_formulation_uses_fallback() -> None:
    """Controller should return fallback result if no formulation is provided."""
    controller = CCMPCController(make_config(horizon=3))
    state = make_state(z=1.0)
    goal = make_goal()

    result = controller.solve(state, goal)

    assert result.success is False
    assert result.used_fallback is True
    assert result.status == "fallback"
    assert result.fallback_result is not None
    assert result.fallback_result.reason == "formulation_missing"
    assert np.allclose(result.command, np.zeros(4))
    assert result.predicted_states.shape == (4, 9)
    assert result.predicted_controls.shape == (3, 4)
    assert controller.previous_result is result
    assert controller.last_info is not None
    assert controller.last_info.fallback_used is True


def test_solve_without_formulation_can_be_required_to_raise() -> None:
    """require_formulation=True should raise instead of fallback."""
    controller = CCMPCController(make_config(require_formulation=True))

    with pytest.raises(CCMPCConfigError, match="formulation"):
        controller.solve(make_state(), make_goal())


def test_solve_success_uses_first_predicted_control() -> None:
    """On solver success, controller should use first predicted control as command."""
    horizon = 4
    state = make_state(x=1.0, z=2.0)
    goal = make_goal()
    controls = make_controls_time_major(horizon)
    states = make_states_time_major(horizon, state)
    problem = FakeProblem(status="optimal", value=12.0, solver_stats=FakeSolverStats(9))
    builder = FakeFormulation(
        problem=problem,
        predicted_states=states,
        predicted_controls=controls,
        metadata={"builder": "ok"},
    )
    controller = CCMPCController(make_config(horizon=horizon), formulation=builder)

    result = controller.solve(state, goal, metadata={"request_id": "abc"})

    assert result.success is True
    assert result.used_fallback is False
    assert result.status == "optimal"
    assert result.solver_result is not None
    assert result.solver_result.success is True
    assert result.objective_value == pytest.approx(12.0)
    assert result.iterations == 9
    assert np.allclose(result.command, controls[0])
    assert np.allclose(result.predicted_states, states)
    assert np.allclose(result.predicted_controls, controls)
    assert result.metadata["builder"] == "ok"
    assert result.metadata["request_id"] == "abc"
    assert result.metadata["obstacle_count"] == 0
    assert problem.solve_calls == 1
    assert builder.calls
    assert controller.previous_result is result
    assert controller.last_info is not None
    assert controller.last_info.success is True
    assert controller.last_info.fallback_used is False


def test_solve_success_uses_explicit_command_value_object() -> None:
    """Controller should extract command from CVXPY-like .value object."""
    horizon = 3
    explicit_command = np.array([0.2, -0.1, 0.05, 0.3], dtype=np.float64)
    controls = make_controls_time_major(horizon, first=np.array([9.0, 9.0, 9.0, 9.0]))
    builder = FakeFormulation(
        command=ValueLike(explicit_command),
        predicted_controls=controls,
    )
    controller = CCMPCController(make_config(horizon=horizon), formulation=builder)

    result = controller.solve(make_state(), make_goal())

    assert result.success is True
    assert np.allclose(result.command, explicit_command)
    assert np.allclose(result.predicted_controls, controls)


def test_solve_success_accepts_state_major_and_control_major_trajectories() -> None:
    """Controller should convert legacy state-major/control-major trajectories.

    Use horizon=5 instead of horizon=4 to avoid the ambiguous square shape
    (4, 4), because ControlCommand4 also has dimension 4.
    """
    horizon = 5
    state = make_state()
    states_time_major = make_states_time_major(horizon, state)
    controls_time_major = make_controls_time_major(horizon)

    builder = FakeFormulation(
        predicted_states=states_time_major.T,
        predicted_controls=controls_time_major.T,
    )
    controller = CCMPCController(make_config(horizon=horizon), formulation=builder)

    result = controller.solve(state, make_goal())

    assert np.allclose(result.predicted_states, states_time_major)
    assert np.allclose(result.predicted_controls, controls_time_major)


def test_solve_success_with_only_command_creates_hold_trajectories() -> None:
    """If only command is provided, controller should create hold trajectories."""
    horizon = 5
    state = make_state(x=1.0, y=2.0, z=3.0)
    command = np.array([0.1, 0.2, -0.1, 0.0], dtype=np.float64)
    builder = FakeFormulation(command=command)
    controller = CCMPCController(make_config(horizon=horizon), formulation=builder)

    result = controller.solve(state, make_goal())

    assert np.allclose(result.command, command)
    assert result.predicted_states.shape == (horizon + 1, 9)
    assert result.predicted_controls.shape == (horizon, 4)
    assert np.allclose(result.predicted_states[0], state)
    assert np.allclose(result.predicted_controls[0], command)


def test_solve_solver_failure_uses_fallback() -> None:
    """Infeasible solver status should route to fallback."""
    problem = FakeProblem(status="infeasible", value=None, raw_return=None)
    builder = FakeFormulation(
        problem=problem,
        predicted_controls=make_controls_time_major(3),
    )
    controller = CCMPCController(
        make_config(horizon=3, fallback_mode="hover"),
        formulation=builder,
    )

    result = controller.solve(make_state(vx=1.0), make_goal())

    assert result.success is False
    assert result.used_fallback is True
    assert result.status == "fallback"
    assert result.solver_result is not None
    assert result.solver_result.status == "infeasible"
    assert result.fallback_result is not None
    assert result.fallback_result.reason == "solver_infeasible"
    assert np.allclose(result.command, np.zeros(4))


def test_solve_solver_failure_can_raise_when_fallback_disabled() -> None:
    """If fallback disabled, solver failure should raise through SolverResult."""
    problem = FakeProblem(status="infeasible", value=None, raw_return=None)
    builder = FakeFormulation(problem=problem, predicted_controls=make_controls_time_major(3))
    controller = CCMPCController(
        make_config(horizon=3, use_fallback_on_failure=False),
        formulation=builder,
    )

    with pytest.raises(Exception, match="infeasible"):
        controller.solve(make_state(), make_goal())


def test_solve_formulation_exception_uses_fallback() -> None:
    """Formulation exceptions should route to fallback by default."""
    builder = RaisingFormulation(RuntimeError("builder failed"))
    controller = CCMPCController(make_config(horizon=3), formulation=builder)

    result = controller.solve(make_state(), make_goal())

    assert result.used_fallback is True
    assert result.fallback_result is not None
    assert result.fallback_result.reason == "formulation_failed"
    assert result.metadata["exception_type"] == "RuntimeError"
    assert result.metadata["error_message"] == "builder failed"


def test_solve_formulation_exception_can_raise_when_fallback_disabled() -> None:
    """Formulation exception should raise when fallback is disabled."""
    builder = RaisingFormulation(RuntimeError("builder failed"))
    controller = CCMPCController(
        make_config(horizon=3, use_fallback_on_failure=False),
        formulation=builder,
    )

    with pytest.raises(CCMPCOutputError, match="builder failed"):
        controller.solve(make_state(), make_goal())


def test_solve_solution_extraction_failure_uses_fallback() -> None:
    """Successful solver but missing command/controls should route to fallback."""
    builder = RawFormulation(FakeProblem(status="optimal", value=1.0))
    controller = CCMPCController(make_config(horizon=3), formulation=builder)

    result = controller.solve(make_state(), make_goal())

    assert result.used_fallback is True
    assert result.fallback_result is not None
    assert result.fallback_result.reason == "solution_extraction_failed"
    assert result.solver_result is not None
    assert result.solver_result.success is True


def test_solve_solution_extraction_failure_can_raise_when_fallback_disabled() -> None:
    """Missing command/controls should raise when fallback is disabled."""
    builder = RawFormulation(FakeProblem(status="optimal", value=1.0))
    controller = CCMPCController(
        make_config(horizon=3, use_fallback_on_failure=False),
        formulation=builder,
    )

    with pytest.raises(CCMPCOutputError, match="command or predicted_controls"):
        controller.solve(make_state(), make_goal())


def test_solve_rejects_missing_state() -> None:
    """solve should require estimated_state or initial_state."""
    controller = CCMPCController(make_config())

    with pytest.raises(CCMPCInputError, match="estimated_state"):
        controller.solve(goal=make_goal())


def test_solve_rejects_missing_goal() -> None:
    """solve should require goal."""
    controller = CCMPCController(make_config())

    with pytest.raises(CCMPCInputError, match="goal"):
        controller.solve(make_state())


def test_solve_rejects_invalid_state_shape() -> None:
    """solve should reject non-State9 input."""
    controller = CCMPCController(make_config())

    with pytest.raises(Exception):
        controller.solve(np.zeros(8, dtype=np.float64), make_goal())


def test_solve_rejects_invalid_goal_shape() -> None:
    """solve should reject non-Goal3 input."""
    controller = CCMPCController(make_config())

    with pytest.raises(Exception):
        controller.solve(make_state(), np.zeros(2, dtype=np.float64))


def test_solve_accepts_legacy_initial_state_and_gamma_0() -> None:
    """solve should accept legacy initial_state and Gamma_0 names."""
    horizon = 3
    gamma = make_gamma()
    builder = FakeFormulation(predicted_controls=make_controls_time_major(horizon))
    controller = CCMPCController(make_config(horizon=horizon), formulation=builder)

    result = controller.solve(
        initial_state=make_state(),
        goal=make_goal(),
        Gamma_0=gamma,
    )

    assert result.success is True
    assert builder.calls
    assert np.allclose(builder.calls[0]["covariance"], gamma)


def test_solve_rejects_mismatched_state_aliases() -> None:
    """estimated_state and initial_state must match if both provided."""
    controller = CCMPCController(make_config())
    state_a = make_state(x=0.0)
    state_b = make_state(x=1.0)

    with pytest.raises(CCMPCInputError, match="differ"):
        controller.solve(state_a, make_goal(), initial_state=state_b)


def test_solve_rejects_mismatched_covariance_aliases() -> None:
    """covariance and Gamma_0 must match if both provided."""
    controller = CCMPCController(make_config())
    gamma_a = make_gamma(0.01)
    gamma_b = make_gamma(0.02)

    with pytest.raises(CCMPCInputError, match="differ"):
        controller.solve(make_state(), make_goal(), covariance=gamma_a, Gamma_0=gamma_b)


def test_solve_passes_obstacles_sequence_to_builder() -> None:
    """obstacles sequence should be passed through as tuple."""
    obstacles = [{"id": "obs_1"}, {"id": "obs_2"}]
    builder = FakeFormulation(predicted_controls=make_controls_time_major(3))
    controller = CCMPCController(make_config(horizon=3), formulation=builder)

    result = controller.solve(make_state(), make_goal(), obstacles=obstacles)

    assert result.success is True
    assert builder.calls[0]["obstacles"] == tuple(obstacles)
    assert result.metadata["obstacle_count"] == 2


def test_solve_accepts_obstacle_manager_obstacles_attr() -> None:
    """obstacle_manager exposing .obstacles should be accepted."""
    obstacles = ({"id": "obs_1"},)
    manager = ObjectWithObstacles(obstacles)
    builder = FakeFormulation(predicted_controls=make_controls_time_major(3))
    controller = CCMPCController(make_config(horizon=3), formulation=builder)

    result = controller.solve(make_state(), make_goal(), obstacle_manager=manager)

    assert result.success is True
    assert builder.calls[0]["obstacles"] == obstacles
    assert result.metadata["obstacle_count"] == 1


def test_solve_accepts_obstacle_manager_active_obstacles_callable() -> None:
    """obstacle_manager exposing active_obstacles() should be accepted."""
    obstacles = ({"id": "obs_1"}, {"id": "obs_2"})
    manager = ObjectWithActiveObstacles(obstacles)
    builder = FakeFormulation(predicted_controls=make_controls_time_major(3))
    controller = CCMPCController(make_config(horizon=3), formulation=builder)

    result = controller.solve(make_state(), make_goal(), obstacle_manager=manager)

    assert result.success is True
    assert builder.calls[0]["obstacles"] == obstacles
    assert result.metadata["obstacle_count"] == 2


def test_solve_rejects_obstacles_and_obstacle_manager_together() -> None:
    """solve should reject ambiguous obstacle inputs."""
    controller = CCMPCController(make_config())

    with pytest.raises(CCMPCInputError, match="either obstacles or obstacle_manager"):
        controller.solve(
            make_state(),
            make_goal(),
            obstacles=[],
            obstacle_manager=ObjectWithObstacles(()),
        )


def test_solve_passes_reference_trajectory_to_builder() -> None:
    """reference_trajectory should be validated and passed to builder."""
    horizon = 3
    reference = make_states_time_major(horizon)
    builder = FakeFormulation(predicted_controls=make_controls_time_major(horizon))
    controller = CCMPCController(make_config(horizon=horizon), formulation=builder)

    result = controller.solve(
        make_state(),
        make_goal(),
        reference_trajectory=reference,
    )

    assert result.success is True
    assert np.allclose(builder.calls[0]["reference_trajectory"], reference)


def test_solve_rejects_invalid_reference_trajectory() -> None:
    """reference_trajectory must be a valid time-major State9 trajectory."""
    controller = CCMPCController(make_config())

    with pytest.raises(Exception):
        controller.solve(
            make_state(),
            make_goal(),
            reference_trajectory=np.zeros((3, 8), dtype=np.float64),
        )


def test_previous_solution_is_passed_to_next_build() -> None:
    """Controller should pass previous_solution to builder for warm-start use."""
    horizon = 3
    builder = FakeFormulation(predicted_controls=make_controls_time_major(horizon))
    controller = CCMPCController(make_config(horizon=horizon), formulation=builder)

    first = controller.solve(make_state(), make_goal())
    second = controller.solve(make_state(x=0.1), make_goal())

    assert first.success is True
    assert second.success is True
    assert builder.calls[0]["previous_solution"] is None
    assert builder.calls[1]["previous_solution"] is first
    assert controller.previous_result is second


def test_reset_clears_previous_result_and_last_info() -> None:
    """reset should clear warm-start/diagnostic state."""
    builder = FakeFormulation(predicted_controls=make_controls_time_major(3))
    controller = CCMPCController(make_config(horizon=3), formulation=builder)

    controller.solve(make_state(), make_goal())
    assert controller.previous_result is not None
    assert controller.last_info is not None

    controller.reset()

    assert controller.previous_result is None
    assert controller.last_info is None


def test_compute_command_returns_command_only() -> None:
    """compute_command should return only ControlCommand4."""
    controls = make_controls_time_major(3)
    builder = FakeFormulation(predicted_controls=controls)
    controller = CCMPCController(make_config(horizon=3), formulation=builder)

    command = controller.compute_command(make_state(), make_goal())

    assert np.allclose(command, controls[0])
    assert command.shape == (4,)


def test_solve_result_info_property() -> None:
    """CCMPCSolveResult.info should expose compact diagnostics."""
    controls = make_controls_time_major(3)
    builder = FakeFormulation(predicted_controls=controls)
    controller = CCMPCController(make_config(horizon=3), formulation=builder)

    result = controller.solve(make_state(), make_goal())
    info = result.info

    assert info.success is True
    assert info.status == "optimal"
    assert info.fallback_used is False
    assert info.solver_name == "CLARABEL"
    assert info.converged is True


def test_solve_result_legacy_tuple_unpacking() -> None:
    """CCMPCSolveResult should support legacy x_traj, u_traj unpacking."""
    horizon = 3
    states = make_states_time_major(horizon)
    controls = make_controls_time_major(horizon)
    builder = FakeFormulation(predicted_states=states, predicted_controls=controls)
    controller = CCMPCController(make_config(horizon=horizon), formulation=builder)

    result = controller.solve(make_state(), make_goal())
    x_traj, u_traj = result

    assert x_traj.shape == (9, horizon + 1)
    assert u_traj.shape == (4, horizon)
    assert np.allclose(x_traj, states.T)
    assert np.allclose(u_traj, controls.T)


def test_ccmpc_solve_result_rejects_bad_command() -> None:
    """CCMPCSolveResult should validate command shape."""
    with pytest.raises(Exception):
        CCMPCSolveResult(
            command=np.zeros(3, dtype=np.float64),
            success=True,
            status="optimal",
            used_fallback=False,
            solve_time_ms=0.1,
            predicted_states=make_states_time_major(3),
            predicted_controls=make_controls_time_major(3),
        )


def test_ccmpc_solve_result_rejects_bad_state_trajectory() -> None:
    """CCMPCSolveResult should validate predicted state trajectory."""
    with pytest.raises(Exception):
        CCMPCSolveResult(
            command=np.zeros(4, dtype=np.float64),
            success=True,
            status="optimal",
            used_fallback=False,
            solve_time_ms=0.1,
            predicted_states=np.zeros((4, 8), dtype=np.float64),
            predicted_controls=make_controls_time_major(3),
        )


def test_ccmpc_solve_result_rejects_bad_control_trajectory() -> None:
    """CCMPCSolveResult should validate predicted control trajectory."""
    with pytest.raises(Exception):
        CCMPCSolveResult(
            command=np.zeros(4, dtype=np.float64),
            success=True,
            status="optimal",
            used_fallback=False,
            solve_time_ms=0.1,
            predicted_states=make_states_time_major(3),
            predicted_controls=np.zeros((3, 3), dtype=np.float64),
        )


def test_ccmpc_solve_result_accepts_solver_result() -> None:
    """CCMPCSolveResult should accept a SolverResult object."""
    solver_result = SolverResult(
        success=True,
        status="optimal",
        solve_time_ms=1.0,
        objective_value=1.5,
        solver_name="CLARABEL",
        adapter_status=SolverStatus.SUCCESS,
        iterations=3,
    )

    result = CCMPCSolveResult(
        command=np.zeros(4, dtype=np.float64),
        success=True,
        status="optimal",
        used_fallback=False,
        solve_time_ms=1.0,
        predicted_states=make_states_time_major(3),
        predicted_controls=make_controls_time_major(3),
        solver_result=solver_result,
        objective_value=solver_result.objective_value,
        iterations=solver_result.iterations,
    )

    assert result.solver_result is solver_result
    assert result.info.objective_value == pytest.approx(1.5)
    assert result.info.iterations == 3
