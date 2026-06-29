"""Orchestrator-style CC-MPC controller.

This module intentionally keeps the main controller thin.  It coordinates:

- canonical input validation from ``ccmpc.types``
- optimization solving through ``SolverAdapter``
- safety fallback through ``FallbackController``
- future optimization construction through an injectable ``problem_builder``

It does not build the CVXPY problem directly.  The CVXPY variables,
constraints, and objective should live in a dedicated problem builder module
such as ``ccmpc/controllers/problem_builder.py``.

Canonical contracts
-------------------
State9:
    [x, y, z, vx, vy, vz, roll, pitch, yaw]

Goal3:
    [x_goal, y_goal, z_goal]

ControlCommand4:
    [phi_c, theta_c, vz_c, psi_dot_c]

Trajectory convention inside this refactored controller:
    predicted_states:   time-major shape (N + 1, 9)
    predicted_controls: time-major shape (N, 4)

For legacy tuple-unpacking compatibility, ``CCMPCSolveResult.__iter__`` yields:
    x_traj: state-major shape (9, N + 1)
    u_traj: control-major shape (4, N)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ccmpc.controllers.fallback_controller import (
    FallbackController,
    FallbackMode,
    FallbackResult,
)
from ccmpc.controllers.solver_adapter import (
    SolverAdapter,
    SolverConfig,
    SolverResult,
)
from ccmpc.types import (
    CONTROL_DIM,
    STATE_DIM,
    FloatArray,
    as_control_command4,
    as_control_trajectory4,
    as_gamma9x9,
    as_goal3,
    as_state9,
    as_trajectory9,
)


class CCMPCError(RuntimeError):
    """Base exception raised by CCMPC controller."""


class CCMPCConfigError(CCMPCError):
    """Raised when CCMPC configuration is invalid."""


class CCMPCInputError(CCMPCError):
    """Raised when solve inputs violate canonical contracts."""


class CCMPCOutputError(CCMPCError):
    """Raised when a builder/solver returns an invalid output contract."""


@dataclass(frozen=True)
class CCMPCConfig:
    """Configuration for the orchestrator-level CC-MPC controller."""

    dt: float = 0.1
    horizon: int = 15
    name: str = "ccmpc"
    fallback_mode: FallbackMode | str = FallbackMode.HOVER
    use_fallback_on_failure: bool = True
    require_problem_builder: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "dt", _as_positive_float(self.dt, "dt"))
        object.__setattr__(self, "horizon", _as_positive_int(self.horizon, "horizon"))

        if not isinstance(self.name, str) or not self.name.strip():
            raise CCMPCConfigError("name must be a non-empty string.")

        if isinstance(self.fallback_mode, str):
            object.__setattr__(self, "fallback_mode", FallbackMode(self.fallback_mode))

        if not isinstance(self.fallback_mode, FallbackMode):
            raise CCMPCConfigError("fallback_mode must be FallbackMode or str.")

        if not isinstance(self.use_fallback_on_failure, bool):
            raise CCMPCConfigError("use_fallback_on_failure must be bool.")

        if not isinstance(self.require_problem_builder, bool):
            raise CCMPCConfigError("require_problem_builder must be bool.")

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "CCMPCConfig":
        """Create config from direct or full project config."""
        if config is None:
            return cls()

        if not isinstance(config, Mapping):
            raise CCMPCConfigError("config must be a mapping or None.")

        if "controller" in config:
            controller_cfg = config.get("controller")
            if not isinstance(controller_cfg, Mapping):
                raise CCMPCConfigError("config['controller'] must be a mapping.")

            prediction_cfg = controller_cfg.get("prediction", {})
            if prediction_cfg is None:
                prediction_cfg = {}
            if not isinstance(prediction_cfg, Mapping):
                raise CCMPCConfigError(
                    "config['controller']['prediction'] must be a mapping."
                )

            fallback_cfg = controller_cfg.get("fallback", {})
            if fallback_cfg is None:
                fallback_cfg = {}
            if not isinstance(fallback_cfg, Mapping):
                raise CCMPCConfigError(
                    "config['controller']['fallback'] must be a mapping."
                )

            dt = prediction_cfg.get("timestep", prediction_cfg.get("dt", 0.1))
            horizon = _resolve_horizon(prediction_cfg, dt)

            return cls(
                dt=dt,
                horizon=horizon,
                name=str(controller_cfg.get("name", "ccmpc")),
                fallback_mode=fallback_cfg.get("mode", FallbackMode.HOVER),
                use_fallback_on_failure=bool(
                    controller_cfg.get("use_fallback_on_failure", True)
                ),
                require_problem_builder=bool(
                    controller_cfg.get("require_problem_builder", False)
                ),
            )

        dt = config.get("timestep", config.get("dt", 0.1))
        horizon = _resolve_horizon(config, dt)

        return cls(
            dt=dt,
            horizon=horizon,
            name=str(config.get("name", "ccmpc")),
            fallback_mode=config.get("fallback_mode", config.get("mode", FallbackMode.HOVER)),
            use_fallback_on_failure=bool(config.get("use_fallback_on_failure", True)),
            require_problem_builder=bool(config.get("require_problem_builder", False)),
        )


@dataclass(frozen=True)
class MPCProblemBundle:
    """Bundle returned by an external problem builder.

    Attributes
    ----------
    problem:
        CVXPY-like problem exposing ``solve(**kwargs)``.
    command:
        Optional command object/array.  If this is a CVXPY variable/expression,
        it should expose ``.value`` after solve.
    predicted_states:
        Optional State9 trajectory.  Accepted as time-major (N+1,9) or
        state-major (9,N+1); conversion happens in the controller.
    predicted_controls:
        Optional ControlCommand4 trajectory.  Accepted as time-major (N,4) or
        control-major (4,N); conversion happens in the controller.
    metadata:
        Builder diagnostics.
    """

    problem: Any
    command: Any | None = None
    predicted_states: Any | None = None
    predicted_controls: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.problem is None:
            raise CCMPCOutputError("MPCProblemBundle.problem must not be None.")

        if not isinstance(self.metadata, dict):
            raise CCMPCOutputError("MPCProblemBundle.metadata must be a dictionary.")

        object.__setattr__(self, "metadata", dict(self.metadata))


@runtime_checkable
class MPCProblemBuilderProtocol(Protocol):
    """Structural protocol for future CC-MPC problem builders."""

    def build(
        self,
        *,
        estimated_state: FloatArray,
        goal: FloatArray,
        obstacles: tuple[Any, ...],
        covariance: FloatArray | None,
        previous_solution: "CCMPCSolveResult | None",
        reference_trajectory: FloatArray | None,
        metadata: dict[str, Any],
    ) -> MPCProblemBundle:
        """Build and return an MPCProblemBundle."""


@dataclass(frozen=True)
class CCMPCSolveInfo:
    """Compact diagnostics compatible with the legacy controller."""

    status: str
    success: bool
    solve_time_ms: float
    iterations: int | None
    objective_value: float | None
    fallback_used: bool
    fallback_reason: str | None = None
    solver_name: str | None = None
    converged: bool = False
    max_deviation: float | None = None


@dataclass(frozen=True)
class CCMPCSolveResult:
    """Structured result returned by CCMPCController.solve()."""

    command: FloatArray
    success: bool
    status: str
    used_fallback: bool
    solve_time_ms: float
    predicted_states: FloatArray
    predicted_controls: FloatArray
    solver_result: SolverResult | None = None
    fallback_result: FallbackResult | None = None
    objective_value: float | None = None
    iterations: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "command", as_control_command4(self.command))

        if not isinstance(self.success, bool):
            raise CCMPCOutputError("success must be bool.")

        if not isinstance(self.status, str) or not self.status:
            raise CCMPCOutputError("status must be a non-empty string.")

        if not isinstance(self.used_fallback, bool):
            raise CCMPCOutputError("used_fallback must be bool.")

        object.__setattr__(
            self,
            "solve_time_ms",
            _as_non_negative_float(self.solve_time_ms, "solve_time_ms"),
        )

        object.__setattr__(
            self,
            "predicted_states",
            as_trajectory9(self.predicted_states, layout="time_major"),
        )
        object.__setattr__(
            self,
            "predicted_controls",
            as_control_trajectory4(self.predicted_controls, layout="time_major"),
        )

        if self.objective_value is not None:
            object.__setattr__(
                self,
                "objective_value",
                _as_finite_float(self.objective_value, "objective_value"),
            )

        if self.iterations is not None:
            object.__setattr__(
                self,
                "iterations",
                _as_non_negative_int(self.iterations, "iterations"),
            )

        if not isinstance(self.metadata, dict):
            raise CCMPCOutputError("metadata must be a dictionary.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def info(self) -> CCMPCSolveInfo:
        """Return compact solve diagnostics."""
        return CCMPCSolveInfo(
            status=self.status,
            success=self.success,
            solve_time_ms=self.solve_time_ms,
            iterations=self.iterations,
            objective_value=self.objective_value,
            fallback_used=self.used_fallback,
            fallback_reason=(
                None if self.fallback_result is None else self.fallback_result.reason
            ),
            solver_name=None if self.solver_result is None else self.solver_result.solver_name,
            converged=self.success and not self.used_fallback,
        )

    @property
    def x_traj_state_major(self) -> FloatArray:
        """Legacy state trajectory shape (9, N+1)."""
        return self.predicted_states.T.copy()

    @property
    def u_traj_control_major(self) -> FloatArray:
        """Legacy control trajectory shape (4, N)."""
        return self.predicted_controls.T.copy()

    def __iter__(self):
        """Allow legacy tuple unpacking: x_traj, u_traj = result."""
        yield self.x_traj_state_major
        yield self.u_traj_control_major


class CCMPCController:
    """Thin orchestrator for chance-constrained MPC control."""

    def __init__(
        self,
        config: str | Path | Mapping[str, Any] | None = None,
        *,
        problem_builder: MPCProblemBuilderProtocol | None = None,
        solver_adapter: SolverAdapter | None = None,
        fallback_controller: FallbackController | None = None,
        horizon: int | None = None,
        timestep: float | None = None,
    ) -> None:
        config_data = load_controller_config(config)
        ccmpc_config = CCMPCConfig.from_config(config_data)

        if horizon is not None:
            ccmpc_config = CCMPCConfig(
                dt=ccmpc_config.dt,
                horizon=horizon,
                name=ccmpc_config.name,
                fallback_mode=ccmpc_config.fallback_mode,
                use_fallback_on_failure=ccmpc_config.use_fallback_on_failure,
                require_problem_builder=ccmpc_config.require_problem_builder,
            )

        if timestep is not None:
            ccmpc_config = CCMPCConfig(
                dt=timestep,
                horizon=ccmpc_config.horizon,
                name=ccmpc_config.name,
                fallback_mode=ccmpc_config.fallback_mode,
                use_fallback_on_failure=ccmpc_config.use_fallback_on_failure,
                require_problem_builder=ccmpc_config.require_problem_builder,
            )

        self.config = ccmpc_config
        self._config_data = config_data
        self.problem_builder = problem_builder
        self.solver_adapter = (
            SolverAdapter.from_config(config_data)
            if solver_adapter is None
            else solver_adapter
        )
        self.fallback_controller = (
            FallbackController.from_config(config_data)
            if fallback_controller is None
            else fallback_controller
        )

        if not isinstance(self.solver_adapter, SolverAdapter):
            raise CCMPCConfigError("solver_adapter must be SolverAdapter.")

        if not isinstance(self.fallback_controller, FallbackController):
            raise CCMPCConfigError("fallback_controller must be FallbackController.")

        self._previous_result: CCMPCSolveResult | None = None
        self._last_info: CCMPCSolveInfo | None = None

    @classmethod
    def from_config(
        cls,
        config: str | Path | Mapping[str, Any] | None,
        *,
        problem_builder: MPCProblemBuilderProtocol | None = None,
    ) -> "CCMPCController":
        """Construct controller from config."""
        return cls(config, problem_builder=problem_builder)

    @property
    def dt(self) -> float:
        """Controller/MPC timestep."""
        return self.config.dt

    @property
    def horizon(self) -> int:
        """Prediction horizon length."""
        return self.config.horizon

    @property
    def last_info(self) -> CCMPCSolveInfo | None:
        """Diagnostics from latest solve call."""
        return self._last_info

    @property
    def previous_result(self) -> CCMPCSolveResult | None:
        """Structured result from latest solve call."""
        return self._previous_result

    def reset(self) -> None:
        """Clear warm-start and diagnostics state."""
        self._previous_result = None
        self._last_info = None

    def solve(
        self,
        estimated_state: FloatArray | None = None,
        goal: FloatArray | None = None,
        *,
        initial_state: FloatArray | None = None,
        obstacles: Sequence[Any] | None = None,
        obstacle_manager: Any | None = None,
        covariance: FloatArray | None = None,
        Gamma_0: FloatArray | None = None,
        reference_trajectory: FloatArray | None = None,
        time_s: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> CCMPCSolveResult:
        """Solve one receding-horizon controller step.

        Parameters
        ----------
        estimated_state / initial_state:
            Current canonical State9.  ``initial_state`` is accepted for legacy
            compatibility.
        goal:
            Goal3.
        obstacles / obstacle_manager:
            Optional obstacle context passed through to the problem builder.
        covariance / Gamma_0:
            Optional Gamma9x9 state covariance.  ``Gamma_0`` is accepted for
            legacy compatibility.
        reference_trajectory:
            Optional time-major State9 reference trajectory.
        time_s:
            Controller timestamp for diagnostics.
        metadata:
            Additional context passed through to the builder and result.
        """
        call_start = time.perf_counter()

        state = _resolve_state(estimated_state, initial_state)
        target = _resolve_goal(goal)
        gamma = _resolve_covariance(covariance, Gamma_0)
        obstacle_tuple = _resolve_obstacles(obstacles, obstacle_manager)
        ref = _resolve_reference_trajectory(reference_trajectory)

        meta = dict(metadata or {})
        meta["time_s"] = _as_non_negative_float(time_s, "time_s")

        if self.problem_builder is None:
            if self.config.require_problem_builder:
                raise CCMPCConfigError(
                    "CCMPCController requires a problem_builder, but none was provided."
                )

            return self._fallback(
                state=state,
                goal=target,
                reason="problem_builder_missing",
                start_time=call_start,
                metadata={
                    **meta,
                    "obstacle_count": len(obstacle_tuple),
                },
            )

        try:
            bundle = self._build_problem_bundle(
                estimated_state=state,
                goal=target,
                obstacles=obstacle_tuple,
                covariance=gamma,
                reference_trajectory=ref,
                metadata=meta,
            )
        except Exception as exc:
            if not self.config.use_fallback_on_failure:
                raise CCMPCOutputError(str(exc)) from exc

            return self._fallback(
                state=state,
                goal=target,
                reason="problem_builder_failed",
                start_time=call_start,
                metadata={
                    **meta,
                    "exception_type": type(exc).__name__,
                    "error_message": str(exc),
                    "obstacle_count": len(obstacle_tuple),
                },
            )

        solver_result = self.solver_adapter.solve(bundle.problem)

        if not solver_result.success:
            if not self.config.use_fallback_on_failure:
                solver_result.raise_for_failure()

            return self._fallback(
                state=state,
                goal=target,
                reason=f"solver_{solver_result.status}",
                start_time=call_start,
                solver_result=solver_result,
                metadata={
                    **meta,
                    **bundle.metadata,
                    "obstacle_count": len(obstacle_tuple),
                    "solver_error_message": solver_result.error_message,
                },
            )

        try:
            result = self._success_from_bundle(
                state=state,
                bundle=bundle,
                solver_result=solver_result,
                start_time=call_start,
                metadata={
                    **meta,
                    **bundle.metadata,
                    "obstacle_count": len(obstacle_tuple),
                },
            )
        except Exception as exc:
            if not self.config.use_fallback_on_failure:
                raise CCMPCOutputError(str(exc)) from exc

            return self._fallback(
                state=state,
                goal=target,
                reason="solution_extraction_failed",
                start_time=call_start,
                solver_result=solver_result,
                metadata={
                    **meta,
                    **bundle.metadata,
                    "exception_type": type(exc).__name__,
                    "error_message": str(exc),
                    "obstacle_count": len(obstacle_tuple),
                },
            )

        self._store_result(result)
        return result

    def compute_command(
        self,
        estimated_state: FloatArray,
        goal: FloatArray,
        **kwargs: Any,
    ) -> FloatArray:
        """Convenience wrapper returning only ControlCommand4."""
        return self.solve(estimated_state, goal, **kwargs).command

    def _build_problem_bundle(
        self,
        *,
        estimated_state: FloatArray,
        goal: FloatArray,
        obstacles: tuple[Any, ...],
        covariance: FloatArray | None,
        reference_trajectory: FloatArray | None,
        metadata: dict[str, Any],
    ) -> MPCProblemBundle:
        """Call external problem builder and normalize its return type."""
        assert self.problem_builder is not None

        raw = self.problem_builder.build(
            estimated_state=estimated_state,
            goal=goal,
            obstacles=obstacles,
            covariance=covariance,
            previous_solution=self._previous_result,
            reference_trajectory=reference_trajectory,
            metadata=metadata,
        )

        if isinstance(raw, MPCProblemBundle):
            return raw

        # Convenience: allow a builder to return only a CVXPY-like problem.
        if hasattr(raw, "solve") and callable(raw.solve):
            return MPCProblemBundle(problem=raw)

        raise CCMPCOutputError(
            "problem_builder.build() must return MPCProblemBundle or a problem "
            "object exposing solve()."
        )

    def _success_from_bundle(
        self,
        *,
        state: FloatArray,
        bundle: MPCProblemBundle,
        solver_result: SolverResult,
        start_time: float,
        metadata: dict[str, Any],
    ) -> CCMPCSolveResult:
        """Extract command/trajectories after successful solve."""
        predicted_controls = _extract_control_trajectory(bundle.predicted_controls)
        predicted_states = _extract_state_trajectory(bundle.predicted_states)

        if bundle.command is not None:
            command = as_control_command4(_extract_value(bundle.command))
        elif predicted_controls is not None:
            command = as_control_command4(predicted_controls[0, :])
        else:
            raise CCMPCOutputError(
                "Solved bundle must provide command or predicted_controls."
            )

        if predicted_controls is None:
            predicted_controls = _hold_control_trajectory(command, self.horizon)

        if predicted_states is None:
            predicted_states = _hold_state_trajectory(state, self.horizon)

        result = CCMPCSolveResult(
            command=command,
            success=True,
            status=solver_result.status,
            used_fallback=False,
            solve_time_ms=(time.perf_counter() - start_time) * 1000.0,
            predicted_states=predicted_states,
            predicted_controls=predicted_controls,
            solver_result=solver_result,
            fallback_result=None,
            objective_value=solver_result.objective_value,
            iterations=solver_result.iterations,
            metadata=metadata,
        )

        return result

    def _fallback(
        self,
        *,
        state: FloatArray,
        goal: FloatArray,
        reason: str,
        start_time: float,
        solver_result: SolverResult | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CCMPCSolveResult:
        """Compute and store fallback result."""
        fallback_result = self.fallback_controller.compute(
            state,
            goal,
            mode=self.config.fallback_mode,
            reason=reason,
            metadata={} if metadata is None else dict(metadata),
        )

        command = fallback_result.command
        predicted_states = _hold_state_trajectory(state, self.horizon)
        predicted_controls = _hold_control_trajectory(command, self.horizon)

        result = CCMPCSolveResult(
            command=command,
            success=False,
            status="fallback",
            used_fallback=True,
            solve_time_ms=(time.perf_counter() - start_time) * 1000.0,
            predicted_states=predicted_states,
            predicted_controls=predicted_controls,
            solver_result=solver_result,
            fallback_result=fallback_result,
            objective_value=None if solver_result is None else solver_result.objective_value,
            iterations=None if solver_result is None else solver_result.iterations,
            metadata={} if metadata is None else dict(metadata),
        )

        self._store_result(result)
        return result

    def _store_result(self, result: CCMPCSolveResult) -> None:
        """Store latest result and compact diagnostics."""
        self._previous_result = result
        self._last_info = result.info


def load_controller_config(
    config: str | Path | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Load controller config from dict/path/None."""
    if config is None:
        return {}

    if isinstance(config, Mapping):
        return dict(config)

    path = Path(config)
    if not path.exists():
        raise CCMPCConfigError(f"Config file not found: {path}")

    try:
        import yaml  # type: ignore[import-not-found]
    except Exception as exc:
        raise CCMPCConfigError("PyYAML is required to load YAML config files.") from exc

    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)

    if loaded is None:
        return {}

    if not isinstance(loaded, Mapping):
        raise CCMPCConfigError("Loaded controller config must be a mapping.")

    return dict(loaded)


def _resolve_horizon(config: Mapping[str, Any], dt: float) -> int:
    """Resolve horizon from horizon/horizon_steps or horizon_time/dt."""
    if "horizon" in config:
        return _as_positive_int(config["horizon"], "horizon")

    if "horizon_steps" in config:
        return _as_positive_int(config["horizon_steps"], "horizon_steps")

    if "N" in config:
        return _as_positive_int(config["N"], "N")

    if "horizon_time" in config:
        dt_value = _as_positive_float(dt, "dt")
        horizon_time = _as_positive_float(config["horizon_time"], "horizon_time")
        return max(1, int(round(horizon_time / dt_value)))

    return 15


def _resolve_state(
    estimated_state: FloatArray | None,
    initial_state: FloatArray | None,
) -> FloatArray:
    """Resolve current state from new or legacy argument names."""
    if estimated_state is None and initial_state is None:
        raise CCMPCInputError("estimated_state or initial_state must be provided.")

    if estimated_state is not None and initial_state is not None:
        a = as_state9(estimated_state)
        b = as_state9(initial_state)
        if not np.allclose(a, b):
            raise CCMPCInputError(
                "estimated_state and initial_state were both provided but differ."
            )
        return a

    return as_state9(estimated_state if estimated_state is not None else initial_state)


def _resolve_goal(goal: FloatArray | None) -> FloatArray:
    """Resolve required Goal3."""
    if goal is None:
        raise CCMPCInputError("goal must be provided.")

    return as_goal3(goal)


def _resolve_covariance(
    covariance: FloatArray | None,
    gamma_0: FloatArray | None,
) -> FloatArray | None:
    """Resolve optional Gamma9x9 from new or legacy argument names."""
    if covariance is None and gamma_0 is None:
        return None

    if covariance is not None and gamma_0 is not None:
        a = as_gamma9x9(covariance)
        b = as_gamma9x9(gamma_0)
        if not np.allclose(a, b):
            raise CCMPCInputError(
                "covariance and Gamma_0 were both provided but differ."
            )
        return a

    return as_gamma9x9(covariance if covariance is not None else gamma_0)


def _resolve_reference_trajectory(value: FloatArray | None) -> FloatArray | None:
    """Validate optional time-major reference trajectory."""
    if value is None:
        return None
    return as_trajectory9(value, layout="time_major")


def _resolve_obstacles(
    obstacles: Sequence[Any] | None,
    obstacle_manager: Any | None,
) -> tuple[Any, ...]:
    """Resolve obstacle context to an immutable tuple."""
    if obstacles is not None and obstacle_manager is not None:
        raise CCMPCInputError("Provide either obstacles or obstacle_manager, not both.")

    if obstacles is not None:
        if isinstance(obstacles, tuple):
            return obstacles
        if isinstance(obstacles, Sequence):
            return tuple(obstacles)
        raise CCMPCInputError("obstacles must be a sequence.")

    if obstacle_manager is None:
        return ()

    if hasattr(obstacle_manager, "active_obstacles"):
        active = obstacle_manager.active_obstacles
        if callable(active):
            return tuple(active())
        return tuple(active)

    if hasattr(obstacle_manager, "obstacles"):
        return tuple(obstacle_manager.obstacles)

    if isinstance(obstacle_manager, Sequence):
        return tuple(obstacle_manager)

    raise CCMPCInputError(
        "obstacle_manager must expose active_obstacles, obstacles, or be a sequence."
    )


def _extract_value(value: Any) -> FloatArray:
    """Extract NumPy value from array-like or CVXPY-like object."""
    if hasattr(value, "value"):
        value = value.value

    if value is None:
        raise CCMPCOutputError("Expected solved value, got None.")

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CCMPCOutputError("Solved value must be convertible to float64 array.") from exc

    if not np.all(np.isfinite(array)):
        raise CCMPCOutputError("Solved value must contain only finite numbers.")

    return array.copy()


def _extract_state_trajectory(value: Any | None) -> FloatArray | None:
    """Extract optional State9 trajectory and convert to time-major."""
    if value is None:
        return None

    array = _extract_value(value)

    if array.ndim != 2:
        raise CCMPCOutputError(
            f"predicted_states must be 2D, got ndim={array.ndim}."
        )

    if array.shape[1] == STATE_DIM:
        return as_trajectory9(array, layout="time_major")

    if array.shape[0] == STATE_DIM:
        return as_trajectory9(array.T, layout="time_major")

    raise CCMPCOutputError(
        "predicted_states must have shape (T,9) or (9,T), "
        f"got {array.shape}."
    )


def _extract_control_trajectory(value: Any | None) -> FloatArray | None:
    """Extract optional ControlCommand4 trajectory and convert to time-major."""
    if value is None:
        return None

    array = _extract_value(value)

    if array.ndim != 2:
        raise CCMPCOutputError(
            f"predicted_controls must be 2D, got ndim={array.ndim}."
        )

    if array.shape[1] == CONTROL_DIM:
        return as_control_trajectory4(array, layout="time_major")

    if array.shape[0] == CONTROL_DIM:
        return as_control_trajectory4(array.T, layout="time_major")

    raise CCMPCOutputError(
        "predicted_controls must have shape (T,4) or (4,T), "
        f"got {array.shape}."
    )


def _hold_state_trajectory(state: FloatArray, horizon: int) -> FloatArray:
    """Create a simple hold-state trajectory for fallback/partial results."""
    x = as_state9(state)
    steps = _as_positive_int(horizon, "horizon") + 1
    return np.tile(x.reshape(1, STATE_DIM), (steps, 1)).astype(np.float64)


def _hold_control_trajectory(command: FloatArray, horizon: int) -> FloatArray:
    """Create a repeated command trajectory for fallback/partial results."""
    u = as_control_command4(command)
    steps = _as_positive_int(horizon, "horizon")
    return np.tile(u.reshape(1, CONTROL_DIM), (steps, 1)).astype(np.float64)


def _as_positive_float(value: Any, name: str) -> float:
    """Validate finite float > 0."""
    if isinstance(value, bool):
        raise CCMPCConfigError(f"{name} must be finite and > 0, got bool.")

    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise CCMPCConfigError(f"{name} must be finite and > 0.") from exc

    if not np.isfinite(scalar) or scalar <= 0.0:
        raise CCMPCConfigError(f"{name} must be finite and > 0.")

    return scalar


def _as_non_negative_float(value: Any, name: str) -> float:
    """Validate finite float >= 0."""
    if isinstance(value, bool):
        raise CCMPCConfigError(f"{name} must be finite and >= 0, got bool.")

    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise CCMPCConfigError(f"{name} must be finite and >= 0.") from exc

    if not np.isfinite(scalar) or scalar < 0.0:
        raise CCMPCConfigError(f"{name} must be finite and >= 0.")

    return scalar


def _as_finite_float(value: Any, name: str) -> float:
    """Validate finite float."""
    if isinstance(value, bool):
        raise CCMPCOutputError(f"{name} must be finite, got bool.")

    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise CCMPCOutputError(f"{name} must be finite.") from exc

    if not np.isfinite(scalar):
        raise CCMPCOutputError(f"{name} must be finite.")

    return scalar


def _as_positive_int(value: Any, name: str) -> int:
    """Validate integer > 0."""
    if isinstance(value, bool):
        raise CCMPCConfigError(f"{name} must be an integer > 0, got bool.")

    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise CCMPCConfigError(f"{name} must be an integer > 0.") from exc

    if integer <= 0:
        raise CCMPCConfigError(f"{name} must be an integer > 0.")

    return integer


def _as_non_negative_int(value: Any, name: str) -> int:
    """Validate integer >= 0."""
    if isinstance(value, bool):
        raise CCMPCOutputError(f"{name} must be an integer >= 0, got bool.")

    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise CCMPCOutputError(f"{name} must be an integer >= 0.") from exc

    if integer < 0:
        raise CCMPCOutputError(f"{name} must be an integer >= 0.")

    return integer


CCMPC = CCMPCController


__all__ = [
    "CCMPC",
    "CCMPCConfig",
    "CCMPCConfigError",
    "CCMPCController",
    "CCMPCError",
    "CCMPCInputError",
    "CCMPCOutputError",
    "CCMPCSolveInfo",
    "CCMPCSolveResult",
    "MPCProblemBuilderProtocol",
    "MPCProblemBundle",
    "load_controller_config",
]
