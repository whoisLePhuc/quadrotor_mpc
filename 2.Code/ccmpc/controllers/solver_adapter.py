"""Solver adapter for CC-MPC controller optimization problems.

The goal of this module is to isolate solver-specific behavior from the core
CCMPCController.

Design
------
- The adapter is intentionally small and unit-testable.
- It does not build CVXPY problems.
- It only calls ``problem.solve(...)`` and normalizes the result.
- Unit tests can use a FakeProblem without importing CVXPY.
- When CVXPY is installed, ``resolve_cvxpy_solver()`` maps solver names to
  CVXPY solver constants when available.

Typical usage
-------------
    adapter = SolverAdapter.from_config(config)
    result = adapter.solve(problem)

    if result.success:
        ...
    else:
        fallback.compute(...)

Supported config shapes
-----------------------
Direct solver config:

    {
        "solver": "CLARABEL",
        "verbose": false,
        "warm_start": true,
        "max_iter": 50,
        "tolerance": 1e-4,
        "solver_options": {...}
    }

Full controller config:

    {
        "controller": {
            "prediction": {
                "solver": "CLARABEL",
                "max_iter": 50,
                "tolerance": 1e-4
            },
            "solver": {
                "verbose": false,
                "warm_start": true,
                "solver_options": {...}
            }
        }
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any

import numpy as np


class SolverError(RuntimeError):
    """Base exception raised by solver adapter."""


class SolverConfigError(SolverError):
    """Raised when solver configuration is invalid."""


class SolverRuntimeError(SolverError):
    """Raised when solver execution fails and caller requested strict mode."""


class SolverStatus(str, Enum):
    """Adapter-level normalized solve status."""

    SUCCESS = "success"
    FAILURE = "failure"
    EXCEPTION = "exception"


SUCCESS_STATUSES = frozenset(
    {
        "optimal",
        "optimal_inaccurate",
    }
)

FAILURE_STATUSES = frozenset(
    {
        "infeasible",
        "infeasible_inaccurate",
        "unbounded",
        "unbounded_inaccurate",
        "infeasible_or_unbounded",
        "solver_error",
        "user_limit",
        "unknown",
        "none",
    }
)


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for optimization solver calls."""

    solver: str = "CLARABEL"
    verbose: bool = False
    warm_start: bool = True
    max_iter: int | None = None
    tolerance: float | None = None
    solver_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate config."""
        object.__setattr__(self, "solver", normalize_solver_name(self.solver))

        if not isinstance(self.verbose, bool):
            raise SolverConfigError("verbose must be bool.")

        if not isinstance(self.warm_start, bool):
            raise SolverConfigError("warm_start must be bool.")

        if self.max_iter is not None:
            object.__setattr__(self, "max_iter", _as_positive_int(self.max_iter, "max_iter"))

        if self.tolerance is not None:
            object.__setattr__(self, "tolerance", _as_positive_float(self.tolerance, "tolerance"))

        if not isinstance(self.solver_options, dict):
            raise SolverConfigError("solver_options must be a dictionary.")
        object.__setattr__(self, "solver_options", dict(self.solver_options))

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "SolverConfig":
        """Create SolverConfig from direct or full project config."""
        if config is None:
            return cls()

        if not isinstance(config, dict):
            raise SolverConfigError("config must be a dictionary or None.")

        if "controller" in config:
            controller_cfg = config.get("controller")
            if not isinstance(controller_cfg, dict):
                raise SolverConfigError("config['controller'] must be a dictionary.")

            prediction_cfg = controller_cfg.get("prediction", {})
            if prediction_cfg is None:
                prediction_cfg = {}
            if not isinstance(prediction_cfg, dict):
                raise SolverConfigError(
                    "config['controller']['prediction'] must be a dictionary."
                )

            solver_cfg = controller_cfg.get("solver", {})
            if solver_cfg is None:
                solver_cfg = {}
            if not isinstance(solver_cfg, dict):
                raise SolverConfigError(
                    "config['controller']['solver'] must be a dictionary."
                )

            solver_options = _merged_options(
                solver_cfg.get("options"),
                solver_cfg.get("solver_options"),
                prediction_cfg.get("solver_options"),
            )

            return cls(
                solver=solver_cfg.get(
                    "solver",
                    prediction_cfg.get("solver", "CLARABEL"),
                ),
                verbose=bool(solver_cfg.get("verbose", prediction_cfg.get("verbose", False))),
                warm_start=bool(solver_cfg.get("warm_start", True)),
                max_iter=_first_present(
                    solver_cfg,
                    prediction_cfg,
                    keys=("max_iter", "max_iters"),
                ),
                tolerance=_first_present(
                    solver_cfg,
                    prediction_cfg,
                    keys=("tolerance", "tol", "eps"),
                ),
                solver_options=solver_options,
            )

        solver_options = _merged_options(
            config.get("options"),
            config.get("solver_options"),
        )

        return cls(
            solver=config.get("solver", "CLARABEL"),
            verbose=bool(config.get("verbose", False)),
            warm_start=bool(config.get("warm_start", True)),
            max_iter=config.get("max_iter", config.get("max_iters")),
            tolerance=config.get("tolerance", config.get("tol", config.get("eps"))),
            solver_options=solver_options,
        )

    def default_options(self) -> dict[str, Any]:
        """Return default solver options for this config."""
        options = default_solver_options(
            self.solver,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
        )
        options.update(self.solver_options)
        return options


@dataclass(frozen=True)
class SolverResult:
    """Normalized result returned by SolverAdapter."""

    success: bool
    status: str
    solve_time_ms: float
    objective_value: float | None
    solver_name: str
    adapter_status: SolverStatus
    error_message: str | None = None
    exception_type: str | None = None
    iterations: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result."""
        if not isinstance(self.success, bool):
            raise SolverConfigError("success must be bool.")

        if not isinstance(self.status, str) or not self.status:
            raise SolverConfigError("status must be a non-empty string.")

        object.__setattr__(
            self,
            "solve_time_ms",
            _as_non_negative_float(self.solve_time_ms, "solve_time_ms"),
        )

        if self.objective_value is not None:
            object.__setattr__(
                self,
                "objective_value",
                _as_optional_finite_float(self.objective_value, "objective_value"),
            )

        if not isinstance(self.solver_name, str) or not self.solver_name:
            raise SolverConfigError("solver_name must be a non-empty string.")

        if isinstance(self.adapter_status, str):
            object.__setattr__(self, "adapter_status", SolverStatus(self.adapter_status))

        if self.error_message is not None and not isinstance(self.error_message, str):
            raise SolverConfigError("error_message must be str or None.")

        if self.exception_type is not None and not isinstance(self.exception_type, str):
            raise SolverConfigError("exception_type must be str or None.")

        if self.iterations is not None:
            object.__setattr__(
                self,
                "iterations",
                _as_non_negative_int(self.iterations, "iterations"),
            )

        if not isinstance(self.extra, dict):
            raise SolverConfigError("extra must be a dictionary.")
        object.__setattr__(self, "extra", dict(self.extra))

    def raise_for_failure(self) -> None:
        """Raise SolverRuntimeError if result is not successful."""
        if self.success:
            return

        message = self.error_message or f"Solver failed with status={self.status!r}."
        raise SolverRuntimeError(message)


class SolverAdapter:
    """Small adapter around CVXPY-like problem.solve()."""

    def __init__(self, config: SolverConfig | None = None) -> None:
        self.config = SolverConfig() if config is None else config

        if not isinstance(self.config, SolverConfig):
            raise SolverConfigError("SolverAdapter expects SolverConfig or None.")

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "SolverAdapter":
        """Create adapter from direct or full project config."""
        return cls(SolverConfig.from_config(config))

    def solve(
        self,
        problem: Any,
        *,
        solver: str | None = None,
        verbose: bool | None = None,
        warm_start: bool | None = None,
        strict: bool = False,
        **solver_options: Any,
    ) -> SolverResult:
        """Solve a CVXPY-like optimization problem.

        The problem object must expose ``solve(**kwargs)``.  CVXPY's
        ``Problem`` satisfies this contract, and unit tests can use a fake
        object with the same method.

        Parameters
        ----------
        problem:
            Object exposing ``solve``.
        solver:
            Optional solver override.
        verbose:
            Optional verbose override.
        warm_start:
            Optional warm-start override.
        strict:
            If True, exceptions are raised through ``SolverRuntimeError``.
            If False, exceptions are converted into failure SolverResult.
        solver_options:
            Extra options merged after config defaults.
        """
        if not hasattr(problem, "solve") or not callable(problem.solve):
            raise SolverRuntimeError("problem must expose a callable solve() method.")

        solver_name = normalize_solver_name(solver or self.config.solver)
        resolved_solver = resolve_cvxpy_solver(solver_name)
        verbose_value = self.config.verbose if verbose is None else _as_bool(verbose, "verbose")
        warm_start_value = (
            self.config.warm_start if warm_start is None else _as_bool(warm_start, "warm_start")
        )

        options = self.config.default_options()
        options.update(solver_options)

        solve_kwargs = {
            "solver": resolved_solver,
            "verbose": verbose_value,
            "warm_start": warm_start_value,
            **options,
        }

        start = time.perf_counter()

        try:
            raw_objective = problem.solve(**solve_kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            status = normalize_status(getattr(problem, "status", None))
            objective_value = objective_from_problem(problem, raw_objective)
            iterations = iterations_from_problem(problem)

            success = is_success_status(status)
            result = SolverResult(
                success=success,
                status=status,
                solve_time_ms=elapsed_ms,
                objective_value=objective_value,
                solver_name=solver_name,
                adapter_status=SolverStatus.SUCCESS if success else SolverStatus.FAILURE,
                error_message=None if success else f"Solver returned status={status!r}.",
                iterations=iterations,
                extra={
                    "solve_kwargs": solve_kwargs,
                    "raw_objective": raw_objective,
                },
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if strict:
                raise SolverRuntimeError(str(exc)) from exc

            result = SolverResult(
                success=False,
                status=SolverStatus.EXCEPTION.value,
                solve_time_ms=elapsed_ms,
                objective_value=None,
                solver_name=solver_name,
                adapter_status=SolverStatus.EXCEPTION,
                error_message=str(exc),
                exception_type=type(exc).__name__,
                iterations=None,
                extra={
                    "solve_kwargs": solve_kwargs,
                },
            )

        return result


def normalize_solver_name(solver: str) -> str:
    """Normalize solver name to uppercase string."""
    if not isinstance(solver, str) or not solver.strip():
        raise SolverConfigError("solver must be a non-empty string.")

    return solver.strip().upper()


def resolve_cvxpy_solver(solver: str) -> Any:
    """Resolve a solver name to a CVXPY constant when available.

    If CVXPY is not installed or the attribute is absent, returns the normalized
    string.  This keeps the adapter importable in lightweight unit tests.
    """
    solver_name = normalize_solver_name(solver)

    try:
        import cvxpy as cp  # type: ignore[import-not-found]
    except Exception:
        return solver_name

    return getattr(cp, solver_name, solver_name)


def default_solver_options(
    solver: str,
    *,
    max_iter: int | None = None,
    tolerance: float | None = None,
) -> dict[str, Any]:
    """Return solver-specific default keyword options.

    Only includes keys when the corresponding values are provided.  This avoids
    passing unsupported options unnecessarily.
    """
    solver_name = normalize_solver_name(solver)
    options: dict[str, Any] = {}

    if max_iter is not None:
        max_iter_value = _as_positive_int(max_iter, "max_iter")

        if solver_name == "SCS":
            options["max_iters"] = max_iter_value
        elif solver_name == "ECOS":
            options["max_iters"] = max_iter_value
        else:
            options["max_iter"] = max_iter_value

    if tolerance is not None:
        tol = _as_positive_float(tolerance, "tolerance")

        if solver_name == "CLARABEL":
            options["tol_gap_abs"] = tol
            options["tol_gap_rel"] = tol
            options["tol_feas"] = tol
        elif solver_name == "OSQP":
            options["eps_abs"] = tol
            options["eps_rel"] = tol
        elif solver_name == "SCS":
            options["eps"] = tol
        elif solver_name == "ECOS":
            options["abstol"] = tol
            options["reltol"] = tol
            options["feastol"] = tol
        else:
            options["tolerance"] = tol

    return options


def is_success_status(status: str | None) -> bool:
    """Return True for statuses considered successful."""
    return normalize_status(status) in SUCCESS_STATUSES


def normalize_status(status: str | None) -> str:
    """Normalize CVXPY-like problem status."""
    if status is None:
        return "none"

    normalized = str(status).strip().lower()
    if not normalized:
        return "none"

    return normalized


def objective_from_problem(problem: Any, raw_objective: Any) -> float | None:
    """Extract objective value from problem.value or solve() return value."""
    value = getattr(problem, "value", None)

    if value is None:
        value = raw_objective

    if value is None:
        return None

    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value_float):
        return None

    return value_float


def iterations_from_problem(problem: Any) -> int | None:
    """Extract iteration count from CVXPY-like solver_stats."""
    solver_stats = getattr(problem, "solver_stats", None)

    if solver_stats is None:
        return None

    num_iters = getattr(solver_stats, "num_iters", None)

    if num_iters is None:
        return None

    try:
        return _as_non_negative_int(num_iters, "iterations")
    except SolverConfigError:
        return None


def installed_solvers() -> tuple[str, ...]:
    """Return installed CVXPY solvers if CVXPY is available."""
    try:
        import cvxpy as cp  # type: ignore[import-not-found]
    except Exception:
        return ()

    try:
        return tuple(str(name).upper() for name in cp.installed_solvers())
    except Exception:
        return ()


def _merged_options(*items: Any) -> dict[str, Any]:
    """Merge optional dict objects from left to right."""
    merged: dict[str, Any] = {}

    for item in items:
        if item is None:
            continue
        if not isinstance(item, dict):
            raise SolverConfigError("solver options must be dictionaries.")
        merged.update(item)

    return merged


def _first_present(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    *,
    keys: tuple[str, ...],
) -> Any:
    """Return first present key from primary then secondary."""
    for key in keys:
        if key in primary:
            return primary[key]

    for key in keys:
        if key in secondary:
            return secondary[key]

    return None


def _as_bool(value: bool, name: str) -> bool:
    """Validate bool."""
    if not isinstance(value, bool):
        raise SolverConfigError(f"{name} must be bool.")
    return value


def _as_positive_int(value: int, name: str) -> int:
    """Validate integer > 0."""
    if isinstance(value, bool):
        raise SolverConfigError(f"{name} must be a positive integer, got bool.")

    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise SolverConfigError(f"{name} must be a positive integer.") from exc

    if integer <= 0:
        raise SolverConfigError(f"{name} must be > 0.")

    return integer


def _as_non_negative_int(value: int, name: str) -> int:
    """Validate integer >= 0."""
    if isinstance(value, bool):
        raise SolverConfigError(f"{name} must be a non-negative integer, got bool.")

    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise SolverConfigError(f"{name} must be a non-negative integer.") from exc

    if integer < 0:
        raise SolverConfigError(f"{name} must be >= 0.")

    return integer


def _as_positive_float(value: float, name: str) -> float:
    """Validate finite float > 0."""
    if isinstance(value, bool):
        raise SolverConfigError(f"{name} must be a positive scalar, got bool.")

    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise SolverConfigError(f"{name} must be a positive scalar.") from exc

    if not np.isfinite(scalar) or scalar <= 0.0:
        raise SolverConfigError(f"{name} must be finite and > 0.")

    return scalar


def _as_non_negative_float(value: float, name: str) -> float:
    """Validate finite float >= 0."""
    if isinstance(value, bool):
        raise SolverConfigError(f"{name} must be a non-negative scalar, got bool.")

    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise SolverConfigError(f"{name} must be a non-negative scalar.") from exc

    if not np.isfinite(scalar) or scalar < 0.0:
        raise SolverConfigError(f"{name} must be finite and >= 0.")

    return scalar


def _as_optional_finite_float(value: float, name: str) -> float:
    """Validate finite float for optional numeric fields."""
    if isinstance(value, bool):
        raise SolverConfigError(f"{name} must be a finite scalar, got bool.")

    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise SolverConfigError(f"{name} must be a finite scalar.") from exc

    if not np.isfinite(scalar):
        raise SolverConfigError(f"{name} must be finite.")

    return scalar


__all__ = [
    "FAILURE_STATUSES",
    "SUCCESS_STATUSES",
    "SolverAdapter",
    "SolverConfig",
    "SolverConfigError",
    "SolverError",
    "SolverResult",
    "SolverRuntimeError",
    "SolverStatus",
    "default_solver_options",
    "installed_solvers",
    "is_success_status",
    "normalize_solver_name",
    "normalize_status",
    "objective_from_problem",
    "resolve_cvxpy_solver",
]
