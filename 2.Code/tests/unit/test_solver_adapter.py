"""Unit tests for CC-MPC solver adapter.

Target module:
    ccmpc.controllers.solver_adapter

These tests use FakeProblem instead of a real CVXPY problem so the adapter can
be tested independently from optimization model construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ccmpc.controllers.solver_adapter import (
    SolverAdapter,
    SolverConfig,
    SolverConfigError,
    SolverResult,
    SolverRuntimeError,
    SolverStatus,
    default_solver_options,
    installed_solvers,
    is_success_status,
    normalize_solver_name,
    normalize_status,
    objective_from_problem,
    resolve_cvxpy_solver,
)


@dataclass
class FakeSolverStats:
    """Minimal CVXPY-like solver_stats object."""

    num_iters: int | None = None


class FakeProblem:
    """Minimal CVXPY-like optimization problem."""

    def __init__(
        self,
        *,
        status: str = "optimal",
        value: float | None = 1.23,
        raw_return: float | None = None,
        solver_stats: FakeSolverStats | None = None,
    ) -> None:
        self.status = status
        self.value = value
        self.raw_return = value if raw_return is None else raw_return
        self.solver_stats = solver_stats
        self.solve_kwargs: dict[str, Any] | None = None
        self.solve_calls = 0

    def solve(self, **kwargs: Any) -> float | None:
        """Record solve kwargs and return configured objective."""
        self.solve_calls += 1
        self.solve_kwargs = kwargs
        return self.raw_return


class RaisingProblem:
    """Fake problem whose solve method raises."""

    status = None
    value = None
    solver_stats = None

    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = RuntimeError("solver exploded") if exc is None else exc
        self.solve_kwargs: dict[str, Any] | None = None

    def solve(self, **kwargs: Any) -> None:
        self.solve_kwargs = kwargs
        raise self.exc


class NoSolveProblem:
    """Object without solve method."""


def make_full_config() -> dict:
    """Create full controller-style solver config."""
    return {
        "controller": {
            "prediction": {
                "solver": "CLARABEL",
                "max_iter": 50,
                "tolerance": 1e-4,
            },
            "solver": {
                "verbose": True,
                "warm_start": False,
                "solver_options": {
                    "custom_option": 123,
                },
            },
        }
    }


def make_direct_config() -> dict:
    """Create solver-only config."""
    return {
        "solver": "OSQP",
        "verbose": True,
        "warm_start": False,
        "max_iter": 100,
        "tolerance": 1e-5,
        "solver_options": {
            "polish": True,
        },
    }


def test_normalize_solver_name() -> None:
    """normalize_solver_name should strip and uppercase names."""
    assert normalize_solver_name("clarabel") == "CLARABEL"
    assert normalize_solver_name("  osqp  ") == "OSQP"


def test_normalize_solver_name_rejects_empty() -> None:
    """normalize_solver_name should reject empty names."""
    with pytest.raises(SolverConfigError, match="solver"):
        normalize_solver_name("")


def test_resolve_cvxpy_solver_returns_something_for_known_solver() -> None:
    """resolve_cvxpy_solver should return a usable solver identifier."""
    resolved = resolve_cvxpy_solver("CLARABEL")

    # CVXPY constants are commonly strings, so only assert semantic identity.
    assert str(resolved).upper() == "CLARABEL"


def test_resolve_cvxpy_solver_unknown_returns_uppercase_string() -> None:
    """Unknown solver names should return normalized strings."""
    resolved = resolve_cvxpy_solver("my_solver")

    assert resolved == "MY_SOLVER"


def test_default_solver_options_clarabel() -> None:
    """CLARABEL tolerance should map to CLARABEL-specific keys."""
    options = default_solver_options(
        "CLARABEL",
        max_iter=50,
        tolerance=1e-4,
    )

    assert options["max_iter"] == 50
    assert options["tol_gap_abs"] == pytest.approx(1e-4)
    assert options["tol_gap_rel"] == pytest.approx(1e-4)
    assert options["tol_feas"] == pytest.approx(1e-4)


def test_default_solver_options_osqp() -> None:
    """OSQP tolerance should map to eps_abs and eps_rel."""
    options = default_solver_options(
        "OSQP",
        max_iter=100,
        tolerance=1e-5,
    )

    assert options["max_iter"] == 100
    assert options["eps_abs"] == pytest.approx(1e-5)
    assert options["eps_rel"] == pytest.approx(1e-5)


def test_default_solver_options_scs() -> None:
    """SCS should use max_iters and eps."""
    options = default_solver_options(
        "SCS",
        max_iter=200,
        tolerance=1e-3,
    )

    assert options["max_iters"] == 200
    assert options["eps"] == pytest.approx(1e-3)


def test_default_solver_options_ecos() -> None:
    """ECOS should use ECOS-style tolerance keys."""
    options = default_solver_options(
        "ECOS",
        max_iter=30,
        tolerance=1e-6,
    )

    assert options["max_iters"] == 30
    assert options["abstol"] == pytest.approx(1e-6)
    assert options["reltol"] == pytest.approx(1e-6)
    assert options["feastol"] == pytest.approx(1e-6)


def test_default_solver_options_unknown() -> None:
    """Unknown solvers should still receive generic option names."""
    options = default_solver_options(
        "CUSTOM",
        max_iter=10,
        tolerance=1e-2,
    )

    assert options["max_iter"] == 10
    assert options["tolerance"] == pytest.approx(1e-2)


def test_default_solver_options_rejects_invalid_max_iter() -> None:
    """default_solver_options should reject invalid max_iter."""
    with pytest.raises(SolverConfigError, match="max_iter"):
        default_solver_options("CLARABEL", max_iter=0)


def test_default_solver_options_rejects_invalid_tolerance() -> None:
    """default_solver_options should reject invalid tolerance."""
    with pytest.raises(SolverConfigError, match="tolerance"):
        default_solver_options("CLARABEL", tolerance=0.0)


def test_solver_config_defaults() -> None:
    """SolverConfig defaults should be valid."""
    config = SolverConfig()

    assert config.solver == "CLARABEL"
    assert config.verbose is False
    assert config.warm_start is True
    assert config.max_iter is None
    assert config.tolerance is None
    assert config.solver_options == {}


def test_solver_config_from_none() -> None:
    """SolverConfig.from_config(None) should return defaults."""
    assert SolverConfig.from_config(None) == SolverConfig()


def test_solver_config_from_direct_config() -> None:
    """SolverConfig.from_config should parse solver-only config."""
    config = SolverConfig.from_config(make_direct_config())

    assert config.solver == "OSQP"
    assert config.verbose is True
    assert config.warm_start is False
    assert config.max_iter == 100
    assert config.tolerance == pytest.approx(1e-5)
    assert config.solver_options == {"polish": True}


def test_solver_config_from_full_config() -> None:
    """SolverConfig.from_config should parse full controller config."""
    config = SolverConfig.from_config(make_full_config())

    assert config.solver == "CLARABEL"
    assert config.verbose is True
    assert config.warm_start is False
    assert config.max_iter == 50
    assert config.tolerance == pytest.approx(1e-4)
    assert config.solver_options == {"custom_option": 123}


def test_solver_config_solver_options_aliases_merge() -> None:
    """Options aliases should merge left-to-right."""
    config = SolverConfig.from_config(
        {
            "solver": "CLARABEL",
            "options": {"a": 1, "shared": "old"},
            "solver_options": {"b": 2, "shared": "new"},
        }
    )

    assert config.solver_options == {"a": 1, "b": 2, "shared": "new"}


def test_solver_config_default_options_merges_custom_options() -> None:
    """SolverConfig.default_options should include defaults and custom overrides."""
    config = SolverConfig(
        solver="CLARABEL",
        max_iter=50,
        tolerance=1e-4,
        solver_options={"tol_gap_abs": 1e-3, "custom": 7},
    )

    options = config.default_options()

    assert options["max_iter"] == 50
    assert options["tol_gap_abs"] == pytest.approx(1e-3)
    assert options["tol_gap_rel"] == pytest.approx(1e-4)
    assert options["custom"] == 7


def test_solver_config_rejects_bad_type() -> None:
    """SolverConfig.from_config should reject non-dict config."""
    with pytest.raises(SolverConfigError, match="dictionary"):
        SolverConfig.from_config(["bad"])  # type: ignore[arg-type]


def test_solver_config_rejects_invalid_solver_options() -> None:
    """solver_options must be a dictionary."""
    with pytest.raises(SolverConfigError, match="solver_options"):
        SolverConfig(solver_options=["bad"])  # type: ignore[arg-type]


def test_solver_config_rejects_invalid_max_iter() -> None:
    """max_iter must be positive when provided."""
    with pytest.raises(SolverConfigError, match="max_iter"):
        SolverConfig(max_iter=0)


def test_solver_config_rejects_invalid_tolerance() -> None:
    """tolerance must be positive when provided."""
    with pytest.raises(SolverConfigError, match="tolerance"):
        SolverConfig(tolerance=0.0)


def test_solver_result_valid_success() -> None:
    """SolverResult should validate successful result."""
    result = SolverResult(
        success=True,
        status="optimal",
        solve_time_ms=1.0,
        objective_value=2.0,
        solver_name="CLARABEL",
        adapter_status=SolverStatus.SUCCESS,
        iterations=5,
        extra={"a": 1},
    )

    assert result.success is True
    assert result.status == "optimal"
    assert result.objective_value == pytest.approx(2.0)
    assert result.adapter_status is SolverStatus.SUCCESS
    assert result.iterations == 5
    assert result.extra == {"a": 1}


def test_solver_result_accepts_string_adapter_status() -> None:
    """SolverResult should parse adapter_status from string."""
    result = SolverResult(
        success=False,
        status="infeasible",
        solve_time_ms=0.1,
        objective_value=None,
        solver_name="CLARABEL",
        adapter_status="failure",
        error_message="bad status",
    )

    assert result.adapter_status is SolverStatus.FAILURE


def test_solver_result_rejects_negative_solve_time() -> None:
    """SolverResult should reject negative solve time."""
    with pytest.raises(SolverConfigError, match="solve_time_ms"):
        SolverResult(
            success=True,
            status="optimal",
            solve_time_ms=-1.0,
            objective_value=1.0,
            solver_name="CLARABEL",
            adapter_status=SolverStatus.SUCCESS,
        )


def test_solver_result_rejects_nonfinite_objective() -> None:
    """SolverResult should reject non-finite objective_value."""
    with pytest.raises(SolverConfigError, match="objective_value"):
        SolverResult(
            success=True,
            status="optimal",
            solve_time_ms=1.0,
            objective_value=float("inf"),
            solver_name="CLARABEL",
            adapter_status=SolverStatus.SUCCESS,
        )


def test_solver_result_raise_for_failure_success_noop() -> None:
    """raise_for_failure should not raise for successful result."""
    result = SolverResult(
        success=True,
        status="optimal",
        solve_time_ms=1.0,
        objective_value=1.0,
        solver_name="CLARABEL",
        adapter_status=SolverStatus.SUCCESS,
    )

    result.raise_for_failure()


def test_solver_result_raise_for_failure_raises() -> None:
    """raise_for_failure should raise SolverRuntimeError on failure."""
    result = SolverResult(
        success=False,
        status="infeasible",
        solve_time_ms=1.0,
        objective_value=None,
        solver_name="CLARABEL",
        adapter_status=SolverStatus.FAILURE,
        error_message="infeasible problem",
    )

    with pytest.raises(SolverRuntimeError, match="infeasible problem"):
        result.raise_for_failure()


def test_normalize_status() -> None:
    """normalize_status should lower-case and handle None/empty."""
    assert normalize_status("OPTIMAL") == "optimal"
    assert normalize_status(" optimal_inaccurate ") == "optimal_inaccurate"
    assert normalize_status(None) == "none"
    assert normalize_status("") == "none"


def test_is_success_status() -> None:
    """Only optimal statuses should be treated as success."""
    assert is_success_status("optimal") is True
    assert is_success_status("optimal_inaccurate") is True
    assert is_success_status("infeasible") is False
    assert is_success_status(None) is False


def test_objective_from_problem_prefers_problem_value() -> None:
    """objective_from_problem should prefer problem.value over raw return."""
    problem = FakeProblem(value=3.0, raw_return=4.0)

    assert objective_from_problem(problem, raw_objective=4.0) == pytest.approx(3.0)


def test_objective_from_problem_uses_raw_return_when_value_none() -> None:
    """objective_from_problem should use raw objective when problem.value is None."""
    problem = FakeProblem(value=None, raw_return=4.0)

    assert objective_from_problem(problem, raw_objective=4.0) == pytest.approx(4.0)


def test_objective_from_problem_returns_none_for_nonfinite() -> None:
    """objective_from_problem should return None for non-finite values."""
    problem = FakeProblem(value=float("inf"), raw_return=1.0)

    assert objective_from_problem(problem, raw_objective=1.0) is None


def test_installed_solvers_returns_tuple() -> None:
    """installed_solvers should always return a tuple."""
    result = installed_solvers()

    assert isinstance(result, tuple)
    assert all(isinstance(item, str) for item in result)


def test_solver_adapter_default() -> None:
    """SolverAdapter default constructor should use default SolverConfig."""
    adapter = SolverAdapter()

    assert adapter.config == SolverConfig()


def test_solver_adapter_from_config() -> None:
    """SolverAdapter.from_config should parse config."""
    adapter = SolverAdapter.from_config(make_direct_config())

    assert adapter.config.solver == "OSQP"
    assert adapter.config.max_iter == 100


def test_solver_adapter_rejects_invalid_config_object() -> None:
    """SolverAdapter should reject non-SolverConfig object."""
    with pytest.raises(SolverConfigError, match="SolverConfig"):
        SolverAdapter(config=object())  # type: ignore[arg-type]


def test_solve_success_with_fake_problem() -> None:
    """solve should return success result for optimal fake problem."""
    adapter = SolverAdapter(
        SolverConfig(
            solver="CLARABEL",
            verbose=False,
            warm_start=True,
            max_iter=20,
            tolerance=1e-4,
            solver_options={"custom": 1},
        )
    )
    problem = FakeProblem(
        status="optimal",
        value=10.0,
        solver_stats=FakeSolverStats(num_iters=7),
    )

    result = adapter.solve(problem)

    assert result.success is True
    assert result.status == "optimal"
    assert result.adapter_status is SolverStatus.SUCCESS
    assert result.objective_value == pytest.approx(10.0)
    assert result.solver_name == "CLARABEL"
    assert result.solve_time_ms >= 0.0
    assert result.iterations == 7
    assert problem.solve_calls == 1
    assert problem.solve_kwargs is not None
    assert str(problem.solve_kwargs["solver"]).upper() == "CLARABEL"
    assert problem.solve_kwargs["verbose"] is False
    assert problem.solve_kwargs["warm_start"] is True
    assert problem.solve_kwargs["max_iter"] == 20
    assert problem.solve_kwargs["tol_gap_abs"] == pytest.approx(1e-4)
    assert problem.solve_kwargs["custom"] == 1


def test_solve_failure_status_with_fake_problem() -> None:
    """solve should return failure result for infeasible fake problem."""
    adapter = SolverAdapter(SolverConfig(solver="CLARABEL"))
    problem = FakeProblem(status="infeasible", value=None, raw_return=None)

    result = adapter.solve(problem)

    assert result.success is False
    assert result.status == "infeasible"
    assert result.adapter_status is SolverStatus.FAILURE
    assert result.objective_value is None
    assert "infeasible" in str(result.error_message)


def test_solve_optimal_inaccurate_is_success() -> None:
    """optimal_inaccurate should be treated as successful."""
    adapter = SolverAdapter()
    problem = FakeProblem(status="optimal_inaccurate", value=2.0)

    result = adapter.solve(problem)

    assert result.success is True
    assert result.status == "optimal_inaccurate"


def test_solve_allows_runtime_overrides() -> None:
    """solve should allow solver, verbose, warm_start, and options overrides."""
    adapter = SolverAdapter(
        SolverConfig(
            solver="CLARABEL",
            verbose=False,
            warm_start=True,
            max_iter=10,
        )
    )
    problem = FakeProblem(status="optimal", value=1.0)

    result = adapter.solve(
        problem,
        solver="OSQP",
        verbose=True,
        warm_start=False,
        max_iter=99,
        eps_abs=1e-6,
    )

    assert result.success is True
    assert result.solver_name == "OSQP"
    assert problem.solve_kwargs is not None
    assert str(problem.solve_kwargs["solver"]).upper() == "OSQP"
    assert problem.solve_kwargs["verbose"] is True
    assert problem.solve_kwargs["warm_start"] is False
    assert problem.solve_kwargs["max_iter"] == 99
    assert problem.solve_kwargs["eps_abs"] == pytest.approx(1e-6)


def test_solve_exception_returns_failure_result() -> None:
    """solve should convert exceptions to failure result by default."""
    adapter = SolverAdapter(SolverConfig(solver="CLARABEL"))
    problem = RaisingProblem(RuntimeError("boom"))

    result = adapter.solve(problem)

    assert result.success is False
    assert result.status == SolverStatus.EXCEPTION.value
    assert result.adapter_status is SolverStatus.EXCEPTION
    assert result.objective_value is None
    assert result.error_message == "boom"
    assert result.exception_type == "RuntimeError"
    assert problem.solve_kwargs is not None


def test_solve_exception_strict_raises() -> None:
    """solve(strict=True) should raise SolverRuntimeError on exceptions."""
    adapter = SolverAdapter(SolverConfig(solver="CLARABEL"))
    problem = RaisingProblem(RuntimeError("boom"))

    with pytest.raises(SolverRuntimeError, match="boom"):
        adapter.solve(problem, strict=True)


def test_solve_rejects_problem_without_solve() -> None:
    """solve should reject object without callable solve."""
    adapter = SolverAdapter()

    with pytest.raises(SolverRuntimeError, match="solve"):
        adapter.solve(NoSolveProblem())


def test_solve_rejects_invalid_verbose_override() -> None:
    """verbose override must be bool."""
    adapter = SolverAdapter()
    problem = FakeProblem()

    with pytest.raises(SolverConfigError, match="verbose"):
        adapter.solve(problem, verbose="yes")  # type: ignore[arg-type]


def test_solve_rejects_invalid_warm_start_override() -> None:
    """warm_start override must be bool."""
    adapter = SolverAdapter()
    problem = FakeProblem()

    with pytest.raises(SolverConfigError, match="warm_start"):
        adapter.solve(problem, warm_start="yes")  # type: ignore[arg-type]
