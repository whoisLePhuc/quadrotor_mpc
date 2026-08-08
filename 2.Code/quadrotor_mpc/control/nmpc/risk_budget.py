"""Risk-budget semantics and allocation for native chance constraints.

Stage 5 supports two explicit meanings:

``individual``
    The same epsilon is attached to every scalar chance constraint.  This is
    backward compatible with Stage 4 and is not a joint-horizon guarantee.

``joint`` with ``uniform`` allocation
    A total receding-horizon budget is split uniformly across every active
    horizon-step/obstacle pair.  Boole's inequality then gives the conservative
    upper bound ``sum(epsilon[k, o]) <= total_epsilon``.

The allocator is intentionally independent of the controller backend so later
geometry-aware or iterative risk allocation can replace it without changing
the belief/controller interface.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

INDIVIDUAL_RISK = "individual"
JOINT_RISK = "joint"
UNIFORM_ALLOCATION = "uniform"


def _epsilon(value: Any, label: str) -> float:
    number = float(value)
    if not np.isfinite(number) or not 0.0 < number < 0.5:
        raise ValueError(f"{label} must be finite and in (0, 0.5)")
    return number


@dataclass(frozen=True, slots=True)
class RiskBudgetOptions:
    """Configuration for individual or joint receding-horizon risk."""

    semantics: str = INDIVIDUAL_RISK
    allocation: str = UNIFORM_ALLOCATION
    total_epsilon: float = 0.10
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        semantics = str(self.semantics).strip().lower()
        if semantics not in {INDIVIDUAL_RISK, JOINT_RISK}:
            raise ValueError(
                "controller.chance_constraints.risk_budget.semantics must be "
                "'individual' or 'joint'"
            )
        allocation = str(self.allocation).strip().lower()
        if allocation != UNIFORM_ALLOCATION:
            raise ValueError(
                "controller.chance_constraints.risk_budget.allocation must be 'uniform' in Stage 5"
            )
        tolerance = float(self.tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "controller.chance_constraints.risk_budget.tolerance must be finite and >= 0"
            )
        object.__setattr__(self, "semantics", semantics)
        object.__setattr__(self, "allocation", allocation)
        object.__setattr__(
            self,
            "total_epsilon",
            _epsilon(
                self.total_epsilon,
                "controller.chance_constraints.risk_budget.total_epsilon",
            ),
        )
        object.__setattr__(self, "tolerance", tolerance)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RiskBudgetOptions:
        return cls(
            semantics=raw.get("semantics", INDIVIDUAL_RISK),
            allocation=raw.get("allocation", UNIFORM_ALLOCATION),
            total_epsilon=raw.get("total_epsilon", 0.10),
            tolerance=raw.get("tolerance", 1e-12),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "semantics": self.semantics,
            "allocation": self.allocation,
            "total_epsilon": self.total_epsilon,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class RiskAllocation:
    """Risk and Gaussian quantile assigned to each scalar constraint."""

    epsilons: np.ndarray
    gaussian_quantiles: np.ndarray
    semantics: str
    allocation: str
    configured_total_epsilon: float | None
    allocated_epsilon: float
    remaining_epsilon: float | None
    active_constraint_count: int
    budget_status: str

    def __post_init__(self) -> None:
        epsilons = np.asarray(self.epsilons, dtype=float)
        quantiles = np.asarray(self.gaussian_quantiles, dtype=float)
        if epsilons.ndim != 2:
            raise ValueError("RiskAllocation.epsilons must have shape (steps, obstacles)")
        if quantiles.shape != epsilons.shape:
            raise ValueError("RiskAllocation.gaussian_quantiles must match epsilons")
        if not np.all(np.isfinite(epsilons)) or np.any(epsilons < 0.0):
            raise ValueError("RiskAllocation.epsilons must be finite and nonnegative")
        if not np.all(np.isfinite(quantiles)) or np.any(quantiles < 0.0):
            raise ValueError("RiskAllocation.gaussian_quantiles must be finite and nonnegative")
        for label, array in (
            ("epsilons", epsilons),
            ("gaussian_quantiles", quantiles),
        ):
            copied = array.copy()
            copied.setflags(write=False)
            object.__setattr__(self, label, copied)
        object.__setattr__(self, "semantics", str(self.semantics))
        object.__setattr__(self, "allocation", str(self.allocation))
        object.__setattr__(
            self,
            "active_constraint_count",
            int(self.active_constraint_count),
        )
        object.__setattr__(self, "budget_status", str(self.budget_status))
        if self.active_constraint_count < 0:
            raise ValueError("RiskAllocation.active_constraint_count must be >= 0")
        for label in (
            "configured_total_epsilon",
            "remaining_epsilon",
        ):
            value = getattr(self, label)
            if value is not None:
                number = float(value)
                if not np.isfinite(number) or number < -1e-12:
                    raise ValueError(f"RiskAllocation.{label} must be finite and nonnegative")
                object.__setattr__(self, label, max(0.0, number))
        allocated = float(self.allocated_epsilon)
        if not np.isfinite(allocated) or allocated < -1e-12:
            raise ValueError("RiskAllocation.allocated_epsilon must be finite and nonnegative")
        object.__setattr__(self, "allocated_epsilon", max(0.0, allocated))


def allocate_risk_budget(
    *,
    steps: int,
    obstacle_count: int,
    enabled: bool,
    individual_epsilon: float,
    options: RiskBudgetOptions,
) -> RiskAllocation:
    """Allocate risk for one receding-horizon solve.

    Every step represented in the NMPC constraint grid is counted.  Thus, for
    the native controller the active count is ``(N + 1) * n_obstacles``.
    """

    steps = int(steps)
    obstacle_count = int(obstacle_count)
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if obstacle_count < 0:
        raise ValueError("obstacle_count must be >= 0")
    shape = (steps, obstacle_count)
    active_count = steps * obstacle_count
    if not enabled:
        return RiskAllocation(
            epsilons=np.zeros(shape, dtype=float),
            gaussian_quantiles=np.zeros(shape, dtype=float),
            semantics="disabled",
            allocation="none",
            configured_total_epsilon=None,
            allocated_epsilon=0.0,
            remaining_epsilon=None,
            active_constraint_count=0,
            budget_status="DISABLED",
        )

    individual = _epsilon(
        individual_epsilon,
        "controller.chance_constraints.individual_epsilon",
    )
    if active_count == 0:
        configured_total = options.total_epsilon if options.semantics == JOINT_RISK else None
        return RiskAllocation(
            epsilons=np.zeros(shape, dtype=float),
            gaussian_quantiles=np.zeros(shape, dtype=float),
            semantics=options.semantics,
            allocation=options.allocation,
            configured_total_epsilon=configured_total,
            allocated_epsilon=0.0,
            remaining_epsilon=configured_total,
            active_constraint_count=0,
            budget_status=("BUDGET_OK" if options.semantics == JOINT_RISK else "INDIVIDUAL_ONLY"),
        )

    if options.semantics == INDIVIDUAL_RISK:
        epsilons = np.full(shape, individual, dtype=float)
        configured_total = None
        remaining = None
        status = "INDIVIDUAL_ONLY"
    else:
        per_constraint = options.total_epsilon / active_count
        # The total is already constrained to < 0.5, so each positive uniform
        # share is a valid one-sided Gaussian epsilon.
        epsilons = np.full(shape, per_constraint, dtype=float)
        configured_total = options.total_epsilon
        remaining = max(0.0, configured_total - float(np.sum(epsilons)))
        status = (
            "BUDGET_OK"
            if float(np.sum(epsilons)) <= configured_total + options.tolerance
            else "BUDGET_EXCEEDED"
        )
        if status != "BUDGET_OK":
            raise RuntimeError("uniform joint-risk allocation exceeded total_epsilon")

    quantiles = np.asarray(
        [NormalDist().inv_cdf(1.0 - epsilon) for epsilon in epsilons.reshape(-1)],
        dtype=float,
    ).reshape(shape)
    return RiskAllocation(
        epsilons=epsilons,
        gaussian_quantiles=quantiles,
        semantics=options.semantics,
        allocation=options.allocation,
        configured_total_epsilon=configured_total,
        allocated_epsilon=float(np.sum(epsilons)),
        remaining_epsilon=remaining,
        active_constraint_count=active_count,
        budget_status=status,
    )
