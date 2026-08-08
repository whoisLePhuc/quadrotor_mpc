"""Horizon-level safety assurance classification.

This module replaces the old ambiguous ``GUARANTEE_ELIGIBLE`` label with an
explicit, testable status that is only produced when every observable
technical gate for joint chance-constraint assurance passes on the *current
prediction horizon*.

``HORIZON_GUARANTEE_ELIGIBLE`` means:

> The primary solution at this control tick satisfies the configured technical
> conditions to be considered eligible for assurance of the joint chance
> constraints enforced on the current prediction horizon.

It does NOT mean an episode-wide collision-probability guarantee, a claim that
the system cannot collide, that chance constraints hold under uncalibrated
covariance, that a fallback carries the same risk guarantee, or that every
predicted future state will be executed unchanged.

The classifier is a pure function with no side effects and a single source of
truth.  Other layers (telemetry, validation, UI, reporting) only consume
:class:`HorizonAssuranceDecision`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Normalized risk semantics
# ---------------------------------------------------------------------------
DETERMINISTIC_SEMANTICS = "deterministic"
INDIVIDUAL_SEMANTICS = "individual"
JOINT_SEMANTICS = "joint"

_RESIDUAL_AVAILABLE = "AVAILABLE"
_RESIDUAL_UNAVAILABLE = "UNAVAILABLE"
_RESIDUAL_INVALID = "INVALID"

# Risk-budget statuses produced by control/nmpc/risk_budget.py.
_BUDGET_OK = "BUDGET_OK"

# Legacy artifacts may still carry the old label.  It must never be promoted
# to the new eligible status because the old data does not contain the gates
# introduced here.
LEGACY_STATUS_MAP: dict[str, str] = {
    "GUARANTEE_ELIGIBLE": "LEGACY_GUARANTEE_ELIGIBLE_UNVERIFIED",
}

ASSURANCE_SCHEMA_VERSION = 2


class HorizonAssuranceStatus(str, Enum):
    """Explicit assurance status for one prediction-horizon solution."""

    NOT_APPLICABLE_DETERMINISTIC = "NOT_APPLICABLE_DETERMINISTIC"
    NOT_JOINT_GUARANTEE_INDIVIDUAL_RISK = (
        "NOT_JOINT_GUARANTEE_INDIVIDUAL_RISK"
    )
    NOT_GUARANTEED_UNKNOWN_RISK_SEMANTICS = (
        "NOT_GUARANTEED_UNKNOWN_RISK_SEMANTICS"
    )
    HORIZON_GUARANTEE_ELIGIBLE = "HORIZON_GUARANTEE_ELIGIBLE"
    NOT_GUARANTEED_RISK_BUDGET_INVALID = (
        "NOT_GUARANTEED_RISK_BUDGET_INVALID"
    )
    NOT_GUARANTEED_PRIMARY_SOLVER_FAILURE = (
        "NOT_GUARANTEED_PRIMARY_SOLVER_FAILURE"
    )
    NOT_GUARANTEED_RESIDUAL_UNAVAILABLE = (
        "NOT_GUARANTEED_RESIDUAL_UNAVAILABLE"
    )
    NOT_GUARANTEED_RESIDUAL_INVALID = "NOT_GUARANTEED_RESIDUAL_INVALID"
    NOT_GUARANTEED_POSITIVE_SLACK = "NOT_GUARANTEED_POSITIVE_SLACK"
    NOT_GUARANTEED_DEADLINE_MISS = "NOT_GUARANTEED_DEADLINE_MISS"
    NOT_GUARANTEED_FALLBACK_ACTIVE = "NOT_GUARANTEED_FALLBACK_ACTIVE"
    NOT_GUARANTEED_INVALID_NUMERICS = "NOT_GUARANTEED_INVALID_NUMERICS"


# Status precedence when several checks fail at once.  The first two entries
# decide whether joint-horizon assurance applies at all; numerics, budget,
# solver and residual destroy the basis for evaluating the solution; slack,
# deadline and fallback are enforcement gates.
_PRECEDENCE: tuple[tuple[HorizonAssuranceStatus, str], ...] = (
    (HorizonAssuranceStatus.NOT_APPLICABLE_DETERMINISTIC, "risk_semantics"),
    (HorizonAssuranceStatus.NOT_JOINT_GUARANTEE_INDIVIDUAL_RISK, "risk_semantics"),
    (HorizonAssuranceStatus.NOT_GUARANTEED_UNKNOWN_RISK_SEMANTICS, "risk_semantics_unknown"),
    (HorizonAssuranceStatus.NOT_GUARANTEED_INVALID_NUMERICS, "invalid_numerics"),
    (HorizonAssuranceStatus.NOT_GUARANTEED_RISK_BUDGET_INVALID, "risk_budget_invalid"),
    (HorizonAssuranceStatus.NOT_GUARANTEED_PRIMARY_SOLVER_FAILURE, "primary_solver_failure"),
    (HorizonAssuranceStatus.NOT_GUARANTEED_RESIDUAL_UNAVAILABLE, "residual_unavailable"),
    (HorizonAssuranceStatus.NOT_GUARANTEED_RESIDUAL_INVALID, "residual_invalid"),
    (HorizonAssuranceStatus.NOT_GUARANTEED_POSITIVE_SLACK, "positive_slack"),
    (HorizonAssuranceStatus.NOT_GUARANTEED_DEADLINE_MISS, "deadline_miss"),
    (HorizonAssuranceStatus.NOT_GUARANTEED_FALLBACK_ACTIVE, "fallback_active"),
)

_ELIGIBLE_STATUS = HorizonAssuranceStatus.HORIZON_GUARANTEE_ELIGIBLE


@dataclass(frozen=True, slots=True)
class HorizonAssuranceInput:
    """Immutable inputs consumed by the assurance classifier."""

    risk_semantics: str
    risk_budget_status: str | None
    primary_solver_success: bool
    residual_status: str
    primal_residual: float | None
    dual_residual: float | None
    primal_residual_tolerance: float
    dual_residual_tolerance: float
    maximum_slack: float
    slack_tolerance: float
    deadline_missed: bool
    fallback_active: bool


@dataclass(frozen=True, slots=True)
class HorizonAssuranceDecision:
    """Result of one assurance classification."""

    status: HorizonAssuranceStatus
    eligible: bool
    reason_code: str
    failed_checks: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.eligible != (self.status == _ELIGIBLE_STATUS):
            raise ValueError(
                "eligible must equal (status == HORIZON_GUARANTEE_ELIGIBLE)"
            )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "horizon_assurance_status": self.status.value,
            "horizon_assurance_eligible": self.eligible,
            "horizon_assurance_reason": self.reason_code,
            "horizon_assurance_failed_checks": list(self.failed_checks),
        }


def _decision(
    status: HorizonAssuranceStatus,
    *,
    reason_code: str,
    failed_checks: tuple[str, ...],
) -> HorizonAssuranceDecision:
    return HorizonAssuranceDecision(
        status=status,
        eligible=status == _ELIGIBLE_STATUS,
        reason_code=reason_code,
        failed_checks=failed_checks,
    )


def _decision_from_precedence(
    failed: list[str],
) -> HorizonAssuranceDecision:
    """Map a list of failed checks to a status using the fixed precedence."""
    failed_set = set(failed)
    for status, reason in _PRECEDENCE:
        if reason in failed_set:
            return _decision(
                status,
                reason_code=reason,
                failed_checks=tuple(failed),
            )
    return _decision(
        _ELIGIBLE_STATUS,
        reason_code="eligible",
        failed_checks=tuple(failed),
    )


def normalize_risk_semantics(value: str) -> str:
    """Map raw risk semantics to the canonical DETERMINISTIC/INDIVIDUAL/JOINT.

    ``disabled`` (chance constraints disabled) and the empty default map to
    deterministic.  Validation modes such as ``joint_uniform`` or
    ``joint_adaptive`` map to joint while their allocation method is preserved
    separately in telemetry.
    """
    normalized = str(value).strip().lower()
    if normalized in {"", "disabled", "deterministic", "none"}:
        return DETERMINISTIC_SEMANTICS
    if normalized in {"individual"}:
        return INDIVIDUAL_SEMANTICS
    if normalized in {"joint", "joint_uniform", "joint_adaptive"}:
        return JOINT_SEMANTICS
    return normalized  # unknown semantics: caller decides


def classify_horizon_assurance(
    data: HorizonAssuranceInput,
) -> HorizonAssuranceDecision:
    """Classify one prediction-horizon solution against the assurance gates.

    The classifier never invents data: a missing residual is never replaced
    with ``0.0``, and NaN/Inf always blocks eligibility.
    """
    failed: list[str] = []
    semantics = normalize_risk_semantics(data.risk_semantics)

    if semantics == DETERMINISTIC_SEMANTICS:
        return _decision(
            HorizonAssuranceStatus.NOT_APPLICABLE_DETERMINISTIC,
            reason_code="risk_semantics",
            failed_checks=("risk_semantics",),
        )

    if semantics == INDIVIDUAL_SEMANTICS:
        return _decision(
            HorizonAssuranceStatus.NOT_JOINT_GUARANTEE_INDIVIDUAL_RISK,
            reason_code="risk_semantics",
            failed_checks=("risk_semantics",),
        )

    if semantics != JOINT_SEMANTICS:
        failed.append("risk_semantics_unknown")

    numeric_values = (
        data.maximum_slack,
        data.slack_tolerance,
        data.primal_residual_tolerance,
        data.dual_residual_tolerance,
    )
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value)
        for value in numeric_values
    ):
        failed.append("invalid_numerics")

    if data.risk_budget_status != _BUDGET_OK:
        failed.append("risk_budget_invalid")

    if not data.primary_solver_success:
        failed.append("primary_solver_failure")

    if data.residual_status == _RESIDUAL_UNAVAILABLE:
        failed.append("residual_unavailable")
    elif data.residual_status != _RESIDUAL_AVAILABLE:
        failed.append("residual_invalid")
    elif (
        data.primal_residual is None
        or data.dual_residual is None
        or not math.isfinite(data.primal_residual)
        or not math.isfinite(data.dual_residual)
    ):
        failed.append("residual_invalid")
    elif (
        data.primal_residual > data.primal_residual_tolerance
        or data.dual_residual > data.dual_residual_tolerance
    ):
        failed.append("residual_invalid")

    if data.maximum_slack > data.slack_tolerance:
        failed.append("positive_slack")

    if data.deadline_missed:
        failed.append("deadline_miss")

    if data.fallback_active:
        failed.append("fallback_active")

    return _decision_from_precedence(failed)
