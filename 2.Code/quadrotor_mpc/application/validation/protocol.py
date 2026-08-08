"""Monte Carlo validation protocols and command disposition semantics.

Two independent validation questions must never be conflated:

- ``ALGORITHMIC_COMPARISON`` measures controller/allocator quality when an
  otherwise valid primary command is actually applied, even if it is late.
- ``REALTIME_QUALIFICATION`` measures the full closed-loop stack under the
  real deadline and fallback policy.

A run that applies late-but-valid primary commands is not evidence of
real-time feasibility, and a run that replaces late commands with fallback is
not a fair allocator comparison.  The applied-command decision is a pure
function of the protocol; every timing fact is recorded identically.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

SCHEMA_VERSION = 2


class MonteCarloProtocol(str, Enum):
    ALGORITHMIC_COMPARISON = "algorithmic_comparison"
    REALTIME_QUALIFICATION = "realtime_qualification"
    UNKNOWN_LEGACY = "UNKNOWN_LEGACY"


class DeadlinePolicy(str, Enum):
    RECORD_ONLY = "record_only"
    REJECT_TO_FALLBACK = "reject_to_fallback"


class DeadlineClock(str, Enum):
    SOLVER_ONLY = "solver_only"
    END_TO_END_CONTROLLER = "end_to_end_controller"


PRIMARY_SOURCE = "PRIMARY"
FALLBACK_SOURCE = "FALLBACK"
ABORT_HOVER_SOURCE = "ABORT_HOVER"
NONE_SOURCE = "NONE"

PRIMARY_APPLIED_ON_TIME = "PRIMARY_APPLIED_ON_TIME"
PRIMARY_APPLIED_LATE = "PRIMARY_APPLIED_LATE"
FALLBACK_APPLIED_DEADLINE_MISS = "FALLBACK_APPLIED_DEADLINE_MISS"
FALLBACK_APPLIED_PRIMARY_INVALID = "FALLBACK_APPLIED_PRIMARY_INVALID"
PRIMARY_NOT_APPLIED = "PRIMARY_NOT_APPLIED"


def deadline_policy_for(protocol: MonteCarloProtocol) -> DeadlinePolicy:
    if protocol is MonteCarloProtocol.ALGORITHMIC_COMPARISON:
        return DeadlinePolicy.RECORD_ONLY
    if protocol is MonteCarloProtocol.REALTIME_QUALIFICATION:
        return DeadlinePolicy.REJECT_TO_FALLBACK
    raise ValueError(f"Unsupported protocol: {protocol}")


def validate_protocol_policy(
    protocol: MonteCarloProtocol,
    deadline_policy: DeadlinePolicy,
) -> None:
    """Reject protocol/policy combinations that cannot be interpreted."""
    derived = deadline_policy_for(protocol)
    if deadline_policy is not derived:
        raise ValueError(
            f"protocol {protocol.value} requires deadline_policy "
            f"{derived.value}, got {deadline_policy.value}"
        )


def validate_control_period(control_period_ms: float) -> float:
    period = float(control_period_ms)
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("control_period_ms must be finite and > 0")
    return period


def deadline_facts(
    total_controller_time_ms: float,
    control_period_ms: float,
) -> tuple[bool, float]:
    """Return ``(deadline_missed, deadline_overrun_ms)``.

    ``deadline_missed`` is strictly ``elapsed > period``; exactly equal to the
    period is not a miss.
    """
    elapsed = float(total_controller_time_ms)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("total_controller_time_ms must be finite and >= 0")
    period = validate_control_period(control_period_ms)
    return elapsed > period, max(0.0, elapsed - period)


@dataclass(frozen=True, slots=True)
class AppliedCommandDecision:
    """Canonical command disposition for one control tick."""

    applied_command_source: str
    primary_disposition: str
    primary_valid: bool
    deadline_missed: bool
    rejection_reasons: tuple[str, ...] = ()


def select_applied_command(
    *,
    protocol: MonteCarloProtocol,
    primary_valid: bool,
    deadline_missed: bool,
    rejection_reasons: tuple[str, ...] = (),
) -> AppliedCommandDecision:
    """Truth-table decision of which command is applied for this protocol.

    Precedence: invalid/non-finite command > solver failure > residual/slack/
    safety rejection > deadline miss > accepted.  Only one canonical
    disposition is returned; all rejection reasons are preserved.
    """
    reasons = tuple(str(reason) for reason in rejection_reasons)
    if not primary_valid:
        return AppliedCommandDecision(
            applied_command_source=FALLBACK_SOURCE,
            primary_disposition=FALLBACK_APPLIED_PRIMARY_INVALID,
            primary_valid=False,
            deadline_missed=deadline_missed,
            rejection_reasons=reasons,
        )
    if deadline_missed:
        if protocol is MonteCarloProtocol.ALGORITHMIC_COMPARISON:
            return AppliedCommandDecision(
                applied_command_source=PRIMARY_SOURCE,
                primary_disposition=PRIMARY_APPLIED_LATE,
                primary_valid=True,
                deadline_missed=True,
                rejection_reasons=reasons,
            )
        if protocol is MonteCarloProtocol.REALTIME_QUALIFICATION:
            return AppliedCommandDecision(
                applied_command_source=FALLBACK_SOURCE,
                primary_disposition=FALLBACK_APPLIED_DEADLINE_MISS,
                primary_valid=True,
                deadline_missed=True,
                rejection_reasons=reasons,
            )
    return AppliedCommandDecision(
        applied_command_source=PRIMARY_SOURCE,
        primary_disposition=PRIMARY_APPLIED_ON_TIME,
        primary_valid=True,
        deadline_missed=deadline_missed,
        rejection_reasons=reasons,
    )


@dataclass(frozen=True, slots=True)
class MonteCarloProtocolMetadata:
    """Run-level provenance for one Monte Carlo campaign."""

    protocol: MonteCarloProtocol
    control_period_ms: float
    deadline_policy: DeadlinePolicy
    deadline_clock: DeadlineClock
    config_hash: str
    schema_version: int = SCHEMA_VERSION

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol": self.protocol.value,
            "deadline_policy": self.deadline_policy.value,
            "deadline_clock": self.deadline_clock.value,
            "control_period_ms": self.control_period_ms,
            "config_hash": self.config_hash,
        }


def config_hash_of(config_mapping: dict[str, Any]) -> str:
    """Deterministic digest over the effective campaign configuration.

    Python's builtin ``hash()`` is process-instance specific and must not be
    used for artifact identity.
    """
    canonical = json.dumps(
        config_mapping,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _json_default(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
