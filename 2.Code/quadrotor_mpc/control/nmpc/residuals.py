"""Honest solver residual semantics for the native NMPC pipeline.

The central invariant of this module:

    missing residual != 0.0

A residual of exactly ``0.0`` means the backend reported a solution that
satisfies the optimality/feasibility condition to machine precision.  A missing
residual means the backend did not expose the value, so the condition cannot be
evaluated at all.  These are different facts and must never be conflated.

Every residual is carried as a :class:`SolverResidual` value object whose
``status`` and ``value`` are kept mutually consistent: ``AVAILABLE`` implies a
finite non-negative number, ``UNAVAILABLE``/``INVALID`` imply ``None``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

_MISSING = object()

# Public sentinel representing "the backend did not expose this field".
MISSING_RESIDUAL = _MISSING


class ResidualStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID = "INVALID"


class ResidualGateStatus(str, Enum):
    PASS = "PASS"
    FAIL_THRESHOLD = "FAIL_THRESHOLD"
    FAIL_INVALID = "FAIL_INVALID"
    UNKNOWN_UNAVAILABLE = "UNKNOWN_UNAVAILABLE"
    NOT_REQUIRED = "NOT_REQUIRED"


@dataclass(frozen=True, slots=True)
class SolverResidual:
    """One residual value with an explicit provenance status."""

    status: ResidualStatus
    value: float | None
    source: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.status is ResidualStatus.AVAILABLE:
            if self.value is None:
                raise ValueError("AVAILABLE residual requires a value")
            if not math.isfinite(self.value) or self.value < 0.0:
                raise ValueError("Residual must be finite and non-negative")
        elif self.value is not None:
            raise ValueError("Unavailable or invalid residual must use None")


@dataclass(frozen=True, slots=True)
class SolverResidualDiagnostics:
    """Independent primal and dual residual observations."""

    primal: SolverResidual
    dual: SolverResidual


def extract_residual(raw_value: Any, *, source: str) -> SolverResidual:
    """Classify a raw backend value into a typed residual.

    ``raw_value`` may be a scalar, a sequence of iteration-history values, or
    the module-level ``_MISSING`` sentinel.  For a collection, every element
    must be finite and non-negative; a single bad element invalidates the whole
    residual (never drop bad elements and aggregate the rest).
    """
    if raw_value is _MISSING:
        return SolverResidual(
            status=ResidualStatus.UNAVAILABLE,
            value=None,
            source=source,
            detail="field_not_exposed_by_backend",
        )

    try:
        values = _normalize_values(raw_value, source=source)
    except (TypeError, ValueError) as exc:
        return SolverResidual(
            status=ResidualStatus.INVALID,
            value=None,
            source=source,
            detail=f"normalization_failed:{type(exc).__name__}",
        )
    if values is None:
        return SolverResidual(
            status=ResidualStatus.UNAVAILABLE,
            value=None,
            source=source,
            detail="backend_returned_no_value",
        )
    if len(values) == 0:
        return SolverResidual(
            status=ResidualStatus.UNAVAILABLE,
            value=None,
            source=source,
            detail="empty_value_collection",
        )
    if any(not math.isfinite(v) or v < 0.0 for v in values):
        return SolverResidual(
            status=ResidualStatus.INVALID,
            value=None,
            source=source,
            detail="non_finite_or_negative_value",
        )
    return SolverResidual(
        status=ResidualStatus.AVAILABLE,
        value=float(values[-1]),
        source=source,
    )


def _normalize_values(raw_value: Any, *, source: str) -> list[float] | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, (int, float)):
        return [float(raw_value)]
    if isinstance(raw_value, (list, tuple)):
        return [float(item) for item in raw_value]
    try:
        import numpy as np

        array = np.asarray(raw_value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"residual normalization failed: {type(exc).__name__}") from exc
    if array.ndim == 0:
        return [float(array)]
    return [float(item) for item in array.reshape(-1)]


def evaluate_residual_gate(
    diagnostics: SolverResidualDiagnostics,
    *,
    primal_tolerance: float,
    dual_tolerance: float,
    required: bool,
) -> ResidualGateStatus:
    """Evaluate the primal/dual residual acceptance gate.

    The gate is ``PASS`` only when every required residual is available and
    within tolerance.  ``FAIL_THRESHOLD`` wins over any other failure when
    direct evidence exceeds a tolerance.  ``FAIL_INVALID`` covers corrupt data.
    ``UNKNOWN_UNAVAILABLE`` means the gate cannot be evaluated because a
    required residual was not observed.
    """
    if not required:
        return ResidualGateStatus.NOT_REQUIRED

    primal = diagnostics.primal
    dual = diagnostics.dual

    def exceeds(residual: SolverResidual, tolerance: float) -> bool:
        return (
            residual.status is ResidualStatus.AVAILABLE
            and residual.value is not None
            and residual.value > tolerance
        )

    if exceeds(primal, primal_tolerance) or exceeds(dual, dual_tolerance):
        return ResidualGateStatus.FAIL_THRESHOLD

    if primal.status is ResidualStatus.INVALID or dual.status is ResidualStatus.INVALID:
        return ResidualGateStatus.FAIL_INVALID

    if primal.status is ResidualStatus.UNAVAILABLE or dual.status is ResidualStatus.UNAVAILABLE:
        return ResidualGateStatus.UNKNOWN_UNAVAILABLE

    if primal.status is ResidualStatus.AVAILABLE and dual.status is ResidualStatus.AVAILABLE:
        return ResidualGateStatus.PASS

    return ResidualGateStatus.UNKNOWN_UNAVAILABLE
