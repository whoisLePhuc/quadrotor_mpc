"""Monotonic high-resolution controller timing contracts.

Per-tick controller phases are measured with ``time.perf_counter_ns`` (or an
injected deterministic clock) and aggregated per episode by the Monte Carlo
pipeline.  ``total_controller_time_ms`` is always measured independently at
the outermost boundary and never derived from the sum of component timings:
component scopes can overlap (a parent chance-profile scope contains the
allocation and tightening child scopes) and some work is not instrumented.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields

import numpy as np

ClockNs = Callable[[], int]

TIMING_FIELD_NAMES = (
    "seed_trajectory_time_ms",
    "covariance_propagation_time_ms",
    "geometry_context_time_ms",
    "risk_allocation_time_ms",
    "tightening_time_ms",
    "chance_profile_time_ms",
    "tvp_update_time_ms",
    "nlp_solve_time_ms",
    "post_solve_diagnostic_time_ms",
    "safety_supervisor_time_ms",
    "total_controller_time_ms",
)

TIMING_SCHEMA_METADATA = {
    "timing_schema_version": 1,
    "clock": "time.perf_counter_ns",
    "unit": "ms",
    "scope_semantics": "mixed_parent_and_child_scopes",
}


@dataclass(frozen=True, slots=True)
class ControllerTiming:
    """One tick's measured controller phase durations in milliseconds.

    ``None`` means the phase did not run or was not applicable; it must never
    be serialized as ``0.0``.  All non-null values are finite and nonnegative.
    ``chance_profile_time_ms`` is a parent scope that contains
    ``risk_allocation_time_ms`` and ``tightening_time_ms``.
    """

    seed_trajectory_time_ms: float | None = None
    covariance_propagation_time_ms: float | None = None
    geometry_context_time_ms: float | None = None
    risk_allocation_time_ms: float | None = None
    tightening_time_ms: float | None = None
    chance_profile_time_ms: float | None = None
    tvp_update_time_ms: float | None = None
    nlp_solve_time_ms: float | None = None
    post_solve_diagnostic_time_ms: float | None = None
    safety_supervisor_time_ms: float | None = None
    total_controller_time_ms: float | None = None

    def __post_init__(self) -> None:
        for name in TIMING_FIELD_NAMES:
            value = getattr(self, name)
            if value is None:
                continue
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"ControllerTiming.{name} must be finite and >= 0 or None")
            object.__setattr__(self, name, number)

    def to_mapping(self) -> dict[str, float | None]:
        return {name: getattr(self, name) for name in TIMING_FIELD_NAMES}

    @classmethod
    def from_mapping(cls, mapping: dict | None) -> ControllerTiming | None:
        if not mapping:
            return None
        return cls(**{name: mapping.get(name) for name in TIMING_FIELD_NAMES})


def merge_controller_timing(
    base: ControllerTiming | None,
    **updates: float | None,
) -> ControllerTiming:
    """Return a timing with ``updates`` applied over ``base``."""
    values: dict[str, float | None] = (
        base.to_mapping() if base is not None else {name: None for name in TIMING_FIELD_NAMES}
    )
    values.update(updates)
    return ControllerTiming(**values)


@dataclass(frozen=True, slots=True)
class TimingFieldStats:
    """Per-tick timing statistics for one controller phase over an episode."""

    count_available: int
    count_missing: int
    mean_ms: float | None
    median_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    max_ms: float | None

    def to_mapping(self) -> dict[str, float | int | None]:
        return {
            "count_available": self.count_available,
            "count_missing": self.count_missing,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "max_ms": self.max_ms,
        }


def summarize_timing_series(
    series: list[dict[str, float | None] | None],
) -> dict[str, dict[str, float | int | None]]:
    """Aggregate one tick's timing mappings into per-field episode statistics.

    Missing (``None``) samples are excluded from percentiles but reported via
    ``count_missing``; they are never replaced by zero.
    """
    result: dict[str, dict[str, float | int | None]] = {}
    for name in TIMING_FIELD_NAMES:
        values: list[float] = []
        for item in series:
            if item is None:
                continue
            value = item.get(name)
            if value is None:
                continue
            number = float(value)
            if math.isfinite(number):
                values.append(number)
        count_available = len(values)
        count_missing = len(series) - count_available
        if values:
            ordered = sorted(values)
            result[name] = {
                "count_available": count_available,
                "count_missing": count_missing,
                "mean_ms": float(np.mean(ordered)),
                "median_ms": float(np.median(ordered)),
                "p95_ms": float(np.percentile(ordered, 95)),
                "p99_ms": float(np.percentile(ordered, 99)),
                "max_ms": float(max(ordered)),
            }
        else:
            result[name] = {
                "count_available": 0,
                "count_missing": count_missing,
                "mean_ms": None,
                "median_ms": None,
                "p95_ms": None,
                "p99_ms": None,
                "max_ms": None,
            }
    return result


class TimingRecorder:
    """Per-tick named-scope timing recorder with a monotonic clock.

    The clock is injectable for deterministic unit tests; production uses
    ``time.perf_counter_ns``.  Duplicate scope names fail fast so a phase can
    never silently overwrite an earlier measurement.
    """

    def __init__(self, clock_ns: ClockNs | None = None) -> None:
        self._clock_ns = clock_ns if clock_ns is not None else time.perf_counter_ns
        self._values_ms: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        if name in self._values_ms:
            raise ValueError(f"timing scope measured twice in one tick: {name!r}")
        start = self._clock_ns()
        try:
            yield
        finally:
            elapsed_ns = self._clock_ns() - start
            self._values_ms[name] = elapsed_ns / 1_000_000.0

    def snapshot(self) -> ControllerTiming:
        known = {field.name for field in fields(ControllerTiming)}
        unknown = set(self._values_ms) - known
        if unknown:
            raise ValueError(f"unrecognized timing scope names: {sorted(unknown)}")
        return ControllerTiming(**self._values_ms)
