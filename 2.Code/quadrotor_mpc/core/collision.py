"""Collision classification contracts for the native validation pipeline.

Separates obstacle collisions, ground collisions and signed clearance so the
Monte Carlo baseline can tell exactly what a controller hit.  Collision
detection stays authoritative in the MuJoCo plant (contact-based); clearance
is a separate signed metric with explicit ``None`` for "no applicable value".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class CollisionType(str, Enum):
    """Episode-level collision classification."""

    NONE = "none"
    OBSTACLE = "obstacle"
    GROUND = "ground"
    OBSTACLE_AND_GROUND = "obstacle_and_ground"
    UNKNOWN_LEGACY = "unknown_legacy"


def classify_collision(
    obstacle_collision_detected: bool,
    ground_collision_detected: bool,
) -> CollisionType:
    """Classify an episode from the two independent detection booleans."""
    if obstacle_collision_detected and ground_collision_detected:
        return CollisionType.OBSTACLE_AND_GROUND
    if obstacle_collision_detected:
        return CollisionType.OBSTACLE
    if ground_collision_detected:
        return CollisionType.GROUND
    return CollisionType.NONE


@dataclass(frozen=True, slots=True)
class CollisionObservation:
    """Per-control-tick collision observation.

    ``minimum_*_clearance_m`` use signed semantics: positive means remaining
    safety distance, zero means boundary contact and negative means
    penetration/collision.  ``None`` means the value is not applicable (for
    example obstacle clearance when the scene has no obstacles), never zero.
    """

    obstacle_collision_detected: bool
    ground_collision_detected: bool
    minimum_obstacle_clearance_m: float | None
    minimum_ground_clearance_m: float | None

    @property
    def collision_detected(self) -> bool:
        """True when any collision type was detected this tick."""
        return self.obstacle_collision_detected or self.ground_collision_detected


@dataclass(frozen=True, slots=True)
class CollisionSummary:
    """Episode-level collision outcome and clearance minima."""

    obstacle_collision_detected: bool
    ground_collision_detected: bool
    collision_type: CollisionType
    first_collision_time_s: float | None
    first_obstacle_collision_time_s: float | None
    first_ground_collision_time_s: float | None
    minimum_obstacle_clearance_m: float | None
    minimum_ground_clearance_m: float | None

    @property
    def collided(self) -> bool:
        """Backward-compatible cumulative flag derived from the two booleans."""
        return self.obstacle_collision_detected or self.ground_collision_detected

    @property
    def first_collision_type(self) -> CollisionType:
        """Type of the collision that occurred first in the episode.

        ``OBSTACLE_AND_GROUND`` is returned only when both types first occur
        on the same tick.  ``NONE`` when no collision ever occurred.
        """
        if self.first_collision_time_s is None:
            return CollisionType.NONE
        obstacle_first = self.first_obstacle_collision_time_s
        ground_first = self.first_ground_collision_time_s
        if obstacle_first is not None and ground_first is not None:
            if obstacle_first == ground_first:
                return CollisionType.OBSTACLE_AND_GROUND
            return CollisionType.OBSTACLE if obstacle_first < ground_first else CollisionType.GROUND
        if obstacle_first is not None:
            return CollisionType.OBSTACLE
        if ground_first is not None:
            return CollisionType.GROUND
        return CollisionType.NONE


class CollisionAccumulator:
    """Stateful episode accumulator for per-tick collision observations.

    Update rules (see the measurement-stabilization spec):

    1. Episode booleans only transition ``False -> True``, never back.
    2. ``first_*_time_s`` is recorded only on the first true condition.
    3. Minimum clearance ignores ``None``; non-finite values are invalid and
       raise ``ValueError`` (the repo contract policy for numeric telemetry).
    4. ``first_collision_time_s`` is the minimum of the two valid first times.
    5. ``collided`` is always derived from the two booleans.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._obstacle_detected = False
        self._ground_detected = False
        self._first_obstacle_time_s: float | None = None
        self._first_ground_time_s: float | None = None
        self._min_obstacle_clearance_m: float | None = None
        self._min_ground_clearance_m: float | None = None

    def observe(self, observation: CollisionObservation, time_s: float) -> None:
        """Fold one per-tick observation into the episode state."""
        if observation.obstacle_collision_detected and not self._obstacle_detected:
            self._obstacle_detected = True
            self._first_obstacle_time_s = float(time_s)
        if observation.ground_collision_detected and not self._ground_detected:
            self._ground_detected = True
            self._first_ground_time_s = float(time_s)
        self._fold_clearance(
            "_min_obstacle_clearance_m",
            observation.minimum_obstacle_clearance_m,
        )
        self._fold_clearance(
            "_min_ground_clearance_m",
            observation.minimum_ground_clearance_m,
        )

    def _fold_clearance(self, attribute: str, candidate: float | None) -> None:
        if candidate is None:
            return
        value = float(candidate)
        if not math.isfinite(value):
            raise ValueError(
                f"CollisionObservation.{attribute[1:]} must be finite or None, got {candidate!r}"
            )
        current = getattr(self, attribute)
        if current is None or value < current:
            setattr(self, attribute, value)

    def finalize(self) -> CollisionSummary:
        """Return the episode summary without mutating accumulated state."""
        candidates = [
            value
            for value in (self._first_obstacle_time_s, self._first_ground_time_s)
            if value is not None
        ]
        first_collision_time_s = min(candidates) if candidates else None
        return CollisionSummary(
            obstacle_collision_detected=self._obstacle_detected,
            ground_collision_detected=self._ground_detected,
            collision_type=classify_collision(
                self._obstacle_detected,
                self._ground_detected,
            ),
            first_collision_time_s=first_collision_time_s,
            first_obstacle_collision_time_s=self._first_obstacle_time_s,
            first_ground_collision_time_s=self._first_ground_time_s,
            minimum_obstacle_clearance_m=self._min_obstacle_clearance_m,
            minimum_ground_clearance_m=self._min_ground_clearance_m,
        )
