"""Temporary perfect-information belief source for deterministic regression.

The native runner uses this adapter only until sensor simulation and estimators
are introduced in Stage 2.  Keeping it outside the controller makes the future
replacement explicit and prevents controller implementations from reading the
MuJoCo plant directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from controller_interface import ObstacleBelief, SphericalObstacle
from obstacle_motion import obstacle_position, predict_obstacle_positions


def _obstacle_velocity(obstacle: Mapping[str, Any], time_s: float, dt: float) -> np.ndarray:
    sample_dt = min(max(float(dt) * 0.1, 1e-4), 1e-2)
    if float(time_s) <= sample_dt:
        return (
            obstacle_position(obstacle, float(time_s) + sample_dt)
            - obstacle_position(obstacle, float(time_s))
        ) / sample_dt
    return (
        obstacle_position(obstacle, float(time_s) + sample_dt)
        - obstacle_position(obstacle, float(time_s) - sample_dt)
    ) / (2.0 * sample_dt)


def exact_obstacle_beliefs(
    obstacles: Sequence[Mapping[str, Any]],
    time_s: float,
    horizon_steps: int,
    dt: float,
) -> tuple[ObstacleBelief, ...]:
    """Create zero-covariance obstacle beliefs for the deterministic baseline."""
    predictions = predict_obstacle_positions(
        obstacles,
        float(time_s),
        int(horizon_steps) + 1,
        float(dt),
    )
    beliefs: list[ObstacleBelief] = []
    for index, obstacle in enumerate(obstacles):
        position = obstacle_position(obstacle, float(time_s))
        velocity = _obstacle_velocity(obstacle, float(time_s), float(dt))
        beliefs.append(
            ObstacleBelief(
                mean_state_6=np.concatenate([position, velocity]),
                covariance_6=np.zeros((6, 6), dtype=float),
                shape=SphericalObstacle(float(obstacle["radius"])),
                name=str(obstacle.get("name", f"obstacle_{index}")),
                predicted_positions=predictions[index],
            )
        )
    return tuple(beliefs)
