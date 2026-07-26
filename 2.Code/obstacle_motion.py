"""Validated obstacle motion models shared by NMPC, MuJoCo and replay.

The functions in this module are deliberately independent of MuJoCo and do-mpc.
This keeps one mathematical definition of obstacle motion for prediction, physics,
visualization and telemetry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

SUPPORTED_MOTIONS = {"sinusoidal", "constant_velocity", "waypoints"}


def _time_scalar(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError("obstacle time must contain one scalar value")
    return float(array.reshape(-1)[0])


def _positive(value: Any, label: str, *, allow_zero: bool = False) -> float:
    number = float(value)
    if number < 0.0 or (number == 0.0 and not allow_zero):
        relation = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{label} must be {relation}")
    return number


def _vector3(value: Any, label: str) -> list[float]:
    if isinstance(value, Mapping):
        try:
            return [float(value[axis]) for axis in ("x", "y", "z")]
        except KeyError as exc:
            raise ValueError(f"{label} requires x, y and z") from exc
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly three values")
    return [float(component) for component in value]


def _legacy_position(raw: Mapping[str, Any], label: str) -> list[float]:
    if "position" in raw:
        return _vector3(raw["position"], f"{label}.position")
    try:
        return [float(raw[axis]) for axis in ("x", "y", "z")]
    except KeyError as exc:
        raise ValueError(f"{label} requires position or x, y and z") from exc


def normalize_obstacle(raw: Mapping[str, Any], index: int = 0) -> dict[str, Any]:
    """Return a validated, serialization-friendly obstacle dictionary.

    The legacy dynamic schema (``x``, ``z``, ``amp`` and ``period``) remains
    accepted and is normalized to a three-axis sinusoidal motion.
    """
    label = f"obstacles[{index}]"
    kind = str(raw.get("type", "static")).lower()
    if kind not in {"static", "dynamic"}:
        raise ValueError(f"{label}.type must be 'static' or 'dynamic'")

    normalized: dict[str, Any] = {
        "name": str(raw.get("name", f"obstacle_{index}")),
        "type": kind,
        "radius": _positive(raw.get("radius"), f"{label}.radius"),
    }
    if kind == "static":
        position = _legacy_position(raw, label)
        normalized.update({"x": position[0], "y": position[1], "z": position[2]})
        return normalized

    motion_raw = raw.get("motion")
    if motion_raw is None:
        # Backward-compatible y(t) = amp*sin(2*pi*t/period) schema.
        try:
            center = [float(raw["x"]), 0.0, float(raw["z"])]
            amplitude = [0.0, float(raw["amp"]), 0.0]
            period = _positive(raw["period"], f"{label}.period")
        except KeyError as exc:
            raise ValueError(f"{label}.motion is required for a dynamic obstacle") from exc
        normalized["motion"] = {
            "type": "sinusoidal",
            "center": center,
            "amplitude": amplitude,
            "period_s": period,
            "phase_rad": 0.0,
        }
        return normalized

    if not isinstance(motion_raw, Mapping):
        raise ValueError(f"{label}.motion must be a mapping")
    motion_type = str(motion_raw.get("type", "")).lower()
    if motion_type not in SUPPORTED_MOTIONS:
        choices = ", ".join(sorted(SUPPORTED_MOTIONS))
        raise ValueError(f"{label}.motion.type must be one of: {choices}")

    if motion_type == "sinusoidal":
        center = _vector3(motion_raw.get("center"), f"{label}.motion.center")
        amplitude = _vector3(
            motion_raw.get("amplitude", [0.0, 0.0, 0.0]),
            f"{label}.motion.amplitude",
        )
        period = _positive(motion_raw.get("period_s"), f"{label}.motion.period_s")
        normalized["motion"] = {
            "type": motion_type,
            "center": center,
            "amplitude": amplitude,
            "period_s": period,
            "phase_rad": float(motion_raw.get("phase_rad", 0.0)),
        }
    elif motion_type == "constant_velocity":
        normalized["motion"] = {
            "type": motion_type,
            "initial_position": _vector3(
                motion_raw.get("initial_position"),
                f"{label}.motion.initial_position",
            ),
            "velocity": _vector3(motion_raw.get("velocity"), f"{label}.motion.velocity"),
        }
    else:
        points_raw = motion_raw.get("points")
        if (
            not isinstance(points_raw, Sequence)
            or isinstance(points_raw, (str, bytes))
            or len(points_raw) < 2
        ):
            raise ValueError(f"{label}.motion.points must contain at least two waypoints")
        points: list[dict[str, Any]] = []
        previous_time = -np.inf
        for point_index, point_raw in enumerate(points_raw):
            if not isinstance(point_raw, Mapping):
                raise ValueError(f"{label}.motion.points[{point_index}] must be a mapping")
            time_s = _positive(
                point_raw.get("time_s"),
                f"{label}.motion.points[{point_index}].time_s",
                allow_zero=True,
            )
            if time_s <= previous_time:
                raise ValueError(f"{label}.motion waypoint times must be strictly increasing")
            previous_time = time_s
            points.append(
                {
                    "time_s": time_s,
                    "position": _vector3(
                        point_raw.get("position"),
                        f"{label}.motion.points[{point_index}].position",
                    ),
                }
            )
        normalized["motion"] = {
            "type": motion_type,
            "repeat": bool(motion_raw.get("repeat", True)),
            "points": points,
        }
    return normalized


def obstacle_position(obstacle: Mapping[str, Any], time_s: float) -> np.ndarray:
    """Evaluate one normalized or legacy obstacle at absolute simulation time."""
    time_value = _time_scalar(time_s)
    if obstacle["type"] == "static":
        return np.array([obstacle["x"], obstacle["y"], obstacle["z"]], dtype=float)

    motion = obstacle.get("motion")
    if motion is None:
        # Keep direct callers using the pre-v1.2 dictionary functional.
        return np.array(
            [
                obstacle["x"],
                obstacle["amp"] * np.sin(2.0 * np.pi * time_value / obstacle["period"]),
                obstacle["z"],
            ],
            dtype=float,
        )

    motion_type = motion["type"]
    if motion_type == "sinusoidal":
        phase = 2.0 * np.pi * time_value / float(motion["period_s"])
        phase += float(motion.get("phase_rad", 0.0))
        return np.asarray(motion["center"], dtype=float) + np.asarray(
            motion["amplitude"], dtype=float
        ) * np.sin(phase)
    if motion_type == "constant_velocity":
        return np.asarray(motion["initial_position"], dtype=float) + time_value * np.asarray(
            motion["velocity"], dtype=float
        )

    points = motion["points"]
    times = np.asarray([point["time_s"] for point in points], dtype=float)
    positions = np.asarray([point["position"] for point in points], dtype=float)
    sample_time = max(0.0, time_value)
    if motion.get("repeat", True) and times[-1] > 0.0:
        sample_time = sample_time % times[-1]
    if sample_time <= times[0]:
        return positions[0].copy()
    if sample_time >= times[-1]:
        return positions[-1].copy()
    upper = int(np.searchsorted(times, sample_time, side="right"))
    lower = upper - 1
    alpha = (sample_time - times[lower]) / (times[upper] - times[lower])
    return (1.0 - alpha) * positions[lower] + alpha * positions[upper]


def predict_obstacle_positions(
    obstacles: Sequence[Mapping[str, Any]], start_time_s: float, steps: int, dt: float
) -> np.ndarray:
    """Return shape ``(n_obstacles, steps, 3)`` for a future time grid."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    times = float(start_time_s) + np.arange(steps, dtype=float) * float(dt)
    if not obstacles:
        return np.empty((0, steps, 3), dtype=float)
    return np.asarray(
        [[obstacle_position(obstacle, time_s) for time_s in times] for obstacle in obstacles],
        dtype=float,
    )
