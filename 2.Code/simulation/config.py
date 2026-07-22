"""Typed scenario configuration and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

Array = npt.NDArray[np.float64]


def _vector(value: Any, length: int, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {result.shape}")
    return result


@dataclass(slots=True)
class NoiseConfig:
    measurement_pos: float = 0.01
    measurement_vel: float = 0.02
    measurement_att: float = 0.005
    process_vel: float = 0.02
    process_att: float = 0.005


@dataclass(slots=True)
class ModelMismatchConfig:
    drag_scale: float = 1.0
    attitude_time_scale: float = 1.0
    vertical_time_scale: float = 1.0


@dataclass(slots=True)
class ObstacleConfig:
    position: Array
    size: Array
    yaw: float = 0.0
    velocity: Array = field(default_factory=lambda: np.zeros(3))
    pos_uncertainty: float = 0.05
    vel_uncertainty: float = 0.10

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ObstacleConfig":
        return cls(
            position=_vector(value["position"], 3, "obstacle.position"),
            size=_vector(value["size"], 3, "obstacle.size"),
            yaw=float(value.get("yaw", 0.0)),
            velocity=_vector(value.get("velocity", [0.0, 0.0, 0.0]), 3, "obstacle.velocity"),
            pos_uncertainty=float(value.get("pos_uncertainty", 0.05)),
            vel_uncertainty=float(value.get("vel_uncertainty", 0.10)),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "position": self.position.tolist(),
            "size": self.size.tolist(),
            "yaw": self.yaw,
            "velocity": self.velocity.tolist(),
            "pos_uncertainty": self.pos_uncertainty,
            "vel_uncertainty": self.vel_uncertainty,
        }


@dataclass(slots=True)
class ScenarioConfig:
    name: str
    start: Array
    goal: Array
    obstacles: list[ObstacleConfig]
    sim_timestep: float = 0.04
    max_time: float = 12.0
    goal_threshold: float = 0.30
    stop_speed: float = 0.35
    seed: int = 1
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    model_mismatch: ModelMismatchConfig = field(default_factory=ModelMismatchConfig)

    @classmethod
    def from_mapping(cls, value: dict[str, Any], name: str = "scenario") -> "ScenarioConfig":
        noise = NoiseConfig(**value.get("noise", {}))
        mismatch = ModelMismatchConfig(**value.get("model_mismatch", {}))
        return cls(
            name=str(value.get("name", name)),
            start=_vector(value["start"], 9, "start"),
            goal=_vector(value["goal"], 3, "goal"),
            obstacles=[ObstacleConfig.from_mapping(item) for item in value.get("obstacles", [])],
            sim_timestep=float(value.get("sim_timestep", 0.04)),
            max_time=float(value.get("max_time", 12.0)),
            goal_threshold=float(value.get("goal_threshold", 0.30)),
            stop_speed=float(value.get("stop_speed", 0.35)),
            seed=int(value.get("seed", 1)),
            noise=noise,
            model_mismatch=mismatch,
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a YAML/JSON-safe representation for manifests and editors."""
        return {
            "name": self.name,
            "start": self.start.tolist(),
            "goal": self.goal.tolist(),
            "sim_timestep": self.sim_timestep,
            "max_time": self.max_time,
            "goal_threshold": self.goal_threshold,
            "stop_speed": self.stop_speed,
            "seed": self.seed,
            "noise": {
                "measurement_pos": self.noise.measurement_pos,
                "measurement_vel": self.noise.measurement_vel,
                "measurement_att": self.noise.measurement_att,
                "process_vel": self.noise.process_vel,
                "process_att": self.noise.process_att,
            },
            "model_mismatch": {
                "drag_scale": self.model_mismatch.drag_scale,
                "attitude_time_scale": self.model_mismatch.attitude_time_scale,
                "vertical_time_scale": self.model_mismatch.vertical_time_scale,
            },
            "obstacles": [item.to_mapping() for item in self.obstacles],
        }


def load_scenario(path: str | Path) -> ScenarioConfig:
    """Load and validate one scenario YAML file."""
    scenario_path = Path(path)
    with scenario_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"scenario must be a YAML mapping: {scenario_path}")
    return ScenarioConfig.from_mapping(data, scenario_path.stem)
