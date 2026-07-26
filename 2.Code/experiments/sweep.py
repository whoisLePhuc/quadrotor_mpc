"""Parameter-sweep engine used by the CLI and dashboard."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from simulation.config import ScenarioConfig
from simulation.runner import SimulationResult, SimulationRunner


@dataclass(slots=True)
class SweepOutcome:
    parameter: str
    value: float
    result: SimulationResult


def _configured(
    base: ScenarioConfig,
    parameter: str,
    value: float,
) -> tuple[ScenarioConfig, dict]:
    scenario = ScenarioConfig.from_mapping(copy.deepcopy(base.to_mapping()), base.name)
    override: dict = {}
    if parameter == "measurement_pos":
        scenario.noise.measurement_pos = value
    elif parameter == "process_vel":
        scenario.noise.process_vel = value
    elif parameter == "drag_scale":
        scenario.model_mismatch.drag_scale = value
    elif parameter == "obstacle_speed_scale":
        for original, changed in zip(base.obstacles, scenario.obstacles):
            changed.velocity = original.velocity * value
    elif parameter == "delta":
        if not 0.0 < value <= 0.5:
            raise ValueError("delta must be in (0, 0.5]")
        override = {"controller": {"obstacle": {"delta": value}}}
    elif parameter == "horizon_steps":
        override = {"runtime": {"horizon_steps": int(value)}}
    else:
        raise ValueError(f"unsupported sweep parameter: {parameter}")
    return scenario, override


def run_parameter_sweep(
    scenario: ScenarioConfig,
    controller_config: str | Path,
    parameter: str,
    values: list[float],
    modes: list[str],
    trials: int,
    backend: str = "scipy",
    seed: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[SweepOutcome]:
    if trials < 1:
        raise ValueError("trials must be >= 1")
    first_seed = scenario.seed if seed is None else seed
    total = len(values) * len(modes) * trials
    complete = 0
    outcomes: list[SweepOutcome] = []
    for value in values:
        configured, override = _configured(scenario, parameter, value)
        for trial in range(trials):
            for mode in modes:
                result = SimulationRunner(
                    configured,
                    controller_config,
                    mode=mode,
                    backend=backend,
                    seed=first_seed + trial,
                    controller_override=override,
                ).run()
                outcomes.append(SweepOutcome(parameter, value, result))
                complete += 1
                if progress is not None:
                    progress(complete, total)
    return outcomes
