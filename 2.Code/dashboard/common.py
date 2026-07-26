"""Shared helpers for Streamlit dashboard pages."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from resource_paths import resource_root
from simulation.config import ScenarioConfig, load_scenario
from simulation.runner import SimulationResult, SimulationRunner

DATA_ROOT = resource_root()


def scenario_files() -> dict[str, Path]:
    return {
        path.stem: path
        for path in sorted((DATA_ROOT / "config/scenarios").glob("*.yaml"))
    }


def load_named_scenario(name: str) -> ScenarioConfig:
    return load_scenario(scenario_files()[name])


def run_many(
    scenario: ScenarioConfig,
    modes: list[str],
    backend: str,
    trials: int,
    seed: int,
    delta: float | None = None,
    fov: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> list[SimulationResult]:
    override: dict = {}
    if delta is not None:
        override.setdefault("controller", {}).setdefault("obstacle", {})["delta"] = delta
    if fov:
        override.setdefault("controller", {}).setdefault("fov", {})["enabled"] = True
    results = []
    total = len(modes) * trials
    done = 0
    for trial in range(trials):
        for mode in modes:
            results.append(SimulationRunner(
                scenario,
                DATA_ROOT / "config/controller.yaml",
                mode=mode,
                backend=backend,
                seed=seed + trial,
                controller_override=override,
            ).run())
            done += 1
            if progress is not None:
                progress(done, total)
    return results


def metrics_frame(results: list[SimulationResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {"controller": result.mode, "seed": result.seed, "backend": result.backend}
        row.update(result.metrics.to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_frame(results: list[SimulationResult]) -> pd.DataFrame:
    frame = metrics_frame(results)
    numeric = frame.select_dtypes(include="number").columns.difference(["seed"])
    means = frame.groupby("controller")[numeric].mean()
    rates = frame.groupby("controller")[["success", "collision"]].mean()
    return rates.join(means, how="outer").reset_index()


def render_kpis(st, result: SimulationResult) -> None:
    metrics = result.metrics
    columns = st.columns(6)
    columns[0].metric("Status", "SUCCESS" if metrics.success else "FAILED")
    columns[1].metric("Final error", f"{metrics.final_error_m:.3f} m")
    columns[2].metric(
        "Min clearance",
        "N/A" if metrics.min_clearance_m is None else f"{metrics.min_clearance_m:.3f} m",
    )
    columns[3].metric("Collision", "YES" if metrics.collision else "NO")
    columns[4].metric("Solver p95", f"{metrics.p95_solver_ms:.1f} ms")
    columns[5].metric("Deadline miss", f"{metrics.deadline_miss_rate:.1%}")
