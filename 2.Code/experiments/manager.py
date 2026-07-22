"""Reproducible run directories and statistical controller comparisons."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from reporting.interactive import save_interactive_report
from simulation.config import ScenarioConfig
from simulation.runner import SimulationResult
from simulation.visualizer import save_report


@dataclass(slots=True)
class ExperimentArtifacts:
    run_id: str
    directory: Path
    manifest: Path
    metrics: Path
    comparison_csv: Path
    static_report: Path
    interactive_report: Path


def _safe_git_commit(root: Path) -> str | None:
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        return process.stdout.strip() if process.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _versions() -> dict[str, str]:
    packages = ("numpy", "scipy", "PyYAML", "matplotlib", "cvxpy", "plotly", "streamlit")
    result: dict[str, str] = {}
    for package in packages:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return result


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    half = z * np.sqrt(
        proportion * (1.0 - proportion) / total + z**2 / (4.0 * total**2)
    ) / denominator
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def _numeric_summary(values: list[float]) -> dict[str, float] | None:
    array = np.asarray(values, dtype=float)
    if not array.size:
        return None
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def aggregate_results(results: list[SimulationResult]) -> dict[str, Any]:
    """Aggregate runs by controller mode with confidence intervals."""
    grouped: dict[str, list[SimulationResult]] = {}
    for result in results:
        grouped.setdefault(result.mode, []).append(result)

    output: dict[str, Any] = {}
    metric_names = (
        "final_error_m",
        "tracking_rmse_m",
        "path_length_m",
        "min_clearance_m",
        "control_effort",
        "control_smoothness",
        "p95_solver_ms",
        "deadline_miss_rate",
        "chance_violation_rate",
        "max_chance_slack",
    )
    for mode, items in grouped.items():
        successes = sum(item.metrics.success for item in items)
        collisions = sum(item.metrics.collision for item in items)
        summary: dict[str, Any] = {
            "trials": len(items),
            "success_rate": successes / len(items),
            "success_rate_ci95": _wilson_interval(successes, len(items)),
            "collision_rate": collisions / len(items),
            "collision_rate_ci95": _wilson_interval(collisions, len(items)),
        }
        for name in metric_names:
            values = [
                float(getattr(item.metrics, name))
                for item in items
                if getattr(item.metrics, name) is not None
            ]
            summary[name] = _numeric_summary(values) if values else None
        output[mode] = summary

    if {"deterministic", "ccmpc"}.issubset(grouped):
        deterministic = {item.seed: item for item in grouped["deterministic"]}
        ccmpc = {item.seed: item for item in grouped["ccmpc"]}
        common = sorted(set(deterministic) & set(ccmpc))
        output["paired_comparison"] = {
            "paired_seeds": common,
            "ccmpc_minus_deterministic": {
                name: _numeric_summary([
                    float(getattr(ccmpc[seed].metrics, name))
                    - float(getattr(deterministic[seed].metrics, name))
                    for seed in common
                    if getattr(ccmpc[seed].metrics, name) is not None
                    and getattr(deterministic[seed].metrics, name) is not None
                ])
                for name in metric_names
            },
        }
    return output


def _comparison_rows(results: list[SimulationResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {
            "scenario": result.scenario_name,
            "mode": result.mode,
            "backend": result.backend,
            "seed": result.seed,
        }
        row.update(result.metrics.to_dict())
        rows.append(row)
    return rows


def save_experiment(
    results: list[SimulationResult],
    scenario: ScenarioConfig,
    controller_config: str | Path,
    output_root: str | Path,
    *,
    command: list[str] | None = None,
    run_id: str | None = None,
) -> ExperimentArtifacts:
    """Persist one single- or multi-controller experiment as a complete run."""
    if not results:
        raise ValueError("at least one result is required")
    root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc)
    if run_id is None:
        modes = "-vs-".join(sorted({item.mode for item in results}))
        run_id = timestamp.strftime("%Y%m%dT%H%M%SZ") + f"-{scenario.name}-{modes}"
    directory = Path(output_root) / "runs" / run_id
    suffix = 1
    while directory.exists():
        directory = Path(output_root) / "runs" / f"{run_id}-{suffix:02d}"
        suffix += 1
    run_id = directory.name
    directory.mkdir(parents=True)

    controller_path = Path(controller_config)
    scenario_snapshot = directory / "scenario.yaml"
    controller_snapshot = directory / "controller.yaml"
    scenario_snapshot.write_text(
        yaml.safe_dump(scenario.to_mapping(), sort_keys=False), encoding="utf-8"
    )
    controller_snapshot.write_text(controller_path.read_text(encoding="utf-8"), encoding="utf-8")

    for result in results:
        result.save(directory)

    aggregate = aggregate_results(results)
    metrics_path = directory / "metrics.json"
    metrics_path.write_text(
        json.dumps(aggregate, indent=2, allow_nan=False), encoding="utf-8"
    )
    rows = _comparison_rows(results)
    comparison_path = directory / "comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": timestamp.isoformat(),
        "scenario": scenario.name,
        "controllers": sorted({item.mode for item in results}),
        "backends": sorted({item.backend for item in results}),
        "seeds": sorted({item.seed for item in results}),
        "controller_timestep_s": sorted({item.controller_dt for item in results}),
        "git_commit": _safe_git_commit(root),
        "command": command if command is not None else sys.argv,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": _versions(),
        },
        "files": {
            "scenario": scenario_snapshot.name,
            "controller": controller_snapshot.name,
            "metrics": metrics_path.name,
            "comparison": comparison_path.name,
            "static_report": "report.png",
            "interactive_report": "report.html",
        },
    }
    manifest_path = directory / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    static_report = save_report(results, scenario, directory / "report.png")
    interactive_report = save_interactive_report(
        results,
        scenario,
        directory / "report.html",
        aggregate=aggregate,
    )
    return ExperimentArtifacts(
        run_id=run_id,
        directory=directory,
        manifest=manifest_path,
        metrics=metrics_path,
        comparison_csv=comparison_path,
        static_report=static_report,
        interactive_report=interactive_report,
    )
