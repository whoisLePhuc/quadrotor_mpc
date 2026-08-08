#!/usr/bin/env python3
"""Run a reproducible noise, risk, mismatch or horizon parameter sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quadrotor_mpc.application.experiments.sweep import SweepOutcome, run_parameter_sweep
from quadrotor_mpc.application.simulation.config import load_scenario
from quadrotor_mpc.infrastructure.resources import resolve_input_path, resolve_output_path

PARAMETERS = (
    "measurement_pos",
    "process_vel",
    "drag_scale",
    "obstacle_speed_scale",
    "delta",
    "horizon_steps",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/scenarios/static_obstacle.yaml")
    parser.add_argument("--controller-config", default="config/controller.yaml")
    parser.add_argument("--parameter", choices=PARAMETERS, default="delta")
    parser.add_argument("--values", nargs="+", type=float, required=True)
    parser.add_argument("--mode", choices=("ccmpc", "deterministic", "both"), default="both")
    parser.add_argument("--backend", choices=("scipy", "cvxpy"), default="scipy")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/sweeps")
    return parser


def _rows(outcomes: list[SweepOutcome]) -> list[dict]:
    rows = []
    for outcome in outcomes:
        row = {
            "parameter": outcome.parameter,
            "value": outcome.value,
            "mode": outcome.result.mode,
            "seed": outcome.result.seed,
        }
        row.update(outcome.result.metrics.to_dict())
        rows.append(row)
    return rows


def _aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[float, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((float(row["value"]), str(row["mode"])), []).append(row)
    aggregate = []
    for (value, mode), items in sorted(groups.items()):
        clearances = [
            item["min_clearance_m"] for item in items if item["min_clearance_m"] is not None
        ]
        aggregate.append(
            {
                "value": value,
                "mode": mode,
                "trials": len(items),
                "success_rate": float(np.mean([item["success"] for item in items])),
                "collision_rate": float(np.mean([item["collision"] for item in items])),
                "tracking_rmse_m": float(np.mean([item["tracking_rmse_m"] for item in items])),
                "min_clearance_m": float(np.mean(clearances)) if clearances else None,
                "p95_solver_ms": float(np.mean([item["p95_solver_ms"] for item in items])),
            }
        )
    return aggregate


def _plot(aggregate: list[dict], parameter: str, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fields = (
        ("success_rate", "Success rate"),
        ("collision_rate", "Collision rate"),
        ("tracking_rmse_m", "Tracking RMSE [m]"),
        ("p95_solver_ms", "Solver p95 [ms]"),
    )
    modes = sorted({row["mode"] for row in aggregate})
    for axis, (field, title) in zip(axes.flat, fields):
        for mode in modes:
            items = [row for row in aggregate if row["mode"] == mode]
            axis.plot(
                [row["value"] for row in items],
                [row[field] for row in items],
                marker="o",
                label=mode,
            )
        axis.set(xlabel=parameter, title=title)
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenario = load_scenario(resolve_input_path(args.config))
    modes = ["deterministic", "ccmpc"] if args.mode == "both" else [args.mode]
    outcomes = run_parameter_sweep(
        scenario,
        resolve_input_path(args.controller_config),
        args.parameter,
        args.values,
        modes,
        args.trials,
        backend=args.backend,
        seed=args.seed,
        progress=lambda done, total: print(f"[{done:03d}/{total:03d}] complete"),
    )
    rows = _rows(outcomes)
    aggregate = _aggregate(rows)
    output = resolve_output_path(args.output_dir) / f"{scenario.name}-{args.parameter}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, allow_nan=False), encoding="utf-8"
    )
    _plot(aggregate, args.parameter, output / "sweep.png")
    print(f"sweep: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
