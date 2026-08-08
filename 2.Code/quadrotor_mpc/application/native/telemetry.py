"""Bounded native telemetry, deterministic recording and replay loading."""

from __future__ import annotations

import csv
import json
import re
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True, slots=True)
class RecordingOptions:
    enabled: bool = True
    output_dir: str = "outputs/native"
    max_buffer_samples: int = 5000

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RecordingOptions:
        maximum = int(raw.get("max_buffer_samples", 5000))
        if maximum < 10:
            raise ValueError("recording.max_buffer_samples must be >= 10")
        return cls(
            enabled=bool(raw.get("enabled", True)),
            output_dir=str(raw.get("output_dir", "outputs/native")),
            max_buffer_samples=maximum,
        )


class TelemetryBuffer:
    """A bounded, locked buffer suitable for readers at a lower UI rate."""

    def __init__(self, max_samples: int = 5000):
        self._samples: deque[dict[str, Any]] = deque(maxlen=max_samples)
        self._lock = threading.Lock()

    def append(self, sample: Mapping[str, Any]) -> None:
        with self._lock:
            self._samples.append(dict(sample))

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._samples)

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()


def step_to_sample(step: Any) -> dict[str, Any]:
    state = np.asarray(step.state_13, dtype=float)
    control = np.asarray(step.control, dtype=float)
    estimated = getattr(step, "estimated_state_13", None)
    covariance = getattr(step, "error_covariance_12", None)
    estimation_error_norm = None
    position_sigma = None
    if estimated is not None:
        estimated = np.asarray(estimated, dtype=float)
        estimation_error_norm = float(np.linalg.norm(estimated[:6] - state[:6]))
    if covariance is not None:
        covariance = np.asarray(covariance, dtype=float)
        position_sigma = np.sqrt(np.maximum(np.diag(covariance)[:3], 0.0)).tolist()
    predicted_covariance = getattr(step, "predicted_covariances", None)
    horizon_terminal_position_sigma = None
    horizon_max_position_sigma = None
    horizon_terminal_covariance_trace = None
    if predicted_covariance is not None:
        predicted_covariance = np.asarray(predicted_covariance, dtype=float)
        if predicted_covariance.size:
            position_variances = np.maximum(
                np.diagonal(predicted_covariance[:, :3, :3], axis1=1, axis2=2),
                0.0,
            )
            horizon_terminal_position_sigma = np.sqrt(position_variances[-1]).tolist()
            horizon_max_position_sigma = np.sqrt(np.max(position_variances, axis=0)).tolist()
            horizon_terminal_covariance_trace = float(np.trace(predicted_covariance[-1]))
    obstacle_measurements = getattr(step, "obstacle_measurement_positions", None)
    serialized_obstacle_measurements = None
    if obstacle_measurements is not None:
        serialized_obstacle_measurements = [
            None if not np.all(np.isfinite(position)) else position.tolist()
            for position in np.asarray(obstacle_measurements, dtype=float)
        ]
    def optional_array(field: str) -> np.ndarray:
        value = getattr(step, field, None)
        return (
            np.empty((0, 0), dtype=float)
            if value is None
            else np.asarray(value, dtype=float)
        )

    chance_residuals = optional_array("chance_margins")
    slacks = optional_array("slacks")
    projected_uncertainties = optional_array("projected_uncertainties")
    tightened_safety_radii = optional_array("tightened_safety_radii")
    return {
        "step_index": int(step.step_index),
        "time_s": float(step.time_s),
        "position": state[:3].tolist(),
        "velocity": state[3:6].tolist(),
        "quaternion": state[6:10].tolist(),
        "body_rate": state[10:13].tolist(),
        "control": control.tolist(),
        "goal_distance_m": float(step.goal_distance_m),
        "min_clearance_m": float(step.min_clearance_m),
        "solver_time_ms": float(step.solver_time_ms),
        "collided": bool(step.collided),
        "paused": bool(getattr(step, "paused", False)),
        "estimated_state": None if estimated is None else estimated.tolist(),
        "position_sigma": position_sigma,
        "horizon_terminal_position_sigma": horizon_terminal_position_sigma,
        "horizon_max_position_sigma": horizon_max_position_sigma,
        "horizon_terminal_covariance_trace": horizon_terminal_covariance_trace,
        "estimation_error_norm": estimation_error_norm,
        "vehicle_measurement_available": bool(getattr(step, "vehicle_measurement_available", True)),
        "obstacle_measurement_available": (
            None
            if getattr(step, "obstacle_measurement_available", None) is None
            else np.asarray(step.obstacle_measurement_available, dtype=bool).tolist()
        ),
        "vehicle_measurement_state": (
            None
            if getattr(step, "vehicle_measurement_state_13", None) is None
            else np.asarray(step.vehicle_measurement_state_13, dtype=float).tolist()
        ),
        "obstacle_measurement_positions": (serialized_obstacle_measurements),
        "solver_status": str(getattr(step, "solver_status", "")),
        "primary_solver_status": str(
            getattr(step, "primary_solver_status", "")
        ),
        "primary_solver_success": bool(
            getattr(step, "primary_solver_success", True)
        ),
        "primary_solver_iterations": int(
            getattr(step, "primary_solver_iterations", 0)
        ),
        "primary_solver_primal_residual": (
            None
            if getattr(step, "primary_solver_primal_residual", None) is None
            else float(step.primary_solver_primal_residual)
        ),
        "primary_solver_dual_residual": (
            None
            if getattr(step, "primary_solver_dual_residual", None) is None
            else float(step.primary_solver_dual_residual)
        ),
        "primary_solver_primal_residual_status": str(
            getattr(step, "primary_solver_primal_residual_status", "UNAVAILABLE")
        ),
        "primary_solver_dual_residual_status": str(
            getattr(step, "primary_solver_dual_residual_status", "UNAVAILABLE")
        ),
        "primary_solver_residual_gate_status": str(
            getattr(step, "primary_solver_residual_gate_status", "UNKNOWN_UNAVAILABLE")
        ),
        "primary_solver_residual_source": str(
            getattr(step, "primary_solver_residual_source", "")
        ),
        "primary_solver_residual_required_for_acceptance": bool(
            getattr(step, "primary_solver_residual_required_for_acceptance", False)
        ),
        "primary_solver_residual_required_for_assurance": bool(
            getattr(step, "primary_solver_residual_required_for_assurance", True)
        ),
        "command_source": str(
            getattr(step, "command_source", "PRIMARY_NMPC")
        ),
        "solution_accepted": bool(
            getattr(step, "solution_accepted", True)
        ),
        "fallback_active": bool(getattr(step, "fallback_active", False)),
        "fallback_level": int(getattr(step, "fallback_level", 0)),
        "fallback_reason": str(getattr(step, "fallback_reason", "")),
        "consecutive_rejections": int(
            getattr(step, "consecutive_rejections", 0)
        ),
        "deadline_missed": bool(getattr(step, "deadline_missed", False)),
        "safety_assurance_status": str(
            getattr(step, "safety_assurance_status", "")
        ),
        "residual_status": str(getattr(step, "residual_status", "UNAVAILABLE")),
        "horizon_assurance_status": str(
            getattr(step, "horizon_assurance_status", "")
        ),
        "horizon_assurance_eligible": bool(
            getattr(step, "horizon_assurance_eligible", False)
        ),
        "horizon_assurance_reason": str(
            getattr(step, "horizon_assurance_reason", "")
        ),
        "horizon_assurance_failed_checks": list(
            getattr(step, "horizon_assurance_failed_checks", ())
        ),
        "assurance_schema_version": int(getattr(step, "assurance_schema_version", 3)),
        "risk_semantics": str(getattr(step, "risk_semantics", "")),
        "risk_allocation_method": str(
            getattr(step, "risk_allocation_method", "")
        ),
        "risk_budget_total": getattr(step, "risk_budget_total", None),
        "risk_budget_allocated": float(
            getattr(step, "risk_budget_allocated", 0.0)
        ),
        "risk_budget_remaining": getattr(
            step,
            "risk_budget_remaining",
            None,
        ),
        "risk_constraint_count": int(
            getattr(step, "risk_constraint_count", 0)
        ),
        "risk_budget_status": str(
            getattr(step, "risk_budget_status", "")
        ),
        "minimum_chance_residual_m": (
            None if chance_residuals.size == 0 else float(np.min(chance_residuals))
        ),
        "maximum_slack_m": None if slacks.size == 0 else float(np.max(slacks)),
        "maximum_projected_uncertainty_m": (
            None
            if projected_uncertainties.size == 0
            else float(np.max(projected_uncertainties))
        ),
        "maximum_tightened_safety_radius_m": (
            None
            if tightened_safety_radii.size == 0
            else float(np.max(tightened_safety_radii))
        ),
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower() or "run"


def _reason_counts(samples: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        reason = str(sample.get("horizon_assurance_reason", ""))
        if not reason or reason == "eligible":
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return counts


class NativeRunRecorder:
    """Collect one run and export a self-contained replay directory on close."""

    def __init__(
        self,
        options: RecordingOptions,
        scenario_name: str,
        scenario_mapping: Mapping[str, Any],
        *,
        base_dir: str | Path,
    ):
        self.options = options
        self.scenario_name = scenario_name
        self.scenario_mapping = dict(scenario_mapping)
        output_root = Path(options.output_dir)
        if not output_root.is_absolute():
            output_root = Path(base_dir) / output_root
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = output_root / f"{stamp}_{_slug(scenario_name)}"
        self.samples: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.state_history: list[np.ndarray] = []
        self.position_predictions: list[np.ndarray] = []
        self.obstacle_predictions: list[np.ndarray] = []
        self.estimated_states: list[np.ndarray] = []
        self.error_covariances: list[np.ndarray] = []
        self.estimated_obstacle_states: list[np.ndarray] = []
        self.obstacle_covariances: list[np.ndarray] = []
        self.estimated_obstacle_predictions: list[np.ndarray] = []
        self.vehicle_measurements: list[np.ndarray] = []
        self.obstacle_measurements: list[np.ndarray] = []
        self.predicted_error_covariance_horizons: list[np.ndarray] = []
        self.predicted_obstacle_covariance_horizons: list[np.ndarray] = []
        self.chance_residual_horizons: list[np.ndarray] = []
        self.risk_allocation_horizons: list[np.ndarray] = []
        self.slack_horizons: list[np.ndarray] = []
        self.projected_uncertainty_horizons: list[np.ndarray] = []
        self.tightened_safety_radius_horizons: list[np.ndarray] = []
        self._finalized = False

    def record_step(self, step: Any, sample: Mapping[str, Any]) -> None:
        if not self.options.enabled:
            return
        self.samples.append(dict(sample))
        self.state_history.append(np.asarray(step.state_13, dtype=float).copy())
        predicted = getattr(step, "predicted_positions", None)
        self.position_predictions.append(
            np.empty((0, 3), dtype=float)
            if predicted is None
            else np.asarray(predicted, dtype=float).copy()
        )
        self.obstacle_predictions.append(np.asarray(step.obstacle_predictions, dtype=float).copy())
        self.estimated_states.append(
            np.asarray(getattr(step, "estimated_state_13", step.state_13), dtype=float).copy()
        )
        covariance = getattr(step, "error_covariance_12", None)
        self.error_covariances.append(
            np.zeros((12, 12), dtype=float)
            if covariance is None
            else np.asarray(covariance, dtype=float).copy()
        )
        obstacle_states = getattr(step, "estimated_obstacle_states", None)
        self.estimated_obstacle_states.append(
            np.empty((0, 6), dtype=float)
            if obstacle_states is None
            else np.asarray(obstacle_states, dtype=float).copy()
        )
        obstacle_covariance = getattr(step, "obstacle_covariances", None)
        self.obstacle_covariances.append(
            np.empty((0, 6, 6), dtype=float)
            if obstacle_covariance is None
            else np.asarray(obstacle_covariance, dtype=float).copy()
        )
        estimated_predictions = getattr(step, "estimated_obstacle_predictions", None)
        self.estimated_obstacle_predictions.append(
            np.asarray(step.obstacle_predictions, dtype=float).copy()
            if estimated_predictions is None
            else np.asarray(estimated_predictions, dtype=float).copy()
        )
        vehicle_measurement = getattr(step, "vehicle_measurement_state_13", None)
        self.vehicle_measurements.append(
            np.full(13, np.nan)
            if vehicle_measurement is None
            else np.asarray(vehicle_measurement, dtype=float).copy()
        )
        obstacle_measurements = getattr(step, "obstacle_measurement_positions", None)
        self.obstacle_measurements.append(
            np.empty((0, 3), dtype=float)
            if obstacle_measurements is None
            else np.asarray(obstacle_measurements, dtype=float).copy()
        )
        predicted_covariance = getattr(step, "predicted_covariances", None)
        self.predicted_error_covariance_horizons.append(
            np.empty((0, 12, 12), dtype=float)
            if predicted_covariance is None
            else np.asarray(predicted_covariance, dtype=float).copy()
        )
        predicted_obstacle_covariance = getattr(
            step,
            "predicted_obstacle_covariances",
            None,
        )
        self.predicted_obstacle_covariance_horizons.append(
            np.empty((0, 0, 6, 6), dtype=float)
            if predicted_obstacle_covariance is None
            else np.asarray(predicted_obstacle_covariance, dtype=float).copy()
        )
        for collection, field in (
            (self.chance_residual_horizons, "chance_margins"),
            (self.risk_allocation_horizons, "risk_allocations"),
            (self.slack_horizons, "slacks"),
            (self.projected_uncertainty_horizons, "projected_uncertainties"),
            (self.tightened_safety_radius_horizons, "tightened_safety_radii"),
        ):
            value = getattr(step, field, None)
            collection.append(
                np.empty((0, 0), dtype=float)
                if value is None
                else np.asarray(value, dtype=float).copy()
            )

    def record_event(
        self, name: str, time_s: float, *, source: str = "runtime", payload: Any = None
    ) -> None:
        if self.options.enabled:
            self.events.append(
                {
                    "time_s": float(time_s),
                    "name": str(name),
                    "source": str(source),
                    "payload": {} if payload is None else payload,
                }
            )

    def reset_episode(self) -> None:
        """Discard pre-reset samples so the exported trajectory stays monotonic."""
        self.samples.clear()
        self.state_history.clear()
        self.position_predictions.clear()
        self.obstacle_predictions.clear()
        self.estimated_states.clear()
        self.error_covariances.clear()
        self.estimated_obstacle_states.clear()
        self.obstacle_covariances.clear()
        self.estimated_obstacle_predictions.clear()
        self.vehicle_measurements.clear()
        self.obstacle_measurements.clear()
        self.predicted_error_covariance_horizons.clear()
        self.predicted_obstacle_covariance_horizons.clear()
        self.chance_residual_horizons.clear()
        self.risk_allocation_horizons.clear()
        self.slack_horizons.clear()
        self.projected_uncertainty_horizons.clear()
        self.tightened_safety_radius_horizons.clear()

    def write_snapshot(self, sample: Mapping[str, Any] | None) -> Path | None:
        if not self.options.enabled or sample is None:
            return None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        index = len(list(self.run_dir.glob("snapshot-*.json"))) + 1
        path = self.run_dir / f"snapshot-{index:03d}.json"
        path.write_text(json.dumps(dict(sample), indent=2), encoding="utf-8")
        return path

    def finalize(self, result: Mapping[str, Any]) -> Path | None:
        if not self.options.enabled or self._finalized:
            return self.run_dir if self._finalized else None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "scenario.yaml").write_text(
            yaml.safe_dump(self.scenario_mapping, sort_keys=False), encoding="utf-8"
        )
        self._write_csv(self.run_dir / "telemetry.csv")
        with (self.run_dir / "events.jsonl").open("w", encoding="utf-8") as stream:
            for event in self.events:
                stream.write(json.dumps(event, separators=(",", ":")) + "\n")
        position_predictions = _stack_or_empty(self.position_predictions, trailing_rank=2)
        obstacle_predictions = _stack_or_empty(self.obstacle_predictions, trailing_rank=3)
        np.savez_compressed(
            self.run_dir / "trajectory.npz",
            state_13=np.asarray(self.state_history, dtype=float),
            predicted_positions=position_predictions,
            obstacle_predictions=obstacle_predictions,
            estimated_state_13=np.asarray(self.estimated_states, dtype=float),
            error_covariance_12=np.asarray(self.error_covariances, dtype=float),
            estimated_obstacle_state_6=_stack_or_empty(
                self.estimated_obstacle_states, trailing_rank=2
            ),
            obstacle_covariance_6=_stack_or_empty(self.obstacle_covariances, trailing_rank=3),
            estimated_obstacle_predictions=_stack_or_empty(
                self.estimated_obstacle_predictions, trailing_rank=3
            ),
            vehicle_measurement_state_13=np.asarray(self.vehicle_measurements, dtype=float),
            obstacle_measurement_positions=_stack_or_empty(
                self.obstacle_measurements, trailing_rank=2
            ),
            predicted_error_covariance_horizon=_stack_or_empty(
                self.predicted_error_covariance_horizons,
                trailing_rank=3,
            ),
            predicted_obstacle_covariance_horizon=_stack_or_empty(
                self.predicted_obstacle_covariance_horizons,
                trailing_rank=4,
            ),
            chance_residual_horizon=_stack_or_empty(
                self.chance_residual_horizons,
                trailing_rank=2,
            ),
            risk_allocation_horizon=_stack_or_empty(
                self.risk_allocation_horizons,
                trailing_rank=2,
            ),
            slack_horizon=_stack_or_empty(self.slack_horizons, trailing_rank=2),
            projected_uncertainty_horizon=_stack_or_empty(
                self.projected_uncertainty_horizons,
                trailing_rank=2,
            ),
            tightened_safety_radius_horizon=_stack_or_empty(
                self.tightened_safety_radius_horizons,
                trailing_rank=2,
            ),
        )
        summary = {
            "scenario": self.scenario_name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "samples": len(self.samples),
            "termination_reason": str(result.get("termination_reason", "unknown")),
            "collision": bool(result.get("collided", False)),
            "min_clearance_m": (
                float(np.min(result["clearance"])) if len(result.get("clearance", [])) else None
            ),
            "risk_semantics": (
                self.samples[-1].get("risk_semantics", "")
                if self.samples
                else ""
            ),
            "risk_allocation_method": (
                self.samples[-1].get("risk_allocation_method", "")
                if self.samples
                else ""
            ),
            "risk_budget_total": (
                self.samples[-1].get("risk_budget_total")
                if self.samples
                else None
            ),
            "risk_budget_status": (
                self.samples[-1].get("risk_budget_status", "")
                if self.samples
                else ""
            ),
            "maximum_slack_m": (
                max(
                    (
                        float(sample["maximum_slack_m"])
                        for sample in self.samples
                        if sample.get("maximum_slack_m") is not None
                    ),
                    default=None,
                )
            ),
            "fallback_activations": sum(
                1
                for sample in self.samples
                if sample.get("fallback_active", False)
            ),
            "deadline_misses": sum(
                1
                for sample in self.samples
                if sample.get("deadline_missed", False)
            ),
            "rejected_primary_solutions": sum(
                1
                for sample in self.samples
                if not sample.get("solution_accepted", True)
            ),
            "horizon_eligible_tick_count": sum(
                1
                for sample in self.samples
                if sample.get("horizon_assurance_eligible", False)
            ),
            "horizon_eligible_tick_rate": (
                sum(
                    1
                    for sample in self.samples
                    if sample.get("horizon_assurance_eligible", False)
                )
                / len(self.samples)
                if self.samples
                else 0.0
            ),
            "horizon_ineligible_reason_counts": _reason_counts(
                self.samples
            ),
            "episode_all_ticks_horizon_eligible": bool(
                self.samples
                and all(
                    sample.get("horizon_assurance_eligible", False)
                    for sample in self.samples
                )
            ),
            "episode_any_fallback": any(
                sample.get("fallback_active", False) for sample in self.samples
            ),
            "episode_any_positive_slack": any(
                sample.get("maximum_slack_m") is not None
                and float(sample["maximum_slack_m"])
                > float(sample.get("slack_tolerance_m", 0.0))
                for sample in self.samples
            ),
            "episode_any_deadline_miss": any(
                sample.get("deadline_missed", False) for sample in self.samples
            ),
            "residual_available_count": sum(
                1
                for sample in self.samples
                if sample.get("primary_solver_primal_residual_status", "")
                == "AVAILABLE"
            ),
            "residual_unavailable_count": sum(
                1
                for sample in self.samples
                if sample.get("primary_solver_primal_residual_status", "")
                == "UNAVAILABLE"
            ),
            "residual_invalid_count": sum(
                1
                for sample in self.samples
                if sample.get("primary_solver_primal_residual_status", "")
                == "INVALID"
            ),
            "residual_gate_pass_rate": (
                sum(
                    1
                    for sample in self.samples
                    if sample.get("primary_solver_residual_gate_status")
                    == "PASS"
                )
                / len(self.samples)
                if self.samples
                else 0.0
            ),
            "residual_gate_unknown_rate": (
                sum(
                    1
                    for sample in self.samples
                    if sample.get("primary_solver_residual_gate_status")
                    == "UNKNOWN_UNAVAILABLE"
                )
                / len(self.samples)
                if self.samples
                else 0.0
            ),
            "maximum_primary_solver_primal_residual": max(
                (
                    float(sample["primary_solver_primal_residual"])
                    for sample in self.samples
                    if sample.get("primary_solver_primal_residual") is not None
                    and sample.get("primary_solver_primal_residual_status")
                    == "AVAILABLE"
                ),
                default=None,
            ),
            "maximum_primary_solver_dual_residual": max(
                (
                    float(sample["primary_solver_dual_residual"])
                    for sample in self.samples
                    if sample.get("primary_solver_dual_residual") is not None
                    and sample.get("primary_solver_dual_residual_status")
                    == "AVAILABLE"
                ),
                default=None,
            ),
        }
        (self.run_dir / "summary.yaml").write_text(
            yaml.safe_dump(summary, sort_keys=False), encoding="utf-8"
        )
        self._finalized = True
        return self.run_dir

    def _write_csv(self, path: Path) -> None:
        fields = [
            "step_index",
            "time_s",
            "px",
            "py",
            "pz",
            "vx",
            "vy",
            "vz",
            "qw",
            "qx",
            "qy",
            "qz",
            "wx",
            "wy",
            "wz",
            "thrust_deviation",
            "tau_x",
            "tau_y",
            "tau_z",
            "goal_distance_m",
            "min_clearance_m",
            "solver_time_ms",
            "solver_status",
            "primary_solver_status",
            "primary_solver_success",
            "primary_solver_iterations",
            "primary_solver_primal_residual",
            "primary_solver_dual_residual",
            "primary_solver_primal_residual_status",
            "primary_solver_dual_residual_status",
            "primary_solver_residual_gate_status",
            "primary_solver_residual_source",
            "command_source",
            "solution_accepted",
            "fallback_active",
            "fallback_level",
            "fallback_reason",
            "consecutive_rejections",
            "deadline_missed",
            "safety_assurance_status",
            "residual_status",
            "horizon_assurance_status",
            "horizon_assurance_eligible",
            "horizon_assurance_reason",
            "horizon_assurance_failed_checks",
            "assurance_schema_version",
            "risk_semantics",
            "risk_allocation_method",
            "risk_budget_total",
            "risk_budget_allocated",
            "risk_budget_remaining",
            "risk_constraint_count",
            "risk_budget_status",
            "minimum_chance_residual_m",
            "maximum_slack_m",
            "maximum_projected_uncertainty_m",
            "maximum_tightened_safety_radius_m",
            "collided",
        ]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for sample in self.samples:
                position = sample["position"]
                velocity = sample["velocity"]
                quaternion = sample["quaternion"]
                body_rate = sample["body_rate"]
                control = sample["control"]
                writer.writerow(
                    {
                        "step_index": sample["step_index"],
                        "time_s": sample["time_s"],
                        "px": position[0],
                        "py": position[1],
                        "pz": position[2],
                        "vx": velocity[0],
                        "vy": velocity[1],
                        "vz": velocity[2],
                        "qw": quaternion[0],
                        "qx": quaternion[1],
                        "qy": quaternion[2],
                        "qz": quaternion[3],
                        "wx": body_rate[0],
                        "wy": body_rate[1],
                        "wz": body_rate[2],
                        "thrust_deviation": control[0],
                        "tau_x": control[1],
                        "tau_y": control[2],
                        "tau_z": control[3],
                        "goal_distance_m": sample["goal_distance_m"],
                        "min_clearance_m": sample["min_clearance_m"],
                        "solver_time_ms": sample["solver_time_ms"],
                        "solver_status": sample.get("solver_status", ""),
                        "primary_solver_status": sample.get(
                            "primary_solver_status",
                            "",
                        ),
                        "primary_solver_success": int(
                            sample.get("primary_solver_success", True)
                        ),
                        "primary_solver_iterations": sample.get(
                            "primary_solver_iterations",
                            0,
                        ),
                        "primary_solver_primal_residual": (
                            ""
                            if sample.get("primary_solver_primal_residual") is None
                            else sample["primary_solver_primal_residual"]
                        ),
                        "primary_solver_dual_residual": (
                            ""
                            if sample.get("primary_solver_dual_residual") is None
                            else sample["primary_solver_dual_residual"]
                        ),
                        "primary_solver_primal_residual_status": sample.get(
                            "primary_solver_primal_residual_status",
                            "UNAVAILABLE",
                        ),
                        "primary_solver_dual_residual_status": sample.get(
                            "primary_solver_dual_residual_status",
                            "UNAVAILABLE",
                        ),
                        "primary_solver_residual_gate_status": sample.get(
                            "primary_solver_residual_gate_status",
                            "UNKNOWN_UNAVAILABLE",
                        ),
                        "primary_solver_residual_source": sample.get(
                            "primary_solver_residual_source",
                            "",
                        ),
                        "command_source": sample.get(
                            "command_source",
                            "PRIMARY_NMPC",
                        ),
                        "solution_accepted": int(
                            sample.get("solution_accepted", True)
                        ),
                        "fallback_active": int(
                            sample.get("fallback_active", False)
                        ),
                        "fallback_level": sample.get("fallback_level", 0),
                        "fallback_reason": sample.get("fallback_reason", ""),
                        "consecutive_rejections": sample.get(
                            "consecutive_rejections",
                            0,
                        ),
                        "deadline_missed": int(
                            sample.get("deadline_missed", False)
                        ),
                        "safety_assurance_status": sample.get(
                            "safety_assurance_status",
                            "",
                        ),
                        "residual_status": sample.get(
                            "residual_status",
                            "UNAVAILABLE",
                        ),
                        "horizon_assurance_status": sample.get(
                            "horizon_assurance_status",
                            "",
                        ),
                        "horizon_assurance_eligible": int(
                            sample.get("horizon_assurance_eligible", False)
                        ),
                        "horizon_assurance_reason": sample.get(
                            "horizon_assurance_reason",
                            "",
                        ),
                        "horizon_assurance_failed_checks": ";".join(
                            sample.get("horizon_assurance_failed_checks", ())
                        ),
                        "assurance_schema_version": int(
                            sample.get("assurance_schema_version", 3)
                        ),
                        "risk_semantics": sample.get("risk_semantics", ""),
                        "risk_allocation_method": sample.get(
                            "risk_allocation_method",
                            "",
                        ),
                        "risk_budget_total": sample.get("risk_budget_total"),
                        "risk_budget_allocated": sample.get(
                            "risk_budget_allocated",
                            0.0,
                        ),
                        "risk_budget_remaining": sample.get(
                            "risk_budget_remaining"
                        ),
                        "risk_constraint_count": sample.get(
                            "risk_constraint_count",
                            0,
                        ),
                        "risk_budget_status": sample.get(
                            "risk_budget_status",
                            "",
                        ),
                        "minimum_chance_residual_m": sample.get(
                            "minimum_chance_residual_m"
                        ),
                        "maximum_slack_m": sample.get("maximum_slack_m"),
                        "maximum_projected_uncertainty_m": sample.get(
                            "maximum_projected_uncertainty_m"
                        ),
                        "maximum_tightened_safety_radius_m": sample.get(
                            "maximum_tightened_safety_radius_m"
                        ),
                        "collided": int(sample["collided"]),
                    }
                )


def load_native_recording(run_dir: str | Path) -> dict[str, Any]:
    """Load a recorder directory with pickle disabled for untrusted arrays."""
    source = Path(run_dir)
    scenario = yaml.safe_load((source / "scenario.yaml").read_text(encoding="utf-8"))
    with (source / "telemetry.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    # State is stored numerically in CSV so replay never needs object/pickle arrays.
    states = np.asarray(
        [
            [
                *[float(row[key]) for key in ("px", "py", "pz", "vx", "vy", "vz")],
                *[float(row[key]) for key in ("qw", "qx", "qy", "qz")],
                *[float(row[key]) for key in ("wx", "wy", "wz")],
            ]
            for row in rows
        ],
        dtype=float,
    )
    with np.load(source / "trajectory.npz", allow_pickle=False) as arrays:
        predicted_positions = arrays["predicted_positions"].copy()
        obstacle_predictions = arrays["obstacle_predictions"].copy()
        estimated_states = (
            arrays["estimated_state_13"].copy() if "estimated_state_13" in arrays else states.copy()
        )
        error_covariances = (
            arrays["error_covariance_12"].copy()
            if "error_covariance_12" in arrays
            else np.zeros((len(states), 12, 12), dtype=float)
        )
        estimated_obstacle_predictions = (
            arrays["estimated_obstacle_predictions"].copy()
            if "estimated_obstacle_predictions" in arrays
            else obstacle_predictions.copy()
        )
        vehicle_measurements = (
            arrays["vehicle_measurement_state_13"].copy()
            if "vehicle_measurement_state_13" in arrays
            else states.copy()
        )
        obstacle_measurements = (
            arrays["obstacle_measurement_positions"].copy()
            if "obstacle_measurement_positions" in arrays
            else obstacle_predictions[:, :, 0, :].copy()
        )
        predicted_error_covariance_horizons = (
            arrays["predicted_error_covariance_horizon"].copy()
            if "predicted_error_covariance_horizon" in arrays
            else np.empty((len(states), 0, 12, 12), dtype=float)
        )
        predicted_obstacle_covariance_horizons = (
            arrays["predicted_obstacle_covariance_horizon"].copy()
            if "predicted_obstacle_covariance_horizon" in arrays
            else np.empty((len(states), 0, 0, 6, 6), dtype=float)
        )
        chance_residual_horizons = (
            arrays["chance_residual_horizon"].copy()
            if "chance_residual_horizon" in arrays
            else np.empty((len(states), 0, 0), dtype=float)
        )
        risk_allocation_horizons = (
            arrays["risk_allocation_horizon"].copy()
            if "risk_allocation_horizon" in arrays
            else np.empty((len(states), 0, 0), dtype=float)
        )
        slack_horizons = (
            arrays["slack_horizon"].copy()
            if "slack_horizon" in arrays
            else np.empty((len(states), 0, 0), dtype=float)
        )
        projected_uncertainty_horizons = (
            arrays["projected_uncertainty_horizon"].copy()
            if "projected_uncertainty_horizon" in arrays
            else np.empty((len(states), 0, 0), dtype=float)
        )
        tightened_safety_radius_horizons = (
            arrays["tightened_safety_radius_horizon"].copy()
            if "tightened_safety_radius_horizon" in arrays
            else np.empty((len(states), 0, 0), dtype=float)
        )
    return {
        "directory": source,
        "scenario": scenario,
        "rows": rows,
        "states": states,
        "predicted_positions": predicted_positions,
        "obstacle_predictions": obstacle_predictions,
        "estimated_states": estimated_states,
        "error_covariances": error_covariances,
        "estimated_obstacle_predictions": estimated_obstacle_predictions,
        "vehicle_measurements": vehicle_measurements,
        "obstacle_measurements": obstacle_measurements,
        "predicted_error_covariance_horizons": (predicted_error_covariance_horizons),
        "predicted_obstacle_covariance_horizons": (predicted_obstacle_covariance_horizons),
        "chance_residual_horizons": chance_residual_horizons,
        "risk_allocation_horizons": risk_allocation_horizons,
        "slack_horizons": slack_horizons,
        "projected_uncertainty_horizons": projected_uncertainty_horizons,
        "tightened_safety_radius_horizons": tightened_safety_radius_horizons,
    }


def _stack_or_empty(items: list[np.ndarray], *, trailing_rank: int) -> np.ndarray:
    if not items:
        return np.empty((0,) + (0,) * trailing_rank, dtype=float)
    shape = items[0].shape
    if any(item.shape != shape for item in items):
        return np.empty((len(items),) + (0,) * trailing_rank, dtype=float)
    return np.stack(items).astype(float, copy=False)

