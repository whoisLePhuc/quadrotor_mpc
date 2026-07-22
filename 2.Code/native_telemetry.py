"""Bounded native telemetry, deterministic recording and replay loading."""

from __future__ import annotations

import csv
import json
import re
import threading
from collections import deque
from collections.abc import Mapping
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
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RecordingOptions":
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
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower() or "run"


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
    return {
        "directory": source,
        "scenario": scenario,
        "rows": rows,
        "states": states,
        "predicted_positions": predicted_positions,
        "obstacle_predictions": obstacle_predictions,
    }


def _stack_or_empty(items: list[np.ndarray], *, trailing_rank: int) -> np.ndarray:
    if not items:
        return np.empty((0,) + (0,) * trailing_rank, dtype=float)
    shape = items[0].shape
    if any(item.shape != shape for item in items):
        return np.empty((len(items),) + (0,) * trailing_rank, dtype=float)
    return np.stack(items).astype(float, copy=False)
