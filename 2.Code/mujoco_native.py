"""Native MuJoCo viewer and configuration for the optional 13-state NMPC track.

The module deliberately keeps MuJoCo imports inside :meth:`NativeMuJoCoViewer.open`
so the deterministic and chance-constrained command-line workflows remain usable
without an OpenGL installation.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import yaml

from native_desktop_panel import DesktopPanelOptions, DesktopPanelProcess
from native_estimation import EstimationOptions
from native_telemetry import (
    NativeRunRecorder,
    RecordingOptions,
    TelemetryBuffer,
    step_to_sample,
)
from obstacle_motion import normalize_obstacle
from runtime_control import CommandName, LocalCommandQueue, RuntimeCommand

if TYPE_CHECKING:
    from mujoco_plant import MuJoCoPlant
    from run_coupled import CoupledRunContext, CoupledStep


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _positive(value: Any, label: str, *, allow_zero: bool = False) -> float:
    number = float(value)
    if number < 0.0 or (not allow_zero and number == 0.0):
        relation = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{label} must be {relation}")
    return number


def _xyz(mapping: Mapping[str, Any], label: str) -> dict[str, float]:
    try:
        return {axis: float(mapping[axis]) for axis in ("x", "y", "z")}
    except KeyError as exc:
        raise ValueError(f"{label} requires x, y and z") from exc


@dataclass(frozen=True, slots=True)
class NativeViewerOptions:
    """Visual and timing settings for the passive desktop viewer."""

    camera_mode: str = "follow"
    distance: float = 7.0
    azimuth: float = 60.0
    elevation: float = -30.0
    realtime_factor: float = 1.0
    show_trail: bool = True
    show_prediction: bool = True
    show_obstacle_prediction: bool = True
    show_safety_envelopes: bool = True
    show_contacts: bool = False
    max_trail_points: int = 600
    max_trail_segments: int = 240
    status_every_steps: int = 10

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "NativeViewerOptions":
        camera_mode = str(raw.get("camera_mode", "follow")).lower()
        if camera_mode not in {"follow", "fixed"}:
            raise ValueError("viewer.camera_mode must be 'follow' or 'fixed'")
        max_trail_points = int(raw.get("max_trail_points", 600))
        max_trail_segments = int(raw.get("max_trail_segments", 240))
        status_every_steps = int(raw.get("status_every_steps", 10))
        if min(max_trail_points, max_trail_segments, status_every_steps) < 1:
            raise ValueError("viewer point, segment and status intervals must be >= 1")
        return cls(
            camera_mode=camera_mode,
            distance=_positive(raw.get("distance", 7.0), "viewer.distance"),
            azimuth=float(raw.get("azimuth", 60.0)),
            elevation=float(raw.get("elevation", -30.0)),
            realtime_factor=_positive(
                raw.get("realtime_factor", 1.0),
                "viewer.realtime_factor",
                allow_zero=True,
            ),
            show_trail=bool(raw.get("show_trail", True)),
            show_prediction=bool(raw.get("show_prediction", True)),
            show_obstacle_prediction=bool(raw.get("show_obstacle_prediction", True)),
            show_safety_envelopes=bool(raw.get("show_safety_envelopes", True)),
            show_contacts=bool(raw.get("show_contacts", False)),
            max_trail_points=max_trail_points,
            max_trail_segments=max_trail_segments,
            status_every_steps=status_every_steps,
        )


@dataclass(frozen=True, slots=True)
class NativeMuJoCoConfig:
    """Validated configuration consumed by ``run_mujoco_native.py``."""

    name: str
    start: dict[str, float]
    goal_position: dict[str, float]
    goal_euler: dict[str, float]
    bounds: dict[str, float]
    obstacles: tuple[dict[str, Any], ...]
    safety_margin: float
    duration_s: float
    mpc_timestep_s: float
    mujoco_timestep_s: float
    horizon_steps: int
    max_solver_iterations: int
    stop_on_goal: bool
    goal_tolerance_m: float
    stop_on_collision: bool
    estimation: EstimationOptions
    viewer: NativeViewerOptions
    panel: DesktopPanelOptions
    recording: RecordingOptions

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "NativeMuJoCoConfig":
        start_raw = _mapping(raw.get("start", {}), "start")
        goal_raw = _mapping(raw.get("goal", {}), "goal")
        position_raw = _mapping(goal_raw.get("position", {}), "goal.position")
        euler_raw = _mapping(goal_raw.get("euler", {}), "goal.euler")
        controller_raw = _mapping(raw.get("controller", {}), "controller")
        bounds_raw = _mapping(controller_raw.get("bounds", {}), "controller.bounds")
        simulation_raw = _mapping(raw.get("simulation", {}), "simulation")
        viewer_raw = _mapping(raw.get("viewer", {}), "viewer")
        panel_raw = _mapping(raw.get("panel", {}), "panel")
        recording_raw = _mapping(raw.get("recording", {}), "recording")
        estimation_raw = _mapping(raw.get("estimation", {}), "estimation")

        start = _xyz(start_raw, "start")
        start.update({axis: float(start_raw.get(axis, 0.0)) for axis in ("roll", "pitch", "yaw")})
        goal_euler = {axis: float(euler_raw.get(axis, 0.0)) for axis in ("roll", "pitch", "yaw")}
        try:
            bounds = {
                "thrust": _positive(bounds_raw["thrust"], "controller.bounds.thrust"),
                "torque_rp": _positive(bounds_raw["torque_rp"], "controller.bounds.torque_rp"),
                "torque_yaw": _positive(bounds_raw["torque_yaw"], "controller.bounds.torque_yaw"),
            }
        except KeyError as exc:
            raise ValueError("controller.bounds requires thrust, torque_rp and torque_yaw") from exc

        obstacles: list[dict[str, Any]] = []
        raw_obstacles = raw.get("obstacles", [])
        if not isinstance(raw_obstacles, Sequence) or isinstance(raw_obstacles, (str, bytes)):
            raise ValueError("obstacles must be a list")
        for index, item in enumerate(raw_obstacles):
            obstacle = _mapping(item, f"obstacles[{index}]")
            obstacles.append(normalize_obstacle(obstacle, index))

        horizon_steps = int(simulation_raw.get("horizon_steps", 20))
        max_solver_iterations = int(simulation_raw.get("max_solver_iterations", 60))
        if horizon_steps < 1 or max_solver_iterations < 1:
            raise ValueError("horizon_steps and max_solver_iterations must be >= 1")

        return cls(
            name=str(raw.get("name", "nmpc-mujoco-native")),
            start=start,
            goal_position=_xyz(position_raw, "goal.position"),
            goal_euler=goal_euler,
            bounds=bounds,
            obstacles=tuple(obstacles),
            safety_margin=_positive(
                controller_raw.get("safety_margin", 0.3),
                "controller.safety_margin",
                allow_zero=True,
            ),
            duration_s=_positive(simulation_raw.get("duration_s", 10.0), "duration_s"),
            mpc_timestep_s=_positive(simulation_raw.get("mpc_timestep_s", 0.05), "mpc_timestep_s"),
            mujoco_timestep_s=_positive(
                simulation_raw.get("mujoco_timestep_s", 0.002), "mujoco_timestep_s"
            ),
            horizon_steps=horizon_steps,
            max_solver_iterations=max_solver_iterations,
            stop_on_goal=bool(simulation_raw.get("stop_on_goal", True)),
            goal_tolerance_m=_positive(
                simulation_raw.get("goal_tolerance_m", 0.25), "goal_tolerance_m"
            ),
            stop_on_collision=bool(simulation_raw.get("stop_on_collision", False)),
            estimation=EstimationOptions.from_mapping(estimation_raw),
            viewer=NativeViewerOptions.from_mapping(viewer_raw),
            panel=DesktopPanelOptions.from_mapping(dict(panel_raw)),
            recording=RecordingOptions.from_mapping(recording_raw),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the effective configuration, including CLI-safe normalized obstacles."""
        return {
            "name": self.name,
            "start": dict(self.start),
            "goal": {
                "position": dict(self.goal_position),
                "euler": dict(self.goal_euler),
            },
            "controller": {
                "bounds": dict(self.bounds),
                "safety_margin": self.safety_margin,
            },
            "simulation": {
                "duration_s": self.duration_s,
                "mpc_timestep_s": self.mpc_timestep_s,
                "mujoco_timestep_s": self.mujoco_timestep_s,
                "horizon_steps": self.horizon_steps,
                "max_solver_iterations": self.max_solver_iterations,
                "stop_on_goal": self.stop_on_goal,
                "goal_tolerance_m": self.goal_tolerance_m,
                "stop_on_collision": self.stop_on_collision,
            },
            "estimation": self.estimation.to_mapping(),
            "viewer": {
                field: getattr(self.viewer, field) for field in self.viewer.__dataclass_fields__
            },
            "panel": {
                field: getattr(self.panel, field) for field in self.panel.__dataclass_fields__
            },
            "recording": {
                field: getattr(self.recording, field)
                for field in self.recording.__dataclass_fields__
            },
            "obstacles": [dict(obstacle) for obstacle in self.obstacles],
        }


def load_native_mujoco_config(path: str | Path) -> NativeMuJoCoConfig:
    """Load and validate a native viewer YAML configuration."""
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"configuration file not found: {source}") from exc
    return NativeMuJoCoConfig.from_mapping(_mapping(raw, "configuration root"))


class NativeMuJoCoViewer:
    """Passive desktop viewer attached to an existing :class:`MuJoCoPlant`.

    Physics and controller execution remain owned by ``run_coupled.py``. This
    class only displays the shared ``MjModel``/``MjData`` pair and returns
    ``False`` when the desktop window has been closed.
    """

    def __init__(
        self,
        options: NativeViewerOptions,
        command_sink: Any | None = None,
    ):
        self.options = options
        self._command_sink = command_sink
        self._mujoco: Any | None = None
        self._session: Any | None = None
        self._viewer: Any | None = None
        self._wall_start = 0.0
        self._idle_wall_time: float | None = None
        self._trail: deque[np.ndarray] = deque(maxlen=options.max_trail_points)
        self._context: CoupledRunContext | None = None
        self._show_trail = options.show_trail
        self._show_prediction = options.show_prediction
        self._show_obstacle_prediction = options.show_obstacle_prediction
        self._show_safety_envelopes = options.show_safety_envelopes
        self._camera_mode = options.camera_mode

    def open(self, plant: "MuJoCoPlant", context: "CoupledRunContext") -> None:
        try:
            import mujoco
            import mujoco.viewer
        except (ModuleNotFoundError, ImportError) as exc:
            raise RuntimeError(
                "Native MuJoCo requires the optional UI dependencies. "
                "Install with `python -m pip install -r requirements-ui.txt`."
            ) from exc

        self._mujoco = mujoco
        self._context = context
        self._session = mujoco.viewer.launch_passive(
            plant.model,
            plant.data,
            key_callback=self._key_callback,
        )
        self._viewer = self._session.__enter__()
        goal = np.asarray(context.goal_position, dtype=float)
        start = np.asarray(context.start_position, dtype=float)
        with self._viewer.lock():
            self._viewer.cam.lookat[:] = (
                start if self._camera_mode == "follow" else 0.5 * (start + goal)
            )
            self._viewer.cam.distance = self.options.distance
            self._viewer.cam.azimuth = self.options.azimuth
            self._viewer.cam.elevation = self.options.elevation
            contact_flag = mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
            self._viewer.opt.flags[contact_flag] = int(self.options.show_contacts)
        self._wall_start = time.perf_counter()
        self._viewer.sync()

    def is_running(self) -> bool:
        return bool(self._viewer is not None and self._viewer.is_running())

    def close(self) -> None:
        if self._session is not None:
            self._session.__exit__(None, None, None)
        self._viewer = None
        self._session = None

    def reset_visuals(self) -> None:
        self._trail.clear()

    def on_idle(self) -> None:
        if self.is_running() and self._viewer is not None:
            now = time.perf_counter()
            if self._idle_wall_time is not None:
                self._wall_start += now - self._idle_wall_time
            self._idle_wall_time = now
            self._viewer.sync()

    def handle_command(self, command: RuntimeCommand) -> None:
        if command.name == CommandName.TOGGLE_TRAIL:
            self._show_trail = not self._show_trail
        elif command.name == CommandName.TOGGLE_PREDICTION:
            self._show_prediction = not self._show_prediction
        elif command.name == CommandName.TOGGLE_SAFETY:
            self._show_safety_envelopes = not self._show_safety_envelopes
        elif command.name == CommandName.TOGGLE_CAMERA:
            self._camera_mode = "fixed" if self._camera_mode == "follow" else "follow"
            if (
                self._viewer is not None
                and self._context is not None
                and self._camera_mode == "fixed"
            ):
                self._viewer.cam.lookat[:] = 0.5 * (
                    self._context.start_position + self._context.goal_position
                )

    def _key_callback(self, keycode: int) -> None:
        if self._command_sink is None:
            return
        mapping = {
            32: CommandName.TOGGLE_PAUSE,
            256: CommandName.STOP,
            ord("N"): CommandName.STEP,
            ord("R"): CommandName.RESET,
            257: CommandName.RUN_AGAIN,
            ord("S"): CommandName.SNAPSHOT,
            ord("T"): CommandName.TOGGLE_TRAIL,
            ord("P"): CommandName.TOGGLE_PREDICTION,
            ord("C"): CommandName.TOGGLE_SAFETY,
            ord("F"): CommandName.TOGGLE_CAMERA,
        }
        name = mapping.get(int(keycode))
        if name is not None:
            self._command_sink(RuntimeCommand(name=name, source="keyboard"))

    def on_step(self, step: "CoupledStep") -> bool:
        if not self.is_running():
            return False
        assert self._viewer is not None
        assert self._mujoco is not None
        assert self._context is not None

        self._idle_wall_time = None

        position = np.asarray(step.state_13[:3], dtype=float)
        self._trail.append(position.copy())
        with self._viewer.lock():
            scene = self._viewer.user_scn
            scene.ngeom = 0
            self._add_sphere(
                np.asarray(self._context.goal_position, dtype=float),
                0.12,
                (1.0, 0.78, 0.05, 0.90),
            )
            if self._show_safety_envelopes:
                for obstacle_position, radius in zip(
                    step.obstacle_positions, self._context.safety_radii
                ):
                    self._add_sphere(
                        np.asarray(obstacle_position, dtype=float),
                        float(radius),
                        (1.0, 0.45, 0.05, 0.12),
                    )
            if self._show_obstacle_prediction:
                displayed_predictions = getattr(
                    step, "estimated_obstacle_predictions", None
                )
                if displayed_predictions is None:
                    displayed_predictions = step.obstacle_predictions
                for prediction in np.asarray(displayed_predictions, dtype=float):
                    self._add_polyline(
                        list(prediction),
                        (1.00, 0.40, 0.15, 0.55),
                        radius=0.009,
                        segment_limit=40,
                    )
            if self._show_trail:
                self._add_polyline(
                    list(self._trail),
                    (0.20, 0.55, 1.00, 0.72),
                    radius=0.018,
                    segment_limit=self.options.max_trail_segments,
                )
            if self._show_prediction and step.predicted_positions is not None:
                self._add_polyline(
                    list(np.asarray(step.predicted_positions, dtype=float)),
                    (0.15, 1.00, 0.35, 0.72),
                    radius=0.012,
                    segment_limit=80,
                )
            if step.collided:
                self._add_sphere(position, 0.46, (1.0, 0.05, 0.05, 0.24))
            if self._camera_mode == "follow":
                self._viewer.cam.lookat[:] = position

        self._viewer.sync()
        if step.step_index % self.options.status_every_steps == 0:
            print(
                f"t={step.time_s:6.2f}s  "
                f"pos=({position[0]:6.2f},{position[1]:6.2f},{position[2]:6.2f})  "
                f"goal={step.goal_distance_m:5.2f}m  "
                f"clearance={step.min_clearance_m:6.3f}m  "
                f"contact={'yes' if step.collided else 'no'}"
            )
        self._pace(step.time_s)
        return self.is_running()

    def _pace(self, simulation_time_s: float) -> None:
        if self.options.realtime_factor <= 0.0:
            return
        target = self._wall_start + simulation_time_s / self.options.realtime_factor
        remaining = target - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)

    def _next_geom(self) -> Any | None:
        assert self._viewer is not None
        scene = self._viewer.user_scn
        if scene.ngeom >= scene.maxgeom:
            return None
        geom = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        return geom

    def _add_sphere(
        self,
        position: np.ndarray,
        radius: float,
        rgba: tuple[float, float, float, float],
    ) -> None:
        geom = self._next_geom()
        if geom is None:
            return
        self._mujoco.mjv_initGeom(
            geom,
            type=self._mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, 0.0, 0.0], dtype=np.float64),
            pos=np.asarray(position, dtype=np.float64),
            mat=np.eye(3, dtype=np.float64).ravel(),
            rgba=np.asarray(rgba, dtype=np.float32),
        )

    def _add_polyline(
        self,
        points: list[np.ndarray],
        rgba: tuple[float, float, float, float],
        *,
        radius: float,
        segment_limit: int,
    ) -> None:
        if len(points) < 2:
            return
        stride = max(1, math.ceil((len(points) - 1) / segment_limit))
        indices = list(range(0, len(points) - 1, stride))
        for index in indices[-segment_limit:]:
            start = np.asarray(points[index], dtype=np.float64)
            end = np.asarray(points[min(index + stride, len(points) - 1)], dtype=np.float64)
            if np.linalg.norm(end - start) < 1e-9:
                continue
            geom = self._next_geom()
            if geom is None:
                return
            self._mujoco.mjv_initGeom(
                geom,
                type=self._mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=np.array([radius, 0.0, 0.0], dtype=np.float64),
                pos=np.zeros(3, dtype=np.float64),
                mat=np.eye(3, dtype=np.float64).ravel(),
                rgba=np.asarray(rgba, dtype=np.float32),
            )
            self._mujoco.mjv_connector(
                geom,
                self._mujoco.mjtGeom.mjGEOM_CAPSULE,
                radius,
                start,
                end,
            )


class InteractiveMuJoCoRuntime:
    """Compose native 3-D viewing, commands, telemetry, panel and recording."""

    def __init__(
        self,
        config: NativeMuJoCoConfig,
        *,
        base_dir: str | Path,
        enable_panel: bool = True,
        enable_recording: bool = True,
    ):
        self.config = config
        self._commands = LocalCommandQueue()
        self.viewer = NativeMuJoCoViewer(config.viewer, self._commands.put)
        self.panel = (
            DesktopPanelProcess(config.panel, config.name)
            if enable_panel and config.panel.enabled
            else None
        )
        recording_options = config.recording
        if not enable_recording:
            recording_options = RecordingOptions(
                enabled=False,
                output_dir=recording_options.output_dir,
                max_buffer_samples=recording_options.max_buffer_samples,
            )
        self.telemetry = TelemetryBuffer(recording_options.max_buffer_samples)
        self.recorder = NativeRunRecorder(
            recording_options,
            config.name,
            config.to_mapping(),
            base_dir=base_dir,
        )
        self._last_sample: dict[str, Any] | None = None
        self._last_time_s = 0.0
        self._panel_was_alive = False
        self._completion_reason: str | None = None

    def open(self, plant: "MuJoCoPlant", context: "CoupledRunContext") -> None:
        if self.panel is not None:
            self.panel.start()
            self._panel_was_alive = True
        self.viewer.open(plant, context)

    def is_running(self) -> bool:
        return self.viewer.is_running()

    def poll_commands(self) -> list[RuntimeCommand]:
        commands = self._commands.drain()
        if self.panel is not None:
            commands.extend(self.panel.drain_commands())
            if (
                self._panel_was_alive
                and not self.panel.is_alive()
                and self.config.panel.stop_when_closed
                and not any(command.name == CommandName.STOP for command in commands)
            ):
                commands.append(RuntimeCommand(CommandName.STOP, source="panel_closed"))
        for command in commands:
            self.viewer.handle_command(command)
            self.recorder.record_event(
                command.name.value,
                self._last_time_s,
                source=command.source,
                payload=command.payload,
            )
            if command.name == CommandName.SNAPSHOT:
                self.recorder.write_snapshot(self._last_sample)
        return commands

    def on_step(self, step: "CoupledStep") -> bool:
        sample = step_to_sample(step)
        self._last_sample = sample
        self._last_time_s = float(step.time_s)
        self.telemetry.append(sample)
        self.recorder.record_step(step, sample)
        if self.panel is not None and self.panel.is_alive():
            self.panel.publish(sample)
        return self.viewer.on_step(step)

    def on_idle(self, paused: bool) -> None:
        self.viewer.on_idle()
        if self._last_sample is not None and self.panel is not None and self.panel.is_alive():
            sample = dict(self._last_sample)
            sample["paused"] = bool(paused)
            sample["completed"] = self._completion_reason is not None
            sample["completion_reason"] = self._completion_reason
            self.panel.publish(sample)

    def on_completed(self, reason: str) -> None:
        """Expose completion without closing either interactive window."""
        self._completion_reason = str(reason)
        self.recorder.record_event(
            "completed", self._last_time_s, payload={"reason": self._completion_reason}
        )
        self.on_idle(True)

    def on_reset(self) -> None:
        self.viewer.reset_visuals()
        self.telemetry.clear()
        self.recorder.reset_episode()
        self._last_sample = None
        self._last_time_s = 0.0
        self._completion_reason = None
        if self.panel is not None and self.panel.is_alive():
            self.panel.reset()

    def close(self) -> None:
        self.viewer.close()
        if self.panel is not None:
            self.panel.close()

    def finalize(self, result: Mapping[str, Any]) -> Path | None:
        return self.recorder.finalize(result)
