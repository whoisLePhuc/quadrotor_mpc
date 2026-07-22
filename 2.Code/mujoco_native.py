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
    viewer: NativeViewerOptions

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

        start = _xyz(start_raw, "start")
        start.update(
            {
                axis: float(start_raw.get(axis, 0.0))
                for axis in ("roll", "pitch", "yaw")
            }
        )
        goal_euler = {
            axis: float(euler_raw.get(axis, 0.0))
            for axis in ("roll", "pitch", "yaw")
        }
        try:
            bounds = {
                "thrust": _positive(bounds_raw["thrust"], "controller.bounds.thrust"),
                "torque_rp": _positive(
                    bounds_raw["torque_rp"], "controller.bounds.torque_rp"
                ),
                "torque_yaw": _positive(
                    bounds_raw["torque_yaw"], "controller.bounds.torque_yaw"
                ),
            }
        except KeyError as exc:
            raise ValueError(
                "controller.bounds requires thrust, torque_rp and torque_yaw"
            ) from exc

        obstacles: list[dict[str, Any]] = []
        raw_obstacles = raw.get("obstacles", [])
        if not isinstance(raw_obstacles, Sequence) or isinstance(raw_obstacles, (str, bytes)):
            raise ValueError("obstacles must be a list")
        for index, item in enumerate(raw_obstacles):
            obstacle = _mapping(item, f"obstacles[{index}]")
            kind = str(obstacle.get("type", "static")).lower()
            if kind not in {"static", "dynamic"}:
                raise ValueError(
                    f"obstacles[{index}].type must be 'static' or 'dynamic'"
                )
            parsed: dict[str, Any] = {
                "type": kind,
                "x": float(obstacle["x"]),
                "z": float(obstacle["z"]),
                "radius": _positive(obstacle["radius"], f"obstacles[{index}].radius"),
            }
            if kind == "static":
                parsed["y"] = float(obstacle["y"])
            else:
                parsed["amp"] = _positive(
                    obstacle["amp"], f"obstacles[{index}].amp", allow_zero=True
                )
                parsed["period"] = _positive(
                    obstacle["period"], f"obstacles[{index}].period"
                )
            obstacles.append(parsed)

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
            mpc_timestep_s=_positive(
                simulation_raw.get("mpc_timestep_s", 0.05), "mpc_timestep_s"
            ),
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
            viewer=NativeViewerOptions.from_mapping(viewer_raw),
        )


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

    def __init__(self, options: NativeViewerOptions):
        self.options = options
        self._mujoco: Any | None = None
        self._session: Any | None = None
        self._viewer: Any | None = None
        self._wall_start = 0.0
        self._trail: deque[np.ndarray] = deque(maxlen=options.max_trail_points)
        self._context: CoupledRunContext | None = None

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
        self._session = mujoco.viewer.launch_passive(plant.model, plant.data)
        self._viewer = self._session.__enter__()
        goal = np.asarray(context.goal_position, dtype=float)
        start = np.asarray(context.start_position, dtype=float)
        with self._viewer.lock():
            self._viewer.cam.lookat[:] = (
                start if self.options.camera_mode == "follow" else 0.5 * (start + goal)
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

    def on_step(self, step: "CoupledStep") -> bool:
        if not self.is_running():
            return False
        assert self._viewer is not None
        assert self._mujoco is not None
        assert self._context is not None

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
            if self.options.show_safety_envelopes:
                for obstacle_position, radius in zip(
                    step.obstacle_positions, self._context.safety_radii
                ):
                    self._add_sphere(
                        np.asarray(obstacle_position, dtype=float),
                        float(radius),
                        (1.0, 0.45, 0.05, 0.12),
                    )
            if self.options.show_trail:
                self._add_polyline(
                    list(self._trail),
                    (0.20, 0.55, 1.00, 0.72),
                    radius=0.018,
                    segment_limit=self.options.max_trail_segments,
                )
            if self.options.show_prediction and step.predicted_positions is not None:
                self._add_polyline(
                    list(np.asarray(step.predicted_positions, dtype=float)),
                    (0.15, 1.00, 0.35, 0.72),
                    radius=0.012,
                    segment_limit=80,
                )
            if step.collided:
                self._add_sphere(position, 0.46, (1.0, 0.05, 0.05, 0.24))
            if self.options.camera_mode == "follow":
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
