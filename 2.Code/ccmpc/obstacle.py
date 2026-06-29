"""Obstacle models and prediction utilities for quadrotor CC-MPC.

This module defines the obstacle-side data contract used by the core
chance-constrained MPC controller.

Legacy-compatible obstacle fields
---------------------------------
The migrated controller expects each obstacle to expose:

    p_hat     : position estimate, shape (3,)
    v_hat     : velocity estimate, shape (3,)
    axes      : ellipsoid semi-axes, shape (3,)
    R_o       : obstacle orientation matrix, shape (3, 3)
    Sigma     : position covariance, shape (3, 3)
    Sigma_v   : velocity covariance / covariance growth term, shape (3, 3)

This file keeps those names intentionally so legacy CC-MPC logic can be
migrated gradually while still using validated data structures.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from ccmpc.types import (
    FloatArray,
    as_position3,
    as_sigma3x3,
)
from ccmpc.utils import (
    Omega_half,
    Omega_matrix,
    as_matrix3x3,
    box_to_ellipsoid_axes,
    yaw_to_rotation,
)


class ObstacleError(ValueError):
    """Base exception raised by obstacle utilities."""


class ObstacleConfigError(ObstacleError):
    """Raised when obstacle configuration is invalid."""


class ObstaclePredictionError(ObstacleError):
    """Raised when obstacle prediction fails."""


class ObstacleShape(str, Enum):
    """Supported obstacle geometry encodings."""

    ELLIPSOID = "ellipsoid"
    BOX = "box"
    SPHERE = "sphere"


class ObstacleMotionModel(str, Enum):
    """Supported obstacle motion models."""

    STATIC = "static"
    CONSTANT_VELOCITY = "constant_velocity"


@dataclass(frozen=True)
class ObstacleState:
    """Validated obstacle state used by CC-MPC.

    Parameters
    ----------
    obstacle_id:
        Stable obstacle identifier.
    p_hat:
        Estimated obstacle center position with shape ``(3,)``.
    v_hat:
        Estimated obstacle velocity with shape ``(3,)``.
    axes:
        Ellipsoid semi-axes with shape ``(3,)``.  For a box obstacle, use
        ``box_to_ellipsoid_axes(size)`` before constructing this object.
    R_o:
        Obstacle orientation matrix with shape ``(3, 3)``.
    Sigma:
        Obstacle position covariance with shape ``(3, 3)``.
    Sigma_v:
        Obstacle velocity covariance / per-step covariance growth term with
        shape ``(3, 3)``.
    active:
        Whether this obstacle should participate in planning.
    shape:
        Source geometry type.
    motion_model:
        Source motion model.
    metadata:
        Optional non-critical metadata for logging/adapters.
    """

    obstacle_id: str
    p_hat: FloatArray
    v_hat: FloatArray
    axes: FloatArray
    R_o: FloatArray
    Sigma: FloatArray
    Sigma_v: FloatArray
    active: bool = True
    shape: ObstacleShape = ObstacleShape.ELLIPSOID
    motion_model: ObstacleMotionModel = ObstacleMotionModel.CONSTANT_VELOCITY
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize obstacle data."""
        if not isinstance(self.obstacle_id, str) or not self.obstacle_id.strip():
            raise ObstacleConfigError("obstacle_id must be a non-empty string.")

        object.__setattr__(self, "p_hat", as_position3(self.p_hat))
        object.__setattr__(self, "v_hat", as_position3(self.v_hat))

        axes = as_position3(self.axes)
        if np.any(axes <= 0.0):
            raise ObstacleConfigError("axes must contain strictly positive values.")
        object.__setattr__(self, "axes", axes)

        R_o = as_matrix3x3(self.R_o, name="R_o")
        if not is_rotation_matrix(R_o):
            raise ObstacleConfigError("R_o must be an orthonormal rotation matrix.")
        object.__setattr__(self, "R_o", R_o)

        object.__setattr__(self, "Sigma", as_sigma3x3(self.Sigma))
        object.__setattr__(self, "Sigma_v", as_sigma3x3(self.Sigma_v))

        if not isinstance(self.active, bool):
            raise ObstacleConfigError("active must be bool.")

        if isinstance(self.shape, str):
            object.__setattr__(self, "shape", ObstacleShape(self.shape))

        if isinstance(self.motion_model, str):
            object.__setattr__(
                self,
                "motion_model",
                ObstacleMotionModel(self.motion_model),
            )

        if not isinstance(self.metadata, dict):
            raise ObstacleConfigError("metadata must be a dictionary.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def position(self) -> FloatArray:
        """Return a copy of obstacle position estimate."""
        return self.p_hat.copy()

    @property
    def velocity(self) -> FloatArray:
        """Return a copy of obstacle velocity estimate."""
        return self.v_hat.copy()

    @property
    def covariance(self) -> FloatArray:
        """Return a copy of obstacle position covariance."""
        return self.Sigma.copy()

    @property
    def velocity_covariance(self) -> FloatArray:
        """Return a copy of obstacle velocity covariance."""
        return self.Sigma_v.copy()

    def distance_to_center(self, position: FloatArray) -> float:
        """Euclidean distance from a position to obstacle center."""
        pos = as_position3(position)
        return float(np.linalg.norm(self.p_hat - pos))

    def distance_to_edge(self, position: FloatArray) -> float:
        """Approximate distance to obstacle edge using largest semi-axis."""
        return max(0.0, self.distance_to_center(position) - float(np.max(self.axes)))

    def omega(self, mav_radius: float = 0.0) -> FloatArray:
        """Return ellipsoidal collision matrix Omega."""
        return Omega_matrix(self.axes, mav_radius, self.R_o)

    def omega_half(self, mav_radius: float = 0.0) -> FloatArray:
        """Return Cholesky factor of collision matrix Omega."""
        return Omega_half(self.omega(mav_radius))

    def predict_at(self, time: float) -> "ObstacleState":
        """Predict obstacle state at continuous time offset.

        Position follows constant-velocity motion:

            p(t) = p_hat + v_hat * t

        Covariance growth follows the conservative legacy-compatible model:

            Sigma(t) = Sigma + Sigma_v * t^2

        For static obstacles, velocity is ignored.
        """
        t = _as_non_negative_float(time, "time")

        if self.motion_model is ObstacleMotionModel.STATIC:
            predicted_position = self.p_hat.copy()
            predicted_velocity = np.zeros(3, dtype=np.float64)
            predicted_sigma = self.Sigma.copy()
        else:
            predicted_position = self.p_hat + self.v_hat * t
            predicted_velocity = self.v_hat.copy()
            predicted_sigma = self.Sigma + self.Sigma_v * (t**2)

        return ObstacleState(
            obstacle_id=self.obstacle_id,
            p_hat=predicted_position,
            v_hat=predicted_velocity,
            axes=self.axes,
            R_o=self.R_o,
            Sigma=predicted_sigma,
            Sigma_v=self.Sigma_v,
            active=self.active,
            shape=self.shape,
            motion_model=self.motion_model,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class ObstaclePrediction:
    """Obstacle prediction at one horizon step."""

    obstacle_id: str
    step_index: int
    time: float
    p_hat: FloatArray
    v_hat: FloatArray
    axes: FloatArray
    R_o: FloatArray
    Sigma: FloatArray
    Sigma_v: FloatArray
    L: FloatArray
    active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate prediction data."""
        if not isinstance(self.obstacle_id, str) or not self.obstacle_id.strip():
            raise ObstaclePredictionError("obstacle_id must be a non-empty string.")

        if isinstance(self.step_index, bool) or self.step_index < 0:
            raise ObstaclePredictionError("step_index must be a non-negative integer.")

        object.__setattr__(self, "time", _as_non_negative_float(self.time, "time"))
        object.__setattr__(self, "p_hat", as_position3(self.p_hat))
        object.__setattr__(self, "v_hat", as_position3(self.v_hat))

        axes = as_position3(self.axes)
        if np.any(axes <= 0.0):
            raise ObstaclePredictionError("axes must contain strictly positive values.")
        object.__setattr__(self, "axes", axes)

        R_o = as_matrix3x3(self.R_o, name="R_o")
        if not is_rotation_matrix(R_o):
            raise ObstaclePredictionError("R_o must be an orthonormal rotation matrix.")
        object.__setattr__(self, "R_o", R_o)

        object.__setattr__(self, "Sigma", as_sigma3x3(self.Sigma))
        object.__setattr__(self, "Sigma_v", as_sigma3x3(self.Sigma_v))

        L = as_matrix3x3(self.L, name="L")
        object.__setattr__(self, "L", L)

        if not isinstance(self.active, bool):
            raise ObstaclePredictionError("active must be bool.")

        if not isinstance(self.metadata, dict):
            raise ObstaclePredictionError("metadata must be a dictionary.")
        object.__setattr__(self, "metadata", dict(self.metadata))


class ObstacleManager:
    """Container and query helper for CC-MPC obstacles."""

    def __init__(self, obstacles: Sequence[ObstacleState] | None = None) -> None:
        self.obstacles: tuple[ObstacleState, ...] = tuple(obstacles or ())
        for index, obstacle in enumerate(self.obstacles):
            if not isinstance(obstacle, ObstacleState):
                raise ObstacleConfigError(
                    f"obstacles[{index}] must be ObstacleState, "
                    f"got {type(obstacle).__name__}."
                )

    def __len__(self) -> int:
        """Return number of stored obstacles."""
        return len(self.obstacles)

    def __iter__(self):
        """Iterate over stored obstacles."""
        return iter(self.obstacles)

    @property
    def active_obstacles(self) -> tuple[ObstacleState, ...]:
        """Return active obstacles only."""
        return tuple(obstacle for obstacle in self.obstacles if obstacle.active)

    def add(self, obstacle: ObstacleState) -> "ObstacleManager":
        """Return a new manager with one obstacle appended."""
        if not isinstance(obstacle, ObstacleState):
            raise ObstacleConfigError("obstacle must be ObstacleState.")
        return ObstacleManager((*self.obstacles, obstacle))

    def get_closest(
        self,
        position: FloatArray,
        k: int = 1,
        *,
        include_inactive: bool = False,
        use_edge_distance: bool = False,
    ) -> list[ObstacleState]:
        """Return the ``k`` closest obstacles to ``position``.

        Parameters
        ----------
        position:
            Query position with shape ``(3,)``.
        k:
            Number of closest obstacles to return.
        include_inactive:
            If False, inactive obstacles are ignored.
        use_edge_distance:
            If True, sort by approximate edge distance.  Otherwise sort by
            center distance.
        """
        pos = as_position3(position)

        if isinstance(k, bool) or k <= 0:
            raise ObstacleConfigError("k must be a positive integer.")

        candidates = self.obstacles if include_inactive else self.active_obstacles

        if use_edge_distance:
            key_fn = lambda obstacle: obstacle.distance_to_edge(pos)
        else:
            key_fn = lambda obstacle: obstacle.distance_to_center(pos)

        return list(sorted(candidates, key=key_fn)[:k])

    def predict_all(
        self,
        *,
        dt: float,
        horizon: int,
        mav_radius: float = 0.0,
        include_initial: bool = False,
        include_inactive: bool = False,
    ) -> tuple[tuple[ObstaclePrediction, ...], ...]:
        """Predict all obstacles over the horizon.

        Returns a time-major tuple:

            predictions[k][i] = prediction of obstacle i at step k

        If ``include_initial`` is False, returned steps are 1..horizon.
        If True, returned steps are 0..horizon.
        """
        dt_value = _as_positive_float(dt, "dt")
        horizon_value = _as_positive_int(horizon, "horizon")
        _as_non_negative_float(mav_radius, "mav_radius")

        obstacles = self.obstacles if include_inactive else self.active_obstacles
        start = 0 if include_initial else 1
        stop = horizon_value + 1

        time_major_predictions: list[tuple[ObstaclePrediction, ...]] = []

        for step in range(start, stop):
            step_predictions = tuple(
                prediction_from_state(
                    obstacle,
                    step_index=step,
                    dt=dt_value,
                    mav_radius=mav_radius,
                )
                for obstacle in obstacles
            )
            time_major_predictions.append(step_predictions)

        return tuple(time_major_predictions)

    @classmethod
    def from_config(cls, config: Any) -> "ObstacleManager":
        """Create manager from a config object/dict/list."""
        return cls(obstacles_from_config(config))


def prediction_from_state(
    obstacle: ObstacleState,
    *,
    step_index: int,
    dt: float,
    mav_radius: float = 0.0,
) -> ObstaclePrediction:
    """Create one obstacle prediction from a state and step index."""
    if not isinstance(obstacle, ObstacleState):
        raise ObstaclePredictionError("obstacle must be ObstacleState.")

    step = _as_non_negative_int(step_index, "step_index")
    dt_value = _as_positive_float(dt, "dt")
    radius = _as_non_negative_float(mav_radius, "mav_radius")

    time_value = step * dt_value
    predicted = obstacle.predict_at(time_value)
    L = predicted.omega_half(radius)

    return ObstaclePrediction(
        obstacle_id=predicted.obstacle_id,
        step_index=step,
        time=time_value,
        p_hat=predicted.p_hat,
        v_hat=predicted.v_hat,
        axes=predicted.axes,
        R_o=predicted.R_o,
        Sigma=predicted.Sigma,
        Sigma_v=predicted.Sigma_v,
        L=L,
        active=predicted.active,
        metadata=predicted.metadata,
    )


def predict_obstacle(
    obstacle: ObstacleState,
    *,
    dt: float,
    horizon: int,
    mav_radius: float = 0.0,
    include_initial: bool = False,
) -> tuple[ObstaclePrediction, ...]:
    """Predict one obstacle over a finite horizon."""
    dt_value = _as_positive_float(dt, "dt")
    horizon_value = _as_positive_int(horizon, "horizon")

    start = 0 if include_initial else 1
    return tuple(
        prediction_from_state(
            obstacle,
            step_index=step,
            dt=dt_value,
            mav_radius=mav_radius,
        )
        for step in range(start, horizon_value + 1)
    )


def make_static_obstacle(
    *,
    obstacle_id: str,
    position: FloatArray,
    axes: FloatArray,
    R_o: FloatArray | None = None,
    Sigma: FloatArray | None = None,
    metadata: dict[str, Any] | None = None,
) -> ObstacleState:
    """Create a static ellipsoidal obstacle."""
    return ObstacleState(
        obstacle_id=obstacle_id,
        p_hat=position,
        v_hat=np.zeros(3, dtype=np.float64),
        axes=axes,
        R_o=np.eye(3, dtype=np.float64) if R_o is None else R_o,
        Sigma=default_position_covariance() if Sigma is None else Sigma,
        Sigma_v=np.zeros((3, 3), dtype=np.float64),
        active=True,
        shape=ObstacleShape.ELLIPSOID,
        motion_model=ObstacleMotionModel.STATIC,
        metadata={} if metadata is None else dict(metadata),
    )


def make_constant_velocity_obstacle(
    *,
    obstacle_id: str,
    position: FloatArray,
    velocity: FloatArray,
    axes: FloatArray,
    R_o: FloatArray | None = None,
    Sigma: FloatArray | None = None,
    Sigma_v: FloatArray | None = None,
    metadata: dict[str, Any] | None = None,
) -> ObstacleState:
    """Create a constant-velocity ellipsoidal obstacle."""
    return ObstacleState(
        obstacle_id=obstacle_id,
        p_hat=position,
        v_hat=velocity,
        axes=axes,
        R_o=np.eye(3, dtype=np.float64) if R_o is None else R_o,
        Sigma=default_position_covariance() if Sigma is None else Sigma,
        Sigma_v=default_velocity_covariance() if Sigma_v is None else Sigma_v,
        active=True,
        shape=ObstacleShape.ELLIPSOID,
        motion_model=ObstacleMotionModel.CONSTANT_VELOCITY,
        metadata={} if metadata is None else dict(metadata),
    )


def obstacles_from_config(config: Any) -> tuple[ObstacleState, ...]:
    """Parse obstacles from dict/list/dataclass-like config.

    Supported inputs
    ----------------
    1. A list/tuple of obstacle dictionaries.
    2. A dict with top-level key ``"obstacles"``.
    3. A dataclass/object exposing ``.obstacles``.
    4. A sequence of already-created ``ObstacleState`` objects.

    Common obstacle dictionary keys
    -------------------------------
    id / obstacle_id / name
    position / p_hat / center
    velocity / v_hat
    axes
    size
    radius
    yaw
    R_o / rotation
    Sigma / covariance
    Sigma_v / velocity_covariance
    active
    shape
    motion_model
    """
    raw_obstacles = _extract_obstacle_sequence(config)
    obstacles: list[ObstacleState] = []

    for index, item in enumerate(raw_obstacles):
        if isinstance(item, ObstacleState):
            obstacles.append(item)
        elif isinstance(item, dict):
            obstacles.append(obstacle_from_dict(item, default_index=index))
        else:
            obstacles.append(obstacle_from_object(item, default_index=index))

    ids = [obstacle.obstacle_id for obstacle in obstacles]
    if len(ids) != len(set(ids)):
        raise ObstacleConfigError("Duplicate obstacle_id detected.")

    return tuple(obstacles)


def obstacle_from_dict(data: dict[str, Any], *, default_index: int = 0) -> ObstacleState:
    """Create ObstacleState from a flexible dictionary."""
    if not isinstance(data, dict):
        raise ObstacleConfigError("obstacle config item must be a dictionary.")

    obstacle_id = str(
        data.get("obstacle_id", data.get("id", data.get("name", f"obs_{default_index:03d}")))
    )

    shape = parse_obstacle_shape(
        data.get("shape", data.get("type", ObstacleShape.ELLIPSOID))
    )
    motion_model = parse_obstacle_motion_model(
        data.get("motion_model", data.get("motion", ObstacleMotionModel.CONSTANT_VELOCITY))
    )

    position = _first_present(data, ("p_hat", "position", "center", "pos"))
    if position is None:
        raise ObstacleConfigError(f"Obstacle {obstacle_id!r} missing position/p_hat.")

    velocity = _first_present(data, ("v_hat", "velocity", "vel"))
    if velocity is None:
        velocity = np.zeros(3, dtype=np.float64)

    axes = _parse_axes(data, shape=shape)

    rotation = _parse_rotation(data)

    Sigma = _first_present(data, ("Sigma", "covariance", "position_covariance"))
    if Sigma is None:
        Sigma = default_position_covariance()

    Sigma_v = _first_present(data, ("Sigma_v", "velocity_covariance", "covariance_velocity"))
    if Sigma_v is None:
        Sigma_v = default_velocity_covariance()

    active = bool(data.get("active", True))

    metadata = dict(data.get("metadata", {}))
    metadata.setdefault("source", "dict")

    return ObstacleState(
        obstacle_id=obstacle_id,
        p_hat=position,
        v_hat=velocity,
        axes=axes,
        R_o=rotation,
        Sigma=Sigma,
        Sigma_v=Sigma_v,
        active=active,
        shape=shape,
        motion_model=motion_model,
        metadata=metadata,
    )


def obstacle_from_object(obj: Any, *, default_index: int = 0) -> ObstacleState:
    """Create ObstacleState from a dataclass-like object."""
    obstacle_id = str(
        getattr(obj, "obstacle_id", getattr(obj, "id", getattr(obj, "name", f"obs_{default_index:03d}")))
    )

    position = _first_attr(obj, ("p_hat", "position", "center", "pos"))
    if position is None:
        raise ObstacleConfigError(f"Obstacle object {obstacle_id!r} missing position/p_hat.")

    velocity = _first_attr(obj, ("v_hat", "velocity", "vel"))
    if velocity is None:
        velocity = np.zeros(3, dtype=np.float64)

    shape_raw = getattr(obj, "shape", getattr(obj, "type", ObstacleShape.ELLIPSOID))
    shape = parse_obstacle_shape(shape_raw)

    axes = _first_attr(obj, ("axes", "semi_axes"))
    if axes is None:
        size = _first_attr(obj, ("size", "dimensions"))
        if size is not None:
            axes = box_to_ellipsoid_axes(size)
            shape = ObstacleShape.BOX
        else:
            radius = getattr(obj, "radius", None)
            if radius is None:
                raise ObstacleConfigError(f"Obstacle object {obstacle_id!r} missing axes/size/radius.")
            radius_value = _as_positive_float(radius, "radius")
            axes = np.full(3, radius_value, dtype=np.float64)
            shape = ObstacleShape.SPHERE

    rotation = _first_attr(obj, ("R_o", "rotation"))
    if rotation is None:
        yaw = getattr(obj, "yaw", 0.0)
        rotation = yaw_to_rotation(float(yaw))

    Sigma = _first_attr(obj, ("Sigma", "covariance", "position_covariance"))
    if Sigma is None:
        Sigma = default_position_covariance()

    Sigma_v = _first_attr(obj, ("Sigma_v", "velocity_covariance", "covariance_velocity"))
    if Sigma_v is None:
        Sigma_v = default_velocity_covariance()

    motion_raw = getattr(obj, "motion_model", getattr(obj, "motion", ObstacleMotionModel.CONSTANT_VELOCITY))
    motion_model = parse_obstacle_motion_model(motion_raw)

    metadata = getattr(obj, "metadata", {})
    if metadata is None:
        metadata = {}

    return ObstacleState(
        obstacle_id=obstacle_id,
        p_hat=position,
        v_hat=velocity,
        axes=axes,
        R_o=rotation,
        Sigma=Sigma,
        Sigma_v=Sigma_v,
        active=bool(getattr(obj, "active", True)),
        shape=shape,
        motion_model=motion_model,
        metadata=dict(metadata),
    )

def parse_obstacle_shape(value: Any) -> ObstacleShape:
    if isinstance(value, ObstacleShape):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if "." in normalized:
            normalized = normalized.split(".")[-1]
        try:
            return ObstacleShape(normalized)
        except ValueError as exc:
            valid = ", ".join(item.value for item in ObstacleShape)
            raise ObstacleConfigError(
                f"Invalid obstacle shape {value!r}. Valid values: {valid}."
            ) from exc

    raise ObstacleConfigError(
        f"Obstacle shape must be ObstacleShape or str, got {type(value).__name__}."
    )


def parse_obstacle_motion_model(value: Any) -> ObstacleMotionModel:
    if isinstance(value, ObstacleMotionModel):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if "." in normalized:
            normalized = normalized.split(".")[-1]
        try:
            return ObstacleMotionModel(normalized)
        except ValueError as exc:
            valid = ", ".join(item.value for item in ObstacleMotionModel)
            raise ObstacleConfigError(
                f"Invalid obstacle motion model {value!r}. Valid values: {valid}."
            ) from exc

    raise ObstacleConfigError(
        "Obstacle motion model must be ObstacleMotionModel or str, "
        f"got {type(value).__name__}."
    )

def default_position_covariance(std: float = 0.05) -> FloatArray:
    """Default obstacle position covariance."""
    std_value = _as_non_negative_float(std, "std")
    return np.eye(3, dtype=np.float64) * (std_value**2)


def default_velocity_covariance(std: float = 0.02) -> FloatArray:
    """Default obstacle velocity covariance/growth term."""
    std_value = _as_non_negative_float(std, "std")
    return np.eye(3, dtype=np.float64) * (std_value**2)


def is_rotation_matrix(R: FloatArray, *, atol: float = 1e-8) -> bool:
    """Return True if ``R`` is approximately a 3D rotation matrix."""
    matrix = as_matrix3x3(R, name="R")
    should_be_identity = matrix.T @ matrix
    determinant = float(np.linalg.det(matrix))

    return bool(
        np.allclose(should_be_identity, np.eye(3), atol=atol, rtol=0.0)
        and np.isclose(determinant, 1.0, atol=atol, rtol=0.0)
    )


def _parse_axes(data: dict[str, Any], *, shape: ObstacleShape) -> FloatArray:
    """Parse obstacle semi-axes from dictionary."""
    axes = _first_present(data, ("axes", "semi_axes"))
    if axes is not None:
        return as_position3(axes)

    size = _first_present(data, ("size", "dimensions"))
    if size is not None:
        return box_to_ellipsoid_axes(size)

    radius = data.get("radius")
    if radius is not None:
        radius_value = _as_positive_float(radius, "radius")
        return np.full(3, radius_value, dtype=np.float64)

    raise ObstacleConfigError(
        f"Obstacle with shape={shape.value!r} must provide axes, size, or radius."
    )


def _parse_rotation(data: dict[str, Any]) -> FloatArray:
    """Parse rotation matrix from dictionary."""
    rotation = _first_present(data, ("R_o", "rotation"))
    if rotation is not None:
        return as_matrix3x3(rotation, name="R_o")

    yaw = data.get("yaw", data.get("yaw_rad", 0.0))
    return yaw_to_rotation(float(yaw))


def _extract_obstacle_sequence(config: Any) -> Sequence[Any]:
    """Extract obstacle sequence from supported config inputs."""
    if config is None:
        return ()

    if isinstance(config, ObstacleManager):
        return config.obstacles

    if isinstance(config, dict):
        if "obstacles" in config:
            obstacles = config["obstacles"]
        elif "scenario" in config and isinstance(config["scenario"], dict) and "obstacles" in config["scenario"]:
            obstacles = config["scenario"]["obstacles"]
        else:
            raise ObstacleConfigError("Config dictionary must contain an 'obstacles' key.")

        if obstacles is None:
            return ()

        if not isinstance(obstacles, Sequence) or isinstance(obstacles, (str, bytes)):
            raise ObstacleConfigError("obstacles must be a sequence.")
        return obstacles

    if isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
        return config

    obstacles_attr = getattr(config, "obstacles", None)
    if obstacles_attr is None:
        raise ObstacleConfigError("Config object must expose an 'obstacles' attribute.")

    if not isinstance(obstacles_attr, Sequence):
        raise ObstacleConfigError("config.obstacles must be a sequence.")

    return obstacles_attr


def _first_present(data: dict[str, Any], keys: Iterable[str]) -> Any | None:
    """Return first present key value from dictionary."""
    for key in keys:
        if key in data:
            return data[key]
    return None


def _first_attr(obj: Any, names: Iterable[str]) -> Any | None:
    """Return first existing attribute value from object."""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _as_non_negative_float(value: float, name: str) -> float:
    """Validate finite scalar >= 0."""
    if isinstance(value, bool):
        raise ObstacleConfigError(f"{name} must be a finite scalar, got bool.")

    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ObstacleConfigError(f"{name} must be a finite scalar.") from exc

    if not np.isfinite(scalar) or scalar < 0.0:
        raise ObstacleConfigError(f"{name} must be finite and >= 0.")

    return scalar


def _as_positive_float(value: float, name: str) -> float:
    """Validate finite scalar > 0."""
    scalar = _as_non_negative_float(value, name)

    if scalar <= 0.0:
        raise ObstacleConfigError(f"{name} must be > 0.")

    return scalar


def _as_non_negative_int(value: int, name: str) -> int:
    """Validate integer >= 0."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ObstacleConfigError(f"{name} must be an integer.")

    if value < 0:
        raise ObstacleConfigError(f"{name} must be >= 0.")

    return value


def _as_positive_int(value: int, name: str) -> int:
    """Validate integer > 0."""
    integer = _as_non_negative_int(value, name)

    if integer <= 0:
        raise ObstacleConfigError(f"{name} must be > 0.")

    return integer


__all__ = [
    "ObstacleConfigError",
    "ObstacleError",
    "ObstacleManager",
    "ObstacleMotionModel",
    "ObstaclePrediction",
    "ObstaclePredictionError",
    "ObstacleShape",
    "ObstacleState",
    "default_position_covariance",
    "default_velocity_covariance",
    "is_rotation_matrix",
    "make_constant_velocity_obstacle",
    "make_static_obstacle",
    "obstacle_from_dict",
    "obstacle_from_object",
    "obstacles_from_config",
    "predict_obstacle",
    "prediction_from_state",
]
