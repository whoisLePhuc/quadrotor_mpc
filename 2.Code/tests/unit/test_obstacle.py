"""Unit tests for obstacle models and prediction utilities.

Target module:
    ccmpc.obstacle

These tests verify the obstacle data contract needed by the migrated CC-MPC
controller:
    obs.p_hat
    obs.v_hat
    obs.axes
    obs.R_o
    obs.Sigma
    obs.Sigma_v

They also test ObstacleManager.get_closest() and prediction helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pytest

from ccmpc.obstacle import (
    ObstacleConfigError,
    ObstacleManager,
    ObstacleMotionModel,
    ObstaclePrediction,
    ObstaclePredictionError,
    ObstacleShape,
    ObstacleState,
    default_position_covariance,
    default_velocity_covariance,
    is_rotation_matrix,
    make_constant_velocity_obstacle,
    make_static_obstacle,
    obstacle_from_dict,
    obstacle_from_object,
    obstacles_from_config,
    predict_obstacle,
    prediction_from_state,
)
from ccmpc.utils import box_to_ellipsoid_axes, yaw_to_rotation


@dataclass(frozen=True)
class ObjectObstacleConfig:
    """Dataclass-like obstacle config for obstacle_from_object tests."""

    obstacle_id: str
    position: np.ndarray
    velocity: np.ndarray
    size: np.ndarray
    yaw: float
    covariance: np.ndarray
    velocity_covariance: np.ndarray
    active: bool = True
    motion_model: str = "constant_velocity"
    metadata: dict | None = None


@dataclass(frozen=True)
class ObjectScenarioConfig:
    """Dataclass-like scenario config exposing .obstacles."""

    obstacles: tuple[ObjectObstacleConfig, ...]


def make_sigma(value: float = 0.01) -> np.ndarray:
    """Create a valid Sigma3x3 covariance."""
    return np.eye(3, dtype=np.float64) * value


def make_obstacle() -> ObstacleState:
    """Create a valid constant-velocity obstacle."""
    return ObstacleState(
        obstacle_id="obs_001",
        p_hat=np.array([2.0, 0.0, 1.0], dtype=np.float64),
        v_hat=np.array([0.1, 0.0, 0.0], dtype=np.float64),
        axes=np.array([0.4, 0.5, 0.8], dtype=np.float64),
        R_o=np.eye(3, dtype=np.float64),
        Sigma=make_sigma(0.01),
        Sigma_v=make_sigma(0.001),
        active=True,
        shape=ObstacleShape.ELLIPSOID,
        motion_model=ObstacleMotionModel.CONSTANT_VELOCITY,
        metadata={"source": "unit-test"},
    )


def test_obstacle_state_valid() -> None:
    """ObstacleState should validate and preserve legacy-compatible fields."""
    obstacle = make_obstacle()

    assert obstacle.obstacle_id == "obs_001"
    assert obstacle.p_hat.shape == (3,)
    assert obstacle.v_hat.shape == (3,)
    assert obstacle.axes.shape == (3,)
    assert obstacle.R_o.shape == (3, 3)
    assert obstacle.Sigma.shape == (3, 3)
    assert obstacle.Sigma_v.shape == (3, 3)
    assert obstacle.active is True
    assert obstacle.shape is ObstacleShape.ELLIPSOID
    assert obstacle.motion_model is ObstacleMotionModel.CONSTANT_VELOCITY
    assert obstacle.metadata == {"source": "unit-test"}


def test_obstacle_state_accepts_string_enums() -> None:
    """ObstacleState should accept string values for shape and motion_model."""
    obstacle = ObstacleState(
        obstacle_id="obs_001",
        p_hat=[0.0, 0.0, 1.0],
        v_hat=[0.0, 0.0, 0.0],
        axes=[0.5, 0.5, 0.5],
        R_o=np.eye(3),
        Sigma=make_sigma(),
        Sigma_v=make_sigma(0.001),
        shape="sphere",
        motion_model="static",
    )

    assert obstacle.shape is ObstacleShape.SPHERE
    assert obstacle.motion_model is ObstacleMotionModel.STATIC


def test_obstacle_state_reject_empty_id() -> None:
    """ObstacleState should reject empty obstacle_id."""
    with pytest.raises(ObstacleConfigError, match="obstacle_id"):
        ObstacleState(
            obstacle_id="",
            p_hat=[0.0, 0.0, 1.0],
            v_hat=[0.0, 0.0, 0.0],
            axes=[0.5, 0.5, 0.5],
            R_o=np.eye(3),
            Sigma=make_sigma(),
            Sigma_v=make_sigma(0.001),
        )


def test_obstacle_state_reject_invalid_axes() -> None:
    """ObstacleState should reject zero or negative semi-axes."""
    with pytest.raises(ObstacleConfigError, match="axes"):
        ObstacleState(
            obstacle_id="obs_bad",
            p_hat=[0.0, 0.0, 1.0],
            v_hat=[0.0, 0.0, 0.0],
            axes=[0.5, 0.0, 0.5],
            R_o=np.eye(3),
            Sigma=make_sigma(),
            Sigma_v=make_sigma(0.001),
        )


def test_obstacle_state_reject_invalid_rotation() -> None:
    """ObstacleState should reject non-orthonormal rotation matrices."""
    bad_rotation = np.eye(3, dtype=np.float64)
    bad_rotation[0, 0] = 2.0

    with pytest.raises(ObstacleConfigError, match="R_o"):
        ObstacleState(
            obstacle_id="obs_bad",
            p_hat=[0.0, 0.0, 1.0],
            v_hat=[0.0, 0.0, 0.0],
            axes=[0.5, 0.5, 0.5],
            R_o=bad_rotation,
            Sigma=make_sigma(),
            Sigma_v=make_sigma(0.001),
        )


def test_obstacle_state_distance_helpers() -> None:
    """distance_to_center and distance_to_edge should be deterministic."""
    obstacle = make_obstacle()

    assert obstacle.distance_to_center([0.0, 0.0, 1.0]) == pytest.approx(2.0)
    assert obstacle.distance_to_edge([0.0, 0.0, 1.0]) == pytest.approx(2.0 - 0.8)


def test_obstacle_state_omega_helpers() -> None:
    """ObstacleState.omega and omega_half should produce valid 3x3 matrices."""
    obstacle = make_obstacle()

    omega = obstacle.omega(mav_radius=0.1)
    L = obstacle.omega_half(mav_radius=0.1)

    assert omega.shape == (3, 3)
    assert L.shape == (3, 3)
    assert np.allclose(L @ L.T, omega)


def test_make_static_obstacle() -> None:
    """make_static_obstacle should create a static obstacle with zero velocity."""
    obstacle = make_static_obstacle(
        obstacle_id="static_001",
        position=[1.0, 2.0, 1.0],
        axes=[0.4, 0.4, 0.8],
    )

    assert obstacle.obstacle_id == "static_001"
    assert np.allclose(obstacle.p_hat, [1.0, 2.0, 1.0])
    assert np.allclose(obstacle.v_hat, [0.0, 0.0, 0.0])
    assert np.allclose(obstacle.Sigma_v, np.zeros((3, 3)))
    assert obstacle.motion_model is ObstacleMotionModel.STATIC


def test_make_constant_velocity_obstacle() -> None:
    """make_constant_velocity_obstacle should preserve position and velocity."""
    obstacle = make_constant_velocity_obstacle(
        obstacle_id="moving_001",
        position=[1.0, 0.0, 1.0],
        velocity=[0.2, 0.0, 0.0],
        axes=[0.4, 0.4, 0.8],
    )

    assert obstacle.obstacle_id == "moving_001"
    assert np.allclose(obstacle.p_hat, [1.0, 0.0, 1.0])
    assert np.allclose(obstacle.v_hat, [0.2, 0.0, 0.0])
    assert obstacle.motion_model is ObstacleMotionModel.CONSTANT_VELOCITY
    assert np.allclose(obstacle.Sigma, default_position_covariance())
    assert np.allclose(obstacle.Sigma_v, default_velocity_covariance())


def test_predict_at_constant_velocity() -> None:
    """ObstacleState.predict_at should update position and covariance for moving obstacles."""
    obstacle = make_obstacle()

    predicted = obstacle.predict_at(2.0)

    assert np.allclose(predicted.p_hat, obstacle.p_hat + obstacle.v_hat * 2.0)
    assert np.allclose(predicted.v_hat, obstacle.v_hat)
    assert np.allclose(predicted.Sigma, obstacle.Sigma + obstacle.Sigma_v * 4.0)
    assert predicted.motion_model is ObstacleMotionModel.CONSTANT_VELOCITY


def test_predict_at_static() -> None:
    """Static obstacles should not move during prediction."""
    obstacle = make_static_obstacle(
        obstacle_id="static_001",
        position=[1.0, 2.0, 1.0],
        axes=[0.4, 0.4, 0.8],
        Sigma=make_sigma(0.02),
    )

    predicted = obstacle.predict_at(2.0)

    assert np.allclose(predicted.p_hat, obstacle.p_hat)
    assert np.allclose(predicted.v_hat, [0.0, 0.0, 0.0])
    assert np.allclose(predicted.Sigma, obstacle.Sigma)


def test_predict_at_rejects_negative_time() -> None:
    """predict_at should reject negative prediction time."""
    obstacle = make_obstacle()

    with pytest.raises(ObstacleConfigError, match="time"):
        obstacle.predict_at(-0.1)


def test_prediction_from_state() -> None:
    """prediction_from_state should create one ObstaclePrediction."""
    obstacle = make_obstacle()

    prediction = prediction_from_state(
        obstacle,
        step_index=2,
        dt=0.1,
        mav_radius=0.2,
    )

    assert isinstance(prediction, ObstaclePrediction)
    assert prediction.obstacle_id == obstacle.obstacle_id
    assert prediction.step_index == 2
    assert prediction.time == pytest.approx(0.2)
    assert np.allclose(prediction.p_hat, obstacle.p_hat + obstacle.v_hat * 0.2)
    assert prediction.L.shape == (3, 3)


def test_prediction_from_state_rejects_negative_step() -> None:
    """prediction_from_state should reject negative step index."""
    with pytest.raises(ObstacleConfigError, match="step_index"):
        prediction_from_state(
            make_obstacle(),
            step_index=-1,
            dt=0.1,
        )


def test_predict_obstacle_constant_velocity() -> None:
    """predict_obstacle should return horizon predictions for steps 1..N by default."""
    obstacle = make_obstacle()

    predictions = predict_obstacle(
        obstacle,
        dt=0.1,
        horizon=3,
        mav_radius=0.1,
    )

    assert len(predictions) == 3
    assert [prediction.step_index for prediction in predictions] == [1, 2, 3]
    assert [prediction.time for prediction in predictions] == pytest.approx([0.1, 0.2, 0.3])
    assert np.allclose(predictions[0].p_hat, obstacle.p_hat + obstacle.v_hat * 0.1)
    assert np.allclose(predictions[-1].p_hat, obstacle.p_hat + obstacle.v_hat * 0.3)


def test_predict_obstacle_include_initial() -> None:
    """predict_obstacle(include_initial=True) should include step 0."""
    predictions = predict_obstacle(
        make_obstacle(),
        dt=0.1,
        horizon=2,
        include_initial=True,
    )

    assert len(predictions) == 3
    assert [prediction.step_index for prediction in predictions] == [0, 1, 2]
    assert [prediction.time for prediction in predictions] == pytest.approx([0.0, 0.1, 0.2])


def test_obstacle_prediction_rejects_invalid_step_index() -> None:
    """ObstaclePrediction should reject invalid step_index."""
    obstacle = make_obstacle()

    with pytest.raises(ObstaclePredictionError, match="step_index"):
        ObstaclePrediction(
            obstacle_id=obstacle.obstacle_id,
            step_index=-1,
            time=0.0,
            p_hat=obstacle.p_hat,
            v_hat=obstacle.v_hat,
            axes=obstacle.axes,
            R_o=obstacle.R_o,
            Sigma=obstacle.Sigma,
            Sigma_v=obstacle.Sigma_v,
            L=np.eye(3),
        )


def test_obstacle_manager_len_iter_active() -> None:
    """ObstacleManager should expose length, iteration, and active filtering."""
    active = make_obstacle()
    inactive = ObstacleState(
        obstacle_id="inactive_001",
        p_hat=[10.0, 0.0, 1.0],
        v_hat=[0.0, 0.0, 0.0],
        axes=[0.5, 0.5, 0.5],
        R_o=np.eye(3),
        Sigma=make_sigma(),
        Sigma_v=make_sigma(0.001),
        active=False,
    )

    manager = ObstacleManager([active, inactive])

    assert len(manager) == 2
    assert tuple(manager) == (active, inactive)
    assert manager.active_obstacles == (active,)


def test_obstacle_manager_add_returns_new_manager() -> None:
    """ObstacleManager.add should return a new manager without mutating original."""
    manager = ObstacleManager()
    obstacle = make_obstacle()

    new_manager = manager.add(obstacle)

    assert len(manager) == 0
    assert len(new_manager) == 1
    assert new_manager.obstacles == (obstacle,)


def test_obstacle_manager_rejects_non_obstacle_state() -> None:
    """ObstacleManager should reject non-ObstacleState entries."""
    with pytest.raises(ObstacleConfigError, match="ObstacleState"):
        ObstacleManager([object()])  # type: ignore[list-item]


def test_obstacle_manager_get_closest() -> None:
    """ObstacleManager.get_closest should return nearest active obstacles."""
    near = make_constant_velocity_obstacle(
        obstacle_id="near",
        position=[1.0, 0.0, 1.0],
        velocity=[0.0, 0.0, 0.0],
        axes=[0.3, 0.3, 0.3],
    )
    far = make_constant_velocity_obstacle(
        obstacle_id="far",
        position=[5.0, 0.0, 1.0],
        velocity=[0.0, 0.0, 0.0],
        axes=[0.3, 0.3, 0.3],
    )
    manager = ObstacleManager([far, near])

    closest = manager.get_closest([0.0, 0.0, 1.0], k=1)

    assert closest == [near]


def test_obstacle_manager_get_closest_ignores_inactive_by_default() -> None:
    """get_closest should ignore inactive obstacles unless requested."""
    active = make_constant_velocity_obstacle(
        obstacle_id="active",
        position=[5.0, 0.0, 1.0],
        velocity=[0.0, 0.0, 0.0],
        axes=[0.3, 0.3, 0.3],
    )
    inactive = ObstacleState(
        obstacle_id="inactive",
        p_hat=[1.0, 0.0, 1.0],
        v_hat=[0.0, 0.0, 0.0],
        axes=[0.3, 0.3, 0.3],
        R_o=np.eye(3),
        Sigma=make_sigma(),
        Sigma_v=make_sigma(0.001),
        active=False,
    )
    manager = ObstacleManager([inactive, active])

    assert manager.get_closest([0.0, 0.0, 1.0], k=1) == [active]
    assert manager.get_closest([0.0, 0.0, 1.0], k=1, include_inactive=True) == [inactive]


def test_obstacle_manager_get_closest_rejects_invalid_k() -> None:
    """get_closest should reject non-positive k."""
    manager = ObstacleManager([make_obstacle()])

    with pytest.raises(ObstacleConfigError, match="k"):
        manager.get_closest([0.0, 0.0, 1.0], k=0)


def test_obstacle_manager_predict_all() -> None:
    """ObstacleManager.predict_all should return time-major predictions."""
    obs_a = make_constant_velocity_obstacle(
        obstacle_id="a",
        position=[1.0, 0.0, 1.0],
        velocity=[0.1, 0.0, 0.0],
        axes=[0.3, 0.3, 0.3],
    )
    obs_b = make_constant_velocity_obstacle(
        obstacle_id="b",
        position=[2.0, 0.0, 1.0],
        velocity=[0.0, 0.1, 0.0],
        axes=[0.4, 0.4, 0.4],
    )
    manager = ObstacleManager([obs_a, obs_b])

    predictions = manager.predict_all(dt=0.1, horizon=3, mav_radius=0.2)

    assert len(predictions) == 3
    assert all(len(step_predictions) == 2 for step_predictions in predictions)
    assert predictions[0][0].step_index == 1
    assert predictions[-1][0].step_index == 3
    assert predictions[0][0].L.shape == (3, 3)


def test_obstacle_manager_predict_all_include_initial() -> None:
    """predict_all(include_initial=True) should include step 0."""
    manager = ObstacleManager([make_obstacle()])

    predictions = manager.predict_all(dt=0.1, horizon=2, include_initial=True)

    assert len(predictions) == 3
    assert [step_predictions[0].step_index for step_predictions in predictions] == [0, 1, 2]


def test_obstacle_manager_from_config() -> None:
    """ObstacleManager.from_config should parse dict config."""
    config = {
        "obstacles": [
            {
                "id": "obs_001",
                "shape": "ellipsoid",
                "motion_model": "constant_velocity",
                "position": [1.0, 0.0, 1.0],
                "velocity": [0.1, 0.0, 0.0],
                "axes": [0.4, 0.4, 0.8],
                "yaw": 0.0,
            }
        ]
    }

    manager = ObstacleManager.from_config(config)

    assert len(manager) == 1
    assert manager.obstacles[0].obstacle_id == "obs_001"


def test_obstacle_from_dict_axes() -> None:
    """obstacle_from_dict should parse explicit ellipsoid axes."""
    obstacle = obstacle_from_dict(
        {
            "id": "obs_axes",
            "shape": "ellipsoid",
            "motion_model": "constant_velocity",
            "position": [1.0, 0.0, 1.0],
            "velocity": [0.1, 0.0, 0.0],
            "axes": [0.4, 0.5, 0.8],
            "yaw": math.pi / 2.0,
            "covariance": make_sigma(0.02),
            "velocity_covariance": make_sigma(0.003),
        }
    )

    assert obstacle.obstacle_id == "obs_axes"
    assert obstacle.shape is ObstacleShape.ELLIPSOID
    assert obstacle.motion_model is ObstacleMotionModel.CONSTANT_VELOCITY
    assert np.allclose(obstacle.axes, [0.4, 0.5, 0.8])
    assert np.allclose(obstacle.R_o, yaw_to_rotation(math.pi / 2.0))


def test_obstacle_from_dict_size_box() -> None:
    """obstacle_from_dict should convert box size to ellipsoid axes."""
    size = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    obstacle = obstacle_from_dict(
        {
            "id": "box_001",
            "shape": "box",
            "motion_model": "static",
            "position": [1.0, 0.0, 1.0],
            "size": size,
        }
    )

    assert obstacle.shape is ObstacleShape.BOX
    assert obstacle.motion_model is ObstacleMotionModel.STATIC
    assert np.allclose(obstacle.axes, box_to_ellipsoid_axes(size))


def test_obstacle_from_dict_radius_sphere() -> None:
    """obstacle_from_dict should convert radius to equal ellipsoid axes."""
    obstacle = obstacle_from_dict(
        {
            "id": "sphere_001",
            "shape": "sphere",
            "motion_model": "static",
            "position": [1.0, 0.0, 1.0],
            "radius": 0.7,
        }
    )

    assert obstacle.shape is ObstacleShape.SPHERE
    assert np.allclose(obstacle.axes, [0.7, 0.7, 0.7])


def test_obstacle_from_dict_rejects_missing_position() -> None:
    """obstacle_from_dict should require a position field."""
    with pytest.raises(ObstacleConfigError, match="position"):
        obstacle_from_dict(
            {
                "id": "bad",
                "shape": "ellipsoid",
                "axes": [0.4, 0.4, 0.8],
            }
        )


def test_obstacle_from_dict_rejects_missing_axes_size_radius() -> None:
    """obstacle_from_dict should require axes, size, or radius."""
    with pytest.raises(ObstacleConfigError, match="axes"):
        obstacle_from_dict(
            {
                "id": "bad",
                "shape": "ellipsoid",
                "position": [1.0, 0.0, 1.0],
            }
        )


def test_obstacle_from_object() -> None:
    """obstacle_from_object should parse dataclass-like obstacle configs."""
    obj = ObjectObstacleConfig(
        obstacle_id="obj_001",
        position=np.array([1.0, 2.0, 1.0], dtype=np.float64),
        velocity=np.array([0.1, 0.0, 0.0], dtype=np.float64),
        size=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        yaw=0.1,
        covariance=make_sigma(0.02),
        velocity_covariance=make_sigma(0.003),
        metadata={"source": "object"},
    )

    obstacle = obstacle_from_object(obj)

    assert obstacle.obstacle_id == "obj_001"
    assert obstacle.shape is ObstacleShape.BOX
    assert np.allclose(obstacle.axes, box_to_ellipsoid_axes(obj.size))
    assert obstacle.metadata == {"source": "object"}


def test_obstacles_from_config_list() -> None:
    """obstacles_from_config should parse list of obstacle dicts."""
    obstacles = obstacles_from_config(
        [
            {
                "id": "obs_001",
                "shape": "ellipsoid",
                "position": [1.0, 0.0, 1.0],
                "velocity": [0.0, 0.0, 0.0],
                "axes": [0.4, 0.4, 0.8],
            },
            {
                "id": "obs_002",
                "shape": "sphere",
                "position": [2.0, 0.0, 1.0],
                "radius": 0.5,
            },
        ]
    )

    assert len(obstacles) == 2
    assert obstacles[0].obstacle_id == "obs_001"
    assert obstacles[1].obstacle_id == "obs_002"


def test_obstacles_from_config_dict() -> None:
    """obstacles_from_config should parse dict with obstacles key."""
    obstacles = obstacles_from_config(
        {
            "obstacles": [
                {
                    "id": "obs_001",
                    "shape": "ellipsoid",
                    "position": [1.0, 0.0, 1.0],
                    "axes": [0.4, 0.4, 0.8],
                }
            ]
        }
    )

    assert len(obstacles) == 1
    assert obstacles[0].obstacle_id == "obs_001"


def test_obstacles_from_config_object() -> None:
    """obstacles_from_config should parse object exposing .obstacles."""
    scenario = ObjectScenarioConfig(
        obstacles=(
            ObjectObstacleConfig(
                obstacle_id="obj_001",
                position=np.array([1.0, 0.0, 1.0], dtype=np.float64),
                velocity=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                size=np.array([1.0, 1.0, 1.0], dtype=np.float64),
                yaw=0.0,
                covariance=make_sigma(),
                velocity_covariance=make_sigma(0.001),
            ),
        )
    )

    obstacles = obstacles_from_config(scenario)

    assert len(obstacles) == 1
    assert obstacles[0].obstacle_id == "obj_001"


def test_obstacles_from_config_rejects_duplicate_ids() -> None:
    """obstacles_from_config should reject duplicate obstacle IDs."""
    with pytest.raises(ObstacleConfigError, match="Duplicate"):
        obstacles_from_config(
            [
                {
                    "id": "dup",
                    "shape": "ellipsoid",
                    "position": [1.0, 0.0, 1.0],
                    "axes": [0.4, 0.4, 0.8],
                },
                {
                    "id": "dup",
                    "shape": "ellipsoid",
                    "position": [2.0, 0.0, 1.0],
                    "axes": [0.4, 0.4, 0.8],
                },
            ]
        )


def test_obstacles_from_config_rejects_invalid_dict_without_obstacles_key() -> None:
    """obstacles_from_config should reject dict without obstacles key."""
    with pytest.raises(ObstacleConfigError, match="obstacles"):
        obstacles_from_config({"not_obstacles": []})


def test_default_covariances() -> None:
    """Default covariance helpers should return Sigma3x3 matrices."""
    Sigma = default_position_covariance(std=0.1)
    Sigma_v = default_velocity_covariance(std=0.2)

    assert Sigma.shape == (3, 3)
    assert Sigma_v.shape == (3, 3)
    assert np.allclose(Sigma, np.eye(3) * 0.01)
    assert np.allclose(Sigma_v, np.eye(3) * 0.04)


def test_default_covariances_reject_negative_std() -> None:
    """Default covariance helpers should reject negative std."""
    with pytest.raises(ObstacleConfigError, match="std"):
        default_position_covariance(std=-0.1)

    with pytest.raises(ObstacleConfigError, match="std"):
        default_velocity_covariance(std=-0.1)


def test_is_rotation_matrix() -> None:
    """is_rotation_matrix should validate SO(3)-like matrices."""
    assert is_rotation_matrix(np.eye(3)) is True
    assert is_rotation_matrix(yaw_to_rotation(0.5)) is True

    bad_rotation = np.eye(3)
    bad_rotation[0, 0] = 2.0

    assert is_rotation_matrix(bad_rotation) is False
