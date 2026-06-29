"""Unit tests for controller base contracts.

Target module:
    simulation.controllers.base

These tests verify the controller abstraction layer before implementing concrete
CC-MPC, PID, LQR, or emergency-stop controllers.
"""

from __future__ import annotations

import numpy as np
import pytest

from ccmpc.types import DataContractError

from simulation.controllers.base import (
    DEFAULT_CCMPC_METADATA,
    ControllerConfigurationError,
    ControllerDiagnostics,
    ControllerInput,
    ControllerInputError,
    ControllerMetadata,
    ControllerOutput,
    ControllerOutputError,
    ControllerStatus,
    ControllerType,
    ObstaclePrediction,
    first_control_from_trajectory,
    make_fallback_output,
    make_success_output,
    validate_controller_time,
)


def make_state() -> np.ndarray:
    """Create a valid canonical State9."""
    return np.array(
        [
            0.0,  # x
            0.0,  # y
            1.0,  # z
            0.1,  # vx
            0.0,  # vy
            0.0,  # vz
            0.0,  # roll
            0.0,  # pitch
            0.0,  # yaw
        ],
        dtype=np.float64,
    )


def make_goal() -> np.ndarray:
    """Create a valid canonical Goal3."""
    return np.array([5.0, 0.0, 2.0], dtype=np.float64)


def make_command() -> np.ndarray:
    """Create a valid canonical ControlCommand4."""
    return np.array(
        [
            0.05,  # phi_c
            -0.04, # theta_c
            0.2,   # vz_c
            0.1,   # psi_dot_c
        ],
        dtype=np.float64,
    )


def make_gamma() -> np.ndarray:
    """Create a valid Gamma9x9 covariance."""
    return np.eye(9, dtype=np.float64) * 0.01


def make_predicted_trajectory() -> np.ndarray:
    """Create a valid time-major State9 trajectory with shape (T, 9)."""
    trajectory = np.zeros((3, 9), dtype=np.float64)
    trajectory[:, 2] = 1.0
    trajectory[:, 0] = [0.0, 0.1, 0.2]
    return trajectory


def make_control_trajectory_time_major() -> np.ndarray:
    """Create a valid time-major ControlCommand4 trajectory with shape (T, 4)."""
    return np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ],
        dtype=np.float64,
    )


def make_control_trajectory_control_major() -> np.ndarray:
    """Create a valid control-major ControlCommand4 trajectory with shape (4, T)."""
    return np.array(
        [
            [0.1, 0.5],
            [0.2, 0.6],
            [0.3, 0.7],
            [0.4, 0.8],
        ],
        dtype=np.float64,
    )


def make_obstacle_prediction() -> ObstaclePrediction:
    """Create a valid obstacle prediction."""
    return ObstaclePrediction(
        obstacle_id="obs_001",
        positions=np.array(
            [
                [2.0, 0.0, 1.0],
                [2.1, 0.0, 1.0],
                [2.2, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        radii=np.array([0.4, 0.4, 0.8], dtype=np.float64),
        covariance=np.eye(3, dtype=np.float64) * 0.01,
        active=True,
        metadata={"source": "unit-test"},
    )


def test_controller_metadata_valid_ccmpc() -> None:
    """Default CC-MPC metadata should describe a State9 -> ControlCommand4 controller."""
    metadata = DEFAULT_CCMPC_METADATA

    assert metadata.controller_type is ControllerType.CCMPC
    assert metadata.name == "CCMPCController"
    assert metadata.state_dim == 9
    assert metadata.goal_dim == 3
    assert metadata.command_dim == 4
    assert metadata.supports_obstacles is True
    assert metadata.supports_covariance is True
    assert metadata.deterministic is True


def test_controller_metadata_reject_wrong_state_dim() -> None:
    """Controller metadata rejects non-State9 public state dimension."""
    with pytest.raises(ControllerConfigurationError, match="state_dim"):
        ControllerMetadata(
            controller_type=ControllerType.CCMPC,
            name="BadController",
            state_dim=8,
            goal_dim=3,
            command_dim=4,
        )


def test_controller_metadata_reject_wrong_goal_dim() -> None:
    """Controller metadata rejects non-Goal3 public goal dimension."""
    with pytest.raises(ControllerConfigurationError, match="goal_dim"):
        ControllerMetadata(
            controller_type=ControllerType.CCMPC,
            name="BadController",
            state_dim=9,
            goal_dim=2,
            command_dim=4,
        )


def test_controller_metadata_reject_wrong_command_dim() -> None:
    """Controller metadata rejects non-ControlCommand4 public command dimension."""
    with pytest.raises(ControllerConfigurationError, match="command_dim"):
        ControllerMetadata(
            controller_type=ControllerType.CCMPC,
            name="BadController",
            state_dim=9,
            goal_dim=3,
            command_dim=3,
        )


def test_controller_metadata_reject_empty_name() -> None:
    """Controller metadata requires a non-empty name."""
    with pytest.raises(ControllerConfigurationError, match="name"):
        ControllerMetadata(
            controller_type=ControllerType.CCMPC,
            name="",
            state_dim=9,
            goal_dim=3,
            command_dim=4,
        )


def test_controller_metadata_reject_invalid_horizon() -> None:
    """Controller metadata rejects non-positive horizon values."""
    with pytest.raises(ControllerConfigurationError, match="horizon"):
        ControllerMetadata(
            controller_type=ControllerType.CCMPC,
            name="BadController",
            horizon=0,
        )


def test_controller_metadata_reject_invalid_dt() -> None:
    """Controller metadata rejects non-positive or non-finite dt values."""
    invalid_dt_values = [0.0, -0.01, np.nan, np.inf, True]

    for invalid_dt in invalid_dt_values:
        with pytest.raises(ControllerConfigurationError, match="dt"):
            ControllerMetadata(
                controller_type=ControllerType.CCMPC,
                name="BadController",
                dt=invalid_dt,
            )


def test_validate_controller_time() -> None:
    """validate_controller_time accepts finite non-negative time."""
    assert validate_controller_time(0.0) == pytest.approx(0.0)
    assert validate_controller_time(1.25) == pytest.approx(1.25)
    assert validate_controller_time(np.float64(0.5)) == pytest.approx(0.5)


def test_validate_controller_time_rejects_invalid_values() -> None:
    """validate_controller_time rejects negative, non-finite, and bool values."""
    invalid_time_values = [-0.01, np.nan, np.inf, True]

    for invalid_time in invalid_time_values:
        with pytest.raises(ControllerInputError, match="time"):
            validate_controller_time(invalid_time)


def test_obstacle_prediction_valid() -> None:
    """ObstaclePrediction accepts position trajectory, radii, and covariance."""
    obstacle = make_obstacle_prediction()

    assert obstacle.obstacle_id == "obs_001"
    assert obstacle.positions.shape == (3, 3)
    assert obstacle.radii is not None
    assert obstacle.radii.shape == (3,)
    assert obstacle.covariance is not None
    assert obstacle.covariance.shape == (3, 3)
    assert obstacle.active is True
    assert obstacle.metadata == {"source": "unit-test"}


def test_obstacle_prediction_accepts_time_varying_covariance() -> None:
    """ObstaclePrediction accepts covariance with shape (T, 3, 3)."""
    obstacle = ObstaclePrediction(
        obstacle_id="obs_001",
        positions=np.zeros((2, 3), dtype=np.float64),
        covariance=np.stack([np.eye(3), np.eye(3) * 2.0], axis=0),
    )

    assert obstacle.covariance is not None
    assert obstacle.covariance.shape == (2, 3, 3)


def test_obstacle_prediction_reject_empty_id() -> None:
    """ObstaclePrediction rejects empty obstacle ID."""
    with pytest.raises(ControllerInputError, match="obstacle_id"):
        ObstaclePrediction(
            obstacle_id="",
            positions=np.zeros((2, 3), dtype=np.float64),
        )


def test_obstacle_prediction_reject_invalid_positions_shape() -> None:
    """ObstaclePrediction rejects positions not shaped as (T, 3)."""
    with pytest.raises(ControllerInputError, match="positions"):
        ObstaclePrediction(
            obstacle_id="obs_001",
            positions=np.zeros((3,), dtype=np.float64),
        )


def test_obstacle_prediction_reject_empty_positions() -> None:
    """ObstaclePrediction rejects empty prediction horizon."""
    with pytest.raises(ControllerInputError, match="at least one"):
        ObstaclePrediction(
            obstacle_id="obs_001",
            positions=np.zeros((0, 3), dtype=np.float64),
        )


def test_obstacle_prediction_reject_non_positive_radii() -> None:
    """ObstaclePrediction rejects zero or negative obstacle radii."""
    with pytest.raises(ControllerInputError, match="radii"):
        ObstaclePrediction(
            obstacle_id="obs_001",
            positions=np.zeros((2, 3), dtype=np.float64),
            radii=np.array([0.4, 0.0, 0.8], dtype=np.float64),
        )


def test_obstacle_prediction_reject_non_symmetric_covariance() -> None:
    """ObstaclePrediction rejects non-symmetric obstacle covariance."""
    covariance = np.eye(3, dtype=np.float64)
    covariance[0, 1] = 0.2

    with pytest.raises(ControllerInputError, match="symmetric"):
        ObstaclePrediction(
            obstacle_id="obs_001",
            positions=np.zeros((2, 3), dtype=np.float64),
            covariance=covariance,
        )


def test_controller_input_valid() -> None:
    """ControllerInput accepts valid canonical controller input."""
    obstacle = make_obstacle_prediction()

    input_data = ControllerInput(
        time=0.1,
        estimated_state=make_state(),
        goal=make_goal(),
        covariance=make_gamma(),
        obstacle_predictions=(obstacle,),
        previous_solution={"warm_start": True},
        reference_trajectory=make_predicted_trajectory(),
        config={"solver": "CLARABEL"},
        metadata={"runtime_mode": "deterministic_single_thread"},
    )

    assert input_data.time == pytest.approx(0.1)
    assert input_data.estimated_state.shape == (9,)
    assert input_data.goal.shape == (3,)
    assert input_data.covariance is not None
    assert input_data.covariance.shape == (9, 9)
    assert input_data.obstacle_predictions == (obstacle,)
    assert input_data.reference_trajectory is not None
    assert input_data.reference_trajectory.shape == (3, 9)
    assert input_data.metadata == {"runtime_mode": "deterministic_single_thread"}


def test_controller_input_reject_invalid_state() -> None:
    """ControllerInput rejects estimated_state that is not State9."""
    with pytest.raises(DataContractError, match="State9"):
        ControllerInput(
            time=0.0,
            estimated_state=np.zeros(8, dtype=np.float64),
            goal=make_goal(),
        )


def test_controller_input_reject_invalid_goal() -> None:
    """ControllerInput rejects goal that is not Goal3."""
    with pytest.raises(DataContractError, match="Goal3"):
        ControllerInput(
            time=0.0,
            estimated_state=make_state(),
            goal=np.zeros(2, dtype=np.float64),
        )


def test_controller_input_reject_invalid_covariance() -> None:
    """ControllerInput rejects covariance that is not Gamma9x9."""
    with pytest.raises(DataContractError, match="Gamma9x9"):
        ControllerInput(
            time=0.0,
            estimated_state=make_state(),
            goal=make_goal(),
            covariance=np.eye(8, dtype=np.float64),
        )


def test_controller_input_reject_non_tuple_obstacles() -> None:
    """ControllerInput requires obstacle_predictions to be tuple."""
    with pytest.raises(ControllerInputError, match="obstacle_predictions"):
        ControllerInput(
            time=0.0,
            estimated_state=make_state(),
            goal=make_goal(),
            obstacle_predictions=[make_obstacle_prediction()],  # type: ignore[arg-type]
        )


def test_controller_input_reject_invalid_reference_trajectory() -> None:
    """ControllerInput rejects reference trajectory not shaped as (T, 9)."""
    with pytest.raises(DataContractError, match="Trajectory9"):
        ControllerInput(
            time=0.0,
            estimated_state=make_state(),
            goal=make_goal(),
            reference_trajectory=np.zeros((9, 3), dtype=np.float64),
        )


def test_controller_diagnostics_valid() -> None:
    """ControllerDiagnostics accepts valid solver diagnostics."""
    diagnostics = ControllerDiagnostics(
        status=ControllerStatus.SUCCESS,
        success=True,
        solve_time_ms=12.5,
        objective_value=42.0,
        iterations=7,
        fallback_used=False,
        max_constraint_violation=0.0,
        min_obstacle_margin=0.5,
        notes=("ok",),
        extra={"solver": "CLARABEL"},
    )

    assert diagnostics.status is ControllerStatus.SUCCESS
    assert diagnostics.success is True
    assert diagnostics.solve_time_ms == pytest.approx(12.5)
    assert diagnostics.iterations == 7
    assert diagnostics.notes == ("ok",)


def test_controller_diagnostics_reject_invalid_solve_time() -> None:
    """ControllerDiagnostics rejects negative solve time."""
    with pytest.raises(ControllerOutputError, match="solve_time_ms"):
        ControllerDiagnostics(
            status=ControllerStatus.SUCCESS,
            success=True,
            solve_time_ms=-1.0,
        )


def test_controller_output_valid() -> None:
    """ControllerOutput accepts valid command, trajectories, and diagnostics."""
    diagnostics = ControllerDiagnostics(
        status=ControllerStatus.SUCCESS,
        success=True,
        solve_time_ms=10.0,
    )

    output = ControllerOutput(
        command=make_command(),
        predicted_trajectory=make_predicted_trajectory(),
        control_trajectory=make_control_trajectory_time_major(),
        diagnostics=diagnostics,
        raw_solution={"status": "optimal"},
        metadata={"controller": "unit-test"},
    )

    assert output.command.shape == (4,)
    assert output.predicted_trajectory is not None
    assert output.predicted_trajectory.shape == (3, 9)
    assert output.control_trajectory is not None
    assert output.control_trajectory.shape == (2, 4)
    assert output.diagnostics is diagnostics
    assert output.raw_solution == {"status": "optimal"}
    assert output.metadata == {"controller": "unit-test"}


def test_controller_output_reject_invalid_command() -> None:
    """ControllerOutput rejects command that is not ControlCommand4."""
    with pytest.raises(DataContractError, match="ControlCommand4"):
        ControllerOutput(command=np.zeros(3, dtype=np.float64))


def test_controller_output_reject_invalid_predicted_trajectory() -> None:
    """ControllerOutput rejects predicted trajectory not shaped as (T, 9)."""
    with pytest.raises(DataContractError, match="Trajectory9"):
        ControllerOutput(
            command=make_command(),
            predicted_trajectory=np.zeros((9, 3), dtype=np.float64),
        )


def test_controller_output_reject_invalid_control_trajectory() -> None:
    """ControllerOutput rejects control trajectory not shaped as (T, 4)."""
    with pytest.raises(DataContractError, match="ControlTrajectory4"):
        ControllerOutput(
            command=make_command(),
            control_trajectory=np.zeros((4, 2), dtype=np.float64),
        )


def test_first_control_from_time_major_trajectory() -> None:
    """first_control_from_trajectory extracts first row from shape (T, 4)."""
    trajectory = make_control_trajectory_time_major()

    command = first_control_from_trajectory(trajectory, layout="time_major")

    assert command.shape == (4,)
    assert np.allclose(command, [0.1, 0.2, 0.3, 0.4])


def test_first_control_from_control_major_trajectory() -> None:
    """first_control_from_trajectory extracts first column from shape (4, T)."""
    trajectory = make_control_trajectory_control_major()

    command = first_control_from_trajectory(trajectory, layout="control_major")

    assert command.shape == (4,)
    assert np.allclose(command, [0.1, 0.2, 0.3, 0.4])


def test_first_control_from_trajectory_reject_invalid_layout() -> None:
    """first_control_from_trajectory rejects unknown layout."""
    with pytest.raises(ControllerOutputError, match="Unsupported"):
        first_control_from_trajectory(
            make_control_trajectory_time_major(),
            layout="bad_layout",
        )


def test_make_success_output() -> None:
    """make_success_output creates a valid successful ControllerOutput."""
    output = make_success_output(
        command=make_command(),
        predicted_trajectory=make_predicted_trajectory(),
        control_trajectory=make_control_trajectory_time_major(),
        solve_time_ms=8.0,
        objective_value=12.0,
        iterations=5,
        raw_solution={"status": "optimal"},
        metadata={"source": "unit-test"},
        diagnostics_extra={"solver": "CLARABEL"},
    )

    assert output.command.shape == (4,)
    assert output.diagnostics.status is ControllerStatus.SUCCESS
    assert output.diagnostics.success is True
    assert output.diagnostics.fallback_used is False
    assert output.diagnostics.solve_time_ms == pytest.approx(8.0)
    assert output.diagnostics.objective_value == pytest.approx(12.0)
    assert output.diagnostics.iterations == 5
    assert output.diagnostics.extra == {"solver": "CLARABEL"}
    assert output.raw_solution == {"status": "optimal"}
    assert output.metadata == {"source": "unit-test"}


def test_make_fallback_output() -> None:
    """make_fallback_output creates a valid fallback ControllerOutput."""
    output = make_fallback_output(
        command=np.zeros(4, dtype=np.float64),
        reason="solver infeasible",
        status=ControllerStatus.FALLBACK,
        metadata={"source": "fallback"},
    )

    assert output.command.shape == (4,)
    assert output.diagnostics.status is ControllerStatus.FALLBACK
    assert output.diagnostics.success is False
    assert output.diagnostics.fallback_used is True
    assert output.diagnostics.fallback_reason == "solver infeasible"
    assert output.metadata == {"source": "fallback"}


def test_make_fallback_output_reject_empty_reason() -> None:
    """make_fallback_output requires a non-empty reason."""
    with pytest.raises(ControllerOutputError, match="reason"):
        make_fallback_output(
            command=np.zeros(4, dtype=np.float64),
            reason="",
        )
