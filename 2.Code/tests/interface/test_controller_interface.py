"""Interface tests for simulation.controllers.metadata.

These tests verify the controller public data contracts defined by the design
interface.  They intentionally do not solve MPC problems, step physics engines,
or depend on runtime orchestration.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulation.controllers.metadata import (
    ControllerDiagnostics,
    ControllerDiagnosticsError,
    ControllerInput,
    ControllerInputError,
    ControllerMetadata,
    ControllerMetadataError,
    ControllerOutput,
    ControllerOutputError,
    ControllerStatus,
    ControllerType,
    make_ccmpc_metadata,
    make_emergency_stop_metadata,
    make_pid_metadata,
    parse_controller_status,
    parse_controller_type,
)


def make_state9(z: float = 1.0) -> np.ndarray:
    """Return a valid canonical State9 vector."""
    return np.array(
        [
            0.0,  # x
            0.0,  # y
            z,    # z
            0.0,  # vx
            0.0,  # vy
            0.0,  # vz
            0.0,  # roll
            0.0,  # pitch
            0.0,  # yaw
        ],
        dtype=float,
    )


def make_goal3() -> np.ndarray:
    """Return a valid canonical Goal3 vector."""
    return np.array([1.0, 2.0, 1.5], dtype=float)


def make_gamma9x9() -> np.ndarray:
    """Return a valid symmetric positive-semidefinite Gamma9x9."""
    return np.diag(
        [
            0.05**2,
            0.05**2,
            0.05**2,
            0.10**2,
            0.10**2,
            0.10**2,
            0.03**2,
            0.03**2,
            0.03**2,
        ]
    )


def make_command4() -> np.ndarray:
    """Return a valid canonical ControlCommand4."""
    return np.array([0.05, -0.03, 0.20, 0.10], dtype=float)


def make_trajectory9(horizon: int = 3) -> np.ndarray:
    """Return a time-major predicted State9 trajectory with shape (N+1, 9)."""
    trajectory = np.zeros((horizon + 1, 9), dtype=float)
    trajectory[:, 2] = 1.0
    return trajectory


def make_control_trajectory4(horizon: int = 3) -> np.ndarray:
    """Return a time-major ControlTrajectory4 with shape (N, 4)."""
    return np.zeros((horizon, 4), dtype=float)


def make_diagnostics(
    status: ControllerStatus | str = ControllerStatus.SUCCESS,
    *,
    success: bool = True,
) -> ControllerDiagnostics:
    """Return a valid ControllerDiagnostics object."""
    return ControllerDiagnostics(
        status=status,
        success=success,
        solve_time_ms=3.5,
        objective_value=12.0,
        iterations=2,
        fallback_used=False,
        fallback_reason=None,
        max_constraint_violation=0.0,
        min_obstacle_margin=0.7,
        notes={"solver": "CLARABEL"},
    )


def test_parse_controller_type_accepts_enum_and_strings() -> None:
    """ControllerType parser should accept enum values and stable strings."""
    assert parse_controller_type(ControllerType.CCMPC) is ControllerType.CCMPC
    assert parse_controller_type("ccmpc") is ControllerType.CCMPC
    assert parse_controller_type("CCMPC") is ControllerType.CCMPC
    assert (
        parse_controller_type("ControllerType.CCMPC")
        is ControllerType.CCMPC
    )


def test_parse_controller_type_rejects_invalid_value() -> None:
    """Unknown controller types should fail at the interface boundary."""
    with pytest.raises(ControllerMetadataError):
        parse_controller_type("unknown_controller")


def test_parse_controller_status_accepts_enum_and_strings() -> None:
    """ControllerStatus parser should accept enum values and stable strings."""
    assert parse_controller_status(ControllerStatus.SUCCESS) is ControllerStatus.SUCCESS
    assert parse_controller_status("success") is ControllerStatus.SUCCESS
    assert parse_controller_status("SOLVER_ERROR") is ControllerStatus.SOLVER_ERROR
    assert (
        parse_controller_status("ControllerStatus.FALLBACK")
        is ControllerStatus.FALLBACK
    )


def test_parse_controller_status_rejects_invalid_value() -> None:
    """Unknown controller status values should fail at diagnostics boundary."""
    with pytest.raises(ControllerDiagnosticsError):
        parse_controller_status("not_a_status")


def test_controller_metadata_valid_ccmpc_contract() -> None:
    """CC-MPC metadata should expose the canonical controller contract."""
    metadata = make_ccmpc_metadata(notes="test")

    assert metadata.controller_type is ControllerType.CCMPC
    assert metadata.input_state_type == "State9"
    assert metadata.output_command_type == "ControlCommand4"
    assert metadata.supports_obstacles is True
    assert metadata.supports_uncertainty is True
    assert metadata.supports_warm_start is True
    assert metadata.supports_predicted_trajectory is True
    assert metadata.notes == "test"


def test_controller_metadata_valid_optional_controllers() -> None:
    """Future controllers should still expose State9 -> ControlCommand4."""
    pid_metadata = make_pid_metadata()
    emergency_metadata = make_emergency_stop_metadata()

    assert pid_metadata.controller_type is ControllerType.PID
    assert pid_metadata.input_state_type == "State9"
    assert pid_metadata.output_command_type == "ControlCommand4"
    assert pid_metadata.supports_obstacles is False

    assert emergency_metadata.controller_type is ControllerType.EMERGENCY_STOP
    assert emergency_metadata.input_state_type == "State9"
    assert emergency_metadata.output_command_type == "ControlCommand4"


def test_controller_metadata_rejects_noncanonical_state_type() -> None:
    """Controller metadata must not advertise non-State9 public input."""
    with pytest.raises(ControllerMetadataError, match="State9"):
        ControllerMetadata(
            controller_type="ccmpc",
            input_state_type="MuJoCoQpos",
            output_command_type="ControlCommand4",
            supports_obstacles=True,
            supports_uncertainty=True,
            supports_warm_start=True,
            supports_predicted_trajectory=True,
        )


def test_controller_metadata_rejects_noncanonical_command_type() -> None:
    """Controller metadata must not advertise actuator commands as output."""
    with pytest.raises(ControllerMetadataError, match="ControlCommand4"):
        ControllerMetadata(
            controller_type="ccmpc",
            input_state_type="State9",
            output_command_type="ActuatorCommand4",
            supports_obstacles=True,
            supports_uncertainty=True,
            supports_warm_start=True,
            supports_predicted_trajectory=True,
        )


def test_controller_diagnostics_valid_success() -> None:
    """Diagnostics should preserve solver/controller information."""
    diagnostics = make_diagnostics(status="success", success=True)

    assert diagnostics.status is ControllerStatus.SUCCESS
    assert diagnostics.success is True
    assert diagnostics.solve_time_ms == pytest.approx(3.5)
    assert diagnostics.objective_value == pytest.approx(12.0)
    assert diagnostics.iterations == 2
    assert diagnostics.fallback_used is False
    assert diagnostics.fallback_reason is None
    assert diagnostics.max_constraint_violation == pytest.approx(0.0)
    assert diagnostics.min_obstacle_margin == pytest.approx(0.7)
    assert diagnostics.notes == {"solver": "CLARABEL"}


def test_controller_diagnostics_valid_fallback_requires_reason() -> None:
    """Fallback diagnostics must include an explicit reason."""
    diagnostics = ControllerDiagnostics(
        status=ControllerStatus.FALLBACK,
        success=False,
        fallback_used=True,
        fallback_reason="solver_failed",
    )

    assert diagnostics.status is ControllerStatus.FALLBACK
    assert diagnostics.success is False
    assert diagnostics.fallback_used is True
    assert diagnostics.fallback_reason == "solver_failed"


def test_controller_diagnostics_rejects_fallback_without_reason() -> None:
    """A fallback flag without a reason is not debuggable."""
    with pytest.raises(ControllerDiagnosticsError, match="fallback_reason"):
        ControllerDiagnostics(
            status=ControllerStatus.FALLBACK,
            success=False,
            fallback_used=True,
            fallback_reason=None,
        )


def test_controller_diagnostics_rejects_negative_solve_time() -> None:
    """Solve time is a duration and must be non-negative."""
    with pytest.raises(ControllerDiagnosticsError, match="solve_time_ms"):
        ControllerDiagnostics(
            status=ControllerStatus.SUCCESS,
            success=True,
            solve_time_ms=-1.0,
        )


def test_controller_diagnostics_rejects_negative_iterations() -> None:
    """Solver iteration count must be non-negative."""
    with pytest.raises(ControllerDiagnosticsError, match="iterations"):
        ControllerDiagnostics(
            status=ControllerStatus.SUCCESS,
            success=True,
            iterations=-1,
        )


def test_controller_input_valid_full_package() -> None:
    """ControllerInput should accept the canonical runtime package."""
    input_data = ControllerInput(
        time=0.12,
        estimated_state=make_state9(),
        goal=make_goal3(),
        covariance=make_gamma9x9(),
        obstacle_predictions={"obstacles": []},
        previous_solution=None,
        reference_trajectory=make_trajectory9(horizon=3),
        config={"horizon": 3, "dt": 0.06},
    )

    assert input_data.time == pytest.approx(0.12)
    assert input_data.estimated_state.shape == (9,)
    assert input_data.goal.shape == (3,)
    assert input_data.covariance is not None
    assert input_data.covariance.shape == (9, 9)
    assert input_data.reference_trajectory is not None
    assert input_data.reference_trajectory.shape == (4, 9)
    assert input_data.obstacle_predictions == {"obstacles": []}


def test_controller_input_accepts_none_covariance_for_generic_controllers() -> None:
    """The interface allows covariance=None for controllers that do not use it."""
    input_data = ControllerInput(
        time=0.0,
        estimated_state=make_state9(),
        goal=make_goal3(),
        covariance=None,
        obstacle_predictions=None,
        previous_solution=None,
        reference_trajectory=None,
        config={},
    )

    assert input_data.covariance is None
    assert input_data.reference_trajectory is None


def test_controller_input_rejects_negative_time() -> None:
    """Runtime/controller time must be non-negative."""
    with pytest.raises(ControllerInputError, match="time"):
        ControllerInput(
            time=-0.01,
            estimated_state=make_state9(),
            goal=make_goal3(),
            covariance=None,
            obstacle_predictions=None,
            previous_solution=None,
            reference_trajectory=None,
            config={},
        )


def test_controller_input_rejects_bad_estimated_state_shape() -> None:
    """estimated_state must be State9-compatible."""
    with pytest.raises(ControllerInputError, match="estimated_state"):
        ControllerInput(
            time=0.0,
            estimated_state=np.zeros(8),
            goal=make_goal3(),
            covariance=None,
            obstacle_predictions=None,
            previous_solution=None,
            reference_trajectory=None,
            config={},
        )


def test_controller_input_rejects_bad_goal_shape() -> None:
    """goal must be Goal3-compatible."""
    with pytest.raises(ControllerInputError, match="goal"):
        ControllerInput(
            time=0.0,
            estimated_state=make_state9(),
            goal=np.zeros(4),
            covariance=None,
            obstacle_predictions=None,
            previous_solution=None,
            reference_trajectory=None,
            config={},
        )


def test_controller_input_rejects_bad_covariance_shape() -> None:
    """covariance must be Gamma9x9 when provided."""
    with pytest.raises(ControllerInputError, match="covariance"):
        ControllerInput(
            time=0.0,
            estimated_state=make_state9(),
            goal=make_goal3(),
            covariance=np.eye(3),
            obstacle_predictions=None,
            previous_solution=None,
            reference_trajectory=None,
            config={},
        )


def test_controller_input_rejects_bad_reference_trajectory_shape() -> None:
    """reference_trajectory must be Trajectory9 when provided."""
    with pytest.raises(ControllerInputError, match="reference_trajectory"):
        ControllerInput(
            time=0.0,
            estimated_state=make_state9(),
            goal=make_goal3(),
            covariance=None,
            obstacle_predictions=None,
            previous_solution=None,
            reference_trajectory=np.zeros((3, 8)),
            config={},
        )


def test_controller_output_valid_full_package() -> None:
    """ControllerOutput should expose command, trajectories, and diagnostics."""
    output = ControllerOutput(
        command=make_command4(),
        predicted_trajectory=make_trajectory9(horizon=3),
        control_trajectory=make_control_trajectory4(horizon=3),
        diagnostics=make_diagnostics(),
    )

    assert output.command.shape == (4,)
    assert output.predicted_trajectory is not None
    assert output.predicted_trajectory.shape == (4, 9)
    assert output.control_trajectory is not None
    assert output.control_trajectory.shape == (3, 4)
    assert output.diagnostics.status is ControllerStatus.SUCCESS


def test_controller_output_accepts_missing_mpc_trajectories() -> None:
    """Non-MPC controllers may return only command and diagnostics."""
    output = ControllerOutput(
        command=make_command4(),
        predicted_trajectory=None,
        control_trajectory=None,
        diagnostics=make_diagnostics(),
    )

    assert output.command.shape == (4,)
    assert output.predicted_trajectory is None
    assert output.control_trajectory is None


def test_controller_output_rejects_bad_command_shape() -> None:
    """command must be ControlCommand4-compatible."""
    with pytest.raises(ControllerOutputError, match="command"):
        ControllerOutput(
            command=np.zeros(3),
            predicted_trajectory=None,
            control_trajectory=None,
            diagnostics=make_diagnostics(),
        )


def test_controller_output_rejects_bad_predicted_trajectory_shape() -> None:
    """predicted_trajectory must be Trajectory9 when provided."""
    with pytest.raises(ControllerOutputError, match="predicted_trajectory"):
        ControllerOutput(
            command=make_command4(),
            predicted_trajectory=np.zeros((4, 8)),
            control_trajectory=None,
            diagnostics=make_diagnostics(),
        )


def test_controller_output_rejects_bad_control_trajectory_shape() -> None:
    """control_trajectory must be ControlTrajectory4 when provided."""
    with pytest.raises(ControllerOutputError, match="control_trajectory"):
        ControllerOutput(
            command=make_command4(),
            predicted_trajectory=None,
            control_trajectory=np.zeros((3, 5)),
            diagnostics=make_diagnostics(),
        )


def test_controller_output_rejects_missing_diagnostics_object() -> None:
    """diagnostics must be the canonical ControllerDiagnostics object."""
    with pytest.raises(ControllerOutputError, match="diagnostics"):
        ControllerOutput(
            command=make_command4(),
            predicted_trajectory=None,
            control_trajectory=None,
            diagnostics={"status": "success"},  # type: ignore[arg-type]
        )
