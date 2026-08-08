"""Native closed-loop application runtime.

The do-mpc/CasADi controller computes one receding-horizon command per MPC tick.
The MuJoCo plant integrates that command with an independent 13-state rigid-body
model, preserving the intended model-mismatch and contact-collision validation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from quadrotor_mpc.application.native.commands import CommandName
from quadrotor_mpc.control.nmpc.chance_constraints import ChanceConstraintOptions
from quadrotor_mpc.control.nmpc.core import (
    DRONE_RADIUS,
    quat_from_euler,
    quat_to_euler,
)
from quadrotor_mpc.control.nmpc.covariance import CovariancePropagationOptions
from quadrotor_mpc.control.nmpc.safety import (
    SafeFallbackController,
    SafetyFallbackOptions,
)
from quadrotor_mpc.core.contracts import ControlGoal, Controller, VehicleBelief
from quadrotor_mpc.core.obstacle_motion import predict_obstacle_positions
from quadrotor_mpc.core.vehicle import DEFAULT_QUADROTOR
from quadrotor_mpc.estimation.native import EstimationOptions, NativeBeliefEstimator
from quadrotor_mpc.estimation.truth import exact_obstacle_beliefs
from quadrotor_mpc.infrastructure.mujoco.plant import MuJoCoPlant


@dataclass(frozen=True, slots=True)
class CoupledRunContext:
    """Immutable scene information shared with an optional runtime observer."""

    start_position: np.ndarray
    goal_position: np.ndarray
    obstacles: tuple[dict, ...]
    safety_radii: np.ndarray
    controller_timestep_s: float
    horizon_steps: int
    total_steps: int


@dataclass(frozen=True, slots=True)
class CoupledStep:
    """One completed MuJoCo control step exposed to a viewer or logger."""

    step_index: int
    time_s: float
    state_13: np.ndarray
    control: np.ndarray
    obstacle_positions: np.ndarray
    obstacle_predictions: np.ndarray
    estimated_state_13: np.ndarray | None
    error_covariance_12: np.ndarray | None
    estimated_obstacle_states: np.ndarray | None
    obstacle_covariances: np.ndarray | None
    estimated_obstacle_predictions: np.ndarray | None
    vehicle_measurement_available: bool
    obstacle_measurement_available: np.ndarray | None
    vehicle_measurement_state_13: np.ndarray | None
    obstacle_measurement_positions: np.ndarray | None
    predicted_positions: np.ndarray | None
    min_clearance_m: float
    goal_distance_m: float
    solver_time_ms: float
    collided: bool
    paused: bool = False
    predicted_covariances: np.ndarray | None = None
    chance_margins: np.ndarray | None = None
    risk_allocations: np.ndarray | None = None
    slacks: np.ndarray | None = None
    solver_status: str = ""
    predicted_obstacle_covariances: np.ndarray | None = None
    projected_uncertainties: np.ndarray | None = None
    tightened_safety_radii: np.ndarray | None = None
    risk_semantics: str = ""
    risk_allocation_method: str = ""
    risk_budget_total: float | None = None
    risk_budget_allocated: float = 0.0
    risk_budget_remaining: float | None = None
    risk_constraint_count: int = 0
    risk_budget_status: str = ""
    primary_solver_status: str = ""
    primary_solver_success: bool = True
    primary_solver_iterations: int = 0
    primary_solver_primal_residual: float = 0.0
    primary_solver_dual_residual: float = 0.0
    residual_status: str = "UNAVAILABLE"
    command_source: str = "PRIMARY_NMPC"
    solution_accepted: bool = True
    fallback_active: bool = False
    fallback_level: int = 0
    fallback_reason: str = ""
    consecutive_rejections: int = 0
    deadline_missed: bool = False
    safety_assurance_status: str = ""
    horizon_assurance_status: str = ""
    horizon_assurance_eligible: bool = False
    horizon_assurance_reason: str = ""
    horizon_assurance_failed_checks: tuple[str, ...] = ()
    assurance_schema_version: int = 2


class CoupledRuntime(Protocol):
    """Lifecycle contract for optional real-time viewers."""

    def open(self, plant: MuJoCoPlant, context: CoupledRunContext) -> None: ...

    def is_running(self) -> bool: ...

    def on_step(self, step: CoupledStep) -> bool: ...

    def poll_commands(self) -> list: ...

    def on_idle(self, paused: bool) -> None: ...

    def on_reset(self) -> None: ...

    def on_completed(self, reason: str) -> None: ...

    def close(self) -> None: ...


def run_coupled_simulation(
    x0_vals,
    goal_pos,
    goal_euler,
    bounds,
    obstacles,
    margin,
    sim_seconds=12.0,
    mpc_dt=0.05,
    n_horizon=20,
    max_iter=60,
    mj_dt=0.002,
    progress_cb=None,
    capture_frames=False,
    render_every=1,
    cached=None,
    runtime: CoupledRuntime | None = None,
    stop_on_goal=False,
    goal_tolerance=0.25,
    stop_on_collision=False,
    controller: Controller | None = None,
    estimation_options: EstimationOptions | None = None,
    covariance_options: CovariancePropagationOptions | None = None,
    chance_options: ChanceConstraintOptions | None = None,
    safety_fallback_options: SafetyFallbackOptions | None = None,
):
    """
    `cached`: optional (model, mpc, dyn_idx, goal_state) tuple from
    quad_mpc_core.build_cached_mpc(...) to reuse an already-built/compiled MPC
    controller across multiple runs (different start/goal) instead of rebuilding
    it every time - see build_cached_mpc's docstring. Profiling showed MuJoCo
    stepping itself costs <1ms/tick, so essentially all per-tick cost is the MPC
    solve; caching (and optionally JIT) is where the real speedup comes from, not
    anything MuJoCo-specific.
    """
    if controller is not None and cached is not None:
        raise ValueError("pass either controller or cached, not both")
    active_controller = controller
    if active_controller is None:
        effective_chance_options = (
            ChanceConstraintOptions() if chance_options is None else chance_options
        )
        if effective_chance_options.enabled:
            from quadrotor_mpc.control.nmpc.chance_constrained import (
                SphericalChanceConstrainedNMPCController,
            )

            controller_type = SphericalChanceConstrainedNMPCController
        else:
            from quadrotor_mpc.control.nmpc.deterministic import DeterministicNMPCController

            controller_type = DeterministicNMPCController
        active_controller = controller_type(
            bounds=bounds,
            obstacle_specs=obstacles,
            margin=margin,
            horizon_steps=n_horizon,
            timestep_s=mpc_dt,
            max_iter=max_iter,
            cached=cached,
            covariance_options=covariance_options,
            chance_options=effective_chance_options,
        )
    elif int(active_controller.horizon_steps) != int(n_horizon):
        raise ValueError(
            "injected controller horizon_steps must match run_coupled_simulation n_horizon"
        )
    effective_fallback_options = (
        SafetyFallbackOptions()
        if safety_fallback_options is None
        else safety_fallback_options
    )
    if effective_fallback_options.enabled and not isinstance(
        active_controller,
        SafeFallbackController,
    ):
        active_controller = SafeFallbackController(
            active_controller,
            options=effective_fallback_options,
            bounds=bounds,
        )

    plant = MuJoCoPlant(x0_vals, goal_pos, obstacles, mj_dt=mj_dt)

    renderer, render_err = (None, "not requested")
    if capture_frames:
        renderer, render_err = plant.try_create_renderer()

    q0 = quat_from_euler(x0_vals.get("roll", 0), x0_vals.get("pitch", 0), x0_vals.get("yaw", 0))
    x0 = np.array([x0_vals["x"], x0_vals["y"], x0_vals["z"], 0, 0, 0, *q0, 0, 0, 0]).reshape(-1, 1)
    estimator: NativeBeliefEstimator | None = None
    if estimation_options is not None and estimation_options.enabled:
        estimator = NativeBeliefEstimator(
            estimation_options,
            obstacles,
            horizon_steps=active_controller.horizon_steps,
            timestep_s=mpc_dt,
        )

    def truth_obstacle_positions(time_s: float) -> np.ndarray:
        if not obstacles:
            return np.empty((0, 3), dtype=float)
        return predict_obstacle_positions(obstacles, time_s, 1, mpc_dt)[:, 0, :]

    def reset_beliefs():
        if estimator is not None:
            return estimator.reset(x0[:, 0], truth_obstacle_positions(0.0))
        exact_vehicle = VehicleBelief.exact(x0[:, 0])
        exact_obstacles = exact_obstacle_beliefs(
            obstacles,
            0.0,
            active_controller.horizon_steps,
            mpc_dt,
        )
        return exact_vehicle, exact_obstacles

    initial_estimation = reset_beliefs()
    if estimator is not None:
        current_belief = initial_estimation.vehicle_belief
        current_obstacle_beliefs = initial_estimation.obstacle_beliefs
        current_vehicle_measurement_available = initial_estimation.vehicle_measurement_available
        current_obstacle_measurement_available = initial_estimation.obstacle_measurement_available
        current_vehicle_measurement_state = initial_estimation.vehicle_measurement_state_13
        current_obstacle_measurement_positions = initial_estimation.obstacle_measurement_positions
    else:
        current_belief, current_obstacle_beliefs = initial_estimation
        current_vehicle_measurement_available = True
        current_obstacle_measurement_available = np.ones(len(obstacles), dtype=bool)
        current_vehicle_measurement_state = x0[:, 0].copy()
        current_obstacle_measurement_positions = truth_obstacle_positions(0.0)
    active_controller.reset(current_belief)
    control_goal = ControlGoal(
        position=np.array([goal_pos["x"], goal_pos["y"], goal_pos["z"]], dtype=float),
        quaternion_wxyz=quat_from_euler(
            goal_euler["roll"],
            goal_euler["pitch"],
            goal_euler["yaw"],
        ),
    )

    n_substeps = max(1, round(mpc_dt / mj_dt))
    n_steps = int(sim_seconds / mpc_dt)

    ts, poss, eulers, us, clearances, frames = [], [], [], [], [], []
    estimated_states, estimation_covariances = [], []
    obstacle_estimated_states, obstacle_estimation_covariances = [], []
    horizon_vehicle_covariances, horizon_obstacle_covariances = [], []
    chance_margin_history, risk_allocation_history, slack_history = [], [], []
    projected_uncertainty_history, tightened_radius_history = [], []
    solver_status_history: list[str] = []
    risk_semantics_history: list[str] = []
    risk_allocation_method_history: list[str] = []
    risk_budget_total_history: list[float] = []
    risk_budget_allocated_history: list[float] = []
    risk_budget_remaining_history: list[float] = []
    risk_constraint_count_history: list[int] = []
    risk_budget_status_history: list[str] = []
    primary_solver_status_history: list[str] = []
    primary_solver_success_history: list[bool] = []
    primary_solver_iteration_history: list[int] = []
    primary_solver_primal_residual_history: list[float] = []
    primary_solver_dual_residual_history: list[float] = []
    command_source_history: list[str] = []
    solution_accepted_history: list[bool] = []
    fallback_active_history: list[bool] = []
    fallback_level_history: list[int] = []
    fallback_reason_history: list[str] = []
    consecutive_rejection_history: list[int] = []
    deadline_missed_history: list[bool] = []
    safety_assurance_status_history: list[str] = []
    horizon_assurance_status_history: list[str] = []
    horizon_assurance_eligible_history: list[bool] = []
    horizon_assurance_reason_history: list[str] = []
    horizon_assurance_failed_checks_history: list[tuple[str, ...]] = []
    residual_status_history: list[str] = []
    solver_time_history: list[float] = []
    vehicle_measurement_history, obstacle_measurement_history = [], []
    vehicle_measurements, obstacle_measurements = [], []
    collided = False
    termination_reason = "completed"
    x_curr = x0
    t = 0.0
    goal_array = np.array([goal_pos["x"], goal_pos["y"], goal_pos["z"]], dtype=float)
    context = CoupledRunContext(
        start_position=np.asarray(x0[:3, 0], dtype=float).copy(),
        goal_position=goal_array,
        obstacles=tuple(dict(item) for item in obstacles),
        safety_radii=np.asarray(
            [item["radius"] + margin + DRONE_RADIUS for item in obstacles], dtype=float
        ),
        controller_timestep_s=float(mpc_dt),
        horizon_steps=int(n_horizon),
        total_steps=n_steps,
    )

    try:
        if runtime is not None:
            runtime.open(plant, context)
        k = 0
        paused = False
        step_once = False
        stop_requested = False
        reset_requested = False
        run_again_requested = False
        holding_reason: str | None = None
        while True:
            if runtime is not None and not runtime.is_running():
                termination_reason = "viewer_closed"
                break

            if runtime is not None:
                poll_commands = getattr(runtime, "poll_commands", None)
                commands = [] if poll_commands is None else poll_commands()
                for command in commands:
                    name = getattr(command, "name", command)
                    if name == CommandName.STOP or name == CommandName.STOP.value:
                        stop_requested = True
                    elif name == CommandName.TOGGLE_PAUSE or name == CommandName.TOGGLE_PAUSE.value:
                        paused = not paused
                    elif name == CommandName.STEP or name == CommandName.STEP.value:
                        paused = True
                        step_once = True
                    elif name == CommandName.RESET or name == CommandName.RESET.value:
                        reset_requested = True
                    elif name == CommandName.RUN_AGAIN or name == CommandName.RUN_AGAIN.value:
                        reset_requested = True
                        run_again_requested = True

            if stop_requested:
                termination_reason = "user_stopped"
                break

            if reset_requested:
                plant.reset(x0_vals)
                x_curr = x0.copy()
                reset_estimation = reset_beliefs()
                if estimator is not None:
                    current_belief = reset_estimation.vehicle_belief
                    current_obstacle_beliefs = reset_estimation.obstacle_beliefs
                    current_vehicle_measurement_available = (
                        reset_estimation.vehicle_measurement_available
                    )
                    current_obstacle_measurement_available = (
                        reset_estimation.obstacle_measurement_available
                    )
                    current_vehicle_measurement_state = (
                        reset_estimation.vehicle_measurement_state_13
                    )
                    current_obstacle_measurement_positions = (
                        reset_estimation.obstacle_measurement_positions
                    )
                else:
                    current_belief, current_obstacle_beliefs = reset_estimation
                    current_vehicle_measurement_available = True
                    current_obstacle_measurement_available = np.ones(len(obstacles), dtype=bool)
                    current_vehicle_measurement_state = x0[:, 0].copy()
                    current_obstacle_measurement_positions = truth_obstacle_positions(0.0)
                active_controller.reset(current_belief)
                ts.clear()
                poss.clear()
                eulers.clear()
                us.clear()
                clearances.clear()
                frames.clear()
                estimated_states.clear()
                estimation_covariances.clear()
                obstacle_estimated_states.clear()
                obstacle_estimation_covariances.clear()
                horizon_vehicle_covariances.clear()
                horizon_obstacle_covariances.clear()
                chance_margin_history.clear()
                risk_allocation_history.clear()
                slack_history.clear()
                projected_uncertainty_history.clear()
                tightened_radius_history.clear()
                solver_status_history.clear()
                risk_semantics_history.clear()
                risk_allocation_method_history.clear()
                risk_budget_total_history.clear()
                risk_budget_allocated_history.clear()
                risk_budget_remaining_history.clear()
                risk_constraint_count_history.clear()
                risk_budget_status_history.clear()
                primary_solver_status_history.clear()
                primary_solver_success_history.clear()
                primary_solver_iteration_history.clear()
                primary_solver_primal_residual_history.clear()
                primary_solver_dual_residual_history.clear()
                command_source_history.clear()
                solution_accepted_history.clear()
                fallback_active_history.clear()
                fallback_level_history.clear()
                fallback_reason_history.clear()
                consecutive_rejection_history.clear()
                deadline_missed_history.clear()
                safety_assurance_status_history.clear()
                horizon_assurance_status_history.clear()
                horizon_assurance_eligible_history.clear()
                horizon_assurance_reason_history.clear()
                horizon_assurance_failed_checks_history.clear()
                residual_status_history.clear()
                solver_time_history.clear()
                vehicle_measurement_history.clear()
                obstacle_measurement_history.clear()
                vehicle_measurements.clear()
                obstacle_measurements.clear()
                collided = False
                t = 0.0
                k = 0
                holding_reason = None
                termination_reason = "completed"
                reset_requested = False
                step_once = False
                if run_again_requested:
                    paused = False
                run_again_requested = False
                on_reset = getattr(runtime, "on_reset", None)
                if on_reset is not None:
                    on_reset()
                continue

            if holding_reason is not None:
                on_idle = getattr(runtime, "on_idle", None)
                if on_idle is not None:
                    on_idle(True)
                time.sleep(0.01)
                continue

            if k >= n_steps:
                if runtime is None:
                    termination_reason = "completed"
                    break
                termination_reason = "duration_reached"
                holding_reason = termination_reason
                paused = True
                on_completed = getattr(runtime, "on_completed", None)
                if on_completed is not None:
                    on_completed(holding_reason)
                continue

            if paused and not step_once:
                on_idle = getattr(runtime, "on_idle", None)
                if on_idle is not None:
                    on_idle(True)
                time.sleep(0.01)
                continue

            solver_start = time.perf_counter()
            solution = active_controller.solve(
                current_belief,
                current_obstacle_beliefs,
                control_goal,
                t,
            )
            u0 = solution.command
            horizon_vehicle_covariances.append(solution.predicted_covariances.copy())
            horizon_obstacle_covariances.append(solution.predicted_obstacle_covariances.copy())
            chance_margin_history.append(solution.chance_margins.copy())
            risk_allocation_history.append(solution.risk_allocations.copy())
            slack_history.append(solution.slacks.copy())
            projected_uncertainty_history.append(solution.projected_uncertainties.copy())
            tightened_radius_history.append(solution.tightened_safety_radii.copy())
            solver_status_history.append(solution.solver_status)
            risk_semantics_history.append(solution.risk_semantics)
            risk_allocation_method_history.append(solution.risk_allocation_method)
            risk_budget_total_history.append(
                np.nan
                if solution.risk_budget_total is None
                else solution.risk_budget_total
            )
            risk_budget_allocated_history.append(solution.risk_budget_allocated)
            risk_budget_remaining_history.append(
                np.nan
                if solution.risk_budget_remaining is None
                else solution.risk_budget_remaining
            )
            risk_constraint_count_history.append(solution.risk_constraint_count)
            risk_budget_status_history.append(solution.risk_budget_status)
            primary_solver_status_history.append(solution.primary_solver_status)
            primary_solver_success_history.append(solution.primary_solver_success)
            primary_solver_iteration_history.append(
                solution.primary_solver_iterations
            )
            primary_solver_primal_residual_history.append(
                solution.primary_solver_primal_residual
            )
            primary_solver_dual_residual_history.append(
                solution.primary_solver_dual_residual
            )
            command_source_history.append(solution.command_source)
            solution_accepted_history.append(solution.solution_accepted)
            fallback_active_history.append(solution.fallback_active)
            fallback_level_history.append(solution.fallback_level)
            fallback_reason_history.append(solution.fallback_reason)
            consecutive_rejection_history.append(
                solution.consecutive_rejections
            )
            deadline_missed_history.append(solution.deadline_missed)
            safety_assurance_status_history.append(
                solution.safety_assurance_status
            )
            horizon_assurance_status_history.append(
                solution.horizon_assurance_status
            )
            horizon_assurance_eligible_history.append(
                solution.horizon_assurance_eligible
            )
            horizon_assurance_reason_history.append(
                solution.horizon_assurance_reason
            )
            horizon_assurance_failed_checks_history.append(
                solution.horizon_assurance_failed_checks
            )
            residual_status_history.append(solution.residual_status)
            measured_solver_time_ms = (
                time.perf_counter() - solver_start
            ) * 1000.0
            solver_time_ms = (
                solution.solve_time_ms
                if solution.solve_time_ms > 0.0
                else measured_solver_time_ms
            )
            solver_time_history.append(solver_time_ms)
            plant.apply_control_and_step(u0, n_substeps, t)  # MuJoCo "true" plant
            x_curr = plant.get_state_13()

            ts.append(t)
            position = x_curr[0:3, 0].copy()
            poss.append(position)
            eulers.append(quat_to_euler(x_curr[6:10, 0]))
            control = u0.copy()
            us.append(control)

            step_time = t + mpc_dt
            obstacle_predictions = predict_obstacle_positions(
                obstacles,
                step_time,
                n_horizon + 1,
                mpc_dt,
            )
            obstacle_positions = obstacle_predictions[:, 0, :]
            if estimator is not None:
                estimation = estimator.advance(
                    x_curr[:, 0],
                    obstacle_positions,
                    u0,
                    dt=mpc_dt,
                )
                current_belief = estimation.vehicle_belief
                current_obstacle_beliefs = estimation.obstacle_beliefs
                current_vehicle_measurement_available = estimation.vehicle_measurement_available
                current_obstacle_measurement_available = estimation.obstacle_measurement_available
                current_vehicle_measurement_state = estimation.vehicle_measurement_state_13
                current_obstacle_measurement_positions = estimation.obstacle_measurement_positions
            else:
                current_belief = VehicleBelief.exact(x_curr[:, 0])
                current_obstacle_beliefs = exact_obstacle_beliefs(
                    obstacles,
                    step_time,
                    active_controller.horizon_steps,
                    mpc_dt,
                )
                current_vehicle_measurement_available = True
                current_obstacle_measurement_available = np.ones(len(obstacles), dtype=bool)
                current_vehicle_measurement_state = x_curr[:, 0].copy()
                current_obstacle_measurement_positions = obstacle_positions.copy()
            estimated_obstacle_states_array = np.asarray(
                [belief.mean_state_6 for belief in current_obstacle_beliefs],
                dtype=float,
            ).reshape(-1, 6)
            obstacle_covariances_array = np.asarray(
                [belief.covariance_6 for belief in current_obstacle_beliefs],
                dtype=float,
            ).reshape(-1, 6, 6)
            estimated_obstacle_predictions_array = np.asarray(
                [
                    belief.mean_positions(
                        active_controller.horizon_steps + 1,
                        mpc_dt,
                    )
                    for belief in current_obstacle_beliefs
                ],
                dtype=float,
            ).reshape(-1, active_controller.horizon_steps + 1, 3)
            estimated_states.append(current_belief.mean_state_13.copy())
            estimation_covariances.append(current_belief.error_covariance_12.copy())
            obstacle_estimated_states.append(estimated_obstacle_states_array.copy())
            obstacle_estimation_covariances.append(obstacle_covariances_array.copy())
            vehicle_measurement_history.append(current_vehicle_measurement_available)
            obstacle_measurement_history.append(current_obstacle_measurement_available.copy())
            vehicle_measurements.append(
                np.full(13, np.nan)
                if current_vehicle_measurement_state is None
                else current_vehicle_measurement_state.copy()
            )
            obstacle_measurements.append(current_obstacle_measurement_positions.copy())
            min_clear = np.inf
            for obs, obstacle_position in zip(obstacles, obstacle_positions):
                d = np.linalg.norm(position - obstacle_position) - obs["radius"] - DRONE_RADIUS
                min_clear = min(min_clear, float(d))
            clearances.append(min_clear)
            collided = collided or plant.check_collision()
            goal_distance = float(np.linalg.norm(position - goal_array))

            if renderer is not None and k % render_every == 0:
                frames.append(plant.render_frame(renderer, camera="fixed_view"))

            if runtime is not None:
                keep_running = runtime.on_step(
                    CoupledStep(
                        step_index=k + 1,
                        time_s=step_time,
                        state_13=x_curr[:, 0].copy(),
                        control=control,
                        obstacle_positions=obstacle_positions,
                        obstacle_predictions=obstacle_predictions,
                        estimated_state_13=current_belief.mean_state_13,
                        error_covariance_12=current_belief.error_covariance_12,
                        estimated_obstacle_states=estimated_obstacle_states_array,
                        obstacle_covariances=obstacle_covariances_array,
                        estimated_obstacle_predictions=estimated_obstacle_predictions_array,
                        vehicle_measurement_available=current_vehicle_measurement_available,
                        obstacle_measurement_available=(current_obstacle_measurement_available),
                        vehicle_measurement_state_13=current_vehicle_measurement_state,
                        obstacle_measurement_positions=(current_obstacle_measurement_positions),
                        predicted_positions=solution.predicted_positions,
                        min_clearance_m=float(min_clear),
                        goal_distance_m=goal_distance,
                        solver_time_ms=solver_time_ms,
                        collided=collided,
                        paused=paused,
                        predicted_covariances=solution.predicted_covariances,
                        chance_margins=solution.chance_margins,
                        risk_allocations=solution.risk_allocations,
                        slacks=solution.slacks,
                        solver_status=solution.solver_status,
                        predicted_obstacle_covariances=(solution.predicted_obstacle_covariances),
                        projected_uncertainties=solution.projected_uncertainties,
                        tightened_safety_radii=solution.tightened_safety_radii,
                        risk_semantics=solution.risk_semantics,
                        risk_allocation_method=solution.risk_allocation_method,
                        risk_budget_total=solution.risk_budget_total,
                        risk_budget_allocated=solution.risk_budget_allocated,
                        risk_budget_remaining=solution.risk_budget_remaining,
                        risk_constraint_count=solution.risk_constraint_count,
                        risk_budget_status=solution.risk_budget_status,
                        primary_solver_status=solution.primary_solver_status,
                        primary_solver_success=solution.primary_solver_success,
                        primary_solver_iterations=(
                            solution.primary_solver_iterations
                        ),
                        primary_solver_primal_residual=(
                            solution.primary_solver_primal_residual
                        ),
                        primary_solver_dual_residual=(
                            solution.primary_solver_dual_residual
                        ),
                        command_source=solution.command_source,
                        solution_accepted=solution.solution_accepted,
                        fallback_active=solution.fallback_active,
                        fallback_level=solution.fallback_level,
                        fallback_reason=solution.fallback_reason,
                        consecutive_rejections=(
                            solution.consecutive_rejections
                        ),
                        deadline_missed=solution.deadline_missed,
                        safety_assurance_status=(
                            solution.safety_assurance_status
                        ),
                        residual_status=solution.residual_status,
                        horizon_assurance_status=(
                            solution.horizon_assurance_status
                        ),
                        horizon_assurance_eligible=(
                            solution.horizon_assurance_eligible
                        ),
                        horizon_assurance_reason=(
                            solution.horizon_assurance_reason
                        ),
                        horizon_assurance_failed_checks=(
                            solution.horizon_assurance_failed_checks
                        ),
                        assurance_schema_version=(
                            solution.assurance_schema_version
                        ),
                    )
                )
                if not keep_running:
                    termination_reason = "viewer_closed"
                    break

            t = step_time
            k += 1
            step_once = False
            if progress_cb is not None:
                progress_cb(k / n_steps)
            if stop_on_collision and collided:
                termination_reason = "collision"
                if runtime is None:
                    break
                holding_reason = termination_reason
                paused = True
                on_completed = getattr(runtime, "on_completed", None)
                if on_completed is not None:
                    on_completed(holding_reason)
                continue
            if stop_on_goal and goal_distance <= float(goal_tolerance):
                termination_reason = "goal_reached"
                if runtime is None:
                    break
                holding_reason = termination_reason
                paused = True
                on_completed = getattr(runtime, "on_completed", None)
                if on_completed is not None:
                    on_completed(holding_reason)
    finally:
        if runtime is not None:
            runtime.close()
        if renderer is not None:
            renderer.close()

    return {
        "t": np.array(ts),
        "pos": np.array(poss),
        "euler": np.array(eulers),
        "u": np.array(us),
        "clearance": np.array(clearances),
        "dt": mpc_dt,
        "collided": collided,
        "frames": frames,
        "render_error": render_err,
        "termination_reason": termination_reason,
        "estimated_state": np.asarray(estimated_states, dtype=float),
        "error_covariance": np.asarray(estimation_covariances, dtype=float),
        "estimated_obstacle_state": np.asarray(obstacle_estimated_states, dtype=float),
        "obstacle_covariance": np.asarray(obstacle_estimation_covariances, dtype=float),
        "predicted_error_covariance_horizon": np.asarray(
            horizon_vehicle_covariances,
            dtype=float,
        ),
        "predicted_obstacle_covariance_horizon": np.asarray(
            horizon_obstacle_covariances,
            dtype=float,
        ),
        "chance_residual_horizon": np.asarray(chance_margin_history, dtype=float),
        "risk_allocation_horizon": np.asarray(risk_allocation_history, dtype=float),
        "slack_horizon": np.asarray(slack_history, dtype=float),
        "projected_uncertainty_horizon": np.asarray(
            projected_uncertainty_history,
            dtype=float,
        ),
        "tightened_safety_radius_horizon": np.asarray(
            tightened_radius_history,
            dtype=float,
        ),
        "solver_status": np.asarray(solver_status_history, dtype=str),
        "risk_semantics": np.asarray(risk_semantics_history, dtype=str),
        "risk_allocation_method": np.asarray(
            risk_allocation_method_history,
            dtype=str,
        ),
        "risk_budget_total": np.asarray(risk_budget_total_history, dtype=float),
        "risk_budget_allocated": np.asarray(
            risk_budget_allocated_history,
            dtype=float,
        ),
        "risk_budget_remaining": np.asarray(
            risk_budget_remaining_history,
            dtype=float,
        ),
        "risk_constraint_count": np.asarray(
            risk_constraint_count_history,
            dtype=int,
        ),
        "risk_budget_status": np.asarray(risk_budget_status_history, dtype=str),
        "primary_solver_status": np.asarray(
            primary_solver_status_history,
            dtype=str,
        ),
        "primary_solver_success": np.asarray(
            primary_solver_success_history,
            dtype=bool,
        ),
        "primary_solver_iterations": np.asarray(
            primary_solver_iteration_history,
            dtype=int,
        ),
        "primary_solver_primal_residual": np.asarray(
            primary_solver_primal_residual_history,
            dtype=float,
        ),
        "primary_solver_dual_residual": np.asarray(
            primary_solver_dual_residual_history,
            dtype=float,
        ),
        "command_source": np.asarray(command_source_history, dtype=str),
        "solution_accepted": np.asarray(
            solution_accepted_history,
            dtype=bool,
        ),
        "fallback_active": np.asarray(fallback_active_history, dtype=bool),
        "fallback_level": np.asarray(fallback_level_history, dtype=int),
        "fallback_reason": np.asarray(fallback_reason_history, dtype=str),
        "consecutive_rejections": np.asarray(
            consecutive_rejection_history,
            dtype=int,
        ),
        "deadline_missed": np.asarray(deadline_missed_history, dtype=bool),
        "safety_assurance_status": np.asarray(
            safety_assurance_status_history,
            dtype=str,
        ),
        "horizon_assurance_status": np.asarray(
            horizon_assurance_status_history,
            dtype=str,
        ),
        "horizon_assurance_eligible": np.asarray(
            horizon_assurance_eligible_history,
            dtype=bool,
        ),
        "horizon_assurance_reason": np.asarray(
            horizon_assurance_reason_history,
            dtype=str,
        ),
        "horizon_assurance_failed_checks": np.asarray(
            horizon_assurance_failed_checks_history,
            dtype=object,
        ),
        "residual_status": np.asarray(residual_status_history, dtype=str),
        "solver_time_ms": np.asarray(solver_time_history, dtype=float),
        "vehicle_measurement_available": np.asarray(vehicle_measurement_history, dtype=bool),
        "obstacle_measurement_available": np.asarray(obstacle_measurement_history, dtype=bool),
        "vehicle_measurement_state": np.asarray(vehicle_measurements, dtype=float),
        "obstacle_measurement_position": np.asarray(obstacle_measurements, dtype=float),
        "estimation_enabled": estimator is not None,
    }


if __name__ == "__main__":
    import time

    t0 = time.time()
    result = run_coupled_simulation(
        x0_vals={"x": 0, "y": 0, "z": 1, "roll": 0, "pitch": 0, "yaw": 0},
        goal_pos={"x": 3, "y": 2, "z": 2.5},
        goal_euler={"roll": 0, "pitch": 0, "yaw": np.deg2rad(45)},
        bounds={
            "thrust": DEFAULT_QUADROTOR.max_upward_thrust_deviation_n,
            "torque_rp": DEFAULT_QUADROTOR.max_roll_pitch_torque_nm,
            "torque_yaw": DEFAULT_QUADROTOR.max_yaw_torque_nm,
        },
        obstacles=[{"type": "static", "x": 1.5, "y": 1.0, "z": 2.0, "radius": 0.5}],
        margin=0.3,
        sim_seconds=8.0,
    )
    print(f"perf: {(time.time() - t0) / len(result['t']) * 1000:.2f} ms/tick (MPC+MuJoCo combined)")
    print("final pos:", result["pos"][-1], " goal: [3,2,2.5]")
    print("min clearance:", result["clearance"].min())
    print("collided (real contact):", result["collided"])
    print("any NaN:", np.isnan(result["pos"]).any())
