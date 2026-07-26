"""
run_coupled.py
==============
Closed-loop simulation: do-mpc/CasADi MPC (unchanged "brain", quad_mpc_core.py)
computes u0 every MPC tick; MuJoCo (mujoco_plant.py) is the "true" plant that
actually integrates that control -- a genuinely different rigid-body integrator
than the one the MPC uses internally to predict, so this is a real model-mismatch
robustness test, plus we get real contact-based collision detection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from belief_from_truth import exact_obstacle_beliefs
from controller_interface import ControlGoal, Controller, VehicleBelief
from mujoco_plant import MuJoCoPlant
from obstacle_motion import predict_obstacle_positions
from quad_mpc_core import (
    DRONE_RADIUS,
    quat_from_euler,
    quat_to_euler,
)
from runtime_control import CommandName
from vehicle import DEFAULT_QUADROTOR


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
        from deterministic_nmpc_controller import DeterministicNMPCController

        active_controller = DeterministicNMPCController(
            bounds=bounds,
            obstacle_specs=obstacles,
            margin=margin,
            horizon_steps=n_horizon,
            timestep_s=mpc_dt,
            max_iter=max_iter,
            cached=cached,
        )
    elif int(active_controller.horizon_steps) != int(n_horizon):
        raise ValueError(
            "injected controller horizon_steps must match run_coupled_simulation n_horizon"
        )

    plant = MuJoCoPlant(x0_vals, goal_pos, obstacles, mj_dt=mj_dt)

    renderer, render_err = (None, "not requested")
    if capture_frames:
        renderer, render_err = plant.try_create_renderer()

    q0 = quat_from_euler(x0_vals.get("roll", 0), x0_vals.get("pitch", 0), x0_vals.get("yaw", 0))
    x0 = np.array([x0_vals["x"], x0_vals["y"], x0_vals["z"], 0, 0, 0, *q0, 0, 0, 0]).reshape(-1, 1)
    initial_belief = VehicleBelief.exact(x0[:, 0])
    active_controller.reset(initial_belief)
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
                active_controller.reset(initial_belief)
                x_curr = x0.copy()
                ts.clear()
                poss.clear()
                eulers.clear()
                us.clear()
                clearances.clear()
                frames.clear()
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
            belief = VehicleBelief.exact(x_curr[:, 0])
            obstacle_beliefs = exact_obstacle_beliefs(
                obstacles,
                t,
                active_controller.horizon_steps,
                mpc_dt,
            )
            solution = active_controller.solve(
                belief,
                obstacle_beliefs,
                control_goal,
                t,
            )
            u0 = solution.command
            solver_time_ms = (time.perf_counter() - solver_start) * 1000.0
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
