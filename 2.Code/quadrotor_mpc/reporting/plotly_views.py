"""Reusable Plotly views for the Streamlit research workbench."""

from __future__ import annotations

import numpy as np

from quadrotor_mpc.application.simulation.config import ScenarioConfig
from quadrotor_mpc.application.simulation.runner import SimulationResult

PALETTE = {"deterministic": "#ff9f43", "ccmpc": "#2e86de", "nmpc": "#20bf6b"}


def trajectory_3d(
    results: list[SimulationResult],
    scenario: ScenarioConfig,
    prediction_index: int | None = None,
):
    import plotly.graph_objects as go

    figure = go.Figure()
    for result in results:
        color = PALETTE.get(result.mode, "#8e44ad")
        figure.add_trace(
            go.Scatter3d(
                x=result.states[:, 0],
                y=result.states[:, 1],
                z=result.states[:, 2],
                mode="lines",
                line={"width": 6, "color": color},
                name=f"actual · {result.mode}",
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=result.estimated_states[:, 0],
                y=result.estimated_states[:, 1],
                z=result.estimated_states[:, 2],
                mode="lines",
                line={"width": 2, "dash": "dot", "color": color},
                opacity=0.55,
                name=f"estimate · {result.mode}",
            )
        )
        if prediction_index is not None and len(result.predicted_trajectories):
            index = int(np.clip(prediction_index, 0, len(result.predicted_trajectories) - 1))
            prediction = result.predicted_trajectories[index]
            figure.add_trace(
                go.Scatter3d(
                    x=prediction[:, 0],
                    y=prediction[:, 1],
                    z=prediction[:, 2],
                    mode="lines+markers",
                    marker={"size": 3},
                    line={"width": 4, "dash": "dash", "color": color},
                    name=f"horizon · {result.mode}",
                )
            )
    figure.add_trace(
        go.Scatter3d(
            x=[scenario.start[0]],
            y=[scenario.start[1]],
            z=[scenario.start[2]],
            mode="markers",
            marker={"size": 7, "color": "#8395a7"},
            name="start",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[scenario.goal[0]],
            y=[scenario.goal[1]],
            z=[scenario.goal[2]],
            mode="markers",
            marker={"size": 10, "color": "#feca57", "symbol": "diamond"},
            name="goal",
        )
    )
    for index, obstacle in enumerate(scenario.obstacles):
        figure.add_trace(
            go.Scatter3d(
                x=[obstacle.position[0]],
                y=[obstacle.position[1]],
                z=[obstacle.position[2]],
                mode="markers",
                marker={
                    "size": max(10, float(np.max(obstacle.size)) * 18),
                    "color": "#ee5253",
                    "opacity": 0.45,
                },
                name=f"obstacle {index + 1}",
            )
        )
    figure.update_layout(
        template="plotly_dark",
        height=650,
        scene={
            "xaxis_title": "x [m]",
            "yaxis_title": "y [m]",
            "zaxis_title": "z [m]",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        legend={"orientation": "h", "y": -0.08},
    )
    return figure


def telemetry_figure(result: SimulationResult, scenario: ScenarioConfig):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Position",
            "Velocity",
            "Attitude",
            "Control",
            "Tracking error",
            "Cost decomposition",
        ),
    )
    for index, label in enumerate(("x", "y", "z")):
        figure.add_trace(
            go.Scatter(x=result.times, y=result.states[:, index], name=label), row=1, col=1
        )
        figure.add_trace(
            go.Scatter(
                x=result.times,
                y=result.reference_positions[:, index],
                name=f"{label} ref",
                line={"dash": "dot"},
            ),
            row=1,
            col=1,
        )
    for index, label in enumerate(("vx", "vy", "vz")):
        figure.add_trace(
            go.Scatter(x=result.times, y=result.states[:, 3 + index], name=label), row=1, col=2
        )
    for index, label in enumerate(("roll", "pitch", "yaw")):
        figure.add_trace(
            go.Scatter(x=result.times, y=np.degrees(result.states[:, 6 + index]), name=label),
            row=2,
            col=1,
        )
    for index, label in enumerate(("roll cmd", "pitch cmd", "vz cmd", "yaw rate")):
        figure.add_trace(
            go.Scatter(x=result.times, y=result.controls[:, index], name=label), row=2, col=2
        )
    error = np.linalg.norm(result.states[:, :3] - scenario.goal, axis=1)
    figure.add_trace(go.Scatter(x=result.times, y=error, name="goal error"), row=3, col=1)
    figure.add_hline(y=scenario.goal_threshold, line_dash="dash", row=3, col=1)
    for name, values in result.cost_terms.items():
        if name != "total":
            figure.add_trace(
                go.Scatter(x=result.times, y=values, name=name, connectgaps=True), row=3, col=2
            )
    figure.update_layout(template="plotly_dark", height=900, legend={"orientation": "h"})
    return figure


def safety_figure(result: SimulationResult):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Clearance",
            "Chance residual",
            "Chance slack",
            "Position uncertainty",
        ),
    )
    figure.add_trace(
        go.Scatter(x=result.times, y=result.clearances, name="clearance"), row=1, col=1
    )
    figure.add_trace(
        go.Scatter(x=result.times, y=result.chance_residuals, name="residual"), row=1, col=2
    )
    figure.add_trace(go.Scatter(x=result.times, y=result.chance_slacks, name="slack"), row=2, col=1)
    sigma = np.sqrt(np.maximum(np.diagonal(result.covariances, axis1=1, axis2=2)[:, :3], 0.0))
    for index, label in enumerate(("sigma x", "sigma y", "sigma z")):
        figure.add_trace(go.Scatter(x=result.times, y=sigma[:, index], name=label), row=2, col=2)
    figure.add_hline(y=0.0, line_dash="dash", row=1, col=1)
    figure.add_hline(y=0.0, line_dash="dash", row=1, col=2)
    figure.update_layout(template="plotly_dark", height=680, legend={"orientation": "h"})
    return figure


def solver_figure(result: SimulationResult):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    mask = result.solver_times_ms > 0.0
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Solve time", "Iterations"))
    figure.add_trace(
        go.Scatter(
            x=result.times[mask],
            y=result.solver_times_ms[mask],
            mode="lines+markers",
            name="solve time",
        ),
        row=1,
        col=1,
    )
    figure.add_hline(y=result.controller_dt * 1000.0, line_dash="dash", row=1, col=1)
    iteration_mask = result.solver_iterations > 0
    figure.add_trace(
        go.Bar(
            x=result.times[iteration_mask],
            y=result.solver_iterations[iteration_mask],
            name="iterations",
        ),
        row=1,
        col=2,
    )
    figure.update_layout(template="plotly_dark", height=420, showlegend=False)
    return figure


def comparison_metrics(results: list[SimulationResult]):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(rows=1, cols=3, subplot_titles=("Tracking", "Safety", "Solver"))
    labels = [f"{item.mode} / {item.seed}" for item in results]
    colors = [PALETTE.get(item.mode, "#8e44ad") for item in results]
    figure.add_trace(
        go.Bar(
            x=labels,
            y=[r.metrics.tracking_rmse_m for r in results],
            marker_color=colors,
            name="RMSE",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=[r.metrics.min_clearance_m or 0.0 for r in results],
            marker_color=colors,
            name="clearance",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Bar(
            x=labels, y=[r.metrics.p95_solver_ms for r in results], marker_color=colors, name="p95"
        ),
        row=1,
        col=3,
    )
    figure.update_layout(template="plotly_dark", height=450, showlegend=False)
    return figure


def monte_carlo_distributions(results: list[SimulationResult]):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Tracking RMSE distribution",
            "ECDF of minimum clearance",
            "Safety–compute tradeoff",
            "Outcome rates",
        ),
    )
    for mode in sorted({item.mode for item in results}):
        items = [item for item in results if item.mode == mode]
        color = PALETTE.get(mode, "#8e44ad")
        figure.add_trace(
            go.Box(y=[r.metrics.tracking_rmse_m for r in items], name=mode, marker_color=color),
            row=1,
            col=1,
        )
        clearance = np.sort(
            np.asarray(
                [r.metrics.min_clearance_m for r in items if r.metrics.min_clearance_m is not None],
                dtype=float,
            )
        )
        if clearance.size:
            figure.add_trace(
                go.Scatter(
                    x=clearance,
                    y=np.arange(1, len(clearance) + 1) / len(clearance),
                    mode="lines+markers",
                    name=mode,
                    line={"color": color},
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
        figure.add_trace(
            go.Scatter(
                x=[r.metrics.p95_solver_ms for r in items],
                y=[r.metrics.min_clearance_m for r in items],
                mode="markers",
                marker={"color": color, "size": 9},
                name=mode,
                showlegend=False,
                text=[f"seed {r.seed}" for r in items],
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Bar(
                x=[f"{mode}<br>success", f"{mode}<br>collision"],
                y=[
                    np.mean([r.metrics.success for r in items]),
                    np.mean([r.metrics.collision for r in items]),
                ],
                marker_color=[color, "#ee5253"],
                name=mode,
                showlegend=False,
            ),
            row=2,
            col=2,
        )
    figure.update_xaxes(title_text="clearance [m]", row=1, col=2)
    figure.update_yaxes(title_text="cumulative probability", row=1, col=2)
    figure.update_xaxes(title_text="solver p95 [ms]", row=2, col=1)
    figure.update_yaxes(title_text="minimum clearance [m]", row=2, col=1)
    figure.update_yaxes(range=[0, 1.05], row=2, col=2)
    figure.update_layout(template="plotly_dark", height=780)
    return figure
