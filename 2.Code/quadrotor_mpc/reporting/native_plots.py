"""Plotly visualizations shared by the legacy Streamlit NMPC applications."""

from __future__ import annotations

import math

import numpy as np


def _obstacle_position(obstacle, time_s: float):
    if obstacle["type"] == "static":
        return obstacle["x"], obstacle["y"], obstacle["z"]
    return (
        obstacle["x"],
        obstacle["amp"] * math.sin(2.0 * math.pi * time_s / obstacle["period"]),
        obstacle["z"],
    )


def build_figure(result, start, goal, obstacles):
    """Build a 3D trajectory animation for a legacy result dictionary."""
    import plotly.graph_objects as go

    positions = np.asarray(result["pos"])
    times = np.asarray(result["t"])
    traces = [
        go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            mode="lines", line=dict(color="#58a6ff", width=5), name="trajectory",
        ),
        go.Scatter3d(
            x=[start["x"]], y=[start["y"]], z=[start["z"]],
            mode="markers", marker=dict(size=5, color="#8b949e"), name="start",
        ),
        go.Scatter3d(
            x=[goal["x"]], y=[goal["y"]], z=[goal["z"]],
            mode="markers", marker=dict(size=8, color="#f2cc60", symbol="diamond"), name="goal",
        ),
        go.Scatter3d(
            x=[positions[0, 0]], y=[positions[0, 1]], z=[positions[0, 2]],
            mode="markers", marker=dict(size=8, color="#2f81f7"), name="quadrotor",
        ),
    ]
    for index, obstacle in enumerate(obstacles):
        x, y, z = _obstacle_position(obstacle, 0.0)
        traces.append(
            go.Scatter3d(
                x=[x], y=[y], z=[z], mode="markers",
                marker=dict(size=max(8, 32 * obstacle["radius"]), color="#f85149", opacity=0.55),
                name=f"obstacle {index + 1}",
            )
        )

    frame_stride = max(1, len(times) // 120)
    frames = []
    for frame_index in range(0, len(times), frame_stride):
        frame_data = [
            go.Scatter3d(
                x=[positions[frame_index, 0]], y=[positions[frame_index, 1]],
                z=[positions[frame_index, 2]], mode="markers",
                marker=dict(size=8, color="#2f81f7"),
            )
        ]
        trace_indices = [3]
        for obstacle_index, obstacle in enumerate(obstacles):
            x, y, z = _obstacle_position(obstacle, float(times[frame_index]))
            frame_data.append(go.Scatter3d(x=[x], y=[y], z=[z], mode="markers"))
            trace_indices.append(4 + obstacle_index)
        frames.append(go.Frame(data=frame_data, traces=trace_indices, name=str(frame_index)))

    figure = go.Figure(data=traces, frames=frames)
    figure.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        updatemenus=[dict(
            type="buttons", showactive=False,
            buttons=[dict(
                label="▶ Play", method="animate",
                args=[None, {"frame": {"duration": 45, "redraw": True}, "fromcurrent": True}],
            )],
        )],
    )
    return figure


def build_timeseries_figure(result, goal):
    """Build position, attitude, control and clearance time-series panels."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    times = np.asarray(result["t"])
    positions = np.asarray(result["pos"])
    euler = np.asarray(result["euler"])
    controls = np.asarray(result["u"])
    clearance = np.asarray(result["clearance"])
    figure = make_subplots(rows=2, cols=2, subplot_titles=("Position", "Euler angles", "Control", "Clearance"))
    for index, label in enumerate(("x", "y", "z")):
        figure.add_trace(go.Scatter(x=times, y=positions[:, index], name=label), row=1, col=1)
        figure.add_hline(y=goal[label], line_dash="dot", row=1, col=1)
    for index, label in enumerate(("roll", "pitch", "yaw")):
        figure.add_trace(go.Scatter(x=times, y=np.degrees(euler[:, index]), name=label), row=1, col=2)
    for index, label in enumerate(("Tdev", "tau_x", "tau_y", "tau_z")):
        figure.add_trace(go.Scatter(x=times, y=controls[:, index], name=label), row=2, col=1)
    figure.add_trace(go.Scatter(x=times, y=clearance, name="clearance"), row=2, col=2)
    figure.add_hline(y=0.0, line_dash="dash", row=2, col=2)
    figure.update_layout(template="plotly_dark", height=720)
    return figure
