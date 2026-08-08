"""Self-contained Plotly HTML reports with a dependency-free fallback."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from quadrotor_mpc.application.simulation.config import ScenarioConfig
from quadrotor_mpc.application.simulation.runner import SimulationResult


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _table(results: list[SimulationResult]) -> str:
    columns = (
        ("Controller", lambda r: r.mode),
        ("Seed", lambda r: r.seed),
        ("Success", lambda r: r.metrics.success),
        ("Collision", lambda r: r.metrics.collision),
        ("Final error [m]", lambda r: r.metrics.final_error_m),
        ("RMSE [m]", lambda r: r.metrics.tracking_rmse_m),
        ("Min clearance [m]", lambda r: r.metrics.min_clearance_m),
        ("Path [m]", lambda r: r.metrics.path_length_m),
        ("Solver p95 [ms]", lambda r: r.metrics.p95_solver_ms),
        ("Deadline miss", lambda r: r.metrics.deadline_miss_rate),
    )
    head = "".join(f"<th>{html.escape(label)}</th>" for label, _ in columns)
    rows = []
    for result in results:
        cells = "".join(f"<td>{html.escape(_fmt(getter(result)))}</td>" for _, getter in columns)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _plotly_figure(results: list[SimulationResult], scenario: ScenarioConfig):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
            [None, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "3-D trajectory",
            "Position error",
            "Safety residual / clearance",
            "Control inputs",
            "Solver timing",
        ),
        vertical_spacing=0.10,
    )
    palette = {"deterministic": "#ff9f43", "ccmpc": "#2e86de", "nmpc": "#20bf6b"}
    for result in results:
        color = palette.get(result.mode, "#8e44ad")
        label = f"{result.mode} / seed {result.seed}"
        position_error = np.linalg.norm(result.states[:, :3] - scenario.goal, axis=1)
        figure.add_trace(
            go.Scatter3d(
                x=result.states[:, 0],
                y=result.states[:, 1],
                z=result.states[:, 2],
                mode="lines",
                name=label,
                line={"color": color, "width": 6},
                legendgroup=label,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=result.times,
                y=position_error,
                name=label,
                line={"color": color},
                legendgroup=label,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        safety = result.clearances if scenario.obstacles else result.chance_residuals
        figure.add_trace(
            go.Scatter(
                x=result.times,
                y=safety,
                name=label,
                line={"color": color},
                legendgroup=label,
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        for index, command_name in enumerate(("roll", "pitch", "vz", "yaw rate")):
            figure.add_trace(
                go.Scatter(
                    x=result.times,
                    y=result.controls[:, index],
                    name=f"{result.mode}: {command_name}",
                    legendgroup=label,
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
        mask = result.solver_times_ms > 0.0
        figure.add_trace(
            go.Scatter(
                x=result.times[mask],
                y=result.solver_times_ms[mask],
                name=label,
                line={"color": color},
                legendgroup=label,
                showlegend=False,
            ),
            row=3,
            col=2,
        )

    figure.add_trace(
        go.Scatter3d(
            x=[scenario.start[0], scenario.goal[0]],
            y=[scenario.start[1], scenario.goal[1]],
            z=[scenario.start[2], scenario.goal[2]],
            mode="markers",
            marker={"size": [5, 8], "color": ["#8395a7", "#feca57"]},
            name="start / goal",
        ),
        row=1,
        col=1,
    )
    for index, obstacle in enumerate(scenario.obstacles):
        figure.add_trace(
            go.Scatter3d(
                x=[obstacle.position[0]],
                y=[obstacle.position[1]],
                z=[obstacle.position[2]],
                mode="markers",
                marker={"size": 12, "color": "#ee5253", "opacity": 0.55},
                name=f"obstacle {index + 1}",
            ),
            row=1,
            col=1,
        )
    maximum_time = max(float(result.times[-1]) for result in results)
    figure.add_trace(
        go.Scatter(
            x=[0.0, maximum_time],
            y=[0.0, 0.0],
            mode="lines",
            line={"dash": "dash", "color": "#ee5253"},
            name="safety boundary",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    for result in results:
        figure.add_trace(
            go.Scatter(
                x=[0.0, maximum_time],
                y=[result.controller_dt * 1000.0] * 2,
                mode="lines",
                line={"dash": "dot", "color": "#8395a7"},
                name=f"deadline · {result.mode}",
                showlegend=False,
            ),
            row=3,
            col=2,
        )
    figure.update_layout(
        template="plotly_white",
        height=1050,
        title=f"Quadrotor MPC experiment — {scenario.name}",
        margin={"l": 45, "r": 25, "t": 90, "b": 45},
        scene={"aspectmode": "data"},
    )
    return figure


def save_interactive_report(
    results: list[SimulationResult],
    scenario: ScenarioConfig,
    output_path: str | Path,
    *,
    aggregate: dict[str, Any] | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    style = """
    body{font-family:Inter,Arial,sans-serif;max-width:1500px;margin:auto;padding:28px;color:#17202a}
    h1{margin-bottom:4px} .note{color:#5d6d7e;margin-top:0}
    table{border-collapse:collapse;width:100%;margin:22px 0} th,td{padding:9px 11px;border:1px solid #d5d8dc;text-align:right}
    th{background:#f4f6f7} th:first-child,td:first-child{text-align:left}
    pre{background:#f4f6f7;padding:14px;border-radius:6px;overflow:auto}
    """
    try:
        from plotly.io import to_html

        plot = to_html(
            _plotly_figure(results, scenario),
            include_plotlyjs=True,
            full_html=False,
        )
        dependency_note = "Interactive plots are embedded in this file."
    except ModuleNotFoundError:
        plot = "<p><strong>Interactive charts unavailable:</strong> install Plotly.</p>"
        dependency_note = "This fallback report contains the complete metrics table."

    aggregate_block = html.escape(json.dumps(aggregate or {}, indent=2))
    document = f"""<!doctype html><html><head><meta charset='utf-8'>
    <title>Quadrotor MPC report — {html.escape(scenario.name)}</title><style>{style}</style></head>
    <body><h1>Quadrotor MPC experiment</h1><p class='note'>{html.escape(scenario.name)} · {dependency_note}</p>
    {_table(results)}{plot}<h2>Aggregate statistics</h2><pre>{aggregate_block}</pre></body></html>"""
    output.write_text(document, encoding="utf-8")
    return output
