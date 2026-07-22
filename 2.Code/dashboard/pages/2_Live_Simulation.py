"""Run and inspect one closed-loop simulation."""

from __future__ import annotations

import streamlit as st

from dashboard.common import load_named_scenario, render_kpis, run_many, scenario_files
from reporting.plotly_views import safety_figure, solver_figure, telemetry_figure, trajectory_3d

st.set_page_config(page_title="Live Simulation", page_icon="▶️", layout="wide")
st.title("Live Simulation")

with st.sidebar:
    scenario_name = st.selectbox("Scenario", list(scenario_files()))
    mode = st.selectbox("Controller", ("ccmpc", "deterministic"))
    backend = st.selectbox("Backend", ("scipy", "cvxpy"))
    seed = st.number_input("Seed", value=1, min_value=0, step=1)
    delta = st.slider("Risk delta", 0.01, 0.50, 0.03, 0.01, disabled=mode != "ccmpc")
    fov = st.checkbox("Enable FOV constraints", value=False)
    run_clicked = st.button("Run simulation", type="primary", use_container_width=True)

if run_clicked:
    scenario = load_named_scenario(scenario_name)
    bar = st.progress(0.0)
    with st.spinner("Solving receding-horizon control..."):
        result = run_many(
            scenario, [mode], backend, 1, int(seed),
            delta=delta if mode == "ccmpc" else None,
            fov=fov,
            progress=lambda done, total: bar.progress(done / total),
        )[0]
    bar.empty()
    st.session_state["live_result"] = result
    st.session_state["live_scenario"] = scenario

if "live_result" in st.session_state:
    result = st.session_state["live_result"]
    scenario = st.session_state["live_scenario"]
    render_kpis(st, result)
    horizon_index = st.slider(
        "Control update / predicted horizon",
        0,
        max(0, len(result.predicted_trajectories) - 1),
        0,
        disabled=len(result.predicted_trajectories) == 0,
    )
    st.plotly_chart(trajectory_3d([result], scenario, horizon_index), use_container_width=True)
    telemetry, safety, solver, events = st.tabs(("Telemetry", "Safety & uncertainty", "Solver", "Events"))
    with telemetry:
        st.plotly_chart(telemetry_figure(result, scenario), use_container_width=True)
    with safety:
        if scenario.obstacles:
            st.plotly_chart(safety_figure(result), use_container_width=True)
        else:
            st.info("Safety clearance is N/A because this scenario has no obstacles.")
    with solver:
        st.plotly_chart(solver_figure(result), use_container_width=True)
    with events:
        st.dataframe(result.events, use_container_width=True)
else:
    st.info("Choose a scenario and run the simulation.")
