"""Paired deterministic MPC versus CC-MPC comparison."""

from __future__ import annotations

import streamlit as st

from quadrotor_mpc.interfaces.dashboard.common import (
    aggregate_frame,
    load_named_scenario,
    metrics_frame,
    run_many,
    scenario_files,
)
from quadrotor_mpc.reporting.plotly_views import comparison_metrics, trajectory_3d

st.set_page_config(page_title="Compare Controllers", page_icon="⚖️", layout="wide")
st.title("Compare Controllers")
st.caption("Both controllers use the same plant, initial state, disturbance sequence and seed.")

with st.sidebar:
    scenario_name = st.selectbox("Scenario", list(scenario_files()), index=min(2, len(scenario_files()) - 1))
    backend = st.selectbox("Backend", ("scipy", "cvxpy"))
    seed = st.number_input("First seed", value=1, min_value=0, step=1)
    trials = st.slider("Paired trials", 1, 30, 1)
    run_clicked = st.button("Run paired comparison", type="primary", use_container_width=True)

if run_clicked:
    scenario = load_named_scenario(scenario_name)
    bar = st.progress(0.0)
    results = run_many(
        scenario, ["deterministic", "ccmpc"], backend, trials, int(seed),
        progress=lambda done, total: bar.progress(done / total),
    )
    bar.empty()
    st.session_state["compare_results"] = results
    st.session_state["compare_scenario"] = scenario

if "compare_results" in st.session_state:
    results = st.session_state["compare_results"]
    scenario = st.session_state["compare_scenario"]
    st.subheader("Aggregate metrics")
    st.dataframe(aggregate_frame(results), use_container_width=True)
    representatives = [next(item for item in results if item.mode == mode) for mode in ("deterministic", "ccmpc")]
    st.plotly_chart(trajectory_3d(representatives, scenario), use_container_width=True)
    st.plotly_chart(comparison_metrics(representatives), use_container_width=True)
    with st.expander("All paired runs"):
        st.dataframe(metrics_frame(results), use_container_width=True)
else:
    st.info("Run a paired comparison to populate this page.")
