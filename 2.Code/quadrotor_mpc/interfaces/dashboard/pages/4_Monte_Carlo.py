"""Monte Carlo robustness study."""

from __future__ import annotations

import streamlit as st

from quadrotor_mpc.interfaces.dashboard.common import (
    aggregate_frame,
    load_named_scenario,
    run_many,
    scenario_files,
)
from quadrotor_mpc.reporting.plotly_views import monte_carlo_distributions

st.set_page_config(page_title="Monte Carlo", page_icon="🎲", layout="wide")
st.title("Monte Carlo Robustness")

with st.sidebar:
    scenario_name = st.selectbox("Scenario", list(scenario_files()))
    controllers = st.multiselect(
        "Controllers", ("deterministic", "ccmpc"), default=("deterministic", "ccmpc")
    )
    trials = st.slider("Trials", 2, 100, 10)
    seed = st.number_input("First seed", value=1, min_value=0, step=1)
    backend = st.selectbox("Backend", ("scipy", "cvxpy"))
    run_clicked = st.button("Run Monte Carlo", type="primary", use_container_width=True)

if run_clicked and controllers:
    scenario = load_named_scenario(scenario_name)
    bar = st.progress(0.0)
    results = run_many(
        scenario,
        list(controllers),
        backend,
        trials,
        int(seed),
        progress=lambda done, total: bar.progress(done / total),
    )
    bar.empty()
    st.session_state["mc_results"] = results

if "mc_results" in st.session_state:
    results = st.session_state["mc_results"]
    st.dataframe(aggregate_frame(results), use_container_width=True)
    st.plotly_chart(monte_carlo_distributions(results), use_container_width=True)
else:
    st.info("Choose controllers and run at least two trials.")
