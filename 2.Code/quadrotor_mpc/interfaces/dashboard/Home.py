"""Home page for the Quadrotor MPC Research Workbench."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Quadrotor MPC Workbench", page_icon="🚁", layout="wide")
st.title("Quadrotor MPC Research Workbench")
st.caption("Learn · Run · Compare · Validate — deterministic MPC and chance-constrained MPC")

left, right = st.columns([1.1, 1.0])
with left:
    st.subheader("Closed-loop architecture")
    st.graphviz_chart(
        """
    digraph G {
      rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#172033", fontcolor="white", color="#4c6fff"];
      scenario [label="Scenario + seed"];
      estimator [label="Estimator + covariance"];
      controller [label="MPC / CC-MPC"];
      plant [label="ODE 9D / MuJoCo 13D"];
      logger [label="Metrics + artifacts"];
      scenario -> estimator -> controller -> plant -> estimator;
      controller -> logger; plant -> logger; estimator -> logger;
    }
    """,
        use_container_width=True,
    )
with right:
    st.subheader("What this workbench records")
    st.markdown("""
    - Actual, estimated and reference states
    - Receding predicted horizon and predicted controls
    - Covariance, clearance, chance residual and slack
    - Cost decomposition, solver status, iterations and deadline misses
    - Reproducible run manifest, CSV/JSON/NPZ, PNG and interactive HTML report
    """)

st.subheader("Workflow")
columns = st.columns(5)
for column, title, body in zip(
    columns,
    ("1 · Build", "2 · Run", "3 · Inspect", "4 · Compare", "5 · Validate"),
    (
        "Create or edit a scenario",
        "Choose controller and backend",
        "Replay horizon and diagnostics",
        "Use paired seeds and statistics",
        "Run sweeps and Monte Carlo",
    ),
):
    column.markdown(f"**{title}**")
    column.caption(body)

st.info(
    "Start with **Live Simulation**, then use **Compare Controllers** on the static-obstacle scenario."
)
