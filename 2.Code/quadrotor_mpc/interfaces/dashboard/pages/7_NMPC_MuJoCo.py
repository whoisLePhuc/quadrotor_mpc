"""Optional 13-state quaternion NMPC on a MuJoCo rigid-body plant."""

from __future__ import annotations

import numpy as np
import streamlit as st

from quadrotor_mpc.interfaces.dashboard import theme

st.set_page_config(page_title="NMPC + MuJoCo", page_icon="🛰️", layout="wide")
theme.apply_theme(st)
theme.register_dashboard_plotly_theme()
st.title("13-State NMPC + MuJoCo Plant")
st.caption("Optional high-fidelity track for model-mismatch and contact-collision studies.")

with st.sidebar:
    st.subheader("Start")
    x0 = st.number_input("x0", value=0.0)
    y0 = st.number_input("y0", value=0.0)
    z0 = st.number_input("z0", value=1.0, min_value=0.2)
    st.subheader("Goal")
    xg = st.number_input("x goal", value=3.0)
    yg = st.number_input("y goal", value=2.0)
    zg = st.number_input("z goal", value=2.5, min_value=0.2)
    duration = st.slider("Duration [s]", 2.0, 15.0, 6.0)
    margin = st.slider("Safety margin [m]", 0.1, 1.0, 0.3)
    obstacle_on = st.checkbox("Static obstacle", value=True)
    run_clicked = st.button("Run NMPC + MuJoCo", type="primary", use_container_width=True)

if run_clicked:
    try:
        from quadrotor_mpc.application.native.runtime import run_coupled_simulation
        from quadrotor_mpc.reporting.native_plots import build_figure, build_timeseries_figure
    except ModuleNotFoundError as exc:
        st.error(f"Optional dependency missing: {exc}. Install `requirements-ui.txt`.")
        st.stop()
    obstacles = []
    if obstacle_on:
        obstacles.append({"type": "static", "x": 1.5, "y": 1.0, "z": 2.0, "radius": 0.5})
    progress = st.progress(0.0)
    with st.spinner("Building and solving NMPC..."):
        result = run_coupled_simulation(
            x0_vals={"x": x0, "y": y0, "z": z0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            goal_pos={"x": xg, "y": yg, "z": zg},
            goal_euler={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            bounds={"thrust": 6.0, "torque_rp": 0.03, "torque_yaw": 0.02},
            obstacles=obstacles,
            margin=margin,
            sim_seconds=duration,
            progress_cb=progress.progress,
        )
    progress.empty()
    st.session_state["mujoco_result"] = result
    st.session_state["mujoco_context"] = (
        {"x": x0, "y": y0, "z": z0},
        {"x": xg, "y": yg, "z": zg},
        obstacles,
        build_figure,
        build_timeseries_figure,
    )

if "mujoco_result" in st.session_state:
    result = st.session_state["mujoco_result"]
    start, goal, obstacles, build_figure, build_timeseries_figure = st.session_state[
        "mujoco_context"
    ]
    final_error = float(np.linalg.norm(result["pos"][-1] - np.array(list(goal.values()))))
    columns = st.columns(4)
    columns[0].metric("Final error", f"{final_error:.3f} m")
    columns[1].metric("Contact collision", "YES" if result["collided"] else "NO")
    columns[2].metric("Duration", f"{result['t'][-1]:.2f} s")
    columns[3].metric("Plant", "MuJoCo 13D")
    tabs = st.tabs(("3-D flight", "Telemetry"))
    with tabs[0]:
        st.plotly_chart(build_figure(result, start, goal, obstacles), use_container_width=True)
    with tabs[1]:
        st.plotly_chart(build_timeseries_figure(result, goal), use_container_width=True)
else:
    st.info("This page requires CasADi, do-mpc and MuJoCo from `requirements-ui.txt`.")
