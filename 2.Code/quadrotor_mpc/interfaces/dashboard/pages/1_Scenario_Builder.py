"""Create and download a validated scenario YAML."""

from __future__ import annotations

import streamlit as st
import yaml

from quadrotor_mpc.application.simulation.config import ScenarioConfig
from quadrotor_mpc.interfaces.dashboard import theme
from quadrotor_mpc.interfaces.dashboard.common import load_named_scenario, scenario_files

st.set_page_config(page_title="Scenario Builder", page_icon="🧩", layout="wide")
theme.apply_theme(st)
st.title("Scenario Builder")
selected = st.selectbox("Template", list(scenario_files()))
base = load_named_scenario(selected).to_mapping()

left, right = st.columns(2)
with left:
    base["name"] = st.text_input("Scenario name", base["name"])
    base["start"][:3] = [
        st.number_input("Start x", value=float(base["start"][0])),
        st.number_input("Start y", value=float(base["start"][1])),
        st.number_input("Start z", value=float(base["start"][2]), min_value=0.0),
    ]
    base["goal"] = [
        st.number_input("Goal x", value=float(base["goal"][0])),
        st.number_input("Goal y", value=float(base["goal"][1])),
        st.number_input("Goal z", value=float(base["goal"][2]), min_value=0.0),
    ]
with right:
    base["max_time"] = st.number_input(
        "Maximum time [s]", value=float(base["max_time"]), min_value=0.1
    )
    base["sim_timestep"] = st.number_input(
        "Plant timestep [s]", value=float(base["sim_timestep"]), min_value=0.001, format="%.3f"
    )
    base["seed"] = st.number_input("Seed", value=int(base["seed"]), min_value=0, step=1)
    base["noise"]["measurement_pos"] = st.number_input(
        "Position measurement noise",
        value=float(base["noise"]["measurement_pos"]),
        min_value=0.0,
        format="%.4f",
    )
    base["noise"]["process_vel"] = st.number_input(
        "Velocity process noise",
        value=float(base["noise"]["process_vel"]),
        min_value=0.0,
        format="%.4f",
    )

st.subheader("Obstacle editor")
obstacle_yaml = st.text_area(
    "Obstacles (YAML list)",
    value=yaml.safe_dump(base["obstacles"], sort_keys=False),
    height=240,
)
try:
    base["obstacles"] = yaml.safe_load(obstacle_yaml) or []
    validated = ScenarioConfig.from_mapping(base, base["name"])
    rendered = yaml.safe_dump(validated.to_mapping(), sort_keys=False)
    st.success("Scenario is valid.")
    st.download_button(
        "Download scenario YAML", rendered, file_name=f"{validated.name}.yaml", mime="text/yaml"
    )
    st.code(rendered, language="yaml")
except (ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
    st.error(f"Invalid scenario: {exc}")
