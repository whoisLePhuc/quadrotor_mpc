"""Browse saved experiment manifests, metrics and reports."""

from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components
import yaml

from quadrotor_mpc.interfaces.dashboard.common import ROOT

st.set_page_config(page_title="Experiment Explorer", page_icon="🗂️", layout="wide")
st.title("Experiment Explorer")
runs_root = ROOT / "outputs/runs"
runs = (
    sorted((path for path in runs_root.glob("*") if path.is_dir()), reverse=True)
    if runs_root.exists()
    else []
)
if not runs:
    st.info("No tracked experiments yet. Run `quadrotor-mpc-run --compare` first.")
    st.stop()

selected = st.selectbox("Run", runs, format_func=lambda path: path.name)
manifest = yaml.safe_load((selected / "manifest.yaml").read_text(encoding="utf-8"))
metrics = json.loads((selected / "metrics.json").read_text(encoding="utf-8"))
left, right = st.columns(2)
with left:
    st.subheader("Manifest")
    st.json(manifest)
with right:
    st.subheader("Aggregate metrics")
    st.json(metrics)

report = selected / "report.html"
if report.exists():
    st.subheader("Interactive report")
    components.html(report.read_text(encoding="utf-8"), height=1150, scrolling=True)
