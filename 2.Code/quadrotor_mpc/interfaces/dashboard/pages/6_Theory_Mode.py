"""Map the implemented equations to code, tests and plots."""

from __future__ import annotations

import streamlit as st

from quadrotor_mpc.interfaces.dashboard import theme

st.set_page_config(page_title="Theory Mode", page_icon="📐", layout="wide")
theme.apply_theme(st)
st.title("Theory ↔ Code Mapping")
st.markdown(r"""
### Chance-constrained obstacle residual

$$
g_k=\lVert y_k\rVert-1-\Phi^{-1}(1-\delta)\,
\sqrt{n_k^T S(\Sigma_k+\Sigma_{o,k})S^T n_k}
$$

The controller treats $g_k\geq0$ as safe under the local Gaussian approximation.
When a soft slack is active, the run is feasible numerically but no longer has an absolute probability guarantee.
""")

st.table(
    [
        {
            "Concept": "9-state dynamics + RK4",
            "Implementation": "ccmpc/dynamics.py",
            "Test": "tests/test_dynamics.py",
            "View": "Telemetry",
        },
        {
            "Concept": "Covariance propagation",
            "Implementation": "ccmpc/uncertainty.py",
            "Test": "tests/test_uncertainty.py",
            "View": "Safety & uncertainty",
        },
        {
            "Concept": "Ellipsoid + chance constraint",
            "Implementation": "ccmpc/risk.py",
            "Test": "tests/test_risk.py",
            "View": "Chance residual",
        },
        {
            "Concept": "Receding-horizon solve",
            "Implementation": "simulation/controllers.py",
            "Test": "tests/test_runner.py",
            "View": "Predicted horizon",
        },
        {
            "Concept": "Experiment protocol",
            "Implementation": "experiments/manager.py",
            "Test": "tests/test_experiments.py",
            "View": "Experiment Explorer",
        },
    ]
)

st.warning(
    "A nonzero chance slack must always be reported as a protocol deviation, not hidden as a successful hard safety guarantee."
)
