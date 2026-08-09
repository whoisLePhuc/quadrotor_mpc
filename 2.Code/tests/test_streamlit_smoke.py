"""Smoke tests: every Streamlit dashboard page must load without raising.

Baseline regression guard for the dark-theme redesign (Task 3). These tests are
STRUCTURAL ONLY - they assert the page renders (crash / no-crash) plus a few
light widget-shape checks. Visual / pixel / color verification is delegated to
Playwright in a later task.

NOTE: AppTest.from_file() resolves paths relative to the process CWD, and the
dashboard pages import ``quadrotor_mpc.interfaces.dashboard.common`` which
calls ``resource_root()``. Run pytest FROM the project root
(``quadrotor_mpc/2.Code``) so package imports and the resource root resolve.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_home_page() -> None:
    at = AppTest.from_file("quadrotor_mpc/interfaces/dashboard/Home.py", default_timeout=30).run()
    # at.exception is an ElementList; empty == no exceptions raised.
    assert not at.exception
    assert len(at.title) == 1
    assert len(at.subheader) >= 3
    assert len(at.info) >= 1
    # Structural: markdown columns (workflow) and markdown content present
    assert len(at.markdown) >= 4


def test_scenario_builder() -> None:
    at = AppTest.from_file("quadrotor_mpc/interfaces/dashboard/pages/1_Scenario_Builder.py", default_timeout=30).run()
    assert not at.exception
    # 3 start + 3 goal + max_time + timestep + seed + 2 noise inputs + text
    assert len(at.number_input) >= 6
    assert len(at.text_input) >= 1
    assert len(at.selectbox) >= 1
    # Structural: success message (validation) and code block (YAML preview) present
    assert len(at.success) >= 1
    assert len(at.code) >= 1


def test_live_simulation() -> None:
    at = AppTest.from_file("quadrotor_mpc/interfaces/dashboard/pages/2_Live_Simulation.py", default_timeout=30).run()
    assert not at.exception
    assert len(at.title) == 1
    assert len(at.sidebar.selectbox) >= 3
    assert len(at.sidebar.button) >= 1
    # Structural: FOV checkbox and Run button present in sidebar
    assert len(at.checkbox) >= 1
    assert len(at.button) >= 1


def test_compare_controllers() -> None:
    at = AppTest.from_file("quadrotor_mpc/interfaces/dashboard/pages/3_Compare_Controllers.py", default_timeout=30).run()
    assert not at.exception
    assert len(at.title) == 1
    assert len(at.sidebar.selectbox) >= 2
    assert len(at.info) >= 1  # idle state hint
    # Structural: Run paired comparison button present
    assert len(at.button) >= 1


def test_monte_carlo() -> None:
    at = AppTest.from_file("quadrotor_mpc/interfaces/dashboard/pages/4_Monte_Carlo.py", default_timeout=30).run()
    assert not at.exception
    assert len(at.title) == 1
    assert len(at.sidebar.multiselect) >= 1
    assert len(at.sidebar.slider) >= 1
    # Structural: Run button and sidebar number input (seed) present
    assert len(at.button) >= 1
    assert len(at.sidebar.number_input) >= 1


def test_experiment_explorer() -> None:
    # The page calls st.stop() when no tracked runs exist. st.stop() is not an
    # exception, so at.exception stays None and the smoke assertion holds.
    # NOTE: a prior wave confirmed a run exists in outputs/runs, so content renders.
    at = AppTest.from_file("quadrotor_mpc/interfaces/dashboard/pages/5_Experiment_Explorer.py", default_timeout=30).run()
    assert not at.exception
    assert len(at.title) == 1
    # Structural: run selector + manifest/metrics JSON panels present
    assert len(at.selectbox) >= 1
    assert len(at.json) >= 1


def test_theory_mode() -> None:
    at = AppTest.from_file("quadrotor_mpc/interfaces/dashboard/pages/6_Theory_Mode.py", default_timeout=30).run()
    assert not at.exception
    assert len(at.title) == 1
    assert len(at.table) >= 1
    assert len(at.warning) >= 1
    # Structural: markdown (LaTeX narrative) present
    assert len(at.markdown) >= 1


def test_nmpc_mujoco() -> None:
    # Guarded optional dependency: missing ModuleNotFoundError deps surface as
    # st.error + st.stop (no exception) rather than a crash.
    at = AppTest.from_file("quadrotor_mpc/interfaces/dashboard/pages/7_NMPC_MuJoCo.py", default_timeout=30).run()
    assert not at.exception
    assert len(at.title) == 1
    assert len(at.sidebar.number_input) >= 6
    assert len(at.sidebar.slider) >= 2
    # Structural: Run button present (MuJoCo installed, page fully rendered)
    assert len(at.button) >= 1
