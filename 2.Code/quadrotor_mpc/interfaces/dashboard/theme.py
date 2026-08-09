"""Shared dark-theme module for the Streamlit dashboard.

Provides:
  - PALETTE: colour constants matching the Qt desktop panel.
  - apply_theme(st): injects a scoped <style> block for CSS-penetrable
    components (probe-verified in docs/UI_CSS_PROBE.md).
  - register_dashboard_plotly_theme(): builds and registers a dark Plotly
    template ("taste_dark") for dashboard charts only.  CLI report modules
    (quadrotor_mpc.reporting.*) are never modified — the template is set at
    the dashboard-process level, not imported by reporting code.

Streamlit 1.60.x  ·  CSS injection via st.markdown(unsafe_allow_html=True)
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Palette — mirrors quadrotor_mpc/interfaces/desktop/panel.py
# ---------------------------------------------------------------------------
PALETTE: dict[str, str] = {
    "bg": "#10151f",
    "bg_secondary": "#172033",
    "text": "#d7e0ea",
    "primary": "#4ea1ff",
    "green": "#62d68b",
    "yellow": "#ffcf5a",
    "red": "#f05a67",
    "cyan": "#58d5e8",
    "purple": "#d783ff",
}

# Derived colours (not in PALETTE dict but used in CSS)
_BORDER = "#26354a"
_LABEL = "#9fb4cf"
_MUTED = "#8ea1ba"

_THEME_VERSION = "taste-dark v1.0  ·  streamlit 1.60.x"

# ---------------------------------------------------------------------------
# Scoped CSS  —  only probe-confirmed PASS components
# stSidebar is intentionally absent (probe FAIL — emotion CSS-in-JS).
# Sidebar colour is handled by .streamlit/config.toml secondaryBackgroundColor.
# ---------------------------------------------------------------------------
_CSS = f"""\
/* {_THEME_VERSION} */

/* === stMetric — KPI cards === streamlit 1.60.x — data-testid=stMetric === */
div[data-testid="stMetric"] {{
    background: {PALETTE["bg_secondary"]} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 8px !important;
    padding: 12px 14px !important;
}}
[data-testid="stMetricLabel"] {{
    color: {_LABEL} !important;
    font-size: 0.78rem !important;
}}
[data-testid="stMetricValue"] {{
    color: {PALETTE["text"]} !important;
    font-size: 1.45rem !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricDelta"] {{
    color: {PALETTE["green"]} !important;
    font-size: 0.82rem !important;
}}

/* === stAlert — info / warning / error / success === streamlit 1.60.x — data-testid=stAlert === */
div[data-testid="stAlert"] {{
    border-radius: 8px !important;
    border: 1px solid !important;
}}
div[data-testid="stAlert"] [data-testid="stAlertContentInfo"] {{
    background: rgba(78,161,255,0.12) !important;
    color: {PALETTE["primary"]} !important;
    border-color: {PALETTE["primary"]} !important;
}}
div[data-testid="stAlert"] [data-testid="stAlertContentWarning"] {{
    background: rgba(255,207,90,0.12) !important;
    color: {PALETTE["yellow"]} !important;
    border-color: {PALETTE["yellow"]} !important;
}}
div[data-testid="stAlert"] [data-testid="stAlertContentError"] {{
    background: rgba(240,90,103,0.12) !important;
    color: {PALETTE["red"]} !important;
    border-color: {PALETTE["red"]} !important;
}}
div[data-testid="stAlert"] [data-testid="stAlertContentSuccess"] {{
    background: rgba(98,214,139,0.12) !important;
    color: {PALETTE["green"]} !important;
    border-color: {PALETTE["green"]} !important;
}}

/* === stTabs — active / inactive states === streamlit 1.60.x — data-testid=stTabs === */
div[data-testid="stTabs"] [role="tablist"] button {{
    color: {_MUTED} !important;
    border-radius: 6px 6px 0 0 !important;
}}
div[data-testid="stTabs"] [role="tablist"] button[aria-selected="true"] {{
    color: {PALETTE["primary"]} !important;
    border-bottom: 2px solid {PALETTE["primary"]} !important;
    font-weight: 600 !important;
}}
div[data-testid="stTabs"] [role="tablist"] button[aria-selected="false"] {{
    color: {_MUTED} !important;
    border-bottom: 2px solid transparent !important;
}}
div[data-testid="stTabs"] [role="tablist"] button:hover {{
    color: {PALETTE["text"]} !important;
    background: rgba(78,161,255,0.08) !important;
}}

/* === stExpander === streamlit 1.60.x — data-testid=stExpander === */
div[data-testid="stExpander"] {{
    border: 1px solid {_BORDER} !important;
    border-radius: 8px !important;
    background: {PALETTE["bg_secondary"]} !important;
}}
div[data-testid="stExpander"] summary {{
    color: {PALETTE["text"]} !important;
    font-weight: 600 !important;
}}

/* === stCode — code blocks === streamlit 1.60.x — data-testid=stCode === */
div[data-testid="stCode"] {{
    background: #0c1119 !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 6px !important;
}}
div[data-testid="stCode"] pre {{
    color: {PALETTE["text"]} !important;
}}

/* === stDataFrame — container === streamlit 1.60.x — data-testid=stDataFrame === */
div[data-testid="stDataFrame"] {{
    border: 1px solid {_BORDER} !important;
    border-radius: 8px !important;
    background: {PALETTE["bg_secondary"]} !important;
}}
div[data-testid="stDataFrame"] .stDataFrame {{
    background: {PALETTE["bg_secondary"]} !important;
}}

/* === stSelectbox === streamlit 1.60.x — data-testid=stSelectbox === */
div[data-testid="stSelectbox"] > div > div {{
    background: {PALETTE["bg_secondary"]} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 6px !important;
    color: {PALETTE["text"]} !important;
}}

/* === stSlider === streamlit 1.60.x — data-testid=stSlider === */
div[data-testid="stSlider"] {{
    color: {PALETTE["text"]} !important;
}}

/* === stNumberInput === streamlit 1.60.x — data-testid=stNumberInput === */
div[data-testid="stNumberInput"] input {{
    background: {PALETTE["bg_secondary"]} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 6px !important;
    color: {PALETTE["text"]} !important;
}}

/* === stCheckbox === streamlit 1.60.x — data-testid=stCheckbox === */
div[data-testid="stCheckbox"] label {{
    color: {PALETTE["text"]} !important;
}}
div[data-testid="stCheckbox"] label:hover {{
    color: {PALETTE["primary"]} !important;
}}

/* === Buttons — primary / secondary === streamlit 1.60.x — button[kind=...] === */
button[kind="primary"] {{
    background: {PALETTE["primary"]} !important;
    border-color: {PALETTE["primary"]} !important;
    color: #0a0f18 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}}
button[kind="primary"]:hover {{
    background: #6bb4ff !important;
    border-color: #6bb4ff !important;
}}
button[kind="secondary"] {{
    background: {PALETTE["bg_secondary"]} !important;
    border: 1px solid {_BORDER} !important;
    color: {PALETTE["text"]} !important;
    border-radius: 6px !important;
}}
button[kind="secondary"]:hover {{
    background: #1e2a3f !important;
    border-color: {PALETTE["primary"]} !important;
    color: {PALETTE["primary"]} !important;
}}
"""


def apply_theme(st: Any) -> None:
    """Inject the scoped dark-theme CSS block into the Streamlit app.

    Call once near the top of each dashboard page, after ``st.set_page_config``::

        from quadrotor_mpc.interfaces.dashboard.theme import apply_theme
        apply_theme(st)

    Parameters
    ----------
    st:
        The ``streamlit`` module (passed explicitly to avoid import-time
        side-effects in non-dashboard contexts).
    """
    st.markdown(
        f"<style>\n{_CSS}</style>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Plotly dark template — dashboard-only
# ---------------------------------------------------------------------------
_TEMPLATE_NAME = "taste_dark"


def register_dashboard_plotly_theme() -> None:
    """Build, register, and set as default a dark Plotly template.

    The template ("taste_dark") is derived from ``plotly_dark`` and recoloured
    with :data:`PALETTE`.  After calling this function every Plotly figure
    created in the **current process** that does not explicitly set
    ``template=`` will use the dark theme.

    **Dashboard pages should call this once at import time** (or at the top
    of the page body).  CLI report modules
    (``quadrotor_mpc.reporting.*``) never import this function, so their
    figures keep the original light Plotly defaults — the template does NOT
    leak across process boundaries.
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    # Start from the built-in dark template
    base = pio.templates["plotly_dark"]

    tpl = go.layout.Template()
    tpl.layout = base.layout.to_plotly_json()  # type: ignore[assignment]

    # Override with palette colours
    tpl.layout.paper_bgcolor = PALETTE["bg"]
    tpl.layout.plot_bgcolor = PALETTE["bg"]
    tpl.layout.font = dict(color=PALETTE["text"], family="sans-serif")
    tpl.layout.title = dict(font=dict(color=PALETTE["text"]))
    tpl.layout.xaxis = dict(
        gridcolor=_BORDER,
        zerolinecolor=_BORDER,
        color=PALETTE["text"],
    )
    tpl.layout.yaxis = dict(
        gridcolor=_BORDER,
        zerolinecolor=_BORDER,
        color=PALETTE["text"],
    )
    tpl.layout.colorway = [
        PALETTE["primary"],
        PALETTE["green"],
        PALETTE["yellow"],
        PALETTE["red"],
        PALETTE["cyan"],
        PALETTE["purple"],
    ]
    tpl.layout.margin = dict(l=50, r=30, t=50, b=40)

    # Register and set as default
    pio.templates[_TEMPLATE_NAME] = tpl
    pio.templates.default = _TEMPLATE_NAME