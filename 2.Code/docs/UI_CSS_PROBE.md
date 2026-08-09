# UI CSS Probe Report — Streamlit 1.60.0

**Streamlit Version:** 1.60.0
**Playwright Version:** 1.62.0
**Probe URL:** http://localhost:8501
**Method:** `st.markdown(..., unsafe_allow_html=True)` injected CSS at top of app

---

## Component Verdict Matrix

| Component | data-testid / Selector | Expected Color | Computed Color | Verdict | Mechanism |
|-----------|------------------------|----------------|----------------|---------|-----------|
| `st.metric` (KPI) | `[data-testid="stMetric"]` | #ff00ff | #ff00ff | **PASS** | CSS penetrates — no shadow DOM |
| `st.dataframe` | `[data-testid="stDataFrame"]` | #ff0000 | #ff0000 | **PASS** | CSS penetrates — no shadow DOM |
| `st.sidebar` | `[data-testid="stSidebar"]` | #ff8800 | #172033 (default dark) | **FAIL** | Emotion CSS-in-JS high-specificity override; NOT shadow-DOM (no shadow roots found) |
| `st.tabs` | `[data-testid="stTabs"]` | #00ff00 | #00ff00 | **PASS** | CSS penetrates — no shadow DOM |
| `st.expander` | `[data-testid="stExpander"]` | #ff00cc | #ff00cc | **PASS** | CSS penetrates — no shadow DOM |
| `st.code` | `[data-testid="stCode"]` | #000088 | #000088 | **PASS** | CSS penetrates — no shadow DOM |
| `st.selectbox` | `[data-testid="stSelectbox"]` | #cccccc | #cccccc | **PASS** | CSS penetrates — no shadow DOM |
| `st.slider` | `[data-testid="stSlider"]` | #008888 | #008888 | **PASS** | CSS penetrates — no shadow DOM |
| `st.number_input` | `[data-testid="stNumberInput"]` | #808000 | #808000 | **PASS** | CSS penetrates — no shadow DOM |
| `st.checkbox` | `[data-testid="stCheckbox"]` | #00aaaa | #00aaaa | **PASS** | CSS penetrates — no shadow DOM |
| `st.info/warning/error/success` | `[data-testid="stAlert"]` | #aaccff | #aaccff | **PASS** | CSS penetrates — no shadow DOM |
| `st.button` (primary) | `button[kind="primary"]` | #ffff00 | #ffff00 | **PASS** | CSS penetrates — no shadow DOM |
| `st.button` (secondary) | `button[kind="secondary"]` | #00ffff | #00ffff | **PASS** | CSS penetrates — no shadow DOM |

---

## Per-Component Fallback Strategy

### PASS Components (use `st.markdown` CSS injection)
These accept injected CSS via `unsafe_allow_html=True` and can be styled directly:

| Component | Fallback Strategy |
|-----------|-------------------|
| `st.metric` | CSS injection works. Use distinctive background to differentiate dark theme KPIs from default. |
| `st.dataframe` | CSS injection works. Style table rows/headers via `div[data-testid="stDataFrame"] table { ... }`. |
| `st.tabs` | CSS injection works. Style tab bar and tab content via `div[data-testid="stTabs"]`. |
| `st.expander` | CSS injection works. Style collapsed/expanded states via `div[data-testid="stExpander"]`. |
| `st.code` | CSS injection works. Override code block background and syntax highlight colors. |
| `st.selectbox` | CSS injection works. Style dropdown via `div[data-testid="stSelectbox"] select { ... }`. |
| `st.slider` | CSS injection works. Style track and thumb via `div[data-testid="stSlider"] input[type=range]`. |
| `st.number_input` | CSS injection works. Style input field via `div[data-testid="stNumberInput"] input`. |
| `st.checkbox` | CSS injection works. Style check indicator via `div[data-testid="stCheckbox"] input`. |
| `st.alert` (info/warning/error/success) | CSS injection works. Override alert background and border colors. |
| `st.button` (primary/secondary) | CSS injection works. Override button background via `button[kind="primary"]`, `button[kind="secondary"]`. |

### FAIL Component — `st.sidebar`
- **Selector:** `div[data-testid="stSidebar"]`
- **Observed computed background:** `rgb(23, 32, 51)` (#172033, Streamlit dark default)
- **Root cause:** Emotion CSS-in-JS generates high-specificity inline styles that override the injected CSS rule despite `!important`. The sidebar element has **no shadow DOM** (confirmed by TreeWalker — zero shadow roots found among 23 child elements).
- **Fallback:** Use Streamlit's native `config.toml` `[theme]` section:
  ```toml
  [theme]
  backgroundColor = "#1a1a2e"
  secondaryBackgroundColor = "#16213e"
  ```
  The sidebar color is controlled by `secondaryBackgroundColor`. Custom HTML/CSS injection CANNOT override sidebar background in Streamlit 1.60.0.

---

## st.metric Deep Dive

`st.metric` is the most critical component for the dark-theme redesign — the quadrotor dashboard's `common.py` `render_kpis()` uses `st.metric` x6 (Velocity X, Velocity Y, Velocity Z, Altitude, Battery, GPS Status).

**Finding: PASS — CSS penetrates cleanly.**

- 3 metric elements found (one per column in the 3-column KPI layout)
- Computed `backgroundColor: rgb(255, 0, 255)` matches injected `#ff00ff` exactly
- No shadow DOM — the metric element renders in the light DOM as a plain `<div data-testid="stMetric">`
- Visibility: visible, Display: block, Opacity: 1 — all normal

**KPI styling approach for dark theme:**
```python
st.markdown("""
<style>
div[data-testid="stMetric"] {
    background: #1e1e3f !important;
    border: 1px solid #3d3d6b !important;
    border-radius: 8px !important;
    padding: 12px !important;
}
div[data-testid="stMetricLabel"] {
    color: #a0a0c0 !important;
}
div[data-testid="stMetricValue"] {
    color: #00e5ff !important;
}
div[data-testid="stMetricDelta"] {
    color: #00ff88 !important;
}
</style>
""", unsafe_allow_html=True)
```

---

## Shadow DOM Scan Results

- **`[data-testid="stSidebar"]`**: 0 shadow roots detected (23 children walked). Shadow DOM is NOT the blocking mechanism for sidebar.
- **`[data-testid="stMetric"]`**: Shadow DOM TreeWalker returned `TypeError` on `node.className.substring` (className is a DOMTokenList in this Streamlit version), but the key finding is that CSS injection worked → light DOM only, no shadow DOM.

---

## Evidence Files

| File | Description |
|------|-------------|
| `task-1-metric-probe.json` | Full computed-style dump for all probed components |
| `task-1-probe-screenshot.png` | Screenshot of the probe app with all colored components |
| `sidebar-structure.json` | DOM tree walk output for stSidebar (23 children, 0 shadow roots) |
| `playwright_probe.py` | Playwright probe script (temporary, deleted post-run) |

---

## Summary

- **Shadow-DOM protected components: NONE** — no web component shadow DOM was found on any probed Streamlit component in 1.60.0.
- **CSS injection blocked: 1** — `st.sidebar` (Emotion CSS-in-JS specificity override, NOT shadow DOM).
- **CSS injection passes: 13** — all other components including `st.metric`.
- **Theme work directive:** Style all PASS components via `st.markdown(..., unsafe_allow_html=True)`. Style sidebar exclusively via `config.toml` `[theme]` section.

---

## Rollback (restore default Streamlit styling)

The dark theme is applied through exactly two mechanisms. Removing both restores
Streamlit's default (light) look with no code changes required beyond the two steps below.

### Step 1 — remove the shared CSS injection

Each dashboard page calls `theme.apply_theme(st)` (and `theme.register_dashboard_plotly_theme()`
on chart pages) right after `st.set_page_config`. Delete those calls from:

```
quadrotor_mpc/interfaces/dashboard/Home.py
quadrotor_mpc/interfaces/dashboard/pages/1_Scenario_Builder.py
quadrotor_mpc/interfaces/dashboard/pages/2_Live_Simulation.py
quadrotor_mpc/interfaces/dashboard/pages/3_Compare_Controllers.py
quadrotor_mpc/interfaces/dashboard/pages/4_Monte_Carlo.py
quadrotor_mpc/interfaces/dashboard/pages/5_Experiment_Explorer.py
quadrotor_mpc/interfaces/dashboard/pages/6_Theory_Mode.py
quadrotor_mpc/interfaces/dashboard/pages/7_NMPC_MuJoCo.py
```

Optionally also remove the `from quadrotor_mpc.interfaces.dashboard import theme` import line
on each page and delete `quadrotor_mpc/interfaces/dashboard/theme.py` itself. Deleting the
`theme` module removes the Plotly `taste_dark` template registration too (CLI report generation
never imports it, so it is unaffected either way).

### Step 2 — remove the config.toml theme

Delete (or empty the `[theme]` section of) `.streamlit/config.toml`:

```
.streamlit/config.toml
```

### Verification after rollback

```bash
# from the project root (C:\Users\Admin\Research\quadrotor_mpc\2.Code)
$env:PYTHONPATH = "C:\Users\Admin\Research\quadrotor_mpc\2.Code"
& "C:\Users\Admin\Research\.venv\Scripts\python.exe" -m pytest tests\test_streamlit_smoke.py -q -o addopts=
# Expected: 8 passed (structural assertions are theme-agnostic)

& "C:\Users\Admin\Research\.venv\Scripts\python.exe" -m pytest tests\ -q --no-header --no-summary
# Expected: exit 0 (full suite unaffected by theme)

# Launch the dashboard and confirm the default light theme renders:
Start-Process -FilePath "C:\Users\Admin\Research\.venv\Scripts\python.exe" `
  -ArgumentList "-m","streamlit","run","quadrotor_mpc\interfaces\dashboard\Home.py",`
  "--server.port=8513","--server.headless=true" -WorkingDirectory "C:\Users\Admin\Research\quadrotor_mpc\2.Code" -WindowStyle Hidden -PassThru
# Expected: body background is the Streamlit default (white / light), not rgb(16,21,31)
```

### Why this works

- No dashboard page contains inline `<style>` or hardcoded colors — all theming flows through
  `theme.py` (CSS) and `.streamlit/config.toml` (native theme). Removing the two call sites and
  the config file fully restores the baseline look.
- The `Home.py` graphviz DOT uses `theme.PALETTE` constants via an f-string; if `theme` is
  deleted, port the three colors back to literals (`#172033` / `#d7e0ea` / `#4ea1ff`) or revert
  the Home.py diff.
- All non-dashboard modules (`reporting/`, `interfaces/desktop/`, `control/`, `core/`,
  `application/`, etc.) were never touched and require no rollback.
