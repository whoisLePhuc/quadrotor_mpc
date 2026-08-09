"""Optional PySide6/pyqtgraph control and telemetry process.

Qt owns the child process main thread while MuJoCo/GLFW remains in the parent.
Only plain dictionaries cross the process boundary.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from queue import Empty, Full
from typing import Any

from quadrotor_mpc.application.native.commands import CommandName, RuntimeCommand
from quadrotor_mpc.interfaces.desktop.model import (
    DANGER,
    INFO,
    MUTED,
    OK,
    WARNING,
    PanelRuntimeContext,
    PanelViewState,
    build_panel_view,
    panel_transition_alerts,
)

# Semantic color tokens (WCAG 2.2 AA targets). Raw hex values belong ONLY here.
# Light theme — verified in Tasks 1-2 (contrast ≥4.5:1 text, ≥3:1 non-text,
# ΔE pass, hues preserved, Okabe-Ito color-blind-safe series).
PALETTE = {
    # ── Surfaces ──────────────────────────────────────────────────────
    "surface":            "#f8f9fa",   # main panel background
    "surface_elevated":   "#ffffff",   # cards, elevated panels
    "surface_inset":      "#e8eaed",   # inset / recessed areas (darker)
    "surface_hover":      "#f1f3f4",   # hover bg for list items, rows

    # ── Text on surface ───────────────────────────────────────────────
    "on_surface":         "#1a1a1a",   # primary text (highest contrast)
    "on_surface_muted":   "#5f6368",   # secondary text
    "on_surface_dim":     "#5b6166",   # tertiary text (still ≥4.5:1 on all surfaces)

    # ── Interactive ────────────────────────────────────────────────────
    "interactive":          "#1558c0", # blue accent — links, focus, plot pens
    "interactive_hover":    "#e8f0fe", # light-blue hover bg for interactive elems
    "interactive_secondary": "#f1f3f4", # secondary button bg (light gray)
    "interactive_focus":    "#1558c0", # focus-ring outline color

    # ── Borders (non-text, ≥3:1) ──────────────────────────────────────
    "border":             "#80868b",  # general UI border
    "button_border":      "#80868b",  # button border

    # ── Buttons (white text on colored bg, ≥4.5:1) ───────────────────
    "button_ok":          "#137333",  # green — white text passes 4.5:1
    "button_danger":      "#c5221f",  # red — white text passes 4.5:1

    # ── Semantic tones (bg, fg) — fg ≥4.5:1 on bg ────────────────────
    "tone_ok":      ("#e6f4ea", "#137333"),   # light-green bg, green fg
    "tone_info":    ("#e8f0fe", "#1558c0"),    # light-blue bg, blue fg
    "tone_warning": ("#fef7e0", "#b45309"),   # light-amber bg, amber fg
    "tone_danger":  ("#fce8e6", "#c5221f"),   # light-red bg, red fg
    "tone_muted":   ("#f1f3f4", "#5f6368"),    # light-gray bg, gray fg

    # ── Badges & status ───────────────────────────────────────────────
    "badge_bg":           "#e8eaed",   # badge / chip background
    "status_bg":          "#f1f3f4",   # status-banner background
    "label_title":        "#202124",   # card-title label (dark, prominent)

    # ── Plot series (Okabe-Ito color-blind-safe) ──────────────────────
    "series_purple":      "#CC79A7",   # Okabe-Ito "reddish purple"
    "series_cyan":        "#56B4E9",   # Okabe-Ito "sky blue"
    "series_orange":      "#E69F00",   # Okabe-Ito "orange"
}

# ── Modular typography scale ──────────────────────────────────────────────
# Minor Third ratio (1.200) — recommended for general desktop UIs (qt-ui-design §1.2).
# base = 16px (WCAG/accessibility minimum for body text).
#   ms(-1) = 16 ÷ 1.2 ≈ 13px  → caption (card title, card detail)
#   ms( 0) = 16px              → body (QWidget base, status banner, alert list)
#   ms(+1) = 16 × 1.2 ≈ 19px  → value (KPI numbers, mode badge)
#   ms(+2) = 19 × 1.2 ≈ 23px  → title (header scenario label)
# ≤4 distinct sizes on screen.  font-size in QSS is device-independent px and
# respects Qt's default DPI scaling — do NOT use pt (would mis-scale on HiDPI).
TYPE_SCALE = {
    "caption": 13,  # ms(-1)
    "body": 16,     # ms( 0)
    "value": 19,    # ms(+1)
    "title": 23,    # ms(+2)
}


@dataclass(frozen=True, slots=True)
class DesktopPanelOptions:
    enabled: bool = True
    update_hz: float = 15.0
    history_seconds: float = 30.0
    stop_when_closed: bool = True
    maximum_alerts: int = 12

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> DesktopPanelOptions:
        update_hz = float(raw.get("update_hz", 15.0))
        history_seconds = float(raw.get("history_seconds", 30.0))
        if update_hz <= 0.0 or history_seconds <= 0.0:
            raise ValueError("panel.update_hz and history_seconds must be > 0")
        maximum_alerts = int(raw.get("maximum_alerts", 12))
        if maximum_alerts < 1:
            raise ValueError("panel.maximum_alerts must be >= 1")
        return cls(
            enabled=bool(raw.get("enabled", True)),
            update_hz=update_hz,
            history_seconds=history_seconds,
            stop_when_closed=bool(raw.get("stop_when_closed", True)),
            maximum_alerts=maximum_alerts,
        )


class DesktopPanelProcess:
    def __init__(
        self,
        options: DesktopPanelOptions,
        title: str,
        context: PanelRuntimeContext,
    ):
        self.options = options
        self.title = title
        self.context = context
        mp_context = mp.get_context("spawn")
        self.command_queue = mp_context.Queue(maxsize=64)
        self.telemetry_queue = mp_context.Queue(maxsize=256)
        self._process = mp_context.Process(
            target=_panel_main,
            args=(
                self.command_queue,
                self.telemetry_queue,
                options,
                title,
                self.context,
            ),
            daemon=True,
            name="quadrotor-telemetry-panel",
        )

    def start(self) -> None:
        self._process.start()

    def is_alive(self) -> bool:
        return self._process.is_alive()

    @property
    def exitcode(self) -> int | None:
        return self._process.exitcode

    def drain_commands(self) -> list[RuntimeCommand]:
        commands: list[RuntimeCommand] = []
        while True:
            try:
                commands.append(
                    RuntimeCommand.from_message(self.command_queue.get_nowait(), source="panel")
                )
            except Empty:
                return commands

    def publish(self, sample: dict[str, Any]) -> None:
        try:
            self.telemetry_queue.put_nowait(sample)
        except Full:
            try:
                self.telemetry_queue.get_nowait()
            except Empty:
                pass
            try:
                self.telemetry_queue.put_nowait(sample)
            except Full:
                pass

    def reset(self) -> None:
        """Clear the child window's plots for a new episode."""
        try:
            self.telemetry_queue.put_nowait({"kind": "reset"})
        except Full:
            pass

    def close(self) -> None:
        if not self._process.is_alive():
            return
        try:
            self.telemetry_queue.put_nowait({"kind": "shutdown"})
        except Full:
            pass
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)


def _panel_main(
    command_queue: Any,
    telemetry_queue: Any,
    options: DesktopPanelOptions,
    title: str,
    context: PanelRuntimeContext,
) -> None:
    import sys
    from collections import deque

    import pyqtgraph as pg
    from PySide6 import QtCore, QtGui, QtWidgets

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(
        antialias=True,
        background=PALETTE["surface"],
        foreground=PALETTE["on_surface"],
    )
    app.setStyleSheet(
        f"""
        QWidget {{ background:{PALETTE["surface"]}; color:{PALETTE["on_surface"]}; font-size:{TYPE_SCALE["body"]}px; }}
        QPushButton {{
            background:{PALETTE["interactive_secondary"]}; border:1px solid {PALETTE["button_border"]}; border-radius:5px;
            padding:7px 11px;
        }}
        QPushButton:hover {{ background:{PALETTE["interactive_hover"]}; }}
        QPushButton:focus {{
            border:2px solid {PALETTE["interactive_focus"]};
            outline:1px solid {PALETTE["interactive_focus"]};
        }}
        QListWidget {{
            background:{PALETTE["surface_inset"]}; border:1px solid {PALETTE["border"]}; border-radius:5px;
        }}
        QListWidget:focus {{
            border:2px solid {PALETTE["interactive_focus"]};
        }}
        """
    )

    class Panel(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(f"{title} — CC-MPC Safety Console")
            self.resize(1480, 930)
            self.times: deque[float] = deque()
            self.px: deque[float] = deque()
            self.py: deque[float] = deque()
            self.pz: deque[float] = deque()
            self.goal_error: deque[float] = deque()
            self.clearance: deque[float] = deque()
            self.chance_residual: deque[float] = deque()
            self.slack: deque[float] = deque()
            self.horizon_position_sigma: deque[float] = deque()
            self.projected_uncertainty: deque[float] = deque()
            self.tightened_radius: deque[float] = deque()
            self.thrust: deque[float] = deque()
            self.tau_norm: deque[float] = deque()
            self.solve_ms: deque[float] = deque()
            self.deadline_ms: deque[float] = deque()
            self.risk_fraction: deque[float] = deque()
            self.solution_accepted: deque[float] = deque()
            self.fallback_level: deque[float] = deque()
            self._last_view: PanelViewState | None = None
            self._card_widgets: dict[
                str,
                tuple[QtWidgets.QFrame, QtWidgets.QLabel, QtWidgets.QLabel],
            ] = {}
            self._build_ui()
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._poll)
            self.timer.start(max(10, round(1000.0 / options.update_hz)))

        def _build_ui(self) -> None:
            central = QtWidgets.QWidget()
            outer = QtWidgets.QVBoxLayout(central)
            outer.setContentsMargins(12, 10, 12, 10)
            outer.setSpacing(8)

            header = QtWidgets.QHBoxLayout()
            scenario = QtWidgets.QLabel(f"<b>{title}</b>")
            scenario.setStyleSheet(f"font-size:{TYPE_SCALE['title']}px")
            mode = QtWidgets.QLabel(context.mode_label)
            mode.setStyleSheet(
                f"padding:5px 9px;background:{PALETTE['badge_bg']};"
                f"color:{PALETTE['on_surface_dim']};border-radius:5px;"
                f"font-size:{TYPE_SCALE['value']}px"
            )
            header.addWidget(scenario)
            header.addStretch(1)
            header.addWidget(mode)
            outer.addLayout(header)

            controls = QtWidgets.QHBoxLayout()
            for label, name in (
                ("Pause / Resume", CommandName.TOGGLE_PAUSE),
                ("Step", CommandName.STEP),
                ("Reset", CommandName.RESET),
                ("Run again", CommandName.RUN_AGAIN),
                ("Snapshot", CommandName.SNAPSHOT),
                ("Stop", CommandName.STOP),
            ):
                button = QtWidgets.QPushButton(label)
                button.clicked.connect(lambda _checked=False, n=name: self._send(n))
                if name == CommandName.RUN_AGAIN:
                    button.setStyleSheet(
                        f"background:{PALETTE['button_ok']};color:white;font-weight:bold"
                    )
                if name == CommandName.STOP:
                    button.setStyleSheet(
                        f"background:{PALETTE['button_danger']};color:white;font-weight:bold"
                    )
                controls.addWidget(button)
            controls.addStretch(1)
            # Visual grouping (Hick's Law / Proximity, qt-ui-design §1): the six
            # control actions and the four view toggles are distinct groups — a
            # thin divider keeps the row from reading as one undifferentiated
            # wall of equal-weight buttons. QFrame is not focusable, so this
            # separator does not affect keyboard tab order.
            group_separator = QtWidgets.QFrame()
            group_separator.setFixedWidth(2)
            group_separator.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
            group_separator.setStyleSheet(
                f"QFrame{{background:{PALETTE['border']};border:none;}}"
            )
            controls.addWidget(group_separator)
            controls.addStretch(1)
            for label, name in (
                ("Trail", CommandName.TOGGLE_TRAIL),
                ("MPC prediction", CommandName.TOGGLE_PREDICTION),
                ("Safety shells", CommandName.TOGGLE_SAFETY),
                ("Follow camera", CommandName.TOGGLE_CAMERA),
            ):
                button = QtWidgets.QPushButton(label)
                button.clicked.connect(lambda _checked=False, n=name: self._send(n))
                controls.addWidget(button)
            outer.addLayout(controls)

            card_grid = QtWidgets.QGridLayout()
            for index, (key, label) in enumerate(
                (
                    ("runtime", "Episode"),
                    ("controller", "Applied control"),
                    ("assurance", "Safety assurance"),
                    ("risk", "Risk budget"),
                    ("slack", "Chance constraint"),
                    ("deadline", "Solver timing"),
                )
            ):
                frame, value, detail = self._create_card(label)
                self._card_widgets[key] = (frame, value, detail)
                card_grid.addWidget(frame, index // 3, index % 3)
            outer.addLayout(card_grid)

            self.status = QtWidgets.QLabel("Waiting for simulation telemetry…")
            self.status.setWordWrap(True)
            self.status.setStyleSheet(
                f"padding:8px;background:{PALETTE['status_bg']};"
                f"border:1px solid {PALETTE['border']};border-radius:5px"
            )
            outer.addWidget(self.status)

            plots = QtWidgets.QGridLayout()
            plots.setSpacing(7)
            self.position_plot = pg.PlotWidget(title="Position (m)")
            self.safety_plot = pg.PlotWidget(title="Goal / clearance / chance residual / slack (m)")
            self.uncertainty_plot = pg.PlotWidget(title="Predicted uncertainty / safety radius (m)")
            self.control_plot = pg.PlotWidget(title="Control input")
            self.solver_plot = pg.PlotWidget(title="NMPC solve time (ms)")
            self.supervisor_plot = pg.PlotWidget(
                title="Risk use / accepted command / fallback level"
            )
            for plot in (
                self.position_plot,
                self.safety_plot,
                self.uncertainty_plot,
                self.control_plot,
                self.solver_plot,
                self.supervisor_plot,
            ):
                plot.showGrid(x=True, y=True, alpha=0.25)
                plot.setLabel("bottom", "Simulation time", units="s")
            self.position_plot.addLegend()
            self.safety_plot.addLegend()
            self.uncertainty_plot.addLegend()
            self.control_plot.addLegend()
            self.solver_plot.addLegend()
            self.supervisor_plot.addLegend()
            self.position_curves = [
                self.position_plot.plot(pen=pg.mkPen(color, width=2), name=axis)
                for color, axis in (
                    (PALETTE["interactive"], "x"),
                    (PALETTE["tone_ok"][1], "y"),
                    (PALETTE["tone_warning"][1], "z"),
                )
            ]
            self.goal_curve = self.safety_plot.plot(
                pen=pg.mkPen(PALETTE["series_purple"], width=2),
                name="goal error",
            )
            self.clearance_curve = self.safety_plot.plot(
                pen=pg.mkPen(PALETTE["tone_ok"][1], width=2), name="clearance"
            )
            self.chance_curve = self.safety_plot.plot(
                pen=pg.mkPen(PALETTE["tone_warning"][1], width=2), name="min chance residual"
            )
            self.slack_curve = self.safety_plot.plot(
                pen=pg.mkPen(PALETTE["tone_danger"][1], width=2), name="max slack"
            )
            self.horizon_sigma_curve = self.uncertainty_plot.plot(
                pen=pg.mkPen(PALETTE["series_cyan"], width=2, style=QtCore.Qt.DashLine),
                name="max terminal σ position",
            )
            self.projected_uncertainty_curve = self.uncertainty_plot.plot(
                pen=pg.mkPen(PALETTE["series_purple"], width=2),
                name="max projected σ",
            )
            self.tightened_radius_curve = self.uncertainty_plot.plot(
                pen=pg.mkPen(PALETTE["series_orange"], width=2),
                name="max tightened radius",
            )
            self.thrust_curve = self.control_plot.plot(
                pen=pg.mkPen(PALETTE["interactive"], width=2), name="thrust dev."
            )
            self.torque_curve = self.control_plot.plot(
                pen=pg.mkPen(PALETTE["tone_warning"][1], width=2), name="||torque||"
            )
            self.solve_curve = self.solver_plot.plot(
                pen=pg.mkPen(PALETTE["tone_ok"][1], width=2),
                name="solve time",
            )
            self.deadline_curve = self.solver_plot.plot(
                pen=pg.mkPen(PALETTE["tone_danger"][1], width=2, style=QtCore.Qt.DashLine),
                name="accept deadline",
            )
            self.risk_curve = self.supervisor_plot.plot(
                pen=pg.mkPen(PALETTE["interactive"], width=2),
                name="joint risk fraction",
            )
            self.accepted_curve = self.supervisor_plot.plot(
                pen=pg.mkPen(PALETTE["tone_ok"][1], width=2),
                name="solution accepted",
            )
            self.fallback_curve = self.supervisor_plot.plot(
                pen=pg.mkPen(PALETTE["tone_danger"][1], width=2),
                name="fallback level",
            )
            plots.addWidget(self.position_plot, 0, 0)
            plots.addWidget(self.safety_plot, 0, 1)
            plots.addWidget(self.uncertainty_plot, 0, 2)
            plots.addWidget(self.control_plot, 1, 0)
            plots.addWidget(self.solver_plot, 1, 1)
            plots.addWidget(self.supervisor_plot, 1, 2)
            outer.addLayout(plots, 1)

            alert_header = QtWidgets.QLabel("<b>Operational transitions</b>")
            outer.addWidget(alert_header)
            self.alert_list = QtWidgets.QListWidget()
            self.alert_list.setMaximumHeight(105)
            outer.addWidget(self.alert_list)
            self.setCentralWidget(central)

        @staticmethod
        def _create_card(
            title_text: str,
        ) -> tuple[QtWidgets.QFrame, QtWidgets.QLabel, QtWidgets.QLabel]:
            frame = QtWidgets.QFrame()
            layout = QtWidgets.QVBoxLayout(frame)
            layout.setContentsMargins(10, 7, 10, 7)
            layout.setSpacing(2)
            title_label = QtWidgets.QLabel(title_text)
            title_label.setStyleSheet(f"color:{PALETTE['label_title']};font-size:{TYPE_SCALE['caption']}px")
            value_label = QtWidgets.QLabel("WAITING")
            value_label.setStyleSheet(f"font-size:{TYPE_SCALE['value']}px;font-weight:bold")
            detail_label = QtWidgets.QLabel("No telemetry")
            detail_label.setStyleSheet(f"color:{PALETTE['on_surface_muted']};font-size:{TYPE_SCALE['caption']}px")
            detail_label.setWordWrap(True)
            layout.addWidget(title_label)
            layout.addWidget(value_label)
            layout.addWidget(detail_label)
            frame.setStyleSheet(
                f"QFrame{{background:{PALETTE['surface_elevated']};"
                f"border:1px solid {PALETTE['border']};border-radius:6px}}"
            )
            return frame, value_label, detail_label

        def _send(self, name: CommandName) -> None:
            try:
                command_queue.put_nowait(RuntimeCommand(name=name, source="panel").as_message())
            except Full:
                self.status.setText("Command queue is full; command was not sent")

        def _poll(self) -> None:
            latest = None
            while True:
                try:
                    message = telemetry_queue.get_nowait()
                except Empty:
                    break
                if message.get("kind") == "shutdown":
                    self.close()
                    return
                if message.get("kind") == "reset":
                    self._clear_history()
                    self.status.setText("Episode reset — simulation running…")
                    continue
                latest = message
                self._append(message)
            if latest is not None:
                current = build_panel_view(latest, context)
                for alert in panel_transition_alerts(self._last_view, current):
                    self._add_alert(alert.time_s, alert.tone, alert.message)
                self._last_view = current
                self._render_view(current)
                self._redraw()

        def _render_view(self, view: PanelViewState) -> None:
            colors = {
                OK: PALETTE["tone_ok"],
                INFO: PALETTE["tone_info"],
                WARNING: PALETTE["tone_warning"],
                DANGER: PALETTE["tone_danger"],
                MUTED: PALETTE["tone_muted"],
            }
            for card in view.cards:
                frame, value, detail = self._card_widgets[card.key]
                background, foreground = colors[card.tone]
                frame.setStyleSheet(
                    "QFrame{"
                    f"background:{background};border:1px solid {foreground};"
                    "border-radius:6px}"
                )
                value.setText(card.value)
                value.setStyleSheet(f"font-size:{TYPE_SCALE['value']}px;font-weight:bold;color:{foreground}")
                detail.setText(card.detail)
            background, foreground = colors[view.runtime_tone]
            self.status.setText(view.banner)
            self.status.setStyleSheet(
                f"padding:8px;background:{background};color:{foreground};"
                f"border:1px solid {foreground};border-radius:5px;font-weight:bold"
            )

        def _add_alert(self, time_s: float, tone: str, message: str) -> None:
            colors = {
                OK: PALETTE["tone_ok"][1],
                INFO: PALETTE["tone_info"][1],
                WARNING: PALETTE["tone_warning"][1],
                DANGER: PALETTE["tone_danger"][1],
                MUTED: PALETTE["tone_muted"][1],
            }
            item = QtWidgets.QListWidgetItem(f"{time_s:7.2f}s  {message}")
            item.setForeground(QtGui.QColor(colors[tone]))
            self.alert_list.insertItem(0, item)
            while self.alert_list.count() > options.maximum_alerts:
                self.alert_list.takeItem(self.alert_list.count() - 1)

        def _clear_history(self) -> None:
            for series in (
                self.times,
                self.px,
                self.py,
                self.pz,
                self.goal_error,
                self.clearance,
                self.chance_residual,
                self.slack,
                self.horizon_position_sigma,
                self.projected_uncertainty,
                self.tightened_radius,
                self.thrust,
                self.tau_norm,
                self.solve_ms,
                self.deadline_ms,
                self.risk_fraction,
                self.solution_accepted,
                self.fallback_level,
            ):
                series.clear()
            self._last_view = None
            self.alert_list.clear()
            self._redraw()

        def _append(self, sample: dict[str, Any]) -> None:
            t = float(sample["time_s"])
            terminal_sigma = sample.get("horizon_terminal_position_sigma")
            terminal_sigma_max = float("nan") if terminal_sigma is None else max(terminal_sigma)
            risk_total = sample.get("risk_budget_total")
            risk_fraction = (
                float("nan")
                if risk_total in (None, 0.0)
                else float(sample.get("risk_budget_allocated", 0.0)) / float(risk_total)
            )
            chance_residual = sample.get("minimum_chance_residual_m")
            slack = sample.get("maximum_slack_m")
            projected = sample.get("maximum_projected_uncertainty_m")
            tightened = sample.get("maximum_tightened_safety_radius_m")
            for series, value in (
                (self.times, t),
                (self.px, sample["position"][0]),
                (self.py, sample["position"][1]),
                (self.pz, sample["position"][2]),
                (self.goal_error, sample["goal_distance_m"]),
                (self.clearance, sample["min_clearance_m"]),
                (
                    self.chance_residual,
                    float("nan") if chance_residual is None else chance_residual,
                ),
                (self.slack, float("nan") if slack is None else slack),
                (self.horizon_position_sigma, terminal_sigma_max),
                (
                    self.projected_uncertainty,
                    float("nan") if projected is None else projected,
                ),
                (
                    self.tightened_radius,
                    float("nan") if tightened is None else tightened,
                ),
                (self.thrust, sample["control"][0]),
                (self.tau_norm, sum(v * v for v in sample["control"][1:]) ** 0.5),
                (self.solve_ms, sample["solver_time_ms"]),
                (
                    self.deadline_ms,
                    context.solve_deadline_ms if context.supervisor_enabled else float("nan"),
                ),
                (self.risk_fraction, risk_fraction),
                (
                    self.solution_accepted,
                    1.0 if sample.get("solution_accepted", True) else 0.0,
                ),
                (self.fallback_level, sample.get("fallback_level", 0)),
            ):
                series.append(float(value))
            cutoff = t - options.history_seconds
            while self.times and self.times[0] < cutoff:
                for series in (
                    self.times,
                    self.px,
                    self.py,
                    self.pz,
                    self.goal_error,
                    self.clearance,
                    self.chance_residual,
                    self.slack,
                    self.horizon_position_sigma,
                    self.projected_uncertainty,
                    self.tightened_radius,
                    self.thrust,
                    self.tau_norm,
                    self.solve_ms,
                    self.deadline_ms,
                    self.risk_fraction,
                    self.solution_accepted,
                    self.fallback_level,
                ):
                    series.popleft()

        def _redraw(self) -> None:
            x = list(self.times)
            for curve, series in zip(self.position_curves, (self.px, self.py, self.pz)):
                curve.setData(x, list(series))
            self.goal_curve.setData(x, list(self.goal_error))
            self.clearance_curve.setData(x, list(self.clearance))
            self.chance_curve.setData(x, list(self.chance_residual))
            self.slack_curve.setData(x, list(self.slack))
            self.horizon_sigma_curve.setData(
                x,
                list(self.horizon_position_sigma),
            )
            self.projected_uncertainty_curve.setData(
                x,
                list(self.projected_uncertainty),
            )
            self.tightened_radius_curve.setData(x, list(self.tightened_radius))
            self.thrust_curve.setData(x, list(self.thrust))
            self.torque_curve.setData(x, list(self.tau_norm))
            self.solve_curve.setData(x, list(self.solve_ms))
            self.deadline_curve.setData(x, list(self.deadline_ms))
            self.risk_curve.setData(x, list(self.risk_fraction))
            self.accepted_curve.setData(x, list(self.solution_accepted))
            self.fallback_curve.setData(x, list(self.fallback_level))

        def closeEvent(self, event: Any) -> None:
            if options.stop_when_closed:
                self._send(CommandName.STOP)
            event.accept()

    window = Panel()
    window.show()
    app.exec()
