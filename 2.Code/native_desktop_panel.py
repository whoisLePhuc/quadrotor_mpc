"""Optional PySide6/pyqtgraph control and telemetry process.

Qt owns the child process main thread while MuJoCo/GLFW remains in the parent.
Only plain dictionaries cross the process boundary.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from queue import Empty, Full
from typing import Any

from runtime_control import CommandName, RuntimeCommand


@dataclass(frozen=True, slots=True)
class DesktopPanelOptions:
    enabled: bool = True
    update_hz: float = 15.0
    history_seconds: float = 30.0
    stop_when_closed: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "DesktopPanelOptions":
        update_hz = float(raw.get("update_hz", 15.0))
        history_seconds = float(raw.get("history_seconds", 30.0))
        if update_hz <= 0.0 or history_seconds <= 0.0:
            raise ValueError("panel.update_hz and history_seconds must be > 0")
        return cls(
            enabled=bool(raw.get("enabled", True)),
            update_hz=update_hz,
            history_seconds=history_seconds,
            stop_when_closed=bool(raw.get("stop_when_closed", True)),
        )


class DesktopPanelProcess:
    def __init__(self, options: DesktopPanelOptions, title: str):
        self.options = options
        self.title = title
        context = mp.get_context("spawn")
        self.command_queue = context.Queue(maxsize=64)
        self.telemetry_queue = context.Queue(maxsize=256)
        self._process = context.Process(
            target=_panel_main,
            args=(self.command_queue, self.telemetry_queue, options, title),
            daemon=True,
            name="quadrotor-telemetry-panel",
        )

    def start(self) -> None:
        self._process.start()

    def is_alive(self) -> bool:
        return self._process.is_alive()

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
    command_queue: Any, telemetry_queue: Any, options: DesktopPanelOptions, title: str
) -> None:
    import sys
    from collections import deque

    import pyqtgraph as pg
    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True, background="#10151f", foreground="#d7e0ea")

    class Panel(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(f"{title} — Control & Telemetry")
            self.resize(1180, 760)
            self.times: deque[float] = deque()
            self.px: deque[float] = deque()
            self.py: deque[float] = deque()
            self.pz: deque[float] = deque()
            self.goal_error: deque[float] = deque()
            self.clearance: deque[float] = deque()
            self.horizon_position_sigma: deque[float] = deque()
            self.thrust: deque[float] = deque()
            self.tau_norm: deque[float] = deque()
            self.solve_ms: deque[float] = deque()
            self._build_ui()
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._poll)
            self.timer.start(max(10, round(1000.0 / options.update_hz)))

        def _build_ui(self) -> None:
            central = QtWidgets.QWidget()
            outer = QtWidgets.QVBoxLayout(central)
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
                    button.setStyleSheet("background:#267a4a;color:white;font-weight:bold")
                if name == CommandName.STOP:
                    button.setStyleSheet("background:#8f2430;color:white;font-weight:bold")
                controls.addWidget(button)
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

            self.status = QtWidgets.QLabel("Waiting for simulation telemetry…")
            self.status.setStyleSheet("padding:8px;background:#1a2230;border-radius:4px")
            outer.addWidget(self.status)

            plots = QtWidgets.QGridLayout()
            self.position_plot = pg.PlotWidget(title="Position (m)")
            self.safety_plot = pg.PlotWidget(title="Goal error / clearance (m)")
            self.control_plot = pg.PlotWidget(title="Control input")
            self.solver_plot = pg.PlotWidget(title="NMPC solve time (ms)")
            for plot in (self.position_plot, self.safety_plot, self.control_plot, self.solver_plot):
                plot.showGrid(x=True, y=True, alpha=0.25)
                plot.setLabel("bottom", "Simulation time", units="s")
            self.position_plot.addLegend()
            self.safety_plot.addLegend()
            self.control_plot.addLegend()
            self.position_curves = [
                self.position_plot.plot(pen=pg.mkPen(color, width=2), name=axis)
                for color, axis in (("#4ea1ff", "x"), ("#62d68b", "y"), ("#ffcf5a", "z"))
            ]
            self.goal_curve = self.safety_plot.plot(
                pen=pg.mkPen("#d783ff", width=2), name="goal error"
            )
            self.clearance_curve = self.safety_plot.plot(
                pen=pg.mkPen("#ff704d", width=2), name="clearance"
            )
            self.horizon_sigma_curve = self.safety_plot.plot(
                pen=pg.mkPen("#58d5e8", width=2, style=QtCore.Qt.DashLine),
                name="max terminal σ position",
            )
            self.thrust_curve = self.control_plot.plot(
                pen=pg.mkPen("#4ea1ff", width=2), name="thrust dev."
            )
            self.torque_curve = self.control_plot.plot(
                pen=pg.mkPen("#ffcf5a", width=2), name="||torque||"
            )
            self.solve_curve = self.solver_plot.plot(pen=pg.mkPen("#62d68b", width=2))
            plots.addWidget(self.position_plot, 0, 0)
            plots.addWidget(self.safety_plot, 0, 1)
            plots.addWidget(self.control_plot, 1, 0)
            plots.addWidget(self.solver_plot, 1, 1)
            outer.addLayout(plots, 1)
            self.setCentralWidget(central)

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
                state = (
                    "COLLISION"
                    if latest["collided"]
                    else "COMPLETED"
                    if latest.get("completed")
                    else "PAUSED"
                    if latest.get("paused")
                    else "RUNNING"
                )
                color = (
                    "#f05a67"
                    if latest["collided"]
                    else "#4ea1ff"
                    if latest.get("completed")
                    else "#ffcf5a"
                    if latest.get("paused")
                    else "#62d68b"
                )
                self.status.setText(
                    f"<b style='color:{color}'>{state}</b> &nbsp; "
                    f"t={latest['time_s']:.2f}s &nbsp; "
                    f"goal={latest['goal_distance_m']:.3f}m &nbsp; "
                    f"clearance={latest['min_clearance_m']:.3f}m &nbsp; "
                    f"solve={latest['solver_time_ms']:.1f}ms &nbsp; "
                    f"solver={latest.get('solver_status', '')}"
                    + (
                        f" &nbsp; <b style='color:#f05a67'>"
                        f"fallback=L{latest.get('fallback_level', 0)}"
                        f"/{latest.get('command_source', '')}"
                        f" ({latest.get('fallback_reason', '')})</b>"
                        if latest.get("fallback_active")
                        else ""
                    )
                    + (
                        f" &nbsp; chance-res={latest['minimum_chance_residual_m']:.3f}m"
                        f" &nbsp; slack={latest['maximum_slack_m']:.3f}m"
                        if latest.get("minimum_chance_residual_m") is not None
                        and latest.get("maximum_slack_m") is not None
                        else ""
                    )
                    + (
                        f" &nbsp; risk={latest.get('risk_semantics', '')}"
                        f"/{latest.get('risk_allocation_method', '')}"
                        f" {latest.get('risk_budget_allocated', 0.0):.6f}"
                        f"/{latest['risk_budget_total']:.6f}"
                        f" ({latest.get('risk_budget_status', '')})"
                        if latest.get("risk_budget_total") is not None
                        else ""
                    )
                    + (
                        f" &nbsp; reason={latest.get('completion_reason', 'completed')}"
                        if latest.get("completed")
                        else ""
                    )
                )
                self._redraw()

        def _clear_history(self) -> None:
            for series in (
                self.times,
                self.px,
                self.py,
                self.pz,
                self.goal_error,
                self.clearance,
                self.horizon_position_sigma,
                self.thrust,
                self.tau_norm,
                self.solve_ms,
            ):
                series.clear()
            self._redraw()

        def _append(self, sample: dict[str, Any]) -> None:
            t = float(sample["time_s"])
            terminal_sigma = sample.get("horizon_terminal_position_sigma")
            terminal_sigma_max = (
                float("nan") if terminal_sigma is None else max(terminal_sigma)
            )
            for series, value in (
                (self.times, t),
                (self.px, sample["position"][0]),
                (self.py, sample["position"][1]),
                (self.pz, sample["position"][2]),
                (self.goal_error, sample["goal_distance_m"]),
                (self.clearance, sample["min_clearance_m"]),
                (self.horizon_position_sigma, terminal_sigma_max),
                (self.thrust, sample["control"][0]),
                (self.tau_norm, sum(v * v for v in sample["control"][1:]) ** 0.5),
                (self.solve_ms, sample["solver_time_ms"]),
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
                    self.horizon_position_sigma,
                    self.thrust,
                    self.tau_norm,
                    self.solve_ms,
                ):
                    series.popleft()

        def _redraw(self) -> None:
            x = list(self.times)
            for curve, series in zip(self.position_curves, (self.px, self.py, self.pz)):
                curve.setData(x, list(series))
            self.goal_curve.setData(x, list(self.goal_error))
            self.clearance_curve.setData(x, list(self.clearance))
            self.horizon_sigma_curve.setData(
                x,
                list(self.horizon_position_sigma),
            )
            self.thrust_curve.setData(x, list(self.thrust))
            self.torque_curve.setData(x, list(self.tau_norm))
            self.solve_curve.setData(x, list(self.solve_ms))

        def closeEvent(self, event: Any) -> None:
            if options.stop_when_closed:
                self._send(CommandName.STOP)
            event.accept()

    window = Panel()
    window.show()
    app.exec()
