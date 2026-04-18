#!/usr/bin/env python3
"""Boat operator UI — Qt tabs for sensors, vehicle, and mission control.

Run with ./run_ui.sh (which activates the ~/karin venv and sources ROS2).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

try:
    import pyqtgraph as pg
    _HAS_PYQTGRAPH = True
except ImportError:   # venv not rebuilt; fall back to a plain label
    _HAS_PYQTGRAPH = False

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import odroid_ssh
from lib.lawnmower import generate as lawnmower_generate
from lib.mavros_launcher import MavrosLauncher
from lib.mission_upload import preview_text
from lib.ros_bridge import RosBridge
from lib.setpoint_streamer import SetpointStreamer
from lib.sonde_fields import DISPLAY_ORDER, display_fields

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("boat_ui")


# =========================================================================== #
#                                SHARED WIDGETS                                #
# =========================================================================== #

class StatusDot(QtWidgets.QLabel):
    """Small coloured circle that indicates connected/disconnected state."""

    def __init__(self, diameter: int = 14):
        super().__init__()
        self._d = diameter
        self.setFixedSize(diameter, diameter)
        self.set_ok(False)

    def set_ok(self, ok: bool) -> None:
        color = QtGui.QColor("#30c030" if ok else "#c03030")
        pm = QtGui.QPixmap(self._d, self._d)
        pm.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setBrush(color)
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(0, 0, self._d, self._d)
        p.end()
        self.setPixmap(pm)


class Sparkline(QtWidgets.QWidget):
    """Tiny line chart of the last N values (for sonar panel)."""

    def __init__(self, width: int = 160, height: int = 40):
        super().__init__()
        self.setFixedSize(width, height)
        self._values: List[float] = []

    def update_values(self, values: List[float]) -> None:
        self._values = list(values)
        self.update()

    def paintEvent(self, _ev: QtGui.QPaintEvent) -> None:
        if not self._values:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QtGui.QColor("#f4f4f4"))
        lo = min(self._values)
        hi = max(self._values)
        span = max(hi - lo, 0.1)
        pen = QtGui.QPen(QtGui.QColor("#2060c0"))
        pen.setWidth(2)
        p.setPen(pen)
        n = len(self._values)
        for i in range(n - 1):
            x0 = int(i * w / max(n - 1, 1))
            x1 = int((i + 1) * w / max(n - 1, 1))
            y0 = h - int((self._values[i] - lo) / span * (h - 4)) - 2
            y1 = h - int((self._values[i + 1] - lo) / span * (h - 4)) - 2
            p.drawLine(x0, y0, x1, y1)
        p.end()


# =========================================================================== #
#                                 SENSOR TAB                                   #
# =========================================================================== #

class SensorTab(QtWidgets.QWidget):
    def __init__(self, bridge: RosBridge):
        super().__init__()
        self._bridge = bridge

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self._build_sonar())
        layout.addWidget(self._build_sonde())
        layout.addWidget(self._build_winch())
        layout.addStretch(1)

    def _build_sonar(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Sonar (Ping1D)")
        v = QtWidgets.QVBoxLayout(box)

        self.sonar_status = QtWidgets.QLabel("stopped")
        self.sonar_value = QtWidgets.QLabel("—")
        font = self.sonar_value.font()
        font.setPointSize(24)
        font.setBold(True)
        self.sonar_value.setFont(font)
        self.sonar_spark = Sparkline()

        btns = QtWidgets.QHBoxLayout()
        start = QtWidgets.QPushButton("Start on Odroid")
        stop = QtWidgets.QPushButton("Stop")
        start.clicked.connect(self._start_sonar)
        stop.clicked.connect(self._stop_sonar)
        btns.addWidget(start)
        btns.addWidget(stop)

        v.addWidget(QtWidgets.QLabel("Range (m)"))
        v.addWidget(self.sonar_value)
        v.addWidget(self.sonar_spark)
        v.addLayout(btns)
        v.addWidget(self.sonar_status)
        v.addStretch(1)
        return box

    def _build_sonde(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Sonde (YSI EXO)")
        v = QtWidgets.QVBoxLayout(box)

        self.sonde_status = QtWidgets.QLabel("stopped")
        self.sonde_table = QtWidgets.QTableWidget(len(DISPLAY_ORDER), 2)
        self.sonde_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.sonde_table.horizontalHeader().setStretchLastSection(True)
        self.sonde_table.verticalHeader().setVisible(False)
        self.sonde_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        for i, (label, _key) in enumerate(DISPLAY_ORDER):
            self.sonde_table.setItem(i, 0, QtWidgets.QTableWidgetItem(label))
            self.sonde_table.setItem(i, 1, QtWidgets.QTableWidgetItem("—"))

        btns = QtWidgets.QHBoxLayout()
        start = QtWidgets.QPushButton("Start on Odroid")
        stop = QtWidgets.QPushButton("Stop")
        start.clicked.connect(self._start_sonde)
        stop.clicked.connect(self._stop_sonde)
        btns.addWidget(start)
        btns.addWidget(stop)

        v.addWidget(self.sonde_table)
        v.addLayout(btns)
        v.addWidget(self.sonde_status)
        return box

    def _build_winch(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Winch (FT232H + AUX1)")
        v = QtWidgets.QVBoxLayout(box)

        self.winch_gpio_label = QtWidgets.QLabel("GPIO: (click Read)")
        self.winch_length_label = QtWidgets.QLabel("spool: — m")

        read_btn = QtWidgets.QPushButton("Read GPIO (SSH)")
        read_btn.clicked.connect(self._read_winch_gpio)

        self.arm_winch = QtWidgets.QCheckBox("ARM WINCH (required for motor)")
        self.arm_winch.setStyleSheet("QCheckBox { color: #a00; font-weight: bold; }")

        # Fine-grain speed buttons. Negative = raise, positive = lower.
        speeds = QtWidgets.QGridLayout()
        speeds.addWidget(QtWidgets.QLabel("<b>Raise</b>"), 0, 0)
        for col, pct in enumerate((-0.40, -0.30, -0.20, -0.10)):
            btn = QtWidgets.QPushButton(f"{int(abs(pct)*100)}%")
            btn.clicked.connect(lambda _, p=pct: self._winch_move(p))
            speeds.addWidget(btn, 0, col + 1)
        stop_btn = QtWidgets.QPushButton("STOP")
        stop_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        stop_btn.clicked.connect(lambda: self._winch_move(0.0))
        speeds.addWidget(stop_btn, 1, 0, 1, 5)
        speeds.addWidget(QtWidgets.QLabel("<b>Lower</b>"), 2, 0)
        for col, pct in enumerate((0.10, 0.20, 0.30, 0.40)):
            btn = QtWidgets.QPushButton(f"{int(pct*100)}%")
            btn.clicked.connect(lambda _, p=pct: self._winch_move(p))
            speeds.addWidget(btn, 2, col + 1)

        # Closed-loop "move to depth" — uses /winch length feedback.
        cl_row = QtWidgets.QHBoxLayout()
        self.winch_target = QtWidgets.QDoubleSpinBox()
        self.winch_target.setRange(0.0, 12.0)
        self.winch_target.setSingleStep(0.1)
        self.winch_target.setSuffix(" m")
        self.winch_target.setValue(1.0)
        goto_btn = QtWidgets.QPushButton("Go to depth")
        goto_btn.clicked.connect(self._winch_goto_depth)
        cl_row.addWidget(QtWidgets.QLabel("Target depth:"))
        cl_row.addWidget(self.winch_target)
        cl_row.addWidget(goto_btn)

        v.addWidget(self.winch_length_label)
        v.addWidget(self.winch_gpio_label)
        v.addWidget(read_btn)
        v.addWidget(self.arm_winch)
        v.addLayout(speeds)
        v.addLayout(cl_row)
        v.addStretch(1)

        # State for the closed-loop controller
        self._winch_goal: Optional[float] = None
        self._winch_poll = QtCore.QTimer(self)
        self._winch_poll.timeout.connect(self._winch_goto_tick)
        return box

    # --- actions --- #

    def _start_sonar(self) -> None:
        self.sonar_status.setText("starting…")
        self._run_async(lambda: odroid_ssh.start_sensor(odroid_ssh.SONAR),
                        lambda r: self.sonar_status.setText(r[1]))

    def _stop_sonar(self) -> None:
        self.sonar_status.setText("stopping…")
        self._run_async(lambda: odroid_ssh.stop_sensor(odroid_ssh.SONAR),
                        lambda r: self.sonar_status.setText(r[1]))

    def _start_sonde(self) -> None:
        self.sonde_status.setText("starting…")
        self._run_async(lambda: odroid_ssh.start_sensor(odroid_ssh.SONDE),
                        lambda r: self.sonde_status.setText(r[1]))

    def _stop_sonde(self) -> None:
        self.sonde_status.setText("stopping…")
        self._run_async(lambda: odroid_ssh.stop_sensor(odroid_ssh.SONDE),
                        lambda r: self.sonde_status.setText(r[1]))

    def _read_winch_gpio(self) -> None:
        self.winch_gpio_label.setText("GPIO: reading…")
        self._run_async(odroid_ssh.read_winch_gpio, self._on_gpio)

    def _on_gpio(self, result: dict) -> None:
        if "error" in result:
            self.winch_gpio_label.setText(f"GPIO error: {result['error']}")
            return
        parts = [f"{k}={v}" for k, v in result.items()]
        self.winch_gpio_label.setText("GPIO: " + "  ".join(parts))

    def _winch_move(self, speed: float) -> None:
        # STOP button always works; any non-zero speed needs ARM WINCH checked
        if speed != 0.0 and not self.arm_winch.isChecked():
            QtWidgets.QMessageBox.warning(self, "Winch locked",
                "Check 'ARM WINCH' before moving the motor.")
            return
        if speed == 0.0:
            # Stop also aborts any closed-loop move in progress
            self._winch_goal = None
            self._winch_poll.stop()
        ok, msg = self._bridge.winch_pwm(speed)
        self.winch_gpio_label.setText(f"winch cmd: {speed:+.2f} {msg}")

    def _winch_goto_depth(self) -> None:
        if not self.arm_winch.isChecked():
            QtWidgets.QMessageBox.warning(self, "Winch locked",
                "Check 'ARM WINCH' before moving the motor.")
            return
        self._winch_goal = float(self.winch_target.value())
        self._winch_poll.start(100)   # 10 Hz closed-loop
        self.winch_gpio_label.setText(f"goto: {self._winch_goal:.2f} m")

    def _winch_goto_tick(self) -> None:
        """10 Hz closed-loop: read /winch, drive motor toward goal."""
        if self._winch_goal is None:
            self._winch_poll.stop()
            return
        length = self._bridge.latest().winch_m
        error = self._winch_goal - length  # positive = lower more
        tol = 0.05   # 5 cm
        if abs(error) < tol:
            self._bridge.winch_pwm(0.0)
            self.winch_gpio_label.setText(
                f"arrived: {length:.2f} m (target {self._winch_goal:.2f})")
            self._winch_goal = None
            self._winch_poll.stop()
            return
        # Proportional speed (saturated)
        speed = max(min(error * 0.8, 0.40), -0.40)
        if abs(speed) < 0.15:           # motor dead-zone
            speed = 0.15 if speed > 0 else -0.15
        self._bridge.winch_pwm(speed)
        self.winch_gpio_label.setText(
            f"goto: now {length:.2f} → {self._winch_goal:.2f} m "
            f"(speed {speed:+.2f})")

    # --- small async helper ---

    def _run_async(self, fn, on_done) -> None:
        """Run fn() on a thread; call on_done(result) on the Qt thread."""
        import threading
        def worker():
            try:
                result = fn()
            except Exception as e:
                result = (False, f"error: {e}")
            QtCore.QMetaObject.invokeMethod(
                self, "_deliver",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(object, (on_done, result)),
            )
        threading.Thread(target=worker, daemon=True).start()

    @QtCore.pyqtSlot(object)
    def _deliver(self, payload):
        cb, result = payload
        cb(result)

    # --- polling --- #

    def refresh(self) -> None:
        s = self._bridge.latest()
        if s.ping_range_m >= 0:
            self.sonar_value.setText(f"{s.ping_range_m:5.2f}")
        self.sonar_spark.update_values(s.ping_samples)

        if s.sonde_raw:
            rows = display_fields(s.sonde_raw)
            if rows is not None:
                for i, (_label, val) in enumerate(rows):
                    self.sonde_table.item(i, 1).setText(str(val))

        self.winch_length_label.setText(f"spool: {s.winch_m:.2f} m")


# =========================================================================== #
#                                 VEHICLE TAB                                  #
# =========================================================================== #

class VehicleTab(QtWidgets.QWidget):
    def __init__(self, bridge: RosBridge, mavros: MavrosLauncher):
        super().__init__()
        self._bridge = bridge
        self._mavros = mavros

        grid = QtWidgets.QGridLayout(self)
        row = 0

        # MAVROS launcher panel (URLs hardcoded to sane defaults — edit
        # DEFAULT_FCU / DEFAULT_GCS in lib/mavros_launcher.py if you need to)
        mav_box = QtWidgets.QGroupBox("MAVROS")
        mg = QtWidgets.QHBoxLayout(mav_box)
        start_mav = QtWidgets.QPushButton("Start MAVROS")
        stop_mav = QtWidgets.QPushButton("Stop MAVROS")
        start_mav.clicked.connect(self._start_mavros)
        stop_mav.clicked.connect(self._stop_mavros)
        self.mavros_dot = StatusDot()
        self.mavros_text = QtWidgets.QLabel("not launched")
        mg.addWidget(start_mav)
        mg.addWidget(stop_mav)
        mg.addSpacing(20)
        mg.addWidget(self.mavros_dot)
        mg.addWidget(self.mavros_text)
        mg.addStretch(1)
        grid.addWidget(mav_box, row, 0, 1, 4)
        row += 1

        # Link
        self.link_dot = StatusDot()
        self.link_text = QtWidgets.QLabel("disconnected")
        grid.addWidget(QtWidgets.QLabel("Link:"), row, 0)
        grid.addWidget(self.link_dot, row, 1)
        grid.addWidget(self.link_text, row, 2, 1, 2)
        row += 1

        # Mode
        self.mode_label = QtWidgets.QLabel("—")
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["MANUAL", "OFFBOARD", "AUTO.MISSION", "AUTO.LOITER", "AUTO.RTL", "POSCTL"])
        set_mode_btn = QtWidgets.QPushButton("Set Mode")
        set_mode_btn.clicked.connect(self._do_set_mode)
        grid.addWidget(QtWidgets.QLabel("Current mode:"), row, 0)
        grid.addWidget(self.mode_label, row, 1)
        grid.addWidget(self.mode_combo, row, 2)
        grid.addWidget(set_mode_btn, row, 3)
        row += 1

        # Arm
        self.arm_btn = QtWidgets.QPushButton("ARM")
        self.arm_btn.setStyleSheet("QPushButton { background: #c03030; color: white; font-weight: bold; padding: 8px; }")
        self.arm_btn.clicked.connect(self._toggle_arm)
        self.armed_label = QtWidgets.QLabel("disarmed")
        grid.addWidget(QtWidgets.QLabel("Armed:"), row, 0)
        grid.addWidget(self.armed_label, row, 1)
        grid.addWidget(self.arm_btn, row, 2, 1, 2)
        row += 1

        # Position
        self.pos_x = QtWidgets.QLabel("—")
        self.pos_y = QtWidgets.QLabel("—")
        self.pos_z = QtWidgets.QLabel("—")
        self.gps = QtWidgets.QLabel("—")
        grid.addWidget(QtWidgets.QLabel("Local X, Y, Z:"), row, 0)
        grid.addWidget(self.pos_x, row, 1)
        grid.addWidget(self.pos_y, row, 2)
        grid.addWidget(self.pos_z, row, 3)
        row += 1
        grid.addWidget(QtWidgets.QLabel("GPS:"), row, 0)
        grid.addWidget(self.gps, row, 1, 1, 3)
        row += 1

        # Battery
        self.battery = QtWidgets.QLabel("—")
        grid.addWidget(QtWidgets.QLabel("Battery:"), row, 0)
        grid.addWidget(self.battery, row, 1, 1, 3)
        row += 1

        # Motor test
        mt_box = QtWidgets.QGroupBox("Motor test (no arm required)")
        mt_grid = QtWidgets.QGridLayout(mt_box)
        self.mt_motor = QtWidgets.QSpinBox(); self.mt_motor.setRange(1, 8); self.mt_motor.setValue(1)
        self.mt_throttle = QtWidgets.QSpinBox(); self.mt_throttle.setRange(0, 80); self.mt_throttle.setValue(20)
        self.mt_duration = QtWidgets.QDoubleSpinBox(); self.mt_duration.setRange(0.1, 10.0); self.mt_duration.setValue(2.0)
        mt_btn = QtWidgets.QPushButton("Spin")
        mt_btn.clicked.connect(self._do_motor_test)
        mt_grid.addWidget(QtWidgets.QLabel("Motor #"), 0, 0); mt_grid.addWidget(self.mt_motor, 0, 1)
        mt_grid.addWidget(QtWidgets.QLabel("Throttle %"), 0, 2); mt_grid.addWidget(self.mt_throttle, 0, 3)
        mt_grid.addWidget(QtWidgets.QLabel("Duration (s)"), 0, 4); mt_grid.addWidget(self.mt_duration, 0, 5)
        mt_grid.addWidget(mt_btn, 0, 6)
        grid.addWidget(mt_box, row, 0, 1, 4)
        row += 1

        grid.setRowStretch(row, 1)

    # --- actions --- #

    def _start_mavros(self) -> None:
        ok, msg = self._mavros.start()   # uses DEFAULT_FCU / DEFAULT_GCS
        QtWidgets.QMessageBox.information(self, "MAVROS", msg)

    def _stop_mavros(self) -> None:
        ok, msg = self._mavros.stop()
        QtWidgets.QMessageBox.information(self, "MAVROS", msg)

    def _do_set_mode(self) -> None:
        mode = self.mode_combo.currentText()
        ok, msg = self._bridge.set_mode(mode)
        QtWidgets.QMessageBox.information(self, "Set mode", f"{mode}: {msg}")

    def _toggle_arm(self) -> None:
        currently_armed = self._bridge.latest().armed
        target = not currently_armed
        if target:
            ans = QtWidgets.QMessageBox.question(
                self, "Arm vehicle?",
                "This will arm the boat. Thrusters become live once setpoints stream or manual input is present. Proceed?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if ans != QtWidgets.QMessageBox.Yes:
                return
        ok, msg = self._bridge.arm(target)
        QtWidgets.QMessageBox.information(self, "Arm/Disarm", f"target={target}: {msg}")

    def _do_motor_test(self) -> None:
        ok, msg = self._bridge.motor_test(
            self.mt_motor.value(),
            float(self.mt_throttle.value()),
            float(self.mt_duration.value()),
        )
        if ok:
            self._info(f"motor test sent: {msg}")
        else:
            QtWidgets.QMessageBox.warning(self, "Motor test", f"failed: {msg}")

    def _info(self, msg: str) -> None:
        win = self.window()
        if hasattr(win, "set_status"):
            win.set_status(msg)

    # --- polling --- #

    def refresh(self) -> None:
        s = self._bridge.latest()
        self.link_dot.set_ok(s.connected)
        self.link_text.setText("connected" if s.connected else "disconnected")
        self.mode_label.setText(s.mode or "—")
        self.armed_label.setText("ARMED" if s.armed else "disarmed")
        self.arm_btn.setText("DISARM" if s.armed else "ARM")
        self.pos_x.setText(f"{s.pose_x:.2f}")
        self.pos_y.setText(f"{s.pose_y:.2f}")
        self.pos_z.setText(f"{s.pose_z:.2f}")
        self.gps.setText(f"{s.gps_lat:.6f}, {s.gps_lon:.6f}"
                         if s.gps_lat or s.gps_lon else "no fix")
        self.battery.setText(f"{s.battery_pct:.0f}%" if s.battery_pct >= 0 else "—")

        running = self._mavros.is_running()
        self.mavros_dot.set_ok(running)
        self.mavros_text.setText("running" if running else "not launched")


# =========================================================================== #
#                                 MISSION TAB                                  #
# =========================================================================== #

class LivePositionPlot(QtWidgets.QWidget):
    """Shows the boat's local position as a moving dot + trail."""

    TRAIL = 200

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.gps_label = QtWidgets.QLabel("GPS: —")
        layout.addWidget(self.gps_label)

        if _HAS_PYQTGRAPH:
            pg.setConfigOptions(antialias=True, background="w", foreground="k")
            self._plot = pg.PlotWidget()
            self._plot.setLabel("bottom", "x (m, local NED)")
            self._plot.setLabel("left", "y (m, local NED)")
            self._plot.showGrid(x=True, y=True, alpha=0.3)
            self._plot.setAspectLocked(True)
            self._trail = self._plot.plot([], [], pen=pg.mkPen(QtGui.QColor("#2060c0"), width=2))
            self._dot = self._plot.plot([], [], pen=None,
                symbol="o", symbolBrush=QtGui.QColor("#c03030"), symbolSize=14)
            self._target_dot = self._plot.plot([], [], pen=None,
                symbol="x", symbolBrush=QtGui.QColor("#30a030"), symbolSize=16)
            layout.addWidget(self._plot)
        else:
            self._plot = None
            layout.addWidget(QtWidgets.QLabel(
                "(install pyqtgraph in the karin venv for the live position plot)"
            ))

        self._xs: List[float] = []
        self._ys: List[float] = []

    def update_position(self, x: float, y: float, target_xy: Optional[Tuple[float, float]] = None):
        self._xs.append(x)
        self._ys.append(y)
        if len(self._xs) > self.TRAIL:
            self._xs = self._xs[-self.TRAIL:]
            self._ys = self._ys[-self.TRAIL:]
        if self._plot is not None:
            self._trail.setData(self._xs, self._ys)
            self._dot.setData([x], [y])
            if target_xy is not None:
                self._target_dot.setData([target_xy[0]], [target_xy[1]])
            else:
                self._target_dot.setData([], [])

    def update_gps(self, lat: float, lon: float) -> None:
        if lat == 0.0 and lon == 0.0:
            self.gps_label.setText("GPS: no fix")
        else:
            self.gps_label.setText(f"GPS: {lat:.6f}°, {lon:.6f}°")


class MissionTab(QtWidgets.QWidget):
    def __init__(self, bridge: RosBridge):
        super().__init__()
        self._bridge = bridge
        self._streamer: Optional[SetpointStreamer] = None
        self._mission: List[Tuple[float, float]] = []

        outer = QtWidgets.QHBoxLayout(self)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(self._build_position_controller())
        left.addWidget(self._build_mission_section())
        left.addStretch(1)

        # Live position panel (right half)
        self.live_plot = LivePositionPlot()
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("<b>Live position</b>"))
        right.addWidget(self.live_plot)

        lw = QtWidgets.QWidget(); lw.setLayout(left)
        rw = QtWidgets.QWidget(); rw.setLayout(right)
        outer.addWidget(lw, 2)
        outer.addWidget(rw, 3)

    # ---------- position controller ---------- #

    def _build_position_controller(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Positional controller (OFFBOARD)")
        g = QtWidgets.QGridLayout(box)

        self.tx = QtWidgets.QDoubleSpinBox(); self.tx.setRange(-1000, 1000); self.tx.setDecimals(2)
        self.ty = QtWidgets.QDoubleSpinBox(); self.ty.setRange(-1000, 1000); self.ty.setDecimals(2)
        self.current_target = QtWidgets.QLabel("target: none")
        self.streamer_dot = StatusDot()
        self.streamer_text = QtWidgets.QLabel("streamer: stopped")

        hold_btn = QtWidgets.QPushButton("Hold current position")
        go_btn = QtWidgets.QPushButton("Go to (x, y)")
        stop_btn = QtWidgets.QPushButton("Stop streaming")
        hold_btn.clicked.connect(self._hold_current)
        go_btn.clicked.connect(self._go_to_xy)
        stop_btn.clicked.connect(self._stop_streaming)

        g.addWidget(QtWidgets.QLabel("Target X (m):"), 0, 0)
        g.addWidget(self.tx, 0, 1)
        g.addWidget(QtWidgets.QLabel("Target Y (m):"), 0, 2)
        g.addWidget(self.ty, 0, 3)
        g.addWidget(hold_btn, 1, 0, 1, 2)
        g.addWidget(go_btn, 1, 2, 1, 2)
        g.addWidget(stop_btn, 2, 0, 1, 2)
        g.addWidget(self.streamer_dot, 2, 2)
        g.addWidget(self.streamer_text, 2, 3)
        g.addWidget(self.current_target, 3, 0, 1, 4)

        return box

    def _hold_current(self) -> None:
        s = self._bridge.latest()
        self._ensure_streamer()
        self._streamer.set_target(s.pose_x, s.pose_y, 0.0)
        self.tx.setValue(s.pose_x)
        self.ty.setValue(s.pose_y)
        self._refresh_streamer_display()

    def _go_to_xy(self) -> None:
        self._ensure_streamer()
        self._streamer.set_target(self.tx.value(), self.ty.value(), 0.0)
        self._refresh_streamer_display()

    def _stop_streaming(self) -> None:
        if self._streamer:
            self._streamer.stop()
            self._streamer = None
        self._refresh_streamer_display()

    def _ensure_streamer(self) -> None:
        if self._streamer is None or not self._streamer.is_running():
            self._streamer = SetpointStreamer(self._bridge.node())
            self._streamer.start()

    def _refresh_streamer_display(self) -> None:
        running = bool(self._streamer and self._streamer.is_running())
        self.streamer_dot.set_ok(running)
        self.streamer_text.setText("streamer: running (20 Hz)" if running else "streamer: stopped")
        if self._streamer:
            t = self._streamer.target()
            self.current_target.setText(f"target: ({t.x:.2f}, {t.y:.2f})")
        else:
            self.current_target.setText("target: none")

    # ---------- mission ---------- #

    def _build_mission_section(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Lawnmower mission")
        g = QtWidgets.QGridLayout(box)

        self.xmin = QtWidgets.QDoubleSpinBox(); self.xmin.setRange(-1000, 1000); self.xmin.setValue(0.0)
        self.xmax = QtWidgets.QDoubleSpinBox(); self.xmax.setRange(-1000, 1000); self.xmax.setValue(25.0)
        self.ymin = QtWidgets.QDoubleSpinBox(); self.ymin.setRange(-1000, 1000); self.ymin.setValue(0.0)
        self.ymax = QtWidgets.QDoubleSpinBox(); self.ymax.setRange(-1000, 1000); self.ymax.setValue(25.0)
        self.spacing = QtWidgets.QDoubleSpinBox(); self.spacing.setRange(0.5, 50); self.spacing.setValue(5.0)
        self.buffer = QtWidgets.QDoubleSpinBox(); self.buffer.setRange(0.0, 50); self.buffer.setValue(2.5)
        self.home_lat = QtWidgets.QDoubleSpinBox(); self.home_lat.setRange(-90, 90); self.home_lat.setDecimals(7)
        self.home_lon = QtWidgets.QDoubleSpinBox(); self.home_lon.setRange(-180, 180); self.home_lon.setDecimals(7)

        preview_btn = QtWidgets.QPushButton("Preview pattern")
        upload_btn = QtWidgets.QPushButton("Upload to Pixhawk")
        start_btn = QtWidgets.QPushButton("Start mission (AUTO.MISSION + arm)")
        pause_btn = QtWidgets.QPushButton("Pause (AUTO.LOITER)")
        preview_btn.clicked.connect(self._preview)
        upload_btn.clicked.connect(self._upload)
        start_btn.clicked.connect(self._start_mission)
        pause_btn.clicked.connect(self._pause_mission)

        self.preview_box = QtWidgets.QPlainTextEdit()
        self.preview_box.setReadOnly(True)
        self.preview_box.setMinimumHeight(200)

        r = 0
        g.addWidget(QtWidgets.QLabel("x min / max:"), r, 0); g.addWidget(self.xmin, r, 1); g.addWidget(self.xmax, r, 2); r += 1
        g.addWidget(QtWidgets.QLabel("y min / max:"), r, 0); g.addWidget(self.ymin, r, 1); g.addWidget(self.ymax, r, 2); r += 1
        g.addWidget(QtWidgets.QLabel("Row spacing / buffer:"), r, 0); g.addWidget(self.spacing, r, 1); g.addWidget(self.buffer, r, 2); r += 1
        g.addWidget(QtWidgets.QLabel("Home lat / lon:"), r, 0); g.addWidget(self.home_lat, r, 1); g.addWidget(self.home_lon, r, 2); r += 1
        g.addWidget(preview_btn, r, 0); g.addWidget(upload_btn, r, 1, 1, 2); r += 1
        g.addWidget(start_btn, r, 0, 1, 2); g.addWidget(pause_btn, r, 2); r += 1
        g.addWidget(self.preview_box, r, 0, 1, 3); r += 1
        g.addWidget(QtWidgets.QLabel(
            "Tip: complex missions — build them in QGC and upload from QGC.\n"
            "This tab handles simple lawnmowers only."), r, 0, 1, 3)

        return box

    def _preview(self) -> None:
        try:
            self._mission = lawnmower_generate(
                self.xmin.value(), self.xmax.value(),
                self.ymin.value(), self.ymax.value(),
                self.spacing.value(), self.buffer.value(),
            )
            self.preview_box.setPlainText(preview_text(self._mission))
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Bad bounds", str(e))

    def _upload(self) -> None:
        if not self._mission:
            QtWidgets.QMessageBox.warning(self, "No mission", "Click Preview first.")
            return
        if self.home_lat.value() == 0.0 and self.home_lon.value() == 0.0:
            s = self._bridge.latest()
            if s.gps_lat == 0.0 and s.gps_lon == 0.0:
                QtWidgets.QMessageBox.warning(self, "Need home position",
                    "Set home lat/lon, or wait for GPS fix (currently 0,0).")
                return
            self.home_lat.setValue(s.gps_lat)
            self.home_lon.setValue(s.gps_lon)
        ok, msg = self._bridge.push_waypoints(
            self._mission, self.home_lat.value(), self.home_lon.value()
        )
        QtWidgets.QMessageBox.information(self, "Upload", f"{'OK' if ok else 'FAILED'}: {msg}")

    def _start_mission(self) -> None:
        if not self._mission:
            QtWidgets.QMessageBox.warning(self, "No mission", "Upload waypoints first.")
            return
        ans = QtWidgets.QMessageBox.question(
            self, "Start mission?",
            "This will set mode AUTO.MISSION and arm. Proceed?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if ans != QtWidgets.QMessageBox.Yes:
            return
        ok1, m1 = self._bridge.set_mode("AUTO.MISSION")
        ok2, m2 = self._bridge.arm(True)
        QtWidgets.QMessageBox.information(self, "Start", f"mode: {m1}\narm: {m2}")

    def _pause_mission(self) -> None:
        ok, msg = self._bridge.set_mode("AUTO.LOITER")
        QtWidgets.QMessageBox.information(self, "Pause", msg)

    # --- polling --- #

    def refresh(self) -> None:
        self._refresh_streamer_display()
        s = self._bridge.latest()
        target_xy = None
        if self._streamer and self._streamer.is_running():
            t = self._streamer.target()
            target_xy = (t.x, t.y)
        self.live_plot.update_position(s.pose_x, s.pose_y, target_xy)
        self.live_plot.update_gps(s.gps_lat, s.gps_lon)


# =========================================================================== #
#                                MAIN WINDOW                                   #
# =========================================================================== #

class BoatUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Karin — Boat Operator UI")
        self.resize(1100, 720)

        self._bridge = RosBridge()
        self._bridge.start()
        self._mavros = MavrosLauncher()

        tabs = QtWidgets.QTabWidget()
        self.sensor_tab = SensorTab(self._bridge)
        self.vehicle_tab = VehicleTab(self._bridge, self._mavros)
        self.mission_tab = MissionTab(self._bridge)
        tabs.addTab(self.vehicle_tab, "Vehicle")
        tabs.addTab(self.sensor_tab, "Sensors")
        tabs.addTab(self.mission_tab, "Mission")
        self.setCentralWidget(tabs)

        self.status = self.statusBar()
        self.status.showMessage("starting…")

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(100)

    def set_status(self, msg: str) -> None:
        self.status.showMessage(msg, 4000)

    def _tick(self) -> None:
        self.sensor_tab.refresh()
        self.vehicle_tab.refresh()
        self.mission_tab.refresh()
        latest = self._bridge.latest()
        conn = "OK" if latest.connected else "link down"
        self.status.showMessage(f"{conn} | mode={latest.mode or '—'} | "
                                f"armed={latest.armed} | bat={latest.battery_pct:.0f}%")

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        try:
            if self.mission_tab._streamer:
                self.mission_tab._streamer.stop()
            if self._mavros.is_running():
                self._mavros.stop()
        finally:
            self._bridge.shutdown()
        super().closeEvent(ev)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = BoatUI()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
