"""rclpy node that fronts MAVROS + sensor topics for the Qt UI.

Runs on its own thread via MultiThreadedExecutor. The Qt main loop reads
`latest()` at 10 Hz for live widget updates. Service calls block briefly in
whatever thread invokes them.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State, Waypoint
from mavros_msgs.srv import CommandBool, CommandLong, SetMode, WaypointPush
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, NavSatFix
from std_msgs.msg import Float32, String

log = logging.getLogger(__name__)


@dataclass
class LatestState:
    """Thread-safe snapshot of every topic the UI cares about."""
    connected: bool = False
    armed: bool = False
    mode: str = ""
    pose_x: float = 0.0
    pose_y: float = 0.0
    pose_z: float = 0.0
    battery_pct: float = -1.0
    gps_lat: float = 0.0
    gps_lon: float = 0.0
    ping_range_m: float = -1.0
    ping_samples: List[float] = field(default_factory=list)
    sonde_raw: str = ""
    winch_m: float = 0.0
    last_log: str = ""


class RosBridge:
    """High-level façade used by the UI. Wraps an rclpy node + executor thread."""

    def __init__(self) -> None:
        if not rclpy.ok():
            rclpy.init()

        self._node = Node("boat_ui_bridge")
        self._lock = threading.Lock()
        self._latest = LatestState()

        # --- subscriptions ---
        self._node.create_subscription(State, "/mavros/state", self._on_state, 10)
        self._node.create_subscription(
            PoseStamped, "/mavros/local_position/pose", self._on_pose, 10
        )
        self._node.create_subscription(
            BatteryState, "/mavros/battery", self._on_battery, 10
        )
        self._node.create_subscription(
            NavSatFix, "/mavros/global_position/global", self._on_gps, 10
        )
        self._node.create_subscription(Float32, "/ping1d/data", self._on_ping, 10)
        self._node.create_subscription(String, "/sonde_data", self._on_sonde, 10)
        self._node.create_subscription(Float32, "/winch", self._on_winch, 10)

        # --- service clients ---
        self._arming = self._node.create_client(CommandBool, "/mavros/cmd/arming")
        self._set_mode = self._node.create_client(SetMode, "/mavros/set_mode")
        self._cmd = self._node.create_client(CommandLong, "/mavros/cmd/command")
        self._wp_push = self._node.create_client(WaypointPush, "/mavros/mission/push")

        # --- executor thread ---
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def node(self) -> Node:
        return self._node

    def shutdown(self) -> None:
        self._executor.shutdown()
        self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def latest(self) -> LatestState:
        with self._lock:
            # return a shallow copy so the UI doesn't fight the callbacks
            return LatestState(
                connected=self._latest.connected,
                armed=self._latest.armed,
                mode=self._latest.mode,
                pose_x=self._latest.pose_x,
                pose_y=self._latest.pose_y,
                pose_z=self._latest.pose_z,
                battery_pct=self._latest.battery_pct,
                gps_lat=self._latest.gps_lat,
                gps_lon=self._latest.gps_lon,
                ping_range_m=self._latest.ping_range_m,
                ping_samples=list(self._latest.ping_samples),
                sonde_raw=self._latest.sonde_raw,
                winch_m=self._latest.winch_m,
                last_log=self._latest.last_log,
            )

    # ------------------------- service wrappers ------------------------- #

    def set_mode(self, mode: str, timeout: float = 3.0) -> Tuple[bool, str]:
        if not self._set_mode.wait_for_service(timeout_sec=timeout):
            return False, "set_mode service unavailable"
        req = SetMode.Request()
        req.custom_mode = mode
        fut = self._set_mode.call_async(req)
        ok = self._wait(fut, timeout)
        if not ok:
            return False, "set_mode timeout"
        res = fut.result()
        return bool(res.mode_sent), f"mode_sent={res.mode_sent}"

    def arm(self, value: bool, timeout: float = 3.0) -> Tuple[bool, str]:
        if not self._arming.wait_for_service(timeout_sec=timeout):
            return False, "arming service unavailable"
        req = CommandBool.Request()
        req.value = bool(value)
        fut = self._arming.call_async(req)
        if not self._wait(fut, timeout):
            return False, "arming timeout"
        res = fut.result()
        return bool(res.success), f"result={res.result}"

    def motor_test(
        self, motor: int, throttle_pct: float, duration_s: float, timeout: float = 3.0
    ) -> Tuple[bool, str]:
        """MAV_CMD_DO_MOTOR_TEST (209). throttle_pct is 0-100, duration_s in seconds."""
        if not self._cmd.wait_for_service(timeout_sec=timeout):
            return False, "cmd service unavailable"
        req = CommandLong.Request()
        req.broadcast = False
        req.command = 209
        req.confirmation = 0
        req.param1 = float(motor)
        req.param2 = 0.0                 # throttle type = percent
        req.param3 = float(throttle_pct)
        req.param4 = float(duration_s)
        fut = self._cmd.call_async(req)
        if not self._wait(fut, timeout):
            return False, "cmd timeout"
        res = fut.result()
        return bool(res.success), f"result={res.result}"

    def winch_pwm(self, value: float, timeout: float = 2.0) -> Tuple[bool, str]:
        """Send raw AUX1 PWM via MAV_CMD_DO_SET_SERVO (187) — matches roswinch.py."""
        if not self._cmd.wait_for_service(timeout_sec=timeout):
            return False, "cmd service unavailable"
        req = CommandLong.Request()
        req.broadcast = False
        req.command = 187
        req.confirmation = 0
        req.param1 = float(-value)        # polarity-flipped like roswinch.py
        req.param2 = float("nan")
        req.param3 = float("nan")
        req.param4 = float("nan")
        req.param5 = float("nan")
        req.param6 = float("nan")
        req.param7 = 0.0
        fut = self._cmd.call_async(req)
        if not self._wait(fut, timeout):
            return False, "cmd timeout"
        res = fut.result()
        return bool(res.success), f"result={res.result}"

    def push_waypoints(
        self,
        waypoints_local_xy: List[Tuple[float, float]],
        home_lat: float,
        home_lon: float,
        altitude: float = 0.0,
        accept_radius: float = 1.0,
        timeout: float = 5.0,
    ) -> Tuple[bool, str]:
        """Upload a list of local (x, y) points as GPS waypoints.

        Local-frame metres are converted to lat/lon using a flat-earth
        approximation around (home_lat, home_lon). Good enough for <1 km surveys.
        """
        if not self._wp_push.wait_for_service(timeout_sec=timeout):
            return False, "waypoint_push service unavailable"

        R = 6371000.0   # metres
        from math import cos, radians
        lat0 = radians(home_lat)

        wps: List[Waypoint] = []
        for i, (x, y) in enumerate(waypoints_local_xy):
            wp = Waypoint()
            wp.frame = 3                 # MAV_FRAME_GLOBAL_RELATIVE_ALT
            wp.command = 16              # MAV_CMD_NAV_WAYPOINT
            wp.is_current = (i == 0)
            wp.autocontinue = True
            wp.param1 = 0.0              # hold time
            wp.param2 = float(accept_radius)
            wp.param3 = 0.0              # pass-through radius
            wp.param4 = float("nan")     # yaw (NaN = keep current)
            # flat-earth delta → lat/lon (local NED: x=north, y=east)
            wp.x_lat = home_lat + (x / R) * (180.0 / 3.141592653589793)
            wp.x_long = home_lon + (y / (R * cos(lat0))) * (180.0 / 3.141592653589793)
            wp.z_alt = float(altitude)
            wps.append(wp)

        req = WaypointPush.Request()
        req.start_index = 0
        req.waypoints = wps
        fut = self._wp_push.call_async(req)
        if not self._wait(fut, timeout):
            return False, "waypoint_push timeout"
        res = fut.result()
        return bool(res.success), f"transferred={res.wp_transfered}"

    # ----------------------------- internals ----------------------------- #

    def _wait(self, fut, timeout: float) -> bool:
        import time
        t0 = time.monotonic()
        while not fut.done() and time.monotonic() - t0 < timeout:
            time.sleep(0.02)
        return fut.done()

    def _spin(self) -> None:
        try:
            self._executor.spin()
        except Exception as e:
            log.exception("executor died: %s", e)

    # --------------------------- subscriptions --------------------------- #

    def _on_state(self, msg: State) -> None:
        with self._lock:
            self._latest.connected = bool(msg.connected)
            self._latest.armed = bool(msg.armed)
            self._latest.mode = str(msg.mode)

    def _on_pose(self, msg: PoseStamped) -> None:
        with self._lock:
            self._latest.pose_x = float(msg.pose.position.x)
            self._latest.pose_y = float(msg.pose.position.y)
            self._latest.pose_z = float(msg.pose.position.z)

    def _on_battery(self, msg: BatteryState) -> None:
        with self._lock:
            self._latest.battery_pct = float(msg.percentage) * 100.0

    def _on_gps(self, msg: NavSatFix) -> None:
        with self._lock:
            self._latest.gps_lat = float(msg.latitude)
            self._latest.gps_lon = float(msg.longitude)

    def _on_ping(self, msg: Float32) -> None:
        with self._lock:
            self._latest.ping_range_m = float(msg.data)
            self._latest.ping_samples.append(float(msg.data))
            if len(self._latest.ping_samples) > 40:
                self._latest.ping_samples = self._latest.ping_samples[-40:]

    def _on_sonde(self, msg: String) -> None:
        with self._lock:
            self._latest.sonde_raw = str(msg.data)

    def _on_winch(self, msg: Float32) -> None:
        with self._lock:
            self._latest.winch_m = float(msg.data)
