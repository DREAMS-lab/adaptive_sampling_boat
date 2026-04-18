"""Background thread that streams setpoints to /mavros/setpoint_position/local.

PX4 drops out of OFFBOARD if setpoints stop for ~500 ms, so this runs at 20 Hz
and accepts a mutable target the UI can update.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


@dataclass
class Target:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw_w: float = 1.0   # quaternion w; identity for "any yaw"


class SetpointStreamer(threading.Thread):
    """Thread that publishes PoseStamped to /mavros/setpoint_position/local at 20 Hz."""

    TOPIC = "/mavros/setpoint_position/local"
    RATE_HZ = 20.0

    def __init__(self, node: Node):
        super().__init__(daemon=True)
        self._node = node
        self._pub = node.create_publisher(PoseStamped, self.TOPIC, 10)
        self._target = Target()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._running = threading.Event()

    def set_target(self, x: float, y: float, z: float = 0.0) -> None:
        with self._lock:
            self._target = Target(x=float(x), y=float(y), z=float(z))

    def target(self) -> Target:
        with self._lock:
            return Target(self._target.x, self._target.y, self._target.z, self._target.yaw_w)

    def is_running(self) -> bool:
        return self._running.is_set() and not self._stop.is_set()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        self._running.set()
        period = 1.0 / self.RATE_HZ
        try:
            while not self._stop.is_set() and rclpy.ok():
                msg = PoseStamped()
                msg.header.stamp = self._node.get_clock().now().to_msg()
                msg.header.frame_id = "map"
                with self._lock:
                    msg.pose.position.x = self._target.x
                    msg.pose.position.y = self._target.y
                    msg.pose.position.z = self._target.z
                    msg.pose.orientation.w = self._target.yaw_w
                self._pub.publish(msg)
                time.sleep(period)
        finally:
            self._running.clear()
