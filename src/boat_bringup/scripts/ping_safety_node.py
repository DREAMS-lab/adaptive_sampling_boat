#!/usr/bin/env python3
"""
Ping1D depth-safety node.

Watches /ping1d/data (Float32, meters below the transducer). If `consecutive`
consecutive readings are below `min_safe_depth + mount_offset`, switches the
autopilot to AUTO.LOITER via /mavros/set_mode and latches a SHALLOW_HOLD
state on /boat_safety/state for the operator dashboard / RViz.

Does NOT auto-recover or auto-disarm — operator restarts the mission.
RC override (pilot on the transmitter) remains the primary safety layer.
"""

from collections import deque
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Float32, String
from mavros_msgs.srv import SetMode


class PingSafetyNode(Node):
    def __init__(self):
        super().__init__('ping_safety_node')

        self.declare_parameter('depth_topic', '/ping1d/data')
        self.declare_parameter('min_safe_depth', 0.5)
        self.declare_parameter('mount_offset', 0.0)
        self.declare_parameter('consecutive', 3)
        self.declare_parameter('hold_mode', 'AUTO.LOITER')

        self.depth_topic = self.get_parameter('depth_topic').value
        self.min_safe_depth = float(self.get_parameter('min_safe_depth').value)
        self.mount_offset = float(self.get_parameter('mount_offset').value)
        self.consecutive = int(self.get_parameter('consecutive').value)
        self.hold_mode = str(self.get_parameter('hold_mode').value)

        self.threshold = self.min_safe_depth + self.mount_offset

        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.state_pub = self.create_publisher(String, '/boat_safety/state', qos_latched)
        self._publish_state('OK')

        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        self.recent = deque(maxlen=self.consecutive)
        self.tripped = False
        self._last_call_time = 0.0

        self.sub = self.create_subscription(Float32, self.depth_topic, self._cb, 10)

        self.get_logger().info(
            f'ping_safety_node watching {self.depth_topic}, '
            f'threshold {self.threshold:.2f} m (= min_safe_depth {self.min_safe_depth:.2f} '
            f'+ mount_offset {self.mount_offset:.2f}), trip after {self.consecutive} samples'
        )

    def _publish_state(self, state: str):
        msg = String()
        msg.data = state
        self.state_pub.publish(msg)

    def _cb(self, msg: Float32):
        d = float(msg.data)
        # Ignore obviously bad readings (ping returns 0.0 when it loses bottom).
        if d <= 0.0:
            return
        self.recent.append(d)
        if len(self.recent) < self.consecutive:
            return
        if all(v < self.threshold for v in self.recent) and not self.tripped:
            self.tripped = True
            self.get_logger().error(
                f'SHALLOW: last {self.consecutive} depths all below {self.threshold:.2f} m '
                f'(values: {[round(v, 2) for v in self.recent]}); requesting {self.hold_mode}'
            )
            self._publish_state('SHALLOW_HOLD')
            self._request_hold()

    def _request_hold(self):
        # Rate-limit so we don't spam the service if it isn't ready
        now = time.time()
        if now - self._last_call_time < 1.0:
            return
        self._last_call_time = now
        if not self.set_mode_client.service_is_ready():
            self.get_logger().warn('/mavros/set_mode service not ready — retrying on next trip')
            return
        req = SetMode.Request()
        req.base_mode = 0
        req.custom_mode = self.hold_mode
        self.set_mode_client.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = PingSafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
