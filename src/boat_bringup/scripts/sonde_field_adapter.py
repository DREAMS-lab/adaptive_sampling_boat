#!/usr/bin/env python3
"""
Sonde -> field adapter for the boat.

Subscribes to `sonde_data` (std_msgs/String) published by the `sonde_read` package's
`read_serial.py`. That node strips the upstream `#DATA: ` prefix and emits
a single whitespace-padded fixed-width row with columns:

    DATA TIME VOID Temp(C) pH Depth(m) SpC HDO_pct HDO_mgL Chl CDOM Turb

This adapter whitespace-splits the row, pulls the configured column index
(default 3 = temperature in °C), validates the value, and republishes as
`std_msgs/Float32` on the topic the planner already expects:

    /gaussian_field/boat/temperature_noisy

No smoothing — pass-through, per the boat-port plan.
"""

import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String


class SondeFieldAdapter(Node):
    def __init__(self):
        super().__init__('sonde_field_adapter')

        self.declare_parameter('input_topic', 'sonde_data')
        self.declare_parameter('output_topic', '/gaussian_field/boat/temperature_noisy')
        self.declare_parameter('column_index', 3)
        self.declare_parameter('temp_min', 0.0)
        self.declare_parameter('temp_max', 50.0)
        self.declare_parameter('expected_column_count', 12)

        self.in_topic = self.get_parameter('input_topic').value
        self.out_topic = self.get_parameter('output_topic').value
        self.col_idx = int(self.get_parameter('column_index').value)
        self.temp_min = float(self.get_parameter('temp_min').value)
        self.temp_max = float(self.get_parameter('temp_max').value)
        self.expected_cols = int(self.get_parameter('expected_column_count').value)

        self.pub = self.create_publisher(Float32, self.out_topic, 10)
        self.sub = self.create_subscription(String, self.in_topic, self._cb, 10)

        self._last_warn = 0.0
        self._n_published = 0
        self._n_rejected = 0

        self.get_logger().info(
            f'sonde_field_adapter: {self.in_topic} (col {self.col_idx}) -> {self.out_topic}, '
            f'range [{self.temp_min}, {self.temp_max}] °C'
        )

    def _warn(self, message: str):
        now = time.time()
        if now - self._last_warn > 5.0:
            self.get_logger().warn(message)
            self._last_warn = now

    def _cb(self, msg: String):
        raw = msg.data
        # Defensive: read_serial strips '#DATA: ' but be tolerant if it ever doesn't.
        if raw.startswith('#DATA:'):
            raw = raw.replace('#DATA:', '', 1).lstrip()
        tokens = raw.split()
        if len(tokens) != self.expected_cols:
            self._warn(
                f'sonde row column count {len(tokens)} != expected {self.expected_cols} '
                f'(payload: "{raw[:80]}")'
            )
            self._n_rejected += 1
            return
        if self.col_idx >= len(tokens):
            self._warn(f'column_index {self.col_idx} out of range for row of {len(tokens)} tokens')
            self._n_rejected += 1
            return
        token = tokens[self.col_idx]
        try:
            temp = float(token)
        except ValueError:
            self._warn(f'sonde temperature token "{token}" is not a float')
            self._n_rejected += 1
            return
        if math.isnan(temp) or math.isinf(temp):
            self._warn(f'sonde temperature is NaN/Inf: {token}')
            self._n_rejected += 1
            return
        if not (self.temp_min <= temp <= self.temp_max):
            self._warn(f'sonde temperature {temp:.3f} outside [{self.temp_min}, {self.temp_max}]')
            self._n_rejected += 1
            return

        out = Float32()
        out.data = temp
        self.pub.publish(out)
        self._n_published += 1


def main(args=None):
    rclpy.init(args=args)
    node = SondeFieldAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f'sonde_field_adapter shutdown: published={node._n_published}, rejected={node._n_rejected}'
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
