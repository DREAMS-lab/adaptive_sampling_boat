#!/usr/bin/env python3
"""
Interactive setpoint commander for bench testing.

Streams a setpoint at 10 Hz so PX4 stays happy in OFFBOARD. Type commands
on stdin:

    x y         set a new target (ENU meters, relative to MAVROS local origin)
    here        snap target to current pose (holds position)
    arm         /mavros/cmd/arming True
    disarm      /mavros/cmd/arming False
    offboard    /mavros/set_mode OFFBOARD
    posctl      /mavros/set_mode POSCTL
    p           print current pose, current target, distance, armed/mode
    q           quit (disarms + posctl on the way out)

Usage:
    ros2 run boat_bringup pose_commander.py
"""

import sys
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
import numpy as np


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


class PoseCommander(Node):
    def __init__(self):
        super().__init__('pose_commander')
        self.target = np.array([0.0, 0.0, 0.0])
        self.pose = None
        self.state = None

        self.sp_pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self._pose_cb, SENSOR_QOS)
        self.create_subscription(State, '/mavros/state', self._state_cb, 10)
        self.arm_cli = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_cli = self.create_client(SetMode, '/mavros/set_mode')

        self.create_timer(0.1, self._stream)
        self.get_logger().info('pose_commander: streaming setpoint at 10 Hz. Type "p" to inspect.')

    def _pose_cb(self, msg):
        self.pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    def _state_cb(self, msg):
        self.state = msg

    def _stream(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(self.target[0])
        msg.pose.position.y = float(self.target[1])
        msg.pose.position.z = float(self.target[2])
        msg.pose.orientation.w = 1.0
        self.sp_pub.publish(msg)

    def set_target(self, x, y, z=0.0):
        self.target = np.array([float(x), float(y), float(z)])
        self.get_logger().info(f'target -> ({x:.2f}, {y:.2f}, {z:.2f})')

    def snap_here(self):
        if self.pose is None:
            self.get_logger().warn('no pose yet')
            return
        self.target = self.pose.copy()
        self.get_logger().info(f'target snapped to current ({self.pose[0]:.2f}, {self.pose[1]:.2f})')

    def call_arm(self, value):
        req = CommandBool.Request()
        req.value = bool(value)
        self.arm_cli.call_async(req)
        self.get_logger().info(f'arm({value}) requested')

    def call_mode(self, custom_mode):
        req = SetMode.Request()
        req.base_mode = 0
        req.custom_mode = custom_mode
        self.mode_cli.call_async(req)
        self.get_logger().info(f'set_mode({custom_mode}) requested')

    def status_line(self):
        if self.pose is None or self.state is None:
            return 'pose=NONE state=NONE'
        d = float(np.linalg.norm(self.pose[:2] - self.target[:2]))
        return (f'pose=({self.pose[0]:+.2f}, {self.pose[1]:+.2f})  '
                f'target=({self.target[0]:+.2f}, {self.target[1]:+.2f})  '
                f'dist={d:.2f} m  '
                f'mode={self.state.mode}  armed={self.state.armed}  '
                f'connected={self.state.connected}')


def repl(node):
    """stdin loop in a background thread"""
    print('cmds: x y | here | arm | disarm | offboard | posctl | p | q')
    while rclpy.ok():
        try:
            line = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        toks = line.split()
        cmd = toks[0].lower()
        try:
            if cmd == 'q':
                node.call_arm(False)
                node.call_mode('POSCTL')
                rclpy.shutdown()
                break
            elif cmd == 'arm':
                node.call_arm(True)
            elif cmd == 'disarm':
                node.call_arm(False)
            elif cmd == 'offboard':
                node.call_mode('OFFBOARD')
            elif cmd == 'posctl':
                node.call_mode('POSCTL')
            elif cmd == 'here':
                node.snap_here()
            elif cmd == 'p':
                print(node.status_line())
            else:
                # try "x y" or "x y z"
                vals = [float(t) for t in toks]
                if len(vals) == 2:
                    node.set_target(vals[0], vals[1])
                elif len(vals) == 3:
                    node.set_target(vals[0], vals[1], vals[2])
                else:
                    print('unknown command')
        except ValueError:
            print('parse error')


def main(args=None):
    rclpy.init(args=args)
    node = PoseCommander()
    t = threading.Thread(target=repl, args=(node,), daemon=True)
    t.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
