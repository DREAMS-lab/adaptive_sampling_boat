#!/usr/bin/env python3
"""Bridge /mavros/local_position/pose into a TF frame so RViz can render the rover.

MAVROS has tf.send=false by default (px4_config.yaml). Rather than override it,
we subscribe to the ENU pose and broadcast world -> base_link directly. This is
the same role rover_monitor.py played for the PX4-direct flow.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster


class MavrosTfBridge(Node):
    def __init__(self):
        super().__init__('mavros_tf_bridge')
        self.br = TransformBroadcaster(self)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.sub = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self.cb, qos
        )
        self.get_logger().info('mavros_tf_bridge: world -> base_link from /mavros/local_position/pose')

    def cb(self, msg: PoseStamped):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = 'world'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation
        self.br.sendTransform(t)


def main():
    rclpy.init()
    node = MavrosTfBridge()
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
