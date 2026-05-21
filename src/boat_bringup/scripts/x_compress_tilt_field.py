#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

class XCompressTiltFieldGenerator(Node):
    def __init__(self):
        super().__init__('x_compress_tilt_field_generator')
        self.width = 25.0
        self.height = 25.0
        self.resolution = 1.0
        self.center_x = 12.5
        self.center_y = 12.5
        self.base_temperature = 20.0
        self.hotspot_amplitude = 10.0
        self.current_pos = None

        # Noise parameter for sensor measurements (configurable)
        self.declare_parameter('noise_std', 0.6)
        self.noise_std = self.get_parameter('noise_std').value
        self.get_logger().info(f"Measurement noise std: {self.noise_std:.3f}°C")

        qos_viz = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        qos_sensor = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self.marker_pub = self.create_publisher(Marker, '/gaussian_field/x_compress_tilt', qos_viz)
        self.temp_pub = self.create_publisher(Float32, '/gaussian_field/x_compress_tilt/temperature', 10)
        self.temp_noisy_pub = self.create_publisher(Float32, '/gaussian_field/x_compress_tilt/temperature_noisy', 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_callback, qos_sensor)
        self.generate_field()
        self.viz_timer = self.create_timer(1.0, self.publish_visualization)
        self.temp_timer = self.create_timer(0.1, self.publish_temperature)
        self.get_logger().info("X-compress-tilt field initialized")
    def generate_field(self):
        sigma_x, sigma_y = 2.5, 7.0
        theta = np.pi / 4.0
        # Include far edge for consistency with ground_truth.py
        self.x_grid = np.arange(0, self.width + 1e-9, self.resolution)
        self.y_grid = np.arange(0, self.height + 1e-9, self.resolution)
        X, Y = np.meshgrid(self.x_grid, self.y_grid, indexing='xy')
        X_rot = (X - self.center_x) * np.cos(theta) + (Y - self.center_y) * np.sin(theta)
        Y_rot = -(X - self.center_x) * np.sin(theta) + (Y - self.center_y) * np.cos(theta)
        gaussian = np.exp(-(X_rot**2 / (2 * sigma_x**2) + Y_rot**2 / (2 * sigma_y**2)))
        self.temperature_field = self.base_temperature + (self.hotspot_amplitude * gaussian)
        self.X, self.Y = X, Y
        self.temp_min = float(np.min(self.temperature_field))
        self.temp_max = float(np.max(self.temperature_field))
        self.get_logger().info(f"X-compress-tilt field: [{self.temp_min:.2f}, {self.temp_max:.2f}]°C")
    def sample_field(self, x, y):
        if x < 0 or x > self.width or y < 0 or y > self.height:
            return self.base_temperature
        # Reuse stored grids and clamp safely
        i = np.searchsorted(self.x_grid, x) - 1
        j = np.searchsorted(self.y_grid, y) - 1
        if i < 0 or i >= len(self.x_grid)-1 or j < 0 or j >= len(self.y_grid)-1:
            return self.base_temperature
        x0, x1 = self.x_grid[i], self.x_grid[i+1]
        y0, y1 = self.y_grid[j], self.y_grid[j+1]
        wx = (x - x0) / (x1 - x0)
        wy = (y - y0) / (y1 - y0)
        temp = (1-wx)*(1-wy)*self.temperature_field[j,i] + wx*(1-wy)*self.temperature_field[j,i+1] + (1-wx)*wy*self.temperature_field[j+1,i] + wx*wy*self.temperature_field[j+1,i+1]
        return float(temp)
    def pose_callback(self, msg):
        p = msg.pose.position
        self.current_pos = np.array([p.x, p.y, p.z])
    def publish_temperature(self):
        if self.current_pos is None:
            return
        temp = self.sample_field(self.current_pos[0], self.current_pos[1])
        
        # Publish clean temperature
        msg_clean = Float32()
        msg_clean.data = temp
        self.temp_pub.publish(msg_clean)
        
        # Publish noisy temperature
        noisy_temp = temp + np.random.normal(0.0, self.noise_std)
        msg_noisy = Float32()
        msg_noisy.data = noisy_temp
        self.temp_noisy_pub.publish(msg_noisy)
    def publish_visualization(self):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "x_compress_tilt_field"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.scale.x = self.resolution
        marker.scale.y = self.resolution
        marker.scale.z = 0.1
        marker.pose.orientation.w = 1.0
        rows, cols = self.X.shape
        for i in range(rows):
            for j in range(cols):
                temp = self.temperature_field[i, j]
                p = Point()
                p.x, p.y, p.z = float(self.X[i, j]), float(self.Y[i, j]), -0.3
                marker.points.append(p)
                normalized = (temp - self.temp_min) / (self.temp_max - self.temp_min)
                color = ColorRGBA()
                color.a = 0.8
                color.r = normalized
                color.g = 1.0
                color.b = 0.0
                marker.colors.append(color)
        self.marker_pub.publish(marker)
def main(args=None):
    rclpy.init(args=args)
    node = XCompressTiltFieldGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()