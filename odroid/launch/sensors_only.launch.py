"""Launch sensors only (sonar + sonde) for bench testing.

No MAVROS, no winch, no rosbag. Useful for verifying serial ports and data flow
before putting the boat in the water.
"""

from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
SENSORS_YAML = str(CONFIG_DIR / "sensors.yaml")


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        Node(
            package="ping_sonar_ros",
            executable="ping1d_node",
            name="ping1d_node",
            parameters=[SENSORS_YAML],
            output="screen",
        ),
        Node(
            package="sonde_read",
            executable="read_serial",
            name="sonde_reader",
            parameters=[SENSORS_YAML],
            output="screen",
        ),
    ])
