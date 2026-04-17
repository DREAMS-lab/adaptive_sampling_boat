"""Launch the full boat mission stack: sonar, MAVROS, sonde, winch, and rosbag record.

Usage:
    ros2 launch boat_control full_mission.launch.py mission_name:=ossabaw_2026_04_17

Config files (parameter sources):
    Boat_control/config/sensors.yaml   # sonar + sonde
    Boat_control/config/winch.yaml     # winch
    Boat_control/config/mission.yaml   # survey area + rosbag base path
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
SENSORS_YAML = str(CONFIG_DIR / "sensors.yaml")
WINCH_YAML = str(CONFIG_DIR / "winch.yaml")


def generate_launch_description() -> LaunchDescription:
    mission_name_arg = DeclareLaunchArgument(
        "mission_name",
        default_value="bench_test",
        description="Short name used in the rosbag directory",
    )
    rosbag_base_arg = DeclareLaunchArgument(
        "rosbag_base",
        default_value="/home/rosbags",
        description="Root directory for rosbag output",
    )
    fcu_url_arg = DeclareLaunchArgument(
        "fcu_url",
        default_value="/dev/boat_fc:921600",
        description="MAVROS flight-controller URL",
    )

    mission_name = LaunchConfiguration("mission_name")
    rosbag_base = LaunchConfiguration("rosbag_base")
    fcu_url = LaunchConfiguration("fcu_url")

    ping1d = Node(
        package="ping_sonar_ros",
        executable="ping1d_node",
        name="ping1d_node",
        parameters=[SENSORS_YAML],
        output="screen",
    )

    mavros = ExecuteProcess(
        cmd=["ros2", "launch", "mavros", "px4.launch", ["fcu_url:=", fcu_url]],
        output="screen",
    )

    sonde = Node(
        package="sonde_read",
        executable="read_serial",
        name="sonde_reader",
        parameters=[SENSORS_YAML],
        output="screen",
    )

    winch = Node(
        package="winch",
        executable="winch",
        name="winch_mission_node",
        parameters=[WINCH_YAML],
        output="screen",
    )

    record_bag = ExecuteProcess(
        cmd=[
            "bash", "-c",
            ["mkdir -p ", rosbag_base, " && ",
             "cd ", rosbag_base, " && ",
             "ros2 bag record -a -s mcap -o ",
             "rosbag2_$(date -u +%Y_%m_%d_%H_%M_%S)_", mission_name],
        ],
        output="screen",
    )

    # Stagger node startup so sonar is publishing before the winch subscribes.
    sonde_delayed = TimerAction(period=2.0, actions=[sonde])
    winch_delayed = TimerAction(period=4.0, actions=[winch])
    mavros_delayed = TimerAction(period=1.0, actions=[mavros])
    recorder_delayed = TimerAction(period=6.0, actions=[record_bag])

    return LaunchDescription([
        mission_name_arg,
        rosbag_base_arg,
        fcu_url_arg,
        ping1d,
        mavros_delayed,
        sonde_delayed,
        winch_delayed,
        recorder_delayed,
    ])
