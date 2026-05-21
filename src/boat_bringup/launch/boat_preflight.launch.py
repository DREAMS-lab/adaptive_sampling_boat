#!/usr/bin/env python3
"""
Standalone pre-flight launch — no planner.

Brings up MAVROS + the boat sensors + the sonde adapter + the depth-safety
node + the preflight_checker. Useful on field-day mornings to confirm the
full stack is healthy before committing to a mission.

The preflight checker publishes its outcome as a latched message on
/preflight/passed (Bool) and human-readable text on /preflight/status (String).

  ros2 launch boat_bringup boat_preflight.launch.py
  ros2 launch boat_bringup boat_preflight.launch.py preflight_arm_test:=true
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    args = [
        DeclareLaunchArgument('fcu_url', default_value='/dev/ttyUSB0:921600'),
        DeclareLaunchArgument('start_mavros', default_value='true'),
        DeclareLaunchArgument('start_sensors', default_value='true'),
        DeclareLaunchArgument('preflight_arm_test', default_value='false',
            description='If true, the checker will arm for ~1s. Boat MUST be blocked.'),
        DeclareLaunchArgument('min_safe_depth', default_value='0.5'),
        DeclareLaunchArgument('mount_offset', default_value='0.0'),
        DeclareLaunchArgument('safety_consecutive', default_value='3'),
    ]

    mavros_px4_launch = os.path.join(
        get_package_share_directory('mavros'), 'launch', 'px4.launch'
    )
    mavros_include = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(mavros_px4_launch),
        launch_arguments={
            'fcu_url': LaunchConfiguration('fcu_url'),
            'gcs_url': '',
            'tgt_system': '1',
            'tgt_component': '1',
        }.items(),
        condition=IfCondition(LaunchConfiguration('start_mavros')),
    )

    ping_node = Node(
        package='ping_sonar_ros', executable='ping1d_node', name='ping1d_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_sensors')),
    )
    sonde_node = Node(
        package='sonde_read', executable='read_serial.py', name='sonde_reader',
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_sensors')),
    )

    sonde_adapter = Node(
        package='boat_bringup',
        executable='sonde_field_adapter.py',
        name='sonde_field_adapter',
        output='screen',
        parameters=[{
            'input_topic': 'sonde_data',
            'output_topic': '/gaussian_field/boat/temperature_noisy',
            'column_index': 3,
        }],
    )

    ping_safety = Node(
        package='boat_bringup',
        executable='ping_safety_node.py',
        name='ping_safety_node',
        output='screen',
        parameters=[{
            'min_safe_depth': LaunchConfiguration('min_safe_depth'),
            'mount_offset': LaunchConfiguration('mount_offset'),
            'consecutive': LaunchConfiguration('safety_consecutive'),
        }],
    )

    preflight = Node(
        package='boat_bringup',
        executable='preflight_checker.py',
        name='preflight_checker',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'preflight_arm_test': LaunchConfiguration('preflight_arm_test'),
        }],
    )

    return LaunchDescription(args + [
        mavros_include, ping_node, sonde_node, sonde_adapter, ping_safety, preflight,
    ])
