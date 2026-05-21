#!/usr/bin/env python3
"""
Nonstationary (Gibbs kernel) exact planner over MAVROS.

Brings up:
  - mavros_node (MAVLink bridge to PX4 SITL or a real autopilot)
  - The selected temperature field publisher
  - The nonstationary exact planner

Usage:
  ros2 launch boat_bringup nonstationary_exact.launch.py
  ros2 launch boat_bringup nonstationary_exact.launch.py field_type:=y_compress trial:=1
  ros2 launch boat_bringup nonstationary_exact.launch.py fcu_url:=/dev/ttyUSB0:921600

Fields: radial, x_compress, y_compress, x_compress_tilt, y_compress_tilt
"""
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory


def _field_node(exec_name, field_value, field_type):
    return Node(
        package='boat_bringup',
        executable=exec_name,
        name=f'{field_value}_field',
        output='screen',
        condition=IfCondition(PythonExpression(["'", field_type, "' == '", field_value, "'"]))
    )


def generate_launch_description():
    field_type_arg = DeclareLaunchArgument('field_type', default_value='radial')
    trial_arg = DeclareLaunchArgument('trial', default_value='-1')
    lambda_cost_arg = DeclareLaunchArgument('lambda_cost', default_value='0.1')
    noise_var_arg = DeclareLaunchArgument('noise_var', default_value='0.36')
    lengthscale_arg = DeclareLaunchArgument('lengthscale', default_value='2.0')
    candidate_resolution_arg = DeclareLaunchArgument('candidate_resolution', default_value='1.0')
    fcu_url_arg = DeclareLaunchArgument(
        'fcu_url',
        default_value='udp://:14540@127.0.0.1:14580',
        description='MAVLink endpoint. Default targets PX4 SITL offboard port 14580. Boat: e.g. /dev/ttyUSB0:921600'
    )
    start_mavros_arg = DeclareLaunchArgument(
        'start_mavros', default_value='true',
        description='Set to false if you start mavros separately'
    )
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Launch RViz2 with the default config showing field marker + rover model'
    )

    field_type = LaunchConfiguration('field_type')
    trial = LaunchConfiguration('trial')
    lambda_cost = LaunchConfiguration('lambda_cost')
    noise_var = LaunchConfiguration('noise_var')
    lengthscale = LaunchConfiguration('lengthscale')
    candidate_resolution = LaunchConfiguration('candidate_resolution')
    fcu_url = LaunchConfiguration('fcu_url')

    pkg_share = get_package_share_directory('boat_bringup')
    urdf_path = os.path.join(pkg_share, 'urdf', 'r1_rover.urdf')
    rviz_config = os.path.join(pkg_share, 'config', 'rviz', 'default.rviz')
    with open(urdf_path, 'r') as f:
        robot_description = f.read()

    mavros_px4_launch = os.path.join(
        get_package_share_directory('mavros'), 'launch', 'px4.launch'
    )
    mavros_include = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(mavros_px4_launch),
        launch_arguments={
            'fcu_url': fcu_url,
            'gcs_url': '',
            'tgt_system': '1',
            'tgt_component': '1',
        }.items(),
        condition=IfCondition(LaunchConfiguration('start_mavros')),
    )

    world_map_static = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_map',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'map'],
        output='screen',
    )

    tf_bridge = Node(
        package='boat_bringup',
        executable='mavros_tf_bridge.py',
        name='mavros_tf_bridge',
        output='screen',
    )

    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'publish_frequency': 20.0,
        }],
        output='screen',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription([
        field_type_arg, trial_arg, lambda_cost_arg, noise_var_arg,
        lengthscale_arg, candidate_resolution_arg, fcu_url_arg,
        start_mavros_arg, rviz_arg,

        mavros_include,
        world_map_static,
        tf_bridge,
        robot_state_pub,
        rviz_node,

        _field_node('radial_field.py', 'radial', field_type),
        _field_node('x_compress_field.py', 'x_compress', field_type),
        _field_node('y_compress_field.py', 'y_compress', field_type),
        _field_node('x_compress_tilt_field.py', 'x_compress_tilt', field_type),
        _field_node('y_compress_tilt_field.py', 'y_compress_tilt', field_type),

        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='nonstationary_planning',
                    executable='nonstationary_exact_planner_mavros.py',
                    name='nonstationary_exact_planner_mavros',
                    output='screen',
                    emulate_tty=True,
                    parameters=[{
                        'field_type': field_type,
                        'trial': trial,
                        'noise_var': noise_var,
                        'lengthscale': lengthscale,
                        'lambda_cost': lambda_cost,
                        'candidate_resolution': candidate_resolution,
                    }]
                )
            ],
        ),
    ])
