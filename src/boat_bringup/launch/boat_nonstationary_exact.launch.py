#!/usr/bin/env python3
"""
Non-stationary (Gibbs kernel) planner over MAVROS — REAL BOAT version.

Same scaffolding as boat_exact.launch.py but spawns
`nonstationary_planning/nonstationary_exact_planner_mavros.py` and exposes
the Gibbs-kernel knobs (grid_size, l_min, l_max).
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
        DeclareLaunchArgument('start_winch', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='true'),

        DeclareLaunchArgument('field_origin_x', default_value='0.0'),
        DeclareLaunchArgument('field_origin_y', default_value='0.0'),
        DeclareLaunchArgument('field_size_x', default_value='25.0'),
        DeclareLaunchArgument('field_size_y', default_value='25.0'),
        DeclareLaunchArgument('candidate_resolution', default_value='1.0'),
        DeclareLaunchArgument('candidate_edge_buffer', default_value='0.5'),
        DeclareLaunchArgument('waypoint_tolerance', default_value='1.5'),
        DeclareLaunchArgument('max_samples', default_value='100'),

        DeclareLaunchArgument('trial', default_value='-1'),
        DeclareLaunchArgument('lambda_cost', default_value='0.1'),
        DeclareLaunchArgument('noise_var', default_value='0.36'),
        DeclareLaunchArgument('lengthscale', default_value='2.0'),

        # Gibbs-kernel specific
        DeclareLaunchArgument('grid_size', default_value='5'),
        DeclareLaunchArgument('l_min', default_value='0.5'),
        DeclareLaunchArgument('l_max', default_value='5.0'),
        DeclareLaunchArgument('optimize_every', default_value='10'),
        DeclareLaunchArgument('optimize_steps', default_value='50'),

        DeclareLaunchArgument('min_safe_depth', default_value='0.5'),
        DeclareLaunchArgument('mount_offset', default_value='0.0'),
        DeclareLaunchArgument('safety_consecutive', default_value='3'),

        DeclareLaunchArgument('preflight_arm_test', default_value='false'),
        DeclareLaunchArgument('preflight_timeout', default_value='90.0'),

        DeclareLaunchArgument('publish_recon', default_value='true'),
        DeclareLaunchArgument('recon_temp_min', default_value='10.0'),
        DeclareLaunchArgument('recon_temp_max', default_value='35.0'),

        DeclareLaunchArgument('output_root',
            default_value='/home/blazair/workspaces/boat_adaptive/data'),
    ]

    pkg_share = get_package_share_directory('boat_bringup')
    urdf_path = os.path.join(pkg_share, 'urdf', 'r1_rover.urdf')
    rviz_config = os.path.join(pkg_share, 'config', 'rviz', 'boat.rviz')
    with open(urdf_path, 'r') as f:
        robot_description = f.read()

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

    world_map_static = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='world_to_map',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'map'],
        output='screen',
    )

    tf_bridge = Node(
        package='boat_bringup', executable='mavros_tf_bridge.py',
        name='mavros_tf_bridge', output='screen',
    )

    robot_state_pub = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'publish_frequency': 20.0}],
        output='screen',
    )

    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', rviz_config], output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
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
    winch_node = Node(
        package='winch', executable='roswinch.py', name='winch',
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_winch')),
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

    planner = Node(
        package='nonstationary_planning',
        executable='nonstationary_exact_planner_mavros.py',
        name='nonstationary_exact_planner_mavros',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'field_type': 'boat',
            'temp_topic': '/gaussian_field/boat/temperature_noisy',
            'trial': LaunchConfiguration('trial'),
            'noise_var': LaunchConfiguration('noise_var'),
            'lengthscale': LaunchConfiguration('lengthscale'),
            'lambda_cost': LaunchConfiguration('lambda_cost'),
            'candidate_resolution': LaunchConfiguration('candidate_resolution'),
            'candidate_edge_buffer': LaunchConfiguration('candidate_edge_buffer'),
            'grid_size': LaunchConfiguration('grid_size'),
            'l_min': LaunchConfiguration('l_min'),
            'l_max': LaunchConfiguration('l_max'),
            'optimize_every': LaunchConfiguration('optimize_every'),
            'optimize_steps': LaunchConfiguration('optimize_steps'),
            'field_origin_x': LaunchConfiguration('field_origin_x'),
            'field_origin_y': LaunchConfiguration('field_origin_y'),
            'field_size_x': LaunchConfiguration('field_size_x'),
            'field_size_y': LaunchConfiguration('field_size_y'),
            'waypoint_tolerance': LaunchConfiguration('waypoint_tolerance'),
            'max_samples': LaunchConfiguration('max_samples'),
            'evaluation_enabled': False,
            'publish_recon': LaunchConfiguration('publish_recon'),
            'recon_temp_min': LaunchConfiguration('recon_temp_min'),
            'recon_temp_max': LaunchConfiguration('recon_temp_max'),
            'wait_for_preflight': True,
            'preflight_timeout': LaunchConfiguration('preflight_timeout'),
            'output_root': LaunchConfiguration('output_root'),
        }],
    )

    return LaunchDescription(args + [
        mavros_include,
        world_map_static,
        tf_bridge,
        robot_state_pub,
        rviz_node,
        ping_node,
        sonde_node,
        winch_node,
        sonde_adapter,
        ping_safety,
        preflight,
        planner,
    ])
