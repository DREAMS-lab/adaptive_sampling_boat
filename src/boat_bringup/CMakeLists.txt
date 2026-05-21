cmake_minimum_required(VERSION 3.8)
project(boat_bringup)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

install(PROGRAMS
  scripts/radial_field.py
  scripts/x_compress_field.py
  scripts/y_compress_field.py
  scripts/x_compress_tilt_field.py
  scripts/y_compress_tilt_field.py
  scripts/mavros_tf_bridge.py
  scripts/sonde_field_adapter.py
  scripts/ping_safety_node.py
  scripts/preflight_checker.py
  scripts/pose_commander.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  urdf
  config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
