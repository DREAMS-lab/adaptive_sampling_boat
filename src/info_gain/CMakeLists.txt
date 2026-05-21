cmake_minimum_required(VERSION 3.8)
project(info_gain)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)

install(DIRECTORY
  info_gain
  DESTINATION lib/python3/dist-packages/
)

install(PROGRAMS
  scripts/exact_planner_mavros.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
