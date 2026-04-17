Boat Control System - ROS 2 Runtime Guide
==========================================

This guide explains how to run all components of the system across multiple terminals.

------------------------------------------------------------
PREREQUISITES (ONE-TIME SETUP)
------------------------------------------------------------

System dependencies:

sudo apt update
sudo apt install libusb-1.0-0-dev

------------------------------------------------------------
FTDI / USB PERMISSIONS (WINCH / FT232H SUPPORT)
------------------------------------------------------------

Create udev rules file:

sudo nano /etc/udev/rules.d/11-ftdi.rules

Add the following:

SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6001", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6011", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6010", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6014", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6015", GROUP="plugdev", MODE="0666"

Reload rules:

sudo udevadm control --reload-rules
sudo udevadm trigger

------------------------------------------------------------
PYTHON VIRTUAL ENVIRONMENT (FT232H / BLINKA)
------------------------------------------------------------

python3 -m venv ~/ros2venv
source ~/ros2venv/bin/activate

pip install pyftdi adafruit-blinka pyusb

------------------------------------------------------------
GENERAL TERMINAL SETUP (EVERY TERMINAL)
------------------------------------------------------------

source /opt/ros/<ROS2_version>/setup.bash
cd ~/ros2_ws
source install/setup.bash

============================================================
TERMINAL 1 - PING SONAR (Ping1D)
============================================================

cd ~/ros2_ws/src/Boat_control/scripts/ping_sonar_ros
ros2 run ping_sonar_ros ping1d_node

============================================================
TERMINAL 2 - MAVROS
============================================================

ros2 launch mavros px4.launch fcu_url:=/dev/ttyUSB0:921600

============================================================
TERMINAL 3 - SONDE READER
============================================================

cd ~/ros2_ws/src/Boat_control/scripts/sonde_read/scripts

chmod +x read_serial.py
./read_serial.py

============================================================
TERMINAL 4 - WINCH (FT232H / BLINKA)
============================================================

source ~/ros2venv/bin/activate

source /opt/ros/<ROS2_version>/setup.bash
cd ~/ros2_ws
source install/setup.bash

export BLINKA_FT232H=1

cd src/Boat_control/scripts/winch/winch

chmod +x roswinch.py
./roswinch.py

============================================================
TERMINAL 5 - ROS 2 BAG RECORDING
============================================================

chmod +x updated_record_and_send_bag.sh

source /opt/ros/<ROS2_version>/setup.bash
./updated_record_and_send_bag.sh

============================================================
NOTES
============================================================

- Always source ROS in every new terminal
- Activate ros2venv only for FT232H / winch usage
- Ensure /dev/ttyUSB0 is correct for MAVROS
- If devices fail, check permissions and udev rules
