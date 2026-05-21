# Boat Control System - ROS 2 Runtime Guide

This guide explains how to run the data collection on the R/V Karin Valentine. All components of the system across multiple terminals.

---

## Prerequisites and System dependencies (ONE-TIME SETUP)

```text
sudo apt update
sudo apt install libusb-1.0-0-dev
```

### FTDI / USB PERMISSIONS (WINCH / FT232H SUPPORT)


Create udev rules file:
```text
sudo nano /etc/udev/rules.d/11-ftdi.rules
```
Add the following:
```text
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6001", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6011", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6010", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6014", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6015", GROUP="plugdev", MODE="0666"
```
Reload rules:
```text
sudo udevadm control --reload-rules
sudo udevadm trigger
```
After setting up the Blinka, you must set up the Python virtual environment:
```text
python3 -m venv ~/ros2venv
source ~/ros2venv/bin/activate
pip install pyftdi adafruit-blinka pyusb
```
In that same terminal, make roswinch.py executable
```text
cd ~/ros2_ws
source install/setup.bash
cd src/winch/winch
chmod +x roswinch.py
```

In another terminal, you should make the updated_record_and_send.sh executable:

```text
chmod +x updated_record_and_send_bag.sh
```
# General Use

###  Ping SonarGeneral Terminal Setup

The first sensor to test is the Ping

```text
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
source install/setup.bash
cd ~/ros2_ws/src/ping_sonar_ros
ros2 run ping_sonar_ros ping1d_node
```
Wait 2 seconds. If you should see:

```text
"Failed to initialize Ping!"
```
That means that you failed to initialize; the most likely culprit is that the wrong port was called. The port order is as follows:

```text
/dev/ttyUSB0 ---> MAVROS
/dev/ttyUSB1 ---> Ping
/dev/ttyUSB2 ---> Sonde
/dev/ttyUSB3 ---> Winch
```
Unplug the cords in order, and plug them in in said order to fix the misplacement.

### MAVROS
In terminal 2, enter the following:
```text
source /opt/ros/jazzy/setup.bash
ros2 launch mavros px4.launch fcu_url:=/dev/ttyUSB0:921600
```
You will see a bunch of topics appear on the screen. After 7 seconds or so, you should see a "IMU Detected" message, and/or "Mission Received"; these are signs that the Pixhawk is talking to the ODROID.

### Sonde Reader

In terminal 3, enter the following:
```text
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
source install/setup.bash
cd ~/ros2_ws/src/sonde_read/scripts
chmod +x read_serial.py
./read_serial.py
```
If successful, you should see data flowing from parameters such as "Temperature", "pH", etc.

### Winch 

In terminal 4, enter the following:
```text
source ~/ros2venv/bin/activate
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
source install/setup.bash
export BLINKA_FT232H=1
cd src/winch/winch
./roswinch.py
```

### ROS2 bag Recording

Finally, in terminal 5, we are able to store all ros2 topics by doing the following:

```text
source /opt/ros/jazzy/setup.bash
./updated_record_and_send_bag.sh
```
