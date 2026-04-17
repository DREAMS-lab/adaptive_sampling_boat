# Boat Control — Odroid Runtime Guide

Runs on the Odroid on the boat. Starts the sensors, MAVROS, winch, and ROS2
bag recorder in one launch file.

## Layout

```
Boat_control/
├── hardware/            Three ROS2 packages that drive the sensors + winch
│   ├── ping_sonar_ros/  Blue Robotics Ping1D depth sonar driver
│   ├── sonde_read/      YSI EXO multiparameter sonde reader
│   └── winch/           FT232H-driven sample winch with state machine
├── config/              Per-mission parameters, loaded at launch time
│   ├── sensors.yaml     Sonar + sonde (serial ports, rates, device params)
│   ├── winch.yaml       Winch (speeds, limits, sonar offset, spool geometry)
│   └── mission.yaml     Mission name, rosbag base, survey area bounds
├── launch/              Launch files
│   ├── full_mission.launch.py   Whole stack: sensors + MAVROS + winch + bag
│   ├── sensors_only.launch.py   Bench-test subset (sonar + sonde only)
│   └── ping_sonar.launch.py     Sonar + RViz visualization
├── scripts/             Shell helpers and udev rules
│   ├── start_mission.sh
│   ├── stop_mission.sh
│   ├── udev/99-boat-serial.rules  Stable /dev/boat_* symlinks
│   └── external/mavros/           Git submodule
└── README.md
```

## One-time setup on a fresh Odroid

### 1. System dependencies

```bash
sudo apt update
sudo apt install libusb-1.0-0-dev
```

### 2. FTDI USB permissions (for the FT232H winch controller)

```bash
sudo tee /etc/udev/rules.d/11-ftdi.rules <<'EOF'
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6001", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6011", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6010", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6014", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6015", GROUP="plugdev", MODE="0666"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 3. Stable serial port symlinks

See [`scripts/udev/README.md`](scripts/udev/README.md) for the per-device
procedure. The short version: fill in the serial numbers in
`scripts/udev/99-boat-serial.rules`, then:

```bash
sudo cp scripts/udev/99-boat-serial.rules /etc/udev/rules.d/
sudo udevadm control --reload
sudo udevadm trigger
```

After that, `/dev/boat_fc`, `/dev/boat_sonde`, and `/dev/boat_sonar` point
to the correct device regardless of plug order.

### 4. Python venv for the FT232H winch

```bash
python3 -m venv ~/ros2venv
source ~/ros2venv/bin/activate
pip install pyftdi adafruit-blinka pyusb
```

### 5. Build the ROS2 workspace

```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Running a mission

### Single command — recommended

```bash
source ~/ros2venv/bin/activate
source /opt/ros/<ros2_version>/setup.bash
source ~/ros2_ws/install/setup.bash

cd ~/ros2_ws/src/Boat_control
./scripts/start_mission.sh ossabaw_2026_04_17
```

This:
1. Exports `BLINKA_FT232H=1` for the winch
2. Launches sonar → MAVROS → sonde → winch → bag recorder (staggered)
3. Loads parameters from `config/sensors.yaml` + `config/winch.yaml`
4. Records a bag to `/home/rosbags/rosbag2_<UTC>_ossabaw_2026_04_17`

To stop:

```bash
Ctrl-C in the launch terminal
./scripts/stop_mission.sh    # cleanup any stragglers
```

### Options

```bash
./scripts/start_mission.sh ossabaw_2026_04_17 \
    rosbag_base:=/mnt/big_drive/rosbags \
    fcu_url:=/dev/boat_fc:921600
```

### Manual per-node launch (for debugging)

Each node can still be started individually — useful when one sensor
is misbehaving and you want it in the foreground:

```bash
ros2 run ping_sonar_ros ping1d_node --ros-args --params-file config/sensors.yaml
ros2 run sonde_read read_serial --ros-args --params-file config/sensors.yaml
ros2 run winch winch --ros-args --params-file config/winch.yaml
ros2 launch mavros px4.launch fcu_url:=/dev/boat_fc:921600
```

## Changing parameters

Every tunable lives in `config/*.yaml`. Edit the YAML, save, restart the stack
— no code edit needed. Examples:

- **Sonar depth offset** (was 0.1651 m hardcoded) → `config/winch.yaml:sonar_shift`
- **Max winch speed** → `config/winch.yaml:max_speed`
- **Sonde serial port** → `config/sensors.yaml:sonde.port`
- **Ping gain** → `config/sensors.yaml:ping1d_node.gain_num`

## After the mission

The rosbag lives at `/home/rosbags/`. Transfer it to the laptop with
`boat_mission/mission_recorder/record_bag.sh` or by SCP'ing directly.
Then post-process with `boat_mission/post_processing/process_mission.py`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| MAVROS silent, sonde logs garbage | `/dev/ttyUSB0` is the sonde, not the Pixhawk | Install the udev rules (step 3 above) |
| Winch homing forever | FT232H not detected | `export BLINKA_FT232H=1` before launch, and check `lsusb` for FTDI |
| Sonar node exits at startup | Sonar on wrong port | Check `config/sensors.yaml:ping1d_node.port`, verify with `ls -l /dev/boat_sonar` |
| Battery readings absent from winch | MAVROS not publishing | Check `/mavros/battery` is actually publishing; some autopilots need `SYS_STATUS` streamed |
