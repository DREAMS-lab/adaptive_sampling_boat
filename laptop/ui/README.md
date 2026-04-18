# karin — boat operator UI

Desktop UI (PyQt5) for running a boat mission from the laptop. Three tabs:

| Tab | What it does |
|-----|-------------|
| **Vehicle** | MAVROS link status, mode selector, arm/disarm, live position + battery, motor test |
| **Sensors** | Start/stop sonar & sonde on the Odroid, live value display. Read winch GPIO. Move winch (behind a safety checkbox). |
| **Mission** | Positional controller (OFFBOARD setpoint streaming) + lawnmower generator + waypoint upload. |

## One-time setup

### Prerequisites

- Ubuntu 22.04
- ROS2 Humble installed at `/opt/ros/humble`
- `ros-humble-mavros` + `mavros-extras` + `mavros-msgs`
- GeographicLib datasets installed (see repo root README)
- `python3-tk` (only if you want to add Tk-based helpers later; **not required** for Qt)
- `python3-venv`
- Telemetry radio plugged into the laptop at `/dev/ttyUSB0`
- SSH key set up on the Odroid (`ssh-copy-id odroid@192.168.1.101` was run once)

### Create the venv

The UI uses a venv called **karin** at `~/karin/` — kept outside the repo so
`colcon` never tries to build it.

```bash
cd laptop/ui
./setup_karin.sh
```

That creates `~/karin/` with `--system-site-packages` (so rclpy and
`mavros_msgs` from `/opt/ros/humble` stay importable) and installs PyQt5
on top.

## Running

Every time you want to launch the UI:

```bash
# from anywhere
~/workspaces/adaptive_sampling_boat/laptop/ui/run_ui.sh
```

The script sources `/opt/ros/humble/setup.bash` and activates the karin
venv before launching `boat_ui.py`.

**Important:** close QGC first. QGC and MAVROS both need exclusive access to
`/dev/ttyUSB0`. The UI expects `ros2 launch mavros px4.launch
fcu_url:=/dev/ttyUSB0:57600` to be running in a separate terminal.

### Typical mission start sequence

1. Plug telemetry radio into laptop, verify `/dev/ttyUSB0` exists.
2. Close QGC.
3. **Terminal 1:** `source /opt/ros/humble/setup.bash && ros2 launch mavros
   px4.launch fcu_url:=/dev/ttyUSB0:57600`
4. **Terminal 2:** `./run_ui.sh`
5. In the UI's **Vehicle** tab, verify the green link dot.
6. In the **Sensors** tab, click *Start on Odroid* for sonar and sonde. Live
   values appear within a few seconds.
7. In the **Mission** tab:
   - Generate a lawnmower pattern (*Preview pattern*).
   - *Upload to Pixhawk* — waypoints push via `/mavros/mission/push`.
   - *Start mission* — sets `AUTO.MISSION` and arms.

## Files

```
laptop/ui/
├── boat_ui.py              main Qt window + tabs
├── run_ui.sh               venv + ROS2 + launch
├── setup_karin.sh          create ~/karin venv
├── requirements.txt        PyQt5, pyqtgraph
├── lib/
│   ├── ros_bridge.py       rclpy node: subscriptions + service wrappers
│   ├── odroid_ssh.py       SSH start/stop of sensor nodes on Odroid
│   ├── setpoint_streamer.py  20 Hz PoseStamped publisher (for OFFBOARD)
│   ├── sonde_fields.py     Reuses post_processing/lib/sonde_parser
│   ├── lawnmower.py        S-shape waypoint generator
│   └── mission_upload.py   Preview text helper
└── tests/
    └── test_lawnmower.py   pytest
```

## Tests

```bash
source ~/karin/bin/activate
cd laptop/ui
python -m pytest tests/
```

## Troubleshooting

**`rclpy` not found when launching** — make sure `/opt/ros/humble/setup.bash`
is sourced *before* the venv is activated, and that the venv was created with
`--system-site-packages` (run `./setup_karin.sh` again if unsure).

**UI shows "link down"** — MAVROS is not running, or it's running against a
different serial port. Check `ls /dev/ttyUSB*` and the MAVROS log.

**Sensor start does nothing** — test SSH from the laptop:
`ssh odroid@192.168.1.101 'echo ok'`. Should print `ok` without a password.
If not, run `ssh-copy-id odroid@192.168.1.101` first.

**Lawnmower upload fails "Need home position"** — enter home lat/lon manually
(the boat's current position, or the survey site centre), or wait for a GPS fix.
