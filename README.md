# boat_adaptive

ROS 2 workspace for adaptive sampling on the R/V Karin Valentine. The laptop
runs the GP planner and MAVROS over the telemetry radio (or WiFi to the
on-boat Odroid); the boat publishes pose + sonde + ping; the planner sends
setpoints back. Originally ported from the SITL workspace at
`~/workspaces/aquatic_mapping_mavros/`.

## Layout

```
boat_adaptive/
├── README.md                                  ← this file
├── docs/
│   └── boat_hardware.md                       ← udev rules, USB port order, sensor wiring
└── src/
    ├── boat_bringup/                          ← launches, URDF, RViz, glue nodes
    │   ├── launch/
    │   │   ├── exact.launch.py                    ← SITL (stationary planner)
    │   │   ├── nonstationary_exact.launch.py      ← SITL (Gibbs kernel)
    │   │   ├── boat_exact.launch.py               ← real boat (stationary)
    │   │   ├── boat_nonstationary_exact.launch.py ← real boat (Gibbs kernel)
    │   │   └── boat_preflight.launch.py           ← pre-flight only, no planner
    │   ├── scripts/
    │   │   ├── radial_field.py …               ← SITL simulated fields
    │   │   ├── mavros_tf_bridge.py             ← /mavros pose → TF
    │   │   ├── sonde_field_adapter.py          ← sonde_data → /gaussian_field/boat/temperature_noisy
    │   │   ├── ping_safety_node.py             ← shallow-depth auto-loiter
    │   │   └── preflight_checker.py            ← pre-mission health checks
    │   ├── urdf/r1_rover.urdf
    │   └── config/rviz/{default,boat}.rviz
    ├── info_gain/                             ← stationary RBF planner
    ├── nonstationary_planning/                ← Gibbs-kernel planner
    ├── ping_sonar_ros/                        ← Ping1D depth sensor driver
    ├── sonde_read/                            ← YSI ProDSS sonde driver
    └── winch/                                 ← FT232H winch controller (off in v1)
```

Six ROS 2 packages, one workspace, one `colcon build`.

## Build

```bash
cd ~/workspaces/boat_adaptive
colcon build --symlink-install
source install/setup.bash
```

Pre-reqs: ROS 2 Jazzy, `ros-jazzy-mavros`, GeographicLib datasets. See
[`docs/boat_hardware.md`](docs/boat_hardware.md) for the FTDI / udev / USB
setup that the sensor drivers depend on. Never use `--break-system-packages`
for Python deps — venv only.

## Run the SITL sim

Bench-test the planner without the boat:

```bash
# Terminal 1
cd ~/PX4-Autopilot && make px4_sitl gz_r1_rover

# Terminal 2
cd ~/workspaces/boat_adaptive && source install/setup.bash
ros2 launch boat_bringup exact.launch.py field_type:=radial trial:=1
```

Outputs land in `data/trials/exact/radial/trial_NNN/`.

## Real-boat workflow

### 1. Field-day pre-flight (always run this first)

```bash
ros2 launch boat_bringup boat_preflight.launch.py
```

Brings up MAVROS + ping1d + sonde + the adapter + safety node + the
preflight checker. Watch `/preflight/status` (String). On full pass,
`/preflight/passed` latches to `Bool(True)`. Checks:

1. `/mavros/state` connected, healthy
2. `/mavros/local_position/pose` publishing
3. `sonde_data` publishing, temperature parses sane
4. `/ping1d/data` publishing, depth plausible
5. Setpoint round-trip + OFFBOARD toggle (no arm)

#### Optional ARM dry-run — DANGEROUS

```bash
ros2 launch boat_bringup boat_preflight.launch.py preflight_arm_test:=true
```

Prompts at the terminal, then arms for ~1 s and disarms. **Props must be
physically removed or the boat blocked / out of water.** Default is off.

### 2. Mission

Stationary planner:

```bash
ros2 launch boat_bringup boat_exact.launch.py \
    field_origin_x:=0 field_origin_y:=0 \
    field_size_x:=25 field_size_y:=25 \
    max_samples:=100
```

Non-stationary (Gibbs kernel):

```bash
ros2 launch boat_bringup boat_nonstationary_exact.launch.py \
    field_origin_x:=0 field_origin_y:=0 \
    field_size_x:=25 field_size_y:=25 \
    max_samples:=100
```

Launch graph: MAVROS → TF bridge → RViz → sensors → sonde adapter →
ping-depth safety → preflight checker → planner. The planner blocks in
`WAIT_PREFLIGHT` until `/preflight/passed` is `True`, then runs the
offboard handshake itself (stream setpoints → set OFFBOARD → arm), visits
three fractional initial waypoints inside the configured box, then runs
the greedy MI acquisition until `max_samples` is reached.

### Sampling box

The box is anchored at the MAVROS local ENU origin (set by PX4 EKF at
power-on / first GPS fix). `field_origin_{x,y}` shifts the box relative
to that origin; `field_size_{x,y}` sets its size in meters. For repeatable
trials launch from the same shore reference.

### Waypoint tolerance

`waypoint_tolerance:=1.5` (boat default). The planner considers a waypoint
reached once it's within this many meters AND a fresh sonde reading is
available. 1.5 m accounts for drift in wind/current.

### Live reconstruction in RViz

`/info_gain/reconstruction` (MarkerArray, CUBE_LIST). One cube per
candidate cell, colored by GP mean temperature, alpha proportional to
confidence (1 − normalized variance). Updates every sample. Tune
`recon_temp_min` / `recon_temp_max` if your expected range is far from
10–35 °C.

### Outputs

`data/trials/<planner>/boat/trial_NNN/`:

- `samples.csv`, `decisions.csv`, `decisions.json`
- `figures/progress.png` (updated each step)
- `summary.json`
- `lengthscale/` (NS planner only) — per-step Gibbs kernel snapshots
- Kac-Rice hotspot analysis runs on the final GP posterior (no ground
  truth needed)

## Operator-side notes

- **RC override is always live.** Flip the mode switch on the transmitter
  and the pilot takes over instantly. This is the primary safety layer.
- **Kill switch** overrides everything; use it if anything goes wrong.
- **QGC** can run on the same laptop in parallel for params, GPS, battery,
  and mode monitoring. PX4 supports multiple GCS clients. QGC is not
  required for the mission.
- **Ping depth safety:** if depth < `min_safe_depth + mount_offset` for
  `safety_consecutive` consecutive samples, `ping_safety_node` calls
  `SetMode → AUTO.LOITER` and latches `SHALLOW_HOLD` on
  `/boat_safety/state`. Does not auto-disarm or auto-recover. Calibrate
  `mount_offset` at the dock with a known-depth pole.
- **Winch** is off by default. The `winch` package's node listens for
  `/mavros/mission/reached`, which OFFBOARD setpoints don't emit. Pilot
  raises / lowers manually in v1.

## SITL parity (sanity check before any field run)

The SITL launches still work in this workspace:

```bash
ros2 launch boat_bringup exact.launch.py field_type:=radial trial:=1
```

Use this to confirm a build hasn't broken baseline behavior before
committing to a real-water trial.
