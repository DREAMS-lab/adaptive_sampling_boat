# adaptive_sampling_boat

Single repo for the DREAMS Lab autonomous boat. Merges the previous
`Boat_control` (Odroid) and `boat_mission` (laptop) repos with full history
preserved.

## Layout

```
adaptive_sampling_boat/
├── odroid/       Code that runs on the boat's Odroid
│   ├── hardware/    ROS2 packages (ping_sonar_ros, sonde_read, winch)
│   ├── config/      Per-mission YAML parameters
│   ├── launch/      ROS2 launch files (single-command mission start)
│   └── scripts/     Shell helpers, udev rules, mavros submodule
└── laptop/       Code that runs on the operator's laptop
    ├── mission_recorder/    Records rosbags on boat, pulls them to laptop
    ├── post_processing/     Bag -> CSV / map / figure pipeline
    ├── RV Karin Valentine/  CAD models + mission debriefs
    └── _archive/            Old scripts kept for reference
```

See `odroid/README.md` and `laptop/README.md` for the details of each side.

## Getting started

### On the Odroid
```bash
git clone https://github.com/DREAMS-lab/adaptive_sampling_boat.git
cd adaptive_sampling_boat
git submodule update --init --recursive
# one-time setup: see odroid/README.md
# run a mission:  odroid/scripts/start_mission.sh <mission_name>
```

### On the laptop
```bash
git clone https://github.com/DREAMS-lab/adaptive_sampling_boat.git
cd adaptive_sampling_boat/laptop/post_processing
python3 process_mission.py --bag /path/to/rosbag2_YYYY_MM_DD_...
```

## History

Commit history from both previous repos is preserved. Use
`git log --follow <file>` to trace any file back across the reorganisation.
