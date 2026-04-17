# boat_mission — Laptop Side

Workflow for running a boat mission from the laptop: record the rosbag on the
Odroid, transfer it to the laptop, then turn it into CSVs / maps / figures.

## Layout

```
boat_mission/
├── RV Karin Valentine/
│   ├── CAD Models/         Hardware drawings (unchanged)
│   └── debriefs/           One markdown debrief per mission day
├── mission_recorder/       Records the rosbag on the boat, SCPs to laptop
│   ├── record_bag.sh
│   └── config.yaml         Laptop/boat hostnames, paths (no more hardcoded IPs)
├── post_processing/        Bag → CSV / map / figure pipeline (was ros2plot.py)
│   ├── process_mission.py  CLI entry point
│   ├── config/mission_config.yaml
│   ├── lib/                Split modules
│   └── tests/              pytest
└── _archive/               Old scripts we no longer run, kept for reference
```

## Workflow

### 1. Record the mission

On the boat Odroid, run the boat's launch file to start sensors and bag recording
(see `Boat_control/README.md`). When the mission is done, transfer the bag to
the laptop with:

```bash
# On the boat
cd ~/ros2_ws/src/boat_mission/mission_recorder
./record_bag.sh
```

The script reads `config.yaml` for the laptop hostname and destination path.
Override any value with an env var, e.g. `LOCAL_HOST=10.0.0.5 ./record_bag.sh`.

### 2. Post-process on the laptop

```bash
cd boat_mission/post_processing

# Edit config/mission_config.yaml to set sonar_offset for this mission,
# then:
python3 process_mission.py --bag ~/boat/boat_results/mission_bags/rosbag2_2026_04_17_...

# Or set the offset on the command line:
python3 process_mission.py --bag <path> --sonar-offset -0.15
```

Outputs (one set per rosbag) land in `./output/`:

| File | Contents |
|------|----------|
| `*_raw_data.csv` | One row per sonde sample: lat, lon, sonar depth, all 9 sonde channels, salinity |
| `*_sonar_raw_data.csv` | Higher-frequency sonar-only data with GPS |
| `*_lake_depth_sonde_map.html` | Interactive Folium map with per-parameter layers |
| `*_Sonde_Water_Quality_vs_GPS_Coordinates.png` | 3×4 grid of sonde parameters over lat/lon |
| `*_Depth_vs_GPS_Coordinates.png` | Sonar bathymetry map |

### 3. Write the debrief

Copy a previous debrief from `RV Karin Valentine/debriefs/` into a new
`<date>/debrief.md`, then fill in objectives, participants, issues, and
embed the PNG figures from post-processing.

## Tests

```bash
cd post_processing
python3 -m pytest
```

## What changed vs the old ros2plot.py

Same numerical output — this is a reorganisation, not a rewrite.

- `ros2plot.py` (696 lines) split into `lib/bag_reader.py`, `sonde_parser.py`,
  `time_align.py`, `salinity.py`, `writers.py`, `plotting.py`
- Sonar offset no longer an interactive prompt — set in `mission_config.yaml`
  or via `--sonar-offset`
- Sonar CSV rows are written once (was twice — silent duplicate rows)
- NaN salinity values still zero-filled but now counted in the log
- Large timestamp gaps now log a warning instead of matching silently
- Unused imports dropped (`plotly`, `Axes3D`, `griddata`, `HeatMap`)
- Topics moved from hardcoded strings into `mission_config.yaml:topics`

The original script is preserved at `post_processing/_ros2plot_original.py`
for byte-for-byte verification against new outputs.
