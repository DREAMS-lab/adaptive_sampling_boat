# post_processing

Turns a boat mission rosbag into CSVs, an interactive map, and figures.

## Quick start

```bash
python3 process_mission.py --bag /path/to/rosbag2_YYYY_MM_DD_...
```

Outputs go to `./output/`. See the per-mission knobs in
`config/mission_config.yaml` (sonar offset, topic names, timestamp gap warning).

## Module map

```
process_mission.py          CLI entry point. Orchestrates the pipeline.
config/mission_config.yaml  Per-run knobs.
lib/
  bag_reader.py             MCAP rosbag → (gps, sonar, sonde) streams
  sonde_parser.py           Parses one #DATA line from the YSI EXO sonde
  time_align.py             Nearest-neighbour timestamp matcher + warnings
  salinity.py               GSW TEOS-10 salinity calc + NaN reporting
  writers.py                CSV writers (merged + sonar-only)
  plotting.py               Folium map + matplotlib figures
tests/
  test_time_align.py        7 sanity checks on the timestamp matcher
```

## Adding a new figure without breaking anything

1. Write a new function in `lib/plotting.py` that takes `(path, data, ...)`
2. Call it from `process_mission.py` after the existing plots
3. Add a row to the output table in `../README.md`

No need to touch `bag_reader.py`, `sonde_parser.py`, etc. — the streams are
already materialised into `BagData` by the time plotting runs.

## Adding a new CSV column

1. Add the column to the header list in `lib/writers.py` (`MERGED_HEADER`)
2. Add the value to the row in `write_merged_csv()` — usually derived from
   one of the sonde fields
3. If the value is expensive to compute, put the maths in its own module
   (`lib/<feature>.py`) and import it

## Changing topic names

They live in `config/mission_config.yaml:topics`. Don't hardcode them elsewhere.

## Verifying against the old script

The original `ros2plot.py` is kept at `_ros2plot_original.py` for reference.
Both should produce identical CSVs apart from:

- The sonar CSV has *half* the rows in the new version — the old one wrote
  each sonar sample twice (see `writers.py` comment).
- The new log shows a "salinity: N/M samples were NaN..." line at the end
  whenever the GSW calculation returned NaN.

Run both on the same rosbag, then compare:

```bash
# old
cd $(mktemp -d) && echo "$BAG_PATH" | python3 /path/to/_ros2plot_original.py

# new
python3 process_mission.py --bag "$BAG_PATH" --output-dir /tmp/new
diff <(head -5 /tmp/old_merged.csv) <(head -5 /tmp/new_merged.csv)
```
