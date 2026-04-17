#!/usr/bin/env python3
"""Process a boat mission rosbag into CSVs, interactive map, and figures.

This replaces the old ros2plot.py monolith. Behaviour is identical except:
  - The duplicate-sonar-row bug is fixed (each sonar sample appears once).
  - NaN salinity values are still zero-filled but are now counted and reported.
  - Timestamp-gap warnings surface clock sync issues that used to be silent.
  - No interactive prompts — everything comes from CLI args or mission_config.yaml.

Usage:
    python process_mission.py --bag /path/to/rosbag2_2025_10_30_...
    python process_mission.py --bag <path> --sonar-offset -0.15 --output-dir out/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import rclpy
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.bag_reader import read_bag
from lib.plotting import make_folium_map, make_sonar_map, make_sonde_grid
from lib.salinity import SalinityCalculator
from lib.time_align import TimeAligner
from lib.writers import write_merged_csv, write_sonar_csv


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "mission_config.yaml"


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def mission_title(bag_path: Path, sonde_records) -> str:
    if sonde_records:
        first = sonde_records[0]
        date = first.get("Date", "unknown").replace("/", "-")
        time = first.get("Time", "unknown").replace(":", "-")
        return f"{date}_{time}"
    return f"{bag_path.name}_unknown_sonde_time"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bag", required=True, type=Path, help="Path to the rosbag directory")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to mission_config.yaml")
    p.add_argument("--sonar-offset", type=float, help="Override sonar_offset from config")
    p.add_argument("--output-dir", type=Path, help="Override output_dir from config")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not (args.bag / "metadata.yaml").exists():
        logging.error("Bag path missing metadata.yaml: %s", args.bag)
        return 1

    cfg = load_config(args.config)
    sonar_offset = args.sonar_offset if args.sonar_offset is not None else cfg["sonar_offset"]
    output_dir = args.output_dir or Path(cfg.get("output_dir", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    try:
        data = read_bag(
            bag_path=str(args.bag),
            gps_topic=cfg["topics"]["gps"],
            sonar_topic=cfg["topics"]["sonar"],
            sonde_topic=cfg["topics"]["sonde"],
            sonar_offset=sonar_offset,
        )
    finally:
        if rclpy.ok():
            rclpy.shutdown()

    max_gap = cfg.get("max_time_gap_seconds")
    gps_aligner = TimeAligner(data.gps, max_gap_seconds=max_gap)
    sonar_aligner = TimeAligner(data.sonar, max_gap_seconds=max_gap)
    salinity = SalinityCalculator()

    title = mission_title(args.bag, data.sonde)

    salinity_series = write_merged_csv(
        output_dir / f"{title}_raw_data.csv",
        data, gps_aligner, sonar_aligner, salinity,
    )
    write_sonar_csv(output_dir / f"{title}_sonar_raw_data.csv", data, gps_aligner)
    make_folium_map(
        output_dir / f"{title}_lake_depth_sonde_map.html",
        data, gps_aligner, sonar_aligner, salinity_series, salinity,
    )
    make_sonde_grid(
        output_dir / f"{title}_Sonde_Water_Quality_vs_GPS_Coordinates.png",
        data, gps_aligner, salinity,
    )
    make_sonar_map(
        output_dir / f"{title}_Depth_vs_GPS_Coordinates.png",
        data, gps_aligner,
    )

    logging.info(salinity.summary())
    logging.info("Outputs written to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
