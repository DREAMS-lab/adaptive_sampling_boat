"""CSV writers for the merged and sonar-only mission outputs."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable, List

from .bag_reader import BagData
from .salinity import SalinityCalculator
from .time_align import TimeAligner

log = logging.getLogger(__name__)


MERGED_HEADER = [
    "Time (UTC)", "Latitude", "Longitude", "Depth (Sonar)",
    "Temperature (°C)", "pH", "Depth (m)", "Conductivity (uS/cm)",
    "Dissolved Oxygen Saturation", "Dissolved Oxygen Concentration (mg/L)",
    "Chlorophyll (ug/L)", "CDOM (ppb)", "Turbidity (NTU)", "Salinity (PSU)",
]

SONAR_HEADER = ["Time (UTC)", "Latitude", "Longitude", "Depth (Sonar)"]


def write_merged_csv(
    path: Path,
    data: BagData,
    gps_aligner: TimeAligner,
    sonar_aligner: TimeAligner,
    salinity: SalinityCalculator,
) -> List[float]:
    """Write the sonde-anchored merged CSV. Returns the salinity series."""
    salinity_series: List[float] = []

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(MERGED_HEADER)

        for sonde in data.sonde:
            gps = gps_aligner.find_closest(sonde["timestamp"])
            sonar = sonar_aligner.find_closest(sonde["timestamp"])
            if gps is None or sonar is None:
                continue

            sal_sp, _ = salinity.compute(
                sonde["Depth m"],
                sonde["SpCond uS/cm"],
                sonde["Temp deg C"],
                gps[1], gps[2],
            )
            salinity_series.append(float(sal_sp))

            writer.writerow([
                sonde["timestamp"], gps[1], gps[2], sonar[1],
                sonde["Temp deg C"], sonde["pH units"], sonde["Depth m"],
                sonde["SpCond uS/cm"], sonde["HDO sat"], sonde["HDO mg/L"],
                sonde["Chl ug/L"], sonde["CDOM ppb"],
                sonde.get("Turb NTU", ""),
                sal_sp,
            ])

    log.info("Wrote %s (%d rows)", path, len(data.sonde))
    return salinity_series


def write_sonar_csv(path: Path, data: BagData, gps_aligner: TimeAligner) -> None:
    """Write the sonar-only CSV (higher frequency than the merged one).

    Bug fix vs ros2plot.py: the original iterated sonar_data twice, writing
    every row twice. This version writes each sonar sample exactly once.
    """
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SONAR_HEADER)

        for sonar in data.sonar:
            gps = gps_aligner.find_closest(sonar[0])
            if gps is None:
                continue
            writer.writerow([sonar[0], gps[1], gps[2], sonar[1]])

    log.info("Wrote %s (%d rows)", path, len(data.sonar))
