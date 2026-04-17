"""Salinity computation via GSW (TEOS-10 oceanographic standard)."""

from __future__ import annotations

import logging
from typing import Tuple

import gsw
import numpy as np

log = logging.getLogger(__name__)


class SalinityCalculator:
    """Compute practical + absolute salinity and track NaN-fill statistics.

    The old code silently replaced NaN values with 0. We keep the zero-fill so
    the CSV format stays identical, but report the count at the end so the
    operator can tell when the sonde's conductivity/temperature inputs were
    out of GSW's valid range (e.g. in very shallow water).
    """

    def __init__(self) -> None:
        self.nan_count = 0
        self.total_count = 0

    def compute(
        self,
        depth_m: float,
        conductivity_us_cm: float,
        temperature_c: float,
        latitude_deg: float,
        longitude_deg: float,
    ) -> Tuple[float, float]:
        depth_m = float(depth_m)
        con = float(conductivity_us_cm)
        temp = float(temperature_c)
        lat = float(latitude_deg)
        lon = float(longitude_deg)

        pressure = gsw.p_from_z(depth_m, lat)
        con_ms = con / 1000
        SP = gsw.SP_from_C(con_ms, temp, pressure)
        SA = gsw.SA_from_SP(SP, pressure, lon, lat)

        self.total_count += 1
        nan_this = int(np.isnan(SP)) + int(np.isnan(SA))
        if nan_this:
            self.nan_count += 1
        if np.isnan(SP):
            SP = 0.0
        if np.isnan(SA):
            SA = 0.0
        return SP, SA

    def summary(self) -> str:
        if self.total_count == 0:
            return "Salinity: no samples computed."
        pct = 100.0 * self.nan_count / self.total_count
        return (f"Salinity: {self.nan_count}/{self.total_count} samples "
                f"({pct:.1f}%) produced NaN and were replaced with 0.")


def pressure_from_depth(depth_m, con, temp, lat_deg, lon_deg):
    """Backwards-compatible shim for the old free function."""
    return SalinityCalculator().compute(depth_m, con, temp, lat_deg, lon_deg)
