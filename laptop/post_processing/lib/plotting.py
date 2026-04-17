"""Folium interactive map + matplotlib static figures for mission data."""

from __future__ import annotations

import logging
import statistics
from pathlib import Path
from typing import List

import branca.colormap as cm
import folium
from folium import Element
import matplotlib.pyplot as plt
import numpy as np

from .bag_reader import BagData
from .salinity import SalinityCalculator
from .sonde_parser import has_turbidity
from .time_align import TimeAligner

log = logging.getLogger(__name__)


def _colormap(values, palette):
    if not values:
        return None
    return palette.scale(min(values), max(values))


def make_folium_map(
    path: Path,
    data: BagData,
    gps_aligner: TimeAligner,
    sonar_aligner: TimeAligner,
    salinity_series: List[float],
    salinity: SalinityCalculator,
) -> None:
    """Produce the interactive HTML map with per-parameter layers."""
    if not data.gps:
        log.warning("No GPS data — skipping folium map.")
        return

    mean_lat = sum(lat for _, lat, _ in data.gps) / len(data.gps)
    mean_lon = sum(lon for _, _, lon in data.gps) / len(data.gps)
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=15, control_scale=True)

    zoom_position = Element("""
    <script>
        var map = document.getElementsByClassName('leaflet-container')[0]._leaflet_map;
        map.zoomControl.setPosition('bottomright');
    </script>
    """)
    m.get_root().html.add_child(zoom_position)

    def floats(key):
        return [float(s[key]) for s in data.sonde if key in s]

    colormaps = {
        "Temp deg C":    _colormap(floats("Temp deg C"),    cm.linear.YlOrRd_09),
        "pH units":      _colormap(floats("pH units"),      cm.linear.PuBu_09),
        "SpCond uS/cm":  _colormap(floats("SpCond uS/cm"),  cm.linear.Oranges_09),
        "HDO mg/L":      _colormap(floats("HDO mg/L"),      cm.linear.Blues_09),
        "Chl ug/L":      _colormap(floats("Chl ug/L"),      cm.linear.Greens_09),
        "CDOM ppb":      _colormap(floats("CDOM ppb"),      cm.linear.Reds_09),
        "Sonar Depth":   _colormap([float(s[1]) for s in data.sonar], cm.linear.Purples_09),
        "Salinity (PSU)": _colormap(salinity_series,        cm.linear.BuGn_09),
    }
    if any(has_turbidity(s) for s in data.sonde):
        colormaps["Turb NTU"] = _colormap(floats("Turb NTU"), cm.linear.Greys_09)

    # Drop params that had no data
    colormaps = {k: v for k, v in colormaps.items() if v is not None}

    layers = {p: folium.FeatureGroup(name=p).add_to(m) for p in colormaps}
    param_stats = {p: [] for p in colormaps}

    for sonde in data.sonde:
        gps = gps_aligner.find_closest(sonde["timestamp"])
        sonar = sonar_aligner.find_closest(sonde["timestamp"])
        if gps is None or sonar is None:
            continue
        lat, lon = gps[1], gps[2]
        depth = float(sonar[1])
        sal_sp, _ = salinity.compute(
            sonde["Depth m"], sonde["SpCond uS/cm"], sonde["Temp deg C"], lat, lon,
        )

        for param, colormap in colormaps.items():
            if param == "Sonar Depth":
                value = depth
            elif param == "Salinity (PSU)":
                value = sal_sp
            elif param in sonde:
                value = float(sonde[param])
            else:
                continue
            param_stats[param].append((value, lat, lon))
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color=colormap(value),
                fill=True,
                fill_color=colormap(value),
                fill_opacity=0.7,
                popup=f"{param}: {value}",
            ).add_to(layers[param])

    sidebar_stats = ""
    for param, values in param_stats.items():
        if not values:
            continue
        vals, lats, lons = zip(*values)
        mean_val = np.mean(vals)
        min_val = min(vals)
        max_val = max(vals)
        try:
            _ = statistics.mode(vals)
        except statistics.StatisticsError:
            pass  # Mode was unused downstream; just skip on tie.

        def idx_of(target):
            return min(range(len(vals)), key=lambda i: abs(vals[i] - target))

        for tag, color, target in (("Mean", "blue", mean_val),
                                   ("Min", "green", min_val),
                                   ("Max", "red", max_val)):
            i = idx_of(target)
            folium.Marker(
                location=(lats[i], lons[i]),
                icon=folium.Icon(color=color, icon="info-sign"),
                popup=f"{param} {tag}: {target:.2f}",
            ).add_to(layers[param])

        sidebar_stats += f"""
            <div style='margin-bottom:10px'>
                <b>{param}</b><br>
                <span style='color:blue'>Mean: {mean_val:.2f}</span><br>
                <span style='color:green'>Min: {min_val:.2f}</span><br>
                <span style='color:red'>Max: {max_val:.2f}</span>
            </div>
        """

    m.get_root().html.add_child(Element(f"""
    <div style="position: fixed; top: 55%; left: 8px; transform: translateY(-50%);
                background-color: white; padding: 9px; border: 2px solid gray;
                border-radius: 10px; max-height: 80%; overflow-y: auto;
                z-index: 9999; font-size: 13px;">
        <h4 style='margin:0 0 10px 0'>Stats</h4>
        {sidebar_stats}
    </div>
    """))

    for param, colormap in colormaps.items():
        colormap.caption = param
        colormap.add_to(m)
    folium.LayerControl().add_to(m)

    m.save(str(path))
    log.info("Wrote %s", path)


def make_sonde_grid(path: Path, data: BagData, gps_aligner: TimeAligner,
                    salinity: SalinityCalculator) -> None:
    """3x4 matplotlib grid of sonde parameters over lat/lon."""
    if not data.sonde:
        log.warning("No sonde data — skipping sonde grid figure.")
        return

    series = {k: [] for k in (
        "Temp deg C", "pH units", "Depth m", "SpCond uS/cm",
        "HDO sat", "HDO mg/L", "Chl ug/L", "CDOM ppb", "Turb NTU",
    )}
    salt: List[float] = []
    lats: List[float] = []
    lons: List[float] = []

    for sonde in data.sonde:
        gps = gps_aligner.find_closest(sonde["timestamp"])
        if gps is None:
            continue
        lats.append(gps[1])
        lons.append(gps[2])
        sal_sp, _ = salinity.compute(
            sonde["Depth m"], sonde["SpCond uS/cm"], sonde["Temp deg C"], gps[1], gps[2],
        )
        salt.append(float(sal_sp))
        for key in series:
            if key in sonde:
                try:
                    series[key].append(float(sonde[key]))
                except ValueError:
                    series[key].append(np.nan)
            else:
                series[key].append(np.nan)

    labels = [
        "Temperature (°C)", "pH", "Depth (m)", "Conductivity (uS/cm)",
        "Dissolved Oxygen Saturation", "Dissolved Oxygen Concentration (mg/L)",
        "Chlorophyll (ug/L)", "CDOM (ppb)", "Turbidity (NTU)", "Salinity (PSU)",
    ]
    data_series = [
        series["Temp deg C"], series["pH units"], series["Depth m"], series["SpCond uS/cm"],
        series["HDO sat"], series["HDO mg/L"], series["Chl ug/L"], series["CDOM ppb"],
        series["Turb NTU"], salt,
    ]

    fig, axes = plt.subplots(3, 4, figsize=(12, 10))
    axes_flat = axes.flatten()

    for ax, label, vals in zip(axes_flat, labels, data_series):
        if not vals or all(np.isnan(v) for v in vals):
            ax.axis("off")
            continue
        scatter = ax.scatter(lons, lats, c=vals, cmap="viridis", s=1)
        ax.set_title(label, fontsize=6)
        ax.set_xlabel("Longitude", fontsize=6)
        ax.set_ylabel("Latitude", fontsize=6)
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.set_xlim([min(lons) - 0.0001, max(lons) + 0.0001])
        ax.set_ylim([min(lats) - 0.0001, max(lats) + 0.0001])
        ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both", useMathText=True)
        ax.xaxis.get_offset_text().set_fontsize(5)
        ax.yaxis.get_offset_text().set_fontsize(5)
        fig.colorbar(scatter, ax=ax).ax.tick_params(labelsize=6)

    for ax in axes_flat[len(labels):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, format="png")
    plt.close(fig)
    log.info("Wrote %s", path)


def make_sonar_map(path: Path, data: BagData, gps_aligner: TimeAligner) -> None:
    """Single scatter plot of sonar depth vs lat/lon."""
    if not data.sonar:
        log.warning("No sonar data — skipping sonar map figure.")
        return

    depths: List[float] = []
    lats: List[float] = []
    lons: List[float] = []
    for sonar in data.sonar:
        gps = gps_aligner.find_closest(sonar[0])
        if gps is None:
            continue
        depths.append(sonar[1])
        lats.append(gps[1])
        lons.append(gps[2])

    fig = plt.figure(figsize=(10, 8))
    scatter = plt.scatter(lons, lats, c=depths, cmap="plasma", s=1)
    plt.colorbar(scatter, label="Sonar Depth (m)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Sonar Depth vs GPS Coordinates")
    plt.xlim([min(lons) - 0.0001, max(lons) + 0.0001])
    plt.ylim([min(lats) - 0.0001, max(lats) + 0.0001])
    plt.grid(True)
    fig.savefig(path, format="png")
    plt.close(fig)
    log.info("Wrote %s", path)
