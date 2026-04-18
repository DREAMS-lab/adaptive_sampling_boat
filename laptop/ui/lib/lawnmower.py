"""Lawnmower waypoint pattern generator.

Matches the S-shape used in the simulation
(aquatic-mapping/src/sampling/scripts/missions/lawnmower.py).
"""

from __future__ import annotations

from typing import List, Tuple


def generate(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    row_spacing: float,
    buffer: float = 2.5,
) -> List[Tuple[float, float]]:
    """Return an S-shaped lawnmower path over the rectangle.

    Args:
        x_min, x_max, y_min, y_max: outer survey bounds (metres, local frame)
        row_spacing: vertical spacing between rows (metres)
        buffer: inset from each side of the rectangle (metres)

    Returns:
        list of (x, y) waypoints in visit order.
    """
    if x_max - x_min < 2 * buffer:
        raise ValueError("x range too small for buffer")
    if y_max - y_min < 2 * buffer:
        raise ValueError("y range too small for buffer")
    if row_spacing <= 0:
        raise ValueError("row_spacing must be positive")

    x_lo = x_min + buffer
    x_hi = x_max - buffer
    y_lo = y_min + buffer
    y_hi = y_max - buffer

    waypoints: List[Tuple[float, float]] = []
    y = y_lo
    row = 0
    while y <= y_hi + 1e-6:
        if row % 2 == 0:
            waypoints.append((x_lo, y))
            waypoints.append((x_hi, y))
        else:
            waypoints.append((x_hi, y))
            waypoints.append((x_lo, y))
        y += row_spacing
        row += 1

    return waypoints


def total_length(waypoints: List[Tuple[float, float]]) -> float:
    """Sum of straight-line distances between consecutive waypoints."""
    total = 0.0
    for (x0, y0), (x1, y1) in zip(waypoints, waypoints[1:]):
        total += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    return total
