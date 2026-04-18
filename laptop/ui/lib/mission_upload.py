"""Mission upload helper — wraps ros_bridge.push_waypoints().

The heavy lifting lives in RosBridge. This file exists so the UI can format
preview text without owning a ROS node.
"""

from __future__ import annotations

from typing import List, Tuple


def preview_text(waypoints: List[Tuple[float, float]]) -> str:
    """Render a human-readable preview for the UI text box."""
    if not waypoints:
        return "(empty)"
    from .lawnmower import total_length
    total = total_length(waypoints)
    lines = [
        f"{len(waypoints)} waypoints, {total:.1f} m total",
        f"  start:  ({waypoints[0][0]:.2f}, {waypoints[0][1]:.2f})",
        f"  end:    ({waypoints[-1][0]:.2f}, {waypoints[-1][1]:.2f})",
        "",
        "  all points:",
    ]
    for i, (x, y) in enumerate(waypoints):
        lines.append(f"    {i:2d}: ({x:7.2f}, {y:7.2f})")
    return "\n".join(lines)
