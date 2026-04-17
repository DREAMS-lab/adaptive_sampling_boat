"""Read a ROS2 MCAP rosbag and return the three sensor streams."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message

from .sonde_parser import parse_sonde_line

log = logging.getLogger(__name__)


@dataclass
class BagData:
    gps: List[Tuple[float, float, float]] = field(default_factory=list)     # (t, lat, lon)
    sonar: List[Tuple[float, float]] = field(default_factory=list)          # (t, range_m)
    sonde: List[dict] = field(default_factory=list)                         # list of sonde records


def read_bag(
    bag_path: str,
    gps_topic: str,
    sonar_topic: str,
    sonde_topic: str,
    sonar_offset: float = 0.0,
) -> BagData:
    """Extract the GPS, sonar, and sonde streams from a rosbag directory."""
    # rosbag2_py requires rclpy init before messages can be deserialised.
    if not rclpy.ok():
        rclpy.init()

    topics_of_interest = {gps_topic, sonar_topic, sonde_topic}

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_path, storage_id="mcap"),
        ConverterOptions("", ""),
    )
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

    out = BagData()

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in topics_of_interest:
            continue

        timestamp = t / 1e9
        msg = deserialize_message(data, get_message(type_map[topic]))

        if topic == gps_topic:
            out.gps.append((timestamp, msg.latitude, msg.longitude))
        elif topic == sonar_topic:
            out.sonar.append((timestamp, msg.data + sonar_offset))
        elif topic == sonde_topic:
            record = parse_sonde_line(timestamp, msg.data)
            if record is not None:
                out.sonde.append(record)

    log.info("Bag read: %d GPS, %d sonar, %d sonde samples",
             len(out.gps), len(out.sonar), len(out.sonde))
    return out
