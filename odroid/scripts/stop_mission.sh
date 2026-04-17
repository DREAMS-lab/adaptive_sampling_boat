#!/usr/bin/env bash
# Cleanly stop all mission nodes and flush the rosbag to disk.
#
# If the launch file is still running in another terminal, Ctrl-C there first.
# This script kills any leftover nodes + the rosbag recorder.

set -uo pipefail

echo "Stopping rosbag recorder..."
pkill -INT -f "ros2 bag record" || true

echo "Stopping sensor nodes..."
pkill -INT -f "ping1d_node" || true
pkill -INT -f "sonde_reader" || true
pkill -INT -f "winch_mission_node" || true

echo "Stopping MAVROS..."
pkill -INT -f "mavros_node" || true

sleep 2

# If anything is still alive, escalate.
for name in "ros2 bag record" "ping1d_node" "sonde_reader" "winch_mission_node" "mavros_node"; do
    if pgrep -f "$name" > /dev/null; then
        echo "Force-killing $name"
        pkill -KILL -f "$name" || true
    fi
done

echo "Done. Check /home/rosbags for the recorded bag."
