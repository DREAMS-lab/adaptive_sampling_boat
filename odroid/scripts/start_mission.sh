#!/usr/bin/env bash
# Start the full boat mission stack in one command.
#
# Usage:   ./start_mission.sh <mission_name>
# Example: ./start_mission.sh ossabaw_2026_04_17
#
# Wraps `ros2 launch` for convenience. If you need to customise the rosbag
# base path or fcu_url, pass them through:
#
#   ./start_mission.sh ossabaw_2026_04_17 \
#       rosbag_base:=/mnt/big_drive/rosbags \
#       fcu_url:=/dev/boat_fc:921600

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <mission_name> [key:=value ...]" >&2
    exit 1
fi

MISSION_NAME="$1"
shift

# FT232H requires this for the winch GPIO
export BLINKA_FT232H=1

exec ros2 launch boat_control full_mission.launch.py \
    "mission_name:=${MISSION_NAME}" \
    "$@"
