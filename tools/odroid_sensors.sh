#!/usr/bin/env bash
# Start MAVROS + ping + sonde on the Odroid in a tmux session.
#
#   bash ~/workspaces/boat_adaptive/tools/odroid_sensors.sh
#
# Then either re-attach with `tmux a -t boat`, or just leave it running.
# Stop with: tmux kill-session -t boat

set -e

if ! command -v tmux >/dev/null; then
    echo "tmux not installed. sudo apt install tmux"
    exit 1
fi

if tmux has-session -t boat 2>/dev/null; then
    echo "Session 'boat' already exists. Attach with: tmux a -t boat"
    exit 0
fi

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="source $WS/tools/setup_env.sh"

tmux new-session -d -s boat -n mavros \
    "$SRC && ros2 launch mavros px4.launch fcu_url:=/dev/ttyUSB0:921600; bash"

tmux new-window -t boat: -n ping \
    "$SRC && ros2 run ping_sonar_ros ping1d_node; bash"

tmux new-window -t boat: -n sonde \
    "$SRC && ros2 run sonde_read read_serial.py; bash"

echo "tmux 'boat' started with 3 windows: mavros, ping, sonde."
echo "Attach: tmux a -t boat   |   Stop: tmux kill-session -t boat"
