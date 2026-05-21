#!/usr/bin/env bash
# Verify the laptop ↔ Odroid ROS 2 link and all expected topics.
#
# Run this on the LAPTOP after sensors + MAVROS have been started on the Odroid.
# Exits 0 on full pass, 1 on first failure.
#
#   bash ~/workspaces/boat_adaptive/tools/check_link.sh
#
# Verbose mode (show topic samples):
#   VERBOSE=1 bash ~/workspaces/boat_adaptive/tools/check_link.sh

set -u

REQUIRED_TOPICS=(
    "/mavros/state"
    "/mavros/local_position/pose"
    "/ping1d/data"
    "sonde_data"
)

REQUIRED_MIN_HZ=(
    "/mavros/state 0.5"
    "/mavros/local_position/pose 5"
    "/ping1d/data 5"
    "sonde_data 5"
)

green() { printf "\033[32m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }
amber() { printf "\033[33m%s\033[0m\n" "$*"; }

fail() { red "FAIL: $*"; exit 1; }

if [ -z "${ROS_DOMAIN_ID:-}" ]; then
    fail "ROS_DOMAIN_ID is unset. Source tools/setup_env.sh first."
fi

echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID  host=$(hostname)"
echo "-----"

echo "1. Topic discovery..."
sleep 1   # let discovery settle
TOPICS=$(ros2 topic list 2>/dev/null) || fail "ros2 topic list failed (ROS not sourced?)"
for t in "${REQUIRED_TOPICS[@]}"; do
    if echo "$TOPICS" | grep -qx "$t"; then
        green "  ok   $t"
    else
        red "  miss $t"
        fail "topic $t not visible. Check ROS_DOMAIN_ID matches on Odroid and that the node is running."
    fi
done
echo

echo "2. Topic rates (5 s sample each)..."
for entry in "${REQUIRED_MIN_HZ[@]}"; do
    topic="${entry%% *}"
    min_hz="${entry##* }"
    out=$(timeout 6 ros2 topic hz "$topic" 2>&1 | grep -m1 "average rate" || true)
    if [ -z "$out" ]; then
        fail "no messages on $topic in 6 s"
    fi
    hz=$(echo "$out" | sed -E 's/.*average rate: ([0-9.]+).*/\1/')
    awk -v h="$hz" -v m="$min_hz" 'BEGIN{exit !(h+0>=m+0)}'
    if [ $? -eq 0 ]; then
        green "  ok   $topic @ ${hz} Hz (>= ${min_hz})"
    else
        red "  slow $topic @ ${hz} Hz (need >= ${min_hz})"
        fail "topic $topic is publishing too slowly"
    fi
done
echo

echo "3. MAVROS state..."
state=$(timeout 3 ros2 topic echo --once /mavros/state 2>/dev/null)
if [ -z "$state" ]; then fail "no /mavros/state message"; fi
connected=$(echo "$state" | awk '/^connected:/{print $2}')
mode=$(echo "$state" | awk '/^mode:/{print $2}')
armed=$(echo "$state" | awk '/^armed:/{print $2}')
if [ "$connected" = "true" ]; then
    green "  connected=true  mode=$mode  armed=$armed"
else
    fail "MAVROS connected=$connected (FCU not talking)"
fi
echo

echo "4. Sample values..."
pose=$(timeout 3 ros2 topic echo --once /mavros/local_position/pose 2>/dev/null | grep -A3 "position:" | head -4)
echo "  pose:"; echo "$pose" | sed 's/^/    /'

depth=$(timeout 3 ros2 topic echo --once /ping1d/data 2>/dev/null | awk '/data:/{print $2}')
echo "  ping depth: $depth m"

sonde=$(timeout 3 ros2 topic echo --once sonde_data 2>/dev/null | grep '^data:' | head -1)
echo "  $sonde"

# pull temp from sonde row (whitespace-padded, col index 3 after stripping "data: \"")
temp=$(echo "$sonde" | sed -E 's/^data: *"//; s/"$//' | awk '{print $4}')
echo "  parsed sonde temp (col 3): $temp °C"

if [ -n "$temp" ]; then
    awk -v t="$temp" 'BEGIN{exit !(t+0>=0 && t+0<=50)}' \
        && green "  temp in [0,50] °C" \
        || amber "  WARN: temp $temp outside [0,50] — sonde wet?"
fi
echo

green "ALL CHECKS PASS"
echo
echo "Next: ros2 run boat_bringup pose_commander.py   (or run a preflight / mission launch)"
