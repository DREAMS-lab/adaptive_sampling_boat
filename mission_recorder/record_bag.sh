#!/usr/bin/env bash
# Record a ROS2 bag on the boat and transfer it to the laptop when done.
#
# Config comes from ./config.yaml next to this script. Override any value by
# exporting it as an env var before running:
#
#     LOCAL_HOST=10.0.0.5 ./record_bag.sh
#
# The script:
#   1. Gets a UTC timestamp from the laptop (authoritative clock)
#   2. Starts `ros2 bag record` on the boat
#   3. Waits for the user to press Enter
#   4. SCPs the bag to the laptop

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/config.yaml"

# Tiny YAML reader — grabs "key: value" lines at the top level.
# Avoids a yq dependency for this simple flat config.
yaml_get() {
    local key="$1"
    awk -F': *' -v k="$key" '$1==k { sub(/[[:space:]]*#.*$/, "", $2); print $2 }' "$CONFIG" | head -1
}

LOCAL_USER="${LOCAL_USER:-$(yaml_get local_user)}"
LOCAL_HOST="${LOCAL_HOST:-$(yaml_get local_host)}"
LOCAL_BASE="${LOCAL_BASE:-$(yaml_get local_base)}"
REMOTE_USER="${REMOTE_USER:-$(yaml_get remote_user)}"
ROSBAG_BASE="${ROSBAG_BASE:-$(yaml_get rosbag_base)}"
RECORD_TOPICS="${RECORD_TOPICS:-$(yaml_get record_topics)}"
STORAGE_FORMAT="${STORAGE_FORMAT:-$(yaml_get storage_format)}"

# Expand ~ in paths
LOCAL_BASE="${LOCAL_BASE/#\~/$HOME}"
ROSBAG_BASE="${ROSBAG_BASE/#\~/$HOME}"

echo "Config:"
echo "  laptop:  ${LOCAL_USER}@${LOCAL_HOST}:${LOCAL_BASE}"
echo "  boat:    ${ROSBAG_BASE}"
echo "  topics:  ${RECORD_TOPICS}"
echo "  format:  ${STORAGE_FORMAT}"
echo

# Step 0: Authoritative timestamp from the laptop.
echo "Getting UTC timestamp from ${LOCAL_USER}@${LOCAL_HOST}..."
if BAG_DIR_BASE=$(ssh "${LOCAL_USER}@${LOCAL_HOST}" 'date -u +rosbag2_%Y_%m_%d_%H_%M_%S'); then
    :
else
    echo "Could not reach laptop, using boat clock."
    BAG_DIR_BASE=$(date -u +rosbag2_%Y_%m_%d_%H_%M_%S)
fi

REMOTE_BAG_DIR="${ROSBAG_BASE}/${BAG_DIR_BASE}"
LOCAL_BAG_DIR="${LOCAL_BASE}/${BAG_DIR_BASE}"

echo "Recording to: ${REMOTE_BAG_DIR}"

# Step 1: Start recording in background.
mkdir -p "${ROSBAG_BASE}"
# shellcheck disable=SC2086
nohup ros2 bag record -o "${REMOTE_BAG_DIR}" -s "${STORAGE_FORMAT}" ${RECORD_TOPICS} \
    > "${ROSBAG_BASE}/ros2_bag.log" 2>&1 &
RECORD_PID=$!

echo "Recording started (PID ${RECORD_PID}, log: ${ROSBAG_BASE}/ros2_bag.log)"
echo "Press [Enter] when the mission is done to stop and transfer..."
read -r

# Step 2: Stop the recorder cleanly.
kill -INT "${RECORD_PID}" 2>/dev/null || true
sleep 2

# Step 3: Transfer.
echo "Transferring to ${LOCAL_USER}@${LOCAL_HOST}:${LOCAL_BASE}..."
ssh "${LOCAL_USER}@${LOCAL_HOST}" "mkdir -p '${LOCAL_BASE}'"
scp -r "${REMOTE_BAG_DIR}" "${LOCAL_USER}@${LOCAL_HOST}:${LOCAL_BASE}/"

echo "Transfer complete: ${LOCAL_BAG_DIR}"
