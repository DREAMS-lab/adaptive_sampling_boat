# Source me — don't execute. Sets up the boat_adaptive shell environment.
#
# Usage:
#   source ~/workspaces/boat_adaptive/tools/setup_env.sh
#
# The ROS_DOMAIN_ID below MUST match on every machine that needs to see
# each other's topics (laptop and Odroid). Change once, propagate to both.

export ROS_DOMAIN_ID=42                  # arbitrary, must match across hosts
export ROS_LOCALHOST_ONLY=0              # allow cross-host DDS
# Uncomment to force Fast-DDS specifically (default on Jazzy):
# export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# ROS 2 base
if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
else
    echo "[setup_env] WARN: /opt/ros/jazzy/setup.bash not found"
fi

# This workspace
_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$_WS/install/setup.bash" ]; then
    source "$_WS/install/setup.bash"
else
    echo "[setup_env] note: $_WS/install/setup.bash not found — run colcon build first"
fi

echo "[setup_env] ROS_DOMAIN_ID=$ROS_DOMAIN_ID  hostname=$(hostname)  ws=$_WS"
