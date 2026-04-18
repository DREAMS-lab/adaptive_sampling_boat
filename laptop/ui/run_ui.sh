#!/usr/bin/env bash
# Launch the boat operator UI.
#
# Activates the ~/karin venv, sources ROS2 Humble, and runs boat_ui.py.
# If you haven't created the venv yet, run ./setup_karin.sh first.

set -eo pipefail   # no -u: ROS2 setup.bash touches unset vars

VENV_DIR="${HOME}/karin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "Venv not found at ${VENV_DIR}. Run ./setup_karin.sh first." >&2
    exit 1
fi

# Source ROS2 before venv so venv's $PATH (and system-site-packages) keeps rclpy.
if [[ -f /opt/ros/humble/setup.bash ]]; then
    # shellcheck source=/dev/null
    source /opt/ros/humble/setup.bash
else
    echo "Warning: /opt/ros/humble/setup.bash not found — ROS2 topics won't work" >&2
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

exec python "${SCRIPT_DIR}/boat_ui.py" "$@"
