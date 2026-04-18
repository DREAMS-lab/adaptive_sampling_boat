#!/usr/bin/env bash
# Create the "karin" Python virtualenv for the boat UI.
# Lives at ~/karin (outside this repo and outside any ROS2 workspace src/
# so colcon never tries to build it).
#
# Inherits system site-packages so rclpy + mavros_msgs (from /opt/ros/humble)
# remain importable. PyQt5 is pip-installed on top.
#
# Usage:
#   cd laptop/ui && ./setup_karin.sh

set -euo pipefail

VENV_DIR="${HOME}/karin"

if ! command -v python3 >/dev/null; then
    echo "python3 not found" >&2
    exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating venv at ${VENV_DIR} (with system site-packages for rclpy)"
    python3 -m venv --system-site-packages "${VENV_DIR}"
else
    echo "Venv ${VENV_DIR} already exists — reusing"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel >/dev/null
python -m pip install -r "$(dirname "${BASH_SOURCE[0]}")/requirements.txt"

echo
echo "Activate with:  source ${VENV_DIR}/bin/activate"
echo "Launch UI with: $(dirname "${BASH_SOURCE[0]}")/run_ui.sh"
