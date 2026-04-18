"""SSH helpers for controlling sensor nodes on the Odroid.

Relies on an SSH key already being set up (ssh-copy-id was run earlier in
the session). No passwords are ever sent over the wire.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

log = logging.getLogger(__name__)

DEFAULT_HOST = "odroid@192.168.1.101"
REPO_PATH_ON_ODROID = "~/boat_ws/src/adaptive_sampling_boat"


@dataclass
class SensorCmd:
    """How to start a sensor node on the Odroid."""
    name: str           # friendly label
    pkg: str            # ros2 package name
    exe: str            # ros2 executable name (from setup.py entry_points)
    config_file: str    # YAML file path (relative to repo root on Odroid)
    log_path: str       # where stdout/stderr go on the Odroid
    pattern: str        # pgrep pattern to kill later


SONAR = SensorCmd(
    name="sonar",
    pkg="ping_sonar_ros",
    exe="ping1d_node",
    config_file=f"{REPO_PATH_ON_ODROID}/odroid/config/sensors.yaml",
    log_path="/tmp/ui_sonar.log",
    pattern="ping1d_node",
)

SONDE = SensorCmd(
    name="sonde",
    pkg="sonde_read",
    exe="read_serial",
    config_file=f"{REPO_PATH_ON_ODROID}/odroid/config/sensors.yaml",
    log_path="/tmp/ui_sonde.log",
    pattern="read_serial",
)


def _ssh(cmd: str, host: str = DEFAULT_HOST, timeout: float = 10.0) -> subprocess.CompletedProcess:
    """Run a command on the Odroid over SSH and return the CompletedProcess."""
    full = ["ssh", "-o", f"ConnectTimeout={int(timeout)}", host, cmd]
    log.debug("ssh: %s", cmd)
    return subprocess.run(full, capture_output=True, text=True, timeout=timeout + 5)


def ping(host: str = DEFAULT_HOST) -> bool:
    """Return True if the Odroid answers SSH within 5 s."""
    try:
        r = _ssh("echo ok", host=host, timeout=5)
        return r.returncode == 0 and "ok" in r.stdout
    except subprocess.TimeoutExpired:
        return False


def start_sensor(sensor: SensorCmd, host: str = DEFAULT_HOST) -> tuple[bool, str]:
    """Start a sensor ROS2 node on the Odroid (idempotent — kills existing first).

    Returns (ok, message).
    """
    stop_sensor(sensor, host=host)   # idempotent
    inner = (
        f"source /opt/ros/jazzy/setup.bash && "
        f"source ~/boat_ws/install/setup.bash && "
        f"nohup ros2 run {sensor.pkg} {sensor.exe} "
        f"--ros-args --params-file {sensor.config_file} "
        f"> {sensor.log_path} 2>&1 &"
    )
    r = _ssh(f"bash -lc '{inner}'", host=host, timeout=15)
    if r.returncode != 0:
        return False, f"ssh start failed: {r.stderr.strip()}"
    return True, f"started {sensor.name}"


def stop_sensor(sensor: SensorCmd, host: str = DEFAULT_HOST) -> tuple[bool, str]:
    """Kill the sensor node by process-name pattern."""
    r = _ssh(f"pkill -f {sensor.pattern}; sleep 0.3; pgrep -f {sensor.pattern} || true",
             host=host, timeout=10)
    return True, f"stopped {sensor.name}"


def read_winch_gpio(host: str = DEFAULT_HOST) -> dict[str, bool | str]:
    """Read the 4 FT232H GPIO inputs (C0-C3) via a one-shot Blinka script."""
    script = (
        "source ~/ros2env/bin/activate && "
        "BLINKA_FT232H=1 python3 -c \""
        "import board, digitalio\\n"
        "out = {}\\n"
        "for n in ['C0', 'C1', 'C2', 'C3']:\\n"
        "    p = digitalio.DigitalInOut(getattr(board, n))\\n"
        "    p.direction = digitalio.Direction.INPUT\\n"
        "    out[n] = p.value\\n"
        "    p.deinit()\\n"
        "print(out)"
        "\""
    )
    try:
        r = _ssh(f"bash -lc '{script}'", host=host, timeout=15)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    if r.returncode != 0:
        return {"error": r.stderr.strip() or "ssh failed"}
    line = (r.stdout or "").strip().splitlines()[-1] if r.stdout else ""
    try:
        return eval(line, {"__builtins__": {}}, {"True": True, "False": False})
    except Exception as e:
        return {"error": f"parse: {e}: {line!r}"}


def is_running(sensor: SensorCmd, host: str = DEFAULT_HOST) -> bool:
    """Check if the given sensor node is currently running on the Odroid."""
    try:
        r = _ssh(f"pgrep -f {sensor.pattern} >/dev/null && echo yes || echo no",
                 host=host, timeout=5)
    except subprocess.TimeoutExpired:
        return False
    return "yes" in (r.stdout or "")
