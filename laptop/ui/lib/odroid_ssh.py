"""SSH helpers for controlling sensor nodes on the Odroid.

Relies on an SSH key already being set up (ssh-copy-id was run earlier).
All SSH commands are wrapped in a background-friendly form so they don't
hang the UI:

- Sensor start uses ssh -f + proper stdin redirection so the background
  ros2 node doesn't keep the TCP connection alive.
- GPIO read pipes a Python script via stdin rather than embedding the
  script as an argument string (avoids shell-quoting hell).
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
    name: str
    pkg: str
    exe: str
    config_file: str
    log_path: str
    pattern: str


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


def _ssh(cmd: str, host: str = DEFAULT_HOST, timeout: float = 10.0,
         stdin: str | None = None) -> subprocess.CompletedProcess:
    full = ["ssh", "-o", f"ConnectTimeout={int(timeout)}",
            "-o", "BatchMode=yes", host, cmd]
    log.debug("ssh: %s", cmd[:120])
    return subprocess.run(full, input=stdin, capture_output=True, text=True,
                          timeout=timeout + 2)


def ping(host: str = DEFAULT_HOST) -> bool:
    try:
        r = _ssh("echo ok", host=host, timeout=5)
        return r.returncode == 0 and "ok" in r.stdout
    except subprocess.TimeoutExpired:
        return False


def start_sensor(sensor: SensorCmd, host: str = DEFAULT_HOST) -> tuple[bool, str]:
    """Start a sensor ROS2 node on the Odroid.

    Uses a subshell-detach pattern — `( nohup ... </dev/null & )` — so the
    remote bash exits immediately and ssh returns right after. No hang.
    """
    stop_sensor(sensor, host=host)

    inner = (
        f"source /opt/ros/jazzy/setup.bash && "
        f"source ~/boat_ws/install/setup.bash && "
        f"( nohup ros2 run {sensor.pkg} {sensor.exe} "
        f"--ros-args --params-file {sensor.config_file} "
        f"> {sensor.log_path} 2>&1 </dev/null & ) && echo spawned"
    )
    try:
        r = _ssh(f"bash -lc '{inner}'", host=host, timeout=8)
    except subprocess.TimeoutExpired:
        return False, "ssh timeout starting sensor"
    if r.returncode != 0 or "spawned" not in (r.stdout or ""):
        return False, f"ssh start failed: {r.stderr.strip() or r.stdout.strip()}"
    return True, f"started {sensor.name}"


def stop_sensor(sensor: SensorCmd, host: str = DEFAULT_HOST) -> tuple[bool, str]:
    try:
        _ssh(f"pkill -f {sensor.pattern}; true", host=host, timeout=8)
    except subprocess.TimeoutExpired:
        return False, "ssh timeout"
    return True, f"stopped {sensor.name}"


_WINCH_GPIO_SCRIPT = """
import board, digitalio
out = {}
for n in ["C0", "C1", "C2", "C3"]:
    p = digitalio.DigitalInOut(getattr(board, n))
    p.direction = digitalio.Direction.INPUT
    out[n] = p.value
    p.deinit()
print(out)
"""


def read_winch_gpio(host: str = DEFAULT_HOST) -> dict:
    """Read the four FT232H GPIO inputs (C0-C3). Script is piped via stdin
    so we don't have to shell-escape Python source."""
    cmd = "source ~/ros2env/bin/activate && BLINKA_FT232H=1 python3 -"
    try:
        r = _ssh(f"bash -lc '{cmd}'", host=host, timeout=12, stdin=_WINCH_GPIO_SCRIPT)
    except subprocess.TimeoutExpired:
        return {"error": "ssh timeout"}
    if r.returncode != 0:
        return {"error": (r.stderr or r.stdout).strip() or "ssh failed"}
    line = (r.stdout or "").strip().splitlines()[-1] if r.stdout else ""
    try:
        return eval(line, {"__builtins__": {}}, {"True": True, "False": False})
    except Exception as e:
        return {"error": f"parse: {e}: {line!r}"}


def is_running(sensor: SensorCmd, host: str = DEFAULT_HOST) -> bool:
    try:
        r = _ssh(f"pgrep -f {sensor.pattern} >/dev/null && echo yes || echo no",
                 host=host, timeout=5)
    except subprocess.TimeoutExpired:
        return False
    return "yes" in (r.stdout or "")
