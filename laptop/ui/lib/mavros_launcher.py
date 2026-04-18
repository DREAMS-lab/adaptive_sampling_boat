"""Launch MAVROS in a terminal window so logs are visible and Ctrl-C works.

We spawn a terminal (konsole / gnome-terminal / xterm / xfce4-terminal,
whichever is on PATH) that runs `ros2 launch mavros px4.launch ...` directly.
The user sees live output, and Ctrl-C in that window shuts MAVROS down the
same way it would from any other terminal.

The UI's Start/Stop buttons are convenience wrappers:
  Start → opens the terminal (or refuses if MAVROS is already running)
  Stop  → SIGINTs the mavros_node process and the terminal

PID tracking uses pgrep/pkill on the process name since the terminal itself
is the subprocess parent we own, not MAVROS directly.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

DEFAULT_FCU = "/dev/ttyUSB0:57600"
DEFAULT_GCS = "udp://@localhost:14550"
MAVROS_PROC_NAME = "mavros_node"

# Terminal emulators to try, in priority order, with the flag for "run command".
TERMINAL_CANDIDATES = [
    ("konsole", "-e"),
    ("gnome-terminal", "--"),
    ("xfce4-terminal", "-e"),
    ("xterm", "-e"),
]


def _mavros_is_running() -> bool:
    r = subprocess.run(["pgrep", "-f", MAVROS_PROC_NAME],
                       capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def _who_has_port(port_dev: str) -> list[str]:
    """Return list of 'pid command' strings for processes holding port_dev."""
    # Extract the device path, drop the ':baud' suffix MAVROS uses
    dev = port_dev.split(":", 1)[0]
    if not os.path.exists(dev):
        return []
    r = subprocess.run(["fuser", dev], capture_output=True, text=True)
    # fuser prints PIDs to stderr when found; stdout stays empty
    pids_raw = (r.stderr + " " + r.stdout).split(":", 1)[-1]
    pids = [p for p in pids_raw.split() if p.isdigit()]
    out = []
    for pid in pids:
        try:
            with open(f"/proc/{pid}/comm") as f:
                name = f.read().strip()
            out.append(f"{pid} ({name})")
        except OSError:
            out.append(pid)
    return out


def _kill_stale_mavros() -> None:
    """Kill any mavros_node that we don't own (leftover from prior runs)."""
    subprocess.run(["pkill", "-9", "-f", MAVROS_PROC_NAME], capture_output=True)
    time.sleep(0.3)


def _find_terminal() -> Optional[tuple[str, str]]:
    for exe, flag in TERMINAL_CANDIDATES:
        if shutil.which(exe):
            return exe, flag
    return None


class MavrosLauncher:
    """Owns the terminal window that runs MAVROS."""

    def __init__(self) -> None:
        self._term: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        return _mavros_is_running()

    def start(
        self,
        fcu_url: str = DEFAULT_FCU,
        gcs_url: str = DEFAULT_GCS,
    ) -> tuple[bool, str]:
        """Open a terminal window running `ros2 launch mavros px4.launch …`.

        Before launching, kills any stale mavros_node AND anything else holding
        the serial port (typically QGC). Aggressive — user asked for it.
        """
        killed_msg = ""
        with self._lock:
            # Clean up any stale MAVROS from a previous run
            _kill_stale_mavros()

            # Kill anything still holding the serial port (usually QGC)
            holders = _who_has_port(fcu_url)
            if holders:
                dev = fcu_url.split(":", 1)[0]
                subprocess.run(["fuser", "-k", dev], capture_output=True)
                time.sleep(0.5)
                still_held = _who_has_port(fcu_url)
                if still_held:
                    return False, (
                        f"{dev} is still held by {', '.join(still_held)} after kill attempt"
                    )
                killed_msg = f"(killed {', '.join(holders)} to free {dev}) "

            term = _find_terminal()
            if term is None:
                return False, "no terminal emulator found (install konsole or xterm)"
            exe, flag = term

            # The inner bash script: run MAVROS and keep the terminal open
            # after it exits so the user can read the final log.
            inner = (
                f"echo '=== MAVROS launch ===' && "
                f"echo 'fcu_url={fcu_url}' && "
                f"echo 'gcs_url={gcs_url}' && "
                f"echo '--- Ctrl-C in this window stops MAVROS ---' && echo && "
                f"ros2 launch mavros px4.launch "
                f"fcu_url:={fcu_url} gcs_url:={gcs_url}; "
                f"echo; echo '[MAVROS exited. This window will close in 3s]'; sleep 3"
            )

            # konsole + gnome-terminal need "bash -c" expressed slightly differently.
            if exe == "gnome-terminal":
                cmd = [exe, "--", "bash", "-c", inner]
            else:
                cmd = [exe, flag, "bash", "-c", inner]

            log.info("spawning MAVROS terminal: %s", " ".join(cmd[:3]))
            try:
                self._term = subprocess.Popen(cmd, preexec_fn=os.setsid)
            except Exception as e:
                return False, f"terminal spawn failed: {e}"

        # Wait up to 5s for mavros_node to appear
        for _ in range(25):
            time.sleep(0.2)
            if _mavros_is_running():
                return True, f"MAVROS running {killed_msg}(check the terminal window)"
        return False, "MAVROS didn't start within 5s — check the terminal for errors"

    def stop(self, timeout: float = 5.0) -> tuple[bool, str]:
        """SIGINT mavros_node and close the terminal window."""
        # Polite interrupt first — ros2 launch catches this and tears down cleanly.
        if _mavros_is_running():
            subprocess.run(["pkill", "-INT", "-f", MAVROS_PROC_NAME],
                           capture_output=True)
            t0 = time.monotonic()
            while _mavros_is_running() and time.monotonic() - t0 < timeout:
                time.sleep(0.2)
            if _mavros_is_running():
                subprocess.run(["pkill", "-KILL", "-f", MAVROS_PROC_NAME],
                               capture_output=True)
        # Close the terminal window too
        with self._lock:
            term = self._term
            self._term = None
        if term is not None and term.poll() is None:
            try:
                os.killpg(os.getpgid(term.pid), signal.SIGTERM)
            except Exception:
                pass
        return True, "stopped"
